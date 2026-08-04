"""SDPA's own GQA path against the materialising one, held to bit-equality.

`_sdpa` used to expand K and V to the query head count with `repeat_kv` before
every attention call. That expansion is a real allocation -- `expand` is a view
but the `reshape` after it cannot be.

`enable_gqa=True` makes SDPA broadcast internally instead. That is a change to
the kernel path, so it is held to the strictest bar available: not "within a
tolerance" but `torch.equal`. If the two ever stop being bit-identical on some
device, this fails and the fused/parity tolerances are not asked to absorb it.

THE SWITCH WAS MADE FOR MEMORY AND THAT REASON WAS WRONG. This file measures
what `repeat_kv` allocates and concluded the switch saved it. It never measured
what `enable_gqa` allocates instead. `kaggle/session4` did, on the T4, and
`enable_gqa` costs 3.3-8.3x MORE transient -- 2567 MiB against 768 MiB at batch
64, kv 2048 -- because MATH expands K/V for its matmul anyway and materialises
the score matrix on top. It is also 2.76-5.67x slower.

`enable_gqa` is kept regardless, for a reason this file cannot see: it is the
path the HF baseline takes, measured in `kaggle/session3`. `docs/AMPERE_RECHECK.md`
items 2 and 6 carry both findings, because both are device-dependent.

WHAT THIS FILE DOES NOT SHOW, and what it must not be read as showing. These
tests pass identical inputs to both paths, so equality here says the EXPANSION
is exact -- which it is, fp16 into fp16. It says nothing about which kernel SDPA
picks. On CUDA the switch improved fp16 parity 6x, and that came from SDPA
dispatching differently for the two input SHAPES, not from the copy. The copy
cost memory only. `kaggle/session2` identifies the backends; the mechanism note
lives in `_sdpa`.

The full parity matrix on CUDA is re-verified separately, on the GPU, because
this file cannot see a device it does not have.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from engine.attention import repeat_kv
from tests.devices import model_devices

# The headline model's shape: 12 query heads over 2 KV heads, head_dim 128.
# Plus a 1x case, where enable_gqa does not apply and must not be passed.
GQA_SHAPES = (
    pytest.param(12, 2, 128, id="headline-6x"),
    pytest.param(14, 2, 64, id="parity-model-7x"),
    pytest.param(4, 4, 32, id="no-expansion-1x"),
)
SEED = 0


def _kernels(device, num_heads, num_kv_heads, head_dim, query_len, kv_len):
    torch.manual_seed(SEED)
    batch = 3
    query = torch.randn(batch, num_heads, query_len, head_dim, device=device)
    keys = torch.randn(batch, num_kv_heads, kv_len, head_dim, device=device)
    values = torch.randn(batch, num_kv_heads, kv_len, head_dim, device=device)
    return query, keys, values


@pytest.mark.parametrize("device", model_devices())
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", GQA_SHAPES)
def test_enable_gqa_is_bit_identical_to_repeat_kv_with_a_mask(
    device, num_heads, num_kv_heads, head_dim
):
    """The fused path's shape: an explicit boolean mask, no causal shortcut."""
    groups = num_heads // num_kv_heads
    query, keys, values = _kernels(device, num_heads, num_kv_heads, head_dim, 5, 9)

    mask = torch.zeros(3, 1, 5, 9, dtype=torch.bool, device=device)
    mask[:, :, :, 0] = True          # every row admits at least one key
    mask[0, 0, :3, :4] = True        # and one row is genuinely restrictive

    expanded = F.scaled_dot_product_attention(
        query, repeat_kv(keys, groups), repeat_kv(values, groups),
        attn_mask=mask, dropout_p=0.0, scale=0.125, is_causal=False,
    )
    broadcast = F.scaled_dot_product_attention(
        query, keys, values, attn_mask=mask, dropout_p=0.0, scale=0.125,
        is_causal=False, enable_gqa=groups > 1,
    )

    deviation = (expanded - broadcast).abs().max().item()
    print(f"\n[{device}] {num_heads}q/{num_kv_heads}kv ({groups}x) masked: "
          f"max|diff|={deviation:.3e} exact={torch.equal(expanded, broadcast)}")
    assert torch.equal(expanded, broadcast), (
        f"[{device}] enable_gqa diverged from repeat_kv by {deviation:.3e} at "
        f"{groups}x expansion; the switch is not free on this device"
    )


@pytest.mark.parametrize("device", model_devices())
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim", GQA_SHAPES)
def test_enable_gqa_is_bit_identical_to_repeat_kv_when_causal(
    device, num_heads, num_kv_heads, head_dim
):
    """The single-sequence backends' shape: SDPA's is_causal shortcut."""
    groups = num_heads // num_kv_heads
    query, keys, values = _kernels(device, num_heads, num_kv_heads, head_dim, 7, 7)

    expanded = F.scaled_dot_product_attention(
        query, repeat_kv(keys, groups), repeat_kv(values, groups),
        dropout_p=0.0, scale=0.125, is_causal=True,
    )
    broadcast = F.scaled_dot_product_attention(
        query, keys, values, dropout_p=0.0, scale=0.125, is_causal=True,
        enable_gqa=groups > 1,
    )

    print(f"[{device}] {num_heads}q/{num_kv_heads}kv ({groups}x) causal: "
          f"exact={torch.equal(expanded, broadcast)}")
    assert torch.equal(expanded, broadcast)


def test_repeat_kv_still_materialises_its_expansion():
    """Half of a comparison, and the half that was mistaken for all of it.

    `repeat_kv` does copy. This test proves it and the size is real: 768 MiB at
    batch 64, kv 2048 on the headline model. What this test CANNOT see is what
    `enable_gqa` allocates instead, and for a while that absence was read as
    zero. It is not zero; it is larger. See the module docstring.

    So this stays, but as a fact about `repeat_kv` alone -- useful because it
    prices the reference path and would catch a future torch making the reshape
    a view. It is not evidence for the switch.
    """
    source = torch.zeros((4, 2, 64, 32), dtype=torch.float32)
    expanded = repeat_kv(source, 6)

    source_bytes = source.numel() * source.element_size()
    expanded_bytes = expanded.numel() * expanded.element_size()
    print(f"\nrepeat_kv 6x: {source_bytes / 2**10:.0f} KiB -> "
          f"{expanded_bytes / 2**10:.0f} KiB, "
          f"same storage={expanded.data_ptr() == source.data_ptr()}")

    assert expanded.shape == (4, 12, 64, 32)
    assert expanded.data_ptr() != source.data_ptr(), (
        "repeat_kv no longer copies, so the transient cost of the reference "
        "path has changed and kaggle/session4's comparison needs re-running"
    )
    assert expanded_bytes == source_bytes * 6


def test_a_one_to_one_ratio_needs_no_expansion_at_all():
    """num_kv_groups == 1 is a no-op both ways, and enable_gqa is not passed."""
    source = torch.randn(2, 4, 8, 16)
    assert repeat_kv(source, 1) is source


# --- the shipped default ---------------------------------------------------


def test_the_shipped_default_is_repeat_kv():
    """`repeat_kv` ships. This is a performance choice with a measured basis.

    It was `sdpa` for the whole benchmark, chosen for SYMMETRY against an HF
    baseline that session 3 measured on MATH. Session 5 showed that symmetry was
    an artifact of the prompt-uniform batch session 3 probed: HF takes
    `enable_gqa` only when `attention_mask is None`, so on a ragged stream the
    baseline already holds EFFICIENT_ATTENTION and the engine on `sdpa` was
    carrying a ~2x handicap nobody chose.

    A shipped engine has no baseline to stay level with, so the only question is
    which kernel is faster and lighter. `repeat_kv` is both: 2.76-5.67x per call
    and 3.3-8.3x less transient memory (kaggle/session4). The benchmark is
    unaffected -- `bench/arms.py` sets the path explicitly per arm.

    Pinned because a silent revert would change every kernel this engine issues
    while every equality test kept passing: the two paths are `torch.equal` given
    identical inputs, so equality cannot detect the swap. Only this can.
    """
    import os

    from engine import attention

    assert "PAGED_ATTENTION_GQA_PATH" not in os.environ, (
        "this test reads the DEFAULT, so it cannot run with the override set"
    )
    assert attention.GQA_PATH == "repeat_kv", (
        f"the shipped GQA path is {attention.GQA_PATH!r}. If this moved back to "
        "'sdpa', note that on sm_75 that is the MATH path -- slower per call and "
        "heavier in transient memory, because MATH expands K/V for its matmul "
        "anyway and materialises the score matrix on top."
    )


def test_both_paths_remain_selectable_and_the_reference_is_not_deleted():
    """`sdpa` stays reachable: bench/arms.py turns this knob, and it is the
    reference the equality tests above are asserted against."""
    from engine import attention

    assert callable(attention.repeat_kv)
    source = Path(attention.__file__).read_text()
    for path in ("repeat_kv", "sdpa"):
        assert f'"{path}"' in source, f"the {path!r} path is no longer selectable"
