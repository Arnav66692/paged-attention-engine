"""SDPA's own GQA path against the materialising one, held to bit-equality.

`_sdpa` used to expand K and V to the query head count with `repeat_kv` before
every attention call. That expansion is a real allocation -- `expand` is a view
but the `reshape` after it cannot be -- and at the headline model's 6x ratio it
measured 1.125 GiB of transient per layer at batch 64 on a card with 14.56 GiB
total and 4.4 GiB already static.

`enable_gqa=True` makes SDPA broadcast internally and allocate nothing. That is
a change to the kernel path, so it is held to the strictest bar available: not
"within a tolerance" but `torch.equal`. If the two ever stop being bit-identical
on some device, this fails and the fused/parity tolerances are not asked to
absorb it.

The full parity matrix on CUDA is re-verified separately, on the GPU, because
this file cannot see a device it does not have.
"""

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


def test_repeat_kv_still_materialises_which_is_why_it_left_the_path():
    """The measurement that motivated the switch, kept runnable.

    `repeat_kv` is retained as the reference and as this probe. If it ever stops
    materialising -- if some future torch makes the reshape a view -- the switch
    stops being a memory win and that should be visible, not assumed.
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
        "repeat_kv no longer copies, so the memory argument for enable_gqa "
        "needs re-checking rather than assuming"
    )
    assert expanded_bytes == source_bytes * 6


def test_a_one_to_one_ratio_needs_no_expansion_at_all():
    """num_kv_groups == 1 is a no-op both ways, and enable_gqa is not passed."""
    source = torch.randn(2, 4, 8, 16)
    assert repeat_kv(source, 1) is source
