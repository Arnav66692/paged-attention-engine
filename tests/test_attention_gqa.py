"""SDPA's own GQA path against the materialising one, held to bit-equality.

`_sdpa` used to expand K and V to the query head count with `repeat_kv` before
every attention call. That expansion is a real allocation -- `expand` is a view
but the `reshape` after it cannot be.

`enable_gqa=True` makes SDPA broadcast internally instead. That is a change to
the kernel path, so it is held to the strictest bar available: not "within a
tolerance" but `torch.equal`.

AND ON CUDA IT DOES NOT HOLD. Measured on a T4 in `kaggle/session8`: where the
two paths dispatch to different backends they cannot be bit-identical, and on
sm_75 they do. Those cases are `xfail(strict=True)` with the mechanism in the
marker -- NOT `allclose`, which would make the suite green by hiding a property
of the hardware. `strict=True` means an XPASS fails too, so if the dispatch ever
changes the fact cannot rot silently. `docs/AMPERE_RECHECK.md` item 7 owns it.

THE SWITCH WAS MADE FOR MEMORY AND THAT REASON WAS WRONG. This file measures
what `repeat_kv` allocates and concluded the switch saved it. It never measured
what `enable_gqa` allocates instead. `kaggle/session4` did, on the T4, and
`enable_gqa` costs 3.3-8.3x MORE transient -- 2567 MiB against 768 MiB at batch
64, kv 2048 -- because MATH expands K/V for its matmul anyway and materialises
the score matrix on top. It is also 2.76-5.67x slower.

SO `repeat_kv` NOW SHIPS. `enable_gqa` was kept for a while on the grounds that
it was the path the HF baseline takes -- true only for prompt-UNIFORM batches,
which is not the realistic stream. Both findings are device-dependent and live
in `docs/AMPERE_RECHECK.md` items 2 and 6. `enable_gqa` stays selectable: it is
the reference these equality tests are asserted against and the knob
`bench/arms.py` turns.

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

# THE ONE DEVICE FACT THAT DEFEATS BIT-EQUALITY, stated in the marker rather
# than hidden behind a loosened comparison.
#
# `torch.equal` STAYS. The alternative -- swapping it for `allclose` -- would
# make the suite green by concealing a measured property of the hardware.
# `xfail(strict=True)` says the opposite: this is expected to fail HERE, for
# this reason, and if it ever starts passing THAT is a failure too, because it
# means the dispatch changed and item 7 needs re-measuring.
CUDA_DISPATCH_DIVERGENCE = (
    "enable_gqa and repeat_kv are NOT bit-identical on a device where the two "
    "paths dispatch differently. They present different input SHAPES -- "
    "unexpanded K/V with enable_gqa=True versus pre-expanded K/V -- and torch "
    "selects MATH for the first and EFFICIENT_ATTENTION for the second. The two "
    "kernels accumulate in different orders, so bit-equality was never available "
    "here; it holds on cpu and mps, where both shapes reach the same kernel. "
    "Measured on a Kaggle T4 sm_75 in kaggle/session8, the first run of the full "
    "default suite on the measurement host. torch.equal is deliberately kept -- "
    "allclose would hide this. strict=True: an XPASS means the dispatch moved "
    "and docs/AMPERE_RECHECK.md item 7 must be re-measured. On sm_80+ FLASH may "
    "accept enable_gqa, in which case both paths could land on one backend and "
    "this is expected to XPASS."
)


def _expect_cuda_divergence(request, device, groups):
    """Mark the known sm_75 dispatch divergence, and ONLY where it applies.

    Not blanket-applied to CUDA: at `groups == 1` the engine does not pass
    `enable_gqa` at all, so both paths are literally the same call and equality
    holds on every device. Marking that case would claim a failure that does not
    exist and `strict=True` would then fail it for passing.
    """
    if device == "cuda" and groups > 1:
        request.applymarker(
            pytest.mark.xfail(strict=True, reason=CUDA_DISPATCH_DIVERGENCE)
        )


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
    request, device, num_heads, num_kv_heads, head_dim
):
    """The fused path's shape: an explicit boolean mask, no causal shortcut."""
    groups = num_heads // num_kv_heads
    _expect_cuda_divergence(request, device, groups)
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
    request, device, num_heads, num_kv_heads, head_dim
):
    """The single-sequence backends' shape: SDPA's is_causal shortcut."""
    groups = num_heads // num_kv_heads
    _expect_cuda_divergence(request, device, groups)
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


def test_the_divergence_marker_applies_exactly_where_the_divergence_is():
    """The xfail must be narrow, and it cannot be checked on a host without CUDA.

    `strict=True` cuts both ways: marking a case that actually passes turns an
    XPASS into a failure. So the predicate has to be exactly right, and on a
    development machine the CUDA branch never executes. This drives the
    predicate directly with a recording stub.
    """
    applied: list = []

    class _Request:
        def applymarker(self, marker):
            applied.append(marker)

    cases = {
        ("cuda", 6): True,    # headline-6x     -- MATH vs EFFICIENT, diverges
        ("cuda", 7): True,    # parity-model-7x -- same
        ("cuda", 1): False,   # no-expansion-1x -- enable_gqa not passed at all
        ("cpu", 6): False,    # both shapes reach the same kernel
        ("mps", 6): False,
        ("cpu", 1): False,
    }
    for (device, groups), expected in cases.items():
        applied.clear()
        _expect_cuda_divergence(_Request(), device, groups)
        assert bool(applied) is expected, (
            f"device={device!r} groups={groups}: marker applied={bool(applied)}, "
            f"expected {expected}. Marking a passing case makes strict=True fail "
            "it for succeeding; leaving a failing case unmarked makes the suite red."
        )
        if expected:
            marker = applied[0]
            assert marker.kwargs["strict"] is True, (
                "strict=False would let the fact rot silently the day the "
                "dispatch changes"
            )
            reason = marker.kwargs["reason"]
            for mechanism in ("MATH", "EFFICIENT_ATTENTION", "accumulate",
                              "SHAPES", "AMPERE_RECHECK"):
                assert mechanism in reason, (
                    f"the xfail reason does not carry {mechanism!r}. The reason "
                    "IS the documentation here -- a bare xfail would hide the "
                    "device fact exactly as allclose would."
                )


def test_no_gqa_test_was_quietly_switched_to_allclose():
    """torch.equal is the bar. Loosening it is the failure mode xfail avoids.

    Checked against the AST, not the text: this file DISCUSSES allclose at
    length in order to explain why it is not used, so a substring search would
    flag its own documentation.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text())
    loosened = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("allclose", "isclose")
    ]
    assert not loosened, (
        f"an equality assertion in this file was relaxed to {loosened}. That "
        "makes the suite green by hiding which kernel ran; use "
        "xfail(strict=True) with the mechanism in the reason instead."
    )
    equals = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "equal"
    ]
    assert len(equals) >= 4, (
        f"only {len(equals)} torch.equal calls remain; the bit-equality claims "
        "are the point of this file"
    )


def test_the_prose_in_attention_py_does_not_contradict_the_shipped_default():
    """The comments described the OTHER configuration as current, in four places.

    Found by handing a fresh clone to a reader with no context. `repeat_kv`'s own
    docstring said "NO LONGER ON THE ATTENTION PATH" while being the shipped
    path; `_sdpa` said "WHY `sdpa` IS THE DEFAULT" while it was not; and the
    kernel bias was described as "UNMEASURED AND OF UNKNOWN SIGN" after both
    arms had measured it at +2.30% and -4.77%.

    That is the worst possible file for stale prose -- the highest comment
    density in the repo, wrapping the one switch everything else depends on --
    and nothing caught it, because pinning the VALUE of GQA_PATH says nothing
    about the paragraph above it claiming the opposite.
    """
    from engine import attention

    source = Path(attention.__file__).read_text()
    shipped = attention.GQA_PATH

    # Phrases that are only true of the configuration that is NOT shipped.
    contradictions = {
        "repeat_kv": [
            "NO LONGER ON THE ATTENTION PATH",
            "WHY `sdpa` IS THE DEFAULT",
            "GQA IS SDPA'S JOB, NOT OURS",
        ],
        "sdpa": [
            "`repeat_kv` SHIPS",
            "THIS IS THE SHIPPED PATH",
        ],
    }
    for phrase in contradictions.get(shipped, []):
        assert phrase not in source, (
            f"engine/attention.py still says {phrase!r}, which describes the "
            f"path that is NOT shipped. GQA_PATH is {shipped!r}. A reader trusts "
            "the prose here more than anywhere else in the repo; it is the file "
            "that argues for its own trustworthiness."
        )

    # A settled measurement must not still be described as open.
    for phrase in ("UNMEASURED AND OF UNKNOWN SIGN",
                   "is UNMEASURED and OF UNKNOWN SIGN"):
        assert phrase not in source, (
            f"engine/attention.py still calls the kernel bias {phrase!r}. Both "
            "arms measured it: +2.30% at rect3072 and -4.77% at rect1757."
        )
