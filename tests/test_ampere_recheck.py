"""The Ampere re-measurement list must exist and must stay complete.

Every measurement in this repo was taken on a Kaggle T4, sm_75. Several results
are properties of that device: TF32 needs sm_80 so the pin is untested, FLASH
rejects every shape so SDPA backend selection is not what it will be on Ampere.
A future run on Modal or Lambda -- both Ampere or newer by default -- would
otherwise inherit those numbers without knowing they need re-checking.

A markdown file nobody is required to read is a file that gets missed. These
tests make dropping an item a test failure.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RECHECK = REPO_ROOT / "docs" / "AMPERE_RECHECK.md"

# Each item must be findable by a phrase that cannot be removed without also
# removing the item. Deliberately not headings -- headings get renamed.
REQUIRED_ITEMS = {
    "tf32_pin": r"TF32 requires sm_80|TF32 pin",
    "sdpa_backend_selection": r"SDPA backend selection",
    "flash_unavailable_on_sm75": r"FLASH_ATTENTION and CUDNN_ATTENTION reject",
    "expandable_segments": r"expandable_segments",
    "fp16_tolerance": r"FP16_LOGIT_ATOL",
    "memory_caps": r"max_num_batched_tokens",
    # Measured in session3/session4/session5. All are facts about which backend
    # `enable_gqa` selects, so all move on Ampere.
    #
    # The baseline's kernel depends on whether the stream's prompts are UNIFORM,
    # because HF only takes enable_gqa when the attention mask is None and the
    # mask is None only when nothing was padded. Session 3 probed a uniform
    # batch and the conclusion was generalised to both streams; it does not
    # generalise. If this item goes missing, that mistake is available again.
    "baseline_kernel_is_stream_dependent": r"use_gqa_in_sdpa",
    "symmetry_is_not_neutrality": r"[Ss]ymmetry is not neutrality|"
                                  r"[Ss]ymmetry.{0,40}not.{0,20}neutral",
    "enable_gqa_memory_claim": r"memory saving does not exist",
}


def test_the_ampere_recheck_list_exists():
    assert RECHECK.exists(), (
        f"{RECHECK} is missing. Every measurement in this repo was taken on "
        "sm_75 and several are device-dependent; the list of what must be "
        "re-measured is not optional."
    )


@pytest.mark.parametrize("item,pattern", sorted(REQUIRED_ITEMS.items()))
def test_every_device_dependent_result_is_listed(item, pattern):
    text = RECHECK.read_text()
    assert re.search(pattern, text), (
        f"the Ampere re-check list no longer mentions {item!r} "
        f"(pattern {pattern!r}). If it was genuinely resolved, remove it from "
        "REQUIRED_ITEMS in this test in the same commit and say why."
    )


def test_the_list_names_the_actual_architectures():
    """sm_75 vs sm_80 is the whole point; a list that omits them is unusable."""
    text = RECHECK.read_text()
    for token in ("sm_75", "sm_80", "sm_86", "T4"):
        assert token in text, f"{token} not named in the re-check list"


def test_the_list_says_the_tolerance_is_not_to_be_tightened():
    """The rule travels with the item, not just in a chat message."""
    text = RECHECK.read_text()
    assert re.search(r"not to be tightened|NOT to be tightened", text), (
        "the list must carry the rule that FP16_LOGIT_ATOL is not adjusted to "
        "suit a measurement, or a future run will quietly fit it"
    )


def test_the_list_records_that_the_two_streams_get_different_baseline_kernels():
    """The asymmetry that was invisible until it was measured.

    On `control` both sides run MATH. On `mixed` the baseline gets
    EFFICIENT_ATTENTION and the engine stays on MATH -- an asymmetry AGAINST the
    engine. A summary that keeps only the control row reads as "symmetric".
    """
    text = RECHECK.read_text()
    for token in ("control", "mixed", "EFFICIENT_ATTENTION", "MATH"):
        assert token in text, f"{token} missing from the backend-selection item"
    assert re.search(r"DEPENDS ON THE STREAM|depends on the stream", text), (
        "the list no longer says the baseline's kernel is stream-dependent, "
        "which is the fact that invalidated the earlier 'they match' reading"
    )


def test_the_kernel_bias_is_recorded_as_unmeasured_and_unsigned():
    """An asserted direction is worse than an admitted gap.

    A previous revision of the list claimed the sm_75 headline was an UPPER
    bound, on the theory that a slow kernel amplifies the baseline's padding
    waste. That was asserted, not measured, and it ignored the opposing
    mechanism: in `ratio = (B_e/B_b)(W + B_b·k)/(W + B_e·k)`, raising the
    per-token cost `k` for both sides drives the ratio toward 1, which would
    make the same number a FLOOR.

    Both mechanisms are real. The rectangle sizing does not settle it -- that is
    a cache allocation and HF masks over the true kv_len -- but static-batch
    stragglers are genuine wasted per-token work. Sign unknown until the two
    arms are run.

    This test fails if anyone reintroduces a direction without a measurement.
    """
    text = RECHECK.read_text()
    assert re.search(r"UNKNOWN SIGN|unknown sign", text), (
        "the list must state that the kernel's effect on the ratio is of "
        "unknown sign until both arms are measured"
    )
    assert re.search(r"UNMEASURED|unmeasured", text), (
        "the list must state the bias is unmeasured rather than implying a size"
    )
    for mechanism in (r"FLOOR|floor", r"CEILING|ceiling"):
        assert re.search(mechanism, text), (
            f"the list dropped one of the two opposing mechanisms ({mechanism}); "
            "keeping only one is how an asserted direction gets back in"
        )
    unhedged = re.search(r"is (?:therefore )?an upper bound", text)
    assert unhedged is None, (
        "the list asserts an upper bound again. That direction is exactly what "
        "arms A and B exist to measure; it may be written down only with the "
        "measurement that establishes it."
    )


def test_claude_md_points_at_the_list():
    """A doc nothing links to is a doc nobody opens."""
    claude_md = REPO_ROOT / "CLAUDE.md"
    assert claude_md.exists()
    assert "AMPERE_RECHECK" in claude_md.read_text(), (
        "CLAUDE.md does not reference docs/AMPERE_RECHECK.md, so a new session "
        "has no reason to find it"
    )
