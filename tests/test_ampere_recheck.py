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


def test_the_kernel_bias_carries_its_measurement_and_the_refuted_mechanism():
    """The bias is MEASURED now. Both halves of that must survive.

    First: the result. Arm A and Arm B on `control-rect3072-C64` gave ratio
    intervals that do not overlap, shifted +2.30%, with the FASTER kernel
    raising the ratio -- so MATH is a floor on this device.

    Second, and more easily lost: the mechanism that was asserted and lost.
    An earlier revision claimed the opposite direction from the straggler
    argument alone, with no measurement. Deleting that once the answer is known
    would leave a file that looks like it was right all along, and the next
    device-dependent question would get the same treatment.
    """
    text = RECHECK.read_text()
    for token in ("floor", "2.30%", "0.7702", "0.7880"):
        assert token in text, (
            f"the measured kernel-bias result no longer carries {token!r}; a "
            "direction without its measurement is what this item exists to stop"
        )
    assert re.search(r"CEILING|ceiling", text), (
        "the refuted straggler mechanism was deleted. Keeping only the winner "
        "makes the file read as though it was never wrong."
    )
    assert re.search(r"asserted rather than measured|asserted a direction|"
                     r"asserted direction is worse", text), (
        "the note that the earlier direction was asserted rather than measured "
        "is gone; that is the transferable part"
    )


def test_the_list_does_not_assert_a_direction_it_has_not_measured():
    """Applies to any FUTURE claim, not the one now backed by both arms."""
    text = RECHECK.read_text()
    for line in text.splitlines():
        if re.search(r"is (?:therefore )?an upper bound", line):
            assert False, (
                f"an upper bound is asserted again: {line.strip()!r}. The "
                "measured result is a FLOOR; if a ceiling is being claimed it "
                "needs the arms re-run, not a sentence."
            )


def test_claude_md_points_at_the_list():
    """A doc nothing links to is a doc nobody opens."""
    claude_md = REPO_ROOT / "CLAUDE.md"
    assert claude_md.exists()
    assert "AMPERE_RECHECK" in claude_md.read_text(), (
        "CLAUDE.md does not reference docs/AMPERE_RECHECK.md, so a new session "
        "has no reason to find it"
    )
