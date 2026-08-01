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
    # Measured in session3/session4. Both are facts about which backend
    # `enable_gqa` selects, so both move on Ampere.
    "baseline_backend_matches": r"baseline_expands_kv",
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


def test_the_list_states_which_way_the_sm75_kernel_biases_the_headline():
    """The direction of a known bias is the part that must not go missing.

    Both sides run MATH, so the comparison is symmetric. Symmetric is not
    unbiased: MATH is expensive per unit of attention work and the baseline does
    more attention work per useful token because it pads, so the slow kernel
    amplifies the baseline's waste. The Gate 1 ratio on sm_75 is an UPPER bound.

    A reader who takes the number without this reads a ceiling as an estimate.
    """
    text = RECHECK.read_text()
    assert re.search(r"upper bound|UPPER bound", text), (
        "the list no longer says the sm_75 headline is an upper bound on the "
        "paging advantage. That is the direction of a known, unquantified bias; "
        "without it the number reads as unbiased."
    )
    assert re.search(r"unmeasured|not.{0,30}quantif", text), (
        "the size of the amplification is unmeasured and the list must say so "
        "rather than implying it is small"
    )


def test_claude_md_points_at_the_list():
    """A doc nothing links to is a doc nobody opens."""
    claude_md = REPO_ROOT / "CLAUDE.md"
    assert claude_md.exists()
    assert "AMPERE_RECHECK" in claude_md.read_text(), (
        "CLAUDE.md does not reference docs/AMPERE_RECHECK.md, so a new session "
        "has no reason to find it"
    )
