"""The `--fast` selection is a filter, not a different suite.

A development shortcut is only safe if it cannot quietly become a weaker gate.
Two things have to stay true, and neither is obvious from reading conftest.py:

  * `--fast` only ever *removes* items. It never adds one, never renames one, and
    never changes what a surviving test asserts.
  * The gate is unaffected by the flag existing at all.

Both are checked here by collecting the suite twice in a subprocess and
comparing the node id sets, which is the only way to observe a collection-time
hook from inside a test.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import conftest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _collect(*extra: str) -> set[str]:
    """Node ids pytest would run, with no marker filtering at all."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "", *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(" ")
    }


@pytest.fixture(scope="module")
def collections():
    return _collect(), _collect("--fast")


def test_fast_is_a_strict_subset_of_the_gate(collections):
    """Nothing runs under --fast that would not run under the gate."""
    gate, fast = collections
    assert fast, "the fast selection collected nothing"
    assert fast < gate, (
        "the fast selection is not a strict subset of the full suite: "
        f"{sorted(fast - gate)[:5]} would run only under --fast"
    )
    print(f"\nfast selection: {len(fast)} of {len(gate)} items "
          f"({len(gate) - len(fast)} deselected)")


def test_fast_drops_only_the_other_device_and_the_other_prompt_lengths(collections):
    """Every dropped item is dropped for one of the two declared reasons.

    If `--fast` ever started dropping something for a third reason -- a whole
    module, an unparametrized test -- this fails, because that would make it a
    different suite rather than a narrower view of the same one.
    """
    gate, fast = collections
    dropped = gate - fast

    other_lengths = {13, 32}
    unexplained = []
    for node_id in dropped:
        params = node_id.partition("[")[2].rstrip("]").split("-") if "[" in node_id else []
        by_device = any(part not in (conftest.FAST_DEVICE,) and part == "mps" for part in params)
        by_length = any(part.isdigit() and int(part) in other_lengths for part in params)
        if not (by_device or by_length):
            unexplained.append(node_id)

    print(f"dropped {len(dropped)} items, all explained by "
          f"device != {conftest.FAST_DEVICE} or prompt_len != {conftest.FAST_PROMPT_LEN}")
    assert not unexplained, (
        f"--fast dropped items for neither declared reason: {sorted(unexplained)[:5]}"
    )


def test_fast_keeps_at_least_one_model_tier_test_per_module(collections):
    """A fast run that skipped a whole module would prove nothing about it."""
    _, fast = collections
    for module in ("test_parity.py", "test_sched.py", "test_fused.py"):
        kept = [node for node in fast if module in node and conftest.FAST_DEVICE in node]
        assert kept, f"--fast collected no {conftest.FAST_DEVICE} test from {module}"


def test_fast_selection_constants_mean_what_conftest_claims():
    """33 is one past a block boundary; cpu is the unconditional device."""
    from engine.cache import BLOCK_SIZE

    assert conftest.FAST_PROMPT_LEN % BLOCK_SIZE == 1, (
        f"FAST_PROMPT_LEN {conftest.FAST_PROMPT_LEN} is no longer one token past "
        f"a {BLOCK_SIZE}-token block boundary, so it is no longer the hardest of "
        "the fixture lengths"
    )
    assert conftest.FAST_DEVICE == "cpu"
