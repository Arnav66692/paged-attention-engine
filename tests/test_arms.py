"""The two-arm kernel experiment: its reading, its comparison, and its gate.

`bench/arms.py` decides whether the Gate 1 headline ships hedged or not. Its
failure mode is not a crash -- it is reporting "not comparable" for every cell,
or "the arms agree" because it read a key that does not exist. Those look like
orderly runs. These tests exist so they cannot happen quietly.
"""

import json

import pytest

from bench import arms
from bench.run import LevelResult, SPREAD_LIMIT


def _level(engine, seconds, tokens, *, unresolved=False, valid=True):
    result = LevelResult(
        engine=engine, stream="control", concurrency=64, num_requests=256,
        throughput_valid=valid,
    )
    result.passes = list(seconds)
    result.generated_tokens = tokens
    result.unresolved = unresolved
    return result.describe()


def _payload(engine_seconds, baseline_seconds, tokens=131072, **kwargs):
    return {
        "levels": [
            _level(arms.ENGINE, engine_seconds, tokens, **kwargs),
            _level(arms.BASELINE, baseline_seconds, tokens, **kwargs),
        ]
    }


# --- the key-name trap -----------------------------------------------------


def test_ratio_reads_the_keys_describe_actually_writes():
    """The whole experiment turns on this and it was wrong once.

    `ratio_for` originally read `level["passes"]`. `describe()` writes
    `pass_seconds`. Every cell would have come back unresolved and the summary
    would have said 'no cell resolved in both arms' -- a clean-looking null
    result from a harness that never read a timing.
    """
    written = set(_level(arms.ENGINE, [1.0, 1.1, 1.2], 100))
    for required in ("pass_seconds", "generated_tokens", "median_tokens_per_second",
                     "unresolved", "throughput_valid", "engine"):
        assert required in written, (
            f"LevelResult.describe() no longer writes {required!r}, which "
            "bench/arms.py reads. Update both together."
        )

    ratio = arms.ratio_for(_payload([2.0, 2.0, 2.0], [4.0, 4.0, 4.0]))
    assert ratio["resolved"], "a perfectly ordinary result must resolve"
    assert ratio["ratio_median"] == pytest.approx(2.0)


def test_a_level_with_no_timings_does_not_silently_resolve():
    assert arms.ratio_for(_payload([], []))["resolved"] is False


# --- the interval ----------------------------------------------------------


def test_the_ratio_interval_pairs_the_extremes_the_right_way_round():
    """ratio_low must be the engine at its worst over the baseline at its best."""
    # engine: 100 tokens over 2..4s  -> 25..50 tps.  baseline: over 8..10s -> 10..12.5
    ratio = arms.ratio_for(_payload([2.0, 3.0, 4.0], [8.0, 9.0, 10.0], tokens=100))

    assert ratio["engine_tps"]["low"] == pytest.approx(25.0)
    assert ratio["engine_tps"]["high"] == pytest.approx(50.0)
    assert ratio["baseline_tps"]["low"] == pytest.approx(10.0)
    assert ratio["baseline_tps"]["high"] == pytest.approx(12.5)

    assert ratio["ratio_low"] == pytest.approx(25.0 / 12.5)
    assert ratio["ratio_high"] == pytest.approx(50.0 / 10.0)
    assert ratio["ratio_low"] < ratio["ratio_median"] < ratio["ratio_high"]


def test_an_unresolved_level_cannot_contribute_a_ratio():
    """UNRESOLVED means the harness declined to report a median. Honour that."""
    ratio = arms.ratio_for(_payload([1.0, 5.0, 9.0], [4.0] * 3, unresolved=True))
    assert ratio["resolved"] is False


def test_a_smoke_run_cannot_contribute_a_ratio():
    ratio = arms.ratio_for(_payload([2.0] * 3, [4.0] * 3, valid=False))
    assert ratio["resolved"] is False


# --- the comparison --------------------------------------------------------


def _ratio(low, median, high):
    return {"resolved": True, "ratio_low": low, "ratio_median": median,
            "ratio_high": high}


def test_overlapping_intervals_are_called_agreement_not_a_difference():
    """Two medians are never equal. Overlap is the test, not equality."""
    comparison = arms.compare_arms(
        {"cell": _ratio(1.9, 2.00, 2.1)},
        {"cell": _ratio(2.0, 2.06, 2.2)},
    )
    assert comparison["cells"]["cell"]["intervals_overlap"] is True
    assert comparison["all_comparable_cells_agree"] is True
    assert "does not bias" in comparison["verdict"]


def test_disjoint_intervals_report_the_direction_of_the_shift():
    """Arm B faster-kernel ratio ABOVE arm A means MATH was a floor."""
    comparison = arms.compare_arms(
        {"cell": _ratio(1.9, 2.0, 2.1)},
        {"cell": _ratio(3.0, 3.2, 3.4)},
    )
    cell = comparison["cells"]["cell"]
    assert cell["intervals_overlap"] is False
    assert "floor" in cell["direction"]
    assert cell["median_shift_b_over_a"] == pytest.approx(1.6)
    assert "DOES move" in comparison["verdict"]


def test_the_other_direction_is_reported_as_a_ceiling():
    comparison = arms.compare_arms(
        {"cell": _ratio(3.0, 3.2, 3.4)},
        {"cell": _ratio(1.9, 2.0, 2.1)},
    )
    assert "ceiling" in comparison["cells"]["cell"]["direction"]


def test_one_cell_disagreeing_is_enough_to_deny_blanket_agreement():
    comparison = arms.compare_arms(
        {"a": _ratio(1.9, 2.0, 2.1), "b": _ratio(1.9, 2.0, 2.1)},
        {"a": _ratio(1.9, 2.0, 2.1), "b": _ratio(5.0, 5.2, 5.4)},
    )
    assert comparison["all_comparable_cells_agree"] is False


def test_no_comparable_cell_is_undetermined_and_never_agreement():
    """The vacuous case must not read as a clean pass."""
    comparison = arms.compare_arms(
        {"cell": {"resolved": False}}, {"cell": {"resolved": False}},
    )
    assert comparison["all_comparable_cells_agree"] is False
    assert comparison["num_comparable"] == 0
    assert "UNDETERMINED" in comparison["verdict"]


# --- the matrix ------------------------------------------------------------


def test_the_matrix_covers_both_headlines_and_both_rectangle_sizings():
    from bench import baseline as baseline_mod
    from bench.workload import PRESSURE_C

    names = {arms.cell_name(cell) for cell in arms.CELLS}
    assert len(arms.CELLS) == 2 * len(baseline_mod.RECTANGLE_SIZINGS)
    for stream in ("control", "mixed"):
        for rectangle in baseline_mod.RECTANGLE_SIZINGS:
            assert f"{stream}-rect{rectangle}-C{PRESSURE_C}" in names

    # The control headline is the quotable number; it must be requested as one.
    control = [c for c in arms.CELLS if c["stream"] == "control"]
    assert control and all(c["headline"] for c in control)
    assert all(c["concurrency"] == PRESSURE_C for c in arms.CELLS), (
        "the control headline is only legal at PRESSURE_C and both arms must "
        "sit at the same level or the two are not comparable"
    )


def test_the_two_arms_differ_in_exactly_the_two_knobs_under_test():
    """An arm that changed anything else would confound the comparison."""
    a, b = arms.ARMS["A"], arms.ARMS["B"]
    differing = {key for key in set(a) | set(b)
                 if key != "intent" and a.get(key) != b.get(key)}
    assert differing == {"gqa_path", "baseline_expand_kv"}, (
        f"the arms differ in {differing}; they may differ only in which kernel "
        "each side runs, or the comparison measures something else too"
    )
    assert a["baseline_expand_kv"] is False and b["baseline_expand_kv"] is True
    assert {a["gqa_path"], b["gqa_path"]} == {"sdpa", "repeat_kv"}


def test_spread_limit_is_shared_with_the_harness_not_redeclared():
    """arms.py must not carry its own idea of what counts as resolved."""
    source = (arms.__file__)
    text = open(source).read()
    assert "SPREAD_LIMIT" not in text.replace("bench.run", ""), (
        "bench/arms.py declares or imports its own spread limit. Resolution is "
        "the harness's decision and arms.py must inherit it, not restate it."
    )
    assert SPREAD_LIMIT == 0.20
