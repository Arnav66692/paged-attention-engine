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


# --- the Arm B attestation -------------------------------------------------


def _patched_payload(identity):
    payload = _payload([2.0] * 3, [4.0] * 3)
    payload["provenance"] = {"baseline_kv_expansion_forced": True}
    payload["baseline_token_identity"] = identity
    return payload


def test_a_patched_baseline_without_its_attestation_is_not_comparable():
    """--compare-only reads whatever files exist. This is the one that matters.

    bench.run cannot write a patched result without the gate -- the gate raises
    -- but the join step reads a directory, and a timing from a patched
    baseline nobody verified is precisely the number Arm B must not report.
    """
    for missing in (None, {}, {"token_identical": False}):
        ratio = arms.ratio_for(_patched_payload(missing))
        assert ratio["resolved"] is False, f"accepted attestation {missing!r}"
        assert "token-identity" in ratio["why"]


def test_a_patched_baseline_with_its_attestation_is_comparable():
    ratio = arms.ratio_for(_patched_payload({"token_identical": True}))
    assert ratio["resolved"] is True
    assert ratio["ratio_median"] == pytest.approx(2.0)


def test_an_unpatched_baseline_needs_no_attestation():
    """Arm A patches nothing, so demanding an attestation would void it."""
    payload = _payload([2.0] * 3, [4.0] * 3)
    payload["provenance"] = {"baseline_kv_expansion_forced": False}
    assert arms.ratio_for(payload)["resolved"] is True


# --- the Arm B patch, against the real HuggingFace code path ---------------
#
# Everything above is arithmetic on dictionaries. This section drives the actual
# `use_gqa_in_sdpa` that transformers ships, because the patch is a monkeypatch
# on a third-party module and the thing most likely to break it is an upstream
# rename, which no amount of dictionary arithmetic would notice.


@pytest.fixture(scope="module")
def tiny_generator():
    """Headline geometry, two layers. Skips if the config cannot be fetched."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from bench import baseline as baseline_mod

    try:
        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"model config unavailable offline: {type(error).__name__}")
    config.num_hidden_layers = 2
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.eval()
    model.set_attn_implementation("sdpa")
    return baseline_mod.HuggingFaceGenerator(model, None, "cpu")


def _requests(lengths, new_tokens=4):
    import torch

    from bench.workload import Request

    torch.manual_seed(1)
    return [
        Request(
            request_id=f"r{index}",
            prompt_token_ids=tuple(int(v) for v in torch.randint(1, 1000, (length,))),
            max_new_tokens=new_tokens,
        )
        for index, length in enumerate(lengths)
    ]


def test_hf_picks_its_gqa_path_from_padding_which_is_why_the_arm_exists(tiny_generator):
    """The finding that reframed this whole experiment, pinned.

    HF takes `enable_gqa` only when the attention mask is None, and the mask is
    None only when nothing was padded. So the BASELINE'S KERNEL DEPENDS ON THE
    STREAM: the control stream's prompts are all 512 tokens and take one path,
    the mixed stream's 224 distinct lengths take the other. An earlier session
    probed a uniform batch, concluded 'the baseline uses enable_gqa', and
    generalised it to both streams. It does not generalise.
    """
    from bench import baseline as baseline_mod

    uniform = baseline_mod.baseline_expands_kv_for(tiny_generator, _requests([24, 24, 24]))
    ragged = baseline_mod.baseline_expands_kv_for(tiny_generator, _requests([8, 15, 22]))

    assert uniform is False, (
        "a prompt-uniform batch no longer takes HF's enable_gqa path. The "
        "control stream's kernel has changed and the arm symmetry analysis in "
        "docs/AMPERE_RECHECK.md item 2 needs re-measuring."
    )
    assert ragged is True, (
        "a ragged batch no longer pre-expands. The mixed stream's kernel has "
        "changed; same re-measurement applies."
    )


def test_the_patch_takes_on_both_stream_shapes_and_undoes_itself(tiny_generator):
    from transformers.integrations import sdpa_attention

    from bench import baseline as baseline_mod

    original = sdpa_attention.use_gqa_in_sdpa
    undo = baseline_mod.force_baseline_kv_expansion()
    try:
        for lengths in ([24, 24, 24], [8, 15, 22]):
            assert baseline_mod.baseline_expands_kv_for(
                tiny_generator, _requests(lengths)
            ) is True, f"the patch did not take for prompt lengths {lengths}"
    finally:
        undo()
    assert sdpa_attention.use_gqa_in_sdpa is original, (
        "force_baseline_kv_expansion leaked past its undo; a later Arm A run in "
        "the same process would silently be Arm B"
    )


def test_force_expansion_raises_rather_than_silently_not_patching(monkeypatch):
    """A patch that no-ops would make Arm B a second copy of Arm A.

    That is the failure that looks like a clean result: two arms agreeing
    perfectly because they were the same arm.
    """
    from transformers.integrations import sdpa_attention

    from bench import baseline as baseline_mod

    monkeypatch.delattr(sdpa_attention, "use_gqa_in_sdpa", raising=False)
    with pytest.raises(RuntimeError, match="use_gqa_in_sdpa"):
        baseline_mod.force_baseline_kv_expansion()


def test_the_gate_refuses_a_baseline_whose_tokens_moved(tiny_generator):
    """The gate must have teeth, so give it something to reject."""
    from bench import baseline as baseline_mod

    batches = [_requests([24, 24, 24]), _requests([8, 15, 22])]
    evidence = baseline_mod.assert_token_identical(tiny_generator, batches)
    assert evidence["token_identical"] is True
    assert evidence["patch_verified_active"] is True
    assert evidence["tokens_compared"] > 0

    real = tiny_generator.generate_batch
    seen = {"calls": 0}

    def perturbed(batch):
        output = real(batch)
        seen["calls"] += 1
        if seen["calls"] > len(batches):  # only the patched half is corrupted
            key = next(iter(output))
            output[key] = (output[key][0] + 1,) + tuple(output[key][1:])
        return output

    tiny_generator.generate_batch = perturbed
    try:
        with pytest.raises(baseline_mod.BaselineNotIdentical, match="diverged at token"):
            baseline_mod.assert_token_identical(tiny_generator, batches)
    finally:
        tiny_generator.generate_batch = real
