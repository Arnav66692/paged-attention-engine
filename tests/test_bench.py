"""Step 6 harness, smoke-tested at tiny scale. NO THROUGHPUT NUMBER IS PRODUCED.

This machine is an M4 Mac whose two devices showed 76% run-to-run spread on the
Step 5b diagnostic. It cannot measure anything, so nothing here measures.
Everything that would otherwise need a stopwatch is driven by an injected clock
returning scripted values, which makes the median, the range, the escalation rule
and the UNRESOLVED verdict all deterministic and checkable without timing a
single forward pass. The one test that does run a real model runs it to check
that tokens come out, and throws the elapsed time away.

`bench/run.py` also refuses to write a result file off a CUDA host unless
`--smoke` is passed, and stamps every smoke record `throughput_valid: false`.
That is tested here too, because the rule is only worth having if it cannot be
forgotten.
"""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from bench import baseline as baseline_mod
from bench import run as run_mod
from bench import workload
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS, PRESSURE_C
from engine.cache import ADMISSION_MARGIN_BLOCKS, BlockAllocator
from engine.scheduler import ExecutionMode, is_admissible
from tests.tiny_model import tiny_engine

REPO_ROOT = Path(__file__).resolve().parent.parent

# The ceiling the Gate 1 spec fixes. Derived in bench/baseline.py from the model
# geometry and the locked constants; restated here so the derivation is checked
# against the number rather than trusted.
KV_POOL_BYTES = 1_644_167_168

# bench/workload.py is frozen, so its content is pinned. See
# `test_bench_workload_is_still_frozen` for why a hash and not a commit count.
FROZEN_WORKLOAD_SHA256 = (
    "bfa3bb5134b11b073802a776dce0aa7d1449213dc8eaf9cefcbf303a93d785cb"
)


# --- the ceiling and the sizings -------------------------------------------


def test_byte_ceiling_is_what_the_locked_constants_imply():
    """1,644,167,168 is not a magic number; it is 3584 x 16 x 28672."""
    assert baseline_mod.BYTES_PER_TOKEN == 2 * 28 * 2 * 128 * 2 == 28_672
    assert baseline_mod.BYTES_PER_BLOCK == 28_672 * BLOCK_SIZE == 458_752
    assert baseline_mod.KV_POOL_TOKEN_SLOTS == NUM_GPU_BLOCKS * BLOCK_SIZE == 57_344
    assert baseline_mod.KV_POOL_BYTES == KV_POOL_BYTES

    print(f"\nbytes/token {baseline_mod.BYTES_PER_TOKEN}, "
          f"bytes/block {baseline_mod.BYTES_PER_BLOCK}, "
          f"pool {baseline_mod.KV_POOL_BYTES} bytes "
          f"({baseline_mod.KV_POOL_BYTES / 1e9:.3f} GB)")


@pytest.mark.parametrize("rectangle", baseline_mod.RECTANGLE_SIZINGS)
def test_both_rectangle_sizings_are_runnable(rectangle):
    """Distribution max and P95 both have to fit, and neither may be free."""
    resident = baseline_mod.max_resident(rectangle)
    used = baseline_mod.rectangle_bytes_used(rectangle)
    print(f"\nrectangle {rectangle} tok -> {resident} resident, "
          f"{used} of {KV_POOL_BYTES} bytes addressable "
          f"({(KV_POOL_BYTES - used) / KV_POOL_BYTES:.2%} stranded)")

    assert resident >= 1
    assert used <= KV_POOL_BYTES
    assert resident * rectangle <= baseline_mod.KV_POOL_TOKEN_SLOTS


def test_the_p95_sizing_buys_residents_and_is_paid_for_in_rejections():
    """The two sizings are a trade, and the harness has to show both halves."""
    wide = baseline_mod.max_resident(baseline_mod.RECTANGLE_DISTRIBUTION_MAX)
    narrow = baseline_mod.max_resident(baseline_mod.RECTANGLE_P95)
    assert narrow > wide, "the P95 sizing must hold more sequences, or it is pointless"

    stream = workload.build_stream("mixed", seed=0, c=8)
    _, rejected_wide = baseline_mod.classify(
        stream, baseline_mod.RECTANGLE_DISTRIBUTION_MAX
    )
    _, rejected_narrow = baseline_mod.classify(stream, baseline_mod.RECTANGLE_P95)

    print(f"\ndistribution max {baseline_mod.RECTANGLE_DISTRIBUTION_MAX}: "
          f"{wide} resident, {len(rejected_wide)}/{len(stream)} rejected")
    print(f"P95 {baseline_mod.RECTANGLE_P95}: "
          f"{narrow} resident, {len(rejected_narrow)}/{len(stream)} rejected")

    # Nothing in the distribution can exceed its own maximum.
    assert not rejected_wide
    # And the P95 sizing's extra residents are paid for in dropped work.
    assert rejected_narrow, (
        "the P95 rectangle rejected nothing, so it is not actually the P95 and "
        "its residency advantage would be free"
    )


def test_baseline_rejects_identically_to_the_engine():
    """Not 'the same rule', the same function.

    A request the scheduler's guard turns away must be turned away here for the
    same reason, and the harness has to be able to say which of the two filters
    fired.
    """
    allocator = BlockAllocator(num_blocks=NUM_GPU_BLOCKS, block_size=BLOCK_SIZE)
    limit_tokens = (NUM_GPU_BLOCKS - ADMISSION_MARGIN_BLOCKS) * BLOCK_SIZE

    monstrous = workload.Request("monster", tuple(range(8)), limit_tokens)
    ordinary = workload.Request("ordinary", tuple(range(8)), 8)
    over_rectangle = workload.Request("wide", tuple(range(4000)), 8)

    assert not is_admissible(monstrous.total_len, allocator, ADMISSION_MARGIN_BLOCKS)
    assert is_admissible(ordinary.total_len, allocator, ADMISSION_MARGIN_BLOCKS)

    accepted, rejected = baseline_mod.classify(
        [monstrous, ordinary, over_rectangle], baseline_mod.RECTANGLE_P95
    )
    reasons = {r.request_id: r.reason for r in rejected}
    print(f"\nrejections attributed: {reasons}")

    assert [r.request_id for r in accepted] == ["ordinary"]
    assert reasons["monster"] == baseline_mod.RejectionReason.ENGINE_GUARD
    assert reasons["wide"] == baseline_mod.RejectionReason.OVER_RECTANGLE


# --- the baseline loop, with a fake generator ------------------------------


class _FakeGenerator:
    """Returns the right shape instantly. This tier is accounting, not model work."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate_batch(self, batch):
        self.batch_sizes.append(len(batch))
        return {
            request.request_id: tuple(range(request.max_new_tokens)) for request in batch
        }


def test_baseline_batches_to_max_resident_and_reports_its_waste():
    """Static batching, and the slot waste that is the reason for the whole gate."""
    stream = workload.build_stream("control", seed=0, c=4)
    generator = _FakeGenerator()
    ticks = iter([0.0, 2.5])

    result = baseline_mod.run_baseline(
        stream, generator, baseline_mod.RECTANGLE_P95, clock=lambda: next(ticks)
    )
    resident = baseline_mod.max_resident(baseline_mod.RECTANGLE_P95)

    print(f"\nbaseline: {result.num_accepted} accepted in {result.num_batches} batches "
          f"of at most {resident}; slot waste {result.slot_waste:.1%}")
    print(f"  rejection rate {result.rejection_rate:.2%}")

    assert all(size <= resident for size in generator.batch_sizes)
    assert sum(generator.batch_sizes) == result.num_accepted
    assert result.num_batches == len(generator.batch_sizes)
    assert result.generated_tokens == sum(
        request.max_new_tokens for request in stream
        if request.total_len <= baseline_mod.RECTANGLE_P95
    )
    # The rectangle charges for slots nobody occupies. That is the point.
    assert 0.0 < result.slot_waste < 1.0
    assert result.elapsed_seconds == 2.5


def test_baseline_result_records_the_rejection_rate():
    stream = workload.build_stream("mixed", seed=0, c=8)
    ticks = iter([0.0, 1.0])
    result = baseline_mod.run_baseline(
        stream, _FakeGenerator(), baseline_mod.RECTANGLE_P95, clock=lambda: next(ticks)
    )
    record = result.describe()
    print(f"\nrecorded: rejection_rate={record['rejection_rate']}, "
          f"over_rectangle={record['rejected_over_rectangle']}, "
          f"engine_guard={record['rejected_by_engine_guard']}")

    assert record["rejection_rate"] > 0
    assert record["num_rejected"] == (
        record["rejected_over_rectangle"] + record["rejected_by_engine_guard"]
    )
    assert record["num_accepted"] + record["num_rejected"] == record["num_offered"]


# --- batch order: the only thing sorting is allowed to change ---------------


class _OrderRecordingGenerator(_FakeGenerator):
    """Also keeps the request ids of each batch, in the order they were cut."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[tuple[str, ...]] = []

    def generate_batch(self, batch):
        self.batches.append(tuple(request.request_id for request in batch))
        return super().generate_batch(batch)


def _run_both_orders(stream, rectangle):
    """The same stream under both orders, with every other constraint held."""
    runs = {}
    for order in baseline_mod.BATCH_ORDERS:
        generator = _OrderRecordingGenerator()
        ticks = iter([0.0, 1.0])
        runs[order] = (
            baseline_mod.run_baseline(
                stream, generator, rectangle,
                max_num_seqs=64, batch_order=order, clock=lambda: next(ticks),
            ),
            generator,
        )
    return runs


def test_sorting_changes_batch_composition_and_nothing_else():
    """The invariant the sorted measurement rests on.

    Same requests completed, same token count for every one of them, same
    acceptance, same rejection, same rectangle accounting. The only difference
    permitted is which requests shared a batch and in what order.
    """
    stream = workload.build_stream("mixed", seed=0, c=64)
    runs = _run_both_orders(stream, baseline_mod.RECTANGLE_P95)
    arrival, arrival_gen = runs[baseline_mod.BATCH_ORDER_ARRIVAL]
    sorted_, sorted_gen = runs[baseline_mod.BATCH_ORDER_SORTED]

    print(f"\naccepted {arrival.num_accepted}, rejected {arrival.num_rejected}, "
          f"{arrival.num_batches} batches either way")
    print(f"  batch compositions differ: {arrival_gen.batches != sorted_gen.batches}")

    # 1. The same request identities completed.
    assert set(arrival.outputs) == set(sorted_.outputs)
    assert len(arrival.outputs) == arrival.num_accepted

    # 2. The same token count for every completed request.
    assert {rid: len(tokens) for rid, tokens in arrival.outputs.items()} == {
        rid: len(tokens) for rid, tokens in sorted_.outputs.items()
    }

    # 3. Every conserved quantity is identical. Acceptance, rejection and its
    #    attribution, the token count, the rectangle accounting and the batch
    #    count are all computed over `accepted`, so none of them may move.
    #    Excluded are the three counters that ARE the order's consequence, and
    #    which are checked for direction below, plus the injected clock.
    order_derived = {
        "batch_order", "elapsed_seconds",
        "forward_passes", "decode_row_steps", "decode_step_efficiency",
    }
    left = {k: v for k, v in arrival.describe().items() if k not in order_derived}
    right = {k: v for k, v in sorted_.describe().items() if k not in order_derived}
    assert left == right, f"sorting moved more than the order: {left} vs {right}"
    assert arrival.describe()["batch_order"] == baseline_mod.BATCH_ORDER_ARRIVAL
    assert sorted_.describe()["batch_order"] == baseline_mod.BATCH_ORDER_SORTED

    # 3b. The counters that may move are the ones grouping is supposed to move,
    #     and they move the only way grouping can: fewer wasted decode steps for
    #     the identical set of kept tokens.
    print(f"  forward passes {arrival.forward_passes} -> {sorted_.forward_passes}; "
          f"decode-step efficiency {arrival.decode_step_efficiency:.4f} -> "
          f"{sorted_.decode_step_efficiency:.4f}")
    assert sorted_.forward_passes < arrival.forward_passes
    assert sorted_.decode_row_steps < arrival.decode_row_steps
    assert sorted_.decode_step_efficiency > arrival.decode_step_efficiency
    assert arrival.generated_tokens == sorted_.generated_tokens
    # Efficiency cannot exceed 1: you never keep more than you computed.
    assert sorted_.decode_step_efficiency <= 1.0

    # 4. The difference is real: the batches were actually cut differently, and
    #    the sorted run's batches are decode-homogeneous where arrival's are not.
    assert arrival_gen.batches != sorted_gen.batches
    assert sorted(sum(arrival_gen.batches, ())) == sorted(sum(sorted_gen.batches, ()))
    assert arrival_gen.batch_sizes == sorted_gen.batch_sizes


def test_sorting_does_not_touch_the_workload_stream():
    """`bench/workload.py` is frozen; the sort must not reach back into it."""
    stream = workload.build_stream("mixed", seed=0, c=32)
    before = workload.stream_digest(stream)
    ids_before = [request.request_id for request in stream]

    baseline_mod.run_baseline(
        stream, _FakeGenerator(), baseline_mod.RECTANGLE_P95,
        max_num_seqs=32, batch_order=baseline_mod.BATCH_ORDER_SORTED,
        clock=iter([0.0, 1.0]).__next__,
    )

    print(f"\nstream digest unchanged: {workload.stream_digest(stream) == before}")
    assert workload.stream_digest(stream) == before
    assert [request.request_id for request in stream] == ids_before


def test_arrival_is_the_default_and_is_the_original_batching():
    """The default must reproduce the batching every committed result used."""
    stream = workload.build_stream("mixed", seed=0, c=32)
    accepted, _ = baseline_mod.classify(stream, baseline_mod.RECTANGLE_P95)

    default_gen = _OrderRecordingGenerator()
    baseline_mod.run_baseline(
        stream, default_gen, baseline_mod.RECTANGLE_P95, max_num_seqs=32,
        clock=iter([0.0, 1.0]).__next__,
    )
    resident = min(baseline_mod.max_resident(baseline_mod.RECTANGLE_P95), 32)
    expected = [
        tuple(r.request_id for r in accepted[i : i + resident])
        for i in range(0, len(accepted), resident)
    ]

    print(f"\ndefault cut {len(default_gen.batches)} batches in arrival order")
    assert default_gen.batches == expected


def test_sorted_is_descending_by_decode_budget_and_stable():
    stream = workload.build_stream("mixed", seed=0, c=32)
    accepted, _ = baseline_mod.classify(stream, baseline_mod.RECTANGLE_P95)
    ordered = baseline_mod.order_for_batching(
        accepted, baseline_mod.BATCH_ORDER_SORTED
    )
    budgets = [request.max_new_tokens for request in ordered]

    print(f"\nsorted budgets: {budgets[:6]} ... {budgets[-6:]}")
    assert budgets == sorted(budgets, reverse=True)
    assert sorted(r.request_id for r in ordered) == sorted(
        r.request_id for r in accepted
    )
    # Stable: equal budgets keep arrival order, so the order is a deterministic
    # function of the stream rather than of the sort implementation.
    arrival_index = {r.request_id: i for i, r in enumerate(accepted)}
    for left, right in zip(ordered, ordered[1:]):
        if left.max_new_tokens == right.max_new_tokens:
            assert arrival_index[left.request_id] < arrival_index[right.request_id]


def test_unknown_batch_order_raises_rather_than_defaulting():
    stream = workload.build_stream("mixed", seed=0, c=8)
    with pytest.raises(ValueError, match="unknown batch_order"):
        baseline_mod.run_baseline(
            stream, _FakeGenerator(), baseline_mod.RECTANGLE_P95,
            batch_order="soretd", clock=iter([0.0, 1.0]).__next__,
        )


def test_session10_kernels_declare_the_batch_order_comparison_they_promise():
    """The batch-order session must be Arm B, both orders, and the two mixed cells.

    Same reason as the session6/session7 slice check: the kernels declare their
    work as literals because Kaggle renames an uploaded script to script.py, so
    the literals have to be checked against the tables they must agree with, in
    the files that own those tables.
    """
    import ast

    from bench.arms import ARMS

    repo_root = Path(__file__).resolve().parent.parent
    tables, declared, orders, paths = {}, {}, {}, {}
    for candidate in sorted((repo_root / "kaggle").glob("session10?/session10?.py")):
        for node in ast.walk(ast.parse(candidate.read_text())):
            if not isinstance(node, ast.Assign):
                continue
            name = getattr(node.targets[0], "id", "")
            if name == "SLICES":
                tables[candidate.stem] = ast.literal_eval(node.value)
            elif name == "SLICE":
                declared[candidate.stem] = ast.literal_eval(node.value)
            elif name == "ORDERS":
                orders[candidate.stem] = ast.literal_eval(node.value)
            elif name == "GQA_PATH":
                paths[candidate.stem] = ast.literal_eval(node.value)
    if not tables:
        pytest.skip("no session10 kernels in this tree")

    print(f"\nsession10 slices: {sorted(declared)}")
    first = next(iter(tables.values()))
    for stem, table in tables.items():
        assert table == first, f"kaggle/{stem} carries a different slice table"
        assert declared.get(stem) == stem, (
            f"kaggle/{stem}/{stem}.py declares SLICE={declared.get(stem)!r} and "
            "would measure another slice's cell"
        )
        # Arm B, because that is the arm the mixed headline is quoted from and
        # the one where both sides run the same kernel.
        assert paths[stem] == ARMS["B"]["gqa_path"], (
            f"kaggle/{stem} declares GQA_PATH={paths[stem]!r}, not arm B's "
            f"{ARMS['B']['gqa_path']!r}"
        )
        # Both orders, or the session measures one of them against nothing.
        assert tuple(orders[stem]) == tuple(baseline_mod.BATCH_ORDERS), (
            f"kaggle/{stem} runs {orders[stem]}, not {list(baseline_mod.BATCH_ORDERS)}"
        )
    assert set(declared) == set(first)

    # The two cells are the mixed Arm B headline cells, at the sweep's own
    # pressure level and both rectangle sizings.
    assert {(c["stream"], c["rectangle"], c["concurrency"]) for c in first.values()} == {
        ("mixed", rectangle, PRESSURE_C)
        for rectangle in baseline_mod.RECTANGLE_SIZINGS
    }


def test_provenance_records_the_batch_order():
    """It travels with the result, because a static tok/s means nothing without it."""
    sorted_record = asdict(run_mod.Provenance.collect(
        baseline_mod.BENCH_MODEL, "torch.float16", "digest",
        baseline_batch_order=baseline_mod.BATCH_ORDER_SORTED,
    ))
    default_record = asdict(run_mod.Provenance.collect(
        baseline_mod.BENCH_MODEL, "torch.float16", "digest",
    ))

    print(f"\nprovenance baseline_batch_order: "
          f"explicit={sorted_record['baseline_batch_order']}, "
          f"default={default_record['baseline_batch_order']}")
    assert sorted_record["baseline_batch_order"] == baseline_mod.BATCH_ORDER_SORTED
    assert default_record["baseline_batch_order"] == baseline_mod.BATCH_ORDER_ARRIVAL


# --- passes, median, range, escalation -------------------------------------


def _scripted_level(elapsed_per_pass, **kwargs):
    """Drive measure_level with a scripted sequence of pass durations."""
    values = iter(list(elapsed_per_pass))

    def run_pass():
        return run_mod.EnginePass(
            elapsed=next(values), generated_tokens=1000, preemptions=7,
            num_rejected=0, peak_blocks_used=100,
        )

    stream = workload.build_stream("control", seed=0, c=1)
    return run_mod.measure_level(
        run_mod.ENGINE_PAGED, "control", 1, stream, run_pass, log=lambda _: None, **kwargs
    )


def test_a_warmup_pass_is_discarded_and_three_are_measured():
    # warmup, then three. The 99.0 is the warmup and must not appear.
    level = _scripted_level([99.0, 1.00, 1.02, 1.04])
    print(f"\npasses recorded: {level.passes}")
    assert len(level.passes) == run_mod.MEASURED_PASSES
    assert 99.0 not in level.passes
    assert level.median_seconds == 1.02
    assert level.range_seconds == pytest.approx(0.04)
    assert not level.unresolved


def test_a_tight_level_does_not_escalate():
    level = _scripted_level([9.0, 1.00, 1.05, 1.10])
    print(f"\nspread {level.spread:.1%} of median, {len(level.passes)} passes")
    assert level.spread <= run_mod.SPREAD_LIMIT
    assert len(level.passes) == run_mod.MEASURED_PASSES
    assert level.median_tokens_per_second is not None


def test_a_wide_level_escalates_to_five_passes_and_reports_both():
    """Range > 20% of median -> rerun at five, and both numbers survive.

    One warmup for the level, not one per attempt: the escalated rerun is five
    measured passes on an already-warm device.
    """
    # warmup, then 3 passes spreading 40%, then the 5 that spread only 10%.
    level = _scripted_level(
        [9.0] + [1.0, 1.2, 1.4] + [1.00, 1.02, 1.05, 1.08, 1.10]
    )
    print(f"\nafter escalation: {len(level.passes)} passes, "
          f"median {level.median_seconds:.3f}s, range {level.range_seconds:.3f}s "
          f"({level.spread:.1%})")

    assert len(level.passes) == run_mod.ESCALATED_PASSES
    assert level.spread <= run_mod.SPREAD_LIMIT
    assert not level.unresolved
    record = level.describe()
    assert record["num_passes"] == run_mod.ESCALATED_PASSES
    assert record["range_seconds"] is not None
    assert record["median_tokens_per_second"] is not None


def test_a_level_that_stays_wide_is_UNRESOLVED_and_reports_no_median():
    """The verdict that matters: a wide level yields no throughput number at all."""
    level = _scripted_level(
        [9.0] + [1.0, 1.5, 2.0] + [1.0, 1.3, 1.6, 1.9, 2.2]
    )
    record = level.describe()
    print(f"\nUNRESOLVED: spread {level.spread:.1%} over {len(level.passes)} passes, "
          f"median_tokens_per_second={record['median_tokens_per_second']}")

    assert level.unresolved
    assert len(level.passes) == run_mod.ESCALATED_PASSES
    assert level.median_tokens_per_second is None
    assert record["median_tokens_per_second"] is None
    # The evidence is still reported. It is the conclusion that is withheld.
    assert record["range_seconds"] is not None
    assert record["spread_fraction"] > run_mod.SPREAD_LIMIT


# --- the guards ------------------------------------------------------------


def test_pressure_c_is_the_oversubscribed_level_the_guard_protects():
    """PRESSURE_C is 64 and its shortfall is 12.5%, from the frozen workload."""
    demand = PRESSURE_C * workload.CONTROL_BLOCKS_PER_REQUEST
    shortfall = (demand - NUM_GPU_BLOCKS) / demand
    print(f"\nPRESSURE_C={PRESSURE_C}: {demand} blocks demanded vs "
          f"{NUM_GPU_BLOCKS} pool, {shortfall:.1%} shortfall")
    assert PRESSURE_C == 64
    assert demand == 4096
    assert shortfall == pytest.approx(0.125)


def test_control_headline_refuses_any_level_other_than_pressure_c():
    run_mod.check_control_headline_level("control", PRESSURE_C, headline=True)

    for level in (1, 32, 63, 65, 128):
        with pytest.raises(run_mod.InvalidRun, match="only defined at PRESSURE_C"):
            run_mod.check_control_headline_level("control", level, headline=True)

    # The sweep itself is not the headline, so it may visit any level.
    run_mod.check_control_headline_level("control", 32, headline=False)
    run_mod.check_control_headline_level("mixed", 32, headline=True)


def test_a_c64_run_with_zero_preemptions_is_invalid_and_raises():
    """Fails loudly. It does not warn and continue."""
    level = run_mod.LevelResult(
        engine=run_mod.ENGINE_PAGED, stream="control", concurrency=PRESSURE_C,
        num_requests=256, passes=[1.0, 1.0, 1.0], preemption_count=0,
    )
    with pytest.raises(run_mod.InvalidRun, match="ZERO preemptions"):
        run_mod.check_preemptions(level, demanded_blocks=4096)

    # One preemption is enough to clear it; the guard is about zero.
    level.preemption_count = 1
    run_mod.check_preemptions(level, demanded_blocks=4096)


def test_the_preemption_guard_also_fires_on_any_oversubscribed_level():
    """Not only at C64: any level demanding more than the pool must evict."""
    level = run_mod.LevelResult(
        engine=run_mod.ENGINE_PAGED, stream="mixed", concurrency=128,
        num_requests=512, passes=[1.0], preemption_count=0,
    )
    with pytest.raises(run_mod.InvalidRun, match="ZERO preemptions"):
        run_mod.check_preemptions(level, demanded_blocks=NUM_GPU_BLOCKS + 1)


def test_the_preemption_guard_does_not_fire_when_the_stream_fits():
    """A level that fits in the pool is allowed to preempt nothing."""
    level = run_mod.LevelResult(
        engine=run_mod.ENGINE_PAGED, stream="mixed", concurrency=4,
        num_requests=64, passes=[1.0], preemption_count=0,
    )
    run_mod.check_preemptions(level, demanded_blocks=NUM_GPU_BLOCKS - 1)

    # And the baseline has nothing to preempt, so it is never subject to it.
    baseline_level = run_mod.LevelResult(
        engine=run_mod.ENGINE_BASELINE, stream="control", concurrency=PRESSURE_C,
        num_requests=256, passes=[1.0], preemption_count=0,
    )
    run_mod.check_preemptions(baseline_level, demanded_blocks=4096)


# --- provenance and the no-numbers-here rule -------------------------------


def test_provenance_records_every_required_field_or_an_honest_none():
    stream = workload.build_stream("control", seed=0, c=1)
    provenance = run_mod.Provenance.collect(
        baseline_mod.BENCH_MODEL, "torch.float16", workload.stream_digest(stream)
    )
    record = provenance.__dict__

    required = {
        "num_gpu_blocks", "dtype", "gpu_name", "driver_version", "torch_version",
        "model_id", "model_sha", "workload_digest", "kv_pool_bytes", "block_size",
    }
    print(f"\nprovenance: gpu={record['gpu_name']} driver={record['driver_version']} "
          f"torch={record['torch_version']} sha={record['model_sha']}")
    assert required <= set(record)
    assert record["num_gpu_blocks"] == NUM_GPU_BLOCKS
    assert record["kv_pool_bytes"] == KV_POOL_BYTES
    assert record["workload_digest"] == workload.stream_digest(stream)
    # On this host there is no GPU, and the harness says so rather than guessing.
    if not torch.cuda.is_available():
        assert record["gpu_name"] is None
        assert record["cuda_available"] is False


def test_writing_a_real_result_off_a_cuda_host_is_refused(tmp_path):
    """The rule that keeps a laptop timing out of a file shaped like a result."""
    if torch.cuda.is_available():
        pytest.skip("this host has CUDA, so the refusal cannot be exercised")

    stream = workload.build_stream("control", seed=0, c=1)
    provenance = run_mod.Provenance.collect(
        baseline_mod.BENCH_MODEL, "torch.float32", workload.stream_digest(stream)
    )
    with pytest.raises(run_mod.InvalidRun, match="refusing to write results"):
        run_mod.write_results(tmp_path / "run.json", provenance, [], smoke=False)

    written = run_mod.write_results(tmp_path / "smoke.json", provenance, [], smoke=True)
    payload = json.loads(written.read_text())
    print(f"\nsmoke record: smoke={payload['smoke']} "
          f"throughput_valid={payload['throughput_valid']}")
    assert payload["smoke"] is True
    assert payload["throughput_valid"] is False


def test_the_cli_refuses_to_measure_without_cuda_or_smoke():
    if torch.cuda.is_available():
        pytest.skip("this host has CUDA")
    result = subprocess.run(
        [sys.executable, "-m", "bench.run", "--stream", "control", "-C", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    print(f"\nCLI without --smoke: exit {result.returncode}, {result.stderr.strip()}")
    assert result.returncode == 2
    assert "without CUDA" in result.stderr


def test_the_cli_takes_the_batch_order_and_rejects_an_unknown_one():
    """Both orders must be reachable from the command line, and typos must not run.

    Neither invocation loads a model: argparse rejects the bad value, and the
    good one reaches the no-CUDA refusal first. That is enough to prove the flag
    is wired, and it costs no forward pass.
    """
    bad = subprocess.run(
        [sys.executable, "-m", "bench.run", "--stream", "mixed", "-C", "1",
         "--smoke", "--baseline-order", "soretd"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    print(f"\nCLI --baseline-order soretd: exit {bad.returncode}")
    assert bad.returncode != 0
    assert "baseline-order" in bad.stderr

    for order in baseline_mod.BATCH_ORDERS:
        good = subprocess.run(
            [sys.executable, "-m", "bench.run", "--stream", "mixed", "-C", "1",
             "--baseline-order", order],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        print(f"  --baseline-order {order}: exit {good.returncode}")
        if torch.cuda.is_available():
            continue
        # Parsed, then stopped by the CUDA rule rather than by the flag.
        assert good.returncode == 2
        assert "without CUDA" in good.stderr


def test_the_cli_rejects_a_headline_at_the_wrong_level():
    result = subprocess.run(
        [sys.executable, "-m", "bench.run", "--stream", "control",
         "-C", "32", "--headline", "--smoke"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    print(f"\nCLI headline at C=32: exit {result.returncode}")
    assert result.returncode != 0
    assert "PRESSURE_C" in (result.stderr + result.stdout)


# --- end to end, tiny, and timed by nobody ---------------------------------


def test_sweep_runs_end_to_end_on_the_stub_scale(tmp_path):
    """Every code path, at a scale a laptop can hold, with the clock ignored.

    The elapsed values this produces are real wall-clock, and they are discarded:
    the record it writes is stamped `throughput_valid: false`, and this test
    asserts that rather than asserting anything about how long it took.
    """
    from engine.model import ModelDims

    torch.manual_seed(0)
    # A tiny model with the real forward pass. Enough to exercise the engine path
    # without loading 1.5B parameters onto a laptop.
    dims = ModelDims(
        num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, hidden_size=32,
        rms_norm_eps=1e-6, vocab_size=64, rope_theta=10000.0,
    )
    engine = tiny_engine(dims)
    pool_blocks = 32

    def cache_factory(allocator):
        return engine.new_paged_cache(allocator.num_blocks, allocator.block_size)

    # What the sweep hands the factory. Captured rather than ignored: the caps
    # existed on run_engine_pass and were passed by nobody, so the sweep ran the
    # engine uncapped and OOM'd on the measurement host at 13.86 of 14.56 GiB
    # before producing one timing. A stub that shrugs off its arguments is how a
    # missing argument survives a green suite.
    handed: dict = {}

    def engine_factory(stream, max_num_seqs, max_num_batched_tokens):
        handed["max_num_seqs"] = max_num_seqs
        handed["max_num_batched_tokens"] = max_num_batched_tokens

        def run_pass():
            def executor_factory(allocator):
                from engine.scheduler import Qwen2Executor

                return Qwen2Executor(engine, cache_factory(allocator), allocator)

            return run_mod.run_engine_pass(
                stream, executor_factory, num_gpu_blocks=pool_blocks,
                mode=ExecutionMode.FUSED,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
            )

        return run_pass

    stream = _tiny_stream(vocab=dims.vocab_size)
    monkey_stream = {1: stream}

    original = workload.build_stream
    workload.build_stream = lambda name, seed, c: monkey_stream[c]
    try:
        results = run_mod.sweep(
            [1], "mixed", 0, engine_factory, None,
            throughput_valid=False, log=lambda _: None,
        )
    finally:
        workload.build_stream = original

    assert handed.get("max_num_seqs") == 1, (
        f"the sweep handed max_num_seqs={handed.get('max_num_seqs')!r} at C=1. "
        "It must pass the level, or C means nothing and the engine runs at "
        "whatever fits the pool while the baseline is capped by its rectangle."
    )
    assert handed.get("max_num_batched_tokens") == run_mod.MAX_NUM_BATCHED_TOKENS, (
        f"the sweep handed max_num_batched_tokens="
        f"{handed.get('max_num_batched_tokens')!r}. Unset, the fused step's "
        "padded rectangle is unbounded -- 102 sequences x 1969 tokens on the "
        "mixed stream, which is how this OOM'd on a T4."
    )

    assert len(results) == 1
    level = results[0]
    print(f"\nsmoke sweep: {level.num_requests} requests, "
          f"{level.generated_tokens} tokens, {len(level.passes)} passes, "
          f"preemptions {level.preemption_count}, "
          f"throughput_valid={level.throughput_valid}")

    assert level.generated_tokens > 0, "the engine produced nothing"
    assert level.throughput_valid is False
    assert level.median_tokens_per_second is None, (
        "a run stamped throughput_valid: false must not yield a throughput number"
    )

    provenance = run_mod.Provenance.collect("tiny", "torch.float32", "smoke")
    written = run_mod.write_results(tmp_path / "smoke.json", provenance, results, smoke=True)
    payload = json.loads(written.read_text())
    assert payload["smoke"] is True
    assert payload["levels"][0]["median_tokens_per_second"] is None


def _tiny_stream(vocab: int):
    return tuple(
        workload.Request(
            request_id=f"tiny-{index}",
            prompt_token_ids=tuple((index * 7 + step) % vocab for step in range(12)),
            max_new_tokens=4 + index,
        )
        for index in range(6)
    )


# --- plots ------------------------------------------------------------------


def test_bench_workload_is_still_frozen():
    """The one file in bench/ that may not change, pinned by content.

    Step 6 adds baseline.py, run.py and plot.py beside it and must not touch it:
    every number the harness produces is indexed by `stream_digest`, and that
    digest is only comparable across runs if the generator behind it has not
    moved. A content hash rather than a commit count, because the freeze is on
    what the file says, not on how its history is shaped.

    If this fails and the change was intended, the workload has been re-based and
    every previously recorded digest refers to a different stream.
    """
    import hashlib

    path = REPO_ROOT / "bench" / "workload.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"\nbench/workload.py sha256 {digest} ({path.stat().st_size} bytes)")
    assert digest == FROZEN_WORKLOAD_SHA256, (
        "bench/workload.py has changed. It is frozen: it is the single source of "
        "the request stream, and stream_digest is only comparable across runs "
        f"while it holds still. Expected {FROZEN_WORKLOAD_SHA256}, got {digest}."
    )


@pytest.mark.skipif(
    not (REPO_ROOT / ".git").exists(),
    reason="no .git in the measurement payload; the CONTENT HASH check above "
           "still runs there and is the one that matters",
)
def test_bench_workload_has_no_uncommitted_edit():
    """And it is frozen in the working tree, not merely in the last commit."""
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "bench/workload.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout.strip(), (
        f"bench/workload.py differs from HEAD:\n{result.stdout}"
    )


def test_passes_that_do_different_amounts_of_work_are_refused():
    """A median over passes that did different work is a median over a mixture."""
    durations = iter([9.0, 1.0, 1.0])
    token_counts = iter([1000, 1000, 999])

    def run_pass():
        return run_mod.EnginePass(
            elapsed=next(durations), generated_tokens=next(token_counts),
            preemptions=1, num_rejected=0, peak_blocks_used=10,
        )

    stream = workload.build_stream("control", seed=0, c=1)
    with pytest.raises(run_mod.InvalidRun, match="different amounts of work"):
        run_mod.measure_level(
            run_mod.ENGINE_PAGED, "control", 1, stream, run_pass, log=lambda _: None
        )


# --- F2: the MEASURED fragmentation plot -----------------------------------




def test_baseline_takes_the_same_concurrency_cap_as_the_engine():
    """Residency becomes min(rectangle limit, C), so a sweep compares like with like.

    Without this the baseline was hard-capped at 18 or 32 by its rectangle while
    the engine admitted whatever fit the pool -- at C=8 that was engine-at-64
    against baseline-at-18, and no ratio out of that means anything.
    """
    stream = workload.build_stream("control", seed=0, c=4)
    rectangle = baseline_mod.RECTANGLE_P95
    structural = baseline_mod.max_resident(rectangle)
    assert structural == 32

    observed = {}
    for cap in (None, 4, 8, 64):
        ticks = iter([0.0, 1.0])
        generator = _FakeGenerator()
        result = baseline_mod.run_baseline(
            stream, generator, rectangle, max_num_seqs=cap,
            clock=lambda: next(ticks),
        )
        observed[cap] = {
            "effective_resident": result.max_resident,
            "widest_batch": max(generator.batch_sizes),
            "num_batches": result.num_batches,
        }
        print(f"\ncap={cap}: {observed[cap]}")
        assert result.max_num_seqs == cap

    # Uncapped, the rectangle's own limit binds.
    assert observed[None]["effective_resident"] == structural
    # A cap below it binds instead, and the batches actually shrink.
    assert observed[4]["effective_resident"] == 4
    assert observed[4]["widest_batch"] == 4
    assert observed[8]["effective_resident"] == 8
    assert observed[8]["widest_batch"] == 8
    # A cap above it does NOT raise residency past what the bytes allow.
    assert observed[64]["effective_resident"] == structural, (
        "max_num_seqs must not let the baseline exceed its own KV byte ceiling"
    )
    # More batches at a tighter cap, same work.
    assert observed[4]["num_batches"] > observed[None]["num_batches"]


def test_provenance_records_sdpa_backends_and_allocator_conf():
    """Two things that move numbers without touching arithmetic.

    SDPA backend selection materially moves fp16 numerics -- the enable_gqa
    switch changed the input shape, torch picked a different kernel, and fp16
    parity improved 6x. That happened silently and was recorded nowhere.

    expandable_segments changes allocation only, so it cannot move a logit, but
    it does move the memory numbers, and a memory number without the allocator
    setting that produced it is not reproducible.
    """
    stream = workload.build_stream("control", seed=0, c=1)
    provenance = run_mod.Provenance.collect(
        baseline_mod.BENCH_MODEL, "torch.float16", workload.stream_digest(stream)
    )
    print(f"\nsdpa backends: {provenance.sdpa_backends_enabled}")
    print(f"allocator conf: {provenance.allocator_conf}")

    assert provenance.sdpa_backends_enabled, (
        "no SDPA backend state recorded; the getter names are wrong again"
    )
    assert set(provenance.sdpa_backends_enabled) >= {"flash", "mem_efficient", "math"}
    assert all(
        isinstance(value, bool)
        for value in provenance.sdpa_backends_enabled.values()
    )
    # Present as keys even when unset, so "not recorded" and "not set" differ.
    assert set(provenance.allocator_conf) == {
        "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"
    }


def test_every_arm_cell_is_assigned_to_exactly_one_kaggle_slice():
    """A cell with no slice never runs, and a matrix with a hole reads as full.

    Two families of kernels, both picking their work from a declared literal
    because Kaggle renames an uploaded script to script.py and a derived
    identity would refuse to start on the measurement host:

      session6?  the primary matrix, one cell each, both arms
      session7?  the gap-fill levels, a LIST of cells each, arm B only

    The tables live in different files from the matrix they must agree with.
    """
    import ast

    from bench.arms import ALL_CELLS, CELLS, GAP_FILL_CELLS, cell_name

    repo_root = Path(__file__).resolve().parent.parent

    def read_tables(pattern):
        tables, declared = {}, {}
        for candidate in sorted((repo_root / "kaggle").glob(pattern)):
            for node in ast.walk(ast.parse(candidate.read_text())):
                if not isinstance(node, ast.Assign):
                    continue
                name = getattr(node.targets[0], "id", "")
                if name == "SLICES":
                    tables[candidate.stem] = ast.literal_eval(node.value)
                elif name == "SLICE":
                    declared[candidate.stem] = ast.literal_eval(node.value)
        return tables, declared

    for pattern, expected_cells, both_arms in (
        ("session6?/session6?.py", CELLS, True),
        ("session7?/session7?.py", GAP_FILL_CELLS, False),
    ):
        tables, declared = read_tables(pattern)
        if not tables:
            pytest.skip(f"no kernels matching {pattern}")

        first = next(iter(tables.values()))
        for stem, table in tables.items():
            assert table == first, (
                f"kaggle/{stem} carries a different slice table; the kernels in "
                "one family must be identical copies"
            )
            assert declared.get(stem) == stem, (
                f"kaggle/{stem}/{stem}.py declares SLICE={declared.get(stem)!r}. "
                "It would measure another slice's cells while its name promises "
                "its own, and two kernels would silently cover the same work."
            )
        assert set(declared) == set(first), (
            f"slice files {sorted(declared)} do not match the table {sorted(first)}"
        )

        # A slice's value is one cell name or a list of them; flatten and check.
        assigned = []
        for value in first.values():
            assigned.extend([value] if isinstance(value, str) else list(value))
        known = {cell_name(cell) for cell in expected_cells}
        assert set(assigned) == known, (
            f"{pattern}: only-in-table={set(assigned) - known}, "
            f"only-in-matrix={known - set(assigned)}"
        )
        assert len(assigned) == len(set(assigned)), (
            f"{pattern}: a cell is assigned twice and would be measured twice: "
            f"{assigned}"
        )

    # Every cell in the full matrix is covered by exactly one family.
    all_names = {cell_name(cell) for cell in ALL_CELLS}
    assert all_names == ({cell_name(c) for c in CELLS}
                         | {cell_name(c) for c in GAP_FILL_CELLS})


def test_gap_fill_levels_sit_above_the_baseline_rectangle_cap():
    """Below the cap both engines are capped by C and the level says nothing.

    `max_resident` is min(rectangle limit, C). At C below the rectangle limit
    the baseline and the engine are both held at C, the engine's residency
    advantage is exactly 1.0x, and the cell would buy a loss that is about the
    cap rather than about paging. The gap-fill levels are chosen to be at or
    above the cap on at least one rectangle, and C=32 on rect1757 sits exactly
    ON it -- the crossover, where a tie or a loss is the expected reading.
    """
    from bench.arms import GAP_FILL_CELLS
    from bench.baseline import RECTANGLE_SIZINGS, max_resident

    caps = {rectangle: max_resident(rectangle) for rectangle in RECTANGLE_SIZINGS}
    assert caps == {3072: 18, 1757: 32}, f"rectangle caps moved: {caps}"

    for cell in GAP_FILL_CELLS:
        concurrency, rectangle = cell["concurrency"], cell["rectangle"]
        assert concurrency >= caps[rectangle], (
            f"gap-fill cell C={concurrency} rect={rectangle} is BELOW the "
            f"baseline cap of {caps[rectangle]}. Both sides would be capped at "
            "C, the engine would have no residency advantage, and the level "
            "would document the cap rather than paging."
        )
        assert cell["arms"] == ("B",), (
            "gap-fill cells run arm B only; arm A mixed is asymmetric (engine "
            "MATH against a baseline already on EFFICIENT) and may not be timed"
        )
