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

    def engine_factory(stream):
        def run_pass():
            def executor_factory(allocator):
                from engine.scheduler import Qwen2Executor

                return Qwen2Executor(engine, cache_factory(allocator), allocator)

            return run_mod.run_engine_pass(
                stream, executor_factory, num_gpu_blocks=pool_blocks,
                mode=ExecutionMode.FUSED,
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

    The four session6 kernels are identical files that pick their cell from
    their own filename. That is only safe while the table and the matrix agree,
    and they live in different files.
    """
    import ast

    from bench.arms import CELLS, cell_name

    repo_root = Path(__file__).resolve().parent.parent
    slices = None
    for candidate in sorted((repo_root / "kaggle").glob("session6?/session6?.py")):
        for node in ast.walk(ast.parse(candidate.read_text())):
            if (isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", "") == "SLICES"):
                table = ast.literal_eval(node.value)
                assert slices is None or table == slices, (
                    f"{candidate.name} carries a different slice table; the "
                    "four kernels must be identical copies"
                )
                slices = table
    if slices is None:
        pytest.skip("session6 slices not present")

    assigned = list(slices.values())
    known = {cell_name(cell) for cell in CELLS}
    assert set(assigned) == known, (
        f"slice table and arm matrix disagree: only-in-table={set(assigned) - known}, "
        f"only-in-matrix={known - set(assigned)}"
    )
    assert len(assigned) == len(set(assigned)), (
        f"a cell is assigned to two slices and would be measured twice: {assigned}"
    )
