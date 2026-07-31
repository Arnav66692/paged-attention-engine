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
from bench import plot as plot_mod
from bench import run as run_mod
from bench import workload
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS, PRESSURE_C
from engine.cache import ADMISSION_MARGIN_BLOCKS, BlockAllocator
from engine.scheduler import ExecutionMode, is_admissible

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
    from engine.model import ModelDims, Qwen2Engine

    torch.manual_seed(0)
    # A tiny model with the real forward pass. Enough to exercise the engine path
    # without loading 1.5B parameters onto a laptop.
    dims = ModelDims(
        num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, hidden_size=32,
        rms_norm_eps=1e-6, vocab_size=64, rope_theta=10000.0,
    )
    engine = _tiny_engine(dims)
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


def _tiny_engine(dims):
    from engine.model import LayerWeights, Qwen2Engine, Qwen2Weights

    def randn(*shape):
        return torch.randn(*shape) * 0.02

    layers = tuple(
        LayerWeights(
            q_proj_weight=randn(dims.num_heads * dims.head_dim, dims.hidden_size),
            q_proj_bias=torch.zeros(dims.num_heads * dims.head_dim),
            k_proj_weight=randn(dims.num_kv_heads * dims.head_dim, dims.hidden_size),
            k_proj_bias=torch.zeros(dims.num_kv_heads * dims.head_dim),
            v_proj_weight=randn(dims.num_kv_heads * dims.head_dim, dims.hidden_size),
            v_proj_bias=torch.zeros(dims.num_kv_heads * dims.head_dim),
            o_proj_weight=randn(dims.hidden_size, dims.num_heads * dims.head_dim),
            gate_proj_weight=randn(64, dims.hidden_size),
            up_proj_weight=randn(64, dims.hidden_size),
            down_proj_weight=randn(dims.hidden_size, 64),
            input_layernorm_weight=torch.ones(dims.hidden_size),
            post_attention_layernorm_weight=torch.ones(dims.hidden_size),
        )
        for _ in range(dims.num_layers)
    )
    weights = Qwen2Weights(
        embed_tokens=randn(dims.vocab_size, dims.hidden_size),
        layers=layers,
        final_norm_weight=torch.ones(dims.hidden_size),
        lm_head_weight=randn(dims.vocab_size, dims.hidden_size),
    )
    return Qwen2Engine(weights, dims, torch.device("cpu"), torch.float32)


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


def _smoke_payload():
    return {
        "smoke": True,
        "throughput_valid": False,
        "provenance": {
            "kv_pool_bytes": KV_POOL_BYTES,
            "bytes_per_token": baseline_mod.BYTES_PER_TOKEN,
            "num_gpu_blocks": NUM_GPU_BLOCKS,
            "block_size": BLOCK_SIZE,
            "gpu_name": None,
            "driver_version": None,
            "torch_version": torch.__version__,
            "dtype": "torch.float32",
            "model_sha": None,
            "workload_digest": "smoke",
            "cuda_available": False,
        },
        "levels": [
            {"engine": "engine", "stream": "control", "concurrency": 16,
             "median_tokens_per_second": 100.0, "median_seconds": 10.0,
             "range_seconds": 0.5, "spread_fraction": 0.05, "generated_tokens": 1000,
             "unresolved": False, "demanded_blocks": 1024, "peak_blocks_used": 900,
             "preemption_count": 0},
            {"engine": "engine", "stream": "control", "concurrency": 64,
             "median_tokens_per_second": None, "median_seconds": 40.0,
             "range_seconds": 12.0, "spread_fraction": 0.30, "generated_tokens": 4000,
             "unresolved": True, "demanded_blocks": 4096, "peak_blocks_used": 3584,
             "preemption_count": 41},
            {"engine": "baseline", "stream": "control", "concurrency": 16,
             "median_tokens_per_second": 40.0, "median_seconds": 25.0,
             "range_seconds": 1.0, "spread_fraction": 0.04, "generated_tokens": 1000,
             "unresolved": False, "demanded_blocks": 1024, "peak_blocks_used": 0,
             "preemption_count": 0},
        ],
    }


def test_an_invalid_run_is_labelled_inside_the_plot_not_in_a_caption():
    payload = _smoke_payload()
    banner = plot_mod._invalid_banner(payload)
    print(f"\nbanner for a smoke run: {banner!r}")
    assert banner and "SMOKE" in banner

    valid = _smoke_payload()
    valid["smoke"] = False
    valid["throughput_valid"] = True
    valid["provenance"]["cuda_available"] = True
    assert plot_mod._invalid_banner(valid) is None


def test_plots_render_from_a_results_file(tmp_path):
    pytest.importorskip("matplotlib", reason="plotting is optional on this host")
    payload = _smoke_payload()

    throughput = plot_mod.throughput_vs_concurrency(
        payload, tmp_path / "throughput.png"
    )
    fragmentation = plot_mod.memory_fragmentation(
        payload, tmp_path / "fragmentation.png"
    )
    print(f"\nrendered {throughput.name} ({throughput.stat().st_size} bytes) and "
          f"{fragmentation.name} ({fragmentation.stat().st_size} bytes)")

    assert throughput.exists() and throughput.stat().st_size > 0
    assert fragmentation.exists() and fragmentation.stat().st_size > 0


def test_the_plot_draws_no_point_for_an_unresolved_level():
    """UNRESOLVED means no median exists, so there is nothing to plot for it."""
    payload = _smoke_payload()
    engine_levels = [lv for lv in payload["levels"] if lv["engine"] == "engine"]
    resolved = [lv for lv in engine_levels if lv["median_tokens_per_second"] is not None]
    unresolved = [lv for lv in engine_levels if lv["unresolved"]]

    assert unresolved, "the fixture must contain an unresolved level"
    assert all(lv["median_tokens_per_second"] is None for lv in unresolved)
    assert all(lv["concurrency"] not in {u["concurrency"] for u in unresolved}
               for lv in resolved)


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


def test_the_fragmentation_split_is_the_argument_for_paging():
    """The left panel's numbers, asserted rather than only drawn.

    A plot is the easiest artefact to get subtly wrong and the hardest to notice
    it in, so the quantity it claims is checked here. The claim is that at the
    workload's own mean request length, a rectangle sized for the longest request
    spends most of its slots on padding and a 16-token block spends almost none.
    """
    budgets = plot_mod.slot_budgets(baseline_mod.KV_POOL_TOKEN_SLOTS)
    by_label = {budget.label.split("\n")[0]: budget for budget in budgets}

    print()
    for budget in budgets:
        flat = budget.label.replace("\n", " ")
        print(f"  {flat:<44} held {budget.held:6.1%}  "
              f"padding {budget.padding:6.1%}  stranded {budget.stranded:6.1%}")

    assert len(budgets) == len(baseline_mod.RECTANGLE_SIZINGS) + 1
    for budget in budgets:
        # Enforced by SlotBudget.__post_init__, restated so the intent is visible.
        assert budget.held + budget.padding + budget.stranded == pytest.approx(1.0)
        assert 0.0 <= budget.held <= 1.0

    engine = by_label["paged engine"]
    wide, narrow = budgets[0], budgets[1]

    # The engine holds nearly everything and strands nothing.
    assert engine.stranded == 0.0
    assert engine.held > 0.95, (
        f"the paged engine only holds {engine.held:.1%} of its slots, which would "
        "mean 16-token blocks fragment nearly as badly as a rectangle"
    )
    # Both rectangles are mostly padding, and the wider one is worse.
    assert wide.padding > 0.5
    assert wide.held < narrow.held < engine.held, (
        "the sizings are not ordered as the argument requires: a rectangle sized "
        "for the distribution maximum must waste more than the P95, which must "
        "waste more than a 16-token block"
    )
    # And a rectangle strands ceiling the engine does not.
    assert wide.stranded > 0.0 and narrow.stranded > 0.0


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
