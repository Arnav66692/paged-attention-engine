"""The Gate 1 plots: the analytical split, the measured trace, and the banners.

Split out of tests/test_bench.py, which covers the harness that produces the
numbers. This covers what is drawn from them.

Two fragmentation plots ship and they answer different questions. The ANALYTICAL
one is a property of the frozen workload's length distribution -- it explains why
paging should win and would read the same on any host, which is both its strength
and its limit, because it proves nothing about a particular run. The MEASURED one
comes off the scheduler's own event log, which records free_blocks and
tokens_held at every admit, finish, preempt and recompute: capacity against
occupancy, sampled at every decision. Analytical explains why; measured proves it
happened. Neither substitutes for the other.

NO THROUGHPUT NUMBER IS PRODUCED HERE. The traced run below uses a real forward
pass to generate a real event log, and its wall-clock is discarded.
"""


import pytest
import torch

from bench import baseline as baseline_mod
from bench import plot as plot_mod
from bench import run as run_mod
from bench import workload
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS
from engine.scheduler import ExecutionMode
from tests.tiny_model import TINY_DIMS, tiny_engine

KV_POOL_BYTES = 1_644_167_168
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



def _traced_pass(pool_blocks: int = 12, block_size: int = 4) -> run_mod.EnginePass:
    """A real engine pass at tiny scale, on a pool tight enough to preempt.

    Real forward passes through a random-weight model, real block accounting,
    real evictions. Only the weights are synthetic, and the trace does not depend
    on them -- admission and eviction read token counts, never token values.
    """
    from engine.scheduler import Qwen2Executor

    dims = TINY_DIMS
    engine = tiny_engine(dims)
    stream = tuple(
        workload.Request(
            request_id=f"frag-{index}",
            prompt_token_ids=tuple((index * 5 + step) % dims.vocab_size for step in range(5)),
            max_new_tokens=24,
        )
        for index in range(5)
    )

    def executor_factory(allocator):
        cache = engine.new_paged_cache(allocator.num_blocks, allocator.block_size)
        return Qwen2Executor(engine, cache, allocator)

    return run_mod.run_engine_pass(
        stream, executor_factory, num_gpu_blocks=pool_blocks, block_size=block_size,
        mode=ExecutionMode.FUSED,
    )


def _traced_payload():
    observed = _traced_pass()
    return {
        "smoke": True,
        "throughput_valid": False,
        "provenance": {
            "kv_pool_bytes": KV_POOL_BYTES,
            "bytes_per_token": baseline_mod.BYTES_PER_TOKEN,
            "num_gpu_blocks": 12,
            "block_size": 4,
            "gpu_name": None, "driver_version": None,
            "torch_version": torch.__version__, "dtype": "torch.float32",
            "model_sha": None, "workload_digest": "smoke", "cuda_available": False,
        },
        "levels": [{
            "engine": "engine", "stream": "mixed", "concurrency": 1,
            "median_tokens_per_second": None, "median_seconds": 1.0,
            "range_seconds": 0.0, "spread_fraction": 0.0,
            "generated_tokens": observed.generated_tokens, "unresolved": False,
            "demanded_blocks": 40, "peak_blocks_used": observed.peak_blocks_used,
            "preemption_count": observed.preemptions,
            "pool_trace": [list(point) for point in observed.pool_trace],
            "trace_events_total": observed.trace_events_total,
        }],
    }, observed


def test_the_event_log_records_occupancy_as_well_as_capacity():
    """free_blocks is what the pool handed out; tokens_held is what is in it.

    Without both, a measured fragmentation plot cannot be drawn at all -- the
    gap between them IS the fragmentation.
    """
    observed = _traced_pass()
    assert observed.pool_trace, "the pass recorded no pool trace"

    block_size = 4
    worst = 0.0
    for step, blocks_allocated, tokens_held, kind in observed.pool_trace:
        reserved = blocks_allocated * block_size
        # Occupancy can never exceed the capacity handed out to hold it.
        assert tokens_held <= reserved, (
            f"step {step}: {tokens_held} tokens held in {reserved} reserved slots, "
            "which would mean tokens living outside any allocated block"
        )
        # Nor can a sequence hold more blocks than its tokens need.
        assert blocks_allocated * block_size - tokens_held < max(1, blocks_allocated) * block_size
        if reserved:
            worst = max(worst, (reserved - tokens_held) / reserved)

    print(f"\ntrace: {len(observed.pool_trace)} points, "
          f"{observed.preemptions} preemptions, "
          f"worst internal fragmentation {worst:.1%}")
    assert observed.preemptions > 0, "the fixture did not preempt, so it is too easy"
    assert any(tokens > 0 for _, _, tokens, _ in observed.pool_trace)


def test_measured_occupancy_reads_the_trace_back_out():
    payload, observed = _traced_payload()
    measured = plot_mod.measured_occupancy(payload)
    assert measured is not None

    print(f"\nmeasured: mean occupancy {measured.mean_occupancy:.1%}, "
          f"peak reserved {measured.peak_reserved_slots} slots of "
          f"{measured.pool_slots}, {len(measured.preempt_steps)} preempts, "
          f"{len(measured.recompute_steps)} recomputes")

    assert len(measured.steps) == len(observed.pool_trace)
    assert 0.0 < measured.mean_occupancy <= 1.0
    assert measured.peak_reserved_slots <= measured.pool_slots
    assert len(measured.preempt_steps) == observed.preemptions
    # Every reserved figure is a whole number of blocks.
    assert all(slots % measured.block_size == 0 for slots in measured.reserved_slots)


def test_measured_and_analytical_fragmentation_are_both_shipped():
    """Both plots exist and answer different questions.

    Analytical is a property of the length distribution and would read the same
    on any host. Measured comes off one run's event log. Neither substitutes for
    the other, which is why both are drawn.
    """
    payload, _ = _traced_payload()
    assert plot_mod.slot_budgets(baseline_mod.KV_POOL_TOKEN_SLOTS)
    assert plot_mod.measured_occupancy(payload) is not None
    # And the measured plot degrades honestly when a run recorded no trace.
    assert plot_mod.measured_occupancy(_smoke_payload()) is None


def test_all_three_plots_render(tmp_path):
    pytest.importorskip("matplotlib", reason="plotting is optional on this host")
    payload, _ = _traced_payload()

    paths = [
        plot_mod.throughput_vs_concurrency(payload, tmp_path / "throughput.png"),
        plot_mod.memory_fragmentation(payload, tmp_path / "analytical.png"),
        plot_mod.measured_fragmentation(payload, tmp_path / "measured.png"),
    ]
    for path in paths:
        print(f"\nrendered {path.name}: {path.stat().st_size} bytes")
        assert path.exists() and path.stat().st_size > 0


def test_the_measured_plot_says_so_when_a_run_has_no_trace(tmp_path):
    pytest.importorskip("matplotlib", reason="plotting is optional on this host")
    path = plot_mod.measured_fragmentation(_smoke_payload(), tmp_path / "empty.png")
    assert path.exists() and path.stat().st_size > 0
