"""Stage 2 of docs/PREREG_arrival.md: measure TTFT and latency under arrivals.

Throughput is NOT the metric here and is recorded only for provenance. Below
saturation both schedulers complete every request, so both match the arrival
rate by construction and tokens/second says nothing. What differs is WHEN each
request is served, which is what this measures.

WHAT IS DRIVEN IN REAL TIME. Arrival times are layered on the frozen stream from
a seeded Poisson process, and both sides are driven against a wall clock started
at the same point in their own run. A scheduler with nothing to do waits for the
next arrival rather than racing ahead, because an idle static batcher waiting is
exactly the behaviour under test.

  engine     admits any arrived request that fits, then steps. A finished slot
             is refilled on the next step.
  baseline   waits for its current batch to fully drain, then takes up to
             `resident` from whatever has queued, decode-length sorted. That
             wait is the mechanism the pre-registration predicts will dominate
             TTFT.

HOW TTFT IS CAPTURED ON EACH SIDE, because they are not symmetric.

  engine     the step that admits a request also prefills it and emits its first
             token, so TTFT is the wall time at the end of that step.
  baseline   `generate` is a single call, so a LogitsProcessor records the wall
             time of its first invocation. That fires once prefill has produced
             logits and before the first token is chosen, which is the same
             instant the engine's admit step reports. Overhead is one comparison
             per generation step.

PAIRING. Both sides run in ONE session on ONE card. Between-session drift on
this project has been measured at 13.6 to 16.3 percent on byte-identical work,
which is larger than several effects reported here, so a cross-session
comparison would be worthless.

bench/workload.py is untouched. Same 256 requests, same digest.
"""

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from bench import arrival as arrival_mod
from bench import baseline as baseline_mod
from bench import workload
from bench.run import MAX_NUM_BATCHED_TOKENS, _cuda_sync, pin_fp16_reduction
from engine.cache import ADMISSION_MARGIN_BLOCKS, BlockAllocator
from engine.model import Qwen2Engine
from engine.scheduler import ExecutionMode, Qwen2Executor, Request, Scheduler
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS

WARMUP_PASSES = 1
MEASURED_PASSES = 3


@dataclass
class Observation:
    request_id: str
    arrives_at: float
    first_token_at: float
    finished_at: float

    @property
    def ttft(self) -> float:
        return self.first_token_at - self.arrives_at

    @property
    def latency(self) -> float:
        return self.finished_at - self.arrives_at


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(observations: Sequence[Observation]) -> dict:
    ttfts = [o.ttft for o in observations]
    lats = [o.latency for o in observations]
    return {
        "requests": len(observations),
        "ttft_p50": percentile(ttfts, 0.50),
        "ttft_p99": percentile(ttfts, 0.99),
        "ttft_mean": statistics.mean(ttfts) if ttfts else float("nan"),
        "ttft_max": max(ttfts) if ttfts else float("nan"),
        "latency_p50": percentile(lats, 0.50),
        "latency_p99": percentile(lats, 0.99),
        "latency_mean": statistics.mean(lats) if lats else float("nan"),
    }


# --- the engine -------------------------------------------------------------


def run_engine(arrivals, executor_factory, max_num_seqs: int):
    """Step the real scheduler against a wall clock, admitting on arrival."""
    allocator = BlockAllocator(num_blocks=NUM_GPU_BLOCKS, block_size=BLOCK_SIZE)
    scheduler = Scheduler(
        executor_factory(allocator), allocator, mode=ExecutionMode.FUSED,
        admission_margin_blocks=ADMISSION_MARGIN_BLOCKS,
        max_num_seqs=max_num_seqs, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )

    pending = list(arrivals)
    first_token: dict[str, float] = {}
    finished: dict[str, float] = {}
    arrival_of = {a.request_id: a.arrives_at for a in arrivals}
    widths: list[int] = []
    rejected = 0

    _cuda_sync()
    started = time.perf_counter()
    # `add_request` enqueues into `waiting`; a request only reaches `running`
    # once a step admits it. Gating the step on `running` alone therefore never
    # steps at all, which is exactly what the dry run caught: zero steps, zero
    # observations, and a clean-looking exit.
    def has_work() -> bool:
        return bool(scheduler.running) or bool(scheduler.waiting)

    while pending or has_work():
        now = time.perf_counter() - started
        # Nothing to run and nothing arrived: wait for the next arrival. An idle
        # scheduler idles; racing ahead would measure a different system.
        if not has_work() and pending and pending[0].arrives_at > now:
            time.sleep(min(pending[0].arrives_at - now, 0.25))
            continue
        while pending and pending[0].arrives_at <= (time.perf_counter() - started):
            a = pending.pop(0)
            if not scheduler.add_request(
                Request(a.request_id, a.prompt_token_ids, a.decode_len)
            ):
                rejected += 1
        if not has_work():
            continue

        result = scheduler.step()
        _cuda_sync()
        stamp = time.perf_counter() - started
        widths.append(result.batch_size)
        for rid in result.admitted:
            first_token.setdefault(rid, stamp)
        for rid in result.finished:
            finished[rid] = stamp
        if not result.batch_size and not pending and not has_work():
            break
    _cuda_sync()
    wall = time.perf_counter() - started

    observations = [
        Observation(rid, arrival_of[rid], first_token[rid], finished[rid])
        for rid in finished if rid in first_token
    ]
    return observations, {
        "wall_seconds": wall,
        "mean_batch_width": statistics.mean(widths) if widths else 0.0,
        "steps": len(widths),
        "rejected": rejected,
    }


# --- the static baseline ----------------------------------------------------


class _FirstCallClock:
    """LogitsProcessor that records when prefill first produced logits."""

    def __init__(self) -> None:
        self.at: float | None = None
        self.origin = 0.0

    def __call__(self, input_ids, scores):
        if self.at is None:
            _cuda_sync()          # no-op off CUDA; the dry run has no device
            self.at = time.perf_counter() - self.origin
        return scores


def run_baseline(arrivals, generator, resident: int, sort_by_decode: bool):
    """Static batching against the same wall clock. Drains before admitting."""
    from transformers import LogitsProcessorList

    pending = list(arrivals)
    queue: list = []
    observations: list[Observation] = []
    depths: list[int] = []
    row_steps = 0
    useful_steps = 0

    _cuda_sync()
    started = time.perf_counter()
    while pending or queue:
        now = time.perf_counter() - started
        if not queue and pending and pending[0].arrives_at > now:
            time.sleep(min(pending[0].arrives_at - now, 0.25))
            continue
        while pending and pending[0].arrives_at <= (time.perf_counter() - started):
            queue.append(pending.pop(0))
        if not queue:
            continue

        depths.append(len(queue))
        # THE SORT ONLY SEES WHAT HAS QUEUED. That is the hypothesis.
        if sort_by_decode:
            queue.sort(key=lambda a: a.decode_len)
        batch = queue[:resident]
        del queue[:len(batch)]

        clock = _FirstCallClock()
        clock.origin = started
        requests = [
            workload.Request(a.request_id, a.prompt_token_ids, a.decode_len)
            for a in batch
        ]
        steps = max(a.decode_len for a in batch)
        row_steps += steps * len(batch)
        useful_steps += sum(a.decode_len for a in batch)

        generator.generate_batch(
            requests, logits_processor=LogitsProcessorList([clock])
        )
        _cuda_sync()
        finished_at = time.perf_counter() - started
        first_at = clock.at if clock.at is not None else finished_at

        for a in batch:
            observations.append(
                Observation(a.request_id, a.arrives_at, first_at, finished_at)
            )

    _cuda_sync()
    wall = time.perf_counter() - started
    return observations, {
        "wall_seconds": wall,
        "row_steps": row_steps,
        "useful_steps": useful_steps,
        "decode_efficiency": useful_steps / row_steps if row_steps else 0.0,
        "queue_depths": depths,
        "mean_queue_depth": statistics.mean(depths) if depths else 0.0,
        "median_queue_depth": statistics.median(depths) if depths else 0.0,
        "batches": len(depths),
    }


# --- the driver -------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rho", type=float, default=0.4)
    parser.add_argument("--rectangle", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--passes", type=int, default=MEASURED_PASSES)
    parser.add_argument("--model", default=baseline_mod.BENCH_MODEL)
    parser.add_argument("--out", type=Path,
                        default=Path("bench/results/arrival/stage2.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if not args.smoke and not torch.cuda.is_available():
        print("refusing to measure without CUDA; pass --smoke", flush=True)
        return 2

    pin_fp16_reduction()
    stream = workload.build_stream("mixed", 0, 64)
    resident = baseline_mod.max_resident(args.rectangle)

    # The offered load is defined against the SORTED batcher's saturated service
    # rate, computed by the Stage 1 simulator so both stages mean the same thing
    # by "rho". Nothing about the arrival times depends on the GPU.
    arrival_mod.calibrate()
    saturated_rate, _ = arrival_mod.sweep(resident, args.max_num_seqs, [args.rho], args.seed)
    rate = args.rho * saturated_rate
    arrivals = arrival_mod.poisson_arrivals(stream, rate=rate, seed=args.seed)
    print(f"rho={args.rho}  lambda={rate:.5f} req/s  "
          f"arrival span {arrivals[-1].arrives_at / 60:.1f} min  "
          f"resident={resident}  digest={workload.stream_digest(stream)[:16]}",
          flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.eval().to(device)
    engine = Qwen2Engine.from_hf_model(model)
    generator = baseline_mod.HuggingFaceGenerator(
        model, AutoTokenizer.from_pretrained(args.model), device)

    def executor_factory(allocator):
        cache = engine.new_paged_cache(allocator.num_blocks, allocator.block_size)
        return Qwen2Executor(engine, cache, allocator)

    # BOTH SIDES, EVERY PASS, ONE SESSION. Interleaved rather than run in two
    # blocks so any slow drift over the session lands on both equally.
    engine_passes, baseline_passes = [], []
    for index in range(WARMUP_PASSES + args.passes):
        measured = index >= WARMUP_PASSES
        tag = "measure" if measured else "warmup "
        e_obs, e_meta = run_engine(arrivals, executor_factory, args.max_num_seqs)
        print(f"  [{tag}] engine   wall {e_meta['wall_seconds'] / 60:5.1f} min  "
              f"ttft p50 {percentile([o.ttft for o in e_obs], .5):7.3f}s  "
              f"p99 {percentile([o.ttft for o in e_obs], .99):8.3f}s", flush=True)
        b_obs, b_meta = run_baseline(arrivals, generator, resident, sort_by_decode=True)
        print(f"  [{tag}] baseline wall {b_meta['wall_seconds'] / 60:5.1f} min  "
              f"ttft p50 {percentile([o.ttft for o in b_obs], .5):7.3f}s  "
              f"p99 {percentile([o.ttft for o in b_obs], .99):8.3f}s  "
              f"eff {b_meta['decode_efficiency']:.3f}", flush=True)
        if measured:
            engine_passes.append({**summarise(e_obs), **e_meta})
            baseline_passes.append({**summarise(b_obs), **b_meta,
                                    "queue_depths": b_meta["queue_depths"]})

    def median_of(passes, key):
        return statistics.median([p[key] for p in passes])

    payload = {
        "prereg": "docs/PREREG_arrival.md",
        "rho": args.rho,
        "lambda_req_per_s": rate,
        "saturated_sorted_service_rate": saturated_rate,
        "rectangle": args.rectangle,
        "resident": resident,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "stream_digest": workload.stream_digest(stream),
        "warmup_passes": WARMUP_PASSES,
        "measured_passes": args.passes,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "engine_passes": engine_passes,
        "baseline_passes": baseline_passes,
        "medians": {
            side: {k: median_of(passes, k) for k in
                   ("ttft_p50", "ttft_p99", "latency_p50", "latency_p99", "wall_seconds")}
            for side, passes in (("engine", engine_passes), ("baseline", baseline_passes))
        },
    }
    payload["medians"]["baseline"]["decode_efficiency"] = median_of(
        baseline_passes, "decode_efficiency")
    payload["medians"]["baseline"]["mean_queue_depth"] = median_of(
        baseline_passes, "mean_queue_depth")
    payload["ttft_p99_ratio_baseline_over_engine"] = (
        payload["medians"]["baseline"]["ttft_p99"] / payload["medians"]["engine"]["ttft_p99"])
    payload["latency_p50_ratio_baseline_over_engine"] = (
        payload["medians"]["baseline"]["latency_p50"] / payload["medians"]["engine"]["latency_p50"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
