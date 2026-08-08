"""Stage 1 of docs/PREREG_arrival.md: simulate arrivals, spend no GPU time.

The question this answers, and the only one: is there an offered load at which a
decode-sorted static batcher loses its sorting advantage while the engine keeps
its refill advantage? If there is no such load, Stage 2 does not run.

WHAT IS SIMULATED AND WHAT IS NOT. This is a discrete-event simulation of
SCHEDULING, not of the GPU. Per-step cost is taken from measurements already on
disk, so the simulation decides only who runs what and when, never how fast a
kernel is. That is deliberate: the hypothesis is about queueing, and a
simulation that also guessed at kernel timings would be answering a different
question badly.

THE FROZEN STREAM IS NOT TOUCHED. `bench/workload.py` supplies the same 256
requests with the same prompts and decode budgets. Arrival times are layered on
top from a seeded Poisson process, and both schedulers see identical arrivals.

THE TWO SCHEDULERS

  StaticBatcher   Fills a batch of up to `resident` from whatever is QUEUED,
                  optionally sorting the queue by decode length first. Runs the
                  batch until its LONGEST member finishes its budget, so every
                  shorter row keeps computing positions nobody reads. Admits
                  nothing until the batch drains. That last property is the one
                  the hypothesis turns on.

  ContinuousEngine  Admits on the next step whenever it has capacity, up to
                  `max_num_seqs`. A finished sequence frees its slot
                  immediately.

Both are given the same capacity so the comparison is about ordering and
admission, not about how many sequences fit.
"""

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from bench import workload

# THE COST MODEL, and the mistake it exists to correct.
#
# A first version of this file gave both schedulers the same per-row-step cost.
# That predicted the engine should beat the sorted batcher at saturation, which
# is the OPPOSITE of what is measured, so every number it produced was an
# artifact of the model rather than a property of the schedulers.
#
# What the measurements on disk actually show:
#
#   static batcher   5046 us per row-step at width 32, 6719 us at width 18
#   engine          10922 us per row-step, stable across all four cells
#
# The batcher's width dependence gives a two-parameter fit that reproduces both
# points exactly:
#
#   step_us = FIXED_US + PER_ROW_US * width      68819 + 2895 * width
#
# FIXED_US is weight loading, paid once per step whatever the width. PER_ROW_US
# is the work that scales. This matters at sub-saturation: a narrow batch pays
# the same FIXED_US over fewer rows, so per-row cost RISES as load falls, and a
# model without it would flatter whichever scheduler runs narrower.
#
# The engine pays the same FIXED_US -- same weights, same model -- and a larger
# PER_ROW_US, solved so that at its measured saturated width it reproduces
# 10922 us per row-step. That extra per-row cost is the paging: block-table
# gather, an explicit mask instead of the causal shortcut, and scatter back.
#
# NOTHING BELOW IS TRUSTED UNTIL `validate()` REPRODUCES ALL FOUR MEASURED CELLS.
FIXED_US = 68819.27
BASELINE_PER_ROW_US = 2895.331
ENGINE_PER_ROW_US = 0.0          # solved in calibrate()
PREFILL_SECONDS_PER_TOKEN = 1.0 / 60000.0


def step_seconds(width: int, per_row_us: float) -> float:
    """One forward step over `width` rows."""
    return (FIXED_US + per_row_us * width) * 1e-6


@dataclass
class Arrival:
    request_id: str
    prompt_len: int
    decode_len: int
    arrives_at: float


@dataclass
class Completion:
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


def poisson_arrivals(stream: Sequence, rate: float, seed: int) -> list[Arrival]:
    """Exponential inter-arrival times over the FROZEN stream, in stream order.

    The stream itself is untouched: same ids, same prompts, same decode budgets.
    Only a timestamp is attached.
    """
    rng = random.Random(seed)
    now = 0.0
    out = []
    for item in stream:
        now += rng.expovariate(rate)
        out.append(Arrival(item.request_id, len(item.prompt_token_ids),
                           item.max_new_tokens, now))
    return out


def run_static(arrivals: Sequence[Arrival], resident: int, sort_by_decode: bool):
    """Static batching. Admits nothing until the current batch fully drains."""
    pending = list(arrivals)
    queue: list[Arrival] = []
    now = 0.0
    done: list[Completion] = []
    row_steps = 0            # rows x steps actually executed
    useful_steps = 0         # rows x steps that produced a token someone wanted
    depths: list[int] = []

    while pending or queue:
        # Nothing queued: jump to the next arrival. An idle batcher waits.
        if not queue:
            if not pending:
                break
            now = max(now, pending[0].arrives_at)
        while pending and pending[0].arrives_at <= now:
            queue.append(pending.pop(0))
        if not queue:
            continue

        depths.append(len(queue))
        # THE SORT ONLY SEES WHAT IS QUEUED. This is the whole hypothesis.
        if sort_by_decode:
            queue.sort(key=lambda a: a.decode_len)
        batch = queue[:resident]
        del queue[:len(batch)]

        prefill = max(a.prompt_len for a in batch) * PREFILL_SECONDS_PER_TOKEN
        first_token_at = now + prefill
        steps = max(a.decode_len for a in batch)      # the straggler sets it
        row_steps += steps * len(batch)
        useful_steps += sum(a.decode_len for a in batch)
        finished_at = first_token_at + steps * step_seconds(len(batch), BASELINE_PER_ROW_US)

        for a in batch:
            done.append(Completion(a.request_id, a.arrives_at,
                                   first_token_at, finished_at))
        now = finished_at

    return done, row_steps, useful_steps, depths


def run_continuous(arrivals: Sequence[Arrival], max_num_seqs: int):
    """Continuous batching. A finished slot is refilled on the next step."""
    pending = list(arrivals)
    running: list[tuple[Arrival, int, float]] = []   # (req, remaining, first_tok)
    now = 0.0
    done: list[Completion] = []
    row_steps = 0
    useful_steps = 0
    widths: list[int] = []

    while pending or running:
        if not running and pending:
            now = max(now, pending[0].arrives_at)
        # Admit whatever has arrived and fits, immediately.
        while pending and len(running) < max_num_seqs and pending[0].arrives_at <= now:
            a = pending.pop(0)
            first = now + a.prompt_len * PREFILL_SECONDS_PER_TOKEN
            running.append((a, a.decode_len, first))
        if not running:
            continue

        # One decode step for every running row. No row runs past its budget,
        # which is the difference from the static batcher.
        width = len(running)
        now += step_seconds(width, ENGINE_PER_ROW_US)
        row_steps += width
        useful_steps += width
        widths.append(width)

        still = []
        for a, remaining, first in running:
            remaining -= 1
            if remaining <= 0:
                done.append(Completion(a.request_id, a.arrives_at, first, now))
            else:
                still.append((a, remaining, first))
        running = still

    return done, row_steps, useful_steps, widths


def calibrate() -> dict:
    """Solve ENGINE_PER_ROW_US so the sim reproduces the measured engine time.

    The engine's average running width is not a free parameter: it is whatever
    the scheduler does with the frozen stream at saturation. So run the engine
    at saturation with a placeholder cost, take the width trace, and solve

        measured_us_per_row_step = FIXED_US / mean_width + ENGINE_PER_ROW_US

    That anchors the engine to its measurement at the width it actually runs,
    and lets per-row cost rise correctly at low load where width is small.
    """
    global ENGINE_PER_ROW_US
    stream = workload.build_stream("mixed", 0, 64)
    useful = sum(r.max_new_tokens for r in stream)

    measured = []
    for path in sorted(Path("bench/results/batch-order").glob("*.json")):
        payload = json.loads(path.read_text())
        engine = next(l for l in payload["levels"] if l["engine"] == "engine")
        measured.append(engine["median_seconds"] / useful * 1e6)
    target_us_per_row = statistics.mean(measured)

    ENGINE_PER_ROW_US = BASELINE_PER_ROW_US          # placeholder for the trace
    everything_now = poisson_arrivals(stream, rate=1e9, seed=0)
    _, _, _, widths = run_continuous(everything_now, 64)
    mean_width = statistics.mean(widths)

    ENGINE_PER_ROW_US = target_us_per_row - FIXED_US / mean_width
    return {"engine_target_us_per_row_step": target_us_per_row,
            "engine_mean_width_at_saturation": mean_width,
            "engine_per_row_us": ENGINE_PER_ROW_US}


def validate() -> list[dict]:
    """Reproduce every measured cell. Nothing downstream counts until this does.

    A simulation that cannot recover a point it was calibrated near has no
    business extrapolating to loads nobody measured. The first version of this
    file predicted the engine beating the sorted batcher at saturation, which is
    the opposite of the measurement, and it would have been caught here.
    """
    stream = workload.build_stream("mixed", 0, 64)
    everything_now = poisson_arrivals(stream, rate=1e9, seed=0)
    useful = sum(r.max_new_tokens for r in stream)

    rows = []
    for path in sorted(Path("bench/results/batch-order").glob("*.json")):
        payload = json.loads(path.read_text())
        levels = {l["engine"]: l for l in payload["levels"]}
        resident = levels["baseline"]["max_resident"]
        sorted_order = "sorted" in path.stem

        b_done, b_rows, _, _ = run_static(everything_now, resident, sorted_order)
        e_done, _, _, _ = run_continuous(everything_now, 64)
        sim_b = max(c.finished_at for c in b_done)
        sim_e = max(c.finished_at for c in e_done)
        got = levels["engine"]["median_tokens_per_second"] / levels["baseline"]["median_tokens_per_second"]
        rows.append({
            "cell": path.stem,
            "measured_ratio": got,
            "simulated_ratio": sim_b / sim_e,
            "measured_baseline_s": levels["baseline"]["median_seconds"],
            "simulated_baseline_s": sim_b,
            "measured_engine_s": levels["engine"]["median_seconds"],
            "simulated_engine_s": sim_e,
        })
        r = rows[-1]
        r["ratio_error_pct"] = 100 * (r["simulated_ratio"] / r["measured_ratio"] - 1)
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def sweep(resident: int, max_num_seqs: int, loads: Sequence[float], seed: int):
    stream = workload.build_stream("mixed", 0, 64)

    # Saturated service rate: everything present at t=0 is the rate=infinity case.
    huge = poisson_arrivals(stream, rate=1e6, seed=seed)
    sat_sorted, _, _, _ = run_static(huge, resident, True)
    saturated_rate = len(stream) / max(c.finished_at for c in sat_sorted)

    rows = []
    for rho in loads:
        arrivals = poisson_arrivals(stream, rate=rho * saturated_rate, seed=seed)
        s_done, s_rows, s_useful, depths = run_static(arrivals, resident, True)
        a_done, a_rows, a_useful, _ = run_static(arrivals, resident, False)
        e_done, e_rows, e_useful, e_widths = run_continuous(arrivals, max_num_seqs)

        rows.append({
            "rho": rho,
            "mean_queue_depth_at_batch_formation": statistics.mean(depths),
            "median_queue_depth": statistics.median(depths),
            "batch_size": resident,
            "sorted_efficiency": s_useful / s_rows,
            "arrival_efficiency": a_useful / a_rows,
            "engine_efficiency": e_useful / e_rows,
            "engine_mean_width": statistics.mean(e_widths),
            "sorted_p50_ttft": percentile([c.ttft for c in s_done], 0.50),
            "sorted_p99_ttft": percentile([c.ttft for c in s_done], 0.99),
            "engine_p50_ttft": percentile([c.ttft for c in e_done], 0.50),
            "engine_p99_ttft": percentile([c.ttft for c in e_done], 0.99),
            "sorted_p50_latency": percentile([c.latency for c in s_done], 0.50),
            "sorted_p99_latency": percentile([c.latency for c in s_done], 0.99),
            "engine_p50_latency": percentile([c.latency for c in e_done], 0.50),
            "engine_p99_latency": percentile([c.latency for c in e_done], 0.99),
        })
        r = rows[-1]
        r["p99_ttft_ratio_sorted_over_engine"] = (
            r["sorted_p99_ttft"] / r["engine_p99_ttft"]
            if r["engine_p99_ttft"] > 0 else float("inf"))
        r["p50_latency_ratio_sorted_over_engine"] = (
            r["sorted_p50_latency"] / r["engine_p50_latency"]
            if r["engine_p50_latency"] > 0 else float("inf"))
    return saturated_rate, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resident", type=int, default=18)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("bench/results/arrival/stage1.json"))
    args = parser.parse_args(argv)

    cal = calibrate()
    print("CALIBRATION")
    for k, v in cal.items():
        print(f"  {k:38s} {v:12.2f}")

    print("\nVALIDATION: does the sim reproduce the four measured cells?")
    print(f"  {'cell':38s} {'measured':>9s} {'simulated':>10s} {'error':>8s}")
    worst = 0.0
    for r in validate():
        print(f"  {r['cell'][:38]:38s} {r['measured_ratio']:9.3f} "
              f"{r['simulated_ratio']:10.3f} {r['ratio_error_pct']:+7.1f}%")
        worst = max(worst, abs(r['ratio_error_pct']))
    print(f"  worst ratio error: {worst:.1f}%")
    if worst > 25.0:
        print("\n  VALIDATION FAILED. The model does not reproduce what was measured,")
        print("  so its sub-saturation predictions mean nothing. Stopping.")
        return 2

    loads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rate, rows = sweep(args.resident, args.max_num_seqs, loads, args.seed)

    print(f"\nsaturated sorted service rate: {rate:.4f} req/s\n")
    print(f"{'rho':>5} {'queue':>7} {'sortEff':>8} {'arrEff':>7} {'engEff':>7} "
          f"{'p99TTFT sorted':>15} {'engine':>9} {'ratio':>7} {'p50lat ratio':>13}")
    for r in rows:
        print(f"{r['rho']:5.2f} {r['mean_queue_depth_at_batch_formation']:7.1f} "
              f"{r['sorted_efficiency']:8.3f} {r['arrival_efficiency']:7.3f} "
              f"{r['engine_efficiency']:7.3f} {r['sorted_p99_ttft']:15.2f} "
              f"{r['engine_p99_ttft']:9.2f} {r['p99_ttft_ratio_sorted_over_engine']:7.2f} "
              f"{r['p50_latency_ratio_sorted_over_engine']:13.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "calibration": cal,
        "validation": validate(),
        "saturated_sorted_service_rate": rate,
        "resident": args.resident,
        "max_num_seqs": args.max_num_seqs,
        "seed": args.seed,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
