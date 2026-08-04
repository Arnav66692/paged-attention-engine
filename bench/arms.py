"""Two sweeps that differ only in which SDPA kernel BOTH sides run.

WHY THIS EXISTS. The Gate 1 ratio is a claim about paging. It is measured as
engine throughput over baseline throughput, and both sides call the same
`scaled_dot_product_attention`. Torch picks the kernel from the input SHAPE, so
the two sides can silently end up on different kernels, and then the ratio is
part paging and part kernel with no way to separate them after the fact.

Worse, the baseline's kernel DEPENDS ON THE STREAM. HuggingFace takes the
`enable_gqa` path only when the attention mask is None, and the mask is None
only when nothing was padded:

    control -- every prompt is 512 tokens, no padding, mask dropped to None,
      HF uses enable_gqa. On a T4 that is MATH. The engine's masked fused call
      is also MATH. Symmetric, by accident.
    mixed -- 224 distinct prompt lengths, real mask, HF pre-expands K/V. On a
      T4 that is EFFICIENT_ATTENTION, the FAST kernel, while the engine stays on
      MATH. Asymmetric, against the engine, and nobody chose it.

THE QUESTION, AND WHY IT CANNOT BE REASONED OUT. Suppose both sides share one
kernel. Does making that kernel slower move the ratio, and which way?

  Toward 1 (a slow kernel makes the T4 number a FLOOR). With `W` the
  batch-independent per-step weight loading and `k` the per-token cost,

      ratio = (B_e / B_b) . (W + B_b.k) / (W + B_e.k)

  and raising `k` on both sides drives that toward 1. The engine's advantage
  comes from amortising `W` over a larger batch; anything that makes per-token
  work dominate `W` shrinks the advantage.

  Away from 1 (a slow kernel makes it a CEILING). This needs the baseline to do
  more attention work PER USEFUL TOKEN. Rectangle sizing does not do that -- it
  is a KV cache allocation and HF masks over the true kv_len. Static-batch
  stragglers do: a finished sequence keeps being recomputed until its whole
  batch drains, and a more expensive kernel makes that waste more expensive.

Both mechanisms are real and they have opposite signs. So: measure.

    ARM A  engine GQA_PATH=sdpa       baseline unpatched
    ARM B  engine GQA_PATH=repeat_kv  baseline forced to pre-expand K/V

Arm B's patch is gated on token identity -- see `baseline.assert_token_identical`.
A baseline whose answers moved is a different baseline and may not be timed.

READING THE RESULT. If the two arms' ratio intervals overlap, kernel choice does
not bias the result at the resolution this harness can see, and the headline
ships with no hedge. If they do not overlap, the direction and size of the gap
IS the finding and goes in the report. Neither outcome is failure; only asserting
one without running both is.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from bench import baseline as baseline_mod
from bench.workload import PRESSURE_C

ENGINE = "engine"
BASELINE = "baseline"

# The two configurations. `gqa_path` is an environment variable because
# `engine.attention.GQA_PATH` is read at import, so an arm is a process.
ARMS = {
    "A": {
        "gqa_path": "sdpa",
        "baseline_expand_kv": False,
        "intent": "engine on enable_gqa; baseline left exactly as HuggingFace ships it",
    },
    "B": {
        "gqa_path": "repeat_kv",
        "baseline_expand_kv": True,
        "intent": "both sides forced onto pre-expanded K/V, hence the same kernel",
    },
}

# Both headlines, both rectangle sizings. The control headline is the quotable
# number and is only legal at PRESSURE_C; mixed is the realistic stream.
CELLS = tuple(
    {
        "stream": stream,
        "rectangle": rectangle,
        "headline": stream == "control",
        "concurrency": PRESSURE_C,
    }
    for stream in ("control", "mixed")
    for rectangle in baseline_mod.RECTANGLE_SIZINGS
)


# GAP FILL. The gate wants a win at three or more levels, and only arm B mixed
# wins, so extra levels go there alone. C=32 and C=87 -- NOT lower.
#
# The reason is the baseline's residency rule: `max_resident` is
# min(rectangle limit, C). Below the rectangle cap BOTH engines are capped
# identically by C, the engine has no residency advantage left, and the level
# would buy a loss that says nothing about paging. Above it the engine keeps
# admitting and the baseline cannot:
#
#   rect3072  C=32 engine 32 vs baseline 18   C=64  64/18   C=87  87/18
#   rect1757  C=32 engine 32 vs baseline 32   C=64  64/32   C=87  87/32
#                       ^^ tied -- THE CROSSOVER
#
# rect1757 at C=32 sits exactly on the cap with both sides at 32 resident. A
# loss or a tie is the EXPECTED result there and it is the point that shows the
# mechanism, not a failure. It is reported as the crossover.
GAP_FILL_CELLS = tuple(
    {
        "stream": "mixed",
        "rectangle": rectangle,
        "headline": False,
        "concurrency": concurrency,
        "arms": ("B",),        # arm A mixed is asymmetric; it may not be timed here
    }
    for concurrency in (32, 87)
    for rectangle in baseline_mod.RECTANGLE_SIZINGS
)

ALL_CELLS = CELLS + GAP_FILL_CELLS


def arms_for(cell: dict) -> tuple:
    """Which arms this cell may be run under. Default both."""
    return tuple(cell.get("arms") or sorted(ARMS))


def cell_name(cell: dict) -> str:
    return f"{cell['stream']}-rect{cell['rectangle']}-C{cell['concurrency']}"


def run_cell(arm: str, cell: dict, out_dir: Path, extra: Sequence[str] = ()) -> Path:
    """One `bench.run` invocation, in its own process so GQA_PATH takes."""
    config = ARMS[arm]
    out = out_dir / f"arm{arm}-{cell_name(cell)}.json"
    argv = [
        sys.executable, "-m", "bench.run",
        "--stream", cell["stream"],
        "-C", str(cell["concurrency"]),
        "--rectangle", str(cell["rectangle"]),
        "--out", str(out),
        *extra,
    ]
    if cell["headline"]:
        argv.append("--headline")
    if config["baseline_expand_kv"]:
        argv.append("--baseline-expand-kv")

    env = dict(os.environ)
    env["PAGED_ATTENTION_GQA_PATH"] = config["gqa_path"]

    print(f"\n=== ARM {arm} :: {cell_name(cell)} "
          f"(GQA_PATH={config['gqa_path']}, "
          f"baseline_expand_kv={config['baseline_expand_kv']}) ===", flush=True)
    completed = subprocess.run(argv, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"arm {arm} cell {cell_name(cell)} exited {completed.returncode}. "
            "A failed cell is not a zero; it is a missing measurement."
        )
    return out


# --- reading a result ------------------------------------------------------


def _tps_bounds(level: dict) -> tuple[float, float, float] | None:
    """(low, median, high) tokens/second for a level, from its own passes.

    The harness reports a median and a range; a ratio built from two medians
    inherits both spreads and reporting it as a point would hide that.
    """
    if level.get("unresolved") or not level.get("throughput_valid"):
        return None
    # KEY NAME IS LOAD-BEARING. `LevelResult.describe()` writes `pass_seconds`.
    # Reading a key that does not exist yields None, every cell reports
    # "not comparable", and the run looks orderly while measuring nothing.
    # `tests/test_arms.py` pins these names against `describe()` itself.
    seconds = level.get("pass_seconds") or []
    tokens = level.get("generated_tokens")
    median = level.get("median_tokens_per_second")
    if not seconds or not tokens or median is None:
        return None
    # Longest pass is the lowest throughput.
    return (tokens / max(seconds), median, tokens / min(seconds))


def ratio_for(payload: dict) -> dict | None:
    """Engine-over-baseline throughput, as an interval, for one result file.

    A result whose baseline was patched must carry the token-identity attestation
    that licensed the patch. `bench.run` cannot produce one without it -- the
    gate raises -- but `--compare-only` reads whatever files are in a directory,
    and a timing from a patched-but-unverified baseline is exactly the number
    this experiment must not report.
    """
    provenance = payload.get("provenance") or {}
    if provenance.get("baseline_kv_expansion_forced"):
        identity = payload.get("baseline_token_identity") or {}
        if not identity.get("token_identical"):
            return {
                "resolved": False,
                "why": (
                    "baseline was patched to pre-expand K/V but the result "
                    "carries no token-identity attestation. Arm B is void for "
                    "this cell; a patched baseline that was never verified is "
                    "not this baseline."
                ),
            }

    levels = payload.get("levels", [])
    engine = next((lv for lv in levels if lv["engine"] == ENGINE), None)
    baseline = next((lv for lv in levels if lv["engine"] == BASELINE), None)
    if engine is None or baseline is None:
        return None
    e_bounds = _tps_bounds(engine)
    b_bounds = _tps_bounds(baseline)
    if e_bounds is None or b_bounds is None:
        return {
            "resolved": False,
            "why": "a level was UNRESOLVED or not throughput_valid",
            "engine_unresolved": engine.get("unresolved"),
            "baseline_unresolved": baseline.get("unresolved"),
        }
    e_lo, e_med, e_hi = e_bounds
    b_lo, b_med, b_hi = b_bounds
    return {
        "resolved": True,
        # Did Arm B move the BASELINE on this cell, or only the engine?
        "baseline_expands_kv_unpatched": provenance.get("baseline_expands_kv_unpatched"),
        "gqa_path": provenance.get("gqa_path"),
        # WHAT THE RUN ACTUALLY DID. Two arms are only comparable if they did
        # the same work; a ratio shift means "the kernel" only once "and nothing
        # else" is established rather than assumed. Decoding is greedy and the
        # workload contract is exact lengths, so every one of these is expected
        # to match exactly across arms, and any that does not voids the cell.
        "work": {
            "workload_digest": provenance.get("workload_digest"),
            "engine_tokens": engine.get("generated_tokens"),
            "baseline_tokens": baseline.get("generated_tokens"),
            "engine_preemptions": engine.get("preemption_count"),
            "engine_peak_blocks": engine.get("peak_blocks_used"),
            "engine_rejected": engine.get("num_rejected"),
            "baseline_rejected": baseline.get("num_rejected"),
            "baseline_max_resident": baseline.get("max_resident"),
        },
        "engine_tps": {"low": e_lo, "median": e_med, "high": e_hi},
        "baseline_tps": {"low": b_lo, "median": b_med, "high": b_hi},
        # Worst case for the engine over best case for the baseline, and back.
        "ratio_low": e_lo / b_hi,
        "ratio_median": e_med / b_med,
        "ratio_high": e_hi / b_lo,
        "engine_preemptions": engine.get("preemption_count"),
        "engine_peak_blocks": engine.get("peak_blocks_used"),
    }


# What an A-vs-B comparison MEANS depends on whether Arm B moved the baseline.
#
# Arm B's patch only bites where HF would otherwise have taken `enable_gqa`,
# which is only where the batch is prompt-uniform. So the cells are not four
# replicates of one contrast, and pooling them into a single verdict would be
# wrong:
#
#   BOTH SIDES MOVED (control: baseline enable_gqa/MATH -> pre-expanded/
#     EFFICIENT, engine likewise). Arm A and Arm B are each internally
#     symmetric, and they differ only in WHICH shared kernel both sides ran.
#     THIS IS THE KERNEL-BIAS QUESTION and only these cells answer it.
#
#   ONLY THE ENGINE MOVED (mixed: HF's ragged batches already pre-expand, so
#     the patch is inert and the baseline runs byte-identical code in both
#     arms). Arm A is ASYMMETRIC -- engine on MATH, baseline already on
#     EFFICIENT -- and Arm B removes that. The A-to-B shift here is a one-sided
#     engine speedup, and it necessarily raises the ratio. Reading it as
#     evidence about kernel bias would be reading a known handicap being lifted.
#     It is still worth reporting: it prices what the unintended asymmetry cost.
BOTH_MOVED = "both_sides_moved"
ONLY_ENGINE_MOVED = "only_engine_moved"
UNKNOWN_MOVE = "unrecorded"


def _classify(arm_a: dict) -> str:
    """Which question this cell answers, from Arm A's measured baseline path."""
    expands = arm_a.get("baseline_expands_kv_unpatched")
    if expands is None:
        return UNKNOWN_MOVE
    return ONLY_ENGINE_MOVED if expands else BOTH_MOVED


def compare_arms(arm_a: dict, arm_b: dict) -> dict:
    """Do the two arms agree, cell by cell? Overlap of the ratio intervals.

    Overlap is the honest test. Two point medians will essentially never be
    equal, and calling a 3% gap a 'difference' when each arm's own passes span
    8% would be reading noise as signal.

    The verdict on kernel bias is drawn ONLY from cells where both sides moved.
    See the note above: cells where the patch was inert on the baseline measure
    something else, and averaging the two would let a lifted handicap masquerade
    as a kernel effect.
    """
    cells = {}
    for name in sorted(set(arm_a) | set(arm_b)):
        a, b = arm_a.get(name), arm_b.get(name)
        if not a or not b or not a.get("resolved") or not b.get("resolved"):
            cells[name] = {"comparable": False,
                           "why": "one or both arms did not resolve this cell"}
            continue
        # THE CELL IS VOID IF THE ARMS DID DIFFERENT WORK. Everything downstream
        # reads a ratio shift as a kernel effect, and that reading is only
        # available once the two arms are known to have consumed the same
        # stream, generated the same tokens, preempted the same number of times
        # and rejected the same requests.
        work_a, work_b = a.get("work") or {}, b.get("work") or {}
        differing = {
            key: (work_a.get(key), work_b.get(key))
            for key in set(work_a) | set(work_b)
            if work_a.get(key) != work_b.get(key)
        }
        if differing:
            cells[name] = {
                "comparable": False,
                "why": (
                    "the two arms did DIFFERENT work, so a ratio shift between "
                    f"them is not attributable to the kernel: {differing}"
                ),
            }
            continue

        overlap = a["ratio_low"] <= b["ratio_high"] and b["ratio_low"] <= a["ratio_high"]
        kind = _classify(a)
        cells[name] = {
            "comparable": True,
            "kind": kind,
            "answers_kernel_bias": kind == BOTH_MOVED,
            "arm_a_ratio": [a["ratio_low"], a["ratio_median"], a["ratio_high"]],
            "arm_b_ratio": [b["ratio_low"], b["ratio_median"], b["ratio_high"]],
            "intervals_overlap": overlap,
            "median_shift_b_over_a": b["ratio_median"] / a["ratio_median"],
            "direction": (
                "arms agree within spread" if overlap
                else ("faster kernel RAISES the ratio (MATH was a floor)"
                      if b["ratio_median"] > a["ratio_median"]
                      else "faster kernel LOWERS the ratio (MATH was a ceiling)")
            ) if kind == BOTH_MOVED else (
                "Arm A was asymmetric here (engine MATH, baseline already "
                "EFFICIENT); this shift is that handicap being lifted, not "
                "kernel bias"
            ),
        }

    comparable = [c for c in cells.values() if c.get("comparable")]
    deciding = [c for c in comparable if c["kind"] == BOTH_MOVED]
    asymmetry = [c for c in comparable if c["kind"] == ONLY_ENGINE_MOVED]
    unknown = [c for c in comparable if c["kind"] == UNKNOWN_MOVE]

    agree = bool(deciding) and all(c["intervals_overlap"] for c in deciding)
    return {
        "cells": cells,
        "num_comparable": len(comparable),
        "num_deciding": len(deciding),
        "kernel_bias": {
            "from_cells": [n for n, c in cells.items()
                           if c.get("kind") == BOTH_MOVED],
            "all_agree": agree,
            "verdict": (
                "UNDETERMINED -- no cell where BOTH sides changed kernel "
                "resolved in both arms" if not deciding
                else "kernel choice does not bias the ratio at this resolution"
                if agree
                else "kernel choice DOES move the ratio; see per-cell direction"
            ),
        },
        "asymmetry_cost": {
            "from_cells": [n for n, c in cells.items()
                           if c.get("kind") == ONLY_ENGINE_MOVED],
            "note": (
                "Arm A ran the engine on MATH against a baseline already on "
                "EFFICIENT_ATTENTION. These shifts price that handicap; they "
                "are NOT evidence about kernel bias."
            ),
            "median_shift_b_over_a": [c["median_shift_b_over_a"] for c in asymmetry],
        },
        "cells_with_unrecorded_baseline_path": len(unknown),
    }


def _compare_only(out_dir: Path) -> int:
    """Join result files written by separate sessions.

    Result filenames carry the arm and the cell -- `armA-control-rect3072-C64`
    -- so the join needs no external index. A cell present in only one arm is
    reported as such rather than dropped, because a silently missing cell is how
    a partial matrix reads as a complete one.
    """
    ratios: dict[str, dict] = {}
    identity: dict[str, dict] = {}
    found = sorted(out_dir.glob("arm*-*.json"))
    if not found:
        print(f"no arm result files in {out_dir}", file=sys.stderr)
        return 2

    for path in found:
        stem = path.stem
        arm, _, cell = stem[3:].partition("-")
        if not arm or not cell:
            continue
        payload = json.loads(path.read_text())
        ratios.setdefault(arm, {})[cell] = ratio_for(payload)
        if payload.get("baseline_token_identity"):
            identity[stem] = payload["baseline_token_identity"]

    everywhere = set.intersection(*(set(v) for v in ratios.values())) if ratios else set()
    partial = sorted({c for v in ratios.values() for c in v} - everywhere)

    summary = {
        "joined_from": [p.name for p in found],
        "ratios": ratios,
        "baseline_token_identity": identity,
        "cells_missing_from_an_arm": partial,
        "comparison": (
            compare_arms(ratios["A"], ratios["B"])
            if "A" in ratios and "B" in ratios
            else {"verdict": f"only arms {sorted(ratios)} present; no comparison"}
        ),
    }
    out = out_dir / "arms_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"joined {len(found)} result files -> {out}")
    if partial:
        print(f"INCOMPLETE: cells present in only one arm: {partial}")
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))
    return 0


def merge_for_plot(out_dir: Path, arm: str, stream: str, rectangle: int) -> dict:
    """One plot-ready payload from the per-cell files of a single config.

    `bench/plot.py` draws throughput against concurrency from ONE result file
    holding every level. The sweep writes one file per cell so a dead session
    loses one cell rather than a matrix, so the levels have to be re-joined
    here. Only cells from the SAME arm, stream and rectangle are merged: a plot
    mixing arms would put two different kernels on one line.

    Provenance is taken from the lowest level and checked against the rest --
    if the cells disagree on anything that decides what was measured, that is
    an error, not something to average over.
    """
    found = []
    for path in sorted(out_dir.glob(f"arm{arm}-{stream}-rect{rectangle}-C*.json")):
        payload = json.loads(path.read_text())
        level = payload["levels"][0]["concurrency"] if payload.get("levels") else 0
        found.append((level, path, payload))
    if not found:
        raise FileNotFoundError(
            f"no arm{arm} {stream} rect{rectangle} cells in {out_dir}"
        )
    found.sort()

    PINNED = ("gqa_path", "baseline_kv_expansion_forced", "dtype", "model_id",
              "model_sha", "num_gpu_blocks", "block_size", "kv_pool_bytes",
              "max_num_batched_tokens", "gpu_name", "torch_version")
    reference = found[0][2]["provenance"]
    for level, path, payload in found[1:]:
        for key in PINNED:
            if payload["provenance"].get(key) != reference.get(key):
                raise ValueError(
                    f"{path.name} disagrees with {found[0][1].name} on "
                    f"{key!r}: {payload['provenance'].get(key)!r} vs "
                    f"{reference.get(key)!r}. These cells did not measure the "
                    "same thing and must not share a plot."
                )

    levels = [lv for _, _, payload in found for lv in payload["levels"]]
    return {
        "smoke": False,
        "throughput_valid": all(lv.get("throughput_valid") for lv in levels),
        "warmup_passes": found[0][2].get("warmup_passes"),
        "measured_passes": found[0][2].get("measured_passes"),
        "escalated_passes": found[0][2].get("escalated_passes"),
        "spread_limit": found[0][2].get("spread_limit"),
        "pressure_c": found[0][2].get("pressure_c"),
        "provenance": reference,
        "levels": levels,
        "merged_from": [path.name for _, path, _ in found],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("bench/results/arms"))
    parser.add_argument("--arm", choices=sorted(ARMS), action="append", default=None)
    parser.add_argument(
        "--cell", action="append", default=None,
        help="restrict to named cells, e.g. control-rect3072-C64",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--merge-for-plot", nargs=3, metavar=("ARM", "STREAM", "RECTANGLE"),
        help="join one config's per-cell files into a plot-ready payload",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help=(
            "run nothing; join the result files already in --out-dir. The "
            "matrix is split across Kaggle sessions, so the comparison has to "
            "be able to happen after the fact."
        ),
    )
    args = parser.parse_args(argv)

    if args.merge_for_plot:
        arm, stream, rectangle = args.merge_for_plot
        payload = merge_for_plot(args.out_dir, arm, stream, int(rectangle))
        out = args.out_dir / f"plot-arm{arm}-{stream}-rect{rectangle}.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"merged {payload['merged_from']} -> {out}")
        return 0

    if args.compare_only:
        return _compare_only(args.out_dir)

    arms = args.arm or sorted(ARMS)
    cells = [c for c in ALL_CELLS if not args.cell or cell_name(c) in args.cell]
    if not cells:
        print(f"no cells matched {args.cell}; known: "
              f"{[cell_name(c) for c in ALL_CELLS]}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra = ["--smoke"] if args.smoke else []

    ratios: dict[str, dict] = {arm: {} for arm in arms}
    identity: dict[str, dict] = {}
    for arm in arms:
        for cell in cells:
            if arm not in arms_for(cell):
                print(f"skipping arm {arm} for {cell_name(cell)}: this cell "
                      f"declares arms={arms_for(cell)}", flush=True)
                continue
            path = run_cell(arm, cell, args.out_dir, extra)
            payload = json.loads(path.read_text())
            ratios[arm][cell_name(cell)] = ratio_for(payload)
            if payload.get("baseline_token_identity"):
                identity[f"arm{arm}-{cell_name(cell)}"] = payload["baseline_token_identity"]

    summary = {
        "arms": {arm: ARMS[arm] for arm in arms},
        "cells": [cell_name(c) for c in cells],
        "ratios": ratios,
        "baseline_token_identity": identity,
        "comparison": (
            compare_arms(ratios["A"], ratios["B"])
            if "A" in ratios and "B" in ratios else
            {"verdict": "only one arm run; no comparison"}
        ),
        "smoke": args.smoke,
    }
    out = args.out_dir / "arms_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
