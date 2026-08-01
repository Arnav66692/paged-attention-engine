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
        "engine_tps": {"low": e_lo, "median": e_med, "high": e_hi},
        "baseline_tps": {"low": b_lo, "median": b_med, "high": b_hi},
        # Worst case for the engine over best case for the baseline, and back.
        "ratio_low": e_lo / b_hi,
        "ratio_median": e_med / b_med,
        "ratio_high": e_hi / b_lo,
        "engine_preemptions": engine.get("preemption_count"),
        "engine_peak_blocks": engine.get("peak_blocks_used"),
    }


def compare_arms(arm_a: dict, arm_b: dict) -> dict:
    """Do the two arms agree, cell by cell? Overlap of the ratio intervals.

    Overlap is the honest test. Two point medians will essentially never be
    equal, and calling a 3% gap a 'difference' when each arm's own passes span
    8% would be reading noise as signal.
    """
    cells = {}
    for name in sorted(set(arm_a) | set(arm_b)):
        a, b = arm_a.get(name), arm_b.get(name)
        if not a or not b or not a.get("resolved") or not b.get("resolved"):
            cells[name] = {"comparable": False,
                           "why": "one or both arms did not resolve this cell"}
            continue
        overlap = a["ratio_low"] <= b["ratio_high"] and b["ratio_low"] <= a["ratio_high"]
        cells[name] = {
            "comparable": True,
            "arm_a_ratio": [a["ratio_low"], a["ratio_median"], a["ratio_high"]],
            "arm_b_ratio": [b["ratio_low"], b["ratio_median"], b["ratio_high"]],
            "intervals_overlap": overlap,
            "median_shift_b_over_a": b["ratio_median"] / a["ratio_median"],
            "direction": (
                "arms agree within spread" if overlap
                else ("faster kernel RAISES the ratio (MATH was a floor)"
                      if b["ratio_median"] > a["ratio_median"]
                      else "faster kernel LOWERS the ratio (MATH was a ceiling)")
            ),
        }
    comparable = [c for c in cells.values() if c.get("comparable")]
    return {
        "cells": cells,
        "all_comparable_cells_agree": bool(comparable) and all(
            c["intervals_overlap"] for c in comparable
        ),
        "num_comparable": len(comparable),
        "verdict": (
            "UNDETERMINED -- no cell resolved in both arms" if not comparable
            else "kernel choice does not bias the ratio at this resolution"
            if all(c["intervals_overlap"] for c in comparable)
            else "kernel choice DOES move the ratio; see per-cell direction"
        ),
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
        "--compare-only", action="store_true",
        help=(
            "run nothing; join the result files already in --out-dir. The "
            "matrix is split across Kaggle sessions, so the comparison has to "
            "be able to happen after the fact."
        ),
    )
    args = parser.parse_args(argv)

    if args.compare_only:
        return _compare_only(args.out_dir)

    arms = args.arm or sorted(ARMS)
    cells = [c for c in CELLS if not args.cell or cell_name(c) in args.cell]
    if not cells:
        print(f"no cells matched {args.cell}; known: "
              f"{[cell_name(c) for c in CELLS]}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra = ["--smoke"] if args.smoke else []

    ratios: dict[str, dict] = {arm: {} for arm in arms}
    identity: dict[str, dict] = {}
    for arm in arms:
        for cell in cells:
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
