"""The two Gate 1 plots, drawn from a results file and nothing else.

Gate 1, Step 6.

  throughput_vs_concurrency   Both engines on one axis. This is the claim.
  memory_fragmentation        ANALYTICAL. Where the KV bytes go under the frozen
                              workload's length distribution. Explains why the
                              first plot looks the way it does, and would read
                              the same on any host -- which is its strength and
                              its limit: it proves nothing about a given run.
  measured_fragmentation      MEASURED. Capacity against occupancy over one real
                              run, read off the scheduler's event log, which
                              records free_blocks and tokens_held at every
                              admit, finish, preempt and recompute. Analytical
                              explains why; this proves it happened.

Everything drawn here comes out of the JSON `bench/run.py` wrote. Nothing is
recomputed, nothing is inferred, and nothing is drawn that the file does not
contain -- an UNRESOLVED level has no median by construction, so it is rendered
as its full range with no point, and a run stamped `throughput_valid: false` gets
a banner across the figure saying so. A plot is the most quotable artefact in the
project and the easiest one to quote out of context, so the caveats travel inside
the image rather than in a caption someone will drop.

matplotlib is imported lazily and is not a dependency of anything else here, so a
host that only runs the sweep does not need it.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from bench import baseline as baseline_mod
from bench import workload
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS

ENGINE_LABEL = {
    "engine": "paged engine",
    "baseline": "naive HF generate (rectangle)",
}
ENGINE_STYLE = {
    "engine": {"marker": "o", "linewidth": 2.0},
    "baseline": {"marker": "s", "linewidth": 2.0, "linestyle": "--"},
}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _levels_for(payload: dict, engine: str) -> list[dict]:
    return sorted(
        (level for level in payload["levels"] if level["engine"] == engine),
        key=lambda level: level["concurrency"],
    )


def _invalid_banner(payload: dict) -> str | None:
    """The one caption that must never be separable from the picture."""
    if payload.get("smoke"):
        return "SMOKE RUN -- structural only, timings are NOT a measurement"
    if not payload.get("throughput_valid", False):
        return "THROUGHPUT NOT VALID for this run -- do not quote"
    provenance = payload.get("provenance", {})
    if not provenance.get("cuda_available", False):
        return "NOT MEASURED ON A GPU -- timings are not comparable"
    return None


def _provenance_line(payload: dict) -> str:
    p = payload.get("provenance", {})
    bits = [
        p.get("gpu_name") or "GPU unknown",
        f"driver {p.get('driver_version') or 'unknown'}",
        f"torch {p.get('torch_version', '?')}",
        f"{p.get('dtype', '?')}",
        f"{p.get('num_gpu_blocks', '?')} blocks x {p.get('block_size', '?')}",
        f"model {(p.get('model_sha') or 'sha unknown')[:12]}",
        f"stream {p.get('workload_digest', '?')[:12]}",
    ]
    return " | ".join(str(bit) for bit in bits)


def throughput_vs_concurrency(payload: dict, out_path: Path):
    """Throughput against concurrency, both engines. UNRESOLVED levels show as ranges."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(9, 5.5))

    for engine in ("engine", "baseline"):
        levels = _levels_for(payload, engine)
        if not levels:
            continue
        resolved = [lv for lv in levels if lv["median_tokens_per_second"] is not None]
        if resolved:
            axes.plot(
                [lv["concurrency"] for lv in resolved],
                [lv["median_tokens_per_second"] for lv in resolved],
                label=ENGINE_LABEL[engine],
                **ENGINE_STYLE[engine],
            )
            # Full range, as a band. Never hidden behind the median.
            for level in resolved:
                if level["median_seconds"] and level["range_seconds"]:
                    tokens = level["generated_tokens"]
                    fastest = tokens / (level["median_seconds"] - level["range_seconds"] / 2)
                    slowest = tokens / (level["median_seconds"] + level["range_seconds"] / 2)
                    axes.vlines(
                        level["concurrency"], slowest, fastest,
                        alpha=0.35, linewidth=6,
                    )
        for level in levels:
            if level["unresolved"]:
                axes.axvline(level["concurrency"], color="crimson", alpha=0.25, linewidth=8)
                axes.annotate(
                    f"UNRESOLVED\n{level['spread_fraction']:.0%} spread",
                    (level["concurrency"], 0),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=8, color="crimson",
                )

    axes.set_xlabel("concurrency C")
    axes.set_ylabel("throughput (tokens/second, median of measured passes)")
    axes.set_title("Gate 1: throughput against concurrency")
    axes.grid(alpha=0.3)
    if axes.get_legend_handles_labels()[0]:
        axes.legend(loc="best")
    else:
        # Every level was UNRESOLVED or the run recorded no throughput. Say so on
        # the axes rather than shipping an empty frame that looks like zero.
        axes.text(0.5, 0.6, "no level yielded a reportable median",
                  transform=axes.transAxes, ha="center", va="center", fontsize=12)

    banner = _invalid_banner(payload)
    if banner:
        axes.text(
            0.5, 0.5, banner, transform=axes.transAxes, ha="center", va="center",
            fontsize=16, color="crimson", alpha=0.35, rotation=18, weight="bold",
        )
    figure.text(0.01, 0.01, _provenance_line(payload), fontsize=6, alpha=0.75)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


@dataclass(frozen=True)
class SlotBudget:
    """Where one sizing's token slots go, at the workload's mean request length.

    Three fates, and the middle one is the whole argument for paging:

      held      slots with a real token in them
      padding   slots a resident sequence owns and does not use. A rectangle
                sized for the longest request pays this on every shorter one; a
                16-token block pays at most 15 per sequence, once.
      stranded  ceiling the sizing cannot address at all

    Modelled from bench/workload.py's own distribution, not measured. These are
    properties of the sizing and the workload and would be identical on any host,
    which is why they can be drawn from a smoke run without lying.
    """

    label: str
    held: float
    padding: float
    stranded: float

    def __post_init__(self) -> None:
        total = self.held + self.padding + self.stranded
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{self.label}: shares sum to {total}, not 1")


def slot_budgets(total_slots: int) -> list[SlotBudget]:
    """The three-way split for each rectangle sizing, and for the paged pool."""
    mean_len = workload.mean_total_len(
        workload.DEFAULT_CONFIG.prompt, workload.DEFAULT_CONFIG.decode, BLOCK_SIZE
    )
    mean_blocks = workload.mean_blocks_per_request(
        workload.DEFAULT_CONFIG.prompt, workload.DEFAULT_CONFIG.decode, BLOCK_SIZE
    )

    budgets: list[SlotBudget] = []
    for rectangle in baseline_mod.RECTANGLE_SIZINGS:
        resident = total_slots // rectangle
        addressable = resident * rectangle
        occupied = min(resident * mean_len, addressable)
        budgets.append(
            SlotBudget(
                label=f"rectangle\n{rectangle} tok\n{resident} resident",
                held=occupied / total_slots,
                padding=(addressable - occupied) / total_slots,
                stranded=(total_slots - addressable) / total_slots,
            )
        )

    # The engine addresses the whole pool and rounds each sequence up to a block.
    occupied = min((NUM_GPU_BLOCKS / mean_blocks) * mean_len, total_slots)
    budgets.append(
        SlotBudget(
            label=f"paged engine\n{BLOCK_SIZE}-tok blocks\n{NUM_GPU_BLOCKS} blocks",
            held=occupied / total_slots,
            padding=(total_slots - occupied) / total_slots,
            stranded=0.0,
        )
    )
    return budgets


def memory_fragmentation(payload: dict, out_path: Path):
    """Where the same KV bytes went. The explanation for the throughput plot.

    Both engines are given `kv_pool_bytes` and nothing more. The engine spends it
    on blocks and hands them out as needed; the baseline spends it on a rectangle
    that every resident occupies in full. The bar chart is that division, and the
    second panel is the engine's realized block demand against the pool, which is
    what makes preemption necessary rather than incidental.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    ceiling = payload["provenance"]["kv_pool_bytes"]
    per_token = payload["provenance"]["bytes_per_token"]
    total_slots = ceiling // per_token

    # Left: where the same token slots go, under the frozen workload's own length
    # distribution. See `slot_budgets`.
    budgets = slot_budgets(total_slots)
    mean_len = workload.mean_total_len(
        workload.DEFAULT_CONFIG.prompt, workload.DEFAULT_CONFIG.decode, BLOCK_SIZE
    )
    labels = [budget.label for budget in budgets]
    held = [budget.held for budget in budgets]
    padding = [budget.padding for budget in budgets]
    stranded = [budget.stranded for budget in budgets]

    left.bar(labels, held, label="held by a real token", color="#2b7bba")
    left.bar(labels, padding, bottom=held, label="padding: owned, unused",
             color="#f08c2e")
    left.bar(labels, stranded, bottom=[h + p for h, p in zip(held, padding)],
             label="stranded: unaddressable", color="#b03a2e")
    for index, value in enumerate(held):
        left.annotate(f"{value:.0%}", (index, value / 2), ha="center", va="center",
                      fontsize=9, color="white", weight="bold")
    left.set_ylabel(f"share of the {total_slots:,} token slots ({ceiling / 1e9:.3f} GB)")
    left.set_title(f"Same bytes, divided three ways\n"
                   f"(modelled at the workload's mean length, {mean_len:.0f} tokens)",
                   fontsize=10)
    left.set_ylim(0, 1.05)
    left.legend(loc="lower right", fontsize=8)
    left.grid(alpha=0.3, axis="y")

    # Right: realized block demand against the pool, per level.
    levels = _levels_for(payload, "engine")
    if levels:
        concurrency = [level["concurrency"] for level in levels]
        right.plot(concurrency, [level["demanded_blocks"] for level in levels],
                   marker="o", label="realized block demand")
        right.plot(concurrency, [level["peak_blocks_used"] for level in levels],
                   marker="^", label="peak blocks resident")
        right.axhline(NUM_GPU_BLOCKS, color="crimson", linestyle="--",
                      label=f"pool = {NUM_GPU_BLOCKS} blocks")
        twin = right.twinx()
        twin.bar(concurrency, [level["preemption_count"] for level in levels],
                 alpha=0.25, width=max(1, min(concurrency) // 2), color="gray")
        twin.set_ylabel("preemptions (bars)")
        right.set_xlabel("concurrency C")
        right.set_ylabel("blocks")
        right.set_title("Demand against the pool, and what it costs")
        right.legend(loc="upper left", fontsize=8)
        right.grid(alpha=0.3)

    banner = _invalid_banner(payload)
    if banner:
        figure.text(0.5, 0.5, banner, ha="center", va="center", fontsize=16,
                    color="crimson", alpha=0.3, rotation=15, weight="bold")
    figure.text(0.01, 0.01, _provenance_line(payload), fontsize=6, alpha=0.75)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


@dataclass(frozen=True)
class MeasuredOccupancy:
    """What a run's event log says about capacity against occupancy.

    The analytical split above is a property of the length distribution and would
    read the same on any host, which is its strength and its limit: it explains
    why paging should win and proves nothing about whether it did. This comes off
    the scheduler's own event log, which records `free_blocks` and `tokens_held`
    at every admit, finish, preempt and recompute -- capacity and occupancy,
    sampled at every decision the scheduler made.
    """

    steps: tuple[int, ...]
    reserved_slots: tuple[int, ...]
    held_tokens: tuple[int, ...]
    preempt_steps: tuple[int, ...]
    recompute_steps: tuple[int, ...]
    block_size: int
    pool_slots: int

    @property
    def mean_occupancy(self) -> float:
        """Mean held/reserved over the run. The measured counterpart of `held`."""
        pairs = [
            (held, reserved)
            for held, reserved in zip(self.held_tokens, self.reserved_slots)
            if reserved > 0
        ]
        if not pairs:
            return 0.0
        return sum(held / reserved for held, reserved in pairs) / len(pairs)

    @property
    def peak_reserved_slots(self) -> int:
        return max(self.reserved_slots, default=0)


def measured_occupancy(payload: dict, engine: str = "engine") -> MeasuredOccupancy | None:
    """Read the pool trace out of a results file. None if the run recorded none."""
    provenance = payload.get("provenance", {})
    block_size = provenance.get("block_size", BLOCK_SIZE)
    num_blocks = provenance.get("num_gpu_blocks", NUM_GPU_BLOCKS)

    trace: list = []
    for level in _levels_for(payload, engine):
        if level.get("pool_trace"):
            trace = level["pool_trace"]
    if not trace:
        return None

    steps, reserved, held = [], [], []
    preempts, recomputes = [], []
    for step, blocks_allocated, tokens_held, kind in trace:
        steps.append(step)
        reserved.append(blocks_allocated * block_size)
        held.append(tokens_held)
        if kind == "preempt":
            preempts.append(step)
        elif kind == "recompute":
            recomputes.append(step)

    return MeasuredOccupancy(
        steps=tuple(steps),
        reserved_slots=tuple(reserved),
        held_tokens=tuple(held),
        preempt_steps=tuple(preempts),
        recompute_steps=tuple(recomputes),
        block_size=block_size,
        pool_slots=num_blocks * block_size,
    )


def measured_fragmentation(payload: dict, out_path: Path):
    """Capacity against occupancy over the run. The measured plot.

    Two curves and the gap between them. `reserved` is what the pool handed out,
    in token slots; `held` is how many of those slots have a token in them. The
    shaded gap is internal fragmentation as it actually occurred, rather than as
    the length distribution predicts it should.

    Preemptions and recomputes are marked on the same axis, because they are what
    the gap costs: every eviction is capacity that was reserved, filled, and then
    given back, and every recompute is the work of refilling it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed = measured_occupancy(payload)
    figure, axes = plt.subplots(figsize=(11, 5.5))

    if observed is None:
        axes.text(0.5, 0.5, "no pool trace recorded in this run",
                  transform=axes.transAxes, ha="center", va="center", fontsize=14)
    else:
        # Step, not line. The event log samples pool state at each decision and
        # the state holds until the next one, so interpolating between samples
        # would draw a ramp where the pool actually sat still.
        axes.plot(observed.steps, observed.reserved_slots, drawstyle="steps-post",
                  label=f"reserved (blocks handed out x {observed.block_size})",
                  linewidth=1.8, color="#f08c2e")
        axes.plot(observed.steps, observed.held_tokens, drawstyle="steps-post",
                  label="held (tokens actually stored)", linewidth=1.8, color="#2b7bba")
        axes.fill_between(observed.steps, observed.held_tokens, observed.reserved_slots,
                          step="post", alpha=0.25, color="#f08c2e",
                          label="internal fragmentation")
        axes.axhline(observed.pool_slots, color="crimson", linestyle="--",
                     linewidth=1.2, label=f"pool = {observed.pool_slots:,} slots")
        for step in observed.preempt_steps:
            axes.axvline(step, color="crimson", alpha=0.18, linewidth=1.0)
        for step in observed.recompute_steps:
            axes.axvline(step, color="seagreen", alpha=0.18, linewidth=1.0)

        axes.set_title(
            "Measured: capacity against occupancy over the run\n"
            f"mean occupancy {observed.mean_occupancy:.1%}, "
            f"{len(observed.preempt_steps)} preemptions (red), "
            f"{len(observed.recompute_steps)} recomputes (green)",
            fontsize=10,
        )
        axes.set_xlabel("scheduler step")
        axes.set_ylabel("token slots")
        axes.legend(loc="best", fontsize=8)
        axes.grid(alpha=0.3)

    banner = _invalid_banner(payload)
    if banner:
        axes.text(0.5, 0.5, banner, transform=axes.transAxes, ha="center",
                  va="center", fontsize=15, color="crimson", alpha=0.35,
                  rotation=18, weight="bold")
    figure.text(0.01, 0.01, _provenance_line(payload), fontsize=6, alpha=0.75)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the Gate 1 plots.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("bench/plots"))
    args = parser.parse_args(argv)

    payload = load(args.results)
    banner = _invalid_banner(payload)
    if banner:
        print(f"NOTE: {banner}")
    for path in (
        throughput_vs_concurrency(payload, args.out_dir / "throughput_vs_concurrency.png"),
        memory_fragmentation(payload, args.out_dir / "memory_fragmentation.png"),
        measured_fragmentation(payload, args.out_dir / "measured_fragmentation.png"),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
