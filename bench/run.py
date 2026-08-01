"""Concurrency sweep for both engines, with the guards that make a run quotable.

Gate 1, Step 6. This file produces the number. Everything in it exists because a
throughput figure is worthless unless you can say what produced it and prove the
run was valid, so the measurement is the small part and the provenance and the
guards are most of it.

MEASUREMENT. Per level: one warmup pass, discarded, then three measured passes.
Median reported, full range reported beside it. The range is not decoration.

    ESCALATION. If a level's full range exceeds 20% of that level's median, the
    level is rerun at five passes and BOTH the new range and the pass count are
    reported. If five passes still spread more than 20%, the level is marked
    UNRESOLVED and its median is not reported as a result. A median over a
    distribution that wide is a number with no content, and printing it anyway
    is how a bad measurement gets quoted.

TWO GUARDS THAT REFUSE RATHER THAN WARN.

    The control headline may only be run at PRESSURE_C. The control stream is
    rectangular 512/512, which is 64 blocks per request, and PRESSURE_C is the
    level where 64 requests demand 4096 blocks against a 3584-block pool. That
    12.5% shortfall is the whole point of the headline: at any lower level the
    stream fits and the headline measures a regime nobody claimed anything
    about. Requesting the headline at another level raises.

    A C64 run recording zero preemptions is INVALID, not suspicious. The
    shortfall above forces eviction arithmetically, so zero preemptions means
    the pool was mis-sized, the stream was not consumed, or the scheduler did not
    run -- and in every one of those cases the throughput number is measuring
    something other than what it claims. It raises. It does not warn and carry on
    with a footnote, because footnotes do not survive being pasted into a README.

NO NUMBERS ON A HOST THAT CANNOT PRODUCE THEM. Writing a result requires CUDA.
On any other host `--smoke` is required, and every record it writes carries
`throughput_valid: false` and `smoke: true`. The development machines here showed
76% run-to-run spread; a number from one of them must not be able to reach a
result file that looks like every other result file.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch

from bench import baseline as baseline_mod
from engine import attention as attention_mod
from bench import workload
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS, PRESSURE_C
from engine.cache import ADMISSION_MARGIN_BLOCKS, BlockAllocator
from engine.scheduler import (
    EventKind,
    ExecutionMode,
    Qwen2Executor,
    Request,
    Scheduler,
)

WARMUP_PASSES = 1
MEASURED_PASSES = 3
ESCALATED_PASSES = 5
# A level whose passes spread wider than this fraction of their own median is not
# a measurement yet.
SPREAD_LIMIT = 0.20

ENGINE_PAGED = "engine"
ENGINE_BASELINE = "baseline"

# THE TWO CAPACITY CAPS, APPLIED TO THE SWEEP.
#
# These existed in the scheduler and in kaggle/session2 and were NOT reaching
# this file: `run_engine_pass` accepted both, `_engine_factory` passed neither,
# and `_baseline_factory` passed no `max_num_seqs` either. The sweep therefore
# ran the engine UNCAPPED -- the configuration session 2 measured OOMing at
# 131,072 padded slots -- and it did OOM, on the mixed stream, at 13.86 GiB of
# 14.56 GiB, before producing a single timing. A capped run is not a tuning
# preference here; the uncapped one does not complete.
#
# max_num_seqs = C. Not device-dependent. It exists so that C means something:
# without it a sweep at C=8 compared an engine running whatever fits the pool
# against a baseline hard-capped by its rectangle. Both sides run at
# min(their own structural limit, C).
#
# max_num_batched_tokens = 8192. EMPIRICAL, measured on a T4 in kaggle/session2,
# not derived. It is the padded-rectangle budget for one fused step. On a larger
# card it is probably conservative -- docs/AMPERE_RECHECK.md item 5.
MAX_NUM_BATCHED_TOKENS = 8192

# Pinned for both engines, and recorded. Session 1 measured this moving the fp16
# deviation by up to 0.75 ulp -- length 33 went 3.00 -> 2.25 ulp with it off --
# so it is not a free knob and the two sides must not differ on it. True is
# torch's own default; FP16_LOGIT_ATOL is NOT adjusted to suit either setting.
ALLOW_FP16_REDUCED_PRECISION_REDUCTION = True


def pin_fp16_reduction(
    enabled: bool = ALLOW_FP16_REDUCED_PRECISION_REDUCTION,
) -> bool:
    """Force the fp16 split-k reduction setting, so both engines share it."""
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = enabled
    except (AttributeError, RuntimeError):
        return False
    return True


class InvalidRun(RuntimeError):
    """A run whose result must not be reported. Raised, never logged and ignored."""


# --- provenance ------------------------------------------------------------


def _driver_version() -> str | None:
    """NVIDIA driver version, or None. Never a guess."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()
    return line[0].strip() if line else None


def _gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def _model_sha(model_id: str) -> str | None:
    """The exact revision of the weights. None if it cannot be established."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:
        pass
    try:
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(model_id, local_files_only=True))
        # Hub caches store snapshots under snapshots/<commit sha>/.
        return local.name if local.parent.name == "snapshots" else None
    except Exception:
        return None


# Read straight off torch. An earlier version read conftest's globals, which
# pytest_configure populates -- so outside pytest it returned {} and provenance
# silently carried no precision settings at all. TF32 has fp16's mantissa; what
# these were set to is part of what produced a number, and a run launched from
# the CLI is exactly the run that matters.
_PRECISION_PROBES = (
    ("torch.backends.cuda.matmul.allow_tf32", "backends.cuda.matmul", "allow_tf32"),
    ("torch.backends.cudnn.allow_tf32", "backends.cudnn", "allow_tf32"),
    ("torch.backends.cuda.matmul.fp32_precision", "backends.cuda.matmul", "fp32_precision"),
    ("torch.backends.cudnn.fp32_precision", "backends.cudnn", "fp32_precision"),
    ("torch.backends.cudnn.conv.fp32_precision", "backends.cudnn.conv", "fp32_precision"),
    ("torch.backends.fp32_precision", "backends", "fp32_precision"),
    (
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
        "backends.cuda.matmul",
        "allow_fp16_reduced_precision_reduction",
    ),
)


def sdpa_backends_available() -> dict:
    """Which SDPA backends this build/device can use, and the global toggles.

    Recorded because SDPA backend selection materially moves fp16 numerics --
    switching to enable_gqa changed the input shape, torch picked a different
    kernel, and fp16 parity improved 6x with no change to the arithmetic this
    repo controls. That happened silently and was recorded nowhere. It is
    recorded now.
    """
    observed: dict = {}
    for name, getter_name in (
        ("flash", "flash_sdp_enabled"),
        ("mem_efficient", "mem_efficient_sdp_enabled"),
        ("math", "math_sdp_enabled"),
        ("cudnn", "cudnn_sdp_enabled"),
    ):
        getter = getattr(torch.backends.cuda, getter_name, None)
        if callable(getter):
            try:
                observed[name] = bool(getter())
            except RuntimeError:
                continue
    return observed


def _precision_settings() -> dict:
    """Whatever precision knobs this torch exposes, read directly from torch."""
    observed: dict = {}
    for label, path, attribute in _PRECISION_PROBES:
        holder = torch
        for part in path.split("."):
            holder = getattr(holder, part, None)
            if holder is None:
                break
        if holder is not None and hasattr(holder, attribute):
            try:
                observed[label] = getattr(holder, attribute)
            except RuntimeError:
                continue
    try:
        observed["torch.get_float32_matmul_precision()"] = (
            torch.get_float32_matmul_precision()
        )
    except (AttributeError, RuntimeError):
        pass
    return observed


@dataclass(frozen=True)
class Provenance:
    """Everything needed to say what produced a number, or that it is unknown."""

    num_gpu_blocks: int
    block_size: int
    dtype: str
    kv_pool_bytes: int
    bytes_per_token: int
    gpu_name: str | None
    driver_version: str | None
    torch_version: str
    python_version: str
    platform: str
    model_id: str
    model_sha: str | None
    workload_digest: str
    cuda_available: bool
    visible_cuda_devices: int
    cuda_visible_devices_env: str | None
    fp32_precision: dict
    sdpa_backends_enabled: dict
    allocator_conf: dict
    # WHICH ARM. The engine's GQA path and whether the baseline was forced onto
    # the expanded-K/V path. These decide which SDPA kernel each side runs, and
    # a ratio is only interpretable next to them.
    gqa_path: str
    baseline_kv_expansion_forced: bool
    max_num_batched_tokens: int | None
    # Whether the UNPATCHED baseline pre-expands K/V on this cell's own stream.
    # Decides whether Arm B moved the baseline or only the engine, which decides
    # which question this cell's A-vs-B comparison answers.
    baseline_expands_kv_unpatched: bool | None

    @classmethod
    def collect(
        cls, model_id: str, dtype: str, stream_digest: str,
        *, baseline_kv_expansion_forced: bool = False,
        baseline_expands_kv_unpatched: bool | None = None,
    ) -> "Provenance":
        return cls(
            num_gpu_blocks=NUM_GPU_BLOCKS,
            block_size=BLOCK_SIZE,
            dtype=dtype,
            kv_pool_bytes=baseline_mod.KV_POOL_BYTES,
            bytes_per_token=baseline_mod.BYTES_PER_TOKEN,
            gpu_name=_gpu_name(),
            driver_version=_driver_version(),
            torch_version=torch.__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
            model_id=model_id,
            model_sha=_model_sha(model_id),
            workload_digest=stream_digest,
            cuda_available=torch.cuda.is_available(),
            visible_cuda_devices=(
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            cuda_visible_devices_env=os.environ.get('CUDA_VISIBLE_DEVICES'),
            # TF32 has fp16's mantissa. An fp32 matmul silently run in TF32
            # deviates at fp16 scale, so what these were set to is part of
            # what produced the number.
            fp32_precision=_precision_settings(),
            # SDPA backend selection moves fp16 numerics. Recorded, because
            # it moved once already without anyone noticing.
            sdpa_backends_enabled=sdpa_backends_available(),
            # expandable_segments changes allocation only, not arithmetic --
            # but it moves the memory numbers, so it travels with them.
            allocator_conf={
                name: os.environ.get(name)
                for name in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')
            },
            # Read from the module rather than the environment: the module is
            # what the kernel calls actually obeyed.
            gqa_path=attention_mod.GQA_PATH,
            baseline_kv_expansion_forced=baseline_kv_expansion_forced,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            baseline_expands_kv_unpatched=baseline_expands_kv_unpatched,
        )


# --- one level -------------------------------------------------------------


@dataclass
class LevelResult:
    """One concurrency level for one engine, with its own verdict."""

    engine: str
    stream: str
    concurrency: int
    num_requests: int
    passes: list[float] = field(default_factory=list)
    generated_tokens: int = 0
    rejection_rate: float = 0.0
    num_rejected: int = 0
    preemption_count: int = 0
    demanded_blocks: int = 0
    peak_blocks_used: int = 0
    rectangle_tokens: int | None = None
    max_resident: int | None = None
    unresolved: bool = False
    throughput_valid: bool = True
    pool_trace: tuple[tuple[int, int, int, str], ...] = ()
    trace_events_total: int = 0

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.passes)

    @property
    def range_seconds(self) -> float:
        return max(self.passes) - min(self.passes)

    @property
    def spread(self) -> float:
        """Full range as a fraction of the median. The escalation trigger."""
        median = self.median_seconds
        return self.range_seconds / median if median > 0 else float("inf")

    @property
    def median_tokens_per_second(self) -> float | None:
        """None when the level is UNRESOLVED. There is no number to report.

        A level with no passes has no throughput either. `collect()` always
        records at least one, so this is unreachable through the sweep -- but
        `describe()` must be total, because the one caller that hits an
        un-passed level is a harness reading a partial run, and raising
        `StatisticsError` out of a serialiser hides which level was empty.
        """
        if self.unresolved or not self.throughput_valid or not self.passes:
            return None
        median = self.median_seconds
        return self.generated_tokens / median if median > 0 else None

    def describe(self) -> dict[str, object]:
        record = {
            "engine": self.engine,
            "stream": self.stream,
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            "num_passes": len(self.passes),
            "pass_seconds": [round(value, 6) for value in self.passes],
            "median_seconds": round(self.median_seconds, 6) if self.passes else None,
            "range_seconds": round(self.range_seconds, 6) if self.passes else None,
            "spread_fraction": round(self.spread, 6) if self.passes else None,
            "spread_limit": SPREAD_LIMIT,
            "generated_tokens": self.generated_tokens,
            "rejection_rate": round(self.rejection_rate, 6),
            "num_rejected": self.num_rejected,
            "preemption_count": self.preemption_count,
            "demanded_blocks": self.demanded_blocks,
            "peak_blocks_used": self.peak_blocks_used,
            "rectangle_tokens": self.rectangle_tokens,
            "max_resident": self.max_resident,
            "unresolved": self.unresolved,
            "throughput_valid": self.throughput_valid,
            "pool_trace": [list(point) for point in self.pool_trace],
            "trace_events_total": self.trace_events_total,
            "trace_downsampled": len(self.pool_trace) < self.trace_events_total,
        }
        tps = self.median_tokens_per_second
        record["median_tokens_per_second"] = round(tps, 4) if tps is not None else None
        return record


# --- the engine side -------------------------------------------------------


def _engine_requests(stream: Sequence) -> list[Request]:
    """workload.Request -> scheduler.Request. Same fields, different owner."""
    return [
        Request(
            request_id=item.request_id,
            prompt_token_ids=item.prompt_token_ids,
            max_new_tokens=item.max_new_tokens,
        )
        for item in stream
    ]


# Pool-trace points kept per level. The trace is the raw material for the
# measured fragmentation plot and a long run produces tens of thousands of
# events, so it is downsampled by striding above this. Truncation is recorded in
# the result and logged, never silent.
MAX_TRACE_POINTS = 4000


@dataclass
class EnginePass:
    elapsed: float
    generated_tokens: int
    preemptions: int
    num_rejected: int
    peak_blocks_used: int
    # (step, blocks_allocated, tokens_held, event kind) at every scheduler
    # decision. Capacity against occupancy, which is the measured fragmentation.
    pool_trace: tuple[tuple[int, int, int, str], ...] = ()
    trace_events_total: int = 0


def run_engine_pass(
    stream: Sequence,
    executor_factory: Callable[[BlockAllocator], object],
    *,
    num_gpu_blocks: int = NUM_GPU_BLOCKS,
    block_size: int = BLOCK_SIZE,
    mode: ExecutionMode = ExecutionMode.FUSED,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] | None = None,
) -> EnginePass:
    """One measured pass of the paged engine over the whole stream."""
    allocator = BlockAllocator(num_blocks=num_gpu_blocks, block_size=block_size)
    scheduler = Scheduler(
        executor_factory(allocator), allocator, mode=mode,
        admission_margin_blocks=ADMISSION_MARGIN_BLOCKS,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    rejected = 0
    for request in _engine_requests(stream):
        if not scheduler.add_request(request):
            rejected += 1

    if synchronize is not None:
        synchronize()
    started = clock()
    outputs = scheduler.run_to_completion()
    if synchronize is not None:
        synchronize()
    elapsed = clock() - started

    # Peak residency as seen at event boundaries -- every admit, finish, preempt
    # and recompute records the pool state at that moment. It is a lower bound on
    # the true peak, not an estimate of it, because blocks are only ever taken
    # between events by a decode that crosses a block boundary. Named for what it
    # is rather than reported as "peak usage".
    peak = max(
        (num_gpu_blocks - event.free_blocks for event in scheduler.events),
        default=0,
    )

    # Capacity against occupancy at every decision. Strided rather than dropped
    # from the tail, so a downsampled trace still spans the whole run.
    stride = max(1, -(-len(scheduler.events) // MAX_TRACE_POINTS))
    trace = tuple(
        (
            event.step,
            num_gpu_blocks - event.free_blocks,
            event.tokens_held,
            event.kind.value,
        )
        for event in scheduler.events[::stride]
    )

    return EnginePass(
        elapsed=elapsed,
        generated_tokens=sum(len(tokens) for tokens in outputs.values()),
        preemptions=len(scheduler.events_of(EventKind.PREEMPT)),
        num_rejected=rejected,
        peak_blocks_used=peak,
        pool_trace=trace,
        trace_events_total=len(scheduler.events),
    )


def realized_block_demand(stream: Sequence, block_size: int = BLOCK_SIZE) -> int:
    """Blocks the stream would hold if every request were resident at its maximum."""
    return sum(-(-item.total_len // block_size) for item in stream)


# --- guards ----------------------------------------------------------------


def check_control_headline_level(stream_name: str, concurrency: int, headline: bool) -> None:
    """GUARD: the control headline is only meaningful at PRESSURE_C."""
    if not headline or stream_name != "control":
        return
    if concurrency != PRESSURE_C:
        raise InvalidRun(
            f"control headline requested at C={concurrency}, but the headline is "
            f"only defined at PRESSURE_C={PRESSURE_C}. At {PRESSURE_C} the control "
            f"stream demands {PRESSURE_C * workload.CONTROL_BLOCKS_PER_REQUEST} "
            f"blocks against a {NUM_GPU_BLOCKS}-block pool, which is the "
            "oversubscribed regime the headline is a claim about. Run the sweep at "
            "other levels if you want the curve; do not label them the headline."
        )


def forced_eviction_demand(stream: Sequence, max_num_seqs: int | None) -> int:
    """Blocks the SMALLEST admissible resident set needs. Eviction is certain
    above the pool and merely possible below it.

    `realized_block_demand` sums the WHOLE stream, which is the wrong quantity
    once `max_num_seqs` exists: only `max_num_seqs` requests are ever resident,
    so a 256-request stream demanding 12,184 blocks says nothing about whether
    64 of them at a time overflow a 3584-block pool. Taking the smallest
    `max_num_seqs` requests makes this a lower bound on any resident set, so
    exceeding the pool means eviction regardless of admission order.

    Measured at C=64: control 4096 (forced, every request is 64 blocks); mixed
    1226 from the smallest 64 and 3310 from the first 64, both under the pool,
    though the largest 64 would want 5766. So the mixed stream MAY preempt and
    is not required to, and a mixed run recording zero preemptions is a fact
    about the stream rather than a broken harness.
    """
    if not stream:
        return 0
    per_request = sorted(
        -(-(len(item.prompt_token_ids) + item.max_new_tokens) // BLOCK_SIZE)
        for item in stream
    )
    resident = len(per_request) if max_num_seqs is None else min(max_num_seqs, len(per_request))
    return sum(per_request[:resident])


def check_preemptions(
    result: LevelResult, demanded_blocks: int, forced_demand: int | None = None
) -> None:
    """GUARD: a run with no preemptions where eviction is FORCED is INVALID.

    Two situations, the same mistake from different sides: the control stream at
    PRESSURE_C, where the 12.5% shortfall forces eviction arithmetically, and
    any level whose smallest resident set already overflows the pool. Either
    way zero preemptions means the pool was mis-sized, the stream was not
    consumed, or the scheduler never ran, and the throughput number is measuring
    something else.

    THE PREDICATE WAS WRONG AND THIS NARROWS IT. It used to test whole-stream
    demand against the pool, which is true for essentially every stream and
    would have raised `InvalidRun` on the mixed cells the moment `max_num_seqs`
    started being applied -- flagging a legitimate zero-preemption run as
    broken. Narrowing a guard is not free and is not done to make a cell pass:
    the control headline's own check is UNCONDITIONAL at PRESSURE_C and is
    untouched, so the quotable number keeps exactly the protection it had.
    """
    if result.engine != ENGINE_PAGED or result.preemption_count > 0:
        return

    at_pressure = result.stream == "control" and result.concurrency == PRESSURE_C
    oversubscribed = (
        forced_demand if forced_demand is not None else demanded_blocks
    ) > NUM_GPU_BLOCKS
    if not (at_pressure or oversubscribed):
        return

    shortfall = (
        f", {(demanded_blocks - NUM_GPU_BLOCKS) / demanded_blocks:.1%} shortfall"
        if demanded_blocks > NUM_GPU_BLOCKS
        else ""
    )
    raise InvalidRun(
        f"INVALID RUN: {result.stream} stream at C={result.concurrency} recorded "
        f"ZERO preemptions while demanding {demanded_blocks} blocks against a "
        f"{NUM_GPU_BLOCKS}-block pool{shortfall}. Eviction is forced here, so this "
        "run did not measure what it claims. Throughput from it must not be "
        "reported."
    )


# --- the sweep -------------------------------------------------------------


def measure_level(
    engine_name: str,
    stream_name: str,
    concurrency: int,
    stream: Sequence,
    run_pass: Callable[[], EnginePass],
    *,
    rectangle_tokens: int | None = None,
    max_resident: int | None = None,
    throughput_valid: bool = True,
    log: Callable[[str], None] = print,
) -> LevelResult:
    """Warmup, three passes, and the escalation rule if they disagree."""
    log(f"  [{engine_name}] C={concurrency}: warmup ({WARMUP_PASSES} pass, discarded)")
    for _ in range(WARMUP_PASSES):
        run_pass()

    result = LevelResult(
        engine=engine_name,
        stream=stream_name,
        concurrency=concurrency,
        num_requests=len(stream),
        rectangle_tokens=rectangle_tokens,
        max_resident=max_resident,
        throughput_valid=throughput_valid,
    )

    def collect(count: int) -> None:
        result.passes = []
        for index in range(count):
            observed = run_pass()
            # Passes must do identical work or their durations are not comparable
            # and the median is over a mixture. Greedy decoding is deterministic
            # and the workload contract is exact lengths, so any difference here
            # is a bug rather than noise.
            if result.passes and observed.generated_tokens != result.generated_tokens:
                raise InvalidRun(
                    f"{engine_name} C={concurrency}: pass {index + 1} produced "
                    f"{observed.generated_tokens} tokens where an earlier pass "
                    f"produced {result.generated_tokens}. The passes did different "
                    "amounts of work, so their timings cannot be compared."
                )
            result.passes.append(observed.elapsed)
            result.generated_tokens = observed.generated_tokens
            result.preemption_count = observed.preemptions
            result.num_rejected = observed.num_rejected
            result.peak_blocks_used = max(result.peak_blocks_used, observed.peak_blocks_used)
            result.rejection_rate = (
                observed.num_rejected / len(stream) if len(stream) else 0.0
            )
            # Decoding is deterministic, so every pass traces the same pool
            # history. Keeping the last is arbitrary and sufficient.
            result.pool_trace = observed.pool_trace
            result.trace_events_total = observed.trace_events_total
            log(f"    pass {index + 1}/{count}: {observed.elapsed:.4f}s")

    collect(MEASURED_PASSES)
    log(f"    median {result.median_seconds:.4f}s, range {result.range_seconds:.4f}s "
        f"({result.spread:.1%} of median)")

    if result.spread > SPREAD_LIMIT:
        log(f"    ESCALATION: {result.spread:.1%} spread exceeds "
            f"{SPREAD_LIMIT:.0%}; rerunning at {ESCALATED_PASSES} passes")
        collect(ESCALATED_PASSES)
        log(f"    median {result.median_seconds:.4f}s, range {result.range_seconds:.4f}s "
            f"({result.spread:.1%} of median) over {len(result.passes)} passes")
        if result.spread > SPREAD_LIMIT:
            result.unresolved = True
            log(f"    UNRESOLVED: {result.spread:.1%} spread over "
                f"{ESCALATED_PASSES} passes. No median reported for this level.")

    return result


def sweep(
    concurrencies: Sequence[int],
    stream_name: str,
    seed: int | str,
    engine_pass_factory: Callable[[Sequence], Callable[[], EnginePass]] | None,
    baseline_pass_factory: Callable[[Sequence, int], Callable[[], EnginePass]] | None = None,
    *,
    rectangle_tokens: int | None = None,
    headline: bool = False,
    throughput_valid: bool = True,
    log: Callable[[str], None] = print,
) -> list[LevelResult]:
    """Sweep the levels, guarding before and after each one."""
    results: list[LevelResult] = []
    for concurrency in concurrencies:
        check_control_headline_level(stream_name, concurrency, headline)
        stream = workload.build_stream(stream_name, seed, concurrency)
        demanded = realized_block_demand(stream)
        forced = forced_eviction_demand(stream, concurrency)
        log(f"C={concurrency}: {len(stream)} requests, "
            f"realized block demand {demanded} vs pool {NUM_GPU_BLOCKS}; "
            f"smallest resident set at C={concurrency} needs {forced} blocks, so "
            f"eviction is {'FORCED' if forced > NUM_GPU_BLOCKS else 'possible but not forced'}")

        if engine_pass_factory is not None:
            level = measure_level(
                ENGINE_PAGED, stream_name, concurrency, stream,
                engine_pass_factory(stream, concurrency, MAX_NUM_BATCHED_TOKENS),
                throughput_valid=throughput_valid, log=log,
            )
            level.demanded_blocks = demanded
            if level.trace_events_total > len(level.pool_trace):
                log(f"    pool trace downsampled: {len(level.pool_trace)} of "
                    f"{level.trace_events_total} events kept (stride sampling, "
                    f"cap {MAX_TRACE_POINTS})")
            check_preemptions(
                level, demanded,
                forced_eviction_demand(stream, concurrency),
            )
            results.append(level)

        if baseline_pass_factory is not None and rectangle_tokens is not None:
            level = measure_level(
                ENGINE_BASELINE, stream_name, concurrency, stream,
                baseline_pass_factory(stream, rectangle_tokens, concurrency),
                rectangle_tokens=rectangle_tokens,
                max_resident=min(
                    baseline_mod.max_resident(rectangle_tokens), concurrency
                ),
                throughput_valid=throughput_valid, log=log,
            )
            level.demanded_blocks = demanded
            results.append(level)

    return results


def write_results(
    path: Path,
    provenance: Provenance,
    results: Sequence[LevelResult],
    *,
    smoke: bool,
    baseline_identity: dict | None = None,
) -> Path:
    """Persist a run. Refuses to write a real result off a CUDA host."""
    if not smoke and not torch.cuda.is_available():
        raise InvalidRun(
            "refusing to write results on a host without CUDA. Timings from a "
            "development machine are not comparable to anything and must not end "
            "up in a file shaped like a real result. Pass --smoke to record a "
            "structural run, which is stamped throughput_valid: false."
        )
    payload = {
        "smoke": smoke,
        "throughput_valid": not smoke and torch.cuda.is_available(),
        "warmup_passes": WARMUP_PASSES,
        "measured_passes": MEASURED_PASSES,
        "escalated_passes": ESCALATED_PASSES,
        "spread_limit": SPREAD_LIMIT,
        "pressure_c": PRESSURE_C,
        "provenance": asdict(provenance),
        # Arm B's gate. None on Arm A, where nothing was patched and there is
        # nothing to attest.
        "baseline_token_identity": baseline_identity,
        "levels": [result.describe() for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


# --- CLI -------------------------------------------------------------------


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _engine_factory(engine, cache_factory):
    def factory(stream, max_num_seqs, max_num_batched_tokens):
        def run_pass() -> EnginePass:
            def executor_factory(allocator):
                cache = cache_factory(allocator)
                return Qwen2Executor(engine, cache, allocator)

            return run_engine_pass(
                stream, executor_factory, synchronize=_cuda_sync,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
            )

        return run_pass

    return factory


def _baseline_factory(generator):
    def factory(stream, rectangle_tokens, max_num_seqs):
        def run_pass() -> EnginePass:
            observed = baseline_mod.run_baseline(
                stream, generator, rectangle_tokens, synchronize=_cuda_sync,
                max_num_seqs=max_num_seqs,
            )
            return EnginePass(
                elapsed=observed.elapsed_seconds,
                generated_tokens=observed.generated_tokens,
                preemptions=0,  # a static batch has nothing to preempt
                num_rejected=observed.num_rejected,
                peak_blocks_used=0,
            )

        return run_pass

    return factory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concurrency sweep for Gate 1.")
    parser.add_argument("--stream", choices=("mixed", "control"), default="control")
    parser.add_argument("--seed", default="0")
    parser.add_argument(
        "-C", "--concurrency", type=int, action="append", default=None,
        help=f"sweep level; repeatable. default: PRESSURE_C={PRESSURE_C}",
    )
    parser.add_argument(
        "--rectangle", type=int, default=baseline_mod.RECTANGLE_DISTRIBUTION_MAX,
        choices=baseline_mod.RECTANGLE_SIZINGS,
    )
    parser.add_argument(
        "--headline", action="store_true",
        help=f"the quotable control number. Only valid at C={PRESSURE_C}.",
    )
    parser.add_argument("--model", default=baseline_mod.BENCH_MODEL)
    parser.add_argument("--out", type=Path, default=Path("bench/results/run.json"))
    parser.add_argument(
        "--smoke", action="store_true",
        help="structural run: no timing claim. Required off a CUDA host.",
    )
    parser.add_argument(
        "--baseline-expand-kv", action="store_true",
        help=(
            "Arm B: force the baseline to pre-expand K/V so it lands on the "
            "same kernel as GQA_PATH=repeat_kv. Gated on token identity."
        ),
    )
    args = parser.parse_args(argv)

    concurrencies = args.concurrency or [PRESSURE_C]
    for concurrency in concurrencies:
        check_control_headline_level(args.stream, concurrency, args.headline)

    if not args.smoke and not torch.cuda.is_available():
        print(
            "refusing to measure on a host without CUDA. Pass --smoke for a "
            "structural run.", file=sys.stderr,
        )
        return 2

    from engine.model import Qwen2Engine
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # One set of weights for both sides. The engine borrows the tensors and the
    # baseline drives the module, so neither gets a different model and neither
    # pays for a second copy.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.eval()
    model.to(device)
    engine = Qwen2Engine.from_hf_model(model)
    generator = baseline_mod.HuggingFaceGenerator(
        model, AutoTokenizer.from_pretrained(args.model), device
    )

    def cache_factory(allocator):
        return engine.new_paged_cache(allocator.num_blocks, allocator.block_size)

    reference = workload.build_stream(args.stream, args.seed, concurrencies[0])

    # ARM B'S PREREQUISITE, BEFORE ANY TIMING. Pre-expanding K/V is a no-op
    # mathematically and a kernel change in practice; if the kernel change moved
    # a token then the thing about to be timed is not this baseline. Checked on
    # the real stream's own batches, and on a prompt-uniform batch specifically,
    # because that is the case where the patch actually changes HF's behaviour
    # (ragged batches already pre-expand, so the patch is inert there).
    # WHICH PATH THE UNPATCHED BASELINE TAKES ON THIS CELL'S OWN STREAM,
    # measured for BOTH arms and recorded. Arm B's patch only bites where HF
    # would otherwise have used `enable_gqa`, i.e. where the batch is
    # prompt-uniform and the mask collapses to None. On a ragged stream HF
    # already pre-expands and the patch is INERT. Without this per cell, a cell
    # where the patch flipped the kernel looks identical in the result file to
    # one where it did nothing -- and those two answer different questions.
    probe_batch = list(reference[: baseline_mod.max_resident(args.rectangle)])[:4]
    baseline_expands_unpatched = baseline_mod.baseline_expands_kv_for(
        generator, probe_batch
    )
    print(f"unpatched baseline on {args.stream}: "
          f"expands_kv={baseline_expands_unpatched} "
          f"({'already on the pre-expanded path; Arm B is inert here' if baseline_expands_unpatched else 'takes enable_gqa; Arm B moves it'})")

    baseline_identity = None
    if args.baseline_expand_kv:
        # Probe the batch the timed pass actually runs. Where the patch is inert
        # on this stream, that batch proves nothing, so borrow a prompt-uniform
        # one -- the case where the patch does flip the path -- rather than
        # comparing the same requests twice and reporting double the evidence.
        batches = [probe_batch]
        if baseline_expands_unpatched:
            batches.append(
                list(workload.build_stream("control", args.seed, concurrencies[0])[:4])
            )
        print("Arm B prerequisite: patched baseline vs unpatched, token identity")
        baseline_identity = baseline_mod.assert_token_identical(generator, batches)
        print(f"  {baseline_identity}")
        baseline_mod.force_baseline_kv_expansion()
        print("  patch applied for the timed passes")

    provenance = Provenance.collect(
        args.model, str(dtype), workload.stream_digest(reference),
        baseline_kv_expansion_forced=args.baseline_expand_kv,
        baseline_expands_kv_unpatched=baseline_expands_unpatched,
    )

    results = sweep(
        concurrencies, args.stream, args.seed,
        _engine_factory(engine, cache_factory),
        _baseline_factory(generator),
        rectangle_tokens=args.rectangle,
        headline=args.headline,
        throughput_valid=not args.smoke and torch.cuda.is_available(),
    )

    written = write_results(
        args.out, provenance, results, smoke=args.smoke,
        baseline_identity=baseline_identity,
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
