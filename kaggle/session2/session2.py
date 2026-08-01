"""SESSION 2: allocator fragmentation, a PRESSURE probe, and SDPA backend identity.

Three items. NO SWEEP. NO THROUGHPUT NUMBER.

  1. expandable_segments. Reserved sat at 98.5% of the card while allocated was
     6-8.4 GiB. That gap is caching-allocator fragmentation, not live tensors.
     Remeasure all three levels with it on and record the setting.

  2. THE PRESSURE PROBE. The last probe measured the easy case -- 25 steps, zero
     preemptions, 3210 of 3584 blocks. Every path that only runs under memory
     pressure went unmeasured. The sweep runs to completion: the pool exhausts,
     recompute fires, and a recompute re-prefills a FULL HISTORY in one forward
     at exactly the moment the pool is fullest and the allocator most fragmented.
     That is the most likely OOM in the real sweep, so run until it happens.

  3. SDPA BACKEND IDENTITY. enable_gqa moved fp16 parity 6x. The earlier
     explanation -- that materialising the expansion cost precision -- is WRONG.
     expand-then-reshape copies fp16 into fp16, which is bit-exact, and the
     torch.equal check in tests/test_attention_gqa.py proves both paths receive
     identical inputs. The only thing left that can move the number is SDPA
     dispatching to a DIFFERENT KERNEL for the two input shapes. So: identify
     which backend actually runs, for the shapes this engine uses.
"""

import json
import os

# Both spellings. torch renamed this and the OOM message on this host named the
# newer one; setting both and recording which took effect beats guessing.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

WORKING = Path(os.environ.get("SESSION2_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION2_INPUT", "/kaggle/input"))
REPO_DEST = WORKING / "repo"
REPORT = WORKING / "session2_report.json"
TRANSFORMERS_PIN = "5.12.1"
HEADLINE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HEADLINE_SHA = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

PRESSURE_STEPS = int(os.environ.get("SESSION2_PRESSURE_STEPS", "400"))
PRESSURE_TARGET_PREEMPTIONS = int(os.environ.get("SESSION2_TARGET_PREEMPTS", "12"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("SESSION2_BATCHED_TOKENS", "8192"))
EASY_STEPS = 25

report: dict = {"items": {}}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def run(command, timeout=5400, cwd=None):
    print(f"$ {' '.join(command)}", flush=True)
    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd)
    except (OSError, subprocess.SubprocessError) as error:
        return subprocess.CompletedProcess(command, 127, "", repr(error))
    if done.stdout:
        print(done.stdout, flush=True)
    if done.stderr:
        print(done.stderr, file=sys.stderr, flush=True)
    return done


section("SETUP")
WORKING.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402

(WORKING / "constraints.txt").write_text(f"torch=={torch.__version__}\n")
run([sys.executable, "-m", "pip", "install", "--quiet", "-c",
     str(WORKING / "constraints.txt"), f"transformers=={TRANSFORMERS_PIN}"], timeout=1800)

source = None
for marker in INPUT_ROOT.rglob("scheduler.py"):
    if marker.parent.name == "engine":
        source = marker.parent.parent
        break
if source is None:
    REPORT.write_text(json.dumps({"fatal": "repo not mounted"}, indent=2))
    raise SystemExit(1)
if REPO_DEST.exists():
    shutil.rmtree(REPO_DEST)
shutil.copytree(source, REPO_DEST)
sys.path.insert(0, str(REPO_DEST))
os.chdir(REPO_DEST)

DEVICE = "cuda"
alloc_conf = {
    "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
}
report["allocator_conf_env"] = alloc_conf
print(f"allocator conf: {alloc_conf}", flush=True)
print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
      f"device_count={torch.cuda.device_count()}", flush=True)

from transformers import AutoModelForCausalLM  # noqa: E402

from bench import run as run_mod  # noqa: E402
from bench import workload  # noqa: E402
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS  # noqa: E402
from engine.cache import BlockAllocator  # noqa: E402
from engine.model import Qwen2Engine  # noqa: E402
from engine.scheduler import (  # noqa: E402
    EventKind, ExecutionMode, Qwen2Executor, Request, Scheduler,
)

run_mod.pin_fp16_reduction()

total_bytes = torch.cuda.get_device_properties(0).total_memory
report["device_total_gib"] = round(total_bytes / 2**30, 2)

torch.cuda.reset_peak_memory_stats()
head_model = AutoModelForCausalLM.from_pretrained(
    HEADLINE_MODEL, dtype=torch.float16, revision=HEADLINE_SHA)
head_model.eval()
head_model.to(DEVICE)
head_engine = Qwen2Engine.from_hf_model(head_model)
report["weights_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
print(f"weights resident {report['weights_gib']} GiB", flush=True)


class StepRecorder:
    """Per-step batch shape, kept whatever happens. Indexed by scheduler step."""

    def __init__(self, inner):
        self.inner = inner
        self.per_step: list[dict] = []
        self.max_batch = 0
        self.max_query = 0
        self.max_padded = 0

    def new_backend(self, seq_id):
        return self.inner.new_backend(seq_id)

    def reserve(self, backend, n):
        return self.inner.reserve(backend, n)

    def forward(self, t, p, b):
        return self.inner.forward(t, p, b)

    def forward_batch(self, inputs):
        widths = [len(i.token_ids) for i in inputs]
        padded = len(widths) * max(widths) if widths else 0
        # Recorded BEFORE delegating, so an OOM here still leaves the shape.
        self.per_step.append({
            "batch": len(inputs), "max_q": max(widths) if widths else 0,
            "padded_slots": padded, "real_tokens": sum(widths),
        })
        self.max_batch = max(self.max_batch, len(inputs))
        self.max_query = max(self.max_query, max(widths) if widths else 0)
        self.max_padded = max(self.max_padded, padded)
        return self.inner.forward_batch(inputs)


def drive(concurrency, max_steps, stop_after_preemptions=None, label=""):
    """Run the real stream at a level, capped, recording everything."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    entry = {
        "concurrency": concurrency, "max_steps": max_steps,
        "max_num_seqs": concurrency,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
    }
    executor = None
    scheduler = None
    try:
        stream = workload.build_stream("mixed", seed=0, c=concurrency)
        entry["num_requests"] = len(stream)
        entry["demanded_blocks"] = sum(-(-r.total_len // BLOCK_SIZE) for r in stream)

        allocator = BlockAllocator(num_blocks=NUM_GPU_BLOCKS, block_size=BLOCK_SIZE)
        cache = head_engine.new_paged_cache(NUM_GPU_BLOCKS, BLOCK_SIZE)
        executor = StepRecorder(Qwen2Executor(head_engine, cache, allocator))
        scheduler = Scheduler(
            executor, allocator, mode=ExecutionMode.FUSED,
            max_num_seqs=concurrency,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        )
        for item in stream:
            scheduler.add_request(Request(
                request_id=item.request_id,
                prompt_token_ids=item.prompt_token_ids,
                max_new_tokens=item.max_new_tokens))

        steps = 0
        started = time.perf_counter()
        step_records = []
        while (scheduler.waiting or scheduler.running) and steps < max_steps:
            result = scheduler.step()
            steps += 1
            step_records.append({
                "step": result.step, "batch_size": result.batch_size,
                "padded": result.batched_token_cost,
                "deferred": len(result.deferred),
                "preempted": len(result.preempted),
                "blocks_used": NUM_GPU_BLOCKS - result.free_blocks,
            })
            preempts = len(scheduler.events_of(EventKind.PREEMPT))
            if stop_after_preemptions and preempts >= stop_after_preemptions:
                print(f"  {label}: reached {preempts} preemptions at step {steps}",
                      flush=True)
                break
        torch.cuda.synchronize()
        entry["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        entry["ok"] = True
    except torch.cuda.OutOfMemoryError as error:
        entry["ok"] = False
        entry["oom"] = True
        entry["error"] = str(error)[:1500]
        print(f"{label}: OUT OF MEMORY\n{error}", file=sys.stderr, flush=True)
    except Exception as error:  # noqa: BLE001
        entry["ok"] = False
        entry["error"] = f"{type(error).__name__}: {error}"[:1500]
        entry["traceback"] = traceback.format_exc()[-2500:]

    if executor is not None:
        entry["max_batch_width"] = executor.max_batch
        entry["max_query_len"] = executor.max_query
        entry["max_padded_slots"] = executor.max_padded
    if scheduler is not None:
        preempt_events = scheduler.events_of(EventKind.PREEMPT)
        recompute_events = scheduler.events_of(EventKind.RECOMPUTE)
        entry["preemptions"] = len(preempt_events)
        entry["recomputes"] = len(recompute_events)
        entry["steps_run"] = scheduler.events[-1].step if scheduler.events else 0
        entry["peak_blocks_used"] = max(
            (NUM_GPU_BLOCKS - e.free_blocks for e in scheduler.events), default=0)
        entry["running_at_stop"] = len(scheduler.running)

        # THE STEP THAT MATTERS: a recompute re-prefills a full history at the
        # moment the pool is fullest. Pull those steps out by name.
        recompute_steps = sorted({e.step for e in recompute_events})
        by_step = {r["step"]: r for r in step_records}
        entry["recompute_steps"] = recompute_steps[:40]
        detail = []
        for step in recompute_steps[:20]:
            record = by_step.get(step)
            if record:
                detail.append(record)
        entry["recompute_step_detail"] = detail
        if detail:
            entry["max_padded_on_a_recompute_step"] = max(
                d["padded"] for d in detail)
            entry["max_batch_on_a_recompute_step"] = max(
                d["batch_size"] for d in detail)
            entry["total_deferred_on_recompute_steps"] = sum(
                d["deferred"] for d in detail)
        entry["total_deferred"] = sum(r["deferred"] for r in step_records)
        entry["peak_blocks_pct"] = round(
            100 * entry["peak_blocks_used"] / NUM_GPU_BLOCKS, 1)

    entry["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    entry["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    entry["reserved_pct_of_card"] = round(
        100 * torch.cuda.max_memory_reserved() / total_bytes, 1)
    entry["headroom_gib"] = round(
        (total_bytes - torch.cuda.max_memory_reserved()) / 2**30, 3)

    print(f"{label} C={concurrency}: ok={entry.get('ok')} "
          f"alloc={entry['peak_allocated_gib']} reserved={entry['peak_reserved_gib']} "
          f"({entry['reserved_pct_of_card']}% of card) "
          f"preempts={entry.get('preemptions')} "
          f"blocks={entry.get('peak_blocks_used')}", flush=True)

    del scheduler, executor
    torch.cuda.empty_cache()
    return entry


# --- ITEM 1: expandable_segments, all three levels ---------------------------

section("ITEM 1: expandable_segments, remeasured at C=8, 64, 87")

item1 = {"steps": EASY_STEPS, "levels": {}}
for concurrency in (8, 64, 87):
    item1["levels"][str(concurrency)] = drive(
        concurrency, EASY_STEPS, label="[item1]")
report["items"]["item1_expandable_segments"] = item1


# --- ITEM 2: the PRESSURE probe ---------------------------------------------

section("ITEM 2: pressure probe -- run until the pool exhausts and recompute fires")

item2 = {
    "max_steps": PRESSURE_STEPS,
    "target_preemptions": PRESSURE_TARGET_PREEMPTIONS,
    "levels": {},
}
for concurrency in (64, 87):
    item2["levels"][str(concurrency)] = drive(
        concurrency, PRESSURE_STEPS,
        stop_after_preemptions=PRESSURE_TARGET_PREEMPTIONS, label="[pressure]")
report["items"]["item2_pressure_probe"] = item2


# --- ITEM 3: which SDPA backend actually runs -------------------------------

section("ITEM 3: SDPA backend identity for the shapes this engine uses")

import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from engine.attention import repeat_kv  # noqa: E402

BACKENDS = {
    "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
    "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
    "MATH": SDPBackend.MATH,
    "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
}


def identify_backend(query, keys, values, **kwargs):
    """Which kernel did the unrestricted call dispatch to?

    Determined by RUNNING it: force each backend in turn and compare bit-exactly
    against the free call. The one that matches is the one that ran. Reported as
    a list because two backends can coincidentally agree, and saying "these
    matched" is honest where "it was flash" would not be.
    """
    reference = F.scaled_dot_product_attention(query, keys, values, **kwargs)
    torch.cuda.synchronize()
    result = {"matches": [], "available": [], "errors": {}}
    for name, backend in BACKENDS.items():
        try:
            with sdpa_kernel(backend):
                candidate = F.scaled_dot_product_attention(
                    query, keys, values, **kwargs)
            torch.cuda.synchronize()
            result["available"].append(name)
            if torch.equal(candidate, reference):
                result["matches"].append(name)
            else:
                result[f"deviation_vs_{name}"] = (
                    (candidate - reference).abs().max().item())
        except Exception as error:  # noqa: BLE001
            result["errors"][name] = f"{type(error).__name__}: {str(error)[:180]}"
    return result


dims = head_engine.dims
groups = dims.num_kv_groups
item3 = {
    "dims": {
        "num_heads": dims.num_heads, "num_kv_heads": dims.num_kv_heads,
        "num_kv_groups": groups, "head_dim": dims.head_dim,
    },
    "shapes": {},
}

# The three shapes the engine actually issues, each measured BOTH ways: with
# enable_gqa (what ships now) and with repeat_kv (what shipped before). If the
# matching backend differs between the two, that IS the mechanism.
SHAPES = (
    ("fused_prefill", 8, 512, 512, False),
    ("fused_decode", 64, 1, 1024, False),
    ("single_causal_prefill", 1, 33, 33, True),
)

for label, batch, query_len, kv_len, causal in SHAPES:
    torch.manual_seed(0)
    query = torch.randn(batch, dims.num_heads, query_len, dims.head_dim,
                        dtype=torch.float16, device=DEVICE)
    keys = torch.randn(batch, dims.num_kv_heads, kv_len, dims.head_dim,
                       dtype=torch.float16, device=DEVICE)
    values = torch.randn(batch, dims.num_kv_heads, kv_len, dims.head_dim,
                         dtype=torch.float16, device=DEVICE)
    mask = None
    if not causal:
        mask = torch.zeros(batch, 1, query_len, kv_len,
                           dtype=torch.bool, device=DEVICE)
        mask[:, :, :, 0] = True
        mask[0, 0, :, : min(kv_len, 8)] = True

    common = dict(dropout_p=0.0, scale=dims.attn_scale)
    gqa_kwargs = dict(common, attn_mask=mask, is_causal=causal, enable_gqa=True)
    rep_kwargs = dict(common, attn_mask=mask, is_causal=causal)

    entry = {
        "batch": batch, "query_len": query_len, "kv_len": kv_len,
        "causal": causal,
        "with_enable_gqa": identify_backend(query, keys, values, **gqa_kwargs),
        "with_repeat_kv": identify_backend(
            query, repeat_kv(keys, groups), repeat_kv(values, groups), **rep_kwargs),
    }
    # And do the two paths agree numerically on THIS shape?
    a = F.scaled_dot_product_attention(query, keys, values, **gqa_kwargs)
    b = F.scaled_dot_product_attention(
        query, repeat_kv(keys, groups), repeat_kv(values, groups), **rep_kwargs)
    entry["gqa_vs_repeat_kv_max_abs"] = (a.float() - b.float()).abs().max().item()
    entry["gqa_vs_repeat_kv_bit_equal"] = bool(torch.equal(a, b))
    entry["same_backend"] = (
        entry["with_enable_gqa"]["matches"] == entry["with_repeat_kv"]["matches"])

    print(f"\n{label}: batch={batch} q={query_len} kv={kv_len} causal={causal}")
    print(f"   enable_gqa  -> matches {entry['with_enable_gqa']['matches']}")
    print(f"   repeat_kv   -> matches {entry['with_repeat_kv']['matches']}")
    print(f"   same backend: {entry['same_backend']}, "
          f"bit-equal: {entry['gqa_vs_repeat_kv_bit_equal']}, "
          f"max|diff|={entry['gqa_vs_repeat_kv_max_abs']:.6e}")
    item3["shapes"][label] = entry
    del query, keys, values, mask, a, b
    torch.cuda.empty_cache()

# The inputs themselves: is repeat_kv bit-exact against the unexpanded source?
# This is the claim the wrong explanation rested on, checked directly.
torch.manual_seed(0)
probe = torch.randn(2, dims.num_kv_heads, 64, dims.head_dim,
                    dtype=torch.float16, device=DEVICE)
expanded = repeat_kv(probe, groups)
manual = probe.repeat_interleave(groups, dim=1)
item3["repeat_kv_is_bit_exact_on_inputs"] = bool(torch.equal(expanded, manual))
item3["repeat_kv_dtype_unchanged"] = str(expanded.dtype) == str(probe.dtype)
print(f"\nrepeat_kv vs repeat_interleave, bit-equal: "
      f"{item3['repeat_kv_is_bit_exact_on_inputs']}, "
      f"dtype unchanged: {item3['repeat_kv_dtype_unchanged']}")
print("-> the expansion itself is exact; anything that moved came from the kernel",
      flush=True)

report["items"]["item3_sdpa_backend"] = item3
report["precision_settings"] = run_mod._precision_settings()

section("DONE")
REPORT.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {REPORT}", flush=True)
