"""SESSION 3: which backend does the BASELINE use, and control@C64 under pressure.

Item 1 decides the GQA question and was missing from the option list. HF's Qwen2
attention also calls SDPA. If it materialises its own expansion it gets
EFFICIENT_ATTENTION, and engine-on-MATH against baseline-on-EFFICIENT is a
KERNEL comparison with paging as a confound -- with the handicap on our side.

So this does not assume the baseline's shapes. It intercepts the real calls HF
makes during a real generate, records them, and identifies the backend for each.

Item 2 then reruns the pressure probe at control@C64 -- rectangular 512+512,
every request exactly 64 blocks, 64 x 64 = 4096 against 3584 -- which is the
headline the zero-preemptions guard protects and which the mixed stream at C=64
never reaches (64 x 47.26 = 3025).
"""

import json
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

WORKING = Path(os.environ.get("SESSION3_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION3_INPUT", "/kaggle/input"))
REPO_DEST = WORKING / "repo"
REPORT = WORKING / "session3_report.json"
HEADLINE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HEADLINE_SHA = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PRESSURE_STEPS = int(os.environ.get("SESSION3_STEPS", "900"))
TARGET_PREEMPTIONS = int(os.environ.get("SESSION3_TARGET_PREEMPTS", "12"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("SESSION3_BATCHED_TOKENS", "8192"))

report: dict = {}


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def sh(cmd, timeout=5400, cwd=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    try:
        d = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess(cmd, 127, "", repr(e))
    if d.stdout:
        print(d.stdout, flush=True)
    if d.stderr:
        print(d.stderr, file=sys.stderr, flush=True)
    return d


section("SETUP")
WORKING.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402

(WORKING / "c.txt").write_text(f"torch=={torch.__version__}\n")
sh([sys.executable, "-m", "pip", "install", "--quiet", "-c", str(WORKING / "c.txt"),
    "transformers==5.12.1"], timeout=1800)

source = None
for marker in INPUT_ROOT.rglob("scheduler.py"):
    if marker.parent.name == "engine":
        source = marker.parent.parent
        break
if source is None:
    REPORT.write_text(json.dumps({"fatal": "repo not mounted"}))
    raise SystemExit(1)
if REPO_DEST.exists():
    shutil.rmtree(REPO_DEST)
shutil.copytree(source, REPO_DEST)
sys.path.insert(0, str(REPO_DEST))
os.chdir(REPO_DEST)

DEVICE = "cuda"
total_bytes = torch.cuda.get_device_properties(0).total_memory
report["device"] = {
    "name": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "total_gib": round(total_bytes / 2**30, 2),
}
print(json.dumps(report["device"]), flush=True)

import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

BACKENDS = {
    "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
    "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
    "MATH": SDPBackend.MATH,
    "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
}
_REAL_SDPA = F.scaled_dot_product_attention


def identify(query, keys, values, **kwargs):
    """Which kernel does the unrestricted call dispatch to? Determined by running it."""
    reference = _REAL_SDPA(query, keys, values, **kwargs)
    torch.cuda.synchronize()
    out = {"matches": [], "rejected": []}
    for name, backend in BACKENDS.items():
        try:
            with sdpa_kernel(backend):
                candidate = _REAL_SDPA(query, keys, values, **kwargs)
            torch.cuda.synchronize()
            if torch.equal(candidate, reference):
                out["matches"].append(name)
        except Exception:  # noqa: BLE001
            out["rejected"].append(name)
    return out


# --- ITEM 1: the baseline's OWN calls, intercepted --------------------------

section("ITEM 1: which backend does the HF baseline dispatch to?")

captured: list[dict] = []


def _recording_sdpa(query, key, value, *args, **kwargs):
    captured.append({
        "q_shape": list(query.shape), "k_shape": list(key.shape),
        "v_shape": list(value.shape),
        "is_causal": bool(kwargs.get("is_causal", False)),
        "has_mask": kwargs.get("attn_mask") is not None,
        "enable_gqa": bool(kwargs.get("enable_gqa", False)),
        "dtype": str(query.dtype),
    })
    return _REAL_SDPA(query, key, value, *args, **kwargs)


tokenizer = AutoTokenizer.from_pretrained(HEADLINE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    HEADLINE_MODEL, dtype=torch.float16, revision=HEADLINE_SHA,
    attn_implementation="sdpa")
model.eval()
model.to(DEVICE)

# Patch every module that holds a reference, not just torch.nn.functional --
# a `from torch.nn.functional import ...` at import time would dodge one patch.
patched = []
for module_name, module in list(sys.modules.items()):
    if module is None:
        continue
    attribute = getattr(module, "scaled_dot_product_attention", None)
    if attribute is _REAL_SDPA:
        setattr(module, "scaled_dot_product_attention", _recording_sdpa)
        patched.append(module_name)
report["patched_modules"] = patched[:40]
print(f"patched {len(patched)} module(s) holding SDPA", flush=True)

prompt_ids = torch.randint(0, 100, (4, 512), device=DEVICE)
with torch.inference_mode():
    model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                   max_new_tokens=4, min_new_tokens=4, do_sample=False,
                   use_cache=True, pad_token_id=0, eos_token_id=None)

for module_name in patched:
    setattr(sys.modules[module_name], "scaled_dot_product_attention", _REAL_SDPA)

print(f"captured {len(captured)} SDPA calls from the baseline", flush=True)
signatures = {}
for call in captured:
    key = json.dumps({k: call[k] for k in
                      ("q_shape", "k_shape", "is_causal", "has_mask", "enable_gqa")})
    signatures.setdefault(key, 0)
    signatures[key] += 1

item1 = {"num_calls": len(captured), "distinct_signatures": len(signatures),
         "shapes": []}
for key, count in list(signatures.items())[:8]:
    call = json.loads(key)
    q_shape, k_shape = call["q_shape"], call["k_shape"]
    kind = "prefill" if q_shape[2] > 1 else "decode"
    # Does the baseline expand its own KV? q heads == k heads means it did.
    expands = q_shape[1] == k_shape[1]
    torch.manual_seed(0)
    query = torch.randn(*q_shape, dtype=torch.float16, device=DEVICE)
    keys = torch.randn(*k_shape, dtype=torch.float16, device=DEVICE)
    values = torch.randn(*k_shape, dtype=torch.float16, device=DEVICE)
    kwargs = {"dropout_p": 0.0, "is_causal": call["is_causal"]}
    if call["enable_gqa"]:
        kwargs["enable_gqa"] = True
    if call["has_mask"]:
        kwargs["attn_mask"] = torch.zeros(
            q_shape[0], 1, q_shape[2], k_shape[2], dtype=torch.bool, device=DEVICE)
        kwargs["attn_mask"][:, :, :, 0] = True
    entry = {
        "kind": kind, "count": count, "q_shape": q_shape, "k_shape": k_shape,
        "is_causal": call["is_causal"], "has_mask": call["has_mask"],
        "enable_gqa": call["enable_gqa"],
        "baseline_expands_kv_itself": expands,
        "backend": identify(query, keys, values, **kwargs),
    }
    print(f"  {kind:8s} q={q_shape} k={k_shape} causal={call['is_causal']} "
          f"mask={call['has_mask']} enable_gqa={call['enable_gqa']} "
          f"expands_kv={expands} -> {entry['backend']['matches']}", flush=True)
    item1["shapes"].append(entry)
    del query, keys, values
    torch.cuda.empty_cache()

baseline_backends = sorted({b for s in item1["shapes"] for b in s["backend"]["matches"]})
item1["baseline_backends"] = baseline_backends
report["item1_baseline_backend"] = item1
print(f"\nBASELINE BACKENDS: {baseline_backends}", flush=True)

del model
torch.cuda.empty_cache()


# --- THE DECISION -----------------------------------------------------------

section("DECISION: does the engine's backend match the baseline's?")

from engine import attention as attention_mod  # noqa: E402
from engine.model import Qwen2Engine  # noqa: E402

head_model = AutoModelForCausalLM.from_pretrained(
    HEADLINE_MODEL, dtype=torch.float16, revision=HEADLINE_SHA)
head_model.eval()
head_model.to(DEVICE)
head_engine = Qwen2Engine.from_hf_model(head_model)
dims = head_engine.dims
groups = dims.num_kv_groups

ENGINE_SHAPES = (
    ("fused_prefill", 8, 512, 512, False, True),
    ("fused_decode", 64, 1, 1024, False, True),
)

decision = {"engine_paths": {}}
for path in ("sdpa", "repeat_kv"):
    per_shape = {}
    for label, batch, query_len, kv_len, causal, masked in ENGINE_SHAPES:
        torch.manual_seed(0)
        query = torch.randn(batch, dims.num_heads, query_len, dims.head_dim,
                            dtype=torch.float16, device=DEVICE)
        keys = torch.randn(batch, dims.num_kv_heads, kv_len, dims.head_dim,
                           dtype=torch.float16, device=DEVICE)
        values = torch.randn(batch, dims.num_kv_heads, kv_len, dims.head_dim,
                             dtype=torch.float16, device=DEVICE)
        mask = None
        if masked:
            mask = torch.zeros(batch, 1, query_len, kv_len, dtype=torch.bool,
                               device=DEVICE)
            mask[:, :, :, 0] = True
        kwargs = {"dropout_p": 0.0, "scale": dims.attn_scale,
                  "attn_mask": mask, "is_causal": causal}
        if path == "sdpa":
            per_shape[label] = identify(query, keys, values, enable_gqa=True, **kwargs)
        else:
            per_shape[label] = identify(
                query,
                attention_mod.repeat_kv(keys, groups),
                attention_mod.repeat_kv(values, groups), **kwargs)
        del query, keys, values, mask
        torch.cuda.empty_cache()
    decision["engine_paths"][path] = per_shape
    backends = sorted({b for s in per_shape.values() for b in s["matches"]})
    print(f"  engine GQA_PATH={path:10s} -> {backends}", flush=True)
    decision[f"backends_{path}"] = backends

matches_baseline = {
    path: sorted(set(decision[f"backends_{path}"])) == baseline_backends
    for path in ("sdpa", "repeat_kv")
}
decision["matches_baseline"] = matches_baseline
if matches_baseline["repeat_kv"] and not matches_baseline["sdpa"]:
    chosen = "repeat_kv"
    reason = ("repeat_kv lands on the same backend as the baseline; enable_gqa "
              "does not, which would hand the engine a kernel handicap")
elif matches_baseline["sdpa"] and not matches_baseline["repeat_kv"]:
    chosen = "sdpa"
    reason = "enable_gqa matches the baseline's backend and repeat_kv does not"
elif matches_baseline["sdpa"] and matches_baseline["repeat_kv"]:
    chosen = "sdpa"
    reason = "both paths match the baseline; keeping the allocation-free one"
else:
    chosen = "repeat_kv"
    reason = ("neither matches the baseline; repeat_kv at least reaches a fused "
              "kernel rather than MATH")
decision["chosen_gqa_path"] = chosen
decision["reason"] = reason
print(f"\nCHOSEN: GQA_PATH={chosen}\n  because {reason}", flush=True)

# Can EFFICIENT be pinned globally without leaving some shape with no kernel?
pin_check = {}
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
for label, batch, query_len, kv_len, causal, masked in ENGINE_SHAPES:
    torch.manual_seed(0)
    query = torch.randn(batch, dims.num_heads, query_len, dims.head_dim,
                        dtype=torch.float16, device=DEVICE)
    keys = torch.randn(batch, dims.num_kv_heads, kv_len, dims.head_dim,
                       dtype=torch.float16, device=DEVICE)
    values = torch.randn(batch, dims.num_kv_heads, kv_len, dims.head_dim,
                         dtype=torch.float16, device=DEVICE)
    mask = torch.zeros(batch, 1, query_len, kv_len, dtype=torch.bool, device=DEVICE)
    mask[:, :, :, 0] = True
    try:
        _REAL_SDPA(query, attention_mod.repeat_kv(keys, groups),
                   attention_mod.repeat_kv(values, groups),
                   attn_mask=mask, dropout_p=0.0, scale=dims.attn_scale)
        pin_check[label] = "ok"
    except Exception as error:  # noqa: BLE001
        pin_check[label] = f"{type(error).__name__}: {str(error)[:120]}"
    del query, keys, values, mask
    torch.cuda.empty_cache()
# restore
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
decision["efficient_only_pin_check"] = pin_check
print(f"pinning EFFICIENT only, repeat_kv path: {pin_check}", flush=True)
report["decision"] = decision


# --- ITEM 2: control@C64 under real pressure --------------------------------

section("ITEM 2: control@C64 pressure probe, the headline the guard protects")

os.environ["PAGED_ATTENTION_GQA_PATH"] = chosen
attention_mod.GQA_PATH = chosen
print(f"engine GQA_PATH set to {attention_mod.GQA_PATH}", flush=True)

from bench import run as run_mod  # noqa: E402
from bench import workload  # noqa: E402
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS, PRESSURE_C  # noqa: E402
from engine.cache import BlockAllocator  # noqa: E402
from engine.scheduler import (  # noqa: E402
    EventKind, ExecutionMode, Qwen2Executor, Request, Scheduler,
)

run_mod.pin_fp16_reduction()


class StepRecorder:
    def __init__(self, inner):
        self.inner = inner
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
        self.max_batch = max(self.max_batch, len(inputs))
        self.max_query = max([self.max_query] + widths)
        if widths:
            self.max_padded = max(self.max_padded, len(widths) * max(widths))
        return self.inner.forward_batch(inputs)


stream = workload.build_stream("control", seed=0, c=PRESSURE_C)
per_request_blocks = -(-stream[0].total_len // BLOCK_SIZE)
resident_demand = PRESSURE_C * per_request_blocks
item2 = {
    "stream": "control", "concurrency": PRESSURE_C,
    "num_requests": len(stream),
    "blocks_per_request": per_request_blocks,
    "resident_demand_at_C": resident_demand,
    "pool": NUM_GPU_BLOCKS,
    "shortfall_pct": round(100 * (resident_demand - NUM_GPU_BLOCKS) / resident_demand, 1),
    "gqa_path": chosen,
    "max_num_seqs": PRESSURE_C,
    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
}
print(f"control@C{PRESSURE_C}: {len(stream)} requests x {per_request_blocks} blocks, "
      f"resident demand {resident_demand} vs pool {NUM_GPU_BLOCKS} "
      f"({item2['shortfall_pct']}% shortfall)", flush=True)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
executor = None
scheduler = None
try:
    allocator = BlockAllocator(num_blocks=NUM_GPU_BLOCKS, block_size=BLOCK_SIZE)
    cache = head_engine.new_paged_cache(NUM_GPU_BLOCKS, BLOCK_SIZE)
    executor = StepRecorder(Qwen2Executor(head_engine, cache, allocator))
    scheduler = Scheduler(executor, allocator, mode=ExecutionMode.FUSED,
                          max_num_seqs=PRESSURE_C,
                          max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS)
    rejected = 0
    for item in stream:
        if not scheduler.add_request(Request(
                request_id=item.request_id,
                prompt_token_ids=item.prompt_token_ids,
                max_new_tokens=item.max_new_tokens)):
            rejected += 1
    item2["rejected"] = rejected

    steps = 0
    started = time.perf_counter()
    records = []
    while (scheduler.waiting or scheduler.running) and steps < PRESSURE_STEPS:
        result = scheduler.step()
        steps += 1
        records.append({"step": result.step, "batch_size": result.batch_size,
                        "padded": result.batched_token_cost,
                        "deferred": len(result.deferred),
                        "blocks_used": NUM_GPU_BLOCKS - result.free_blocks})
        if len(scheduler.events_of(EventKind.PREEMPT)) >= TARGET_PREEMPTIONS:
            print(f"  reached {TARGET_PREEMPTIONS} preemptions at step {steps}",
                  flush=True)
            break
    torch.cuda.synchronize()
    item2["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    item2["ok"] = True
except torch.cuda.OutOfMemoryError as error:
    item2["ok"] = False
    item2["oom"] = True
    item2["error"] = str(error)[:1500]
    print(f"control@C64: OUT OF MEMORY\n{error}", file=sys.stderr, flush=True)
except Exception as error:  # noqa: BLE001
    item2["ok"] = False
    item2["error"] = f"{type(error).__name__}: {error}"[:1500]
    item2["traceback"] = traceback.format_exc()[-2500:]

if executor is not None:
    item2["max_batch_width"] = executor.max_batch
    item2["max_query_len"] = executor.max_query
    item2["max_padded_slots"] = executor.max_padded
if scheduler is not None:
    preempts = scheduler.events_of(EventKind.PREEMPT)
    recomputes = scheduler.events_of(EventKind.RECOMPUTE)
    item2["preemptions"] = len(preempts)
    item2["recomputes"] = len(recomputes)
    item2["peak_blocks_used"] = max(
        (NUM_GPU_BLOCKS - e.free_blocks for e in scheduler.events), default=0)
    item2["peak_blocks_pct"] = round(100 * item2["peak_blocks_used"] / NUM_GPU_BLOCKS, 1)
    item2["steps_run"] = len(records)
    item2["total_deferred"] = sum(r["deferred"] for r in records)
    by_step = {r["step"]: r for r in records}
    recompute_steps = sorted({e.step for e in recomputes})
    item2["recompute_steps"] = recompute_steps[:40]
    detail = [by_step[s] for s in recompute_steps[:20] if s in by_step]
    item2["recompute_step_detail"] = detail
    if detail:
        item2["max_padded_on_a_recompute_step"] = max(d["padded"] for d in detail)
        item2["max_batch_on_a_recompute_step"] = max(d["batch_size"] for d in detail)

item2["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
item2["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
item2["reserved_pct_of_card"] = round(
    100 * torch.cuda.max_memory_reserved() / total_bytes, 1)
item2["headroom_gib"] = round(
    (total_bytes - torch.cuda.max_memory_reserved()) / 2**30, 3)

print(f"\ncontrol@C64: ok={item2.get('ok')} oom={item2.get('oom', False)} "
      f"preempts={item2.get('preemptions')} recomputes={item2.get('recomputes')} "
      f"blocks={item2.get('peak_blocks_used')} ({item2.get('peak_blocks_pct')}%) "
      f"alloc={item2['peak_allocated_gib']} reserved={item2['peak_reserved_gib']} "
      f"({item2['reserved_pct_of_card']}%)", flush=True)
if item2.get("recompute_step_detail"):
    print(f"  {'step':>6} {'batch':>6} {'padded':>8} {'deferred':>9} {'blocks':>7}")
    for d in item2["recompute_step_detail"]:
        print(f"  {d['step']:>6} {d['batch_size']:>6} {d['padded']:>8} "
              f"{d['deferred']:>9} {d['blocks_used']:>7}", flush=True)

report["item2_control_c64_pressure"] = item2
report["precision_settings"] = run_mod._precision_settings()
report["sdpa_backends_enabled"] = run_mod.sdpa_backends_available()

section("DONE")
REPORT.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {REPORT}", flush=True)
