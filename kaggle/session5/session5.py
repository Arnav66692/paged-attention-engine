"""SESSION 5: the gates for the two-arm kernel experiment. No throughput claim.

Before spending hours of GPU on two full sweeps, three things must be true, and
each of them can void the experiment:

  GATE 1  Arm B's patched baseline must be TOKEN-IDENTICAL to the unpatched one.
          Pre-expanding K/V is mathematically a no-op but it changes which
          kernel torch selects, and kernels differ in accumulation order. In
          fp16 that can move a token. If it does, Arm B is timing a different
          baseline and the arm is void.

  GATE 2  Which backend each side ACTUALLY selects, per arm, per stream. This is
          not assumable. HF takes enable_gqa only when the attention mask is
          None, which happens only when nothing was padded -- so the baseline's
          kernel depends on whether the stream's prompts are uniform. Measured
          on CPU already; confirmed here on the device that matters.

  GATE 3  The budget. Seconds per step for each side at the real shapes, so the
          full matrix can be costed before it is launched rather than after it
          runs out of quota.

Nothing here is a Gate 1 throughput number and nothing here is compared across
engine and baseline for a claim.
"""

import json
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
GQA_PATH_ENV = "PAGED_ATTENTION_GQA_PATH"

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

WORKING = Path(os.environ.get("SESSION5_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION5_INPUT", "/kaggle/input"))
REPORT = WORKING / "session5_report.json"
HEADLINE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

report: dict = {"disclaimer": "gates only; NO throughput number is produced here"}
WORKING.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402

(WORKING / "c.txt").write_text(f"torch=={torch.__version__}\n")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-c",
                str(WORKING / "c.txt"), "transformers==5.12.1"], timeout=1800)

source = None
for marker in INPUT_ROOT.rglob("scheduler.py"):
    if marker.parent.name == "engine":
        source = marker.parent.parent
        break
dest = WORKING / "repo"
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(source, dest)
sys.path.insert(0, str(dest))
os.chdir(dest)


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def save():
    REPORT.write_text(json.dumps(report, indent=2, default=str))


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

report["device"] = torch.cuda.get_device_name(0)
report["capability"] = list(torch.cuda.get_device_capability(0))
report["torch"] = torch.__version__

from bench import baseline as baseline_mod  # noqa: E402
from bench import workload  # noqa: E402
from bench.run import pin_fp16_reduction  # noqa: E402

pin_fp16_reduction()

section("loading the headline model, fp16, once")
model = AutoModelForCausalLM.from_pretrained(HEADLINE_MODEL, dtype=torch.float16)
model.eval()
model.to("cuda")
generator = baseline_mod.HuggingFaceGenerator(
    model, AutoTokenizer.from_pretrained(HEADLINE_MODEL), "cuda")
print(f"loaded. allocated {torch.cuda.memory_allocated() / 2**30:.3f} GiB", flush=True)

control = workload.build_stream("control", 0, 64)
mixed = workload.build_stream("mixed", 0, 64)
# Small batches: the gates are structural, not timing.
UNIFORM = list(control[:4])
RAGGED = list(mixed[:4])
print(f"uniform batch prompt lens: {[r.prompt_len for r in UNIFORM]}")
print(f"ragged  batch prompt lens: {[r.prompt_len for r in RAGGED]}", flush=True)

# ---------------------------------------------------------------- GATE 1
section("GATE 1: is Arm B's patched baseline token-identical?")
gate1 = {}
try:
    gate1 = baseline_mod.assert_token_identical(generator, [UNIFORM, RAGGED])
    gate1["arm_b_viable"] = True
    print(f"PASS  {gate1}")
except baseline_mod.BaselineNotIdentical as error:
    gate1 = {"arm_b_viable": False, "error": str(error)}
    print(f"ARM B VOID: {error}")
except Exception as error:  # noqa: BLE001
    gate1 = {"arm_b_viable": False, "error": f"{type(error).__name__}: {error}"}
    print(f"ARM B VOID (unexpected): {error}")
report["gate1_token_identity"] = gate1
save()

# ---------------------------------------------------------------- GATE 2
section("GATE 2: which backend does each side actually select?")


def identify(query, keys, values, **kwargs):
    reference = _REAL_SDPA(query, keys, values, **kwargs)
    torch.cuda.synchronize()
    matches = []
    for name, backend in BACKENDS.items():
        try:
            with sdpa_kernel(backend):
                candidate = _REAL_SDPA(query, keys, values, **kwargs)
            torch.cuda.synchronize()
            if torch.equal(candidate, reference):
                matches.append(name)
        except Exception:  # noqa: BLE001
            continue
    return matches


def observe_baseline(batch, expand_kv):
    """Distinct (shape, flags) signatures the baseline issues, each identified."""
    seen: dict = {}
    undo = baseline_mod.force_baseline_kv_expansion() if expand_kv else (lambda: None)

    def recording(query, key, value, *args, **kwargs):
        signature = (
            tuple(query.shape), tuple(key.shape),
            bool(kwargs.get("is_causal", False)),
            kwargs.get("attn_mask") is not None,
            bool(kwargs.get("enable_gqa", False)),
        )
        if signature not in seen:
            seen[signature] = {
                "q": list(query.shape), "k": list(key.shape),
                "is_causal": signature[2], "has_mask": signature[3],
                "enable_gqa": signature[4],
                "kv_heads_equal_q_heads": query.shape[1] == key.shape[1],
                "backends": identify(query, key, value, *args, **kwargs),
                "count": 0,
            }
        seen[signature]["count"] += 1
        return _REAL_SDPA(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = recording
    try:
        generator.generate_batch(batch)
    finally:
        F.scaled_dot_product_attention = _REAL_SDPA
        undo()
    return list(seen.values())


gate2: dict = {}
for stream_name, batch in (("control_uniform", UNIFORM), ("mixed_ragged", RAGGED)):
    for arm, expand in (("A_unpatched", False), ("B_patched", True)):
        key = f"baseline/{stream_name}/{arm}"
        try:
            calls = observe_baseline(batch, expand)
            backends = sorted({b for c in calls for b in c["backends"]})
            expands = [not c["enable_gqa"] for c in calls]
            gate2[key] = {"signatures": calls, "backends": backends,
                          "expands_kv": all(expands)}
            print(f"{key}: backends={backends} expands_kv={all(expands)} "
                  f"({len(calls)} distinct shapes)", flush=True)
        except Exception as error:  # noqa: BLE001
            gate2[key] = {"error": f"{type(error).__name__}: {error}"[:300]}
            print(f"{key}: ERROR {error}")
report["gate2_backends"] = gate2
save()

# The engine side has to run in its own process: GQA_PATH is read at import.
section("GATE 2b: the ENGINE's backends, one subprocess per GQA path")
PROBE = WORKING / "engine_probe.py"
PROBE.write_text('''
import json, os, sys, torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, os.environ["REPO"])
REAL = F.scaled_dot_product_attention
BACKENDS = {"FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
            "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
            "MATH": SDPBackend.MATH,
            "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION}
def identify(q, k, v, **kw):
    ref = REAL(q, k, v, **kw); torch.cuda.synchronize(); out = []
    for name, be in BACKENDS.items():
        try:
            with sdpa_kernel(be):
                cand = REAL(q, k, v, **kw)
            torch.cuda.synchronize()
            if torch.equal(cand, ref): out.append(name)
        except Exception: continue
    return out
from transformers import AutoModelForCausalLM
from engine import attention as attn
from engine.model import Qwen2Engine
from engine.cache import ADMISSION_MARGIN_BLOCKS, BlockAllocator
from engine.scheduler import ExecutionMode, Qwen2Executor, Scheduler
from bench import workload
from bench.run import _engine_requests
from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS
model = AutoModelForCausalLM.from_pretrained(os.environ["MODEL"], dtype=torch.float16)
model.eval(); model.to("cuda")
engine = Qwen2Engine.from_hf_model(model)
seen = {}
def recording(q, k, v, *a, **kw):
    sig = (tuple(q.shape), tuple(k.shape), bool(kw.get("is_causal", False)),
           kw.get("attn_mask") is not None, bool(kw.get("enable_gqa", False)))
    if sig not in seen:
        seen[sig] = {"q": list(q.shape), "k": list(k.shape), "is_causal": sig[2],
                     "has_mask": sig[3], "enable_gqa": sig[4],
                     "backends": identify(q, k, v, *a, **kw), "count": 0}
    seen[sig]["count"] += 1
    return REAL(q, k, v, *a, **kw)
# The harness's own construction, not a hand-rolled copy of it.
stream = workload.build_stream(os.environ["STREAM"], 0, 64)[:4]
allocator = BlockAllocator(num_blocks=NUM_GPU_BLOCKS, block_size=BLOCK_SIZE)
cache = engine.new_paged_cache(allocator.num_blocks, allocator.block_size)
scheduler = Scheduler(Qwen2Executor(engine, cache, allocator), allocator,
                      mode=ExecutionMode.FUSED,
                      admission_margin_blocks=ADMISSION_MARGIN_BLOCKS)
for request in _engine_requests(stream):
    scheduler.add_request(request)
F.scaled_dot_product_attention = recording
try:
    # A prefill step and a few decodes is every distinct shape the engine issues.
    for _ in range(int(os.environ.get("STEPS", "6"))):
        if not scheduler.step().batch_size: break
finally:
    F.scaled_dot_product_attention = REAL
print("PROBE_RESULT " + json.dumps(
    {"gqa_path": attn.GQA_PATH, "signatures": list(seen.values()),
     "backends": sorted({b for s in seen.values() for b in s["backends"]})}))
''')
gate2b: dict = {}
for gqa_path in ("sdpa", "repeat_kv"):
    for stream_name in ("control", "mixed"):
        key = f"engine/{stream_name}/{gqa_path}"
        env = dict(os.environ, REPO=str(dest), MODEL=HEADLINE_MODEL,
                   STREAM=stream_name, **{GQA_PATH_ENV: gqa_path})
        done = subprocess.run([sys.executable, str(PROBE)], env=env,
                              capture_output=True, text=True, timeout=2400)
        line = next((ln for ln in reversed((done.stdout or "").splitlines())
                     if ln.startswith("PROBE_RESULT ")), None)
        if line:
            gate2b[key] = json.loads(line[len("PROBE_RESULT "):])
            print(f"{key}: backends={gate2b[key]['backends']} "
                  f"({len(gate2b[key]['signatures'])} distinct shapes)", flush=True)
        else:
            err = "\n".join(
                ln for ln in (done.stderr or "").splitlines()
                if "it/s" not in ln and "Loading weights" not in ln)
            gate2b[key] = {"error": err[-1500:]}
            print(f"{key}: ERROR\n{err[-1500:]}", flush=True)
report["gate2b_engine_backends"] = gate2b
save()

# ---------------------------------------------------------------- summary
section("SUMMARY: is each arm symmetric?")
summary = {}
for arm, engine_path, baseline_key in (
    ("A", "sdpa", "A_unpatched"), ("B", "repeat_kv", "B_patched"),
):
    for stream_name, baseline_stream in (
        ("control", "control_uniform"), ("mixed", "mixed_ragged"),
    ):
        eng = gate2b.get(f"engine/{stream_name}/{engine_path}", {}).get("backends")
        base = gate2.get(f"baseline/{baseline_stream}/{baseline_key}", {}).get("backends")
        summary[f"arm{arm}/{stream_name}"] = {
            "engine_backends": eng, "baseline_backends": base,
            "symmetric": bool(eng) and bool(base) and set(eng) == set(base),
        }
        print(f"arm{arm}/{stream_name}: engine={eng} baseline={base} "
              f"symmetric={summary[f'arm{arm}/{stream_name}']['symmetric']}", flush=True)
report["summary_symmetry"] = summary
report["arm_b_viable"] = bool(gate1.get("arm_b_viable"))
save()
print(f"\nwrote {REPORT}", flush=True)
