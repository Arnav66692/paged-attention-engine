"""SESSION 4: the speed half of the GQA decision. A KERNEL DIAGNOSTIC, NOT GATE 1.

Session 3 settled the fairness question: the HF baseline passes unexpanded K/V
with enable_gqa=True and lands on MATH, exactly as this engine does. They MATCH.
So the choice is no longer constrained by fairness -- provided both sides move
together -- and it falls to speed.

WHAT THIS MEASURES: two GQA paths through this engine's own attention, at the
shapes it actually issues. Nothing here is compared against the baseline and
nothing here is a throughput number. It exists to answer one implementation
question and to price what the current choice costs.

THE ASYMMETRY TRAP, stated so it cannot be walked into. Moving the ENGINE to
repeat_kv/EFFICIENT while the baseline stays on MATH would hand the engine the
fast kernel and leave the baseline on the slow one -- the same confound Session 3
was run to avoid, pointing the other way. Any move has to move both, and the
baseline's kernel is chosen by HuggingFace's own code.
"""

import json
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

WORKING = Path(os.environ.get("SESSION4_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION4_INPUT", "/kaggle/input"))
REPORT = WORKING / "session4_report.json"
HEADLINE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HEADLINE_SHA = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WARMUP = 3
REPEATS = 20

report: dict = {"disclaimer": "kernel diagnostic; NOT the Gate 1 throughput number"}
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

DEVICE = "cuda"
import torch.nn.functional as F  # noqa: E402

from engine.attention import repeat_kv  # noqa: E402

report["device"] = torch.cuda.get_device_name(0)
report["capability"] = list(torch.cuda.get_device_capability(0))

# Qwen2.5-1.5B geometry, so the shapes are the real ones.
HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128
GROUPS = HEADS // KV_HEADS
SCALE = HEAD_DIM ** -0.5

SHAPES = (
    ("fused_prefill_8x512", 8, 512, 512),
    ("fused_decode_64x1024", 64, 1, 1024),
    ("fused_decode_64x2048", 64, 1, 2048),
    ("pressure_prefill_16x512", 16, 512, 512),
)


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(REPEATS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / REPEATS


results = {}
for label, batch, query_len, kv_len in SHAPES:
    torch.manual_seed(0)
    query = torch.randn(batch, HEADS, query_len, HEAD_DIM,
                        dtype=torch.float16, device=DEVICE)
    keys = torch.randn(batch, KV_HEADS, kv_len, HEAD_DIM,
                       dtype=torch.float16, device=DEVICE)
    values = torch.randn(batch, KV_HEADS, kv_len, HEAD_DIM,
                         dtype=torch.float16, device=DEVICE)
    mask = torch.zeros(batch, 1, query_len, kv_len, dtype=torch.bool, device=DEVICE)
    mask[:, :, :, 0] = True

    def call_sdpa():
        return F.scaled_dot_product_attention(
            query, keys, values, attn_mask=mask, dropout_p=0.0,
            scale=SCALE, enable_gqa=True)

    def call_repeat():
        return F.scaled_dot_product_attention(
            query, repeat_kv(keys, GROUPS), repeat_kv(values, GROUPS),
            attn_mask=mask, dropout_p=0.0, scale=SCALE)

    entry = {"batch": batch, "query_len": query_len, "kv_len": kv_len}
    for name, fn in (("sdpa_enable_gqa", call_sdpa), ("repeat_kv", call_repeat)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            seconds = timeit(fn)
            entry[name] = {
                "seconds_per_call": seconds,
                "ms_per_call": round(seconds * 1e3, 4),
                "peak_transient_mib": round(
                    (torch.cuda.max_memory_allocated() - base) / 2**20, 1),
                "ok": True,
            }
        except Exception as error:  # noqa: BLE001
            entry[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"[:300]}

    if entry["sdpa_enable_gqa"].get("ok") and entry["repeat_kv"].get("ok"):
        a = entry["sdpa_enable_gqa"]["seconds_per_call"]
        b = entry["repeat_kv"]["seconds_per_call"]
        entry["repeat_kv_speedup_vs_sdpa"] = round(a / b, 3)
    results[label] = entry
    print(f"\n{label}: batch={batch} q={query_len} kv={kv_len}")
    print(f"   enable_gqa (MATH)      {entry['sdpa_enable_gqa'].get('ms_per_call')} ms  "
          f"transient {entry['sdpa_enable_gqa'].get('peak_transient_mib')} MiB")
    print(f"   repeat_kv  (EFFICIENT) {entry['repeat_kv'].get('ms_per_call')} ms  "
          f"transient {entry['repeat_kv'].get('peak_transient_mib')} MiB")
    print(f"   repeat_kv is {entry.get('repeat_kv_speedup_vs_sdpa')}x the speed "
          f"of enable_gqa", flush=True)
    del query, keys, values, mask
    torch.cuda.empty_cache()

report["shapes"] = results
speedups = [v["repeat_kv_speedup_vs_sdpa"] for v in results.values()
            if "repeat_kv_speedup_vs_sdpa" in v]
if speedups:
    report["median_repeat_kv_speedup"] = sorted(speedups)[len(speedups) // 2]
    report["range_repeat_kv_speedup"] = [min(speedups), max(speedups)]
    print(f"\nrepeat_kv/EFFICIENT vs enable_gqa/MATH: "
          f"median {report['median_repeat_kv_speedup']}x, "
          f"range {report['range_repeat_kv_speedup']}", flush=True)

REPORT.write_text(json.dumps(report, indent=2, default=str))
print(f"\nwrote {REPORT}", flush=True)
