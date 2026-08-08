"""SESSION 10: Stage 2 of docs/PREREG_arrival.md. rho = 0.4, one point.

Both sides in ONE session on ONE card, passes interleaved, because
between-session drift on this project measures 13.6 to 16.3 percent on
byte-identical work and that is larger than several effects reported here.

THROUGHPUT IS NOT THE METRIC. Below saturation both schedulers complete every
request and both match the arrival rate by construction. What is reported is
TTFT and end-to-end latency at p50 and p99, the baseline's realized decode
efficiency against the simulated 0.369, and the queue depth distribution.

The prediction being tested was written down first, in docs/PREREG_arrival.md
at commit a9dfd71, before bench/arrival.py existed.
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

WORKING = Path(os.environ.get("SESSION10_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION10_INPUT", "/kaggle/input"))
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

print(f"device: {torch.cuda.get_device_name(0)} "
      f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}", flush=True)

out = WORKING / "stage2.json"
started = time.perf_counter()
done = subprocess.run(
    [sys.executable, "-m", "bench.arrival_bench",
     "--rho", "0.4", "--rectangle", "3072", "--seed", "0",
     "--max-num-seqs", "64", "--passes", "3",
     "--rate", "0.196279",          # rho=0.4 x the Stage 1 saturated rate
     "--out", str(out)],
    env=dict(os.environ),
)
elapsed = time.perf_counter() - started
print(f"\nelapsed {elapsed / 60:.1f} min, rc={done.returncode}", flush=True)

report = {"returncode": done.returncode, "elapsed_seconds": round(elapsed, 1),
          "device": torch.cuda.get_device_name(0)}
if out.exists():
    report["stage2"] = json.loads(out.read_text())
(WORKING / "session10_report.json").write_text(json.dumps(report, indent=2))
print(f"wrote {WORKING / 'session10_report.json'}", flush=True)
if done.returncode != 0:
    raise SystemExit(done.returncode)
