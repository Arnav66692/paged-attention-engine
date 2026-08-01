"""SESSION 6: the two-arm sweep. THIS ONE PRODUCES THE GATE 1 NUMBERS.

Both arms, both headlines, both rectangle sizings, three passes with the
escalation rule, run through `bench/arms.py` so the arms differ in exactly two
knobs and nothing else.

    ARM A  engine GQA_PATH=sdpa       baseline unpatched
    ARM B  engine GQA_PATH=repeat_kv  baseline forced to pre-expand K/V

WHY TWICE. Both sides call the same SDPA, torch picks the kernel from input
SHAPE, and the two sides can land on different kernels without anyone choosing
it. Whether that biases the ratio -- and in which direction -- is unmeasured and
the two mechanisms known to be in play have opposite signs. Running both
configurations is the only way to find out, so the answer is a measurement
rather than a hedge in a docstring.

SPLIT ACROSS KERNELS. Kaggle caps a session well below the full matrix, so
SESSION6_ARMS and SESSION6_CELLS select a slice. Each cell writes its own result
file; `bench/arms.py --compare-only` joins them afterwards. A slice that dies
loses its own cell and nothing else.
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

WORKING = Path(os.environ.get("SESSION6_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION6_INPUT", "/kaggle/input"))
RESULTS = WORKING / "arms"
REPORT = WORKING / "session6_report.json"
ARMS = [a for a in os.environ.get("SESSION6_ARMS", "A,B").split(",") if a]
CELLS = [c for c in os.environ.get("SESSION6_CELLS", "").split(",") if c]

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
RESULTS.mkdir(parents=True, exist_ok=True)

print(f"device: {torch.cuda.get_device_name(0)} "
      f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}", flush=True)
print(f"arms={ARMS} cells={CELLS or 'ALL'}", flush=True)

argv = [sys.executable, "-m", "bench.arms", "--out-dir", str(RESULTS)]
for arm in ARMS:
    argv += ["--arm", arm]
for cell in CELLS:
    argv += ["--cell", cell]

started = time.perf_counter()
done = subprocess.run(argv, env=dict(os.environ))
elapsed = time.perf_counter() - started

payload = {
    "arms_requested": ARMS,
    "cells_requested": CELLS or "ALL",
    "returncode": done.returncode,
    "elapsed_seconds": round(elapsed, 2),
    "device": torch.cuda.get_device_name(0),
    "result_files": sorted(p.name for p in RESULTS.glob("*.json")),
}
summary = RESULTS / "arms_summary.json"
if summary.exists():
    payload["summary"] = json.loads(summary.read_text())
REPORT.write_text(json.dumps(payload, indent=2, default=str))
print(f"\nelapsed {elapsed / 60:.1f} min, returncode {done.returncode}")
print(f"wrote {REPORT}", flush=True)
if done.returncode != 0:
    raise SystemExit(done.returncode)
