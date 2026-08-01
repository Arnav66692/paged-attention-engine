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

SPLIT ACROSS KERNELS, ONE CELL PER KERNEL, BOTH ARMS INSIDE IT. Kaggle caps a
session below the full matrix, so the matrix is sliced -- but the slice boundary
is chosen to protect the comparison. Arm A and Arm B for a given cell run
back to back in the SAME session on the SAME physical T4, because the question
is whether A and B differ, and splitting them across two Kaggle hosts would fold
host-to-host variance into exactly the difference being measured. Different
CELLS may land on different hosts; that is fine, since cells are never compared
to each other.

The slice is taken from this file's own name, so the four kernels are identical
copies and there is no environment to get wrong. Each cell writes its own result
file as it finishes, so a session that dies after Arm A still leaves Arm A's
data, and `bench/arms.py --compare-only` joins whatever exists.
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
# Which cell this kernel owns, keyed by its own filename. Both arms, always.
SLICES = {
    "session6a": "control-rect3072-C64",
    "session6b": "control-rect1757-C64",
    "session6c": "mixed-rect3072-C64",
    "session6d": "mixed-rect1757-C64",
}
# DECLARED, NOT DERIVED. Kaggle renames an uploaded script to `script.py`
# before running it, so `Path(__file__).stem` is "script" on the measurement
# host and every slice would refuse to start. The four files are otherwise
# identical copies; this line is the only difference between them, and
# tests/test_bench.py checks it against the directory it lives in.
SLICE = "session6c"
if SLICE not in SLICES:
    raise SystemExit(
        f"{SLICE} is not a known slice. Known: {sorted(SLICES)}. The cell is "
        "taken from the filename so the four kernels can be identical copies."
    )
ARMS = ["A", "B"]
CELLS = [SLICES[SLICE]]

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
print(f"slice={SLICE}: arms={ARMS} cell={CELLS[0]}", flush=True)

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
    "slice": SLICE,
    "cells_requested": CELLS,
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
