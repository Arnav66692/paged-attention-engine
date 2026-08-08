"""SESSION 10: does sorting the static baseline remove the mixed-stream win?

The headline compares the engine against a static batcher that batches in
ARRIVAL order. A static batch runs every row until the batch's longest member
finishes its budget, so a 1024-token decode sharing a batch with an 8-token
decode makes the short row compute 1016 positions nobody reads. Grouping the
accepted list by decode budget is the cheapest thing a static batcher can do
about that, and it is one line. Until it is measured, the headline is a claim
about continuous batching against the WEAKER of two static batchers.

This session answers that, and only that.

    baseline_order=arrival   the batching every committed result used
    baseline_order=sorted    the accepted list grouped by decode budget first

Both are ARM B (`GQA_PATH=repeat_kv`, baseline forced to pre-expand K/V), which
is the arm the mixed headline is quoted from and the one where both sides run
the same SDPA kernel.

WHY THE ENGINE IS RERUN RATHER THAN REUSED. The committed Arm B engine level for
this cell was measured in session 6, and byte-identical engine work has been
observed 13.6% and 16.3% apart between two Kaggle sessions (7a vs 7b). A sorted
baseline from this host divided by an engine from that one would fold host drift
straight into the answer. So both baseline modes AND the engine run here, in one
session on one physical T4, and every ratio reported is same-session.

The engine therefore runs twice, once per `bench.run` invocation. That is not
waste: it pairs each baseline with an engine measured in its own process, and
the two engine levels bound this session's own reproducibility for free.

NOTHING ELSE MOVES. Same stream, same seed, same rectangle, same `max_resident`,
same KV ceiling, same rejection rule, same pass count and warmup. `bench/
workload.py` is untouched and `bench/baseline.py::order_for_batching` reorders
the ALREADY-ACCEPTED list, so acceptance, rejection and every token count are
computed before the sort and are order-invariant after it.

SLICE DECLARED, NOT DERIVED. Kaggle renames an uploaded script to `script.py`,
so `Path(__file__).stem` is "script" on the measurement host. The two files are
identical copies and the SLICE line is the only difference between them.
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
RESULTS = WORKING / "arms"
REPORT = WORKING / "session10_report.json"

# One cell per kernel, both baseline orders inside it, because the orders are
# what this session compares and splitting them across hosts would put host
# drift inside the comparison.
SLICES = {
    "session10a": {"stream": "mixed", "rectangle": 3072, "concurrency": 64},
    "session10b": {"stream": "mixed", "rectangle": 1757, "concurrency": 64},
}
SLICE = "session10a"
if SLICE not in SLICES:
    raise SystemExit(
        f"{SLICE} is not a known slice. Known: {sorted(SLICES)}. The cell is "
        "declared so the kernels can be identical copies."
    )
CELL = SLICES[SLICE]
ORDERS = ["arrival", "sorted"]
# Arm B, restated here rather than imported, because this file runs before the
# repo is on sys.path. `bench/arms.py::ARMS["B"]` is the source of truth and
# tests/test_bench.py checks these against it.
GQA_PATH = "repeat_kv"

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
if source is None:
    raise SystemExit("the repo is not in the dataset")
dest = WORKING / "repo"
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(source, dest)
sys.path.insert(0, str(dest))
os.chdir(dest)
RESULTS.mkdir(parents=True, exist_ok=True)

print(f"device: {torch.cuda.get_device_name(0)} "
      f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}", flush=True)
print(f"slice={SLICE}: cell={CELL} orders={ORDERS} arm=B", flush=True)

env = dict(os.environ)
env["PAGED_ATTENTION_GQA_PATH"] = GQA_PATH

runs = []
overall = time.perf_counter()
for order in ORDERS:
    out = RESULTS / (
        f"armB-{CELL['stream']}-rect{CELL['rectangle']}"
        f"-C{CELL['concurrency']}-order{order}.json"
    )
    argv = [
        sys.executable, "-m", "bench.run",
        "--stream", CELL["stream"],
        "-C", str(CELL["concurrency"]),
        "--rectangle", str(CELL["rectangle"]),
        "--baseline-expand-kv",
        "--baseline-order", order,
        "--out", str(out),
    ]
    print(f"\n=== baseline_order={order} (GQA_PATH={GQA_PATH}) ===", flush=True)
    started = time.perf_counter()
    done = subprocess.run(argv, env=env)
    elapsed = time.perf_counter() - started
    runs.append({
        "batch_order": order,
        "returncode": done.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "out": out.name,
        "written": out.exists(),
    })
    print(f"--- {order}: rc={done.returncode} in {elapsed / 60:.1f} min", flush=True)
    if done.returncode != 0:
        # A failed order is not a zero; it is a missing half of the comparison.
        break

payload = {
    "slice": SLICE,
    "cell": CELL,
    "arm": "B",
    "gqa_path": GQA_PATH,
    "orders_requested": ORDERS,
    "runs": runs,
    "elapsed_seconds": round(time.perf_counter() - overall, 2),
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "torch_version": torch.__version__,
    "result_files": sorted(p.name for p in RESULTS.glob("*.json")),
}
REPORT.write_text(json.dumps(payload, indent=2, default=str))
print(f"\ntotal {payload['elapsed_seconds'] / 60:.1f} min")
print(f"wrote {REPORT}", flush=True)
failed = [r for r in runs if r["returncode"] != 0]
if failed or len(runs) != len(ORDERS):
    raise SystemExit(failed[0]["returncode"] if failed else 1)
