"""PREFLIGHT: is the accelerator actually available, and does the transport work?

Two minutes of quota to answer three questions that would otherwise be answered
thirty minutes into Session 0:

  1. Is there a GPU? Phone verification gates the accelerator on Kaggle, not just
     internet, so an unverified account fails HERE rather than at runtime.
  2. Is there internet? Session 0 needs it for pip and the model pull.
  3. Did the repo dataset attach, and can it be read?

Nothing is installed and nothing is measured. This exists purely so that a
failed precondition costs two minutes instead of a session.
"""

import json
import subprocess
import sys
from pathlib import Path

report: dict = {}


def run(command):
    print(f"$ {' '.join(command)}", flush=True)
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"  ({error!r})", flush=True)
        return None
    print(done.stdout or done.stderr, flush=True)
    return done


print("=" * 70)
print("1. ACCELERATOR")
print("=" * 70)

smi = run(["nvidia-smi"])
report["nvidia_smi_ok"] = bool(smi and smi.returncode == 0)

query = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"])
report["gpu_query"] = query.stdout.strip() if query and query.returncode == 0 else None

try:
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    report.update({
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": count,
        "devices": [torch.cuda.get_device_name(i) for i in range(count)],
        "capabilities": [list(torch.cuda.get_device_capability(i)) for i in range(count)],
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if count else False,
    })
    # The two shapes Kaggle's free tier serves. Named, because they are not
    # interchangeable: 2x T4 is two devices, 1x P100 is one bigger one.
    names = " ".join(report["devices"]).upper()
    if count == 2 and "T4" in names:
        report["accelerator_shape"] = "dual T4"
    elif count == 1 and "P100" in names:
        report["accelerator_shape"] = "single P100"
    elif count:
        report["accelerator_shape"] = f"{count}x {report['devices'][0]}"
    else:
        report["accelerator_shape"] = "NONE"

    if count:
        # Prove a kernel actually launches, not just that the device enumerates.
        probe = (torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")).sum()
        torch.cuda.synchronize()
        report["matmul_probe_finite"] = bool(torch.isfinite(probe).item())
except Exception as error:  # noqa: BLE001
    report["torch_error"] = repr(error)

print(json.dumps({k: v for k, v in report.items()}, indent=2), flush=True)

print("=" * 70)
print("2. INTERNET")
print("=" * 70)
try:
    import urllib.request

    with urllib.request.urlopen("https://pypi.org/simple/", timeout=30) as response:
        report["internet_ok"] = response.status == 200
        print(f"pypi.org/simple -> HTTP {response.status}", flush=True)
except Exception as error:  # noqa: BLE001
    report["internet_ok"] = False
    report["internet_error"] = repr(error)
    print(f"NO INTERNET: {error!r}", file=sys.stderr, flush=True)

print("=" * 70)
print("3. REPO TRANSPORT")
print("=" * 70)
root = Path("/kaggle/input")
print(f"/kaggle/input exists: {root.exists()}", flush=True)

# Walk it. The mount layout is not what the docs imply -- the first two runs
# found a bare directory called "datasets" and nothing else -- so stop guessing
# at the shape and just look. Bounded so a huge mount cannot flood the log.
shown = 0
for path in sorted(root.rglob("*")) if root.exists() else []:
    if shown >= 60:
        print("  ... (truncated at 60 entries)", flush=True)
        break
    kind = "d" if path.is_dir() else "f"
    print(f"  {kind} {path}", flush=True)
    shown += 1

# Find the repo wherever it actually landed, by looking for the file that
# identifies it rather than by assuming a path.
found = None
if root.exists():
    for marker in root.rglob("scheduler.py"):
        if marker.parent.name == "engine":
            found = marker.parent.parent
            break
print(f"\nrepo root resolved to: {found}", flush=True)
report["repo_mounted"] = str(found) if found else None
if found:
    files = sum(1 for _ in found.rglob("*.py"))
    report["repo_python_files"] = files
    print(f"repo at {found}, {files} python files", flush=True)
else:
    print("REPO NOT MOUNTED", file=sys.stderr, flush=True)

print("=" * 70)
print("PREFLIGHT VERDICT")
print("=" * 70)
ready = (
    report.get("cuda_available") and report.get("device_count", 0) >= 1
    and report.get("internet_ok") and report.get("repo_mounted")
)
report["ready_for_session0"] = bool(ready)
for key in ("accelerator_shape", "internet_ok", "repo_mounted", "ready_for_session0"):
    print(f"  {key}: {report.get(key)}", flush=True)

Path("/kaggle/working/preflight_report.json").write_text(json.dumps(report, indent=2))
print("\nwrote /kaggle/working/preflight_report.json", flush=True)
