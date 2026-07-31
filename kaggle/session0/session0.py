"""SESSION 0: PORT VERIFICATION ONLY. No sweep, no timing, no throughput number.

Runs on Kaggle's free tier (T4 or P100). Thirty minutes of work whose entire
purpose is to find out what breaks when this engine meets CUDA for the first
time. The gather path in engine/attention.py touches device placement, dtype
promotion, contiguity and index arithmetic, and all four are cheap to diagnose
here and expensive to diagnose halfway through a sweep.

The order below is deliberate:

  1. Record the environment. GPU, driver, torch, CUDA. Recorded, never assumed.
  2. Do NOT install torch. Kaggle ships its own CUDA build and forcing this
     repo's macOS arm64 pin would break it. transformers is installed under a
     pip CONSTRAINT that names the torch already present, and the version is
     re-checked afterwards -- "do not install torch" is verified, not hoped for.
  3. Install transformers==5.12.1 explicitly. engine/model.py was written
     against that source. If the image ships 4.x, parity fails immediately,
     which is exactly what this session exists to discover.
  4. pytest --fast, then pytest -m slow. Every failure reported verbatim.

Nothing here is allowed to fail silently. A step that cannot run says so and the
script keeps going, because a single broken step should not cost the whole
session.
"""

import json
import os

# ONE GPU, pinned before anything can import torch. Kaggle hands out dual T4 and
# the whole design is single-GPU: the KV ceiling of 1,644,167,168 bytes is sized
# against ONE 15360 MiB card. torch caches the device list at CUDA init, so this
# is only effective here, above the imports. conftest.py does the same for a
# direct pytest run.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# Kaggle's real mounts, overridable so this script can be dry-run off-Kaggle.
# tests/test_kaggle_session.py drives the whole thing against a fake mount on
# CPU, because a kernel script that dies in its first minute wastes a session
# and there is no reason to discover that on the GPU.
WORKING = Path(os.environ.get("SESSION0_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION0_INPUT", "/kaggle/input"))

REPO_DEST = WORKING / "repo"
REPORT = WORKING / "session0_report.json"
TRANSFORMERS_PIN = "5.12.1"

# The dry run points these at a fast subset. On Kaggle they are the real gate.
FAST_ARGS = json.loads(os.environ.get("SESSION0_FAST_ARGS", '["--fast", "-q", "-rA", "--tb=long"]'))
SLOW_ARGS = json.loads(os.environ.get("SESSION0_SLOW_ARGS", '["-m", "slow", "-q", "-rA", "--tb=long"]'))

report: dict = {"steps": [], "failures": []}


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def run(command, *, check=False, timeout=3600, cwd=None):
    """Run a command, echo everything, and keep the transcript.

    A missing binary or a timeout comes back as a failed result rather than an
    exception. That is the whole contract of this script: one step that cannot
    run must not take the session with it. Found by the local dry run, where
    nvidia-smi does not exist and the unguarded version died on line one.
    """
    printable = " ".join(command) if isinstance(command, list) else command
    print(f"$ {printable}", flush=True)
    try:
        completed = subprocess.run(
            command, shell=isinstance(command, str), capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
    except FileNotFoundError as error:
        print(f"  (not found: {error})", file=sys.stderr, flush=True)
        return subprocess.CompletedProcess(command, 127, "", f"not found: {error}")
    except subprocess.TimeoutExpired as error:
        print(f"  (timed out after {timeout}s)", file=sys.stderr, flush=True)
        return subprocess.CompletedProcess(
            command, 124, (error.stdout or b"").decode(errors="replace")
            if isinstance(error.stdout, bytes) else (error.stdout or ""),
            f"timed out after {timeout}s",
        )
    except OSError as error:
        print(f"  (could not execute: {error!r})", file=sys.stderr, flush=True)
        return subprocess.CompletedProcess(command, 126, "", repr(error))

    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, flush=True)
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed with {completed.returncode}")
    return completed


def record(name: str, ok: bool, detail) -> None:
    report["steps"].append({"step": name, "ok": ok, "detail": detail})
    if not ok:
        report["failures"].append({"step": name, "detail": detail})


# --- 1. environment ---------------------------------------------------------

section("1. ENVIRONMENT -- recorded, not assumed")

smi = run(["nvidia-smi"])
record("nvidia-smi", smi.returncode == 0, smi.stdout or smi.stderr)

driver = run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
     "--format=csv,noheader"]
)
record("gpu_query", driver.returncode == 0, driver.stdout.strip())

env: dict = {"python": sys.version}
try:
    import torch

    env.update({
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_bf16_supported())
        if torch.cuda.is_available() else False,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })

    # TF32 carries 10 explicit mantissa bits -- exactly fp16's. An fp32 matmul
    # silently executed in TF32 deviates at fp16 scale, roughly 4.7e-2 for this
    # model, against LOGIT_ATOL of 1e-3. Forty-seven times over, on the float32
    # parity tests, and indistinguishable from a paging bug on a new device.
    # conftest.py pins these before any test runs; this records what the host
    # had BEFORE that, which is the number worth reporting.
    precision = {}
    for label, path, attribute in (
        ("torch.backends.cuda.matmul.allow_tf32", "backends.cuda.matmul", "allow_tf32"),
        ("torch.backends.cudnn.allow_tf32", "backends.cudnn", "allow_tf32"),
        ("torch.backends.cuda.matmul.fp32_precision", "backends.cuda.matmul", "fp32_precision"),
        ("torch.backends.cudnn.fp32_precision", "backends.cudnn", "fp32_precision"),
        ("torch.backends.cudnn.conv.fp32_precision", "backends.cudnn.conv", "fp32_precision"),
        ("torch.backends.fp32_precision", "backends", "fp32_precision"),
        ("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
         "backends.cuda.matmul", "allow_fp16_reduced_precision_reduction"),
    ):
        holder = torch
        for part in path.split("."):
            holder = getattr(holder, part, None)
            if holder is None:
                break
        if holder is not None and hasattr(holder, attribute):
            try:
                precision[label] = getattr(holder, attribute)
            except RuntimeError:
                pass
    try:
        precision["torch.get_float32_matmul_precision()"] = (
            torch.get_float32_matmul_precision()
        )
    except (AttributeError, RuntimeError):
        pass
    env["fp32_precision_defaults"] = precision
except Exception as error:  # noqa: BLE001 -- record anything, diagnose later
    env["torch_import_error"] = repr(error)

print(json.dumps(env, indent=2), flush=True)
record("torch_environment", "torch" in env, env)

# The whole reason this project moved off bf16. Recorded so the claim is
# evidence rather than an assumption about what Kaggle serves.
if env.get("cuda_available"):
    print(f"\nbf16 supported on this device: {env['bf16_supported']} "
          f"(compute capability {env['capability']}). "
          f"fp16 is the two-byte format under test either way.", flush=True)


# --- 2 & 3. dependencies ----------------------------------------------------

section("2. TORCH IS NOT TOUCHED / 3. transformers is pinned explicitly")

torch_before = env.get("torch")
transformers_before = None
try:
    import transformers

    transformers_before = transformers.__version__
except Exception as error:  # noqa: BLE001
    transformers_before = f"<not importable: {error!r}>"
print(f"transformers before: {transformers_before}", flush=True)
print(f"torch before       : {torch_before}", flush=True)

WORKING.mkdir(parents=True, exist_ok=True)
constraints = WORKING / "constraints.txt"
if torch_before:
    # A pip constraint naming the torch already installed. pip may not move it.
    constraints.write_text(f"torch=={torch_before}\n")
    print(f"pip constraint: torch=={torch_before}", flush=True)

install = run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     *(["-c", str(constraints)] if torch_before else []),
     f"transformers=={TRANSFORMERS_PIN}"],
    timeout=1800,
)
record("pip_install_transformers", install.returncode == 0,
       (install.stdout + install.stderr)[-4000:])

# Re-import in a FRESH interpreter. This one already imported the old
# transformers, so asking it would report the version that is in memory rather
# than the version that is now on disk.
VERSION_PROBE = (
    "import json, torch, transformers; "
    "print(json.dumps({'torch': torch.__version__, "
    "'transformers': transformers.__version__}))"
)
check = run([sys.executable, "-c", VERSION_PROBE])
after = {}
try:
    after = json.loads(check.stdout.strip().splitlines()[-1])
except Exception as error:  # noqa: BLE001
    record("version_recheck", False, f"could not parse: {check.stdout!r} {error!r}")

if after:
    print(f"\ntorch after        : {after.get('torch')}", flush=True)
    print(f"transformers after : {after.get('transformers')}", flush=True)
    torch_unchanged = after.get("torch") == torch_before
    transformers_pinned = after.get("transformers") == TRANSFORMERS_PIN
    if not torch_unchanged:
        print(f"!!! TORCH WAS REPLACED: {torch_before} -> {after.get('torch')}. "
              "The constraint did not hold. Every number from this session is "
              "suspect.", file=sys.stderr, flush=True)
    record("torch_unchanged", torch_unchanged,
           {"before": torch_before, "after": after.get("torch")})
    record("transformers_pinned", transformers_pinned,
           {"before": transformers_before, "after": after.get("transformers"),
            "wanted": TRANSFORMERS_PIN})


# --- the repo ---------------------------------------------------------------

section("REPO -- copied out of the read-only input mount")

candidates = sorted(INPUT_ROOT.glob("*")) if INPUT_ROOT.exists() else []
print(f"{INPUT_ROOT} contains: {[str(path) for path in candidates]}", flush=True)

# Find the repo by the file that identifies it, not by a path shape. The mount
# layout is NOT the documented /kaggle/input/<slug>/ -- the preflight found it at
# /kaggle/input/datasets/<username>/<slug>/, three levels down. An earlier
# version of this checked one and two levels and would have reported the repo
# missing on a host where it was present the whole time.
source = None
if INPUT_ROOT.exists():
    for marker in INPUT_ROOT.rglob("scheduler.py"):
        if marker.parent.name == "engine":
            source = marker.parent.parent
            break

if source is None:
    record("repo_located", False,
           f"no engine/scheduler.py under {INPUT_ROOT}; saw {[str(p) for p in candidates]}")
    print("FATAL: the repo dataset is not attached. Nothing further can run.",
          file=sys.stderr, flush=True)
else:
    record("repo_located", True, str(source))
    if REPO_DEST.exists():
        shutil.rmtree(REPO_DEST)
    shutil.copytree(source, REPO_DEST)
    print(f"copied {source} -> {REPO_DEST}", flush=True)
    run(["ls", "-la", str(REPO_DEST)])


# --- 4. the tests -----------------------------------------------------------

def pytest_run(label: str, args: list[str], timeout: int):
    section(f"4. pytest {' '.join(args)}")
    if source is None:
        record(label, False, "skipped: repo not attached")
        return
    started = time.perf_counter()
    completed = run(
        [sys.executable, "-m", "pytest", *args],
        cwd=str(REPO_DEST), timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    tail = (completed.stdout + "\n" + completed.stderr)[-20000:]
    # pytest exit 5 is "no tests collected", which is neither a pass nor a test
    # failure. It is still wrong here -- a selection that matches nothing means
    # the marker or the filter broke -- so it is recorded as a failure with its
    # own name rather than being reported as a crash or, worse, as green.
    collected_nothing = completed.returncode == 5
    record(label, completed.returncode == 0, {
        "returncode": completed.returncode,
        "collected_nothing": collected_nothing,
        "elapsed_seconds": round(elapsed, 1),
        "output_tail": tail,
    })
    verdict = "no tests collected" if collected_nothing else f"exit {completed.returncode}"
    print(f"\n{label}: {verdict} in {elapsed:.1f}s", flush=True)


pytest_run("pytest_fast", FAST_ARGS, timeout=3600)
pytest_run("pytest_slow", SLOW_ARGS, timeout=7200)


# --- report -----------------------------------------------------------------

section("SESSION 0 SUMMARY")

report["environment"] = env
report["transformers_before"] = transformers_before
report["torch_before"] = torch_before
report["versions_after"] = after

for step in report["steps"]:
    print(f"  {'ok  ' if step['ok'] else 'FAIL'}  {step['step']}", flush=True)

if report["failures"]:
    print(f"\n{len(report['failures'])} FAILURE(S). Verbatim detail is in the log "
          f"above and in {REPORT.name}.", flush=True)
    for failure in report["failures"]:
        print(f"\n--- {failure['step']} ---", flush=True)
        print(failure["detail"], flush=True)
else:
    print("\nno failures", flush=True)

REPORT.write_text(json.dumps(report, indent=2, default=str))
print(f"\nwrote {REPORT}", flush=True)
