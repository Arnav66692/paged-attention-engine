"""SESSION 8: re-verify the FULL parity matrix on CUDA after the default moved.

`engine/attention.py::GQA_PATH` shipped as `sdpa` through the entire benchmark
and is now `repeat_kv`. That is a change to the kernel path -- on sm_75 it moves
every attention call this engine issues from MATH to EFFICIENT_ATTENTION -- so
every equality and tolerance claim in the repo has to be re-checked against the
device, not assumed to carry over.

WHAT IS AT RISK AND WHY THIS IS NOT A FORMALITY. `tests/test_attention_gqa.py`
holds the two paths to `torch.equal` given identical inputs, but that is the
EXPANSION being exact, not the kernel being identical. The kernels differ in
accumulation order, and this repo has already measured that difference moving
fp16 parity by 6x. So:

  - the fp32 matrix (LOGIT_ATOL 1e-3) should be unaffected but must be shown so;
  - the fp16 matrix (FP16_LOGIT_ATOL 6.25e-2) is the one that can move, and it
    is the one that moved 6x last time the kernel changed;
  - the fused-vs-unfused equality (FUSED_LOGIT_ATOL 5e-3) now compares two paths
    that BOTH moved, so it should hold, but "should" is not a measurement.

FP16_LOGIT_ATOL IS NOT TO BE TIGHTENED OR WIDENED. If a bound is exceeded that
is the result and it gets reported as one. The whole point of running this is
that the answer is allowed to be "the switch broke parity".

NO THROUGHPUT NUMBER IS PRODUCED HERE. Arm B already measured this exact
configuration on both streams; nothing is re-swept.
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

WORKING = Path(os.environ.get("SESSION8_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION8_INPUT", "/kaggle/input"))
REPORT = WORKING / "session8_report.json"

report: dict = {"disclaimer": "parity re-verification only; NO throughput number"}
WORKING.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402

(WORKING / "c.txt").write_text(f"torch=={torch.__version__}\n")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-c",
                str(WORKING / "c.txt"), "transformers==5.12.1", "pytest"], timeout=1800)

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


report["device"] = torch.cuda.get_device_name(0)
report["capability"] = list(torch.cuda.get_device_capability(0))
report["torch"] = torch.__version__

# THE DEFAULT ITSELF, read from the module rather than assumed from the diff.
# A switch that silently did not take would make every test below pass for the
# wrong reason, which is the one outcome that looks like success.
section("0. what default is actually in force?")
probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, '.'); "
     "from engine import attention; "
     "print('GQA_PATH=' + attention.GQA_PATH)"],
    capture_output=True, text=True, cwd=dest,
    env={k: v for k, v in os.environ.items() if k != "PAGED_ATTENTION_GQA_PATH"},
)
line = next((ln for ln in (probe.stdout or "").splitlines()
             if ln.startswith("GQA_PATH=")), None)
in_force = line.split("=", 1)[1] if line else None
report["gqa_path_in_force"] = in_force
print(f"GQA_PATH in force with no environment override: {in_force!r}")
if in_force != "repeat_kv":
    report["FATAL"] = (
        f"expected the shipped default to be 'repeat_kv', got {in_force!r}. "
        "Every result below would be from the OLD path."
    )
    save()
    raise SystemExit(report["FATAL"])
save()


def pytest_run(label: str, args: list[str], timeout: int):
    section(f"pytest {' '.join(args)}")
    started = time.perf_counter()
    try:
        done = subprocess.run([sys.executable, "-m", "pytest", *args],
                              capture_output=True, text=True, timeout=timeout, cwd=dest)
        out = (done.stdout or "") + (done.stderr or "")
        code = done.returncode
    except subprocess.TimeoutExpired as expired:
        out = f"TIMEOUT after {timeout}s\n{expired.stdout or ''}"
        code = -9
    elapsed = time.perf_counter() - started

    tail = [ln for ln in out.splitlines() if ln.strip()][-40:]
    summary = next((ln for ln in reversed(out.splitlines())
                    if " passed" in ln or " failed" in ln or "error" in ln.lower()), "")
    failures = [ln for ln in out.splitlines()
                if ln.startswith("FAILED") or ln.startswith("ERROR")]
    # Exit 5 is "no tests collected" -- not a pass, and it fooled an earlier session.
    report[label] = {
        "returncode": code,
        "passed": code == 0,
        "no_tests_collected": code == 5,
        "seconds": round(elapsed, 1),
        "summary": summary.strip(),
        "failures": failures,
        "tail": tail,
    }
    print("\n".join(tail))
    print(f"\n[{label}] rc={code} in {elapsed / 60:.1f} min :: {summary.strip()}",
          flush=True)
    save()
    return code == 0


# The whole gate, both halves. --fast is a filter and is NOT the gate.
default_ok = pytest_run("pytest_default", ["-q", "--no-header"], timeout=10800)
slow_ok = pytest_run("pytest_slow", ["-q", "--no-header", "-m", "slow"], timeout=10800)

# The three tolerance owners, run alone with output shown, so the actual
# deviations are in the record rather than only a pass/fail.
for label, path in (
    ("parity_fp32", "tests/test_parity.py"),
    ("parity_fp16", "tests/test_parity_fp16.py"),
    ("fused", "tests/test_fused.py"),
    ("gqa_equality", "tests/test_attention_gqa.py"),
):
    pytest_run(f"detail_{label}", ["-q", "--no-header", "-s", path], timeout=7200)

report["verdict"] = {
    "gqa_path_in_force": in_force,
    "default_suite_passed": default_ok,
    "slow_suite_passed": slow_ok,
    "parity_holds_under_repeat_kv": bool(default_ok and slow_ok),
}
save()
section("VERDICT")
print(json.dumps(report["verdict"], indent=2))
print(f"\nwrote {REPORT}", flush=True)
