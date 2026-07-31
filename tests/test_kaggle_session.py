"""The Kaggle session scripts, dry-run locally before they cost GPU quota.

Quota is 30 GPU-hours a week and a kernel script that dies in its first minute
spends a whole session finding out. Everything here runs on this machine, on
CPU, against a fake /kaggle mount, so the failure modes that are cheap to find
(a typo, a bad path, a wrong flag, a report that never gets written) are found
here and not there.

What this CANNOT check is the thing Session 0 exists for: device placement,
dtype promotion, contiguity and index arithmetic on a real CUDA device. Those
need the GPU. This only guarantees the script gets far enough to test them.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = REPO_ROOT / "kaggle"
SESSION0 = KAGGLE_DIR / "session0" / "session0.py"

# The subset the dry run points pytest at: fast, and needs no model download.
DRY_RUN_TESTS = ("test_alloc.py",)


def test_session0_script_exists_and_parses():
    import ast

    assert SESSION0.exists(), f"{SESSION0} is missing"
    ast.parse(SESSION0.read_text())


def test_kernel_metadata_requests_gpu_and_internet():
    """Both flags are required and neither is the default."""
    metadata = json.loads((KAGGLE_DIR / "session0" / "kernel-metadata.json").read_text())
    print(f"\nkernel metadata: {json.dumps(metadata, indent=2)}")

    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True, (
        "internet is needed to pip install transformers==5.12.1 and to pull the "
        "model from the hub; without it Session 0 cannot run at all"
    )
    assert metadata["is_private"] is True
    assert metadata["kernel_type"] == "script"
    assert metadata["code_file"] == "session0.py"
    # The repo arrives through the input mount, not over the network.
    assert metadata["dataset_sources"], "no dataset attached, so no repo on the host"


def test_metadata_templates_carry_no_real_username():
    """The account is not known yet; kaggle/push.py substitutes at stage time."""
    for name in ("session0/kernel-metadata.json", "dataset/dataset-metadata.json"):
        text = (KAGGLE_DIR / name).read_text()
        assert "KAGGLE_USERNAME" in text, f"{name} lost its placeholder"


def test_push_stages_a_payload_without_credentials(tmp_path):
    """Staging must work while the account is still blocked."""
    result = subprocess.run(
        [sys.executable, str(KAGGLE_DIR / "push.py"), "session0", "--stage-only"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    print(f"\n{result.stdout}{result.stderr}")
    assert result.returncode == 0, result.stderr

    build = KAGGLE_DIR / "_build"
    payload = build / "dataset" / "repo"
    assert (payload / "engine" / "scheduler.py").exists()
    assert (payload / "tests" / "test_parity_fp16.py").exists()
    assert (payload / "conftest.py").exists()
    assert (payload / "pytest.ini").exists()

    # And it must NOT carry the things that would break a CUDA host or bloat it.
    assert not list(payload.rglob("__pycache__"))
    assert not list(payload.rglob("*.pyc"))
    assert not (payload / ".venv").exists(), (
        "the macOS arm64 venv would be actively harmful on a CUDA host"
    )
    assert not (payload / ".git").exists()


def test_push_reports_the_credential_blocker_precisely():
    """A blocked run must say what is missing and who can fix it."""
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        pytest.skip("credentials are present on this host")

    result = subprocess.run(
        [sys.executable, str(KAGGLE_DIR / "push.py"), "session0"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    print(f"\nexit {result.returncode}\n{result.stderr}")
    assert result.returncode == 3, "a missing credential is not a crash, it is a blocker"
    assert "BLOCKED" in result.stderr
    assert "kaggle.json" in result.stderr


@pytest.mark.slow
def test_session0_runs_end_to_end_against_a_fake_mount(tmp_path):
    """The whole kernel script, on CPU, against a fake /kaggle layout.

    Proves the script reaches its own pytest invocations and writes its report.
    It will record failures -- there is no nvidia-smi here and no CUDA -- and
    that is the point: a step that cannot run must say so and let the rest
    continue, rather than taking the session down with it.
    """
    working = tmp_path / "working"
    mount = tmp_path / "input" / "paged-attention-repo"
    (mount / "repo").mkdir(parents=True)
    working.mkdir(parents=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for name in ("engine", "bench"):
        shutil.copytree(REPO_ROOT / name, mount / "repo" / name, ignore=ignore)
    (mount / "repo" / "tests").mkdir()
    for name in DRY_RUN_TESTS:
        shutil.copy2(REPO_ROOT / "tests" / name, mount / "repo" / "tests" / name)
    for name in ("conftest.py", "pytest.ini"):
        shutil.copy2(REPO_ROOT / name, mount / "repo" / name)

    # One slow-marked test, so `-m slow` selects something. Without it pytest
    # exits 5 ("no tests collected") and the dry run would be checking the
    # wrong thing -- the point here is that the marker selection works, not that
    # this repo's real slow tests pass on CPU.
    (mount / "repo" / "tests" / "test_dryrun_marker.py").write_text(
        "import pytest\n\n\n"
        "@pytest.mark.slow\n"
        "def test_the_slow_marker_selects_this():\n"
        "    assert True\n"
    )

    environment = dict(os.environ)
    environment.update({
        "SESSION0_WORKING": str(working),
        "SESSION0_INPUT": str(tmp_path / "input"),
        # A fast subset. The real session runs the whole matrix.
        "SESSION0_FAST_ARGS": json.dumps(["--fast", "-q", "tests"]),
        "SESSION0_SLOW_ARGS": json.dumps(["-m", "slow", "-q", "tests"]),
    })

    result = subprocess.run(
        [sys.executable, str(SESSION0)], cwd=REPO_ROOT, capture_output=True,
        text=True, timeout=2400, env=environment,
    )
    transcript = result.stdout + result.stderr
    print(f"\n--- session0 dry run, exit {result.returncode} ---")
    print(transcript[-3000:])

    report_path = working / "session0_report.json"
    assert report_path.exists(), (
        "the script did not write its report, so a real session would produce "
        f"nothing to read back. transcript tail:\n{transcript[-3000:]}"
    )
    report = json.loads(report_path.read_text())
    steps = {step["step"]: step["ok"] for step in report["steps"]}
    print(f"\nsteps: {json.dumps(steps, indent=2)}")

    # The parts that do not need a GPU must all have worked.
    assert steps.get("repo_located") is True, "the script could not find the repo"
    assert steps.get("pytest_fast") is True, "the fast subset failed on CPU"
    assert steps.get("pytest_slow") is True
    assert steps.get("transformers_pinned") is True, (
        "the pin did not hold even locally, where it is already satisfied"
    )
    assert steps.get("torch_unchanged") is True, "the constraint let torch move"

    # And the GPU-only parts must have been recorded as failures, not crashed.
    assert "nvidia-smi" in steps
    assert report["environment"]["cuda_available"] is False
