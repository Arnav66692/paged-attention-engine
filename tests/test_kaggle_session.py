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
import re
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

# Package infrastructure the fake mount must also carry. conftest.py imports
# tests.devices, so a payload with conftest.py but without these fails to load
# conftest at all -- pytest exit 4, before a single test is collected. The real
# dataset ships the whole tests/ directory so it is fine there; this list keeps
# the dry run honest about what conftest actually depends on.
DRY_RUN_SUPPORT = ("__init__.py", "devices.py")

# The measurement payload ships engine/, bench/, tests/ and the two config files
# -- see REPO_INCLUDE in kaggle/push.py -- and deliberately NOT kaggle/ or .git.
# Tests that need either are about the local development environment, not about
# the engine, so on the measurement host they SKIP. They failed there before, 13
# of them, and 13 red lines that mean "you are not on a dev box" drown the ones
# that mean something.
#
# tests/test_bench.py keeps the same distinction on purpose: the frozen-workload
# CONTENT HASH runs everywhere including the measurement host, because that is
# the check that matters there. Only its .git-dependent sibling skips.
HAVE_KAGGLE_DIR = (KAGGLE_DIR / "session0" / "session0.py").exists()
HAVE_GIT = (REPO_ROOT / ".git").exists()

needs_kaggle_dir = pytest.mark.skipif(
    not HAVE_KAGGLE_DIR, reason="kaggle/ is not shipped in the measurement payload"
)
needs_git = pytest.mark.skipif(
    not HAVE_GIT, reason="no .git in the measurement payload"
)


@needs_kaggle_dir
def test_session0_script_exists_and_parses():
    import ast

    assert SESSION0.exists(), f"{SESSION0} is missing"
    ast.parse(SESSION0.read_text())


@needs_kaggle_dir
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


@needs_kaggle_dir
def test_metadata_templates_carry_no_real_username():
    """The account is not known yet; kaggle/push.py substitutes at stage time."""
    for name in ("session0/kernel-metadata.json", "dataset/dataset-metadata.json"):
        text = (KAGGLE_DIR / name).read_text()
        assert "KAGGLE_USERNAME" in text, f"{name} lost its placeholder"


@needs_kaggle_dir
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


@needs_kaggle_dir
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
@needs_kaggle_dir
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
    for name in DRY_RUN_TESTS + DRY_RUN_SUPPORT:
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


# --- credential hygiene -----------------------------------------------------


CREDENTIAL_PATHS = (
    "kaggle.json",
    "kaggle/kaggle.json",
    ".kaggle/kaggle.json",
    "some/deep/path/kaggle.json",
    "my_credentials.json",
)


@needs_git
def test_git_tracks_no_kaggle_credential():
    """A tracked API token is an account takeover. Checked, not assumed.

    The real token lives at ~/.kaggle/kaggle.json, outside this repo, so this
    cannot fail today -- which is exactly why it is worth asserting. It fails the
    day someone copies one in "just to test something".
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert tracked.returncode == 0, tracked.stderr
    offenders = [
        line for line in tracked.stdout.splitlines()
        if "kaggle.json" in line.lower() or line.startswith(".kaggle/")
    ]
    assert not offenders, f"git is tracking credential file(s): {offenders}"


@pytest.mark.parametrize("path", CREDENTIAL_PATHS)
@needs_git
def test_gitignore_would_catch_a_stray_credential(path):
    """Every plausible place a token could land must already be ignored."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"{path} is NOT gitignored, so a token dropped there could be committed"
    )


def test_the_repo_tree_holds_no_credential_file():
    """Nothing that looks like a token is sitting in the working tree either."""
    found = [
        path.relative_to(REPO_ROOT)
        for path in REPO_ROOT.rglob("kaggle*.json")
        if ".venv" not in path.parts and ".git" not in path.parts
    ]
    assert not found, f"credential-shaped file(s) in the repo tree: {found}"


def _slugify(title: str) -> str:
    """Kaggle's own rule: lowercase, non-alphanumerics collapse to hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


@pytest.mark.parametrize("session", ("preflight", "session0"))
@needs_kaggle_dir
def test_kernel_id_matches_the_slug_kaggle_will_derive(session):
    """The id must equal slugify(title), or the push 409s.

    Kaggle derives a NEW kernel's slug from its TITLE and ignores the id. With
    id `paged-attention-session0` and title "paged attention session0 port
    verification" it created `paged-attention-session0-port-verification`, so
    polling the id queried a kernel that did not exist -- and `kernels status`
    answered with a permission error containing none of the words the poller
    looks for, so it would have spun for twelve hours on a run that had already
    finished. The next push then 409'd, because the id it asked to create was a
    slug the title had already claimed.
    """
    metadata = json.loads(
        (KAGGLE_DIR / session / "kernel-metadata.json").read_text()
    )
    owner, _, slug = metadata["id"].partition("/")
    expected = _slugify(metadata["title"])
    print(f"\n{session}: title={metadata['title']!r} -> slug {expected!r}, id slug {slug!r}")
    assert slug == expected, (
        f"kernel id slug {slug!r} != slugify(title) {expected!r}; Kaggle will "
        "create the title's slug and the id will point at nothing"
    )
    assert owner == "KAGGLE_USERNAME"
