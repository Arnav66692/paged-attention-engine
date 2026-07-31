"""Drive a Kaggle session end to end through the API. No browser.

    python kaggle/push.py session0

Does the whole cycle: stage the repo as a private dataset, push it, push the
kernel, poll until it stops, pull the output and the log back. Every Kaggle call
goes through the CLI, so the only credential involved is ~/.kaggle/kaggle.json.

WHY A DATASET AND NOT A GIT CLONE. Two reasons, and the second is the one that
matters. This repo has no git remote, so a clone inside the notebook would need a
GitHub repo created first -- another thing that needs a human at a keyboard,
which is exactly what this file exists to avoid. And a dataset keeps working if
internet-enabled kernels stay blocked behind phone verification: the repo arrives
through the input mount rather than over the network. A clone would fail there
and take the session with it.

The cost is that the dataset must be re-pushed when the repo changes, which is
one command and is what `stage` below does.

USERNAME SUBSTITUTION. The metadata files in this directory carry the literal
KAGGLE_USERNAME, because the account is not known until credentials exist. This
script reads the username out of the credentials and writes real metadata into a
scratch copy. The templates in git stay generic.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BUILD = HERE / "_build"
PLACEHOLDER = "KAGGLE_USERNAME"

# What the engine needs to run its tests, and nothing else. No .git (large, and
# the kernel has no use for history), no .venv (macOS arm64 wheels would be
# actively harmful on a CUDA host), no caches.
REPO_INCLUDE = ("engine", "bench", "tests", "conftest.py", "pytest.ini",
                "requirements.txt")
REPO_EXCLUDE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", ".venv", ".git", "_build", "results",
    "plots",
)

POLL_SECONDS = 30
POLL_TIMEOUT_SECONDS = 12 * 3600  # Kaggle's own kernel ceiling


class Blocked(RuntimeError):
    """Something only the account owner can clear."""


def kaggle(*args: str, check: bool = True, capture: bool = True):
    command = [sys.executable, "-m", "kaggle", *args]
    print(f"$ kaggle {' '.join(args)}", flush=True)
    completed = subprocess.run(
        command, capture_output=capture, text=True, cwd=str(REPO_ROOT)
    )
    if capture:
        if completed.stdout:
            print(completed.stdout, flush=True)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, flush=True)
    if check and completed.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed ({completed.returncode})")
    return completed


def credentials() -> dict:
    """Read ~/.kaggle/kaggle.json, or say precisely what is missing."""
    path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not path.exists():
        raise Blocked(
            f"no Kaggle credentials at {path}.\n"
            "This needs the account owner: create the account, then\n"
            "  Kaggle -> Settings -> API -> Create New Token\n"
            "which downloads kaggle.json. Place it at that path and\n"
            "  chmod 600 " + str(path) + "\n"
            "Nothing in this script can do that step."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise Blocked(f"{path} is not valid JSON: {error}") from error
    if not data.get("username") or not data.get("key"):
        raise Blocked(f"{path} has no username/key pair")
    return data


def stage(username: str, session: str) -> tuple[Path, Path]:
    """Build a dataset directory and a kernel directory with real metadata."""
    if BUILD.exists():
        shutil.rmtree(BUILD)
    dataset_dir = BUILD / "dataset"
    kernel_dir = BUILD / session
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)

    # The repo snapshot the kernel will import.
    payload = dataset_dir / "repo"
    payload.mkdir()
    for name in REPO_INCLUDE:
        source = REPO_ROOT / name
        if not source.exists():
            raise RuntimeError(f"{source} is missing; the dataset would be incomplete")
        if source.is_dir():
            shutil.copytree(source, payload / name, ignore=REPO_EXCLUDE)
        else:
            shutil.copy2(source, payload / name)

    for template, destination in (
        (HERE / "dataset" / "dataset-metadata.json", dataset_dir / "dataset-metadata.json"),
        (HERE / session / "kernel-metadata.json", kernel_dir / "kernel-metadata.json"),
    ):
        destination.write_text(template.read_text().replace(PLACEHOLDER, username))

    shutil.copy2(HERE / session / f"{session}.py", kernel_dir / f"{session}.py")

    files = sum(1 for _ in payload.rglob("*") if _.is_file())
    size = sum(path.stat().st_size for path in payload.rglob("*") if path.is_file())
    print(f"staged {files} repo files ({size / 1024:.0f} KB) into {dataset_dir}")
    return dataset_dir, kernel_dir


def push_dataset(dataset_dir: Path, username: str) -> None:
    """Create the dataset, or add a version if it already exists."""
    existing = kaggle("datasets", "list", "--mine", "-s", "paged-attention-repo",
                      check=False)
    slug = f"{username}/paged-attention-repo"
    if slug in (existing.stdout or ""):
        kaggle("datasets", "version", "-p", str(dataset_dir),
               "-m", "repo sync", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(dataset_dir), "--dir-mode", "zip")
    # Kaggle needs a moment before a new version is attachable.
    time.sleep(10)


def push_kernel(kernel_dir: Path) -> None:
    kaggle("kernels", "push", "-p", str(kernel_dir))


def poll(slug: str) -> str:
    """Block until the kernel stops. Returns its final status."""
    started = time.perf_counter()
    while True:
        status = kaggle("kernels", "status", slug, check=False)
        text = (status.stdout or "") + (status.stderr or "")
        lowered = text.lower()
        if "complete" in lowered or "error" in lowered or "cancel" in lowered:
            return text.strip()
        elapsed = time.perf_counter() - started
        if elapsed > POLL_TIMEOUT_SECONDS:
            raise RuntimeError(f"kernel still running after {elapsed / 3600:.1f}h")
        print(f"  ... {elapsed / 60:.1f} min elapsed", flush=True)
        time.sleep(POLL_SECONDS)


def fetch(slug: str, session: str) -> Path:
    out = REPO_ROOT / "kaggle" / "_output" / session
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", slug, "-p", str(out), check=False)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a Kaggle session via the API.")
    parser.add_argument("session", default="session0", nargs="?")
    parser.add_argument("--stage-only", action="store_true",
                        help="build the payload and stop; needs no credentials")
    parser.add_argument("--no-wait", action="store_true", help="push and return")
    args = parser.parse_args(argv)

    if args.stage_only:
        # Deliberately before the credential check, so the payload can be built
        # and inspected while the account is still blocked.
        stage(PLACEHOLDER, args.session)
        print("staged with the placeholder username; credentials not required")
        return 0

    try:
        creds = credentials()
    except Blocked as blocker:
        print(f"\nBLOCKED:\n{blocker}", file=sys.stderr)
        return 3

    username = creds["username"]
    slug = f"{username}/paged-attention-{args.session}"
    dataset_dir, kernel_dir = stage(username, args.session)

    push_dataset(dataset_dir, username)
    push_kernel(kernel_dir)
    if args.no_wait:
        print(f"pushed {slug}; not waiting")
        return 0

    final = poll(slug)
    print(f"\nfinal status: {final}")
    destination = fetch(slug, args.session)
    print(f"output in {destination}")
    return 0 if "error" not in final.lower() else 1


if __name__ == "__main__":
    raise SystemExit(main())
