# Running on Kaggle, through the API

No browser. Every step below is a CLI call against the Kaggle API.

## What is blocked, and by whom

Two things need the account owner at a keyboard. Nothing in this directory can
do either, and both are checked for rather than assumed.

1. **The account and its API token.** `kaggle/push.py` exits `3` with a precise
   message if `~/.kaggle/kaggle.json` is absent. Create the account, then
   *Settings → API → Create New Token*, place the file, `chmod 600` it.
2. **Phone verification.** This gates internet-enabled kernels. Session 0 needs
   internet for two things — `pip install transformers==5.12.1` and pulling the
   model from the hub — so **until phone verification is cleared, Session 0
   cannot run at all**. `enable_internet` is asserted `true` by
   `tests/test_kaggle_session.py`; a kernel that quietly ran without it would
   fail in a way that looked like a port bug.

Everything else in this directory works today, with no credentials:

```bash
python kaggle/push.py session0 --stage-only
```

## Transport: a private Dataset, not a git clone

The repo reaches the host as a private Kaggle Dataset attached to the kernel,
not by cloning inside the notebook. Two reasons, and the second is the one that
decided it:

- This repo has **no git remote**. A clone would need a GitHub repo created
  first, which is one more thing needing a human — exactly what the API route
  exists to avoid.
- A dataset **still works if internet stays blocked**. The repo arrives through
  the read-only input mount rather than over the network, so if phone
  verification is pending, the failure is confined to the pip install and the
  model pull instead of taking the repo with it.

The cost is that the dataset must be re-pushed whenever the repo changes. That
is one command, and `push.py` does it automatically on every run.

## The cycle

```bash
python kaggle/push.py session0
```

Stages the repo, pushes the dataset (create, or a new version if it exists),
pushes the kernel, polls `kernels status` every 30s until it stops, then pulls
the output into `kaggle/_output/session0/`. Both `_build/` and `_output/` are
gitignored — they are rebuilt, never committed.

## Session 0 is verification only

Thirty minutes, no sweep, no timing, no throughput number. It records the
environment, installs `transformers==5.12.1` under a pip constraint naming the
torch already present, **verifies torch did not move**, then runs `pytest --fast`
and `pytest -m slow` and reports every failure verbatim.

The expected failure classes are device placement, dtype promotion, contiguity
and index arithmetic. `PagedBackend.attend` and `FusedPagedAttention.attend`
touch all four — a `.to(device)` that was a no-op on CPU, an index tensor built
without a device, a `.view()` on something CUDA made non-contiguous. Cheap to
find here, expensive to find mid-sweep.

`FP16_LOGIT_ATOL = 6.25e-2` is an untested prediction; no CUDA device existed
when it was derived. **If it fails, the measured deviation is the result and the
constant does not move.**

## Dry-run first

`tests/test_kaggle_session.py` runs `session0.py` end to end on this machine
against a fake `/kaggle` mount, on CPU. It cannot check anything CUDA-specific —
that is the entire point of Session 0 — but it does guarantee the script gets far
enough to test them.

It has already earned its place twice: it caught `nvidia-smi` raising
`FileNotFoundError` and killing the script on its first line, and it caught
`pytest -m slow` exiting `5` on an empty selection being reported as a crash.
Both would have burned a session.

```bash
pytest tests/test_kaggle_session.py -m ""
```

## Two GPU sessions at a time, and the slice boundary respects it

`kaggle kernels push` refuses a third concurrent GPU kernel:

```
Kernel push error: Maximum batch GPU session count of 2 reached.
```

So the four `session6?` slices run two at a time. That is not a problem for the
comparison, because **each slice already contains both arms** — Arm A and Arm B
for a cell run back to back in the same session on the same physical T4, and the
only thing split across hosts is different cells, which are never compared to
each other.

Pushing a new dataset version while kernels are running is safe: a kernel copies
the repo out of the mounted dataset at startup and does not re-read it.

Kaggle also renames an uploaded script to `script.py` before running it, so
nothing may derive its identity from `__file__` — the session6 slices each
declare their cell as a literal, checked against the directory name by
`tests/test_bench.py`.
