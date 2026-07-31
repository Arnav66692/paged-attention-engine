"""Repo root on sys.path, and the `--fast` development selection.

Two selections exist and they are not interchangeable.

  default + `-m slow`   THE GATE. The full parity matrix, every prompt length on
                        every device this host has. This is what runs before a
                        commit and what any claim about the engine rests on.
  `--fast`              A development inner loop. One prompt length on one
                        device, so an edit can be checked in seconds instead of
                        minutes. It is a filter over the same tests, never a
                        different or weakened set of them, and it proves strictly
                        less than the gate does.

`--fast` narrows by real parameter values read off each item's callspec, not by
matching test-id strings, so a test that grows a new parameter is unaffected and
a test carrying neither parameter is always kept.

This file also does two things before torch is imported anywhere, because both
are only effective if they happen first. See CUDA_VISIBLE_DEVICES and the fp32
precision pinning below.
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# ONE GPU. Set before torch is imported, because torch caches the device list at
# CUDA init and a later change does nothing. The whole design is single-GPU: the
# KV ceiling of 1,644,167,168 bytes is sized against ONE card, so a run that
# quietly spread over two would be measuring a machine that does not exist as far
# as any of the locked constants are concerned. `setdefault`, so an explicit
# choice by the caller still wins.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402 -- must follow the environment pin above

from tests.devices import model_devices  # noqa: E402


def _accelerator() -> str:
    """The device this host is actually for.

    Was hardcoded to "cpu" on the grounds that every host has one. That was
    wrong in the way that mattered: on the Kaggle T4 host it made `--fast` run
    the entire model tier on CPU, so a session whose whole purpose was to find
    CUDA failures could not have found one. The fast selection is one device --
    it should be the device the host exists to exercise.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


FAST_DEVICE = _accelerator()

# 33 = one token past a block boundary, with BLOCK_SIZE 16. Of the three fixture
# lengths it is the one that exercises a partially filled final block and a
# multi-block table at the same time, so it is the single length most likely to
# catch a paging mistake. 13 sits inside one block and 32 fills two exactly; both
# are easier cases.
FAST_PROMPT_LEN = 33

# The item parameters the fast selection filters on. `runtime` is the device
# fixture in every model-tier module, and its param value *is* the device string.
_DEVICE_PARAMS = ("runtime",)
_PROMPT_PARAMS = ("prompt_len",)


# --- the TF32 trap ----------------------------------------------------------
#
# TF32 carries 10 explicit mantissa bits. That is EXACTLY fp16's mantissa. So an
# fp32 matmul silently executed in TF32 does not deviate a little -- it deviates
# at fp16 scale, which for this model is around 4.7e-2 against a LOGIT_ATOL of
# 1e-3. Forty-seven times over, on the float32 parity tests, and it looks
# identical to a paging bug on a new device. It is not one.
#
# So the settings are READ first and RECORDED, then pinned. Defaults are not
# assumed: torch has moved this API twice, the two legacy booleans disagree with
# each other out of the box (matmul False, cudnn True on 2.12), and the newer
# fp32_precision strings do not exist at all on older builds. Whatever this host
# actually had is reported in the header and travels into run metadata.
#
# `allow_fp16_reduced_precision_reduction` is read and recorded here but NOT
# pinned. It governs fp16 split-k reduction, which is the term the fp16 tolerance
# derivation assumes is negligible -- that is Session 1's experiment to run
# deliberately, not something to silently change underneath it.
_PRECISION_PROBES = (
    ("torch.backends.cuda.matmul.allow_tf32", "backends.cuda.matmul", "allow_tf32"),
    ("torch.backends.cudnn.allow_tf32", "backends.cudnn", "allow_tf32"),
    ("torch.backends.cuda.matmul.fp32_precision", "backends.cuda.matmul", "fp32_precision"),
    ("torch.backends.cudnn.fp32_precision", "backends.cudnn", "fp32_precision"),
    ("torch.backends.cudnn.conv.fp32_precision", "backends.cudnn.conv", "fp32_precision"),
    ("torch.backends.fp32_precision", "backends", "fp32_precision"),
    (
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
        "backends.cuda.matmul",
        "allow_fp16_reduced_precision_reduction",
    ),
)

# What we pin fp32 to, and why: "highest"/False/"ieee" all mean the same thing --
# fp32 matmuls execute in fp32.
_PRECISION_PINS = (
    ("backends.cuda.matmul", "allow_tf32", False),
    ("backends.cudnn", "allow_tf32", False),
    ("backends.cuda.matmul", "fp32_precision", "ieee"),
    ("backends.cudnn", "fp32_precision", "ieee"),
    ("backends.cudnn.conv", "fp32_precision", "ieee"),
)

FP32_PRECISION_DEFAULTS: dict[str, object] = {}
FP32_PRECISION_PINNED: dict[str, object] = {}


def _resolve(path: str):
    obj = torch
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _read_precision() -> dict[str, object]:
    """Every precision knob that exists on this torch, by its real name."""
    observed: dict[str, object] = {}
    for label, path, attribute in _PRECISION_PROBES:
        holder = _resolve(path)
        if holder is None:
            continue
        try:
            observed[label] = getattr(holder, attribute)
        except (AttributeError, RuntimeError):
            continue
    try:
        observed["torch.get_float32_matmul_precision()"] = (
            torch.get_float32_matmul_precision()
        )
    except (AttributeError, RuntimeError):
        pass
    return observed


def pin_fp32_precision() -> tuple[dict[str, object], dict[str, object]]:
    """Record the defaults, then force fp32 to mean fp32. Returns (before, after)."""
    before = _read_precision()

    try:
        torch.set_float32_matmul_precision("highest")
    except (AttributeError, RuntimeError):
        pass
    for path, attribute, value in _PRECISION_PINS:
        holder = _resolve(path)
        if holder is None or not hasattr(holder, attribute):
            continue
        try:
            setattr(holder, attribute, value)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Some builds expose the name but reject the value. Recorded by the
            # after-read below rather than guessed at.
            continue

    return before, _read_precision()


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help=(
            f"Development selection: prompt_len={FAST_PROMPT_LEN} on "
            f"{FAST_DEVICE} only. NOT the gate -- run the full suite and "
            f"-m slow before committing."
        ),
    )


def pytest_configure(config):
    """Pin fp32 precision before a single test runs, and remember the defaults."""
    before, after = pin_fp32_precision()
    FP32_PRECISION_DEFAULTS.clear()
    FP32_PRECISION_DEFAULTS.update(before)
    FP32_PRECISION_PINNED.clear()
    FP32_PRECISION_PINNED.update(after)


def pytest_report_header(config):
    lines = [
        f"devices: {', '.join(model_devices())}"
        f" | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        f" | visible CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
    ]
    changed = [
        f"{name}: {FP32_PRECISION_DEFAULTS[name]!r} -> {FP32_PRECISION_PINNED.get(name)!r}"
        for name in FP32_PRECISION_DEFAULTS
        if FP32_PRECISION_DEFAULTS[name] != FP32_PRECISION_PINNED.get(name)
    ]
    lines.append(
        "fp32 precision pinned: " + ("; ".join(changed) if changed else "(already ieee)")
    )
    lines.append(f"fp32 precision defaults were: {FP32_PRECISION_DEFAULTS}")
    if config.getoption("--fast"):
        lines += [
            f"FAST SELECTION: prompt_len={FAST_PROMPT_LEN}, device={FAST_DEVICE}",
            "development inner loop, NOT the gate -- the full matrix and -m slow are",
        ]
    return lines


def _params(item) -> dict:
    callspec = getattr(item, "callspec", None)
    return dict(callspec.params) if callspec is not None else {}


def _wanted(item) -> bool:
    params = _params(item)
    for name in _DEVICE_PARAMS:
        if name in params and params[name] != FAST_DEVICE:
            return False
    for name in _PROMPT_PARAMS:
        if name in params and params[name] != FAST_PROMPT_LEN:
            return False
    return True


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--fast"):
        return
    keep, drop = [], []
    for item in items:
        (keep if _wanted(item) else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
