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
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# cpu, because every host has one and it is the only device whose presence is
# unconditional. Narrowing to mps would make the fast selection mean something
# different on a machine without one, which is the opposite of what a
# development loop wants.
FAST_DEVICE = "cpu"

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


def pytest_report_header(config):
    if config.getoption("--fast"):
        return [
            f"FAST SELECTION: prompt_len={FAST_PROMPT_LEN}, device={FAST_DEVICE}",
            "development inner loop, NOT the gate -- the full matrix and -m slow are",
        ]
    return []


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
