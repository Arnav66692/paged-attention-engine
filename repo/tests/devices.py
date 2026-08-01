"""The devices the model tier runs on. One definition, three consumers.

Was duplicated in test_parity.py, test_sched.py and test_fused.py as a private
`_devices()` that read:

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices

which returns ["cpu"] on a CUDA host. Every model-tier test on Kaggle therefore
ran on CPU, and Session 0 -- whose entire purpose was to surface device
placement, dtype promotion, contiguity and index arithmetic failures -- could not
have surfaced one, because the engine never touched the GPU. Triplicated code
that is wrong in three places at once is why this now lives in one.

cpu stays FIRST and stays the reference. Every byte-identity claim in this suite
is asserted device-by-device, so cpu remains the path whose numbers are known,
and an accelerator is an additional column rather than a replacement.
"""

import torch


def model_devices() -> list[str]:
    """cpu, plus whatever accelerator this host has. cpu is always the reference."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices
