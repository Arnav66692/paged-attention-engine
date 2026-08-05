"""SESSION 9: run the STANDALONE upstream reproduction, end to end.

`docs/upstream/repro_enable_gqa_padding_cliff.py` is what would be attached to a
comment on huggingface/transformers#45776. It has not been run since two figures
were pulled from the draft, so this runs it exactly as a stranger would -- from
its own file, with no dependency on this repo -- and captures its real output.

Nothing here is a Gate 1 number. The repo is frozen; this is verification of a
document, not new work.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

WORKING = Path(os.environ.get("SESSION9_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION9_INPUT", "/kaggle/input"))
REPORT = WORKING / "session9_report.json"

WORKING.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402

(WORKING / "c.txt").write_text(f"torch=={torch.__version__}\n")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-c",
                str(WORKING / "c.txt"), "transformers==5.12.1"], timeout=1800)

source = None
for marker in INPUT_ROOT.rglob("repro_enable_gqa_padding_cliff.py"):
    source = marker
    break
if source is None:
    raise SystemExit("the repro script is not in the dataset")

# Copy it OUT of the repo, to a bare directory, and run it there. If it has an
# accidental dependency on this repo, that is exactly what must fail here.
sandbox = WORKING / "standalone"
if sandbox.exists():
    shutil.rmtree(sandbox)
sandbox.mkdir()
script = sandbox / source.name
shutil.copy2(source, script)

print(f"running {script} from a bare directory, cwd={sandbox}", flush=True)
done = subprocess.run([sys.executable, str(script)], capture_output=True,
                      text=True, timeout=3600, cwd=sandbox,
                      env={k: v for k, v in os.environ.items()
                           if not k.startswith("PAGED_ATTENTION")})
out = (done.stdout or "")
err = (done.stderr or "")
clean = "\n".join(l for l in (out + "\n" + err).splitlines()
                  if "it/s" not in l and "Loading weights" not in l and l.strip())
print(clean, flush=True)

REPORT.write_text(json.dumps({
    "returncode": done.returncode,
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "stdout": out,
    "stderr_tail": err[-3000:],
    "clean": clean,
}, indent=2))
print(f"\nrc={done.returncode}; wrote {REPORT}", flush=True)
