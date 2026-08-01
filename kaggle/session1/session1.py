"""SESSION 1+2: parity matrix, the TF32 and fp16-reduction counterfactuals,
memory headroom on 15 GB, and a harness smoke. NO SWEEP. NO THROUGHPUT RESULT.

Five parts, reported separately. Everything here MEASURES; nothing is asserted
from arithmetic that was not run on this host.
"""

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

WORKING = Path(os.environ.get("SESSION1_WORKING", "/kaggle/working"))
INPUT_ROOT = Path(os.environ.get("SESSION1_INPUT", "/kaggle/input"))
REPO_DEST = WORKING / "repo"
REPORT = WORKING / "session1_report.json"
TRANSFORMERS_PIN = "5.12.1"

# Resolved from the hub during Step 6 and recorded then. Confirmed, not trusted.
HEADLINE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HEADLINE_SHA_EXPECTED = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

report: dict = {"parts": {}}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def run(command, timeout=5400, cwd=None):
    printable = " ".join(command) if isinstance(command, list) else command
    print(f"$ {printable}", flush=True)
    try:
        done = subprocess.run(command, shell=isinstance(command, str),
                              capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"  ({error!r})", flush=True)
        return subprocess.CompletedProcess(command, 127, "", repr(error))
    if done.stdout:
        print(done.stdout, flush=True)
    if done.stderr:
        print(done.stderr, file=sys.stderr, flush=True)
    return done


# --- setup ------------------------------------------------------------------

section("SETUP")
WORKING.mkdir(parents=True, exist_ok=True)
constraints = WORKING / "constraints.txt"
import torch  # noqa: E402

constraints.write_text(f"torch=={torch.__version__}\n")
run([sys.executable, "-m", "pip", "install", "--quiet", "-c", str(constraints),
     f"transformers=={TRANSFORMERS_PIN}"], timeout=1800)

source = None
for marker in INPUT_ROOT.rglob("scheduler.py"):
    if marker.parent.name == "engine":
        source = marker.parent.parent
        break
if source is None:
    print("FATAL: repo not mounted", file=sys.stderr, flush=True)
    REPORT.write_text(json.dumps({"fatal": "repo not mounted"}, indent=2))
    raise SystemExit(1)
if REPO_DEST.exists():
    shutil.rmtree(REPO_DEST)
shutil.copytree(source, REPO_DEST)
sys.path.insert(0, str(REPO_DEST))
os.chdir(REPO_DEST)
print(f"repo at {REPO_DEST}", flush=True)

DEVICE = "cuda"
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"device_count={torch.cuda.device_count()}, "
      f"name={torch.cuda.get_device_name(0)}", flush=True)


# --- PART 1: full parity matrix on CUDA -------------------------------------

section("PART 1: full parity matrix on CUDA, prompt lengths 13 / 32 / 33")

DEVIATION = re.compile(r"MAX ABS DEVIATION\s*:\s*([0-9.e+-]+)")


def pytest_matrix(label, args, timeout=5400):
    done = run([sys.executable, "-m", "pytest", *args, "-q", "-rA", "-s"],
               timeout=timeout, cwd=str(REPO_DEST))
    return {"returncode": done.returncode,
            "output_tail": (done.stdout + "\n" + done.stderr)[-30000:]}


part1 = {}
part1["fp32_cuda"] = pytest_matrix(
    "fp32", ["tests/test_parity.py", "-k", "cuda"])
part1["fp16_cuda"] = pytest_matrix("fp16", ["tests/test_parity_fp16.py"])
report["parts"]["part1_parity_matrix"] = part1


# --- PART 2: the TF32 counterfactual, MEASURED ------------------------------

section("PART 2: TF32 counterfactual -- measured, not asserted")

from transformers import AutoModelForCausalLM  # noqa: E402

from engine.cache import BLOCK_SIZE, BlockAllocator  # noqa: E402
from engine.model import Qwen2Engine  # noqa: E402
from tests.test_parity import (  # noqa: E402
    NUM_GENERATED_TOKENS, PROMPT_LENGTHS, _prompt_ids, hf_greedy_cached,
)

PARITY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PAGED_POOL = 64


def precision_snapshot():
    snap = {}
    for label, path, attribute in (
        ("cuda.matmul.allow_tf32", "backends.cuda.matmul", "allow_tf32"),
        ("cudnn.allow_tf32", "backends.cudnn", "allow_tf32"),
        ("cuda.matmul.fp32_precision", "backends.cuda.matmul", "fp32_precision"),
        ("cudnn.fp32_precision", "backends.cudnn", "fp32_precision"),
        ("cudnn.conv.fp32_precision", "backends.cudnn.conv", "fp32_precision"),
        ("cuda.matmul.allow_fp16_reduced_precision_reduction",
         "backends.cuda.matmul", "allow_fp16_reduced_precision_reduction"),
    ):
        holder = torch
        for part in path.split("."):
            holder = getattr(holder, part, None)
            if holder is None:
                break
        if holder is not None and hasattr(holder, attribute):
            try:
                snap[label] = getattr(holder, attribute)
            except RuntimeError:
                pass
    try:
        snap["get_float32_matmul_precision()"] = torch.get_float32_matmul_precision()
    except (AttributeError, RuntimeError):
        pass
    return snap


def apply_precision(settings):
    for path, attribute, value in settings:
        holder = torch
        for part in path.split("."):
            holder = getattr(holder, part, None)
            if holder is None:
                break
        if holder is not None and hasattr(holder, attribute):
            try:
                setattr(holder, attribute, value)
            except (RuntimeError, TypeError, ValueError):
                pass


ARM_RECORDED_DEFAULTS = [
    ("backends.cuda.matmul", "allow_tf32", False),
    ("backends.cudnn", "allow_tf32", True),
    ("backends.cuda.matmul", "fp32_precision", "none"),
    ("backends.cudnn", "fp32_precision", "none"),
    ("backends.cudnn.conv", "fp32_precision", "tf32"),
]
ARM_PINNED_IEEE = [
    ("backends.cuda.matmul", "allow_tf32", False),
    ("backends.cudnn", "allow_tf32", False),
    ("backends.cuda.matmul", "fp32_precision", "ieee"),
    ("backends.cudnn", "fp32_precision", "ieee"),
    ("backends.cudnn.conv", "fp32_precision", "ieee"),
]
ARM_TF32_FORCED_ON = [
    ("backends.cuda.matmul", "allow_tf32", True),
    ("backends.cudnn", "allow_tf32", True),
    ("backends.cuda.matmul", "fp32_precision", "tf32"),
    ("backends.cudnn", "fp32_precision", "tf32"),
    ("backends.cudnn.conv", "fp32_precision", "tf32"),
]


def fp32_parity_once(model, engine, prompt_len, path):
    """engine vs HuggingFace, both under whatever precision is currently set."""
    prompt_ids = _prompt_ids(prompt_len)
    hf_tokens, hf_logits = hf_greedy_cached(
        model, prompt_ids, NUM_GENERATED_TOKENS, DEVICE
    )
    if path == "contiguous":
        backend = engine.new_contiguous_backend(prompt_len + NUM_GENERATED_TOKENS)
    else:
        cache = engine.new_paged_cache(PAGED_POOL)
        allocator = BlockAllocator(num_blocks=PAGED_POOL, block_size=BLOCK_SIZE)
        backend = engine.new_paged_backend(cache, allocator, seq_id=0)
    result = engine.generate_greedy(list(prompt_ids), NUM_GENERATED_TOKENS, backend=backend)
    deviation = (result.step_logits.float() - hf_logits.float()).abs().max().item()
    return {"max_abs_deviation": deviation,
            "tokens_identical": result.token_ids == hf_tokens}


fp32_model = AutoModelForCausalLM.from_pretrained(PARITY_MODEL, dtype=torch.float32)
fp32_model.eval()
fp32_model.to(DEVICE)
fp32_engine = Qwen2Engine.from_hf_model(fp32_model)

part2 = {"arms": {}}
for arm_name, arm in (("recorded_defaults", ARM_RECORDED_DEFAULTS),
                      ("pinned_ieee", ARM_PINNED_IEEE),
                      ("tf32_forced_on", ARM_TF32_FORCED_ON)):
    apply_precision(arm)
    observed = {"settings": precision_snapshot(), "measurements": {}}
    for prompt_len in PROMPT_LENGTHS:
        for path in ("contiguous", "paged"):
            key = f"{prompt_len}-{path}"
            observed["measurements"][key] = fp32_parity_once(
                fp32_model, fp32_engine, prompt_len, path)
            m = observed["measurements"][key]
            print(f"[{arm_name}] {key}: dev={m['max_abs_deviation']:.6e} "
                  f"tokens_identical={m['tokens_identical']}", flush=True)
    worst = max(v["max_abs_deviation"] for v in observed["measurements"].values())
    observed["worst_deviation"] = worst
    observed["all_tokens_identical"] = all(
        v["tokens_identical"] for v in observed["measurements"].values())
    print(f"[{arm_name}] WORST = {worst:.6e}", flush=True)
    part2["arms"][arm_name] = observed

part2["logit_atol"] = 1e-3
report["parts"]["part2_tf32_counterfactual"] = part2

apply_precision(ARM_PINNED_IEEE)
del fp32_model, fp32_engine
torch.cuda.empty_cache()


# --- PART 3: allow_fp16_reduced_precision_reduction -------------------------

section("PART 3: allow_fp16_reduced_precision_reduction, both ways")

fp16_model = AutoModelForCausalLM.from_pretrained(PARITY_MODEL, dtype=torch.float16)
fp16_model.eval()
fp16_model.to(DEVICE)
fp16_engine = Qwen2Engine.from_hf_model(fp16_model)

part3 = {"arms": {}}
for enabled in (True, False):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = enabled
    arm = {"allow_fp16_reduced_precision_reduction": enabled,
           "settings": precision_snapshot(), "measurements": {}}
    for prompt_len in PROMPT_LENGTHS:
        for path in ("contiguous", "paged"):
            key = f"{prompt_len}-{path}"
            arm["measurements"][key] = fp32_parity_once(
                fp16_model, fp16_engine, prompt_len, path)
            m = arm["measurements"][key]
            print(f"[fp16 reduced={enabled}] {key}: dev={m['max_abs_deviation']:.6e} "
                  f"tokens_identical={m['tokens_identical']}", flush=True)
    arm["worst_deviation"] = max(
        v["max_abs_deviation"] for v in arm["measurements"].values())
    arm["all_tokens_identical"] = all(
        v["tokens_identical"] for v in arm["measurements"].values())
    print(f"[fp16 reduced={enabled}] WORST = {arm['worst_deviation']:.6e}", flush=True)
    part3["arms"][str(enabled)] = arm

part3["fp16_logit_atol"] = 6.25e-2
part3["moved_the_number"] = (
    part3["arms"]["True"]["worst_deviation"] != part3["arms"]["False"]["worst_deviation"])
report["parts"]["part3_fp16_reduced_reduction"] = part3

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
del fp16_model, fp16_engine
torch.cuda.empty_cache()


# --- PART 4: memory headroom on 15 GB ---------------------------------------

section("PART 4: memory headroom on 15 GB, headline model")

# Pinned for BOTH engines and recorded. Session 1 measured this moving the fp16
# deviation by up to 0.75 ulp, so it is not a free knob.
run_mod_pin = None
try:
    from bench.run import ALLOW_FP16_REDUCED_PRECISION_REDUCTION, pin_fp16_reduction

    run_mod_pin = pin_fp16_reduction()
    print(f"pinned allow_fp16_reduced_precision_reduction="
          f"{ALLOW_FP16_REDUCED_PRECISION_REDUCTION} (applied={run_mod_pin})", flush=True)
except Exception as error:  # noqa: BLE001
    print(f"could not pin fp16 reduction: {error!r}", file=sys.stderr, flush=True)

from bench import baseline as baseline_mod  # noqa: E402
from bench import workload  # noqa: E402
from bench.workload import NUM_GPU_BLOCKS  # noqa: E402
from engine.attention import repeat_kv  # noqa: E402
from engine.scheduler import (  # noqa: E402
    EventKind, ExecutionMode, Qwen2Executor, Request, Scheduler,
)

part4 = {"fp16_reduction_pinned": run_mod_pin}
total_bytes = torch.cuda.get_device_properties(0).total_memory
part4["device_total_bytes"] = total_bytes
part4["device_total_gib"] = round(total_bytes / 2**30, 2)
print(f"device total: {total_bytes / 2**30:.2f} GiB", flush=True)

# --- the pinned SHA, confirmed rather than trusted
sha_observed = None
try:
    from huggingface_hub import HfApi

    sha_observed = HfApi().model_info(HEADLINE_MODEL).sha
except Exception as error:  # noqa: BLE001
    sha_observed = f"<unresolved: {error!r}>"
part4["headline_model"] = HEADLINE_MODEL
part4["sha_expected"] = HEADLINE_SHA_EXPECTED
part4["sha_observed"] = sha_observed
part4["sha_matches"] = sha_observed == HEADLINE_SHA_EXPECTED
print(f"headline SHA expected {HEADLINE_SHA_EXPECTED}\n"
      f"                observed {sha_observed}\n"
      f"                matches  {part4['sha_matches']}", flush=True)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
head_model = AutoModelForCausalLM.from_pretrained(
    HEADLINE_MODEL, dtype=torch.float16, revision=HEADLINE_SHA_EXPECTED)
head_model.eval()
head_model.to(DEVICE)
head_engine = Qwen2Engine.from_hf_model(head_model)
part4["weights_bytes"] = torch.cuda.memory_allocated()
part4["weights_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
print(f"weights resident: {part4['weights_gib']} GiB", flush=True)
part4["dims"] = {
    "num_layers": head_engine.dims.num_layers,
    "num_heads": head_engine.dims.num_heads,
    "num_kv_heads": head_engine.dims.num_kv_heads,
    "num_kv_groups": head_engine.dims.num_kv_groups,
    "head_dim": head_engine.dims.head_dim,
}
print(f"dims: {part4['dims']}", flush=True)


# --- does repeat_kv materialize?
section("PART 4a: does repeat_kv materialize the GQA expansion?")
groups = head_engine.dims.num_kv_groups
kv_heads = head_engine.dims.num_kv_heads
head_dim = head_engine.dims.head_dim

repeat_probe = {}
for label, (batch, kv_len) in (("C8-ish", (8, 3072)), ("C64-ish", (64, 3072))):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    sample = torch.zeros((batch, kv_heads, kv_len, head_dim),
                         dtype=torch.float16, device=DEVICE)
    before = torch.cuda.memory_allocated()
    expanded = repeat_kv(sample, groups)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    materialized = expanded.data_ptr() != sample.data_ptr()
    repeat_probe[label] = {
        "batch": batch, "kv_len": kv_len,
        "source_bytes": sample.numel() * sample.element_size(),
        "materialized": bool(materialized),
        "extra_bytes_measured": after - before,
        "extra_mib_measured": round((after - before) / 2**20, 1),
        "is_view": bool(expanded.data_ptr() == sample.data_ptr()),
    }
    print(f"{label}: materialized={materialized}, "
          f"extra={(after - before) / 2**20:.1f} MiB per tensor "
          f"(K and V both, per layer)", flush=True)
    del sample, expanded
    torch.cuda.empty_cache()

# What SDPA's own GQA path would save: it broadcasts internally instead of
# copying, so the saving is exactly the materialized bytes, for K and for V.
repeat_probe["sdpa_enable_gqa_available"] = "enable_gqa" in getattr(
    torch.nn.functional.scaled_dot_product_attention, "__doc__", "") or hasattr(
    torch.nn.functional, "scaled_dot_product_attention")
part4["repeat_kv"] = repeat_probe


# --- peak memory at three concurrency levels
section("PART 4b: peak allocated / reserved at C=8, 64, 87")

MEASURE_STEPS = int(os.environ.get("SESSION1_STEPS", "25"))

# The padded-rectangle budget. The uncapped run OOM'd at C=8 materialising
# 64 rows x 2048 = 131,072 padded slots; this is 16x smaller. It is a starting
# point to be measured, not a derived constant -- if it still OOMs that is the
# result to report.
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("SESSION1_BATCHED_TOKENS", "8192"))


class ShapeRecorder:
    """Records the batch shapes a run actually reached.

    Its counters live on the recorder, not in the try block's locals, because
    the previous session assigned them into the result dict only on the SUCCESS
    path -- so all three OOM runs lost max_batch_width and max_query_len, which
    were the two numbers that would have explained the OOM.
    """

    def __init__(self, inner):
        self.inner = inner
        self.max_batch = 0
        self.max_query = 0
        self.max_padded_slots = 0
        self.calls = 0

    def new_backend(self, seq_id):
        return self.inner.new_backend(seq_id)

    def reserve(self, backend, n):
        return self.inner.reserve(backend, n)

    def forward(self, t, p, b):
        return self.inner.forward(t, p, b)

    def forward_batch(self, inputs):
        # Record BEFORE delegating. If the forward OOMs, these are the shapes
        # that caused it and they must already be saved.
        self.calls += 1
        widths = [len(i.token_ids) for i in inputs]
        self.max_batch = max(self.max_batch, len(inputs))
        self.max_query = max([self.max_query] + widths)
        if widths:
            self.max_padded_slots = max(
                self.max_padded_slots, len(widths) * max(widths))
        return self.inner.forward_batch(inputs)


levels = {}
for concurrency in (8, 64, 87):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    entry = {"concurrency": concurrency, "steps_requested": MEASURE_STEPS}
    try:
        stream = workload.build_stream("mixed", seed=0, c=concurrency)
        entry["num_requests"] = len(stream)
        entry["demanded_blocks"] = sum(
            -(-r.total_len // BLOCK_SIZE) for r in stream)

        allocator = BlockAllocator(num_blocks=NUM_GPU_BLOCKS, block_size=BLOCK_SIZE)
        cache = head_engine.new_paged_cache(NUM_GPU_BLOCKS, BLOCK_SIZE)
        entry["kv_pool_bytes"] = cache.storage_bytes()
        entry["kv_pool_gib"] = round(cache.storage_bytes() / 2**30, 3)
        executor = ShapeRecorder(Qwen2Executor(head_engine, cache, allocator))
        # THE CAPS. max_num_seqs = C is what makes C mean anything: without it
        # admission was by pool fit alone, so C=1,2,4,8,16 all admitted the same
        # 64 requests. max_num_batched_tokens bounds the padded rectangle, which
        # is what actually allocates.
        scheduler = Scheduler(
            executor, allocator, mode=ExecutionMode.FUSED,
            max_num_seqs=concurrency,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        )
        entry["max_num_seqs"] = concurrency
        entry["max_num_batched_tokens"] = MAX_NUM_BATCHED_TOKENS

        rejected = 0
        for item in stream:
            if not scheduler.add_request(Request(
                    request_id=item.request_id,
                    prompt_token_ids=item.prompt_token_ids,
                    max_new_tokens=item.max_new_tokens)):
                rejected += 1
        entry["rejected"] = rejected

        steps = 0
        while (scheduler.waiting or scheduler.running) and steps < MEASURE_STEPS:
            scheduler.step()
            steps += 1
        torch.cuda.synchronize()
        entry["steps_run"] = steps
        entry["running_at_stop"] = len(scheduler.running)
        entry["preemptions"] = len(scheduler.events_of(EventKind.PREEMPT))
        entry["peak_blocks_used"] = max(
            (NUM_GPU_BLOCKS - e.free_blocks for e in scheduler.events), default=0)
        entry["ok"] = True
    except torch.cuda.OutOfMemoryError as error:
        entry["ok"] = False
        entry["oom"] = True
        entry["error"] = str(error)[:2000]
        print(f"C={concurrency}: OUT OF MEMORY\n{error}", file=sys.stderr, flush=True)
    except Exception as error:  # noqa: BLE001
        entry["ok"] = False
        entry["error"] = f"{type(error).__name__}: {error}"[:2000]
        entry["traceback"] = traceback.format_exc()[-3000:]
        print(f"C={concurrency}: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)

    # ALWAYS, whatever happened. These are the shapes that explain an OOM.
    try:
        entry["max_batch_width"] = executor.max_batch
        entry["max_query_len"] = executor.max_query
        entry["max_padded_slots"] = executor.max_padded_slots
        entry["forward_batch_calls"] = executor.calls
    except NameError:
        entry["max_batch_width"] = None
    try:
        entry["steps_run"] = entry.get("steps_run", steps)
        entry["preemptions"] = entry.get(
            "preemptions", len(scheduler.events_of(EventKind.PREEMPT)))
        entry["peak_blocks_used"] = entry.get("peak_blocks_used", max(
            (NUM_GPU_BLOCKS - e.free_blocks for e in scheduler.events), default=0))
    except NameError:
        pass

    entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    entry["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    entry["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    entry["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    entry["headroom_gib"] = round(
        (total_bytes - torch.cuda.max_memory_reserved()) / 2**30, 3)
    print(f"C={concurrency}: ok={entry.get('ok')} "
          f"peak_allocated={entry['peak_allocated_gib']} GiB "
          f"peak_reserved={entry['peak_reserved_gib']} GiB "
          f"headroom={entry['headroom_gib']} GiB "
          f"max_batch={entry.get('max_batch_width')} "
          f"max_q={entry.get('max_query_len')} "
          f"max_padded_slots={entry.get('max_padded_slots')}", flush=True)
    levels[str(concurrency)] = entry
    try:
        del scheduler, executor, cache, allocator
    except NameError:
        pass
    torch.cuda.empty_cache()

part4["levels"] = levels
report["parts"]["part4_memory_headroom"] = part4


# --- PART 5: harness smoke at tiny scale on the GPU -------------------------

section("PART 5: harness smoke on the GPU. No sweep, no throughput result.")

from bench import run as run_mod  # noqa: E402

part5 = {}

# The two rectangle sizings, and a real generate through each.
sizings = {}
tiny_stream = workload.build_stream("mixed", seed=0, c=1)[:2]
tiny_stream = tuple(
    workload.Request(request_id=r.request_id,
                     prompt_token_ids=r.prompt_token_ids[:64],
                     max_new_tokens=4)
    for r in tiny_stream
)
generator = baseline_mod.HuggingFaceGenerator(head_model, None, DEVICE)
for rectangle in baseline_mod.RECTANGLE_SIZINGS:
    entry = {"rectangle_tokens": rectangle,
             "max_resident": baseline_mod.max_resident(rectangle)}
    try:
        observed = baseline_mod.run_baseline(
            tiny_stream, generator, rectangle, max_num_seqs=2)
        entry.update({
            "ok": True,
            "num_accepted": observed.num_accepted,
            "num_rejected": observed.num_rejected,
            "rejection_rate": observed.rejection_rate,
            "generated_tokens": observed.generated_tokens,
            "num_batches": observed.num_batches,
            "max_num_seqs": observed.max_num_seqs,
            "effective_resident": observed.max_resident,
            "slot_waste": observed.slot_waste,
        })
    except Exception as error:  # noqa: BLE001
        entry.update({"ok": False, "error": f"{type(error).__name__}: {error}"[:1000]})
    print(f"rectangle {rectangle}: {entry}", flush=True)
    sizings[str(rectangle)] = entry
part5["rectangle_sizings"] = sizings

# GUARD 1: the control headline refuses any level other than PRESSURE_C.
guard1 = {}
try:
    run_mod.check_control_headline_level("control", workload.PRESSURE_C, headline=True)
    guard1["accepts_pressure_c"] = True
except run_mod.InvalidRun as error:
    guard1["accepts_pressure_c"] = False
    guard1["unexpected"] = str(error)
try:
    run_mod.check_control_headline_level("control", 32, headline=True)
    guard1["refused_wrong_level"] = False
except run_mod.InvalidRun as error:
    guard1["refused_wrong_level"] = True
    guard1["message"] = str(error)[:600]
print(f"GUARD control-headline: {guard1}", flush=True)
part5["guard_control_headline"] = guard1

# GUARD 2: a C64 run with zero preemptions is INVALID.
guard2 = {}
level = run_mod.LevelResult(
    engine=run_mod.ENGINE_PAGED, stream="control", concurrency=workload.PRESSURE_C,
    num_requests=256, passes=[1.0, 1.0, 1.0], preemption_count=0)
try:
    run_mod.check_preemptions(level, demanded_blocks=4096)
    guard2["failed_loudly"] = False
except run_mod.InvalidRun as error:
    guard2["failed_loudly"] = True
    guard2["message"] = str(error)[:600]
level.preemption_count = 1
try:
    run_mod.check_preemptions(level, demanded_blocks=4096)
    guard2["passes_with_one_preemption"] = True
except run_mod.InvalidRun:
    guard2["passes_with_one_preemption"] = False
print(f"GUARD zero-preemptions-at-C64: {guard2}", flush=True)
part5["guard_zero_preemptions"] = guard2

# --smoke records must still carry throughput_valid false, ON A CUDA HOST.
smoke = {}
provenance = run_mod.Provenance.collect(
    HEADLINE_MODEL, "torch.float16",
    workload.stream_digest(workload.build_stream("control", seed=0, c=1)))
written = run_mod.write_results(WORKING / "smoke_run.json", provenance, [], smoke=True)
payload = json.loads(written.read_text())
smoke["cuda_available"] = payload["provenance"]["cuda_available"]
smoke["smoke"] = payload["smoke"]
smoke["throughput_valid"] = payload["throughput_valid"]
smoke["visible_cuda_devices"] = payload["provenance"].get("visible_cuda_devices")
smoke["cuda_visible_devices_env"] = payload["provenance"].get("cuda_visible_devices_env")
smoke["fp32_precision_recorded"] = payload["provenance"].get("fp32_precision")
smoke["gpu_name"] = payload["provenance"].get("gpu_name")
smoke["driver_version"] = payload["provenance"].get("driver_version")
smoke["model_sha"] = payload["provenance"].get("model_sha")
print(f"--smoke record: {json.dumps(smoke, indent=2)}", flush=True)
part5["smoke_record"] = smoke

report["parts"]["part5_harness_smoke"] = part5


# --- done -------------------------------------------------------------------

section("DONE")
REPORT.write_text(json.dumps(report, indent=2, default=str))
print(f"wrote {REPORT}", flush=True)
for name in report["parts"]:
    print(f"  recorded: {name}", flush=True)
