"""Naive HuggingFace generate loop, held to the engine's own KV byte ceiling.

Gate 1, Step 6. This is the thing the engine is measured against, so the only
interesting question about it is whether the comparison is fair. It is made fair
in one way and one way only: BOTH SIDES GET THE SAME NUMBER OF KV BYTES.

    KV_POOL_BYTES = num_gpu_blocks * block_size * bytes_per_token
                  = 3584 * 16 * 28672
                  = 1,644,167,168

The engine spends that on 3584 blocks of 16 tokens and hands them out a block at
a time. The baseline spends it on a rectangle: every resident sequence gets
`rectangle_tokens` slots whether it needs them or not, because that is what a
static batch is. So

    max_resident = KV_POOL_BYTES // (rectangle_tokens * bytes_per_token)

and the whole of Gate 1 is the claim that dividing the same bytes the second way
is worse. Nothing else about the baseline is handicapped, and nothing is padded
in its favour either.

THE CEILING IS A BUDGETING RULE, NOT AN OBSERVED LIMIT, AND ON ONE SIZING IT IS
EXCEEDED. `max_resident` charges every resident sequence `rectangle_tokens`
slots, but HuggingFace's cache actually holds
`resident * (longest_prompt_in_batch + longest_decode_in_batch)` slots, and those
two maxima come from DIFFERENT requests -- so a batch whose members each fit the
rectangle can still exceed `resident * rectangle_tokens`. Computed directly from
the frozen stream at C=64: on `control` no batch exceeds the ceiling at either
sizing, and on `mixed` at rect3072 none does (worst batch 55,296 of 57,344
slots), but on `mixed` at rect1757 SEVEN OF EIGHT BATCHES DO, the mean batch
demanding 68,364 slots and the worst 77,216 -- 34.7% over. Nothing here or in
`bench/run.py` records `torch.cuda.max_memory_allocated`, so neither side's real
KV residency was ever observed. "Both sides get the same KV bytes" holds as
arithmetic on the rectangle, and holds on the device only where the realized
demand above stays under the ceiling.

NOT A PROCESS-WIDE MEMORY FRACTION. It would be much easier to cap the baseline
with `torch.cuda.set_per_process_memory_fraction` and let it OOM, and it would
be wrong: that budget also has to hold the weights, the activations and every
cuBLAS workspace, so the baseline would lose because it had less room for
*everything*, not because a rectangle wastes KV. The ceiling here constrains KV
residency and nothing else.

TWO SIZINGS, both runnable, and they are not the same experiment.

    3072  the distribution's maximum. No request can exceed the rectangle, so
          nothing is rejected and the baseline runs the whole stream -- but 18
          residents is all the ceiling buys.
    1757  the P95. 32 residents, nearly twice as many, paid for by the ~5% of
          requests that no longer fit and are rejected. `rejection_rate` is
          recorded for exactly this reason: the P95 sizing's advantage is not
          free and the number that says so travels with the result.

REJECTION IS SHARED CODE, not a reimplementation. `engine.scheduler.is_admissible`
is the predicate the Scheduler applies, and this module calls that same function,
so "rejected identically to the engine" is literally true rather than a claim
about two pieces of code that look alike. On top of it the baseline applies its
own rectangle limit, which the engine has no analogue for; the two counts are
recorded separately so a rejection can always be attributed.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from bench.workload import BLOCK_SIZE, NUM_GPU_BLOCKS, Request
from engine.cache import ADMISSION_MARGIN_BLOCKS, BlockAllocator
from engine.scheduler import is_admissible

# --- the ceiling -----------------------------------------------------------

# Qwen2.5-1.5B-Instruct KV geometry. Stated as fields rather than a single magic
# number so `kv_pool_bytes` below is derived and can be checked against the
# locked constants instead of being asserted.
BENCH_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_LAYERS = 28
NUM_KV_HEADS = 2
HEAD_DIM = 128
# fp16 on a T4 or a P100. bf16 is not available on either -- see
# tests/test_parity_fp16.py -- and it would not change this arithmetic anyway,
# because both formats are two bytes per element.
BYTES_PER_ELEMENT = 2

# K and V, every layer, every KV head, every element of the head dimension.
BYTES_PER_TOKEN = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_ELEMENT
BYTES_PER_BLOCK = BYTES_PER_TOKEN * BLOCK_SIZE
KV_POOL_TOKEN_SLOTS = NUM_GPU_BLOCKS * BLOCK_SIZE
KV_POOL_BYTES = KV_POOL_TOKEN_SLOTS * BYTES_PER_TOKEN

# The two sizings. Named, because "3072" on its own does not say why.
RECTANGLE_DISTRIBUTION_MAX = 3072  # prompt.hi 2048 + decode.hi 1024
RECTANGLE_P95 = 1757
RECTANGLE_SIZINGS = (RECTANGLE_DISTRIBUTION_MAX, RECTANGLE_P95)


def max_resident(rectangle_tokens: int, ceiling_bytes: int = KV_POOL_BYTES) -> int:
    """How many sequences of `rectangle_tokens` slots the ceiling pays for."""
    if rectangle_tokens < 1:
        raise ValueError(f"rectangle_tokens must be at least 1, got {rectangle_tokens}")
    resident = ceiling_bytes // (rectangle_tokens * BYTES_PER_TOKEN)
    if resident < 1:
        raise ValueError(
            f"a {rectangle_tokens}-token rectangle needs "
            f"{rectangle_tokens * BYTES_PER_TOKEN} bytes per sequence, which does "
            f"not fit in the {ceiling_bytes}-byte ceiling even once"
        )
    return int(resident)


def rectangle_bytes_used(rectangle_tokens: int, ceiling_bytes: int = KV_POOL_BYTES) -> int:
    """Bytes the rectangle actually occupies. The remainder is unusable."""
    return max_resident(rectangle_tokens, ceiling_bytes) * rectangle_tokens * BYTES_PER_TOKEN


# --- rejection -------------------------------------------------------------


class RejectionReason:
    ENGINE_GUARD = "engine_admission_guard"
    OVER_RECTANGLE = "longer_than_rectangle"


# --- batch order -----------------------------------------------------------
#
# THE ONE THING THE BASELINE IS ALLOWED TO VARY, AND IT IS NOT FREE TO THE
# COMPARISON. A static batch runs every row until the batch's longest member
# finishes its budget, so mixing a 1024-token decode with an 8-token decode
# makes the short row compute 1016 positions nobody reads. Grouping by decode
# length is the cheapest thing a static batcher can do about that, so a headline
# measured against the arrival-order form is measured against the weaker of two
# static batchers, and the gap between them is a number rather than an argument.
#
# `arrival` is the default and is the original behaviour, byte for byte.
# `sorted` reorders the ALREADY-ACCEPTED list and nothing else: acceptance,
# rejection, the rectangle, `max_resident`, the KV ceiling and every token count
# are computed before it and are order-invariant afterwards. The workload stream
# is not touched -- `classify` returns a fresh tuple and this sorts a copy of it.
#
# Descending, and via a stable sort, so requests of equal budget keep their
# arrival order and the batching is a deterministic function of the stream.
BATCH_ORDER_ARRIVAL = "arrival"
BATCH_ORDER_SORTED = "sorted"
BATCH_ORDERS = (BATCH_ORDER_ARRIVAL, BATCH_ORDER_SORTED)


def order_for_batching(accepted: Sequence[Request], batch_order: str) -> tuple[Request, ...]:
    """The accepted list in the order batches are cut from it.

    Raises rather than defaulting on an unknown value: a typo that silently
    fell back to `arrival` would report a sorted measurement that never sorted,
    which is the one failure mode that looks like a clean result.
    """
    if batch_order == BATCH_ORDER_ARRIVAL:
        return tuple(accepted)
    if batch_order == BATCH_ORDER_SORTED:
        return tuple(sorted(accepted, key=lambda request: -request.max_new_tokens))
    raise ValueError(
        f"unknown batch_order {batch_order!r}, expected one of {list(BATCH_ORDERS)}"
    )


@dataclass(frozen=True)
class Rejection:
    request_id: str
    reason: str
    total_len: int
    limit: int


def classify(
    requests: Sequence[Request],
    rectangle_tokens: int,
    num_gpu_blocks: int = NUM_GPU_BLOCKS,
    block_size: int = BLOCK_SIZE,
    admission_margin_blocks: int = ADMISSION_MARGIN_BLOCKS,
) -> tuple[tuple[Request, ...], tuple[Rejection, ...]]:
    """Split a stream into what the baseline will run and what it turns away.

    Two independent filters, kept separate so a rejection can be attributed:

      * the engine's own admission guard, applied through the engine's own
        function, so both sides reject exactly the same requests for that reason;
      * the rectangle, which is the baseline's alone -- a sequence longer than
        the rectangle has nowhere to live in a static batch.
    """
    allocator = BlockAllocator(num_blocks=num_gpu_blocks, block_size=block_size)
    engine_limit_tokens = (num_gpu_blocks - admission_margin_blocks) * block_size

    accepted: list[Request] = []
    rejected: list[Rejection] = []
    for request in requests:
        if not is_admissible(request.total_len, allocator, admission_margin_blocks):
            rejected.append(
                Rejection(request.request_id, RejectionReason.ENGINE_GUARD,
                          request.total_len, engine_limit_tokens)
            )
        elif request.total_len > rectangle_tokens:
            rejected.append(
                Rejection(request.request_id, RejectionReason.OVER_RECTANGLE,
                          request.total_len, rectangle_tokens)
            )
        else:
            accepted.append(request)
    return tuple(accepted), tuple(rejected)


# --- result ----------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """One measured pass. Timing is separate from everything that describes it."""

    rectangle_tokens: int
    max_resident: int
    max_num_seqs: int | None
    batch_order: str
    num_offered: int
    num_accepted: int
    num_rejected: int
    rejected_by_engine_guard: int
    rejected_over_rectangle: int
    generated_tokens: int
    num_batches: int
    # WHAT THE BATCHING ACTUALLY COST, so a static tok/s can be read against the
    # work it issued rather than against the work it was credited for. Counted,
    # not derived from the timing: `forward_passes` is one prefill plus `steps`
    # decode steps per batch, and `decode_row_steps` is those decode steps times
    # the rows that sat through them. `generated_tokens` is what was kept.
    forward_passes: int
    decode_row_steps: int
    padded_token_slots: int
    used_token_slots: int
    elapsed_seconds: float
    outputs: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)

    @property
    def rejection_rate(self) -> float:
        return self.num_rejected / self.num_offered if self.num_offered else 0.0

    @property
    def decode_step_efficiency(self) -> float:
        """Kept decode tokens over decode row-steps computed. 1.0 wastes nothing."""
        return (
            self.generated_tokens / self.decode_row_steps
            if self.decode_row_steps
            else 0.0
        )

    @property
    def slot_waste(self) -> float:
        """Fraction of the rectangle's slots that held nothing. The whole point."""
        return (
            (self.padded_token_slots - self.used_token_slots) / self.padded_token_slots
            if self.padded_token_slots
            else 0.0
        )

    def describe(self) -> dict[str, object]:
        return {
            "rectangle_tokens": self.rectangle_tokens,
            "max_resident": self.max_resident,
            "max_num_seqs": self.max_num_seqs,
            "batch_order": self.batch_order,
            "num_offered": self.num_offered,
            "num_accepted": self.num_accepted,
            "num_rejected": self.num_rejected,
            "rejected_by_engine_guard": self.rejected_by_engine_guard,
            "rejected_over_rectangle": self.rejected_over_rectangle,
            "rejection_rate": round(self.rejection_rate, 6),
            "generated_tokens": self.generated_tokens,
            "num_batches": self.num_batches,
            "forward_passes": self.forward_passes,
            "decode_row_steps": self.decode_row_steps,
            "decode_step_efficiency": round(self.decode_step_efficiency, 6),
            "padded_token_slots": self.padded_token_slots,
            "used_token_slots": self.used_token_slots,
            "slot_waste": round(self.slot_waste, 6),
            "elapsed_seconds": self.elapsed_seconds,
        }


# --- the loop --------------------------------------------------------------


class HuggingFaceGenerator:
    """The naive side: `model.generate`, one static batch at a time.

    Deliberately not clever. Requests are taken in stream order, `max_resident`
    at a time, and every member of a batch is held until the longest member of
    that batch is done. That is the cost paged attention exists to remove, so
    removing it here would be measuring the wrong thing.

    The workload contract is honoured exactly as the engine honours it: greedy,
    EOS ignored, exactly `max_new_tokens` produced. `min_new_tokens` is pinned to
    `max_new_tokens` for that reason -- a baseline that stopped early would be
    doing less work and the comparison would be void.
    """

    def __init__(self, model, tokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer is not None and tokenizer.pad_token_id is not None
            else 0
        )

    @torch.inference_mode()
    def generate_batch(self, batch: Sequence[Request], **generate_kwargs) -> dict[str, tuple[int, ...]]:
        longest_prompt = max(request.prompt_len for request in batch)
        steps = max(request.max_new_tokens for request in batch)

        # Left padding, so every sequence's last real token is at the same index
        # and one `generate` call serves the whole batch.
        input_ids = torch.full(
            (len(batch), longest_prompt), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(batch), longest_prompt), dtype=torch.long)
        for row, request in enumerate(batch):
            start = longest_prompt - request.prompt_len
            input_ids[row, start:] = torch.tensor(request.prompt_token_ids, dtype=torch.long)
            attention_mask[row, start:] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        produced = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=steps,
            min_new_tokens=steps,  # EOS ignored: the contract is exact lengths
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=None,
            # Stage 2 injects a LogitsProcessor here to timestamp the first
            # token. Nothing else passes anything, and the default call is
            # unchanged, so every existing measurement stays comparable.
            **generate_kwargs,
        )

        outputs: dict[str, tuple[int, ...]] = {}
        for row, request in enumerate(batch):
            new_tokens = produced[row, longest_prompt:].tolist()
            # Each request keeps only its own budget. The rest of the row is work
            # the batch did on its behalf and is not part of its answer.
            outputs[request.request_id] = tuple(new_tokens[: request.max_new_tokens])
        return outputs


def run_baseline(
    requests: Sequence[Request],
    generator: HuggingFaceGenerator,
    rectangle_tokens: int,
    *,
    ceiling_bytes: int = KV_POOL_BYTES,
    max_num_seqs: int | None = None,
    batch_order: str = BATCH_ORDER_ARRIVAL,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] | None = None,
) -> BaselineResult:
    """One measured pass of the naive baseline over `requests`.

    `clock` and `synchronize` are injected so a test can drive this with a fake
    generator and a fake clock and check the accounting without measuring
    anything. `batch_order` is the one knob: `arrival` is the naive batching, on
    purpose and by default, and `sorted` groups the accepted list by decode
    budget before cutting batches. Nothing else here is configurable.
    """
    resident = max_resident(rectangle_tokens, ceiling_bytes)
    # THE SAME CONCURRENCY CAP THE ENGINE GETS. Without it a sweep at C compares
    # an engine running whatever fits the pool against a baseline hard-capped at
    # 18 or 32 by its rectangle -- at C=8 that was engine-at-64 against
    # baseline-at-18, and no ratio out of that means anything. Both sides now run
    # at min(their own structural limit, C).
    if max_num_seqs is not None:
        resident = min(resident, max_num_seqs)
    accepted, rejected = classify(requests, rectangle_tokens)

    # ORDER ONLY, AND ONLY HERE. Acceptance is already decided above and every
    # count below is taken over `accepted`, not over `ordered`, so the two modes
    # differ in which requests share a batch and in nothing else.
    ordered = order_for_batching(accepted, batch_order)
    batches = [
        ordered[start : start + resident] for start in range(0, len(ordered), resident)
    ]

    if synchronize is not None:
        synchronize()
    started = clock()
    outputs: dict[str, tuple[int, ...]] = {}
    for batch in batches:
        outputs.update(generator.generate_batch(batch))
    if synchronize is not None:
        synchronize()
    elapsed = clock() - started

    # What the rectangle charged for, against what was actually occupied. A batch
    # holds `resident * rectangle_tokens` slots even when it is short a member.
    padded = len(batches) * resident * rectangle_tokens
    used = sum(request.total_len for request in accepted)

    # What the batching issued. One prefill plus `steps` decode steps per batch,
    # where `steps` is the batch's longest budget -- the same rule
    # `generate_batch` runs by, restated here rather than measured, so the count
    # is a property of the batching and not of the model.
    steps_per_batch = [max(request.max_new_tokens for request in batch) for batch in batches]
    forward_passes = sum(1 + steps for steps in steps_per_batch)
    decode_row_steps = sum(
        steps * len(batch) for steps, batch in zip(steps_per_batch, batches)
    )

    return BaselineResult(
        rectangle_tokens=rectangle_tokens,
        max_resident=resident,
        max_num_seqs=max_num_seqs,
        batch_order=batch_order,
        num_offered=len(requests),
        num_accepted=len(accepted),
        num_rejected=len(rejected),
        rejected_by_engine_guard=sum(
            1 for r in rejected if r.reason == RejectionReason.ENGINE_GUARD
        ),
        rejected_over_rectangle=sum(
            1 for r in rejected if r.reason == RejectionReason.OVER_RECTANGLE
        ),
        generated_tokens=sum(len(tokens) for tokens in outputs.values()),
        num_batches=len(batches),
        forward_passes=forward_passes,
        decode_row_steps=decode_row_steps,
        padded_token_slots=padded,
        used_token_slots=used,
        elapsed_seconds=elapsed,
        outputs=outputs,
    )


def load_baseline(model_id: str = BENCH_MODEL, device: str = "cuda", dtype=torch.float16):
    """Load the model the baseline runs. Imported lazily by callers that need it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval()
    model.to(device)
    return HuggingFaceGenerator(model, tokenizer, device)


# --- Arm B: forcing the baseline onto the same kernel as the engine ---------
#
# The engine and the baseline must run the SAME SDPA kernel or the Gate 1 ratio
# is partly a kernel comparison. Which kernel each side gets is decided by input
# SHAPE, and for the baseline that shape is decided by one HuggingFace helper:
#
#     transformers.integrations.sdpa_attention.use_gqa_in_sdpa(mask, key) -> bool
#
#     True  -> HF passes unexpanded K/V with enable_gqa=True
#     False -> HF calls repeat_kv first and passes expanded K/V
#
# and HF returns True only when `attention_mask is None`. That makes the
# baseline's kernel DEPEND ON THE STREAM, which is not obvious and was missed
# once already:
#
#   control stream -- every prompt is 512 tokens, so left padding adds nothing,
#     transformers drops the all-ones mask to None, and HF takes enable_gqa.
#   mixed stream -- 224 distinct prompt lengths, so the mask is real, and HF
#     pre-expands. On a T4 that is EFFICIENT_ATTENTION, the fast kernel, while
#     the engine's masked fused call sits on MATH.
#
# Forcing this helper to False makes the baseline pre-expand unconditionally,
# which is what `GQA_PATH=repeat_kv` does on the engine side. It changes the
# input SHAPE and nothing else -- not the arithmetic HF intends, not the mask,
# not `is_causal`, whose expression does not reference this at all.
#
# This is a patch applied to the baseline and it must therefore clear a bar:
# `assert_token_identical` below. A baseline whose answers moved is a different
# baseline and may not be timed.


class BaselineNotIdentical(RuntimeError):
    """The patched baseline answers differently. Arm B may not be timed."""


def force_baseline_kv_expansion():
    """Make HF pre-expand K/V on every call. Returns an undo callable.

    Raises rather than degrading if the helper is not where it is expected --
    a silently unapplied patch would leave Arm B measuring Arm A twice, which
    is the one failure mode that looks like a clean result.
    """
    from transformers.integrations import sdpa_attention

    original = getattr(sdpa_attention, "use_gqa_in_sdpa", None)
    if not callable(original):
        raise RuntimeError(
            "transformers.integrations.sdpa_attention.use_gqa_in_sdpa is missing "
            f"(transformers has {sorted(n for n in dir(sdpa_attention) if 'gqa' in n.lower())}). "
            "Arm B cannot force the baseline onto the expanded-K/V path without "
            "it, and guessing another patch point would be measuring something "
            "unknown. Re-derive the patch for this transformers version."
        )

    def never_use_gqa(attention_mask, key):
        return False

    sdpa_attention.use_gqa_in_sdpa = never_use_gqa

    def undo():
        sdpa_attention.use_gqa_in_sdpa = original

    return undo


def baseline_expands_kv_for(generator: "HuggingFaceGenerator", batch) -> bool:
    """Does the baseline pre-expand K/V for THIS batch? Observed, not assumed."""
    from transformers.integrations import sdpa_attention

    observed: list[bool] = []
    current = sdpa_attention.use_gqa_in_sdpa

    def spy(attention_mask, key):
        result = current(attention_mask, key)
        observed.append(not result)
        return result

    sdpa_attention.use_gqa_in_sdpa = spy
    try:
        generator.generate_batch(batch)
    finally:
        sdpa_attention.use_gqa_in_sdpa = current
    if not observed:
        raise RuntimeError(
            "no SDPA GQA decision was observed during a baseline batch; the "
            "attention implementation is not the sdpa one this probe assumes"
        )
    return all(observed)


def assert_token_identical(generator: "HuggingFaceGenerator", batches) -> dict:
    """Arm B's prerequisite: the patched baseline must answer identically.

    Pre-expanding K/V is mathematically a no-op -- `tests/test_attention_gqa.py`
    holds the two expansions to `torch.equal` -- but it changes which kernel
    torch selects, and kernels differ in accumulation order. In fp16 that can
    move a token. If it does, Arm B is timing a baseline that gives different
    answers, which is not the baseline, and the arm is void.

    Returns the evidence. Raises `BaselineNotIdentical` on any mismatch.
    """
    unpatched = [generator.generate_batch(batch) for batch in batches]

    undo = force_baseline_kv_expansion()
    try:
        patched = [generator.generate_batch(batch) for batch in batches]
        expanded = [baseline_expands_kv_for(generator, batch) for batch in batches]
    finally:
        undo()

    compared = 0
    for index, (before, after) in enumerate(zip(unpatched, patched)):
        if before.keys() != after.keys():
            raise BaselineNotIdentical(
                f"batch {index}: patched baseline returned a different request set"
            )
        for request_id, tokens in before.items():
            compared += len(tokens)
            if tokens != after[request_id]:
                first = next(
                    (position for position, (a, b) in enumerate(zip(tokens, after[request_id])) if a != b),
                    min(len(tokens), len(after[request_id])),
                )
                raise BaselineNotIdentical(
                    f"batch {index}, request {request_id}: patched baseline "
                    f"diverged at token {first} of {len(tokens)} "
                    f"({tokens[max(0, first - 2):first + 2]} vs "
                    f"{after[request_id][max(0, first - 2):first + 2]}). "
                    "Pre-expanding K/V changed the baseline's answers, so Arm B "
                    "would be timing a different baseline. Arm B is VOID."
                )

    if not all(expanded):
        raise BaselineNotIdentical(
            "the patch did not take: the baseline still chose enable_gqa on at "
            "least one probed batch. Arm B would silently repeat Arm A."
        )

    return {
        "batches_compared": len(batches),
        "tokens_compared": compared,
        "token_identical": True,
        "patch_verified_active": True,
    }
