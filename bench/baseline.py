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
    num_offered: int
    num_accepted: int
    num_rejected: int
    rejected_by_engine_guard: int
    rejected_over_rectangle: int
    generated_tokens: int
    num_batches: int
    padded_token_slots: int
    used_token_slots: int
    elapsed_seconds: float
    outputs: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)

    @property
    def rejection_rate(self) -> float:
        return self.num_rejected / self.num_offered if self.num_offered else 0.0

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
            "num_offered": self.num_offered,
            "num_accepted": self.num_accepted,
            "num_rejected": self.num_rejected,
            "rejected_by_engine_guard": self.rejected_by_engine_guard,
            "rejected_over_rectangle": self.rejected_over_rectangle,
            "rejection_rate": round(self.rejection_rate, 6),
            "generated_tokens": self.generated_tokens,
            "num_batches": self.num_batches,
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
    def generate_batch(self, batch: Sequence[Request]) -> dict[str, tuple[int, ...]]:
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
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] | None = None,
) -> BaselineResult:
    """One measured pass of the naive baseline over `requests`.

    `clock` and `synchronize` are injected so a test can drive this with a fake
    generator and a fake clock and check the accounting without measuring
    anything. Nothing else in this function is configurable: the batching is the
    naive batching on purpose.
    """
    resident = max_resident(rectangle_tokens, ceiling_bytes)
    accepted, rejected = classify(requests, rectangle_tokens)

    batches = [
        accepted[start : start + resident] for start in range(0, len(accepted), resident)
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

    return BaselineResult(
        rectangle_tokens=rectangle_tokens,
        max_resident=resident,
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
