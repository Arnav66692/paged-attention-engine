"""Step 5b: batched execution. The fused path against the unfused reference.

Steps 4 and 5 ran one forward per sequence per step. That is correct and it is
byte-identical to running each request alone, but N sequences in a step cost N
forwards, so concurrency bought nothing -- and Gate 1 is defined by a throughput
number. This step fuses a step's whole running set into one batched forward.

The correctness chain is deliberately two links, not one:

  sequential  ==  unfused-concurrent   byte-identical. Asserted in
                                       tests/test_sched.py, unchanged, and that
                                       path is never removed.
  unfused     ~=  fused                within FUSED_LOGIT_ATOL below, declared
                                       from the arithmetic in this file and
                                       nowhere else, with token IDs asserted
                                       exactly equal.

THE NEGATIVE CONTROL is the point of this file. A `forward_batch` that loops over
its inputs and stacks the results passes every numerical test here -- it does not
merely come close to the unfused path, it *is* the unfused path -- so no
tolerance, no token comparison and no logit deviation can distinguish fusion from
a loop with better naming. Only counting passes can.
`test_a_looping_pseudo_fusion_is_numerically_perfect_and_still_caught` builds
exactly that impostor and shows the same assertion the real path satisfies fails
for it.

NO BUCKETING. Grouping sequences by length would cut padding waste, and the spec
permits it. It is not done, because a step that issues one call per bucket is
indistinguishable by call count from a loop, which would cost the negative
control its teeth. The padding this leaves on the table is measured and printed
rather than left unsaid.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import pytest
import torch

from engine.attention import BatchInput, FusedPagedAttention, PagedBackend
from engine.cache import (
    ADMISSION_MARGIN_BLOCKS,
    BLOCK_SIZE,
    BlockAllocator,
    PagedCacheSpec,
    PagedKVCache,
)
from tests.devices import model_devices
from engine.scheduler import (
    EventKind,
    ExecutionMode,
    Qwen2Executor,
    Request,
    Scheduler,
    StepResult,
)

DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# The same text as tests/test_parity.py and tests/test_sched.py, so a fused
# failure can be compared against the float32 parity result for the same tokens.
BASE_PROMPT = (
    "The capital of France is Paris, and the capital of Japan is Tokyo. "
    "Rivers shape the cities built beside them, carrying trade inland long "
    "before roads were paved, and the oldest bridges still mark where the "
    "first crossings were cut into the banks."
)

# Against the locked block size of 16: under one block, an exact multiple, and
# one token past a boundary. The three lengths the deviation is reported at.
PROMPT_LENGTHS = (13, 32, 33)
SEED = 0

# Absolute tolerance on logits, unfused vs fused, in float32. Declared here,
# derived from the arithmetic, and fixed before anything was measured.
#
# It is NOT LOGIT_ATOL and NOT FP16_LOGIT_ATOL. Those bound different
# comparisons -- this engine against HuggingFace, in two different dtypes -- and
# carrying a constant across comparisons is how a tolerance stops being a
# prediction and becomes a description of whatever the code happened to produce.
# Neither is imported into this module, and
# `test_fused_tolerance_is_declared_here` asserts both names are absent.
#
# Both sides of this comparison are this engine: same weights, same dtype, same
# device, the same `_run_layers`. They differ in two places and only two.
#
#   1. Every GEMM changes shape. The unfused path multiplies (n, 896) activations
#      against each projection; the fused path multiplies
#      (batch * max_query_len, 896). A different shape is a different tiling,
#      which is a different order of summation.
#   2. Softmax reduces over max_kv_len instead of kv_len. The padded terms are
#      masked to -inf and contribute exactly zero, but they move where the
#      partial sums are cut.
#
# float32 carries a 24-bit mantissa, so eps = 2**-24 = 5.96e-8. The longest
# reduction in a layer is the MLP down-projection over 4864 terms, and two
# orderings of that sum differ by roughly sqrt(4864) * eps = 4.2e-6 relative.
# Taking the 24 layers to accumulate linearly rather than as a random walk --
# the pessimistic assumption, and the right one for a bound -- gives 1.0e-4
# relative in the residual stream. The lm_head is a length-896 reduction, which
# preserves relative error rather than amplifying it. This model's logits reach
# magnitudes near 2.2e1, so the expected absolute deviation is near 2.2e-3.
# 5e-3 is that with a factor of two in hand.
#
# TOKEN EQUALITY IS THE GATE and is asserted exactly. This constant only bounds
# how far the logits may drift while still producing the same argmax, and the
# measured deviation is printed for every request. It is not widened if a test
# fails; a failure is reported.
FUSED_LOGIT_ATOL = 5e-3

# Pool for the model-tier fixtures. Comfortably more than the 13 blocks the whole
# request set needs, so nothing is preempted and the two modes take the same
# scheduling decisions -- which is what makes the logit comparison step-aligned.
FUSED_POOL_BLOCKS = 24

# The request set. Four distinct prompt lengths covering all three of
# PROMPT_LENGTHS, with four distinct decode budgets so the running set shrinks
# 4 -> 3 -> 2 as they finish, plus one late arrival admitted while the others are
# already decoding. The late arrival is what forces a step to mix a 33-token
# prefill with single-token decodes, which is the only way the query axis gets
# padded in earnest.
UPFRONT_SHAPES = ((13, 5), (20, 4), (32, 6), (33, 3))
LATE_SHAPE = (33, 4)
LATE_PROMPT_OFFSET = 8  # a different slice, so the late request is not a prefix
LATE_ARRIVAL_STEP = 2

# Forced-preemption fixture, so recompute is exercised in fused mode too. Small
# blocks and a tight pool; the numbers come from tests/test_sched.py's Step 5
# recompute case.
RECOMPUTE_BLOCK_SIZE = 4
RECOMPUTE_POOL_BLOCKS = 18
RECOMPUTE_PROMPT_LEN = 33
RECOMPUTE_MAX_NEW = 16
RECOMPUTE_FILLER_PROMPT_LEN = 16
RECOMPUTE_FILLER_MAX_NEW = 16

# The thrashing fixture, shape-identical to the one tests/test_sched.py uses for
# test_recompute_folds_into_the_same_batched_call. One of its steps readmits two
# preempted sequences and evicts one of them again, all before the flush.
CO_POOL_BLOCKS = 12
CO_BLOCK_SIZE = 4
CO_MARGIN = 0
CO_REQUESTS = 5
CO_PROMPT_LEN = 5
CO_MAX_NEW = 24

# Throughput fixture. Big enough that the measurement is the model rather than
# the scheduler, small enough to belong in a test.
THROUGHPUT_REQUESTS = 8
THROUGHPUT_MAX_NEW = 48
THROUGHPUT_POOL_BLOCKS = 128


def _devices() -> list[str]:
    """cpu plus this host's accelerator. See tests/devices.py for why."""
    return model_devices()


@pytest.fixture(scope="module", params=_devices())
def runtime(request):
    from transformers import AutoModelForCausalLM

    from engine.model import Qwen2Engine

    device = request.param
    model = AutoModelForCausalLM.from_pretrained(DEV_MODEL, dtype=torch.float32)
    model.eval()
    model.to(device)
    return device, Qwen2Engine.from_hf_model(model)


@lru_cache(maxsize=None)
def _base_ids() -> tuple[int, ...]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL)
    return tuple(tokenizer(BASE_PROMPT, add_special_tokens=False)["input_ids"])


# --- instrumentation --------------------------------------------------------


class InstrumentedExecutor:
    """Wraps an executor and records what each step actually asked of it.

    Everything the fusion gate needs to be non-vacuous is observed here rather
    than assumed: how many calls a step made, how wide each batch was, how ragged
    it was, and where in the pool the batch declared it would write.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.single_calls = 0
        self.batch_calls = 0
        self.batch_shapes: list[tuple[int, ...]] = []
        self.batch_write_slots: list[frozenset[int]] = []

    def new_backend(self, seq_id: int):
        return self.inner.new_backend(seq_id)

    def reserve(self, backend, num_tokens):
        return self.inner.reserve(backend, num_tokens)

    def forward(self, token_ids, positions, backend):
        self.single_calls += 1
        return self.inner.forward(token_ids, positions, backend)

    def forward_batch(self, inputs: Sequence[BatchInput]):
        self.batch_calls += 1
        self.batch_shapes.append(tuple(len(item.token_ids) for item in inputs))
        self.batch_write_slots.append(_declared_write_slots(inputs))
        return self.inner.forward_batch(inputs)

    @property
    def calls(self) -> int:
        return self.single_calls + self.batch_calls


class LoopingPseudoFusedExecutor:
    """`forward_batch` implemented as a loop. Correct, and not fused.

    Broken in exactly the way no numerical test can see: it runs each sequence
    through the single-sequence path and stacks the results, so its output is not
    close to the unfused reference, it is bit-for-bit the unfused reference. Every
    tolerance in this file passes. Every token comparison passes. The pass counter
    is the only thing that knows.
    """

    def __init__(self, inner) -> None:
        self.inner = inner

    def new_backend(self, seq_id: int):
        return self.inner.new_backend(seq_id)

    def reserve(self, backend, num_tokens):
        return self.inner.reserve(backend, num_tokens)

    def forward(self, token_ids, positions, backend):
        return self.inner.forward(token_ids, positions, backend)

    def forward_batch(self, inputs: Sequence[BatchInput]):
        return torch.stack(
            [
                self.inner.forward(item.token_ids, item.positions, item.backend)
                for item in inputs
            ]
        )


def _declared_write_slots(inputs: Sequence[BatchInput]) -> frozenset[int]:
    """Where this batch says it is going to store new K/V.

    Read through the backends' own translation, which is the definition of "this
    batch's destinations". Whether those destinations are the *right* ones is a
    separate question, settled by
    `test_fused_write_slots_are_unique_and_match_the_block_tables`.
    """
    slots: list[int] = []
    for item in inputs:
        past = item.backend.past_len
        slots.extend(
            int(slot) for slot in item.backend.write_slots(past, past + len(item.token_ids))
        )
    return frozenset(slots)


@dataclass
class Run:
    scheduler: Scheduler
    executor: InstrumentedExecutor
    allocator: BlockAllocator
    cache: PagedKVCache
    outputs: dict[str, tuple[int, ...]]
    steps: list[StepResult]


def _requests():
    ids = _base_ids()
    upfront = [
        Request(
            request_id=f"u{prompt_len}",
            prompt_token_ids=tuple(ids[:prompt_len]),
            max_new_tokens=max_new,
        )
        for prompt_len, max_new in UPFRONT_SHAPES
    ]
    prompt_len, max_new = LATE_SHAPE
    late = Request(
        request_id=f"late{prompt_len}",
        prompt_token_ids=tuple(
            ids[LATE_PROMPT_OFFSET : LATE_PROMPT_OFFSET + prompt_len]
        ),
        max_new_tokens=max_new,
    )
    return upfront, late


def _build(
    engine,
    mode: ExecutionMode,
    *,
    pool_blocks: int = FUSED_POOL_BLOCKS,
    block_size: int = BLOCK_SIZE,
    margin: int = ADMISSION_MARGIN_BLOCKS,
    record_logits: bool = False,
    wrapper=InstrumentedExecutor,
):
    allocator = BlockAllocator(num_blocks=pool_blocks, block_size=block_size)
    cache = engine.new_paged_cache(pool_blocks, block_size)
    executor = wrapper(Qwen2Executor(engine, cache, allocator))
    scheduler = Scheduler(
        executor,
        allocator,
        admission_margin_blocks=margin,
        mode=mode,
        record_logits=record_logits,
    )
    return scheduler, executor, allocator, cache


def _drive(engine, mode: ExecutionMode, *, record_logits: bool = False, wrapper=InstrumentedExecutor):
    """Run the standard request set, injecting the late arrival mid-flight."""
    scheduler, executor, allocator, cache = _build(
        engine, mode, record_logits=record_logits, wrapper=wrapper
    )
    upfront, late = _requests()
    for request in upfront:
        assert scheduler.add_request(request)

    steps: list[StepResult] = []
    while scheduler.waiting or scheduler.running:
        if len(steps) == LATE_ARRIVAL_STEP:
            assert scheduler.add_request(late)
        steps.append(scheduler.step())
        assert len(steps) < 200, "fixture did not converge"

    assert late.request_id in scheduler.finished, "the late arrival never ran"
    return Run(
        scheduler=scheduler,
        executor=executor,
        allocator=allocator,
        cache=cache,
        outputs={rid: tuple(seq.generated) for rid, seq in scheduler.finished.items()},
        steps=steps,
    )


def _max_abs_deviation(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _stacked_logits(scheduler: Scheduler, request_id: str) -> torch.Tensor:
    seq = scheduler.finished[request_id]
    assert len(seq.step_logits) == len(seq.generated), (
        f"{request_id}: {len(seq.step_logits)} logit rows for "
        f"{len(seq.generated)} tokens; the two fell out of alignment"
    )
    return torch.stack(seq.step_logits)


# --- the mask and the addressing, with no model involved --------------------


def _tiny_pool(num_blocks: int, block_size: int):
    cache = PagedKVCache(
        PagedCacheSpec(
            num_layers=1,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=1,
            head_dim=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )
    return cache, BlockAllocator(num_blocks=num_blocks, block_size=block_size)


def _prepared(cache, allocator, seq_id: int, past: int, new: int) -> PagedBackend:
    """A backend holding `past` committed tokens with `new` more just reserved."""
    allocator.create_sequence(seq_id)
    backend = PagedBackend(cache, allocator, seq_id, num_kv_groups=1, scale=1.0)
    if past:
        assert backend.begin_forward(past).ok
        backend.end_forward(past)
    assert backend.begin_forward(new).ok
    assert backend.past_len == past
    return backend


def _batch_inputs(backends, shapes) -> tuple[BatchInput, ...]:
    return tuple(
        BatchInput(
            token_ids=tuple(range(1, new + 1)),
            positions=tuple(range(past, past + new)),
            backend=backend,
        )
        for backend, (past, new) in zip(backends, shapes)
    )


# A prefill from empty, a decode deep into a sequence, a short prefill, and a
# decode near a block boundary. Every combination of "is padded" x "is a prefill".
MASK_SHAPES = ((0, 9), (14, 1), (0, 3), (7, 1))
MASK_BLOCK_SIZE = 4
MASK_NUM_BLOCKS = 32


def test_fused_mask_matches_an_independent_reference():
    """The mask, checked against causality restated as a double loop.

    The implementation builds the mask with broadcast comparisons. The reference
    here is written the other way -- one Python loop per row, straight from the
    definition -- so an off-by-one in the vectorised form has nowhere to hide. No
    model, no floating point.
    """
    cache, allocator = _tiny_pool(MASK_NUM_BLOCKS, MASK_BLOCK_SIZE)
    backends = [
        _prepared(cache, allocator, index, past, new)
        for index, (past, new) in enumerate(MASK_SHAPES)
    ]
    fused = FusedPagedAttention(
        _batch_inputs(backends, MASK_SHAPES), num_kv_groups=1, scale=1.0
    )

    max_query = max(new for _, new in MASK_SHAPES)
    max_kv = max(past + new for past, new in MASK_SHAPES)
    assert fused.max_query_len == max_query
    assert fused.max_kv_len == max_kv
    assert fused.mask.shape == (len(MASK_SHAPES), 1, max_query, max_kv)

    print(f"\nmask fixture {MASK_SHAPES}: {len(MASK_SHAPES)}x{max_query}x{max_kv}, "
          f"{fused.num_real_tokens} real / {fused.num_padded_tokens} padded query slots")

    for row, (past, new) in enumerate(MASK_SHAPES):
        kv_len = past + new
        for query in range(max_query):
            for key in range(max_kv):
                if query < new:
                    # A real query token sits at logical position past + query and
                    # may look at everything up to and including itself.
                    want = key <= past + query
                else:
                    # A padded row is discarded, but it must still be allowed one
                    # key: a fully masked row makes softmax produce NaN.
                    want = key == 0
                got = bool(fused.mask[row, 0, query, key])
                assert got == want, (
                    f"mask[{row}, 0, {query}, {key}] is {got}, expected {want} "
                    f"(past={past}, new={new}, kv_len={kv_len})"
                )

    # Real rows must be genuinely restrictive, or the loop above proves nothing.
    assert not bool(fused.mask.all()), "the mask admits everything"
    assert bool(fused.mask.any(dim=-1).all()), "some row is masked to nothing"


def test_fused_padded_rows_do_not_leave_a_row_fully_masked():
    """The specific NaN this guards, made explicit.

    A batch whose widest member is much wider than the rest is where padding is
    at its most extreme. Every row of the mask -- real or pad -- must admit at
    least one key.
    """
    shapes = ((0, 40), (33, 1), (5, 1))
    cache, allocator = _tiny_pool(MASK_NUM_BLOCKS, MASK_BLOCK_SIZE)
    backends = [
        _prepared(cache, allocator, index, past, new)
        for index, (past, new) in enumerate(shapes)
    ]
    fused = FusedPagedAttention(_batch_inputs(backends, shapes), num_kv_groups=1, scale=1.0)

    admitted = fused.mask.sum(dim=-1)
    print(f"\nkeys admitted per row, worst case: min={int(admitted.min())} "
          f"max={int(admitted.max())} over {tuple(fused.mask.shape)}")
    assert int(admitted.min()) >= 1


def _fragment_pool(allocator: BlockAllocator, num_blocks: int) -> None:
    """Consume the pool with decoys and release them from both ends inward.

    Same construction as tests/test_parity.py: freed blocks go to the back of the
    free list, so the next sequence receives blocks that descend and are never
    adjacent.
    """
    decoy_base = 1_000_000
    for index in range(num_blocks):
        allocator.create_sequence(decoy_base + index)
        assert allocator.append_tokens(decoy_base + index, 1).ok

    release_order: list[int] = []
    low, high = 0, num_blocks - 1
    while low <= high:
        release_order.append(high)
        if low != high:
            release_order.append(low)
        low += 1
        high -= 1
    for block_id in release_order:
        allocator.free_sequence(decoy_base + block_id)
    assert allocator.free_block_ids() == tuple(release_order)


def test_fused_write_slots_are_unique_and_match_the_block_tables():
    """Addressing, on a fragmented pool where reading straight through is wrong.

    Two properties. Every destination the batch names is the one the allocator
    says that token belongs in -- computed here from `physical_slot`, which knows
    nothing about the fused path -- and no two tokens in the batch name the same
    destination, which is what makes the scatter safe.
    """
    shapes = ((0, 9), (14, 1), (0, 17))
    cache, allocator = _tiny_pool(MASK_NUM_BLOCKS, MASK_BLOCK_SIZE)
    _fragment_pool(allocator, MASK_NUM_BLOCKS)

    backends = [
        _prepared(cache, allocator, index, past, new)
        for index, (past, new) in enumerate(shapes)
    ]
    tables = [allocator.block_table(index) for index in range(len(shapes))]
    print(f"\nfragmented block tables: {tables}")
    assert any(
        table[i + 1] < table[i] for table in tables for i in range(len(table) - 1)
    ), "no block table descends: the pool was not fragmented and this is vacuous"

    fused = FusedPagedAttention(_batch_inputs(backends, shapes), num_kv_groups=1, scale=1.0)

    expected: list[int] = []
    for seq_id, (past, new) in enumerate(shapes):
        for position in range(past, past + new):
            block_id, offset = allocator.physical_slot(seq_id, position)
            expected.append(block_id * MASK_BLOCK_SIZE + offset)

    observed = [int(slot) for slot in fused.write_slots]
    assert observed == expected, (
        f"write slots {observed} do not match the block tables' {expected}"
    )
    assert len(set(observed)) == len(observed), (
        "two tokens in one batch would be scattered to the same slot"
    )

    # The read side must name the same slots for the positions it re-reads.
    read = fused.read_slots.view(len(shapes), fused.max_kv_len)
    for seq_id, (past, new) in enumerate(shapes):
        for position in range(past + new):
            block_id, offset = allocator.physical_slot(seq_id, position)
            assert int(read[seq_id, position]) == block_id * MASK_BLOCK_SIZE + offset


def test_fused_batch_rejects_a_split_pool():
    """Two pools in one batch would gather from the wrong tensor."""
    cache_a, allocator_a = _tiny_pool(8, MASK_BLOCK_SIZE)
    cache_b, allocator_b = _tiny_pool(8, MASK_BLOCK_SIZE)
    backend_a = _prepared(cache_a, allocator_a, 0, 0, 3)
    backend_b = _prepared(cache_b, allocator_b, 0, 0, 3)

    with pytest.raises(ValueError, match="one block pool"):
        FusedPagedAttention(
            _batch_inputs([backend_a, backend_b], ((0, 3), (0, 3))),
            num_kv_groups=1,
            scale=1.0,
        )


def test_fused_batch_rejects_non_contiguous_positions():
    """Positions that do not match the cache offset would desync mask and rope."""
    cache, allocator = _tiny_pool(8, MASK_BLOCK_SIZE)
    backend = _prepared(cache, allocator, seq_id=0, past=4, new=1)
    bad = (BatchInput(token_ids=(7,), positions=(99,), backend=backend),)

    with pytest.raises(NotImplementedError, match="contiguous run"):
        FusedPagedAttention(bad, num_kv_groups=1, scale=1.0)


# --- the model tier ---------------------------------------------------------


def test_fused_matches_unfused_on_the_model(runtime):
    """THE Step 5b numerical gate. Tokens exactly equal, logits within tolerance.

    Reported per request, so the three prompt lengths of PROMPT_LENGTHS each get
    their own deviation on each device. The pool is ample, so neither mode
    preempts and both take the same scheduling decisions -- asserted below, which
    is what makes this a comparison of arithmetic rather than of two different
    runs that happen to agree.
    """
    device, engine = runtime
    torch.manual_seed(SEED)

    unfused = _drive(engine, ExecutionMode.UNFUSED, record_logits=True)
    fused = _drive(engine, ExecutionMode.FUSED, record_logits=True)

    widths_unfused = [result.batch_size for result in unfused.steps]
    widths_fused = [result.batch_size for result in fused.steps]
    shapes = fused.executor.batch_shapes
    real = sum(sum(shape) for shape in shapes)
    padded = sum(len(shape) * max(shape) for shape in shapes) - real

    print(f"\n[{device}] fused vs unfused, {len(fused.outputs)} requests, "
          f"pool {FUSED_POOL_BLOCKS} blocks of {BLOCK_SIZE}")
    print(f"[{device}]   steps                 : {len(widths_fused)} "
          f"(unfused {len(widths_unfused)})")
    print(f"[{device}]   batch widths per step : {widths_fused}")
    print(f"[{device}]   query lengths per step: {[list(s) for s in shapes]}")
    print(f"[{device}]   query slots           : {real} real, {padded} padded "
          f"({padded / (real + padded):.1%} waste, no bucketing)")
    print(f"[{device}]   tolerance (FUSED_LOGIT_ATOL): {FUSED_LOGIT_ATOL:.1e}")

    # The fixture has to exercise what the tolerance is for.
    assert widths_fused == widths_unfused, (
        "the two modes took different scheduling decisions, so this is not an "
        f"aligned comparison: {widths_fused} vs {widths_unfused}"
    )
    assert not fused.scheduler.events_of(EventKind.PREEMPT), "pool was meant to be ample"
    assert max(widths_fused) > 1, "no step ever batched more than one sequence"
    assert len(set(widths_fused)) > 1, "the running-set size never changed"
    assert any(len(set(shape)) > 1 for shape in shapes), (
        "no batch was ragged, so the query axis was never padded and padding is untested"
    )
    assert any(min(shape) == 1 and max(shape) > 1 for shape in shapes), (
        "no step mixed a multi-token prefill with a single-token decode"
    )
    assert padded > 0

    worst = 0.0
    for request_id in sorted(unfused.outputs):
        unfused_logits = _stacked_logits(unfused.scheduler, request_id)
        fused_logits = _stacked_logits(fused.scheduler, request_id)
        deviation = _max_abs_deviation(fused_logits, unfused_logits)
        worst = max(worst, deviation)
        prompt_len = unfused.scheduler.finished[request_id].request.prompt_len
        print(
            f"[{device}]   {request_id:<8} prompt_len={prompt_len:<3} "
            f"tokens identical={fused.outputs[request_id] == unfused.outputs[request_id]}  "
            f"MAX ABS DEVIATION={deviation:.6e}"
        )

        assert not torch.isnan(fused_logits).any(), f"{request_id}: fused logits contain NaN"
        assert fused.outputs[request_id] == unfused.outputs[request_id], (
            f"[{device}] {request_id} token mismatch at prompt_len={prompt_len}: "
            f"fused {fused.outputs[request_id]} vs unfused {unfused.outputs[request_id]}"
        )
        assert deviation <= FUSED_LOGIT_ATOL, (
            f"[{device}] {request_id} max abs logit deviation {deviation:.6e} "
            f"exceeds FUSED_LOGIT_ATOL {FUSED_LOGIT_ATOL:.1e}"
        )

    print(f"[{device}]   WORST OVER ALL REQUESTS: {worst:.6e}")

    covered = {
        unfused.scheduler.finished[rid].request.prompt_len for rid in unfused.outputs
    }
    assert set(PROMPT_LENGTHS) <= covered, (
        f"prompt lengths {sorted(set(PROMPT_LENGTHS) - covered)} were not reported"
    )


def test_fused_batch_of_one_matches_unfused(runtime):
    """The mask alone, with no padding in play.

    A batch of one has max_query_len == its own length and max_kv_len == its own
    kv length, so nothing is padded and the *only* difference from the unfused
    path is that attention is masked explicitly instead of taking SDPA's
    `is_causal` shortcut. If the mask were wrong by a position this would diverge
    with padding out of the picture.
    """
    device, engine = runtime
    torch.manual_seed(SEED)
    upfront, _ = _requests()

    worst = 0.0
    print(f"\n[{device}] batch-of-one: explicit mask vs is_causal, no padding")
    for request in upfront:
        results = {}
        for mode in (ExecutionMode.UNFUSED, ExecutionMode.FUSED):
            scheduler, executor, _, _ = _build(engine, mode, record_logits=True)
            assert scheduler.add_request(request)
            scheduler.run_to_completion()
            results[mode] = scheduler
            if mode is ExecutionMode.FUSED:
                assert executor.batch_calls > 0
                assert all(len(shape) == 1 for shape in executor.batch_shapes)

        request_id = request.request_id
        unfused_logits = _stacked_logits(results[ExecutionMode.UNFUSED], request_id)
        fused_logits = _stacked_logits(results[ExecutionMode.FUSED], request_id)
        deviation = _max_abs_deviation(fused_logits, unfused_logits)
        worst = max(worst, deviation)
        unfused_tokens = tuple(results[ExecutionMode.UNFUSED].finished[request_id].generated)
        fused_tokens = tuple(results[ExecutionMode.FUSED].finished[request_id].generated)

        print(f"[{device}]   {request_id:<8} tokens identical={fused_tokens == unfused_tokens}"
              f"  MAX ABS DEVIATION={deviation:.6e}")
        assert fused_tokens == unfused_tokens, (
            f"[{device}] {request_id}: an unpadded fused batch of one diverged from "
            f"the unfused path, which points at the mask: {fused_tokens} vs {unfused_tokens}"
        )
        assert deviation <= FUSED_LOGIT_ATOL

    print(f"[{device}]   WORST: {worst:.6e}")


def test_fused_writes_nothing_outside_the_batchs_own_slots(runtime):
    """Padded rows are computed and discarded; they must not exist in the pool.

    A pad row goes through every projection and every norm exactly as a real one
    does. If the scatter stored it, it would land in whatever slot the padded
    position translated to -- which is some other sequence's block. Snapshot the
    whole pool before every step and assert that the only slots that moved are the
    ones the batch declared.
    """
    device, engine = runtime
    torch.manual_seed(SEED)

    scheduler, executor, allocator, cache = _build(engine, ExecutionMode.FUSED)
    upfront, late = _requests()
    for request in upfront:
        assert scheduler.add_request(request)

    num_layers = engine.dims.num_layers
    num_slots = cache.spec.num_slots
    checked = 0
    padded_steps = 0
    steps = 0

    while scheduler.waiting or scheduler.running:
        if steps == LATE_ARRIVAL_STEP:
            assert scheduler.add_request(late)
        before = [
            (cache.flat_keys(layer).clone(), cache.flat_values(layer).clone())
            for layer in range(num_layers)
        ]
        calls_before = executor.batch_calls
        scheduler.step()
        steps += 1
        assert steps < 200

        if executor.batch_calls == calls_before:
            continue  # nothing ran, nothing to check
        written = executor.batch_write_slots[-1]
        shape = executor.batch_shapes[-1]
        if len(set(shape)) > 1:
            padded_steps += 1

        untouched = torch.ones(num_slots, dtype=torch.bool)
        untouched[torch.tensor(sorted(written), dtype=torch.long)] = False
        for layer in range(num_layers):
            keys_before, values_before = before[layer]
            assert torch.equal(
                cache.flat_keys(layer)[untouched], keys_before[untouched]
            ), f"step {steps}, layer {layer}: a key slot outside the batch changed"
            assert torch.equal(
                cache.flat_values(layer)[untouched], values_before[untouched]
            ), f"step {steps}, layer {layer}: a value slot outside the batch changed"
        # And the batch did write where it said it would, or the check is vacuous.
        assert not torch.equal(
            cache.flat_keys(0)[~untouched], before[0][0][~untouched]
        ), f"step {steps}: the batch's own slots did not change"
        checked += 1

    print(f"\n[{device}] pool containment: {checked} steps checked across "
          f"{num_layers} layers, {padded_steps} of them with a ragged batch")
    assert checked >= 2
    assert padded_steps >= 1, "no ragged batch occurred, so padding was not exercised"
    assert allocator.num_free == FUSED_POOL_BLOCKS
    assert allocator.num_allocated == 0


def test_fused_recompute_matches_unfused(runtime):
    """Step 5's gate carried into the fused path: recompute is still transparent.

    A preempted request replays its whole history in one multi-token prefill. In
    fused mode that prefill also shares a batch with other sequences' decodes, so
    the replay is padded as well as reordered. The tokens must not move.
    """
    device, engine = runtime
    torch.manual_seed(SEED)
    ids = _base_ids()

    target = Request(
        request_id="B",
        prompt_token_ids=tuple(ids[:RECOMPUTE_PROMPT_LEN]),
        max_new_tokens=RECOMPUTE_MAX_NEW,
    )
    filler = Request(
        request_id="filler",
        prompt_token_ids=tuple(ids[:RECOMPUTE_FILLER_PROMPT_LEN]),
        max_new_tokens=RECOMPUTE_FILLER_MAX_NEW,
    )

    results = {}
    for mode in (ExecutionMode.UNFUSED, ExecutionMode.FUSED):
        scheduler, _, allocator, _ = _build(
            engine,
            mode,
            pool_blocks=RECOMPUTE_POOL_BLOCKS,
            block_size=RECOMPUTE_BLOCK_SIZE,
            record_logits=True,
        )
        assert scheduler.add_request(filler)
        assert scheduler.add_request(target)
        scheduler.run_to_completion()
        assert allocator.num_free == RECOMPUTE_POOL_BLOCKS
        results[mode] = scheduler

        preempts = scheduler.events_of(EventKind.PREEMPT)
        recomputes = scheduler.events_of(EventKind.RECOMPUTE)
        print(f"\n[{device}] {mode.value}: {len(preempts)} preemptions, "
              f"{len(recomputes)} recomputes, "
              f"B preempted {scheduler.finished['B'].preemption_count}x")
        assert preempts, f"[{mode.value}] the pool was not tight enough to preempt"
        assert recomputes, f"[{mode.value}] nothing was ever replayed"

    for request_id in ("B", "filler"):
        unfused_logits = _stacked_logits(results[ExecutionMode.UNFUSED], request_id)
        fused_logits = _stacked_logits(results[ExecutionMode.FUSED], request_id)
        deviation = _max_abs_deviation(fused_logits, unfused_logits)
        unfused_tokens = tuple(results[ExecutionMode.UNFUSED].finished[request_id].generated)
        fused_tokens = tuple(results[ExecutionMode.FUSED].finished[request_id].generated)

        print(f"[{device}]   {request_id:<8} tokens identical={fused_tokens == unfused_tokens}"
              f"  MAX ABS DEVIATION={deviation:.6e}")
        assert fused_tokens == unfused_tokens, (
            f"[{device}] {request_id}: fused recompute changed the answer: "
            f"{fused_tokens} vs {unfused_tokens}"
        )
        assert deviation <= FUSED_LOGIT_ATOL


# --- THE NEGATIVE CONTROL ---------------------------------------------------


def _forward_passes_per_step(engine, run_mode: ExecutionMode, wrapper) -> list[tuple[int, int]]:
    """(sequences that ran, layer-stack traversals) for every step of a run."""
    scheduler, executor, _, _ = _build(engine, run_mode, wrapper=wrapper)
    upfront, late = _requests()
    for request in upfront:
        assert scheduler.add_request(request)

    observed: list[tuple[int, int]] = []
    steps = 0
    while scheduler.waiting or scheduler.running:
        if steps == LATE_ARRIVAL_STEP:
            assert scheduler.add_request(late)
        before = engine.forward_pass_count
        result = scheduler.step()
        steps += 1
        assert steps < 200
        observed.append((result.batch_size, engine.forward_pass_count - before))
    return observed


def _assert_one_pass_per_step(observed: list[tuple[int, int]]) -> None:
    """The fusion assertion: one traversal of the model per step, whatever the width.

    Used by the positive test and, inside `pytest.raises`, by the impostor below.
    A single assertion shared by both is the only way to show the control has
    teeth rather than merely asserting something true.
    """
    assert observed, "no steps ran"
    for index, (width, passes) in enumerate(observed, start=1):
        assert width >= 1, f"step {index} ran nothing"
        assert passes == 1, (
            f"step {index} ran {width} sequences in {passes} forward passes; "
            "fusion means one pass per step regardless of running-set size"
        )


def test_fused_model_makes_exactly_one_forward_pass_per_step(runtime):
    """REQUIRED NEGATIVE CONTROL. One pass per step, whatever the running set.

    `forward_pass_count` is incremented once per traversal of the layer stack, so
    it cannot be satisfied by naming: a `forward_batch` that loops raises it once
    per sequence. Counted here across a run whose running set is 4, then 5 as the
    late request joins, then 4, 3 and 2 as budgets run out -- which is what makes
    "regardless of running-set size" a measurement and not a phrase.
    """
    device, engine = runtime
    torch.manual_seed(SEED)

    observed = _forward_passes_per_step(engine, ExecutionMode.FUSED, InstrumentedExecutor)
    widths = [width for width, _ in observed]
    passes = [count for _, count in observed]

    print(f"\n[{device}] fused: running-set width per step {widths}")
    print(f"[{device}] fused: forward passes per step     {passes}")
    print(f"[{device}] total: {sum(widths)} sequence-steps in {sum(passes)} forward passes")

    _assert_one_pass_per_step(observed)
    assert max(widths) > 1, "the running set was never larger than one"
    assert len(set(widths)) > 1, "the running-set size never varied"
    assert sum(passes) < sum(widths), (
        "fusion saved no forward passes at all, which cannot be right if any step "
        "ran more than one sequence"
    )


def test_fused_model_makes_one_pass_per_step_under_preemption(runtime):
    """One pass per step still, on a step that preempts AND replays at once.

    A recompute re-prefills a whole history, which is a forward. The scheduler
    routes it through `_submit` like any decode, so in fused mode it joins the
    step's batch -- but that is a claim about the scheduler, and this checks the
    consequence at the model, where `forward_pass_count` counts traversals of the
    layer stack and cannot be satisfied by naming.

    The fixture is the shape
    `tests/test_sched.py::test_recompute_folds_into_the_same_batched_call` found:
    twelve blocks of four, no margin, five requests that each grow to six blocks.
    It thrashes, and one of its steps readmits two preempted sequences and then
    evicts one of them again while planning the decodes. Same shape here, real
    tokens, so the schedule is the same -- admission and eviction read token
    counts, never token values.
    """
    device, engine = runtime
    torch.manual_seed(SEED)
    ids = _base_ids()

    scheduler, _, allocator, _ = _build(
        engine,
        ExecutionMode.FUSED,
        pool_blocks=CO_POOL_BLOCKS,
        block_size=CO_BLOCK_SIZE,
        margin=CO_MARGIN,
    )
    # Distinct slices, so the five requests are different sequences of the same
    # length rather than five copies of one.
    for index in range(CO_REQUESTS):
        assert scheduler.add_request(
            Request(
                request_id=f"r{index}",
                prompt_token_ids=tuple(ids[index : index + CO_PROMPT_LEN]),
                max_new_tokens=CO_MAX_NEW,
            )
        )

    observed: list[tuple[int, int]] = []
    replay_steps: list[int] = []
    preempt_steps: list[int] = []
    both_steps: list[int] = []
    while scheduler.waiting or scheduler.running:
        before_passes = engine.forward_pass_count
        before_events = len(scheduler.events)
        result = scheduler.step()
        raised = [event.kind for event in scheduler.events[before_events:]]
        if EventKind.RECOMPUTE in raised:
            replay_steps.append(result.step)
        if EventKind.PREEMPT in raised:
            preempt_steps.append(result.step)
        if EventKind.RECOMPUTE in raised and EventKind.PREEMPT in raised:
            both_steps.append(result.step)
        observed.append((result.batch_size, engine.forward_pass_count - before_passes))
        assert result.step < 400, "fixture did not converge"

    print(f"\n[{device}] thrashing fixture: {len(observed)} steps, "
          f"preempt on {preempt_steps}, recompute on {replay_steps}")
    print(f"[{device}] steps with BOTH: {both_steps}")
    for step in both_steps:
        width, passes = observed[step - 1]
        print(f"[{device}]   step {step}: width={width}, forward passes={passes}")

    assert preempt_steps, "the pool was not tight enough to preempt"
    assert replay_steps, "nothing was ever replayed, so the question is untested"
    assert both_steps, (
        "no step contained both a preempt and a recompute, so this fixture "
        "cannot answer the question it exists to answer"
    )
    _assert_one_pass_per_step(observed)

    # The co-occurrence step must carry a replay prefill *alongside* other work,
    # or "it folded in" would be a claim about a batch of one.
    for step in both_steps:
        width, passes = observed[step - 1]
        assert passes == 1, f"step {step} cost {passes} passes"
        assert width >= 2, (
            f"step {step} batched only {width} sequence, so the replay did not "
            "actually share a call with anything"
        )
    assert allocator.num_free == CO_POOL_BLOCKS


def test_unfused_makes_one_forward_pass_per_sequence(runtime):
    """The counter's calibration: the reference mode fails the fused assertion.

    Without this, `passes == 1` might just be what the counter always reports.
    """
    device, engine = runtime
    torch.manual_seed(SEED)

    observed = _forward_passes_per_step(engine, ExecutionMode.UNFUSED, InstrumentedExecutor)
    print(f"\n[{device}] unfused: (width, passes) per step {observed}")

    for width, passes in observed:
        assert passes == width, "the unfused path is not one pass per sequence"
    with pytest.raises(AssertionError, match="fusion means one pass per step"):
        _assert_one_pass_per_step(observed)


def test_a_looping_pseudo_fusion_is_numerically_perfect_and_still_caught(runtime):
    """REQUIRED NEGATIVE CONTROL, the half with teeth.

    `LoopingPseudoFusedExecutor` implements `forward_batch` as a loop over the
    single-sequence path. Its output is not close to the unfused reference, it is
    byte-for-byte the unfused reference -- asserted below -- so every tolerance,
    every token comparison and every logit deviation in this file passes for it.
    It is a loop with better naming and nothing numerical can say so.

    The same assertion the real fused path satisfies fails for it. That is the
    whole content of the control: the test discriminates.
    """
    device, engine = runtime
    torch.manual_seed(SEED)

    honest = _drive(engine, ExecutionMode.UNFUSED)
    impostor = _drive(engine, ExecutionMode.FUSED, wrapper=LoopingPseudoFusedExecutor)

    print(f"\n[{device}] looping pseudo-fusion: byte-identical to unfused = "
          f"{impostor.outputs == honest.outputs}")

    # It passes the numerical gate outright -- exactly, not within tolerance.
    assert impostor.outputs == honest.outputs, (
        "the impostor was expected to reproduce the unfused path exactly; if it "
        "does not, this test is no longer demonstrating an undetectable-by-numbers "
        "failure"
    )

    observed = _forward_passes_per_step(
        engine, ExecutionMode.FUSED, LoopingPseudoFusedExecutor
    )
    print(f"[{device}] looping pseudo-fusion: (width, passes) per step {observed}")

    for width, passes in observed:
        assert passes == width, (
            "the impostor did not loop, so it is not the thing this test needs "
            "to catch"
        )
    with pytest.raises(AssertionError, match="fusion means one pass per step"):
        _assert_one_pass_per_step(observed)


def test_fused_tolerance_is_declared_here():
    """Runs everywhere: proves the tolerance above is this file's own.

    A test file that quietly imported LOGIT_ATOL and called it a fused tolerance
    would look identical from the outside to one that derived its own.
    """
    assert FUSED_LOGIT_ATOL == 5e-3
    assert "LOGIT_ATOL" not in globals()
    assert "FP16_LOGIT_ATOL" not in globals()

    # Distinct from both of the other declared tolerances, checked against the
    # files that own them rather than against remembered literals.
    import tests.test_parity as parity
    import tests.test_parity_fp16 as parity_fp16

    assert FUSED_LOGIT_ATOL != parity.LOGIT_ATOL
    assert FUSED_LOGIT_ATOL != parity_fp16.FP16_LOGIT_ATOL


# --- throughput -------------------------------------------------------------


def _throughput_requests():
    """Requests long enough that the measurement is the model, not the scheduler."""
    ids = _base_ids()
    repeated = ids * 3
    return [
        Request(
            request_id=f"t{index}",
            prompt_token_ids=tuple(repeated[index : index + len(ids)]),
            max_new_tokens=THROUGHPUT_MAX_NEW,
        )
        for index in range(THROUGHPUT_REQUESTS)
    ]


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


@pytest.mark.slow
def test_fused_throughput_against_unfused(runtime):
    """DIAGNOSTIC ONLY. This engine against its own sequential path.

    NOT THE GATE 1 NUMBER, and nothing may quote it as one. Gate 1 is this engine
    against naive HuggingFace `generate`, measured on the benchmark host. What is
    below is this engine's fused mode against this engine's unfused mode, on
    whatever laptop happens to be running the tests -- it says fusion did
    something, and it says nothing at all about the gate. The two are different
    comparisons against different baselines on different hardware, and the ratio
    printed here is not comparable to either.

    No assertion on the ratio: a speedup is hardware, not correctness, so nothing
    here fails if the machine is slow or busy. What is asserted is that both modes
    produced the same tokens, which is what makes the two timings comparable to
    each other at all, and that fused mode really did run one pass per step.
    """
    import time

    device, engine = runtime
    torch.manual_seed(SEED)
    requests = _throughput_requests()
    total_tokens = sum(request.max_new_tokens for request in requests)

    timings: dict[ExecutionMode, tuple[float, int, int]] = {}
    outputs: dict[ExecutionMode, dict[str, tuple[int, ...]]] = {}

    for mode in (ExecutionMode.UNFUSED, ExecutionMode.FUSED):
        # Warm up kernels and any lazy device init, off the clock.
        warmup, _, _, _ = _build(
            engine, mode, pool_blocks=THROUGHPUT_POOL_BLOCKS
        )
        assert warmup.add_request(
            Request("warmup", requests[0].prompt_token_ids, max_new_tokens=2)
        )
        warmup.run_to_completion()
        _sync(device)

        scheduler, _, _, _ = _build(engine, mode, pool_blocks=THROUGHPUT_POOL_BLOCKS)
        for request in requests:
            assert scheduler.add_request(request)

        passes_before = engine.forward_pass_count
        _sync(device)
        started = time.perf_counter()
        produced = scheduler.run_to_completion()
        _sync(device)
        elapsed = time.perf_counter() - started

        steps = scheduler.events[-1].step
        timings[mode] = (elapsed, engine.forward_pass_count - passes_before, steps)
        outputs[mode] = produced

    print(f"\n[{device}] DIAGNOSTIC (fused vs this engine's own unfused path -- "
          f"NOT the Gate 1 number, which is vs naive HF generate)")
    print(f"[{device}] {THROUGHPUT_REQUESTS} requests x "
          f"{THROUGHPUT_MAX_NEW} new tokens = {total_tokens} tokens, "
          f"prompt_len={len(requests[0].prompt_token_ids)}")
    for mode in (ExecutionMode.UNFUSED, ExecutionMode.FUSED):
        elapsed, passes, steps = timings[mode]
        print(f"[{device}]   {mode.value:<8} {elapsed:7.2f}s  "
              f"{total_tokens / elapsed:8.2f} tok/s  "
              f"{steps} steps, {passes} forward passes")
    unfused_elapsed = timings[ExecutionMode.UNFUSED][0]
    fused_elapsed = timings[ExecutionMode.FUSED][0]
    print(f"[{device}]   diagnostic ratio  {unfused_elapsed / fused_elapsed:.2f}x "
          f"(self-vs-self, not a result)")

    assert outputs[ExecutionMode.FUSED] == outputs[ExecutionMode.UNFUSED], (
        "the two modes produced different tokens, so the timings are not comparable"
    )
    assert timings[ExecutionMode.FUSED][1] == timings[ExecutionMode.FUSED][2], (
        "fused mode did not run exactly one forward pass per step"
    )
