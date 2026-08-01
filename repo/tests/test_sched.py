"""Step 4: continuous batching scheduler, and Step 5b's second execution mode.

The gate is equality: running N requests concurrently must produce byte-identical
tokens to running each one alone. Two tiers of test assert it.

The stub tier drives the real BlockAllocator and real PagedBackend against a
one-element KV pool, with a "model" whose next token is a pure function of the
sequence's own last token and position. Block accounting is therefore real while
the arithmetic is exactly reproducible, so any concurrent-vs-sequential
difference is a scheduling bug and cannot be floating point. That also makes the
stub exact under recompute, which lets the forced-preemption case assert equality
rather than a tolerance. These run in milliseconds.

The model tier runs the same comparison through Qwen2.5-0.5B on both devices.

`test_oversized_request_cannot_wedge_the_queue` is the admission guard's reason
for existing: without it that request livelocks the scheduler at zero throughput.

EXECUTION MODES. Step 5b added `ExecutionMode.FUSED`, which runs a step in one
batched forward instead of one forward per sequence. Because the stub's
arithmetic is exact, every stub-tier test here is parametrized over both modes:
the scheduler's decisions must not depend on how the forwards are issued, and
byte-identity must survive fusion where the arithmetic itself is exact.

The model-tier byte-identity tests are pinned to UNFUSED on purpose. Fusion pads
the batch and changes every reduction's shape, so on real floating point it
agrees with the reference to a declared tolerance rather than exactly. That
comparison lives in `tests/test_fused.py`, where the tolerance is declared. This
file keeps the exact claim against the exact path.
"""

import time
from functools import lru_cache

import pytest
import torch

from engine.attention import PagedBackend
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
    SequenceStatus,
)

STUB_VOCAB = 64
SEED = 0

MODES = (ExecutionMode.UNFUSED, ExecutionMode.FUSED)


class StubExecutor:
    """Real block accounting, deterministic sequence-local logits.

    The next token depends only on this sequence's own last token and position,
    never on what else is running. Concurrency-independence is therefore a
    property of the stub, so the equality tests below isolate the scheduler.

    `forward_batch` is a Python loop here, and that is fine: this tier exists to
    test the *scheduler*, and what it proves about fusion is only that a step
    issues one call. Whether that call is really one forward pass is a claim about
    the model, and `tests/test_fused.py` counts the engine's own passes to settle
    it.
    """

    def __init__(self, allocator: BlockAllocator) -> None:
        self.allocator = allocator
        self.cache = PagedKVCache(
            PagedCacheSpec(
                num_layers=1,
                num_blocks=allocator.num_blocks,
                block_size=allocator.block_size,
                num_kv_heads=1,
                head_dim=1,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        )
        self.forward_calls = 0
        self.batch_calls = 0

    def new_backend(self, seq_id: int) -> PagedBackend:
        self.allocator.create_sequence(seq_id)
        return PagedBackend(self.cache, self.allocator, seq_id, num_kv_groups=1, scale=1.0)

    def reserve(self, backend, num_tokens):
        reservation = backend.begin_forward(num_tokens)
        if not reservation.ok:
            raise RuntimeError(
                f"stub could not reserve {num_tokens} tokens: "
                f"{reservation.status.value}; the scheduler should have checked first"
            )
        return reservation

    def _logits(self, token_ids, positions) -> torch.Tensor:
        next_token = (token_ids[-1] * 7 + positions[-1] * 13 + 1) % STUB_VOCAB
        logits = torch.full((STUB_VOCAB,), -1.0)
        logits[next_token] = 1.0
        return logits

    def forward(self, token_ids, positions, backend) -> torch.Tensor:
        self.forward_calls += 1
        backend.end_forward(len(token_ids))
        return self._logits(token_ids, positions)

    def forward_batch(self, inputs) -> torch.Tensor:
        self.batch_calls += 1
        rows = []
        for item in inputs:
            item.backend.end_forward(len(item.token_ids))
            rows.append(self._logits(item.token_ids, item.positions))
        return torch.stack(rows)

    @property
    def calls(self) -> int:
        return self.forward_calls + self.batch_calls


def _stub_scheduler(
    num_blocks: int,
    margin: int = ADMISSION_MARGIN_BLOCKS,
    block_size: int = BLOCK_SIZE,
    mode: ExecutionMode = ExecutionMode.UNFUSED,
):
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    executor = StubExecutor(allocator)
    return (
        Scheduler(executor, allocator, admission_margin_blocks=margin, mode=mode),
        allocator,
    )


def _request(request_id: str, prompt_len: int, max_new_tokens: int) -> Request:
    # Prompt content derived from the id so different requests differ, and
    # deterministically so the fixture is reproducible.
    seed = sum(ord(character) for character in request_id)
    prompt = tuple((seed + index * 3) % STUB_VOCAB for index in range(prompt_len))
    return Request(
        request_id=request_id, prompt_token_ids=prompt, max_new_tokens=max_new_tokens
    )


def _run_sequentially(
    requests,
    num_blocks: int,
    margin: int = ADMISSION_MARGIN_BLOCKS,
    mode: ExecutionMode = ExecutionMode.UNFUSED,
):
    """Each request alone, on a fresh scheduler and a fresh pool."""
    outputs = {}
    for request in requests:
        scheduler, _ = _stub_scheduler(num_blocks, margin, mode=mode)
        assert scheduler.add_request(request)
        outputs.update(scheduler.run_to_completion())
    return outputs


def _run_concurrently(
    requests,
    num_blocks: int,
    margin: int = ADMISSION_MARGIN_BLOCKS,
    mode: ExecutionMode = ExecutionMode.UNFUSED,
):
    scheduler, _ = _stub_scheduler(num_blocks, margin, mode=mode)
    for request in requests:
        assert scheduler.add_request(request)
    return scheduler.run_to_completion(), scheduler


MIXED_REQUESTS = [
    _request("r0", 13, 9),
    _request("r1", 32, 5),
    _request("r2", 33, 12),
    _request("r3", 16, 7),
    _request("r4", 5, 20),
    _request("r5", 48, 4),
]

# Admission only reserves the *current* history, so a short prompt is cheap to
# admit and then grows far past what it was charged for. That gap is what makes
# preemption reachable at all: four of these are admitted on one block each and
# then demand four blocks apiece out of a six-block pool.
GROWTH_REQUESTS = [_request(f"g{index}", 4, 60) for index in range(4)]
GROWTH_POOL_BLOCKS = 6
AMPLE_POOL_BLOCKS = 64


# --- the gate ---------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_concurrent_matches_sequential(mode):
    """Byte-identical, not close: the whole point of iteration-level scheduling.

    Asserted in both execution modes. The stub's arithmetic does not depend on
    batch shape, so fusion must not disturb this even though on the real model it
    only holds to a tolerance -- any difference here would be the scheduler.
    """
    sequential = _run_sequentially(MIXED_REQUESTS, num_blocks=AMPLE_POOL_BLOCKS, mode=mode)
    concurrent, scheduler = _run_concurrently(
        MIXED_REQUESTS, num_blocks=AMPLE_POOL_BLOCKS, mode=mode
    )

    assert set(concurrent) == set(sequential)
    for request_id in sorted(sequential):
        assert concurrent[request_id] == sequential[request_id], (
            f"[{mode.value}] {request_id} diverged: concurrent {concurrent[request_id]} "
            f"vs sequential {sequential[request_id]}"
        )
    print(
        f"\n[{mode.value}] concurrent == sequential over {len(MIXED_REQUESTS)} requests, "
        f"{sum(len(v) for v in concurrent.values())} tokens total"
    )
    assert not scheduler.events_of(EventKind.PREEMPT), "pool was meant to be ample here"


@pytest.mark.parametrize("mode", MODES)
def test_concurrent_matches_sequential_under_forced_preemption(mode):
    """Same equality when the pool is too small and sequences get recomputed."""
    sequential = _run_sequentially(GROWTH_REQUESTS, num_blocks=AMPLE_POOL_BLOCKS, mode=mode)
    concurrent, scheduler = _run_concurrently(
        GROWTH_REQUESTS, num_blocks=GROWTH_POOL_BLOCKS, mode=mode
    )

    preemptions = scheduler.events_of(EventKind.PREEMPT)
    recomputes = scheduler.events_of(EventKind.RECOMPUTE)
    print(
        f"\n[{mode.value}] forced preemption: {len(preemptions)} preempts, "
        f"{len(recomputes)} recomputes on a {GROWTH_POOL_BLOCKS}-block pool"
    )
    assert preemptions, "pool was not small enough to force preemption"
    assert recomputes, "preempted sequences were never recomputed"

    for request_id in sorted(sequential):
        assert concurrent[request_id] == sequential[request_id], (
            f"[{mode.value}] {request_id} diverged under preemption: concurrent "
            f"{concurrent[request_id]} vs sequential {sequential[request_id]}"
        )


# --- admission guard --------------------------------------------------------


def test_admission_guard_rejects_oversized_request():
    scheduler, allocator = _stub_scheduler(num_blocks=8)
    limit_tokens = (allocator.num_blocks - ADMISSION_MARGIN_BLOCKS) * BLOCK_SIZE

    ok = _request("fits", limit_tokens - 1, 1)
    too_big = _request("too_big", limit_tokens, 1)

    assert scheduler.add_request(ok) is True
    assert scheduler.add_request(too_big) is False

    assert "too_big" in scheduler.rejected
    assert scheduler.rejected["too_big"].status is SequenceStatus.REJECTED
    # Rejected before it could ever occupy the queue head.
    assert all(seq.request_id != "too_big" for seq in scheduler.waiting)

    rejects = scheduler.events_of(EventKind.REJECT)
    assert len(rejects) == 1
    assert rejects[0].request_id == "too_big"


def test_oversized_request_cannot_wedge_the_queue():
    """The livelock the guard exists to prevent.

    An oversized request would be the only thing running, fail to grow, be its
    own eviction victim, requeue to the FCFS head, be readmitted into an empty
    pool, and exhaust again -- forever. With the guard it never enters the queue,
    so the requests behind it still complete.
    """
    scheduler, allocator = _stub_scheduler(num_blocks=8)
    limit_tokens = (allocator.num_blocks - ADMISSION_MARGIN_BLOCKS) * BLOCK_SIZE

    assert scheduler.add_request(_request("monster", limit_tokens + 64, 8)) is False
    assert scheduler.add_request(_request("small_a", 12, 4)) is True
    assert scheduler.add_request(_request("small_b", 20, 4)) is True

    outputs = scheduler.run_to_completion(max_steps=500)

    assert set(outputs) == {"small_a", "small_b"}
    assert "monster" not in outputs
    assert not scheduler.events_of(EventKind.PREEMPT)
    assert len(outputs["small_a"]) == 4
    assert len(outputs["small_b"]) == 4


def test_guard_boundary_is_exactly_num_blocks_minus_margin():
    scheduler, allocator = _stub_scheduler(num_blocks=8, margin=ADMISSION_MARGIN_BLOCKS)
    admissible_blocks = allocator.num_blocks - ADMISSION_MARGIN_BLOCKS

    exactly = _request("exactly", admissible_blocks * BLOCK_SIZE - 1, 1)
    one_over = _request("one_over", admissible_blocks * BLOCK_SIZE, 1)

    assert allocator.blocks_needed(exactly.max_total_len) == admissible_blocks
    assert allocator.blocks_needed(one_over.max_total_len) == admissible_blocks + 1
    assert scheduler.add_request(exactly) is True
    assert scheduler.add_request(one_over) is False


# --- FCFS -------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_strict_fcfs_no_backfilling(mode):
    """A head that does not fit blocks everything behind it, including one that would."""
    scheduler, allocator = _stub_scheduler(num_blocks=8, mode=mode)

    # Occupies 6 of 8 blocks once admitted, leaving 2 free.
    assert scheduler.add_request(_request("big", 6 * BLOCK_SIZE, 4))
    # Needs 2 blocks + 1 margin = 3 > 2 free, so it cannot be admitted.
    assert scheduler.add_request(_request("head", 2 * BLOCK_SIZE, 2))
    # Needs 1 block + 1 margin = 2 <= 2 free, so it *would* fit if backfilled.
    assert scheduler.add_request(_request("tail", 8, 2))

    scheduler.step()
    assert [seq.request_id for seq in scheduler.running] == ["big"]
    assert allocator.num_free == 2

    scheduler.step()
    waiting_ids = [seq.request_id for seq in scheduler.waiting]
    assert waiting_ids == ["head", "tail"], (
        f"expected head and tail both still queued in arrival order, got {waiting_ids}"
    )
    assert [seq.request_id for seq in scheduler.running] == ["big"], (
        "tail was backfilled past a head that did not fit"
    )


@pytest.mark.parametrize("mode", MODES)
def test_finished_sequence_frees_blocks_immediately(mode):
    """Blocks come back as soon as a sequence finishes, and the pool ends whole.

    True in both modes. *When* within a step they come back differs, and
    `test_fused_defers_mid_step_reclamation` is the test for that.
    """
    scheduler, allocator = _stub_scheduler(num_blocks=AMPLE_POOL_BLOCKS, mode=mode)
    assert scheduler.add_request(_request("short", 16, 2))
    assert scheduler.add_request(_request("long", 16, 6))

    free_before = allocator.num_free
    while scheduler.running or scheduler.waiting:
        result = scheduler.step()
        if result.finished:
            # The finish event records the pool state at the moment of the
            # finish, so freed blocks are visible without waiting for the step
            # to end.
            finish = [e for e in scheduler.events_of(EventKind.FINISH)][-1]
            assert finish.free_blocks > 0

    assert allocator.num_free == free_before == allocator.num_blocks
    assert allocator.num_allocated == 0


# --- preemption -------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_victim_set_is_never_the_entire_running_set(mode):
    """Checked per step, on a pool small enough to preempt constantly.

    Parametrized because FUSED mode holds every batch member's blocks across the
    whole forward, so it reaches exhaustion sooner and evicts more. The invariant
    still follows from the admission guard -- everything holding a block is in the
    running set and therefore evictable -- but the harder case deserves the test.
    """
    scheduler, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS, mode=mode)
    for request in GROWTH_REQUESTS:
        scheduler.add_request(request)

    steps = 0
    while scheduler.waiting or scheduler.running:
        running_before = len(scheduler.running)
        result = scheduler.step()
        if result.preempted:
            assert len(result.preempted) < max(running_before, 1), (
                f"step {result.step}: preempted {len(result.preempted)} of "
                f"{running_before} running -- the entire running set"
            )
        steps += 1
        assert steps < 20_000, "scheduler made no progress"

    assert scheduler.events_of(EventKind.PREEMPT), "test did not exercise preemption"
    print(
        f"\n[{mode.value}] victim-set invariant held over {steps} steps and "
        f"{len(scheduler.events_of(EventKind.PREEMPT))} preemptions"
    )


@pytest.mark.parametrize("mode", MODES)
def test_preemption_count_is_durable(mode):
    scheduler, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS, mode=mode)
    for request in GROWTH_REQUESTS:
        scheduler.add_request(request)
    scheduler.run_to_completion()

    counts = {rid: seq.preemption_count for rid, seq in scheduler.finished.items()}
    preempt_events = scheduler.events_of(EventKind.PREEMPT)
    tally: dict[str, int] = {}
    for event in preempt_events:
        tally[event.request_id] = tally.get(event.request_id, 0) + 1

    print(f"\n[{mode.value}] preemption_count per request: {counts}")
    assert any(count > 0 for count in counts.values())
    # Survived every requeue and readmission: the count equals the number of
    # preempt events for that request, not just the last one.
    for request_id, count in counts.items():
        assert count == tally.get(request_id, 0)


@pytest.mark.parametrize("mode", MODES)
def test_preemption_count_is_never_consulted(mode):
    """Scheduling must be invariant to the telemetry counter's value."""
    baseline, scheduler_a = _run_concurrently(
        GROWTH_REQUESTS, num_blocks=GROWTH_POOL_BLOCKS, mode=mode
    )
    trace_a = [(e.kind, e.request_id) for e in scheduler_a.events]

    scheduler_b, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS, mode=mode)
    for index, request in enumerate(GROWTH_REQUESTS):
        assert scheduler_b.add_request(request)
        # Seed wildly different histories before anything runs.
        scheduler_b.waiting[-1].preemption_count = (index + 1) * 100
    seeded = scheduler_b.run_to_completion()
    trace_b = [(e.kind, e.request_id) for e in scheduler_b.events]

    assert seeded == baseline
    assert trace_b == trace_a, "scheduling changed when preemption_count changed"


# --- event log --------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_event_log_records_every_decision_with_timestamps(mode):
    scheduler, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS, mode=mode)
    assert scheduler.add_request(_request("huge", 4096, 4)) is False
    for request in GROWTH_REQUESTS:
        scheduler.add_request(request)
    scheduler.run_to_completion()

    kinds = {event.kind for event in scheduler.events}
    assert EventKind.ADMIT in kinds
    assert EventKind.FINISH in kinds
    assert EventKind.PREEMPT in kinds
    assert EventKind.RECOMPUTE in kinds
    assert EventKind.REJECT in kinds

    timestamps = [event.timestamp for event in scheduler.events]
    assert all(isinstance(value, float) for value in timestamps)
    assert timestamps == sorted(timestamps), "event log is not in time order"
    assert all(event.step >= 0 for event in scheduler.events)

    # Every finished request was admitted at least once, and admitted once more
    # for each time it was preempted.
    for request_id, seq in scheduler.finished.items():
        admits = [e for e in scheduler.events_of(EventKind.ADMIT) if e.request_id == request_id]
        assert len(admits) == 1 + seq.preemption_count

    print(f"\n[{mode.value}] event log: {len(scheduler.events)} events "
          f"across {len(kinds)} kinds")


def test_event_log_uses_an_injectable_clock():
    """So a test can be deterministic and a plot can use real time."""
    ticks = iter(range(1000))
    allocator = BlockAllocator(num_blocks=32, block_size=BLOCK_SIZE)
    scheduler = Scheduler(
        StubExecutor(allocator), allocator, clock=lambda: float(next(ticks))
    )
    scheduler.add_request(_request("a", 16, 3))
    scheduler.run_to_completion()

    stamps = [event.timestamp for event in scheduler.events]
    assert stamps == [float(index) for index in range(len(stamps))]
    assert Scheduler(StubExecutor(allocator), allocator).clock is time.monotonic


# --- model tier -------------------------------------------------------------

DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_PROMPT = (
    "The capital of France is Paris, and the capital of Japan is Tokyo. "
    "Rivers shape the cities built beside them, carrying trade inland long "
    "before roads were paved, and the oldest bridges still mark where the "
    "first crossings were cut into the banks."
)
MODEL_POOL_BLOCKS = 64
MODEL_REQUEST_SHAPES = ((13, 6), (20, 5), (32, 7), (33, 4))


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
    """Token IDs of BASE_PROMPT, tokenized once per session."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL)
    return tuple(tokenizer(BASE_PROMPT, add_special_tokens=False)["input_ids"])


def _model_requests():
    ids = _base_ids()
    return [
        Request(
            request_id=f"m{index}",
            prompt_token_ids=tuple(ids[:prompt_len]),
            max_new_tokens=max_new,
        )
        for index, (prompt_len, max_new) in enumerate(MODEL_REQUEST_SHAPES)
    ]


def _model_scheduler(
    engine,
    num_blocks: int,
    block_size: int = BLOCK_SIZE,
    mode: ExecutionMode = ExecutionMode.UNFUSED,
):
    """Model-tier scheduler. Defaults to UNFUSED: this file's model-tier claims
    are exact equalities, which is a property of the unfused path."""
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    cache = engine.new_paged_cache(num_blocks, block_size)
    return (
        Scheduler(Qwen2Executor(engine, cache, allocator), allocator, mode=mode),
        allocator,
    )


def test_model_concurrent_matches_sequential(runtime):
    """The real gate, on the real model: identical tokens, not close.

    UNFUSED. Byte-identity holds because no sequence's arithmetic depends on what
    else is resident. `tests/test_fused.py` is where the fused path is held to the
    unfused one, to a tolerance declared there.
    """
    device, engine = runtime
    torch.manual_seed(SEED)
    requests = _model_requests()

    sequential = {}
    for request in requests:
        scheduler, _ = _model_scheduler(engine, MODEL_POOL_BLOCKS)
        assert scheduler.add_request(request)
        sequential.update(scheduler.run_to_completion())

    scheduler, _ = _model_scheduler(engine, MODEL_POOL_BLOCKS)
    for request in requests:
        assert scheduler.add_request(request)
    concurrent = scheduler.run_to_completion()

    print(f"\n[{device}] scheduler: {len(requests)} requests concurrent vs sequential")
    for request_id in sorted(sequential):
        identical = concurrent[request_id] == sequential[request_id]
        print(f"[{device}]   {request_id}: identical={identical}")
    print(f"[{device}]   max concurrent running: "
          f"{max(e.running for e in scheduler.events)}")

    assert set(concurrent) == set(sequential)
    for request_id in sorted(sequential):
        assert concurrent[request_id] == sequential[request_id], (
            f"[{device}] {request_id} diverged: concurrent "
            f"{concurrent[request_id]} vs sequential {sequential[request_id]}"
        )
    # It must actually have batched, or the test proves nothing.
    assert max(event.running for event in scheduler.events) > 1


def test_model_scheduler_matches_generate_greedy(runtime):
    """Cross-check against the Step 3 path, which parity already pinned to HF."""
    device, engine = runtime
    torch.manual_seed(SEED)
    requests = _model_requests()

    scheduler, _ = _model_scheduler(engine, MODEL_POOL_BLOCKS)
    for request in requests:
        assert scheduler.add_request(request)
    concurrent = scheduler.run_to_completion()

    for request in requests:
        allocator = BlockAllocator(num_blocks=MODEL_POOL_BLOCKS, block_size=BLOCK_SIZE)
        cache = engine.new_paged_cache(MODEL_POOL_BLOCKS)
        backend = engine.new_paged_backend(cache, allocator, seq_id=0)
        expected = engine.generate_greedy(
            list(request.prompt_token_ids), request.max_new_tokens, backend=backend
        )
        assert concurrent[request.request_id] == expected.token_ids, (
            f"[{device}] {request.request_id}: scheduler output differs from "
            "generate_greedy, which parity pinned to HuggingFace"
        )


# --- Step 5: preemption -----------------------------------------------------
#
# The gate is that recompute is transparent: a request that was preempted and
# replayed must produce the identical tokens it would have produced untouched.
# If that fails, preemption is not an optimisation with a cost, it is a
# correctness bug, and every throughput number downstream is measuring a
# different computation than the reference.

# The required fixture, exactly as specified.
FIXTURE_NUM_BLOCKS = 3
FIXTURE_BLOCK_SIZE = 4
FIXTURE_PROMPT_LEN = 4  # one whole block: both A and B start on a boundary
FIXTURE_MAX_NEW = 4
FIXTURE_MAX_TURNS = 8

# Real-model recompute gate, as (B prompt_len, pool_blocks). Derived with the
# stub, which schedules identically because admission and eviction depend only
# on token *counts*, never on token values -- so the cheap tier can be used to
# find a pool size that forces exactly one preemption, and the expensive tier
# then only has to check the arithmetic.
RECOMPUTE_BLOCK_SIZE = 4
RECOMPUTE_FILLER_PROMPT_LEN = 16
RECOMPUTE_FILLER_MAX_NEW = 16
RECOMPUTE_B_MAX_NEW = 16
RECOMPUTE_CASES = ((13, 13), (32, 18), (33, 18))
RECOMPUTE_AMPLE_BLOCKS = 64

# Real configured pool, driven into exhaustion.
EXHAUSTION_REQUESTS = 40
EXHAUSTION_PROMPT_LEN = 1600
EXHAUSTION_MAX_NEW = 1600


def _event_index(events, kind: EventKind, request_id: str) -> list[int]:
    return [
        index
        for index, event in enumerate(events)
        if event.kind is kind and event.request_id == request_id
    ]


@pytest.mark.parametrize("mode", MODES)
def test_required_preemption_fixture(mode):
    """Three blocks, block size four. A older, B newer, both on block boundaries.

    B is squeezed out exactly once, waits while A runs to completion, and is only
    readmitted once A's blocks come back. Every assertion below is on the event
    log rather than on internal state, so it describes observable behaviour.

    The trace is identical in both execution modes. It happens not to depend on
    mid-step reclamation: A is alone by the time it finishes, so there is nobody
    behind it in that step to hand blocks to.
    """
    scheduler, allocator = _stub_scheduler(
        num_blocks=FIXTURE_NUM_BLOCKS, block_size=FIXTURE_BLOCK_SIZE, mode=mode
    )
    a = _request("A", FIXTURE_PROMPT_LEN, FIXTURE_MAX_NEW)
    b = _request("B", FIXTURE_PROMPT_LEN, FIXTURE_MAX_NEW)

    # Both start on a block boundary, which is what makes the trace exact.
    assert a.prompt_len % FIXTURE_BLOCK_SIZE == 0
    assert b.prompt_len % FIXTURE_BLOCK_SIZE == 0
    assert scheduler.add_request(a)  # older
    assert scheduler.add_request(b)  # newer

    turns = 0
    while scheduler.waiting or scheduler.running:
        scheduler.step()
        turns += 1
        assert turns <= FIXTURE_MAX_TURNS, (
            f"did not complete within {FIXTURE_MAX_TURNS} turns: "
            f"{len(scheduler.waiting)} waiting, {len(scheduler.running)} running"
        )

    events = scheduler.events
    admits_b = _event_index(events, EventKind.ADMIT, "B")
    preempts_b = _event_index(events, EventKind.PREEMPT, "B")
    finishes_a = _event_index(events, EventKind.FINISH, "A")

    print(f"\n[{mode.value}] required fixture: {turns} turns, "
          f"{FIXTURE_NUM_BLOCKS} blocks of {FIXTURE_BLOCK_SIZE}")
    for event in events:
        print(f"  step {event.step}  {event.kind.value:<9} {event.request_id}  "
              f"free={event.free_blocks} running={event.running} waiting={event.waiting}")

    # B is preempted exactly once. The bound of one is a property of this
    # fixture, not a cap the scheduler enforces.
    assert len(preempts_b) == 1, f"expected exactly one preemption of B, got {len(preempts_b)}"
    assert not _event_index(events, EventKind.PREEMPT, "A"), "A must never be preempted"
    assert len(admits_b) == 2, f"expected admit + one readmit for B, got {len(admits_b)}"

    preempt_b, finish_a, readmit_b = preempts_b[0], finishes_a[0], admits_b[1]
    assert preempt_b < finish_a < readmit_b, (
        f"ordering violated: PREEMPT(B)={preempt_b}, FINISH(A)={finish_a}, "
        f"READMIT(B)={readmit_b}"
    )
    # No readmission of B in the window between losing its blocks and A's finish
    # returning them: B genuinely waits rather than thrashing.
    assert not [index for index in admits_b if preempt_b < index < finish_a], (
        "B was readmitted before A finished"
    )
    # The readmission is a replay, not a fresh start.
    recomputes_b = _event_index(events, EventKind.RECOMPUTE, "B")
    assert len(recomputes_b) == 1
    assert recomputes_b[0] > preempt_b

    # All three blocks came back.
    assert allocator.num_free == FIXTURE_NUM_BLOCKS
    assert allocator.num_allocated == 0

    assert scheduler.finished["A"].preemption_count == 0
    assert scheduler.finished["B"].preemption_count == 1
    assert len(scheduler.finished["A"].generated) == FIXTURE_MAX_NEW
    assert len(scheduler.finished["B"].generated) == FIXTURE_MAX_NEW


@pytest.mark.parametrize("mode", MODES)
def test_required_fixture_recompute_is_byte_identical(mode):
    """B's output in the fixture equals B's output when never preempted."""
    scheduler, _ = _stub_scheduler(
        num_blocks=FIXTURE_NUM_BLOCKS, block_size=FIXTURE_BLOCK_SIZE, mode=mode
    )
    assert scheduler.add_request(_request("A", FIXTURE_PROMPT_LEN, FIXTURE_MAX_NEW))
    assert scheduler.add_request(_request("B", FIXTURE_PROMPT_LEN, FIXTURE_MAX_NEW))
    preempted = scheduler.run_to_completion()
    assert scheduler.finished["B"].preemption_count == 1

    alone, reference = _stub_scheduler(
        num_blocks=RECOMPUTE_AMPLE_BLOCKS, block_size=FIXTURE_BLOCK_SIZE, mode=mode
    )
    assert alone.add_request(_request("B", FIXTURE_PROMPT_LEN, FIXTURE_MAX_NEW))
    untouched = alone.run_to_completion()
    assert alone.finished["B"].preemption_count == 0

    assert preempted["B"] == untouched["B"], (
        f"recompute changed the answer: preempted {preempted['B']} "
        f"vs untouched {untouched['B']}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_one_preemption_is_fixture_specific_not_a_global_cap(mode):
    """The scheduler imposes no limit; the fixture's bound of one is its own."""
    scheduler, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS, mode=mode)
    for request in GROWTH_REQUESTS:
        assert scheduler.add_request(request)
    scheduler.run_to_completion()

    counts = {rid: seq.preemption_count for rid, seq in scheduler.finished.items()}
    print(f"\n[{mode.value}] no global cap: preemption counts {counts}")
    assert max(counts.values()) > 1, (
        "no request was preempted more than once, so this cannot distinguish "
        "'no cap' from 'a cap of one'"
    )


@pytest.mark.parametrize("mode", MODES)
def test_real_pool_exhaustion_completes_every_request(mode):
    """Exhaust the real configured pool: no deadlock, no loss, no truncation."""
    from engine.cache import NUM_GPU_BLOCKS

    requests = [
        _request(f"x{index}", EXHAUSTION_PROMPT_LEN, EXHAUSTION_MAX_NEW)
        for index in range(EXHAUSTION_REQUESTS)
    ]
    scheduler, allocator = _stub_scheduler(
        num_blocks=NUM_GPU_BLOCKS, block_size=BLOCK_SIZE, mode=mode
    )
    assert allocator.num_blocks == NUM_GPU_BLOCKS
    assert allocator.block_size == BLOCK_SIZE
    for request in requests:
        assert scheduler.add_request(request), "the guard should not reject these"

    outputs = scheduler.run_to_completion()

    demanded = sum(
        allocator.blocks_needed(request.max_total_len) for request in requests
    )
    preempts = scheduler.events_of(EventKind.PREEMPT)
    print(
        f"\n[{mode.value}] real pool {NUM_GPU_BLOCKS} blocks x {BLOCK_SIZE}: "
        f"{EXHAUSTION_REQUESTS} requests demanding {demanded} blocks "
        f"({demanded / NUM_GPU_BLOCKS:.1f}x the pool), "
        f"{len(preempts)} preemptions, "
        f"max running {max(event.running for event in scheduler.events)}"
    )
    assert demanded > NUM_GPU_BLOCKS, "the pool was never actually oversubscribed"
    assert preempts, "exhaustion was not reached"

    # No request lost.
    assert set(outputs) == {request.request_id for request in requests}
    # No silent truncation.
    for request in requests:
        assert len(outputs[request.request_id]) == request.max_new_tokens, (
            f"{request.request_id} produced {len(outputs[request.request_id])} tokens, "
            f"expected {request.max_new_tokens}"
        )
    # Correct output, against each request run alone.
    sequential = _run_sequentially(requests, num_blocks=NUM_GPU_BLOCKS, mode=mode)
    for request in requests:
        assert outputs[request.request_id] == sequential[request.request_id], (
            f"{request.request_id} diverged under exhaustion"
        )
    # Pool fully reclaimed.
    assert allocator.num_free == NUM_GPU_BLOCKS
    assert allocator.num_allocated == 0


@pytest.mark.parametrize("prompt_len,pool_blocks", RECOMPUTE_CASES)
def test_model_recompute_is_byte_identical(runtime, prompt_len, pool_blocks):
    """THE Step 5 gate, on the real model. UNFUSED, where the claim is exact.

    A preempted-and-replayed request must produce exactly the tokens it produces
    when it is never touched. This is the one place the numerics of recompute are
    actually at risk: the replay re-derives the whole history in a single
    multi-token prefill, where the untouched run built the same KV one token at a
    time. Those are different reduction orders, so the KV is not bitwise equal,
    and a token could flip. Only measurement settles it.

    `tests/test_fused.py::test_fused_recompute_matches_unfused` carries the same
    claim into the fused path, where it is a tolerance rather than an equality.
    """
    device, engine = runtime
    torch.manual_seed(SEED)
    ids = _base_ids()

    b = Request(
        request_id="B",
        prompt_token_ids=tuple(ids[:prompt_len]),
        max_new_tokens=RECOMPUTE_B_MAX_NEW,
    )
    filler = Request(
        request_id="filler",
        prompt_token_ids=tuple(ids[:RECOMPUTE_FILLER_PROMPT_LEN]),
        max_new_tokens=RECOMPUTE_FILLER_MAX_NEW,
    )

    # Reference: B alone on an ample pool, never preempted.
    reference_scheduler, _ = _model_scheduler(
        engine, RECOMPUTE_AMPLE_BLOCKS, RECOMPUTE_BLOCK_SIZE
    )
    assert reference_scheduler.add_request(b)
    untouched = reference_scheduler.run_to_completion()
    assert reference_scheduler.finished["B"].preemption_count == 0

    # Preempted: an older filler holds blocks while B tries to grow.
    scheduler, allocator = _model_scheduler(engine, pool_blocks, RECOMPUTE_BLOCK_SIZE)
    assert scheduler.add_request(filler)
    assert scheduler.add_request(b)
    preempted = scheduler.run_to_completion()

    sequence = scheduler.finished["B"]
    replay = [
        event for event in scheduler.events_of(EventKind.RECOMPUTE)
        if event.request_id == "B"
    ]
    print(
        f"\n[{device}] recompute gate prompt_len={prompt_len} pool={pool_blocks} "
        f"blocks of {RECOMPUTE_BLOCK_SIZE}"
    )
    print(f"[{device}]   B preemption_count : {sequence.preemption_count}")
    print(f"[{device}]   replay             : {replay[0].detail if replay else 'none'}")
    print(f"[{device}]   identical to untouched: {preempted['B'] == untouched['B']}")

    assert sequence.preemption_count == 1, (
        f"fixture did not preempt B exactly once (got {sequence.preemption_count}); "
        "the gate would be vacuous"
    )
    assert replay, "B was readmitted without a recompute event"
    assert allocator.num_free == pool_blocks

    assert preempted["B"] == untouched["B"], (
        f"[{device}] recompute changed the answer at prompt_len={prompt_len}: "
        f"preempted {preempted['B']} vs untouched {untouched['B']}"
    )


# --- Step 5b: what the two execution modes share, and what they do not -------


@pytest.mark.parametrize("mode", MODES)
def test_a_step_issues_one_call_per_running_sequence_or_one_in_total(mode):
    """The scheduler's half of the fusion claim: how many calls a step makes.

    FUSED must make exactly one, whatever the running-set size. UNFUSED must make
    one per sequence that ran, which is what gives the counter teeth -- if both
    modes made the same number of calls, counting them would prove nothing.

    This tier can only see calls into the executor. Whether one call is really one
    forward pass is a claim about the model, and
    `tests/test_fused.py::test_fused_model_makes_exactly_one_forward_pass_per_step`
    counts the engine's own passes to settle it.
    """
    scheduler, _ = _stub_scheduler(num_blocks=AMPLE_POOL_BLOCKS, mode=mode)
    for request in MIXED_REQUESTS:
        assert scheduler.add_request(request)

    executor = scheduler.executor
    widths: list[int] = []
    while scheduler.waiting or scheduler.running:
        before = executor.calls
        result = scheduler.step()
        made = executor.calls - before
        widths.append(result.batch_size)

        if mode is ExecutionMode.FUSED:
            assert made == 1, (
                f"step {result.step} ran {result.batch_size} sequences in {made} calls; "
                "fusion means one"
            )
        else:
            assert made == result.batch_size, (
                f"step {result.step} ran {result.batch_size} sequences in {made} calls"
            )

    print(f"\n[{mode.value}] running-set widths per step: {widths}")
    print(f"[{mode.value}] executor calls: single={executor.forward_calls} "
          f"batched={executor.batch_calls}")

    # The fixture has to actually vary the running-set size, or "regardless of
    # running-set size" is untested.
    assert max(widths) > 1, "no step ever ran more than one sequence"
    assert len(set(widths)) > 1, "the running-set size never changed"

    if mode is ExecutionMode.FUSED:
        assert executor.forward_calls == 0, "fused mode fell back to single forwards"
        assert executor.batch_calls == len(widths)
    else:
        assert executor.batch_calls == 0
        assert executor.forward_calls == sum(widths) > len(widths)


# Three blocks of four, no admission margin, and A finishing on its prefill.
# UNFUSED reclaims A's two blocks inside step 1 and admits B behind it; FUSED
# cannot, because A holds those blocks for the whole of the step's one forward.
RECLAIM_NUM_BLOCKS = 3
RECLAIM_BLOCK_SIZE = 4
RECLAIM_PROMPT_LEN = 8


def _reclaim_requests():
    return (
        _request("A", RECLAIM_PROMPT_LEN, 1),
        _request("B", RECLAIM_PROMPT_LEN, 4),
    )


def _run_reclaim_fixture(mode: ExecutionMode):
    scheduler, allocator = _stub_scheduler(
        num_blocks=RECLAIM_NUM_BLOCKS,
        margin=0,
        block_size=RECLAIM_BLOCK_SIZE,
        mode=mode,
    )
    for request in _reclaim_requests():
        assert scheduler.add_request(request), "the guard should not reject these"

    admitted_per_step: list[tuple[str, ...]] = []
    while scheduler.waiting or scheduler.running:
        result = scheduler.step()
        admitted_per_step.append(result.admitted)
        assert len(admitted_per_step) <= 12, "fixture did not converge"
    return scheduler, allocator, admitted_per_step


def test_fused_defers_mid_step_reclamation():
    """The one behavioural cost of fusion, pinned to a fixture rather than prose.

    UNFUSED runs each sequence's forward as it is scheduled, so A -- which
    finishes on its own prefill -- returns its blocks before the queue head is
    looked at again, and B is admitted in the same step. FUSED plans the whole
    step before running anything, so A still holds those blocks when B's
    admission is tested, and B waits a step.

    This costs a step. It does not cost correctness: both modes produce the same
    tokens, because a deferred admission is just a later admission and recompute
    is transparent.
    """
    unfused, unfused_pool, unfused_admits = _run_reclaim_fixture(ExecutionMode.UNFUSED)
    fused, fused_pool, fused_admits = _run_reclaim_fixture(ExecutionMode.FUSED)

    print(f"\nadmissions per step, unfused: {unfused_admits}")
    print(f"admissions per step, fused  : {fused_admits}")

    assert unfused_admits[0] == ("A", "B"), (
        "unfused did not reuse A's blocks inside step 1, so this fixture no "
        "longer demonstrates mid-step reclamation"
    )
    assert fused_admits[0] == ("A",)
    assert fused_admits[1] == ("B",)
    assert len(fused_admits) == len(unfused_admits) + 1

    # Same answers, one step later.
    unfused_out = {rid: tuple(seq.generated) for rid, seq in unfused.finished.items()}
    fused_out = {rid: tuple(seq.generated) for rid, seq in fused.finished.items()}
    assert fused_out == unfused_out

    for pool in (unfused_pool, fused_pool):
        assert pool.num_free == RECLAIM_NUM_BLOCKS
        assert pool.num_allocated == 0


# --- Step 5b, the question the fusion report left open ----------------------
#
# Does one-pass-per-step survive a step where a preemption AND a recompute both
# happen? A recompute re-prefills the whole history, which is a forward, so the
# question is whether that forward joins the step's batch or costs its own call.
#
# ANSWER: it joins. A readmission goes through `_start`, which calls `_submit`
# like any decode, so in FUSED mode it lands in `_pending` and is flushed with
# everything else. There is no path by which a recompute issues its own forward.
# Asserted below on a fixture where both events land in one step, rather than
# argued from the source.

# Twelve blocks of four, no margin, five requests that each grow to six blocks.
# Step 29 of this run readmits two preempted sequences -- two recomputes -- and
# then evicts one of them again while planning the decodes, all in one step.
CO_NUM_BLOCKS = 12
CO_BLOCK_SIZE = 4
CO_MARGIN = 0
CO_REQUESTS = 5
CO_PROMPT_LEN = 5
CO_MAX_NEW = 24


def _co_requests():
    return [_request(f"r{index}", CO_PROMPT_LEN, CO_MAX_NEW) for index in range(CO_REQUESTS)]


def _run_co_fixture(mode: ExecutionMode):
    """Step through the fixture, recording per-step calls and the events raised."""
    scheduler, allocator = _stub_scheduler(
        num_blocks=CO_NUM_BLOCKS, margin=CO_MARGIN, block_size=CO_BLOCK_SIZE, mode=mode
    )
    for request in _co_requests():
        assert scheduler.add_request(request), "the guard should not reject these"

    executor = scheduler.executor
    rows = []
    while scheduler.waiting or scheduler.running:
        calls_before = executor.calls
        events_before = len(scheduler.events)
        result = scheduler.step()
        rows.append(
            (
                result.step,
                result.batch_size,
                executor.calls - calls_before,
                tuple((e.kind, e.request_id) for e in scheduler.events[events_before:]),
            )
        )
        assert result.step < 400, "fixture did not converge"
    return scheduler, allocator, rows


def _steps_with(rows, *kinds):
    return [row for row in rows if all(any(k is kind for k, _ in row[3]) for kind in kinds)]


def test_recompute_folds_into_the_same_batched_call():
    """A step containing BOTH a preempt and a recompute still costs ONE call.

    The step this pins does more than the name suggests. It readmits two
    preempted sequences, so two recompute prefills are planned, and then evicts
    one of them *again* while planning the decodes -- so the batch also loses a
    member it had already reserved blocks for. One call covers all of it.
    """
    scheduler, allocator, rows = _run_co_fixture(ExecutionMode.FUSED)
    executor = scheduler.executor

    both = _steps_with(rows, EventKind.PREEMPT, EventKind.RECOMPUTE)
    print(f"\nfused: {len(rows)} steps, {len(both)} with both a preempt and a recompute")
    for step, width, calls, events in both:
        print(f"  step {step}  width={width}  calls={calls}")
        for kind, request_id in events:
            print(f"      {kind.value:<9} {request_id}")

    assert both, (
        "no step contained both a preempt and a recompute, so this fixture "
        "cannot answer the question it exists to answer"
    )
    for step, width, calls, events in both:
        assert calls == 1, (
            f"step {step} raised {[k.value for k, _ in events]} and made {calls} "
            "calls; a recompute must join the step's batch, not run beside it"
        )
        assert width >= 1

    # Not just on those steps: one call per step for the whole run, and the
    # recompute prefills are genuinely inside those batches rather than skipped.
    assert all(calls == 1 for _, _, calls, _ in rows)
    assert executor.forward_calls == 0, "fused mode fell back to single forwards"
    assert executor.batch_calls == len(rows)

    recomputes = scheduler.events_of(EventKind.RECOMPUTE)
    preempts = scheduler.events_of(EventKind.PREEMPT)
    print(f"fused: {len(preempts)} preemptions and {len(recomputes)} recomputes "
          f"across {len(rows)} steps, {executor.batch_calls} batched calls, "
          f"0 single calls")
    assert len(recomputes) >= 2
    assert allocator.num_free == CO_NUM_BLOCKS


def test_the_co_occurrence_fixture_costs_several_calls_unfused():
    """Calibration: the same step is several calls in the reference mode.

    Without this, `calls == 1` above could be a property of the fixture rather
    than of fusion.
    """
    scheduler, _, rows = _run_co_fixture(ExecutionMode.UNFUSED)
    both = _steps_with(rows, EventKind.PREEMPT, EventKind.RECOMPUTE)

    print(f"\nunfused: {len(rows)} steps, {len(both)} with both")
    for step, width, calls, _ in both:
        print(f"  step {step}  width={width}  calls={calls}")

    assert both
    assert any(calls > 1 for _, _, calls, _ in both), (
        "the co-occurrence step was a single call even unfused, so the fused "
        "result proves nothing"
    )
    assert max(calls for _, _, calls, _ in rows) > 1
    assert scheduler.executor.batch_calls == 0


def test_fused_and_unfused_agree_on_the_co_occurrence_fixture():
    """Same tokens out of a run that preempts one request twice."""
    fused, _, _ = _run_co_fixture(ExecutionMode.FUSED)
    unfused, _, _ = _run_co_fixture(ExecutionMode.UNFUSED)

    fused_out = {rid: tuple(seq.generated) for rid, seq in fused.finished.items()}
    unfused_out = {rid: tuple(seq.generated) for rid, seq in unfused.finished.items()}
    counts = {rid: seq.preemption_count for rid, seq in sorted(fused.finished.items())}

    print(f"\nco-occurrence fixture preemption counts: {counts}")
    assert fused_out == unfused_out
    assert all(len(tokens) == CO_MAX_NEW for tokens in fused_out.values())
    # Somebody was preempted more than once, so the fixture really is thrashing.
    assert max(counts.values()) > 1


def test_a_step_that_evicts_everything_it_planned_makes_no_call():
    """The other edge of the invariant: at most one call, not exactly one.

    If every planned sequence is evicted before the flush, `_pending` is empty
    and the step issues no forward at all. No fixture in this file reaches that
    state, so it is provoked directly -- an invariant with an unreachable branch
    is still an invariant, and leaving it unpinned would mean `calls == 1`
    silently became `calls == 0` one day.
    """
    scheduler, _ = _stub_scheduler(num_blocks=8, mode=ExecutionMode.FUSED)
    assert scheduler.add_request(_request("only", 16, 4))
    scheduler.step()

    executor = scheduler.executor
    calls_before = executor.calls
    # Plan the step, then take the planned work away before it can be flushed.
    scheduler._step += 1
    scheduler._pending = []
    scheduler._ran = []
    scheduler._plan_decode(scheduler.running[0])
    assert scheduler._pending, "nothing was planned, so there is nothing to drop"
    scheduler._evict(scheduler.running[0])
    assert not scheduler._pending, "eviction did not drop the planned work"
    scheduler._flush()

    print(f"\nempty batch: {executor.calls - calls_before} calls issued")
    assert executor.calls == calls_before, "an empty batch still issued a forward"
    assert not scheduler._ran


# --- capacity caps ----------------------------------------------------------
#
# Two caps, the vLLM pattern, both capacity conditions on the same strict-FCFS
# order. Neither reorders the queue and neither lets anything jump the head.
#
# They exist because C never capped anything. Admission was by pool fit alone,
# so at C=1, 2, 4, 8 and 16 the mixed stream is the same 64 requests, they all
# fit the 3584-block pool, and all five levels admitted everything -- five
# identical runs and no low end to the sweep. Worse, the baseline has a hard
# resident cap of 18 or 32 from its rectangle sizing, so a "C=8" comparison was
# engine-at-64 against baseline-at-18.

CAP_REQUESTS = [_request(f"c{index}", 8, 12) for index in range(12)]


def test_max_num_seqs_bounds_the_running_set():
    """The running set never exceeds the cap, at any step."""
    for cap in (1, 2, 4, 8):
        scheduler, _ = _stub_scheduler(num_blocks=AMPLE_POOL_BLOCKS)
        scheduler.max_num_seqs = cap
        for request in CAP_REQUESTS:
            assert scheduler.add_request(request)
        widest = 0
        while scheduler.waiting or scheduler.running:
            result = scheduler.step()
            widest = max(widest, result.running)
            assert result.running <= cap, (
                f"cap {cap}: {result.running} running at step {result.step}"
            )
            assert result.step < 500
        print(f"\nmax_num_seqs={cap}: widest running set observed {widest}")
        assert widest == cap, (
            f"cap {cap} was never actually reached (widest {widest}), so this "
            "does not demonstrate the cap binds"
        )


def test_without_the_cap_every_low_level_admits_everything():
    """The bug the cap fixes, pinned so it cannot come back silently."""
    widths = {}
    for concurrency in (1, 2, 4, 8, 16):
        scheduler, _ = _stub_scheduler(num_blocks=AMPLE_POOL_BLOCKS)
        for request in CAP_REQUESTS:
            assert scheduler.add_request(request)
        result = scheduler.step()
        widths[concurrency] = result.running

    print(f"\nuncapped, running set after step 1 at each C: {widths}")
    assert len(set(widths.values())) == 1, (
        "uncapped levels already differ, so this fixture no longer demonstrates "
        "the collapse"
    )

    # With the cap set to C they must differ.
    capped = {}
    for concurrency in (1, 2, 4, 8, 16):
        scheduler, _ = _stub_scheduler(num_blocks=AMPLE_POOL_BLOCKS)
        scheduler.max_num_seqs = concurrency
        for request in CAP_REQUESTS:
            assert scheduler.add_request(request)
        capped[concurrency] = scheduler.step().running
    print(f"capped at C,  running set after step 1 at each C: {capped}")
    assert len(set(capped.values())) > 1, "the cap did not separate the levels"
    assert capped[1] == 1 and capped[2] == 2 and capped[4] == 4


def test_max_num_batched_tokens_bounds_the_padded_rectangle():
    """The cap is on PADDED slots, which is what actually allocates memory.

    A step mixing one long prefill with many single-token decodes costs
    (rows x longest), not the sum. Bounding the real token count would not bound
    this engine's memory at all.
    """
    scheduler, _ = _stub_scheduler(num_blocks=AMPLE_POOL_BLOCKS, mode=ExecutionMode.FUSED)
    scheduler.max_num_batched_tokens = 64
    for request in [_request("long", 48, 6)] + [_request(f"s{i}", 4, 6) for i in range(8)]:
        assert scheduler.add_request(request)

    costs = []
    while scheduler.waiting or scheduler.running:
        result = scheduler.step()
        costs.append(result.batched_token_cost)
        assert result.batched_token_cost <= scheduler.max_num_batched_tokens, (
            f"step {result.step} materialised {result.batched_token_cost} padded "
            f"slots against a cap of {scheduler.max_num_batched_tokens}"
        )
        assert result.step < 500
    print(f"\npadded cost per step under a 64-slot cap: {costs}")
    assert max(costs) > 0


def test_an_empty_batch_admits_the_head_whatever_its_length():
    """Otherwise a request longer than the cap could never run at all.

    That is a livelock arriving by a different route than the admission guard's,
    and it needs the same kind of escape.
    """
    scheduler, _ = _stub_scheduler(num_blocks=AMPLE_POOL_BLOCKS, mode=ExecutionMode.FUSED)
    scheduler.max_num_batched_tokens = 16
    assert scheduler.add_request(_request("huge", 100, 3))

    outputs = scheduler.run_to_completion(max_steps=200)
    print(f"\nsequence of 100 tokens under a 16-slot cap: "
          f"{len(outputs['huge'])} tokens generated")
    assert len(outputs["huge"]) == 3


def test_deferred_decodes_are_not_preemptions():
    """A decode refused for budget keeps its blocks and runs next step.

    The fixture has to put decodes in flight BEFORE the long prefill arrives.
    An earlier version admitted the long request first with nothing else
    running, so the budget never bound and the test proved nothing -- it
    asserted so, and failed, which is why it is written this way now.
    """
    scheduler, allocator = _stub_scheduler(
        num_blocks=AMPLE_POOL_BLOCKS, mode=ExecutionMode.FUSED)
    scheduler.max_num_batched_tokens = 120

    for index in range(6):
        assert scheduler.add_request(_request(f"s{index}", 4, 8))
    for _ in range(3):
        scheduler.step()
    assert len(scheduler.running) == 6, "the short sequences are not in flight"

    # Now a long one arrives. Admitted alone by the empty-batch escape at
    # 1 x 100 slots; a single decode row would take it to 2 x 100 = 200 > 120.
    assert scheduler.add_request(_request("long", 100, 4))

    deferred_total = 0
    while scheduler.waiting or scheduler.running:
        result = scheduler.step()
        deferred_total += len(result.deferred)
        assert result.batched_token_cost <= scheduler.max_num_batched_tokens
        assert result.step < 500

    print(f"\n{deferred_total} decode(s) deferred for budget; "
          f"{len(scheduler.events_of(EventKind.PREEMPT))} preemptions")
    assert deferred_total > 0, "the budget never bound, so this proves nothing"
    # Deferral is not eviction: nobody lost blocks for it, and nobody lost work.
    assert not scheduler.events_of(EventKind.PREEMPT), (
        "a budget deferral was turned into a preemption"
    )
    for sequence in scheduler.finished.values():
        assert len(sequence.generated) == sequence.request.max_new_tokens
    assert allocator.num_free == AMPLE_POOL_BLOCKS


def test_the_headlines_still_preempt_under_the_caps():
    """Capping at C must not defuse either headline's oversubscription.

    control@C64  = 64 x 64 blocks    = 4096 against 3584
    mixed@C87    = 87 x 47.26 blocks = 4111 against 3584
    """
    from bench import workload
    from engine.cache import NUM_GPU_BLOCKS

    control = workload.PRESSURE_C * workload.CONTROL_BLOCKS_PER_REQUEST
    mixed = workload.FRAG_C * workload.MEAN_BLOCKS_PER_REQUEST
    print(f"\ncontrol@C{workload.PRESSURE_C}: {control} blocks vs {NUM_GPU_BLOCKS}")
    print(f"mixed@C{workload.FRAG_C}: {int(mixed)} blocks vs {NUM_GPU_BLOCKS}")

    assert control == 4096 and control > NUM_GPU_BLOCKS
    assert int(mixed) == 4111 and mixed > NUM_GPU_BLOCKS

    # And a capped scheduler really does still evict, on a fixture whose
    # demand AT THE CAP exceeds the pool -- 8 sequences needing 9 blocks each is
    # 72 against 64. An earlier version used 5-block sequences, so 8 of them fit
    # in 64 with room to spare and nothing was ever oversubscribed; it reported
    # zero preemptions and failed, which is the assertion doing its job.
    capped_pool, capped_seqs = 64, 8
    requests = [_request(f"p{i}", 16, 120) for i in range(capped_seqs)]
    per_request_blocks = -(-requests[0].max_total_len // BLOCK_SIZE)
    demand = capped_seqs * per_request_blocks
    print(f"capped fixture: {capped_seqs} x {per_request_blocks} = {demand} blocks "
          f"vs a {capped_pool}-block pool")
    assert demand > capped_pool, "the capped fixture is not oversubscribed"

    scheduler, _ = _stub_scheduler(num_blocks=capped_pool, mode=ExecutionMode.FUSED)
    scheduler.max_num_seqs = capped_seqs
    for request in requests:
        assert scheduler.add_request(request)
    scheduler.run_to_completion(max_steps=20_000)
    preempts = scheduler.events_of(EventKind.PREEMPT)
    print(f"capped at {capped_seqs}: {len(preempts)} preemptions")
    assert preempts, "the cap defused oversubscription entirely"
