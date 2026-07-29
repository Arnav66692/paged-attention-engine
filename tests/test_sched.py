"""Step 4: continuous batching scheduler.

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
"""

import time

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
from engine.scheduler import (
    EventKind,
    Qwen2Executor,
    Request,
    Scheduler,
    SequenceStatus,
)

STUB_VOCAB = 64
SEED = 0


class StubExecutor:
    """Real block accounting, deterministic sequence-local logits.

    The next token depends only on this sequence's own last token and position,
    never on what else is running. Concurrency-independence is therefore a
    property of the stub, so the equality tests below isolate the scheduler.
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

    def new_backend(self, seq_id: int) -> PagedBackend:
        self.allocator.create_sequence(seq_id)
        return PagedBackend(self.cache, self.allocator, seq_id, num_kv_groups=1, scale=1.0)

    def forward(self, token_ids, positions, backend) -> torch.Tensor:
        self.forward_calls += 1
        reservation = backend.begin_forward(len(token_ids))
        if not reservation.ok:
            raise RuntimeError(
                f"stub forward could not reserve {len(token_ids)} tokens: "
                f"{reservation.status.value}; the scheduler should have checked first"
            )
        backend.end_forward(len(token_ids))

        next_token = (token_ids[-1] * 7 + positions[-1] * 13 + 1) % STUB_VOCAB
        logits = torch.full((STUB_VOCAB,), -1.0)
        logits[next_token] = 1.0
        return logits


def _stub_scheduler(num_blocks: int, margin: int = ADMISSION_MARGIN_BLOCKS):
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    executor = StubExecutor(allocator)
    return Scheduler(executor, allocator, admission_margin_blocks=margin), allocator


def _request(request_id: str, prompt_len: int, max_new_tokens: int) -> Request:
    # Prompt content derived from the id so different requests differ, and
    # deterministically so the fixture is reproducible.
    seed = sum(ord(character) for character in request_id)
    prompt = tuple((seed + index * 3) % STUB_VOCAB for index in range(prompt_len))
    return Request(
        request_id=request_id, prompt_token_ids=prompt, max_new_tokens=max_new_tokens
    )


def _run_sequentially(requests, num_blocks: int, margin: int = ADMISSION_MARGIN_BLOCKS):
    """Each request alone, on a fresh scheduler and a fresh pool."""
    outputs = {}
    for request in requests:
        scheduler, _ = _stub_scheduler(num_blocks, margin)
        assert scheduler.add_request(request)
        outputs.update(scheduler.run_to_completion())
    return outputs


def _run_concurrently(requests, num_blocks: int, margin: int = ADMISSION_MARGIN_BLOCKS):
    scheduler, _ = _stub_scheduler(num_blocks, margin)
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


def test_concurrent_matches_sequential():
    """Byte-identical, not close: the whole point of iteration-level scheduling."""
    sequential = _run_sequentially(MIXED_REQUESTS, num_blocks=AMPLE_POOL_BLOCKS)
    concurrent, scheduler = _run_concurrently(MIXED_REQUESTS, num_blocks=AMPLE_POOL_BLOCKS)

    assert set(concurrent) == set(sequential)
    for request_id in sorted(sequential):
        assert concurrent[request_id] == sequential[request_id], (
            f"{request_id} diverged: concurrent {concurrent[request_id]} "
            f"vs sequential {sequential[request_id]}"
        )
    print(
        f"\nconcurrent == sequential over {len(MIXED_REQUESTS)} requests, "
        f"{sum(len(v) for v in concurrent.values())} tokens total"
    )
    assert not scheduler.events_of(EventKind.PREEMPT), "pool was meant to be ample here"


def test_concurrent_matches_sequential_under_forced_preemption():
    """Same equality when the pool is too small and sequences get recomputed."""
    sequential = _run_sequentially(GROWTH_REQUESTS, num_blocks=AMPLE_POOL_BLOCKS)
    concurrent, scheduler = _run_concurrently(GROWTH_REQUESTS, num_blocks=GROWTH_POOL_BLOCKS)

    preemptions = scheduler.events_of(EventKind.PREEMPT)
    recomputes = scheduler.events_of(EventKind.RECOMPUTE)
    print(
        f"\nforced preemption: {len(preemptions)} preempts, {len(recomputes)} recomputes "
        f"on a {GROWTH_POOL_BLOCKS}-block pool"
    )
    assert preemptions, "pool was not small enough to force preemption"
    assert recomputes, "preempted sequences were never recomputed"

    for request_id in sorted(sequential):
        assert concurrent[request_id] == sequential[request_id], (
            f"{request_id} diverged under preemption: concurrent "
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


def test_strict_fcfs_no_backfilling():
    """A head that does not fit blocks everything behind it, including one that would."""
    scheduler, allocator = _stub_scheduler(num_blocks=8)

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


def test_finished_sequence_frees_blocks_immediately():
    scheduler, allocator = _stub_scheduler(num_blocks=AMPLE_POOL_BLOCKS)
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


def test_victim_set_is_never_the_entire_running_set():
    """Checked per step, on a pool small enough to preempt constantly."""
    scheduler, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS)
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
        f"\nvictim-set invariant held over {steps} steps and "
        f"{len(scheduler.events_of(EventKind.PREEMPT))} preemptions"
    )


def test_preemption_count_is_durable():
    scheduler, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS)
    for request in GROWTH_REQUESTS:
        scheduler.add_request(request)
    scheduler.run_to_completion()

    counts = {rid: seq.preemption_count for rid, seq in scheduler.finished.items()}
    preempt_events = scheduler.events_of(EventKind.PREEMPT)
    tally: dict[str, int] = {}
    for event in preempt_events:
        tally[event.request_id] = tally.get(event.request_id, 0) + 1

    print(f"\npreemption_count per request: {counts}")
    assert any(count > 0 for count in counts.values())
    # Survived every requeue and readmission: the count equals the number of
    # preempt events for that request, not just the last one.
    for request_id, count in counts.items():
        assert count == tally.get(request_id, 0)


def test_preemption_count_is_never_consulted():
    """Scheduling must be invariant to the telemetry counter's value."""
    baseline, scheduler_a = _run_concurrently(GROWTH_REQUESTS, num_blocks=GROWTH_POOL_BLOCKS)
    trace_a = [(e.kind, e.request_id) for e in scheduler_a.events]

    scheduler_b, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS)
    for index, request in enumerate(GROWTH_REQUESTS):
        assert scheduler_b.add_request(request)
        # Seed wildly different histories before anything runs.
        scheduler_b.waiting[-1].preemption_count = (index + 1) * 100
    seeded = scheduler_b.run_to_completion()
    trace_b = [(e.kind, e.request_id) for e in scheduler_b.events]

    assert seeded == baseline
    assert trace_b == trace_a, "scheduling changed when preemption_count changed"


# --- event log --------------------------------------------------------------


def test_event_log_records_every_decision_with_timestamps():
    scheduler, _ = _stub_scheduler(num_blocks=GROWTH_POOL_BLOCKS)
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

    print(f"\nevent log: {len(scheduler.events)} events across {len(kinds)} kinds")


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
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(scope="module", params=_devices())
def runtime(request):
    from transformers import AutoModelForCausalLM

    from engine.model import Qwen2Engine

    device = request.param
    model = AutoModelForCausalLM.from_pretrained(DEV_MODEL, dtype=torch.float32)
    model.eval()
    model.to(device)
    return device, Qwen2Engine.from_hf_model(model)


def _model_requests():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL)
    ids = tokenizer(BASE_PROMPT, add_special_tokens=False)["input_ids"]
    return [
        Request(
            request_id=f"m{index}",
            prompt_token_ids=tuple(ids[:prompt_len]),
            max_new_tokens=max_new,
        )
        for index, (prompt_len, max_new) in enumerate(MODEL_REQUEST_SHAPES)
    ]


def _model_scheduler(engine, num_blocks: int):
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    cache = engine.new_paged_cache(num_blocks)
    return Scheduler(Qwen2Executor(engine, cache, allocator), allocator), allocator


def test_model_concurrent_matches_sequential(runtime):
    """The real gate, on the real model: identical tokens, not close."""
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
