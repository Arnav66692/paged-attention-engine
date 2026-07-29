"""Continuous batching scheduler: iteration-level scheduling over the block pool.

Gate 1, Step 4. Every step decides what runs. A step is:

  1. Admit from the head of the waiting queue while the head fits.
  2. Newly admitted sequences prefill their complete current history and join
     the running set, producing their first token in that same step.
  3. Every other running sequence decodes one token.
  4. A sequence that reaches its budget leaves the running set immediately and
     its blocks return to the free list mid-step, where a later sequence in the
     same step can take them.

Admission is strict FCFS with no backfilling. If the head does not fit, the step
admits nothing further -- the head keeps its place and is not skipped, and no
smaller request behind it jumps the queue.

Two capacity checks, deliberately different:

  admission   needs blocks_needed(history) + admission_margin_blocks free.
  decode      needs only the one block the next token might cross into.

The margin is headroom reserved so already-running sequences can grow. Spending
it at admission would defeat its purpose, so decode is allowed into it and
admission is not.

THE ADMISSION GUARD. A request whose maximum possible length cannot fit in
num_gpu_blocks - admission_margin_blocks is rejected outright and never enters
the queue. Without this it becomes the only running sequence, fails to grow,
is selected as its own eviction victim, requeues to the FCFS head, is readmitted
because the pool is empty, and exhausts again -- forever, at zero throughput,
with the event log filling with preempt/admit pairs. The fix has to be here.
Putting it in the eviction path would only convert a livelock into a crash.

That guard is also what makes the eviction invariant true: because any single
admitted sequence fits in the pool alone, evicting down to one leaves that one
able to allocate, so the victim set is never the entire running set.

`preemption_count` is durable across requeue and readmission. It is telemetry
for the Step 6 plots. Nothing in this file reads it to make a decision, and
`tests/test_sched.py` asserts scheduling is invariant to it.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

import torch

from engine.attention import AttentionBackend
from engine.cache import ADMISSION_MARGIN_BLOCKS, BlockAllocator


class EventKind(Enum):
    ADMIT = "admit"
    FINISH = "finish"
    PREEMPT = "preempt"
    RECOMPUTE = "recompute"
    REJECT = "reject"


@dataclass(frozen=True)
class Event:
    """One scheduler decision, timestamped.

    This log is how the scheduler is debugged and where the Step 6 fragmentation
    plot gets its data, so it records pool state at the moment of the decision
    rather than leaving it to be reconstructed.
    """

    timestamp: float
    step: int
    kind: EventKind
    request_id: str
    detail: str
    free_blocks: int
    running: int
    waiting: int


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError(f"{self.request_id}: prompt must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError(f"{self.request_id}: max_new_tokens must be at least 1")

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def max_total_len(self) -> int:
        """The most tokens this request could ever hold in the cache."""
        return self.prompt_len + self.max_new_tokens


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    REJECTED = "rejected"


@dataclass
class Sequence:
    """Runtime state for one request, surviving preemption and readmission."""

    request: Request
    arrival_index: int
    generated: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    seq_id: int | None = None
    backend: AttentionBackend | None = None

    # Durable telemetry. Survives requeue and readmission; never consulted by
    # any scheduling decision.
    preemption_count: int = 0

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def history(self) -> list[int]:
        """Prompt plus everything generated so far -- what a recompute replays."""
        return list(self.request.prompt_token_ids) + self.generated

    @property
    def history_len(self) -> int:
        return self.request.prompt_len + len(self.generated)

    @property
    def is_finished(self) -> bool:
        return len(self.generated) >= self.request.max_new_tokens


@dataclass(frozen=True)
class StepResult:
    step: int
    admitted: tuple[str, ...]
    decoded: tuple[str, ...]
    finished: tuple[str, ...]
    preempted: tuple[str, ...]
    running: int
    waiting: int
    free_blocks: int


class Executor(Protocol):
    """What the scheduler needs from a model. Narrow so tests can stub it."""

    def new_backend(self, seq_id: int) -> AttentionBackend: ...

    def forward(
        self, token_ids: list[int], positions: list[int], backend: AttentionBackend
    ) -> torch.Tensor: ...


class Qwen2Executor:
    """Drives `Qwen2Engine.forward` over a paged cache for the scheduler."""

    def __init__(self, engine, paged_cache, allocator: BlockAllocator) -> None:
        self.engine = engine
        self.cache = paged_cache
        self.allocator = allocator

    def new_backend(self, seq_id: int) -> AttentionBackend:
        return self.engine.new_paged_backend(self.cache, self.allocator, seq_id)

    def forward(
        self, token_ids: list[int], positions: list[int], backend: AttentionBackend
    ) -> torch.Tensor:
        device = self.engine.device
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        pos = torch.tensor(positions, dtype=torch.long, device=device)
        return self.engine.forward(ids, pos, backend)


class Scheduler:
    """Iteration-level scheduler over a single block pool.

    One forward pass per running sequence per step. The sequences in a step are
    *not* fused into a single batched matmul, so each sequence's arithmetic is
    bit-for-bit what it would be running alone -- which is what makes the
    concurrent-equals-sequential test an equality rather than a tolerance.
    Fusing them is a throughput optimization that would trade that equality for
    a tolerance, and it is not done here.
    """

    def __init__(
        self,
        executor: Executor,
        allocator: BlockAllocator,
        admission_margin_blocks: int = ADMISSION_MARGIN_BLOCKS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if admission_margin_blocks < 0:
            raise ValueError(
                f"admission_margin_blocks must be non-negative, got {admission_margin_blocks}"
            )
        self.executor = executor
        self.allocator = allocator
        self.admission_margin_blocks = admission_margin_blocks
        self.clock = clock

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: dict[str, Sequence] = {}
        self.rejected: dict[str, Sequence] = {}
        self.events: list[Event] = []

        self._step = 0
        self._next_arrival = 0
        self._next_seq_id = 0

    # --- capacity ----------------------------------------------------------

    @property
    def max_admissible_blocks(self) -> int:
        """The largest a request may ever become and still be accepted."""
        return self.allocator.num_blocks - self.admission_margin_blocks

    def _admission_fits(self, seq: Sequence) -> bool:
        """Room for the complete current history, keeping the margin free."""
        needed = self.allocator.blocks_needed(seq.history_len)
        return needed + self.admission_margin_blocks <= self.allocator.num_free

    def _decode_fits(self, seq: Sequence) -> bool:
        """Room for one more token. Allowed to spend the margin; that is its job."""
        held = len(self.allocator.block_table(seq.seq_id))
        needed = self.allocator.blocks_needed(self.allocator.num_tokens(seq.seq_id) + 1)
        return (needed - held) <= self.allocator.num_free

    # --- event log ---------------------------------------------------------

    def _log(self, kind: EventKind, seq: Sequence, detail: str = "") -> None:
        self.events.append(
            Event(
                timestamp=self.clock(),
                step=self._step,
                kind=kind,
                request_id=seq.request_id,
                detail=detail,
                free_blocks=self.allocator.num_free,
                running=len(self.running),
                waiting=len(self.waiting),
            )
        )

    def events_of(self, kind: EventKind) -> list[Event]:
        return [event for event in self.events if event.kind is kind]

    # --- intake ------------------------------------------------------------

    def add_request(self, request: Request) -> bool:
        """Offer a request. Returns False if the admission guard rejects it.

        The guard is applied here, before the request can ever reach the head of
        the queue, so an impossible request can neither be admitted nor block
        anything behind it.
        """
        seq = Sequence(request=request, arrival_index=self._next_arrival)
        self._next_arrival += 1

        required = self.allocator.blocks_needed(request.max_total_len)
        if required > self.max_admissible_blocks:
            seq.status = SequenceStatus.REJECTED
            self.rejected[request.request_id] = seq
            self._log(
                EventKind.REJECT,
                seq,
                f"max length {request.max_total_len} needs {required} blocks, "
                f"limit is {self.max_admissible_blocks} "
                f"({self.allocator.num_blocks} - {self.admission_margin_blocks} margin)",
            )
            return False

        self.waiting.append(seq)
        return True

    # --- scheduling --------------------------------------------------------

    def _admit(self) -> list[Sequence]:
        """Admit from the head while it fits. Strict FCFS, no backfilling."""
        admitted: list[Sequence] = []
        while self.waiting:
            head = self.waiting[0]
            if not self._admission_fits(head):
                # The head keeps its place and nothing behind it is considered.
                break
            self.waiting.popleft()
            self._start(head)
            admitted.append(head)
        return admitted

    def _start(self, seq: Sequence) -> None:
        """Register with the allocator, prefill the whole history, take a token."""
        is_recompute = bool(seq.generated)
        seq.seq_id = self._next_seq_id
        self._next_seq_id += 1
        seq.backend = self.executor.new_backend(seq.seq_id)
        seq.status = SequenceStatus.RUNNING
        self.running.append(seq)

        history = seq.history
        self._log(
            EventKind.ADMIT,
            seq,
            f"prefill {len(history)} tokens"
            + (f", readmission #{seq.preemption_count}" if is_recompute else ""),
        )
        if is_recompute:
            self._log(
                EventKind.RECOMPUTE,
                seq,
                f"replaying {len(history)} tokens "
                f"({seq.request.prompt_len} prompt + {len(seq.generated)} generated)",
            )

        logits = self.executor.forward(history, list(range(len(history))), seq.backend)
        self._append_token(seq, logits)

    def _append_token(self, seq: Sequence, logits: torch.Tensor) -> None:
        seq.generated.append(int(torch.argmax(logits).item()))
        if seq.is_finished:
            self._finish(seq)

    def _finish(self, seq: Sequence) -> None:
        """Leave the running set and return blocks immediately, mid-step."""
        self.running.remove(seq)
        freed = self.allocator.free_sequence(seq.seq_id)
        seq.seq_id = None
        seq.backend = None
        seq.status = SequenceStatus.FINISHED
        self.finished[seq.request_id] = seq
        self._log(EventKind.FINISH, seq, f"{len(seq.generated)} tokens, freed {len(freed)} blocks")

    def _evict(self, seq: Sequence) -> None:
        """Preempt a running sequence: free its blocks, requeue it in FCFS order."""
        self.running.remove(seq)
        freed = self.allocator.free_sequence(seq.seq_id)
        seq.seq_id = None
        seq.backend = None
        seq.status = SequenceStatus.WAITING
        seq.preemption_count += 1
        self._log(
            EventKind.PREEMPT,
            seq,
            f"freed {len(freed)} blocks, {len(seq.generated)} generated tokens to replay",
        )
        # Back into the queue by arrival order, so FCFS is unaffected by having
        # been preempted. This is also why preemption_count cannot influence
        # position: order comes from arrival_index alone.
        self.waiting.append(seq)
        self.waiting = deque(sorted(self.waiting, key=lambda s: s.arrival_index))

    def _make_room_for(self, seq: Sequence) -> list[Sequence]:
        """Evict newest-first until `seq` can grow, or until `seq` is the victim.

        Returns the victims. The victim set is never the entire running set: the
        admission guard bounds every admitted sequence at
        num_gpu_blocks - admission_margin_blocks, so once everything else has
        been evicted the survivor necessarily fits and stops the loop.
        """
        running_before = len(self.running)
        victims: list[Sequence] = []

        while not self._decode_fits(seq):
            if not self.running:
                raise AssertionError(
                    f"{seq.request_id}: nothing left to evict but still cannot "
                    "allocate; the admission guard should have made this impossible"
                )
            victim = max(self.running, key=lambda s: s.arrival_index)
            victims.append(victim)
            self._evict(victim)
            if victim is seq:
                break

        assert len(victims) < running_before, (
            f"eviction victim set ({len(victims)}) is the entire running set "
            f"({running_before}); the admission guard failed to bound request sizes"
        )
        return victims

    def _decode(self, seq: Sequence) -> None:
        """One token for an already-resident sequence."""
        last_token = seq.generated[-1]
        # That token occupies the position right after everything already cached.
        position = seq.history_len - 1
        logits = self.executor.forward([last_token], [position], seq.backend)
        self._append_token(seq, logits)

    def step(self) -> StepResult:
        """One scheduling iteration."""
        self._step += 1

        admitted = self._admit()
        admitted_ids = {id(seq) for seq in admitted}

        decoded: list[str] = []
        preempted: list[str] = []
        finished_before = set(self.finished)

        # Snapshot: the running set is mutated mid-loop by finishes and evictions.
        for seq in list(self.running):
            if id(seq) in admitted_ids:
                continue  # already produced its token during prefill this step
            if seq.status is not SequenceStatus.RUNNING:
                continue  # finished or evicted earlier in this same step

            if not self._decode_fits(seq):
                victims = self._make_room_for(seq)
                preempted.extend(victim.request_id for victim in victims)
                if seq.status is not SequenceStatus.RUNNING:
                    continue  # it was its own victim

            self._decode(seq)
            decoded.append(seq.request_id)

        return StepResult(
            step=self._step,
            admitted=tuple(seq.request_id for seq in admitted),
            decoded=tuple(decoded),
            finished=tuple(sorted(set(self.finished) - finished_before)),
            preempted=tuple(preempted),
            running=len(self.running),
            waiting=len(self.waiting),
            free_blocks=self.allocator.num_free,
        )

    def run_to_completion(self, max_steps: int = 100_000) -> dict[str, tuple[int, ...]]:
        """Step until nothing is waiting or running. Returns generated tokens.

        `max_steps` is a livelock backstop, not a policy. If it trips, the
        admission guard has failed and the scheduler is making no progress.
        """
        steps = 0
        while self.waiting or self.running:
            self.step()
            steps += 1
            if steps > max_steps:
                raise RuntimeError(
                    f"no progress after {max_steps} steps: "
                    f"{len(self.waiting)} waiting, {len(self.running)} running"
                )
        return {
            request_id: tuple(seq.generated) for request_id, seq in self.finished.items()
        }
