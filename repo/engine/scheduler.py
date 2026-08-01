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

STEP 5B: TWO EXECUTION MODES, one set of decisions. `ExecutionMode.UNFUSED` is
Step 4's path -- one forward per sequence per step -- and it is kept as the
reference, not as a fallback. `ExecutionMode.FUSED` is the shipping path and
runs the whole step in one batched forward. Which sequences run, which are
admitted and which are evicted is decided by the same code either way; the modes
differ only in *when* the forwards happen, which is `_submit` and `_flush`.

That timing difference has one visible consequence, and it is not hidden. In
UNFUSED mode a sequence that finishes returns its blocks before the next
sequence in the same step is considered, so a later sequence can reuse them
immediately. In FUSED mode there is one forward and every member of the batch
holds its blocks for the whole of it, so reclamation cannot happen until the
step ends. Under memory pressure FUSED therefore admits slightly later and
preempts slightly more. It does not change any sequence's output -- recompute is
transparent, which is the Step 5 gate -- and
`tests/test_sched.py::test_fused_defers_mid_step_reclamation` pins the
difference to a three-block fixture rather than leaving it as prose.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, Sequence as TypingSequence

import torch

from engine.attention import AttentionBackend, BatchInput
from engine.cache import ADMISSION_MARGIN_BLOCKS, AllocResult, BlockAllocator


def admission_limit_blocks(num_blocks: int, admission_margin_blocks: int) -> int:
    """The most blocks any single request may ever be allowed to hold."""
    return num_blocks - admission_margin_blocks


def is_admissible(
    max_total_len: int,
    allocator: BlockAllocator,
    admission_margin_blocks: int = ADMISSION_MARGIN_BLOCKS,
) -> bool:
    """THE admission guard, as a predicate anything can ask.

    Module-level rather than a method because `bench/baseline.py` has to reject
    exactly what the scheduler rejects, and the only way to make "identically"
    literally true is for both to call this. `Scheduler.add_request` does.
    """
    return allocator.blocks_needed(max_total_len) <= admission_limit_blocks(
        allocator.num_blocks, admission_margin_blocks
    )


class EventKind(Enum):
    ADMIT = "admit"
    FINISH = "finish"
    PREEMPT = "preempt"
    RECOMPUTE = "recompute"
    REJECT = "reject"


class ExecutionMode(Enum):
    """How a step's forwards are issued. Not a policy -- both are supported.

    UNFUSED  One forward per sequence per step. Step 4's path. Every sequence's
             arithmetic is bit-for-bit what it would be running alone, which is
             what makes concurrent-equals-sequential an equality rather than a
             tolerance. Kept permanently as the reference the fused path is
             checked against.
    FUSED    One batched forward per step, whatever the running-set size. Step
             5b, and the path the throughput number is measured on. Padding and
             batched reductions mean it agrees with UNFUSED to a tolerance rather
             than exactly, so UNFUSED remains the thing equality is asserted
             against.
    """

    UNFUSED = "unfused"
    FUSED = "fused"


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
    # Tokens actually stored across every running sequence at this moment.
    #
    # `free_blocks` reports CAPACITY -- what the pool has handed out. This is
    # OCCUPANCY -- what is in it. The gap between them is internal fragmentation,
    # and recording both here is what lets bench/plot.py draw a *measured*
    # fragmentation plot rather than only the analytical one derived from the
    # length distribution. Analytical explains why; this proves it happened.
    tokens_held: int = 0


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

    # Populated only when the scheduler is built with record_logits=True.
    # `step_logits[k]` is the logit vector that `generated[k]` was taken from,
    # which stays aligned across preemption because a recompute appends exactly
    # one token, just as a decode does. That alignment is what lets the fused and
    # unfused runs be compared per token when their step boundaries differ.
    step_logits: list[torch.Tensor] = field(default_factory=list)

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
class ScheduledWork:
    """One sequence's work for one step: the tokens to run and where they sit."""

    seq: Sequence
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    is_prefill: bool

    def as_input(self) -> BatchInput:
        return BatchInput(
            token_ids=self.token_ids, positions=self.positions, backend=self.seq.backend
        )


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
    # Sequences that actually ran a forward this step. In FUSED mode this is the
    # width of the one batched call; in UNFUSED mode it is how many separate calls
    # were made. A sequence evicted after being planned but before the flush is
    # not counted, because it did not run.
    batch_size: int = 0
    # Sequences that were ready to decode but did not fit this step's token
    # budget. Not preempted -- they keep their blocks and run next step.
    deferred: tuple[str, ...] = ()
    # Padded token slots the step's forward materialised. See
    # `Scheduler.batched_token_cost`.
    batched_token_cost: int = 0


class Executor(Protocol):
    """What the scheduler needs from a model. Narrow so tests can stub it.

    Reservation is separate from execution because capacity is the scheduler's
    decision: it has to know a sequence's blocks are taken before it decides what
    else fits in the same step, and in FUSED mode nothing runs until the step is
    fully planned.
    """

    def new_backend(self, seq_id: int) -> AttentionBackend: ...

    def reserve(self, backend: AttentionBackend, num_tokens: int) -> AllocResult: ...

    def forward(
        self, token_ids: tuple[int, ...], positions: tuple[int, ...], backend: AttentionBackend
    ) -> torch.Tensor: ...

    def forward_batch(self, inputs: TypingSequence[BatchInput]) -> torch.Tensor: ...


class Qwen2Executor:
    """Drives `Qwen2Engine` over a paged cache for the scheduler."""

    def __init__(self, engine, paged_cache, allocator: BlockAllocator) -> None:
        self.engine = engine
        self.cache = paged_cache
        self.allocator = allocator

    def new_backend(self, seq_id: int) -> AttentionBackend:
        return self.engine.new_paged_backend(self.cache, self.allocator, seq_id)

    def reserve(self, backend: AttentionBackend, num_tokens: int) -> AllocResult:
        return backend.begin_forward(num_tokens)

    def forward(
        self, token_ids: tuple[int, ...], positions: tuple[int, ...], backend: AttentionBackend
    ) -> torch.Tensor:
        device = self.engine.device
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        pos = torch.tensor(positions, dtype=torch.long, device=device)
        return self.engine.forward_prereserved(ids, pos, backend)

    def forward_batch(self, inputs: TypingSequence[BatchInput]) -> torch.Tensor:
        return self.engine.forward_batch_prereserved(inputs)


class Scheduler:
    """Iteration-level scheduler over a single block pool, in two execution modes.

    `ExecutionMode.UNFUSED` runs one forward per sequence per step, so each
    sequence's arithmetic is bit-for-bit what it would be running alone -- which
    is what makes the concurrent-equals-sequential test an equality rather than a
    tolerance. It is the reference and is never removed.

    `ExecutionMode.FUSED` runs the whole step in a single batched forward, which
    is where the throughput comes from. It trades that equality for a declared
    tolerance, so it does not replace the reference.

    Both modes share every decision in this class. See the module docstring for
    the one behavioural consequence of the difference.
    """

    def __init__(
        self,
        executor: Executor,
        allocator: BlockAllocator,
        admission_margin_blocks: int = ADMISSION_MARGIN_BLOCKS,
        clock: Callable[[], float] = time.monotonic,
        mode: ExecutionMode = ExecutionMode.FUSED,
        record_logits: bool = False,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        if admission_margin_blocks < 0:
            raise ValueError(
                f"admission_margin_blocks must be non-negative, got {admission_margin_blocks}"
            )
        self.executor = executor
        self.allocator = allocator
        self.admission_margin_blocks = admission_margin_blocks
        self.clock = clock
        self.mode = mode
        self.record_logits = record_logits
        # Two capacity caps, the vLLM pattern. Both are capacity conditions on
        # the same strict-FCFS order -- neither reorders the queue and neither
        # lets anything jump the head. None means unbounded, which is what the
        # scheduler did before and is kept so existing fixtures are unchanged.
        if max_num_seqs is not None and max_num_seqs < 1:
            raise ValueError(f'max_num_seqs must be at least 1, got {max_num_seqs}')
        if max_num_batched_tokens is not None and max_num_batched_tokens < 1:
            raise ValueError(
                f'max_num_batched_tokens must be at least 1, got {max_num_batched_tokens}'
            )
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: dict[str, Sequence] = {}
        self.rejected: dict[str, Sequence] = {}
        self.events: list[Event] = []

        self._step = 0
        self._next_arrival = 0
        self._next_seq_id = 0

        # This step's planned and completed work. `_pending` is only ever
        # non-empty in FUSED mode, between planning and the flush.
        self._pending: list[ScheduledWork] = []
        self._ran: list[ScheduledWork] = []
        self._last_token_cost = 0

    # --- capacity ----------------------------------------------------------

    @property
    def max_admissible_blocks(self) -> int:
        """The largest a request may ever become and still be accepted."""
        return admission_limit_blocks(
            self.allocator.num_blocks, self.admission_margin_blocks
        )

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

    def _tokens_held(self) -> int:
        """Tokens stored in the pool right now, across every running sequence.

        Every call site logs after the running set and the allocator agree --
        `_evict` and `_finish` both remove the sequence and free its blocks before
        logging, and `_start` appends before the sequence has reserved anything --
        so this never asks the allocator about a sequence it has dropped.
        """
        return sum(
            self.allocator.num_tokens(seq.seq_id)
            for seq in self.running
            if seq.seq_id is not None
        )

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
                tokens_held=self._tokens_held(),
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
        if not is_admissible(
            request.max_total_len, self.allocator, self.admission_margin_blocks
        ):
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

    # --- capacity caps -----------------------------------------------------

    def _seq_cap_reached(self) -> bool:
        """True once the running set is at `max_num_seqs`."""
        return (
            self.max_num_seqs is not None
            and len(self.running) >= self.max_num_seqs
        )

    def batched_token_cost(self, extra_len: int | None = None) -> int:
        """Token slots this step's forward will MATERIALISE, not merely contain.

        The distinction is the whole point. In FUSED mode the batch is padded to
        its widest member, so N sequences of one token each sharing a step with a
        2048-token prefill cost (N + 1) * 2048 slots, not N + 2048. Bounding the
        *real* token count -- which is what vLLM's max_num_batched_tokens does,
        because its kernels are varlen and never pad -- would not bound this
        engine's memory at all. So the cap here is on the padded rectangle, and
        that deviation is deliberate.

        In UNFUSED mode each sequence runs alone, so the cost is the plain sum.
        """
        lengths = [len(work.token_ids) for work in self._pending]
        if extra_len is not None:
            lengths.append(extra_len)
        if not lengths:
            return 0
        if self.mode is ExecutionMode.FUSED:
            return len(lengths) * max(lengths)
        return sum(lengths)

    def _token_budget_allows(self, extra_len: int) -> bool:
        """Room for one more row of `extra_len` tokens in this step's forward.

        An EMPTY batch always admits the head, whatever its length. Without that
        escape a request longer than the cap could never run at any point, which
        is a livelock -- the same failure the admission guard exists to prevent,
        arriving by a different route.
        """
        if self.max_num_batched_tokens is None or not self._pending:
            return True
        return self.batched_token_cost(extra_len) <= self.max_num_batched_tokens

    def _admit(self) -> list[Sequence]:
        """Admit from the head while it fits. Strict FCFS, no backfilling.

        Three capacity conditions now, all applied to the same FCFS order and
        none of them reordering it: blocks must be free, the running set must be
        under `max_num_seqs`, and the step's padded forward must stay under
        `max_num_batched_tokens`. A head that fails any of them stops the step
        and keeps its place; nothing behind it is considered.
        """
        admitted: list[Sequence] = []
        while self.waiting:
            head = self.waiting[0]
            if self._seq_cap_reached():
                break
            if not self._admission_fits(head):
                # The head keeps its place and nothing behind it is considered.
                break
            if not self._token_budget_allows(head.history_len):
                break
            self.waiting.popleft()
            self._start(head)
            admitted.append(head)
        return admitted

    def _start(self, seq: Sequence) -> None:
        """Register with the allocator and plan a prefill of the whole history."""
        is_recompute = bool(seq.generated)
        seq.seq_id = self._next_seq_id
        self._next_seq_id += 1
        seq.backend = self.executor.new_backend(seq.seq_id)
        seq.status = SequenceStatus.RUNNING
        self.running.append(seq)

        history = tuple(seq.history)
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

        self._reserve(seq, len(history))
        self._submit(
            ScheduledWork(
                seq=seq,
                token_ids=history,
                positions=tuple(range(len(history))),
                is_prefill=True,
            )
        )

    # --- execution ---------------------------------------------------------

    def _reserve(self, seq: Sequence, num_tokens: int) -> None:
        """Take the blocks this work needs, now that the capacity check passed.

        Reserving here rather than inside the forward is what lets the next
        sequence in the same step see the blocks as taken, in both modes. A
        failure is a bug in the capacity checks above, not an expected outcome, so
        it is an assertion and not a preemption.
        """
        reservation = self.executor.reserve(seq.backend, num_tokens)
        if not reservation.ok:
            raise AssertionError(
                f"{seq.request_id}: reserving {num_tokens} token(s) failed with "
                f"{reservation.status.value} after the capacity check passed"
            )

    def _submit(self, work: ScheduledWork) -> None:
        """Queue this step's work, and in UNFUSED mode run it immediately.

        Running immediately is what preserves Step 4's interleaving exactly: a
        sequence that finishes here returns its blocks before the next sequence in
        the same step is considered. In FUSED mode the work waits for `_flush`,
        because there is one forward and every sequence in it holds its blocks for
        the duration.
        """
        if self.mode is ExecutionMode.FUSED:
            self._pending.append(work)
            return

        self._last_token_cost += len(work.token_ids)
        logits = self.executor.forward(work.token_ids, work.positions, work.seq.backend)
        self._ran.append(work)
        self._append_token(work.seq, logits)

    def _flush(self) -> None:
        """FUSED mode: one batched forward for the whole step, then take tokens."""
        if self.mode is not ExecutionMode.FUSED or not self._pending:
            return

        batch = self._pending
        self._last_token_cost = self.batched_token_cost()
        self._pending = []
        logits = self.executor.forward_batch(tuple(work.as_input() for work in batch))
        if logits.shape[0] != len(batch):
            raise AssertionError(
                f"fused forward returned {logits.shape[0]} rows for a batch of {len(batch)}"
            )

        self._ran.extend(batch)
        for row, work in enumerate(batch):
            self._append_token(work.seq, logits[row])

    def _append_token(self, seq: Sequence, logits: torch.Tensor) -> None:
        if self.record_logits:
            seq.step_logits.append(logits.detach().cpu().clone())
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
        # In FUSED mode this sequence may already be planned into the step's batch.
        # It is losing the blocks that batch would read, so the work goes with it.
        # In UNFUSED mode its forward has already happened and `_pending` is empty.
        self._pending = [work for work in self._pending if work.seq is not seq]
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

    def _plan_decode(self, seq: Sequence) -> None:
        """Plan one token for an already-resident sequence."""
        last_token = seq.generated[-1]
        # That token occupies the position right after everything already cached.
        position = seq.history_len - 1
        self._reserve(seq, 1)
        self._submit(
            ScheduledWork(
                seq=seq,
                token_ids=(last_token,),
                positions=(position,),
                is_prefill=False,
            )
        )

    def step(self) -> StepResult:
        """One scheduling iteration.

        The decisions below are identical in both execution modes. What differs is
        that `_submit` runs each piece of work immediately in UNFUSED mode and
        defers it to the single `_flush` forward in FUSED mode.
        """
        self._step += 1
        self._pending = []
        self._ran = []
        self._last_token_cost = 0

        admitted = self._admit()
        admitted_ids = {id(seq) for seq in admitted}

        preempted: list[str] = []
        deferred: list[str] = []
        finished_before = set(self.finished)

        # Snapshot: the running set is mutated mid-loop by finishes and evictions.
        for seq in list(self.running):
            if id(seq) in admitted_ids:
                continue  # its prefill already covers this step's token
            if seq.status is not SequenceStatus.RUNNING:
                continue  # finished or evicted earlier in this same step

            # The token budget binds decodes too, and it has to. A decode adds
            # one real token but a whole ROW of the padded rectangle, so 60
            # decodes joining a 2048-token prefill cost 61 * 2048 slots -- which
            # is exactly the shape that ran a 15 GB card out of memory. A
            # sequence refused here is not preempted and keeps its blocks; it
            # simply decodes on the next step.
            if not self._token_budget_allows(1):
                deferred.append(seq.request_id)
                continue

            if not self._decode_fits(seq):
                victims = self._make_room_for(seq)
                preempted.extend(victim.request_id for victim in victims)
                if seq.status is not SequenceStatus.RUNNING:
                    continue  # it was its own victim

            self._plan_decode(seq)

        self._flush()

        return StepResult(
            step=self._step,
            admitted=tuple(seq.request_id for seq in admitted),
            # Only work that actually ran. A sequence planned and then evicted by a
            # later sequence's `_make_room_for` never reaches `_ran`.
            decoded=tuple(work.seq.request_id for work in self._ran if not work.is_prefill),
            finished=tuple(sorted(set(self.finished) - finished_before)),
            preempted=tuple(preempted),
            running=len(self.running),
            waiting=len(self.waiting),
            free_blocks=self.allocator.num_free,
            batch_size=len(self._ran),
            deferred=tuple(deferred),
            batched_token_cost=self._last_token_cost,
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
