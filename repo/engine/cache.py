"""KV cache storage: the contiguous single-sequence cache, and the block allocator.

Step 1 built `ContiguousKVCache` -- one contiguous K tensor and one contiguous V
tensor per layer, allocated up front for a single sequence. Nothing here is a
HuggingFace cache object and nothing here subclasses one; replacing that was the
point of that step.

Step 2 adds `BlockAllocator` beside it, not in place of it. The contiguous cache
stays as the reference implementation Step 3's paged attention is checked
against, so both must keep working.

The two differ in exactly one respect. The contiguous cache maps logical
position p to physical slot p. The allocator breaks that identity: a sequence
holds a block table mapping logical block index to a physical block id drawn
from a shared free list, so consecutive positions need not be adjacent in
memory. Everything else about the storage is unchanged.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

# --- LOCKED (Gate 1 spec) --------------------------------------------------
# These are the engine's own copies. bench/workload.py restates them because it
# predates engine/, and it is frozen; tests/test_alloc.py asserts the two agree
# so they cannot drift.
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 3584

# Locked, and not yet used: admission is a later step, and there is no scheduler
# at Step 2. Recorded here so the value lives with the rest of the cache config
# rather than being introduced from nowhere when admission lands.
ADMISSION_MARGIN_BLOCKS = 1


@dataclass(frozen=True)
class CacheSpec:
    """Shape and placement of the KV cache. All fields come from the model config."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_kv_heads", "head_dim", "max_seq_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")


class ContiguousKVCache:
    """Per-layer contiguous K/V storage for exactly one sequence.

    Each layer owns two tensors of shape (1, num_kv_heads, max_seq_len, head_dim).
    The leading 1 is the batch dimension that scaled_dot_product_attention wants;
    Step 1 is single-request, so it is always 1.

    Slots are assigned strictly in position order: token at position p lives in
    slot p of every layer. That one-to-one mapping between logical position and
    physical slot is exactly what the block table replaces at Step 2.
    """

    def __init__(self, spec: CacheSpec) -> None:
        self.spec = spec
        shape = (1, spec.num_kv_heads, spec.max_seq_len, spec.head_dim)
        self._keys = [
            torch.zeros(shape, dtype=spec.dtype, device=spec.device) for _ in range(spec.num_layers)
        ]
        self._values = [
            torch.zeros(shape, dtype=spec.dtype, device=spec.device) for _ in range(spec.num_layers)
        ]
        self._length = 0

    @property
    def length(self) -> int:
        """Number of positions currently committed."""
        return self._length

    def reset(self) -> None:
        """Forget the sequence. Storage is retained; stale slots are overwritten."""
        self._length = 0

    def write(self, layer_idx: int, start: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write `key`/`value` of shape (1, num_kv_heads, n, head_dim) at slot `start`.

        Writes are not committed until `advance` is called, so every layer writes
        at the same `start` for a given forward pass.
        """
        if not 0 <= layer_idx < self.spec.num_layers:
            raise IndexError(f"layer {layer_idx} out of range for {self.spec.num_layers} layers")
        n = key.shape[2]
        if value.shape[2] != n:
            raise ValueError(f"key has {n} positions but value has {value.shape[2]}")
        if start + n > self.spec.max_seq_len:
            raise ValueError(
                f"writing {n} positions at slot {start} exceeds max_seq_len {self.spec.max_seq_len}"
            )
        self._keys[layer_idx][:, :, start : start + n, :] = key
        self._values[layer_idx][:, :, start : start + n, :] = value

    def advance(self, n: int) -> None:
        """Commit `n` positions, after every layer has written them."""
        if n < 0:
            raise ValueError(f"cannot advance by {n}")
        if self._length + n > self.spec.max_seq_len:
            raise ValueError(
                f"advancing to {self._length + n} exceeds max_seq_len {self.spec.max_seq_len}"
            )
        self._length += n

    def keys(self, layer_idx: int, length: int) -> torch.Tensor:
        """A view of the first `length` key slots. No copy."""
        return self._keys[layer_idx][:, :, :length, :]

    def values(self, layer_idx: int, length: int) -> torch.Tensor:
        """A view of the first `length` value slots. No copy."""
        return self._values[layer_idx][:, :, :length, :]

    def storage_bytes(self) -> int:
        """Total bytes held across all layers, K and V."""
        per_tensor = self._keys[0].numel() * self._keys[0].element_size()
        return 2 * self.spec.num_layers * per_tensor


# --- Block allocator -------------------------------------------------------


class AllocStatus(Enum):
    """Outcome of an allocation request.

    Capacity exhaustion is a value, not an exception: the scheduler will ask for
    blocks on every decode step and a full pool is an ordinary, expected answer
    that it handles by preempting. Exceptions here are reserved for caller bugs
    -- an unknown sequence id, a double free -- which are not recoverable and
    should not be caught.
    """

    OK = "ok"
    OUT_OF_BLOCKS = "out_of_blocks"


@dataclass(frozen=True)
class AllocResult:
    """What an allocation did. `new_block_ids` is empty when nothing was needed."""

    status: AllocStatus
    new_block_ids: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is AllocStatus.OK


@dataclass
class BlockTable:
    """One sequence's logical-block-index to physical-block-id mapping.

    Mutated only by the `BlockAllocator` that issued it. Read it through the
    allocator's accessors, which hand back copies.
    """

    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0

    def __len__(self) -> int:
        return len(self.block_ids)


class BlockAllocator:
    """A free list of physical blocks and the per-sequence tables that hold them.

    Every physical block is in exactly one of two places at all times: the free
    list, or exactly one sequence's block table. That is the invariant the whole
    design rests on, and `tests/test_alloc.py` checks it after every operation.

    Freed blocks go to the back of the free list rather than the front. Reuse is
    therefore as delayed as possible, which makes a use-after-free show up as
    corrupted data in a test instead of being masked by handing the same block
    straight back out.
    """

    def __init__(self, num_blocks: int = NUM_GPU_BLOCKS, block_size: int = BLOCK_SIZE) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be at least 1, got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be at least 1, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._tables: dict[int, BlockTable] = {}

    # --- pool state --------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return sum(len(table) for table in self._tables.values())

    def free_block_ids(self) -> tuple[int, ...]:
        """Copy of the free list, in the order blocks would be handed out."""
        return tuple(self._free)

    def sequence_ids(self) -> tuple[int, ...]:
        return tuple(self._tables)

    def blocks_needed(self, num_tokens: int) -> int:
        """Blocks required to hold `num_tokens`, i.e. ceil division."""
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        return -(-num_tokens // self.block_size)

    # --- per-sequence lifecycle -------------------------------------------

    def create_sequence(self, seq_id: int) -> None:
        """Register an empty sequence. Holds no blocks until tokens are appended."""
        if seq_id in self._tables:
            raise KeyError(f"sequence {seq_id} already exists")
        self._tables[seq_id] = BlockTable()

    def block_table(self, seq_id: int) -> tuple[int, ...]:
        """Copy of the sequence's logical-to-physical mapping."""
        return tuple(self._tables[seq_id].block_ids)

    def num_tokens(self, seq_id: int) -> int:
        return self._tables[seq_id].num_tokens

    def append_token(self, seq_id: int) -> AllocResult:
        """Append one token, taking a block only if the last one just filled."""
        return self.append_tokens(seq_id, 1)

    def append_tokens(self, seq_id: int, count: int) -> AllocResult:
        """Append `count` tokens, taking whatever blocks that needs.

        All or nothing. If the free list cannot cover the whole request this
        returns OUT_OF_BLOCKS having touched nothing -- no partial allocation,
        no tokens recorded -- so a caller that retries later sees the pool
        exactly as it left it.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        table = self._tables[seq_id]

        required = self.blocks_needed(table.num_tokens + count) - len(table)
        if required > len(self._free):
            return AllocResult(AllocStatus.OUT_OF_BLOCKS)

        taken = tuple(self._free.popleft() for _ in range(required))
        table.block_ids.extend(taken)
        table.num_tokens += count
        return AllocResult(AllocStatus.OK, new_block_ids=taken)

    def free_sequence(self, seq_id: int) -> tuple[int, ...]:
        """Drop the whole table and return every block it held, in table order."""
        table = self._tables.pop(seq_id)
        freed = tuple(table.block_ids)
        self._free.extend(freed)
        return freed

    # --- address translation ----------------------------------------------

    def physical_slot(self, seq_id: int, position: int) -> tuple[int, int]:
        """Map a logical token position to (physical block id, offset in block)."""
        table = self._tables[seq_id]
        if not 0 <= position < table.num_tokens:
            raise IndexError(
                f"position {position} out of range for sequence {seq_id} "
                f"holding {table.num_tokens} tokens"
            )
        logical_block, offset = divmod(position, self.block_size)
        return table.block_ids[logical_block], offset


# --- Paged block storage ---------------------------------------------------


@dataclass(frozen=True)
class PagedCacheSpec:
    """Shape and placement of the physical block pool."""

    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_blocks", "block_size", "num_kv_heads", "head_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size


class PagedKVCache:
    """The physical block pool: per layer, one K tensor and one V tensor of blocks.

    Each is (num_blocks, block_size, num_kv_heads, head_dim). Block ids index the
    first axis, so blocks belonging to one sequence need not be adjacent -- that
    is the whole point, and it is what `BlockAllocator` hands out.

    `flat_keys` reshapes to (num_blocks * block_size, num_kv_heads, head_dim) so a
    token can be addressed by a single slot index, block_id * block_size + offset.
    The reshape is a view of contiguous storage, so it costs nothing, and it turns
    both the scatter of new KV and the gather of a sequence into one indexing op
    each rather than a Python loop over blocks.
    """

    def __init__(self, spec: PagedCacheSpec) -> None:
        self.spec = spec
        shape = (spec.num_blocks, spec.block_size, spec.num_kv_heads, spec.head_dim)
        self._keys = [
            torch.zeros(shape, dtype=spec.dtype, device=spec.device)
            for _ in range(spec.num_layers)
        ]
        self._values = [
            torch.zeros(shape, dtype=spec.dtype, device=spec.device)
            for _ in range(spec.num_layers)
        ]
        flat = (spec.num_slots, spec.num_kv_heads, spec.head_dim)
        # Views onto the same storage. The backing tensors are never reallocated,
        # so these stay valid for the life of the cache.
        self._flat_keys = [tensor.view(flat) for tensor in self._keys]
        self._flat_values = [tensor.view(flat) for tensor in self._values]

    def flat_keys(self, layer_idx: int) -> torch.Tensor:
        """Slot-addressed view of this layer's keys. Indexed by block*size+offset."""
        return self._flat_keys[layer_idx]

    def flat_values(self, layer_idx: int) -> torch.Tensor:
        return self._flat_values[layer_idx]

    def block_keys(self, layer_idx: int, block_id: int) -> torch.Tensor:
        """One physical block's keys, (block_size, num_kv_heads, head_dim)."""
        return self._keys[layer_idx][block_id]

    def block_values(self, layer_idx: int, block_id: int) -> torch.Tensor:
        return self._values[layer_idx][block_id]

    def storage_bytes(self) -> int:
        per_tensor = self._keys[0].numel() * self._keys[0].element_size()
        return 2 * self.spec.num_layers * per_tensor
