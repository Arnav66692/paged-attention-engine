"""Attention over a KV cache, in two interchangeable backends.

Gate 1, Step 3. `model.py` projects Q/K/V and applies rotary embeddings, then
hands them to a backend that owns everything about where KV lives:

  ContiguousBackend  Step 1's storage. Logical position p is physical slot p, so
                     the "gather" is a slice. Kept as the reference the paged
                     path is checked against, never deleted.
  PagedBackend       Step 3's storage. A block table maps logical block index to
                     a physical block id, so consecutive positions can live
                     anywhere in the pool. Gather is a real gather.

Both end in the same scaled_dot_product_attention call with the same scale and
the same causal handling, so a parity failure between them is a paging bug and
nothing else.

The gather is naive on purpose: every decode step re-reads the whole sequence
into a fresh contiguous buffer rather than attending to the blocks in place. It
is one indexing op rather than a Python loop over blocks, but it is still O(seq)
work per step that a fused kernel would avoid. Correctness first; no Triton.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from engine.cache import (
    AllocResult,
    AllocStatus,
    BlockAllocator,
    ContiguousKVCache,
    PagedKVCache,
)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to query heads for GQA. Equivalent to repeat_interleave on dim 1."""
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    expanded = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _sdpa(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    num_kv_groups: int,
    scale: float,
    past_len: int,
) -> torch.Tensor:
    """The one attention call both backends share.

    A multi-token query is only ever a prefill from an empty cache here, where
    q_len == kv_len and SDPA's top-left causal alignment is correct. Guarded
    rather than assumed, because a partial prefill into a warm cache would
    silently attend to the wrong positions instead of failing.
    """
    num_query_tokens = query.shape[2]
    is_causal = num_query_tokens > 1
    if is_causal and past_len != 0:
        raise NotImplementedError(
            f"multi-token forward at cache offset {past_len}: is_causal alignment "
            "only holds for a prefill from an empty cache"
        )
    return F.scaled_dot_product_attention(
        query,
        repeat_kv(keys, num_kv_groups),
        repeat_kv(values, num_kv_groups),
        attn_mask=None,
        dropout_p=0.0,
        scale=scale,
        is_causal=is_causal,
    )


class AttentionBackend(ABC):
    """Owns KV storage for one sequence and answers attention over it.

    A forward pass is `begin_forward`, then `attend` once per layer, then
    `end_forward`. Splitting begin from attend is what lets every layer write at
    the same offset before the sequence length moves.
    """

    def __init__(self, num_kv_groups: int, scale: float) -> None:
        self.num_kv_groups = num_kv_groups
        self.scale = scale

    @property
    @abstractmethod
    def length(self) -> int:
        """Tokens committed so far."""

    @abstractmethod
    def begin_forward(self, num_tokens: int) -> AllocResult:
        """Reserve room for `num_tokens`. May return OUT_OF_BLOCKS."""

    @abstractmethod
    def attend(
        self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Store this layer's new K/V, then attend over everything held so far."""

    @abstractmethod
    def end_forward(self, num_tokens: int) -> None:
        """Commit the forward pass."""


class ContiguousBackend(AttentionBackend):
    """Step 1's path: position p lives at slot p, so the gather is a slice."""

    def __init__(self, cache: ContiguousKVCache, num_kv_groups: int, scale: float) -> None:
        super().__init__(num_kv_groups, scale)
        self.cache = cache
        self._past_len = 0

    @property
    def length(self) -> int:
        return self.cache.length

    def begin_forward(self, num_tokens: int) -> AllocResult:
        self._past_len = self.cache.length
        if self._past_len + num_tokens > self.cache.spec.max_seq_len:
            return AllocResult(AllocStatus.OUT_OF_BLOCKS)
        return AllocResult(AllocStatus.OK)

    def attend(
        self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        total = self._past_len + key.shape[2]
        self.cache.write(layer_idx, self._past_len, key, value)
        return _sdpa(
            query,
            self.cache.keys(layer_idx, total),
            self.cache.values(layer_idx, total),
            self.num_kv_groups,
            self.scale,
            self._past_len,
        )

    def end_forward(self, num_tokens: int) -> None:
        self.cache.advance(num_tokens)


class PagedBackend(AttentionBackend):
    """Step 3's path: KV lives in scattered physical blocks named by a block table.

    Nothing in here may assume the block table is sorted or contiguous. The only
    thing that maps a logical position to memory is `_slot_indices`, and if that
    were bypassed -- reading the pool straight through from the sequence's first
    block -- a sequentially allocated sequence would still produce correct
    output. `tests/test_parity.py::test_parity_on_a_fragmented_pool` fragments
    the pool first so that shortcut produces garbage.
    """

    def __init__(
        self,
        cache: PagedKVCache,
        allocator: BlockAllocator,
        seq_id: int,
        num_kv_groups: int,
        scale: float,
    ) -> None:
        super().__init__(num_kv_groups, scale)
        if cache.spec.block_size != allocator.block_size:
            raise ValueError(
                f"cache block_size {cache.spec.block_size} != "
                f"allocator block_size {allocator.block_size}"
            )
        if cache.spec.num_blocks != allocator.num_blocks:
            raise ValueError(
                f"cache has {cache.spec.num_blocks} blocks but allocator manages "
                f"{allocator.num_blocks}"
            )
        self.cache = cache
        self.allocator = allocator
        self.seq_id = seq_id
        self.block_size = cache.spec.block_size
        self.device = cache.spec.device
        self._past_len = 0

    @property
    def length(self) -> int:
        return self.allocator.num_tokens(self.seq_id)

    @property
    def block_table(self) -> tuple[int, ...]:
        return self.allocator.block_table(self.seq_id)

    def begin_forward(self, num_tokens: int) -> AllocResult:
        self._past_len = self.allocator.num_tokens(self.seq_id)
        return self.allocator.append_tokens(self.seq_id, num_tokens)

    def _slot_indices(self, start: int, stop: int) -> torch.Tensor:
        """Physical slot index for each logical position in [start, stop).

        This is the address translation, and the only place it happens:
            slot = block_table[position // block_size] * block_size
                   + position % block_size
        """
        table = torch.tensor(self.block_table, dtype=torch.long, device=self.device)
        positions = torch.arange(start, stop, dtype=torch.long, device=self.device)
        logical_blocks = torch.div(positions, self.block_size, rounding_mode="floor")
        offsets = positions - logical_blocks * self.block_size
        return table[logical_blocks] * self.block_size + offsets

    def write_slots(self, start: int, stop: int) -> torch.Tensor:
        """Where this forward pass's new K/V go."""
        return self._slot_indices(start, stop)

    def read_slots(self, total: int) -> torch.Tensor:
        """Where the whole sequence currently lives.

        Split from `write_slots` so a test can subclass this alone and read the
        pool straight through, proving the fragmented-pool fixture would catch a
        block table that is written but never consulted on read.
        """
        return self._slot_indices(0, total)

    def attend(
        self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        num_new = key.shape[2]
        total = self._past_len + num_new

        # Write new K/V into the correct slot of the correct physical block.
        # (1, kv_heads, n, head_dim) -> (n, kv_heads, head_dim) to match the pool.
        write_slots = self.write_slots(self._past_len, total)
        self.cache.flat_keys(layer_idx)[write_slots] = key[0].transpose(0, 1)
        self.cache.flat_values(layer_idx)[write_slots] = value[0].transpose(0, 1)

        # Gather the whole sequence out of its blocks into a contiguous buffer.
        read_slots = self.read_slots(total)
        keys = self.cache.flat_keys(layer_idx)[read_slots]
        values = self.cache.flat_values(layer_idx)[read_slots]

        # (total, kv_heads, head_dim) -> (1, kv_heads, total, head_dim)
        keys = keys.permute(1, 0, 2).unsqueeze(0).contiguous()
        values = values.permute(1, 0, 2).unsqueeze(0).contiguous()

        return _sdpa(query, keys, values, self.num_kv_groups, self.scale, self._past_len)

    def end_forward(self, num_tokens: int) -> None:
        # `begin_forward` already committed through the allocator, which owns the
        # token count. Nothing further to do.
        return None
