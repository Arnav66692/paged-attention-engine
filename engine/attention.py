"""Attention over a KV cache: two single-sequence backends and one fused batch.

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

Step 5b adds `FusedPagedAttention`, which is not a backend: it is one step's
whole running set attended in a single call. It reads KV through the
participating `PagedBackend`s rather than re-deriving addresses, so all three
paths translate logical positions to slots in exactly one place, and all three
reach the same `_sdpa`.

The gather is naive on purpose: every decode step re-reads the whole sequence
into a fresh contiguous buffer rather than attending to the blocks in place. It
is one indexing op rather than a Python loop over blocks, but it is still O(seq)
work per step that a fused kernel would avoid. Correctness first; no Triton.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

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


def _causal_shortcut(num_query_tokens: int, past_len: int) -> bool:
    """Whether SDPA's `is_causal` flag is the right masking for this pass.

    A multi-token query is only ever a prefill from an empty cache in the
    single-sequence backends, where q_len == kv_len and SDPA's top-left causal
    alignment is correct. Guarded rather than assumed, because a partial prefill
    into a warm cache would silently attend to the wrong positions instead of
    failing.

    `FusedPagedAttention` does not go through here: its batch is padded on both
    axes, so it builds the mask itself and has no alignment shortcut to protect.
    """
    is_causal = num_query_tokens > 1
    if is_causal and past_len != 0:
        raise NotImplementedError(
            f"multi-token forward at cache offset {past_len}: is_causal alignment "
            "only holds for a prefill from an empty cache"
        )
    return is_causal


def _sdpa(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    num_kv_groups: int,
    scale: float,
    *,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The one attention call every path in this file shares.

    Contiguous, paged and fused all arrive here with the same scale and the same
    GQA expansion, so a disagreement between them is masking or addressing and
    never the attention itself. Exactly one of `is_causal` and `attn_mask` is
    ever set: the single-sequence backends take the causal shortcut, and the
    fused batch supplies an explicit mask because padding makes the shortcut
    wrong.
    """
    return F.scaled_dot_product_attention(
        query,
        repeat_kv(keys, num_kv_groups),
        repeat_kv(values, num_kv_groups),
        attn_mask=attn_mask,
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
            is_causal=_causal_shortcut(query.shape[2], self._past_len),
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
    def past_len(self) -> int:
        """Tokens held before the reservation `begin_forward` most recently made.

        Read by `FusedPagedAttention`, which is built after every sequence in the
        step has reserved and therefore cannot recover this from the allocator.
        """
        return self._past_len

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

        return _sdpa(
            query,
            keys,
            values,
            self.num_kv_groups,
            self.scale,
            is_causal=_causal_shortcut(num_new, self._past_len),
        )

    def end_forward(self, num_tokens: int) -> None:
        # `begin_forward` already committed through the allocator, which owns the
        # token count. Nothing further to do.
        return None


# --- Fused batched attention ------------------------------------------------


@dataclass(frozen=True)
class BatchInput:
    """One sequence's contribution to a fused forward pass.

    `positions` must be the contiguous run
    [backend.past_len, backend.past_len + len(token_ids)). That is what both the
    rotary tables in `model.py` and the mask below assume, and it is checked
    rather than trusted -- a mismatch would attend to the wrong positions and
    still produce plausible-looking logits.
    """

    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    backend: AttentionBackend


class FusedPagedAttention:
    """A whole step's running set, attended in one call.

    Step 5b. The backends above run one forward per sequence per step. That is
    the reference and it is correct, but N sequences in a step cost N forwards,
    so concurrency buys nothing -- and Gate 1 is a throughput number. This fuses
    the step into one padded batch.

    It is not an `AttentionBackend`: a backend owns storage for one sequence,
    while this owns nothing and is rebuilt every step from the sequences taking
    part. Two things have to be right, and both are index arithmetic rather than
    new attention math.

    ADDRESSING is not re-derived here. Every slot index comes from the
    participating backend's own `write_slots` / `read_slots`, so the block table
    is consulted through exactly the code the unfused path uses and there is
    still only one place a logical position becomes a physical slot. A backend
    that ignored its block table would produce garbage here too.

    THE MASK. Sequences in a step have different lengths and a step mixes
    multi-token prefills with single-token decodes, so both the query axis and
    the key axis are padded. One rule covers every row:

        may_attend[i, r, j] = (j < kv_len[i]) and (j <= query_pos[i, r])

    For a prefill (past 0, query_pos = r) that is ordinary causal masking. For a
    decode (past P, query_pos = P, kv_len = P + 1) it admits the whole history.
    For a partial prefill into a warm cache -- which the scheduler never issues,
    but which the single-sequence backends have to refuse -- it is also correct,
    because query_pos carries the offset. For a padded query row query_pos is 0,
    so the row still admits exactly one key: masking a row to nothing makes its
    softmax NaN, and a NaN in a discarded row is still a NaN when something
    downstream checks the tensor.

    A padded row cannot contaminate a real one. Every other operation in the
    layer stack is position-local, and the scatter below writes only real tokens
    into the pool, so a pad row's K/V never exists.

    No bucketing. Grouping sequences by length would cut padding, but it would
    also make a step issue one call per bucket, which is exactly what a loop
    dressed up as fusion looks like from the outside. One call per step is the
    property `tests/test_fused.py` asserts, so it is the property this keeps.

    What that costs, stated rather than discovered later: the mask is
    batch x max_query_len x max_kv_len booleans, so a step that admits one long
    prompt alongside many decodes materialises a grid sized by the longest member
    on both axes. `tests/test_fused.py` prints the padding fraction for its
    fixture. Length bucketing, or a varlen kernel that needs no padding at all,
    are the ways out, and neither is done here.
    """

    def __init__(
        self,
        inputs: Sequence[BatchInput],
        num_kv_groups: int,
        scale: float,
    ) -> None:
        if not inputs:
            raise ValueError("a fused batch needs at least one sequence")

        backends: list[PagedBackend] = []
        for item in inputs:
            if not isinstance(item.backend, PagedBackend):
                raise TypeError(
                    f"fused attention needs PagedBackend, got {type(item.backend).__name__}; "
                    "contiguous storage holds one sequence by construction"
                )
            backends.append(item.backend)

        cache = backends[0].cache
        for backend in backends[1:]:
            if backend.cache is not cache:
                raise ValueError("every sequence in a fused batch must share one block pool")

        self.cache = cache
        self.backends = tuple(backends)
        self.num_kv_groups = num_kv_groups
        self.scale = scale

        device = cache.spec.device
        query_lens = [len(item.token_ids) for item in inputs]
        past_lens = [backend.past_len for backend in backends]
        kv_lens = [past + count for past, count in zip(past_lens, query_lens)]

        self.batch_size = len(inputs)
        self.query_lens = tuple(query_lens)
        self.kv_lens = tuple(kv_lens)
        self.max_query_len = max(query_lens)
        self.max_kv_len = max(kv_lens)

        read_rows = torch.zeros(
            (self.batch_size, self.max_kv_len), dtype=torch.long, device=device
        )
        # Pads stay 0, a real slot, so the gather never reads out of bounds. Its
        # contents are irrelevant: the mask discards them.
        query_pos = torch.zeros(
            (self.batch_size, self.max_query_len), dtype=torch.long, device=device
        )
        write_slots: list[torch.Tensor] = []
        write_rows: list[torch.Tensor] = []

        for row, item in enumerate(inputs):
            past, count, kv_len = past_lens[row], query_lens[row], kv_lens[row]
            expected = tuple(range(past, kv_len))
            if tuple(item.positions) != expected:
                raise NotImplementedError(
                    f"positions {tuple(item.positions)} are not the contiguous run "
                    f"{expected} this sequence's cache offset implies; the fused mask "
                    "and the rotary tables would disagree"
                )
            read_rows[row, :kv_len] = item.backend.read_slots(kv_len)
            write_slots.append(item.backend.write_slots(past, kv_len))
            write_rows.append(
                torch.arange(count, dtype=torch.long, device=device) + row * self.max_query_len
            )
            query_pos[row, :count] = torch.arange(
                past, kv_len, dtype=torch.long, device=device
            )

        self.read_slots = read_rows.reshape(-1)
        self.write_slots = torch.cat(write_slots)
        self.write_rows = torch.cat(write_rows)
        self.last_query_row = torch.tensor(
            [count - 1 for count in query_lens], dtype=torch.long, device=device
        )

        key_index = torch.arange(self.max_kv_len, dtype=torch.long, device=device)
        kv_len_column = torch.tensor(kv_lens, dtype=torch.long, device=device)[:, None, None]
        self.mask = (
            (key_index[None, None, :] < kv_len_column)
            & (key_index[None, None, :] <= query_pos[:, :, None])
        ).unsqueeze(1)

    @property
    def num_real_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def num_padded_tokens(self) -> int:
        """Query slots the batch computes and throws away. Reported, not hidden."""
        return self.batch_size * self.max_query_len - self.num_real_tokens

    def attend(
        self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Scatter this layer's new K/V, gather every sequence, attend once.

        `query` is (batch, num_heads, max_query_len, head_dim) and `key`/`value`
        are (batch, num_kv_heads, max_query_len, head_dim), padded rows included.
        """
        num_kv_heads, head_dim = key.shape[1], key.shape[3]
        flat_keys = self.cache.flat_keys(layer_idx)
        flat_values = self.cache.flat_values(layer_idx)

        # Only real tokens are stored. `write_rows` selects them out of the padded
        # (batch, max_query_len) grid in the same order `write_slots` names their
        # destinations, and those destinations are distinct because a physical
        # block belongs to exactly one sequence.
        new_keys = key.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)
        new_values = value.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)
        flat_keys[self.write_slots] = new_keys[self.write_rows]
        flat_values[self.write_slots] = new_values[self.write_rows]

        shape = (self.batch_size, self.max_kv_len, num_kv_heads, head_dim)
        keys = flat_keys[self.read_slots].view(shape).permute(0, 2, 1, 3).contiguous()
        values = flat_values[self.read_slots].view(shape).permute(0, 2, 1, 3).contiguous()

        return _sdpa(
            query, keys, values, self.num_kv_groups, self.scale, attn_mask=self.mask
        )

    def end_forward(self) -> None:
        for backend, count in zip(self.backends, self.query_lens):
            backend.end_forward(count)
