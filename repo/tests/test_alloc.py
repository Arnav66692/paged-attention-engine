"""Step 2: block allocator invariants under randomized allocate and free.

The whole paging design rests on one invariant: every physical block is in
exactly one place at any moment, either the free list or exactly one sequence's
block table. If that can be violated, two sequences read each other's KV and
the corruption surfaces far away from the cause. So it is checked after every
single operation of a randomized sequence, not just at the end.

These tests load no model and touch no tensors, so they run in milliseconds.
"""

import random

import pytest

from bench import workload
from engine.cache import (
    ADMISSION_MARGIN_BLOCKS,
    BLOCK_SIZE,
    NUM_GPU_BLOCKS,
    AllocStatus,
    BlockAllocator,
    CacheSpec,
    ContiguousKVCache,
)

SEED = 0
RANDOM_OPS = 4000

# Small enough that random operation sequences actually exhaust it, which is the
# only way the out-of-blocks path gets exercised.
SMALL_POOL = 24


def _snapshot(allocator: BlockAllocator):
    """Everything observable about the pool, for before/after comparison."""
    return (
        allocator.free_block_ids(),
        {sid: allocator.block_table(sid) for sid in allocator.sequence_ids()},
        {sid: allocator.num_tokens(sid) for sid in allocator.sequence_ids()},
    )


def _assert_invariants(allocator: BlockAllocator, label: str) -> None:
    tables = {sid: allocator.block_table(sid) for sid in allocator.sequence_ids()}
    held = [block for ids in tables.values() for block in ids]
    free = allocator.free_block_ids()

    # Every physical block sits in at most one block table.
    assert len(held) == len(set(held)), f"{label}: a physical block is in two block tables"
    assert not (set(held) & set(free)), f"{label}: a block is both held and free"
    assert len(free) == len(set(free)), f"{label}: duplicate entry in the free list"

    # len(free_list) + allocated == num_gpu_blocks, after every operation.
    assert len(free) + len(held) == allocator.num_blocks, f"{label}: blocks leaked or invented"
    assert allocator.num_free + allocator.num_allocated == allocator.num_blocks, label
    assert set(free) | set(held) == set(range(allocator.num_blocks)), f"{label}: pool not whole"

    # A table holds exactly as many blocks as its token count requires.
    for seq_id, ids in tables.items():
        expected = allocator.blocks_needed(allocator.num_tokens(seq_id))
        assert len(ids) == expected, (
            f"{label}: sequence {seq_id} holds {len(ids)} blocks for "
            f"{allocator.num_tokens(seq_id)} tokens, expected {expected}"
        )


def test_engine_constants_match_bench():
    """bench/workload.py is frozen and restates these, so the check lives here."""
    assert workload.BLOCK_SIZE == BLOCK_SIZE
    assert workload.NUM_GPU_BLOCKS == NUM_GPU_BLOCKS
    # Locked Gate 1 values, asserted so a later edit to either file is caught.
    assert BLOCK_SIZE == 16
    assert NUM_GPU_BLOCKS == 3584
    assert ADMISSION_MARGIN_BLOCKS == 1


def test_fresh_pool_is_whole():
    allocator = BlockAllocator(num_blocks=SMALL_POOL, block_size=BLOCK_SIZE)
    assert allocator.num_free == SMALL_POOL
    assert allocator.num_allocated == 0
    _assert_invariants(allocator, "fresh")


def test_randomized_allocate_and_free():
    """Invariants hold after every operation of a long random sequence."""
    rng = random.Random(SEED)
    allocator = BlockAllocator(num_blocks=SMALL_POOL, block_size=BLOCK_SIZE)
    live: list[int] = []
    next_seq_id = 0

    failures = 0
    frees = 0
    appends = 0

    for step in range(RANDOM_OPS):
        choice = rng.random()

        if choice < 0.25 or not live:
            allocator.create_sequence(next_seq_id)
            live.append(next_seq_id)
            next_seq_id += 1
            _assert_invariants(allocator, f"step {step} after create")

        elif choice < 0.80:
            seq_id = rng.choice(live)
            count = rng.randint(1, 2 * BLOCK_SIZE)
            before = _snapshot(allocator)
            result = allocator.append_tokens(seq_id, count)
            if result.ok:
                appends += 1
                # Newly taken blocks must be new to this table and to everyone.
                assert len(set(result.new_block_ids)) == len(result.new_block_ids)
            else:
                failures += 1
                assert result.status is AllocStatus.OUT_OF_BLOCKS
                assert result.new_block_ids == ()
                # Allocation failure leaves the pool untouched.
                assert _snapshot(allocator) == before, (
                    f"step {step}: failed allocation mutated the pool"
                )
            _assert_invariants(allocator, f"step {step} after append")

        else:
            seq_id = rng.choice(live)
            expected = allocator.block_table(seq_id)
            freed = allocator.free_sequence(seq_id)
            live.remove(seq_id)
            frees += 1
            # Freeing a sequence returns exactly the blocks it held.
            assert freed == expected, f"step {step}: free returned {freed}, held {expected}"
            _assert_invariants(allocator, f"step {step} after free")

    print(
        f"\nrandomized: {RANDOM_OPS} ops on a {SMALL_POOL}-block pool -- "
        f"{appends} appends, {frees} frees, {failures} out-of-block signals"
    )
    # The run is only meaningful if it actually hit the exhaustion path.
    assert failures > 0, "pool never filled; the out-of-blocks path went untested"
    assert frees > 0


def test_free_returns_exactly_the_blocks_held():
    allocator = BlockAllocator(num_blocks=SMALL_POOL, block_size=BLOCK_SIZE)
    allocator.create_sequence(0)
    allocator.create_sequence(1)
    allocator.append_tokens(0, 3 * BLOCK_SIZE)
    allocator.append_tokens(1, BLOCK_SIZE + 1)

    held_zero = allocator.block_table(0)
    held_one = allocator.block_table(1)
    assert len(held_zero) == 3
    assert len(held_one) == 2
    assert not set(held_zero) & set(held_one)

    freed = allocator.free_sequence(0)
    assert freed == held_zero
    assert set(freed).isdisjoint(allocator.block_table(1))
    assert set(freed).issubset(set(allocator.free_block_ids()))
    _assert_invariants(allocator, "after partial free")

    # The surviving sequence is untouched by its neighbour's free.
    assert allocator.block_table(1) == held_one


def test_allocation_failure_is_a_signal_not_an_exception():
    """A full pool returns OUT_OF_BLOCKS and leaves everything exactly as it was."""
    allocator = BlockAllocator(num_blocks=2, block_size=BLOCK_SIZE)
    allocator.create_sequence(0)
    assert allocator.append_tokens(0, 2 * BLOCK_SIZE).ok
    assert allocator.num_free == 0

    before = _snapshot(allocator)
    result = allocator.append_token(0)  # would need a third block

    assert result.status is AllocStatus.OUT_OF_BLOCKS
    assert not result.ok
    assert result.new_block_ids == ()
    assert _snapshot(allocator) == before
    assert allocator.num_tokens(0) == 2 * BLOCK_SIZE
    _assert_invariants(allocator, "after refused append")


def test_bulk_allocation_failure_is_all_or_nothing():
    """A request larger than the pool takes no blocks at all, not a partial run."""
    allocator = BlockAllocator(num_blocks=4, block_size=BLOCK_SIZE)
    allocator.create_sequence(0)
    before = _snapshot(allocator)

    result = allocator.append_tokens(0, 5 * BLOCK_SIZE)  # needs 5, pool has 4

    assert result.status is AllocStatus.OUT_OF_BLOCKS
    assert _snapshot(allocator) == before
    assert allocator.num_free == 4
    assert allocator.block_table(0) == ()
    _assert_invariants(allocator, "after refused bulk append")


def test_freed_blocks_are_reusable():
    """Capacity actually comes back, so exhaustion is recoverable."""
    allocator = BlockAllocator(num_blocks=2, block_size=BLOCK_SIZE)
    allocator.create_sequence(0)
    allocator.append_tokens(0, 2 * BLOCK_SIZE)
    assert not allocator.append_token(0).ok

    allocator.create_sequence(1)
    assert not allocator.append_token(1).ok  # nothing spare

    allocator.free_sequence(0)
    assert allocator.append_tokens(1, 2 * BLOCK_SIZE).ok
    assert allocator.num_free == 0
    _assert_invariants(allocator, "after reuse")


def test_block_is_taken_exactly_when_the_last_one_fills():
    """One block per BLOCK_SIZE tokens, taken on the token that overflows."""
    allocator = BlockAllocator(num_blocks=8, block_size=BLOCK_SIZE)
    allocator.create_sequence(0)

    for token_index in range(4 * BLOCK_SIZE):
        result = allocator.append_token(0)
        assert result.ok
        crosses_boundary = token_index % BLOCK_SIZE == 0
        assert len(result.new_block_ids) == (1 if crosses_boundary else 0), (
            f"token {token_index}: took {len(result.new_block_ids)} blocks"
        )
        assert len(allocator.block_table(0)) == (token_index // BLOCK_SIZE) + 1
        _assert_invariants(allocator, f"token {token_index}")


def test_physical_slot_maps_logical_positions_into_blocks():
    allocator = BlockAllocator(num_blocks=SMALL_POOL, block_size=BLOCK_SIZE)
    # A decoy sequence first, so sequence 1's blocks are not trivially 0,1,2.
    allocator.create_sequence(0)
    allocator.append_tokens(0, BLOCK_SIZE)
    allocator.create_sequence(1)
    allocator.append_tokens(1, 2 * BLOCK_SIZE + 5)

    table = allocator.block_table(1)
    for position in range(allocator.num_tokens(1)):
        block_id, offset = allocator.physical_slot(1, position)
        assert block_id == table[position // BLOCK_SIZE]
        assert offset == position % BLOCK_SIZE
        assert 0 <= offset < BLOCK_SIZE

    with pytest.raises(IndexError):
        allocator.physical_slot(1, allocator.num_tokens(1))


def test_caller_bugs_still_raise():
    """Capacity is a signal; misuse is not. Those must stay distinguishable."""
    allocator = BlockAllocator(num_blocks=4, block_size=BLOCK_SIZE)
    with pytest.raises(KeyError):
        allocator.append_token(99)
    with pytest.raises(KeyError):
        allocator.free_sequence(99)
    allocator.create_sequence(0)
    with pytest.raises(KeyError):
        allocator.create_sequence(0)
    allocator.free_sequence(0)
    with pytest.raises(KeyError):
        allocator.free_sequence(0)  # double free


def test_full_pool_at_locked_size():
    """The real pool fills to exactly NUM_GPU_BLOCKS and refuses the next block."""
    allocator = BlockAllocator()
    assert allocator.num_blocks == NUM_GPU_BLOCKS
    assert allocator.block_size == BLOCK_SIZE

    allocator.create_sequence(0)
    assert allocator.append_tokens(0, NUM_GPU_BLOCKS * BLOCK_SIZE).ok
    assert allocator.num_free == 0
    assert allocator.num_allocated == NUM_GPU_BLOCKS
    assert not allocator.append_token(0).ok
    _assert_invariants(allocator, "full pool")

    assert allocator.free_sequence(0) == tuple(range(NUM_GPU_BLOCKS))
    assert allocator.num_free == NUM_GPU_BLOCKS
    _assert_invariants(allocator, "drained pool")


def test_contiguous_cache_still_works():
    """Step 3 checks paged attention against the contiguous path, so it must live."""
    import torch

    spec = CacheSpec(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = ContiguousKVCache(spec)
    assert cache.length == 0

    key = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    value = key + 100.0
    for layer in range(spec.num_layers):
        cache.write(layer, 0, key, value)
    cache.advance(3)

    assert cache.length == 3
    assert torch.equal(cache.keys(0, 3), key)
    assert torch.equal(cache.values(1, 3), value)
    assert cache.keys(0, 3).shape == (1, 2, 3, 4)

    cache.reset()
    assert cache.length == 0
