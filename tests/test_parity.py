"""Parity: our decode loop against HuggingFace, on both storage backends.

The engine owns its KV storage outright and runs a forward pass written out in
engine/model.py. This test asserts that swapping HuggingFace's decode path for
ours changes nothing observable:

  * all 32 generated token IDs are exactly equal, and
  * per-step logits agree within LOGIT_ATOL.

Both backends are held to the same expected logits from the same fixture:
`contiguous` (Step 1, position p at slot p) and `paged` (Step 3, KV scattered
across physical blocks named by a block table).

Prompt lengths are chosen against the locked block size of 16: a prompt shorter
than one block, an exact block multiple, and one that ends mid-block.

`test_parity_on_a_fragmented_pool` is the test that matters for Step 3.
Sequential allocation hands a sequence blocks 0,1,2,... so reading the pool
straight through from its first block gives the right answer anyway, and a paged
path that ignores its block table passes. That test fragments the pool first so
the sequence receives non-adjacent blocks in descending-then-ascending order,
and asserts the resulting block table really is non-monotonic so the fixture
cannot silently stop fragmenting.

Two independent baselines exist. `hf_greedy_cached` is HuggingFace driving its
own DynamicCache, the realistic path and the one parity is asserted against.
`hf_greedy_recompute` uses no cache at all and re-runs the full prefix every
step; it is O(n^2) and marked slow, so it is opt-in via `-m slow`. When both
run, the cross-check answers "is the engine wrong, or is the baseline wrong?".

Tolerance is fixed up front from the arithmetic, never fitted to an observed
result. The max observed deviation is printed for every configuration.
"""

from dataclasses import dataclass
from functools import lru_cache

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from engine.attention import PagedBackend
from engine.cache import BLOCK_SIZE, BlockAllocator
from engine.model import Qwen2Engine

DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Sliced to each length below, so every fixture is a prefix of one fixed text.
BASE_PROMPT = (
    "The capital of France is Paris, and the capital of Japan is Tokyo. "
    "Rivers shape the cities built beside them, carrying trade inland long "
    "before roads were paved, and the oldest bridges still mark where the "
    "first crossings were cut into the banks."
)

# Against the locked block size of 16: under one block, an exact multiple, and
# one token past a boundary. Asserted against BLOCK_SIZE so they cannot quietly
# stop meaning that.
PROMPT_LENGTHS = (13, 32, 33)
SEED = 0
NUM_GENERATED_TOKENS = 32

PATHS = ("contiguous", "paged")

# Comfortably more than the 5 blocks the longest fixture needs (33 + 32 tokens).
PAGED_POOL_BLOCKS = 64
# Small enough that fragmenting it is cheap, large enough to still hold the
# longest fixture after every block has been recycled.
FRAGMENTED_POOL_BLOCKS = 16

# Absolute tolerance on logits, in float32.
#
# Both sides run the same weights in the same dtype on the same device, so the
# only source of disagreement is the order of floating point reductions: our
# attention reads a slice of a preallocated cache tensor, or gathers scattered
# blocks into a fresh one, where HuggingFace concatenates. All three reach
# scaled_dot_product_attention with different memory layouts. float32 has ~1.2e-7
# relative resolution; across 24 residual layers that compounds to order 1e-5
# relative, and this model's logits run to roughly 1e1, giving expected deviation
# near 1e-4. 1e-3 is that with headroom. Unchanged since Step 1, and it is not
# widened if the test fails -- a failure is reported.
LOGIT_ATOL = 1e-3

# Greedy decoding is deterministic, so nothing here actually samples. The seed
# is set anyway so that any incidental nondeterminism would be pinned.
PARITY_DTYPE = torch.float32


@dataclass
class Runtime:
    """A loaded model and an engine sharing its weight tensors."""

    device: str
    model: object
    engine: Qwen2Engine


def _devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(scope="module", params=_devices())
def runtime(request) -> Runtime:
    """Load the model once per device for the whole module."""
    device = request.param
    model = AutoModelForCausalLM.from_pretrained(DEV_MODEL, dtype=PARITY_DTYPE)
    model.eval()
    model.to(device)
    return Runtime(device=device, model=model, engine=Qwen2Engine.from_hf_model(model))


@lru_cache(maxsize=None)
def _prompt_ids(length: int) -> tuple[int, ...]:
    """The first `length` token IDs of BASE_PROMPT."""
    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL)
    ids = tokenizer(BASE_PROMPT, add_special_tokens=False)["input_ids"]
    if len(ids) < length:
        raise ValueError(f"BASE_PROMPT is {len(ids)} tokens, need at least {length}")
    return tuple(ids[:length])


@torch.inference_mode()
def hf_greedy_cached(model, prompt_ids, num_steps, device):
    """HuggingFace greedy decode driving its own DynamicCache."""
    cache = DynamicCache(config=model.config)
    logits = model(
        input_ids=torch.tensor([list(prompt_ids)], device=device),
        past_key_values=cache,
        use_cache=True,
    ).logits[0, -1]

    tokens, step_logits = [], []
    for step in range(num_steps):
        step_logits.append(logits)
        token = int(torch.argmax(logits).item())
        tokens.append(token)
        if step + 1 == num_steps:
            break
        logits = model(
            input_ids=torch.tensor([[token]], device=device),
            past_key_values=cache,
            use_cache=True,
        ).logits[0, -1]
    return tuple(tokens), torch.stack(step_logits)


@torch.inference_mode()
def hf_greedy_recompute(model, prompt_ids, num_steps, device):
    """HuggingFace greedy decode with no cache: full prefix re-run every step."""
    sequence = list(prompt_ids)
    tokens, step_logits = [], []
    for _ in range(num_steps):
        logits = model(
            input_ids=torch.tensor([sequence], device=device), use_cache=False
        ).logits[0, -1]
        step_logits.append(logits)
        token = int(torch.argmax(logits).item())
        tokens.append(token)
        sequence.append(token)
    return tuple(tokens), torch.stack(step_logits)


_HF_REFERENCE: dict[tuple[str, int], tuple] = {}


def _hf_reference(runtime: Runtime, prompt_len: int):
    """The expected tokens and logits. Computed once per (device, prompt_len).

    Every backend and the fragmented-pool test are held to this same result, so
    "the same expected logits, the same fixture" is literally true.
    """
    key = (runtime.device, prompt_len)
    if key not in _HF_REFERENCE:
        _HF_REFERENCE[key] = hf_greedy_cached(
            runtime.model, _prompt_ids(prompt_len), NUM_GENERATED_TOKENS, runtime.device
        )
    return _HF_REFERENCE[key]


def _max_abs_deviation(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _make_backend(engine: Qwen2Engine, path: str, max_seq_len: int):
    if path == "contiguous":
        return engine.new_contiguous_backend(max_seq_len)
    if path == "paged":
        cache = engine.new_paged_cache(PAGED_POOL_BLOCKS)
        allocator = BlockAllocator(num_blocks=PAGED_POOL_BLOCKS, block_size=BLOCK_SIZE)
        return engine.new_paged_backend(cache, allocator, seq_id=0)
    raise ValueError(f"unknown path {path!r}")


def _fragment_pool(allocator: BlockAllocator, num_blocks: int) -> None:
    """Consume the whole pool with decoys, then free them from both ends inward.

    Freed blocks go to the back of the free list, so the free list ends up in
    exactly the order the decoys were released: [n-1, 0, n-2, 1, ...]. The next
    sequence therefore gets blocks that descend immediately and are never
    adjacent. Deterministic rather than shuffled, so the fixture cannot pass or
    fail depending on a seed.
    """
    decoy_base = 1_000_000
    for i in range(num_blocks):
        allocator.create_sequence(decoy_base + i)
        assert allocator.append_tokens(decoy_base + i, 1).ok, "pool too small to fragment"
    assert allocator.num_free == 0, "decoys did not consume the pool"

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
    assert allocator.num_free == num_blocks
    assert allocator.free_block_ids() == tuple(release_order)


def test_prompt_lengths_cover_block_boundaries():
    """The fixture lengths mean what the docstring claims, against BLOCK_SIZE."""
    sub_block, exact_multiple, mid_block = PROMPT_LENGTHS
    assert sub_block < BLOCK_SIZE
    assert exact_multiple % BLOCK_SIZE == 0
    assert mid_block % BLOCK_SIZE == 1


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("prompt_len", PROMPT_LENGTHS)
def test_parity_tokens_and_logits(runtime, prompt_len, path):
    """Both backends reproduce HuggingFace exactly on tokens."""
    torch.manual_seed(SEED)
    device = runtime.device
    prompt_ids = _prompt_ids(prompt_len)
    assert len(prompt_ids) == prompt_len

    hf_tokens, hf_logits = _hf_reference(runtime, prompt_len)
    backend = _make_backend(
        runtime.engine, path, prompt_len + NUM_GENERATED_TOKENS
    )
    result = runtime.engine.generate_greedy(
        list(prompt_ids), NUM_GENERATED_TOKENS, backend=backend
    )

    assert result.step_logits.shape[0] == NUM_GENERATED_TOKENS
    assert len(result.token_ids) == NUM_GENERATED_TOKENS
    # One fewer forward than tokens generated: the last token is read off the
    # final forward's logits and never fed back, so it occupies no cache slot.
    assert backend.length == prompt_len + NUM_GENERATED_TOKENS - 1

    deviation = _max_abs_deviation(result.step_logits, hf_logits)
    per_step = (result.step_logits.float() - hf_logits.float()).abs().amax(dim=-1)

    print(
        f"\n[{device}] path={path} prompt_len={prompt_len} "
        f"({prompt_len / BLOCK_SIZE:.2f} blocks), {NUM_GENERATED_TOKENS} steps"
    )
    print(f"[{device}]   tokens identical      : {result.token_ids == hf_tokens}")
    print(f"[{device}]   MAX ABS DEVIATION     : {deviation:.6e}")
    print(f"[{device}]   tolerance (LOGIT_ATOL): {LOGIT_ATOL:.1e}")
    print(f"[{device}]   worst step            : {int(per_step.argmax())}")

    assert result.token_ids == hf_tokens, (
        f"token mismatch on {path} at prompt_len={prompt_len}: "
        f"engine {result.token_ids} vs HF {hf_tokens}"
    )
    assert deviation <= LOGIT_ATOL, (
        f"{path} max abs logit deviation {deviation:.6e} exceeds "
        f"LOGIT_ATOL {LOGIT_ATOL:.1e}"
    )


@pytest.mark.parametrize("prompt_len", PROMPT_LENGTHS)
def test_parity_on_a_fragmented_pool(runtime, prompt_len):
    """Parity when the block table is scrambled, not sequential.

    This is the test Step 3 exists for. With sequential allocation a paged path
    that ignores its block table and reads the pool straight through still gets
    the right answer, so parity alone proves nothing. Here the sequence's blocks
    descend and are non-adjacent, so reading through would gather the wrong
    tokens and both the logits and the generated text would diverge.
    """
    torch.manual_seed(SEED)
    device = runtime.device
    engine = runtime.engine
    prompt_ids = _prompt_ids(prompt_len)

    hf_tokens, hf_logits = _hf_reference(runtime, prompt_len)

    cache = engine.new_paged_cache(FRAGMENTED_POOL_BLOCKS)
    allocator = BlockAllocator(num_blocks=FRAGMENTED_POOL_BLOCKS, block_size=BLOCK_SIZE)
    _fragment_pool(allocator, FRAGMENTED_POOL_BLOCKS)

    backend = engine.new_paged_backend(cache, allocator, seq_id=0)
    result = engine.generate_greedy(
        list(prompt_ids), NUM_GENERATED_TOKENS, backend=backend
    )

    table = backend.block_table
    descents = [i for i in range(len(table) - 1) if table[i + 1] < table[i]]
    non_adjacent = [i for i in range(len(table) - 1) if abs(table[i + 1] - table[i]) != 1]

    deviation = _max_abs_deviation(result.step_logits, hf_logits)
    print(
        f"\n[{device}] FRAGMENTED prompt_len={prompt_len}, block_table={table}"
    )
    print(f"[{device}]   descents / non-adjacent: {len(descents)} / {len(non_adjacent)}")
    print(f"[{device}]   tokens identical      : {result.token_ids == hf_tokens}")
    print(f"[{device}]   MAX ABS DEVIATION     : {deviation:.6e}")
    print(f"[{device}]   tolerance (LOGIT_ATOL): {LOGIT_ATOL:.1e}")

    # The fixture must actually be fragmenting, or the test below is vacuous.
    assert len(table) >= 3, f"block table {table} too short to be meaningfully scrambled"
    assert descents, f"block table {table} is monotonic: the pool was not fragmented"
    assert non_adjacent, f"block table {table} is contiguous: the pool was not fragmented"

    assert result.token_ids == hf_tokens, (
        f"token mismatch on a fragmented pool at prompt_len={prompt_len} "
        f"with block_table={table}: engine {result.token_ids} vs HF {hf_tokens}"
    )
    assert deviation <= LOGIT_ATOL, (
        f"fragmented-pool max abs logit deviation {deviation:.6e} exceeds "
        f"LOGIT_ATOL {LOGIT_ATOL:.1e}"
    )


class _BlockTableIgnoringBackend(PagedBackend):
    """Deliberately broken: writes by the block table, reads straight through.

    This is precisely the bug the fragmented-pool test exists to catch -- a paged
    path where KV is stored correctly but the gather assumes logical position p
    lives at physical slot p. On a sequentially allocated pool that assumption
    happens to hold, so this subclass would pass parity there.
    """

    def read_slots(self, total: int) -> torch.Tensor:
        return torch.arange(total, dtype=torch.long, device=self.device)


def test_fragmented_fixture_would_catch_a_ignored_block_table(runtime):
    """Negative control: the fixture has teeth.

    A passing test proves nothing unless the same fixture fails when the thing
    under test is broken. Here the block table is written but ignored on read.
    On a fragmented pool that must produce different tokens; if it does not, the
    fragmentation is not actually forcing the gather to consult the table and
    `test_parity_on_a_fragmented_pool` is vacuous.
    """
    torch.manual_seed(SEED)
    engine = runtime.engine
    prompt_len = PROMPT_LENGTHS[-1]
    prompt_ids = _prompt_ids(prompt_len)
    hf_tokens, _ = _hf_reference(runtime, prompt_len)

    cache = engine.new_paged_cache(FRAGMENTED_POOL_BLOCKS)
    allocator = BlockAllocator(num_blocks=FRAGMENTED_POOL_BLOCKS, block_size=BLOCK_SIZE)
    _fragment_pool(allocator, FRAGMENTED_POOL_BLOCKS)
    allocator.create_sequence(0)
    broken = _BlockTableIgnoringBackend(
        cache, allocator, 0, engine.dims.num_kv_groups, engine.dims.attn_scale
    )

    result = engine.generate_greedy(list(prompt_ids), NUM_GENERATED_TOKENS, backend=broken)

    print(
        f"\n[{runtime.device}] NEGATIVE CONTROL, block_table={broken.block_table}"
    )
    print(f"[{runtime.device}]   ignoring the block table changed the output: "
          f"{result.token_ids != hf_tokens}")

    assert result.token_ids != hf_tokens, (
        "a paged backend that ignores its block table still reproduced HF exactly, "
        "so the fragmented-pool fixture cannot detect that bug and proves nothing"
    )


@pytest.mark.parametrize("prompt_len", PROMPT_LENGTHS)
def test_paged_and_contiguous_agree_exactly_on_tokens(runtime, prompt_len):
    """The two backends must not merely both be close to HF; they must agree."""
    torch.manual_seed(SEED)
    prompt_ids = _prompt_ids(prompt_len)
    max_seq_len = prompt_len + NUM_GENERATED_TOKENS

    contiguous = runtime.engine.generate_greedy(
        list(prompt_ids),
        NUM_GENERATED_TOKENS,
        backend=_make_backend(runtime.engine, "contiguous", max_seq_len),
    )
    paged = runtime.engine.generate_greedy(
        list(prompt_ids),
        NUM_GENERATED_TOKENS,
        backend=_make_backend(runtime.engine, "paged", max_seq_len),
    )

    deviation = _max_abs_deviation(paged.step_logits, contiguous.step_logits)
    print(
        f"\n[{runtime.device}] paged vs contiguous, prompt_len={prompt_len}: "
        f"max |dlogit| = {deviation:.6e}"
    )
    assert paged.token_ids == contiguous.token_ids
    assert deviation <= LOGIT_ATOL


@pytest.mark.slow
@pytest.mark.parametrize("prompt_len", PROMPT_LENGTHS)
def test_baselines_agree(runtime, prompt_len):
    """The two HuggingFace baselines agree, so either can be trusted as ground truth.

    Opt-in: the recompute baseline re-runs the whole prefix every step.
    """
    torch.manual_seed(SEED)
    device = runtime.device
    prompt_ids = _prompt_ids(prompt_len)

    cached_tokens, cached_logits = hf_greedy_cached(
        runtime.model, prompt_ids, NUM_GENERATED_TOKENS, device
    )
    recompute_tokens, recompute_logits = hf_greedy_recompute(
        runtime.model, prompt_ids, NUM_GENERATED_TOKENS, device
    )

    deviation = _max_abs_deviation(cached_logits, recompute_logits)
    print(
        f"\n[{device}] prompt_len={prompt_len} baseline cross-check: "
        f"HF-cached vs HF-recompute max |dlogit| = {deviation:.3e} (atol {LOGIT_ATOL:.1e})"
    )
    assert cached_tokens == recompute_tokens
    assert deviation <= LOGIT_ATOL


def test_cache_is_ours_and_contiguous(runtime):
    """The path under test must not be holding a transformers cache object."""
    engine = runtime.engine
    cache = engine.new_cache(max_seq_len=64)

    assert type(cache).__module__.startswith("engine.")
    assert not isinstance(cache, DynamicCache)

    assert cache.length == 0
    assert cache.keys(0, 0).shape == (1, engine.dims.num_kv_heads, 0, engine.dims.head_dim)
    full = cache.keys(0, 64)
    assert full.is_contiguous()
    assert full.shape == (1, engine.dims.num_kv_heads, 64, engine.dims.head_dim)


def test_paged_cache_writes_land_in_the_right_physical_block(runtime):
    """A token's KV must be findable in the block its table names, not slot p."""
    engine = runtime.engine
    cache = engine.new_paged_cache(FRAGMENTED_POOL_BLOCKS)
    allocator = BlockAllocator(num_blocks=FRAGMENTED_POOL_BLOCKS, block_size=BLOCK_SIZE)
    _fragment_pool(allocator, FRAGMENTED_POOL_BLOCKS)
    backend = engine.new_paged_backend(cache, allocator, seq_id=0)

    prompt_ids = _prompt_ids(PROMPT_LENGTHS[-1])
    engine.generate_greedy(list(prompt_ids), NUM_GENERATED_TOKENS, backend=backend)

    table = backend.block_table
    assert table[0] != 0, "fixture failed: sequence still starts at physical block 0"

    # Position 0's key must live in block table[0] at offset 0, and nowhere near
    # the slot a straight-through read would use.
    block_id, offset = allocator.physical_slot(0, 0)
    assert (block_id, offset) == (table[0], 0)
    stored = cache.block_keys(0, block_id)[offset]
    assert torch.any(stored != 0), "no key was written into the mapped block"
