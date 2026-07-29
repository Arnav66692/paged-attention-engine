"""Step 1 parity: our decode loop against HuggingFace, same weights, same prompt.

The engine under test owns its KV cache outright (engine/cache.py) and runs a
forward pass written out in engine/model.py. This test asserts that swapping
HuggingFace's decode path for ours changes nothing observable:

  * all 32 generated token IDs are exactly equal, and
  * per-step logits agree within LOGIT_ATOL.

Prompt lengths are chosen against the locked block size of 16, so the fixtures
already cover the cases that will matter once attention is paged at Step 3: a
prompt shorter than one block, a prompt that is an exact block multiple, and a
prompt that ends mid-block.

Two independent baselines exist. `hf_greedy_cached` is HuggingFace driving its
own DynamicCache, the realistic path and the one parity is asserted against.
`hf_greedy_recompute` uses no cache at all and re-runs the full prefix every
step, so it depends on no cache implementation whatsoever; it is O(n^2) and
marked slow, so it is opt-in via `-m slow`. When both run, the cross-check
answers "is the engine wrong, or is the baseline wrong?".

Tolerance is fixed up front from the arithmetic, never fitted to an observed
result. The max observed deviation is printed for every configuration.
"""

from dataclasses import dataclass
from functools import lru_cache

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from engine.cache import BLOCK_SIZE
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

# Absolute tolerance on logits, in float32.
#
# Both sides run the same weights in the same dtype on the same device, so the
# only source of disagreement is the order of floating point reductions: our
# attention reads a slice of a preallocated cache tensor where HuggingFace
# concatenates a fresh one, and the two reach scaled_dot_product_attention with
# different memory layouts. float32 has ~1.2e-7 relative resolution; across 24
# residual layers that compounds to order 1e-5 relative, and this model's logits
# run to roughly 1e1, giving expected deviation near 1e-4. 1e-3 is that with
# headroom. It is not widened if the test fails -- a failure is reported.
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


def _max_abs_deviation(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def test_prompt_lengths_cover_block_boundaries():
    """The fixture lengths mean what the docstring claims, against BLOCK_SIZE."""
    sub_block, exact_multiple, mid_block = PROMPT_LENGTHS
    assert sub_block < BLOCK_SIZE
    assert exact_multiple % BLOCK_SIZE == 0
    assert mid_block % BLOCK_SIZE == 1


@pytest.mark.parametrize("prompt_len", PROMPT_LENGTHS)
def test_parity_tokens_and_logits(runtime, prompt_len):
    """Our decode loop and cache reproduce HuggingFace exactly on tokens."""
    torch.manual_seed(SEED)
    device = runtime.device
    prompt_ids = _prompt_ids(prompt_len)
    assert len(prompt_ids) == prompt_len

    hf_tokens, hf_logits = hf_greedy_cached(
        runtime.model, prompt_ids, NUM_GENERATED_TOKENS, device
    )
    result = runtime.engine.generate_greedy(list(prompt_ids), NUM_GENERATED_TOKENS)

    assert result.step_logits.shape[0] == NUM_GENERATED_TOKENS
    assert len(result.token_ids) == NUM_GENERATED_TOKENS

    deviation = _max_abs_deviation(result.step_logits, hf_logits)
    per_step = (result.step_logits.float() - hf_logits.float()).abs().amax(dim=-1)

    print(
        f"\n[{device}] prompt_len={prompt_len} ({prompt_len / BLOCK_SIZE:.2f} blocks), "
        f"{NUM_GENERATED_TOKENS} steps, dtype={PARITY_DTYPE}"
    )
    print(f"[{device}]   tokens identical      : {result.token_ids == hf_tokens}")
    print(f"[{device}]   MAX ABS DEVIATION     : {deviation:.6e}")
    print(f"[{device}]   tolerance (LOGIT_ATOL): {LOGIT_ATOL:.1e}")
    print(f"[{device}]   worst step            : {int(per_step.argmax())}")

    assert result.token_ids == hf_tokens, (
        f"token mismatch at prompt_len={prompt_len}: "
        f"engine {result.token_ids} vs HF {hf_tokens}"
    )
    assert deviation <= LOGIT_ATOL, (
        f"max abs logit deviation {deviation:.6e} exceeds LOGIT_ATOL {LOGIT_ATOL:.1e}"
    )


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
