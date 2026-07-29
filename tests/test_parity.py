"""Step 1 parity: our decode loop against HuggingFace, same weights, same prompt.

The engine under test owns its KV cache outright (engine/cache.py) and runs a
forward pass written out in engine/model.py. This test asserts that swapping
HuggingFace's decode path for ours changes nothing observable:

  * all 32 generated token IDs are exactly equal, and
  * per-step logits agree within LOGIT_ATOL.

Two independent baselines are run. `hf_greedy_cached` is HuggingFace driving its
own DynamicCache, the realistic path. `hf_greedy_recompute` uses no cache at all
and re-runs the full prefix every step, so it depends on no cache implementation
whatsoever. They are asserted equal to each other first: if that fails, the
baseline is the suspect, not the engine.

Tolerance is fixed up front from the arithmetic, never fitted to an observed
result. The max observed deviation is printed for every configuration.
"""

import sys
from functools import lru_cache

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from engine.model import Qwen2Engine

DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

FIXED_PROMPT = "The capital of France is Paris, and the capital of Japan is"
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


def _devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@lru_cache(maxsize=None)
def _load(device: str):
    """Load the model once per device and build an engine sharing its weights."""
    model = AutoModelForCausalLM.from_pretrained(DEV_MODEL, dtype=PARITY_DTYPE)
    model.eval()
    model.to(device)
    engine = Qwen2Engine.from_hf_model(model)
    return model, engine


@lru_cache(maxsize=None)
def _prompt_ids() -> tuple[int, ...]:
    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL)
    return tuple(tokenizer(FIXED_PROMPT, add_special_tokens=False)["input_ids"])


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


@pytest.mark.parametrize("device", _devices())
def test_baselines_agree(device):
    """The two HuggingFace baselines must agree before either is trusted."""
    torch.manual_seed(SEED)
    model, _ = _load(device)
    prompt_ids = _prompt_ids()

    cached_tokens, cached_logits = hf_greedy_cached(
        model, prompt_ids, NUM_GENERATED_TOKENS, device
    )
    recompute_tokens, recompute_logits = hf_greedy_recompute(
        model, prompt_ids, NUM_GENERATED_TOKENS, device
    )

    deviation = _max_abs_deviation(cached_logits, recompute_logits)
    print(
        f"\n[{device}] baseline cross-check: HF-cached vs HF-recompute "
        f"max |dlogit| = {deviation:.3e} (atol {LOGIT_ATOL:.1e})"
    )
    assert cached_tokens == recompute_tokens
    assert deviation <= LOGIT_ATOL


@pytest.mark.parametrize("device", _devices())
def test_parity_tokens_and_logits(device):
    """Our decode loop and cache reproduce HuggingFace exactly on tokens."""
    torch.manual_seed(SEED)
    model, engine = _load(device)
    prompt_ids = _prompt_ids()

    hf_tokens, hf_logits = hf_greedy_cached(
        model, prompt_ids, NUM_GENERATED_TOKENS, device
    )
    result = engine.generate_greedy(list(prompt_ids), NUM_GENERATED_TOKENS)

    assert result.step_logits.shape[0] == NUM_GENERATED_TOKENS
    assert len(result.token_ids) == NUM_GENERATED_TOKENS

    deviation = _max_abs_deviation(result.step_logits, hf_logits)
    per_step = (result.step_logits.float() - hf_logits.float()).abs().amax(dim=-1)

    print(
        f"\n[{device}] engine vs HF over {NUM_GENERATED_TOKENS} steps, "
        f"prompt_len={len(prompt_ids)}, dtype={PARITY_DTYPE}"
    )
    print(f"[{device}]   tokens identical      : {result.token_ids == hf_tokens}")
    print(f"[{device}]   MAX ABS DEVIATION     : {deviation:.6e}")
    print(f"[{device}]   tolerance (LOGIT_ATOL): {LOGIT_ATOL:.1e}")
    print(f"[{device}]   worst step            : {int(per_step.argmax())}")
    print(f"[{device}]   first-step deviation  : {per_step[0].item():.6e}")
    print(f"[{device}]   last-step deviation   : {per_step[-1].item():.6e}")

    assert result.token_ids == hf_tokens, (
        f"token mismatch: engine {result.token_ids} vs HF {hf_tokens}"
    )
    assert deviation <= LOGIT_ATOL, (
        f"max abs logit deviation {deviation:.6e} exceeds LOGIT_ATOL {LOGIT_ATOL:.1e}"
    )


@pytest.mark.parametrize("device", _devices())
def test_cache_is_ours_and_contiguous(device):
    """The path under test must not be holding a transformers cache object."""
    _, engine = _load(device)
    cache = engine.new_cache(max_seq_len=64)

    assert type(cache).__module__.startswith("engine.")
    assert not isinstance(cache, DynamicCache)
    for name, module in sys.modules.items():
        if name == "engine.cache":
            assert "cache_utils" not in getattr(module, "__dict__", {})

    # One contiguous tensor per layer for K and for V.
    assert cache.length == 0
    keys = cache.keys(0, 0)
    assert keys.shape == (1, engine.dims.num_kv_heads, 0, engine.dims.head_dim)
    full = cache.keys(0, 64)
    assert full.is_contiguous()
    assert full.shape == (1, engine.dims.num_kv_heads, 64, engine.dims.head_dim)
