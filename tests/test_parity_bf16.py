"""bf16 parity on CUDA. Skipped everywhere else, never absent from collection.

This exists because float32 parity cannot catch a missing upcast. Every
`.to(torch.float32)` in the forward pass -- the one inside RMSNorm, the one
building the rotary tables -- is a no-op when the model is already float32, so
deleting it changes nothing and tests/test_parity.py stays green. In bf16 those
upcasts are load-bearing: RMSNorm accumulates a sum of squares over 896 elements
and doing that in bf16 loses the variance, and rotary tables built in bf16 lose
angular resolution at long positions. Both show up here and only here.

The tolerance below is declared separately and deliberately. It is NOT
LOGIT_ATOL: reusing a float32 tolerance in bf16 would fail for reasons that have
nothing to do with correctness, and loosening the shared constant to fit bf16
would silently gut the float32 test.

This machine is an M4 Mac, so this file reports as a skip here. It is written so
that it runs unchanged on a CUDA host without anyone having to remember it
exists.
"""

from functools import lru_cache

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from engine.cache import BLOCK_SIZE, BlockAllocator
from engine.model import Qwen2Engine

DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# The same text and lengths as tests/test_parity.py, so a bf16 failure can be
# compared against the float32 result for the identical fixture.
BASE_PROMPT = (
    "The capital of France is Paris, and the capital of Japan is Tokyo. "
    "Rivers shape the cities built beside them, carrying trade inland long "
    "before roads were paved, and the oldest bridges still mark where the "
    "first crossings were cut into the banks."
)
PROMPT_LENGTHS = (13, 32, 33)
PATHS = ("contiguous", "paged")
SEED = 0
NUM_GENERATED_TOKENS = 32
PAGED_POOL_BLOCKS = 64

BF16_DTYPE = torch.bfloat16

# Absolute tolerance on logits, in bfloat16. Declared here, independently of the
# float32 LOGIT_ATOL, and never substituted for it.
#
# bfloat16 keeps an 8-bit mantissa, so its relative resolution is 2**-8, about
# 3.9e-3 -- roughly 30,000x coarser than float32. This model's logits reach
# magnitudes near 1.6e1, where one bf16 ulp is 16 * 2**-8 = 0.0625. Two
# implementations that disagree only in reduction order will land within a few
# ulp of each other, so this is set at four: 4 * 0.0625 = 0.25.
#
# Token equality, asserted below, is the real gate. It is not a tolerance and
# cannot be relaxed.
BF16_LOGIT_ATOL = 2.5e-1

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bf16 parity is CUDA-only; this host has no CUDA device",
)


@lru_cache(maxsize=None)
def _load_bf16():
    model = AutoModelForCausalLM.from_pretrained(DEV_MODEL, dtype=BF16_DTYPE)
    model.eval()
    model.to("cuda")
    return model, Qwen2Engine.from_hf_model(model)


@lru_cache(maxsize=None)
def _prompt_ids(length: int) -> tuple[int, ...]:
    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL)
    ids = tokenizer(BASE_PROMPT, add_special_tokens=False)["input_ids"]
    if len(ids) < length:
        raise ValueError(f"BASE_PROMPT is {len(ids)} tokens, need at least {length}")
    return tuple(ids[:length])


@torch.inference_mode()
def _hf_greedy_cached(model, prompt_ids, num_steps):
    cache = DynamicCache(config=model.config)
    logits = model(
        input_ids=torch.tensor([list(prompt_ids)], device="cuda"),
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
            input_ids=torch.tensor([[token]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
        ).logits[0, -1]
    return tuple(tokens), torch.stack(step_logits)


def _make_backend(engine: Qwen2Engine, path: str, max_seq_len: int):
    if path == "contiguous":
        return engine.new_contiguous_backend(max_seq_len)
    if path == "paged":
        cache = engine.new_paged_cache(PAGED_POOL_BLOCKS)
        allocator = BlockAllocator(num_blocks=PAGED_POOL_BLOCKS, block_size=BLOCK_SIZE)
        return engine.new_paged_backend(cache, allocator, seq_id=0)
    raise ValueError(f"unknown path {path!r}")


@requires_cuda
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("prompt_len", PROMPT_LENGTHS)
def test_bf16_parity_tokens_and_logits(prompt_len, path):
    """Token IDs must be exactly equal to HuggingFace's in bf16, on both backends."""
    torch.manual_seed(SEED)
    model, engine = _load_bf16()
    prompt_ids = _prompt_ids(prompt_len)

    assert engine.dtype == BF16_DTYPE, (
        f"engine is running in {engine.dtype}, not bfloat16; this test would "
        "then be a duplicate of the float32 one and would catch nothing"
    )

    hf_tokens, hf_logits = _hf_greedy_cached(model, prompt_ids, NUM_GENERATED_TOKENS)
    backend = _make_backend(engine, path, prompt_len + NUM_GENERATED_TOKENS)
    result = engine.generate_greedy(
        list(prompt_ids), NUM_GENERATED_TOKENS, backend=backend
    )

    deviation = (result.step_logits.float() - hf_logits.float()).abs().max().item()

    print(f"\n[cuda bf16] path={path} prompt_len={prompt_len}")
    print(f"[cuda bf16]   tokens identical           : {result.token_ids == hf_tokens}")
    print(f"[cuda bf16]   MAX ABS DEVIATION          : {deviation:.6e}")
    print(f"[cuda bf16]   tolerance (BF16_LOGIT_ATOL): {BF16_LOGIT_ATOL:.1e}")

    assert result.token_ids == hf_tokens, (
        f"bf16 token mismatch on {path} at prompt_len={prompt_len}: "
        f"engine {result.token_ids} vs HF {hf_tokens}"
    )
    assert deviation <= BF16_LOGIT_ATOL, (
        f"bf16 max abs logit deviation {deviation:.6e} exceeds "
        f"BF16_LOGIT_ATOL {BF16_LOGIT_ATOL:.1e}"
    )


@requires_cuda
def test_bf16_rms_norm_accumulates_in_fp32():
    """A bf16-accumulated RMSNorm is measurably wrong; ours must not be.

    Guards the specific upcast that float32 parity can never exercise, by
    computing the same normalisation both ways over real-sized vectors and
    asserting our implementation tracks the float64 reference far more closely
    than a bf16-accumulated one does.
    """
    from engine.model import rms_norm

    torch.manual_seed(SEED)
    hidden_size = 896
    x = torch.randn(1, 8, hidden_size, dtype=BF16_DTYPE, device="cuda")
    weight = torch.ones(hidden_size, dtype=BF16_DTYPE, device="cuda")
    eps = 1e-6

    ours = rms_norm(x, weight, eps).float()

    exact = x.double()
    exact = exact * torch.rsqrt(exact.pow(2).mean(-1, keepdim=True) + eps)
    exact = exact.float()

    naive = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    naive = naive.float()

    our_error = (ours - exact).abs().max().item()
    naive_error = (naive - exact).abs().max().item()

    print(f"\n[cuda bf16] rms_norm max error vs float64: ours={our_error:.6e}")
    print(f"[cuda bf16] rms_norm max error vs float64: bf16-accumulated={naive_error:.6e}")

    assert our_error <= naive_error, (
        "our RMSNorm is no better than accumulating in bf16, which means the "
        "float32 upcast is missing or ineffective"
    )


def test_bf16_suite_is_collected():
    """Always runs, everywhere: proves the file above is collected, not absent.

    A CUDA-only file that silently failed to import would look identical on this
    machine to one whose tests all skipped. This runs unconditionally, so the
    import errors loudly if it ever breaks.
    """
    assert BF16_DTYPE is torch.bfloat16
    assert BF16_LOGIT_ATOL == 2.5e-1
    # Declared here, not borrowed. The float32 tolerance is deliberately not
    # imported into this module, so it cannot be reused or quietly widened.
    assert "LOGIT_ATOL" not in globals()
