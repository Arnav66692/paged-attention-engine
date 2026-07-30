"""fp16 parity on CUDA. Skipped everywhere else, never absent from collection.

This exists because float32 parity cannot catch a missing upcast. Every
`.to(torch.float32)` in the forward pass -- the one inside RMSNorm, the one
building the rotary tables -- is a no-op when the model is already float32, so
deleting it changes nothing and tests/test_parity.py stays green. In a two-byte
format those upcasts are load-bearing: RMSNorm accumulates a sum of squares over
896 elements and doing that in fp16 loses the variance, and rotary tables built
in fp16 lose angular resolution at long positions. Both show up here and only
here.

WHY fp16 AND NOT bf16. The measurement host is Kaggle's free tier, which serves
a T4 (Turing, SM 7.5) or a P100 (Pascal, SM 6.0). Both predate Ampere, and
neither has bfloat16 hardware -- on those cards a bf16 model either refuses to
run or falls back to emulation that measures nothing anyone wants measured. So
the two-byte format under test is fp16.

The swap is not free, and the difference is not what it first looks like. fp16
keeps 10 explicit mantissa bits against bf16's 8, so it is the *finer* of the two
by a factor of four. What it gives up is exponent range: 5 exponent bits against
bf16's 8, so fp16 saturates at 65504 where bf16 reaches float32's range. bf16
cannot overflow anywhere float32 would not; fp16 can. The parity test below
therefore asserts the logits are finite as well as close, which is a check the
bf16 version had no reason to make.

NOTHING LOCKED MOVES. fp16 and bf16 are both two bytes per element, so bytes per
token, bytes per block, num_gpu_blocks, kv_pool_bytes, the shortfall at
PRESSURE_C and both baseline residencies are all exactly what they were.

The tolerance below is declared separately and deliberately. It is NOT
LOGIT_ATOL and NOT FUSED_LOGIT_ATOL: reusing a float32 tolerance in fp16 would
fail for reasons that have nothing to do with correctness, and loosening a shared
constant to fit fp16 would silently gut the float32 tests. It is also not the
2.5e-1 the bf16 version carried, which was derived for a coarser mantissa and
would be the wrong bound here.

SCOPE. This file pins the two single-sequence storage paths against HuggingFace.
The fused batched path added at Step 5b agrees with the unfused one to
FUSED_LOGIT_ATOL in float32; what it agrees to in fp16 is a different question
needing its own declared tolerance, and this file does not answer it.

This machine is an M4 Mac, so this file reports as a skip here. It is written so
that it runs unchanged on the Kaggle host without anyone having to remember it
exists.
"""

from functools import lru_cache

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from engine.cache import BLOCK_SIZE, BlockAllocator
from engine.model import Qwen2Engine

DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# The same text and lengths as tests/test_parity.py, so an fp16 failure can be
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

FP16_DTYPE = torch.float16

# The largest finite value fp16 can represent. bf16 reaches ~3.4e38, so this
# ceiling is the one genuinely new hazard in the swap.
FP16_MAX = 65504.0

# Absolute tolerance on logits, in float16. Declared here, derived from fp16's
# own numbers, and carried across from nothing.
#
# It is NOT LOGIT_ATOL and NOT FUSED_LOGIT_ATOL. Neither is imported into this
# module and `test_fp16_suite_is_collected` asserts both names are absent. It is
# also not the bf16 file's 2.5e-1: that was derived for an 8-bit mantissa, and
# fp16's is 10, so reusing it would be the wrong bound for the wrong format.
#
#   1. Resolution. fp16 carries 10 explicit mantissa bits, so its unit roundoff
#      is 2**-11 = 4.9e-4 and consecutive representable values are 2**-10 = 9.8e-4
#      apart in relative terms. Four times finer than bfloat16's 2**-8, and about
#      16,000 times coarser than float32's 2**-24.
#
#   2. Reductions are NOT the dominant term here, which is what makes this
#      derivation different in kind from the float32 ones. cuBLAS accumulates
#      fp16 GEMMs in float32 on both a T4 and a P100, so reordering the
#      4864-term MLP projection perturbs it at float32 resolution -- the order
#      1e-4 absolute that tests/test_parity.py already measures. Negligible at
#      this scale.
#
#   3. What fp16 actually costs is that the two implementations round each
#      operation's output back into fp16 independently. Wherever their float32-
#      level values straddle a rounding boundary the stored values differ by one
#      ulp, and the residual stream carries that difference forward.
#
#   4. This model's logits reach magnitudes near 2.2e1, measured on the float32
#      fixtures. 22 sits in the binade [16, 32), where one fp16 ulp is
#      16 * 2**-10 = 1.5625e-2.
#
#   5. Take one such divergence per decoder layer and let the 24 of them
#      accumulate as a random walk: sqrt(24) = 4.9, so about 5 ulp, or 7.7e-2.
#      Double it for the attention softmax and the lm_head projection, two more
#      places the two paths round independently, giving 1.6e-1.
#
# 2e-1 is that, rounded up. It sits below the bf16 file's 2.5e-1, which is the
# right direction for a format with four times the mantissa resolution.
#
# TOKEN EQUALITY, asserted below, is the real gate. It is not a tolerance and
# cannot be relaxed. This constant only bounds how far the logits may drift while
# still producing the same argmax, and the measured deviation is printed for
# every configuration. No CUDA device was available when it was derived, so it is
# a prediction rather than a fitted value; if the benchmark host disagrees, the
# measurement is reported and this number is not moved to accommodate it.
FP16_LOGIT_ATOL = 2e-1

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fp16 parity is CUDA-only; this host has no CUDA device",
)


@lru_cache(maxsize=None)
def _load_fp16():
    model = AutoModelForCausalLM.from_pretrained(DEV_MODEL, dtype=FP16_DTYPE)
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
def test_fp16_parity_tokens_and_logits(prompt_len, path):
    """Token IDs must be exactly equal to HuggingFace's in fp16, on both backends."""
    torch.manual_seed(SEED)
    model, engine = _load_fp16()
    prompt_ids = _prompt_ids(prompt_len)

    assert engine.dtype == FP16_DTYPE, (
        f"engine is running in {engine.dtype}, not float16; this test would "
        "then be a duplicate of the float32 one and would catch nothing"
    )

    hf_tokens, hf_logits = _hf_greedy_cached(model, prompt_ids, NUM_GENERATED_TOKENS)
    backend = _make_backend(engine, path, prompt_len + NUM_GENERATED_TOKENS)
    result = engine.generate_greedy(
        list(prompt_ids), NUM_GENERATED_TOKENS, backend=backend
    )

    deviation = (result.step_logits.float() - hf_logits.float()).abs().max().item()
    largest = result.step_logits.float().abs().max().item()

    print(f"\n[cuda fp16] path={path} prompt_len={prompt_len}")
    print(f"[cuda fp16]   tokens identical           : {result.token_ids == hf_tokens}")
    print(f"[cuda fp16]   MAX ABS DEVIATION          : {deviation:.6e}")
    print(f"[cuda fp16]   tolerance (FP16_LOGIT_ATOL): {FP16_LOGIT_ATOL:.1e}")
    print(f"[cuda fp16]   largest |logit|            : {largest:.3f} "
          f"(fp16 saturates at {FP16_MAX:.0f})")

    # fp16's narrow exponent range is the hazard bf16 did not have. An overflow
    # surfaces as inf here and as a token mismatch below, so name it separately
    # rather than leaving it to be misdiagnosed as a tolerance failure.
    assert torch.isfinite(result.step_logits).all(), (
        f"fp16 logits contain inf or NaN on {path} at prompt_len={prompt_len}; "
        f"something in the forward pass overflowed fp16's {FP16_MAX:.0f} ceiling"
    )

    assert result.token_ids == hf_tokens, (
        f"fp16 token mismatch on {path} at prompt_len={prompt_len}: "
        f"engine {result.token_ids} vs HF {hf_tokens}"
    )
    assert deviation <= FP16_LOGIT_ATOL, (
        f"fp16 max abs logit deviation {deviation:.6e} exceeds "
        f"FP16_LOGIT_ATOL {FP16_LOGIT_ATOL:.1e}"
    )


@requires_cuda
def test_fp16_rms_norm_accumulates_in_fp32():
    """An fp16-accumulated RMSNorm is measurably wrong; ours must not be.

    Guards the specific upcast that float32 parity can never exercise, by
    computing the same normalisation both ways over real-sized vectors and
    asserting our implementation tracks the float64 reference far more closely
    than an fp16-accumulated one does.

    fp16 makes this sharper than bf16 did, for a reason worth naming. The sum of
    squares over 896 elements of a unit-normal vector lands near 896, and each
    term is squared before it is added, so an fp16 accumulator is both losing
    mantissa on every add and running within two orders of magnitude of its own
    65504 ceiling. bf16 had the first problem and not the second.
    """
    from engine.model import rms_norm

    torch.manual_seed(SEED)
    hidden_size = 896
    x = torch.randn(1, 8, hidden_size, dtype=FP16_DTYPE, device="cuda")
    weight = torch.ones(hidden_size, dtype=FP16_DTYPE, device="cuda")
    eps = 1e-6

    ours = rms_norm(x, weight, eps).float()

    exact = x.double()
    exact = exact * torch.rsqrt(exact.pow(2).mean(-1, keepdim=True) + eps)
    exact = exact.float()

    # The same expression with no upcast: squared, summed and averaged entirely
    # in fp16, which is exactly what deleting the `.to(torch.float32)` produces.
    naive = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    naive = naive.float()

    our_error = (ours - exact).abs().max().item()
    naive_error = (naive - exact).abs().max().item()

    print(f"\n[cuda fp16] rms_norm max error vs float64: ours={our_error:.6e}")
    print(f"[cuda fp16] rms_norm max error vs float64: fp16-accumulated={naive_error:.6e}")

    assert torch.isfinite(ours).all(), "our RMSNorm produced inf or NaN in fp16"
    assert our_error <= naive_error, (
        "our RMSNorm is no better than accumulating in fp16, which means the "
        "float32 upcast is missing or ineffective"
    )


def test_fp16_suite_is_collected():
    """Always runs, everywhere: proves the file above is collected, not absent.

    A CUDA-only file that silently failed to import would look identical on this
    machine to one whose tests all skipped. This runs unconditionally, so the
    import errors loudly if it ever breaks.
    """
    assert FP16_DTYPE is torch.float16
    assert FP16_LOGIT_ATOL == 2e-1
    # Declared here, not borrowed. Neither float32 tolerance is imported into
    # this module, so neither can be reused or quietly widened.
    assert "LOGIT_ATOL" not in globals()
    assert "FUSED_LOGIT_ATOL" not in globals()
    # And not carried across from the bf16 version this file replaced.
    assert FP16_LOGIT_ATOL != 2.5e-1

    # fp16 and bf16 are both two bytes, so nothing sized in bytes moved with the
    # swap. The mantissa is the whole difference, and it is why the tolerance is
    # tighter rather than equal.
    assert torch.finfo(FP16_DTYPE).bits == torch.finfo(torch.bfloat16).bits == 16
    assert torch.finfo(FP16_DTYPE).max == FP16_MAX
    assert torch.finfo(FP16_DTYPE).eps < torch.finfo(torch.bfloat16).eps
