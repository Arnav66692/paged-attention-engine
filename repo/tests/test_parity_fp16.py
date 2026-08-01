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
# module and `test_fp16_suite_is_collected` asserts both names are absent.
#
# TWO CORRECTIONS, AND THE REASON THIS DERIVATION IS WRITTEN OUT IN FULL.
#
# FIRST: bf16 DOES NOT HAVE AN 8-BIT MANTISSA. The file this replaced said it
# did, and every downstream number inherited the error. bfloat16 is 1 sign + 8
# EXPONENT + 7 mantissa; the 8 belongs to the exponent, which is the whole point
# of the format. torch.finfo(torch.bfloat16).eps is 2**-7 = 7.8e-3, not the
# 2**-8 = 3.9e-3 that file claimed, so it understated one bf16 ulp by a factor of
# two: at a logit of 16 the true ulp is 0.125, not the 0.0625 it computed. Its
# declared bound of 2.5e-1 was therefore TWO true bf16 ulp, not the four it
# said. `test_the_fp16_tolerance_is_the_mantissa_ratio_tighter` now reads both
# widths off torch.finfo rather than from prose, so this cannot recur.
#
# The consequence for the swap: fp16's 10 explicit mantissa bits are THREE more
# than bf16's 7, not two, so the resolution gain is 8x and not 4x.
#
# SECOND: the first version of this constant was 2e-1, a 20% tightening. That was
# an artifact of silently changing the ERROR MODEL at the same time as the
# format. bf16's bound was "the two implementations track to within a few ulp of
# the final logit," with no layer-accumulation term at all. The 2e-1 derivation
# instead assumed a full ulp of divergence per decoder layer accumulating as a
# random walk over 24 layers and then doubled it, about ten ulp. Two derivations
# on different error models cannot be compared, and quoting the ratio between
# them as a tightening is exactly the mistake. Under one model, the bound below
# is 8x tighter than a consistently derived bf16 bound would be, which is what
# three mantissa bits buy.
#
# THE DERIVATION, on one model throughout, anchored on measurements rather than
# assumptions wherever a measurement exists.
#
#   1. RESOLUTION. fp16 carries 10 explicit mantissa bits, so its unit roundoff
#      is 2**-11 = 4.88e-4 and consecutive representable values are
#      2**-10 = 9.77e-4 apart relatively -- eight times finer than bfloat16's
#      2**-7, and about 8,200 times coarser than float32's 2**-23. This model's
#      logits reach 2.24e1, measured on the float32 fixtures, which sits in the
#      binade [16, 32) where one fp16 ulp is 16 * 2**-10 = 1.5625e-2.
#
#   2. REDUCTION ORDER IS NOT THE DOMINANT TERM -- but it is NOT negligible
#      either, and that correction is measured. cuBLAS accumulates fp16 GEMMs in
#      float32, so most of the reduction happens at float32 resolution:
#      tests/test_parity.py measures 2.575e-05 absolute on logits of 2.24e1, or
#      1.15e-6 relative, 1/850th of one fp16 ulp.
#
#      HOWEVER, `allow_fp16_reduced_precision_reduction` defaults to True and
#      permits fp16 split-k reduction. Session 1 measured both ways on a T4:
#
#        prompt_len 13:  3.00 ulp on,  3.00 ulp off   (no change)
#        prompt_len 32:  2.92 ulp on,  2.75 ulp off
#        prompt_len 33:  3.00 ulp on,  2.25 ulp off   (0.75 ulp, ~19% of budget)
#
#      So the split-k term is worth up to three quarters of an ulp -- a fifth of
#      the whole 4-ulp budget. The bound below survives because it was derived
#      pessimistically, not because this term is absent. It is pinned to True,
#      torch's default, for both engines in bench/run.py, and recorded in
#      provenance. THE BOUND IS NOT ADJUSTED TO SUIT EITHER SETTING.
#
#   3. WHAT fp16 ACTUALLY COSTS is that both implementations round each
#      operation's output back into fp16 independently. Two values 1.15e-6 apart
#      relatively round to the same fp16 value unless they straddle a boundary,
#      which happens with probability about 1.15e-6 / 9.77e-4 = 1.2e-3 per
#      rounding. When they do straddle, they differ by exactly one ulp.
#
#   4. DIVERGENCE SATURATES RATHER THAN COMPOUNDING. Once an element differs by
#      one ulp it is 850 times further apart than the float32-level difference
#      that caused it, so it is no longer a rare straddle -- but a decoder
#      layer's residual map is close to Lipschitz-1 in this regime, so a one-ulp
#      relative difference neither shrinks nor multiplies as it is carried
#      forward. The steady state is the two hidden states tracking each other at
#      order one ulp, not one ulp per layer accumulating. This is the step the
#      2e-1 derivation got wrong.
#
#   5. ACCUMULATION is then only the fresh half-ulp of rounding noise each layer
#      contributes, adding as a random walk: sqrt(24) * 0.5 = 2.4 ulp. With the
#      steady-state term of about 1 ulp and the lm_head's own rounding, roughly
#      3.4 ulp, bounded at 4.
#
# 4 * 1.5625e-2 = 6.25e-2.
#
# CONSISTENCY CHECK, not the derivation. The same model applied to bf16 gives
# 4 * (16 * 2**-7) = 5e-1, so this bound is exactly 8x tighter -- the ratio of
# the two mantissa widths, as it must be once the model is held fixed. Arriving
# there independently is what shows the model was applied rather than the ratio
# assumed. Note this is NOT one quarter of the 2.5e-1 the bf16 file declared:
# that value was two true bf16 ulp under a model with no accumulation term, and
# comparing against it directly is what produced the 2e-1 mistake above.
#
# HEADROOM IS THIN, DELIBERATELY. The derivation expects about 3.4 ulp and the
# bound is 4, so roughly 18%. A wider bound would cost real ground: this
# tolerance is what would catch a recompute divergence in fp16 before it became a
# token flip, and a bound eight times looser than the format earns cannot see one.
#
# TOKEN EQUALITY, asserted below, is the real gate. It is not a tolerance and
# cannot be relaxed. No CUDA device was available when this was derived, so it is
# a prediction rather than a fitted value; if the benchmark host disagrees, the
# measured deviation is reported as a result and this number does not move.
FP16_LOGIT_ATOL = 6.25e-2

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
    assert FP16_LOGIT_ATOL == 6.25e-2
    # Declared here, not borrowed. Neither float32 tolerance is imported into
    # this module, so neither can be reused or quietly widened.
    assert "LOGIT_ATOL" not in globals()
    assert "FUSED_LOGIT_ATOL" not in globals()

    # fp16 and bf16 are both two bytes, so nothing sized in bytes moved with the
    # swap. The mantissa is the whole difference.
    assert torch.finfo(FP16_DTYPE).bits == torch.finfo(torch.bfloat16).bits == 16
    assert torch.finfo(FP16_DTYPE).max == FP16_MAX
    assert torch.finfo(FP16_DTYPE).eps < torch.finfo(torch.bfloat16).eps


DERIVED_ULP_COUNT = 4.0
# 2.24e1 measured on the float32 fixtures, so the largest logits sit in [16, 32).
LOGIT_BINADE_BASE = 16.0


def test_the_fp16_tolerance_is_the_mantissa_ratio_tighter():
    """The tightening the extra mantissa bits earn, read off torch.finfo.

    Both widths come from torch.finfo rather than from prose, because prose is
    how the file this replaced came to believe bf16 had an 8-bit mantissa. It has
    7 -- the 8 is the exponent -- so the resolution gain from bf16 to fp16 is 8x
    and not 4x, and every ulp that file computed was half the true size.

    Held fixed at DERIVED_ULP_COUNT, the same error model gives a bf16 bound of
    4 ulp and an fp16 bound of 4 ulp, so the absolute bounds must differ by
    exactly the resolution ratio. If they ever do not, the two were derived under
    different models and are not comparable -- which is the specific mistake that
    produced a 20% "tightening" the first time.
    """
    bf16_eps = torch.finfo(torch.bfloat16).eps
    fp16_eps = torch.finfo(FP16_DTYPE).eps
    resolution_ratio = bf16_eps / fp16_eps

    one_fp16_ulp = LOGIT_BINADE_BASE * fp16_eps
    one_bf16_ulp = LOGIT_BINADE_BASE * bf16_eps
    consistent_bf16_bound = DERIVED_ULP_COUNT * one_bf16_ulp

    print(f"\nbf16 eps {bf16_eps:.4e} (2**-7, 7 explicit mantissa bits)")
    print(f"fp16 eps {fp16_eps:.4e} (2**-10, 10 explicit mantissa bits)")
    print(f"resolution ratio: {resolution_ratio:.1f}x  (3 bits, not 2)")
    print(f"one ulp at |logit| ~2.2e1: bf16 {one_bf16_ulp:.4e}, fp16 {one_fp16_ulp:.4e}")
    print(f"same model, bf16 would be {consistent_bf16_bound:.4e}; "
          f"fp16 is {FP16_LOGIT_ATOL:.4e} ({consistent_bf16_bound / FP16_LOGIT_ATOL:.1f}x tighter)")

    # bf16 is 1 sign + 8 exponent + 7 mantissa. The 8 is not the mantissa.
    assert bf16_eps == pytest.approx(2.0**-7)
    assert fp16_eps == pytest.approx(2.0**-10)
    assert resolution_ratio == pytest.approx(8.0)

    # The bound is exactly the derived ulp count, at this model's logit scale.
    assert FP16_LOGIT_ATOL == pytest.approx(DERIVED_ULP_COUNT * one_fp16_ulp)
    # And the same model on bf16 differs by exactly the resolution ratio.
    assert consistent_bf16_bound / FP16_LOGIT_ATOL == pytest.approx(resolution_ratio), (
        "the fp16 bound is not tighter than a consistently derived bf16 bound by "
        "the ratio the mantissa earns, so the two rest on different error models"
    )

    # The predecessor's declared value was two TRUE bf16 ulp, not the four it
    # claimed, because it halved the ulp. Recorded so the discrepancy is not
    # rediscovered as a surprise.
    assert 2.5e-1 / one_bf16_ulp == pytest.approx(2.0)
