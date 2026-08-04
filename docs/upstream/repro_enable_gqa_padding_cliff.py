#!/usr/bin/env python3
"""Standalone reproduction: on sm < 80, PADDING A BATCH MAKES HUGGINGFACE FASTER.

Depends on nothing but torch + transformers. Copy this file anywhere and run it.

    pip install "torch>=2.5" "transformers>=5.0"
    python repro_enable_gqa_padding_cliff.py

WHAT IT SHOWS

`transformers.integrations.sdpa_attention.use_gqa_in_sdpa` returns True -- and
so `sdpa_attention_forward` passes unexpanded K/V with `enable_gqa=True` --
exactly when `attention_mask is None`, which for `generate()` means exactly when
nothing was left-padded.

On a GPU where FLASH is available that is the fast path. On a GPU where it is
NOT (sm < 80: T4, V100, RTX 20xx, GTX 16xx), torch's SDPA cannot honour
`enable_gqa` on any remaining backend and dispatches to MATH. EFFICIENT_ATTENTION
would have run, and does run, as soon as K/V are pre-expanded.

The consequence is inverted from what anyone would guess: a batch of
EQUAL-LENGTH prompts takes the slow kernel, and adding one shorter prompt --
which forces a mask, which makes HF pre-expand -- takes the fast one.

WHAT IT PRINTS

  1. which SDPA backend each configuration actually dispatches to, determined by
     running the call under each backend in turn and comparing bit-for-bit
     against the unrestricted call, not by inference from shapes;
  2. wall time of two real `model.generate()` calls doing IDENTICAL work, one
     with a uniform batch (mask None -> enable_gqa) and one with the same batch
     forced onto the pre-expanded path.

Both timings generate the same number of tokens from the same prompts, so the
ratio is not a workload difference.
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import sdpa_attention

REAL_SDPA = F.scaled_dot_product_attention
BACKENDS = {
    "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
    "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
    "MATH": SDPBackend.MATH,
    "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
}


def identify(query, key, value, **kwargs):
    """Which kernel did the unrestricted call use? Determined by running it.

    Pinning one backend at a time and comparing bit-for-bit against the free
    call is the only way to know; the dispatch rules are not public API and
    inferring them from shapes is how this gets misdiagnosed.
    """
    reference = REAL_SDPA(query, key, value, **kwargs)
    torch.cuda.synchronize()
    matches = []
    for name, backend in BACKENDS.items():
        try:
            with sdpa_kernel(backend):
                candidate = REAL_SDPA(query, key, value, **kwargs)
            torch.cuda.synchronize()
            if torch.equal(candidate, reference):
                matches.append(name)
        except Exception:      # noqa: BLE001 -- "no available kernel" is a result
            continue
    return matches


def force_pre_expansion():
    """Make HF take the repeat_kv path unconditionally. Returns an undo."""
    original = sdpa_attention.use_gqa_in_sdpa

    def never(*args, **kwargs):
        return False

    sdpa_attention.use_gqa_in_sdpa = never

    def undo():
        sdpa_attention.use_gqa_in_sdpa = original

    return undo


def observe(model, input_ids, attention_mask, steps, label):
    """Run one generate, recording each distinct SDPA shape and its backend."""
    seen: dict = {}

    def recording(query, key, value, *args, **kwargs):
        signature = (tuple(query.shape), tuple(key.shape),
                     kwargs.get("attn_mask") is not None,
                     bool(kwargs.get("enable_gqa", False)))
        if signature not in seen:
            seen[signature] = {
                "q": list(query.shape), "k": list(key.shape),
                "enable_gqa": signature[3],
                "kv_expanded": query.shape[1] == key.shape[1],
                "backends": identify(query, key, value, *args, **kwargs),
            }
        return REAL_SDPA(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = recording
    try:
        with torch.inference_mode():
            model.generate(input_ids=input_ids, attention_mask=attention_mask,
                           max_new_tokens=steps, min_new_tokens=steps,
                           do_sample=False, num_beams=1, use_cache=True,
                           pad_token_id=0, eos_token_id=None)
    finally:
        F.scaled_dot_product_attention = REAL_SDPA

    backends = sorted({b for s in seen.values() for b in s["backends"]})
    expanded = all(s["kv_expanded"] for s in seen.values())
    print(f"  {label:34s} enable_gqa={not expanded!s:5s} "
          f"kv_pre_expanded={expanded!s:5s} -> {backends}")
    return backends


def timed(model, input_ids, attention_mask, steps, repeats):
    with torch.inference_mode():                       # warmup, discarded
        model.generate(input_ids=input_ids, attention_mask=attention_mask,
                       max_new_tokens=steps, min_new_tokens=steps, do_sample=False,
                       num_beams=1, use_cache=True, pad_token_id=0, eos_token_id=None)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.inference_mode():
            model.generate(input_ids=input_ids, attention_mask=attention_mask,
                           max_new_tokens=steps, min_new_tokens=steps,
                           do_sample=False, num_beams=1, use_cache=True,
                           pad_token_id=0, eos_token_id=None)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), min(samples), max(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="any GQA model; default has 12 q heads over 2 kv heads")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required: this is about GPU kernel dispatch.")
        return 2

    major, minor = torch.cuda.get_device_capability(0)
    print(f"device      : {torch.cuda.get_device_name(0)} (sm_{major}{minor})")
    print(f"torch       : {torch.__version__}")
    import transformers
    print(f"transformers: {transformers.__version__}")
    print(f"flash available for this device: {major >= 8}"
          f"{'  <-- sm < 80, so enable_gqa has no fast backend' if major < 8 else ''}\n")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.eval().to("cuda")
    model.set_attn_implementation("sdpa")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    torch.manual_seed(0)
    vocab = min(getattr(tokenizer, "vocab_size", 30000) or 30000, 30000)
    input_ids = torch.randint(1, vocab, (args.batch, args.prompt_len), device="cuda")
    # UNIFORM lengths -> transformers drops the all-ones mask to None.
    attention_mask = torch.ones_like(input_ids)

    print("BACKEND SELECTED, measured by pinning each backend and comparing bitwise")
    unpadded = observe(model, input_ids, attention_mask, 4,
                       "uniform batch, as HF ships it")
    undo = force_pre_expansion()
    try:
        padded = observe(model, input_ids, attention_mask, 4,
                         "same batch, K/V pre-expanded")
    finally:
        undo()

    print(f"\nTIMING, {args.batch}x{args.prompt_len} prompt, {args.new_tokens} new "
          f"tokens, median of {args.repeats} after 1 warmup")
    slow = timed(model, input_ids, attention_mask, args.new_tokens, args.repeats)
    undo = force_pre_expansion()
    try:
        fast = timed(model, input_ids, attention_mask, args.new_tokens, args.repeats)
    finally:
        undo()

    print(f"  enable_gqa   (as shipped) : {slow[0]:7.3f}s  "
          f"[{slow[1]:.3f}, {slow[2]:.3f}]  {unpadded}")
    print(f"  pre-expanded K/V          : {fast[0]:7.3f}s  "
          f"[{fast[1]:.3f}, {fast[2]:.3f}]  {padded}")
    print(f"\n  pre-expanding is {slow[0] / fast[0]:.2f}x faster on IDENTICAL work")

    if unpadded == padded:
        print("\n  Both configurations chose the same backend on this device, so "
              "there is no cliff here. Expected on sm >= 80, where FLASH can "
              "honour enable_gqa.")
    else:
        print(f"\n  The only difference is the K/V input SHAPE. HuggingFace picks "
              f"the first form whenever attention_mask is None -- i.e. whenever "
              f"nothing was padded -- so on this device an UNPADDED batch is "
              f"{slow[0] / fast[0]:.2f}x SLOWER than the same batch padded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
