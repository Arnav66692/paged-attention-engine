# DRAFT — comment for huggingface/transformers#45776

**Status: NOT POSTED. Nav reviews and posts this, under his own name, or not.**

- Target: <https://github.com/huggingface/transformers/issues/45776> —
  *"Gate `enable_gqa=True` on actual flash-attention eligibility"* (open, draft PR)
- **Do not open a new issue.** #45776 already describes this mechanism. This is
  supporting data for the existing thread.
- The repro script is INLINED in the comment body in a collapsed details
  block. GitHub comments cannot attach .py files, and a comment that says
  "attached" with nothing attached is the fastest way to be dismissed.

## Why comment rather than file

#45776 already identifies the mechanism and lists `sm < 80` among the FA
constraints that cause the silent MATH fallback. What it does not have:

1. **any sm_75 data** — its benchmarks are on sm_86 and sm_89 hardware, both of
   which *have* FLASH, so `enable_gqa` is honoured there and the fallback never
   fires;
2. **any measurement of how large the fallback is.** Its `+2.6%` (22.943 ms ->
   23.535 ms, SmolLM2-135M, sm_89) is the OVERHEAD OF THE ELIGIBILITY PROBE on
   hardware where FLASH exists and the fallback never fires — that is the PR
   being careful about the cost of its own fix, and it is not a measurement of
   the problem. Nothing in the thread sizes the fallback itself;
3. **the inverted user-facing consequence** — on sm < 80 the
   `attention_mask is None` path selects the *slower* kernel, so padding your
   inputs makes HuggingFace faster. Nothing in the thread says this.

The measurements below are on a device where FLASH does not exist at all, which
is the regime the gating change is meant to protect and the one with no data.

---

## Draft comment text

> Some data from a device class this thread doesn't cover yet: **Turing, sm_75
> (Tesla T4), where FLASH is unavailable for every shape**, so `enable_gqa` has
> no backend that can honour it and the fallback fires unconditionally rather
> than at the margins.
>
> **Which backend gets selected.** Determined by pinning each backend with
> `sdpa_kernel(...)` and comparing bit-for-bit against the unrestricted call:
>
> | K/V form | `enable_gqa` | backend selected on sm_75 |
> |---|---|---|
> | unexpanded (what `use_gqa_in_sdpa() == True` produces) | `True` | `MATH` |
> | pre-expanded via `repeat_kv` | not set | `EFFICIENT_ATTENTION` |
>
> `FLASH_ATTENTION` and `CUDNN_ATTENTION` reject every shape tested on this
> device.
>
> Everything below: torch 2.10.0+cu128, transformers 5.12.1, fp16, Kaggle
> Tesla T4.
>
> **What it costs, end to end.** Qwen2.5-1.5B-Instruct, fp16, real
> `model.generate()`, 256 requests of 512 prompt + 512 decode tokens, median of
> 3 measured passes after a discarded warmup (pass spread under 0.2%). The only
> change is forcing `use_gqa_in_sdpa` to return `False`:
>
> | concurrent sequences | as shipped | K/V pre-expanded | speedup |
> |---|---|---|---|
> | 18 | 177.8 tok/s | 288.9 tok/s | **1.62x** |
> | 32 | 202.0 tok/s | 370.9 tok/s | **1.84x** |
>
> Identical work in both columns. Same prompts, same token counts, greedy,
> `min_new_tokens == max_new_tokens`, and the batch is uniform so
> `attention_mask` is `None` on both sides. The only difference is the return
> value of `use_gqa_in_sdpa`.
>
> **The part I think is worth calling out separately.** On CUDA with torch 2.5 or newer,
> `use_gqa_in_sdpa()` returns `True` exactly when `attention_mask is None`, which
> for `generate()`
> means exactly when nothing was left-padded. So on sm < 80 the consequence is
> inverted from what a user would ever guess:
>
> - a batch of **equal-length** prompts takes the slow kernel;
> - adding one shorter prompt forces a mask, HF pre-expands, and the same batch
>   takes the **fast** one.
>
> **Padding your inputs makes it faster.** That is invisible from the outside.
> No warning, no fallback message, and the configuration that looks optimal,
> a uniform batch with no padding waste, is the one that loses 1.6 to 1.8x.
>
> **torch says why, in its own words.** Running the pinned-backend probe emits
> these two warnings, rewrapped here and trimmed after `instead.`:
>
> ```
> Flash attention only supports gpu architectures in the range [sm80, sm121].
>   Attempting to run on a sm 7.5 gpu.
> For dense input, both fused kernels require query, key and value to have the
>   same num_heads. Query.sizes(): [8, 12, 512, 128],
>   Key sizes(): [8, 2, 512, 128], Value sizes(): [8, 2, 512, 128] instead.
> ```
>
> The second warning is the mechanism this PR already describes: the
> unexpanded K/V shape is *why* both
> fused kernels decline, and `enable_gqa=True` is what produces that shape.
>
> **Standalone repro, inlined at the bottom of this comment.** Two `generate()`
> calls doing identical work on the same uniform batch, one as shipped and one with pre-expansion forced,
> printing the selected backend for each and the timing delta. Its own default
> shape is smaller than the table above, batch 8 with 512-token prompts and 64
> new tokens, and on the T4 it reports (abridged):
>
> ```
> uniform batch, as HF ships it    enable_gqa=True   -> ['MATH']
> same batch, K/V pre-expanded     enable_gqa=False  -> ['EFFICIENT_ATTENTION']
> enable_gqa   (as shipped) :   3.374s
> pre-expanded K/V          :   2.683s
> pre-expanding is 1.26x faster on IDENTICAL work
> ```
>
> The backend selection is stable. The ratio is not: two runs of this script on
> the same T4 gave 1.35x and 1.26x, so expect roughly 1.3x rather than a precise
> figure, and treat anything in that neighbourhood as agreement. It is smaller
> than the 1.6 to 1.8x above because at 64 new tokens prefill is a larger share
> of the run; the gap widens with decode length and batch. The repro is
> deliberately small so it finishes in seconds. Pass
> `--batch 32 --new-tokens 512` to move it toward the table.
>
> **I have not run any of this on Ampere or later.** No such device was
> available, so what the script reports on sm_86 or sm_89 is unknown to me. If
> FLASH can honour `enable_gqa` there, both columns should name the same backend
> and the delta should collapse; that is a prediction, not a measurement.
>
> No opinion on the fix itself. One thing that may have changed since this
> thread went quiet: the `SDPAParams` / fullgraph blocker it references,
> pytorch#182684, is now closed as completed (2026-07-25). I have not verified
> where the fix landed, but it may be worth a re-check of the torch-native
> eligibility path. The `+2.6%` probe overhead in the
> thread is a different quantity from anything above, since it is the cost of
> the check on hardware where the fallback never fires. The only thing I am adding is that on
> sm < 80 the fallback is not an edge case but the default path for unpadded
> batches, and that it is worth 1.6 to 1.8x there.
>
> <details>
> <summary>repro_enable_gqa_padding_cliff.py (torch + transformers only, no other dependencies)</summary>
>
> ```python
> #!/usr/bin/env python3
> """Standalone reproduction: on sm < 80, HuggingFace's no-padding path is the SLOW one.
>
> Depends on nothing but torch + transformers. Copy this file anywhere and run it.
>
>     pip install "torch>=2.5" "transformers>=5.0"
>     python repro_enable_gqa_padding_cliff.py
>
> WHAT IT SHOWS
>
> `transformers.integrations.sdpa_attention.use_gqa_in_sdpa` returns True -- and
> so `sdpa_attention_forward` passes unexpanded K/V with `enable_gqa=True` --
> exactly when `attention_mask is None`, which for `generate()` means exactly when
> nothing was left-padded.
>
> On a GPU where FLASH is available that is the fast path. On a GPU where it is
> NOT (sm < 80: T4, V100, RTX 20xx, GTX 16xx), torch's SDPA cannot honour
> `enable_gqa` on any remaining backend and dispatches to MATH. EFFICIENT_ATTENTION
> would have run, and does run, as soon as K/V are pre-expanded.
>
> The consequence is inverted from what anyone would guess: a batch of
> EQUAL-LENGTH prompts takes the SLOW kernel, because no padding means no mask
> means `enable_gqa`. A ragged batch forces a mask, so HF pre-expands, so it takes
> the fast one.
>
> WHAT THIS SCRIPT DOES AND DOES NOT MEASURE. It holds ONE uniform batch fixed and
> flips `use_gqa_in_sdpa` to force the other path, so both timings do byte-for-byte
> identical work and the delta is attributable to the kernel alone. It does NOT
> time a ragged batch against a uniform one -- those would be different workloads
> and the comparison would prove nothing. The ragged case is covered by the
> backend readout, which shows which path HF takes for each.
>
> WHAT IT PRINTS
>
>   1. which SDPA backend each configuration actually dispatches to, determined by
>      running the call under each backend in turn and comparing bit-for-bit
>      against the unrestricted call, not by inference from shapes;
>   2. wall time of two real `model.generate()` calls doing IDENTICAL work, one
>      with a uniform batch (mask None -> enable_gqa) and one with the same batch
>      forced onto the pre-expanded path.
>
> Both timings generate the same number of tokens from the same prompts, so the
> ratio is not a workload difference.
> """
>
> import argparse
> import statistics
> import time
>
> import torch
> import torch.nn.functional as F
> from torch.nn.attention import SDPBackend, sdpa_kernel
> from transformers import AutoModelForCausalLM, AutoTokenizer
> from transformers.integrations import sdpa_attention
>
> REAL_SDPA = F.scaled_dot_product_attention
> BACKENDS = {
>     "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
>     "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
>     "MATH": SDPBackend.MATH,
>     "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
> }
>
>
> def identify(query, key, value, **kwargs):
>     """Which kernel did the unrestricted call use? Determined by running it.
>
>     Pinning one backend at a time and comparing bit-for-bit against the free
>     call is the only way to know; the dispatch rules are not public API and
>     inferring them from shapes is how this gets misdiagnosed.
>     """
>     reference = REAL_SDPA(query, key, value, **kwargs)
>     torch.cuda.synchronize()
>     matches = []
>     for name, backend in BACKENDS.items():
>         try:
>             with sdpa_kernel(backend):
>                 candidate = REAL_SDPA(query, key, value, **kwargs)
>             torch.cuda.synchronize()
>             if torch.equal(candidate, reference):
>                 matches.append(name)
>         except Exception:      # noqa: BLE001 -- "no available kernel" is a result
>             continue
>     return matches
>
>
> def force_pre_expansion():
>     """Make HF take the repeat_kv path unconditionally. Returns an undo."""
>     original = sdpa_attention.use_gqa_in_sdpa
>
>     def never(*args, **kwargs):
>         return False
>
>     sdpa_attention.use_gqa_in_sdpa = never
>
>     def undo():
>         sdpa_attention.use_gqa_in_sdpa = original
>
>     return undo
>
>
> def observe(model, input_ids, attention_mask, steps, label):
>     """Run one generate, recording each distinct SDPA shape and its backend."""
>     seen: dict = {}
>
>     def recording(query, key, value, *args, **kwargs):
>         signature = (tuple(query.shape), tuple(key.shape),
>                      kwargs.get("attn_mask") is not None,
>                      bool(kwargs.get("enable_gqa", False)))
>         if signature not in seen:
>             seen[signature] = {
>                 "q": list(query.shape), "k": list(key.shape),
>                 "enable_gqa": signature[3],
>                 "kv_expanded": query.shape[1] == key.shape[1],
>                 "backends": identify(query, key, value, *args, **kwargs),
>             }
>         return REAL_SDPA(query, key, value, *args, **kwargs)
>
>     F.scaled_dot_product_attention = recording
>     try:
>         with torch.inference_mode():
>             model.generate(input_ids=input_ids, attention_mask=attention_mask,
>                            max_new_tokens=steps, min_new_tokens=steps,
>                            do_sample=False, num_beams=1, use_cache=True,
>                            pad_token_id=0, eos_token_id=None)
>     finally:
>         F.scaled_dot_product_attention = REAL_SDPA
>
>     backends = sorted({b for s in seen.values() for b in s["backends"]})
>     expanded = all(s["kv_expanded"] for s in seen.values())
>     print(f"  {label:34s} enable_gqa={not expanded!s:5s} "
>           f"kv_pre_expanded={expanded!s:5s} -> {backends}")
>     return backends
>
>
> def timed(model, input_ids, attention_mask, steps, repeats):
>     with torch.inference_mode():                       # warmup, discarded
>         model.generate(input_ids=input_ids, attention_mask=attention_mask,
>                        max_new_tokens=steps, min_new_tokens=steps, do_sample=False,
>                        num_beams=1, use_cache=True, pad_token_id=0, eos_token_id=None)
>     torch.cuda.synchronize()
>     samples = []
>     for _ in range(repeats):
>         started = time.perf_counter()
>         with torch.inference_mode():
>             model.generate(input_ids=input_ids, attention_mask=attention_mask,
>                            max_new_tokens=steps, min_new_tokens=steps,
>                            do_sample=False, num_beams=1, use_cache=True,
>                            pad_token_id=0, eos_token_id=None)
>         torch.cuda.synchronize()
>         samples.append(time.perf_counter() - started)
>     return statistics.median(samples), min(samples), max(samples)
>
>
> def main() -> int:
>     parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
>     parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
>                         help="any GQA model; default has 12 q heads over 2 kv heads")
>     parser.add_argument("--batch", type=int, default=8)
>     parser.add_argument("--prompt-len", type=int, default=512)
>     parser.add_argument("--new-tokens", type=int, default=64)
>     parser.add_argument("--repeats", type=int, default=3)
>     args = parser.parse_args()
>
>     if not torch.cuda.is_available():
>         print("CUDA required: this is about GPU kernel dispatch.")
>         return 2
>
>     major, minor = torch.cuda.get_device_capability(0)
>     print(f"device      : {torch.cuda.get_device_name(0)} (sm_{major}{minor})")
>     print(f"torch       : {torch.__version__}")
>     import transformers
>     print(f"transformers: {transformers.__version__}")
>     print(f"flash available for this device: {major >= 8}"
>           f"{'  <-- sm < 80, so enable_gqa has no fast backend' if major < 8 else ''}\n")
>
>     model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
>     model.eval().to("cuda")
>     model.set_attn_implementation("sdpa")
>     tokenizer = AutoTokenizer.from_pretrained(args.model)
>
>     torch.manual_seed(0)
>     vocab = min(getattr(tokenizer, "vocab_size", 30000) or 30000, 30000)
>     input_ids = torch.randint(1, vocab, (args.batch, args.prompt_len), device="cuda")
>     # UNIFORM lengths -> transformers drops the all-ones mask to None.
>     attention_mask = torch.ones_like(input_ids)
>
>     print("BACKEND SELECTED, measured by pinning each backend and comparing bitwise")
>     unpadded = observe(model, input_ids, attention_mask, 4,
>                        "uniform batch, as HF ships it")
>     undo = force_pre_expansion()
>     try:
>         padded = observe(model, input_ids, attention_mask, 4,
>                          "same batch, K/V pre-expanded")
>     finally:
>         undo()
>
>     print(f"\nTIMING, {args.batch}x{args.prompt_len} prompt, {args.new_tokens} new "
>           f"tokens, median of {args.repeats} after 1 warmup")
>     slow = timed(model, input_ids, attention_mask, args.new_tokens, args.repeats)
>     undo = force_pre_expansion()
>     try:
>         fast = timed(model, input_ids, attention_mask, args.new_tokens, args.repeats)
>     finally:
>         undo()
>
>     print(f"  enable_gqa   (as shipped) : {slow[0]:7.3f}s  "
>           f"[{slow[1]:.3f}, {slow[2]:.3f}]  {unpadded}")
>     print(f"  pre-expanded K/V          : {fast[0]:7.3f}s  "
>           f"[{fast[1]:.3f}, {fast[2]:.3f}]  {padded}")
>     print(f"\n  pre-expanding is {slow[0] / fast[0]:.2f}x faster on IDENTICAL work")
>
>     if unpadded == padded:
>         print("\n  Both configurations chose the same backend on this device, so "
>               "there is no cliff here. That is the expected outcome wherever "
>               "FLASH can honour enable_gqa -- but this script's author only ran "
>               "it on sm_75, so treat sm >= 80 behaviour as unverified.")
>     else:
>         print(f"\n  Both timings above ran the SAME uniform batch; the only "
>               f"difference is the K/V input shape, forced by the monkeypatch. "
>               f"HuggingFace picks the first form whenever attention_mask is None "
>               f"-- i.e. whenever nothing was padded -- so on this device the "
>               f"no-padding case is the {slow[0] / fast[0]:.2f}x SLOWER one. A "
>               f"genuinely ragged batch would take the fast path here, but it "
>               f"would also be different work, which is why this script holds the "
>               f"batch fixed and flips the flag instead.")
>     return 0
>
>
> if __name__ == "__main__":
>     raise SystemExit(main())
> ```
>
> </details>

---

## Provenance for anything in the comment

| claim | source |
|---|---|
| backends selected, both K/V forms | `kaggle/session5` gate 2, `kaggle/session3` |
| FLASH/CUDNN reject every shape on sm_75 | `docs/AMPERE_RECHECK.md` item 2 |
| 1.62x / 1.84x end to end, uniform mask-free batch | `kaggle/session6a/arm{A,B}-control-rect3072-C64.json` and `kaggle/session6b/arm{A,B}-control-rect1757-C64.json`, baseline levels (the `bench/results/arms/` working copies are byte-identical but untracked) |
| `use_gqa_in_sdpa` returns True iff mask is None | transformers 5.12.1 and `main`, verified 2026-08 |
| repro output, 1.35x and 1.26x on two runs | `kaggle/session9/session9_report.json` and `session9_report_run2.json`, both standalone from a bare directory, rc=0 |
| torch's own warning text | same run, verbatim from stderr |

**Deliberately NOT included:** the per-attention-call gap (2.76 to 5.67x) and the
transient-memory ratio (3.3 to 8.3x) from `kaggle/session4`. Those were measured
with an explicit boolean mask on **both** paths, which is this engine's shape,
not HuggingFace's — HF's `enable_gqa` path only ever runs with `attn_mask=None`.
They do not characterise HF's two shipped paths and would be wrong to offer
upstream as if they did.

Every figure is from a run that happened. The 1.62x / 1.84x pair are **baseline
levels only** — the paged engine is not part of this comparison and should not
be mentioned upstream; it is the HuggingFace baseline measured against itself
with one boolean flipped.
