# DRAFT — comment for huggingface/transformers#45776

**Status: NOT POSTED. Nav reviews and posts this, under his own name, or not.**

- Target: <https://github.com/huggingface/transformers/issues/45776> —
  *"Gate `enable_gqa=True` on actual flash-attention eligibility"* (open, draft PR)
- **Do not open a new issue.** #45776 already describes this mechanism. This is
  supporting data for the existing thread.
- Attach: [`repro_enable_gqa_padding_cliff.py`](repro_enable_gqa_padding_cliff.py)
  (standalone, torch + transformers only, no dependency on this repo).

## Why comment rather than file

#45776 already identifies the mechanism and lists `sm < 80` among the FA
constraints that cause the silent MATH fallback. What it does not have:

1. **any sm_75 data** — its benchmarks are RTX 3090 (sm_86) and RTX 4090
   (sm_89), both of which *have* FLASH, so `enable_gqa` is honoured there and the
   fallback never fires;
2. **an end-to-end figure larger than ~2.5%** — that is its own number, from a
   135M-parameter model;
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
> **Which backend actually runs.** Determined by pinning each backend with
> `sdpa_kernel(...)` and comparing bit-for-bit against the unrestricted call,
> not inferred from shapes:
>
> | K/V form | `enable_gqa` | backend selected on sm_75 |
> |---|---|---|
> | unexpanded (what `use_gqa_in_sdpa() == True` produces) | `True` | `MATH` |
> | pre-expanded via `repeat_kv` | — | `EFFICIENT_ATTENTION` |
>
> `FLASH_ATTENTION` and `CUDNN_ATTENTION` reject every shape tested on this
> device.
>
> **What it costs, end to end.** Qwen2.5-1.5B-Instruct, fp16, real
> `model.generate()`, 256 requests of 512 prompt + 512 decode tokens, median of
> 3 measured passes after a discarded warmup (pass spread ≤ 0.2%). The only
> change is forcing `use_gqa_in_sdpa` to return `False`:
>
> | batch shape | as shipped | K/V pre-expanded | speedup |
> |---|---|---|---|
> | 18 resident | 177.8 tok/s | 288.9 tok/s | **1.62×** |
> | 32 resident | 202.0 tok/s | 370.9 tok/s | **1.84×** |
>
> Identical work in both columns — same prompts, same token counts, greedy,
> `min_new_tokens == max_new_tokens`, and the batch is uniform so
> `attention_mask` is `None` on both sides. The only difference is the return
> value of `use_gqa_in_sdpa`.
>
> **The part I think is worth calling out separately.** On CUDA with torch >= 2.5,
> `use_gqa_in_sdpa()` returns `True` exactly when `attention_mask is None`, which
> for `generate()`
> means exactly when nothing was left-padded. So on sm < 80 the consequence is
> inverted from what a user would ever guess:
>
> - a batch of **equal-length** prompts takes the slow kernel;
> - adding one shorter prompt forces a mask, HF pre-expands, and the same batch
>   takes the **fast** one.
>
> **Padding your inputs makes it faster.** That is invisible from the outside —
> no warning, no fallback message, and the config that looks obviously optimal
> (uniform batch, no padding waste) is the one that loses ~1.6–1.8× end to end.
>
> **torch says why, in its own words.** Running the pinned-backend probe emits,
> verbatim:
>
> ```
> Flash attention only supports gpu architectures in the range [sm80, sm121].
>   Attempting to run on a sm 7.5 gpu.
> For dense input, both fused kernels require query, key and value to have the
>   same num_heads. Query.sizes(): [8, 12, 512, 128],
>   Key sizes(): [8, 2, 512, 128], Value sizes(): [8, 2, 512, 128] instead.
> ```
>
> The second line is the whole thing: the unexpanded K/V shape is *why* both
> fused kernels decline, and `enable_gqa=True` is what produces that shape.
>
> **Standalone repro attached.** Two `generate()` calls doing identical work on
> the same uniform batch, one as shipped and one with pre-expansion forced,
> printing the selected backend for each and the timing delta. Its own default
> shape is smaller than the table above — batch 8, 512-token prompts, 64 new
> tokens — and on the T4 it reports:
>
> ```
> uniform batch, as HF ships it    enable_gqa=True   -> ['MATH']
> same batch, K/V pre-expanded     enable_gqa=False  -> ['EFFICIENT_ATTENTION']
> enable_gqa   (as shipped) :   3.394s  [3.391, 3.415]
> pre-expanded K/V          :   2.520s  [2.497, 2.522]
> pre-expanding is 1.35x faster on IDENTICAL work
> ```
>
> **1.35×, not the 1.6–1.8× above** — the gap narrows at 64 new tokens because
> prefill is a larger share of the run, and widens with decode length and batch.
> Both are real; the repro is deliberately small so it finishes in seconds.
> Pass `--batch 32 --new-tokens 512` to approach the table.
>
> **I have not run any of this on sm ≥ 80** — no Ampere-or-later device was
> available — so what the script reports on your 3090/4090 is unknown to me. If
> FLASH can honour `enable_gqa` there, both columns should name the same backend
> and the delta should collapse; that is a prediction, not a measurement.
>
> Not attached to any opinion on the fix — the `SDPAParams` / fullgraph blocker
> in this thread is real and I have nothing to add on it. Flagging only that on
> sm < 80 the fallback is not an edge case, it is the default path for unpadded
> batches, and the size is closer to 1.6–1.8× end to end than to 2.5%.

---

## Provenance for anything in the comment

| claim | source |
|---|---|
| backends selected, both K/V forms | `kaggle/session5` gate 2, `kaggle/session3` |
| FLASH/CUDNN reject every shape on sm_75 | `docs/AMPERE_RECHECK.md` item 2 |
| 1.62× / 1.84× end to end, uniform mask-free batch | `bench/results/arms/arm{A,B}-control-rect{3072,1757}-C64.json`, baseline levels |
| `use_gqa_in_sdpa` returns True iff mask is None | transformers 5.12.1 and `main`, verified 2026-08 |
| repro output, 1.35× at batch 8 / 64 new tokens | `kaggle/session9/session9_report.json` — script run standalone from a bare directory, rc=0 |
| torch's own warning text | same run, verbatim from stderr |

**Deliberately NOT included:** the per-attention-call gap (2.76×–5.67×) and the
transient-memory ratio (3.3×–8.3×) from `kaggle/session4`. Those were measured
with an explicit boolean mask on **both** paths, which is this engine's shape,
not HuggingFace's — HF's `enable_gqa` path only ever runs with `attn_mask=None`.
They do not characterise HF's two shipped paths and would be wrong to offer
upstream as if they did.

Every figure is from a run that happened. The 1.62×/1.84× pair are **baseline
levels only** — the paged engine is not part of this comparison and should not
be mentioned upstream; it is the HuggingFace baseline measured against itself
with one boolean flipped.
