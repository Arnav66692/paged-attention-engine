# A paged-attention inference engine, built from scratch

A block-allocated paged KV cache, continuous batching, preemption with
recompute, and a fused batched forward. Written to find out what an inference
engine is actually made of, and benchmarked against naive HuggingFace
`generate` at an identical KV byte ceiling.

```bash
pip install -r requirements-dev.txt
pytest                    # ~12 min. `pytest -m slow` is a second ~3.
```

```python
from transformers import AutoModelForCausalLM
from engine.model import Qwen2Engine
from engine.cache import BlockAllocator
from engine.scheduler import ExecutionMode, Qwen2Executor, Request, Scheduler

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").eval().cuda()
engine = Qwen2Engine.from_hf_model(model)

allocator = BlockAllocator(num_blocks=3584, block_size=16)
cache = engine.new_paged_cache(allocator.num_blocks, allocator.block_size)
scheduler = Scheduler(Qwen2Executor(engine, cache, allocator), allocator,
                      mode=ExecutionMode.FUSED, max_num_seqs=64,
                      max_num_batched_tokens=8192)

scheduler.add_request(Request("r0", prompt_token_ids=(1, 2, 3), max_new_tokens=32))
outputs = scheduler.run_to_completion()      # {request_id: (token, ...)}
```

---

## The result

Engine tokens/s over baseline tokens/s, Kaggle Tesla T4, fp16, both sides
holding the same 1,644,167,168-byte KV ceiling.

| request stream | ratio |
|---|---|
| **ragged** — 120/224/288 distinct prompt lengths, 40–2048 tokens | **1.89× – 2.06×** at C=32, 64, 87 and both cache sizings |
| **uniform** — every prompt 512 + 512 | **0.66× – 0.79×** — a loss |

**The opponent is weak and that matters.** Naive static-batch `generate` is not
vLLM, TGI, or anything in production, and none of those were ever benchmarked
here. Beating static batching on a stream whose prompts span 40 to 2048 tokens
is close to the arithmetic you would estimate on a napkin. The uniform-stream
loss is the more informative half: with no straggler waste and no fragmentation
there is nothing for the engine to reclaim, so it pays its own bookkeeping for
no return.

**The ragged ratio depends on both sides running the same attention backend, and
getting that wrong in either direction moves it by 2×.** Under this engine's
previous default it sat on `MATH` while HuggingFace's ragged batches already had
`EFFICIENT_ATTENTION` — same code, same stream, 0.94× instead of 1.89×. Any
benchmark of this kind is one silent kernel selection away from being a kernel
comparison. [`bench/arms.py`](bench/arms.py) runs the whole sweep twice to bound
it: the shared kernel is worth a measured ±5%, and the sign is not constant
(+2.30% at one cache sizing, −4.77% at the other).

Full tables, the refuted hypotheses, and the rejection confound:
**[`docs/FINDINGS.md`](docs/FINDINGS.md)**.

---

## What the benchmark found in HuggingFace

`transformers` selects a slower attention kernel when your prompts are all the
same length.

`use_gqa_in_sdpa()` returns `True` exactly when `attention_mask is None`, which
for `generate()` means exactly when nothing was left-padded. It then passes
unexpanded K/V with `enable_gqa=True`. On any GPU where FLASH is unavailable —
sm < 80: T4, V100, RTX 20xx — no remaining backend can honour that flag, so
torch falls back to `MATH`. Pre-expanding K/V reaches `EFFICIENT_ATTENTION`
instead.

So a batch of equal-length prompts takes the slow path, and adding one shorter
prompt forces a mask, makes HF pre-expand, and takes the fast one. Measured on a
T4 against HuggingFace with one boolean flipped and nothing else changed:

| | as shipped | K/V pre-expanded | |
|---|---|---|---|
| 18 resident | 177.8 tok/s | 288.9 tok/s | **1.62×** |
| 32 resident | 202.0 tok/s | 370.9 tok/s | **1.84×** |

torch says why itself:

```
Flash attention only supports gpu architectures in the range [sm80, sm121].
  Attempting to run on a sm 7.5 gpu.
For dense input, both fused kernels require query, key and value to have the
  same num_heads. Query.sizes(): [8, 12, 512, 128],
  Key sizes(): [8, 2, 512, 128], Value sizes(): [8, 2, 512, 128] instead.
```

Standalone reproduction, torch + transformers only:
[`docs/upstream/repro_enable_gqa_padding_cliff.py`](docs/upstream/repro_enable_gqa_padding_cliff.py).
Its defaults are smaller than the table — batch 8, 64 new tokens — and it reports
roughly **1.3×** (1.35 and 1.26 on two runs); the gap widens with decode length and batch. Written up on its own at
[`docs/hf-padding-kernel-cliff.md`](docs/hf-padding-kernel-cliff.md) and reported
upstream on
[huggingface/transformers#45776](https://github.com/huggingface/transformers/issues/45776),
which already describes the mechanism but has no sm_75 data.

---

## The negative controls

Tests that pass prove little. These are the ones that fail when the thing they
guard is broken.

| control | what it would catch |
|---|---|
| [`LoopingPseudoFusedExecutor`](tests/test_fused.py) | An impostor that produces **byte-identical output** to the fused executor by looping one sequence at a time. The fusion tests must reject it, which means they are testing fusion and not output. |
| [`_BlockTableIgnoringBackend`](tests/test_parity.py) | A paged backend that reads slot `p` for position `p`. It must **diverge** from the real one; if the block table is never consulted, addressing is untested. |
| [`--fast` is a strict subset](tests/test_selection.py) | The development selection filtering, never weakening. It once silently picked `cpu` on a CUDA host and a whole measurement session verified nothing. |
| [frozen workload hash](tests/test_bench.py) | `bench/workload.py` pinned by content hash. `stream_digest` is only comparable across runs while the stream holds still. |
| [arms differ in exactly two knobs](tests/test_arms.py) | The two benchmark arms may differ in which kernel each side runs and in nothing else. |
| [four `xfail(strict=True)`](tests/test_attention_gqa.py) | On sm_75 the two GQA paths dispatch to different kernels, so `torch.equal` between them cannot hold. `strict=True` means an XPASS fails too. |

---

## Method

- **Median of 3 timed passes** after a discarded warmup. If the range exceeds
  20% of the median, the level reruns at 5 passes; if it still does, the level
  is `UNRESOLVED` and no median is reported.
- **Both engines are held to the same KV byte ceiling**, not the same process
  memory — otherwise the baseline loses for having less room for everything,
  rather than for what its rectangle wastes.
- **Rejection is shared code.** `is_admissible` is called by the engine and the
  baseline, so "rejected identically" is literally true.
- **Guards refuse rather than warn.** A C=64 control run recording zero
  preemptions raises: eviction is arithmetically forced there, so zero means the
  run measured something else.
- **No result file can be written without CUDA.** The development machine showed
  ~76% run-to-run spread; `--smoke` records a structural run stamped
  `throughput_valid: false`.

Three tolerances, each derived independently in the file that owns it:
`LOGIT_ATOL` 1e-3, `FUSED_LOGIT_ATOL` 5e-3, `FP16_LOGIT_ATOL` 6.25e-2. Token
equality is the real gate everywhere; the tolerances only bound how far logits
may drift while producing the same argmax.

---

## Environments

Two, and they are not the same one.

| file | what it is |
|---|---|
| [`requirements-dev.txt`](requirements-dev.txt) | M4 Mac, CPU/MPS, torch 2.12.1, Python 3.14.6. Runs the correctness suite. **Produced no number in this repo.** |
| [`requirements-measurement.txt`](requirements-measurement.txt) | Kaggle Tesla T4, torch 2.10.0+cu128, Python 3.12.13, Linux, driver 580.159.04. **Produced every number**, and all twelve result files agree on it exactly. |

## Reproducing

```bash
PAGED_ATTENTION_GQA_PATH=repeat_kv python -m bench.run --stream mixed -C 64 \
  --rectangle 3072 --baseline-expand-kv --out bench/results/repro.json
```

Needs CUDA. To redraw the analysis from the committed results with no GPU:

```bash
mkdir -p bench/results/arms && cp kaggle/session[67]*/arm[AB]-*.json bench/results/arms/
python -m bench.arms --compare-only --out-dir bench/results/arms
python -m bench.arms --merge-for-plot B mixed 3072 --out-dir bench/results/arms
python -m bench.plot bench/results/arms/plot-armB-mixed-rect3072.json --out-dir bench/plots
```

## Before running on a different GPU

**[`docs/AMPERE_RECHECK.md`](docs/AMPERE_RECHECK.md).** Every measurement was
taken on a T4, sm_75, and seven results are properties of that device rather
than of this code — TF32 is inert below sm_80 so the fp32 pin is untested
against what it defends, and FLASH and CUDNN reject every shape this engine
issues. `tests/test_ampere_recheck.py` fails if an item is dropped. No
sm_80-or-later device was ever run here.

That list is also why `pytest` on a CUDA host shows four `xfail`s.

## Scope

Gate 1 only. Prefix caching, Triton kernels, and a serving front end are not in
this repo and are not started.

Named for what it is: the paging is real, correct, and load-bearing — 87
unequal-length sequences do not fit a 3584-block pool without it. What earns the
throughput is a separate question, and the answer is in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

MIT licensed.
