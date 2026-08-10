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

## The result: the engine loses to a static batcher that sorts its input

Engine tokens/s over baseline tokens/s, Kaggle Tesla T4, fp16, both sides
holding the same 1,644,167,168-byte KV ceiling.

The ragged-stream win this project was built around holds only against a static
batcher that batches in **arrival order**. Sorting the baseline's accepted
requests by decode length before batching — one line, no paging, no continuous
batching — reverses it:

| request stream | baseline batches in arrival order | baseline batches decode-sorted |
|---|---|---|
| **ragged**, rect3072, C=64 | **1.91×** | **0.73×** — a loss |
| **ragged**, rect1757, C=64 | **1.89×** | **0.67×** — a loss |
| **uniform** — every prompt 512 + 512 | **0.66× – 0.79×** — a loss | not measured |

Both columns of a row were measured in one Kaggle session on one T4, with the
engine rerun alongside each baseline; the engine landed within 0.03% of itself
across the two runs of each cell. `bench/results/batch-order/`.

**What the gap between the two columns measures.** A static batch runs every row
until the batch's longest member finishes its budget, so a 1024-token decode
sharing a batch with an 8-token decode makes the short row compute 1016
positions nobody reads. Grouping by decode length removes most of that: measured
decode-step efficiency goes from 0.313 to 0.862 at rect3072 and from 0.254 to
0.732 at rect1757, and baseline forward passes fall 2.8× and 2.9×. The engine's
own throughput does not move. So the ragged-stream number is a measurement of
how much waste the baseline's batching order leaves on the table, and it is
sensitive to a choice the baseline was making badly.

**The opponent is weak and that matters.** Naive static-batch `generate` is not
vLLM, TGI, or anything in production, and none of those were ever benchmarked
here. The uniform-stream loss was already the more informative half: with no
straggler waste and no fragmentation there is nothing for the engine to reclaim,
so it pays its own bookkeeping for no return. The sorted-baseline result extends
that to the ragged stream — once the baseline stops wasting decode steps, the
engine's bookkeeping is not paid for there either.

### The other regime: under an arrival process, the engine wins on latency

The table above gives the sorted batcher every request at `t=0`, which is the
best case it will ever have: it selects 18 near-identical decode lengths out of
a pool of 256. A serving engine's actual job is requests that arrive over time,
and a sorted batcher cannot sort what has not arrived.

Measured with a seeded Poisson arrival process layered over the same frozen
stream, at 40% of the sorted batcher's saturated service rate. One T4, one
session, passes interleaved, 1 warmup and 3 measured:

| metric | sorted static batcher | engine | ratio |
|---|---|---|---|
| TTFT p50 | 9.464 s | 0.176 s | **53.8×** |
| TTFT p99 | 76.281 s | 1.132 s | **67.4×** |
| latency p50 | 28.728 s | 5.747 s | **5.00×** |
| latency p99 | 111.240 s | 35.961 s | **3.09×** |
| wall clock | 23.1 min | 23.1 min | 1.00× |

Wall clock is identical because below saturation both schedulers complete every
request and both match the arrival rate. Throughput is not the metric here and
would say nothing.

**The mechanism, measured.** Mean queue depth at batch formation was **3.0**
against a batch size of 18, and 36 of 87 batches formed with a single request
queued. The sort is not merely weakened, it is frequently a literal no-op.
Meanwhile the static batcher admits nothing until its current batch drains, so a
request arriving just after a batch forms waits for that batch's longest member
before it is even prefilled. Realized decode efficiency was 0.495, against 0.862
with the whole stream present.

**This is a textbook result, not a discovery.** Admitting into free slots while a
static batcher drains first is the entire reason continuous batching exists, and
every serving system built in the last three years does it. This measures that
the engine implements it correctly. It does not make the engine competitive with
anything in production, and it does not rehabilitate the throughput claim above.

It was **pre-registered** before the run, with the prediction, the threshold and
the refutation branch written down first: [`docs/PREREG_arrival.md`](docs/PREREG_arrival.md).
The simulation that justified spending the GPU time overestimated the TTFT ratio
by an order of magnitude, and that is recorded there too.

**Both regimes together:** the engine loses on throughput when the whole workload
is known in advance, and wins on latency when it is not.

**What is not claimed.** This does not show that paging or continuous batching
are worthless; it shows that this benchmark never isolated their contribution,
because the margin it reported is recoverable by reordering a list. A comparison
that would isolate it has not been run.

**The ragged ratio depends on both sides running the same attention backend, and
getting that wrong in either direction moves it by 2×.** Under this engine's
previous default it sat on `MATH` while HuggingFace's ragged batches already had
`EFFICIENT_ATTENTION` — same code, same stream, 0.94× instead of 1.89× against
the arrival-order baseline. Any
benchmark of this kind is one silent kernel selection away from being a kernel
comparison. [`bench/arms.py`](bench/arms.py) runs both arms on the four `C=64`
cells to bound it — the `C=32` and `C=87` cells are Arm B only. On the two
`control` cells, the only ones where the patch moves the baseline as well as the
engine, the shared kernel moved the ratio by +2.30% and −4.77%: a few percent,
and the sign is not constant.

Full tables, the refuted hypotheses, and the rejection confound:
**[`docs/FINDINGS.md`](docs/FINDINGS.md)**.

### What this benchmark does not measure

Every measurement here was taken on one Kaggle Tesla T4 (sm_75). No
Ampere-or-later device has run this code, and FLASH is unavailable for every
shape this benchmark issues on sm_75, so the ratio should not be expected to
transfer to hardware where it is available.

One model was measured: Qwen2.5-1.5B-Instruct in fp16, 28 layers over 2 KV
heads. KV is cheap relative to compute at that shape, and no second model, size,
or dtype was run.

The baseline is naive static-batch `model.generate()` with EOS disabled. No
production serving engine was benchmarked. Both batching orders are now
measured, and the ragged-stream margin does not survive the sorted one.

The KV ceiling of 1,644,167,168 bytes is a chosen constant, roughly a tenth of
the card. Every stream measured would fit in the T4's free memory at once, so
the capacity pressure is a parameter of the benchmark rather than a property of
the device, and no sensitivity analysis of it exists.

The request stream is synthetic: lognormal prompt and decode lengths with no
trace behind the parameters, uniformly random token IDs so no two requests share
a prefix, all requests present at t=0, and one seed. Sensitivity to the
dispersion, to prefix sharing, to an arrival process, and to the seed is
unmeasured.

Most of the ragged-stream margin against the arrival-order baseline is one
attention-kernel choice: switching the engine's GQA path from `sdpa` to
`repeat_kv` roughly halved its time while the baseline moved 0.04%, and under
the previous default the engine lost at 0.94x. That choice matters here because
FLASH is unavailable on sm_75; on hardware where it is available the effect is
unmeasured.

Latency is measured at one offered load on one stream: rho=0.4 on the ragged
stream at rect3072. There is no latency measurement at any other load, on the
uniform stream, or at the other cache sizing, and no single-request number
anywhere in this repository.

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
[`docs/hf-padding-kernel-cliff.md`](docs/hf-padding-kernel-cliff.md). Reported
upstream on 2026-08-09 in
[a comment on huggingface/transformers#45776](https://github.com/huggingface/transformers/pull/45776#issuecomment-5230620009),
which thread already describes the mechanism but had no sm_75 data.

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
  baseline, so a request that predicate refuses is refused identically on both
  sides. In these runs it refused nothing: every rejection in the result files is
  the baseline's own rectangle limit, which the engine has no analogue for.
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
