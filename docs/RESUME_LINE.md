# Gate 1 resume line

Three sentences. Engine first, mechanism second, and the loss stated rather than
omitted — a claim that survives someone opening the repo is worth more than a
bigger one that does not. Drafted 2026-08.

> Built an LLM inference engine from scratch with a block-allocated paged KV
> cache, continuous batching, and recompute-based preemption. Sustains 1.89 to
> 2.06x the throughput of HuggingFace generate at an identical 1.64 GB KV
> ceiling on a T4, across three concurrency levels and two cache sizings. Loses
> at 0.66 to 0.79x on uniform-length streams, where there is no straggler waste
> to reclaim.

## Every figure, and where it comes from

| figure | source |
|---|---|
| **1.89 to 2.06×** | `bench/results/arms/armB-mixed-rect{3072,1757}-C{32,64,87}.json`. Six cells: 2.050, 1.890, 1.996, 1.965, 1.897, 2.061. Min 1.890, max 2.061 — quoted without rounding outward. |
| **identical 1.64 GB KV ceiling** | `1,644,167,168` bytes = 3584 blocks × 16 tokens × 28,672 bytes/token, derived in `bench/baseline.py`, asserted in `tests/test_bench.py`. Both engines are held to it; the rectangle sizing is only what the baseline spends it on. |
| **T4** | `provenance.gpu_name` = `Tesla T4`, sm_75, in all 12 result files. |
| **three concurrency levels** | C = 32, 64, 87. C=87 is `FRAG_C`. |
| **two cache sizings** | Rectangles 3072 (distribution max) and 1757 (P95), giving the baseline 18 and 32 resident. |
| **0.66 to 0.79×** | `arm{A,B}-control-rect{3072,1757}-C64.json`: 0.6598, 0.6928, 0.7702, 0.7880. |
| **no straggler waste to reclaim** | The control stream is 512 + 512 for every request, so every member of a static batch finishes on the same step. **Supported by elimination, not by a static-batching ablation** — see the README's mechanism section. |

## The conditions it holds under

The line cannot carry these, so anyone who asks should get them immediately
rather than discover them:

- **Kaggle Tesla T4, sm_75, fp16.** No Ampere-or-later device was ever run.
  `docs/AMPERE_RECHECK.md` lists what must be re-measured first.
- **Both sides on `EFFICIENT_ATTENTION`** — the engine's shipping default
  (`GQA_PATH=repeat_kv`) and the kernel unpatched HuggingFace already reaches on
  ragged batches. Nothing was done to the baseline.
- **The shared kernel moves the ratio by a measured ±5%, sign not constant**
  (+2.30% at rect3072, −4.77% at rect1757).
- **At rect1757 the baseline rejects requests over its rectangle** — 7, 10 and
  14 at C=32/64/87 — so it serves 8.8–14.4% less work than the engine there.
  Those are the longest requests by construction, so the comparison
  **understates** the engine.
- **Median of 3 measured passes** after 1 discarded warmup, spread 0.05–0.25%,
  with escalation to 5 passes and an UNRESOLVED verdict above 20%.

## Why the third sentence stays in

It is what makes the first two checkable. An engine claiming a win on
uniform-length work would be claiming to have removed a cost that is not there;
the loss is the evidence that the win is mechanical rather than tuned. Anyone
who asks about it gets a better answer than the headline.

## What it must not be shortened to

- **"2× faster than HuggingFace"** — drops the stream condition, which is the
  whole content. The same engine is 0.66× on a different stream.
- Dropping **"at an identical KV ceiling"** — without it the comparison is
  against an arbitrarily-sized baseline and means nothing.
- Dropping the device — every number here is sm_75 and several are properties of
  that device.
- **"paged attention makes it 2× faster."** Paging makes the residency
  affordable; continuous batching earns the throughput. The repo's own data
  rules residency out as the mechanism in both directions.
- A vLLM comparison. This was never benchmarked against a production engine.
