# Gate 1 resume line

One sentence, the number, and the condition it holds under. Drafted 2026-08.

> Built a paged-attention inference engine from scratch — block-allocated KV
> cache, continuous batching, preemption with recompute, fused batched execution
> — that sustains **1.89–2.06× the throughput of naive HuggingFace `generate`**
> at an identical 1.64 GB KV ceiling on a Tesla T4, **measured on
> variable-length request streams at three concurrency levels and two cache
> sizings**, and **0.66–0.79× on uniform-length streams**, where there is no
> straggler or fragmentation waste for paging to remove.

## Why the second clause stays in

It is the part that makes the first clause checkable. A paged engine that
claimed a win on uniform-length work would be claiming to have removed a cost
that is not there; the loss is the evidence that the win is mechanical rather
than tuned. Anyone who asks about it gets a better answer than the headline.

## What the sentence must not be shortened to

- **"2× faster than HuggingFace"** — drops the stream condition, which is the
  whole content. The same engine is 0.66× on a different stream.
- Omitting **"at an identical KV ceiling"** — without it the comparison is
  against an arbitrarily-sized baseline and means nothing.
- Omitting the device — every number here is sm_75 and several are properties
  of that device. See `docs/AMPERE_RECHECK.md`.

## Caveat to have ready, not to hide

The win requires `PAGED_ATTENTION_GQA_PATH=repeat_kv`, which is **not** the
engine's shipped default. Under the default the engine sits on the `MATH` kernel
while HuggingFace's ragged batches already hold `EFFICIENT_ATTENTION`, and the
same cells measure 0.94× and 0.99×. The README states this above the fold.
