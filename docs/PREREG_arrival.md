# Pre-registration: does the engine's advantage return under an arrival process?

**Written and committed before any simulation was run and before any GPU time was
spent.** Git history is the evidence: this file's commit precedes every commit
that touches `bench/arrival.py` or any result under `bench/results/arrival/`.

## Where this starts

The ragged-stream win this project was built around does not survive sorting the
baseline's input. Measured, one T4, one session, engine rerun beside each
baseline:

| cell | baseline in arrival order | baseline decode-sorted |
|---|---|---|
| `mixed-rect3072-C64` | 1.914x | **0.731x** |
| `mixed-rect1757-C64` | 1.885x | **0.669x** |

A static batcher that groups its accepted requests by decode length beats this
engine. Decode-step efficiency goes from 0.313 to 0.862 at rect3072. That is the
current headline and the current conclusion.

## The hypothesis

**A length-sorted static batcher cannot sort requests that have not arrived
yet.**

Every measurement in this repository presents all 256 requests at `t=0`. That
hands the sorted batcher the entire distribution up front, which is the best
case that will ever exist for it: it can select 18 or 32 near-identical decode
lengths out of a pool of 256. Under a real arrival process the pool is whatever
happens to be queued, and at sub-saturation load that is small.

Two effects should separate under arrivals, in opposite directions:

1. **The sort degrades.** Sorting can only help when the queue is deeper than
   the batch. When queue depth approaches batch size, the sorted batcher selects
   every request it has and the sort becomes a no-op, returning it to arrival
   order.
2. **The refill does not degrade.** The engine admits a new request on the next
   step. A static batcher admits nothing until its current batch drains, so a
   request arriving just after a batch forms waits for that batch's *longest*
   member before it is even prefilled.

Effect 2 is a latency claim, not a throughput claim, and it is the reason this
experiment reports latency.

## The prediction

Registered before running. Offered load `rho` is arrival rate over the
measured saturated service rate of the configuration being tested.

**PRIMARY.** At some offered load in `rho` in [0.3, 0.7], the engine's **p99
time-to-first-token is at least 2x lower** than the decode-sorted static
batcher's.

**SECONDARY.** At `rho <= 0.5`, the sorted batcher's realized decode-step
efficiency falls **below 0.60**, from the 0.862 it achieves at rect3072 with the
whole stream present at `t=0`.

**TERTIARY, and the one I am least confident in.** The engine's p50 end-to-end
request latency is lower than the sorted batcher's at `rho <= 0.7`. I expect
TTFT to move much further than p50 latency, and it is possible p50 latency
favours the sorted batcher everywhere while TTFT does not.

## The acceptance criterion

**CONFIRMED** if PRIMARY holds. The README gains a section stating that the
sorted-baseline result holds only when the whole stream is available at once,
and that under arrivals the engine wins on tail TTFT at the loads named. The
sorted-baseline loss stays in the README exactly as it is now, alongside it. Two
regimes, both published, neither deleted.

**REFUTED** if PRIMARY fails at every load tested. Then:

> The README headline becomes that this engine is slower than a static batcher
> that sorts by decode length, under an arrival process as well as with the whole
> stream present, and that the project's conclusion is that the engine does not
> beat static batching on this workload. Full stop. No further conditions get
> tested looking for one where it wins.

**AMBIGUOUS** is not an outcome. If the simulation in Stage 1 shows no window at
any load, that is REFUTED for the purposes of this document and Stage 2 does not
run.

Both outcomes are committed to publication before the result is known. That is
the point of the file.

## Why this is not motivated benchmarking

It partly is, and pretending otherwise would be worse than saying so.

The honest description: a result came back against the project, and the next
thing tested is a condition chosen because it is expected to be favourable. That
is the shape of motivated benchmarking, and no amount of pre-registration
changes the shape. What pre-registration changes is the ability to fit the
claim to the result afterwards. The prediction above has a number and a
threshold, both written down before the first run, and the refutation branch has
a specific README edit committed to it in advance.

Two further constraints, because the shape is bad enough to need them. First,
the sorted-baseline loss is not removed or softened under any outcome; it stays
in the README as the saturated-load result it is. Second, this is the last
condition tested. If PRIMARY fails, the conclusion is that the engine loses, and
the search stops there rather than moving on to a third workload.

The defence for running it at all is that all-requests-at-`t=0` is not a
neutral choice either. It was inherited from the original frozen workload, it is
the best case for a sorted batcher by construction, and an inference engine's
actual job is serving requests that arrive over time. A benchmark that only ever
measures the batch case is not more honest than one that measures both, provided
both are reported. Both will be.

## What is fixed and may not move

- `bench/workload.py` is untouched. Arrival times **layer on top** of the frozen
  stream. Same 256 requests, same prompts, same decode budgets, same
  `stream_digest`.
- Arrival times are generated from a seeded Poisson process. The seed is
  recorded.
- The reported metric is **latency**: TTFT and end-to-end per request, p50 and
  p99. Throughput may be recorded but is not the answer to this question and is
  not the headline of it.
- Both schedulers see identical arrival times.

## Procedure

**Stage 1, no GPU.** Layer Poisson arrivals over the frozen stream at a grid of
offered loads. Compute, analytically, the sorted batcher's steady-state queue
depth, how much of its sorting advantage survives at that depth, and where the
engine's admit-on-next-step behaviour starts to matter. **If Stage 1 shows no
window at any load, stop and report REFUTED.** No GPU time is spent.

**Stage 2, GPU, only if Stage 1 identifies a window.** Instrument TTFT and
per-request latency on both sides, run the loads Stage 1 named, report p50 and
p99. Projected GPU-hours reported before launching.

---

# OUTCOME

**Written after the run. Everything above this line predates it.** Measured on a
Kaggle T4, one session, passes interleaved, 1 warmup and 3 measured.
`bench/results/arrival/stage2-rho0.4.json`.

| metric | sorted static batcher | engine | ratio |
|---|---|---|---|
| TTFT p50 | 9.464 s | 0.176 s | **53.8x** |
| TTFT p99 | 76.281 s | 1.132 s | **67.4x** |
| latency p50 | 28.728 s | 5.747 s | **5.00x** |
| latency p99 | 111.240 s | 35.961 s | **3.09x** |
| wall per pass | 23.1 min | 23.1 min | 1.00x |

Per-pass spread on the headline figure was 1.0% for the engine and 2.3% for the
baseline, and both sides completed all 256 requests in every pass.

## Against the registered predictions

**PRIMARY: CONFIRMED.** Registered as "engine p99 TTFT at least 2x lower at some
load in [0.3, 0.7]". Measured **67.4x** at rho=0.4.

**SECONDARY: CONFIRMED.** Registered as "sorted decode efficiency below 0.60 at
rho <= 0.5". Measured **0.495**, against 0.862 with the whole stream present.

**TERTIARY: CONFIRMED**, and it was the one I said I was least confident in.
Engine p50 latency was 5.00x lower. p99 latency, which the simulation put at
only 1.7x and which I flagged as the narrowest margin, measured 3.09x.

## What the simulation got wrong, in the honest direction

Stage 1 predicted a mean queue depth of 11.2 at batch formation. **The measured
depth was 3.0**, and 36 of 87 batches formed with a single request queued. The
queue is far shallower than simulated, so the sort is not merely degraded, it is
frequently a literal no-op: a one-request batch has nothing to sort.

That also explains why measured efficiency (0.495) came out ABOVE the simulated
0.369. A batch of one wastes no decode steps at all, so shallow queues destroy
the sorting advantage and reduce straggler waste at the same time. The
simulation captured the first effect and underweighted the second.

The simulated TTFT ratio of 750x overstated the measured 67.4x by an order of
magnitude, exactly as flagged before the run: the model charged prefill at a
flat per-token rate and did not make an arriving request wait for a step
boundary or share a fused rectangle with running decodes. The registered
threshold was 2x and the measurement is judged against that.

## What this is, and what it is not

**This is a textbook result reproduced on this implementation, not a discovery.**
Continuous batching admitting into free slots while a static batcher drains
first is the entire reason continuous batching exists, and every serving system
built in the last three years does it. The measurement confirms this engine
implements it correctly. It does not make the engine competitive with anything
in production, and it does not rehabilitate the throughput claim.

**The sorted-baseline loss stays the headline.** At saturation, with every
request available, a static batcher that sorts by decode length beats this
engine at 0.73x and 0.67x. That is unchanged, it is still the first thing the
README says, and this result sits beside it as the other regime rather than
above it as a correction.

The honest summary of both together: the engine loses on throughput when the
whole workload is known in advance, and wins on latency when it is not.
