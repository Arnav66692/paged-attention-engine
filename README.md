# A paged-attention inference engine, built from scratch and audited

Block-allocated KV cache, continuous batching, preemption with recompute, and a
fused batched forward — measured against naive HuggingFace `generate` at an
identical KV byte ceiling on a Kaggle Tesla T4.

**The repo is organised so the throughput claim can be audited rather than
believed.** Every number below came from a run that happened, on hardware named
in the provenance of the file that holds it. Where something is unmeasured, it
says unmeasured. Where the engine loses, it says so, in the same size type.

---

## Gate 1 outcome, stated plainly

**Passed on the ragged stream. Failed on the uniform control stream. Both are
published.**

| stream | prompt lengths | result |
|---|---|---|
| `mixed` | 224 distinct, 40–2048 tokens | **1.89× – 2.06× in Arm B**, winning at all three levels on both sizings — but **0.94× and 0.99× in Arm A**, a loss |
| `control` | all 512 tokens | **0.66× – 0.79×** — a loss, in both arms, at every sizing |

**Arm A loses `mixed` as well, and that belongs here rather than in a footnote:**
0.9409× at `mixed-rect3072-C64` and 0.9859× at `mixed-rect1757-C64`. Arm A is the
engine's *shipped default* (`PAGED_ATTENTION_GQA_PATH=sdpa`), and under it the
engine sits on `MATH` while HF's ragged batches already hold
`EFFICIENT_ATTENTION` — the asymmetry described below, priced. Arm A was run on
`mixed` at `C=64` only; `C=32` and `C=87` are **unmeasured** under the default.

So the win requires the engine to be on `repeat_kv`, which is not what it ships
with today. Every number here is real, and the ones that are not wins are the
more informative.

---

## The headline: `mixed`, both sides on `EFFICIENT_ATTENTION`

Engine tokens/s over baseline tokens/s. Median of 3 measured passes after 1
discarded warmup; pass spread 0.05–0.25%, so these ratios are resolved to well
under a percent.

| level | engine cap | baseline resident | engine tok/s | baseline tok/s | **ratio** | preemptions |
|---|---|---|---|---|---|---|
| `mixed-rect3072-C32` | 32 | 18 | 77.658 | 37.881 | **2.050** | 0 |
| `mixed-rect3072-C64` | 64 | 18 | 89.487 | 47.339 | **1.890** | 0 |
| `mixed-rect3072-C87` | 87 | 18 | 81.553 | 40.860 | **1.996** | 70 |
| `mixed-rect1757-C32` | 32 | 32 | 88.235 | 44.897 | **1.965** | 0 |
| `mixed-rect1757-C64` | 64 | 32 | 83.537 | 44.041 | **1.897** | 0 |
| `mixed-rect1757-C87` | 87 | 32 | 94.844 | 46.016 | **2.061** | 70 |

**The ratio is not monotonic in `C`.** It dips at `C=64` on both sizings —
2.050 → 1.890 → 1.996 at rect3072, and 1.965 → 1.897 → 2.061 at rect1757. That
shape is unexplained and is reported rather than smoothed; three levels are not
enough to attribute it, and no mechanism here has been measured.

What `C=87` does establish — it is `FRAG_C`, the level sized to stress
fragmentation — is that the scheduler **does not degrade under maximum pressure**:
99.9% of the block pool held (3580 of 3584 blocks), 70 preemptions, 69 recomputes,
and the ratio still ~2× on both sizings. It is the best cell on rect1757 (2.061)
but not on rect3072, where `C=32` is higher (2.050 vs 1.996).

### Why this comparison is symmetric, and what "symmetric" required

Both sides call the same `torch.scaled_dot_product_attention`, and **torch picks
the kernel from the input shape**. Two engines can therefore land on different
kernels with nobody choosing it, and the ratio silently becomes part paging and
part kernel.

Worse, **the baseline's kernel depends on the stream.** HuggingFace takes the
`enable_gqa` path only when `attention_mask is None`, and the mask is `None` only
when nothing was left-padded:

| stream | prompts | HF's path | baseline backend |
|---|---|---|---|
| `control` | all 512, no padding | mask → `None` → `enable_gqa` | `MATH` |
| `mixed` | 224 lengths, real mask | pre-expands K/V | `EFFICIENT_ATTENTION` |

So the shipping configuration is **asymmetric on `mixed`, against the engine** —
HF already held the fast kernel while the engine sat on `MATH`. That was nobody's
decision and was invisible until measured.

The sweep therefore runs twice (`bench/arms.py`):

- **Arm A** — engine `GQA_PATH=sdpa`, baseline untouched.
- **Arm B** — engine `GQA_PATH=repeat_kv`, baseline forced to pre-expand.

**The headline above is Arm B, and on `mixed` its baseline is effectively
unpatched.** HF's ragged batches already pre-expand, so forcing pre-expansion
changes nothing on that side — measured, not assumed: baseline median 1249.92 s
(Arm A) vs 1250.45 s (Arm B) at rect3072, and 1208.61 s vs 1207.34 s at rect1757.
**0.04% and 0.10% apart.** Both sides run `EFFICIENT_ATTENTION`; the baseline is
the one HuggingFace ships, running the code HuggingFace chose for it.

Arm B's patch is gated on the patched baseline being **token-identical** to the
unpatched one before it may be timed. A baseline whose answers moved is a
different baseline.

---

## The control-stream loss, and why it is here

**On uniform-length prompts the engine loses, 0.66× to 0.79×, in both arms.**

| cell | Arm A (both `MATH`) | Arm B (both `EFFICIENT`) |
|---|---|---|
| `control-rect3072-C64` | 0.7702 | 0.7880 |
| `control-rect1757-C64` | 0.6928 | 0.6598 |

**The mechanism.** The control stream is 256 requests of exactly 512 prompt and
512 decode tokens. Every sequence starts together, grows at the same rate, and
finishes together. That means:

- **no stragglers** — a static batch is never held hostage by one long member,
  because every member is the same length;
- **no fragmentation** — uniform lengths pack a contiguous rectangle perfectly,
  so the rectangle wastes nothing;
- **no rejections** — every request fits both sizings.

Continuous batching and block allocation exist to eliminate exactly those three
costs. When they are all zero there is nothing to eliminate, and paging is left
paying its own overheads — per-step block bookkeeping, an explicit attention mask
instead of the causal shortcut, gather/scatter through a block table, and 32
preemptions with their recomputes at `C=64`.

**A paged engine should lose here.** A version of this repo that reported a win
on the control stream would be reporting a bug.

---

## Which cell may decide what

Arm B's patch only bites where HF would otherwise have taken `enable_gqa`. On
`mixed` it is inert, so **only the `control` cells can answer whether the shared
kernel biases the ratio** — those are the only ones where both sides moved.

### Kernel bias: −4.77% and +2.30% at the two cells that can decide it, sign not constant

| deciding cell | baseline resident | Arm A (`MATH`) | Arm B (`EFFICIENT`) | shift |
|---|---|---|---|---|
| `control-rect3072-C64` | 18 | 0.7699 – **0.7702** – 0.7713 | 0.7862 – **0.7880** – 0.7886 | **+2.30%** |
| `control-rect1757-C64` | 32 | 0.6927 – **0.6928** – 0.6929 | 0.6594 – **0.6598** – 0.6607 | **−4.77%** |

Intervals do not overlap either time, and both arms did provably identical work
(same tokens, preemptions, peak blocks, rejections, workload digest). So the
disagreement is real: **at the two configurations able to decide this, the kernel
moved the ratio by −4.77% and +2.30%, and which way it moves depends on the
configuration.** These are the only two deciding cells that exist — `control` was
never run at `C=32` or `C=87` — so the magnitude at any other configuration is
**unmeasured**, and no general bound is claimed. It follows whichever side gains more from the
switch — engine 1.662× vs baseline 1.625× at resident 18 (ratio up), engine
1.749× vs baseline 1.836× at resident 32 (ratio down).

**Both candidate models were refuted.** Batch amortisation,
`ratio = (B_e/B_b)(W + B_b·k)/(W + B_e·k)`, is monotonically decreasing in the
per-token cost `k` whenever `B_e > B_b`, predicting Arm B above Arm A always —
rect1757 contradicts it, and the flaw is assuming one `k` serves both sides.
Static-batch stragglers predict the reverse always — rect3072 contradicts that.
Neither dominates. Two directions were asserted in this repo before the
measurement existed, and both were wrong; `docs/AMPERE_RECHECK.md` keeps the
record of that rather than only the answer.

---

## The crossover — and the hypothesis it refuted

The baseline is resident at `min(rectangle limit, C)`, so its residency saturates
at 18 (rect3072) or 32 (rect1757) while the engine keeps admitting. That predicts
a **crossover**: at `C` equal to the baseline's cap both sides hold the same
number of sequences, the engine's residency advantage is exactly 1.00×, and the
engine should tie or lose.

`mixed-rect1757-C32` is that point exactly — both sides capped at 32 resident.

**The prediction is refuted by measurement.**

| | tokens | median | tok/s |
|---|---|---|---|
| engine | 30,429 | 344.86 s | 88.235 |
| baseline | 26,042 | 580.04 s | 44.897 |

The engine produced **4,387 more tokens in 235 s less time — ratio 1.965**, one
of the highest in the matrix. Residency parity is not throughput parity.

**So residency is not the mechanism, and it cannot be in either direction:** the
engine also *loses* the control stream at `C=64` while holding roughly 55
sequences against the baseline's 18, roughly a 3× residency advantage. (Derived,
not directly recorded: peak blocks held was **3548**,
measured, and a control request is 512 + 512 tokens ÷ `BLOCK_SIZE` 16 = 64
blocks, so 3548 / 64 ≈ 55.4.) That is a smaller advantage than `mixed-rect3072-C87`
enjoys — cap 87 against 18, 4.8× — but the control cell loses while the mixed one
wins by 2×. What separates a win from a loss is **the shape of the stream, not the
size of the running set.** Ragged lengths mean a static batch is held by its
longest member; that waste is what continuous batching removes, and it is present
at equal residency. Uniform lengths mean there is no such waste at any residency.

---

## The rejection confound, and which way it points

The two rectangle sizings trade residency against admissions, and the P95 sizing
charges for its extra residents in rejected work.

| sizing | baseline resident | `control` rejections | `mixed` rejections |
|---|---|---|---|
| 3072 (distribution max) | 18 | 0 / 256 (0.00%) | 0 at every level |
| 1757 (P95) | 32 | 0 / 256 (0.00%) | 7/128 (5.47%), 10/256 (3.91%), 14/348 (4.02%) |

**This makes the rect1757 `mixed` cells an unequal-work comparison and it must be
stated, not left to be discovered.** At `mixed-rect1757-C64` the baseline rejects
10 requests and generates **53,172 tokens against the engine's 59,195 — 10.2%
less work.** The engine serves every request; the baseline cannot.

**The direction matters and it is favourable to the baseline.** Rejected requests
are the ones over the rectangle — the longest, hence the worst straggler
generators. The baseline therefore ran the *easier* subset, with its most
expensive work removed, and was still slower per token. **The reported ratio
understates the engine's advantage on those cells.** An unstated asymmetry in a
headline is what gets caught; this one is stated, and it points against us.

The rect3072 cells have no rejections at any level and are the clean comparison.

---

## The negative controls

Tests that pass are weak evidence; tests that would fail if the thing under test
were broken are the point. These are the ones worth reading first.

- **`LoopingPseudoFusedExecutor`** — `tests/test_fused.py`. A byte-identical
  impostor: it produces exactly the same logits as the fused executor by looping
  over the running set one sequence at a time. Every correctness assertion in the
  fused suite passes against it. Only the forward-call counter fails it. Without
  this, "fusion" is indistinguishable from a loop with better naming, and the
  one-pass-per-step claim is unfalsifiable.
- **`_BlockTableIgnoringBackend`** — `tests/test_parity.py`, in
  `test_fragmented_fixture_would_catch_a_ignored_block_table`. A paged backend
  that ignores its block table entirely. The test asserts it **diverges** from
  HuggingFace. If it did not, the fragmented-pool fixture could not detect address
  translation bugs and every parity pass built on that fixture would prove
  nothing.
- **Arm B's token-identity gate** — `bench/baseline.py::assert_token_identical`.
  The patched baseline must reproduce the unpatched one token for token before it
  may be timed, verified against a deliberately perturbed baseline in
  `tests/test_arms.py`.
- **The zero-preemption guard** — `bench/run.py::check_preemptions`. A run where
  eviction is arithmetically forced but no preemption was recorded raises
  `InvalidRun` rather than reporting a number.

---

## Reproducing

```bash
PAGED_ATTENTION_GQA_PATH=repeat_kv python -m bench.run --stream mixed -C 64 --rectangle 3072 --baseline-expand-kv --out bench/results/repro.json
```

Requires a CUDA host — `bench/run.py` refuses to write a result file without one,
because both development machines here showed 76% run-to-run spread. Add
`--smoke` for a structural run, which is stamped `throughput_valid: false`.

Plots: `python -m bench.arms --merge-for-plot B mixed 3072 --out-dir bench/results/arms`
then `python -m bench.plot bench/results/arms/plot-armB-mixed-rect3072.json`.

Tests: `pytest` and `pytest -m slow` — both halves run before every commit.

---

## Before running on a different GPU

**Read [`docs/AMPERE_RECHECK.md`](docs/AMPERE_RECHECK.md).** Every measurement
here was taken on a Kaggle **T4, sm_75**, and six results are properties of that
device rather than of this code — TF32 is inert below sm_80 so the fp32 pin is
untested against what it defends; FLASH and CUDNN reject every shape this engine
issues on sm_75, so backend selection will move on Ampere; and the kernel-bias
result above is a fact about which backend `enable_gqa` selects, which changes
with the device. `tests/test_ampere_recheck.py` fails if any item is dropped.

---

## Scope

Gate 1 only: paged KV cache, continuous batching, preemption and recompute, fused
batched execution, and an audited throughput comparison. Prefix caching, Triton
kernels, and a serving front end are **not** in this repo and are not started.
