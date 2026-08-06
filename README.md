# A paged-attention inference engine, built from scratch and audited

Block-allocated KV cache, continuous batching, preemption with recompute, and a
fused batched forward — measured against naive HuggingFace `generate` at an
identical KV byte ceiling on a Kaggle Tesla T4.

**A note on the name.** This is a paged-attention engine and the paging is real,
correct, and load-bearing: without a block allocator the engine could not hold
87 unequal-length sequences in a 3584-block pool at all. But **the measured
throughput win is attributable to continuous batching, not to paging** — the
repo's own data rules residency out as the mechanism, in both directions. Paging
is what makes the residency affordable; refilling a finished sequence's slot
immediately is what earns the 2×. The reasoning is
[set out in full below](#so-what-is-the-mechanism-continuous-batching-not-paging),
including which part of it is measured and which part is elimination.

```bash
pip install -r requirements.txt        # or requirements.lock.txt for the exact pins
pytest                                  # ~12 minutes; -m slow is a second ~3
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

**What it is measured against.** Naive static-batch HuggingFace `generate` — a
deliberately weak opponent. It is not vLLM, TGI, or any production engine, and
nothing here was ever benchmarked against one. Beating static batching on a
stream with 224 distinct prompt lengths is close to the expected result; the
interesting part is the methodology and the kernel finding, not the ratio.

---

## Gate 1 outcome, stated plainly

**Passed on the ragged stream. Failed on the uniform control stream. Both are
published.**

| stream | prompt lengths | result |
|---|---|---|
| `mixed` | 120 / 224 / 288 distinct lengths at `C=32`/`64`/`87`, all within 40–2048 | **1.89× – 2.06×**, winning at all three levels on both sizings |
| `control` | all 512 tokens | **0.66× – 0.79×** — a loss, at every sizing |

**The headline is the shipping configuration.** The engine ships
`PAGED_ATTENTION_GQA_PATH=repeat_kv`, which puts it on `EFFICIENT_ATTENTION`,
and HuggingFace's ragged batches already pre-expand K/V and land there too. So
the `mixed` numbers below are this engine as shipped against **unpatched**
HuggingFace, both sides on the same kernel, with nothing done to either.

For most of this project the default was `sdpa` instead, chosen to match a
baseline that an early measurement had found on `MATH`. That match turned out to
hold only for prompt-uniform batches, and under it the engine sat on the slow
kernel while HF held the fast one — 0.9409× and 0.9859× on `mixed`, a loss
bought by a choice nobody made. The default moved; those two numbers are kept in
[the kernel section](#kernel-bias-477-and-230-at-the-two-cells-that-can-decide-it-sign-not-constant)
because they are what the handicap cost.

**On `control` the engine holds the faster kernel and still loses.** That is not
a kernel artifact and it is not being explained away.

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

**And this is not the kernel.** On `control` the engine ships
`repeat_kv`/`EFFICIENT_ATTENTION` while HuggingFace — whose uniform prompts
collapse the mask to `None` — takes `enable_gqa` and lands on `MATH`. So in the
shipping configuration **the engine holds the faster kernel here and still
loses.** Arm B, where the baseline is forced onto `EFFICIENT` too and the
advantage is removed, is the fairer of the two columns and the loss survives it
at both sizings.

**The mechanism.** The control stream is 256 requests of exactly 512 prompt and
512 decode tokens. Every sequence starts together, grows at the same rate, and
finishes together. That means:

- **no stragglers** — a static batch is never held hostage by one long member,
  because every member is the same length;
- **no fragmentation** — uniform lengths pack a contiguous rectangle perfectly,
  so the rectangle wastes nothing;
- **no rejections** — every request fits both sizings.

Continuous batching and block allocation exist to eliminate exactly those three
costs. When all three are zero there is nothing to eliminate, and the engine is
left paying its own overheads for no return — per-step block bookkeeping, an
explicit attention mask instead of SDPA's causal shortcut, gather/scatter through
a block table, and 32 preemptions with their recomputes at `C=64`.

A paged engine should lose here.

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
engine also *loses* the control stream at `C=64` while holding **64**
sequences against the baseline's 18 — a **3.56×** residency advantage. (Counted
from the engine's own `pool_trace`, a complete 608-event record that closes at
zero: the running set reaches the `max_num_seqs` cap of 64 and stands at 63 at
the step where `peak_blocks_used` hits its measured 3548. Dividing peak blocks by
64 blocks-per-request gives 55.4 and is **wrong** — sequences hold short block
runs until they reach full length, so it undercounts the set.) That is still a
smaller advantage than `mixed-rect3072-C87` enjoys — cap 87 against 18, 4.83× —
and the control cell loses while the mixed one wins by 2×.

### So what *is* the mechanism: continuous batching, not paging

Residency is eliminated in both directions. What survives is this:

**The baseline runs STATIC batches.** A sequence that finishes early leaves its
slot idle until every other member of its batch drains. The engine refills that
slot on the next step. So the quantity that decides the outcome is *how much
idle slot-time a static batch accumulates*, and that is a property of the spread
of lengths within a batch:

- **Ragged stream** — 120 to 288 distinct lengths per level, 40 to 2048 tokens.
  Stragglers are severe, the baseline's batches spend a large fraction of their
  slot-time idle, and refilling immediately is worth ~2×.
- **Uniform stream** — every request 512 + 512. Every member of a batch finishes
  on the same step. There is no idle slot-time to reclaim, so continuous
  batching has nothing to do, and the engine pays its own scheduling, block
  translation and preemption overheads for nothing. Hence 0.66–0.79×.

**Paging is what makes high residency affordable; it is not what wins.** The
block allocator is real, correct, and load-bearing — without it the engine could
not hold 87 sequences of unequal length in a 3584-block pool at all. But
residency is not the mechanism, and this repo's own data rules it out.

**This is supported by ELIMINATION, not measured.** Residency is excluded by two
measurements pointing opposite ways, and continuous batching is the explanation
left standing. **No static-batching ablation was run on this engine** — that
would need a contiguous-KV plus fused-batching path, which does not exist here
(`ContiguousKVCache` is single-sequence, kept as the parity reference). The
fused/unfused ablation that *does* exist would not settle it either: both modes
share the paged cache and the scheduler, so it separates the batched forward and
nothing else. Treat the mechanism as the surviving hypothesis, and treat a
future static-batching ablation as the thing that would confirm it.

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

**The direction favours the baseline.** Rejection is
`total_len > rectangle`, so the rejected requests are the longest in the stream
**by construction**, and they carry 6,023 of the 59,195 decode tokens. That they
are therefore the *worst straggler generators* is an inference, not a
measurement — but the arithmetic direction does not depend on it: the baseline
ran a strictly smaller and shorter subset and was still slower per token. **The reported ratio
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

Requires a CUDA host — `bench/run.py` refuses to write a result file without one.
The rule exists because this project's development machine (an M4 Mac) showed
~76% run-to-run spread across its two devices on the Step 5b diagnostic; that
figure is recorded in `bench/run.py`'s docstring as prose and **no result file
in this repo holds it**, so treat it as the reason for the rule rather than as a
measurement you can check here. Add `--smoke` for a structural run, which is
stamped `throughput_valid: false`.

**Plots, from the published results — no GPU needed.** `bench/results/` is
gitignored (it is an output directory), so the measured result files ship under
`kaggle/session6*/` and `kaggle/session7*/` instead. Gather them first:

```bash
mkdir -p bench/results/arms && cp kaggle/session[67]*/arm[AB]-*.json bench/results/arms/
python -m bench.arms --compare-only --out-dir bench/results/arms
python -m bench.arms --merge-for-plot B mixed 3072 --out-dir bench/results/arms
python -m bench.plot bench/results/arms/plot-armB-mixed-rect3072.json --out-dir bench/plots
```

That reproduces the comparison table and both plots from the committed
measurements, on any machine.

Tests: `pytest` and `pytest -m slow` — both halves run before every commit.

---

## Before running on a different GPU

**Read [`docs/AMPERE_RECHECK.md`](docs/AMPERE_RECHECK.md).** Every measurement
here was taken on a Kaggle **T4, sm_75**, and six results are properties of that
device rather than of this code — TF32 is inert below sm_80 so the fp32 pin is
untested against what it defends; FLASH and CUDNN reject every shape this engine
issues on sm_75, so backend selection is expected to move on Ampere — **expected,
not measured; no sm_80-or-later device was ever run here**; and the kernel-bias
result above is a fact about which backend `enable_gqa` selects, which changes
with the device. `tests/test_ampere_recheck.py` fails if any item is dropped.

Item 7 is why `pytest` on a CUDA host shows **four `xfail`s**: on sm_75 the two
GQA paths dispatch to different backends — `MATH` and `EFFICIENT_ATTENTION` —
which accumulate in different orders, so `torch.equal` between them was never
available there. `torch.equal` is kept and the cases are marked
`xfail(strict=True)` with the mechanism in the reason, so an XPASS fails too and
the fact cannot rot the day the dispatch changes. Swapping in `allclose` would
have made the suite green by hiding a property of the hardware.

---

## Lessons

- **A verification step must state which selection it ran.** The full default
  suite first ran on the measurement host at **Session 8** — Session 0's "port
  verification" ran `--fast` and `-m slow`, and neither selection covers the
  middle, so the non-slow non-fast half of the gate had only ever run on cpu and
  mps. That is where the four CUDA `xfail`s had been hiding. Same shape as
  `_devices()` returning `["cpu"]` on a CUDA host: a step that reports "the
  suite passed" says nothing without naming the selection, and both times the
  gap was invisible precisely because something green was being reported.

---

## Scope

Gate 1 only: paged KV cache, continuous batching, preemption and recompute, fused
batched execution, and an audited throughput comparison. Prefix caching, Triton
kernels, and a serving front end are **not** in this repo and are not started.
