# Things that MUST be re-measured on the first Ampere-or-later host

Every measurement in this repo so far was taken on a **Kaggle Tesla T4, sm_75
(Turing)**. Several results are properties of that device rather than of this
code, and they will move on Ampere (sm_80+) or later. An A10 is **sm_86**. An
A100 is sm_80. An L4 is sm_89. Modal and Lambda both default to Ampere or newer.

**Nothing on this list is a bug. Every item is a measurement whose answer is
device-dependent and currently known only for sm_75.** Running the sweep on a
different card without re-measuring these would silently change what the numbers
mean.

`tests/test_ampere_recheck.py` asserts this file exists and still lists every
item, so it cannot be quietly dropped.

---

## 1. The TF32 pin — UNMEASURED on Ampere

`conftest.py` pins fp32 matmuls to ieee. On the T4 that pin is **inert**: all
three arms (host defaults, pinned ieee, TF32 forced fully on with
`float32_matmul_precision='high'`) came out byte-identical at `3.051758e-05`,
because **TF32 requires sm_80 and a T4 is sm_75** — the tensor cores that would
execute it do not exist there.

So the pin is **not** shown to be unnecessary. It is untested against the thing
it defends against.

**Why it matters.** TF32 carries 10 explicit mantissa bits — exactly fp16's. An
fp32 matmul silently run in TF32 does not drift a little, it drifts at fp16
scale, roughly `4.7e-2` for this model against a `LOGIT_ATOL` of `1e-3`. Forty-
seven times over, on the float32 parity tests, and indistinguishable from a
paging bug on a new device.

**To re-measure.** Run `kaggle/session1`'s Part 2 (three arms: recorded
defaults, pinned ieee, TF32 forced on). On Ampere the arms **must** separate. If
they do not, the pin is not being applied and that is the finding.

---

## 2. SDPA backend selection — the fast kernels are unavailable on sm_75

On the T4, **FLASH_ATTENTION and CUDNN_ATTENTION reject every shape this engine
issues**, with `"No available kernel"`. That is a property of sm_75, not of this
code. On Ampere FLASH becomes available and **backend selection moves again**.

Measured on the T4, engine side:

| path | backend selected |
|---|---|
| `enable_gqa=True` (unexpanded K/V) | `MATH` |
| `repeat_kv` (pre-expanded K/V) | `EFFICIENT_ATTENTION` |

That difference is worth 6× in fp16 parity — `MATH` is the accurate reference
implementation — and it changed **silently** when the input shape changed.
`engine/attention.py::GQA_PATH` exists because of it.

**Why it matters.** The engine and the HF baseline must run on the **same**
kernel or the Gate 1 comparison is a kernel comparison with paging as a
confound. `bench/run.py` records `sdpa_backends_enabled`, but *enabled* is not
*selected* — the selection is shape-dependent.

### The baseline side — its kernel DEPENDS ON THE STREAM

`kaggle/session3` Item 1 intercepted the baseline's **real** SDPA calls during
`model.generate`: 112 calls, all passing **unexpanded** K/V with
`enable_gqa=True` and selecting `MATH`. That was read as "the baseline matches
the engine". **It does not generalise, and the reason is padding.**

HuggingFace takes the `enable_gqa` path only when `attention_mask is None`
(`transformers.integrations.sdpa_attention.use_gqa_in_sdpa`), and the mask is
None only when nothing was left-padded. Session 3 happened to probe a
prompt-uniform batch. The two sweep streams are not the same shape:

| stream | prompts | mask | HF path | baseline backend (T4, measured) |
|---|---|---|---|---|
| `control` | all 256 are 512 tokens | dropped to `None` | `enable_gqa` | **`MATH`** |
| `mixed` | 224 distinct lengths | real | pre-expands K/V | **`EFFICIENT_ATTENTION`** |

Measured in `kaggle/session5` gate 2, on the T4, against the real streams, and
reproduced on CPU in `tests/test_arms.py`.

**So the shipping configuration is not symmetric.** On `control` both sides are
`MATH`. On `mixed` the baseline gets `EFFICIENT_ATTENTION`, the fast kernel,
while the engine's masked fused call stays on `MATH` — an asymmetry **against**
the engine that nobody chose and that was invisible until it was measured.

That is why `sdpa` is the default — it matches on the stream the headline is
quoted from — and it is **not** a memory saving; see item 6.

### Symmetry is not neutrality — MEASURED, and the SIGN IS NOT CONSTANT

**Both arms have been run on both deciding cells.** `bench/arms.py`, Kaggle T4,
both arms of a cell back to back on the same card, 1 warmup + 3 passes, pass
spread 0.01–0.17%. Only the `control` cells decide this, because only there does
Arm B move the baseline as well as the engine.

| deciding cell | baseline resident | Arm A (`MATH`) | Arm B (`EFFICIENT`) | shift | overlap |
|---|---|---|---|---|---|
| `control-rect3072-C64` | 18 | 0.7699 – **0.7702** – 0.7713 | 0.7862 – **0.7880** – 0.7886 | **+2.30%** | no |
| `control-rect1757-C64` | 32 | 0.6927 – **0.6928** – 0.6929 | 0.6594 – **0.6598** – 0.6607 | **−4.77%** | no |

**The kernel does move the ratio, by at most ~5%, and the SIGN IS NOT A PROPERTY
OF THE KERNEL — it flips between the two rectangle sizings.** Neither "floor" nor
"ceiling" is the right word for it. Both arms did provably identical work on both
cells — same generated tokens, preemptions, peak blocks, rejections and
`workload_digest`, no provenance drift outside the two knobs — so the shifts are
attributable to the kernel and the disagreement is real, not noise. The intervals
are tight enough (≤0.2%) that a 2–5% shift is comfortably resolvable.

**What actually decides the sign: which side gains more from the switch.**

| cell | engine A→B | baseline A→B | winner | ratio moves |
|---|---|---|---|---|
| rect3072 (resident 18) | 1.662× | 1.625× | engine | up |
| rect1757 (resident 32) | 1.749× | **1.836×** | baseline | down |

So the kernel switch is **not** a uniform change in a shared per-token cost `k`.
The two sides issue different shapes — the engine a padded rectangle over a
64-sequence running set, the baseline a static batch of 18 or 32 — and
`EFFICIENT_ATTENTION` rewards those shapes by different factors. At resident 32
the baseline benefits more than the engine does; at 18 it benefits less.

**This refutes both of the models that were argued from theory.**

*Mechanism 1 — batch amortisation.* With `W` the batch-independent per-step
weight loading, `ratio = (B_e/B_b)(W + B_b·k)/(W + B_e·k)` is monotonically
**decreasing** in `k` whenever `B_e > B_b`, so it predicts Arm B above Arm A
**always**. rect3072 agrees; rect1757 contradicts it. The model's flaw is its
assumption that one `k` serves both sides.

*Mechanism 2 — static-batch stragglers.* A finished sequence keeps being computed
until its batch drains, which is real wasted per-token work that a costlier
kernel makes costlier, predicting Arm B **below** Arm A. rect1757 agrees;
rect3072 contradicts it. (The rectangle sizing itself was never a candidate — it
is a KV cache **allocation**, and HF masks over the true `kv_len`.)

Both effects are real and neither dominates universally. An earlier revision of
this file asserted mechanism 2 alone and called the headline an upper bound;
a later one, after only the first cell had run, called it a floor. **Both were
wrong, and wrong the same way: a direction stated from less evidence than the
question needed.** The transferable lesson is not which mechanism wins — nothing
wins — but that the sign here is a property of the *configuration* and has to be
measured per cell.

**Consequence for the headline.** The kernel contributes a bounded ±5% that
cannot be signed a priori, so it must be quoted with the arm and the rectangle it
came from. It is not a hedge to be waved at; it is a measured term.

**The `mixed` cells decide nothing about this** and must not be read as if they
did. There HF's ragged batches already pre-expand, so Arm B's patch is inert on
the baseline — measured rather than assumed, on both cells:

| cell | baseline Arm A | baseline Arm B | apart | engine A→B |
|---|---|---|---|---|
| `mixed-rect3072-C64` | 1249.92 s | 1250.45 s | 0.04% | 0.9409 → 1.8903 (2.01×) |
| `mixed-rect1757-C64` | 1208.61 s | 1207.34 s | 0.10% | 0.9859 → 1.8968 (1.92×) |

The baseline is invariant to within a tenth of a percent, which is what "inert"
has to mean if it is to be claimed. Only the engine moves, and it roughly
doubles. **That prices the handicap Arm A was carrying — the engine sitting on
`MATH` while HuggingFace already held `EFFICIENT_ATTENTION` — and it is not
evidence about kernel bias.** It is also the largest number in this experiment,
and it was nobody's decision.

**The procedure, kept runnable.** `python -m bench.arms` runs both arms;
`--compare-only` joins slices measured in separate sessions. Arm B is **void
unless** the patched baseline is token-identical to the unpatched one, and cells
whose two arms did different work are refused rather than compared.

**To re-measure.** Re-run `kaggle/session3` Item 1 and `kaggle/session4`.
On Ampere, confirm:

- the engine and the baseline still land on the same backend;
- whether FLASH accepts `enable_gqa` on sm_80+, which would make the matching
  path also the fast path and dissolve this entire tradeoff;
- whether the +2.30% floor effect measured here holds, grows or reverses. It is
  small on sm_75 only because both kernels are slow relative to `W`; on a card
  where FLASH is available the gap between the available kernels is different
  and so is the size of this term. Re-run `python -m bench.arms`.

---

## 3. `expandable_segments` — measured only against sm_75 fragmentation

`PYTORCH_ALLOC_CONF=expandable_segments:True` took peak *reserved* from 98.5% of
the card to 48.5% at C=64 with peak *allocated* unchanged to three decimals. It
changes allocation only and cannot move a logit, but every memory number in this
repo was taken with it on, and it is recorded in `Provenance.allocator_conf`.

**To re-measure.** A larger card changes the fragmentation picture and may change
whether the padded-rectangle cap needs to be 8192. Re-run `kaggle/session2`
Item 1 at the three levels.

---

## 4. `FP16_LOGIT_ATOL` is a prediction that has only been checked on sm_75

`6.25e-2`, derived in `tests/test_parity_fp16.py` from fp16's mantissa. Measured
on the T4 at 3.00 ulp of the 4 allowed with `repeat_kv`, and 0.50 ulp with
`enable_gqa` — the difference being the backend, item 2 above.

**It is not to be tightened to suit a measurement**, on this card or any other.
If Ampere lands outside the bound, report the measurement.

---

## 5. The memory caps were sized against 14.56 GiB

`max_num_batched_tokens = 8192` was chosen empirically against a T4 after the
uncapped run OOM'd at 131,072 padded slots. It is a starting point that was
measured, not a derived constant. On a 24 GB A10 it is probably conservative.

`max_num_seqs = C` is **not** device-dependent and does not belong on this list —
it exists so that C means something, and it must stay whatever the card is.

---

## 6. The `enable_gqa` memory saving does not exist on sm_75

`enable_gqa` was adopted to avoid `repeat_kv`'s materialised expansion. That
expansion is real — 768 MiB at batch 64, kv 2048. What was never measured is
what the alternative allocates. `kaggle/session4` measured it, and the switch
costs **3.3–8.3× more** transient memory, not less:

| shape | `enable_gqa` (MATH) | `repeat_kv` (EFFICIENT) |
|---|---|---|
| prefill 8×512 | 332.2 MiB | 40.0 MiB |
| decode 64×1024 | 1283.9 MiB | 384.3 MiB |
| decode 64×2048 | 2567.0 MiB | 768.4 MiB |
| prefill 16×512 | 648.1 MiB | 80.0 MiB |

MATH expands K/V for its matmul anyway and materialises the score matrix on top.

**The switch is kept anyway**, for the reason in item 2: it is the path the HF
baseline takes. **This is device-dependent** — it is a fact about which backend
`enable_gqa` selects, and on Ampere that selection moves. Re-measure with
`kaggle/session4` before repeating the memory claim either way.
