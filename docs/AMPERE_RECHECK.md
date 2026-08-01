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

Measured on the T4:

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

**To re-measure.** Run `kaggle/session3` Item 1, which intercepts the baseline's
real SDPA calls and identifies the backend for each. Re-confirm on Ampere that:

- the engine and the baseline still land on the same backend, and
- `GQA_PATH` is still the right choice — FLASH may accept `enable_gqa` on sm_80+,
  which would make the allocation-free path also the fast path and settle it.

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
