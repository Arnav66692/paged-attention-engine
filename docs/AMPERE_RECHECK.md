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

### Symmetry is not neutrality — the bias is UNMEASURED AND OF UNKNOWN SIGN

`kaggle/session4` timed the two paths on the shapes the engine actually issues:

| shape | `enable_gqa` (MATH) | `repeat_kv` (EFFICIENT) | ratio |
|---|---|---|---|
| prefill 8×512 | 9.20 ms | 2.05 ms | 4.48× |
| decode 64×1024 | 11.62 ms | 4.21 ms | 2.76× |
| decode 64×2048 | 24.47 ms | 8.78 ms | 2.79× |
| prefill 16×512 | 14.93 ms | 2.63 ms | 5.67× |

Both sides being on MATH makes the comparison **symmetric**. It does not
establish that the kernel is neutral. **Which way it biases the ratio — if at
all — is UNMEASURED AND OF UNKNOWN SIGN.** Two mechanisms are known, they pull
in opposite directions, and neither can be called from theory.

**Mechanism 1 — a slow kernel shrinks the ratio (T4 number would be a FLOOR).**
Model the throughput ratio as

```
ratio = (B_e / B_b) · (W + B_b·k) / (W + B_e·k)
```

with `W` the batch-independent per-step weight loading, `k` the per-token cost,
`B_e` and `B_b` the engine's and baseline's batch sizes. As `k` rises the ratio
falls toward 1. A slower attention kernel raises `k` for **both** sides, so MATH
would *compress* the advantage and the sm_75 number would understate it.

**Mechanism 2 — a slow kernel inflates the ratio (T4 number would be a CEILING).**
This needs the baseline to do more attention work *per useful token*. The
rectangle sizing alone does **not** do that — it is a KV cache **allocation**,
and HF masks over the actual `kv_len`. **Static-batch stragglers do**: a finished
sequence keeps being computed until the whole batch drains, and that is real
wasted per-token work that a more expensive kernel makes more expensive.

An earlier revision of this file asserted mechanism 2 and labelled the headline
an upper bound. That was a direction claimed without measurement, and it ignored
mechanism 1 entirely. **An asserted direction is worse than an admitted gap.**

**To resolve.** Run the sweep twice, same host, same session:

- **Arm A** — both sides on `MATH`: `GQA_PATH=sdpa`, baseline unpatched.
- **Arm B** — both sides on `EFFICIENT_ATTENTION`: `GQA_PATH=repeat_kv`, baseline
  patched to pre-expand K/V. Arm B is **void unless** the patched baseline is
  token-identical to the unpatched one.

If the two ratios agree within measurement spread, kernel choice does not bias
the result and the headline ships with no hedge at all. If they differ, the
direction and magnitude *is* the finding. Until then this entry says only that
the question is open.

**To re-measure.** Re-run `kaggle/session3` Item 1 and `kaggle/session4`.
On Ampere, confirm:

- the engine and the baseline still land on the same backend;
- whether FLASH accepts `enable_gqa` on sm_80+, which would make the matching
  path also the fast path and dissolve this entire tradeoff;
- if they still split, whether the MATH amplification above is large enough that
  the headline must be qualified or retaken.

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
