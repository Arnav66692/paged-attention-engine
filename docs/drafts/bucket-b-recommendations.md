# Bucket B — recommended disclosures, not yet applied

**Status: LOCAL DRAFT. Not published, not submitted, not sent anywhere.**

Three disclosure recommendations from the pre-publication audit, with their
numbers updated against the session10 measurement. Each names the file and
section it belongs in. None has been applied.

---

## B1. Between-session drift on byte-identical work

**Recommended text** (new subsection in `docs/FINDINGS.md`, immediately after the
headline table; one sentence of it also belongs in the README method bullets):

> **Between-session reproducibility.** Pass spread within a level is 0.02% to
> 0.25%, and that is the only error bar the escalation rule can see. It does not
> bound host-to-host variation. Four pairs of runs did byte-identical engine work
> in two different Kaggle sessions:
>
> | cell | session | session | apart |
> |---|---|---|---|
> | engine, `mixed` C=32 | 7a: 391.833 s | 7b: 344.862 s | 13.62% |
> | engine, `mixed` C=87 | 7a: 944.769 s | 7b: 812.373 s | 16.30% |
> | engine, `mixed` C=64 Arm A | 6c: 1328.373 s | 6d: 1364.788 s | 2.74% |
> | engine, `mixed` C=64 Arm B | 6c: 661.495 s | 6d: 708.612 s | 7.12% |
>
> Same generated tokens, same peak blocks, same preemption count, same
> `workload_digest`, no provenance drift. Absolute tok/s columns are therefore
> not commensurable across rows drawn from different sessions.
>
> Ratios are more robust than absolutes, because engine and baseline share a
> session and a card. Session 10 measured that directly: it reran the Arm B
> engine for both `mixed` C=64 cells and landed at 91.562 and 90.268 tok/s
> against session 6's 89.487 and 83.537 — 2.3% and 8.1% apart on the absolutes —
> while the arrival-order ratios came out 1.9139 and 1.8850 against session 6's
> 1.890 and 1.897, that is 1.3% and 0.6% apart. Within one session the engine
> reproduced to 0.01% and 0.03% across two separate `bench.run` processes.

**Numbers updated from the audit draft:** the session-10 engine and ratio
reproducibility figures are new; the four drift pairs are unchanged.

---

## B2. The `C` dip crosses a session boundary

**Recommended text** (append to the "not monotonic in `C`" paragraph in
`docs/FINDINGS.md`):

> The `C=64` cells were measured in sessions 6c and 6d; `C=32` and `C=87` came
> from 7a and 7b. Byte-identical engine work moved 13.6% and 16.3% between those
> two sessions, and the dip is 7.8% at rect3072 and 3.5% at rect1757. The dip is
> therefore not separable from between-session variation with the runs on hand,
> and no cell has been repeated in a second session to bound it.

**Numbers updated from the audit draft:** the dip magnitudes are now stated
explicitly (2.050 to 1.890 is 7.8%; 1.965 to 1.897 is 3.5%).

---

## B3. The length-sorted baseline — MEASURED, no longer a gap

This item was "no length-sorted static-batching control exists." It has been
measured. The recommendation is no longer a disclosure of absence but of result,
and the README headline has already been rewritten around it.

**Recommended text** (new subsection in `docs/FINDINGS.md`, before the headline
table, since it governs how that table should be read):

> **The headline is against an arrival-order static batcher, and it does not
> survive a sorted one.** `bench/baseline.py` batches accepted requests in stream
> order and runs every row until the batch's longest member finishes its budget.
> Sorting the accepted list by decode budget first — `batch_order=sorted`, which
> changes ordering and nothing else — was measured in session 10, with the engine
> rerun alongside each baseline in the same session on the same T4:
>
> | cell | baseline arrival | baseline sorted | engine | ratio arrival | ratio sorted |
> |---|---|---|---|---|---|
> | `mixed-rect3072-C64` | 47.841 tok/s | 125.227 tok/s | 91.56 tok/s | 1.9139 | **0.7311** |
> | `mixed-rect1757-C64` | 47.887 tok/s | 134.951 tok/s | 90.28 tok/s | 1.8850 | **0.6691** |
>
> Baseline forward passes fall from 10,822 to 3,843 and from 6,743 to 2,291;
> decode-step efficiency rises from 0.313 to 0.862 and from 0.254 to 0.732. The
> engine reproduced to 0.01% and 0.03% across the two runs of each cell, so the
> whole of the movement is the baseline's.
>
> The ragged-stream margin is therefore recoverable by reordering a list, and
> nothing in this repository isolates a contribution from paging or continuous
> batching. `bench/results/batch-order/`.

**Numbers updated from the audit draft:** every figure here is new. The audit's
projection was that sorting would put the ratio at 0.68 to 0.80 and flip the
sign; the measured values are 0.7311 and 0.6691. The sign flip held; the
projected magnitude for rect3072 was 0.80 against 0.7311 measured, low by 0.0689.
