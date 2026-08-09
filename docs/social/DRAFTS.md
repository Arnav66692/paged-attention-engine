# Publication drafts and order

Everything Nav posts, in the order he posts it. Nothing here is posted by
anyone else. Drafted against the two-regime story; figures audited against
committed result files on 2026-08-09.

## Link targets

| surface | target |
|---|---|
| HN submission, X post 5, email [LINK] | docs/hf-padding-kernel-cliff.md (the finding page) |
| X post 7 | the repo root |
| upstream comment | huggingface/transformers#45776, attach docs/upstream/repro_enable_gqa_padding_cliff.py |

## Publication order

1. Post the upstream comment on #45776, attaching the repro script. Text in
   docs/upstream/DRAFT_comment_on_transformers_45776.md.
2. Update the finding page: the line "There is no sm_75 data in the thread."
   is now false. Replace with "The sm_75 data above is now in the thread:
   [link to the posted comment]." Commit and push.
3. Flip the repo public. The finding page must resolve before anything links it.
4. Submit the HN link to the finding page, then post the author comment below.
5. Post the X thread. Image on post 3, finding page on post 5, repo on post 7.
6. Send THREE emails, not ten. Wait a week, fix what the replies expose, then
   send the rest.

Step 2 sits between 1 and 3 because the email and the finding page both say the
repro is on the thread, and the page must not go public asserting an absence
that step 1 just filled.

## B1: Hacker News

Regular link submission, not Show HN. Title (78 chars):

    HuggingFace picks a slower attention kernel when your prompts are equal length

Author comment, posted immediately after submitting:

Author here. I hit this while benchmarking a paged attention engine against HuggingFace generate. The ratio kept moving for reasons that had nothing to do with my code, so I went looking.

The short version: use_gqa_in_sdpa() gates on attention_mask is None, which for generate() means "nothing was left padded", which means "all prompts the same length". It then passes unexpanded K/V with enable_gqa=True, and on sm < 80 no backend can honour that, so torch falls back to MATH.

The upstream thread at #45776 already had the mechanism and even lists sm < 80 among the constraints, but it benchmarks on sm_86 and sm_89 hardware, where FLASH is available, so the fallback is not expected to fire. There was no sm_75 data. I only have a T4, so I do not know what this does on Ampere.

The engine itself is linked at the bottom of the page: it loses on throughput when the whole workload is known in advance, running at 0.67 to 0.73x a decode-sorted static batcher, and wins on latency when it is not, with its overhead measured at 1.6 to 2.2x per row-step.

## B2: X thread, 7 posts

**1/7** (195 chars)

HuggingFace transformers picks a slower attention kernel when all your prompts are the same length.

Padding your inputs makes it faster.

On a T4: 177.8 -> 288.9 tok/s from flipping one boolean.

---

**2/7** (204 chars)

The mechanism. use_gqa_in_sdpa() returns True exactly when attention_mask is None, which for generate() means exactly when nothing got left padded.

It then hands SDPA unexpanded K/V with enable_gqa=True.

---

**3/7** (attach docs/social/torch-warning.png) (154 chars)

That flag needs a backend that can honour it. On sm < 80, so T4, V100, RTX 20xx, FLASH is unavailable and no remaining backend can.

torch says it itself:

---

**4/7** (219 chars)

Pre-expand K/V and the same call reaches EFFICIENT_ATTENTION.

Uniform batch, slow path. Add one shorter prompt, a mask appears, HF pre-expands, fast path.

The config that looks obviously optimal is the one that loses.

---

**5/7** (273 chars)

Standalone repro, torch and transformers only. Its defaults are small, batch 8 and 64 new tokens, and it reports roughly 1.3x across runs. The 177.8 -> 288.9 in the first post is 1.62x, at batch 18 with 512 decode tokens, and the gap widens with both.

[LINK: finding page]

---

**6/7** (267 chars)

Found this benchmarking a paged attention engine I wrote from scratch.

Offline, whole workload known: against a static batcher that sorts by decode length, my engine runs at 0.67 to 0.73x its throughput. It pays 1.6 to 2.2x per row-step for gather, masking, scatter.

---

**7/7** (259 chars)

Online at 40 percent load: the engine wins TTFT p99 by 67x and p50 latency by 5x. 36 of 87 batches formed on one queued request, and you cannot sort a queue of one.

Textbook continuous batching, confirmed on this implementation. Pre-registered.

[LINK: repo]

## B3: outreach email

Subject: transformers picks a slow attention kernel on unpadded batches

Hi [NAME], you wrote [SPECIFIC POST OR TALK], so this may be worth ninety seconds. transformers selects a slower SDPA kernel when every prompt in a batch is the same length, because use_gqa_in_sdpa() gates on attention_mask is None and on sm < 80 nothing can honour enable_gqa, and on a T4 with Qwen2.5 1.5B, forcing K/V pre-expansion and changing nothing else took it from 177.8 to 288.9 tok/s, with a 20 line repro on the writeup at [LINK: finding page] plus the upstream thread at #45776. I found it while benchmarking a paged attention engine I wrote from scratch, linked at the bottom, which loses on throughput to a decode-sorted static batcher when the whole workload is known in advance, and wins on latency under an arrival process, with its overhead measured at 1.6 to 2.2x per row-step. If you have a spare minute I would value knowing whether the kernel cliff reproduces on [THEIR HARDWARE], since I only had a T4. I'm eighteen.

Recipients are Nav's choice. The kind of person: whoever bylines a serving
company's inference or kernel engineering posts, maintainers visible in
transformers SDPA and attention PR review, authors of public latency writeups.
