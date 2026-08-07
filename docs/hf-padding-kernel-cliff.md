# HuggingFace picks a slower attention kernel when your prompts are equal length

On GPUs without FLASH, `transformers` runs a uniform batch on the slow kernel
and a padded one on the fast kernel. Padding your inputs makes it faster. This
is the opposite of what anyone would guess, and it is invisible from the
outside: no warning, no fallback message.

## The mechanism

`transformers.integrations.sdpa_attention.use_gqa_in_sdpa()` returns `True`
exactly when `attention_mask is None`. For `generate()` that means exactly when
nothing was left padded, which means when every prompt in the batch is the same
length. It then hands SDPA unexpanded K/V with `enable_gqa=True`.

That flag needs a backend that can honour it. On sm < 80, which covers the T4,
V100 and RTX 20xx, FLASH is unavailable, so no remaining backend can, and torch
falls back to `MATH`. Pre-expanding K/V with `repeat_kv` first reaches
`EFFICIENT_ATTENTION` instead.

torch says this itself. Both lines come from a real run:

```
Flash attention only supports gpu architectures in the range [sm80, sm121].
  Attempting to run on a sm 7.5 gpu.
For dense input, both fused kernels require query, key and value to have the
  same num_heads. Query.sizes(): [8, 12, 512, 128],
  Key sizes(): [8, 2, 512, 128], Value sizes(): [8, 2, 512, 128] instead.
```

The second line is the whole thing. The unexpanded K/V shape is why both fused
kernels decline, and `enable_gqa=True` is what produces that shape.

## What it costs

Qwen2.5 1.5B Instruct, fp16, Tesla T4, real `model.generate()`, 256 requests of
512 prompt plus 512 decode tokens. Median of 3 passes after a discarded warmup,
pass spread under 0.2%. The only change is forcing `use_gqa_in_sdpa` to return
`False`:

| batch | as shipped | K/V pre-expanded | |
|---|---|---|---|
| 18 resident | 177.8 tok/s | 288.9 tok/s | **1.62x** |
| 32 resident | 202.0 tok/s | 370.9 tok/s | **1.84x** |

Identical work in both columns. Same prompts, same token counts, greedy,
`min_new_tokens == max_new_tokens`, and the batch is uniform so `attention_mask`
is `None` on both sides.

## Reproducing it

```python
import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(8, 12, 512, 128, dtype=torch.float16, device="cuda")
k = torch.randn(8,  2, 512, 128, dtype=torch.float16, device="cuda")
v = torch.randn_like(k)

def which(*args, **kw):
    ref = F.scaled_dot_product_attention(*args, **kw)
    out = []
    for name, be in (("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                     ("MATH", SDPBackend.MATH)):
        try:
            with sdpa_kernel(be):
                if torch.equal(F.scaled_dot_product_attention(*args, **kw), ref):
                    out.append(name)
        except Exception:
            pass
    return out

rep = lambda x: x.repeat_interleave(6, dim=1)
print("enable_gqa   ->", which(q, k, v, enable_gqa=True))   # ['MATH']
print("pre-expanded ->", which(q, rep(k), rep(v)))          # ['EFFICIENT']
```

A fuller version that times two real `generate()` calls is
[here](upstream/repro_enable_gqa_padding_cliff.py). Its defaults are smaller,
batch 8 and 64 new tokens, and it reports **1.35x**. The gap widens with decode
length and batch, which is why the table above is larger.

I have not run any of this on sm >= 80. If FLASH can honour `enable_gqa` there,
the two paths should land on the same backend and the gap should close, but that
is a prediction and not something I measured.

## Where it goes

Reported on
[huggingface/transformers#45776](https://github.com/huggingface/transformers/issues/45776),
which already describes the mechanism and lists `sm < 80` among the constraints,
but benchmarks on sm_86 and sm_89 hardware, where FLASH exists and the fallback
never fires. There was no sm_75 data in the thread.

Found while benchmarking a paged attention inference engine I wrote from
scratch. That is [here](https://github.com/Arnav66692/paged-attention-engine).
