"""Qwen2 forward pass and greedy decode loop, owned by this engine.

Gate 1, Step 1. The forward pass is written out here rather than delegated to
transformers so that the KV cache and, at Step 3, the attention itself are ours
to replace. No HuggingFace cache object appears anywhere in this path: the only
thing taken from transformers is the weight tensors.

The arithmetic mirrors transformers 5.12.1 `models/qwen2/modeling_qwen2.py` as
read from the installed package, including the details that matter numerically:
RMSNorm accumulates in float32 and casts back, the rotary tables are built in
float32 by the same matmul-then-concat expression, and attention scales by
head_dim ** -0.5.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from engine.attention import AttentionBackend, ContiguousBackend, PagedBackend
from engine.cache import (
    BLOCK_SIZE,
    BlockAllocator,
    CacheSpec,
    ContiguousKVCache,
    PagedCacheSpec,
    PagedKVCache,
)


@dataclass(frozen=True)
class ModelDims:
    """Everything the forward pass needs to know about the checkpoint."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5

    @classmethod
    def from_config(cls, config) -> "ModelDims":
        # head_dim is absent from the Qwen2.5-0.5B config; transformers falls
        # back to hidden_size // num_attention_heads, so do the same.
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        return cls(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            rope_theta=config.rope_parameters["rope_theta"],
        )


@dataclass(frozen=True)
class LayerWeights:
    """Views onto one decoder layer's tensors. Not copies."""

    q_proj_weight: torch.Tensor
    q_proj_bias: torch.Tensor
    k_proj_weight: torch.Tensor
    k_proj_bias: torch.Tensor
    v_proj_weight: torch.Tensor
    v_proj_bias: torch.Tensor
    o_proj_weight: torch.Tensor
    gate_proj_weight: torch.Tensor
    up_proj_weight: torch.Tensor
    down_proj_weight: torch.Tensor
    input_layernorm_weight: torch.Tensor
    post_attention_layernorm_weight: torch.Tensor


@dataclass(frozen=True)
class Qwen2Weights:
    embed_tokens: torch.Tensor
    layers: tuple[LayerWeights, ...]
    final_norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor


@dataclass(frozen=True)
class GenerationResult:
    """Greedy decode output. `step_logits[i]` is what produced `token_ids[i]`."""

    token_ids: tuple[int, ...]
    step_logits: torch.Tensor  # (num_steps, vocab_size)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen2RMSNorm: accumulate the variance in float32, cast back, then scale."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the halves of the last dimension, as Qwen2 defines it."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def build_inv_freq(head_dim: int, rope_theta: float, device: torch.device) -> torch.Tensor:
    """Inverse frequencies for the `default` rope type."""
    exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
    return 1.0 / (rope_theta ** (exponent / head_dim))


def build_cos_sin(
    inv_freq: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary tables for `positions`, built in float32 then cast to `dtype`.

    The matmul-then-concat shape is deliberate: it is the expression
    transformers uses, and reproducing it avoids a needless rounding difference
    against the baseline. `attention_scaling` is 1.0 for the `default` rope type
    Qwen2.5 uses, so it is not applied.
    """
    inv_freq_expanded = inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1)
    positions_expanded = positions[:, None, :].float()
    freqs = (inv_freq_expanded @ positions_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to q and k, which are (batch, heads, seq, head_dim)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Qwen2Engine:
    """Single-sequence greedy decoding over a KV cache this engine allocates."""

    def __init__(
        self,
        weights: Qwen2Weights,
        dims: ModelDims,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.weights = weights
        self.dims = dims
        self.device = device
        self.dtype = dtype
        self.inv_freq = build_inv_freq(dims.head_dim, dims.rope_theta, device)

    # --- construction ------------------------------------------------------

    @classmethod
    def from_hf_model(cls, hf_model) -> "Qwen2Engine":
        """Borrow weight tensors from a loaded transformers model.

        Views, not copies, so the parity test can hold one set of weights and
        compare two forward passes over them. Only tensors cross this boundary;
        no transformers module or cache object is retained.
        """
        state = hf_model.state_dict()
        dims = ModelDims.from_config(hf_model.config)
        embed = state["model.embed_tokens.weight"]
        layers = tuple(
            LayerWeights(
                q_proj_weight=state[f"model.layers.{i}.self_attn.q_proj.weight"],
                q_proj_bias=state[f"model.layers.{i}.self_attn.q_proj.bias"],
                k_proj_weight=state[f"model.layers.{i}.self_attn.k_proj.weight"],
                k_proj_bias=state[f"model.layers.{i}.self_attn.k_proj.bias"],
                v_proj_weight=state[f"model.layers.{i}.self_attn.v_proj.weight"],
                v_proj_bias=state[f"model.layers.{i}.self_attn.v_proj.bias"],
                o_proj_weight=state[f"model.layers.{i}.self_attn.o_proj.weight"],
                gate_proj_weight=state[f"model.layers.{i}.mlp.gate_proj.weight"],
                up_proj_weight=state[f"model.layers.{i}.mlp.up_proj.weight"],
                down_proj_weight=state[f"model.layers.{i}.mlp.down_proj.weight"],
                input_layernorm_weight=state[f"model.layers.{i}.input_layernorm.weight"],
                post_attention_layernorm_weight=state[
                    f"model.layers.{i}.post_attention_layernorm.weight"
                ],
            )
            for i in range(dims.num_layers)
        )
        # Qwen2.5-0.5B ties lm_head to the embedding matrix, so the key is
        # absent from the state dict. Fall back rather than assume either way.
        lm_head = state.get("lm_head.weight", embed)
        weights = Qwen2Weights(
            embed_tokens=embed,
            layers=layers,
            final_norm_weight=state["model.norm.weight"],
            lm_head_weight=lm_head,
        )
        return cls(weights, dims, embed.device, embed.dtype)

    def new_cache(self, max_seq_len: int) -> ContiguousKVCache:
        """Allocate a contiguous cache sized for one sequence of at most `max_seq_len`."""
        return ContiguousKVCache(
            CacheSpec(
                num_layers=self.dims.num_layers,
                num_kv_heads=self.dims.num_kv_heads,
                head_dim=self.dims.head_dim,
                max_seq_len=max_seq_len,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def new_contiguous_backend(self, max_seq_len: int) -> ContiguousBackend:
        """Step 1's storage, still the reference the paged path is checked against."""
        return ContiguousBackend(
            self.new_cache(max_seq_len), self.dims.num_kv_groups, self.dims.attn_scale
        )

    def new_paged_cache(
        self, num_blocks: int, block_size: int = BLOCK_SIZE
    ) -> PagedKVCache:
        """Allocate a physical block pool matching this model's KV geometry."""
        return PagedKVCache(
            PagedCacheSpec(
                num_layers=self.dims.num_layers,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=self.dims.num_kv_heads,
                head_dim=self.dims.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def new_paged_backend(
        self, cache: PagedKVCache, allocator: BlockAllocator, seq_id: int
    ) -> PagedBackend:
        """Register `seq_id` with the allocator and return a backend over it.

        The caller owns the pool and the allocator, so a test can fragment the
        pool before the sequence is ever registered.
        """
        allocator.create_sequence(seq_id)
        return PagedBackend(
            cache, allocator, seq_id, self.dims.num_kv_groups, self.dims.attn_scale
        )

    # --- forward -----------------------------------------------------------

    def _attention(
        self,
        layer: LayerWeights,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        backend: AttentionBackend,
        layer_idx: int,
    ) -> torch.Tensor:
        """Project and rotate Q/K/V, then let the backend decide where KV lives.

        Everything below the rotary embedding is storage-agnostic: the backend
        stores this layer's K/V and attends over the whole sequence. Contiguous
        and paged differ only inside `attend`.
        """
        dims = self.dims
        n = hidden.shape[1]

        q = F.linear(hidden, layer.q_proj_weight, layer.q_proj_bias)
        k = F.linear(hidden, layer.k_proj_weight, layer.k_proj_bias)
        v = F.linear(hidden, layer.v_proj_weight, layer.v_proj_bias)

        q = q.view(1, n, dims.num_heads, dims.head_dim).transpose(1, 2)
        k = k.view(1, n, dims.num_kv_heads, dims.head_dim).transpose(1, 2)
        v = v.view(1, n, dims.num_kv_heads, dims.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        attn = backend.attend(layer_idx, q, k, v)
        attn = attn.transpose(1, 2).reshape(1, n, dims.num_heads * dims.head_dim)
        return F.linear(attn, layer.o_proj_weight)

    def _mlp(self, layer: LayerWeights, hidden: torch.Tensor) -> torch.Tensor:
        gate = F.linear(hidden, layer.gate_proj_weight)
        up = F.linear(hidden, layer.up_proj_weight)
        return F.linear(F.silu(gate) * up, layer.down_proj_weight)

    @torch.inference_mode()
    def forward(
        self, token_ids: torch.Tensor, positions: torch.Tensor, backend: AttentionBackend
    ) -> torch.Tensor:
        """Run one forward pass and return logits for the final query position.

        `token_ids` and `positions` are 1-D and the same length. The backend
        reserves room first so every layer writes at the same offset, and the
        sequence length only moves once all of them have.
        """
        if token_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("token_ids and positions must be 1-D")
        if token_ids.shape[0] != positions.shape[0]:
            raise ValueError(
                f"{token_ids.shape[0]} tokens but {positions.shape[0]} positions"
            )

        n = token_ids.shape[0]
        reservation = backend.begin_forward(n)
        if not reservation.ok:
            # The allocator reports exhaustion as a value, which is what lets a
            # scheduler preempt instead of unwinding. There is no scheduler yet,
            # so at this layer it becomes an error rather than being swallowed.
            raise RuntimeError(
                f"cache exhausted reserving {n} token(s): {reservation.status.value}"
            )

        hidden = self.weights.embed_tokens[token_ids].unsqueeze(0)
        cos, sin = build_cos_sin(self.inv_freq, positions.unsqueeze(0), hidden.dtype)

        for layer_idx, layer in enumerate(self.weights.layers):
            residual = hidden
            hidden = rms_norm(hidden, layer.input_layernorm_weight, self.dims.rms_norm_eps)
            hidden = residual + self._attention(layer, hidden, cos, sin, backend, layer_idx)

            residual = hidden
            hidden = rms_norm(
                hidden, layer.post_attention_layernorm_weight, self.dims.rms_norm_eps
            )
            hidden = residual + self._mlp(layer, hidden)

        backend.end_forward(n)

        hidden = rms_norm(hidden, self.weights.final_norm_weight, self.dims.rms_norm_eps)
        return F.linear(hidden[0, -1], self.weights.lm_head_weight)

    # --- decode loop -------------------------------------------------------

    @torch.inference_mode()
    def generate_greedy(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        backend: AttentionBackend | None = None,
    ) -> GenerationResult:
        """Greedy decode. EOS is ignored: exactly `max_new_tokens` are produced.

        Ignoring EOS is the workload contract from bench/workload.py, and it
        keeps this comparable against a baseline doing the same amount of work.

        `backend` selects where KV lives. Default is the contiguous cache; pass a
        `PagedBackend` to run the same decode over scattered physical blocks.
        """
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, got {max_new_tokens}")
        if not prompt_token_ids:
            raise ValueError("prompt must be non-empty")

        prompt_len = len(prompt_token_ids)
        if backend is None:
            backend = self.new_contiguous_backend(prompt_len + max_new_tokens)

        prompt = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(prompt_len, dtype=torch.long, device=self.device)
        logits = self.forward(prompt, positions, backend)

        generated: list[int] = []
        step_logits: list[torch.Tensor] = []
        for step in range(max_new_tokens):
            step_logits.append(logits)
            token = int(torch.argmax(logits).item())
            generated.append(token)
            if step + 1 == max_new_tokens:
                break
            next_ids = torch.tensor([token], dtype=torch.long, device=self.device)
            next_pos = torch.tensor(
                [prompt_len + step], dtype=torch.long, device=self.device
            )
            logits = self.forward(next_ids, next_pos, backend)

        return GenerationResult(
            token_ids=tuple(generated), step_logits=torch.stack(step_logits)
        )
