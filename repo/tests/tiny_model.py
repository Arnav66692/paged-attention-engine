"""A tiny Qwen2 engine with random weights, shared by the bench smoke tests.

Real forward passes, real block accounting, real evictions -- two layers and a
32-wide hidden state instead of 1.5B parameters, so a laptop can run a whole
sweep in seconds. Nothing that depends on the weights is asserted anywhere it is
used: admission and eviction read token counts, never token values, so a random
model schedules exactly as the real one does.
"""

import torch

from engine.model import LayerWeights, ModelDims, Qwen2Engine, Qwen2Weights

TINY_DIMS = ModelDims(
    num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, hidden_size=32,
    rms_norm_eps=1e-6, vocab_size=64, rope_theta=10000.0,
)
TINY_INTERMEDIATE = 64


def tiny_engine(dims: ModelDims = TINY_DIMS, seed: int = 0) -> Qwen2Engine:
    torch.manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape) * 0.02

    layers = tuple(
        LayerWeights(
            q_proj_weight=randn(dims.num_heads * dims.head_dim, dims.hidden_size),
            q_proj_bias=torch.zeros(dims.num_heads * dims.head_dim),
            k_proj_weight=randn(dims.num_kv_heads * dims.head_dim, dims.hidden_size),
            k_proj_bias=torch.zeros(dims.num_kv_heads * dims.head_dim),
            v_proj_weight=randn(dims.num_kv_heads * dims.head_dim, dims.hidden_size),
            v_proj_bias=torch.zeros(dims.num_kv_heads * dims.head_dim),
            o_proj_weight=randn(dims.hidden_size, dims.num_heads * dims.head_dim),
            gate_proj_weight=randn(TINY_INTERMEDIATE, dims.hidden_size),
            up_proj_weight=randn(TINY_INTERMEDIATE, dims.hidden_size),
            down_proj_weight=randn(dims.hidden_size, TINY_INTERMEDIATE),
            input_layernorm_weight=torch.ones(dims.hidden_size),
            post_attention_layernorm_weight=torch.ones(dims.hidden_size),
        )
        for _ in range(dims.num_layers)
    )
    weights = Qwen2Weights(
        embed_tokens=randn(dims.vocab_size, dims.hidden_size),
        layers=layers,
        final_norm_weight=torch.ones(dims.hidden_size),
        lm_head_weight=randn(dims.vocab_size, dims.hidden_size),
    )
    return Qwen2Engine(weights, dims, torch.device("cpu"), torch.float32)
