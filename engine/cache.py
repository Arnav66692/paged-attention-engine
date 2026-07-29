"""KV cache storage.

Gate 1, Step 1: one contiguous K tensor and one contiguous V tensor per layer,
allocated up front for a single sequence. Nothing here is a HuggingFace cache
object and nothing here subclasses one -- replacing that is the point of this
step. The block allocator that supersedes this arrives at Step 2.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheSpec:
    """Shape and placement of the KV cache. All fields come from the model config."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_kv_heads", "head_dim", "max_seq_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")


class ContiguousKVCache:
    """Per-layer contiguous K/V storage for exactly one sequence.

    Each layer owns two tensors of shape (1, num_kv_heads, max_seq_len, head_dim).
    The leading 1 is the batch dimension that scaled_dot_product_attention wants;
    Step 1 is single-request, so it is always 1.

    Slots are assigned strictly in position order: token at position p lives in
    slot p of every layer. That one-to-one mapping between logical position and
    physical slot is exactly what the block table replaces at Step 2.
    """

    def __init__(self, spec: CacheSpec) -> None:
        self.spec = spec
        shape = (1, spec.num_kv_heads, spec.max_seq_len, spec.head_dim)
        self._keys = [
            torch.zeros(shape, dtype=spec.dtype, device=spec.device) for _ in range(spec.num_layers)
        ]
        self._values = [
            torch.zeros(shape, dtype=spec.dtype, device=spec.device) for _ in range(spec.num_layers)
        ]
        self._length = 0

    @property
    def length(self) -> int:
        """Number of positions currently committed."""
        return self._length

    def reset(self) -> None:
        """Forget the sequence. Storage is retained; stale slots are overwritten."""
        self._length = 0

    def write(self, layer_idx: int, start: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write `key`/`value` of shape (1, num_kv_heads, n, head_dim) at slot `start`.

        Writes are not committed until `advance` is called, so every layer writes
        at the same `start` for a given forward pass.
        """
        if not 0 <= layer_idx < self.spec.num_layers:
            raise IndexError(f"layer {layer_idx} out of range for {self.spec.num_layers} layers")
        n = key.shape[2]
        if value.shape[2] != n:
            raise ValueError(f"key has {n} positions but value has {value.shape[2]}")
        if start + n > self.spec.max_seq_len:
            raise ValueError(
                f"writing {n} positions at slot {start} exceeds max_seq_len {self.spec.max_seq_len}"
            )
        self._keys[layer_idx][:, :, start : start + n, :] = key
        self._values[layer_idx][:, :, start : start + n, :] = value

    def advance(self, n: int) -> None:
        """Commit `n` positions, after every layer has written them."""
        if n < 0:
            raise ValueError(f"cannot advance by {n}")
        if self._length + n > self.spec.max_seq_len:
            raise ValueError(
                f"advancing to {self._length + n} exceeds max_seq_len {self.spec.max_seq_len}"
            )
        self._length += n

    def keys(self, layer_idx: int, length: int) -> torch.Tensor:
        """A view of the first `length` key slots. No copy."""
        return self._keys[layer_idx][:, :, :length, :]

    def values(self, layer_idx: int, length: int) -> torch.Tensor:
        """A view of the first `length` value slots. No copy."""
        return self._values[layer_idx][:, :, :length, :]

    def storage_bytes(self) -> int:
        """Total bytes held across all layers, K and V."""
        per_tensor = self._keys[0].numel() * self._keys[0].element_size()
        return 2 * self.spec.num_layers * per_tensor
