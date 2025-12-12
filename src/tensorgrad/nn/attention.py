"""Scaled dot-product attention and multi-head self-attention.

The computation itself is the fused op in :mod:`tensorgrad.ops.attention`; this module adds
the learnable projections.
"""

from __future__ import annotations

from dataclasses import dataclass

from tensorgrad._types import Array
from tensorgrad.autograd import no_grad
from tensorgrad.nn.layers import Dropout, Linear
from tensorgrad.nn.module import Module
from tensorgrad.ops.attention import (
    _split_heads,
    attention_forward,
    causal_mask,
    packed_self_attention,
    scaled_dot_product_attention,
)
from tensorgrad.tensor import Tensor

__all__ = ["LayerCache", "MultiHeadAttention", "causal_mask", "scaled_dot_product_attention"]


@dataclass
class LayerCache:
    """One attention layer's keys and values for the positions processed so far.

    ``keys`` and ``values`` are preallocated ``(B, H, capacity, head_dim)`` buffers of which
    the first ``length`` positions are filled; see :meth:`tensorgrad.nn.GPT.make_cache`.
    """

    keys: Array
    values: Array
    length: int = 0


class MultiHeadAttention(Module):
    """Multi-head self-attention with a fused QKV projection.

    Args:
        embed_dim: Model width ``D``; must be divisible by ``num_heads``.
        num_heads: Number of attention heads ``H`` (head size ``D / H``).
        dropout: Dropout on attention weights and on the output projection.
        causal: Apply a causal mask (autoregressive models).
        bias: Use biases in the projections.

    Input and output are ``(B, T, D)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = Linear(embed_dim, embed_dim, bias=bias)
        self.resid_dropout = Dropout(dropout)

    def forward(self, x: Tensor, cache: LayerCache | None = None) -> Tensor:
        """Attend over ``x`` (``(B, T, D)``) or, with a ``cache``, over cached plus new keys.

        With a cache the call is for inference: ``x`` holds only the new positions, their keys
        and values are appended to the cache, and no graph is recorded.
        """
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"expected (B, T, {self.embed_dim}) input, got {x.shape}")
        if cache is not None:
            return self._forward_cached(x, cache)
        # One fused node from the packed (B, T, 3D) projections to merged heads (B, T, D).
        y = packed_self_attention(
            self.qkv(x),
            self.num_heads,
            causal=self.causal,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.resid_dropout(self.proj(y))

    def _forward_cached(self, x: Tensor, cache: LayerCache) -> Tensor:
        b, t, d = x.shape
        start, end = cache.length, cache.length + t
        if end > cache.keys.shape[2]:
            raise ValueError(f"KV cache holds {cache.keys.shape[2]} positions; {end} requested")
        with no_grad():
            q, k, v = _split_heads(self.qkv(x).data, self.num_heads)
            cache.keys[:, :, start:end] = k
            cache.values[:, :, start:end] = v
            cache.length = end
            # The new queries sit at positions start..end-1 and may see every earlier key.
            forbidden = causal_mask(t, end) if self.causal else None
            keys, values = cache.keys[:, :, :end], cache.values[:, :, :end]
            y, _ = attention_forward(q, keys, values, forbidden, None, 1.0)
            merged = Tensor._wrap(y.transpose(0, 2, 1, 3).reshape(b, t, d))
            return self.resid_dropout(self.proj(merged))

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, causal={self.causal}"
