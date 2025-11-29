"""Scaled dot-product attention and multi-head self-attention.

The computation itself is the fused op in :mod:`tensorgrad.ops.attention`; this module adds
the learnable projections.
"""

from __future__ import annotations

from tensorgrad.nn.layers import Dropout, Linear
from tensorgrad.nn.module import Module
from tensorgrad.ops.attention import (
    causal_mask,
    packed_self_attention,
    scaled_dot_product_attention,
)
from tensorgrad.tensor import Tensor

__all__ = ["MultiHeadAttention", "causal_mask", "scaled_dot_product_attention"]


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

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"expected (B, T, {self.embed_dim}) input, got {x.shape}")
        # One fused node from the packed (B, T, 3D) projections to merged heads (B, T, D).
        y = packed_self_attention(
            self.qkv(x),
            self.num_heads,
            causal=self.causal,
            dropout_p=self.dropout,
            training=self.training,
        )
        return self.resid_dropout(self.proj(y))

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, causal={self.causal}"
