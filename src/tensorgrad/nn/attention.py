"""Scaled dot-product attention and multi-head self-attention."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from tensorgrad import ops
from tensorgrad._types import Array
from tensorgrad.nn.layers import Dropout, Linear
from tensorgrad.nn.module import Module
from tensorgrad.tensor import Tensor

__all__ = ["MultiHeadAttention", "causal_mask", "scaled_dot_product_attention"]


@lru_cache(maxsize=32)
def causal_mask(q_len: int, k_len: int) -> Array:
    """Boolean ``(q_len, k_len)`` mask that is ``True`` where attention is *forbidden*.

    Query ``i`` may attend to keys ``0 .. i + (k_len - q_len)``; with equal lengths that is
    "no peeking at future positions". The cached array is read-only.
    """
    mask = np.triu(np.ones((q_len, k_len), dtype=np.bool_), k=1 + k_len - q_len)
    mask.flags.writeable = False
    return mask


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    mask: Array | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tensor:
    """``softmax(q k^T / sqrt(d) + mask) v`` over the last two axes.

    Args:
        q: Queries ``(..., T_q, d)``.
        k: Keys ``(..., T_k, d)``.
        v: Values ``(..., T_k, d_v)``.
        causal: Forbid attention to future positions.
        mask: Extra boolean mask (broadcastable to ``(..., T_q, T_k)``), ``True`` = forbidden.
        dropout_p: Dropout on the attention weights (training only).
        training: Whether dropout is active.
    """
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    forbidden = mask
    if causal:
        cm = causal_mask(q.shape[-2], k.shape[-2])
        forbidden = cm if forbidden is None else (forbidden | cm)
    if forbidden is not None:
        scores = scores.masked_fill(forbidden, -np.inf)
    weights = ops.softmax(scores, axis=-1)
    weights = ops.dropout(weights, dropout_p, training)
    return weights @ v


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
        b, t, d = x.shape
        # (B, T, 3D) -> (3, B, H, T, head_dim)
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = scaled_dot_product_attention(
            q, k, v, causal=self.causal, dropout_p=self.dropout, training=self.training
        )
        y = y.transpose(1, 2).reshape(b, t, d)  # merge heads
        return self.resid_dropout(self.proj(y))

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, causal={self.causal}"
