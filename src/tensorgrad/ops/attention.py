"""Fused scaled-dot-product attention.

``softmax(q k^T / sqrt(d) + mask) v`` as a single graph node. Compared with composing
matmul, scaling, masking, softmax, dropout and matmul, the fused op keeps one ``(T_q, T_k)``
array per head instead of four, applies the mask and the softmax in place, and has a
hand-derived backward pass::

    dV = P^T dO,   dP = dO V^T,   dS = P * (dP - rowsum(dP * P)) / sqrt(d),
    dQ = dS K,     dK = dS^T Q

(with the dropout mask applied to ``P`` and ``dP`` when active). :func:`packed_self_attention`
takes the packed ``(B, T, 3D)`` output of a QKV projection and returns merged heads, so the
multi-head split and merge cost no extra graph nodes and no copies of Q, K and V.

The differentiable VJPs used under ``create_graph=True`` evaluate the same formulas with
tensor ops (softmax recomputed as a differentiable op), so higher-order gradients stay exact;
``attention_reference`` is the unfused composition the tests compare both against.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from tensorgrad._random import get_rng, keep_mask
from tensorgrad._types import Array
from tensorgrad.ops import activation, shape
from tensorgrad.ops._util import as_tensor, make_result, unbroadcast
from tensorgrad.profiler import profiled
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["causal_mask", "packed_self_attention", "scaled_dot_product_attention"]

Need = tuple[bool, ...]


@lru_cache(maxsize=32)
def causal_mask(q_len: int, k_len: int) -> Array:
    """Boolean ``(q_len, k_len)`` mask that is ``True`` where attention is *forbidden*.

    Query ``i`` may attend to keys ``0 .. i + (k_len - q_len)``; with equal lengths that is
    "no peeking at future positions". The cached array is read-only.
    """
    mask = np.triu(np.ones((q_len, k_len), dtype=np.bool_), k=1 + k_len - q_len)
    mask.flags.writeable = False
    return mask


def _forbidden(q_len: int, k_len: int, causal: bool, mask: Array | None) -> Array | None:
    if not causal:
        return None if mask is None else np.asarray(mask, dtype=np.bool_)
    cm = causal_mask(q_len, k_len)
    return cm if mask is None else (np.asarray(mask, dtype=np.bool_) | cm)


def _dropout_keep(
    shape_: tuple[int, ...], p: float, training: bool, rng: np.random.Generator | None
) -> tuple[Array | None, float]:
    """Keep-mask for dropout on the attention weights (``None`` when inactive) and its scale."""
    if not training or p == 0.0:
        return None, 1.0
    if not 0.0 <= p < 1.0:
        raise ValueError(f"attention dropout must be in [0, 1), got {p}")
    return keep_mask(shape_, p, rng or get_rng())


# ------------------------------------------------------------------ array kernels
def attention_forward(
    q: Array, k: Array, v: Array, forbidden: Array | None, keep: Array | None, keep_scale: float
) -> tuple[Array, Array]:
    """``(output, probabilities)`` of attention on arrays shaped ``(..., T, d)``."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = q @ np.swapaxes(k, -1, -2)
    s *= scale
    if forbidden is not None:
        np.copyto(s, -np.inf, where=forbidden)
    m = s.max(axis=-1, keepdims=True)
    # A row that is entirely masked would give inf - inf; shift it by 0 (it becomes NaN, as
    # in the unfused version and in PyTorch).
    np.copyto(m, 0.0, where=~np.isfinite(m))
    s -= m
    np.exp(s, out=s)
    s /= s.sum(axis=-1, keepdims=True)
    weights = s if keep is None else s * keep * keep_scale
    return weights @ v, s


def attention_backward(
    g: Array,
    q: Array,
    k: Array,
    v: Array,
    probs: Array,
    keep: Array | None,
    keep_scale: float,
    need: Need,
) -> tuple[Array | None, Array | None, Array | None]:
    """``(dQ, dK, dV)`` for the upstream gradient ``g`` of the attention output."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    weights = probs if keep is None else probs * keep * keep_scale
    dv = np.swapaxes(weights, -1, -2) @ g if need[2] else None
    if not (need[0] or need[1]):
        return None, None, dv
    ds = g @ np.swapaxes(v, -1, -2)
    if keep is not None:
        ds *= keep
        ds *= keep_scale
    ds -= (ds * probs).sum(axis=-1, keepdims=True)
    ds *= probs
    ds *= scale
    dq = ds @ k if need[0] else None
    dk = np.swapaxes(ds, -1, -2) @ q if need[1] else None
    return dq, dk, dv


def _probabilities(q: Tensor, k: Tensor, forbidden: Array | None) -> Tensor:
    scores = (q @ shape.transpose(k, -1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    if forbidden is not None:
        scores = shape.masked_fill(scores, forbidden, -np.inf)
    return activation.softmax(scores, axis=-1)


def attention_reference(
    q: Tensor, k: Tensor, v: Tensor, forbidden: Array | None, keep: Array | None, keep_scale: float
) -> Tensor:
    """The same function composed of primitive differentiable ops."""
    weights = _probabilities(q, k, forbidden)
    if keep is not None:
        weights = weights * (keep * keep_scale)
    return weights @ v


def attention_backward_graph(
    g: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    forbidden: Array | None,
    keep: Array | None,
    keep_scale: float,
    need: Need,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """:func:`attention_backward` written with differentiable tensor ops."""
    probs = _probabilities(q, k, forbidden)
    drop = None if keep is None else keep * keep_scale
    weights = probs if drop is None else probs * drop
    dv = shape.transpose(weights, -1, -2) @ g if need[2] else None
    if not (need[0] or need[1]):
        return None, None, dv
    dp = g @ shape.transpose(v, -1, -2)
    if drop is not None:
        dp = dp * drop
    ds = probs * (dp - (dp * probs).sum(axis=-1, keepdims=True)) * (1.0 / math.sqrt(q.shape[-1]))
    dq = ds @ k if need[0] else None
    dk = shape.transpose(ds, -1, -2) @ q if need[1] else None
    return dq, dk, dv


# ------------------------------------------------------------------ differentiable ops
@profiled("attention")
def scaled_dot_product_attention(
    q: TensorLike,
    k: TensorLike,
    v: TensorLike,
    *,
    causal: bool = False,
    mask: Array | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> Tensor:
    """``softmax(q k^T / sqrt(d) + mask) v`` over the last two axes, fused.

    Args:
        q: Queries ``(..., T_q, d)``.
        k: Keys ``(..., T_k, d)``.
        v: Values ``(..., T_k, d_v)``.
        causal: Forbid attention to future positions.
        mask: Extra boolean mask (broadcastable to ``(..., T_q, T_k)``), ``True`` = forbidden.
        dropout_p: Dropout on the attention weights (training only).
        training: Whether dropout is active.
        rng: Generator for the dropout mask (default: the global one).
    """
    tq, tk, tv = as_tensor(q), as_tensor(k), as_tensor(v)
    forbidden = _forbidden(tq.shape[-2], tk.shape[-2], causal, mask)
    batch = np.broadcast_shapes(tq.shape[:-2], tk.shape[:-2])
    score_shape = (*batch, tq.shape[-2], tk.shape[-2])
    keep, keep_scale = _dropout_keep(score_shape, dropout_p, training, rng)
    out, probs = attention_forward(tq.data, tk.data, tv.data, forbidden, keep, keep_scale)

    def backward(g: Array, need: Need) -> tuple[Array | None, ...]:
        grads = attention_backward(g, tq.data, tk.data, tv.data, probs, keep, keep_scale, need)
        return tuple(
            None if gr is None else _sum_like(gr, t.shape)
            for gr, t in zip(grads, (tq, tk, tv), strict=True)
        )

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, ...]:
        grads = attention_backward_graph(g, tq, tk, tv, forbidden, keep, keep_scale, need)
        return tuple(
            None if gr is None else shape.sum_to(gr, t.shape)
            for gr, t in zip(grads, (tq, tk, tv), strict=True)
        )

    return make_result(out, (tq, tk, tv), backward, "attention", graph=graph)


def _sum_like(grad: Array, target: tuple[int, ...]) -> Array:
    """Undo batch-axis broadcasting of an input (e.g. keys shared across heads)."""
    return grad if grad.shape == target else unbroadcast(grad, target)


def _split_heads(qkv: Array, num_heads: int) -> tuple[Array, Array, Array]:
    """Views ``(B, H, T, hd)`` of Q, K and V inside a packed ``(B, T, 3 * H * hd)`` array."""
    b, t, d3 = qkv.shape
    parts = qkv.reshape(b, t, 3, num_heads, d3 // (3 * num_heads))
    return tuple(parts[:, :, i].transpose(0, 2, 1, 3) for i in range(3))  # type: ignore[return-value]


@profiled("self_attention")
def packed_self_attention(
    qkv: TensorLike,
    num_heads: int,
    *,
    causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> Tensor:
    """Multi-head self-attention on a packed QKV projection.

    ``qkv`` has shape ``(B, T, 3 * D)``: queries, keys and values side by side, each split
    into ``num_heads`` heads of size ``D / num_heads`` (the layout of one ``Linear(D, 3 D)``).
    Returns the attention output with the heads merged back, shape ``(B, T, D)``. Q, K and V
    are strided views of the input, and the backward pass writes their gradients straight
    into one ``(B, T, 3 * D)`` array.
    """
    tqkv = as_tensor(qkv)
    if tqkv.ndim != 3 or tqkv.shape[-1] % (3 * num_heads):
        raise ValueError(
            f"expected (B, T, 3 * D) with D divisible by {num_heads}, got {tqkv.shape}"
        )
    b, t, d3 = tqkv.shape
    d = d3 // 3
    forbidden = _forbidden(t, t, causal, None)
    keep, keep_scale = _dropout_keep((b, num_heads, t, t), dropout_p, training, rng)
    q, k, v = _split_heads(tqkv.data, num_heads)
    y, probs = attention_forward(q, k, v, forbidden, keep, keep_scale)
    out = y.transpose(0, 2, 1, 3).reshape(b, t, d)

    def backward(g: Array, need: Need) -> tuple[Array]:
        gy = g.reshape(b, t, num_heads, d // num_heads).transpose(0, 2, 1, 3)
        grads = attention_backward(gy, q, k, v, probs, keep, keep_scale, (True, True, True))
        dqkv = np.empty((b, t, 3, num_heads, d // num_heads), dtype=np.result_type(*grads))
        for i, gr in enumerate(grads):
            dqkv[:, :, i] = gr.transpose(0, 2, 1, 3)  # type: ignore[union-attr]
        return (dqkv.reshape(b, t, d3),)

    def graph(g: Tensor, y_: Tensor, need: Need) -> tuple[Tensor]:
        parts = shape.reshape(tqkv, (b, t, 3, num_heads, d // num_heads))
        heads: list[Any] = [shape.permute(parts[:, :, i], (0, 2, 1, 3)) for i in range(3)]
        gy = shape.permute(shape.reshape(g, (b, t, num_heads, d // num_heads)), (0, 2, 1, 3))
        grads = attention_backward_graph(
            gy, heads[0], heads[1], heads[2], forbidden, keep, keep_scale, (True, True, True)
        )
        merged = shape.stack([shape.permute(gr, (0, 2, 1, 3)) for gr in grads if gr is not None], 2)
        return (shape.reshape(merged, (b, t, d3)),)

    return make_result(out, (tqkv,), backward, "self_attention", graph=graph)
