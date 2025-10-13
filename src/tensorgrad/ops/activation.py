"""Softmax and log-softmax, both fused and numerically stable."""

from __future__ import annotations

import numpy as np

from tensorgrad._types import Array
from tensorgrad.ops._util import as_tensor, make_result, normalize_axis
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["log_softmax", "logsumexp", "softmax"]


def _shifted(x: Array, axis: int) -> Array:
    # Subtracting the row max leaves softmax unchanged but keeps exp() from overflowing.
    # Rows that are entirely -inf would give inf - inf = nan; shift them by 0 instead.
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0)
    return x - m


def softmax(a: TensorLike, axis: int = -1) -> Tensor:
    """``exp(x) / sum(exp(x))`` along ``axis``.

    Backward uses the closed form ``s * (g - sum(g * s))`` rather than materialising the
    per-row Jacobian ``diag(s) - s s^T``.
    """
    ta = as_tensor(a)
    ax = normalize_axis(axis, ta.ndim)
    e = np.exp(_shifted(ta.data, ax))
    s = e / np.sum(e, axis=ax, keepdims=True)

    def backward(g: Array) -> tuple[Array]:
        return (s * (g - np.sum(g * s, axis=ax, keepdims=True)),)

    return make_result(s, (ta,), backward, "softmax")


def _log_softmax_array(x: Array, axis: int) -> Array:
    shifted = _shifted(x, axis)
    lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out: Array = shifted - lse
    return out


def log_softmax(a: TensorLike, axis: int = -1) -> Tensor:
    """``x - logsumexp(x)`` along ``axis``, computed with the max-shift trick.

    Never evaluates ``log(softmax(x))`` directly, which underflows to ``log(0) = -inf`` for
    strongly negative logits.
    """
    ta = as_tensor(a)
    ax = normalize_axis(axis, ta.ndim)
    out = _log_softmax_array(ta.data, ax)

    def backward(g: Array) -> tuple[Array]:
        return (g - np.exp(out) * np.sum(g, axis=ax, keepdims=True),)

    return make_result(out, (ta,), backward, "log_softmax")


def logsumexp(a: TensorLike, axis: int = -1, keepdims: bool = False) -> Tensor:
    """Stable ``log(sum(exp(x)))`` along ``axis``; its gradient is ``softmax(x)``."""
    ta = as_tensor(a)
    ax = normalize_axis(axis, ta.ndim)
    m = np.max(ta.data, axis=ax, keepdims=True)
    m = np.where(np.isfinite(m), m, 0)
    kept = np.log(np.sum(np.exp(ta.data - m), axis=ax, keepdims=True)) + m
    out = kept if keepdims else np.squeeze(kept, axis=ax)

    def backward(g: Array) -> tuple[Array]:
        g_kept = g if keepdims else np.expand_dims(g, ax)
        return (g_kept * np.exp(ta.data - kept),)

    return make_result(out, (ta,), backward, "logsumexp")
