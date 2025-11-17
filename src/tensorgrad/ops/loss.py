"""Loss functions."""

from __future__ import annotations

from typing import Literal

import numpy as np

from tensorgrad._types import Array
from tensorgrad.ops import reduce
from tensorgrad.ops._util import as_tensor, coerce_pair, make_result
from tensorgrad.ops.activation import _log_softmax_array, softmax
from tensorgrad.ops.shape import reshape
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["Reduction", "cross_entropy", "mse_loss"]

Reduction = Literal["mean", "sum", "none"]


def cross_entropy(
    logits: TensorLike,
    target: Tensor | Array,
    *,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
) -> Tensor:
    """Softmax cross-entropy computed directly from unnormalised logits.

    Args:
        logits: Shape ``(*, C)`` -- the class axis is **last** (convenient for sequence
            models, where logits are ``(batch, time, vocab)``).
        target: Integer class indices of shape ``(*)``.
        ignore_index: Targets equal to this value contribute neither loss nor gradient.
        reduction: ``"mean"`` averages over non-ignored targets (0 if every target is
            ignored), ``"sum"`` adds them, ``"none"`` returns the per-element losses.

    The forward pass uses log-softmax (never ``log(softmax(x))``) and the backward pass the
    fused gradient ``softmax(x) - one_hot(target)``.
    """
    tl = as_tensor(logits)
    t = np.asarray(target.data if isinstance(target, Tensor) else target)
    if not np.issubdtype(t.dtype, np.integer):
        raise TypeError(f"cross_entropy targets must be integers, got {t.dtype}")
    if tl.shape[:-1] != t.shape:
        raise ValueError(f"logits {tl.shape} and targets {t.shape} are incompatible")
    num_classes = tl.shape[-1]
    flat_logits = tl.data.reshape(-1, num_classes)
    flat_t = t.reshape(-1)
    valid = flat_t != ignore_index
    safe_t = np.where(valid, flat_t, 0)
    if np.any((safe_t < 0) | (safe_t >= num_classes)):
        raise IndexError(f"target out of range for {num_classes} classes")
    logp = _log_softmax_array(flat_logits, axis=1)
    rows = np.arange(flat_t.size)
    losses = np.where(valid, -logp[rows, safe_t], 0).astype(tl.dtype)
    n_valid = int(valid.sum())
    if reduction == "mean":
        out: Array = np.asarray(losses.sum() / max(n_valid, 1), dtype=tl.dtype)
    elif reduction == "sum":
        out = np.asarray(losses.sum(), dtype=tl.dtype)
    elif reduction == "none":
        out = losses.reshape(t.shape)
    else:
        raise ValueError(f"unknown reduction {reduction!r}")

    def backward(g: Array, need: tuple[bool, ...]) -> tuple[Array]:
        grad = np.exp(logp)
        grad[rows, safe_t] -= 1
        grad *= valid[:, None]
        if reduction == "none":
            grad *= g.reshape(-1, 1)
        elif reduction == "mean":
            grad *= g / max(n_valid, 1)
        else:
            grad *= g
        return (grad.reshape(tl.shape).astype(tl.dtype, copy=False),)

    def graph(g: Tensor, y: Tensor, need: tuple[bool, ...]) -> tuple[Tensor]:
        # The same fused gradient, with softmax(logits) recomputed as a differentiable op
        # (the Hessian of cross-entropy is diag(p) - p p^T per row).
        onehot = np.zeros((rows.size, num_classes), dtype=tl.dtype)
        onehot[rows, safe_t] = 1
        weight = valid[:, None].astype(tl.dtype)
        grad = (softmax(reshape(tl, (-1, num_classes)), axis=1) - onehot) * weight
        if reduction == "none":
            grad = grad * reshape(g, (-1, 1))
        elif reduction == "mean":
            grad = grad * (g / max(n_valid, 1))
        else:
            grad = grad * g
        return (reshape(grad, tl.shape),)

    return make_result(out, (tl,), backward, "cross_entropy", graph=graph)


def mse_loss(
    prediction: TensorLike, target: TensorLike, *, reduction: Reduction = "mean"
) -> Tensor:
    """Squared error ``(prediction - target)**2``, averaged, summed or returned elementwise."""
    tp, tt = coerce_pair(prediction, target)
    sq = (tp - tt) ** 2
    if reduction == "mean":
        return reduce.mean(sq)
    if reduction == "sum":
        return reduce.sum(sq)
    if reduction == "none":
        return sq
    raise ValueError(f"unknown reduction {reduction!r}")
