"""Fused building blocks used by :mod:`tensorgrad.nn`: linear, normalisation, embedding, dropout."""

from __future__ import annotations

import numpy as np

from tensorgrad._random import get_rng
from tensorgrad._types import Array
from tensorgrad.autograd import no_grad
from tensorgrad.ops import elementwise, reduce
from tensorgrad.ops._util import as_tensor, make_result
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["batch_norm", "dropout", "embedding", "layer_norm", "linear"]


def linear(x: TensorLike, weight: TensorLike, bias: TensorLike | None = None) -> Tensor:
    """``x @ weight.T + bias`` for ``x`` of shape ``(*, in)`` and ``weight`` ``(out, in)``.

    Fused into a single graph node (instead of transpose + matmul + add).
    """
    tx, tw = as_tensor(x), as_tensor(weight)
    tb = None if bias is None else as_tensor(bias)
    if tw.ndim != 2 or tx.ndim == 0 or tx.shape[-1] != tw.shape[1]:
        raise ValueError(f"linear: input {tx.shape} does not match weight {tw.shape}")
    out = tx.data @ tw.data.T
    if tb is not None:
        out = out + tb.data

    def backward(g: Array) -> tuple[Array | None, ...]:
        g2 = g.reshape(-1, g.shape[-1])
        gx = (g @ tw.data) if tx.requires_grad else None
        gw = (g2.T @ tx.data.reshape(-1, tx.shape[-1])) if tw.requires_grad else None
        if tb is None:
            return gx, gw
        return gx, gw, (g2.sum(axis=0) if tb.requires_grad else None)

    parents = (tx, tw) if tb is None else (tx, tw, tb)
    return make_result(out, parents, backward, "linear")


def layer_norm(
    x: TensorLike,
    weight: TensorLike | None = None,
    bias: TensorLike | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Normalise over the last axis to zero mean / unit variance, then scale and shift.

    Fused forward and backward. With ``xhat = (x - mu) * rstd`` and ``gh = g * weight``, the
    input gradient is ``rstd * (gh - mean(gh) - xhat * mean(gh * xhat))``.
    """
    tx = as_tensor(x)
    tw = None if weight is None else as_tensor(weight)
    tb = None if bias is None else as_tensor(bias)
    d = tx.shape[-1]
    mu = tx.data.mean(axis=-1, keepdims=True)
    centered = tx.data - mu
    rstd = 1.0 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    xhat = centered * rstd
    out = xhat if tw is None else xhat * tw.data
    if tb is not None:
        out = out + tb.data
    lead = tuple(range(tx.ndim - 1))

    def backward(g: Array) -> tuple[Array | None, ...]:
        grads: list[Array | None] = []
        gh = g if tw is None else g * tw.data
        if tx.requires_grad:
            mean_gh = gh.sum(axis=-1, keepdims=True) / d
            mean_ghx = (gh * xhat).sum(axis=-1, keepdims=True) / d
            grads.append(rstd * (gh - mean_gh - xhat * mean_ghx))
        else:
            grads.append(None)
        if tw is not None:
            grads.append((g * xhat).sum(axis=lead) if tw.requires_grad else None)
        if tb is not None:
            grads.append(g.sum(axis=lead) if tb.requires_grad else None)
        return tuple(grads)

    parents = tuple(t for t in (tx, tw, tb) if t is not None)
    return make_result(out.astype(tx.dtype, copy=False), parents, backward, "layer_norm")


def batch_norm(
    x: Tensor,
    running_mean: Array | None,
    running_var: Array | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    *,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Batch normalisation for ``(N, C)`` or ``(N, C, L)`` input (statistics per channel).

    In training mode the batch statistics are used and the running estimates (if given) are
    updated in place with ``momentum``, using the unbiased batch variance as PyTorch does. In
    evaluation mode the running estimates are used instead. Built from primitive ops, so the
    backward pass comes for free from the graph.
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"batch_norm expects (N, C) or (N, C, L) input, got {x.shape}")
    axes = (0,) if x.ndim == 2 else (0, 2)
    view = (1, -1) if x.ndim == 2 else (1, -1, 1)
    if training:
        mean = reduce.mean(x, axes, keepdims=True)
        var = reduce.var(x, axes, keepdims=True, correction=0)
        if running_mean is not None and running_var is not None:
            n = x.size // x.shape[1]
            with no_grad():
                unbiased = var.data.reshape(-1) * (n / max(n - 1, 1))
                running_mean *= 1 - momentum
                running_mean += momentum * mean.data.reshape(-1)
                running_var *= 1 - momentum
                running_var += momentum * unbiased
    else:
        if running_mean is None or running_var is None:
            raise ValueError("evaluation-mode batch_norm needs running statistics")
        mean = Tensor._wrap(running_mean.reshape(view).astype(x.dtype, copy=False))
        var = Tensor._wrap(running_var.reshape(view).astype(x.dtype, copy=False))
    out = (x - mean) / elementwise.sqrt(var + eps)
    if weight is not None:
        out = out * weight.reshape(view)
    if bias is not None:
        out = out + bias.reshape(view)
    return out


def embedding(indices: Tensor | Array, weight: TensorLike) -> Tensor:
    """Look up rows of ``weight`` (``(num_embeddings, dim)``) for integer ``indices``.

    The backward pass is a scatter-add: rows looked up several times accumulate gradient.
    """
    tw = as_tensor(weight)
    idx = np.asarray(indices.data if isinstance(indices, Tensor) else indices)
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"embedding indices must be integers, got {idx.dtype}")
    # NumPy would silently wrap negative ids around to the last rows.
    if idx.size and (idx.min() < 0 or idx.max() >= tw.shape[0]):
        raise IndexError(
            f"embedding ids must be in [0, {tw.shape[0]}), got {idx.min()}..{idx.max()}"
        )
    out = tw.data[idx]

    def backward(g: Array) -> tuple[Array]:
        grad = np.zeros_like(tw.data)
        np.add.at(grad, idx.reshape(-1), g.reshape(-1, tw.shape[1]))
        return (grad,)

    return make_result(out, (tw,), backward, "embedding")


def dropout(
    x: TensorLike, p: float = 0.5, training: bool = True, rng: np.random.Generator | None = None
) -> Tensor:
    """Inverted dropout: zero each element with probability ``p`` and scale the rest by
    ``1 / (1 - p)`` so the expected activation is unchanged; the identity when not training.
    """
    tx = as_tensor(x)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {p}")
    if not training or p == 0.0:
        return tx
    if p == 1.0:
        mask = np.zeros_like(tx.data)
    else:
        keep = (rng or get_rng()).random(tx.shape, dtype=np.float32) >= p
        mask = keep.astype(tx.dtype) / np.asarray(1.0 - p, dtype=tx.dtype)

    def backward(g: Array) -> tuple[Array]:
        return (g * mask,)

    return make_result(tx.data * mask, (tx,), backward, "dropout")
