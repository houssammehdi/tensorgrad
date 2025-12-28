"""Fused building blocks used by :mod:`tensorgrad.nn`: linear, normalisation, embedding, dropout."""

from __future__ import annotations

import numpy as np

from tensorgrad._random import get_rng, keep_mask
from tensorgrad._types import Array
from tensorgrad.autograd import no_grad
from tensorgrad.ops import elementwise, reduce
from tensorgrad.ops._util import as_tensor, make_result
from tensorgrad.ops.linalg import matmul
from tensorgrad.ops.shape import _scatter_add, reshape, transpose
from tensorgrad.profiler import profiled
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["batch_norm", "dropout", "embedding", "layer_norm", "linear"]

Need = tuple[bool, ...]


@profiled("linear")
def linear(x: TensorLike, weight: TensorLike, bias: TensorLike | None = None) -> Tensor:
    """``x @ weight.T + bias`` for ``x`` of shape ``(*, in)`` and ``weight`` ``(out, in)``.

    Fused into a single graph node (instead of transpose + matmul + add).
    """
    tx, tw = as_tensor(x), as_tensor(weight)
    tb = None if bias is None else as_tensor(bias)
    if tw.ndim != 2 or tx.ndim == 0 or tx.shape[-1] != tw.shape[1]:
        raise ValueError(f"linear: input {tx.shape} does not match weight {tw.shape}")
    # Flatten the leading axes so the product is one large GEMM: NumPy would otherwise loop
    # over them and call BLAS once per (T, in) slice.
    x2 = tx.data.reshape(-1, tw.shape[1])
    out = x2 @ tw.data.T
    if tb is not None:
        if np.result_type(out, tb.data) == out.dtype:
            out += tb.data  # ``out`` is fresh from the matmul: safe to update in place
        else:
            out = out + tb.data
    out = out.reshape(*tx.shape[:-1], tw.shape[0])

    def backward(g: Array, need: Need) -> tuple[Array | None, ...]:
        g2 = g.reshape(-1, g.shape[-1])
        gx = (g2 @ tw.data).reshape(tx.shape) if need[0] else None
        gw = (g2.T @ x2) if need[1] else None
        if tb is None:
            return gx, gw
        return gx, gw, (g2.sum(axis=0) if need[2] else None)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, ...]:
        g2 = reshape(g, (-1, g.shape[-1]))
        gx = matmul(g, tw) if need[0] else None
        gw = None
        if need[1]:
            gw = matmul(transpose(g2, 0, 1), reshape(tx, (-1, tx.shape[-1])))
        if tb is None:
            return gx, gw
        return gx, gw, (g2.sum(axis=0) if need[2] else None)

    parents = (tx, tw) if tb is None else (tx, tw, tb)
    return make_result(out, parents, backward, "linear", graph=graph)


@profiled("layer_norm")
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
    data = tx.data if tx.dtype.kind == "f" else tx.data.astype(np.float64)
    # In place on two buffers: the centred input becomes xhat, the squares become the output.
    xhat = data - data.mean(axis=-1, keepdims=True)
    scratch = np.multiply(xhat, xhat, out=np.empty_like(xhat))
    rstd = 1.0 / np.sqrt(scratch.mean(axis=-1, keepdims=True) + eps)
    xhat *= rstd
    if tw is None and tb is None:
        out = xhat
    else:
        out = scratch
        if tw is None:
            out[...] = xhat
        else:
            np.multiply(xhat, tw.data, out=out)
        if tb is not None:
            out += tb.data
    lead = tuple(range(tx.ndim - 1))

    def backward(g: Array, need: Need) -> tuple[Array | None, ...]:
        tmp = np.empty(g.shape, dtype=np.result_type(g, xhat))
        k = 1  # parents are (x, [weight], [bias]); need[k] follows that order
        grads: list[Array | None] = [None]
        if tw is not None:
            if need[k]:
                grads.append(np.multiply(g, xhat, out=tmp).sum(axis=lead))
            else:
                grads.append(None)
            k += 1
        if tb is not None:
            grads.append(g.sum(axis=lead) if need[k] else None)
        if need[0]:
            gh = g * tw.data if tw is not None else np.array(g, dtype=tmp.dtype, copy=True)
            mean_ghx = np.multiply(gh, xhat, out=tmp).mean(axis=-1, keepdims=True)
            gh -= gh.mean(axis=-1, keepdims=True)
            gh -= np.multiply(xhat, mean_ghx, out=tmp)
            gh *= rstd
            grads[0] = gh
        return tuple(grads)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, ...]:
        # The fused formula again, with the statistics recomputed as differentiable ops.
        c = tx - tx.mean(axis=-1, keepdims=True)
        r = ((c * c).mean(axis=-1, keepdims=True) + eps) ** -0.5
        xh = c * r
        grads: list[Tensor | None] = []
        if need[0]:
            gh = g if tw is None else g * tw
            mean_gh = gh.mean(axis=-1, keepdims=True)
            mean_ghx = (gh * xh).mean(axis=-1, keepdims=True)
            grads.append(r * (gh - mean_gh - xh * mean_ghx))
        else:
            grads.append(None)
        k = 1
        if tw is not None:
            grads.append((g * xh).sum(axis=lead) if need[k] else None)
            k += 1
        if tb is not None:
            grads.append(g.sum(axis=lead) if need[k] else None)
        return tuple(grads)

    parents = tuple(t for t in (tx, tw, tb) if t is not None)
    return make_result(out, parents, backward, "layer_norm", graph=graph)


@profiled("batch_norm")
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


def _sum_rows(ids: Array, rows: Array, num: int) -> Array:
    """``out[ids[i]] += rows[i]`` for every ``i`` (``numpy.add.at`` semantics).

    Implemented as a stable sort of the ids followed by ``numpy.add.reduceat`` over each run
    of equal ids: 7x faster than ``numpy.add.at`` for a character vocabulary and about 2x for
    a 50,000-token one.
    """
    out = np.zeros((num, *rows.shape[1:]), dtype=rows.dtype)
    if ids.size == 0:
        return out
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1])))
    out[sorted_ids[starts]] = np.add.reduceat(rows[order], starts, axis=0)
    return out


@profiled("embedding")
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

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (_sum_rows(idx.reshape(-1), g.reshape(-1, tw.shape[1]), tw.shape[0]),)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        return (_scatter_add(g, (idx,), True, tw.shape),)

    return make_result(out, (tw,), backward, "embedding", graph=graph)


@profiled("dropout")
def dropout(
    x: TensorLike, p: float = 0.5, training: bool = True, rng: np.random.Generator | None = None
) -> Tensor:
    """Inverted dropout: zero each element with probability ``p`` and scale the rest by
    ``1 / (1 - p)`` so the expected activation is unchanged; the identity when not training.

    ``p`` is quantised to a multiple of ``2**-16`` and the scale matches it exactly (see
    :func:`tensorgrad._random.keep_mask`).
    """
    tx = as_tensor(x)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {p}")
    if not training or p == 0.0:
        return tx
    keep, scale = keep_mask(tx.shape, p, rng or get_rng())
    mask = keep * np.asarray(scale, dtype=tx.dtype)

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (g * mask,)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        return (g * mask,)

    return make_result(tx.data * mask, (tx,), backward, "dropout", graph=graph)
