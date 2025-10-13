"""Shape manipulation, indexing, joining and selection ops."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from tensorgrad._types import Array
from tensorgrad.ops._util import (
    as_tensor,
    coerce_pair,
    make_result,
    normalize_axis,
    unbroadcast,
)
from tensorgrad.tensor import IndexLike, Tensor, TensorLike

__all__ = [
    "concat",
    "flatten",
    "getitem",
    "masked_fill",
    "permute",
    "reshape",
    "squeeze",
    "stack",
    "transpose",
    "unsqueeze",
    "where",
]


def reshape(a: TensorLike, shape: Sequence[int]) -> Tensor:
    """Return the same elements with a new shape (one entry may be ``-1``)."""
    ta = as_tensor(a)

    def backward(g: Array) -> tuple[Array]:
        return (g.reshape(ta.shape),)

    return make_result(ta.data.reshape(tuple(shape)), (ta,), backward, "reshape")


def permute(a: TensorLike, dims: Sequence[int]) -> Tensor:
    """Reorder the axes of ``a`` so that output axis ``i`` is input axis ``dims[i]``."""
    ta = as_tensor(a)
    order = tuple(normalize_axis(d, ta.ndim) for d in dims)
    if sorted(order) != list(range(ta.ndim)):
        raise ValueError(f"dims {tuple(dims)} is not a permutation of {ta.ndim} axes")
    inverse = tuple(int(i) for i in np.argsort(order))

    def backward(g: Array) -> tuple[Array]:
        return (g.transpose(inverse),)

    return make_result(ta.data.transpose(order), (ta,), backward, "permute")


def transpose(a: TensorLike, axis0: int, axis1: int) -> Tensor:
    """Swap two axes of ``a``."""
    ta = as_tensor(a)
    order = list(range(ta.ndim))
    i, j = normalize_axis(axis0, ta.ndim), normalize_axis(axis1, ta.ndim)
    order[i], order[j] = order[j], order[i]
    return permute(ta, order)


def unsqueeze(a: TensorLike, axis: int) -> Tensor:
    """Insert a size-1 axis at position ``axis``."""
    ta = as_tensor(a)
    pos = normalize_axis(axis, ta.ndim + 1)
    return reshape(ta, (*ta.shape[:pos], 1, *ta.shape[pos:]))


def squeeze(a: TensorLike, axis: int | None = None) -> Tensor:
    """Remove size-1 axes (only ``axis`` if given; a no-op if that axis is not size 1)."""
    ta = as_tensor(a)
    if axis is None:
        shape = tuple(n for n in ta.shape if n != 1)
    else:
        ax = normalize_axis(axis, ta.ndim)
        shape = ta.shape if ta.shape[ax] != 1 else ta.shape[:ax] + ta.shape[ax + 1 :]
    return reshape(ta, shape)


def flatten(a: TensorLike, start_axis: int = 0, end_axis: int = -1) -> Tensor:
    """Merge axes ``start_axis`` through ``end_axis`` (inclusive) into one."""
    ta = as_tensor(a)
    if ta.ndim == 0:
        return reshape(ta, (1,))
    start, end = normalize_axis(start_axis, ta.ndim), normalize_axis(end_axis, ta.ndim)
    if start > end:
        raise ValueError("flatten: start_axis must not come after end_axis")
    merged = int(np.prod(ta.shape[start : end + 1]))
    return reshape(ta, (*ta.shape[:start], merged, *ta.shape[end + 1 :]))


def _normalize_index(index: IndexLike) -> tuple[tuple[Any, ...], bool]:
    """Convert tensors/lists/bool masks in an index to NumPy form; report advanced indexing."""
    items = index if isinstance(index, tuple) else (index,)
    out: list[Any] = []
    advanced = False
    for item in items:
        value: Any = item.data if isinstance(item, Tensor) else item
        if isinstance(value, list):
            value = np.asarray(value)
        if isinstance(value, np.ndarray):
            advanced = True
            if value.dtype == np.bool_:
                out.extend(np.nonzero(value))  # a[mask] == a[mask.nonzero()]
                continue
            if not np.issubdtype(value.dtype, np.integer):
                raise IndexError(f"index arrays must be integer or boolean, got {value.dtype}")
        out.append(value)
    return tuple(out), advanced


def getitem(a: TensorLike, index: IndexLike) -> Tensor:
    """NumPy-style indexing: basic slicing plus integer and boolean array indexing.

    The backward pass scatters the gradient back into a zero array; with advanced indexing
    the same element can be selected several times, so contributions are summed with
    :func:`numpy.add.at` (scatter-add) instead of plain assignment.
    """
    ta = as_tensor(a)
    idx, advanced = _normalize_index(index)
    out = ta.data[idx]

    def backward(g: Array) -> tuple[Array]:
        grad = np.zeros_like(ta.data)
        if advanced:
            np.add.at(grad, idx, g)
        else:
            grad[idx] = g
        return (grad,)

    return make_result(np.asarray(out), (ta,), backward, "getitem")


def concat(tensors: Sequence[TensorLike], axis: int = 0) -> Tensor:
    """Join tensors along an existing axis."""
    if not tensors:
        raise ValueError("concat needs at least one tensor")
    ts = [as_tensor(t) for t in tensors]
    ax = normalize_axis(axis, ts[0].ndim)
    boundaries = np.cumsum([t.shape[ax] for t in ts])[:-1]

    def backward(g: Array) -> tuple[Array | None, ...]:
        pieces = np.split(g, boundaries, axis=ax)
        return tuple(p if t.requires_grad else None for p, t in zip(pieces, ts, strict=True))

    data = np.concatenate([t.data for t in ts], axis=ax)
    return make_result(data, tuple(ts), backward, "concat")


def stack(tensors: Sequence[TensorLike], axis: int = 0) -> Tensor:
    """Join equally shaped tensors along a new axis."""
    if not tensors:
        raise ValueError("stack needs at least one tensor")
    ts = [as_tensor(t) for t in tensors]
    ax = normalize_axis(axis, ts[0].ndim + 1)

    def backward(g: Array) -> tuple[Array | None, ...]:
        return tuple(np.take(g, i, axis=ax) if t.requires_grad else None for i, t in enumerate(ts))

    data = np.stack([t.data for t in ts], axis=ax)
    return make_result(data, tuple(ts), backward, "stack")


def _mask_array(mask: Tensor | Array | bool) -> Array:
    arr = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
    return arr.astype(np.bool_, copy=False)


def where(condition: Tensor | Array | bool, a: TensorLike, b: TensorLike) -> Tensor:
    """Pick elements from ``a`` where ``condition`` is true and from ``b`` elsewhere."""
    cond = _mask_array(condition)
    ta, tb = coerce_pair(a, b)
    zero = np.zeros((), dtype=np.result_type(ta.dtype, tb.dtype))

    def backward(g: Array) -> tuple[Array | None, Array | None]:
        return (
            unbroadcast(np.where(cond, g, zero), ta.shape) if ta.requires_grad else None,
            unbroadcast(np.where(cond, zero, g), tb.shape) if tb.requires_grad else None,
        )

    return make_result(np.where(cond, ta.data, tb.data), (ta, tb), backward, "where")


def masked_fill(a: TensorLike, mask: Tensor | Array, value: float) -> Tensor:
    """Replace entries of ``a`` where ``mask`` (broadcastable to ``a``) is true by ``value``.

    Filled positions receive zero gradient. Filling with ``-inf`` before a softmax is how
    attention masks out forbidden positions.
    """
    ta = as_tensor(a)
    m = _mask_array(mask)
    out = np.where(m, np.asarray(value, dtype=ta.dtype), ta.data)

    def backward(g: Array) -> tuple[Array]:
        return (unbroadcast(np.where(m, np.zeros((), dtype=g.dtype), g), ta.shape),)

    return make_result(out, (ta,), backward, "masked_fill")
