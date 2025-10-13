"""Helpers shared by the op implementations."""

from __future__ import annotations

import numpy as np

from tensorgrad._types import Array, Axis
from tensorgrad.autograd import BackwardFn, Node, is_grad_enabled
from tensorgrad.tensor import Tensor, TensorLike


def as_tensor(x: TensorLike, like: Tensor | None = None) -> Tensor:
    """Return ``x`` as a tensor; Python scalars adopt the floating dtype of ``like``."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, int | float) and like is not None and np.issubdtype(like.dtype, np.floating):
        return Tensor._wrap(np.asarray(x, dtype=like.dtype))
    return Tensor(x)


def coerce_pair(a: TensorLike, b: TensorLike) -> tuple[Tensor, Tensor]:
    """Convert both operands of a binary op, letting scalars follow the tensor operand."""
    ta = a if isinstance(a, Tensor) else None
    tb = b if isinstance(b, Tensor) else None
    return as_tensor(a, tb), as_tensor(b, ta)


def make_result(data: Array, parents: tuple[Tensor, ...], backward: BackwardFn, op: str) -> Tensor:
    """Wrap an op's output, recording a graph node if any input requires grad."""
    out = Tensor._wrap(data)
    if is_grad_enabled() and any(p.requires_grad for p in parents):
        out._requires_grad = True
        out._node = Node(op, parents, backward)
    return out


def unbroadcast(grad: Array, shape: tuple[int, ...]) -> Array:
    """Sum ``grad`` down to ``shape``: the adjoint of NumPy broadcasting.

    Broadcasting prepends axes and stretches size-1 axes; the gradient of a broadcast input
    is therefore the output gradient summed over every such axis.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


def normalize_axes(axis: Axis, ndim: int) -> tuple[int, ...]:
    """Turn an axis specification into a sorted tuple of non-negative axes."""
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    out = []
    for ax in axes:
        if not -ndim <= ax < max(ndim, 1):
            raise np.exceptions.AxisError(ax, ndim)
        out.append(ax % ndim if ndim else 0)
    return tuple(sorted(set(out)))


def normalize_axis(axis: int, ndim: int) -> int:
    """Turn a single (possibly negative) axis into a non-negative one."""
    if not -ndim <= axis < ndim:
        raise np.exceptions.AxisError(axis, ndim)
    return axis % ndim


def expand_reduced(
    grad: Array, shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool
) -> Array:
    """Re-insert reduced axes into ``grad`` (if dropped) and broadcast it back to ``shape``."""
    if not keepdims:
        grad = np.expand_dims(grad, axes)
    return np.broadcast_to(grad, shape)
