"""Matrix multiplication."""

from __future__ import annotations

import numpy as np

from tensorgrad._types import Array
from tensorgrad.ops._util import coerce_pair, make_result, unbroadcast
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["matmul"]


def matmul(a: TensorLike, b: TensorLike) -> Tensor:
    """Matrix product with :func:`numpy.matmul` semantics.

    Leading (batch) dimensions broadcast; 1-D operands are treated as a row vector (``a``) or
    column vector (``b``) whose extra axis is removed from the result.
    """
    ta, tb = coerce_pair(a, b)
    if ta.ndim == 0 or tb.ndim == 0:
        raise ValueError("matmul does not accept 0-d tensors; use mul instead")
    a2 = ta.data[None, :] if ta.ndim == 1 else ta.data
    b2 = tb.data[:, None] if tb.ndim == 1 else tb.data
    out = a2 @ b2
    if tb.ndim == 1:
        out = out[..., 0]
    if ta.ndim == 1:
        out = out[..., 0, :] if tb.ndim > 1 else out[..., 0]

    def backward(g: Array, need: tuple[bool, ...]) -> tuple[Array | None, Array | None]:
        g2 = g
        if tb.ndim == 1:
            g2 = g2[..., None]
        if ta.ndim == 1:
            g2 = np.expand_dims(g2, -2)
        ga = gb = None
        if need[0]:
            ga = unbroadcast(g2 @ np.swapaxes(b2, -1, -2), a2.shape).reshape(ta.shape)
        if need[1]:
            gb = unbroadcast(np.swapaxes(a2, -1, -2) @ g2, b2.shape).reshape(tb.shape)
        return ga, gb

    return make_result(np.asarray(out), (ta, tb), backward, "matmul")
