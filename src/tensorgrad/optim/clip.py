"""Gradient clipping."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from tensorgrad.tensor import Tensor

__all__ = ["clip_grad_norm_"]


def clip_grad_norm_(params: Iterable[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    """Rescale gradients in place so their global L2 norm is at most ``max_norm``.

    The norm is taken over all gradients concatenated, so the update *direction* is kept.

    Returns:
        The total norm before clipping.
    """
    with_grad = [(p, p.grad.data) for p in params if p.grad is not None]
    total = math.sqrt(
        math.fsum(float(np.sum(np.square(g, dtype=np.float64))) for _, g in with_grad)
    )
    scale = max_norm / (total + eps)
    if scale < 1.0:
        for p, g in with_grad:
            p.grad = g * np.asarray(scale, dtype=g.dtype)
    return total
