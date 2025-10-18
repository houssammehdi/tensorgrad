"""In-place parameter initialisers (all draw from the global generator)."""

from __future__ import annotations

import math
from typing import TypeVar

from tensorgrad._random import get_rng
from tensorgrad.tensor import Tensor

T = TypeVar("T", bound=Tensor)

__all__ = [
    "calculate_fans",
    "kaiming_uniform_",
    "normal_",
    "ones_",
    "uniform_",
    "xavier_uniform_",
    "zeros_",
]


def calculate_fans(t: Tensor) -> tuple[int, int]:
    """``(fan_in, fan_out)`` of a ``(out, in, *kernel)`` weight."""
    if t.ndim < 2:
        raise ValueError("fans are defined for tensors with at least 2 dimensions")
    receptive = math.prod(t.shape[2:])
    return t.shape[1] * receptive, t.shape[0] * receptive


def uniform_(t: T, low: float = 0.0, high: float = 1.0) -> T:
    """Fill ``t`` with samples from ``U(low, high)``."""
    t.data[...] = get_rng().uniform(low, high, size=t.shape)
    return t


def normal_(t: T, mean: float = 0.0, std: float = 1.0) -> T:
    """Fill ``t`` with samples from ``N(mean, std**2)``."""
    t.data[...] = get_rng().normal(mean, std, size=t.shape)
    return t


def zeros_(t: T) -> T:
    """Fill ``t`` with zeros."""
    t.data[...] = 0
    return t


def ones_(t: T) -> T:
    """Fill ``t`` with ones."""
    t.data[...] = 1
    return t


def xavier_uniform_(t: T, gain: float = 1.0) -> T:
    """Glorot/Xavier uniform: ``U(-a, a)`` with ``a = gain * sqrt(6 / (fan_in + fan_out))``."""
    fan_in, fan_out = calculate_fans(t)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(t, -bound, bound)


def kaiming_uniform_(t: T, nonlinearity_gain: float = math.sqrt(2.0)) -> T:
    """He/Kaiming uniform: ``U(-b, b)`` with ``b = gain * sqrt(3 / fan_in)``."""
    fan_in, _ = calculate_fans(t)
    bound = nonlinearity_gain * math.sqrt(3.0 / fan_in)
    return uniform_(t, -bound, bound)
