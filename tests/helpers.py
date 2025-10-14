"""Helpers for building float64 test inputs."""

from __future__ import annotations

import numpy as np

import tensorgrad as tg


def leaf(
    rng: np.random.Generator,
    *shape: int,
    positive: bool = False,
    away_from_zero: bool = False,
) -> tg.Tensor:
    """A float64 leaf tensor with standard-normal entries.

    ``positive`` maps entries to ``|x| + 0.5`` (for log / sqrt / division); ``away_from_zero``
    pushes entries at least 0.1 away from 0 so finite differences never straddle a kink.
    """
    data = np.asarray(rng.standard_normal(shape))
    if positive:
        data = np.abs(data) + 0.5
    if away_from_zero:
        data = data + np.where(data >= 0, 0.1, -0.1)
    return tg.Tensor(data, requires_grad=True)


def distinct(rng: np.random.Generator, *shape: int) -> tg.Tensor:
    """A float64 leaf whose entries are well separated (no near-ties for max / pooling)."""
    n = int(np.prod(shape))
    data = rng.permutation(n).reshape(shape) * 0.1 + rng.uniform(-0.01, 0.01, size=shape)
    return tg.Tensor(data, requires_grad=True)
