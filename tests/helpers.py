"""Helpers for building float64 test inputs and checking their derivatives."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

import tensorgrad as tg
from tensorgrad.utils import gradcheck, gradgradcheck


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


def check_gradients(
    fn: Callable[..., tg.Tensor],
    inputs: Sequence[tg.Tensor],
    *,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> bool:
    """First- and second-order finite-difference checks of ``fn`` at ``inputs``.

    :func:`gradcheck` verifies the ordinary backward pass; :func:`gradgradcheck` verifies that
    the ``create_graph`` VJPs agree with it and that their own derivatives (double backward,
    with respect to the inputs and to the upstream gradient) match finite differences.
    """
    assert gradcheck(fn, inputs, eps=eps, atol=atol, rtol=rtol)
    assert gradgradcheck(fn, inputs, eps=eps, atol=max(atol, 1e-5), rtol=rtol)
    return True
