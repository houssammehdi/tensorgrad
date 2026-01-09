"""Global random-number state.

All randomness inside tensorgrad (parameter initialisation, dropout masks, random tensor
factories) draws from a single :class:`numpy.random.Generator`, so one call to
:func:`manual_seed` makes a whole training run reproducible.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator

import numpy as np

from tensorgrad._types import Array


class _RngState:
    """Holder for the process-wide generator (a class avoids ``global`` statements)."""

    generator: np.random.Generator = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Re-seed the global generator used for initialisation, dropout and random factories."""
    _RngState.generator = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """Return the global generator."""
    return _RngState.generator


@contextlib.contextmanager
def replaying(generator: np.random.Generator) -> Iterator[None]:
    """Make ``generator`` the global generator inside the block, then restore the previous one.

    Gradient checkpointing uses it to repeat the forward pass's random draws.
    """
    previous = _RngState.generator
    _RngState.generator = generator
    try:
        yield
    finally:
        _RngState.generator = previous


#: Bit generators whose raw output words carry 64 uniform bits.
_WIDE_GENERATORS = (np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64)


def keep_mask(shape: tuple[int, ...], p: float, rng: np.random.Generator) -> tuple[Array, float]:
    """Dropout keep-mask for drop probability ``p``, and the matching inverse scale.

    The drop probability is quantised to ``p' = round(p * 2**16) / 2**16`` (within
    ``2**-17`` of ``p``): the mask compares uniform 16-bit words with a threshold, which is
    about three times faster than drawing floats. Scaling survivors by exactly
    ``1 / (1 - p')`` keeps inverted dropout unbiased.
    """
    threshold = round(p * 65536)
    if threshold >= 65536:
        return np.zeros(shape, dtype=np.bool_), 0.0
    n = math.prod(shape)
    if isinstance(rng.bit_generator, _WIDE_GENERATORS):
        words = rng.bit_generator.random_raw((n + 3) // 4).view(np.uint16)[:n]
    else:
        words = np.frombuffer(rng.bytes(2 * n), dtype=np.uint16)
    keep: Array = (words >= threshold).reshape(shape)
    return keep, 65536.0 / (65536 - threshold)
