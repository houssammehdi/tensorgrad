"""Global random-number state.

All randomness inside tensorgrad (parameter initialisation, dropout masks, random tensor
factories) draws from a single :class:`numpy.random.Generator`, so one call to
:func:`manual_seed` makes a whole training run reproducible.
"""

from __future__ import annotations

import numpy as np


class _RngState:
    """Holder for the process-wide generator (a class avoids ``global`` statements)."""

    generator: np.random.Generator = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Re-seed the global generator used for initialisation, dropout and random factories."""
    _RngState.generator = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """Return the global generator."""
    return _RngState.generator
