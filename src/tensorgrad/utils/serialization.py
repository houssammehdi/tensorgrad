"""Checkpointing state dicts with :func:`numpy.savez` (no pickle involved)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from tensorgrad._types import Array

if TYPE_CHECKING:
    from tensorgrad.nn.module import Module

__all__ = ["load", "save"]


def save(obj: Module | Mapping[str, Array], path: str | os.PathLike[str]) -> None:
    """Write a module's ``state_dict()`` (or any name -> array mapping) to ``path``.

    The file is a standard ``.npz`` archive written to exactly ``path`` (NumPy's automatic
    ``.npz`` suffix is not added).
    """
    state = obj if isinstance(obj, Mapping) else obj.state_dict()
    arrays = {name: np.asarray(value) for name, value in state.items()}
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)  # type: ignore[arg-type]


def load(path: str | os.PathLike[str]) -> dict[str, Array]:
    """Read a checkpoint written by :func:`save` into a plain ``dict`` of arrays."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
