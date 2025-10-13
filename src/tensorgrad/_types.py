"""Shared type aliases used across the package."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

#: Any NumPy array. Tensors are thin wrappers around one of these.
Array: TypeAlias = npt.NDArray[Any]

#: Axis specification accepted by reductions: one axis, several axes, or all axes (``None``).
Axis: TypeAlias = int | tuple[int, ...] | None

#: Shape specification accepted by factory functions.
ShapeLike: TypeAlias = int | tuple[int, ...] | list[int]

#: Python scalars that ops accept in place of a tensor operand.
Scalar: TypeAlias = int | float

DTypeLike: TypeAlias = npt.DTypeLike

FloatDType: TypeAlias = type[np.floating[Any]]
