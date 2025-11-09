"""Reductions: sum, mean, max, min and var over arbitrary axes."""

from __future__ import annotations

import builtins
from collections.abc import Callable

import numpy as np

from tensorgrad._types import Array, Axis
from tensorgrad.ops._util import as_tensor, expand_reduced, make_result, normalize_axes
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["max", "mean", "min", "sum", "var"]


def _drop_axes(kept: Array, axes: tuple[int, ...], keepdims: bool) -> Array:
    """Squeeze reduced axes unless ``keepdims`` (reductions always run with keepdims=True)."""
    return kept if keepdims else np.squeeze(kept, axis=axes)


def sum(a: TensorLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Sum over ``axis`` (all axes when ``None``)."""
    ta = as_tensor(a)
    axes = normalize_axes(axis, ta.ndim)
    out = _drop_axes(np.sum(ta.data, axis=axes, keepdims=True), axes, keepdims)

    def backward(g: Array, need: tuple[bool, ...]) -> tuple[Array]:
        return (expand_reduced(g, ta.shape, axes, keepdims),)

    return make_result(np.asarray(out), (ta,), backward, "sum")


def _count(shape: tuple[int, ...], axes: tuple[int, ...]) -> int:
    count = 1
    for ax in axes:
        count *= shape[ax]
    return count


def mean(a: TensorLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Arithmetic mean over ``axis``."""
    ta = as_tensor(a)
    axes = normalize_axes(axis, ta.ndim)
    n = _count(ta.shape, axes)
    out = _drop_axes(np.mean(ta.data, axis=axes, keepdims=True), axes, keepdims)

    def backward(g: Array, need: tuple[bool, ...]) -> tuple[Array]:
        return (expand_reduced(g / n, ta.shape, axes, keepdims),)

    return make_result(np.asarray(out), (ta,), backward, "mean")


def _extremum(
    a: TensorLike, axis: Axis, keepdims: bool, reducer: Callable[..., Array], op: str
) -> Tensor:
    ta = as_tensor(a)
    axes = normalize_axes(axis, ta.ndim)
    kept = reducer(ta.data, axis=axes, keepdims=True)
    out = _drop_axes(kept, axes, keepdims)

    def backward(g: Array, need: tuple[bool, ...]) -> tuple[Array]:
        # Split the gradient evenly between tied extrema (a valid subgradient).
        winners = ta.data == kept
        share = winners / winners.sum(axis=axes, keepdims=True)
        return (expand_reduced(g, ta.shape, axes, keepdims) * share,)

    return make_result(np.asarray(out), (ta,), backward, op)


def max(a: TensorLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Maximum over ``axis``; tied maxima share the gradient equally."""
    return _extremum(a, axis, keepdims, np.max, "max")


def min(a: TensorLike, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Minimum over ``axis``; tied minima share the gradient equally."""
    return _extremum(a, axis, keepdims, np.min, "min")


def var(a: TensorLike, axis: Axis = None, keepdims: bool = False, correction: int = 1) -> Tensor:
    """Variance over ``axis``: ``sum((x - mean)**2) / (N - correction)``.

    ``correction=1`` (the default, as in PyTorch) gives the unbiased estimator; normalisation
    layers use ``correction=0``.
    """
    ta = as_tensor(a)
    axes = normalize_axes(axis, ta.ndim)
    denom = builtins.max(_count(ta.shape, axes) - correction, 0)
    centered = ta.data - np.mean(ta.data, axis=axes, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        kept = np.sum(centered * centered, axis=axes, keepdims=True) / denom
    out = _drop_axes(kept, axes, keepdims)
    # With N <= correction the estimator is undefined; like PyTorch, the value and the
    # gradient are NaN (rather than a ZeroDivisionError during backward).
    scale = 2 / denom if denom else float("nan")

    def backward(g: Array, need: tuple[bool, ...]) -> tuple[Array]:
        return (expand_reduced(g, ta.shape, axes, keepdims) * scale * centered,)

    return make_result(np.asarray(out), (ta,), backward, "var")
