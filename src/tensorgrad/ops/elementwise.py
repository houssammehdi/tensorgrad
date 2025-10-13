"""Elementwise arithmetic and pointwise non-linearities."""

from __future__ import annotations

import math

import numpy as np

from tensorgrad._types import Array, DTypeLike
from tensorgrad.ops._util import as_tensor, coerce_pair, make_result, unbroadcast
from tensorgrad.tensor import Tensor, TensorLike

__all__ = [
    "add",
    "astype",
    "div",
    "exp",
    "gelu",
    "log",
    "mul",
    "neg",
    "pow",
    "relu",
    "sigmoid",
    "sqrt",
    "sub",
    "tanh",
]


def add(a: TensorLike, b: TensorLike) -> Tensor:
    """``a + b`` with NumPy broadcasting."""
    ta, tb = coerce_pair(a, b)

    def backward(g: Array) -> tuple[Array | None, Array | None]:
        return (
            unbroadcast(g, ta.shape) if ta.requires_grad else None,
            unbroadcast(g, tb.shape) if tb.requires_grad else None,
        )

    return make_result(ta.data + tb.data, (ta, tb), backward, "add")


def sub(a: TensorLike, b: TensorLike) -> Tensor:
    """``a - b`` with NumPy broadcasting."""
    ta, tb = coerce_pair(a, b)

    def backward(g: Array) -> tuple[Array | None, Array | None]:
        return (
            unbroadcast(g, ta.shape) if ta.requires_grad else None,
            unbroadcast(-g, tb.shape) if tb.requires_grad else None,
        )

    return make_result(ta.data - tb.data, (ta, tb), backward, "sub")


def mul(a: TensorLike, b: TensorLike) -> Tensor:
    """``a * b`` with NumPy broadcasting."""
    ta, tb = coerce_pair(a, b)

    def backward(g: Array) -> tuple[Array | None, Array | None]:
        return (
            unbroadcast(g * tb.data, ta.shape) if ta.requires_grad else None,
            unbroadcast(g * ta.data, tb.shape) if tb.requires_grad else None,
        )

    return make_result(ta.data * tb.data, (ta, tb), backward, "mul")


def div(a: TensorLike, b: TensorLike) -> Tensor:
    """``a / b`` with NumPy broadcasting."""
    ta, tb = coerce_pair(a, b)
    out = ta.data / tb.data

    def backward(g: Array) -> tuple[Array | None, Array | None]:
        return (
            unbroadcast(g / tb.data, ta.shape) if ta.requires_grad else None,
            unbroadcast(-g * out / tb.data, tb.shape) if tb.requires_grad else None,
        )

    return make_result(out, (ta, tb), backward, "div")


def neg(a: TensorLike) -> Tensor:
    """``-a``."""
    ta = as_tensor(a)

    def backward(g: Array) -> tuple[Array]:
        return (-g,)

    return make_result(-ta.data, (ta,), backward, "neg")


def pow(base: TensorLike, exponent: TensorLike) -> Tensor:
    """``base ** exponent``.

    The gradient with respect to a tensor exponent, ``out * log(base)``, is only defined for
    positive bases; it is set to zero elsewhere.
    """
    tb, te = coerce_pair(base, exponent)
    out = tb.data**te.data

    def backward(g: Array) -> tuple[Array | None, Array | None]:
        g_base = None
        if tb.requires_grad:
            g_base = unbroadcast(g * te.data * tb.data ** (te.data - 1), tb.shape)
        g_exp = None
        if te.requires_grad:
            positive = tb.data > 0
            log_base = np.log(np.where(positive, tb.data, 1))
            g_exp = unbroadcast(np.where(positive, g * out * log_base, 0), te.shape)
        return g_base, g_exp

    return make_result(out, (tb, te), backward, "pow")


def exp(a: TensorLike) -> Tensor:
    """Elementwise ``e**a``."""
    ta = as_tensor(a)
    out = np.exp(ta.data)

    def backward(g: Array) -> tuple[Array]:
        return (g * out,)

    return make_result(out, (ta,), backward, "exp")


def log(a: TensorLike) -> Tensor:
    """Elementwise natural logarithm."""
    ta = as_tensor(a)

    def backward(g: Array) -> tuple[Array]:
        return (g / ta.data,)

    return make_result(np.log(ta.data), (ta,), backward, "log")


def sqrt(a: TensorLike) -> Tensor:
    """Elementwise square root."""
    ta = as_tensor(a)
    out = np.sqrt(ta.data)

    def backward(g: Array) -> tuple[Array]:
        return (g * 0.5 / out,)

    return make_result(out, (ta,), backward, "sqrt")


def tanh(a: TensorLike) -> Tensor:
    """Elementwise hyperbolic tangent."""
    ta = as_tensor(a)
    out = np.tanh(ta.data)

    def backward(g: Array) -> tuple[Array]:
        return (g * (1 - out * out),)

    return make_result(out, (ta,), backward, "tanh")


def sigmoid(a: TensorLike) -> Tensor:
    """Elementwise logistic sigmoid, evaluated without overflow for large ``|a|``."""
    ta = as_tensor(a)
    x = ta.data
    # exp(-|x|) never overflows; pick the algebraically equivalent branch per sign.
    z = np.exp(-np.abs(x))
    out = np.where(x >= 0, 1 / (1 + z), z / (1 + z)).astype(x.dtype, copy=False)

    def backward(g: Array) -> tuple[Array]:
        return (g * out * (1 - out),)

    return make_result(out, (ta,), backward, "sigmoid")


def relu(a: TensorLike) -> Tensor:
    """Elementwise ``max(a, 0)`` (gradient 0 at exactly 0)."""
    ta = as_tensor(a)
    positive = ta.data > 0

    def backward(g: Array) -> tuple[Array]:
        return (g * positive,)

    return make_result(np.maximum(ta.data, 0), (ta,), backward, "relu")


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(a: TensorLike) -> Tensor:
    """GELU with the tanh approximation used by GPT-2.

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``
    """
    ta = as_tensor(a)
    x = ta.data
    inner = _GELU_C * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    out = 0.5 * x * (1 + t)

    def backward(g: Array) -> tuple[Array]:
        d_inner = _GELU_C * (1 + 3 * 0.044715 * x * x)
        return (g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * d_inner),)

    return make_result(out, (ta,), backward, "gelu")


def astype(a: TensorLike, dtype: DTypeLike) -> Tensor:
    """Differentiable dtype conversion; the gradient is cast back to the input dtype."""
    ta = as_tensor(a)
    source = ta.dtype

    def backward(g: Array) -> tuple[Array]:
        return (g.astype(source),)

    return make_result(ta.data.astype(dtype), (ta,), backward, "astype")
