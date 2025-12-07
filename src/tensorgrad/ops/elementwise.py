"""Elementwise arithmetic and pointwise non-linearities.

Each op has two VJPs: ``backward`` on NumPy arrays (an ordinary backward pass) and ``graph``,
the same formula written with tensor ops for ``create_graph=True``. ``graph`` may reuse the
op's output tensor (for ``exp``, ``tanh``, ...): it is itself differentiable, so reusing it
costs nothing and keeps higher derivatives exact.
"""

from __future__ import annotations

import math

import numpy as np

from tensorgrad._types import Array, DTypeLike
from tensorgrad.ops._util import as_tensor, coerce_pair, make_result, unbroadcast
from tensorgrad.ops.shape import sum_to, where
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

Need = tuple[bool, ...]
Pair = tuple[Array | None, Array | None]
TensorPair = tuple[Tensor | None, Tensor | None]


def add(a: TensorLike, b: TensorLike) -> Tensor:
    """``a + b`` with NumPy broadcasting."""
    ta, tb = coerce_pair(a, b)

    def backward(g: Array, need: Need) -> Pair:
        return (
            unbroadcast(g, ta.shape) if need[0] else None,
            unbroadcast(g, tb.shape) if need[1] else None,
        )

    def graph(g: Tensor, out: Tensor, need: Need) -> TensorPair:
        return (
            sum_to(g, ta.shape) if need[0] else None,
            sum_to(g, tb.shape) if need[1] else None,
        )

    return make_result(ta.data + tb.data, (ta, tb), backward, "add", graph=graph)


def sub(a: TensorLike, b: TensorLike) -> Tensor:
    """``a - b`` with NumPy broadcasting."""
    ta, tb = coerce_pair(a, b)

    def backward(g: Array, need: Need) -> Pair:
        return (
            unbroadcast(g, ta.shape) if need[0] else None,
            unbroadcast(-g, tb.shape) if need[1] else None,
        )

    def graph(g: Tensor, out: Tensor, need: Need) -> TensorPair:
        return (
            sum_to(g, ta.shape) if need[0] else None,
            sum_to(-g, tb.shape) if need[1] else None,
        )

    return make_result(ta.data - tb.data, (ta, tb), backward, "sub", graph=graph)


def mul(a: TensorLike, b: TensorLike) -> Tensor:
    """``a * b`` with NumPy broadcasting."""
    ta, tb = coerce_pair(a, b)

    def backward(g: Array, need: Need) -> Pair:
        return (
            unbroadcast(g * tb.data, ta.shape) if need[0] else None,
            unbroadcast(g * ta.data, tb.shape) if need[1] else None,
        )

    def graph(g: Tensor, out: Tensor, need: Need) -> TensorPair:
        return (
            sum_to(g * tb, ta.shape) if need[0] else None,
            sum_to(g * ta, tb.shape) if need[1] else None,
        )

    return make_result(ta.data * tb.data, (ta, tb), backward, "mul", graph=graph)


def div(a: TensorLike, b: TensorLike) -> Tensor:
    """``a / b`` with NumPy broadcasting."""
    ta, tb = coerce_pair(a, b)
    out = ta.data / tb.data

    def backward(g: Array, need: Need) -> Pair:
        return (
            unbroadcast(g / tb.data, ta.shape) if need[0] else None,
            unbroadcast(-g * out / tb.data, tb.shape) if need[1] else None,
        )

    def graph(g: Tensor, y: Tensor, need: Need) -> TensorPair:
        return (
            sum_to(g / tb, ta.shape) if need[0] else None,
            sum_to(-g * y / tb, tb.shape) if need[1] else None,
        )

    return make_result(out, (ta, tb), backward, "div", graph=graph)


def neg(a: TensorLike) -> Tensor:
    """``-a``."""
    ta = as_tensor(a)

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (-g,)

    def graph(g: Tensor, out: Tensor, need: Need) -> tuple[Tensor]:
        return (-g,)

    return make_result(-ta.data, (ta,), backward, "neg", graph=graph)


def pow(base: TensorLike, exponent: TensorLike) -> Tensor:
    """``base ** exponent``.

    The gradient with respect to a tensor exponent, ``out * log(base)``, is only defined for
    positive bases; it is set to zero elsewhere.
    """
    tb, te = coerce_pair(base, exponent)
    out = tb.data**te.data

    def backward(g: Array, need: Need) -> Pair:
        g_base = None
        if need[0]:
            g_base = unbroadcast(g * te.data * tb.data ** (te.data - 1), tb.shape)
        g_exp = None
        if need[1]:
            positive = tb.data > 0
            log_base = np.log(np.where(positive, tb.data, 1))
            g_exp = unbroadcast(np.where(positive, g * out * log_base, 0), te.shape)
        return g_base, g_exp

    def graph(g: Tensor, y: Tensor, need: Need) -> TensorPair:
        g_base = sum_to(g * te * tb ** (te - 1), tb.shape) if need[0] else None
        g_exp = None
        if need[1]:
            positive = tb.data > 0  # constant mask: log(base) only exists for base > 0
            log_base = log(where(positive, tb, 1.0))
            g_exp = sum_to(where(positive, g * y * log_base, 0.0), te.shape)
        return g_base, g_exp

    return make_result(out, (tb, te), backward, "pow", graph=graph)


def exp(a: TensorLike) -> Tensor:
    """Elementwise ``e**a``."""
    ta = as_tensor(a)
    out = np.exp(ta.data)

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (g * out,)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        return (g * y,)

    return make_result(out, (ta,), backward, "exp", graph=graph)


def log(a: TensorLike) -> Tensor:
    """Elementwise natural logarithm."""
    ta = as_tensor(a)

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (g / ta.data,)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        return (g / ta,)

    return make_result(np.log(ta.data), (ta,), backward, "log", graph=graph)


def sqrt(a: TensorLike) -> Tensor:
    """Elementwise square root."""
    ta = as_tensor(a)
    out = np.sqrt(ta.data)

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (g * 0.5 / out,)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        return (g * 0.5 / y,)

    return make_result(out, (ta,), backward, "sqrt", graph=graph)


def tanh(a: TensorLike) -> Tensor:
    """Elementwise hyperbolic tangent."""
    ta = as_tensor(a)
    out = np.tanh(ta.data)

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (g * (1 - out * out),)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        return (g * (1 - y * y),)

    return make_result(out, (ta,), backward, "tanh", graph=graph)


def sigmoid(a: TensorLike) -> Tensor:
    """Elementwise logistic sigmoid, evaluated without overflow for large ``|a|``."""
    ta = as_tensor(a)
    x = ta.data
    # exp(-|x|) never overflows; pick the algebraically equivalent branch per sign.
    z = np.exp(-np.abs(x))
    out = np.where(x >= 0, 1 / (1 + z), z / (1 + z)).astype(x.dtype, copy=False)

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (g * out * (1 - out),)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        return (g * y * (1 - y),)

    return make_result(out, (ta,), backward, "sigmoid", graph=graph)


def relu(a: TensorLike) -> Tensor:
    """Elementwise ``max(a, 0)`` (gradient 0 at exactly 0)."""
    ta = as_tensor(a)
    positive = ta.data > 0

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (g * positive,)

    def graph(g: Tensor, out: Tensor, need: Need) -> tuple[Tensor]:
        # The mask is piecewise constant, so relu's second derivative is zero (a.e.).
        return (where(positive, g, 0.0),)

    return make_result(np.maximum(ta.data, 0), (ta,), backward, "relu", graph=graph)


_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_A = 0.044715


def gelu(a: TensorLike) -> Tensor:
    """GELU with the tanh approximation used by GPT-2.

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``

    Forward and backward are written in place: they allocate two arrays each instead of one
    per arithmetic step, which matters for the large activations of a transformer MLP.
    """
    ta = as_tensor(a)
    x = ta.data if ta.dtype.kind == "f" else ta.data.astype(np.float64)
    # Explicit output buffers keep 0-d inputs arrays (plain ``x * x`` would give a scalar).
    t = np.multiply(x, x, out=np.empty_like(x))
    t *= _GELU_A
    t += 1.0
    t *= x
    t *= _GELU_C  # sqrt(2/pi) * (x + 0.044715 x^3), without NumPy's slow float32 power
    np.tanh(t, out=t)
    out = np.add(t, 1.0, out=np.empty_like(t))
    out *= x
    out *= 0.5

    def backward(g: Array, need: Need) -> tuple[Array]:
        # d/dx = 0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2)
        d = np.multiply(x, x, out=np.empty_like(x))
        d *= 3 * _GELU_A * _GELU_C
        d += _GELU_C
        s = np.multiply(t, t, out=np.empty_like(t))
        np.subtract(1.0, s, out=s)
        s *= d
        s *= x
        s += t
        s += 1.0
        s *= 0.5
        if np.result_type(s, g) == s.dtype:
            s *= g
            return (s,)
        return (s * g,)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        x2 = ta * ta
        th = tanh(_GELU_C * (ta + _GELU_A * x2 * ta))
        d_inner = _GELU_C * (1 + 3 * _GELU_A * x2)
        return (g * (0.5 * (1 + th) + 0.5 * ta * (1 - th * th) * d_inner),)

    return make_result(out, (ta,), backward, "gelu", graph=graph)


def astype(a: TensorLike, dtype: DTypeLike) -> Tensor:
    """Dtype conversion; the gradient is cast back to the input dtype.

    Only floating-point results stay in the graph. Casting to an integer or boolean dtype
    is piecewise constant (its derivative is zero almost everywhere), so, as in PyTorch, the
    result is a constant that does not require grad.
    """
    ta = as_tensor(a)
    source = ta.dtype
    out = ta.data.astype(dtype)
    if not np.issubdtype(out.dtype, np.floating):
        return Tensor._wrap(out)

    def backward(g: Array, need: Need) -> tuple[Array]:
        return (g.astype(source),)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        return (astype(g, source),)

    return make_result(out, (ta,), backward, "astype", graph=graph)
