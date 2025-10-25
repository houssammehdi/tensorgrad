"""The :class:`Tensor` type, dtype defaults and tensor factory functions."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, SupportsIndex, TypeAlias, cast

import numpy as np

from tensorgrad._random import get_rng
from tensorgrad._types import Array, Axis, DTypeLike, FloatDType, Scalar, ShapeLike
from tensorgrad.autograd import Node, run_backward

if TYPE_CHECKING:
    from types import EllipsisType

__all__ = [
    "IndexLike",
    "Tensor",
    "TensorLike",
    "arange",
    "full",
    "get_default_dtype",
    "ones",
    "ones_like",
    "rand",
    "randn",
    "set_default_dtype",
    "tensor",
    "zeros",
    "zeros_like",
]


class _DTypeState:
    default: FloatDType = np.float32


def set_default_dtype(dtype: FloatDType) -> None:
    """Set the floating dtype used for tensors built from Python floats (default float32)."""
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"default dtype must be a floating type, got {dtype}")
    _DTypeState.default = dtype


def get_default_dtype() -> FloatDType:
    """Return the floating dtype used for tensors built from Python floats."""
    return _DTypeState.default


def _to_array(data: object, dtype: DTypeLike | None) -> Array:
    if isinstance(data, Tensor):
        data = data.data
    if isinstance(data, np.ndarray | np.generic):
        return np.array(data, dtype=dtype, copy=True)
    arr = np.asarray(data, dtype=dtype)
    if dtype is None and arr.dtype == np.float64:
        # Python floats become the default dtype; explicit float64 NumPy arrays stay float64.
        arr = arr.astype(get_default_dtype())
    return arr


def _shape_tuple(shape: ShapeLike | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(s) for s in shape)


class Tensor:
    """An n-dimensional array that records the ops applied to it for reverse-mode autodiff.

    Args:
        data: Anything :func:`numpy.asarray` accepts, or another tensor (its data is copied).
            Python floats become :func:`get_default_dtype` (float32); NumPy arrays keep their
            dtype, so float64 arrays give float64 tensors (what :func:`gradcheck` wants).
        requires_grad: Record ops on this tensor so ``backward`` can populate ``.grad``.
            Only floating-point tensors can require gradients.
        dtype: Optional dtype to cast ``data`` to.

    Attributes:
        data: The underlying :class:`numpy.ndarray`.
        grad: Accumulated gradient (same shape and dtype as ``data``) or ``None``.
    """

    __slots__ = ("__weakref__", "_node", "_requires_grad", "_retains_grad", "data", "grad")

    # Make NumPy defer to our reflected operators: ``ndarray + Tensor`` -> Tensor.__radd__.
    __array_ufunc__ = None

    data: Array
    grad: Array | None

    def __init__(
        self,
        data: object,
        requires_grad: bool = False,
        dtype: DTypeLike | None = None,
    ) -> None:
        self.data = _to_array(data, dtype)
        self.grad = None
        self._node: Node | None = None
        self._retains_grad = False
        self._requires_grad = False
        self.requires_grad = requires_grad

    @classmethod
    def _wrap(cls, data: Array) -> Tensor:
        """Wrap an existing array without copying or converting it (internal fast path)."""
        out = Tensor.__new__(Tensor)
        out.data = data
        out.grad = None
        out._node = None
        out._retains_grad = False
        out._requires_grad = False
        return out

    # ------------------------------------------------------------------ properties
    @property
    def requires_grad(self) -> bool:
        """Whether gradients flow to (and, for leaves, accumulate in) this tensor."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if self._node is not None:
            raise RuntimeError("requires_grad can only be changed on leaf tensors")
        if value and not np.issubdtype(self.data.dtype, np.floating):
            raise TypeError(
                f"only floating-point tensors can require gradients (got {self.data.dtype})"
            )
        self._requires_grad = bool(value)

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor's shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.data.size

    @property
    def dtype(self) -> np.dtype[Any]:
        """The NumPy dtype of the underlying data."""
        return self.data.dtype

    @property
    def is_leaf(self) -> bool:
        """``True`` for tensors created by the user rather than by a recorded op."""
        return self._node is None

    @property
    def grad_fn(self) -> Node | None:
        """The graph node that produced this tensor, or ``None`` for leaves."""
        return self._node

    @property
    def T(self) -> Tensor:
        """Reverse all axes (a matrix transpose for 2-D tensors)."""
        return _shape.permute(self, tuple(reversed(range(self.ndim))))

    # ------------------------------------------------------------------ autograd
    def backward(self, grad: Tensor | Array | None = None, retain_graph: bool = False) -> None:
        """Backpropagate from this tensor, accumulating gradients into the leaves' ``.grad``.

        Args:
            grad: d(loss)/d(self). May be omitted only when the tensor has a single element.
            retain_graph: Keep the graph's saved values so ``backward`` can run again.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for single-element outputs; "
                    f"got shape {self.shape}"
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad.data if isinstance(grad, Tensor) else grad, dtype=self.dtype)
            if seed.shape != self.shape:
                raise ValueError(f"grad has shape {seed.shape}, expected {self.shape}")
        run_backward(self, seed, retain_graph=retain_graph)

    def _accumulate_grad(self, g: Array) -> None:
        g = g.astype(self.dtype, copy=False)
        # Copy on first write: ``g`` may be a read-only broadcast view or alias another array.
        self.grad = np.array(g, copy=True) if self.grad is None else self.grad + g

    def retain_grad(self) -> None:
        """Keep ``.grad`` for this non-leaf tensor during the next backward pass."""
        if not self.requires_grad:
            raise RuntimeError("cannot retain_grad() on a tensor that does not require grad")
        if self._node is not None:
            self._retains_grad = True

    def zero_grad(self) -> None:
        """Reset the accumulated gradient."""
        self.grad = None

    def detach(self) -> Tensor:
        """Return a tensor sharing this data but cut off from the graph."""
        return Tensor._wrap(self.data)

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set :attr:`requires_grad` in place and return ``self``."""
        self.requires_grad = requires_grad
        return self

    # ------------------------------------------------------------------ conversion
    def numpy(self) -> Array:
        """Return the underlying array (shared memory, not a copy)."""
        return self.data

    def item(self) -> Any:
        """Return the value of a single-element tensor as a Python scalar."""
        return self.data.item()

    def tolist(self) -> Any:
        """Return the data as (nested) Python lists."""
        return self.data.tolist()

    def astype(self, dtype: DTypeLike) -> Tensor:
        """Differentiable dtype cast."""
        return _elementwise.astype(self, dtype)

    # ------------------------------------------------------------------ dunder helpers
    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __bool__(self) -> bool:
        return bool(self.data)

    def __repr__(self) -> str:
        body = np.array2string(self.data, separator=", ", prefix="Tensor(", precision=4)
        extras = []
        if self.dtype != get_default_dtype():
            extras.append(f"dtype={self.dtype}")
        if self._node is not None:
            extras.append(f"grad_fn=<{self._node.op}>")
        elif self.requires_grad:
            extras.append("requires_grad=True")
        suffix = "".join(f", {e}" for e in extras)
        return f"{type(self).__name__}({body}{suffix})"

    # ------------------------------------------------------------------ arithmetic
    def __add__(self, other: TensorLike) -> Tensor:
        return _elementwise.add(self, other)

    def __radd__(self, other: TensorLike) -> Tensor:
        return _elementwise.add(other, self)

    def __sub__(self, other: TensorLike) -> Tensor:
        return _elementwise.sub(self, other)

    def __rsub__(self, other: TensorLike) -> Tensor:
        return _elementwise.sub(other, self)

    def __mul__(self, other: TensorLike) -> Tensor:
        return _elementwise.mul(self, other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        return _elementwise.mul(other, self)

    def __truediv__(self, other: TensorLike) -> Tensor:
        return _elementwise.div(self, other)

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        return _elementwise.div(other, self)

    def __pow__(self, exponent: TensorLike) -> Tensor:
        return _elementwise.pow(self, exponent)

    def __rpow__(self, base: TensorLike) -> Tensor:
        return _elementwise.pow(base, self)

    def __neg__(self) -> Tensor:
        return _elementwise.neg(self)

    def __matmul__(self, other: TensorLike) -> Tensor:
        return _linalg.matmul(self, other)

    def __rmatmul__(self, other: TensorLike) -> Tensor:
        return _linalg.matmul(other, self)

    def __getitem__(self, index: IndexLike) -> Tensor:
        return _shape.getitem(self, index)

    # Comparisons are not differentiable and return boolean tensors. ``__eq__`` is deliberately
    # left alone so tensors stay hashable by identity (the autograd engine relies on it).
    def __lt__(self, other: object) -> Tensor:
        return Tensor._wrap(self.data < _raw(other))

    def __le__(self, other: object) -> Tensor:
        return Tensor._wrap(self.data <= _raw(other))

    def __gt__(self, other: object) -> Tensor:
        return Tensor._wrap(self.data > _raw(other))

    def __ge__(self, other: object) -> Tensor:
        return Tensor._wrap(self.data >= _raw(other))

    # ------------------------------------------------------------------ op methods
    def matmul(self, other: TensorLike) -> Tensor:
        """Matrix product (see :func:`tensorgrad.matmul`)."""
        return _linalg.matmul(self, other)

    def exp(self) -> Tensor:
        """Elementwise ``e**x``."""
        return _elementwise.exp(self)

    def log(self) -> Tensor:
        """Elementwise natural logarithm."""
        return _elementwise.log(self)

    def sqrt(self) -> Tensor:
        """Elementwise square root."""
        return _elementwise.sqrt(self)

    def tanh(self) -> Tensor:
        """Elementwise hyperbolic tangent."""
        return _elementwise.tanh(self)

    def sigmoid(self) -> Tensor:
        """Elementwise logistic sigmoid."""
        return _elementwise.sigmoid(self)

    def relu(self) -> Tensor:
        """Elementwise ``max(x, 0)``."""
        return _elementwise.relu(self)

    def gelu(self) -> Tensor:
        """Elementwise GELU (tanh approximation)."""
        return _elementwise.gelu(self)

    def sum(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        """Sum over ``axis`` (all axes by default)."""
        return _reduce.sum(self, axis, keepdims)

    def mean(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        """Arithmetic mean over ``axis``."""
        return _reduce.mean(self, axis, keepdims)

    def max(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        """Maximum over ``axis``; the gradient is shared equally between tied maxima."""
        return _reduce.max(self, axis, keepdims)

    def min(self, axis: Axis = None, keepdims: bool = False) -> Tensor:
        """Minimum over ``axis``; the gradient is shared equally between tied minima."""
        return _reduce.min(self, axis, keepdims)

    def var(self, axis: Axis = None, keepdims: bool = False, correction: int = 1) -> Tensor:
        """Variance over ``axis`` with Bessel ``correction`` (1 = unbiased, 0 = population)."""
        return _reduce.var(self, axis, keepdims, correction)

    def argmax(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Indices of the maxima (not differentiable)."""
        return Tensor._wrap(np.argmax(self.data, axis=axis, keepdims=keepdims))

    def reshape(self, *shape: int | Sequence[int]) -> Tensor:
        """Return a tensor with the same data and a new shape (one entry may be ``-1``).

        Accepts ``t.reshape(2, 3)`` as well as ``t.reshape((2, 3))``.
        """
        dims: Sequence[object] = (
            shape[0] if len(shape) == 1 and isinstance(shape[0], Sequence) else shape
        )
        return _shape.reshape(self, tuple(operator.index(cast(SupportsIndex, d)) for d in dims))

    def transpose(self, axis0: int, axis1: int) -> Tensor:
        """Swap two axes."""
        return _shape.transpose(self, axis0, axis1)

    def permute(self, *dims: int) -> Tensor:
        """Reorder all axes."""
        return _shape.permute(self, dims)

    def flatten(self, start_axis: int = 0, end_axis: int = -1) -> Tensor:
        """Merge axes ``start_axis..end_axis`` (inclusive) into one."""
        return _shape.flatten(self, start_axis, end_axis)

    def unsqueeze(self, axis: int) -> Tensor:
        """Insert a size-1 axis at ``axis``."""
        return _shape.unsqueeze(self, axis)

    def squeeze(self, axis: int | None = None) -> Tensor:
        """Remove size-1 axes (only ``axis`` if given)."""
        return _shape.squeeze(self, axis)

    def softmax(self, axis: int = -1) -> Tensor:
        """Numerically stable softmax along ``axis``."""
        return _activation.softmax(self, axis)

    def log_softmax(self, axis: int = -1) -> Tensor:
        """Numerically stable log-softmax along ``axis``."""
        return _activation.log_softmax(self, axis)

    def masked_fill(self, mask: Tensor | Array, value: float) -> Tensor:
        """Replace entries where ``mask`` is true by ``value`` (no gradient flows there)."""
        return _shape.masked_fill(self, mask, value)


#: Anything accepted where a tensor operand is expected.
TensorLike: TypeAlias = Tensor | Array | Scalar

_IndexItem: TypeAlias = (
    "int | np.integer[Any] | slice | EllipsisType | Array | Tensor | list[int] | None"
)

#: Indices accepted by ``Tensor.__getitem__`` (basic and integer/boolean array indexing).
IndexLike: TypeAlias = "_IndexItem | tuple[_IndexItem, ...]"


def _raw(x: object) -> Array:
    return x.data if isinstance(x, Tensor) else np.asarray(x)


# ---------------------------------------------------------------------- factories
def tensor(data: object, requires_grad: bool = False, dtype: DTypeLike | None = None) -> Tensor:
    """Construct a tensor (alias for the :class:`Tensor` constructor)."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def _float_dtype(dtype: DTypeLike | None) -> DTypeLike:
    return get_default_dtype() if dtype is None else dtype


def zeros(
    shape: ShapeLike, *, dtype: DTypeLike | None = None, requires_grad: bool = False
) -> Tensor:
    """A tensor filled with zeros."""
    return Tensor(
        np.zeros(_shape_tuple(shape), dtype=_float_dtype(dtype)), requires_grad=requires_grad
    )


def ones(
    shape: ShapeLike, *, dtype: DTypeLike | None = None, requires_grad: bool = False
) -> Tensor:
    """A tensor filled with ones."""
    return Tensor(
        np.ones(_shape_tuple(shape), dtype=_float_dtype(dtype)), requires_grad=requires_grad
    )


def full(
    shape: ShapeLike,
    fill_value: float,
    *,
    dtype: DTypeLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """A tensor filled with ``fill_value``."""
    return Tensor(
        np.full(_shape_tuple(shape), fill_value, dtype=_float_dtype(dtype)),
        requires_grad=requires_grad,
    )


def zeros_like(t: Tensor, *, requires_grad: bool = False) -> Tensor:
    """Zeros with the shape and dtype of ``t``."""
    return Tensor(np.zeros_like(t.data), requires_grad=requires_grad)


def ones_like(t: Tensor, *, requires_grad: bool = False) -> Tensor:
    """Ones with the shape and dtype of ``t``."""
    return Tensor(np.ones_like(t.data), requires_grad=requires_grad)


def arange(start: float, stop: float | None = None, step: float = 1) -> Tensor:
    """Evenly spaced values in ``[start, stop)`` (integer dtype for integer arguments)."""
    values = np.arange(start) if stop is None else np.arange(start, stop, step)
    return Tensor._wrap(values)


def randn(
    shape: ShapeLike, *, dtype: DTypeLike | None = None, requires_grad: bool = False
) -> Tensor:
    """Standard-normal samples drawn from the global generator."""
    data = get_rng().standard_normal(_shape_tuple(shape)).astype(_float_dtype(dtype))
    return Tensor(data, requires_grad=requires_grad)


def rand(
    shape: ShapeLike, *, dtype: DTypeLike | None = None, requires_grad: bool = False
) -> Tensor:
    """Uniform ``[0, 1)`` samples drawn from the global generator."""
    data = get_rng().random(_shape_tuple(shape)).astype(_float_dtype(dtype))
    return Tensor(data, requires_grad=requires_grad)


# The op modules import ``Tensor`` from this module, so they are bound only after the class
# exists; Tensor methods look them up at call time.
from tensorgrad.ops import activation as _activation  # noqa: E402
from tensorgrad.ops import elementwise as _elementwise  # noqa: E402
from tensorgrad.ops import linalg as _linalg  # noqa: E402
from tensorgrad.ops import reduce as _reduce  # noqa: E402
from tensorgrad.ops import shape as _shape  # noqa: E402
