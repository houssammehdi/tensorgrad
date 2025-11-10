"""Finite-difference checks of first derivatives (:func:`gradcheck`) and second derivatives
(:func:`gradgradcheck`)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tensorgrad import autograd
from tensorgrad._types import Array
from tensorgrad.autograd import enable_grad, no_grad
from tensorgrad.ops import concat, reshape
from tensorgrad.tensor import Tensor

__all__ = ["GradcheckError", "gradcheck", "gradgradcheck", "numerical_grad"]


class GradcheckError(AssertionError):
    """Raised when analytical and numerical gradients disagree."""


def _projected(fn: Callable[..., Tensor], inputs: Sequence[Tensor], weights: Array) -> float:
    with no_grad():
        return float(np.sum(fn(*inputs).data * weights))


def numerical_grad(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    index: int,
    weights: Array,
    eps: float = 1e-6,
) -> Array:
    """Central-difference estimate of ``d sum(fn(*inputs) * weights) / d inputs[index]``."""
    x = inputs[index].data
    grad = np.zeros_like(x, dtype=np.float64)
    flat = x.reshape(-1)  # a view: perturbing it perturbs the input in place
    if not np.shares_memory(flat, x):
        raise ValueError("gradcheck inputs must be contiguous arrays")
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = _projected(fn, inputs, weights)
        flat[i] = original - eps
        minus = _projected(fn, inputs, weights)
        flat[i] = original
        grad.reshape(-1)[i] = (plus - minus) / (2 * eps)
    return grad


def _matches(
    analytical: Array, expected: Array, what: str, atol: float, rtol: float, raise_exception: bool
) -> bool:
    if np.allclose(analytical, expected, atol=atol, rtol=rtol):
        return True
    if not raise_exception:
        return False
    diff = np.abs(analytical - expected)
    worst = np.unravel_index(int(np.argmax(diff)), diff.shape) if diff.ndim else ()
    raise GradcheckError(
        f"{what}: max abs diff {diff.max():.3e} at {worst}; "
        f"analytical={np.asarray(analytical)[worst]!r}, expected={np.asarray(expected)[worst]!r}"
    )


def _checked_inputs(inputs: Sequence[Tensor], name: str) -> list[int]:
    checked = [i for i, t in enumerate(inputs) if t.requires_grad]
    if not checked:
        raise ValueError(f"{name} needs at least one input with requires_grad=True")
    for i in checked:
        if inputs[i].dtype != np.float64:
            raise TypeError(f"{name} input {i} is {inputs[i].dtype}; use float64")
    return checked


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
    seed: int = 0,
    raise_exception: bool = True,
) -> bool:
    """Compare backpropagated gradients of ``fn`` with central finite differences.

    ``fn`` may return a tensor of any shape: it is reduced to a scalar by a dot product with
    fixed random weights, which checks the full vector-Jacobian product rather than just the
    gradient of ``sum(fn(...))`` (a plain sum would hide errors that cancel out).

    Args:
        fn: Deterministic function of ``inputs``.
        inputs: Tensors to differentiate; those with ``requires_grad`` are checked and must be
            float64 (float32 finite differences are too noisy to be meaningful).
        eps: Finite-difference step.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        seed: Seed for the projection weights.
        raise_exception: Raise :class:`GradcheckError` on mismatch instead of returning False.

    Returns:
        ``True`` if every checked gradient matches.
    """
    checked = _checked_inputs(inputs, "gradcheck")
    for t in inputs:
        t.zero_grad()
    out = fn(*inputs)
    weights = np.random.default_rng(seed).standard_normal(out.shape)
    (out * Tensor(weights)).sum().backward()

    for i in checked:
        grad = inputs[i].grad
        analytical = np.zeros_like(inputs[i].data) if grad is None else grad.data
        numerical = numerical_grad(fn, inputs, i, weights, eps)
        what = f"gradient mismatch for input {i} (shape {inputs[i].shape})"
        if not _matches(analytical, numerical, what, atol, rtol, raise_exception):
            return False
    return True


def gradgradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    seed: int = 0,
    raise_exception: bool = True,
) -> bool:
    """Check second derivatives of ``fn`` against finite differences of its gradient.

    With a fixed random upstream gradient ``v``, the first derivative is the map
    ``G(x, v) = J(x)^T v``, computed with ``create_graph=True`` (the differentiable VJPs).
    Two things are verified:

    1. ``G`` equals the gradient of the ordinary (NumPy) backward pass, so the two VJP
       implementations of every op agree.
    2. :func:`gradcheck` of ``G`` with respect to both ``x`` and ``v``: backpropagating
       through the gradient graph (double backward) must match central differences of
       ``G``. The ``x`` part checks second derivatives; the ``v`` part checks that the
       gradient graph is correctly differentiable in the upstream gradient too (which is
       what forward-mode products via the double-VJP trick rely on).

    Arguments are as in :func:`gradcheck`; inputs that require grad must be float64.
    """
    checked = _checked_inputs(inputs, "gradgradcheck")
    wrt = [inputs[i] for i in checked]
    with enable_grad():
        out = fn(*inputs)
    v = np.random.default_rng(seed + 1).standard_normal(out.shape)

    with enable_grad():
        fast = autograd.grad(fn(*inputs), wrt, grad_outputs=v, allow_unused=True)
        graph = autograd.grad(
            fn(*inputs), wrt, grad_outputs=Tensor(v), create_graph=True, allow_unused=True
        )
    for i, f, g in zip(checked, fast, graph, strict=True):
        zeros = np.zeros_like(inputs[i].data)
        what = f"create_graph VJP disagrees with the NumPy VJP for input {i}"
        f_arr = zeros if f is None else f.data
        g_arr = zeros if g is None else g.data
        if not _matches(g_arr, f_arr, what, 1e-12, 1e-10, raise_exception):
            return False

    def first_derivative(*args: Tensor) -> Tensor:
        *xs, cotangent = args
        with enable_grad():
            grads = autograd.grad(
                fn(*xs),
                [xs[i] for i in checked],
                grad_outputs=cotangent,
                create_graph=True,
                allow_unused=True,
            )
        flat = [
            reshape(Tensor(np.zeros(xs[i].shape)) if g is None else g, (-1,))
            for i, g in zip(checked, grads, strict=True)
        ]
        return concat(flat)

    return gradcheck(
        first_derivative,
        [*inputs, Tensor(v, requires_grad=True)],
        eps=eps,
        atol=atol,
        rtol=rtol,
        seed=seed + 2,
        raise_exception=raise_exception,
    )
