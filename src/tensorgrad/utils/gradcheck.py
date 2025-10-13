"""Finite-difference gradient checking."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tensorgrad._types import Array
from tensorgrad.autograd import no_grad
from tensorgrad.tensor import Tensor

__all__ = ["GradcheckError", "gradcheck", "numerical_grad"]


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
    checked = [i for i, t in enumerate(inputs) if t.requires_grad]
    if not checked:
        raise ValueError("gradcheck needs at least one input with requires_grad=True")
    for i in checked:
        if inputs[i].dtype != np.float64:
            raise TypeError(f"gradcheck input {i} is {inputs[i].dtype}; use float64")

    for t in inputs:
        t.zero_grad()
    out = fn(*inputs)
    weights = np.random.default_rng(seed).standard_normal(out.shape)
    (out * Tensor(weights)).sum().backward()

    for i in checked:
        analytical = inputs[i].grad
        if analytical is None:
            analytical = np.zeros_like(inputs[i].data)
        numerical = numerical_grad(fn, inputs, i, weights, eps)
        if not np.allclose(analytical, numerical, atol=atol, rtol=rtol):
            if not raise_exception:
                return False
            diff = np.abs(analytical - numerical)
            worst = np.unravel_index(int(np.argmax(diff)), diff.shape) if diff.ndim else ()
            raise GradcheckError(
                f"gradient mismatch for input {i} (shape {inputs[i].shape}): "
                f"max abs diff {diff.max():.3e} at {worst}; "
                f"analytical={np.asarray(analytical)[worst]!r}, "
                f"numerical={numerical[worst]!r}"
            )
    return True
