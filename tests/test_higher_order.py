"""Higher-order differentiation through the engine: create_graph, double backward, JVPs."""

from __future__ import annotations

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad.autograd import grad


def test_gradients_of_gradients_match_closed_forms() -> None:
    x = tg.Tensor(np.array([0.3, -1.2, 2.0]), requires_grad=True)
    (g1,) = grad((x**4).sum(), x, create_graph=True)
    (g2,) = grad(g1.sum(), x, create_graph=True)
    (g3,) = grad(g2.sum(), x, create_graph=True)
    (g4,) = grad(g3.sum(), x, allow_unused=True)
    assert g1 is not None and g2 is not None and g3 is not None
    np.testing.assert_allclose(g1.data, 4 * x.data**3)
    np.testing.assert_allclose(g2.data, 12 * x.data**2)
    np.testing.assert_allclose(g3.data, 24 * x.data)
    assert g4 is not None
    np.testing.assert_allclose(g4.data, 24.0)


def test_backward_with_create_graph_gives_differentiable_grads() -> None:
    x = tg.Tensor(np.array([1.5, -0.5]), requires_grad=True)
    tg.exp(x * x).sum().backward(create_graph=True)
    assert x.grad is not None and x.grad.requires_grad and not x.grad.is_leaf
    first = x.grad
    np.testing.assert_allclose(first.data, 2 * x.data * np.exp(x.data**2))
    x.grad = None
    first.sum().backward()  # the forward graph was retained: create_graph implies it
    expected = (2 + 4 * x.data**2) * np.exp(x.data**2)
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.data, expected)


def test_create_graph_accumulates_into_existing_grad() -> None:
    x = tg.Tensor(np.array([2.0]), requires_grad=True)
    (x * x).sum().backward(create_graph=True)
    (x * x * x).sum().backward(create_graph=True)
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.data, 2 * 2.0 + 3 * 4.0)
    (d2,) = grad(x.grad.sum(), x)
    assert d2 is not None
    np.testing.assert_allclose(d2.data, 2 + 6 * 2.0)


def test_plain_backward_leaves_no_graph_on_grads() -> None:
    x = tg.Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (x * x).sum().backward()
    assert x.grad is not None and not x.grad.requires_grad
    (g,) = grad((x * x).sum(), x)
    assert g is not None and not g.requires_grad


def test_create_graph_works_inside_no_grad() -> None:
    # The backward pass decides about recording, not the caller's grad mode.
    x = tg.Tensor(np.array([3.0]), requires_grad=True)
    y = (x**3).sum()
    with tg.no_grad():
        (g,) = grad(y, x, create_graph=True)
    assert g is not None and g.requires_grad
    (h,) = grad(g.sum(), x)
    assert h is not None
    np.testing.assert_allclose(h.data, 6 * 3.0)


def test_jacobian_vector_product_by_the_double_vjp_trick() -> None:
    """J v = d/du <J^T u, v>: the VJP is linear in u, so differentiating it gives the JVP."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((3, 4))
    x = tg.Tensor(rng.standard_normal(4), requires_grad=True)
    f = lambda t: tg.tanh(tg.Tensor(w) * t).sum(axis=1)  # R^4 -> R^3
    v = rng.standard_normal(4)
    u = tg.Tensor(np.zeros(3), requires_grad=True)  # dummy cotangent; its value is irrelevant
    (vjp,) = grad(f(x), x, grad_outputs=u, create_graph=True)
    assert vjp is not None
    (jvp,) = grad(vjp, u, grad_outputs=v)
    assert jvp is not None
    eps = 1e-6
    fd = (f(tg.Tensor(x.data + eps * v)).data - f(tg.Tensor(x.data - eps * v)).data) / (2 * eps)
    np.testing.assert_allclose(jvp.data, fd, rtol=1e-7, atol=1e-9)


def test_graph_can_be_freed_explicitly_under_create_graph() -> None:
    x = tg.Tensor(np.array([1.0]), requires_grad=True)
    y = (x * x).sum()
    (g,) = grad(y, x, create_graph=True, retain_graph=False)
    assert g is not None
    with pytest.raises(RuntimeError, match="second time"):
        grad(y, x)
