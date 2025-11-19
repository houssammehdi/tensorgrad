"""Function transforms: grad, vjp, jvp, jacobian, hessian and hvp.

Second derivatives are checked three independent ways: against closed-form Hessians of
known functions, against finite differences of the (independently verified) gradient, and
through structural identities (Hessian symmetry, <u, J v> = <J^T u, v>).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import tensorgrad as tg
from tensorgrad import func

seeds = st.integers(min_value=0, max_value=2**32 - 1)


def fd_gradient_directional(
    grad_fn: Callable[[tg.Tensor], tg.Tensor], x: np.ndarray, v: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central difference of the gradient along ``v``: approximately H v."""
    with tg.no_grad():
        plus = grad_fn(tg.Tensor(x + eps * v)).data
        minus = grad_fn(tg.Tensor(x - eps * v)).data
    return (plus - minus) / (2 * eps)


# ---------------------------------------------------------------------- closed forms
def test_hessian_of_a_cubic_is_diagonal() -> None:
    x = np.array([0.5, -1.0, 2.0])
    h = func.hessian(lambda t: (t**3).sum())(x)
    np.testing.assert_allclose(h.data, np.diag(6 * x))


def test_hessian_of_a_quadratic_form() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    h = func.hessian(lambda t: t @ (tg.Tensor(a) @ t))(rng.standard_normal(4))
    np.testing.assert_allclose(h.data, a + a.T, atol=1e-12)


def test_hessian_of_logsumexp_is_the_softmax_covariance() -> None:
    x = np.array([0.3, -1.2, 2.0, 0.1])
    h = func.hessian(lambda t: tg.logsumexp(t, axis=0))(x)
    p = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    np.testing.assert_allclose(h.data, np.diag(p) - np.outer(p, p), atol=1e-12)


def test_hessian_of_the_rosenbrock_function() -> None:
    def rosenbrock(v: tg.Tensor) -> tg.Tensor:
        x, y = v[0], v[1]
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    x, y = 0.7, -0.4
    h = func.hessian(rosenbrock)(np.array([x, y]))
    expected = [[2 - 400 * (y - x**2) + 800 * x**2, -400 * x], [-400 * x, 200.0]]
    np.testing.assert_allclose(h.data, expected, rtol=1e-12)


def test_hessian_of_logistic_regression_loss() -> None:
    rng = np.random.default_rng(1)
    features = rng.standard_normal((20, 3))
    labels = (rng.random(20) > 0.5).astype(np.float64)
    w = rng.standard_normal(3)

    def loss(weights: tg.Tensor) -> tg.Tensor:
        z = tg.Tensor(features) @ weights
        # Numerically stable binary cross-entropy: softplus(z) - y z.
        return (tg.logsumexp(tg.stack([z * 0.0, z], axis=1), axis=1) - labels * z).mean()

    p = 1 / (1 + np.exp(-features @ w))
    expected = features.T @ (features * (p * (1 - p))[:, None]) / len(labels)
    np.testing.assert_allclose(func.hessian(loss)(w).data, expected, atol=1e-12)
    np.testing.assert_allclose(
        func.grad(loss)(w).data, features.T @ (p - labels) / len(labels), atol=1e-12
    )


def test_mixed_partials_with_several_arguments() -> None:
    x, y = np.array([1.0, 2.0]), np.array([0.5, -1.5])
    f = lambda a, b: (a * b * b).sum()
    (h_xx, h_xy), (h_yx, h_yy) = func.hessian(f, argnums=(0, 1))(x, y)
    np.testing.assert_allclose(h_xx.data, np.zeros((2, 2)))
    np.testing.assert_allclose(h_xy.data, np.diag(2 * y))
    np.testing.assert_allclose(h_yx.data, np.diag(2 * y))
    np.testing.assert_allclose(h_yy.data, np.diag(2 * x))


def test_third_derivative_by_nesting_grad() -> None:
    # d^3/dx^3 tanh(x) = -2 (1 - t^2)(1 - 3 t^2) with t = tanh(x)
    d1 = func.grad(lambda x: tg.tanh(x).sum())
    d2 = func.grad(lambda x: d1(x).sum())
    d3 = func.grad(lambda x: d2(x).sum())
    x = np.array([0.3, -0.8])
    t = np.tanh(x)
    np.testing.assert_allclose(d3(x).data, -2 * (1 - t**2) * (1 - 3 * t**2), rtol=1e-10)


# ---------------------------------------------------------------------- finite differences
def small_network(params: tg.Tensor, inputs: np.ndarray) -> tg.Tensor:
    """A scalar loss mixing matmul, tanh, softmax, cross-entropy and layer norm."""
    w1 = params[:12].reshape(3, 4)
    w2 = params[12:20].reshape(4, 2)
    hidden = tg.layer_norm(tg.tanh(tg.Tensor(inputs) @ w1))
    logits = hidden @ w2
    return tg.cross_entropy(logits, np.array([0, 1, 1, 0, 1])) + (tg.softmax(logits) ** 2).sum()


@given(seed=seeds)
def test_hvp_matches_finite_differences_of_the_gradient(seed: int) -> None:
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((5, 3))
    params, v = rng.standard_normal(20), rng.standard_normal(20)
    f = lambda p: small_network(p, inputs)
    value, (hv,) = func.hvp(f, (params,), (v,))
    np.testing.assert_allclose(value.data, f(tg.Tensor(params)).data)
    expected = fd_gradient_directional(func.grad(f), params, v)
    np.testing.assert_allclose(hv.data, expected, rtol=1e-5, atol=1e-7)


@given(seed=seeds)
def test_hessian_is_symmetric_and_consistent_with_hvp(seed: int) -> None:
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((5, 3))
    params = rng.standard_normal(20)
    f = lambda p: small_network(p, inputs)
    h = func.hessian(f)(params).data
    np.testing.assert_allclose(h, h.T, atol=1e-10)
    v = rng.standard_normal(20)
    (hv,) = func.hvp(f, (params,), (v,))[1]
    np.testing.assert_allclose(h @ v, hv.data, rtol=1e-9, atol=1e-11)


def test_jvp_matches_finite_differences_and_the_jacobian() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((3, 4))
    f = lambda x: tg.softmax(tg.Tensor(a) @ tg.tanh(x), axis=0)
    x, v = rng.standard_normal(4), rng.standard_normal(4)
    out, jv = func.jvp(f, (x,), (v,))
    eps = 1e-6
    with tg.no_grad():
        fd = (f(tg.Tensor(x + eps * v)).data - f(tg.Tensor(x - eps * v)).data) / (2 * eps)
    np.testing.assert_allclose(jv.data, fd, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(func.jacobian(f)(x).data @ v, jv.data, atol=1e-12)
    np.testing.assert_allclose(out.data, f(tg.Tensor(x)).data)


def test_vjp_and_jvp_are_adjoint() -> None:
    rng = np.random.default_rng(3)
    f = lambda x, y: tg.tanh(x[:, None] * y[None, :]).sum(axis=0) + y * y
    x, y = rng.standard_normal(3), rng.standard_normal(2)
    vx, vy, u = rng.standard_normal(3), rng.standard_normal(2), rng.standard_normal(2)
    _, jv = func.jvp(f, (x, y), (vx, vy))
    _, vjp_fn = func.vjp(f, x, y)
    ux, uy = vjp_fn(u)
    lhs = float(u @ jv.data)
    rhs = float(ux.data @ vx + uy.data @ vy)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    ux_again, _ = vjp_fn(2 * u)  # the pullback can be reused
    np.testing.assert_allclose(ux_again.data, 2 * ux.data)


def test_jacobian_of_softmax() -> None:
    x = np.array([0.2, -0.3, 1.1])
    s = np.exp(x) / np.exp(x).sum()
    j = func.jacobian(lambda t: tg.softmax(t, axis=0))(x)
    np.testing.assert_allclose(j.data, np.diag(s) - np.outer(s, s), atol=1e-12)


def test_jacobian_shapes_and_tuple_outputs() -> None:
    x = np.ones((2, 3))
    j = func.jacobian(lambda t: t.sum(axis=1))(x)
    assert j.shape == (2, 2, 3)
    ja, jb = func.jacobian(lambda t: (t * 2, t.sum()))(x)
    assert ja.shape == (2, 3, 2, 3) and jb.shape == (2, 3)
    np.testing.assert_allclose(jb.data, np.ones((2, 3)))


# ---------------------------------------------------------------------- composition rules
def test_nested_grad_avoids_perturbation_confusion() -> None:
    """d/dx [x * d/dy (x + y) at y = x] = d/dx [x * 1] = 1.

    A tape that differentiated the inner function with respect to the tensor ``x`` itself
    (rather than an alias of the argument) would see ``d(x + x)/dx = 2`` and return 2.
    """
    inner = lambda x: func.grad(lambda y: (x + y).sum())(x)
    outer = func.grad(lambda x: (x * inner(x)).sum())
    assert outer(tg.Tensor(np.array(3.0))).item() == pytest.approx(1.0)


def test_results_carry_a_graph_only_when_grad_mode_is_on() -> None:
    f = lambda x: (x**3).sum()
    x = tg.Tensor(np.array([1.0, 2.0]))
    assert func.grad(f)(x).requires_grad
    with tg.no_grad():
        g = func.grad(f)(x)
        value, h = func.hvp(f, (x,), (np.ones(2),))
    assert not g.requires_grad and not value.requires_grad and not h[0].requires_grad
    np.testing.assert_allclose(g.data, 3 * x.data**2)


def test_gradient_flows_to_closed_over_parameters() -> None:
    # A gradient penalty: the input gradient must itself be differentiable in the weights.
    w = tg.Tensor(np.array([0.5, -2.0]), requires_grad=True)
    x = np.array([1.0, 3.0])
    gx = func.grad(lambda t: (tg.tanh(t * w)).sum())(x)  # = w (1 - tanh(w x)^2)
    ((gx * gx).sum()).backward()
    t = np.tanh(w.data * x)
    d = 2 * gx.data * ((1 - t**2) - 2 * w.data * x * t * (1 - t**2))
    assert w.grad is not None
    np.testing.assert_allclose(w.grad.data, d, rtol=1e-10)


def test_value_and_grad_and_argnums() -> None:
    f = lambda a, b: (a * b).sum()
    value, (ga, gb) = func.value_and_grad(f, argnums=(0, 1))(
        np.array([1.0, 2.0]), np.array([3.0, 4.0])
    )
    assert value.item() == 11.0
    assert ga.tolist() == [3.0, 4.0] and gb.tolist() == [1.0, 2.0]
    assert func.grad(f, argnums=1)(np.array([1.0, 2.0]), np.array([3.0, 4.0])).tolist() == [
        1.0,
        2.0,
    ]


def test_constant_functions_have_zero_derivatives() -> None:
    x = np.array([1.0, 2.0])
    assert func.grad(lambda t: tg.Tensor(np.array(5.0)))(x).tolist() == [0.0, 0.0]
    assert func.hessian(lambda t: t.sum())(x).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert func.jvp(lambda t: tg.Tensor(np.ones(3)), (x,), (x,))[1].tolist() == [0.0] * 3


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="one scalar"):
        func.grad(lambda t: t * 2)(np.ones(3))
    with pytest.raises(ValueError, match="tangent"):
        func.jvp(lambda t: t, (np.ones(3),), (np.ones(2),))
    with pytest.raises(IndexError, match="argnums"):
        func.grad(lambda t: t.sum(), argnums=2)(np.ones(3))
