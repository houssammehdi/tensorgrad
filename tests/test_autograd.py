"""Behaviour of the Tensor type and the reverse-mode engine."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad.utils import GradcheckError, gradcheck


class TestConstruction:
    def test_python_floats_use_default_dtype(self) -> None:
        assert tg.tensor([1.0, 2.0]).dtype == np.float32
        assert tg.Tensor(3.5).dtype == np.float32

    def test_numpy_arrays_keep_their_dtype(self) -> None:
        assert tg.Tensor(np.zeros(3)).dtype == np.float64
        assert tg.Tensor(np.float64(1.0)).dtype == np.float64
        assert tg.Tensor(np.arange(3)).dtype == np.int64

    def test_explicit_dtype_and_copy_semantics(self) -> None:
        base = np.ones(3)
        t = tg.Tensor(base, dtype=np.float32)
        assert t.dtype == np.float32
        base[0] = 5.0
        assert t.data[0] == 1.0  # the constructor copies

    def test_set_default_dtype(self) -> None:
        tg.set_default_dtype(np.float64)
        try:
            assert tg.tensor([1.0]).dtype == np.float64
            assert tg.zeros(2).dtype == np.float64
        finally:
            tg.set_default_dtype(np.float32)
        with pytest.raises(TypeError):
            tg.set_default_dtype(np.int32)

    def test_only_floats_can_require_grad(self) -> None:
        with pytest.raises(TypeError):
            tg.Tensor(np.arange(3), requires_grad=True)

    def test_factories(self) -> None:
        assert tg.zeros((2, 3)).shape == (2, 3)
        assert tg.ones(4).data.sum() == 4
        assert tg.full((2,), 7.0).tolist() == [7.0, 7.0]
        assert tg.arange(5).tolist() == [0, 1, 2, 3, 4]
        assert tg.randn((3, 2), requires_grad=True).requires_grad
        r = tg.rand(100)
        assert r.data.min() >= 0.0 and r.data.max() < 1.0
        assert tg.zeros_like(r).shape == r.shape and tg.ones_like(r).data.all()

    def test_manual_seed_is_reproducible(self) -> None:
        tg.manual_seed(42)
        a = tg.randn(5).data
        tg.manual_seed(42)
        np.testing.assert_array_equal(a, tg.randn(5).data)

    def test_repr_and_scalars(self) -> None:
        t = tg.Tensor([[1.0, 2.0]], requires_grad=True)
        assert "requires_grad=True" in repr(t)
        assert "grad_fn=<mul>" in repr(t * 2)
        assert tg.Tensor(3.0).item() == 3.0
        assert len(t) == 1
        with pytest.raises(TypeError):
            len(tg.Tensor(1.0))

    def test_zero_dim_results_are_arrays(self) -> None:
        # Regression: ufuncs on 0-d arrays return NumPy scalars, which leaked into ``data``,
        # so ``numpy()`` was not an ndarray and in-place writes raised TypeError.
        a = tg.Tensor(np.array(2.0), requires_grad=True)
        for result in (a * 3, a + a, tg.exp(a), a.sum(), -a):
            assert type(result.data) is np.ndarray and result.shape == ()
            assert isinstance(result.numpy(), np.ndarray)
            result.data[...] = 1.0  # writable, like any other tensor's data

    def test_numpy_defers_to_tensor_operators(self) -> None:
        t = tg.Tensor(np.ones(3), requires_grad=True)
        out = np.full(3, 2.0) * t  # ndarray on the left
        assert isinstance(out, tg.Tensor)
        out.sum().backward()
        np.testing.assert_allclose(t.grad, 2.0)

    def test_comparisons_are_not_differentiable(self) -> None:
        t = tg.Tensor([1.0, -1.0], requires_grad=True)
        mask = t > 0
        assert mask.dtype == np.bool_ and not mask.requires_grad
        assert (t <= 0).tolist() == [False, True]

    def test_equality_is_elementwise_and_hashing_is_by_identity(self) -> None:
        a, b = tg.Tensor([1.0, 2.0]), tg.Tensor([1.0, 3.0])
        assert (a == b).tolist() == [True, False]
        assert (a != b).tolist() == [False, True]
        assert (a == 1.0).any() and not (a == b).all()
        lookup = {a: "a", b: "b"}  # tensors remain usable as dict keys and set members
        assert lookup[a] == "a" and len({a, b, a}) == 2

    def test_numpy_interop(self) -> None:
        t = tg.Tensor(np.arange(3.0))
        assert np.asarray(t) is t.data  # no copy unless asked for
        assert np.array(t, copy=True) is not t.data
        assert np.asarray(t, dtype=np.float32).dtype == np.float32
        np.testing.assert_allclose(t, [0.0, 1.0, 2.0])
        assert float(tg.Tensor(2.5)) == 2.5 and int(tg.Tensor(np.array([7]))) == 7


class TestGradAttribute:
    def test_grad_is_a_constant_tensor(self) -> None:
        x = tg.Tensor(np.array([1.0, 2.0]), requires_grad=True)
        (x * x).sum().backward()
        assert isinstance(x.grad, tg.Tensor)
        assert not x.grad.requires_grad and x.grad.is_leaf
        np.testing.assert_allclose(x.grad.data, [2.0, 4.0])

    def test_assigning_an_array_wraps_it_and_checks_the_shape(self) -> None:
        x = tg.Tensor(np.zeros(3), requires_grad=True)
        x.grad = np.ones(3)
        assert isinstance(x.grad, tg.Tensor) and x.grad.tolist() == [1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="shape"):
            x.grad = np.ones(4)
        x.grad = None
        assert x.grad is None

    def test_accumulated_gradients_do_not_alias(self) -> None:
        # ``add`` hands the same upstream array to both inputs; each leaf must own a copy.
        a = tg.Tensor(np.zeros(2), requires_grad=True)
        b = tg.Tensor(np.zeros(2), requires_grad=True)
        (a + b).sum().backward()
        assert a.grad is not None and b.grad is not None
        assert not np.shares_memory(a.grad.data, b.grad.data)
        a.grad.data[0] = 5.0
        assert b.grad.data[0] == 1.0


class TestBackward:
    def test_simple_chain_rule(self) -> None:
        x = tg.Tensor(np.array(3.0), requires_grad=True)
        y = x * x * x + 2 * x  # dy/dx = 3x^2 + 2 = 29
        y.backward()
        assert x.grad == pytest.approx(29.0)

    def test_gradients_accumulate_across_backward_calls(self) -> None:
        x = tg.Tensor(np.array([1.0, 2.0]), requires_grad=True)
        (x * 3).sum().backward()
        (x * 3).sum().backward()
        np.testing.assert_allclose(x.grad, [6.0, 6.0])
        x.zero_grad()
        assert x.grad is None

    def test_diamond_graph_sums_both_paths(self) -> None:
        x = tg.Tensor(np.array(2.0), requires_grad=True)
        a = x * 3
        b = x * x
        (a + b).backward()  # d/dx (3x + x^2) = 3 + 2x = 7
        assert x.grad == pytest.approx(7.0)

    def test_same_tensor_used_twice_in_one_op(self) -> None:
        x = tg.Tensor(np.array([1.0, -2.0]), requires_grad=True)
        (x * x).sum().backward()
        np.testing.assert_allclose(x.grad, [2.0, -4.0])

    def test_broadcast_gradient_is_summed_to_input_shape(self) -> None:
        bias = tg.Tensor(np.zeros((1, 3)), requires_grad=True)
        x = tg.Tensor(np.ones((4, 2, 3)))
        (x + bias).sum().backward()
        assert bias.grad is not None and bias.grad.shape == (1, 3)
        np.testing.assert_allclose(bias.grad, 8.0)

    def test_explicit_upstream_gradient(self) -> None:
        x = tg.Tensor(np.ones((2, 2)), requires_grad=True)
        (x * 2).backward(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(x.grad, [[2.0, 4.0], [6.0, 8.0]])
        with pytest.raises(ValueError, match="shape"):
            (x * 2).backward(np.ones(3))

    def test_non_scalar_backward_requires_gradient(self) -> None:
        x = tg.Tensor(np.ones(3), requires_grad=True)
        with pytest.raises(RuntimeError, match="single-element"):
            (x * 2).backward()

    def test_backward_on_constant_raises(self) -> None:
        with pytest.raises(RuntimeError, match="does not require grad"):
            tg.Tensor(np.ones(1)).backward()

    def test_graph_is_freed_after_backward(self) -> None:
        x = tg.Tensor(np.ones(3), requires_grad=True)
        y = (x * 2).sum()
        y.backward()
        with pytest.raises(RuntimeError, match="second time"):
            y.backward()

    def test_retain_graph_allows_second_backward(self) -> None:
        x = tg.Tensor(np.ones(3), requires_grad=True)
        y = (x * 2).sum()
        y.backward(retain_graph=True)
        y.backward()
        np.testing.assert_allclose(x.grad, 4.0)

    def test_released_graph_drops_saved_arrays(self) -> None:
        x = tg.Tensor(np.ones(1000), requires_grad=True)
        hidden = x.exp()
        ref = weakref.ref(hidden)
        loss = hidden.sum()
        del hidden
        gc.collect()
        assert ref() is not None  # kept alive by the graph
        loss.backward()
        gc.collect()
        assert ref() is None  # freed together with the node's closure

    def test_non_leaf_grad_only_when_retained(self) -> None:
        x = tg.Tensor(np.ones(2), requires_grad=True)
        h = x * 3
        h.retain_grad()
        k = h * 2
        (k + h).sum().backward()
        np.testing.assert_allclose(h.grad, 3.0)
        assert k.grad is None
        np.testing.assert_allclose(x.grad, 9.0)

    def test_leaf_properties(self) -> None:
        x = tg.Tensor(np.ones(2), requires_grad=True)
        y = x * 2
        assert x.is_leaf and not y.is_leaf
        assert y.grad_fn is not None and y.grad_fn.op == "mul"
        with pytest.raises(RuntimeError, match="leaf"):
            y.requires_grad = False

    def test_constant_inputs_get_no_gradient(self) -> None:
        x = tg.Tensor(np.ones(2), requires_grad=True)
        c = tg.Tensor(np.ones(2))
        (x * c).sum().backward()
        assert c.grad is None

    def test_deep_graph_does_not_hit_recursion_limit(self) -> None:
        x = tg.Tensor(np.array(0.0), requires_grad=True)
        y = x
        for _ in range(5000):
            y = y + 1.0
        y.backward()
        assert x.grad == pytest.approx(1.0)

    def test_gradient_dtype_matches_parameter(self) -> None:
        w = tg.Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
        (w * np.ones(3, dtype=np.float64)).sum().backward()  # result promotes to float64
        assert w.grad is not None and w.grad.dtype == np.float32

    def test_integer_cast_cuts_the_graph(self) -> None:
        # Regression: astype(int) returned an integer tensor that still required grad, and
        # backward() then produced a gradient of 2 for a piecewise-constant function.
        x = tg.Tensor(np.array([1.5, 2.5]), requires_grad=True)
        ids = x.astype(np.int64)
        assert ids.dtype == np.int64 and not ids.requires_grad and ids.is_leaf
        with pytest.raises(RuntimeError, match="does not require grad"):
            (ids * 2).sum().backward()
        assert not x.astype(np.bool_).requires_grad
        assert x.astype(np.float32).requires_grad  # float casts stay differentiable


def recording_op(a: tg.Tensor, b: tg.Tensor, log: list[tuple[bool, ...]]) -> tg.Tensor:
    """``a * b`` whose backward records which input gradients the engine asked for."""
    from tensorgrad.ops._util import make_result

    def backward(g: np.ndarray, need: tuple[bool, ...]) -> tuple[np.ndarray | None, ...]:
        log.append(need)
        return (g * b.data if need[0] else None, g * a.data if need[1] else None)

    def graph(g: tg.Tensor, out: tg.Tensor, need: tuple[bool, ...]) -> tuple[tg.Tensor | None, ...]:
        log.append(need)
        return (g * b if need[0] else None, g * a if need[1] else None)

    return make_result(a.data * b.data, (a, b), backward, "recording_mul", graph=graph)


class TestFunctionalGrad:
    def test_returns_gradients_without_touching_dot_grad(self) -> None:
        x = tg.Tensor(np.array([1.0, 2.0]), requires_grad=True)
        w = tg.Tensor(np.array([3.0, 4.0]), requires_grad=True)
        (gx,) = tg.autograd.grad((x * w).sum(), x)
        assert isinstance(gx, tg.Tensor) and gx.tolist() == [3.0, 4.0]
        assert x.grad is None and w.grad is None

    def test_gradient_with_respect_to_an_intermediate_tensor(self) -> None:
        x = tg.Tensor(np.array(2.0), requires_grad=True)
        h = x * x  # dL/dh = 3h^2 = 48 for L = h^3
        gh, gx = tg.autograd.grad(h**3, [h, x])
        assert gh is not None and gx is not None
        assert gh.item() == pytest.approx(48.0)
        assert gx.item() == pytest.approx(48.0 * 2 * 2.0)  # chain rule through h

    def test_multiple_outputs_and_explicit_grad_outputs(self) -> None:
        x = tg.Tensor(np.array([1.0, -1.0]), requires_grad=True)
        (g,) = tg.autograd.grad([x * 2, x * x], x, grad_outputs=[np.ones(2), np.array([1.0, 3.0])])
        assert g is not None
        np.testing.assert_allclose(g.data, [2 + 2 * 1.0, 2 + 3 * 2 * -1.0])

    def test_only_the_needed_gradients_are_computed(self) -> None:
        x = tg.Tensor(np.array([1.0, 2.0]), requires_grad=True)
        w = tg.Tensor(np.array([3.0, 4.0]), requires_grad=True)
        log: list[tuple[bool, ...]] = []
        tg.autograd.grad(recording_op(x, w, log).sum(), x)
        assert log == [(True, False)]  # the weight gradient is never evaluated
        recording_op(x, w, log).sum().backward()
        assert log[-1] == (True, True)

    def test_unused_inputs(self) -> None:
        x = tg.Tensor(np.ones(2), requires_grad=True)
        unused = tg.Tensor(np.ones(2), requires_grad=True)
        with pytest.raises(RuntimeError, match="allow_unused"):
            tg.autograd.grad((x * 2).sum(), [x, unused])
        gx, gu = tg.autograd.grad((x * 2).sum(), [x, unused], allow_unused=True)
        assert gx is not None and gu is None

    def test_rejects_tensors_without_a_graph(self) -> None:
        x = tg.Tensor(np.ones(2), requires_grad=True)
        with pytest.raises(RuntimeError, match="input 0 does not require grad"):
            tg.autograd.grad((x * 2).sum(), tg.Tensor(np.ones(2)))
        with pytest.raises(RuntimeError, match="output 0 does not require grad"):
            tg.autograd.grad(tg.Tensor(np.ones(1)), x)
        with pytest.raises(RuntimeError, match="single-element"):
            tg.autograd.grad(x * 2, x)

    def test_graph_is_freed_unless_retained(self) -> None:
        x = tg.Tensor(np.ones(2), requires_grad=True)
        y = (x * x).sum()
        tg.autograd.grad(y, x, retain_graph=True)
        tg.autograd.grad(y, x)
        with pytest.raises(RuntimeError, match="second time"):
            tg.autograd.grad(y, x)

    def test_returned_gradients_are_fresh_arrays(self) -> None:
        # The sum's VJP is a read-only broadcast view of the seed; the caller gets a copy.
        x = tg.Tensor(np.zeros((2, 3)), requires_grad=True)
        (g,) = tg.autograd.grad(x.sum(), x)
        assert g is not None and g.data.flags.writeable and g.data.flags.owndata
        g.data[0, 0] = 7.0

    def test_backward_accepts_several_roots(self) -> None:
        x = tg.Tensor(np.array([1.0, 2.0]), requires_grad=True)
        tg.autograd.backward([(x * 3).sum(), (x * x).sum()])
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.data, [3 + 2.0, 3 + 4.0])


class TestGradMode:
    def test_no_grad_context(self) -> None:
        x = tg.Tensor(np.ones(2), requires_grad=True)
        with tg.no_grad():
            y = x * 2
            assert not tg.is_grad_enabled()
        assert tg.is_grad_enabled()
        assert not y.requires_grad and y.grad_fn is None

    def test_no_grad_decorator_and_nesting(self) -> None:
        @tg.no_grad()
        def infer(t: tg.Tensor) -> tg.Tensor:
            with tg.enable_grad():
                inner = t * 1
            assert inner.requires_grad
            return t * 2

        x = tg.Tensor(np.ones(2), requires_grad=True)
        assert not infer(x).requires_grad
        assert tg.is_grad_enabled()

    def test_no_grad_restores_state_after_exception(self) -> None:
        with pytest.raises(ValueError, match="boom"), tg.no_grad():
            raise ValueError("boom")
        assert tg.is_grad_enabled()

    def test_detach_cuts_the_graph(self) -> None:
        x = tg.Tensor(np.ones(2), requires_grad=True)
        d = (x * 2).detach()
        assert not d.requires_grad
        (d * x).sum().backward()
        np.testing.assert_allclose(x.grad, 2.0)  # no gradient through the detached branch

    def test_requires_grad_toggle(self) -> None:
        x = tg.Tensor(np.ones(2)).requires_grad_()
        assert x.requires_grad
        x.requires_grad_(False)
        assert not (x * 2).requires_grad


class TestGradcheckUtility:
    def test_detects_a_wrong_gradient(self) -> None:
        from tensorgrad.ops._util import as_tensor, make_result

        def bad_square(a: tg.Tensor) -> tg.Tensor:
            t = as_tensor(a)
            return make_result(
                t.data**2,
                (t,),
                lambda g, need: (g * t.data,),  # wrong: should be 2 * t
                "bad_square",
                graph=lambda g, out, need: (g * t,),
            )

        x = tg.Tensor(np.random.default_rng(0).standard_normal(4), requires_grad=True)
        with pytest.raises(GradcheckError, match="mismatch"):
            gradcheck(bad_square, [x])
        assert gradcheck(bad_square, [x], raise_exception=False) is False

    def test_rejects_float32_inputs(self) -> None:
        with pytest.raises(TypeError, match="float64"):
            gradcheck(lambda a: a * 2, [tg.Tensor([1.0], requires_grad=True)])

    def test_needs_an_input_that_requires_grad(self) -> None:
        with pytest.raises(ValueError, match="requires_grad"):
            gradcheck(lambda a: a * 2, [tg.Tensor(np.ones(2))])
