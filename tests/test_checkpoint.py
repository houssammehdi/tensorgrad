"""Gradient checkpointing: same values and derivatives as without it, less memory."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable

import numpy as np
import pytest

import tensorgrad as tg
from helpers import check_gradients, leaf
from tensorgrad import func, nn
from tensorgrad.utils import checkpoint


@pytest.fixture(autouse=True)
def _float64() -> object:
    tg.set_default_dtype(np.float64)
    yield
    tg.set_default_dtype(np.float32)


def block_with_dropout() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Dropout(0.3), nn.Linear(16, 4))


def gradients(
    model: nn.Module, x: tg.Tensor, forward: Callable[[tg.Tensor], tg.Tensor]
) -> list[np.ndarray]:
    model.zero_grad()
    x.zero_grad()
    tg.manual_seed(7)  # the same dropout masks for both variants
    (forward(x) ** 2).sum().backward()
    assert x.grad is not None
    return [p.grad.data.copy() for p in model.parameters() if p.grad is not None] + [
        x.grad.data.copy()
    ]


def test_values_and_gradients_match_the_plain_forward_pass() -> None:
    model = block_with_dropout()
    x = tg.randn((5, 4), requires_grad=True)
    plain = gradients(model, x, model)
    checkpointed = gradients(model, x, lambda t: checkpoint(model, t))
    assert len(plain) == len(checkpointed) == 5
    for a, b in zip(plain, checkpointed, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    tg.manual_seed(7)
    expected = model(x).data
    tg.manual_seed(7)
    np.testing.assert_allclose(checkpoint(model, x).data, expected)


def test_first_and_second_order_finite_differences() -> None:
    rng = np.random.default_rng(0)
    w = leaf(rng, 3, 3)
    fn = lambda x: tg.tanh(x @ w) * x
    assert check_gradients(lambda x, w_: checkpoint(fn, x, params=[w_]), [leaf(rng, 2, 3), w])


def test_hessian_vector_products_match() -> None:
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))
    x = tg.randn((4, 3))
    params = list(model.parameters())
    v = [tg.Tensor(np.random.default_rng(1).standard_normal(p.shape)) for p in params]

    def hvp(forward: Callable[[tg.Tensor], tg.Tensor]) -> list[np.ndarray]:
        grads = tg.autograd.grad(forward(x).sum(), params, create_graph=True)
        dot = sum(((g * u).sum() for g, u in zip(grads, v, strict=True)), tg.Tensor(0.0))
        hs = tg.autograd.grad(dot, params, allow_unused=True)  # the last bias: no curvature
        return [np.zeros(p.shape) if h is None else h.data for h, p in zip(hs, params, strict=True)]

    for a, b in zip(hvp(model), hvp(lambda t: checkpoint(model, t)), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_autograd_grad_through_a_checkpoint_leaves_dot_grad_alone() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    x = tg.randn((2, 3), requires_grad=True)
    params = list(model.parameters())
    expected = tg.autograd.grad(model(x).sum(), [x, *params])
    got = tg.autograd.grad(checkpoint(model, x).sum(), [x, *params])
    for a, b in zip(expected, got, strict=True):
        assert a is not None and b is not None
        np.testing.assert_allclose(a.data, b.data)
    assert all(p.grad is None for p in params)


def test_retained_graph_replays_the_same_dropout_masks() -> None:
    model = block_with_dropout()
    x = tg.randn((5, 4), requires_grad=True)
    loss = checkpoint(model, x).sum()
    loss.backward(retain_graph=True)
    assert x.grad is not None
    first = x.grad.data.copy()
    loss.backward()
    np.testing.assert_allclose(x.grad.data, 2 * first)


def test_tensors_used_but_not_declared_are_an_error() -> None:
    w = tg.randn((3, 3), requires_grad=True)
    out = checkpoint(lambda x: x @ w, tg.randn((2, 3), requires_grad=True))
    with pytest.raises(RuntimeError, match="neither an argument nor in params"):
        out.sum().backward()


def test_argument_validation() -> None:
    w = tg.randn((3, 3), requires_grad=True)
    with pytest.raises(ValueError, match="leaf"):
        checkpoint(lambda x: x, tg.randn(3), params=[w * 2])
    with pytest.raises(TypeError, match="must return a Tensor"):
        checkpoint(lambda x: x.data, tg.randn(3))  # type: ignore[arg-type, return-value]


def test_checkpointing_every_block_lowers_peak_memory() -> None:
    blocks = [nn.MLP([64, 512, 64], activation="gelu") for _ in range(8)]
    x = tg.randn((256, 64), requires_grad=True)

    def peak(use_checkpoint: bool) -> int:
        tracemalloc.start()
        h = x
        for block in blocks:
            h = checkpoint(block, h) if use_checkpoint else block(h)
        h.sum().backward()
        size = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return size

    plain, checkpointed = peak(False), peak(True)
    assert checkpointed < 0.5 * plain


def test_gpt_gradient_checkpointing_changes_memory_not_results() -> None:
    config = nn.GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.1)
    tg.manual_seed(0)
    model = nn.GPT(config)
    ids = np.random.default_rng(2).integers(0, 11, size=(3, 8))

    def run() -> tuple[float, list[np.ndarray]]:
        model.zero_grad()
        tg.manual_seed(3)
        loss = tg.cross_entropy(model(ids[:, :-1]), ids[:, 1:])
        loss.backward()
        return loss.item(), [p.grad.data.copy() for p in model.parameters()]  # type: ignore[union-attr]

    loss_plain, grads_plain = run()
    model.gradient_checkpointing = True
    loss_ckpt, grads_ckpt = run()
    assert loss_ckpt == pytest.approx(loss_plain, rel=1e-12)
    for a, b in zip(grads_plain, grads_ckpt, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
    with tg.no_grad():  # inference is unaffected
        model(ids)


def test_func_transforms_see_through_checkpoints() -> None:
    f = lambda x: checkpoint(lambda y: tg.tanh(y) ** 3, x).sum()
    x = tg.Tensor(np.array([0.3, -0.7]))
    t = np.tanh(x.data)
    np.testing.assert_allclose(func.grad(f)(x).data, 3 * t**2 * (1 - t**2))
