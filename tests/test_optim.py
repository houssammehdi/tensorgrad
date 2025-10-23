"""Optimisers, gradient clipping and learning-rate schedules against hand-computed values."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import optim


def param(values: list[float]) -> tg.Tensor:
    return tg.Tensor(np.array(values, dtype=np.float64), requires_grad=True)


def run_steps(opt: optim.Optimizer[object], p: tg.Tensor, grads: list[list[float]]) -> None:
    for g in grads:
        p.grad = np.array(g, dtype=np.float64)
        opt.step()


GRADS = [[0.5, -1.0], [0.1, 0.3], [-0.4, 0.2]]


class TestSGD:
    def test_plain_sgd(self) -> None:
        p = param([1.0, 2.0])
        run_steps(optim.SGD([p], lr=0.1), p, GRADS)
        expected = np.array([1.0, 2.0]) - 0.1 * np.sum(GRADS, axis=0)
        np.testing.assert_allclose(p.data, expected)

    @pytest.mark.parametrize("nesterov", [False, True])
    def test_momentum_and_weight_decay_reference(self, nesterov: bool) -> None:
        lr, mu, wd = 0.1, 0.9, 0.01
        p = param([1.0, 2.0])
        run_steps(optim.SGD([p], lr=lr, momentum=mu, nesterov=nesterov, weight_decay=wd), p, GRADS)
        ref, buf = np.array([1.0, 2.0]), None
        for g in GRADS:
            d = np.array(g) + wd * ref
            buf = d.copy() if buf is None else mu * buf + d
            ref = ref - lr * (d + mu * buf if nesterov else buf)
        np.testing.assert_allclose(p.data, ref)

    def test_invalid_hyperparameters(self) -> None:
        p = param([1.0])
        with pytest.raises(ValueError, match="Nesterov"):
            optim.SGD([p], nesterov=True)
        with pytest.raises(ValueError, match="learning rate"):
            optim.SGD([p], lr=-1)
        with pytest.raises(ValueError, match="momentum"):
            optim.SGD([p], momentum=-0.5)


class TestAdam:
    def test_first_step_moves_each_coordinate_by_lr(self) -> None:
        # With bias correction, step 1 is lr * g / (|g| + eps) = lr * sign(g).
        p = param([1.0, 2.0, 3.0])
        p.grad = np.array([0.5, -2.0, 1e-3])
        optim.Adam([p], lr=0.01).step()
        np.testing.assert_allclose(p.data, [0.99, 2.01, 2.99], rtol=1e-6)

    def test_matches_hand_computed_reference(self) -> None:
        lr, (b1, b2), eps = 0.05, (0.8, 0.95), 1e-8
        p = param([1.0, -1.0])
        run_steps(optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps), p, GRADS)
        # Hand-unrolled Adam, written out coordinate by coordinate.
        ref = [1.0, -1.0]
        for i in range(2):
            m = v = 0.0
            x = ref[i]
            for t, g in enumerate((row[i] for row in GRADS), start=1):
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                x -= lr * m_hat / (math.sqrt(v_hat) + eps)
            ref[i] = x
        np.testing.assert_allclose(p.data, ref, rtol=1e-7)

    def test_adam_l2_versus_adamw_decoupled_decay(self) -> None:
        g = [[0.0, 0.0]]  # zero gradient isolates the weight-decay behaviour
        adam_p, adamw_p = param([1.0, -2.0]), param([1.0, -2.0])
        run_steps(optim.Adam([adam_p], lr=0.1, weight_decay=0.5), adam_p, g)
        run_steps(optim.AdamW([adamw_p], lr=0.1, weight_decay=0.5), adamw_p, g)
        # L2: the decay term 0.5 * p goes through Adam's normalisation -> step of lr * sign(p).
        np.testing.assert_allclose(adam_p.data, [0.9, -1.9], rtol=1e-6)
        # Decoupled: p *= (1 - lr * wd) and the zero gradient contributes nothing.
        np.testing.assert_allclose(adamw_p.data, [0.95, -1.9])

    def test_state_is_per_parameter(self) -> None:
        a, b = param([1.0]), param([1.0])
        opt = optim.Adam([a, b], lr=0.1)
        a.grad = np.array([1.0])
        opt.step()  # b has no gradient: untouched, no state
        assert b.data[0] == 1.0
        assert len(opt.state) == 1

    def test_invalid_betas(self) -> None:
        with pytest.raises(ValueError, match="betas"):
            optim.Adam([param([1.0])], betas=(1.0, 0.9))


class TestParamGroups:
    def test_group_overrides(self) -> None:
        a, b = param([1.0]), param([1.0])
        opt = optim.SGD([{"params": [a]}, {"params": [b], "lr": 1.0}], lr=0.1)
        a.grad = b.grad = np.array([1.0])
        opt.step()
        assert a.data[0] == pytest.approx(0.9)
        assert b.data[0] == pytest.approx(0.0)

    def test_zero_grad(self) -> None:
        p = param([1.0])
        p.grad = np.array([1.0])
        opt = optim.SGD([p])
        opt.zero_grad()
        assert p.grad is None

    def test_rejects_bad_parameter_lists(self) -> None:
        p = param([1.0])
        with pytest.raises(ValueError, match="empty"):
            optim.SGD([])
        with pytest.raises(ValueError, match="more than one"):
            optim.SGD([{"params": [p]}, {"params": [p]}])
        with pytest.raises(TypeError, match="requires_grad"):
            optim.SGD([tg.Tensor(np.ones(1))])

    def test_float32_parameters_stay_float32(self) -> None:
        p = tg.Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
        p.grad = np.ones(3, dtype=np.float64)
        optim.Adam([p]).step()
        assert p.dtype == np.float32


class TestClipGradNorm:
    def test_scales_down_to_max_norm(self) -> None:
        a, b = param([3.0]), param([0.0])
        a.grad, b.grad = np.array([3.0]), np.array([4.0])
        total = optim.clip_grad_norm_([a, b], max_norm=1.0)
        assert total == pytest.approx(5.0)
        assert a.grad is not None and b.grad is not None
        np.testing.assert_allclose([a.grad[0], b.grad[0]], [0.6, 0.8], rtol=1e-5)

    def test_leaves_small_gradients_alone(self) -> None:
        a = param([1.0])
        a.grad = np.array([0.5])
        assert optim.clip_grad_norm_([a, param([2.0])], max_norm=1.0) == pytest.approx(0.5)
        np.testing.assert_array_equal(a.grad, [0.5])


class TestSchedulers:
    def test_step_lr(self) -> None:
        opt = optim.SGD([param([1.0])], lr=1.0)
        sched = optim.StepLR(opt, step_size=2, gamma=0.5)
        lrs = []
        for _ in range(6):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        assert lrs == [1.0, 1.0, 0.5, 0.5, 0.25, 0.25]

    def test_cosine_with_warmup(self) -> None:
        opt = optim.AdamW([param([1.0])], lr=0.1)
        sched = optim.CosineWarmupLR(opt, warmup_steps=4, total_steps=12, min_lr_ratio=0.1)
        lrs = []
        for _ in range(14):
            lrs.append(opt.param_groups[0].lr)
            sched.step()
        np.testing.assert_allclose(lrs[:4], [0.025, 0.05, 0.075, 0.1])
        assert lrs[4] == pytest.approx(0.1)  # peak right after warm-up
        assert lrs[8] == pytest.approx(0.01 + 0.09 * 0.5)  # halfway through the cosine
        assert lrs[12] == pytest.approx(0.01)
        assert lrs[13] == pytest.approx(0.01)  # stays at the floor
        assert all(x >= y for x, y in itertools.pairwise(lrs[4:]))

    def test_scheduler_respects_group_lrs(self) -> None:
        opt = optim.SGD([{"params": [param([1.0])], "lr": 1.0}, {"params": [param([1.0])]}], lr=0.1)
        sched = optim.StepLR(opt, step_size=1, gamma=0.1)
        sched.step()
        np.testing.assert_allclose(sched.get_last_lr(), [0.1, 0.01])

    def test_invalid_schedules(self) -> None:
        opt = optim.SGD([param([1.0])])
        with pytest.raises(ValueError, match="warmup"):
            optim.CosineWarmupLR(opt, warmup_steps=5, total_steps=3)
        with pytest.raises(ValueError, match="step_size"):
            optim.StepLR(opt, step_size=0)
