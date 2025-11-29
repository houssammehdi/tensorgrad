"""Optimisers, gradient clipping and step schedules against torch.optim on equal gradients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tensorgrad as tg  # noqa: E402
from tensorgrad import optim  # noqa: E402

pytestmark = pytest.mark.torch

STEPS = 25


def run_both(
    make_ours: Callable[[list[tg.Tensor]], Any],
    make_theirs: Callable[[list[torch.Tensor]], Any],
    *,
    clip: float | None = None,
) -> None:
    """Apply the same sequence of random gradients to two parameters in both frameworks."""
    rng = np.random.default_rng(0)
    init = [rng.standard_normal((3, 4)), rng.standard_normal(5)]
    ours = [tg.Tensor(a.copy(), requires_grad=True) for a in init]
    theirs = [torch.tensor(a.copy(), requires_grad=True) for a in init]
    opt_ours, opt_theirs = make_ours(ours), make_theirs(theirs)
    for _ in range(STEPS):
        grads = [rng.standard_normal(a.shape) * 3 for a in init]
        for p, g in zip(ours, grads, strict=True):
            p.grad = g.copy()
        for p, g in zip(theirs, grads, strict=True):
            p.grad = torch.tensor(g.copy())
        if clip is not None:
            norm_ours = optim.clip_grad_norm_(ours, clip)
            norm_theirs = torch.nn.utils.clip_grad_norm_(theirs, clip)
            assert norm_ours == pytest.approx(float(norm_theirs), rel=1e-12)
        opt_ours.step()
        opt_theirs.step()
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(a.data, b.detach().numpy(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("momentum", "nesterov", "weight_decay"),
    [(0.0, False, 0.0), (0.9, False, 0.0), (0.9, True, 0.01), (0.5, False, 0.1)],
)
def test_sgd(momentum: float, nesterov: bool, weight_decay: float) -> None:
    kwargs = {"lr": 0.05, "momentum": momentum, "nesterov": nesterov, "weight_decay": weight_decay}
    run_both(lambda ps: optim.SGD(ps, **kwargs), lambda ps: torch.optim.SGD(ps, **kwargs))


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_adam(weight_decay: float) -> None:
    kwargs = {"lr": 0.01, "betas": (0.8, 0.95), "eps": 1e-8, "weight_decay": weight_decay}
    run_both(lambda ps: optim.Adam(ps, **kwargs), lambda ps: torch.optim.Adam(ps, **kwargs))


def test_adamw_with_clipping() -> None:
    kwargs = {"lr": 0.01, "betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.1}
    run_both(
        lambda ps: optim.AdamW(ps, **kwargs),
        lambda ps: torch.optim.AdamW(ps, **kwargs),
        clip=1.0,
    )


def test_step_lr_schedule() -> None:
    p_ours, p_theirs = tg.Tensor(np.ones(1), requires_grad=True), torch.ones(1, requires_grad=True)
    opt_ours = optim.SGD([p_ours], lr=0.3)
    opt_theirs = torch.optim.SGD([p_theirs], lr=0.3)
    sched_ours = optim.StepLR(opt_ours, step_size=3, gamma=0.5)
    sched_theirs = torch.optim.lr_scheduler.StepLR(opt_theirs, step_size=3, gamma=0.5)
    for _ in range(10):
        assert sched_ours.get_last_lr() == pytest.approx(sched_theirs.get_last_lr())
        opt_theirs.step()  # torch warns if the scheduler steps before the optimiser
        sched_ours.step()
        sched_theirs.step()
