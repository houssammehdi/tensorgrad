"""Learning-rate schedules that rescale each parameter group's ``initial_lr``."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from tensorgrad.optim.optimizer import Optimizer

__all__ = ["CosineWarmupLR", "LRScheduler", "StepLR"]


class LRScheduler(ABC):
    """Sets ``group.lr = group.initial_lr * factor(step)``; call :meth:`step` once per update.

    The factor for step 0 is applied on construction.
    """

    def __init__(self, optimizer: Optimizer[Any]) -> None:
        self.optimizer = optimizer
        self.last_step = 0
        self._apply()

    @abstractmethod
    def factor(self, step: int) -> float:
        """Multiplier of the initial learning rate at ``step``."""

    def step(self) -> None:
        """Advance the schedule by one step."""
        self.last_step += 1
        self._apply()

    def get_last_lr(self) -> list[float]:
        """Current learning rate of every parameter group."""
        return [group.lr for group in self.optimizer.param_groups]

    def _apply(self) -> None:
        f = self.factor(self.last_step)
        for group in self.optimizer.param_groups:
            group.lr = group.initial_lr * f


class StepLR(LRScheduler):
    """Multiply the learning rate by ``gamma`` every ``step_size`` steps."""

    def __init__(self, optimizer: Optimizer[Any], step_size: int, gamma: float = 0.1) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def factor(self, step: int) -> float:
        return float(self.gamma ** (step // self.step_size))


class CosineWarmupLR(LRScheduler):
    """Linear warm-up followed by cosine decay.

    For ``step < warmup_steps`` the factor rises linearly as ``(step + 1) / warmup_steps``;
    it then follows half a cosine from 1 down to ``min_lr_ratio`` at ``total_steps`` and
    stays there.
    """

    def __init__(
        self,
        optimizer: Optimizer[Any],
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ) -> None:
        if warmup_steps < 0 or total_steps <= 0 or warmup_steps > total_steps:
            raise ValueError("need 0 <= warmup_steps <= total_steps and total_steps > 0")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        span = max(self.total_steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / span, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
