"""Adam and AdamW."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from tensorgrad._types import Array
from tensorgrad.optim.optimizer import Optimizer, ParamGroup, ParamGroupSpec
from tensorgrad.tensor import Tensor

__all__ = ["Adam", "AdamW"]


@dataclass
class _AdamState:
    exp_avg: Array
    exp_avg_sq: Array
    step: int = 0


class Adam(Optimizer[_AdamState]):
    """Adam (Kingma & Ba, 2015) with bias correction.

    Per step ``t`` with gradient ``g``::

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        p = p - lr * (m / (1 - b1**t)) / (sqrt(v / (1 - b2**t)) + eps)

    ``weight_decay`` here is classic L2 regularisation (added to ``g``); use :class:`AdamW`
    for decoupled weight decay.
    """

    decoupled_weight_decay = False

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[ParamGroupSpec],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas {betas}")
        if eps < 0:
            raise ValueError(f"invalid eps {eps}")
        super().__init__(params, lr, weight_decay)
        self.betas = betas
        self.eps = eps

    def _init_state(self, param: Tensor) -> _AdamState:
        return _AdamState(np.zeros_like(param.data), np.zeros_like(param.data))

    def _update(self, param: Tensor, grad: Array, group: ParamGroup, state: _AdamState) -> None:
        beta1, beta2 = self.betas
        if group.weight_decay:
            if self.decoupled_weight_decay:
                param.data *= 1.0 - group.lr * group.weight_decay
            else:
                grad = grad + group.weight_decay * param.data
        state.step += 1
        state.exp_avg *= beta1
        state.exp_avg += (1.0 - beta1) * grad
        state.exp_avg_sq *= beta2
        state.exp_avg_sq += (1.0 - beta2) * grad * grad
        bias_correction1 = 1.0 - beta1**state.step
        bias_correction2 = 1.0 - beta2**state.step
        denom = np.sqrt(state.exp_avg_sq) / math.sqrt(bias_correction2) + self.eps
        param.data -= (group.lr / bias_correction1) * state.exp_avg / denom


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    The parameters shrink by ``lr * weight_decay`` directly, instead of the decay being
    folded into the gradient where Adam's per-coordinate scaling would distort it.
    """

    decoupled_weight_decay = True

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[ParamGroupSpec],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay)
