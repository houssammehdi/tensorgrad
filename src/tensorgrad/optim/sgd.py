"""Stochastic gradient descent with momentum."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tensorgrad._types import Array
from tensorgrad.optim.optimizer import Optimizer, ParamGroup, ParamGroupSpec
from tensorgrad.tensor import Tensor

__all__ = ["SGD"]


@dataclass
class _SGDState:
    momentum_buffer: Array | None = None


class SGD(Optimizer[_SGDState]):
    """SGD with optional (Nesterov) momentum and L2 weight decay.

    Per step, with gradient ``g``::

        g = g + weight_decay * p
        b = momentum * b + g                   (b = g on the first step)
        d = g + momentum * b  if nesterov else b
        p = p - lr * d
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[ParamGroupSpec],
        lr: float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        if momentum < 0:
            raise ValueError(f"invalid momentum {momentum}")
        if nesterov and momentum == 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov

    def _init_state(self, param: Tensor) -> _SGDState:
        return _SGDState()

    def _update(self, param: Tensor, grad: Array, group: ParamGroup, state: _SGDState) -> None:
        if group.weight_decay:
            grad = grad + group.weight_decay * param.data
        if self.momentum:
            buf = state.momentum_buffer
            buf = grad.copy() if buf is None else self.momentum * buf + grad
            state.momentum_buffer = buf
            grad = grad + self.momentum * buf if self.nesterov else buf
        param.data -= group.lr * grad
