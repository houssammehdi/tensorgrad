"""Optimizer base class and parameter groups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic, NotRequired, TypedDict, TypeVar, cast

from tensorgrad._types import Array
from tensorgrad.tensor import Tensor

__all__ = ["Optimizer", "ParamGroup", "ParamGroupSpec"]


class ParamGroupSpec(TypedDict):
    """User-facing description of a parameter group with optional per-group overrides."""

    params: Iterable[Tensor]
    lr: NotRequired[float]
    weight_decay: NotRequired[float]


@dataclass(eq=False)
class ParamGroup:
    """Parameters sharing a learning rate and weight decay.

    ``initial_lr`` remembers the learning rate at construction; LR schedulers scale it.
    """

    params: list[Tensor]
    lr: float
    weight_decay: float
    initial_lr: float = field(init=False)

    def __post_init__(self) -> None:
        self.initial_lr = self.lr


StateT = TypeVar("StateT")


class Optimizer(ABC, Generic[StateT]):
    """Base class: owns parameter groups and per-parameter state, updates ``param.data``.

    Args:
        params: Either an iterable of tensors, or of :class:`ParamGroupSpec` dicts
            (``{"params": [...], "lr": ..., "weight_decay": ...}``) for per-group settings.
        lr: Default learning rate.
        weight_decay: Default weight decay.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[ParamGroupSpec],
        lr: float,
        weight_decay: float,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate {lr}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight decay {weight_decay}")
        items = list(params)
        if not items:
            raise ValueError("optimizer got an empty parameter list")
        specs: list[ParamGroupSpec]
        if all(isinstance(item, dict) for item in items):
            specs = cast(list[ParamGroupSpec], items)
        elif any(isinstance(item, dict) for item in items):
            raise TypeError("mix of tensors and parameter-group dicts")
        else:
            specs = [{"params": cast(list[Tensor], items)}]
        self.param_groups: list[ParamGroup] = []
        seen: set[int] = set()
        for spec in specs:
            group_params = list(spec["params"])
            for p in group_params:
                if not isinstance(p, Tensor) or not p.requires_grad:
                    raise TypeError("optimizers can only update tensors with requires_grad=True")
                if id(p) in seen:
                    raise ValueError("a parameter appears in more than one parameter group")
                seen.add(id(p))
            self.param_groups.append(
                ParamGroup(
                    group_params,
                    lr=spec.get("lr", lr),
                    weight_decay=spec.get("weight_decay", weight_decay),
                )
            )
        self.state: dict[int, StateT] = {}

    def zero_grad(self) -> None:
        """Reset the gradient of every managed parameter."""
        for group in self.param_groups:
            for p in group.params:
                p.grad = None

    def step(self) -> None:
        """Apply one update to every parameter that has a gradient."""
        for group in self.param_groups:
            for p in group.params:
                if p.grad is None:
                    continue
                state = self.state.get(id(p))
                if state is None:
                    state = self.state[id(p)] = self._init_state(p)
                self._update(p, p.grad.astype(p.dtype, copy=False), group, state)

    @abstractmethod
    def _init_state(self, param: Tensor) -> StateT:
        """Create the per-parameter state before the first update."""

    @abstractmethod
    def _update(self, param: Tensor, grad: Array, group: ParamGroup, state: StateT) -> None:
        """Update ``param.data`` in place."""
