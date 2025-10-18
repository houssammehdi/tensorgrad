"""The :class:`Module` base class and :class:`Parameter`."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple, Self

import numpy as np

from tensorgrad._types import Array
from tensorgrad.tensor import Tensor

__all__ = ["IncompatibleKeys", "Module", "Parameter"]


class Parameter(Tensor):
    """A tensor that a :class:`Module` registers as trainable (``requires_grad`` by default)."""

    __slots__ = ()

    def __init__(self, data: object, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad)


class IncompatibleKeys(NamedTuple):
    """Result of a non-strict :meth:`Module.load_state_dict`."""

    missing_keys: list[str]
    unexpected_keys: list[str]


class Module:
    """Base class for layers and models.

    Assigning a :class:`Parameter`, a sub-:class:`Module` or a registered buffer to an
    attribute registers it, so :meth:`parameters`, :meth:`state_dict`, :meth:`train` and
    friends see the whole tree. Subclasses implement :meth:`forward` and are called directly.

    Example:
        >>> import tensorgrad as tg
        >>> from tensorgrad import nn
        >>> class Affine(nn.Module):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.scale = nn.Parameter([2.0])
        ...     def forward(self, x: tg.Tensor) -> tg.Tensor:
        ...         return x * self.scale
        >>> [name for name, _ in Affine().named_parameters()]
        ['scale']
    """

    training: bool

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    # ------------------------------------------------------------------ registration
    def __setattr__(self, name: str, value: object) -> None:
        state = self.__dict__
        if "_parameters" not in state:
            if isinstance(value, Parameter | Module):
                raise AttributeError("call Module.__init__() before assigning parameters/modules")
            object.__setattr__(self, name, value)
            return
        params, modules, buffers = state["_parameters"], state["_modules"], state["_buffers"]
        if name in buffers:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"buffer '{name}' must be a numpy array")
            buffers[name] = value
        else:
            params.pop(name, None)
            modules.pop(name, None)
            if isinstance(value, Parameter):
                params[name] = value
            elif isinstance(value, Module):
                modules[name] = value
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        for registry in ("_parameters", "_modules", "_buffers"):
            self.__dict__[registry].pop(name, None)
        object.__delattr__(self, name)

    def register_buffer(self, name: str, value: Array) -> None:
        """Register a non-trainable array (e.g. running statistics) saved in the state dict."""
        self.__dict__["_buffers"][name] = value
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------ traversal
    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Direct sub-modules with their attribute names."""
        yield from self.__dict__["_modules"].items()

    def children(self) -> Iterator[Module]:
        """Direct sub-modules."""
        for _, module in self.named_children():
            yield module

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        """This module and all descendants (each once), with dotted names."""
        seen: set[int] = set()

        def walk(module: Module, name: str) -> Iterator[tuple[str, Module]]:
            if id(module) in seen:
                return
            seen.add(id(module))
            yield name, module
            for child_name, child in module.named_children():
                yield from walk(child, f"{name}.{child_name}" if name else child_name)

        yield from walk(self, prefix)

    def modules(self) -> Iterator[Module]:
        """This module and all descendants."""
        for _, module in self.named_modules():
            yield module

    def _named_members(
        self, registry: str, prefix: str, recurse: bool, dedupe: bool
    ) -> Iterator[tuple[str, Any]]:
        seen: set[int] = set()

        def walk(module: Module, name: str) -> Iterator[tuple[str, Any]]:
            for key, value in module.__dict__[registry].items():
                if dedupe and id(value) in seen:
                    continue
                seen.add(id(value))
                yield (f"{name}.{key}" if name else key), value
            if recurse:
                for child_name, child in module.named_children():
                    yield from walk(child, f"{name}.{child_name}" if name else child_name)

        yield from walk(self, prefix)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        """Parameters with dotted names; a parameter shared by several modules appears once."""
        yield from self._named_members("_parameters", prefix, recurse, dedupe=True)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """All trainable parameters (shared ones once)."""
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, Array]]:
        """Registered buffers with dotted names."""
        yield from self._named_members("_buffers", prefix, recurse, dedupe=True)

    def num_parameters(self) -> int:
        """Total number of scalar parameters (shared parameters counted once)."""
        return sum(p.size for p in self.parameters())

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict[str, Array]:
        """Copies of every parameter and buffer, keyed by dotted name.

        Like PyTorch, a parameter shared by two modules (weight tying) appears under both
        names, so the dict mirrors the module structure exactly.
        """
        state: dict[str, Array] = {}
        for name, param in self._named_members("_parameters", "", True, dedupe=False):
            state[name] = param.data.copy()
        for name, buffer in self._named_members("_buffers", "", True, dedupe=False):
            state[name] = np.array(buffer, copy=True)
        return state

    def load_state_dict(self, state: Mapping[str, Array], strict: bool = True) -> IncompatibleKeys:
        """Copy arrays from ``state`` into this module's parameters and buffers (in place).

        Args:
            state: Mapping produced by :meth:`state_dict` (or :func:`tensorgrad.utils.load`).
            strict: Raise ``KeyError`` on missing or unexpected keys.

        Raises:
            ValueError: If a shape does not match.
        """
        targets: dict[str, Array] = {}
        for name, param in self._named_members("_parameters", "", True, dedupe=False):
            targets[name] = param.data
        for name, buffer in self._named_members("_buffers", "", True, dedupe=False):
            targets[name] = buffer
        missing = [k for k in targets if k not in state]
        unexpected = [k for k in state if k not in targets]
        if strict and (missing or unexpected):
            raise KeyError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
        for name, target in targets.items():
            if name not in state:
                continue
            value = np.asarray(state[name])
            if value.shape != target.shape:
                raise ValueError(
                    f"shape mismatch for '{name}': expected {target.shape}, got {value.shape}"
                )
            target[...] = value
        return IncompatibleKeys(missing, unexpected)

    # ------------------------------------------------------------------ modes / helpers
    def train(self, mode: bool = True) -> Self:
        """Put this module and all descendants into training (or evaluation) mode."""
        for module in self.modules():
            object.__setattr__(module, "training", mode)
        return self

    def eval(self) -> Self:
        """Equivalent to ``train(False)``: disables dropout, uses running statistics."""
        return self.train(False)

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter."""
        for param in self.parameters():
            param.grad = None

    def apply(self, fn: Callable[[Module], None]) -> Self:
        """Call ``fn`` on every descendant and then on ``self`` (e.g. custom initialisation)."""
        for module in reversed(list(self.modules())):
            fn(module)
        return self

    # ------------------------------------------------------------------ calling
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute the module's output. Subclasses must override this."""
        raise NotImplementedError(f"{type(self).__name__} does not implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """Extra text shown inside this module's ``repr`` (hyper-parameters)."""
        return ""

    def __repr__(self) -> str:
        children = list(self.named_children())
        head = f"{type(self).__name__}({self.extra_repr()}"
        if not children:
            return head + ")"
        lines = [head]
        for name, child in children:
            body = repr(child).replace("\n", "\n  ")
            lines.append(f"  ({name}): {body}")
        return "\n".join(lines) + "\n)"
