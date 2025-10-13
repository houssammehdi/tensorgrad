"""The reverse-mode autodiff engine.

A :class:`~tensorgrad.Tensor` produced by a differentiable op holds a :class:`Node` that
remembers the op's inputs (``parents``) and a closure mapping the gradient of the output to
the gradients of each input (the vector-Jacobian product). :func:`run_backward` walks this
dynamic graph in reverse topological order, accumulating gradients as it goes.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from tensorgrad._types import Array

if TYPE_CHECKING:
    from types import TracebackType

    from tensorgrad.tensor import Tensor

__all__ = [
    "BackwardFn",
    "Node",
    "enable_grad",
    "is_grad_enabled",
    "no_grad",
    "run_backward",
]

#: Vector-Jacobian product of one op: maps d(loss)/d(output) to d(loss)/d(input_i) for each
#: input, returning ``None`` for inputs that need no gradient.
BackwardFn = Callable[[Array], tuple[Array | None, ...]]

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _GradMode(threading.local):
    enabled: bool = True


_grad_mode = _GradMode()


def is_grad_enabled() -> bool:
    """Return ``True`` if ops currently record the graph needed for ``backward``."""
    return _grad_mode.enabled


class _GradModeContext:
    """Base for context managers / decorators that switch graph recording on or off."""

    _target: bool

    def __init__(self) -> None:
        self._previous: list[bool] = []

    def __enter__(self) -> None:
        self._previous.append(_grad_mode.enabled)
        _grad_mode.enabled = self._target

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _grad_mode.enabled = self._previous.pop()

    def __call__(self, fn: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with type(self)():
                return fn(*args, **kwargs)

        return wrapper


class no_grad(_GradModeContext):
    """Disable graph recording (context manager or decorator).

    Inside the block, op results never require grad, so no graph nodes (and no saved
    activations) are kept alive -- use it for evaluation, sampling and parameter updates.
    """

    _target = False


class enable_grad(_GradModeContext):
    """Re-enable graph recording inside a :class:`no_grad` region."""

    _target = True


class Node:
    """A recorded op in the dynamic graph: its inputs and its vector-Jacobian product."""

    __slots__ = ("_backward", "op", "parents")

    def __init__(self, op: str, parents: tuple[Tensor, ...], backward: BackwardFn) -> None:
        self.op = op
        self.parents = parents
        self._backward: BackwardFn | None = backward

    @property
    def released(self) -> bool:
        """``True`` once the saved state was freed by a backward pass without retain_graph."""
        return self._backward is None

    def apply(self, grad: Array) -> tuple[Array | None, ...]:
        """Map the output gradient to one gradient (or ``None``) per parent."""
        if self._backward is None:
            raise RuntimeError(
                f"Trying to backward through the graph a second time (op '{self.op}'); "
                "the saved intermediate values were freed. Pass retain_graph=True to the "
                "first backward() call if you need to backpropagate through it again."
            )
        return self._backward(grad)

    def release(self) -> None:
        """Drop the closure (and every array it captured) and the references to the parents."""
        self._backward = None
        self.parents = ()

    def __repr__(self) -> str:
        return f"<Node {self.op}>"


def _topological_order(root: Tensor) -> list[Tensor]:
    """Return every tensor ``root`` depends on (that requires grad), inputs before outputs.

    Iterative post-order DFS: graphs of deep models easily exceed Python's recursion limit.
    """
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        tensor, expanded = stack.pop()
        if expanded:
            order.append(tensor)
            continue
        if id(tensor) in visited:
            continue
        visited.add(id(tensor))
        stack.append((tensor, True))
        node = tensor._node
        if node is not None:
            for parent in node.parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))
    return order


def run_backward(root: Tensor, grad: Array, *, retain_graph: bool = False) -> None:
    """Backpropagate ``grad`` (d(loss)/d(root)) through the graph that produced ``root``.

    Gradients of leaves that require grad are accumulated into their ``.grad`` attribute;
    non-leaf tensors only keep theirs if :meth:`~tensorgrad.Tensor.retain_grad` was called.
    Unless ``retain_graph`` is set, every visited node frees its saved state afterwards.
    """
    order = _topological_order(root)
    pending: dict[int, Array] = {id(root): grad}
    for tensor in reversed(order):
        g = pending.pop(id(tensor), None)
        if g is None:
            continue
        node = tensor._node
        if node is None or tensor._retains_grad:
            tensor._accumulate_grad(g)
        if node is None:
            continue
        parent_grads = node.apply(g)
        if len(parent_grads) != len(node.parents):
            raise RuntimeError(
                f"op '{node.op}' returned {len(parent_grads)} gradients "
                f"for {len(node.parents)} inputs"
            )
        for parent, parent_grad in zip(node.parents, parent_grads, strict=True):
            if parent_grad is None or not parent.requires_grad:
                continue
            if parent_grad.shape != parent.shape:
                raise RuntimeError(
                    f"op '{node.op}' produced a gradient of shape {parent_grad.shape} "
                    f"for an input of shape {parent.shape}"
                )
            previous = pending.get(id(parent))
            pending[id(parent)] = parent_grad if previous is None else previous + parent_grad
        if not retain_graph:
            node.release()
