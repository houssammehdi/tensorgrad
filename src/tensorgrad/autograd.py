"""The reverse-mode autodiff engine.

A :class:`~tensorgrad.Tensor` produced by a differentiable op holds a :class:`Node` that
remembers the op's inputs (``parents``) and a closure mapping the gradient of the output to
the gradients of each input (the vector-Jacobian product, VJP). The engine walks this dynamic
graph in reverse topological order, accumulating gradients as it goes.

Two entry points share one engine:

* :func:`backward` (and :meth:`Tensor.backward <tensorgrad.Tensor.backward>`) accumulates
  gradients into the ``.grad`` of every leaf the outputs depend on;
* :func:`grad` returns the gradients with respect to chosen ``inputs`` (leaves or not)
  without touching any ``.grad``, and computes only the part of the graph that leads to
  them.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ParamSpec, TypeAlias, TypeVar

import numpy as np

from tensorgrad import profiler as _profiler
from tensorgrad._types import Array

if TYPE_CHECKING:
    from types import TracebackType

    from tensorgrad.tensor import Tensor

__all__ = [
    "BackwardFn",
    "GradOutputs",
    "Node",
    "backward",
    "enable_grad",
    "grad",
    "is_grad_enabled",
    "no_grad",
]

#: Vector-Jacobian product of one op on NumPy arrays: given d(loss)/d(output) and, per input,
#: whether its gradient is needed, return d(loss)/d(input_i) (``None`` where not needed).
BackwardFn = Callable[[Array, tuple[bool, ...]], tuple[Array | None, ...]]

#: The same VJP written with differentiable tensor ops, for ``create_graph=True``: it gets the
#: output gradient and the op's output tensor, and returns gradients that carry a graph.
GraphBackwardFn = Callable[["Tensor", "Tensor", tuple[bool, ...]], tuple["Tensor | None", ...]]

#: Upstream gradients accepted by :func:`grad` / :func:`backward`: one per output, where
#: ``None`` means "ones" (allowed for single-element outputs only).
GradOutputs: TypeAlias = "Tensor | Array | Sequence[Tensor | Array | None] | None"

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
    """A recorded op in the dynamic graph: its inputs and its vector-Jacobian products.

    ``backward`` works on NumPy arrays and is what an ordinary backward pass runs.
    ``graph_backward`` computes the same VJP with differentiable tensor ops; the engine uses
    it when asked to ``create_graph``, so that gradients can themselves be differentiated.
    """

    __slots__ = ("_backward", "_graph_backward", "op", "parents")

    def __init__(
        self,
        op: str,
        parents: tuple[Tensor, ...],
        backward: BackwardFn,
        graph_backward: GraphBackwardFn,
    ) -> None:
        self.op = op
        self.parents = parents
        self._backward: BackwardFn | None = backward
        self._graph_backward: GraphBackwardFn | None = graph_backward

    @property
    def released(self) -> bool:
        """``True`` once the saved state was freed by a backward pass without retain_graph."""
        return self._backward is None

    def check_alive(self) -> None:
        """Raise if a backward pass without ``retain_graph`` already freed this node."""
        if self._backward is None:
            raise self._freed_error()

    def _freed_error(self) -> RuntimeError:
        return RuntimeError(
            f"Trying to backward through the graph a second time (op '{self.op}'); "
            "the saved intermediate values were freed. Pass retain_graph=True to the "
            "first backward() call if you need to backpropagate through it again."
        )

    def apply(self, grad: Array, need: tuple[bool, ...]) -> tuple[Array | None, ...]:
        """Map the output gradient to one gradient (or ``None``) per parent."""
        self.check_alive()
        assert self._backward is not None
        return self._backward(grad, need)

    def apply_graph(
        self, grad: Tensor, output: Tensor, need: tuple[bool, ...]
    ) -> tuple[Tensor | None, ...]:
        """Differentiable version of :meth:`apply`: the returned gradients carry a graph."""
        self.check_alive()
        assert self._graph_backward is not None
        return self._graph_backward(grad, output, need)

    def release(self) -> None:
        """Drop the closures (and every array they captured) and the references to parents."""
        self._backward = None
        self._graph_backward = None
        self.parents = ()

    def __repr__(self) -> str:
        return f"<Node {self.op}>"


# ---------------------------------------------------------------------------- engine
def _topological_order(roots: Sequence[Tensor]) -> list[Tensor]:
    """Return every tensor the roots depend on (that requires grad), inputs before outputs.

    Iterative post-order DFS: graphs of deep models easily exceed Python's recursion limit.
    """
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False) for root in roots]
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


def _leads_to(order: list[Tensor], targets: set[int]) -> set[int]:
    """Ids of the tensors in ``order`` that are a target or have a target among their inputs.

    Gradients only need to flow into these tensors; ``order`` lists inputs before outputs,
    so one forward sweep suffices.
    """
    live: set[int] = set()
    for tensor in order:
        node = tensor._node
        if id(tensor) in targets or (node is not None and any(id(p) in live for p in node.parents)):
            live.add(id(tensor))
    return live


GradValue: TypeAlias = "Array | Tensor"


def _run_backward(
    roots: Sequence[Tensor],
    seeds: Sequence[GradValue],
    *,
    retain_graph: bool,
    create_graph: bool,
    targets: Sequence[Tensor] | None,
) -> dict[int, GradValue]:
    """Backpropagate ``seeds`` (d(out)/d(root) for each root) through the recorded graph.

    Without ``targets`` the gradients of leaves (and of non-leaves that called
    ``retain_grad``) are accumulated into their ``.grad``. With ``targets`` nothing is
    accumulated; instead the total gradient of each target is returned, keyed by ``id``, and
    only nodes that lead to a target are evaluated. Unless ``retain_graph``, every evaluated
    node frees its saved state. With ``create_graph`` the seeds are tensors and every VJP is
    built from differentiable ops (grad mode on), so the returned gradients have a graph.
    """
    prof = _profiler.active()
    if prof is None:
        return _backward_pass(roots, seeds, retain_graph, create_graph, targets, None)
    return prof.timed(
        _profiler.ENGINE,
        True,
        _backward_pass,
        roots,
        seeds,
        retain_graph,
        create_graph,
        targets,
        prof,
    )


def _backward_pass(
    roots: Sequence[Tensor],
    seeds: Sequence[GradValue],
    retain_graph: bool,
    create_graph: bool,
    targets: Sequence[Tensor] | None,
    prof: _profiler.Profile | None,
) -> dict[int, GradValue]:
    """The body of :func:`_run_backward`; VJP calls are timed when ``prof`` is given."""
    order = _topological_order(roots)
    target_ids = None if targets is None else {id(t) for t in targets}
    live = None if target_ids is None else _leads_to(order, target_ids)
    pending: dict[int, GradValue] = {}
    for root, seed in zip(roots, seeds, strict=True):
        previous = pending.get(id(root))
        pending[id(root)] = seed if previous is None else previous + seed
    captured: dict[int, GradValue] = {}

    with enable_grad() if create_graph else contextlib.nullcontext():
        for tensor in reversed(order):
            g = pending.pop(id(tensor), None)
            if g is None:
                continue
            node = tensor._node
            if target_ids is not None:
                if id(tensor) in target_ids:
                    captured[id(tensor)] = g
            elif node is None or tensor._retains_grad:
                tensor._accumulate_grad(g)
            if node is None:
                continue
            backward_fn = node._backward
            if backward_fn is None:  # freed: it has no parents left, so fail loudly
                raise node._freed_error()
            parents = node.parents
            # List comprehensions over the slot: this loop is the hot path of every step.
            if live is None:
                need = tuple([p._requires_grad for p in parents])
            else:
                need = tuple([p._requires_grad and id(p) in live for p in parents])
            if not any(need):
                continue
            parent_grads: tuple[GradValue | None, ...]
            if create_graph:
                if prof is None:
                    parent_grads = node.apply_graph(g, tensor, need)  # type: ignore[arg-type]
                else:
                    parent_grads = prof.timed(node.op, True, node.apply_graph, g, tensor, need)  # type: ignore[arg-type]
            elif prof is None:
                parent_grads = backward_fn(g, need)  # type: ignore[arg-type]
            else:
                parent_grads = prof.timed(node.op, True, backward_fn, g, need)  # type: ignore[arg-type]
            if len(parent_grads) != len(parents):
                raise RuntimeError(
                    f"op '{node.op}' returned {len(parent_grads)} gradients "
                    f"for {len(parents)} inputs"
                )
            for parent, parent_grad, wanted in zip(parents, parent_grads, need, strict=True):
                if parent_grad is None or not wanted:
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
    return captured


# ---------------------------------------------------------------------------- public API
def _as_tensors(value: Tensor | Sequence[Tensor], what: str) -> tuple[Tensor, ...]:
    from tensorgrad.tensor import Tensor

    items = (value,) if isinstance(value, Tensor) else tuple(value)
    if not items:
        raise ValueError(f"{what} must contain at least one tensor")
    for item in items:
        if not isinstance(item, Tensor):
            raise TypeError(f"{what} must be tensors, got {type(item).__name__}")
    return items


def _seeds(
    outputs: tuple[Tensor, ...], grad_outputs: GradOutputs, create_graph: bool
) -> list[GradValue]:
    """One upstream gradient per output: arrays normally, tensors when building a graph."""
    from tensorgrad.tensor import Tensor

    if grad_outputs is None:
        given: list[Tensor | Array | None] = [None] * len(outputs)
    elif isinstance(grad_outputs, Tensor | np.ndarray):
        given = [grad_outputs]
    else:
        given = list(grad_outputs)
    if len(given) != len(outputs):
        raise ValueError(f"got {len(given)} grad_outputs for {len(outputs)} outputs")
    seeds: list[GradValue] = []
    for out, g in zip(outputs, given, strict=True):
        if g is None:
            if out.size != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for single-element outputs; "
                    f"got shape {out.shape}"
                )
            ones = np.ones_like(out.data)
            seeds.append(Tensor._wrap(ones) if create_graph else ones)
            continue
        if isinstance(g, Tensor) and create_graph:
            # The seed may itself require grad: that is how the double-VJP trick works.
            seed: GradValue = g
        else:
            arr = np.asarray(g.data if isinstance(g, Tensor) else g, dtype=out.dtype)
            seed = Tensor._wrap(arr) if create_graph else arr
        if seed.shape != out.shape:
            raise ValueError(f"grad has shape {seed.shape}, expected {out.shape}")
        seeds.append(seed)
    return seeds


def backward(
    tensors: Tensor | Sequence[Tensor],
    grad_tensors: GradOutputs = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
) -> None:
    """Accumulate d(sum of ``tensors``)/d(leaf) into the ``.grad`` of every leaf.

    Args:
        tensors: Outputs to differentiate (each must require grad).
        grad_tensors: Upstream gradient per output (``None`` = ones, single elements only).
        retain_graph: Keep the graph's saved values so it can be backpropagated again.
            Defaults to ``create_graph``.
        create_graph: Build the graph of the backward pass itself, so the accumulated
            ``.grad`` tensors can be differentiated again (higher-order derivatives).
    """
    outputs = _as_tensors(tensors, "tensors")
    for out in outputs:
        if not out.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
    seeds = _seeds(outputs, grad_tensors, create_graph)
    retain = create_graph if retain_graph is None else retain_graph
    _run_backward(outputs, seeds, retain_graph=retain, create_graph=create_graph, targets=None)


def grad(
    outputs: Tensor | Sequence[Tensor],
    inputs: Tensor | Sequence[Tensor],
    grad_outputs: GradOutputs = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[Tensor | None, ...]:
    """Gradients of ``outputs`` with respect to ``inputs``, returned instead of accumulated.

    Computes ``sum_i grad_outputs[i]^T d outputs[i] / d inputs[j]`` for every input ``j``.
    ``inputs`` may be leaves or intermediate tensors; no ``.grad`` attribute is modified, and
    only the part of the graph between the outputs and the inputs is evaluated.

    Args:
        outputs: Tensors to differentiate (each must require grad).
        inputs: Tensors to differentiate with respect to (each must require grad).
        grad_outputs: Upstream gradient per output (``None`` = ones, single elements only).
            With ``create_graph`` these may require grad themselves; the result is then
            differentiable with respect to them too (the basis of the double-VJP trick).
        retain_graph: Keep the graph so it can be backpropagated again. Defaults to
            ``create_graph``.
        create_graph: Return gradients that carry their own graph, so they can be
            differentiated again (Hessian-vector products, gradient penalties, MAML...).
        allow_unused: Return ``None`` for inputs the outputs do not depend on, instead of
            raising.

    Returns:
        One gradient per input, with the input's shape and dtype.
    """
    from tensorgrad.tensor import Tensor

    outs = _as_tensors(outputs, "outputs")
    ins = _as_tensors(inputs, "inputs")
    for i, out in enumerate(outs):
        if not out.requires_grad:
            raise RuntimeError(f"output {i} does not require grad and has no graph")
    for i, inp in enumerate(ins):
        if not inp.requires_grad:
            raise RuntimeError(f"input {i} does not require grad")
    seeds = _seeds(outs, grad_outputs, create_graph)
    retain = create_graph if retain_graph is None else retain_graph
    captured = _run_backward(
        outs, seeds, retain_graph=retain, create_graph=create_graph, targets=ins
    )
    result: list[Tensor | None] = []
    for i, inp in enumerate(ins):
        g = captured.get(id(inp))
        if g is None:
            if not allow_unused:
                raise RuntimeError(
                    f"input {i} was not used to compute the outputs; pass allow_unused=True "
                    "to get None for it instead"
                )
            result.append(None)
        elif isinstance(g, Tensor):
            result.append(g if g.dtype == inp.dtype else g.astype(inp.dtype))
        else:
            # A fresh array: the engine may hand back read-only broadcast views or aliases.
            result.append(Tensor._wrap(np.array(g, dtype=inp.dtype, copy=True)))
    return tuple(result)
