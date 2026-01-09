"""Gradient checkpointing: recompute activations in the backward pass instead of storing them.

    y = checkpoint(block, x)          # block is a Module: its parameters are found for you
    y = checkpoint(f, x, params=[w])  # any function of tensors, plus the tensors it closes over

The forward pass runs the function under :class:`~tensorgrad.no_grad`, so none of its
intermediate activations are kept; the result is a single graph node whose parents are the
inputs and the parameters. Its backward pass runs the function again with grad enabled and
backpropagates through that fresh graph. Checkpointing every block of a deep network keeps
only the block boundaries alive, at the cost of one extra forward pass per block.

The recomputation replays the global random generator's state from the forward pass, so
dropout masks are identical in both passes. Higher-order derivatives work: under
``create_graph=True`` the recomputed graph is kept and differentiated like any other.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable

from tensorgrad import autograd
from tensorgrad._random import get_rng, replaying
from tensorgrad._types import Array
from tensorgrad.autograd import enable_grad, no_grad
from tensorgrad.ops._util import as_tensor, make_result
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["checkpoint"]

Need = tuple[bool, ...]


def _alias(x: Tensor) -> Tensor:
    """An identity node on ``x``: gradients taken with respect to it flow on into ``x``."""
    return make_result(x.data, (x,), lambda g, need: (g,), "alias", graph=lambda g, y, need: (g,))


def checkpoint(
    fn: Callable[..., Tensor],
    *args: TensorLike,
    params: Iterable[Tensor] | None = None,
    preserve_rng_state: bool = True,
) -> Tensor:
    """``fn(*args)``, without keeping ``fn``'s intermediate activations for the backward pass.

    Args:
        fn: A function of tensors returning one tensor, or a :class:`~tensorgrad.nn.Module`.
        args: Its tensor arguments.
        params: Tensors that ``fn`` uses besides its arguments and that need gradients (leaf
            tensors such as parameters). Defaults to ``fn.parameters()`` if ``fn`` has that
            method, and to none otherwise.
        preserve_rng_state: Replay the global generator during recomputation so that random
            ops such as dropout make the same draws as in the forward pass.

    Raises:
        TypeError: If ``fn`` does not return a tensor.
        ValueError: If a parameter is not a leaf tensor.
        RuntimeError: In the backward pass, if ``fn`` used a tensor that requires grad but
            is neither an argument nor in ``params`` (its gradient would be lost).
    """
    inputs = [as_tensor(a) for a in args]
    if params is None:
        get_params = getattr(fn, "parameters", None)
        params = get_params() if callable(get_params) else ()
    weights = list(params)
    for p in weights:
        if p._node is not None:
            raise ValueError("checkpoint params must be leaf tensors (e.g. parameters)")
    snapshot = copy.deepcopy(get_rng()) if preserve_rng_state else None
    with no_grad():
        out = fn(*inputs)
    if not isinstance(out, Tensor):
        raise TypeError(f"checkpointed function must return a Tensor, got {type(out).__name__}")
    parents = (*inputs, *weights)

    def recompute(connect: bool) -> tuple[list[Tensor], Tensor]:
        """Rerun ``fn``: returns the tensors to differentiate with respect to (one per parent)
        and the output. With ``connect`` the inputs are identity nodes on the originals, so
        the result stays differentiable with respect to them; otherwise fresh leaves."""
        with enable_grad():
            if connect:
                leaves = [_alias(x) if x.requires_grad else x for x in inputs]
            else:
                leaves = [
                    Tensor._wrap(x.data).requires_grad_() if x.requires_grad else x for x in inputs
                ]
            if snapshot is None:
                y = fn(*leaves)
            else:  # a fresh copy each time, so repeated backward passes replay the same draws
                with replaying(copy.deepcopy(snapshot)):
                    y = fn(*leaves)
        _check_captures(y, [*leaves, *weights])
        return [*leaves, *weights], y

    def vjp(g: Tensor | Array, need: Need, create_graph: bool) -> list[Tensor | None]:
        wrt, y = recompute(connect=create_graph)
        wanted = [t for t, n in zip(wrt, need, strict=True) if n]
        grads: list[Tensor | None] = [None] * len(wrt)
        if not wanted or not y.requires_grad:
            return grads
        found = iter(
            autograd.grad(y, wanted, grad_outputs=g, create_graph=create_graph, allow_unused=True)
        )
        for i, n in enumerate(need):
            if n:
                grads[i] = next(found)
        return grads

    def backward(g: Array, need: Need) -> tuple[Array | None, ...]:
        return tuple(None if t is None else t.data for t in vjp(g, need, False))

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, ...]:
        return tuple(vjp(g, need, True))

    return make_result(out.data, parents, backward, "checkpoint", graph=graph)


def _check_captures(y: Tensor, allowed: list[Tensor]) -> None:
    """Raise if the recomputed graph reaches a tensor requiring grad that is not allowed."""
    known = {id(t) for t in allowed}
    stack, seen = [y], set()
    while stack:
        t = stack.pop()
        if id(t) in seen or id(t) in known:
            continue
        seen.add(id(t))
        if t._node is None:
            if t.requires_grad:
                raise RuntimeError(
                    f"the checkpointed function uses a tensor of shape {t.shape} that requires "
                    "grad but is neither an argument nor in params; its gradient would be lost"
                )
            continue
        stack.extend(p for p in t._node.parents if p.requires_grad)
