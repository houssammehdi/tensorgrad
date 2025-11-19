"""Function transforms built on the reverse-mode engine: grad, vjp, jvp, jacobian, hessian.

The transforms take a Python function of tensors and return a new function (or evaluate it
at a point), in the style of JAX and ``torch.func``::

    import tensorgrad as tg
    from tensorgrad import func

    f = lambda x: (x ** 3).sum()
    x = tg.Tensor([1.0, 2.0])
    func.grad(f)(x)               # 3 x^2
    func.hessian(f)(x)            # diag(6 x)
    func.hvp(f, (x,), (v,))       # (f(x), (H v,))

Two rules make them compose (``hessian`` is literally ``jacobian(grad(f))``):

* **Results are differentiable exactly when grad mode is on.** Inside the transform the
  function always runs with grad enabled; whether the *returned* values carry a graph (so an
  outer transform or ``backward`` can differentiate them again) follows
  :func:`~tensorgrad.is_grad_enabled` at call time. Call a transform under
  :class:`~tensorgrad.no_grad` when you only need numbers.
* **Each differentiated argument gets a fresh alias.** The function receives an identity
  view of every argument, and derivatives are taken with respect to that view. A nested
  transform therefore differentiates only through its own argument, never through other
  uses of the same tensor captured by a closure -- the "perturbation confusion" that naive
  tape-based nesting gets wrong.

:func:`jvp` (forward-mode products) uses the double-VJP trick: the VJP ``u -> J^T u`` is
linear in ``u``, so differentiating it with respect to ``u`` in direction ``v`` gives
``J v`` -- two reverse passes instead of a separate forward-mode implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np

from tensorgrad import autograd
from tensorgrad._types import Array
from tensorgrad.autograd import enable_grad, is_grad_enabled
from tensorgrad.ops import stack
from tensorgrad.ops._util import make_result
from tensorgrad.ops.shape import reshape
from tensorgrad.tensor import Tensor

__all__ = ["grad", "hessian", "hvp", "jacobian", "jvp", "value_and_grad", "vjp"]

#: Values accepted where a tensor argument is expected (arrays and scalars are converted).
TensorLike: TypeAlias = Tensor | Array | float
Argnums: TypeAlias = int | tuple[int, ...]


def _as_tensor(value: TensorLike) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


def _alias(x: Tensor) -> Tensor:
    """A tensor to differentiate with respect to that shares ``x``'s data.

    If ``x`` is already part of a graph the alias is an identity op on top of it, so results
    stay differentiable with respect to ``x``; otherwise it is a fresh leaf.
    """
    if not x.requires_grad:
        return Tensor._wrap(x.data).requires_grad_()
    return make_result(
        x.data,
        (x,),
        lambda g, need: (g,),
        "alias",
        graph=lambda g, out, need: (g,),
    )


def _zeros_like(x: Tensor) -> Tensor:
    return Tensor._wrap(np.zeros_like(x.data))


def _grads(
    outputs: Sequence[Tensor],
    inputs: Sequence[Tensor],
    grad_outputs: Sequence[Tensor | Array | None] | None,
    *,
    create_graph: bool,
    retain_graph: bool | None = None,
) -> tuple[Tensor, ...]:
    """``autograd.grad`` that returns zeros (instead of failing) where nothing depends on x."""
    live = [i for i, out in enumerate(outputs) if out.requires_grad]
    if not live:
        return tuple(_zeros_like(x) for x in inputs)
    seeds = None if grad_outputs is None else [grad_outputs[i] for i in live]
    grads = autograd.grad(
        [outputs[i] for i in live],
        inputs,
        grad_outputs=seeds,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    return tuple(_zeros_like(x) if g is None else g for x, g in zip(inputs, grads, strict=True))


def _prepare(args: Sequence[TensorLike], argnums: Argnums) -> tuple[list[Tensor], list[Tensor]]:
    """Convert the arguments to tensors and replace the differentiated ones by aliases."""
    positions = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    if not positions:
        raise ValueError("argnums must name at least one argument")
    tensors = [_as_tensor(a) for a in args]
    for pos in positions:
        if not -len(tensors) <= pos < len(tensors):
            raise IndexError(f"argnums {pos} is out of range for {len(tensors)} arguments")
        tensors[pos] = _alias(tensors[pos])
    return tensors, [tensors[pos] for pos in positions]


def _unpack(values: tuple[Tensor, ...], argnums: Argnums) -> Tensor | tuple[Tensor, ...]:
    return values[0] if isinstance(argnums, int) else values


def _finish(value: Tensor, create_graph: bool) -> Tensor:
    return value if create_graph else value.detach()


def value_and_grad(
    f: Callable[..., Tensor], argnums: Argnums = 0
) -> Callable[..., tuple[Tensor, Tensor | tuple[Tensor, ...]]]:
    """Like :func:`grad`, but the new function returns ``(f(*args), gradient)``."""

    def value_and_grad_f(*args: TensorLike) -> tuple[Tensor, Tensor | tuple[Tensor, ...]]:
        create_graph = is_grad_enabled()
        with enable_grad():
            tensors, wrt = _prepare(args, argnums)
            out = f(*tensors)
            if not isinstance(out, Tensor) or out.size != 1:
                shape = getattr(out, "shape", type(out).__name__)
                raise ValueError(f"grad needs a function returning one scalar, got {shape}")
            grads = _grads([out], wrt, None, create_graph=create_graph)
        return _finish(out, create_graph), _unpack(tuple(grads), argnums)

    return value_and_grad_f


def grad(f: Callable[..., Tensor], argnums: Argnums = 0) -> Callable[..., Any]:
    """Return a function computing the gradient of the scalar function ``f``.

    Args:
        f: Function of tensors returning a single-element tensor.
        argnums: Which positional argument(s) to differentiate with respect to. An int gives
            one gradient tensor, a tuple gives a tuple of them.

    Example:
        >>> import tensorgrad as tg
        >>> from tensorgrad import func
        >>> func.grad(lambda x: (x * x).sum())(tg.Tensor([1.0, -2.0])).tolist()
        [2.0, -4.0]
    """
    vg = value_and_grad(f, argnums)

    def grad_f(*args: TensorLike) -> Tensor | tuple[Tensor, ...]:
        return vg(*args)[1]

    return grad_f


def vjp(
    f: Callable[..., Tensor], *primals: TensorLike
) -> tuple[Tensor, Callable[[TensorLike], tuple[Tensor, ...]]]:
    """Evaluate ``f(*primals)`` and return it with its vector-Jacobian-product function.

    ``vjp_fn(u)`` returns ``u^T df/dprimal_i`` for every primal; it can be called repeatedly
    (the forward graph is kept alive by ``vjp_fn``).
    """
    create_graph = is_grad_enabled()
    with enable_grad():
        xs = [_alias(_as_tensor(p)) for p in primals]
        out = f(*xs)

    def vjp_fn(cotangent: TensorLike) -> tuple[Tensor, ...]:
        with enable_grad():
            u = _as_tensor(cotangent)
            return _grads([out], xs, [u], create_graph=is_grad_enabled(), retain_graph=True)

    return _finish(out, create_graph), vjp_fn


def jvp(
    f: Callable[..., Tensor],
    primals: Sequence[TensorLike],
    tangents: Sequence[TensorLike],
) -> tuple[Tensor, Tensor]:
    """Evaluate ``f`` at ``primals`` and its Jacobian-vector product ``sum_i J_i v_i``.

    Computed with the double-VJP trick: for a dummy cotangent ``u`` (its value is
    irrelevant), ``g(u) = J^T u`` is built with ``create_graph=True``; it is linear in ``u``,
    so ``d<g(u), v>/du = J v``.
    """
    if len(primals) != len(tangents):
        raise ValueError(f"got {len(primals)} primals but {len(tangents)} tangents")
    create_graph = is_grad_enabled()
    with enable_grad():
        xs = [_alias(_as_tensor(p)) for p in primals]
        vs = [_as_tensor(t) for t in tangents]
        for x, v in zip(xs, vs, strict=True):
            if x.shape != v.shape:
                raise ValueError(f"tangent shape {v.shape} does not match primal {x.shape}")
        out = f(*xs)
        u = Tensor._wrap(np.zeros_like(out.data)).requires_grad_()
        pulled = _grads([out], xs, [u], create_graph=True)
        pairs = [(g, v) for g, v in zip(pulled, vs, strict=True) if g.requires_grad]
        if not pairs:
            return _finish(out, create_graph), _zeros_like(out)
        (tangent_out,) = _grads(
            [g for g, _ in pairs], [u], [v for _, v in pairs], create_graph=create_graph
        )
    return _finish(out, create_graph), tangent_out


def jacobian(f: Callable[..., Any], argnums: Argnums = 0) -> Callable[..., Any]:
    """Return a function computing the Jacobian of ``f`` by reverse mode.

    For an output of shape ``O`` and an argument of shape ``I`` the Jacobian has shape
    ``O + I``. One backward pass is made per output element, so this suits functions with
    few outputs (use :func:`vjp` / :func:`jvp` for products with large Jacobians). If ``f``
    returns a tuple of tensors, the result is a tuple with one entry per output.
    """

    def jacobian_f(*args: TensorLike) -> Any:
        create_graph = is_grad_enabled()
        with enable_grad():
            tensors, wrt = _prepare(args, argnums)
            result = f(*tensors)
            outputs = result if isinstance(result, tuple) else (result,)
            per_output = []
            for out in outputs:
                rows = []
                for i in range(out.size):
                    onehot = np.zeros(out.size, dtype=out.dtype)
                    onehot[i] = 1.0
                    seed = onehot.reshape(out.shape)
                    rows.append(
                        _grads([out], wrt, [seed], create_graph=create_graph, retain_graph=True)
                    )
                blocks = tuple(
                    reshape(stack([row[j] for row in rows]), out.shape + x.shape)
                    if rows
                    else Tensor._wrap(np.zeros(out.shape + x.shape, dtype=out.dtype))
                    for j, x in enumerate(wrt)
                )
                per_output.append(_unpack(blocks, argnums))
        if not create_graph:
            per_output = [_detached(block) for block in per_output]
        return tuple(per_output) if isinstance(result, tuple) else per_output[0]

    return jacobian_f


def _detached(value: Tensor | tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return value.detach()
    return tuple(v.detach() for v in value)


def hessian(f: Callable[..., Tensor], argnums: Argnums = 0) -> Callable[..., Any]:
    """Return a function computing the Hessian of the scalar function ``f``.

    ``hessian(f) = jacobian(grad(f))``: the gradient is built with a graph and differentiated
    once more per entry. For an argument of shape ``S`` the Hessian has shape ``S + S``; with
    a tuple ``argnums`` it is a nested tuple of blocks ``H[i][j] = d^2 f / dx_i dx_j``.
    """
    return jacobian(grad(f, argnums), argnums)


def hvp(
    f: Callable[..., Tensor],
    primals: Sequence[TensorLike],
    tangents: Sequence[TensorLike],
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Evaluate the scalar ``f`` at ``primals`` and the Hessian-vector product ``H v``.

    Reverse-over-reverse: ``H v = d<grad f(x), v>/dx``, two backward passes and no Hessian
    in memory, so it scales to many parameters (Newton-CG, curvature estimates, ...).
    """
    if len(primals) != len(tangents):
        raise ValueError(f"got {len(primals)} primals but {len(tangents)} tangents")
    create_graph = is_grad_enabled()
    with enable_grad():
        xs = [_alias(_as_tensor(p)) for p in primals]
        vs = [_as_tensor(t) for t in tangents]
        out = f(*xs)
        if out.size != 1:
            raise ValueError(f"hvp needs a function returning one scalar, got shape {out.shape}")
        grads = _grads([out], xs, None, create_graph=True)
        terms = [(g * v).sum() for g, v in zip(grads, vs, strict=True) if g.requires_grad]
        if not terms:
            return _finish(out, create_graph), tuple(_zeros_like(x) for x in xs)
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        products = _grads([total], xs, None, create_graph=create_graph)
    return _finish(out, create_graph), products
