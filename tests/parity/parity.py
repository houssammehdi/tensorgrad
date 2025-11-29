"""Compare tensorgrad with PyTorch: forward values, VJPs and second-order VJPs (float64).

Only imported by modules that have already called ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

import tensorgrad as tg

RTOL = 1e-10
ATOL = 1e-12


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().numpy()
    if isinstance(value, tg.Tensor):
        return value.data
    return np.asarray(value)


def assert_close(ours: Any, theirs: Any, what: str, rtol: float = RTOL, atol: float = ATOL) -> None:
    np.testing.assert_allclose(to_numpy(ours), to_numpy(theirs), rtol=rtol, atol=atol, err_msg=what)


def _dense(grads: Sequence[Any], like: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [
        np.zeros_like(x) if g is None else to_numpy(g) for g, x in zip(grads, like, strict=True)
    ]


def compare(
    tg_fn: Callable[..., tg.Tensor],
    torch_fn: Callable[..., torch.Tensor],
    arrays: Sequence[np.ndarray],
    differentiable: Sequence[bool] | None = None,
    *,
    second_order: bool = True,
    rtol: float = RTOL,
    atol: float = ATOL,
    seed: int = 0,
) -> None:
    """Check ``tg_fn`` against ``torch_fn`` on the same float64 inputs.

    1. Forward values agree.
    2. VJPs agree for a random upstream gradient ``v``.
    3. With ``second_order``: for random ``w``, the gradients of ``<VJP(v), w>`` with respect
       to every differentiable input *and* to ``v`` agree (Hessian-vector products and the
       Jacobian-vector product, i.e. double backward in both directions).
    """
    rng = np.random.default_rng(seed)
    flags = list(differentiable) if differentiable is not None else [True] * len(arrays)
    ours = [
        tg.Tensor(np.array(a, copy=True), requires_grad=f)
        for a, f in zip(arrays, flags, strict=True)
    ]
    theirs = [
        torch.tensor(np.array(a, copy=True), requires_grad=f)
        for a, f in zip(arrays, flags, strict=True)
    ]
    out_ours, out_theirs = tg_fn(*ours), torch_fn(*theirs)
    assert_close(out_ours, out_theirs, "forward values", rtol, atol)

    wrt_ours = [t for t, f in zip(ours, flags, strict=True) if f]
    wrt_theirs = [t for t, f in zip(theirs, flags, strict=True) if f]
    shapes = [a for a, f in zip(arrays, flags, strict=True) if f]
    if not wrt_ours:
        return
    v = rng.standard_normal(out_ours.shape)
    v_ours = tg.Tensor(v, requires_grad=second_order)
    v_theirs = torch.tensor(v, requires_grad=second_order)
    # The ordinary backward pass (NumPy VJPs)...
    fast = tg.autograd.grad(
        out_ours, wrt_ours, grad_outputs=v, retain_graph=True, allow_unused=True
    )
    g_theirs = torch.autograd.grad(
        out_theirs, wrt_theirs, grad_outputs=v_theirs, create_graph=second_order, allow_unused=True
    )
    expected = _dense(g_theirs, shapes)
    for i, (a, b) in enumerate(zip(_dense(fast, shapes), expected, strict=True)):
        assert_close(a, b, f"gradient of input {i}", rtol, atol)
    if not second_order:
        return
    # ...and the differentiable VJPs that create_graph uses.
    g_ours = tg.autograd.grad(
        out_ours, wrt_ours, grad_outputs=v_ours, create_graph=True, allow_unused=True
    )
    for i, (a, b) in enumerate(zip(_dense(g_ours, shapes), expected, strict=True)):
        assert_close(a, b, f"create_graph gradient of input {i}", rtol, atol)

    weights = [rng.standard_normal(a.shape) for a in shapes]
    s_ours = [(g * w).sum() for g, w in zip(g_ours, weights, strict=True) if g is not None]
    s_theirs = [
        (g * torch.tensor(w)).sum() for g, w in zip(g_theirs, weights, strict=True) if g is not None
    ]
    total_ours = sum(s_ours[1:], s_ours[0]) if s_ours else None
    total_theirs = sum(s_theirs[1:], s_theirs[0]) if s_theirs else None
    targets = [*shapes, v]
    if total_ours is not None and total_ours.requires_grad:
        h_ours = tg.autograd.grad(total_ours, [*wrt_ours, v_ours], allow_unused=True)
    else:
        h_ours = (None,) * len(targets)
    if total_theirs is not None and total_theirs.requires_grad:
        h_theirs = torch.autograd.grad(total_theirs, [*wrt_theirs, v_theirs], allow_unused=True)
    else:
        h_theirs = (None,) * len(targets)
    names = [f"second derivative w.r.t. input {i}" for i in range(len(shapes))]
    names.append("second derivative w.r.t. the upstream gradient")
    for name, a, b in zip(names, _dense(h_ours, targets), _dense(h_theirs, targets), strict=True):
        assert_close(a, b, name, rtol, atol)
