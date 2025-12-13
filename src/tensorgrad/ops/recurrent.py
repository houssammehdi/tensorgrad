"""Recurrent sequence ops (Elman RNN, LSTM, GRU) with explicit backpropagation through time.

Each op runs a whole sequence as one graph node. The forward pass computes the input
projections of all time steps with one GEMM, then runs the recurrence and keeps the gate
activations. The backward pass walks the sequence in reverse (backpropagation through time),
carrying the hidden- and cell-state gradients from step to step, and ends with one GEMM per
weight matrix. Sequences are time-major: ``x`` is ``(T, B, input_size)``.

Weight layouts are PyTorch's, so parameters copy over from ``torch.nn.RNN``, ``LSTM`` and
``GRU``. The ``G * H`` rows of ``w_ih`` and ``w_hh`` hold the gates in the order (input,
forget, cell, output) for the LSTM and (reset, update, new) for the GRU.

Under ``create_graph=True`` the VJP runs the same reverse recursion with differentiable
tensor ops on a recomputed forward pass, so higher-order derivatives are exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from tensorgrad._types import Array
from tensorgrad.ops import shape
from tensorgrad.ops._util import as_tensor, make_result
from tensorgrad.ops.elementwise import relu, sigmoid, sigmoid_array, tanh
from tensorgrad.ops.layers import linear
from tensorgrad.ops.linalg import matmul
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["gru", "lstm", "rnn"]

Need = tuple[bool, ...]
Nonlinearity = Literal["tanh", "relu"]
_GATES = {"rnn": 1, "lstm": 4, "gru": 3}


def _prepare(
    kind: str,
    x: TensorLike,
    states: Sequence[TensorLike],
    w_ih: TensorLike,
    w_hh: TensorLike,
    b_ih: TensorLike | None,
    b_hh: TensorLike | None,
) -> tuple[tuple[Tensor, ...], list[Array]]:
    """Validate shapes and return the parents ``(x, *states, w_ih, w_hh[, b_ih, b_hh])``
    with their data converted to one floating dtype."""
    if (b_ih is None) != (b_hh is None):
        raise ValueError(f"{kind}: pass both biases or neither")
    tx, tw_ih, tw_hh = as_tensor(x), as_tensor(w_ih), as_tensor(w_hh)
    tstates = [as_tensor(s) for s in states]
    tbiases = [] if b_ih is None or b_hh is None else [as_tensor(b_ih), as_tensor(b_hh)]
    if tx.ndim != 3 or tx.shape[0] == 0:
        raise ValueError(
            f"{kind}: expected a non-empty (T, B, input_size) sequence, got {tx.shape}"
        )
    hidden = tw_hh.shape[-1] if tw_hh.ndim == 2 else -1
    rows = _GATES[kind] * hidden
    if tw_hh.shape != (rows, hidden) or tw_ih.shape != (rows, tx.shape[2]):
        raise ValueError(
            f"{kind}: weights {tw_ih.shape} and {tw_hh.shape} do not fit input size "
            f"{tx.shape[2]} ({_GATES[kind]} gate(s) of the hidden size {hidden})"
        )
    for b in tbiases:
        if b.shape != (rows,):
            raise ValueError(f"{kind}: bias of shape {b.shape} should be ({rows},)")
    for s in tstates:
        if s.shape != (tx.shape[1], hidden):
            raise ValueError(f"{kind}: initial state {s.shape} should be {(tx.shape[1], hidden)}")
    parents = (tx, *tstates, tw_ih, tw_hh, *tbiases)
    dtype = np.result_type(np.float16, *(p.dtype for p in parents))
    return parents, [p.data.astype(dtype, copy=False) for p in parents]


def _project(x: Array, w: Array, bias: Array | None) -> Array:
    """``x_t W^T + bias`` for all steps at once, as a fresh ``(T, B, rows)`` array."""
    t, b, width = x.shape
    out = (x.reshape(t * b, width) @ w.T).reshape(t, b, w.shape[0])
    if bias is not None:
        out += bias
    return out


def _previous(h0: Array, hs: Array) -> Array:
    """The hidden state each step started from (``h_0 .. h_{T-1}``) as ``(T * B, H)``."""
    return np.concatenate((h0[None], hs[:-1])).reshape(-1, h0.shape[-1])


def _parameter_grads(
    dz: Array, dz_hh: Array, x: Array, h_prev: Array, w_ih: Array, first: int, need: Need
) -> list[Array | None]:
    """Gradients of ``x, w_ih, w_hh[, b_ih, b_hh]`` from the gradients of the gates'
    pre-activations: ``dz`` for the input projection, ``dz_hh`` for the recurrent one (the
    same array except in the GRU). ``first`` is the position of ``w_ih`` among the parents.
    """
    rows = dz.shape[-1]
    dz2, dh2 = dz.reshape(-1, rows), dz_hh.reshape(-1, rows)
    grads: list[Array | None] = [
        (dz2 @ w_ih).reshape(x.shape) if need[0] else None,
        dz2.T @ x.reshape(-1, x.shape[-1]) if need[first] else None,
        dh2.T @ h_prev if need[first + 1] else None,
    ]
    if len(need) > first + 2:
        grads.append(dz2.sum(axis=0) if need[first + 2] else None)
        grads.append(dh2.sum(axis=0) if need[first + 3] else None)
    return grads


def _parameter_grads_graph(
    dz: Tensor, dz_hh: Tensor, x: Tensor, h_prev: Tensor, w_ih: Tensor, first: int, need: Need
) -> list[Tensor | None]:
    """:func:`_parameter_grads` with differentiable ops (``h_prev`` is ``(T, B, H)``)."""
    rows = dz.shape[-1]
    dz2, dh2 = shape.reshape(dz, (-1, rows)), shape.reshape(dz_hh, (-1, rows))
    grads: list[Tensor | None] = [
        matmul(dz, w_ih) if need[0] else None,
        matmul(dz2.T, shape.reshape(x, (-1, x.shape[-1]))) if need[first] else None,
        matmul(dh2.T, shape.reshape(h_prev, (-1, h_prev.shape[-1]))) if need[first + 1] else None,
    ]
    if len(need) > first + 2:
        grads.append(dz2.sum(axis=0) if need[first + 2] else None)
        grads.append(dh2.sum(axis=0) if need[first + 3] else None)
    return grads


def _summed_bias(arrays: Sequence[Array], first: int) -> Array | None:
    """``b_ih + b_hh`` if the op has biases (in the RNN and LSTM they only occur summed)."""
    return arrays[first + 2] + arrays[first + 3] if len(arrays) > first + 2 else None


def _projection_graph(x: Tensor, w_ih: Tensor, biases: Sequence[Tensor]) -> Tensor:
    """Differentiable ``x W_ih^T + b_ih + b_hh`` (see :func:`_summed_bias`)."""
    return linear(x, w_ih) if not biases else linear(x, w_ih, biases[0]) + biases[1]


# --------------------------------------------------------------------------- Elman RNN
def rnn(
    x: TensorLike,
    h0: TensorLike,
    w_ih: TensorLike,
    w_hh: TensorLike,
    b_ih: TensorLike | None = None,
    b_hh: TensorLike | None = None,
    *,
    nonlinearity: Nonlinearity = "tanh",
) -> Tensor:
    """Elman RNN ``h_t = act(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh)`` over a sequence.

    Args:
        x: Input sequence ``(T, B, input_size)``.
        h0: Initial hidden state ``(B, H)``.
        w_ih: Input weights ``(H, input_size)``.
        w_hh: Recurrent weights ``(H, H)``.
        b_ih: Input bias ``(H,)``; pass both biases or neither.
        b_hh: Recurrent bias ``(H,)``.
        nonlinearity: ``"tanh"`` or ``"relu"``.

    Returns:
        The hidden states ``h_1 .. h_T``, shape ``(T, B, H)``.
    """
    if nonlinearity not in ("tanh", "relu"):
        raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
    parents, arrays = _prepare("rnn", x, (h0,), w_ih, w_hh, b_ih, b_hh)
    xd, h0d, w_ihd, w_hhd = arrays[:4]
    steps = len(xd)
    out = _project(xd, w_ihd, _summed_bias(arrays, 2))  # pre-activations, then the states
    h = h0d
    for t in range(steps):
        a = out[t]
        a += h @ w_hhd.T
        if nonlinearity == "tanh":
            np.tanh(a, out=a)
        else:
            np.maximum(a, 0.0, out=a)
        h = a

    def backward(g: Array, need: Need) -> tuple[Array | None, ...]:
        da = np.empty_like(out)
        dh = np.zeros_like(h0d)
        for t in range(steps - 1, -1, -1):
            dh += g[t]
            if nonlinearity == "tanh":
                np.multiply(dh, 1.0 - out[t] * out[t], out=da[t])
            else:
                np.multiply(dh, out[t] > 0, out=da[t])
            if t or need[1]:
                dh = da[t] @ w_hhd
        grads = _parameter_grads(da, da, xd, _previous(h0d, out), w_ihd, 2, need)
        return (grads[0], dh if need[1] else None, *grads[1:])

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, ...]:
        tx, th0, tw_ih, tw_hh = parents[:4]
        xw = _projection_graph(tx, tw_ih, parents[4:])
        hs: list[Tensor] = []
        h = th0
        for t in range(steps):
            pre = xw[t] + matmul(h, tw_hh.T)
            h = tanh(pre) if nonlinearity == "tanh" else relu(pre)
            hs.append(h)
        das: list[Tensor] = []
        dh: Tensor | None = None
        for t in range(steps - 1, -1, -1):
            dht = g[t] if dh is None else g[t] + dh
            if nonlinearity == "tanh":
                da = dht * (1.0 - hs[t] * hs[t])
            else:
                da = shape.where(hs[t].data > 0, dht, 0.0)
            das.append(da)
            dh = matmul(da, tw_hh)
        dz = shape.stack(das[::-1], 0)
        h_prev = shape.stack([th0, *hs[:-1]], 0)
        grads = _parameter_grads_graph(dz, dz, tx, h_prev, tw_ih, 2, need)
        return (grads[0], dh if need[1] else None, *grads[1:])

    return make_result(out, parents, backward, "rnn", graph=graph)


# --------------------------------------------------------------------------- LSTM
def lstm(
    x: TensorLike,
    h0: TensorLike,
    c0: TensorLike,
    w_ih: TensorLike,
    w_hh: TensorLike,
    b_ih: TensorLike | None = None,
    b_hh: TensorLike | None = None,
) -> Tensor:
    """Long short-term memory over a sequence (Hochreiter and Schmidhuber, 1997).

    With ``z = x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh`` split into four gates::

        i, f, o = sigmoid(z_i), sigmoid(z_f), sigmoid(z_o);   g = tanh(z_g)
        c_t = f * c_{t-1} + i * g;                             h_t = o * tanh(c_t)

    Args:
        x: Input sequence ``(T, B, input_size)``.
        h0: Initial hidden state ``(B, H)``.
        c0: Initial cell state ``(B, H)``.
        w_ih: Input weights ``(4 H, input_size)``, gate rows ordered (i, f, g, o).
        w_hh: Recurrent weights ``(4 H, H)``.
        b_ih: Input bias ``(4 H,)``; pass both biases or neither.
        b_hh: Recurrent bias ``(4 H,)``.

    Returns:
        A ``(T, B, 2, H)`` tensor: ``[:, :, 0]`` holds the hidden states ``h_1 .. h_T`` and
        ``[:, :, 1]`` the cell states ``c_1 .. c_T`` (one op, one output, so both stay
        differentiable).
    """
    parents, arrays = _prepare("lstm", x, (h0, c0), w_ih, w_hh, b_ih, b_hh)
    xd, h0d, c0d, w_ihd, w_hhd = arrays[:5]
    steps, batch, hid = len(xd), xd.shape[1], h0d.shape[1]
    acts = _project(xd, w_ihd, _summed_bias(arrays, 3))  # pre-activations, then the gates
    out = np.empty((steps, batch, 2, hid), dtype=acts.dtype)
    tanh_c = np.empty((steps, batch, hid), dtype=acts.dtype)
    h, c = h0d, c0d
    for t in range(steps):
        z = acts[t]
        z += h @ w_hhd.T
        z[:, : 2 * hid] = sigmoid_array(z[:, : 2 * hid])
        z[:, 3 * hid :] = sigmoid_array(z[:, 3 * hid :])
        np.tanh(z[:, 2 * hid : 3 * hid], out=z[:, 2 * hid : 3 * hid])
        c_new = out[t, :, 1]
        np.multiply(z[:, hid : 2 * hid], c, out=c_new)
        c_new += z[:, :hid] * z[:, 2 * hid : 3 * hid]
        np.tanh(c_new, out=tanh_c[t])
        np.multiply(z[:, 3 * hid :], tanh_c[t], out=out[t, :, 0])
        h, c = out[t, :, 0], c_new

    def backward(g: Array, need: Need) -> tuple[Array | None, ...]:
        dz = np.empty_like(acts)
        dh, dc = np.zeros_like(h0d), np.zeros_like(c0d)
        for t in range(steps - 1, -1, -1):
            dh += g[t, :, 0]
            dc += g[t, :, 1]
            a, d, tc = acts[t], dz[t], tanh_c[t]
            i, f, gg, o = (a[:, k * hid : (k + 1) * hid] for k in range(4))
            c_prev = c0d if t == 0 else out[t - 1, :, 1]
            dc += dh * o * (1.0 - tc * tc)
            np.multiply(dc * gg, i * (1.0 - i), out=d[:, :hid])
            np.multiply(dc * c_prev, f * (1.0 - f), out=d[:, hid : 2 * hid])
            np.multiply(dc * i, 1.0 - gg * gg, out=d[:, 2 * hid : 3 * hid])
            np.multiply(dh * tc, o * (1.0 - o), out=d[:, 3 * hid :])
            dc *= f
            if t or need[1]:
                dh = d @ w_hhd
        grads = _parameter_grads(dz, dz, xd, _previous(h0d, out[:, :, 0]), w_ihd, 3, need)
        return (grads[0], dh if need[1] else None, dc if need[2] else None, *grads[1:])

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, ...]:
        tx, th0, tc0, tw_ih, tw_hh = parents[:5]
        xw = _projection_graph(tx, tw_ih, parents[5:])
        saved: list[tuple[Tensor, ...]] = []
        h, c = th0, tc0
        for t in range(steps):
            z = xw[t] + matmul(h, tw_hh.T)
            i, f = sigmoid(z[:, :hid]), sigmoid(z[:, hid : 2 * hid])
            gg, o = tanh(z[:, 2 * hid : 3 * hid]), sigmoid(z[:, 3 * hid :])
            c_new = f * c + i * gg
            tc = tanh(c_new)
            saved.append((h, c, i, f, gg, o, tc))
            h, c = o * tc, c_new
        dzs: list[Tensor] = []
        dh: Tensor | None = None
        dc: Tensor | None = None
        for t in range(steps - 1, -1, -1):
            _, c_prev, i, f, gg, o, tc = saved[t]
            dht = g[t, :, 0] if dh is None else g[t, :, 0] + dh
            dct = g[t, :, 1] if dc is None else g[t, :, 1] + dc
            dct = dct + dht * o * (1.0 - tc * tc)
            d = shape.concat(
                [
                    dct * gg * i * (1.0 - i),
                    dct * c_prev * f * (1.0 - f),
                    dct * i * (1.0 - gg * gg),
                    dht * tc * o * (1.0 - o),
                ],
                axis=-1,
            )
            dzs.append(d)
            dc = dct * f
            dh = matmul(d, tw_hh)
        dz_all = shape.stack(dzs[::-1], 0)
        h_prev = shape.stack([s[0] for s in saved], 0)
        grads = _parameter_grads_graph(dz_all, dz_all, tx, h_prev, tw_ih, 3, need)
        return (grads[0], dh if need[1] else None, dc if need[2] else None, *grads[1:])

    return make_result(out, parents, backward, "lstm", graph=graph)


# --------------------------------------------------------------------------- GRU
def gru(
    x: TensorLike,
    h0: TensorLike,
    w_ih: TensorLike,
    w_hh: TensorLike,
    b_ih: TensorLike | None = None,
    b_hh: TensorLike | None = None,
) -> Tensor:
    """Gated recurrent unit over a sequence (Cho et al., 2014), in PyTorch's formulation.

    With ``a = x_t W_ih^T + b_ih`` and ``b = h_{t-1} W_hh^T + b_hh`` split into three gates::

        r = sigmoid(a_r + b_r),   u = sigmoid(a_u + b_u),   n = tanh(a_n + r * b_n)
        h_t = (1 - u) * n + u * h_{t-1}

    Args:
        x: Input sequence ``(T, B, input_size)``.
        h0: Initial hidden state ``(B, H)``.
        w_ih: Input weights ``(3 H, input_size)``, gate rows ordered (r, u, n).
        w_hh: Recurrent weights ``(3 H, H)``.
        b_ih: Input bias ``(3 H,)``; pass both biases or neither.
        b_hh: Recurrent bias ``(3 H,)`` (its ``n`` part is scaled by the reset gate).

    Returns:
        The hidden states ``h_1 .. h_T``, shape ``(T, B, H)``.
    """
    parents, arrays = _prepare("gru", x, (h0,), w_ih, w_hh, b_ih, b_hh)
    xd, h0d, w_ihd, w_hhd = arrays[:4]
    b_hhd = arrays[5] if len(arrays) > 4 else None
    steps, batch, hid = len(xd), xd.shape[1], h0d.shape[1]
    acts = _project(xd, w_ihd, arrays[4] if len(arrays) > 4 else None)  # a, then the gates
    hn = np.empty((steps, batch, hid), dtype=acts.dtype)  # b_n, which the backward needs
    out = np.empty((steps, batch, hid), dtype=acts.dtype)
    h = h0d
    for t in range(steps):
        hw = h @ w_hhd.T
        if b_hhd is not None:
            hw += b_hhd
        a = acts[t]
        a[:, : 2 * hid] += hw[:, : 2 * hid]
        a[:, : 2 * hid] = sigmoid_array(a[:, : 2 * hid])
        hn[t] = hw[:, 2 * hid :]
        n = a[:, 2 * hid :]
        n += a[:, :hid] * hn[t]
        np.tanh(n, out=n)
        np.subtract(h, n, out=out[t])
        out[t] *= a[:, hid : 2 * hid]
        out[t] += n
        h = out[t]

    def backward(g: Array, need: Need) -> tuple[Array | None, ...]:
        dz_ih, dz_hh = np.empty_like(acts), np.empty_like(acts)
        dh = np.zeros_like(h0d)
        for t in range(steps - 1, -1, -1):
            dh += g[t]
            a, d_in, d_rec = acts[t], dz_ih[t], dz_hh[t]
            r, u, n = (a[:, k * hid : (k + 1) * hid] for k in range(3))
            h_prev = h0d if t == 0 else out[t - 1]
            np.multiply(dh * (1.0 - u), 1.0 - n * n, out=d_in[:, 2 * hid :])
            np.multiply(dh * (h_prev - n), u * (1.0 - u), out=d_in[:, hid : 2 * hid])
            np.multiply(d_in[:, 2 * hid :] * hn[t], r * (1.0 - r), out=d_in[:, :hid])
            d_rec[:, : 2 * hid] = d_in[:, : 2 * hid]
            np.multiply(d_in[:, 2 * hid :], r, out=d_rec[:, 2 * hid :])
            if t or need[1]:
                dh = dh * u + d_rec @ w_hhd
        grads = _parameter_grads(dz_ih, dz_hh, xd, _previous(h0d, out), w_ihd, 2, need)
        return (grads[0], dh if need[1] else None, *grads[1:])

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, ...]:
        tx, th0, tw_ih, tw_hh = parents[:4]
        tb_ih, tb_hh = (parents[4], parents[5]) if len(parents) > 4 else (None, None)
        xw = linear(tx, tw_ih, tb_ih)
        saved: list[tuple[Tensor, ...]] = []
        h = th0
        for t in range(steps):
            hw = linear(h, tw_hh, tb_hh)
            r = sigmoid(xw[t, :, :hid] + hw[:, :hid])
            u = sigmoid(xw[t, :, hid : 2 * hid] + hw[:, hid : 2 * hid])
            b_n = hw[:, 2 * hid :]
            n = tanh(xw[t, :, 2 * hid :] + r * b_n)
            saved.append((h, r, u, n, b_n))
            h = n + u * (h - n)
        d_ih: list[Tensor] = []
        d_hh: list[Tensor] = []
        dh: Tensor | None = None
        for t in range(steps - 1, -1, -1):
            h_prev, r, u, n, b_n = saved[t]
            dht = g[t] if dh is None else g[t] + dh
            dan = dht * (1.0 - u) * (1.0 - n * n)
            dau = dht * (h_prev - n) * u * (1.0 - u)
            dar = dan * b_n * r * (1.0 - r)
            d_ih.append(shape.concat([dar, dau, dan], axis=-1))
            d_hh.append(shape.concat([dar, dau, dan * r], axis=-1))
            dh = dht * u + matmul(d_hh[-1], tw_hh)
        h_prev_all = shape.stack([s[0] for s in saved], 0)
        grads = _parameter_grads_graph(
            shape.stack(d_ih[::-1], 0), shape.stack(d_hh[::-1], 0), tx, h_prev_all, tw_ih, 2, need
        )
        return (grads[0], dh if need[1] else None, *grads[1:])

    return make_result(out, parents, backward, "gru", graph=graph)
