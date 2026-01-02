"""2-D convolution (via im2col) and max pooling for ``(N, C, H, W)`` tensors.

Higher-order derivatives of convolution come from a closure argument. Write the convolution
as the trilinear form ``T(x, w, y) = <conv(x, w), y>``. Its three partial derivatives are
bilinear maps implemented with the same im2col / col2im kernels:

* ``dT/dy = conv2d(x, w)``            -- the forward pass,
* ``dT/dx = conv2d_input_grad(y, w)`` -- a transposed convolution,
* ``dT/dw = conv2d_weight_grad(x, y)`` -- the correlation of input and output gradient.

Because ``T`` is linear in each argument, the VJP of each map is one of the other two (for
example ``<conv2d_input_grad(y, w), u> = T(u, w, y)``, whose gradients are
``conv2d(u, w)`` and ``conv2d_weight_grad(u, y)``). So every VJP below is built from these
three ops, and derivatives of any order stay fast im2col matrix products.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tensorgrad._types import Array
from tensorgrad.ops._util import as_tensor, make_result
from tensorgrad.ops.shape import _scatter_add, reshape
from tensorgrad.profiler import profiled
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["conv2d", "conv2d_input_grad", "conv2d_weight_grad", "im2col", "max_pool2d"]

IntPair = int | tuple[int, int]
Pair = tuple[int, int]
Need = tuple[bool, ...]


def _pair(value: IntPair, name: str) -> Pair:
    pair = (value, value) if isinstance(value, int) else (int(value[0]), int(value[1]))
    if len(pair) != 2:
        raise ValueError(f"{name} must be an int or a pair of ints")
    return pair


def _out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    out = (size + 2 * padding - kernel) // stride + 1
    if out <= 0:
        raise ValueError(
            f"kernel {kernel} with padding {padding} does not fit an input of size {size}"
        )
    return out


def im2col(x: Array, kernel: Pair, stride: Pair) -> Array:
    """Return every receptive field of an (already padded) ``(N, C, H, W)`` array.

    The result has shape ``(N, OH, OW, C, KH, KW)`` and is a strided *view* -- no data is
    copied until the caller reshapes it into the ``(N*OH*OW, C*KH*KW)`` patch matrix.
    """
    windows = sliding_window_view(x, kernel, axis=(2, 3))[:, :, :: stride[0], :: stride[1]]
    return windows.transpose(0, 2, 3, 1, 4, 5)


def _col2im_add(dest: Array, cols: Array, kernel: Pair, stride: Pair, out_hw: Pair) -> None:
    """Adjoint of :func:`im2col`: scatter-add patches ``(N, OH, OW, C, KH, KW)`` into ``dest``.

    Loops over the ``KH*KW`` kernel offsets only; each iteration is one vectorised strided add.
    Overlapping windows (stride < kernel) accumulate correctly.
    """
    (kh, kw), (sh, sw), (oh, ow) = kernel, stride, out_hw
    for i in range(kh):
        for j in range(kw):
            dest[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw] += cols[..., i, j].transpose(
                0, 3, 1, 2
            )


# ------------------------------------------------------------------ array kernels
def _patches(x: Array, kernel: Pair, stride: Pair, padding: Pair) -> Array:
    """The ``(N*OH*OW, C*KH*KW)`` im2col patch matrix of the zero-padded input."""
    (ph, pw), n, c = padding, x.shape[0], x.shape[1]
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if ph or pw else x
    windows = im2col(xp, kernel, stride)
    return windows.reshape(n * windows.shape[1] * windows.shape[2], c * kernel[0] * kernel[1])


def _output_matrix(g: Array) -> Array:
    """``(N, C_out, OH, OW)`` output gradient as the ``(N*OH*OW, C_out)`` matrix."""
    return g.transpose(0, 2, 3, 1).reshape(-1, g.shape[1])


def _conv_kernel(x: Array, w: Array, stride: Pair, padding: Pair) -> tuple[Array, Array]:
    """``conv(x, w)`` without bias, plus the patch matrix (reused by the weight gradient)."""
    n, c_out = x.shape[0], w.shape[0]
    oh = _out_size(x.shape[2], w.shape[2], stride[0], padding[0])
    ow = _out_size(x.shape[3], w.shape[3], stride[1], padding[1])
    cols = _patches(x, (w.shape[2], w.shape[3]), stride, padding)
    out = (cols @ w.reshape(c_out, -1).T).reshape(n, oh, ow, c_out).transpose(0, 3, 1, 2)
    return out, cols


def _input_grad_kernel(
    g_mat: Array, w: Array, x_shape: tuple[int, ...], out_hw: Pair, stride: Pair, padding: Pair
) -> Array:
    """``dT/dx``: scatter ``g_mat @ W`` (one patch per output position) back onto the input."""
    n, c, h, width = x_shape
    c_out, _, kh, kw = w.shape
    (ph, pw), (oh, ow) = padding, out_hw
    d_cols = (g_mat @ w.reshape(c_out, -1)).reshape(n, oh, ow, c, kh, kw)
    dxp = np.zeros((n, c, h + 2 * ph, width + 2 * pw), dtype=d_cols.dtype)
    _col2im_add(dxp, d_cols, (kh, kw), stride, out_hw)
    return dxp[:, :, ph : ph + h, pw : pw + width]


def _weight_grad_kernel(g_mat: Array, cols: Array, w_shape: tuple[int, ...]) -> Array:
    """``dT/dw``: every patch weighted by the output gradient at its position, summed."""
    return (g_mat.T @ cols).reshape(w_shape)


def _check_conv_args(x: Tensor, w: Tensor, stride: IntPair, padding: IntPair) -> tuple[Pair, Pair]:
    if x.ndim != 4 or w.ndim != 4:
        raise ValueError(f"conv2d expects 4-D input and weight, got {x.shape} and {w.shape}")
    if x.shape[1] != w.shape[1]:
        raise ValueError(f"input has {x.shape[1]} channels but weight expects {w.shape[1]}")
    return _pair(stride, "stride"), _pair(padding, "padding")


# ------------------------------------------------------------------ differentiable ops
@profiled("conv2d")
def conv2d(
    x: TensorLike,
    weight: TensorLike,
    bias: TensorLike | None = None,
    stride: IntPair = 1,
    padding: IntPair = 0,
) -> Tensor:
    """2-D cross-correlation (what deep-learning libraries call convolution).

    Args:
        x: Input of shape ``(N, C_in, H, W)``.
        weight: Filters of shape ``(C_out, C_in, KH, KW)``.
        bias: Optional ``(C_out,)`` bias.
        stride: Step between receptive fields.
        padding: Zero padding added to each spatial border.

    Returns:
        ``(N, C_out, OH, OW)`` output. The forward pass lowers the convolution to one matrix
        multiplication of the im2col patch matrix with the flattened filters.
    """
    tx, tw = as_tensor(x), as_tensor(weight)
    tb = None if bias is None else as_tensor(bias)
    st, pad = _check_conv_args(tx, tw, stride, padding)
    c_out = tw.shape[0]
    if tb is not None and tb.shape != (c_out,):
        raise ValueError(f"bias must have shape ({c_out},), got {tb.shape}")
    out, cols = _conv_kernel(tx.data, tw.data, st, pad)
    if tb is not None:
        out = out + tb.data.reshape(1, -1, 1, 1)

    def backward(g: Array, need: Need) -> tuple[Array | None, ...]:
        g_mat = _output_matrix(g)
        gx = gw = gb = None
        if need[0]:
            gx = _input_grad_kernel(g_mat, tw.data, tx.shape, g.shape[2:], st, pad)
        if need[1]:
            gw = _weight_grad_kernel(g_mat, cols, tw.shape)
        if tb is not None and need[2]:
            gb = g_mat.sum(axis=0)
        return (gx, gw) if tb is None else (gx, gw, gb)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, ...]:
        gx = conv2d_input_grad(g, tw, tx.shape, st, pad) if need[0] else None
        gw = conv2d_weight_grad(tx, g, tw.shape, st, pad) if need[1] else None
        gb = g.sum(axis=(0, 2, 3)) if tb is not None and need[2] else None
        return (gx, gw) if tb is None else (gx, gw, gb)

    parents = (tx, tw) if tb is None else (tx, tw, tb)
    return make_result(out, parents, backward, "conv2d", graph=graph)


@profiled("conv2d_input_grad")
def conv2d_input_grad(
    grad_output: TensorLike,
    weight: TensorLike,
    input_shape: tuple[int, ...],
    stride: IntPair = 1,
    padding: IntPair = 0,
) -> Tensor:
    """Gradient of ``conv2d(x, weight)`` with respect to ``x`` (a transposed convolution).

    Maps a ``(N, C_out, OH, OW)`` output gradient to an input-shaped ``(N, C_in, H, W)``
    tensor. It is differentiable itself (see the module docstring).
    """
    tg_, tw = as_tensor(grad_output), as_tensor(weight)
    st, pad = _pair(stride, "stride"), _pair(padding, "padding")
    out_hw = (tg_.shape[2], tg_.shape[3])
    out = _input_grad_kernel(_output_matrix(tg_.data), tw.data, input_shape, out_hw, st, pad)

    def backward(u: Array, need: Need) -> tuple[Array | None, Array | None]:
        g_u = w_u = None
        if need[0]:  # d/dy T(u, w, y) = conv(u, w)
            g_u = _conv_kernel(u, tw.data, st, pad)[0]
        if need[1]:  # d/dw T(u, w, y) = weight gradient of (u, y)
            patches = _patches(u, (tw.shape[2], tw.shape[3]), st, pad)
            w_u = _weight_grad_kernel(_output_matrix(tg_.data), patches, tw.shape)
        return g_u, w_u

    def graph(u: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, Tensor | None]:
        return (
            conv2d(u, tw, stride=st, padding=pad) if need[0] else None,
            conv2d_weight_grad(u, tg_, tw.shape, st, pad) if need[1] else None,
        )

    return make_result(out, (tg_, tw), backward, "conv2d_input_grad", graph=graph)


@profiled("conv2d_weight_grad")
def conv2d_weight_grad(
    x: TensorLike,
    grad_output: TensorLike,
    weight_shape: tuple[int, ...],
    stride: IntPair = 1,
    padding: IntPair = 0,
) -> Tensor:
    """Gradient of ``conv2d(x, w)`` with respect to ``w``, for an output gradient.

    Returns a ``weight_shape`` tensor. It is differentiable itself (see the module docstring).
    """
    tx, tg_ = as_tensor(x), as_tensor(grad_output)
    st, pad = _pair(stride, "stride"), _pair(padding, "padding")
    kernel = (weight_shape[2], weight_shape[3])
    out = _weight_grad_kernel(
        _output_matrix(tg_.data), _patches(tx.data, kernel, st, pad), weight_shape
    )

    def backward(m: Array, need: Need) -> tuple[Array | None, Array | None]:
        x_m = g_m = None
        if need[0]:  # d/dx T(x, m, y) = input gradient of (y, m)
            out_hw = (tg_.shape[2], tg_.shape[3])
            x_m = _input_grad_kernel(_output_matrix(tg_.data), m, tx.shape, out_hw, st, pad)
        if need[1]:  # d/dy T(x, m, y) = conv(x, m)
            g_m = _conv_kernel(tx.data, m, st, pad)[0]
        return x_m, g_m

    def graph(m: Tensor, y: Tensor, need: Need) -> tuple[Tensor | None, Tensor | None]:
        return (
            conv2d_input_grad(tg_, m, tx.shape, st, pad) if need[0] else None,
            conv2d(tx, m, stride=st, padding=pad) if need[1] else None,
        )

    return make_result(out, (tx, tg_), backward, "conv2d_weight_grad", graph=graph)


def _window_offsets(
    kernel: Pair, stride: Pair, out_hw: Pair
) -> list[tuple[int, tuple[slice, ...]]]:
    """``(k, index)`` for each kernel offset ``k``: ``xp[index]`` is the ``(N, C, OH, OW)``
    array of the ``k``-th cell (row-major within the window) of every pooling window."""
    (kh, kw), (sh, sw), (oh, ow) = kernel, stride, out_hw
    return [
        (i * kw + j, (slice(None), slice(None), slice(i, i + sh * (oh - 1) + 1, sh),
                      slice(j, j + sw * (ow - 1) + 1, sw)))
        for i in range(kh)
        for j in range(kw)
    ]  # fmt: skip


def _first_hits(
    xp: Array, out: Array, offsets: list[tuple[int, tuple[slice, ...]]]
) -> Iterator[tuple[int, tuple[slice, ...], Array]]:
    """For each kernel offset, the mask of windows whose maximum is that offset's cell.

    Each window counts only its first maximum in row-major order (the cell
    ``numpy.argmax`` picks, where a NaN counts as the maximum), as PyTorch does.
    """
    open_ = np.ones(out.shape, dtype=np.bool_)
    nan_out = np.isnan(out)
    any_nan = bool(nan_out.any())
    for k, index in offsets:
        v = xp[index]
        hit = v == out
        if any_nan:
            hit |= np.isnan(v) & nan_out
        hit &= open_
        open_ ^= hit
        yield k, index, hit


def _winner_positions(winner: Array, kw: int, stride: Pair, padded: tuple[int, ...]) -> Array:
    """Flat index into the padded input of every window's maximum."""
    n, c, oh, ow = winner.shape
    hp, wp = padded[2], padded[3]
    ki, kj = np.divmod(winner, kw)
    rows = np.arange(oh).reshape(1, 1, oh, 1) * stride[0] + ki
    cols = np.arange(ow).reshape(1, 1, 1, ow) * stride[1] + kj
    planes = np.arange(n * c).reshape(n, c, 1, 1) * (hp * wp)
    positions: Array = planes + rows * wp + cols
    return positions


@profiled("max_pool2d")
def max_pool2d(
    x: TensorLike,
    kernel_size: IntPair,
    stride: IntPair | None = None,
    padding: IntPair = 0,
) -> Tensor:
    """Max pooling over ``(N, C, H, W)`` input; ``stride`` defaults to ``kernel_size``.

    Padding uses ``-inf`` so padded cells never win; as in PyTorch it may be at most half the
    kernel size, so every window contains at least one real input cell. The gradient goes to
    the first maximum of each window (the same tie-breaking rule as PyTorch).
    """
    tx = as_tensor(x)
    if tx.ndim != 4:
        raise ValueError(f"max_pool2d expects 4-D input, got {tx.shape}")
    kh, kw = _pair(kernel_size, "kernel_size")
    sh, sw = _pair(kernel_size if stride is None else stride, "stride")
    ph, pw = _pair(padding, "padding")
    if ph > kh // 2 or pw > kw // 2:
        raise ValueError(
            f"padding {(ph, pw)} must be at most half the kernel size {(kh, kw)}; larger "
            "padding creates windows that contain only padding"
        )
    h, w = tx.shape[2:]
    oh, ow = _out_size(h, kh, sh, ph), _out_size(w, kw, sw, pw)
    xp = tx.data
    if ph or pw:
        xp = np.pad(xp, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-np.inf)
    # A running maximum over the kh * kw strided views of the input: one vectorised pass
    # per kernel offset (np.maximum propagates NaN, like argmax). Which cell won is only
    # needed by the backward pass, which recovers it by comparison.
    offsets = _window_offsets((kh, kw), (sh, sw), (oh, ow))
    out = xp[offsets[0][1]].copy()
    for _, index in offsets[1:]:
        np.maximum(out, xp[index], out=out)

    def backward(g: Array, need: Need) -> tuple[Array]:
        dxp = np.zeros(xp.shape, dtype=g.dtype)
        overlapping = sh < kh or sw < kw
        for _, index, hit in _first_hits(xp, out, offsets):
            if overlapping:
                dxp[index] += g * hit
            else:  # every input cell is in at most one window: write in place
                np.multiply(g, hit, out=dxp[index])
        return (dxp[:, :, ph : ph + h, pw : pw + w],)

    def graph(g: Tensor, y: Tensor, need: Need) -> tuple[Tensor]:
        # Pooling selects fixed positions (a.e.), so its VJP is a scatter-add of g into
        # them -- a linear op whose own VJP is the matching gather.
        winner = np.zeros(out.shape, dtype=np.intp)
        for k, _, hit in _first_hits(xp, out, offsets):
            winner += k * hit
        positions = _winner_positions(winner, kw, (sh, sw), xp.shape)
        dxp = _scatter_add(reshape(g, (-1,)), (positions.reshape(-1),), True, (xp.size,))
        return (reshape(dxp, xp.shape)[:, :, ph : ph + h, pw : pw + w],)

    return make_result(out, (tx,), backward, "max_pool2d", graph=graph)
