"""2-D convolution (via im2col) and max pooling for ``(N, C, H, W)`` tensors."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tensorgrad._types import Array
from tensorgrad.ops._util import as_tensor, make_result
from tensorgrad.tensor import Tensor, TensorLike

__all__ = ["conv2d", "im2col", "max_pool2d"]

IntPair = int | tuple[int, int]


def _pair(value: IntPair, name: str) -> tuple[int, int]:
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


def im2col(x: Array, kernel: tuple[int, int], stride: tuple[int, int]) -> Array:
    """Return every receptive field of an (already padded) ``(N, C, H, W)`` array.

    The result has shape ``(N, OH, OW, C, KH, KW)`` and is a strided *view* -- no data is
    copied until the caller reshapes it into the ``(N*OH*OW, C*KH*KW)`` patch matrix.
    """
    windows = sliding_window_view(x, kernel, axis=(2, 3))[:, :, :: stride[0], :: stride[1]]
    return windows.transpose(0, 2, 3, 1, 4, 5)


def _col2im_add(
    dest: Array,
    cols: Array,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    out_hw: tuple[int, int],
) -> None:
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
    if tx.ndim != 4 or tw.ndim != 4:
        raise ValueError(f"conv2d expects 4-D input and weight, got {tx.shape} and {tw.shape}")
    n, c, h, w = tx.shape
    c_out, c_in, kh, kw = tw.shape
    if c != c_in:
        raise ValueError(f"input has {c} channels but weight expects {c_in}")
    (sh, sw), (ph, pw) = _pair(stride, "stride"), _pair(padding, "padding")
    oh, ow = _out_size(h, kh, sh, ph), _out_size(w, kw, sw, pw)

    xp = np.pad(tx.data, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if ph or pw else tx.data
    cols = im2col(xp, (kh, kw), (sh, sw)).reshape(n * oh * ow, c * kh * kw)
    w_mat = tw.data.reshape(c_out, -1)
    out = cols @ w_mat.T
    if tb is not None:
        out = out + tb.data
    result = out.reshape(n, oh, ow, c_out).transpose(0, 3, 1, 2)

    def backward(g: Array) -> tuple[Array | None, ...]:
        g_mat = g.transpose(0, 2, 3, 1).reshape(-1, c_out)
        gx = gw = gb = None
        if tx.requires_grad:
            d_cols = (g_mat @ w_mat).reshape(n, oh, ow, c, kh, kw)
            dxp = np.zeros(xp.shape, dtype=g.dtype)
            _col2im_add(dxp, d_cols, (kh, kw), (sh, sw), (oh, ow))
            gx = dxp[:, :, ph : ph + h, pw : pw + w]
        if tw.requires_grad:
            gw = (g_mat.T @ cols).reshape(tw.shape)
        if tb is not None and tb.requires_grad:
            gb = g_mat.sum(axis=0)
        return (gx, gw) if tb is None else (gx, gw, gb)

    parents = (tx, tw) if tb is None else (tx, tw, tb)
    return make_result(result, parents, backward, "conv2d")


def max_pool2d(
    x: TensorLike,
    kernel_size: IntPair,
    stride: IntPair | None = None,
    padding: IntPair = 0,
) -> Tensor:
    """Max pooling over ``(N, C, H, W)`` input; ``stride`` defaults to ``kernel_size``.

    Padding uses ``-inf`` so padded cells never win. The gradient goes to the first maximum
    of each window (the same tie-breaking rule as PyTorch).
    """
    tx = as_tensor(x)
    if tx.ndim != 4:
        raise ValueError(f"max_pool2d expects 4-D input, got {tx.shape}")
    kh, kw = _pair(kernel_size, "kernel_size")
    sh, sw = _pair(kernel_size if stride is None else stride, "stride")
    ph, pw = _pair(padding, "padding")
    n, c, h, w = tx.shape
    oh, ow = _out_size(h, kh, sh, ph), _out_size(w, kw, sw, pw)
    xp = tx.data
    if ph or pw:
        xp = np.pad(xp, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-np.inf)
    windows = sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
    flat = windows.reshape(n, c, oh, ow, kh * kw)
    winner = np.argmax(flat, axis=-1)
    out = np.take_along_axis(flat, winner[..., None], axis=-1)[..., 0]

    def backward(g: Array) -> tuple[Array]:
        dxp = np.zeros(xp.shape, dtype=g.dtype)
        for i in range(kh):
            for j in range(kw):
                hit = winner == i * kw + j
                dxp[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw] += g * hit
        return (dxp[:, :, ph : ph + h, pw : pw + w],)

    return make_result(out, (tx,), backward, "max_pool2d")
