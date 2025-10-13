"""Differentiable ops. Each function returns a new tensor and records how to backpropagate."""

# ``tensor`` must finish loading before any op module: it binds the op modules at its end.
from tensorgrad import tensor as _tensor  # noqa: F401
from tensorgrad.ops.activation import log_softmax, logsumexp, softmax
from tensorgrad.ops.conv import conv2d, max_pool2d
from tensorgrad.ops.elementwise import (
    add,
    astype,
    div,
    exp,
    gelu,
    log,
    mul,
    neg,
    pow,
    relu,
    sigmoid,
    sqrt,
    sub,
    tanh,
)
from tensorgrad.ops.layers import batch_norm, dropout, embedding, layer_norm, linear
from tensorgrad.ops.linalg import matmul
from tensorgrad.ops.loss import cross_entropy, mse_loss
from tensorgrad.ops.reduce import max, mean, min, sum, var
from tensorgrad.ops.shape import (
    concat,
    flatten,
    getitem,
    masked_fill,
    permute,
    reshape,
    squeeze,
    stack,
    transpose,
    unsqueeze,
    where,
)

__all__ = [
    "add",
    "astype",
    "batch_norm",
    "concat",
    "conv2d",
    "cross_entropy",
    "div",
    "dropout",
    "embedding",
    "exp",
    "flatten",
    "gelu",
    "getitem",
    "layer_norm",
    "linear",
    "log",
    "log_softmax",
    "logsumexp",
    "masked_fill",
    "matmul",
    "max",
    "max_pool2d",
    "mean",
    "min",
    "mse_loss",
    "mul",
    "neg",
    "permute",
    "pow",
    "relu",
    "reshape",
    "sigmoid",
    "softmax",
    "sqrt",
    "squeeze",
    "stack",
    "sub",
    "sum",
    "tanh",
    "transpose",
    "unsqueeze",
    "var",
    "where",
]
