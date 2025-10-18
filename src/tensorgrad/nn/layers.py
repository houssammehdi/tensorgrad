"""Standard layers: linear, convolution, embedding, normalisation, dropout and containers."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np

from tensorgrad import ops
from tensorgrad._types import Array
from tensorgrad.nn import init
from tensorgrad.nn.module import Module, Parameter
from tensorgrad.tensor import Tensor, get_default_dtype

__all__ = [
    "GELU",
    "MLP",
    "BatchNorm1d",
    "Conv2d",
    "Dropout",
    "Embedding",
    "Flatten",
    "Identity",
    "LayerNorm",
    "Linear",
    "MaxPool2d",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
]


def _empty(*shape: int) -> Parameter:
    return Parameter(np.empty(shape, dtype=get_default_dtype()))


class Linear(Module):
    """Affine map ``y = x W^T + b`` over the last axis.

    Weights and bias are initialised from ``U(-1/sqrt(in), 1/sqrt(in))`` (PyTorch's default).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = init.uniform_(_empty(out_features, in_features), -bound, bound)
        self.bias: Parameter | None = (
            init.uniform_(_empty(out_features), -bound, bound) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return ops.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class Conv2d(Module):
    """2-D convolution over ``(N, C, H, W)`` input (see :func:`tensorgrad.conv2d`)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        bound = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = init.uniform_(
            _empty(out_channels, in_channels, kernel_size, kernel_size), -bound, bound
        )
        self.bias: Parameter | None = (
            init.uniform_(_empty(out_channels), -bound, bound) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return ops.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class MaxPool2d(Module):
    """Max pooling over ``(N, C, H, W)`` input."""

    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return ops.max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride or self.kernel_size}"


class Embedding(Module):
    """Lookup table mapping integer ids to ``embedding_dim`` vectors (init ``N(0, 1)``)."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = init.normal_(_empty(num_embeddings, embedding_dim))

    def forward(self, indices: Tensor | Array) -> Tensor:
        return ops.embedding(indices, self.weight)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class LayerNorm(Module):
    """Layer normalisation over the last axis with learnable scale and shift."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight: Parameter | None = None
        self.bias: Parameter | None = None
        if elementwise_affine:
            self.weight = init.ones_(_empty(normalized_shape))
            self.bias = init.zeros_(_empty(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(f"expected last dimension {self.normalized_shape}, got {x.shape}")
        return ops.layer_norm(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class BatchNorm1d(Module):
    """Batch normalisation over ``(N, C)`` or ``(N, C, L)`` input.

    Keeps ``running_mean`` / ``running_var`` buffers (exponential moving averages with
    ``momentum``) that replace the batch statistics in evaluation mode.
    """

    running_mean: Array
    running_var: Array

    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight: Parameter | None = None
        self.bias: Parameter | None = None
        if affine:
            self.weight = init.ones_(_empty(num_features))
            self.bias = init.zeros_(_empty(num_features))
        self.register_buffer("running_mean", np.zeros(num_features, dtype=get_default_dtype()))
        self.register_buffer("running_var", np.ones(num_features, dtype=get_default_dtype()))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in (2, 3) or x.shape[1] != self.num_features:
            raise ValueError(f"expected (N, {self.num_features}[, L]) input, got {x.shape}")
        return ops.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"


class Dropout(Module):
    """Inverted dropout with drop probability ``p``; the identity in evaluation mode."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return ops.dropout(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class ReLU(Module):
    """Elementwise ``max(x, 0)``."""

    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class GELU(Module):
    """Elementwise GELU (tanh approximation)."""

    def forward(self, x: Tensor) -> Tensor:
        return ops.gelu(x)


class Tanh(Module):
    """Elementwise hyperbolic tangent."""

    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    """Elementwise logistic sigmoid."""

    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


class Identity(Module):
    """Returns its input unchanged."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class Flatten(Module):
    """Flatten all axes from ``start_axis`` on (keeps the batch axis by default)."""

    def __init__(self, start_axis: int = 1) -> None:
        super().__init__()
        self.start_axis = start_axis

    def forward(self, x: Tensor) -> Tensor:
        return ops.flatten(x, self.start_axis)


class Sequential(Module):
    """Chain modules: the output of each is the input of the next."""

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self._length = 0
        for layer in layers:
            self.append(layer)

    def append(self, layer: Module) -> Sequential:
        """Add a layer at the end."""
        setattr(self, str(self._length), layer)
        self._length += 1
        return self

    def __getitem__(self, index: int) -> Module:
        if not -self._length <= index < self._length:
            raise IndexError(f"index {index} out of range for Sequential of length {len(self)}")
        return self.__dict__["_modules"][str(index % self._length)]  # type: ignore[no-any-return]

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Module]:
        return self.children()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self:
            x = layer(x)
        return x


_ACTIVATIONS: dict[str, type[Module]] = {"relu": ReLU, "gelu": GELU, "tanh": Tanh}


class MLP(Sequential):
    """Multi-layer perceptron: ``Linear -> activation [-> Dropout]`` blocks.

    Args:
        sizes: Layer widths including input and output, e.g. ``[2, 64, 64, 3]``.
        activation: Non-linearity between layers.
        dropout: Dropout probability after each hidden activation.
        final_activation: Also apply the activation after the last layer.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        activation: Literal["relu", "gelu", "tanh"] = "relu",
        dropout: float = 0.0,
        final_activation: bool = False,
    ) -> None:
        if len(sizes) < 2:
            raise ValueError("an MLP needs at least an input and an output size")
        layers: list[Module] = []
        n_linear = len(sizes) - 1
        for i, (fan_in, fan_out) in enumerate(itertools.pairwise(sizes)):
            layers.append(Linear(fan_in, fan_out))
            if i < n_linear - 1 or final_activation:
                layers.append(_ACTIVATIONS[activation]())
                if dropout > 0 and i < n_linear - 1:
                    layers.append(Dropout(dropout))
        super().__init__(*layers)
