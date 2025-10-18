"""Neural-network building blocks with a PyTorch-like API."""

from tensorgrad.nn import functional, init
from tensorgrad.nn.attention import MultiHeadAttention
from tensorgrad.nn.layers import (
    GELU,
    MLP,
    BatchNorm1d,
    Conv2d,
    Dropout,
    Embedding,
    Flatten,
    Identity,
    LayerNorm,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from tensorgrad.nn.module import IncompatibleKeys, Module, Parameter
from tensorgrad.nn.transformer import GPT, GPTConfig, TransformerBlock

__all__ = [
    "GELU",
    "GPT",
    "MLP",
    "BatchNorm1d",
    "Conv2d",
    "Dropout",
    "Embedding",
    "Flatten",
    "GPTConfig",
    "Identity",
    "IncompatibleKeys",
    "LayerNorm",
    "Linear",
    "MaxPool2d",
    "Module",
    "MultiHeadAttention",
    "Parameter",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "TransformerBlock",
    "functional",
    "init",
]
