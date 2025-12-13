"""Recurrent layers: Elman RNN, LSTM and GRU, with PyTorch's parameters and call signature.

The recurrence of each layer is one fused op from :mod:`tensorgrad.ops.recurrent`, whose
backward pass is backpropagation through time. Layers are unidirectional; stacking
``num_layers`` of them feeds each layer's hidden states to the next.
"""

from __future__ import annotations

import math

import numpy as np

from tensorgrad import ops
from tensorgrad.nn import init
from tensorgrad.nn.layers import _empty
from tensorgrad.nn.module import Module
from tensorgrad.ops.recurrent import Nonlinearity, gru, lstm, rnn
from tensorgrad.tensor import Tensor

__all__ = ["GRU", "LSTM", "RNN"]


class _Recurrent(Module):
    """Parameters and layer stacking shared by :class:`RNN`, :class:`LSTM` and :class:`GRU`.

    Each layer ``k`` has ``weight_ih_lk`` ``(G * H, in)``, ``weight_hh_lk`` ``(G * H, H)``
    and, with ``bias``, ``bias_ih_lk`` and ``bias_hh_lk`` ``(G * H,)``, all initialised from
    ``U(-1/sqrt(H), 1/sqrt(H))`` as in PyTorch (``G`` gates, hidden size ``H``).
    """

    _gates = 1

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size, num_layers) < 1:
            raise ValueError("input_size, hidden_size and num_layers must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {dropout}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        bound = 1.0 / math.sqrt(hidden_size)
        rows = self._gates * hidden_size
        for layer in range(num_layers):
            width = input_size if layer == 0 else hidden_size
            shapes: dict[str, tuple[int, ...]] = {
                "weight_ih": (rows, width),
                "weight_hh": (rows, hidden_size),
            }
            if bias:
                shapes |= {"bias_ih": (rows,), "bias_hh": (rows,)}
            for name, shape in shapes.items():
                setattr(self, f"{name}_l{layer}", init.uniform_(_empty(*shape), -bound, bound))

    def layer_weights(self, layer: int) -> list[Tensor]:
        """``[weight_ih, weight_hh]`` of one layer, followed by its two biases if any."""
        names = ["weight_ih", "weight_hh"] + (["bias_ih", "bias_hh"] if self.bias else [])
        return [getattr(self, f"{name}_l{layer}") for name in names]

    def _time_major(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            layout = "(B, T, input_size)" if self.batch_first else "(T, B, input_size)"
            raise ValueError(
                f"{type(self).__name__} expects {layout} input with input_size "
                f"{self.input_size}, got {x.shape}"
            )
        return ops.transpose(x, 0, 1) if self.batch_first else x

    def _initial_state(self, state: Tensor | None, like: Tensor) -> Tensor:
        expected = (self.num_layers, like.shape[1], self.hidden_size)
        if state is None:
            return Tensor(np.zeros(expected, dtype=like.dtype))
        if state.shape != expected:
            raise ValueError(f"initial state should have shape {expected}, got {state.shape}")
        return state

    def _next_input(self, hs: Tensor, layer: int) -> Tensor:
        """Dropout between stacked layers (never after the last one), as in PyTorch."""
        if self.dropout > 0 and self.training and layer < self.num_layers - 1:
            return ops.dropout(hs, self.dropout, True)
        return hs

    def _output(self, hs: Tensor) -> Tensor:
        return ops.transpose(hs, 0, 1) if self.batch_first else hs

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"bias={self.bias}, batch_first={self.batch_first}, dropout={self.dropout}"
        )


class RNN(_Recurrent):
    """Multi-layer Elman RNN, ``h_t = act(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh)``.

    Args:
        input_size: Features per input step.
        hidden_size: Size ``H`` of the hidden state.
        num_layers: Number of stacked layers.
        nonlinearity: ``"tanh"`` or ``"relu"``.
        bias: Use the biases ``b_ih`` and ``b_hh``.
        batch_first: Inputs and outputs are ``(B, T, features)`` instead of time-major.
        dropout: Dropout on the outputs of every layer except the last (training only).

    Called with ``x`` (``(T, B, input_size)``) and an optional initial state ``h0``
    (``(num_layers, B, H)``, zeros by default), it returns ``(output, h_n)``: the last
    layer's hidden state at every step and every layer's final hidden state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: Nonlinearity = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ) -> None:
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout)
        self.nonlinearity: Nonlinearity = nonlinearity

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        seq = self._time_major(x)
        state = self._initial_state(h0, seq)
        finals = []
        hs = seq
        for layer in range(self.num_layers):
            hs = rnn(seq, state[layer], *self.layer_weights(layer), nonlinearity=self.nonlinearity)
            finals.append(hs[-1])
            seq = self._next_input(hs, layer)
        return self._output(hs), ops.stack(finals, 0)

    def __call__(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        return self.forward(x, h0)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, nonlinearity={self.nonlinearity!r}"


class LSTM(_Recurrent):
    """Multi-layer long short-term memory (see :func:`tensorgrad.ops.recurrent.lstm`).

    Takes the arguments of :class:`RNN` except ``nonlinearity``. Called with ``x`` and an
    optional ``(h0, c0)`` pair (each ``(num_layers, B, H)``, zeros by default), it returns
    ``(output, (h_n, c_n))``.
    """

    _gates = 4

    def forward(  # type: ignore[override]
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        seq = self._time_major(x)
        h0 = self._initial_state(None if state is None else state[0], seq)
        c0 = self._initial_state(None if state is None else state[1], seq)
        final_h, final_c = [], []
        hs = seq
        for layer in range(self.num_layers):
            packed = lstm(seq, h0[layer], c0[layer], *self.layer_weights(layer))
            hs = packed[:, :, 0]
            final_h.append(packed[-1, :, 0])
            final_c.append(packed[-1, :, 1])
            seq = self._next_input(hs, layer)
        return self._output(hs), (ops.stack(final_h, 0), ops.stack(final_c, 0))

    def __call__(  # type: ignore[override]
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return self.forward(x, state)


class GRU(_Recurrent):
    """Multi-layer gated recurrent unit (see :func:`tensorgrad.ops.recurrent.gru`).

    Takes the arguments of :class:`RNN` except ``nonlinearity`` and is called like it,
    returning ``(output, h_n)``.
    """

    _gates = 3

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        seq = self._time_major(x)
        state = self._initial_state(h0, seq)
        finals = []
        hs = seq
        for layer in range(self.num_layers):
            hs = gru(seq, state[layer], *self.layer_weights(layer))
            finals.append(hs[-1])
            seq = self._next_input(hs, layer)
        return self._output(hs), ops.stack(finals, 0)

    def __call__(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        return self.forward(x, h0)
