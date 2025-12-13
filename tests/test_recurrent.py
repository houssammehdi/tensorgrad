"""Recurrent ops and layers: values against plain NumPy loops, BPTT gradients, module API."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tensorgrad as tg
from helpers import check_gradients, leaf
from tensorgrad import nn, optim
from tensorgrad.ops.recurrent import gru, lstm, rnn
from tensorgrad.utils import gradcheck

# Every example runs first- and second-order finite differences through the unrolled
# sequence, so these tests draw a quarter of the profile's examples.
FEWER = settings(max_examples=max(4, settings().max_examples // 4))
GATES = {"rnn": 1, "rnn_relu": 1, "lstm": 4, "gru": 3}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def reference(
    kind: str, x: np.ndarray, h0: np.ndarray, c0: np.ndarray, *w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The textbook recurrences, one step at a time (PyTorch's gate layouts)."""
    w_ih, w_hh = w[:2]
    b_ih, b_hh = w[2:] if len(w) == 4 else (0.0, 0.0)
    h, c, hs, cs = h0, c0, [], []
    for x_t in x:
        a, b = x_t @ w_ih.T + b_ih, h @ w_hh.T + b_hh
        if kind == "lstm":
            i, f, g, o = np.split(a + b, 4, axis=1)
            c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
            h = sigmoid(o) * np.tanh(c)
        elif kind == "gru":
            (ar, au, an), (br, bu, bn) = np.split(a, 3, axis=1), np.split(b, 3, axis=1)
            r, u = sigmoid(ar + br), sigmoid(au + bu)
            n = np.tanh(an + r * bn)
            h = (1 - u) * n + u * h
        else:
            h = np.tanh(a + b) if kind == "rnn" else np.maximum(a + b, 0.0)
        hs.append(h)
        cs.append(c)
    return np.stack(hs), np.stack(cs)


def sequence_op(kind: str) -> Callable[..., tg.Tensor]:
    """The fused op for ``kind`` with a uniform ``(x, h0, c0, *weights)`` signature."""
    if kind == "lstm":
        return lambda x, h0, c0, *w: lstm(x, h0, c0, *w)
    if kind == "gru":
        return lambda x, h0, c0, *w: gru(x, h0, *w)
    nonlinearity = "relu" if kind == "rnn_relu" else "tanh"
    return lambda x, h0, c0, *w: rnn(x, h0, *w, nonlinearity=nonlinearity)  # type: ignore[arg-type]


def make_inputs(
    rng: np.random.Generator, kind: str, t: int, b: int, i: int, h: int, bias: bool
) -> list[tg.Tensor]:
    rows = GATES[kind] * h
    shapes = [(t, b, i), (b, h), (b, h), (rows, i), (rows, h)] + ([(rows,)] * 2 if bias else [])
    inputs = [leaf(rng, *s) for s in shapes]
    if kind == "rnn_relu":  # keep pre-activations away from the kink at 0
        inputs[0] = tg.Tensor(inputs[0].data + np.sign(inputs[0].data), requires_grad=True)
    return inputs


@pytest.mark.parametrize("kind", sorted(GATES))
@pytest.mark.parametrize("bias", [True, False])
def test_fused_ops_match_step_by_step_numpy(kind: str, bias: bool) -> None:
    rng = np.random.default_rng(0)
    inputs = make_inputs(rng, kind, 7, 3, 4, 5, bias)
    out = sequence_op(kind)(*inputs).data
    hs, cs = reference(kind, *(t.data for t in inputs))
    if kind == "lstm":
        np.testing.assert_allclose(out[:, :, 0], hs, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(out[:, :, 1], cs, rtol=1e-12, atol=1e-14)
    else:
        np.testing.assert_allclose(out, hs, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kind", sorted(GATES))
@FEWER
@given(
    t=st.integers(1, 3),
    b=st.integers(1, 2),
    i=st.integers(1, 3),
    h=st.integers(1, 2),
    bias=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_bptt_first_and_second_order(
    kind: str, t: int, b: int, i: int, h: int, bias: bool, seed: int
) -> None:
    inputs = make_inputs(np.random.default_rng(seed), kind, t, b, i, h, bias)
    if kind != "lstm":  # c0 is unused: keep it out of the check
        inputs[2].requires_grad = False
    assert check_gradients(sequence_op(kind), inputs)


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_gradients_reach_the_start_of_a_long_sequence(kind: str) -> None:
    rng = np.random.default_rng(1)
    inputs = make_inputs(rng, kind, 25, 2, 2, 3, True)
    op = sequence_op(kind)
    only_last = lambda *args: op(*args)[-1]
    assert gradcheck(only_last, inputs)
    out = only_last(*inputs)
    out.sum().backward()
    assert inputs[0].grad is not None
    assert np.abs(inputs[0].grad.data[0]).max() > 1e-8  # BPTT reached step 0


def test_gradient_masks_skip_unneeded_work() -> None:
    rng = np.random.default_rng(2)
    x, h0, c0, w_ih, w_hh, b_ih, b_hh = make_inputs(rng, "lstm", 4, 2, 3, 2, True)
    x.requires_grad = h0.requires_grad = c0.requires_grad = False
    lstm(x, h0, c0, w_ih, w_hh, b_ih, b_hh).sum().backward()
    assert all(p.grad is not None for p in (w_ih, w_hh, b_ih, b_hh))
    assert x.grad is None and h0.grad is None


def test_validation() -> None:
    x, h0 = tg.zeros((3, 2, 4)), tg.zeros((2, 5))
    w_ih, w_hh = tg.zeros((20, 4)), tg.zeros((20, 5))
    with pytest.raises(ValueError, match="both biases"):
        lstm(x, h0, h0, w_ih, w_hh, tg.zeros(20))
    with pytest.raises(ValueError, match="non-empty"):
        lstm(tg.zeros((0, 2, 4)), h0, h0, w_ih, w_hh)
    with pytest.raises(ValueError, match="do not fit"):
        gru(x, h0, w_ih, w_hh)  # four gates' worth of rows for a three-gate cell
    with pytest.raises(ValueError, match="initial state"):
        lstm(x, tg.zeros((3, 5)), h0, w_ih, w_hh)
    with pytest.raises(ValueError, match="nonlinearity"):
        rnn(x, h0, tg.zeros((5, 4)), tg.zeros((5, 5)), nonlinearity="sigmoid")  # type: ignore[arg-type]


class TestModules:
    @pytest.mark.parametrize("cls", [nn.RNN, nn.LSTM, nn.GRU])
    @pytest.mark.parametrize("batch_first", [False, True])
    def test_shapes_and_final_states(self, cls: type, batch_first: bool) -> None:
        model = cls(3, 5, num_layers=2, batch_first=batch_first)
        x = tg.randn((4, 6, 3))  # (B, T) if batch_first else (T, B)
        output, state = model(x)
        assert output.shape == (4, 6, 5)
        h_n = state[0] if cls is nn.LSTM else state
        assert h_n.shape == (2, 4 if batch_first else 6, 5)
        last = output.data[:, -1] if batch_first else output.data[-1]
        np.testing.assert_allclose(h_n.data[-1], last)

    def test_parameter_names_follow_pytorch(self) -> None:
        names = [n for n, _ in nn.LSTM(3, 4, num_layers=2).named_parameters()]
        assert names == [
            f"{kind}_{part}_l{layer}"
            for layer in range(2)
            for kind, part in (("weight", "ih"), ("weight", "hh"), ("bias", "ih"), ("bias", "hh"))
        ]
        assert nn.GRU(3, 4).weight_ih_l0.shape == (12, 3)
        assert [n for n, _ in nn.RNN(3, 4, bias=False).named_parameters()] == [
            "weight_ih_l0",
            "weight_hh_l0",
        ]

    def test_initial_state_is_used_and_receives_gradient(self) -> None:
        model = nn.LSTM(2, 3)
        x = tg.randn((5, 4, 2))
        h0 = tg.Tensor(np.full((1, 4, 3), 0.5, dtype=np.float32), requires_grad=True)
        c0 = tg.Tensor(np.full((1, 4, 3), -0.5, dtype=np.float32), requires_grad=True)
        with_state, _ = model(x, (h0, c0))
        without, _ = model(x)
        assert not np.allclose(with_state.data, without.data)
        with_state.sum().backward()
        assert h0.grad is not None and c0.grad is not None
        with pytest.raises(ValueError, match="initial state"):
            model(x, (h0[:, :2], c0))

    def test_dropout_only_between_layers_and_only_in_training(self) -> None:
        model = nn.GRU(3, 4, num_layers=2, dropout=0.5)
        x = tg.randn((6, 2, 3))
        tg.manual_seed(1)
        a, _ = model(x)
        tg.manual_seed(2)
        b, _ = model(x)
        assert not np.allclose(a.data, b.data)  # different masks between the layers
        model.eval()
        np.testing.assert_array_equal(model(x)[0].data, model(x)[0].data)
        single = nn.GRU(3, 4, dropout=0.5)  # one layer: nothing to drop between
        np.testing.assert_array_equal(single(x)[0].data, single(x)[0].data)

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="input_size 3"):
            nn.RNN(3, 4)(tg.randn((5, 2, 2)))
        with pytest.raises(ValueError, match="positive"):
            nn.LSTM(3, 0)
        with pytest.raises(ValueError, match="nonlinearity"):
            nn.RNN(3, 4, nonlinearity="gelu")  # type: ignore[arg-type]
        assert "num_layers=1" in repr(nn.RNN(3, 4))

    @pytest.mark.parametrize("cls", [nn.LSTM, nn.GRU])
    def test_learns_the_adding_problem(self, cls: type) -> None:
        # Sum the two marked values of a length-10 sequence: predicting the mean scores 1/6,
        # so a low loss needs gradients carried back to the marked steps.
        rng = np.random.default_rng(0)
        model, head = cls(2, 16), nn.Linear(16, 1)
        opt = optim.Adam([*model.parameters(), *head.parameters()], lr=0.01)
        losses = []
        for _ in range(400):
            values = rng.uniform(0, 1, (10, 32))
            marks = np.zeros((10, 32))
            for b in range(32):
                marks[rng.choice(10, 2, replace=False), b] = 1.0
            x = np.stack([values, marks], axis=-1).astype(np.float32)
            output, _ = model(tg.Tensor(x))
            loss = tg.mse_loss(head(output[-1]), (values * marks).sum(0, keepdims=True).T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert np.mean(losses[-50:]) < 0.04
