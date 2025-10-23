"""End-to-end training: the whole stack (ops, engine, nn, optim) has to cooperate.

The fast tests run in CI on every push; the ``slow`` ones are longer runs enabled with
``pytest --run-slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import nn, optim
from tensorgrad.datasets import CharTokenizer, make_shapes, make_spiral
from tensorgrad.utils import DataLoader


def accuracy(logits: tg.Tensor, labels: np.ndarray) -> float:
    return float((logits.data.argmax(axis=-1) == labels).mean())


def train_spiral(steps: int) -> tuple[float, float]:
    x, y = make_spiral(100, 3, noise=0.2, seed=0)
    model = nn.MLP([2, 64, 64, 3])
    opt = optim.Adam(model.parameters(), lr=0.01)
    xt = tg.Tensor(x)
    for _ in range(steps):
        loss = tg.cross_entropy(model(xt), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with tg.no_grad():
        return accuracy(model(xt), y), loss.item()


def test_spiral_mlp_reaches_high_accuracy_in_seconds() -> None:
    # 200 full-batch Adam steps on 300 points: well under a second single-threaded.
    acc, loss = train_spiral(steps=200)
    assert acc > 0.9
    assert loss < 0.2


def test_linear_regression_recovers_weights_with_sgd() -> None:
    rng = np.random.default_rng(0)
    true_w = np.array([[2.0], [-3.0], [0.5]], dtype=np.float32)
    x = rng.standard_normal((256, 3)).astype(np.float32)
    y = x @ true_w + 1.0
    layer = nn.Linear(3, 1)
    opt = optim.SGD(layer.parameters(), lr=0.05, momentum=0.9, nesterov=True)
    for xb, yb in [b for _ in range(30) for b in DataLoader(x, y, batch_size=32, shuffle=True)]:
        loss = tg.mse_loss(layer(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    np.testing.assert_allclose(layer.weight.data.T, true_w, atol=1e-3)
    assert layer.bias is not None
    np.testing.assert_allclose(layer.bias.data, [1.0], atol=1e-3)


def small_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 4),
    )


def train_cnn(n_train: int, epochs: int) -> tuple[list[float], float]:
    x, y = make_shapes(n_train, seed=0)
    x_test, y_test = make_shapes(400, seed=1)
    model = small_cnn()
    opt = optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in DataLoader(x, y, batch_size=32, shuffle=True, seed=0):
            loss = tg.cross_entropy(model(xb), yb.data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        losses.append(total / n_train)
    with tg.no_grad():
        return losses, accuracy(model(tg.Tensor(x_test)), y_test)


def test_cnn_loss_decreases_on_shapes() -> None:
    losses, test_acc = train_cnn(n_train=320, epochs=3)
    assert losses[-1] < losses[0]
    assert test_acc > 0.4  # well above the 25% chance level after a few hundred images


TEXT = "to be, or not to be, that is the question. " * 20


def train_char_gpt(text: str, steps: int, seed: int = 0) -> tuple[float, float]:
    tok = CharTokenizer(text)
    data = tok.encode(text)
    config = nn.GPTConfig(tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = nn.GPT(config)
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    first = last = 0.0
    for step in range(steps):
        starts = rng.integers(0, len(data) - config.block_size - 1, size=16)
        xb = np.stack([data[s : s + config.block_size] for s in starts])
        yb = np.stack([data[s + 1 : s + 1 + config.block_size] for s in starts])
        loss = tg.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    return first, last


def test_char_gpt_loss_decreases() -> None:
    first, last = train_char_gpt(TEXT, steps=60)
    assert first == pytest.approx(np.log(len(set(TEXT))), rel=0.1)  # starts near uniform
    assert last < 0.6 * first


@pytest.mark.slow
def test_cnn_reaches_high_test_accuracy() -> None:
    _, test_acc = train_cnn(n_train=3000, epochs=6)
    assert test_acc > 0.9


@pytest.mark.slow
def test_char_gpt_memorises_a_repeated_phrase() -> None:
    _, last = train_char_gpt(TEXT, steps=400)
    assert last < 0.3
