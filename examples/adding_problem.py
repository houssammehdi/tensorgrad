"""The adding problem: an LSTM and a GRU learn a long-range dependency that a plain RNN cannot.

    python examples/adding_problem.py --seed 0

Each input is a sequence of ``T`` steps with two features: a value drawn from ``U(0, 1)``
and a marker that is 1 at exactly two steps, one in each half of the sequence. The target is
the sum of the two marked values (Hochreiter and Schmidhuber, 1997). Predicting the mean,
1, scores a mean squared error of 1/6; beating that requires carrying the first marked value
across at least ``T / 2`` steps, and training requires gradients that survive being
backpropagated through as many.

The script trains an Elman RNN (tanh), a GRU and an LSTM of the same width with the same
optimiser, batches and seed, reports the test error, and measures how much gradient reaches
each input step, which shows why the gated cells win. The LSTM's forget-gate bias starts at
+1 (``--forget-bias 0`` keeps PyTorch's initialisation, which learns the task more slowly).
With matplotlib installed it saves both as docs/adding_problem.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import tensorgrad as tg
from _style import INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE, import_pyplot, style_axes
from tensorgrad import nn, optim

ROOT = Path(__file__).resolve().parent
DEFAULT_PLOT = ROOT.parent / "docs" / "adding_problem.png"
MODELS = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
COLORS = {"LSTM": SERIES[0], "GRU": SERIES[1], "RNN": SERIES[2]}
BASELINE = 1.0 / 6.0  # MSE of always predicting 1 (the variance of a sum of two U(0, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--length", type=int, default=100, help="sequence length T")
    parser.add_argument("--steps", type=int, default=3000, help="optimisation steps per model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default="RNN,GRU,LSTM", help="comma-separated subset")
    parser.add_argument(
        "--forget-bias", type=float, default=1.0, help="added to the LSTM forget-gate bias"
    )
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def make_batch(rng: np.random.Generator, length: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """``(T, size, 2)`` inputs (value, marker) and ``(size, 1)`` targets."""
    values = rng.uniform(0.0, 1.0, size=(length, size))
    half = length // 2
    first = rng.integers(0, half, size=size)
    second = rng.integers(half, length, size=size)
    markers = np.zeros((length, size))
    markers[first, np.arange(size)] = 1.0
    markers[second, np.arange(size)] = 1.0
    x = np.stack([values, markers], axis=-1).astype(np.float32)
    y = (values * markers).sum(axis=0)[:, None].astype(np.float32)
    return x, y


class Regressor(nn.Module):
    """A recurrent layer read out by a linear map from its final hidden state."""

    def __init__(self, cell: type[nn.RNN] | type[nn.GRU] | type[nn.LSTM], hidden: int) -> None:
        super().__init__()
        self.rnn = cell(2, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: tg.Tensor) -> tg.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[-1])


def test_error(model: Regressor, x: np.ndarray, y: np.ndarray) -> float:
    with tg.no_grad():
        return tg.mse_loss(model(tg.Tensor(x)), y).item()


def gradient_profile(model: Regressor, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean ``|dL/dx_t|`` over the batch and features for every step ``t``."""
    xt = tg.Tensor(x, requires_grad=True)
    tg.mse_loss(model(xt), y).backward()
    assert xt.grad is not None
    return np.abs(xt.grad.data).mean(axis=(1, 2))  # type: ignore[no-any-return]


def train(
    name: str, args: argparse.Namespace, test: tuple[np.ndarray, np.ndarray]
) -> dict[str, object]:
    tg.manual_seed(args.seed)  # the same initial weights for every width-matched model
    model = Regressor(MODELS[name], args.hidden)
    if isinstance(model.rnn, nn.LSTM):
        # Start with the forget gate mostly open so the cell state (and its gradient) is
        # carried across many steps (Gers et al., 2000; Jozefowicz et al., 2015).
        model.rnn.bias_ih_l0.data[args.hidden : 2 * args.hidden] += args.forget_bias
    params = list(model.parameters())
    opt = optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)  # the same batches for every model
    probe = make_batch(np.random.default_rng(args.seed + 2), args.length, 256)
    initial_profile = gradient_profile(model, *probe)
    curve: list[tuple[int, float]] = [(0, test_error(model, *test))]
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = make_batch(rng, args.length, args.batch_size)
        loss = tg.mse_loss(model(tg.Tensor(x)), y)
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.eval_interval == 0 or step == args.steps:
            curve.append((step, test_error(model, *test)))
    seconds = time.perf_counter() - start
    solved = next((s for s, e in curve if e < 0.01), None)
    print(
        f"{name:5s} test MSE {curve[-1][1]:.4f} (baseline {BASELINE:.4f}); "
        f"below 0.01 at step {solved if solved is not None else '-'}; "
        f"{1000 * seconds / args.steps:.0f} ms/step"
    )
    return {
        "curve": curve,
        "initial_profile": initial_profile,
        "final_profile": gradient_profile(model, *probe),
    }


def save_plot(results: dict[str, dict[str, object]], args: argparse.Namespace, path: Path) -> bool:
    plt = import_pyplot()
    if plt is None:
        return False
    from matplotlib.lines import Line2D

    fig, (ax_mse, ax_grad) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for name, result in results.items():
        steps, errors = np.array(result["curve"]).T
        ax_mse.plot(steps, errors, color=COLORS[name], linewidth=2, label=name)
        t = np.arange(1, args.length + 1)
        ax_grad.plot(t, np.asarray(result["final_profile"]), color=COLORS[name], linewidth=2)
        ax_grad.plot(
            t, np.asarray(result["initial_profile"]), color=COLORS[name], linewidth=1, ls="--"
        )
    ax_mse.axhline(BASELINE, color=INK_MUTED, linewidth=1, linestyle=":")
    ax_mse.annotate(
        "predicting the mean",
        (args.steps, BASELINE),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("training step", fontsize=9)
    ax_mse.set_ylabel("test MSE", fontsize=9)
    style_axes(ax_mse, f"Adding problem, T = {args.length}: test error")
    ax_mse.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower left")
    ax_grad.set_yscale("log")
    ax_grad.set_xlabel("input step t (the loss is computed after step T)", fontsize=9)
    ax_grad.set_ylabel("mean |dL / dx_t|", fontsize=9)
    style_axes(ax_grad, "Gradient reaching each input step")
    styles = [
        Line2D([], [], color=INK_SECONDARY, linewidth=2, label="after training"),
        Line2D([], [], color=INK_SECONDARY, linewidth=1, ls="--", label="at initialisation"),
    ]
    ax_grad.legend(handles=styles, frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    for ax in (ax_mse, ax_grad):
        ax.title.set_color(INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = sorted(set(names) - set(MODELS))
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {sorted(MODELS)}")
    test = make_batch(np.random.default_rng(args.seed + 1), args.length, args.test_size)
    print(
        f"adding problem: T = {args.length}, hidden {args.hidden}, {args.steps} steps of "
        f"batch {args.batch_size}; predicting the mean scores {BASELINE:.4f}"
    )
    results = {name: train(name, args, test) for name in names}
    for name, result in results.items():
        before, after = np.asarray(result["initial_profile"]), np.asarray(result["final_profile"])
        print(
            f"{name:5s} |dL/dx_1| / |dL/dx_T|: {before[0] / before[-1]:.1e} at initialisation, "
            f"{after[0] / after[-1]:.1e} after training"
        )
    if not args.no_plot and save_plot(results, args, args.plot):
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
