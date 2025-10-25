"""Classify a three-armed 2-D spiral with an MLP.

    python examples/spiral_mlp.py --epochs 300 --seed 0

Trains full-batch with Adam, reports train/test accuracy and (if matplotlib is installed)
saves the learned decision boundary to docs/spiral_decision_boundary.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import tensorgrad as tg
from tensorgrad import nn, optim
from tensorgrad.datasets import make_spiral

DEFAULT_PLOT = Path(__file__).resolve().parent.parent / "docs" / "spiral_decision_boundary.png"
# Reference palette: first three categorical slots (validated for all-pairs use).
CLASS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=300, help="full-batch training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--points", type=int, default=200, help="points per class")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="output PNG path")
    parser.add_argument("--no-plot", action="store_true", help="skip the decision-boundary plot")
    return parser.parse_args()


def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    with tg.no_grad():
        return float((model(tg.Tensor(x)).data.argmax(axis=1) == y).mean())


def save_plot(model: nn.Module, x: np.ndarray, y: np.ndarray, path: Path, title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return False
    grid = np.linspace(-1.2, 1.2, 300, dtype=np.float32)
    gx, gy = np.meshgrid(grid, grid)
    with tg.no_grad():
        pred = model(tg.Tensor(np.stack([gx.ravel(), gy.ravel()], axis=1))).data.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.contourf(
        gx,
        gy,
        pred.reshape(gx.shape),
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=ListedColormap(CLASS_COLORS),
        alpha=0.18,
    )
    for c, color in enumerate(CLASS_COLORS):
        pts = x[y == c]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=14,
            color=color,
            edgecolors="#fcfcfb",
            linewidths=0.6,
            label=f"class {c}",
        )
    ax.set_title(title, color="#0b0b0b", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d9d8d4")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor="#52514e",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    tg.manual_seed(args.seed)
    x, y = make_spiral(args.points, 3, noise=0.2, seed=args.seed)
    x_test, y_test = make_spiral(args.points, 3, noise=0.2, seed=args.seed + 1)

    model = nn.MLP([2, args.hidden, args.hidden, 3])
    opt = optim.Adam(model.parameters(), lr=args.lr)
    xt = tg.Tensor(x)
    print(f"MLP with {model.num_parameters()} parameters, {len(x)} training points")

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = tg.cross_entropy(model(xt), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % max(args.epochs // 6, 1) == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:4d}  loss {loss.item():.4f}  train acc {accuracy(model, x, y):.3f}"
            )
    elapsed = time.perf_counter() - start

    train_acc, test_acc = accuracy(model, x, y), accuracy(model, x_test, y_test)
    print(f"train accuracy {train_acc:.3f} | test accuracy {test_acc:.3f} | {elapsed:.2f}s")
    if not args.no_plot:
        title = f"MLP decision boundary (train {train_acc:.1%}, test {test_acc:.1%})"
        if save_plot(model, x, y, args.plot, title):
            print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
