"""Train a small CNN on procedurally drawn 16x16 shapes (circle, square, triangle, cross).

    python examples/shapes_cnn.py --epochs 6 --seed 0

The images are generated in code (random position, size, rotation, stroke width and pixel
noise), so nothing is downloaded. Prints per-epoch metrics and a confusion matrix and, if
matplotlib is installed, saves a grid of test predictions to docs/shapes_predictions.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import tensorgrad as tg
from tensorgrad import nn, optim
from tensorgrad.datasets import SHAPE_CLASSES, make_shapes
from tensorgrad.utils import DataLoader

DEFAULT_PLOT = Path(__file__).resolve().parent.parent / "docs" / "shapes_predictions.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="output PNG path")
    parser.add_argument("--no-plot", action="store_true", help="skip the prediction grid")
    return parser.parse_args()


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),  # 16x16
        nn.ReLU(),
        nn.MaxPool2d(2),  # 8x8
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 4x4
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, len(SHAPE_CLASSES)),
    )


def predict(model: nn.Module, images: np.ndarray, batch_size: int = 500) -> np.ndarray:
    model.eval()
    preds = []
    with tg.no_grad():
        for (xb,) in DataLoader(images, batch_size=batch_size):
            preds.append(model(xb).data.argmax(axis=1))
    model.train()
    return np.concatenate(preds)


def save_grid(images: np.ndarray, labels: np.ndarray, preds: np.ndarray, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return False
    rows, cols = 3, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05, rows * 1.3), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    # Show every mistake (up to 4) alongside correctly classified examples.
    wrong = np.flatnonzero(preds != labels)[:4]
    right = np.flatnonzero(preds == labels)[: rows * cols - len(wrong)]
    order = np.concatenate([right, wrong]).astype(int)
    for i, ax in zip(order, axes.ravel(), strict=True):
        ax.imshow(images[i, 0], cmap="gray_r", vmin=0, vmax=1)
        ok = preds[i] == labels[i]
        text = (
            SHAPE_CLASSES[preds[i]]
            if ok
            else f"{SHAPE_CLASSES[preds[i]]} (true: {SHAPE_CLASSES[labels[i]]})"
        )
        ax.set_title(text, fontsize=7, color="#52514e" if ok else "#e34948")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d9d8d4")
    fig.suptitle("CNN predictions on held-out synthetic shapes", fontsize=9, color="#0b0b0b")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    tg.manual_seed(args.seed)
    x_train, y_train = make_shapes(args.train_size, seed=args.seed)
    x_test, y_test = make_shapes(args.test_size, seed=args.seed + 10_000)
    model = build_model()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(x_train, y_train, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    sched = optim.CosineWarmupLR(
        opt, warmup_steps=len(loader), total_steps=args.epochs * len(loader)
    )
    print(
        f"CNN with {model.num_parameters()} parameters; {len(x_train)} train / {len(x_test)} test"
    )

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        total, correct = 0.0, 0
        for xb, yb in loader:
            logits = model(xb)
            loss = tg.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
            correct += int((logits.data.argmax(axis=1) == yb.data).sum())
        test_acc = float((predict(model, x_test) == y_test).mean())
        print(
            f"epoch {epoch}  loss {total / len(x_train):.4f}  "
            f"train acc {correct / len(x_train):.3f}  test acc {test_acc:.3f}  "
            f"({time.perf_counter() - start:.1f}s)"
        )

    preds = predict(model, x_test)
    confusion = np.zeros((len(SHAPE_CLASSES),) * 2, dtype=int)
    np.add.at(confusion, (y_test, preds), 1)
    print("\nconfusion matrix (rows = true, columns = predicted)")
    print(" " * 10 + "".join(f"{name:>10}" for name in SHAPE_CLASSES))
    for name, row in zip(SHAPE_CLASSES, confusion, strict=True):
        print(f"{name:>10}" + "".join(f"{v:>10d}" for v in row))
    print(f"\nfinal test accuracy {float((preds == y_test).mean()):.3f}")
    if not args.no_plot and save_grid(x_test, y_test, preds, args.plot):
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
