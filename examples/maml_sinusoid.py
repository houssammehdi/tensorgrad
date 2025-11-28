"""MAML on sinusoid regression: meta-learning through a gradient step (second-order autodiff).

    python examples/maml_sinusoid.py --seed 0

The few-shot regression benchmark of Finn, Abbeel & Levine (2017): every task is a sinusoid
``y = A sin(x - phase)`` with ``A ~ U[0.1, 5]``, ``phase ~ U[0, pi]``, ``x ~ U[-5, 5]``. A
``1 -> 40 -> 40 -> 1`` ReLU network sees ``K = 10`` points of a new task and takes one step of
SGD on them. MAML learns the *initialisation* such that this one step fits the task well: the
meta-loss is the query loss *after* the inner step, so its gradient flows back through the
inner gradient -- that is a Hessian-vector product, which ``create_graph=True`` provides.

Three initialisations are compared on the same held-out tasks after 0-10 SGD steps:

* **MAML**: the exact meta-gradient (second order),
* **first-order MAML**: the inner gradient treated as a constant (no second derivatives),
* **pretrained**: one network regressed on all tasks jointly, then fine-tuned the same way.

All tasks of a meta-batch are processed at once: the shared weights are broadcast to a
leading task axis, so one ``autograd.grad`` yields every task's inner gradient, and the
meta-gradient is summed back through ``broadcast_to``'s VJP.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tensorgrad as tg
from _style import INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE, import_pyplot, style_axes
from tensorgrad import autograd, nn, optim

DEFAULT_PLOT = Path(__file__).resolve().parent.parent / "docs" / "maml_sinusoid.png"
HIDDEN = 40
Params = list[tg.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=20000, help="meta-training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meta-batch", type=int, default=25, help="tasks per meta-step")
    parser.add_argument("--shots", type=int, default=10, help="K support points per task")
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--meta-lr", type=float, default=1e-3)
    parser.add_argument("--test-tasks", type=int, default=600)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="output PNG path")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


@dataclass
class Tasks:
    """A batch of sinusoid tasks: support and query points, ``(T, n, 1)`` arrays."""

    amplitude: np.ndarray
    phase: np.ndarray
    x_support: np.ndarray
    y_support: np.ndarray
    x_query: np.ndarray
    y_query: np.ndarray


def sample_tasks(
    rng: np.random.Generator, n_tasks: int, shots: int, query: np.ndarray | int
) -> Tasks:
    """``query`` is either a number of random query points or a fixed grid of x values."""
    amplitude = rng.uniform(0.1, 5.0, (n_tasks, 1, 1))
    phase = rng.uniform(0.0, np.pi, (n_tasks, 1, 1))
    x_support = rng.uniform(-5.0, 5.0, (n_tasks, shots, 1))
    if isinstance(query, int):
        x_query = rng.uniform(-5.0, 5.0, (n_tasks, query, 1))
    else:
        x_query = np.broadcast_to(query.reshape(1, -1, 1), (n_tasks, query.size, 1)).copy()
    return Tasks(
        amplitude,
        phase,
        x_support,
        amplitude * np.sin(x_support - phase),
        x_query,
        amplitude * np.sin(x_query - phase),
    )


def init_params(rng: np.random.Generator) -> Params:
    """PyTorch-style uniform initialisation; biases stored as ``(1, n)`` rows."""
    shapes = [(1, HIDDEN), (HIDDEN, HIDDEN), (HIDDEN, 1)]
    params: Params = []
    for fan_in, fan_out in shapes:
        bound = 1.0 / np.sqrt(fan_in)
        params.append(nn.Parameter(rng.uniform(-bound, bound, (fan_in, fan_out))))
        params.append(nn.Parameter(rng.uniform(-bound, bound, (1, fan_out))))
    return params


def forward(params: Params, x: np.ndarray) -> tg.Tensor:
    """The MLP with an explicit (possibly per-task) set of weights; ``x`` is ``(T, n, 1)``."""
    w1, b1, w2, b2, w3, b3 = params
    h = tg.relu(tg.Tensor(x) @ w1 + b1)
    h = tg.relu(h @ w2 + b2)
    return h @ w3 + b3


def task_losses(params: Params, x: np.ndarray, y: np.ndarray) -> tg.Tensor:
    """Mean squared error of every task, shape ``(T,)``."""
    return ((forward(params, x) - y) ** 2).mean(axis=(1, 2))


def per_task(params: Params, n_tasks: int) -> Params:
    """Broadcast shared weights to a leading task axis: one copy (view) per task."""
    return [tg.broadcast_to(p, (n_tasks, *p.shape)) for p in params]


def adapt(params: Params, tasks: Tasks, lr: float, steps: int, second_order: bool) -> Params:
    """``steps`` of SGD on each task's support loss, starting from ``params`` (per task)."""
    fast = params
    for _ in range(steps):
        support = task_losses(fast, tasks.x_support, tasks.y_support).sum()
        grads = autograd.grad(support, fast, create_graph=second_order, retain_graph=True)
        fast = [p - lr * g for p, g in zip(fast, grads, strict=True)]
    return fast


Trainer = Callable[[Params, Tasks], tg.Tensor]


def maml_objective(inner_lr: float, second_order: bool) -> Trainer:
    def objective(params: Params, tasks: Tasks) -> tg.Tensor:
        fast = adapt(per_task(params, len(tasks.amplitude)), tasks, inner_lr, 1, second_order)
        return task_losses(fast, tasks.x_query, tasks.y_query).mean()

    return objective


def pretrain_objective(params: Params, tasks: Tasks) -> tg.Tensor:
    # Joint regression on all points of all tasks: the usual "pretrain, then fine-tune".
    x = np.concatenate([tasks.x_support, tasks.x_query], axis=1)
    y = np.concatenate([tasks.y_support, tasks.y_query], axis=1)
    return task_losses(per_task(params, len(x)), x, y).mean()


def train(objective: Trainer, args: argparse.Namespace) -> tuple[Params, float]:
    # Every method starts from the same weights and sees the same stream of tasks.
    rng = np.random.default_rng(args.seed + 100)
    params = init_params(np.random.default_rng(args.seed))
    opt = optim.Adam(params, lr=args.meta_lr)
    start = time.perf_counter()
    for _ in range(args.iterations):
        tasks = sample_tasks(rng, args.meta_batch, args.shots, args.shots)
        loss = objective(params, tasks)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return params, time.perf_counter() - start


def evaluate(params: Params, tasks: Tasks, lr: float, max_steps: int) -> np.ndarray:
    """Query MSE of every task after 0..max_steps SGD steps, shape ``(max_steps + 1, T)``."""
    mse = []
    fast = per_task(params, len(tasks.amplitude))
    for step in range(max_steps + 1):
        if step:
            fast = [p.detach().requires_grad_() for p in adapt(fast, tasks, lr, 1, False)]
        with tg.no_grad():
            mse.append(task_losses(fast, tasks.x_query, tasks.y_query).data)
    return np.array(mse)


def save_plot(
    models: dict[str, Params], curves: dict[str, np.ndarray], lr: float, shots: int, path: Path
) -> bool:
    plt = import_pyplot()
    if plt is None:
        return False
    # The illustrated task is fixed (amplitude 3, phase 0.5) rather than picked for a good fit;
    # the right-hand panel summarises all held-out tasks.
    grid = np.linspace(-5, 5, 200)
    demo = sample_tasks(np.random.default_rng(12345), 1, shots, grid)
    demo.amplitude[...], demo.phase[...] = 3.0, 0.5
    demo.y_support = 3.0 * np.sin(demo.x_support - 0.5)
    demo.y_query = 3.0 * np.sin(demo.x_query - 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    colors = dict(zip(curves, SERIES, strict=False))
    for ax, name in zip(axes[:2], ("MAML", "pretrained"), strict=True):
        ax.plot(grid, demo.y_query[0, :, 0], color=INK_MUTED, linewidth=2, label="true function")
        for steps, alpha in ((0, 0.35), (1, 0.65), (10, 1.0)):
            fast = per_task(models[name], 1)
            if steps:
                fast = adapt(fast, demo, lr, steps, second_order=False)
            with tg.no_grad():
                pred = forward(fast, demo.x_query).data[0, :, 0]
            label = "before adaptation" if steps == 0 else f"after {steps} step{'s' * (steps > 1)}"
            ax.plot(grid, pred, color=colors[name], alpha=alpha, linewidth=2, label=label)
        ax.plot(
            demo.x_support[0, :, 0],
            demo.y_support[0, :, 0],
            "o",
            color=INK,
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=f"{shots} support points",
        )
        style_axes(ax, f"{name} initialisation")
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylim(-5.0, 7.5)  # headroom above the curves for the legend
        ax.legend(frameon=False, fontsize=7, labelcolor=INK_SECONDARY, loc="upper center", ncol=2)
    ax = axes[2]
    for name, mse in curves.items():
        steps = np.arange(len(mse))
        mean = mse.mean(axis=1)
        half = 1.96 * mse.std(axis=1) / np.sqrt(mse.shape[1])
        ax.fill_between(steps, mean - half, mean + half, color=colors[name], alpha=0.12, lw=0)
        ax.plot(steps, mean, color=colors[name], linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, label=name)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("SGD steps on the 10 support points", fontsize=9)
    ax.set_ylabel("query MSE (held-out tasks)", fontsize=9)
    style_axes(ax, "Error after adaptation (95% CI)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    trainers: dict[str, Trainer] = {
        "MAML": maml_objective(args.inner_lr, second_order=True),
        "first-order MAML": maml_objective(args.inner_lr, second_order=False),
        "pretrained": pretrain_objective,
    }
    test_tasks = sample_tasks(
        np.random.default_rng(args.seed + 1), args.test_tasks, args.shots, np.linspace(-5, 5, 100)
    )
    models: dict[str, Params] = {}
    curves: dict[str, np.ndarray] = {}
    print(
        f"{args.iterations} meta-steps of {args.meta_batch} tasks, K = {args.shots}, "
        f"inner lr {args.inner_lr}; evaluated on {args.test_tasks} held-out tasks"
    )
    print(f"{'method':18s} {'train s':>8s}  query MSE after 0 / 1 / 10 steps (mean ± 95% CI)")
    for name, objective in trainers.items():
        params, seconds = train(objective, args)
        mse = evaluate(params, test_tasks, args.inner_lr, max_steps=10)
        models[name], curves[name] = params, mse
        cells = []
        for k in (0, 1, 10):
            half = 1.96 * mse[k].std() / np.sqrt(mse.shape[1])
            cells.append(f"{mse[k].mean():.3f} ± {half:.3f}")
        print(f"{name:18s} {seconds:8.1f}  " + " / ".join(cells))
    if not args.no_plot and save_plot(models, curves, args.inner_lr, args.shots, args.plot):
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
