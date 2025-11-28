"""Exact Newton's method versus gradient descent on L2-regularised logistic regression.

    python examples/newton_logreg.py --seed 0

Second-order autodiff at work. The loss is the mean binary cross-entropy of a logistic model
plus ``lam/2 * |w|^2`` on an ill-conditioned problem (correlated features on scales from 1 to
30). Three optimisers start from ``w = 0``:

* **gradient descent** with the classical step ``1/L`` (``L`` bounds the Hessian),
* **Newton** with the exact Hessian from ``tensorgrad.func.hessian`` (one extra backward pass
  per parameter) and a backtracking line search,
* **Newton-CG**, which never forms the Hessian: it builds the gradient's graph once per
  iteration (``autograd.grad(..., create_graph=True)``) and gets every Hessian-vector product
  from one more backward pass through it, inside conjugate gradients.

It reports iterations, backward passes and wall time to reach a suboptimality of 1e-10 and
(if matplotlib is installed) plots the convergence to docs/newton_logreg.png.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import tensorgrad as tg
from _style import INK, INK_SECONDARY, SERIES, SURFACE, import_pyplot, style_axes
from tensorgrad import autograd, func

DEFAULT_PLOT = Path(__file__).resolve().parent.parent / "docs" / "newton_logreg.png"
Loss = Callable[[tg.Tensor], tg.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--lam", type=float, default=1e-3, help="L2 regularisation")
    parser.add_argument("--gd-steps", type=int, default=5000, help="gradient-descent budget")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="output PNG path")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def make_problem(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Features with AR(1) correlation 0.8 and geometric scales 1..30, plus an intercept."""
    rng = np.random.default_rng(seed)
    corr = 0.8 ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    scales = np.geomspace(1.0, 30.0, d)
    x = rng.standard_normal((n, d)) @ np.linalg.cholesky(corr).T * scales
    w_true = 2.0 * rng.standard_normal(d) / scales
    p = 1.0 / (1.0 + np.exp(-(x @ w_true + 0.5)))
    y = (rng.random(n) < p).astype(np.float64)
    return np.hstack([x, np.ones((n, 1))]), y


def make_loss(x: np.ndarray, y: np.ndarray, lam: float) -> Loss:
    features = tg.Tensor(x)

    def loss(w: tg.Tensor) -> tg.Tensor:
        return tg.binary_cross_entropy_with_logits(features @ w, y) + 0.5 * lam * (w * w).sum()

    return loss


@dataclass
class Trace:
    """Loss value, cumulative backward passes and wall time after every iteration."""

    name: str
    values: list[float] = field(default_factory=list)
    backward_passes: list[int] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)

    def record(self, value: float, passes: int, start: float) -> None:
        self.values.append(value)
        self.backward_passes.append(passes)
        self.seconds.append(time.perf_counter() - start)


def value_and_grad(loss: Loss, w: np.ndarray) -> tuple[float, np.ndarray]:
    with tg.no_grad():  # plain numbers: no graph is kept for the result
        value, g = func.value_and_grad(loss)(w)
    return value.item(), g.data


def backtracking(
    loss: Loss, w: np.ndarray, step: np.ndarray, value: float, slope: float, t: float = 1.0
) -> tuple[float, int]:
    """Armijo backtracking (c = 1e-4) from ``t``; returns the step and the loss evaluations."""
    evaluations = 0
    with tg.no_grad():
        while t > 1e-12:
            evaluations += 1
            if loss(tg.Tensor(w + t * step)).item() <= value + 1e-4 * t * slope:
                break
            t *= 0.5
    return t, evaluations


def gradient_descent(loss: Loss, w0: np.ndarray, lr: float, steps: int) -> Trace:
    """Fixed step ``lr = 1/L``: the classical guarantee for an L-smooth convex loss."""
    trace, w, start = Trace("gradient descent, step 1/L"), w0.copy(), time.perf_counter()
    for k in range(steps + 1):
        value, g = value_and_grad(loss, w)
        trace.record(value, k + 1, start)
        w = w - lr * g
    return trace


def gradient_descent_line_search(loss: Loss, w0: np.ndarray, steps: int) -> Trace:
    """Steepest descent with Armijo backtracking, starting each search at twice the last step."""
    trace, w, start = Trace("gradient descent, line search"), w0.copy(), time.perf_counter()
    t = 1.0
    for k in range(steps + 1):
        value, g = value_and_grad(loss, w)
        trace.record(value, k + 1, start)
        t, _ = backtracking(loss, w, -g, value, -float(g @ g), t=2 * t)
        w = w - t * g
    return trace


def newton(loss: Loss, w0: np.ndarray, iterations: int, tol: float) -> tuple[Trace, np.ndarray]:
    """Damped Newton with the exact Hessian; returns the trace and the final iterate."""
    trace, w, start, passes = Trace("Newton, exact Hessian"), w0.copy(), time.perf_counter(), 0
    hessian = func.hessian(loss)
    for _ in range(iterations):
        value, g = value_and_grad(loss, w)
        passes += 1
        trace.record(value, passes, start)
        if np.linalg.norm(g) < tol:
            break
        with tg.no_grad():
            h = hessian(w).data  # gradient with a graph, then one backward pass per parameter
        passes += 1 + w.size
        step = -np.linalg.solve(h, g)
        t, _ = backtracking(loss, w, step, value, float(g @ step))
        w = w + t * step
    return trace, w


def newton_cg(loss: Loss, w0: np.ndarray, iterations: int, tol: float) -> Trace:
    """Truncated Newton: CG on ``H d = -g`` with Hessian-vector products, no Hessian."""
    trace, w, start, passes = (
        Trace("Newton-CG, Hessian-vector products"),
        w0.copy(),
        time.perf_counter(),
        0,
    )
    for _ in range(iterations):
        wt = tg.Tensor(w, requires_grad=True)
        value = loss(wt)
        (grad_t,) = autograd.grad(value, wt, create_graph=True)  # the gradient *and* its graph
        assert grad_t is not None
        passes += 1
        g = grad_t.data
        trace.record(value.item(), passes, start)
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            break
        d, r = np.zeros_like(g), -g.copy()
        p, rs = r.copy(), float(r @ r)
        forcing = min(0.5, np.sqrt(gnorm)) * gnorm  # superlinear convergence (Nocedal & Wright)
        for _ in range(w.size):
            # H p: one more backward pass through the retained graph of the gradient.
            (hp_t,) = autograd.grad(grad_t, wt, grad_outputs=p, retain_graph=True)
            assert hp_t is not None
            hp = hp_t.data
            passes += 1
            alpha = rs / float(p @ hp)
            d += alpha * p
            r -= alpha * hp
            rs_new = float(r @ r)
            if np.sqrt(rs_new) < forcing:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        t, _ = backtracking(loss, w, d, value.item(), float(g @ d))
        w = w + t * d
    return trace


def first_below(trace: Trace, optimum: float, threshold: float) -> int | None:
    for i, v in enumerate(trace.values):
        if v - optimum < threshold:
            return i
    return None


def save_plot(traces: list[Trace], optimum: float, kappa: float, path: Path) -> bool:
    plt = import_pyplot()
    if plt is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=110, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    floor = 1e-16  # float64 cannot resolve smaller gaps at a loss of about 0.2
    short = ("Newton", "Newton-CG", "GD, line search", "GD, step 1/L")
    for trace, color, name in zip(traces, SERIES, short, strict=False):
        gap = np.maximum(np.array(trace.values) - optimum, floor)
        iters = np.arange(1, len(gap) + 1)
        seconds = np.maximum(np.array(trace.seconds), 1e-4)
        for ax, x in ((axes[0], iters), (axes[1], seconds)):
            ax.plot(x, gap, color=color, linewidth=2, label=trace.name, solid_capstyle="round")
            ax.plot(x[-1], gap[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=2)  # fmt: skip
        # Direct labels on the iteration panel: Newton curves where they plunge, GD at the end.
        if name.startswith("GD"):
            below = name == "GD, line search"  # keep the two end labels off each other's curve
            axes[0].annotate(f"{name}\n{gap[-1]:.0e}", (iters[-1], gap[-1]),
                             xytext=(-8, -8 if below else 8), textcoords="offset points",
                             ha="right", va="top" if below else "bottom", fontsize=8,
                             color=INK_SECONDARY)  # fmt: skip
        else:
            k = int(np.argmax(gap < 1e-7))
            dx = -6 if name == "Newton" else 6
            axes[0].annotate(name, (iters[k], gap[k]), xytext=(dx, 0), textcoords="offset points",
                             ha="right" if dx < 0 else "left", va="center", fontsize=8,
                             color=INK_SECONDARY)  # fmt: skip
    for ax, xlabel in ((axes[0], "iteration"), (axes[1], "wall-clock seconds")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylim(floor / 3, 3)
        style_axes(ax)
    axes[0].set_ylabel("loss - optimal loss", fontsize=9)
    fig.suptitle(
        f"L2-regularised logistic regression, 11 parameters, Hessian condition number {kappa:,.0f}",
        color=INK, fontsize=10, x=0.01, ha="left",
    )  # fmt: skip
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    x, y = make_problem(args.samples, args.features, args.seed)
    loss = make_loss(x, y, args.lam)
    w0 = np.zeros(x.shape[1])
    # Global bound on the curvature of the logistic loss (sigmoid' <= 1/4).
    lipschitz = np.linalg.eigvalsh(x.T @ x).max() / (4 * len(x)) + args.lam

    newton_trace, w_star = newton(loss, w0, iterations=50, tol=1e-10)
    traces = [
        newton_trace,
        newton_cg(loss, w0, iterations=100, tol=1e-10),
        gradient_descent_line_search(loss, w0, steps=args.gd_steps),
        gradient_descent(loss, w0, lr=1.0 / lipschitz, steps=args.gd_steps),
    ]
    optimum = min(min(t.values) for t in traces)
    with tg.no_grad():
        eigs = np.linalg.eigvalsh(func.hessian(loss)(w_star).data)
    kappa = float(eigs.max() / eigs.min())
    print(f"{len(x)} samples, {x.shape[1]} parameters, lam = {args.lam:g}")
    print(f"Hessian at the optimum: condition number {kappa:,.0f}")
    print(f"{'method':36s} {'iterations':>10s} {'backward passes':>16s} {'seconds':>8s}")
    print(f"{'(to reach loss - optimum < 1e-10)':36s}")
    for t in traces:
        hit = first_below(t, optimum, 1e-10)
        if hit is None:
            gap, steps, secs = t.values[-1] - optimum, len(t.values) - 1, t.seconds[-1]
            print(f"{t.name:36s}   gap {gap:.1e} after {steps} steps ({secs:.2f} s)")
        else:
            print(f"{t.name:36s} {hit:>10d} {t.backward_passes[hit]:>16d} {t.seconds[hit]:>8.3f}")
    if not args.no_plot and save_plot(traces, optimum, kappa, args.plot):
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
