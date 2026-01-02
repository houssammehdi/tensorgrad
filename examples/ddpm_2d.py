"""A denoising diffusion probabilistic model (DDPM) that learns a 2-D swiss roll.

    python examples/ddpm_2d.py --seed 0

The forward process gradually adds Gaussian noise to data points over ``T = 1000`` steps
with the linear variance schedule of Ho et al. (2020), until they are indistinguishable from
``N(0, I)``. A small MLP learns to predict the added noise from the noisy point and the step
(the simplified training objective, Algorithm 1 of the paper). Sampling runs the learned
reverse process from pure noise (Algorithm 2).

The script reports how close the samples are to the data (the energy distance to fresh data,
next to the values for pure noise and for a second draw of real data, and the distance of
each sample to the noiseless spiral), and with matplotlib installed it draws snapshots of the
reverse process to docs/ddpm_reverse_process.png.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import tensorgrad as tg
from _style import GRID, INK, INK_SECONDARY, SURFACE, import_pyplot
from tensorgrad import nn, optim

ROOT = Path(__file__).resolve().parent
DEFAULT_PLOT = ROOT.parent / "docs" / "ddpm_reverse_process.png"
DATA_NOISE = 0.05  # standard deviation of the noise around the spiral
SEQUENTIAL_LIGHT, SEQUENTIAL_DARK = "#9ec2ef", "#0d2f63"  # the blue ramp's ends


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=8000, help="optimisation steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=1000, help="T")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--samples", type=int, default=2000, help="points to generate")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def spiral_curve(u: np.ndarray) -> np.ndarray:
    """Points of the noiseless swiss roll at curve parameters ``u`` in ``[0, 1]``."""
    angle = 1.5 * math.pi * (1 + 2 * u)
    return np.stack([angle * np.cos(angle), angle * np.sin(angle)], axis=-1) / 7.0


def sample_data(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` points of a 2-D swiss roll (roughly unit scale) with Gaussian noise."""
    points = spiral_curve(rng.uniform(0.0, 1.0, size=n))
    return points + DATA_NOISE * rng.standard_normal((n, 2))


class Schedule:
    """The linear variance schedule ``beta_t`` and the products derived from it."""

    def __init__(self, steps: int, beta_1: float = 1e-4, beta_t: float = 0.02) -> None:
        # Ho et al. use beta_1 = 1e-4 and beta_T = 0.02 for T = 1000; other T keep the sum.
        scale = 1000 / steps
        self.steps = steps
        self.betas = np.linspace(beta_1 * scale, beta_t * scale, steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)


def time_features(t: np.ndarray, dim: int, steps: int) -> np.ndarray:
    """Sinusoidal features of the (integer) diffusion step, as in the Transformer."""
    half = dim // 2
    freqs = np.exp(-math.log(steps) * np.arange(half) / half)
    angles = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


class Denoiser(nn.Module):
    """``eps_theta(x_t, t)``: an MLP on the noisy point and features of the step."""

    def __init__(self, hidden: int, steps: int, time_dim: int = 32) -> None:
        super().__init__()
        self.steps = steps
        self.time_dim = time_dim
        self.net = nn.MLP([2 + time_dim, hidden, hidden, hidden, 2], activation="gelu")

    def forward(self, x: tg.Tensor, t: np.ndarray) -> tg.Tensor:
        features = time_features(t, self.time_dim, self.steps).astype(x.dtype)
        return self.net(tg.concat([x, tg.Tensor(features)], axis=1))


def train(model: Denoiser, schedule: Schedule, args: argparse.Namespace) -> list[float]:
    rng = np.random.default_rng(args.seed)
    params = list(model.parameters())
    opt = optim.Adam(params, lr=args.lr)
    sched = optim.CosineWarmupLR(opt, min(100, args.steps), args.steps, min_lr_ratio=0.05)
    losses = []
    for _ in range(args.steps):
        x0 = sample_data(rng, args.batch_size)
        t = rng.integers(0, schedule.steps, size=args.batch_size)
        eps = rng.standard_normal(x0.shape)
        ab = schedule.alpha_bars[t][:, None]
        xt = np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps
        loss = tg.mse_loss(model(tg.Tensor(xt.astype(np.float32)), t), eps.astype(np.float32))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


def reverse_process(
    model: Denoiser, schedule: Schedule, n: int, rng: np.random.Generator, keep: set[int]
) -> dict[int, np.ndarray]:
    """Algorithm 2 with ``sigma_t^2 = beta_t``: the points ``x_t`` for each ``t`` in ``keep``
    (``x_T`` is always kept)."""
    x = rng.standard_normal((n, 2))
    snapshots = {schedule.steps: x.copy()}
    with tg.no_grad():
        for t in range(schedule.steps - 1, -1, -1):  # index t holds step t + 1 of the paper
            eps = model(tg.Tensor(x.astype(np.float32)), np.full(n, t)).data.astype(np.float64)
            beta, alpha, ab = schedule.betas[t], schedule.alphas[t], schedule.alpha_bars[t]
            x = (x - beta / math.sqrt(1.0 - ab) * eps) / math.sqrt(alpha)
            if t > 0:
                x = x + math.sqrt(beta) * rng.standard_normal(x.shape)
            if t in keep:
                snapshots[t] = x.copy()
    return snapshots


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``2 E|A - B| - E|A - A'| - E|B - B'|``: zero iff the distributions are equal."""

    def mean_dist(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1)).mean())

    return 2 * mean_dist(a, b) - mean_dist(a, a) - mean_dist(b, b)


def nearest_on_spiral(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to the noiseless curve, and the curve parameter ``u`` of
    the nearest curve point (the curve is sampled densely)."""
    u = np.linspace(0.0, 1.0, 2000)
    dist = np.sqrt(((points[:, None, :] - spiral_curve(u)[None, :, :]) ** 2).sum(-1))
    return dist.min(axis=1), u[dist.argmin(axis=1)]


def save_plot(snapshots: dict[int, np.ndarray], data: np.ndarray, path: Path) -> bool:
    plt = import_pyplot()
    if plt is None:
        return False
    from matplotlib.colors import LinearSegmentedColormap

    # One hue, light to dark: where along the spiral each point ends up.
    ramp = LinearSegmentedColormap.from_list("ends", [SEQUENTIAL_LIGHT, SEQUENTIAL_DARK])
    _, ends = nearest_on_spiral(snapshots[0])
    order = np.argsort(ends)  # draw dark (outer) points last
    times = sorted(snapshots, reverse=True)
    fig, axes = plt.subplots(1, len(times), figsize=(2.05 * len(times), 2.45), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    lim = 2.45
    for ax, t in zip(axes, times, strict=True):
        ax.scatter(data[:, 0], data[:, 1], s=3, color=GRID, linewidths=0)
        pts = snapshots[t][order]
        ax.scatter(pts[:, 0], pts[:, 1], s=2.5, c=ends[order], cmap=ramp, linewidths=0)
        label = {times[0]: f"t = {t}  (noise)", 0: "t = 0  (samples)"}.get(t, f"t = {t}")
        ax.set_title(label, color=INK, fontsize=9, loc="left")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.text(
        0.005,
        -0.03,
        "The learned reverse process, from Gaussian noise (left) to samples (right). Grey: "
        "the training distribution. Each point is shaded by where it ends on the spiral "
        "(light: inner end, dark: outer end).",
        fontsize=8,
        color=INK_SECONDARY,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    tg.manual_seed(args.seed)
    schedule = Schedule(args.diffusion_steps)
    print(
        f"T = {schedule.steps}, beta {schedule.betas[0]:.1e} .. {schedule.betas[-1]:.1e}; "
        f"alpha_bar_T = {schedule.alpha_bars[-1]:.1e} (x_T is almost pure noise)"
    )
    model = Denoiser(args.hidden, schedule.steps)
    print(f"denoiser: {model.num_parameters():,} parameters")
    start = time.perf_counter()
    losses = train(model, schedule, args)
    seconds = time.perf_counter() - start
    print(
        f"trained {args.steps} steps in {seconds:.1f}s; noise-prediction MSE "
        f"{np.mean(losses[:100]):.3f} (first 100) -> {np.mean(losses[-100:]):.3f} (last 100)"
    )

    rng = np.random.default_rng(args.seed + 1)
    t_max = schedule.steps
    keep = {int(f * t_max) for f in (0.6, 0.3, 0.1, 0.03)} | {0}
    begin = time.perf_counter()
    snapshots = reverse_process(model, schedule, args.samples, rng, keep)
    print(f"sampled {args.samples} points in {time.perf_counter() - begin:.1f}s")
    samples = snapshots[0]
    reference = sample_data(np.random.default_rng(args.seed + 2), args.samples)
    fresh = sample_data(np.random.default_rng(args.seed + 3), args.samples)
    noise = np.random.default_rng(args.seed + 4).standard_normal((args.samples, 2))
    print(
        "energy distance to fresh data: "
        f"samples {energy_distance(samples, reference):.4f}, "
        f"a second draw of real data {energy_distance(fresh, reference):.4f}, "
        f"pure noise {energy_distance(noise, reference):.4f}"
    )
    dist, _ = nearest_on_spiral(samples)
    data_dist, _ = nearest_on_spiral(fresh)
    print(
        f"distance to the noiseless spiral: median {np.median(dist):.3f} for samples, "
        f"{np.median(data_dist):.3f} for real data; within 0.15: {np.mean(dist < 0.15):.1%} "
        f"of samples, {np.mean(data_dist < 0.15):.1%} of real data"
    )
    if not args.no_plot and save_plot(snapshots, reference, args.plot):
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
