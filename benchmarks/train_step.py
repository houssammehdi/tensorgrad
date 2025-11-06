"""Median wall-clock time of one training step (forward, backward, optimiser update).

    OPENBLAS_NUM_THREADS=1 python benchmarks/train_step.py

Three workloads, each built exactly like the corresponding example:

* ``spiral``: the ``2 -> 64 -> 64 -> 3`` ReLU MLP of ``examples/spiral_mlp.py``, full batch of
  600 points, Adam.
* ``cnn``: the two-conv-layer network of ``examples/shapes_cnn.py`` on a batch of 64 16x16
  images, Adam.
* ``gpt``: a 3-layer, 4-head, width-96 GPT with a 64-token context and dropout 0.1 (the
  round-1 ``examples/char_gpt.py`` model) on a batch of 32 sequences, AdamW with gradient
  clipping.

Each workload runs ``--warmup`` untimed steps, then ``--repeats`` timed ones; the median is
reported. Timings on a shared machine are indicative only: check ``/proc/loadavg`` first.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable

import numpy as np

import tensorgrad as tg
from tensorgrad import nn, optim
from tensorgrad.datasets import make_shapes, make_spiral

Step = Callable[[], None]


def spiral_step() -> Step:
    x, y = make_spiral(200, 3, noise=0.2, seed=0)
    model = nn.MLP([2, 64, 64, 3])
    opt = optim.Adam(model.parameters(), lr=0.01)
    xt = tg.Tensor(x)

    def step() -> None:
        loss = tg.cross_entropy(model(xt), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return step


def cnn_step() -> Step:
    images, labels = make_shapes(64, seed=0)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 4),
    )
    opt = optim.Adam(model.parameters(), lr=3e-3)
    xb = tg.Tensor(images)

    def step() -> None:
        loss = tg.cross_entropy(model(xb), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return step


def gpt_step() -> Step:
    config = nn.GPTConfig(vocab_size=65, block_size=64, n_layer=3, n_head=4, n_embd=96, dropout=0.1)
    model = nn.GPT(config)
    params = list(model.parameters())
    opt = optim.AdamW(params, lr=2e-3, betas=(0.9, 0.99), weight_decay=0.1)
    rng = np.random.default_rng(0)
    data = rng.integers(0, config.vocab_size, size=100_000)

    def step() -> None:
        starts = rng.integers(0, len(data) - config.block_size - 1, size=32)
        idx = starts[:, None] + np.arange(config.block_size)
        loss = tg.cross_entropy(model(data[idx]), data[idx + 1])
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(params, 1.0)
        opt.step()

    return step


WORKLOADS: dict[str, tuple[Callable[[], Step], int, int]] = {
    # name: (factory, default warm-up steps, default timed steps)
    "spiral": (spiral_step, 20, 300),
    "cnn": (cnn_step, 5, 60),
    "gpt": (gpt_step, 3, 25),
}


def time_steps(step: Step, warmup: int, repeats: int) -> list[float]:
    """Run ``warmup`` untimed and ``repeats`` timed steps; return the timings in seconds."""
    for _ in range(warmup):
        step()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        timings.append(time.perf_counter() - start)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS), choices=WORKLOADS)
    parser.add_argument("--repeats", type=int, default=None, help="override timed steps")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args()

    results = {}
    for name in args.workloads:
        factory, warmup, repeats = WORKLOADS[name]
        tg.manual_seed(0)
        timings = time_steps(factory(), warmup, args.repeats or repeats)
        results[name] = {
            "median_ms": 1000 * statistics.median(timings),
            "min_ms": 1000 * min(timings),
            "steps": len(timings),
        }
    meta = {
        "tensorgrad": tg.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "loadavg_1min": os.getloadavg()[0],
    }
    if args.json:
        print(json.dumps({"meta": meta, "results": results}))
        return
    print(" ".join(f"{k}={v}" for k, v in meta.items()))
    for name, r in results.items():
        print(
            f"{name:>7}: median {r['median_ms']:8.2f} ms  min {r['min_ms']:8.2f} ms  "
            f"({r['steps']} steps)"
        )


if __name__ == "__main__":
    main()
