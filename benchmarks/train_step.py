"""Median wall-clock time of one training step (forward, backward, optimiser update).

    OPENBLAS_NUM_THREADS=1 python benchmarks/train_step.py

Three workloads (and a variant), built like the examples of version 0.1.0:

* ``spiral``: the ``2 -> 64 -> 64 -> 3`` ReLU MLP of ``examples/spiral_mlp.py``, full batch of
  600 points, Adam.
* ``cnn``: the two-conv-layer network of ``examples/shapes_cnn.py --arch plain`` on a batch
  of 64 16x16 images, Adam.
* ``gpt``: a 3-layer, 4-head, width-96 GPT with a 64-token context and dropout 0.1 (the
  round-1 ``examples/char_gpt.py`` model) on a batch of 32 sequences, AdamW with gradient
  clipping; ``gpt-checkpointed`` is the same with gradient checkpointing of every block.

Each workload runs ``--warmup`` untimed steps, then ``--repeats`` timed ones; the median is
reported. ``--retain-freed-memory`` applies :func:`tensorgrad.utils.retain_freed_memory`
first, ``--profile`` also prints where the time of a few more steps goes, per op
(:mod:`tensorgrad.profiler`), and ``--peak-memory`` reports the peak memory NumPy
allocates during one more step (traced with :mod:`tracemalloc`). Timings on a shared
machine are indicative only: check ``/proc/loadavg`` first.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable

import numpy as np

import tensorgrad as tg
from tensorgrad import nn, optim
from tensorgrad.datasets import make_shapes, make_spiral
from tensorgrad.utils import retain_freed_memory

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


def gpt_step(gradient_checkpointing: bool = False) -> Step:
    config = nn.GPTConfig(vocab_size=65, block_size=64, n_layer=3, n_head=4, n_embd=96, dropout=0.1)
    model = nn.GPT(config)
    model.gradient_checkpointing = gradient_checkpointing
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
    "gpt-checkpointed": (lambda: gpt_step(gradient_checkpointing=True), 3, 25),
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
    parser.add_argument(
        "--workloads", nargs="+", default=["spiral", "cnn", "gpt"], choices=WORKLOADS
    )
    parser.add_argument("--repeats", type=int, default=None, help="override timed steps")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    parser.add_argument(
        "--retain-freed-memory", action="store_true", help="tune glibc malloc first"
    )
    parser.add_argument("--profile", action="store_true", help="print a per-op profile")
    parser.add_argument("--peak-memory", action="store_true", help="trace one step's memory")
    args = parser.parse_args()
    retained = args.retain_freed_memory and retain_freed_memory()

    results = {}
    profiles = {}
    for name in args.workloads:
        factory, warmup, repeats = WORKLOADS[name]
        tg.manual_seed(0)
        step = factory()
        timings = time_steps(step, warmup, args.repeats or repeats)
        results[name] = {
            "median_ms": 1000 * statistics.median(timings),
            "min_ms": 1000 * min(timings),
            "steps": len(timings),
        }
        if args.profile:
            with tg.profiler.profile() as prof:
                for _ in range(5):
                    step()
            profiles[name] = prof.table(limit=12)
        if args.peak_memory:
            tracemalloc.start()
            step()
            results[name]["peak_mb"] = tracemalloc.get_traced_memory()[1] / 2**20
            tracemalloc.stop()
    meta = {
        "tensorgrad": tg.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "retain_freed_memory": retained,
        "loadavg_1min": os.getloadavg()[0],
    }
    if args.json:
        print(json.dumps({"meta": meta, "results": results}))
        return
    print(" ".join(f"{k}={v}" for k, v in meta.items()))
    for name, r in results.items():
        peak = f"  peak {r['peak_mb']:.1f} MB" if "peak_mb" in r else ""
        print(
            f"{name:>16}: median {r['median_ms']:8.2f} ms  min {r['min_ms']:8.2f} ms  "
            f"({r['steps']} steps){peak}"
        )
    for name, table in profiles.items():
        print(f"\n{name}: 5 steps under the profiler\n{table}")


if __name__ == "__main__":
    main()
