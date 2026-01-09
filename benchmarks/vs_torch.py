"""The training-step workloads of ``train_step.py`` in tensorgrad and in PyTorch (CPU).

    OPENBLAS_NUM_THREADS=1 python benchmarks/vs_torch.py --threads 1

Needs PyTorch. Both frameworks run float32 with the same number of threads: ``--threads``
sets ``torch.set_num_threads`` and must match ``OPENBLAS_NUM_THREADS`` (NumPy's BLAS reads
it at import, so it has to be set in the environment; the script refuses a mismatch). The
PyTorch models mirror the tensorgrad ones layer for layer: same shapes, initialisation
scheme, optimiser, dropout and (for the GPT) a fused causal attention kernel. Frameworks and
workloads are interleaved over ``--rounds`` rounds so that a change in machine load affects
both, and the median over all timed steps is reported. ``--retain-freed-memory`` applies
:func:`tensorgrad.utils.retain_freed_memory`, which changes glibc's settings for the whole
process and so for both frameworks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_step  # the tensorgrad side of each workload

import tensorgrad as tg
from tensorgrad.datasets import make_shapes, make_spiral
from tensorgrad.utils import retain_freed_memory

Step = Callable[[], None]


def torch_workloads() -> dict[str, Callable[[], Step]]:
    """PyTorch versions of the three workloads (imported lazily: torch is optional)."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    def spiral() -> Step:
        x, y = make_spiral(200, 3, noise=0.2, seed=0)
        model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                              nn.Linear(64, 3))  # fmt: skip
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        xt, yt = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)

        def step() -> None:
            loss = F.cross_entropy(model(xt), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()

        return step

    def cnn() -> Step:
        images, labels = make_shapes(64, seed=0)
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 4),
        )  # fmt: skip
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        xb, yb = torch.from_numpy(images), torch.from_numpy(labels)

        def step() -> None:
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        return step

    class Block(nn.Module):  # type: ignore[misc]
        def __init__(self, dim: int, heads: int, dropout: float) -> None:
            super().__init__()
            self.heads, self.dropout = heads, dropout
            self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
            self.fc, self.out = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, d = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, dropout_p=self.dropout if self.training else 0.0,
            )  # fmt: skip
            x = x + self.drop(self.proj(y.transpose(1, 2).reshape(b, t, d)))
            return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))

    class GPT(nn.Module):  # type: ignore[misc]
        def __init__(self, vocab: int, block: int, layers: int, dim: int, heads: int) -> None:
            super().__init__()
            self.tok, self.pos = nn.Embedding(vocab, dim), nn.Embedding(block, dim)
            self.drop = nn.Dropout(0.1)
            self.blocks = nn.ModuleList(Block(dim, heads, 0.1) for _ in range(layers))
            self.ln_f = nn.LayerNorm(dim)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            x = self.drop(self.tok(idx) + self.pos(torch.arange(idx.shape[1])))
            for block in self.blocks:
                x = block(x)
            return self.ln_f(x) @ self.tok.weight.T  # tied output head

    def gpt() -> Step:
        model = GPT(65, 64, 3, 96, 4)
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=2e-3, betas=(0.9, 0.99), weight_decay=0.1)
        rng = np.random.default_rng(0)
        data = torch.from_numpy(rng.integers(0, 65, size=100_000))

        def step() -> None:
            starts = torch.from_numpy(rng.integers(0, len(data) - 65, size=32))
            idx = starts[:, None] + torch.arange(64)
            logits = model(data[idx])
            loss = F.cross_entropy(logits.reshape(-1, 65), data[idx + 1].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        return step

    return {"spiral": spiral, "cnn": cnn, "gpt": gpt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3, help="interleaved repetitions")
    parser.add_argument(
        "--workloads", nargs="+", default=["spiral", "cnn", "gpt"], choices=["spiral", "cnn", "gpt"]
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--retain-freed-memory",
        action="store_true",
        help="tune glibc malloc first (process-wide: applies to both frameworks)",
    )
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != str(args.threads):
        raise SystemExit(f"set OPENBLAS_NUM_THREADS={args.threads} to match --threads")
    retained = args.retain_freed_memory and retain_freed_memory()
    import torch

    torch.set_num_threads(args.threads)
    torch_factories = torch_workloads()
    timings: dict[str, dict[str, list[float]]] = {
        w: {"tensorgrad": [], "torch": []} for w in args.workloads
    }
    for _ in range(args.rounds):
        for name in args.workloads:
            factory, warmup, repeats = train_step.WORKLOADS[name]
            tg.manual_seed(0)
            timings[name]["tensorgrad"] += train_step.time_steps(factory(), warmup, repeats)
            torch.manual_seed(0)
            torch_step = torch_factories[name]()
            timings[name]["torch"] += train_step.time_steps(torch_step, warmup, repeats)
    results = {
        name: {fw: 1000 * statistics.median(ts) for fw, ts in by_fw.items()}
        for name, by_fw in timings.items()
    }
    meta = {
        "threads": args.threads,
        "retain_freed_memory": retained,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "loadavg_1min_at_end": os.getloadavg()[0],
    }
    if args.json:
        print(json.dumps({"meta": meta, "results": results}))
        return
    print(" ".join(f"{k}={v}" for k, v in meta.items()))
    for name, r in results.items():
        print(
            f"{name:>7}: tensorgrad {r['tensorgrad']:8.2f} ms  torch {r['torch']:8.2f} ms  "
            f"ratio {r['tensorgrad'] / r['torch']:5.1f}x"
        )


if __name__ == "__main__":
    main()
