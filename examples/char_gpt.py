"""Train a small character-level GPT on bundled Shakespeare and sample from it.

    python examples/char_gpt.py --steps 1200 --seed 0

The corpus (examples/data/shakespeare.txt, ~13 KB of public-domain speeches and sonnets) is
split 90/10 into train/validation text. Training uses AdamW with decoupled weight decay on
the weight matrices only, a cosine schedule with warm-up and global-norm gradient clipping.
At the end the script samples text, optionally saves a checkpoint, and (if matplotlib is
installed) plots the loss curves to docs/gpt_loss.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import tensorgrad as tg
from tensorgrad import nn, optim
from tensorgrad.datasets import CharTokenizer
from tensorgrad.utils import load, save

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "shakespeare.txt"
DEFAULT_PLOT = ROOT.parent / "docs" / "gpt_loss.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=1200, help="optimisation steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--block-size", type=int, default=64, help="context length")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=None, help="stop early after this")
    parser.add_argument("--sample-chars", type=int, default=600)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="save weights here")
    parser.add_argument("--resume", type=Path, default=None, help="load weights before training")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="loss-curve PNG path")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random windows of ``block_size`` tokens and the same windows shifted by one."""
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    offsets = np.arange(block_size)
    x = data[starts[:, None] + offsets]
    return x, data[starts[:, None] + offsets + 1]


def estimate_loss(
    model: nn.GPT, data: np.ndarray, args: argparse.Namespace, rng: np.random.Generator
) -> float:
    model.eval()
    losses = []
    with tg.no_grad():
        for _ in range(args.eval_batches):
            x, y = get_batch(data, args.block_size, args.batch_size, rng)
            losses.append(tg.cross_entropy(model(x), y).item())
    model.train()
    return float(np.mean(losses))


def save_plot(history: list[tuple[int, float, float]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return False
    steps, train, val = (np.array(col) for col in zip(*history, strict=True))
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for values, color, label in ((train, "#2a78d6", "train"), (val, "#eb6834", "validation")):
        ax.plot(steps, values, color=color, linewidth=2, label=label)
        ax.annotate(
            f"{label} {values[-1]:.2f}",
            (steps[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#52514e",
        )
    ax.set_xlabel("step", color="#52514e", fontsize=9)
    ax.set_ylabel("cross-entropy (nats / char)", color="#52514e", fontsize=9)
    ax.set_title("Character-level GPT on Shakespeare", color="#0b0b0b", fontsize=10)
    ax.grid(color="#e8e7e3", linewidth=0.8)
    ax.tick_params(colors="#52514e", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d9d8d4")
    ax.legend(frameon=False, fontsize=8, labelcolor="#52514e")
    ax.set_xlim(0, steps[-1] * 1.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    tg.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    text = args.corpus.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = tok.encode(text)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    config = nn.GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = nn.GPT(config)
    if args.resume is not None:
        model.load_state_dict(load(args.resume))
    # Decay weight matrices (and the tied embedding) but not biases or LayerNorm gains.
    params = list(model.parameters())
    opt = optim.AdamW(
        [
            {"params": [p for p in params if p.ndim >= 2]},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    sched = optim.CosineWarmupLR(opt, args.warmup, max(args.steps, args.warmup), min_lr_ratio=0.1)
    print(
        f"corpus {len(text)} chars, vocab {tok.vocab_size}, "
        f"train {len(train_data)} / val {len(val_data)} tokens"
    )
    print(f"GPT {config} -> {model.num_parameters():,} parameters")

    history: list[tuple[int, float, float]] = []
    start = time.perf_counter()
    step = 0
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, rng)
        loss = tg.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        elapsed = time.perf_counter() - start
        out_of_time = args.max_minutes is not None and elapsed > 60 * args.max_minutes
        if step % args.eval_interval == 0 or step == args.steps or out_of_time:
            train_loss = estimate_loss(model, train_data, args, rng)
            val_loss = estimate_loss(model, val_data, args, rng)
            history.append((step, train_loss, val_loss))
            print(
                f"step {step:5d}  train {train_loss:.3f}  val {val_loss:.3f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  {elapsed:6.1f}s"
            )
        if out_of_time:
            print(f"time budget of {args.max_minutes} min reached")
            break
    total = time.perf_counter() - start
    print(f"trained {step} steps in {total:.1f}s ({1000 * total / max(step, 1):.0f} ms/step)")

    if args.checkpoint is not None:
        save(model, args.checkpoint)
        print(f"saved checkpoint to {args.checkpoint}")
    if not args.no_plot and history and save_plot(history, args.plot):
        print(f"saved {args.plot}")

    prompt = tok.encode(args.prompt)[None, :]
    sample = model.generate(
        prompt,
        args.sample_chars,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=np.random.default_rng(args.seed),
    )
    print("\n--- sample ---")
    print(tok.decode(sample[0]))


if __name__ == "__main__":
    main()
