"""Train a character-level GPT on tiny-shakespeare and sample from it.

    python examples/char_gpt.py --seed 0

The corpus is Andrej Karpathy's tiny-shakespeare (1,115,394 characters of Shakespeare's
plays), downloaded once, verified against a pinned SHA-256 and cached in
``~/.cache/tensorgrad`` (see ``tensorgrad.datasets.tiny_shakespeare``). Without network
access the script falls back to the 13 KB excerpt bundled in ``examples/data`` and says so.
The text is split 90/10 into training and validation characters.

Training uses AdamW (decoupled weight decay on the weight matrices only), a cosine schedule
with linear warm-up and global-norm gradient clipping. Train and validation losses are
measured on fixed random windows so the curves are not dominated by sampling noise. At the
end the script samples text with the KV cache, optionally saves a checkpoint, and (if
matplotlib is installed) plots the loss curves to docs/gpt_loss.png.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from pathlib import Path

import numpy as np

import tensorgrad as tg
from _style import INK, INK_SECONDARY, SERIES, SURFACE, import_pyplot, style_axes
from tensorgrad import nn, optim
from tensorgrad.datasets import CharTokenizer, tiny_shakespeare
from tensorgrad.utils import load, retain_freed_memory, save

ROOT = Path(__file__).resolve().parent
BUNDLED_CORPUS = ROOT / "data" / "shakespeare.txt"
DEFAULT_PLOT = ROOT.parent / "docs" / "gpt_loss.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=4000, help="optimisation steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus", type=Path, default=None, help="text file (default: download)")
    parser.add_argument("--block-size", type=int, default=128, help="context length")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20, help="fixed windows x batch")
    parser.add_argument("--max-minutes", type=float, default=None, help="stop early after this")
    parser.add_argument("--sample-chars", type=int, default=600)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="save weights here")
    parser.add_argument("--resume", type=Path, default=None, help="load weights before training")
    parser.add_argument("--history", type=Path, default=None, help="write the curves as JSON")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="loss-curve PNG path")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_corpus(path: Path | None) -> tuple[str, str]:
    """Text and a description of where it came from."""
    if path is not None:
        return path.read_text(encoding="utf-8"), str(path)
    try:
        return tiny_shakespeare(), "tiny-shakespeare (SHA-256 verified)"
    except (urllib.error.URLError, OSError) as exc:
        print(f"could not download tiny-shakespeare ({exc}); using the bundled 13 KB excerpt")
        return BUNDLED_CORPUS.read_text(encoding="utf-8"), f"{BUNDLED_CORPUS} (offline fallback)"


def windows(data: np.ndarray, starts: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs ``data[s : s + T]`` and next-character targets for each start ``s``."""
    idx = starts[:, None] + np.arange(block_size)
    return data[idx], data[idx + 1]


def random_starts(
    rng: np.random.Generator, n_tokens: int, block_size: int, count: int
) -> np.ndarray:
    """``count`` uniform window starts; the last valid one is ``n_tokens - block_size - 1``
    (its final target is the last token)."""
    return rng.integers(0, n_tokens - block_size, size=count)


def estimate_loss(
    model: nn.GPT, data: np.ndarray, starts: np.ndarray, args: argparse.Namespace
) -> float:
    """Mean cross-entropy over the fixed windows at ``starts`` (dropout off, no graph)."""
    model.eval()
    total = 0.0
    with tg.no_grad():
        for batch in np.array_split(starts, max(len(starts) // args.batch_size, 1)):
            x, y = windows(data, batch, args.block_size)
            total += tg.cross_entropy(model(x), y).item() * len(batch)
    model.train()
    return total / len(starts)


def save_plot(history: list[dict[str, float]], path: Path) -> bool:
    plt = import_pyplot()
    if plt is None:
        return False
    steps = np.array([h["step"] for h in history])
    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for key, color, label in (("train", SERIES[0], "train"), ("val", SERIES[1], "validation")):
        values = np.array([h[key] for h in history])
        ax.plot(steps, values, color=color, linewidth=2, label=label)
        ax.annotate(
            f"{label} {values[-1]:.3f}",
            (steps[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("cross-entropy (nats / char)", fontsize=9)
    style_axes(ax)
    ax.set_title("Character-level GPT on tiny-shakespeare", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    ax.set_xlim(0, steps[-1] * 1.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    retain_freed_memory()  # keep freed activations mapped: saves ~20% of a step (see docs)
    tg.manual_seed(args.seed)
    text, source = load_corpus(args.corpus)
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
    # Separate generators: evaluation never changes which batches training sees.
    train_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)
    n_eval = args.eval_batches * args.batch_size
    eval_starts = {
        "train": random_starts(eval_rng, len(train_data), args.block_size, n_eval),
        "val": random_starts(eval_rng, len(val_data), args.block_size, n_eval),
    }
    print(f"corpus: {source}")
    print(
        f"{len(text):,} chars, vocab {tok.vocab_size}, "
        f"train {len(train_data):,} / val {len(val_data):,} tokens"
    )
    print(f"GPT {config} -> {model.num_parameters():,} parameters")

    history: list[dict[str, float]] = []
    start, cpu_start = time.perf_counter(), time.process_time()
    step = 0
    for step in range(1, args.steps + 1):
        starts = random_starts(train_rng, len(train_data), args.block_size, args.batch_size)
        x, y = windows(train_data, starts, args.block_size)
        loss = tg.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        elapsed = time.perf_counter() - start
        out_of_time = args.max_minutes is not None and elapsed > 60 * args.max_minutes
        if step % args.eval_interval == 0 or step == args.steps or out_of_time:
            record = {
                "step": step,
                "train": estimate_loss(model, train_data, eval_starts["train"], args),
                "val": estimate_loss(model, val_data, eval_starts["val"], args),
                "seconds": elapsed,
                "cpu_seconds": time.process_time() - cpu_start,
            }
            history.append(record)
            print(
                f"step {step:5d}  train {record['train']:.3f}  val {record['val']:.3f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  {elapsed:7.1f}s"
            )
        if out_of_time:
            print(f"time budget of {args.max_minutes} min reached")
            break
    total, cpu = time.perf_counter() - start, time.process_time() - cpu_start
    print(
        f"trained {step} steps in {total:.1f}s wall, {cpu:.1f}s CPU "
        f"({1000 * total / max(step, 1):.0f} ms/step wall, {1000 * cpu / max(step, 1):.0f} CPU)"
    )

    if args.checkpoint is not None:
        save(model, args.checkpoint)
        print(f"saved checkpoint to {args.checkpoint}")
    if args.history is not None and history:
        args.history.write_text(json.dumps(history, indent=1))
    if not args.no_plot and history and save_plot(history, args.plot):
        print(f"saved {args.plot}")

    prompt = tok.encode(args.prompt)[None, :]
    begin = time.perf_counter()
    sample = model.generate(
        prompt,
        args.sample_chars,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=np.random.default_rng(args.seed),
    )
    rate = args.sample_chars / (time.perf_counter() - begin)
    print(f"\n--- sample ({args.sample_chars} chars at {rate:.0f} chars/s, KV cache) ---")
    print(tok.decode(sample[0]))


if __name__ == "__main__":
    main()
