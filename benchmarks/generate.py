"""Tokens per second of GPT sampling with and without the KV cache.

    OPENBLAS_NUM_THREADS=1 python benchmarks/generate.py

Uses an untrained model with the shape of ``examples/char_gpt.py``'s default (the weights do
not affect the cost). Two lengths are timed: filling the context window from a one-token
prompt, where the cache turns every step into a one-position update, and running past the
window, where the shifted window has to be re-encoded with or without the cache (absolute
position embeddings). Both settings produce identical tokens.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import tensorgrad as tg
from tensorgrad import nn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    args = parser.parse_args()

    tg.manual_seed(0)
    config = nn.GPTConfig(
        vocab_size=65,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = nn.GPT(config)
    prompt = np.zeros((1, 1), dtype=np.int64)
    for new_tokens in (args.block_size - 1, 3 * args.block_size):
        results = {}
        for use_cache in (True, False):
            start = time.perf_counter()
            out = model.generate(
                prompt, new_tokens, rng=np.random.default_rng(0), use_cache=use_cache
            )
            results[use_cache] = (time.perf_counter() - start, out)
        assert np.array_equal(results[True][1], results[False][1])  # same tokens either way
        cached, plain = results[True][0], results[False][0]
        print(
            f"{new_tokens:5d} new tokens: with cache {new_tokens / cached:7.1f} tok/s, "
            f"without {new_tokens / plain:7.1f} tok/s ({plain / cached:.1f}x)"
        )


if __name__ == "__main__":
    main()
