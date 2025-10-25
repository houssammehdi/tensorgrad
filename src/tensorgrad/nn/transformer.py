"""Pre-LayerNorm transformer block and a small GPT-style language model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tensorgrad import ops
from tensorgrad._random import get_rng
from tensorgrad._types import Array
from tensorgrad.autograd import no_grad
from tensorgrad.nn import init
from tensorgrad.nn.attention import MultiHeadAttention
from tensorgrad.nn.layers import GELU, Dropout, Embedding, LayerNorm, Linear, Sequential
from tensorgrad.nn.module import Module
from tensorgrad.tensor import Tensor

__all__ = ["GPT", "GPTConfig", "TransformerBlock"]


class TransformerBlock(Module):
    """``x + attn(ln1(x))`` followed by ``x + mlp(ln2(x))`` (pre-LN, as in GPT-2).

    Normalising *inside* the residual branch keeps the residual stream an identity path,
    which trains stably without learning-rate warm-up tricks at small scale.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: int = 4
    ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, causal=True)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = Sequential(
            Linear(embed_dim, mlp_ratio * embed_dim),
            GELU(),
            Linear(mlp_ratio * embed_dim, embed_dim),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


@dataclass(frozen=True)
class GPTConfig:
    """Hyper-parameters of :class:`GPT`."""

    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    tie_weights: bool = True


class GPT(Module):
    """Decoder-only transformer language model.

    Token + learned position embeddings, ``n_layer`` pre-LN :class:`TransformerBlock` s, a
    final LayerNorm and a linear head producing next-token logits. With ``tie_weights`` the
    head reuses the token-embedding matrix (one :class:`~tensorgrad.nn.Parameter` used in two
    places; the autograd engine sums both gradient contributions).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = Embedding(config.block_size, config.n_embd)
        self.drop = Dropout(config.dropout)
        self.blocks = Sequential(
            *(
                TransformerBlock(config.n_embd, config.n_head, config.dropout)
                for _ in range(config.n_layer)
            )
        )
        self.ln_f = LayerNorm(config.n_embd)
        self.head = Linear(config.n_embd, config.vocab_size, bias=False)
        self._init_weights()
        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

    def _init_weights(self) -> None:
        # GPT-2 initialisation: N(0, 0.02) everywhere, residual output projections scaled by
        # 1/sqrt(2 * n_layer) so the residual stream's variance does not grow with depth.
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, module in self.named_modules():
            if isinstance(module, Linear):
                is_residual = name.endswith(("attn.proj", "mlp.2"))
                init.normal_(module.weight, 0.0, residual_std if is_residual else 0.02)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, Embedding):
                init.normal_(module.weight, 0.0, 0.02)

    def forward(self, idx: Tensor | Array) -> Tensor:
        """Map token ids ``(B, T)`` to next-token logits ``(B, T, vocab_size)``."""
        tokens = np.asarray(idx.data if isinstance(idx, Tensor) else idx)
        if tokens.ndim != 2:
            raise ValueError(f"expected (B, T) token ids, got shape {tokens.shape}")
        t = tokens.shape[1]
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")
        x = self.tok_emb(tokens) + self.pos_emb(np.arange(t))
        x = self.blocks(self.drop(x))
        return self.head(self.ln_f(x))

    def generate(
        self,
        idx: Array,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> Array:
        """Autoregressively extend ``idx`` (``(B, T)`` ids) by ``max_new_tokens`` samples.

        Runs in evaluation mode under :class:`~tensorgrad.no_grad`; the context is cropped to
        the last ``block_size`` tokens. ``top_k`` restricts sampling to the k likeliest tokens;
        ``rng`` defaults to the global generator (see :func:`tensorgrad.manual_seed`).
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        rng = rng or get_rng()
        was_training = self.training
        self.eval()
        out = np.asarray(idx, dtype=np.int64)
        try:
            with no_grad():
                for _ in range(max_new_tokens):
                    context = out[:, -self.config.block_size :]
                    logits = self(context).data[:, -1, :].astype(np.float64) / temperature
                    if top_k is not None:
                        kth = np.sort(logits, axis=-1)[:, -min(top_k, logits.shape[-1])]
                        logits = np.where(logits < kth[:, None], -np.inf, logits)
                    probs = ops.softmax(Tensor._wrap(logits), axis=-1).data
                    nxt = np.array([rng.choice(probs.shape[-1], p=row) for row in probs])
                    out = np.concatenate([out, nxt[:, None]], axis=1)
        finally:
            self.train(was_training)
        return out
