"""nn modules against PyTorch with identical weights: outputs, all gradients and HVPs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

import tensorgrad as tg  # noqa: E402
from parity import assert_close  # noqa: E402
from tensorgrad import nn  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(autouse=True)
def _float64_parameters() -> object:
    tg.set_default_dtype(np.float64)
    yield
    tg.set_default_dtype(np.float32)


@pytest.fixture(autouse=True)
def _torch_math_attention() -> object:
    # PyTorch's fused CPU attention kernel has no double backward; its reference "math"
    # implementation does, so the Hessian-vector products can be compared.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        yield


def copy_weights(pairs: Sequence[tuple[tg.Tensor, torch.Tensor]]) -> None:
    with torch.no_grad():
        for ours, theirs in pairs:
            assert tuple(ours.shape) == tuple(theirs.shape), (ours.shape, theirs.shape)
            theirs.copy_(torch.from_numpy(ours.data))


def check_module(
    ours: nn.Module,
    theirs: torch.nn.Module,
    pairs: Sequence[tuple[tg.Tensor, torch.Tensor]],
    x: np.ndarray,
    *,
    input_grad: bool = True,
    loss: tuple[Callable[[tg.Tensor], tg.Tensor], Callable[[torch.Tensor], torch.Tensor]]
    | None = None,
    hvp: bool = True,
    seed: int = 0,
) -> None:
    """Outputs, input/parameter gradients and a parameter Hessian-vector product agree."""
    rng = np.random.default_rng(seed)
    copy_weights(pairs)
    params_ours = [p for p, _ in pairs]
    params_theirs = [p for _, p in pairs]
    x_ours = tg.Tensor(x, requires_grad=input_grad) if x.dtype.kind == "f" else x
    x_theirs = (
        torch.tensor(x, requires_grad=input_grad) if x.dtype.kind == "f" else torch.from_numpy(x)
    )
    out_ours, out_theirs = ours(x_ours), theirs(x_theirs)
    assert_close(out_ours, out_theirs, "module output")
    if loss is None:
        v = rng.standard_normal(out_ours.shape)
        scalar_ours = (out_ours * v).sum()
        scalar_theirs = (out_theirs * torch.tensor(v)).sum()
    else:
        scalar_ours, scalar_theirs = loss[0](out_ours), loss[1](out_theirs)
        assert_close(scalar_ours, scalar_theirs, "loss")
    wrt_ours = params_ours + ([x_ours] if input_grad else [])
    wrt_theirs = params_theirs + ([x_theirs] if input_grad else [])
    g_ours = tg.autograd.grad(scalar_ours, wrt_ours, create_graph=hvp, allow_unused=True)
    g_theirs = torch.autograd.grad(scalar_theirs, wrt_theirs, create_graph=hvp, allow_unused=True)
    for i, (a, b) in enumerate(zip(g_ours, g_theirs, strict=True)):
        assert a is not None and b is not None
        assert_close(a, b, f"gradient {i}")
    if not hvp:
        return
    vs = [rng.standard_normal(p.shape) for p in params_ours]
    dot_ours = sum(((g * v).sum() for g, v in zip(g_ours, vs, strict=False)), tg.Tensor(0.0))
    dot_theirs = sum(
        ((g * torch.tensor(v)).sum() for g, v in zip(g_theirs, vs, strict=False)), torch.tensor(0.0)
    )
    # A module that is linear in its parameters (e.g. an embedding) has a zero Hessian: the
    # dot product then carries no graph on either side.
    none: tuple[None, ...] = (None,) * len(params_ours)
    h_ours = (
        tg.autograd.grad(dot_ours, params_ours, allow_unused=True)
        if dot_ours.requires_grad
        else none
    )
    h_theirs = (
        torch.autograd.grad(dot_theirs, params_theirs, allow_unused=True)
        if dot_theirs.requires_grad
        else none
    )
    for i, (a, b) in enumerate(zip(h_ours, h_theirs, strict=True)):
        a_arr = np.zeros(params_ours[i].shape) if a is None else a.data
        b_arr = np.zeros(params_ours[i].shape) if b is None else b.detach().numpy()
        assert_close(a_arr, b_arr, f"Hessian-vector product, parameter {i}")


def test_linear() -> None:
    ours, theirs = nn.Linear(4, 3), torch.nn.Linear(4, 3).double()
    x = np.random.default_rng(0).standard_normal((2, 5, 4))
    check_module(ours, theirs, [(ours.weight, theirs.weight), (ours.bias, theirs.bias)], x)


def test_mlp() -> None:
    ours = nn.MLP([3, 8, 8, 2], activation="tanh")
    theirs = torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    ).double()
    pairs = [(ours[i].weight, theirs[i].weight) for i in (0, 2, 4)] + [
        (ours[i].bias, theirs[i].bias) for i in (0, 2, 4)
    ]
    x = np.random.default_rng(1).standard_normal((6, 3))
    targets = np.array([0, 1, 1, 0, 1, 0])
    check_module(
        ours,
        theirs,
        pairs,
        x,
        loss=(
            lambda z: tg.cross_entropy(z, targets),
            lambda z: F.cross_entropy(z, torch.from_numpy(targets)),
        ),
    )


@pytest.mark.parametrize(("stride", "padding"), [(1, 0), (2, 1)])
def test_conv2d(stride: int, padding: int) -> None:
    ours = nn.Conv2d(2, 3, 3, stride=stride, padding=padding)
    theirs = torch.nn.Conv2d(2, 3, 3, stride=stride, padding=padding).double()
    x = np.random.default_rng(2).standard_normal((2, 2, 6, 5))
    check_module(ours, theirs, [(ours.weight, theirs.weight), (ours.bias, theirs.bias)], x)


def test_embedding() -> None:
    ours, theirs = nn.Embedding(7, 4), torch.nn.Embedding(7, 4).double()
    ids = np.array([[0, 3, 3, 6], [1, 3, 0, 2]])
    check_module(ours, theirs, [(ours.weight, theirs.weight)], ids, input_grad=False)


def test_layer_norm() -> None:
    ours, theirs = nn.LayerNorm(6), torch.nn.LayerNorm(6).double()
    ours.weight.data[...] = np.linspace(0.5, 1.5, 6)  # non-trivial affine parameters
    ours.bias.data[...] = np.linspace(-0.2, 0.3, 6)
    x = np.random.default_rng(3).standard_normal((2, 3, 6)) * 3 + 1
    check_module(ours, theirs, [(ours.weight, theirs.weight), (ours.bias, theirs.bias)], x)


@pytest.mark.parametrize("shape", [(8, 3), (4, 3, 5)])
def test_batch_norm_1d_training_and_evaluation(shape: tuple[int, ...]) -> None:
    ours, theirs = nn.BatchNorm1d(3, momentum=0.3), torch.nn.BatchNorm1d(3, momentum=0.3).double()
    ours.weight.data[...] = [0.5, 1.0, 2.0]
    pairs = [(ours.weight, theirs.weight), (ours.bias, theirs.bias)]
    rng = np.random.default_rng(4)
    for _ in range(3):  # several training steps update the running statistics
        check_module(ours, theirs, pairs, rng.standard_normal(shape) * 2 + 1, seed=1)
    np.testing.assert_allclose(ours.running_mean, theirs.running_mean.numpy(), rtol=1e-6)
    np.testing.assert_allclose(ours.running_var, theirs.running_var.numpy(), rtol=1e-6)
    ours.eval()
    theirs.eval()
    # tensorgrad keeps running statistics in the default dtype of their creation; compare
    # the evaluation-mode output with the same (float64) statistics on both sides.
    with torch.no_grad():
        theirs.running_mean.copy_(torch.from_numpy(ours.running_mean.astype(np.float64)))
        theirs.running_var.copy_(torch.from_numpy(ours.running_var.astype(np.float64)))
    check_module(ours, theirs, pairs, rng.standard_normal(shape))


def test_multi_head_attention_matches_torch_nn_multihead_attention() -> None:
    ours = nn.MultiHeadAttention(8, 2, causal=True)
    theirs = torch.nn.MultiheadAttention(8, 2, batch_first=True).double()
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)  # True = not allowed

    def torch_forward(x: torch.Tensor) -> torch.Tensor:
        return theirs(x, x, x, attn_mask=mask, need_weights=False)[0]

    pairs = [
        (ours.qkv.weight, theirs.in_proj_weight),
        (ours.qkv.bias, theirs.in_proj_bias),
        (ours.proj.weight, theirs.out_proj.weight),
        (ours.proj.bias, theirs.out_proj.bias),
    ]
    x = np.random.default_rng(5).standard_normal((2, 5, 8))
    check_module(ours, torch_forward, pairs, x)  # type: ignore[arg-type]


class TorchBlock(torch.nn.Module):  # type: ignore[misc]
    """Reference pre-LN block written directly in PyTorch."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.fc = torch.nn.Linear(dim, 4 * dim)
        self.out = torch.nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


def block_pairs(
    ours: nn.TransformerBlock, theirs: TorchBlock
) -> list[tuple[tg.Tensor, torch.Tensor]]:
    assert ours.ln1.weight is not None and ours.ln1.bias is not None
    assert ours.ln2.weight is not None and ours.ln2.bias is not None
    fc, out = ours.mlp[0], ours.mlp[2]
    assert isinstance(fc, nn.Linear) and isinstance(out, nn.Linear)
    assert fc.bias is not None and out.bias is not None
    assert ours.attn.qkv.bias is not None and ours.attn.proj.bias is not None
    return [
        (ours.ln1.weight, theirs.ln1.weight),
        (ours.ln1.bias, theirs.ln1.bias),
        (ours.attn.qkv.weight, theirs.attn.in_proj_weight),
        (ours.attn.qkv.bias, theirs.attn.in_proj_bias),
        (ours.attn.proj.weight, theirs.attn.out_proj.weight),
        (ours.attn.proj.bias, theirs.attn.out_proj.bias),
        (ours.ln2.weight, theirs.ln2.weight),
        (ours.ln2.bias, theirs.ln2.bias),
        (fc.weight, theirs.fc.weight),
        (fc.bias, theirs.fc.bias),
        (out.weight, theirs.out.weight),
        (out.bias, theirs.out.bias),
    ]


def test_transformer_block() -> None:
    ours, theirs = nn.TransformerBlock(8, 2), TorchBlock(8, 2).double()
    x = np.random.default_rng(6).standard_normal((2, 4, 8))
    check_module(ours, theirs, block_pairs(ours, theirs), x)


class TorchGPT(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, vocab: int, block: int, layers: int, dim: int, heads: int) -> None:
        super().__init__()
        self.tok = torch.nn.Embedding(vocab, dim)
        self.pos = torch.nn.Embedding(block, dim)
        self.blocks = torch.nn.ModuleList(TorchBlock(dim, heads) for _ in range(layers))
        self.ln_f = torch.nn.LayerNorm(dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.tok.weight.T  # tied output head


def test_gpt_with_tied_embeddings() -> None:
    config = nn.GPTConfig(vocab_size=11, block_size=6, n_layer=2, n_head=2, n_embd=8)
    ours, theirs = nn.GPT(config), TorchGPT(11, 6, 2, 8, 2).double()
    assert ours.ln_f.weight is not None and ours.ln_f.bias is not None
    pairs = [(ours.tok_emb.weight, theirs.tok.weight), (ours.pos_emb.weight, theirs.pos.weight)]
    for mine, ref in zip(ours.blocks, theirs.blocks, strict=True):
        assert isinstance(mine, nn.TransformerBlock)
        pairs += block_pairs(mine, ref)
    pairs += [(ours.ln_f.weight, theirs.ln_f.weight), (ours.ln_f.bias, theirs.ln_f.bias)]
    rng = np.random.default_rng(7)
    ids, targets = rng.integers(0, 11, size=(2, 6)), rng.integers(0, 11, size=(2, 6))
    check_module(
        ours,
        theirs,
        pairs,
        ids,
        input_grad=False,
        loss=(
            lambda z: tg.cross_entropy(z, targets),
            lambda z: F.cross_entropy(z.permute(0, 2, 1), torch.from_numpy(targets)),
        ),
    )


def flatten_outputs(output: Any, state: Any, cat: Callable[..., Any]) -> Any:
    """A recurrent module's output and final state(s) as one flat tensor."""
    states = state if isinstance(state, tuple) else (state,)
    return cat([t.reshape(-1) for t in (output, *states)])


RECURRENT = [("RNN", {}), ("RNN", {"nonlinearity": "relu"}), ("LSTM", {}), ("GRU", {})]


@pytest.mark.parametrize(("name", "kwargs"), RECURRENT, ids=["rnn", "rnn_relu", "lstm", "gru"])
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bias", [True, False])
def test_recurrent_layers(name: str, kwargs: dict[str, Any], batch_first: bool, bias: bool) -> None:
    options = {"num_layers": 2, "bias": bias, "batch_first": batch_first, **kwargs}
    ours = getattr(nn, name)(3, 4, **options)
    theirs = getattr(torch.nn, name)(3, 4, **options).double()
    reference = dict(theirs.named_parameters())
    assert list(reference) == [n for n, _ in ours.named_parameters()]
    pairs = [(p, reference[n]) for n, p in ours.named_parameters()]
    x = np.random.default_rng(8).standard_normal((5, 2, 3))
    check_module(
        lambda x: flatten_outputs(*ours(x), tg.concat),  # type: ignore[arg-type]
        lambda x: flatten_outputs(*theirs(x), torch.cat),
        pairs,
        x,
    )


@pytest.mark.parametrize("name", ["RNN", "LSTM", "GRU"])
def test_recurrent_initial_state_gradients(name: str) -> None:
    ours, theirs = getattr(nn, name)(3, 4, num_layers=2), getattr(torch.nn, name)(3, 4, 2).double()
    copy_weights([(p, dict(theirs.named_parameters())[n]) for n, p in ours.named_parameters()])
    rng = np.random.default_rng(9)
    x = rng.standard_normal((6, 2, 3))
    states = [rng.standard_normal((2, 2, 4)) for _ in range(2 if name == "LSTM" else 1)]
    s_ours = [tg.Tensor(s, requires_grad=True) for s in states]
    s_theirs = [torch.tensor(s, requires_grad=True) for s in states]
    hx_ours = tuple(s_ours) if name == "LSTM" else s_ours[0]
    hx_theirs = tuple(s_theirs) if name == "LSTM" else s_theirs[0]
    out_ours = flatten_outputs(*ours(tg.Tensor(x), hx_ours), tg.concat)
    out_theirs = flatten_outputs(*theirs(torch.tensor(x), hx_theirs), torch.cat)
    assert_close(out_ours, out_theirs, "output")
    v = rng.standard_normal(out_ours.shape)
    g_ours = tg.autograd.grad((out_ours * v).sum(), s_ours)
    g_theirs = torch.autograd.grad((out_theirs * torch.tensor(v)).sum(), s_theirs)
    for a, b in zip(g_ours, g_theirs, strict=True):
        assert_close(a, b, "initial-state gradient")
