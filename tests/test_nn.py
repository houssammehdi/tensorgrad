"""Module system, layers, attention and the GPT model."""

from __future__ import annotations

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import nn
from tensorgrad.utils import gradcheck


@pytest.fixture
def float64_default() -> object:
    tg.set_default_dtype(np.float64)
    yield
    tg.set_default_dtype(np.float32)


class TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.norm = nn.BatchNorm1d(4)
        self.decoder = nn.Linear(4, 2, bias=False)
        self.scale = nn.Parameter(np.ones(1, dtype=np.float32))

    def forward(self, x: tg.Tensor) -> tg.Tensor:
        return self.decoder(self.norm(self.encoder(x)).relu()) * self.scale


class TestModule:
    def test_parameter_registration_and_names(self) -> None:
        model = TwoLayer()
        names = [name for name, _ in model.named_parameters()]
        assert names == ["scale", "encoder.weight", "encoder.bias", "norm.weight", "norm.bias",
                         "decoder.weight"]  # fmt: skip
        assert model.num_parameters() == 1 + 12 + 4 + 4 + 4 + 8
        assert [n for n, _ in model.named_buffers()] == ["norm.running_mean", "norm.running_var"]
        assert [n for n, _ in model.named_modules()] == ["", "encoder", "norm", "decoder"]

    def test_reassignment_and_deletion_update_registry(self) -> None:
        model = TwoLayer()
        model.decoder = nn.Linear(4, 2)  # replaced module (now with a bias)
        assert "decoder.bias" in dict(model.named_parameters())
        del model.scale
        assert "scale" not in dict(model.named_parameters())
        model.encoder.bias = None
        assert "encoder.bias" not in model.state_dict()

    def test_assigning_before_init_fails(self) -> None:
        class Broken(nn.Module):
            def __init__(self) -> None:
                self.w = nn.Parameter([1.0])

        with pytest.raises(AttributeError, match="__init__"):
            Broken()

    def test_buffers_must_be_arrays(self) -> None:
        with pytest.raises(TypeError, match="buffer"):
            nn.BatchNorm1d(3).running_mean = [0.0]

    def test_state_dict_round_trip(self) -> None:
        a, b = TwoLayer(), TwoLayer()
        x = tg.Tensor(np.random.default_rng(0).standard_normal((8, 3)).astype(np.float32))
        a(x)  # update running statistics so buffers differ from their defaults
        state = a.state_dict()
        assert set(state) == {
            "scale", "encoder.weight", "encoder.bias", "norm.weight", "norm.bias",
            "norm.running_mean", "norm.running_var", "decoder.weight",
        }  # fmt: skip
        b.load_state_dict(state)
        a.eval()
        b.eval()
        np.testing.assert_array_equal(a(x).data, b(x).data)
        state["scale"][...] = 5.0  # state_dict returns copies
        assert a.scale.item() == 1.0

    def test_load_state_dict_validation(self) -> None:
        model = TwoLayer()
        state = model.state_dict()
        state.pop("scale")
        state["extra"] = np.zeros(1)
        with pytest.raises(KeyError, match="missing"):
            model.load_state_dict(state)
        result = model.load_state_dict(state, strict=False)
        assert result.missing_keys == ["scale"]
        assert result.unexpected_keys == ["extra"]
        state = model.state_dict()
        state["encoder.weight"] = np.zeros((2, 2))
        with pytest.raises(ValueError, match="shape mismatch"):
            model.load_state_dict(state)

    def test_train_eval_propagates(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.Dropout(0.5)))
        model.eval()
        assert all(not m.training for m in model.modules())
        model.train()
        assert all(m.training for m in model.modules())

    def test_zero_grad_and_apply(self) -> None:
        model = TwoLayer()
        model(tg.randn((4, 3))).sum().backward()
        assert all(p.grad is not None for p in model.parameters())
        model.zero_grad()
        assert all(p.grad is None for p in model.parameters())
        visited: list[str] = []
        model.apply(lambda m: visited.append(type(m).__name__))
        assert visited[-1] == "TwoLayer"

    def test_repr_shows_tree(self) -> None:
        text = repr(nn.Sequential(nn.Linear(2, 3), nn.ReLU()))
        assert "(0): Linear(2, 3, bias=True)" in text
        assert "(1): ReLU()" in text

    def test_forward_must_be_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            nn.Module()(tg.zeros(1))


class TestLayers:
    def test_linear_shapes_and_init_bounds(self) -> None:
        layer = nn.Linear(16, 5)
        assert layer(tg.randn((2, 7, 16))).shape == (2, 7, 5)
        assert np.abs(layer.weight.data).max() <= 1 / 4
        assert layer.weight.dtype == np.float32

    def test_conv_pool_flatten_pipeline(self) -> None:
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64, 3),
        )
        assert model(tg.randn((5, 1, 8, 8))).shape == (5, 3)

    def test_embedding_layer(self) -> None:
        emb = nn.Embedding(10, 4)
        assert emb(np.array([[1, 2, 3]])).shape == (1, 3, 4)

    def test_layer_norm_module(self) -> None:
        ln = nn.LayerNorm(6)
        out = ln(tg.randn((3, 6)) * 4 + 2)
        np.testing.assert_allclose(out.data.mean(axis=-1), 0.0, atol=1e-5)
        with pytest.raises(ValueError, match="last dimension"):
            ln(tg.randn((3, 5)))
        assert nn.LayerNorm(4, elementwise_affine=False).num_parameters() == 0

    def test_batchnorm_running_stats_and_eval(self) -> None:
        bn = nn.BatchNorm1d(3, momentum=0.5)
        x = tg.Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]], dtype=np.float32))
        out = bn(x)
        np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(bn.running_mean, [1.0, 2.0, 3.0])  # 0.5 * batch mean
        np.testing.assert_allclose(bn.running_var, [1.5, 4.5, 9.5])  # 0.5 + 0.5 * unbiased var
        bn.eval()
        expected = (x.data - bn.running_mean) / np.sqrt(bn.running_var + bn.eps)
        np.testing.assert_allclose(bn(x).data, expected, rtol=1e-6)
        seq = nn.BatchNorm1d(3)(tg.randn((4, 3, 5)))  # (N, C, L) input
        assert seq.shape == (4, 3, 5)

    def test_dropout_module_respects_mode(self) -> None:
        drop = nn.Dropout(0.5)
        x = tg.ones((1000,))
        assert (drop(x).data == 0).any()
        drop.eval()
        assert drop(x) is x
        with pytest.raises(ValueError, match="probability"):
            nn.Dropout(-0.1)

    def test_sequential_indexing(self) -> None:
        seq = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
        assert len(seq) == 3
        assert isinstance(seq[-1], nn.Linear)
        with pytest.raises(IndexError):
            seq[3]

    @pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
    def test_mlp_structure(self, activation: str) -> None:
        mlp = nn.MLP([2, 8, 8, 3], activation=activation, dropout=0.1)
        kinds = [type(m).__name__ for m in mlp]
        assert kinds.count("Linear") == 3
        assert kinds.count("Dropout") == 2
        assert kinds[-1] == "Linear"
        assert mlp(tg.randn((4, 2))).shape == (4, 3)
        with pytest.raises(ValueError, match="at least"):
            nn.MLP([3])

    def test_whole_module_gradcheck(self, float64_default: object) -> None:
        mlp = nn.MLP([3, 4, 2], activation="tanh")
        x = tg.randn((5, 3))
        assert gradcheck(lambda *_: mlp(x), list(mlp.parameters()))


class TestAttention:
    def test_output_shape_and_validation(self) -> None:
        mha = nn.MultiHeadAttention(16, 4)
        assert mha(tg.randn((2, 5, 16))).shape == (2, 5, 16)
        with pytest.raises(ValueError, match="divisible"):
            nn.MultiHeadAttention(10, 3)
        with pytest.raises(ValueError, match="expected"):
            mha(tg.randn((2, 5, 8)))

    @pytest.mark.parametrize("position", [0, 3, 6])
    def test_causal_mask_blocks_future_tokens(self, position: int) -> None:
        """The output at position t must not depend on inputs at positions > t."""
        mha = nn.MultiHeadAttention(8, 2, causal=True)
        x = tg.Tensor(np.random.default_rng(0).standard_normal((2, 7, 8)), requires_grad=True)
        mha(x)[:, position].sum().backward()
        assert x.grad is not None
        assert np.all(x.grad[:, position + 1 :] == 0.0)  # exactly zero: no leakage at all
        assert np.all(np.abs(x.grad[:, : position + 1]).sum(axis=-1) > 0)  # past does matter

    def test_non_causal_attention_sees_the_future(self) -> None:
        mha = nn.MultiHeadAttention(8, 2, causal=False)
        x = tg.Tensor(np.random.default_rng(0).standard_normal((1, 5, 8)), requires_grad=True)
        mha(x)[:, 0].sum().backward()
        assert x.grad is not None
        assert np.all(np.abs(x.grad[:, 1:]).sum(axis=-1) > 0)

    def test_causal_mask_helper(self) -> None:
        mask = nn.attention.causal_mask(3, 3)
        np.testing.assert_array_equal(mask, np.triu(np.ones((3, 3), dtype=bool), k=1))
        assert not mask.flags.writeable
        # With a longer key sequence (cached prefix), the last query sees every key.
        assert not nn.attention.causal_mask(2, 4)[-1].any()

    def test_attention_heads_match_manual_computation(self, float64_default: object) -> None:
        mha = nn.MultiHeadAttention(4, 2, causal=True)
        x = tg.randn((1, 3, 4))
        qkv = x.data @ mha.qkv.weight.data.T + mha.qkv.bias.data
        q, k, v = np.split(qkv, 3, axis=-1)
        heads = []
        for h in range(2):
            sl = slice(2 * h, 2 * h + 2)
            scores = q[0, :, sl] @ k[0, :, sl].T / np.sqrt(2)
            scores[np.triu_indices(3, 1)] = -np.inf
            w = np.exp(scores - scores.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            heads.append(w @ v[0, :, sl])
        expected = np.concatenate(heads, axis=-1) @ mha.proj.weight.data.T + mha.proj.bias.data
        np.testing.assert_allclose(mha(x).data[0], expected, atol=1e-12)

    def test_transformer_block_gradcheck(self, float64_default: object) -> None:
        block = nn.TransformerBlock(8, 2)
        x = tg.randn((2, 3, 8))
        params = list(block.parameters())
        assert gradcheck(lambda *_: block(x), [*params, x.requires_grad_()])


class TestGPT:
    def config(self, **overrides: object) -> nn.GPTConfig:
        base: dict[str, object] = {
            "vocab_size": 11, "block_size": 8, "n_layer": 2, "n_head": 2, "n_embd": 16
        }  # fmt: skip
        base.update(overrides)
        return nn.GPTConfig(**base)

    def test_forward_shape_and_length_check(self) -> None:
        model = nn.GPT(self.config())
        assert model(np.zeros((3, 8), dtype=np.int64)).shape == (3, 8, 11)
        with pytest.raises(ValueError, match="block_size"):
            model(np.zeros((1, 9), dtype=np.int64))
        with pytest.raises(ValueError, match="token ids"):
            model(np.zeros(4, dtype=np.int64))

    def test_weight_tying_shares_one_parameter(self) -> None:
        tied = nn.GPT(self.config())
        untied = nn.GPT(self.config(tie_weights=False))
        assert tied.head.weight is tied.tok_emb.weight
        assert untied.num_parameters() - tied.num_parameters() == 11 * 16
        names = [n for n, _ in tied.named_parameters()]
        assert "head.weight" not in names  # deduplicated
        assert "head.weight" in tied.state_dict()  # but still addressable by name

    def test_tied_weight_receives_both_gradients(self) -> None:
        model = nn.GPT(self.config())
        idx = np.array([[1, 2, 3, 4]])
        tg.cross_entropy(model(idx), np.array([[2, 3, 4, 5]])).backward()
        grad = model.tok_emb.weight.grad
        assert grad is not None
        # Rows never used as inputs still get gradient through the output head.
        assert np.abs(grad[7]).sum() > 0

    def test_future_tokens_do_not_change_past_logits(self) -> None:
        model = nn.GPT(self.config())
        a = np.array([[1, 2, 3, 4, 5]])
        b = a.copy()
        b[0, 3:] = [9, 10]
        la, lb = model(a).data, model(b).data
        np.testing.assert_allclose(la[:, :3], lb[:, :3], atol=1e-6)
        assert not np.allclose(la[:, 3:], lb[:, 3:])

    def test_generate(self) -> None:
        model = nn.GPT(self.config(dropout=0.1))
        out = model.generate(np.array([[1, 2]]), 10, top_k=3, rng=np.random.default_rng(0))
        assert out.shape == (1, 12)
        assert out.max() < 11
        assert model.training  # mode restored
        greedy_a = model.generate(np.array([[1]]), 5, temperature=1e-4)
        greedy_b = model.generate(np.array([[1]]), 5, temperature=1e-4)
        np.testing.assert_array_equal(greedy_a, greedy_b)
        with pytest.raises(ValueError, match="temperature"):
            model.generate(np.array([[1]]), 1, temperature=0.0)

    def test_generate_crops_context_to_block_size(self) -> None:
        model = nn.GPT(self.config(block_size=4))
        out = model.generate(np.array([[1, 2, 3]]), 6, rng=np.random.default_rng(1))
        assert out.shape == (1, 9)
