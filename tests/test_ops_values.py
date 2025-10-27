"""Forward-pass correctness against straightforward reference implementations."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad.ops._util import unbroadcast


def naive_conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, pad: int) -> np.ndarray:
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    n, _, h, wd = xp.shape
    c_out, _, kh, kw = w.shape
    oh, ow = (h - kh) // stride + 1, (wd - kw) // stride + 1
    out = np.zeros((n, c_out, oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def naive_max_pool(x: np.ndarray, k: int, stride: int) -> np.ndarray:
    n, c, h, w = x.shape
    oh, ow = (h - k) // stride + 1, (w - k) // stride + 1
    out = np.zeros((n, c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i * stride : i * stride + k, j * stride : j * stride + k].max(
                axis=(2, 3)
            )
    return out


@pytest.mark.parametrize(("stride", "pad"), [(1, 0), (1, 1), (2, 1), (3, 0)])
def test_conv2d_matches_naive_loops(rng: np.random.Generator, stride: int, pad: int) -> None:
    x = rng.standard_normal((2, 3, 7, 6))
    w = rng.standard_normal((4, 3, 3, 3))
    b = rng.standard_normal(4)
    out = tg.conv2d(tg.Tensor(x), tg.Tensor(w), tg.Tensor(b), stride=stride, padding=pad)
    np.testing.assert_allclose(out.data, naive_conv2d(x, w, b, stride, pad), atol=1e-12)


@pytest.mark.parametrize(("k", "stride"), [(2, 2), (3, 1), (2, 1)])
def test_max_pool2d_matches_naive_loops(rng: np.random.Generator, k: int, stride: int) -> None:
    x = rng.standard_normal((2, 3, 6, 7))
    out = tg.max_pool2d(tg.Tensor(x), k, stride)
    np.testing.assert_allclose(out.data, naive_max_pool(x, k, stride))


def test_max_pool2d_padding_never_wins() -> None:
    x = -np.ones((1, 1, 2, 2))
    out = tg.max_pool2d(tg.Tensor(x), 2, stride=1, padding=1)
    assert np.all(out.data == -1.0)


def test_conv_and_pool_validate_shapes() -> None:
    with pytest.raises(ValueError, match="channels"):
        tg.conv2d(tg.zeros((1, 2, 4, 4)), tg.zeros((1, 3, 3, 3)))
    with pytest.raises(ValueError, match="does not fit"):
        tg.conv2d(tg.zeros((1, 1, 2, 2)), tg.zeros((1, 1, 3, 3)))
    with pytest.raises(ValueError, match="4-D"):
        tg.max_pool2d(tg.zeros((2, 2)), 2)


def test_softmax_is_stable_for_huge_logits() -> None:
    logits = tg.Tensor(np.array([[1000.0, 1000.0, -1000.0], [-1e4, 0.0, 1e4]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no overflow warnings allowed
        s = tg.softmax(logits).data
        ls = tg.log_softmax(logits).data
        lse = tg.logsumexp(logits).data
    np.testing.assert_allclose(s.sum(axis=-1), 1.0)
    np.testing.assert_allclose(s[0], [0.5, 0.5, 0.0], atol=1e-12)
    assert np.all(np.isfinite(ls))  # log(softmax) would give -inf here
    np.testing.assert_allclose(ls[1], [-2e4, -1e4, 0.0])
    np.testing.assert_allclose(lse, [1000.0 + math.log(2.0), 1e4])


def test_softmax_handles_masked_rows() -> None:
    x = tg.Tensor(np.array([[0.0, -np.inf, 1.0]]), requires_grad=True)
    s = tg.softmax(x)
    assert s.data[0, 1] == 0.0
    s[:, 0].sum().backward()
    assert x.grad is not None
    assert x.grad[0, 1] == 0.0


def test_cross_entropy_matches_manual_computation(rng: np.random.Generator) -> None:
    logits = rng.standard_normal((5, 4))
    target = np.array([0, 3, 1, 1, 2])
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    expected = -logp[np.arange(5), target]
    got = tg.cross_entropy(tg.Tensor(logits), target, reduction="none")
    np.testing.assert_allclose(got.data, expected)
    np.testing.assert_allclose(tg.cross_entropy(tg.Tensor(logits), target).item(), expected.mean())
    np.testing.assert_allclose(
        tg.cross_entropy(tg.Tensor(logits), target, reduction="sum").item(), expected.sum()
    )


def test_cross_entropy_ignore_index() -> None:
    logits = tg.Tensor(np.zeros((3, 4)), requires_grad=True)
    loss = tg.cross_entropy(logits, np.array([1, -100, 2]))
    assert loss.item() == pytest.approx(math.log(4.0))  # mean over the 2 valid rows only
    loss.backward()
    assert logits.grad is not None
    np.testing.assert_array_equal(logits.grad[1], 0.0)

    all_ignored = tg.cross_entropy(tg.Tensor(np.zeros((2, 3))), np.array([7, 7]), ignore_index=7)
    assert all_ignored.item() == 0.0


def test_cross_entropy_validates_targets() -> None:
    logits = tg.Tensor(np.zeros((2, 3)))
    with pytest.raises(TypeError, match="integers"):
        tg.cross_entropy(logits, np.array([0.0, 1.0]))
    with pytest.raises(IndexError, match="out of range"):
        tg.cross_entropy(logits, np.array([0, 3]))
    with pytest.raises(ValueError, match="incompatible"):
        tg.cross_entropy(logits, np.array([0, 1, 2]))


def test_cross_entropy_of_confident_correct_logits_is_small() -> None:
    logits = tg.Tensor(np.array([[50.0, -50.0]]))
    assert tg.cross_entropy(logits, np.array([0])).item() < 1e-12


def test_sigmoid_is_stable_and_symmetric() -> None:
    x = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = tg.sigmoid(tg.Tensor(x)).data
    assert s[0] == 0.0 or s[0] < 1e-300
    assert s[-1] == 1.0
    np.testing.assert_allclose(s[1:4], 1 / (1 + np.exp(-x[1:4])))
    np.testing.assert_allclose(s + s[::-1], 1.0)


def test_gelu_reference_values() -> None:
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    c = math.sqrt(2 / math.pi)
    expected = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))
    np.testing.assert_allclose(tg.gelu(tg.Tensor(x)).data, expected)
    assert tg.gelu(tg.Tensor(np.array(1.0))).item() == pytest.approx(0.841192, abs=1e-6)


def test_relu_and_where() -> None:
    x = tg.Tensor(np.array([-1.0, 0.0, 2.0]))
    assert tg.relu(x).tolist() == [0.0, 0.0, 2.0]
    assert tg.where(x.data > 0, x, 10.0).tolist() == [10.0, 10.0, 2.0]


def test_layer_norm_normalises_last_axis(rng: np.random.Generator) -> None:
    x = rng.standard_normal((4, 16)) * 5 + 3
    out = tg.layer_norm(tg.Tensor(x)).data
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-7)
    np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-3)


def test_dropout_statistics_and_eval_identity(rng: np.random.Generator) -> None:
    x = tg.Tensor(np.ones(100_000))
    out = tg.dropout(x, 0.25, rng=np.random.default_rng(0)).data
    np.testing.assert_allclose(np.unique(out), [0.0, 1 / 0.75])
    assert out.mean() == pytest.approx(1.0, abs=0.01)  # inverted dropout keeps the expectation
    assert tg.dropout(x, 0.25, training=False) is x
    assert np.all(tg.dropout(x, 1.0).data == 0.0)
    with pytest.raises(ValueError, match="probability"):
        tg.dropout(x, 1.5)


def test_embedding_lookup() -> None:
    w = tg.Tensor(np.arange(12.0).reshape(4, 3))
    out = tg.embedding(np.array([[3, 0]]), w)
    assert out.shape == (1, 2, 3)
    np.testing.assert_array_equal(out.data[0, 0], [9.0, 10.0, 11.0])
    with pytest.raises(TypeError, match="integers"):
        tg.embedding(np.array([0.5]), w)
    assert tg.embedding(np.zeros((2, 0), dtype=np.int64), w).shape == (2, 0, 3)


@pytest.mark.parametrize("bad_id", [-1, 4])
def test_embedding_rejects_out_of_range_ids(bad_id: int) -> None:
    # Regression: id -1 silently returned the last row (NumPy's negative indexing).
    w = tg.Tensor(np.arange(12.0).reshape(4, 3))
    with pytest.raises(IndexError, match="embedding ids"):
        tg.embedding(np.array([0, bad_id]), w)


def test_reductions_keep_float32_and_promote_integers() -> None:
    x = tg.Tensor(np.ones((2, 3), dtype=np.float32))
    assert x.mean().dtype == x.var().dtype == x.sum(axis=0).dtype == np.float32
    ints = tg.Tensor(np.array([1, 2]))
    assert ints.mean().item() == 1.5


def test_var_matches_numpy(rng: np.random.Generator) -> None:
    x = rng.standard_normal((3, 5))
    np.testing.assert_allclose(tg.Tensor(x).var(axis=1).data, x.var(axis=1, ddof=1))
    np.testing.assert_allclose(tg.Tensor(x).var(correction=0).data, x.var())


@pytest.mark.parametrize("shape", [(1,), (1, 3)])
def test_unbiased_var_of_a_single_sample_is_nan_not_an_error(shape: tuple[int, ...]) -> None:
    # Regression: backward divided a Python int by zero (ZeroDivisionError). PyTorch returns
    # NaN for both the value and the gradient when N <= correction.
    x = tg.Tensor(np.ones(shape), requires_grad=True)
    v = x.var(axis=0)
    assert np.all(np.isnan(v.data))
    v.sum().backward()
    assert x.grad is not None and np.all(np.isnan(x.grad))


@pytest.mark.parametrize(
    ("grad_shape", "target", "expected"),
    [
        ((2, 3), (2, 3), np.ones((2, 3))),
        ((2, 3), (3,), np.full(3, 2.0)),
        ((2, 3), (1, 3), np.full((1, 3), 2.0)),
        ((2, 3), (2, 1), np.full((2, 1), 3.0)),
        ((4, 2, 3), (), np.array(24.0)),
        ((4, 2, 3), (2, 1), np.full((2, 1), 12.0)),
    ],
)
def test_unbroadcast(
    grad_shape: tuple[int, ...], target: tuple[int, ...], expected: np.ndarray
) -> None:
    out = unbroadcast(np.ones(grad_shape), target)
    assert out.shape == target
    np.testing.assert_allclose(out, expected)


def test_shape_op_errors() -> None:
    t = tg.zeros((2, 3))
    with pytest.raises(ValueError, match="permutation"):
        t.permute(0, 0)
    with pytest.raises(ValueError, match="0-d"):
        tg.matmul(tg.Tensor(1.0), t)
    with pytest.raises(ValueError, match="at least one"):
        tg.concat([])
    with pytest.raises(IndexError, match="integer or boolean"):
        t[np.array([0.5])]
    assert t.reshape((3, 2)).shape == t.reshape(3, 2).shape == (3, 2)
    assert t.reshape(np.int64(3), -1).shape == (3, 2)  # NumPy integers are accepted too
    assert t.squeeze(0).shape == (2, 3)  # not size 1: unchanged
