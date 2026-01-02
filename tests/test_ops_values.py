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


def argmax_pool_reference(
    x: np.ndarray, g: np.ndarray, k: int, stride: int, pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Output and input gradient of max pooling where each window's gradient goes to the
    cell numpy.argmax picks (the first maximum, or the first NaN)."""
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=-np.inf)
    n, c, hp, wp = xp.shape
    oh, ow = (hp - k) // stride + 1, (wp - k) // stride + 1
    out = np.empty((n, c, oh, ow))
    dxp = np.zeros_like(xp)
    for b, ch, i, j in np.ndindex(n, c, oh, ow):
        window = xp[b, ch, i * stride : i * stride + k, j * stride : j * stride + k]
        r, q = divmod(int(np.argmax(window)), k)
        out[b, ch, i, j] = window[r, q]
        dxp[b, ch, i * stride + r, j * stride + q] += g[b, ch, i, j]
    return out, dxp[:, :, pad : pad + x.shape[2], pad : pad + x.shape[3]]


@pytest.mark.parametrize(("k", "stride", "pad"), [(2, 2, 0), (3, 1, 1), (3, 2, 1), (2, 1, 0)])
def test_max_pool2d_ties_and_nans_follow_argmax(k: int, stride: int, pad: int) -> None:
    # Values on a coarse grid make ties common; the gradient must go to the first maximum
    # of each window, and a NaN must win its window (as numpy.argmax and PyTorch do).
    rng = np.random.default_rng(k * 10 + stride)
    x = np.round(rng.standard_normal((2, 3, 7, 6)))
    x.reshape(-1)[::13] = np.nan
    t = tg.Tensor(x, requires_grad=True)
    out = tg.max_pool2d(t, k, stride, pad)
    g = rng.standard_normal(out.shape)
    out.backward(g)
    expected_out, expected_grad = argmax_pool_reference(x, g, k, stride, pad)
    np.testing.assert_array_equal(out.data, expected_out)
    assert t.grad is not None
    np.testing.assert_allclose(t.grad.data, expected_grad)
    (graph_grad,) = tg.autograd.grad(
        tg.max_pool2d(t, k, stride, pad), [t], grad_outputs=tg.Tensor(g), create_graph=True
    )
    np.testing.assert_allclose(graph_grad.data, expected_grad)


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


def test_conv2d_rejects_a_misshaped_bias() -> None:
    # Regression: a (1,) bias broadcast silently in the forward pass and only failed inside
    # backward() with an internal gradient-shape error.
    x = tg.Tensor(np.ones((1, 1, 3, 3)), requires_grad=True)
    w = tg.Tensor(np.ones((2, 1, 2, 2)), requires_grad=True)
    with pytest.raises(ValueError, match="bias must have shape"):
        tg.conv2d(x, w, tg.Tensor(np.ones(1)))


def test_max_pool2d_rejects_padding_above_half_the_kernel() -> None:
    # Regression: padding 2 with a 2x2 kernel produced windows made only of padding, whose
    # output was -inf (PyTorch rejects this configuration).
    with pytest.raises(ValueError, match="at most half"):
        tg.max_pool2d(tg.zeros((1, 1, 4, 4)), 2, stride=1, padding=2)


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


def test_binary_cross_entropy_matches_the_definition_and_is_stable() -> None:
    z = np.array([-3.0, -0.5, 0.0, 1.5, 4.0])
    y = np.array([0.0, 1.0, 0.3, 1.0, 0.0])
    p = 1 / (1 + np.exp(-z))
    expected = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    got = tg.binary_cross_entropy_with_logits(tg.Tensor(z), y, reduction="none")
    np.testing.assert_allclose(got.data, expected, rtol=1e-12)
    assert tg.binary_cross_entropy_with_logits(tg.Tensor(z), y).item() == pytest.approx(
        expected.mean()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no overflow in exp or log(0)
        extreme = tg.Tensor(np.array([-1000.0, 1000.0]), requires_grad=True)
        loss = tg.binary_cross_entropy_with_logits(extreme, np.array([1.0, 0.0]), reduction="sum")
        loss.backward()
    assert loss.item() == pytest.approx(2000.0)
    assert extreme.grad is not None
    np.testing.assert_allclose(extreme.grad.data, [-1.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        tg.binary_cross_entropy_with_logits(tg.Tensor(z), np.zeros(3))


def test_sigmoid_is_stable_and_symmetric() -> None:
    x = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = tg.sigmoid(tg.Tensor(x)).data
    assert s[0] == 0.0 or s[0] < 1e-300
    assert s[-1] == 1.0
    np.testing.assert_allclose(s[1:4], 1 / (1 + np.exp(-x[1:4])))
    np.testing.assert_allclose(s + s[::-1], 1.0)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_sigmoid_keeps_floating_dtypes(dtype: type) -> None:
    assert tg.sigmoid(tg.Tensor(np.array([-2.0, 0.5], dtype=dtype))).dtype == dtype


def test_sigmoid_of_integers_is_a_float() -> None:
    # Regression: the result was cast back to the input dtype, so integers truncated to 0.
    x = np.array([1, 2, -3])
    s = tg.sigmoid(tg.Tensor(x))
    assert s.dtype == np.float64
    np.testing.assert_allclose(s.data, 1 / (1 + np.exp(-x)))


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


@pytest.mark.parametrize("generator", [np.random.PCG64, np.random.MT19937])
def test_dropout_masks_are_unbiased_for_any_bit_generator(generator: type) -> None:
    # PCG64 takes the fast path (raw 64-bit words split into four 16-bit uniforms); MT19937
    # produces only 32 bits per raw word and must use the portable path instead.
    from tensorgrad._random import keep_mask

    rng = np.random.Generator(generator(0))
    keep, scale = keep_mask((400_000,), 0.1, rng)
    threshold = round(0.1 * 65536)
    assert scale == 65536 / (65536 - threshold)  # matches the quantised keep probability
    assert keep.mean() == pytest.approx(1 - threshold / 65536, abs=3e-3)
    assert (keep * scale).mean() == pytest.approx(1.0, abs=4e-3)
    again, _ = keep_mask((400_000,), 0.1, np.random.Generator(generator(0)))
    np.testing.assert_array_equal(keep, again)  # deterministic for a seeded generator


def test_dropout_probability_quantisation_is_exact_for_dyadic_rates() -> None:
    from tensorgrad._random import keep_mask

    rng = np.random.default_rng(0)
    for p, scale in ((0.5, 2.0), (0.25, 4 / 3), (0.0, 1.0)):
        assert keep_mask((10,), p, rng)[1] == pytest.approx(scale, rel=1e-15)
    keep, scale = keep_mask((10,), 1.0, rng)
    assert not keep.any() and scale == 0.0


def test_embedding_lookup() -> None:
    w = tg.Tensor(np.arange(12.0).reshape(4, 3))
    out = tg.embedding(np.array([[3, 0]]), w)
    assert out.shape == (1, 2, 3)
    np.testing.assert_array_equal(out.data[0, 0], [9.0, 10.0, 11.0])
    with pytest.raises(TypeError, match="integers"):
        tg.embedding(np.array([0.5]), w)
    assert tg.embedding(np.zeros((2, 0), dtype=np.int64), w).shape == (2, 0, 3)


@pytest.mark.parametrize("vocab", [1, 5, 300])
def test_embedding_gradient_sums_repeated_rows_like_add_at(vocab: int) -> None:
    rng = np.random.default_rng(vocab)
    ids = rng.integers(0, vocab, size=(7, 11))
    w = tg.Tensor(rng.standard_normal((vocab, 3)), requires_grad=True)
    g = rng.standard_normal((7, 11, 3))
    tg.embedding(ids, w).backward(g)
    expected = np.zeros((vocab, 3))
    np.add.at(expected, ids.reshape(-1), g.reshape(-1, 3))
    assert w.grad is not None
    np.testing.assert_allclose(w.grad.data, expected, rtol=1e-12, atol=1e-12)


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
    assert x.grad is not None and np.all(np.isnan(x.grad.data))


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
