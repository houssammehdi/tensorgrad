"""Finite-difference gradient checks for every differentiable op.

Each test compares the analytical vector-Jacobian product produced by ``backward`` with
central differences in float64 (see :func:`tensorgrad.utils.gradcheck`). Hypothesis draws
shapes -- including every flavour of broadcasting -- and seeds for the data.
"""

from __future__ import annotations

from collections.abc import Callable

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import tensorgrad as tg
from helpers import distinct, leaf
from tensorgrad.nn.functional import scaled_dot_product_attention
from tensorgrad.utils import gradcheck

seeds = st.integers(min_value=0, max_value=2**32 - 1)
small_shapes = hnp.array_shapes(min_dims=0, max_dims=3, min_side=1, max_side=4)
broadcast_pairs = hnp.mutually_broadcastable_shapes(
    num_shapes=2, min_dims=0, max_dims=3, max_side=3
)


# --------------------------------------------------------------------------- elementwise
BINARY: dict[str, tuple[Callable[[tg.Tensor, tg.Tensor], tg.Tensor], bool]] = {
    # name: (fn, second operand must be positive)
    "add": (lambda a, b: a + b, False),
    "sub": (lambda a, b: a - b, False),
    "mul": (lambda a, b: a * b, False),
    "div": (lambda a, b: a / b, True),
}


@pytest.mark.parametrize("name", sorted(BINARY))
@given(shapes=broadcast_pairs, seed=seeds)
def test_binary_ops_with_broadcasting(
    name: str, shapes: hnp.BroadcastableShapes, seed: int
) -> None:
    fn, positive = BINARY[name]
    rng = np.random.default_rng(seed)
    a = leaf(rng, *shapes.input_shapes[0])
    b = leaf(rng, *shapes.input_shapes[1], positive=positive)
    assert gradcheck(fn, [a, b])


@given(shapes=broadcast_pairs, seed=seeds)
def test_pow_tensor_exponent_with_broadcasting(shapes: hnp.BroadcastableShapes, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = leaf(rng, *shapes.input_shapes[0], positive=True)
    exponent = leaf(rng, *shapes.input_shapes[1])
    assert gradcheck(lambda a, b: a**b, [base, exponent])


@pytest.mark.parametrize("exponent", [2, 3, 0.5, -1.5])
def test_pow_scalar_exponent(rng: np.random.Generator, exponent: float) -> None:
    assert gradcheck(lambda a: a**exponent, [leaf(rng, 3, 4, positive=True)])


def test_rpow_scalar_base(rng: np.random.Generator) -> None:
    assert gradcheck(lambda a: 2.0**a, [leaf(rng, 3, 4)])


def test_scalar_operands_and_reflected_ops(rng: np.random.Generator) -> None:
    fn = lambda a: (1.0 - a) * 3.0 / (a * a + 1.0) + 2.0 / (a * a + 1.0) - (-a)
    assert gradcheck(fn, [leaf(rng, 5)])


def test_numpy_array_operands(rng: np.random.Generator) -> None:
    const = rng.standard_normal((3, 1))
    assert gradcheck(lambda a: const * a + const - a / (const**2 + 1), [leaf(rng, 3, 4)])


UNARY: dict[str, tuple[Callable[[tg.Tensor], tg.Tensor], dict[str, bool]]] = {
    "neg": (lambda a: -a, {}),
    "exp": (tg.exp, {}),
    "log": (tg.log, {"positive": True}),
    "sqrt": (tg.sqrt, {"positive": True}),
    "tanh": (tg.tanh, {}),
    "sigmoid": (tg.sigmoid, {}),
    "relu": (tg.relu, {"away_from_zero": True}),
    "gelu": (tg.gelu, {}),
}


@pytest.mark.parametrize("name", sorted(UNARY))
@given(shape=small_shapes, seed=seeds)
def test_unary_ops(name: str, shape: tuple[int, ...], seed: int) -> None:
    fn, kwargs = UNARY[name]
    assert gradcheck(fn, [leaf(np.random.default_rng(seed), *shape, **kwargs)])


def test_astype_roundtrip_gradient(rng: np.random.Generator) -> None:
    assert gradcheck(
        lambda a: a.astype(np.float32).astype(np.float64) * 2,
        [leaf(rng, 3)],
        eps=1e-3,
        atol=1e-3,
        rtol=1e-3,
    )


# --------------------------------------------------------------------------- reductions
@st.composite
def shape_and_axis(draw: st.DrawFn) -> tuple[tuple[int, ...], int | tuple[int, ...] | None]:
    shape = draw(hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4))
    ndim = len(shape)
    axis = draw(
        st.one_of(
            st.none(),
            st.integers(min_value=-ndim, max_value=ndim - 1),
            st.lists(st.integers(0, ndim - 1), min_size=1, max_size=ndim, unique=True).map(tuple),
        )
    )
    return shape, axis


@pytest.mark.parametrize("name", ["sum", "mean", "max", "min", "var"])
@given(spec=shape_and_axis(), keepdims=st.booleans(), seed=seeds)
def test_reductions(
    name: str,
    spec: tuple[tuple[int, ...], int | tuple[int, ...] | None],
    keepdims: bool,
    seed: int,
) -> None:
    shape, axis = spec
    rng = np.random.default_rng(seed)
    x = distinct(rng, *shape) if name in ("max", "min") else leaf(rng, *shape)
    if name == "var":
        axes = range(x.ndim) if axis is None else ([axis] if isinstance(axis, int) else axis)
        if int(np.prod([shape[a] for a in axes])) < 2:
            return  # unbiased variance of a single element is undefined
        fn = lambda t: t.var(axis=axis, keepdims=keepdims)
    else:
        fn = lambda t: getattr(t, name)(axis=axis, keepdims=keepdims)
    assert gradcheck(fn, [x])


def test_var_population_correction(rng: np.random.Generator) -> None:
    assert gradcheck(lambda t: t.var(axis=1, correction=0), [leaf(rng, 3, 5)])


def test_max_splits_gradient_between_ties() -> None:
    x = tg.Tensor(np.array([[1.0, 3.0, 3.0], [2.0, 0.0, 1.0]]), requires_grad=True)
    x.max(axis=1).sum().backward()
    np.testing.assert_allclose(x.grad, [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])


# --------------------------------------------------------------------------- matmul
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        ((3, 4), (4, 2)),  # plain matrices
        ((2, 3, 4), (2, 4, 5)),  # batched
        ((2, 1, 3, 4), (5, 4, 2)),  # batch dims broadcast against each other
        ((3, 4), (2, 4, 5)),  # matrix against a batch
        ((4,), (4, 3)),  # vector @ matrix
        ((3, 4), (4,)),  # matrix @ vector
        ((4,), (4,)),  # dot product
        ((4,), (2, 4, 3)),  # vector @ batch
        ((2, 3, 4), (4,)),  # batch @ vector
    ],
)
def test_matmul(
    rng: np.random.Generator, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
) -> None:
    a, b = leaf(rng, *shape_a), leaf(rng, *shape_b)
    np.testing.assert_allclose((a @ b).data, a.data @ b.data)
    assert gradcheck(lambda x, y: x @ y, [a, b])


# --------------------------------------------------------------------------- shape ops
@given(shape=hnp.array_shapes(min_dims=1, max_dims=4, max_side=3), seed=seeds, data=st.data())
def test_reshape_and_permute(shape: tuple[int, ...], seed: int, data: st.DataObject) -> None:
    x = leaf(np.random.default_rng(seed), *shape)
    dims = data.draw(st.permutations(range(len(shape))))
    assert gradcheck(lambda t: t.permute(*dims).reshape(-1) * np.arange(t.size), [x])


def test_transpose_flatten_squeeze_unsqueeze(rng: np.random.Generator) -> None:
    x = leaf(rng, 2, 1, 3)
    assert gradcheck(lambda t: t.transpose(0, 2), [x])
    assert gradcheck(lambda t: t.T, [x])
    assert gradcheck(lambda t: t.flatten(1), [x])
    assert gradcheck(lambda t: t.squeeze(1), [x])
    assert gradcheck(lambda t: t.squeeze(), [x])
    assert gradcheck(lambda t: t.unsqueeze(-1), [x])


@pytest.mark.parametrize(
    "index",
    [
        (1,),
        (slice(1, None), slice(None, None, 2)),
        (Ellipsis, 0),
        (None, slice(None), 1),
        (np.array([0, 2, 2, 0]),),  # repeated rows: backward must scatter-add
        (slice(None), np.array([[0, 1], [1, 1]])),
        (np.array([0, 1]), np.array([2, 2])),
        (np.array([True, False, True]),),
        ([2, 0],),
    ],
    ids=repr,
)
def test_getitem(rng: np.random.Generator, index: tuple[object, ...]) -> None:
    x = leaf(rng, 3, 4)
    idx = index[0] if len(index) == 1 else index
    np.testing.assert_allclose(
        x[idx].data, x.data[idx if not isinstance(idx, list) else np.array(idx)]
    )  # type: ignore[index]
    assert gradcheck(lambda t: t[idx], [x])  # type: ignore[index]


def test_getitem_repeated_index_accumulates() -> None:
    x = tg.Tensor(np.zeros(3), requires_grad=True)
    x[np.array([0, 0, 0, 2])].sum().backward()
    np.testing.assert_allclose(x.grad, [3.0, 0.0, 1.0])


def test_getitem_with_boolean_tensor_mask(rng: np.random.Generator) -> None:
    x = leaf(rng, 3, 4)
    assert gradcheck(lambda t: t[t > 0], [x])


@given(seed=seeds, axis=st.integers(-2, 1))
def test_concat(seed: int, axis: int) -> None:
    rng = np.random.default_rng(seed)
    shapes = [[2, 3], [2, 3], [2, 3]]
    for k, s in enumerate(shapes):
        s[axis] += k  # different sizes along the concatenation axis
    inputs = [leaf(rng, *s) for s in shapes]
    assert gradcheck(lambda *ts: tg.concat(ts, axis=axis), inputs)


@given(seed=seeds, axis=st.integers(-3, 2))
def test_stack(seed: int, axis: int) -> None:
    rng = np.random.default_rng(seed)
    inputs = [leaf(rng, 2, 3) for _ in range(3)]
    assert gradcheck(lambda *ts: tg.stack(ts, axis=axis), inputs)


def test_concat_with_constant_input(rng: np.random.Generator) -> None:
    const = tg.Tensor(rng.standard_normal((2, 2)))
    assert gradcheck(lambda t: tg.concat([const, t, const], axis=1), [leaf(rng, 2, 3)])


@given(shapes=hnp.mutually_broadcastable_shapes(num_shapes=3, max_dims=3, max_side=3), seed=seeds)
def test_where_with_broadcasting(shapes: hnp.BroadcastableShapes, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cond = rng.random(shapes.input_shapes[0]) > 0.5
    a, b = leaf(rng, *shapes.input_shapes[1]), leaf(rng, *shapes.input_shapes[2])
    assert gradcheck(lambda x, y: tg.where(cond, x, y), [a, b])


def test_masked_fill(rng: np.random.Generator) -> None:
    mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
    x = leaf(rng, 2, 4, 4)
    assert gradcheck(lambda t: t.masked_fill(mask, -3.0), [x])
    out = x.masked_fill(mask, -np.inf)
    assert np.all(np.isneginf(out.data[:, mask]))


# --------------------------------------------------------------------------- nn ops
@given(shape=hnp.array_shapes(min_dims=1, max_dims=3, max_side=4), seed=seeds, data=st.data())
def test_softmax_family(shape: tuple[int, ...], seed: int, data: st.DataObject) -> None:
    axis = data.draw(st.integers(-len(shape), len(shape) - 1))
    x = leaf(np.random.default_rng(seed), *shape)
    assert gradcheck(lambda t: tg.softmax(t, axis=axis), [x])
    assert gradcheck(lambda t: tg.log_softmax(t, axis=axis), [x])
    assert gradcheck(lambda t: tg.logsumexp(t, axis=axis), [x])
    assert gradcheck(lambda t: tg.logsumexp(t, axis=axis, keepdims=True), [x])


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy(rng: np.random.Generator, reduction: str) -> None:
    logits = leaf(rng, 6, 5)
    target = np.array([0, 4, -100, 2, 2, -100])
    assert gradcheck(
        lambda t: tg.cross_entropy(t, target, reduction=reduction),
        [logits],
    )


def test_cross_entropy_sequence_logits(rng: np.random.Generator) -> None:
    logits = leaf(rng, 2, 3, 7)  # (batch, time, vocab)
    target = rng.integers(0, 7, size=(2, 3))
    assert gradcheck(lambda t: tg.cross_entropy(t, target), [logits])


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_loss(rng: np.random.Generator, reduction: str) -> None:
    pred, target = leaf(rng, 4, 3), leaf(rng, 4, 3)
    assert gradcheck(
        lambda p, t: tg.mse_loss(p, t, reduction=reduction),
        [pred, target],
    )


@pytest.mark.parametrize(
    ("x_shape", "w_shape", "stride", "padding", "bias"),
    [
        ((2, 3, 5, 5), (4, 3, 3, 3), 1, 0, True),
        ((1, 2, 6, 5), (3, 2, 3, 3), 2, 1, True),
        ((2, 1, 5, 6), (2, 1, 2, 3), (1, 2), (0, 1), False),
        ((1, 2, 4, 4), (2, 2, 1, 1), 1, 0, True),  # 1x1 convolution
        ((1, 1, 7, 7), (1, 1, 3, 3), 3, 2, False),  # stride larger than kernel overlap
    ],
)
def test_conv2d(
    rng: np.random.Generator,
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    bias: bool,
) -> None:
    x, w = leaf(rng, *x_shape), leaf(rng, *w_shape)
    inputs = [x, w] + ([leaf(rng, w_shape[0])] if bias else [])
    assert gradcheck(lambda *ts: tg.conv2d(*ts, stride=stride, padding=padding), inputs)


@pytest.mark.parametrize(
    ("kernel", "stride", "padding"), [(2, None, 0), (3, 1, 1), (2, 1, 0), ((2, 3), (2, 1), 0)]
)
def test_max_pool2d(
    rng: np.random.Generator,
    kernel: int | tuple[int, int],
    stride: int | tuple[int, int] | None,
    padding: int,
) -> None:
    x = distinct(rng, 2, 2, 4, 5)
    assert gradcheck(lambda t: tg.max_pool2d(t, kernel, stride, padding), [x])


def test_embedding_scatter_adds_repeated_ids(rng: np.random.Generator) -> None:
    weight = leaf(rng, 5, 3)
    ids = np.array([[0, 3, 3], [4, 3, 0]])
    assert gradcheck(lambda w: tg.embedding(ids, w), [weight])


@pytest.mark.parametrize("bias", [True, False])
def test_linear(rng: np.random.Generator, bias: bool) -> None:
    inputs = [leaf(rng, 2, 3, 4), leaf(rng, 5, 4)] + ([leaf(rng, 5)] if bias else [])
    assert gradcheck(lambda *ts: tg.linear(*ts), inputs)


@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm(rng: np.random.Generator, affine: bool) -> None:
    x = leaf(rng, 2, 3, 6)
    if affine:
        assert gradcheck(tg.layer_norm, [x, leaf(rng, 6), leaf(rng, 6)])
    else:
        assert gradcheck(tg.layer_norm, [x])


@pytest.mark.parametrize("shape", [(6, 3), (4, 3, 5)])
def test_batch_norm_training(rng: np.random.Generator, shape: tuple[int, ...]) -> None:
    x, w, b = leaf(rng, *shape), leaf(rng, 3), leaf(rng, 3)
    fn = lambda *ts: tg.ops.batch_norm(ts[0], None, None, ts[1], ts[2], training=True)
    assert gradcheck(fn, [x, w, b])


def test_dropout_with_fixed_mask(rng: np.random.Generator) -> None:
    x = leaf(rng, 4, 5)
    # A fresh generator with the same seed on every call keeps the mask fixed.
    fn = lambda t: tg.dropout(t, 0.3, rng=np.random.default_rng(7))
    assert gradcheck(fn, [x])


@pytest.mark.parametrize("causal", [False, True])
def test_scaled_dot_product_attention(rng: np.random.Generator, causal: bool) -> None:
    q, k, v = leaf(rng, 2, 4, 3), leaf(rng, 2, 4, 3), leaf(rng, 2, 4, 2)
    fn = lambda *ts: scaled_dot_product_attention(*ts, causal=causal)
    assert gradcheck(fn, [q, k, v])
