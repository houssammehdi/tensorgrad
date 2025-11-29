"""Every differentiable op against PyTorch: values, gradients and second derivatives.

Skipped when PyTorch is not installed (it is not a dependency); CI runs these in a separate
job with the CPU build of torch. All comparisons are in float64.
"""

from __future__ import annotations

from collections.abc import Callable

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")
F = torch.nn.functional

import tensorgrad as tg  # noqa: E402
from parity import compare  # noqa: E402
from tensorgrad.nn.functional import scaled_dot_product_attention  # noqa: E402
from tensorgrad.ops.conv import conv2d_input_grad, conv2d_weight_grad  # noqa: E402

pytestmark = pytest.mark.torch
seeds = st.integers(min_value=0, max_value=2**32 - 1)
few = settings(max_examples=15, deadline=None)


def normal(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return np.asarray(rng.standard_normal(shape))


def positive(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return np.abs(normal(rng, *shape)) + 0.5


def distinct(rng: np.random.Generator, *shape: int) -> np.ndarray:
    n = int(np.prod(shape))
    return rng.permutation(n).reshape(shape) * 0.1 + rng.uniform(-0.01, 0.01, size=shape)


# ---------------------------------------------------------------------- elementwise
UNARY: dict[str, tuple[Callable[..., tg.Tensor], Callable[..., torch.Tensor], str]] = {
    "neg": (lambda x: -x, lambda x: -x, "normal"),
    "exp": (tg.exp, torch.exp, "normal"),
    "log": (tg.log, torch.log, "positive"),
    "sqrt": (tg.sqrt, torch.sqrt, "positive"),
    "tanh": (tg.tanh, torch.tanh, "normal"),
    "sigmoid": (tg.sigmoid, torch.sigmoid, "normal"),
    "relu": (tg.relu, torch.relu, "away"),
    "gelu": (tg.gelu, lambda x: F.gelu(x, approximate="tanh"), "normal"),
}


@pytest.mark.parametrize("name", sorted(UNARY))
@few
@given(shape=hnp.array_shapes(min_dims=0, max_dims=3, max_side=4), seed=seeds)
def test_unary(name: str, shape: tuple[int, ...], seed: int) -> None:
    ours, theirs, kind = UNARY[name]
    rng = np.random.default_rng(seed)
    x = positive(rng, *shape) if kind == "positive" else normal(rng, *shape)
    if kind == "away":
        x = x + np.where(x >= 0, 0.1, -0.1)
    compare(ours, theirs, [x], seed=seed)


BINARY: dict[str, tuple[Callable[..., tg.Tensor], Callable[..., torch.Tensor]]] = {
    "add": (lambda a, b: a + b, lambda a, b: a + b),
    "sub": (lambda a, b: a - b, lambda a, b: a - b),
    "mul": (lambda a, b: a * b, lambda a, b: a * b),
    "div": (lambda a, b: a / b, lambda a, b: a / b),
    "pow": (lambda a, b: a**b, lambda a, b: a**b),
}


@pytest.mark.parametrize("name", sorted(BINARY))
@few
@given(shapes=hnp.mutually_broadcastable_shapes(num_shapes=2, max_dims=3, max_side=3), seed=seeds)
def test_binary_with_broadcasting(name: str, shapes: hnp.BroadcastableShapes, seed: int) -> None:
    ours, theirs = BINARY[name]
    rng = np.random.default_rng(seed)
    a_shape, b_shape = shapes.input_shapes
    a = positive(rng, *a_shape) if name == "pow" else normal(rng, *a_shape)
    b = positive(rng, *b_shape) if name == "div" else normal(rng, *b_shape)
    compare(ours, theirs, [a, b], seed=seed)


@pytest.mark.parametrize("exponent", [2, 3, 0.5, -1.5])
def test_pow_with_a_scalar_exponent(exponent: float) -> None:
    x = positive(np.random.default_rng(0), 3, 4)
    compare(lambda t: t**exponent, lambda t: t**exponent, [x])


def test_astype_round_trip() -> None:
    x = normal(np.random.default_rng(0), 5)
    compare(
        lambda t: t.astype(np.float32).astype(np.float64) * 3,
        lambda t: t.to(torch.float32).to(torch.float64) * 3,
        [x],
        rtol=1e-6,
        atol=1e-6,
    )


# ---------------------------------------------------------------------- reductions
@st.composite
def shape_and_axis(draw: st.DrawFn) -> tuple[tuple[int, ...], int | tuple[int, ...] | None]:
    shape = draw(hnp.array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=4))
    axis = draw(
        st.one_of(
            st.none(),
            st.integers(-len(shape), len(shape) - 1),
            st.lists(st.integers(0, len(shape) - 1), min_size=1, unique=True).map(tuple),
        )
    )
    return shape, axis


def torch_dims(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    return (axis,) if isinstance(axis, int) else axis


@pytest.mark.parametrize("name", ["sum", "mean", "max", "min", "var0", "var1"])
@few
@given(spec=shape_and_axis(), keepdims=st.booleans(), seed=seeds)
def test_reductions(
    name: str, spec: tuple[tuple[int, ...], int | tuple[int, ...] | None], keepdims: bool, seed: int
) -> None:
    shape, axis = spec
    rng = np.random.default_rng(seed)
    x = distinct(rng, *shape) if name in ("max", "min") else normal(rng, *shape)
    dims = torch_dims(axis, len(shape))
    if name.startswith("var"):
        c = int(name[-1])
        ours = lambda t: t.var(axis=axis, keepdims=keepdims, correction=c)
        theirs = lambda t: torch.var(t, dim=dims, keepdim=keepdims, correction=c)
    else:
        ours = lambda t: getattr(t, name)(axis=axis, keepdims=keepdims)
        # Our max/min split the gradient between ties like torch.amax/amin (all equal here).
        reducer = {"sum": torch.sum, "mean": torch.mean, "max": torch.amax, "min": torch.amin}
        theirs = lambda t: reducer[name](t, dim=dims, keepdim=keepdims)
    compare(ours, theirs, [x], seed=seed)


def test_max_with_ties_splits_the_gradient_like_amax() -> None:
    x = np.array([[1.0, 3.0, 3.0], [2.0, 2.0, 0.0]])
    compare(lambda t: t.max(axis=1), lambda t: torch.amax(t, dim=1), [x], second_order=False)


# ---------------------------------------------------------------------- linear algebra
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        ((3, 4), (4, 2)),
        ((2, 3, 4), (2, 4, 5)),
        ((2, 1, 3, 4), (5, 4, 2)),
        ((3, 4), (2, 4, 5)),
        ((4,), (4, 3)),
        ((3, 4), (4,)),
        ((4,), (4,)),
        ((4,), (2, 4, 3)),
        ((2, 3, 4), (4,)),
    ],
)
def test_matmul(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    compare(lambda a, b: a @ b, lambda a, b: a @ b, [normal(rng, *shape_a), normal(rng, *shape_b)])


# ---------------------------------------------------------------------- shape ops
SHAPE_OPS: dict[str, tuple[Callable[..., tg.Tensor], Callable[..., torch.Tensor]]] = {
    "reshape": (lambda t: t.reshape(4, 6), lambda t: t.reshape(4, 6)),
    "permute": (lambda t: t.permute(2, 0, 1), lambda t: t.permute(2, 0, 1)),
    "transpose": (lambda t: t.transpose(0, 2), lambda t: t.transpose(0, 2)),
    "T": (lambda t: t.T, lambda t: t.permute(2, 1, 0)),
    "flatten": (lambda t: t.flatten(1), lambda t: t.flatten(1)),
    "unsqueeze": (lambda t: t.unsqueeze(1), lambda t: t.unsqueeze(1)),
    "squeeze": (lambda t: t.reshape(2, 1, 12).squeeze(1), lambda t: t.reshape(2, 1, 12).squeeze(1)),
    "broadcast_to": (
        lambda t: tg.broadcast_to(t[:, :1], (5, 2, 3, 4)),
        lambda t: torch.broadcast_to(t[:, :1], (5, 2, 3, 4)),
    ),
    "sum_to": (lambda t: tg.ops.sum_to(t, (3, 1)), lambda t: t.sum_to_size(3, 1)),
}


@pytest.mark.parametrize("name", sorted(SHAPE_OPS))
def test_shape_ops(name: str) -> None:
    ours, theirs = SHAPE_OPS[name]
    compare(ours, theirs, [normal(np.random.default_rng(0), 2, 3, 4)])


@pytest.mark.parametrize(
    "index",
    [
        (1,),
        (slice(1, None), slice(None, None, 2)),
        (Ellipsis, 0),
        (None, slice(None), 1),
        (np.array([0, 2, 2, 0]),),  # repeated rows: gradients must accumulate
        (slice(None), np.array([[0, 1], [1, 1]])),
        (np.array([0, 1]), np.array([2, 2])),
        (np.array([True, False, True]),),
    ],
    ids=repr,
)
def test_indexing(index: tuple[object, ...]) -> None:
    idx = index[0] if len(index) == 1 else index

    def torch_index(value: object) -> object:
        return torch.from_numpy(value) if isinstance(value, np.ndarray) else value

    t_idx = tuple(torch_index(i) for i in idx) if isinstance(idx, tuple) else torch_index(idx)
    compare(lambda t: t[idx], lambda t: t[t_idx], [normal(np.random.default_rng(0), 3, 4)])


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_concat_and_stack(axis: int) -> None:
    rng = np.random.default_rng(axis + 5)
    sizes = [[2, 3], [2, 3], [2, 3]]
    for k, s in enumerate(sizes):
        s[axis] += k
    parts = [normal(rng, *s) for s in sizes]
    compare(lambda *ts: tg.concat(ts, axis=axis), lambda *ts: torch.cat(ts, dim=axis), parts)
    same = [normal(rng, 2, 3) for _ in range(3)]
    compare(lambda *ts: tg.stack(ts, axis=axis), lambda *ts: torch.stack(ts, dim=axis), same)


def test_where_and_masked_fill() -> None:
    rng = np.random.default_rng(3)
    cond = rng.random((3, 1, 4)) > 0.5
    a, b = normal(rng, 2, 4), normal(rng, 3, 2, 1)
    compare(
        lambda x, y: tg.where(cond, x, y),
        lambda x, y: torch.where(torch.from_numpy(cond), x, y),
        [a, b],
    )
    mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
    compare(
        lambda x: x.masked_fill(mask, -3.0),
        lambda x: x.masked_fill(torch.from_numpy(mask), -3.0),
        [normal(rng, 2, 4, 4)],
    )


# ---------------------------------------------------------------------- softmax family, losses
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_family(axis: int) -> None:
    x = normal(np.random.default_rng(axis + 10), 3, 4, 5)
    compare(lambda t: tg.softmax(t, axis), lambda t: torch.softmax(t, axis), [x])
    compare(lambda t: tg.log_softmax(t, axis), lambda t: torch.log_softmax(t, axis), [x])
    compare(lambda t: tg.logsumexp(t, axis), lambda t: torch.logsumexp(t, axis), [x])
    compare(
        lambda t: tg.logsumexp(t, axis, keepdims=True),
        lambda t: torch.logsumexp(t, axis, keepdim=True),
        [x],
    )


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy(reduction: str) -> None:
    rng = np.random.default_rng(4)
    target = np.array([0, 4, -100, 2, 2, -100])
    compare(
        lambda z: tg.cross_entropy(z, target, reduction=reduction),
        lambda z: F.cross_entropy(z, torch.from_numpy(target), reduction=reduction),
        [normal(rng, 6, 5)],
    )


def test_cross_entropy_on_sequences_puts_the_class_axis_last() -> None:
    # tensorgrad: (batch, time, classes); PyTorch: (batch, classes, time).
    rng = np.random.default_rng(5)
    target = rng.integers(0, 7, size=(2, 3))
    compare(
        lambda z: tg.cross_entropy(z, target),
        lambda z: F.cross_entropy(z.permute(0, 2, 1), torch.from_numpy(target)),
        [normal(rng, 2, 3, 7)],
    )


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_binary_cross_entropy_with_logits(reduction: str) -> None:
    rng = np.random.default_rng(6)
    targets = 1 / (1 + np.exp(-normal(rng, 4, 3)))
    compare(
        lambda z, y: tg.binary_cross_entropy_with_logits(z, y, reduction=reduction),
        lambda z, y: F.binary_cross_entropy_with_logits(z, y, reduction=reduction),
        [normal(rng, 4, 3) * 3, targets],
    )


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_loss(reduction: str) -> None:
    rng = np.random.default_rng(7)
    compare(
        lambda p, t: tg.mse_loss(p, t, reduction=reduction),
        lambda p, t: F.mse_loss(p, t, reduction=reduction),
        [normal(rng, 4, 3), normal(rng, 4, 3)],
    )


# ---------------------------------------------------------------------- convolution, pooling
CONV_CASES = [
    ((2, 3, 5, 5), (4, 3, 3, 3), 1, 0, True),
    ((1, 2, 6, 5), (3, 2, 3, 3), 2, 1, True),
    ((2, 1, 5, 6), (2, 1, 2, 3), (1, 2), (0, 1), False),
    ((1, 2, 4, 4), (2, 2, 1, 1), 1, 0, True),
    ((1, 1, 7, 7), (1, 1, 3, 3), 3, 1, False),
]


@pytest.mark.parametrize(("x_shape", "w_shape", "stride", "padding", "bias"), CONV_CASES)
def test_conv2d(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    bias: bool,
) -> None:
    rng = np.random.default_rng(8)
    arrays = [normal(rng, *x_shape), normal(rng, *w_shape)] + (
        [normal(rng, w_shape[0])] if bias else []
    )
    compare(
        lambda *ts: tg.conv2d(*ts, stride=stride, padding=padding),
        lambda *ts: F.conv2d(*ts, stride=stride, padding=padding),
        arrays,
    )


@pytest.mark.parametrize(("x_shape", "w_shape", "stride", "padding", "bias"), CONV_CASES[:3])
def test_convolution_gradient_ops(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, ...],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    bias: bool,
) -> None:
    rng = np.random.default_rng(9)
    out_shape = F.conv2d(
        torch.zeros(x_shape), torch.zeros(w_shape), stride=stride, padding=padding
    ).shape
    g, w, x = normal(rng, *out_shape), normal(rng, *w_shape), normal(rng, *x_shape)
    compare(
        lambda gy, ww: conv2d_input_grad(gy, ww, x_shape, stride, padding),
        lambda gy, ww: torch.nn.grad.conv2d_input(x_shape, ww, gy, stride=stride, padding=padding),
        [g, w],
    )
    compare(
        lambda xx, gy: conv2d_weight_grad(xx, gy, w_shape, stride, padding),
        lambda xx, gy: torch.nn.grad.conv2d_weight(xx, w_shape, gy, stride=stride, padding=padding),
        [x, g],
    )


@pytest.mark.parametrize(
    ("kernel", "stride", "padding"), [(2, None, 0), (3, 1, 1), (2, 1, 0), ((2, 3), (2, 1), (1, 1))]
)
def test_max_pool2d(
    kernel: int | tuple[int, int],
    stride: int | tuple[int, int] | None,
    padding: int | tuple[int, int],
) -> None:
    x = distinct(np.random.default_rng(10), 2, 2, 5, 6)
    compare(
        lambda t: tg.max_pool2d(t, kernel, stride, padding),
        lambda t: F.max_pool2d(t, kernel, stride, padding),
        [x],
    )


# ---------------------------------------------------------------------- fused layers
@pytest.mark.parametrize("bias", [True, False])
def test_linear(bias: bool) -> None:
    rng = np.random.default_rng(11)
    arrays = [normal(rng, 2, 3, 4), normal(rng, 5, 4)] + ([normal(rng, 5)] if bias else [])
    compare(lambda *ts: tg.linear(*ts), lambda *ts: F.linear(*ts), arrays)


@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm(affine: bool) -> None:
    rng = np.random.default_rng(12)
    if affine:
        compare(
            lambda x, w, b: tg.layer_norm(x, w, b),
            lambda x, w, b: F.layer_norm(x, (6,), w, b),
            [normal(rng, 2, 3, 6), normal(rng, 6), normal(rng, 6)],
        )
    else:
        compare(tg.layer_norm, lambda x: F.layer_norm(x, (6,)), [normal(rng, 2, 3, 6)])


def test_embedding() -> None:
    ids = np.array([[0, 3, 3], [4, 3, 0]])
    compare(
        lambda w: tg.embedding(ids, w),
        lambda w: F.embedding(torch.from_numpy(ids), w),
        [normal(np.random.default_rng(13), 5, 3)],
    )


@pytest.mark.parametrize("shape", [(6, 3), (4, 3, 5)])
def test_batch_norm_training_statistics_and_output(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(14)
    x, w, b = normal(rng, *shape), normal(rng, 3), normal(rng, 3)
    ours_mean, ours_var = np.zeros(3), np.ones(3)
    theirs_mean, theirs_var = (
        torch.zeros(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
    )
    compare(
        lambda *ts: tg.ops.batch_norm(ts[0], ours_mean, ours_var, ts[1], ts[2], training=True),
        lambda *ts: F.batch_norm(ts[0], theirs_mean, theirs_var, ts[1], ts[2], training=True),
        [x, w, b],
    )
    # compare() ran both forward passes exactly once: the running estimates must agree too.
    np.testing.assert_allclose(ours_mean, theirs_mean.numpy(), rtol=1e-12)
    np.testing.assert_allclose(ours_var, theirs_var.numpy(), rtol=1e-12)
    compare(
        lambda *ts: tg.ops.batch_norm(ts[0], ours_mean, ours_var, ts[1], ts[2], training=False),
        lambda *ts: F.batch_norm(ts[0], theirs_mean, theirs_var, ts[1], ts[2], training=False),
        [x, w, b],
    )


@pytest.mark.parametrize("causal", [False, True])
def test_scaled_dot_product_attention(causal: bool) -> None:
    rng = np.random.default_rng(15)
    q, k, v = normal(rng, 2, 2, 4, 3), normal(rng, 2, 2, 4, 3), normal(rng, 2, 2, 4, 2)
    compare(
        lambda *ts: scaled_dot_product_attention(*ts, causal=causal),
        lambda *ts: F.scaled_dot_product_attention(*ts, is_causal=causal),
        [q, k, v],
    )


def test_scaled_dot_product_attention_with_an_explicit_mask() -> None:
    rng = np.random.default_rng(16)
    q, k, v = normal(rng, 2, 3, 4), normal(rng, 2, 5, 4), normal(rng, 2, 5, 2)
    forbidden = rng.random((3, 5)) > 0.6
    forbidden[:, 0] = False  # keep at least one visible key per query
    # tensorgrad masks mark *forbidden* positions; PyTorch boolean masks mark allowed ones.
    compare(
        lambda *ts: scaled_dot_product_attention(*ts, mask=forbidden),
        lambda *ts: F.scaled_dot_product_attention(*ts, attn_mask=torch.from_numpy(~forbidden)),
        [q, k, v],
    )


# ---------------------------------------------------------------------- deliberate differences
def test_documented_differences_from_pytorch() -> None:
    """Each divergence below is intentional and documented in docs/parity.md."""
    # 1. A mean cross-entropy over targets that are all ignored is 0, not NaN, so a batch
    #    without valid labels cannot poison the parameters with NaN gradients.
    logits = np.zeros((2, 3))
    ignored = np.array([-100, -100])
    ours = tg.cross_entropy(tg.Tensor(logits), ignored).item()
    theirs = F.cross_entropy(torch.tensor(logits), torch.from_numpy(ignored)).item()
    assert ours == 0.0 and np.isnan(theirs)

    # 2. d(base ** e)/de needs log(base); for base <= 0 tensorgrad returns 0, PyTorch NaN
    #    (for base < 0; both return 0 for base == 0 with a positive exponent).
    base = np.array([-2.0, 0.0, 2.0])
    e_ours = tg.Tensor(np.array([3.0, 2.0, 2.0]), requires_grad=True)
    e_theirs = torch.tensor([3.0, 2.0, 2.0], dtype=torch.float64, requires_grad=True)
    (tg.Tensor(base) ** e_ours).sum().backward()
    (torch.tensor(base) ** e_theirs).sum().backward()
    assert e_ours.grad is not None and e_theirs.grad is not None
    assert e_ours.grad.data[0] == 0.0 and np.isnan(e_theirs.grad[0].item())
    np.testing.assert_allclose(e_ours.grad.data[1:], e_theirs.grad[1:].numpy())

    # 3. max/min over an axis split the gradient between ties like torch.amax/amin;
    #    torch.max(dim=...) gives all of it to a single index.
    x = np.array([[1.0, 3.0, 3.0]])
    t_ours, t_theirs = tg.Tensor(x, requires_grad=True), torch.tensor(x, requires_grad=True)
    t_ours.max(axis=1).sum().backward()
    t_theirs.max(dim=1).values.sum().backward()
    assert t_ours.grad is not None and t_theirs.grad is not None
    assert t_ours.grad.tolist() == [[0.0, 0.5, 0.5]]
    assert t_theirs.grad.tolist() == [[0.0, 1.0, 0.0]]


def test_attention_mask_convention_is_forbidden_positions() -> None:
    # tensorgrad and torch.nn.MultiheadAttention: True = may *not* attend.
    # torch.nn.functional.scaled_dot_product_attention: True = may attend.
    rng = np.random.default_rng(17)
    q, k, v = (torch.tensor(normal(rng, 1, 2, 3)) for _ in range(3))
    forbidden = np.array([[False, True], [False, False]])
    ours = scaled_dot_product_attention(
        tg.Tensor(q.numpy()), tg.Tensor(k.numpy()), tg.Tensor(v.numpy()), mask=forbidden
    )
    theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.from_numpy(~forbidden))
    np.testing.assert_allclose(ours.data, theirs.numpy(), rtol=1e-12)
