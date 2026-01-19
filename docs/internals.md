# tensorgrad internals

How the engine works, how higher-order derivatives are built on it, and why the code looks the
way it does. The [README](../README.md) is the front page; [performance.md](performance.md)
has the measurements and [parity.md](parity.md) the comparison with PyTorch.

- [1. Tensors, nodes and the backward pass](#1-tensors-nodes-and-the-backward-pass)
- [2. Two VJPs per op](#2-two-vjps-per-op)
- [3. Higher-order derivatives](#3-higher-order-derivatives)
- [4. Function transforms](#4-function-transforms)
- [5. Fused ops](#5-fused-ops)
- [6. Memory](#6-memory)
- [7. Numerical stability](#7-numerical-stability)
- [8. Tooling: profiler and graph drawing](#8-tooling-profiler-and-graph-drawing)
- [9. How the derivatives are verified](#9-how-the-derivatives-are-verified)

## 1. Tensors, nodes and the backward pass

A `Tensor` is a NumPy array (`.data`) plus four slots: `_requires_grad`, `_grad`, `_node` (the
op that produced it, `None` for leaves) and `_retains_grad`. Every op computes its output
with NumPy and passes it to `ops._util.make_result`, which attaches a `Node` if grad mode is
on and any input requires grad. A node stores the op name, the parent tensors and two
closures, described in the next section. Nothing else is recorded, so the graph is exactly
the computation that ran, Python control flow included.

`loss.backward()` (and `autograd.grad`, which shares the engine) then:

1. **Sorts** every tensor the outputs depend on, among those that require grad, with an
   iterative post-order DFS (no recursion, so a 5,000-node chain is fine; the tests build
   one).
2. **Seeds** d(loss)/d(loss) = 1 and walks the order in reverse. Each tensor's gradient is
   the sum of what its consumers sent it; its node's VJP turns that into one gradient per
   parent.
3. **Accumulates** into `.grad` only at leaves, and at non-leaves that called
   `retain_grad()`, adding to what is already there. That is why optimisers call
   `zero_grad()`. `autograd.grad` instead returns the totals for the requested inputs and
   touches no `.grad`.
4. **Frees** each node once used: the closures, and the activations they captured, are
   dropped and the parent references cleared, unless `retain_graph=True`. A second backward
   through a freed node raises an error naming the op.

Worked example, `L = (x * y + x) ** 2` at `x = 2`, `y = 3`:

| Forward | Value | Backward (VJP) | Gradient |
|---|---|---|---|
| `a = x * y` | 6 | `dL/da = dL/db` | 16 |
| `b = a + x` | 8 | `dL/db = 2b` | 16 |
| `L = b ** 2` | 64 | seed | 1 |
| | | `dL/dx = dL/da * y + dL/db` (two paths, summed) | 16·3 + 16 = **64** |
| | | `dL/dy = dL/da * x` | 16·2 = **32** |

**Broadcasting** is the subtle part of elementwise ops. If `w` of shape `(1, 3)` meets an
operand of shape `(4, 3)`, its gradient must be summed back over the stretched axis:
`unbroadcast(grad, shape)` sums over prepended axes and over axes that were size 1.

**Weight tying** needs nothing special: the GPT head and the token embedding are one
`Parameter` used twice, and the engine sums both contributions like any other fan-out.

## 2. Two VJPs per op

Every kind of graph node, the 48 ops of `tensorgrad.ops` and the identity and checkpoint
nodes of `func` and `utils.checkpoint`, supplies its vector-Jacobian product twice:

```python
def backward(g: Array, need: tuple[bool, ...]) -> tuple[Array | None, ...]: ...
def graph(g: Tensor, out: Tensor, need: tuple[bool, ...]) -> tuple[Tensor | None, ...]: ...
```

- `backward` works on NumPy arrays. It is all that an ordinary `loss.backward()` runs, so the
  first-order path never builds graph nodes and can use in-place kernels, reuse forward
  buffers and skip work.
- `graph` computes the same VJP with differentiable tensor ops. The engine calls it instead
  when asked to `create_graph=True`, with grad mode on, so the gradients it returns have a
  graph of their own and can be differentiated again.

Keeping both costs code, and it is tested (section 9), but it means higher-order support
costs nothing when it is not used: the spiral MLP step is as fast as in v0.1.0 or faster
(see [performance.md](performance.md)).

**Needs-input-grad masks.** Both closures receive `need`, one flag per parent: whether that
parent requires grad and, for `autograd.grad`, whether it lies on a path to one of the
requested inputs. The engine computes the second part once per call (`_leads_to`, a
forward sweep over the sorted tensors) and skips nodes whose mask is all false. Ops use the
mask to skip work: `linear` does not compute the input gradient of the first layer (the
data), attention skips `dQ`/`dK` when only `dV` is needed, and a recurrent layer skips the
weight GEMMs of frozen weights.

**`.grad` is a `Tensor`.** An ordinary backward stores a constant tensor wrapping the
accumulated array. Under `create_graph=True` the accumulated value is the gradient tensor
itself, graph included, so `x.grad` can be backpropagated again, as in PyTorch.

## 3. Higher-order derivatives

`backward(create_graph=True)` and `autograd.grad(..., create_graph=True)` run the `graph`
VJPs. The result is an ordinary differentiable tensor: Hessian-vector products, gradient
penalties and meta-learning are all "differentiate the gradient again". `retain_graph`
defaults to `create_graph`, because the second pass usually reuses the forward graph.

For this to be exact, the graph VJP of each op must be built from ops that are themselves
differentiable to every order. Most are obvious (the VJP of `exp` is `g * exp(x)`, which
reuses the op's output tensor). The interesting cases are the linear ops, whose VJPs are
other linear ops, and which must therefore come in closed families:

| Op | VJP with respect to its input | VJP of that VJP |
|---|---|---|
| `broadcast_to(x, shape)` | `sum_to(g, x.shape)` | `broadcast_to` |
| `x[index]` (`getitem`) | `scatter_add(g, index, x.shape)` | `getitem` |
| `reshape`, `permute` | `reshape` back, the inverse permutation | the op itself |
| `concat`, `stack` | slices of `g` (`getitem`) | `scatter_add` |
| `conv2d(x, w)` | `conv2d_input_grad(g, w)` and `conv2d_weight_grad(x, g)` | each other (below) |

**Convolution as a trilinear form.** Write `T(x, w, y) = <conv2d(x, w), y>`. It is linear in
each argument, and the three partial derivatives are `conv2d(x, w)` (with respect to `y`),
`conv2d_input_grad(y, w)` (with respect to `x`) and `conv2d_weight_grad(x, y)` (with respect
to `w`). So each of the three ops is the VJP of the other two: the backward of
`conv2d_input_grad` needs only `conv2d` and `conv2d_weight_grad`, and so on. The three are
public differentiable ops sharing the im2col and col2im kernels, and double backward through
a CNN closes over them.

**Piecewise-linear ops** (`relu`, `max`, `max_pool2d`) have zero second derivative almost
everywhere. Their graph VJPs route `g` through a constant mask (`where(x > 0, g, 0)`, a
scatter to the pooling winners), which is differentiable in `g` and constant in `x`: exactly
right away from ties and kinks.

**Fused ops** (section 5) recompute what they need. The graph VJP of the fused attention op
rebuilds the softmax probabilities with differentiable ops and applies the same formulas as
the NumPy backward, and the recurrent ops rerun the forward recurrence with tensor ops and
then run backpropagation through time with tensor ops. An earlier design called
`autograd.grad` on the unfused composition from inside the VJP. It was correct, but every
nested call re-sorted the whole graph below it, which is quadratic in depth.

## 4. Function transforms

`tensorgrad.func` wraps the engine in JAX-style transforms: `grad`, `value_and_grad`, `vjp`
(which returns a reusable pullback), `jvp`, `jacobian`, `hessian` and `hvp`.
`hessian(f)` is literally `jacobian(grad(f))`. Two rules make them compose:

- **Results carry a graph exactly when grad mode is on.** Inside a transform the function
  always runs with grad enabled; the returned values are detached only if the caller is under
  `no_grad`. So `func.grad(func.grad(f))` works, and `with tg.no_grad(): func.hessian(f)(x)`
  returns plain numbers.
- **Each differentiated argument gets a fresh alias** (an identity op, or a new leaf if the
  argument has no graph), and derivatives are taken with respect to the alias. Without it,
  a nested transform that closes over the outer variable differentiates through both uses:
  the classic "perturbation confusion" of tape-based nesting. The test is
  `d/dx [x * d/dy (x + y)]` at any `x`, which is 1; a naive implementation returns 2.
  Removing the alias makes `tests/test_func.py` fail.

**Forward mode without forward mode.** `jvp(f, x, v)` uses the double-VJP trick. The VJP
`u -> J^T u` is linear in `u`, so differentiating `<J^T u, v>` with respect to `u` gives
`J v`: two reverse passes with a dummy `u` that requires grad, instead of a separate
forward-mode implementation. This is also why `gradgradcheck` differentiates the gradient
graph with respect to the upstream gradient (section 9). `hvp` is reverse-over-reverse:
`grad(<grad f(x), v>)`.

## 5. Fused ops

Several ops are one node instead of a chain of primitives, with a hand-derived NumPy backward
(and, per section 2, a differentiable graph VJP):

- **`linear`**: `x @ W.T + b` as one GEMM over the flattened leading axes (NumPy would
  otherwise loop over them), with the bias added in place.
- **`layer_norm`**: the forward pass works in place on two buffers; the backward pass is
  `rstd * (gh - mean(gh) - xhat * mean(gh * xhat))` with `gh = g * weight`.
- **Attention** (`ops/attention.py`): `softmax(q k^T / sqrt(d) + mask) v` in one node that
  keeps one `(T, T)` array per head and applies the mask and the softmax in place:
  `dV = P^T dO`, `dP = dO V^T`, `dS = P * (dP - rowsum(dP * P)) / sqrt(d)`, `dQ = dS K`,
  `dK = dS^T Q`. `packed_self_attention` takes the `(B, T, 3D)` output of the fused QKV
  projection, reads Q, K and V as strided views, and writes their gradients straight into one
  `(B, T, 3D)` array, so splitting and merging heads costs no nodes and no copies.
- **Recurrent layers** (`ops/recurrent.py`): `rnn`, `lstm` and `gru` each process a whole
  time-major sequence in one node. The forward pass projects all input steps with one GEMM
  and keeps the gate activations. The backward pass is explicit backpropagation through time:
  a reverse loop that carries the hidden-state gradient (and the LSTM's cell-state gradient)
  from step to step, followed by one GEMM per weight matrix over all steps. The LSTM returns
  a `(T, B, 2, H)` tensor holding both hidden and cell states, so a single-output node can
  expose `c_n` differentiably. Gate layouts and parameter names match `torch.nn`, and
  outputs, gradients and Hessian-vector products agree with PyTorch to about 1e-15.
- **`cross_entropy`** from logits: log-sum-exp in the forward pass and the fused gradient
  `softmax(z) - onehot(y)`, with `ignore_index`.
- **`embedding`** backward: a stable argsort of the ids followed by `np.add.reduceat` over
  runs of equal ids, about 7x faster than `np.add.at` for a character vocabulary.

`batch_norm`, `mse_loss`, `flatten` and `transpose` are compositions of primitives, which
shows that composition works as well; their derivatives of every order come from the graph.

## 6. Memory

- **Graph lifetime.** A node's closures capture exactly what its VJP needs (its inputs, or
  its output where that is cheaper, as for `exp`, `tanh` and `sigmoid`). Freeing the node
  after use drops them, so a training loop holds one forward pass of activations at a time.
  `no_grad()` records nothing and is used for evaluation and generation.
- **The allocator.** NumPy allocates array buffers with `malloc`. glibc serves large requests
  with `mmap` and returns freed memory at the top of the heap to the kernel, so a step that
  frees its activations (84 MB at the peak of the round-1 GPT step) page-faults them back in
  on the next step: about 11,000 to 13,000 minor faults and a fifth of a GPT step. `utils.retain_freed_memory()` raises glibc's
  mmap and trim thresholds with `mallopt` (both must move: raising only the trim threshold
  freezes the adaptive mmap threshold at 128 KB, which is worse than doing nothing). The
  training examples call it; it is opt-in because the setting is process-wide, and it returns
  `False` without doing anything on other C libraries.
- **Gradient checkpointing.** `utils.checkpoint(fn, *args)` runs `fn` under `no_grad`, so
  none of its intermediate activations are kept, and records one node whose parents are the
  arguments and the tensors `fn` closes over (a module's parameters, found automatically).
  The node's VJP reruns `fn` with grad enabled and backpropagates through the fresh graph.
  The global generator is snapshotted in the forward pass and replayed, so dropout draws
  the same masks. Under `create_graph=True` the inputs are aliased rather than detached, so
  the recomputed graph stays connected and higher-order derivatives work. A tensor that
  requires grad but was not declared raises an error, because its gradient would otherwise
  be lost. `GPT.gradient_checkpointing = True` checkpoints every block: on the round-1 GPT
  step the peak traced memory halves for a third more time (see
  [performance.md](performance.md#gradient-checkpointing)).
- **KV cache.** `GPT.make_cache(batch)` preallocates `(B, heads, block_size, head_dim)` key
  and value buffers per layer. With the cache, `forward` processes only the new tokens:
  positions continue from `cache.length` and queries attend to cached plus new keys through
  an offset causal mask. The position embeddings are absolute, so once a sequence outgrows
  `block_size` the shifted window is re-encoded from scratch, exactly as without the cache.
  Tests require identical tokens and logits with and without it.
- **Dropout masks** come from raw 16-bit uniforms: `p` is quantised to a multiple of
  2^-16 and the scale is `1 / (1 - p')` for the quantised `p'`, so the expectation is exact.
  The raw-bits path is used only for 64-bit bit generators (PCG64, PCG64DXSM, Philox,
  SFC64). For 32-bit ones such as MT19937 it would be biased, so the mask falls back to
  `Generator.bytes`.

## 7. Numerical stability

- `softmax`, `log_softmax` and `logsumexp` subtract the row maximum before `exp`, with a
  guard for rows that are entirely `-inf` (fully masked attention rows become NaN, as in
  PyTorch, instead of raising).
- `cross_entropy` never evaluates `log(softmax(z))`, which underflows to `-inf`.
  `binary_cross_entropy_with_logits` uses `max(z, 0) - z y + log1p(exp(-|z|))`.
- `sigmoid` evaluates `exp(-|x|)` and picks the algebraically equivalent branch per sign,
  so neither can overflow.
- `var` with `N <= correction` returns NaN (as NumPy and PyTorch do) instead of dividing by
  zero in the backward pass.
- GELU is the tanh approximation (`approximate="tanh"` in PyTorch).
- Gradient checks run in float64; training runs in float32.

## 8. Tooling: profiler and graph drawing

**Profiler.** Every public op function is wrapped by `@profiled(name)`. Outside a
`profiler.profile()` block the wrapper is one global lookup and a call; inside, it times the
op. The engine times each VJP call the same way, and the whole backward pass as the
"autograd engine" row. A stack of child-time accumulators turns total times into self
times: an op called from inside another op (a composite, or a graph VJP of a
`create_graph` backward) is charged to itself and subtracted from its caller, so the rows
add up to the time spent in ops and the engine. Example, one GPT training step:

```python
with tg.profiler.profile() as prof:
    loss = tg.cross_entropy(model(x), y)
    loss.backward()
print(prof.table(limit=8))
```

**Graph drawing.** `viz.to_mermaid(*outputs, names=...)` and `viz.to_dot(...)` walk the
recorded graph (the same DFS as the engine, but through constants as well) and emit Mermaid
or Graphviz source. Parameters are shaded, constants dashed, and ops are labelled with their
output shape. The README's diagram is generated this way, and a test keeps it in sync.

## 9. How the derivatives are verified

- **`gradcheck`** compares the analytical VJP with central differences in float64, on a
  random projection of the output (one backward pass instead of one per output element).
- **`gradgradcheck`** checks second derivatives in two steps. With a fixed random upstream
  gradient `v`, the first derivative is `G(x, v) = J(x)^T v`, computed with
  `create_graph=True`. First, `G` must equal the NumPy VJP (rtol 1e-10), so the two VJP
  implementations of every op agree. Then `G` is gradchecked with respect to both `x`
  (second derivatives) and `v` (differentiability in the upstream gradient, which the
  double-VJP trick relies on).
- **Hypothesis** draws shapes and data for both checks, covering every broadcasting
  pattern, reduction axes, permutations and recurrent sequence lengths; indexing is checked
  on a fixed list of basic, integer-array and boolean index expressions. The default profile
  runs 25 examples per property (a quarter of that for the slow recurrent checks), and
  `pytest --hypothesis-profile=thorough` runs 200.
- **Analytic results**: closed-form higher derivatives (up to the fourth), Hessians of
  quadratics and log-sum-exp, Hessian symmetry, HVPs against finite differences of
  gradients, and JVPs against the full Jacobian.
- **PyTorch parity** (`tests/parity`, separate CI job): forward values, first-order
  gradients through both VJP paths and second derivatives of every op and module, and the
  trajectories of the optimisers, in float64. See [parity.md](parity.md).
- **Mutation testing** was used to check that these suites catch bugs. Perturbing a
  constant in a VJP (such as GELU's, a gate derivative in the LSTM, or the GRU's recurrent
  gradient), removing the alias in `func`, or mis-scaling a dropout mask each fails at least
  one test.
