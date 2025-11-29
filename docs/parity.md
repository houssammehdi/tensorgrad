# PyTorch parity

`tests/parity/` checks tensorgrad against PyTorch on identical float64 inputs. PyTorch is
not a dependency: the tests are marked `torch` and skip themselves when it is not
installed, and CI runs them in a separate job with the CPU build of torch.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
OMP_NUM_THREADS=1 pytest tests/parity -m torch
```

With torch 2.14.0 all 111 parity tests pass (about 5 seconds), with tolerances of
`rtol=1e-10`, `atol=1e-12`.

## What is compared

**Every differentiable op** (`tests/parity/test_parity_ops.py`), each at three levels:

1. forward values;
2. the vector-Jacobian product for a random upstream gradient `v`, through both of
   tensorgrad's VJP implementations (the NumPy one used by ordinary backward passes and the
   differentiable one used under `create_graph=True`);
3. second order: for random `w`, the gradient of `<VJP(v), w>` with respect to every input
   *and* to `v`, which covers Hessian-vector products and double backward in the
   upstream-gradient direction (what forward-mode products via the double-VJP trick use).

Shapes for the elementwise ops and reductions are drawn by Hypothesis, including every
kind of broadcasting. The ops are the elementwise arithmetic and non-linearities, all
reductions over arbitrary axes, `matmul` for all nine shape combinations, the shape ops
(`reshape`, `permute`, `broadcast_to`, `sum_to` = `sum_to_size`, basic/integer/boolean
indexing, `concat`, `stack`, `where`, `masked_fill`), the softmax family, the four losses,
`conv2d` and its two gradient ops (checked against `torch.nn.grad.conv2d_input` /
`conv2d_weight`), `max_pool2d`, `linear`, `layer_norm`, `embedding`, `batch_norm` (output
and running statistics) and scaled-dot-product attention (causal and with explicit masks).

**nn modules** (`test_parity_nn.py`), with weights copied into the PyTorch equivalent
(`Linear`, `Conv2d`, `Embedding`, `LayerNorm`, `BatchNorm1d`, `nn.MultiheadAttention`, a
`Sequential` MLP) or into a reference written directly in PyTorch (the pre-LN transformer
block and the GPT with tied embeddings): outputs, the input gradient, every parameter
gradient, and a Hessian-vector product with respect to all parameters. PyTorch's fused CPU
attention kernel has no double backward, so these tests select its "math" attention
backend.

**Optimisers** (`test_parity_optim.py`): SGD (momentum, Nesterov, weight decay), Adam with
L2 decay, AdamW, `clip_grad_norm_` and `StepLR` follow the same trajectories as
`torch.optim` over 25 steps of identical random gradients.

## Deliberate differences

Each of these is pinned by `test_documented_differences_from_pytorch` or
`test_attention_mask_convention_is_forbidden_positions`.

| tensorgrad | PyTorch | Why |
|---|---|---|
| `cross_entropy` takes logits `(*, C)`: the class axis is **last** | `(N, C, *)`: class axis second | Sequence models produce `(batch, time, vocab)`; no transpose needed. |
| Mean cross-entropy over targets that are *all* ignored is `0` | `nan` | One batch without labels should not turn every parameter into NaN. |
| `d(b**e)/de` is `0` where `b <= 0` | `nan` for `b < 0` | `log(b)` is undefined there; tensorgrad documents the zero. For `b == 0` both give 0. |
| `max`/`min` over an axis split the gradient evenly between tied extrema | `torch.max(dim=...)` gives it all to one index (`amax`/`amin` split it) | A symmetric subgradient; the parity tests compare with `amax`/`amin`. |
| Attention masks mark **forbidden** positions (`True` = may not attend) | Same for `nn.MultiheadAttention`; the opposite for `F.scaled_dot_product_attention` | Matches `masked_fill(mask, -inf)` semantics. |
| `gelu` is the tanh approximation | `F.gelu` defaults to the exact erf form | GPT-2 uses the tanh form; compare with `approximate="tanh"`. |
| `MultiHeadAttention(causal=True)` by default | no default mask | tensorgrad's attention exists for decoder models. |
| Dropout masks come from NumPy's generator | torch's generator | Same distribution, different random streams: masks are not bitwise comparable. |
