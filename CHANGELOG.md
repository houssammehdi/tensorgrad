# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (before 1.0, minor versions may
break the API).

## [0.2.0] - 2026-09-25

### Added

- **Higher-order autodiff.** `Tensor.backward(create_graph=True)` and
  `tensorgrad.autograd.grad(outputs, inputs, grad_outputs, retain_graph, create_graph,
  allow_unused)`. Every op has a second, differentiable VJP, so derivatives of any order
  work through the whole library, convolutions, pooling, attention and recurrent layers
  included. The first-order path is unchanged.
- `tensorgrad.func`: `grad`, `value_and_grad`, `vjp`, `jvp` (by the double-VJP trick),
  `jacobian`, `hessian` and `hvp`, composable and safe against perturbation confusion.
- `utils.gradgradcheck` for second derivatives.
- Recurrent layers `nn.RNN` (tanh or ReLU), `nn.LSTM` and `nn.GRU`, and the fused ops
  `ops.rnn`, `ops.lstm` and `ops.gru` with explicit backpropagation through time. Parameter
  names, gate layouts and the call signature follow `torch.nn`.
- `nn.BatchNorm2d`; `ops.batch_norm` accepts `(N, C, *spatial)` input.
- A KV cache for GPT sampling: `GPT.make_cache`, `GPT.forward(idx, cache=...)` and
  `generate(use_cache=True)` (the default).
- Fused `scaled_dot_product_attention` and `packed_self_attention` ops.
- New ops: `binary_cross_entropy_with_logits`, `broadcast_to`, `sum_to`, and
  `conv2d_input_grad` / `conv2d_weight_grad` (the VJPs of `conv2d`, differentiable too).
- `datasets.tiny_shakespeare()`, `datasets.fetch()` and `datasets.cache_dir()`: downloads
  pinned by SHA-256 and cached in `~/.cache/tensorgrad` (or `$TENSORGRAD_CACHE`).
- `utils.retain_freed_memory()`: keeps glibc from returning freed activations to the kernel
  between steps.
- Gradient checkpointing: `utils.checkpoint(fn, *args, params=None)` recomputes activations
  in the backward pass (dropout masks replayed, higher-order derivatives supported), and
  `GPT.gradient_checkpointing` applies it to every block.
- `tensorgrad.profiler`: per-op forward and backward self time and call counts.
- `tensorgrad.viz`: `to_mermaid` and `to_dot` draw the recorded graph.
- Examples: exact Newton and Newton-CG (`newton_logreg.py`), MAML (`maml_sinusoid.py`),
  the adding problem (`adding_problem.py`) and a 2-D DDPM (`ddpm_2d.py`). The char-level
  GPT now trains on tiny-shakespeare.
- Benchmarks: `train_step.py` (with `--profile` and `--peak-memory`), `generate.py` and
  `vs_torch.py`.
- PyTorch parity tests in `tests/parity` (skipped without torch) and a CI job that runs
  them with the CPU build of torch.
- Documentation: `docs/internals.md`, `docs/performance.md` and `docs/parity.md`.

### Changed

- **Breaking:** `Tensor.grad` is a `Tensor`, not a NumPy array (use `.grad.data` for the
  array). Under `create_graph=True` it carries a graph.
- **Breaking:** `==` and `!=` compare elementwise and return boolean tensors; tensors hash
  by identity. `Tensor.all()` and `Tensor.any()` were added for the common checks.
- Faster training steps (1.2 to 1.7 times as fast as 0.1.0 on the benchmark workloads): fused
  attention, in-place GELU and LayerNorm kernels, one GEMM per linear layer, dropout masks
  from raw 16-bit uniforms, a sort-based embedding backward and a running-maximum pooling
  kernel. See `docs/performance.md` for the measurements.
- `examples/char_gpt.py` defaults to a 4-layer, 128-wide model with a 128-character context,
  measures losses on fixed evaluation windows from its own generator, and falls back to the
  bundled excerpt when offline. `examples/shapes_cnn.py` uses `BatchNorm2d`.
- The README is a front page; the long-form material moved to `docs/internals.md`.

### Fixed

- The backward pass of `var` raised `ZeroDivisionError` when `N <= correction`; value and
  gradient are now NaN, as in PyTorch.
- Casting to an integer or boolean dtype returned a tensor that still required grad.
- `Tensor.data` could hold a NumPy scalar instead of a 0-d array after some ops.
- `embedding` silently wrapped negative ids; out-of-range ids now raise `IndexError`.
- `conv2d` accepted a misshaped bias and `max_pool2d` padding above half the kernel.
- `sigmoid` of an integer tensor truncated every value to 0.
- The GPT example never sampled the last training window, and raised for a corpus exactly
  one window long.
- The README described the bundled 13 KB corpus inaccurately; it is now documented as a
  transcribed excerpt used only offline and in tests.

## [0.1.0] - 2026-09-25

### Added

- `Tensor` with reverse-mode autodiff over NumPy: dynamic graphs, broadcasting-aware
  gradients, `no_grad` / `enable_grad`, `retain_graph`, graph freeing.
- 43 differentiable ops, each checked against finite differences with Hypothesis-drawn
  shapes.
- `nn` (Module system, Linear, Conv2d, MaxPool2d, Embedding, LayerNorm, BatchNorm1d,
  Dropout, activations, MLP, causal multi-head attention, pre-LN transformer block, GPT),
  `optim` (SGD, Adam, AdamW, gradient clipping, cosine and step schedules) and `utils`
  (gradcheck, DataLoader, npz checkpoints).
- Examples: spiral MLP, synthetic-shapes CNN and a character-level GPT on a bundled
  Shakespeare excerpt.
