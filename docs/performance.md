# Performance

tensorgrad runs every op as NumPy calls plus Python bookkeeping. This page measures what that
costs: training steps against version 0.1.0 and against PyTorch on the same CPU, the effect
of BLAS threading and of the glibc allocator, and generation with the KV cache.

**Machine and method.** All numbers come from a shared 4-vCPU Linux VM (Python 3.11.15,
NumPy 2.4.6 with OpenBLAS, PyTorch 2.14.0) on which other jobs were running: the 1-minute
load average during these measurements was between 3.5 and 10, so treat them as
indicative.
To keep comparisons fair under a changing load, the variants being compared are
interleaved (A, B, A, B, ...) and the tables report medians. Every command is given; the
benchmark scripts print the load average with their results.

## Training steps

`benchmarks/train_step.py` times one full step (forward, backward, optimiser update) of the
three example workloads, built exactly as in the round-1 examples so that versions can be
compared:

- **spiral**: the `2 -> 64 -> 64 -> 3` ReLU MLP of `spiral_mlp.py`, a full batch of 600
  points, Adam;
- **cnn**: the two-convolution network of `shapes_cnn.py --arch plain` on 64 images of
  16x16, Adam;
- **gpt**: a 3-layer, 4-head, width-96 GPT with a 64-token context and dropout 0.1 on 32
  sequences, AdamW with gradient clipping (the round-1 `char_gpt.py` model).

### Against version 0.1.0

Five interleaved rounds of `OPENBLAS_NUM_THREADS=1 python benchmarks/train_step.py --json`,
with the source tree of 0.1.0 (commit `c6921d5`) or of this version on `PYTHONPATH`. The last
column adds `--retain-freed-memory` (see [the allocator](#the-allocator)):

| Workload | 0.1.0 | 0.2.0 | 0.2.0 with `--retain-freed-memory` |
|---|---|---|---|
| spiral | 1.62 ms (best 1.31) | 1.38 ms (best 1.26): 1.2x | 1.06 ms (best 1.00): 1.5x |
| cnn | 23.4 ms (best 21.4) | 19.9 ms (best 17.8): 1.2x | 15.8 ms (best 14.5): 1.5x |
| gpt | 255 ms (best 223) | 177 ms (best 147): 1.4x | 149 ms (best 140): 1.7x |

Each cell is the median over the five runs of each run's median step time, with the fastest
single step in brackets; the factors compare medians with 0.1.0. The load average was
between 3.5 and 5.3 during this comparison. A run of the same code at a load of 6.8 to 10.4
gave factors of 1.2, 1.2 and 1.6 (without the allocator tuning) and 1.6, 1.5 and 1.6 (with
it): at high load the medians move by up to 20% between runs.

The GPT step is faster because of the transformer work of this release (each change was
measured on its own when it was made, by interleaved runs of this benchmark):

| Change | Effect on the GPT step |
|---|---|
| Fused attention: one node from the packed QKV projection to merged heads, with an in-place masked softmax and a hand-derived backward pass | about 7% faster |
| In-place GELU kernels, and one GEMM per linear layer over the flattened batch and time axes | about 20% faster |
| Dropout masks from raw 16-bit random integers instead of `random()` doubles | masks 3.4x faster |
| LayerNorm computed in place on two buffers | LayerNorm 10% faster |
| Embedding backward by a stable sort and `np.add.reduceat` instead of `np.add.at` | the scatter 7x faster |

The CNN step gained from a new max-pooling kernel (a running maximum over strided views
instead of an argmax over copied windows: 20.2 ms to 16.8 ms per step, measured on its own).
Higher-order support costs nothing when unused: an ordinary backward pass never builds a
graph (see [internals](internals.md#2-two-vjps-per-op)).

### Where the time goes

`--profile` adds a per-op breakdown from `tensorgrad.profiler` (self time over five steps):

```
$ OPENBLAS_NUM_THREADS=1 python benchmarks/train_step.py --workloads gpt cnn \
      --retain-freed-memory --profile --repeats 10
    gpt: median   155.00 ms  min   147.47 ms  (10 steps)
    cnn: median    16.82 ms  min    16.03 ms  (10 steps)

gpt: 5 steps under the profiler
op               fwd calls     fwd ms  bwd calls     bwd ms   total ms   share
------------------------------------------------------------------------------
linear                  65     101.34         65     182.33     283.67   39.3%
self_attention          15      92.82         15      91.69     184.51   25.6%
gelu                    15      46.70         15      63.11     109.81   15.2%
layer_norm              35      35.37         35      49.15      84.51   11.7%
dropout                 35      23.34         35       5.51      28.86    4.0%
autograd engine          0       0.00          5      11.47      11.47    1.6%
add                     35       7.26         35       0.53       7.79    1.1%
cross_entropy            5       5.12          5       1.83       6.95    1.0%
embedding               10       1.25         10       3.26       4.50    0.6%
ops and engine: 722.07 ms of 740.89 ms wall time (18.81 ms elsewhere: Python code between
ops, optimiser, data)

cnn: 5 steps under the profiler
op               fwd calls     fwd ms  bwd calls     bwd ms   total ms   share
------------------------------------------------------------------------------
conv2d                  10      20.31         10      29.61      49.93   61.9%
max_pool2d              10       4.32         10      12.57      16.89   20.9%
relu                    15       2.77         15       6.31       9.08   11.3%
linear                  10       0.86         10       0.95       1.82    2.3%
autograd engine          0       0.00          5       1.42       1.42    1.8%
...
```

(Load average 6.2. The footer line is wrapped here; the profiled steps run a little slower
than unprofiled ones.)

In the GPT step, the linear layers (their GEMMs) and the fused attention op take about two
thirds of the time, and the element-wise work of GELU and LayerNorm most of the rest. The
engine's own bookkeeping (sorting the graph, summing gradients) is under 2%. In the CNN
step, the convolutions dominate.

### Against PyTorch

`benchmarks/vs_torch.py` builds the same three workloads in PyTorch (same layers and
shapes, the same optimisers and dropout, PyTorch's fused `scaled_dot_product_attention`),
runs both frameworks in one process with one thread each (`OPENBLAS_NUM_THREADS=1` and
`torch.set_num_threads(1)`), and interleaves them over three rounds:

| Workload | tensorgrad | PyTorch | tensorgrad / PyTorch |
|---|---|---|---|
| spiral | 1.10 ms | 1.44 ms | 0.76 |
| cnn | 17.1 ms | 8.6 ms | 2.0 |
| gpt | 163 ms | 155 ms | 1.05 |

These are medians over all timed steps with `--retain-freed-memory` (load average 5.9 at the
end). Without it (load average 6.3) the ratios were 0.73, 2.4 and 1.05. Two earlier pairs of
runs, at load averages of 7 to 11, gave 0.72 to 0.74 for the MLP, 2.0 to 2.6 for the CNN and
1.02 to 1.07 for the GPT. PyTorch 2.14.0 here uses MKL for matrix products and oneDNN for
convolutions, with AVX-512.

- **The tiny MLP is faster in tensorgrad.** Its matrices are 600x64, so a step is dominated
  by per-op overhead, and a NumPy call plus tensorgrad's bookkeeping is cheaper than
  PyTorch's dispatcher and autograd machinery.
- **The GPT step is on par.** At this size the step is dominated by matrix products, which
  both frameworks hand to an optimised BLAS (OpenBLAS for NumPy, MKL for PyTorch), and
  tensorgrad's fused attention, GELU, LayerNorm and linear ops keep the rest small.
- **Convolutions are PyTorch's clear win.** tensorgrad's im2col copies every receptive field
  into a patch matrix and scatters gradients back with `KH*KW` strided additions, while
  PyTorch uses oneDNN's convolution kernels.
- These are single-thread numbers. With more threads PyTorch's kernels scale on large
  operations and tensorgrad's per-op overhead does not shrink, so the gap would widen for
  bigger models; on this shared machine multi-threaded timings are not meaningful (next
  section).

## BLAS threads

Round 1 found that OpenBLAS's default thread pool made small workloads much slower and
recommended `OPENBLAS_NUM_THREADS=1`. Re-checked for this release by running
`benchmarks/train_step.py --retain-freed-memory` with `OPENBLAS_NUM_THREADS=1` and `=4`,
two rounds each, at a load average of 5.4 to 7.5:

| Workload | 1 thread | 4 threads | Slowdown |
|---|---|---|---|
| spiral | 1.07, 1.09 ms | 64.0, 40.1 ms | 37x to 60x |
| cnn | 16.3, 16.9 ms | 341, 237 ms | 14x to 21x |
| gpt | 148, 159 ms | 1090, 896 ms | 6x to 7x |

An earlier run at a load of 6.2 to 6.5 gave slowdowns of 41x to 74x, about 10x and about 4x.

On an idle machine, earlier in this release's development, four threads were within about
10% of one thread for these workloads: the matrices are too small to split profitably. On a
busy machine, OpenBLAS's worker threads compete with everything else for the cores and a
step becomes 4 to 70 times slower, the smallest workload suffering most. One thread is
never much worse and sometimes far better, so it stays the recommendation, and CI sets it.

## The allocator

NumPy allocates array buffers with `malloc`. glibc serves large requests with `mmap` and
hands freed memory at the top of the heap back to the kernel, so a training step that frees
its activations at the end page-faults all of them back in on the next step. For the GPT
step that was about 11,000 to 13,000 minor page faults and a fifth of the step time
(measured with `getrusage`).

`tensorgrad.utils.retain_freed_memory()` raises glibc's mmap threshold to 64 MB and its trim
threshold to 1 GB with `mallopt`, so freed buffers stay mapped and are reused: faults drop
to about 30 per step. Both thresholds must change together, because raising only the
trim threshold also freezes glibc's adaptive mmap threshold at 128 KB, which is slower than
doing nothing. The setting is process-wide, so it is opt-in: the training examples call it,
the benchmarks take `--retain-freed-memory`, and it returns `False` without doing anything
on other C libraries. Its effect is the difference between the last two columns of the
version table above.

## Gradient checkpointing

`GPT.gradient_checkpointing = True` recomputes each block's activations in the backward pass
(`tensorgrad.utils.checkpoint`). On the round-1 GPT step, `OPENBLAS_NUM_THREADS=1 python
benchmarks/train_step.py --workloads gpt gpt-checkpointed --retain-freed-memory
--peak-memory` (load average 5.9; a run at 4.5 gave 157 and 207 ms and the same memory):

| | Median step | Peak memory traced by `tracemalloc` |
|---|---|---|
| without checkpointing | 152 ms | 84.1 MB |
| every block checkpointed | 210 ms | 40.1 MB |

Between the two passes only each block's input is kept, so the peak roughly halves; the
price is one more forward pass through the blocks: 32% and 38% more time per step in the two
runs.

## Generation with the KV cache

`OPENBLAS_NUM_THREADS=1 python benchmarks/generate.py` samples from an untrained model of
`char_gpt.py`'s default shape (4 layers, width 128, 128-token context), with and without the
cache. Median of three runs:

| Tokens generated | With the cache | Without | Speed-up |
|---|---|---|---|
| 127 (filling the 128-token window) | 668 tokens/s | 200 tokens/s | 3.3x |
| 384 (past the window) | 155 tokens/s | 133 tokens/s | 1.2x |

(The three runs gave 689, 662 and 668 tokens/s with the cache and 199, 200 and 205 without
for the first row; 155, 156 and 148 against 137, 133 and 127 for the second. Load average
about 6. Runs at a load of 7.6 to 9.1 gave 3.5x and 1.2x.)

Within the context window the cache turns every step into a one-position update. Past it,
the learned position embeddings are absolute, so the shifted window has to be re-encoded
from scratch either way, and the cache only saves the first 127 steps. Both settings
produce identical tokens.
