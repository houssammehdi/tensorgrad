"""Minimal mini-batch iteration over in-memory NumPy arrays."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from tensorgrad._types import Array
from tensorgrad.tensor import Tensor

__all__ = ["DataLoader"]


class DataLoader:
    """Iterate over aligned arrays in mini-batches, yielding tuples of tensors.

    Args:
        *arrays: Arrays sharing the same first dimension (e.g. inputs and labels).
        batch_size: Examples per batch.
        shuffle: Draw a fresh permutation at the start of every epoch.
        drop_last: Skip the final batch if it is smaller than ``batch_size``.
        seed: Seed for the shuffling generator (independent of the global generator).

    Example:
        >>> import numpy as np
        >>> loader = DataLoader(np.zeros((10, 3)), np.arange(10), batch_size=4)
        >>> [xb.shape for xb, _ in loader]
        [(4, 3), (4, 3), (2, 3)]
    """

    def __init__(
        self,
        *arrays: Array,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if not arrays:
            raise ValueError("DataLoader needs at least one array")
        lengths = {len(a) for a in arrays}
        if len(lengths) != 1:
            raise ValueError(f"arrays have different lengths: {sorted(lengths)}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.arrays = tuple(np.asarray(a) for a in arrays)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)

    @property
    def num_examples(self) -> int:
        """Number of examples per epoch (before batching)."""
        return len(self.arrays[0])

    def __len__(self) -> int:
        full, rest = divmod(self.num_examples, self.batch_size)
        return full if self.drop_last or rest == 0 else full + 1

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        n = self.num_examples
        order = self._rng.permutation(n) if self.shuffle else np.arange(n)
        stop = n - n % self.batch_size if self.drop_last else n
        for start in range(0, stop, self.batch_size):
            idx = order[start : start + self.batch_size]
            yield tuple(Tensor._wrap(a[idx]) for a in self.arrays)
