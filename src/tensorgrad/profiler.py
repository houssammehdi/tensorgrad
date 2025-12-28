"""Per-op timing: where the time of a forward and backward pass goes.

    with tensorgrad.profiler.profile() as prof:
        loss = tg.cross_entropy(model(x), y)
        loss.backward()
    print(prof.table())

Every differentiable op is timed on entry and exit, and every VJP the backward pass runs is
timed too. Time spent in ops that an op calls (a composite op, or the differentiable VJP of a
``create_graph`` backward) is charged to those ops, so each row is *self* time and the rows
add up. The "autograd engine" row is what the backward pass spends outside VJPs: sorting the
graph and summing gradients. Profiling costs a few microseconds per op call; use it to find
hot spots rather than for absolute timings.
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

__all__ = ["ENGINE", "OpStats", "Profile", "profile", "profiled"]

P = ParamSpec("P")
R = TypeVar("R")

#: Name of the row that holds the backward engine's own time.
ENGINE = "autograd engine"


@dataclass
class OpStats:
    """Call counts and self time (in seconds) of one op."""

    forward_calls: int = 0
    forward_seconds: float = 0.0
    backward_calls: int = 0
    backward_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        """Forward plus backward self time."""
        return self.forward_seconds + self.backward_seconds


class Profile:
    """Per-op timings collected by :func:`profile`, keyed by op name."""

    def __init__(self) -> None:
        self.ops: dict[str, OpStats] = {}
        #: Wall-clock duration of the ``with`` block.
        self.wall_seconds = 0.0
        # Time spent in nested timed calls, one entry per timed call in progress.
        self._children: list[float] = []

    def timed(
        self, name: str, backward: bool, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        """Run ``fn(*args, **kwargs)`` and charge its self time to ``name``."""
        self._children.append(0.0)
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            own = elapsed - self._children.pop()
            if self._children:
                self._children[-1] += elapsed
            stats = self.ops.get(name)
            if stats is None:
                stats = self.ops[name] = OpStats()
            if backward:
                stats.backward_calls += 1
                stats.backward_seconds += own
            else:
                stats.forward_calls += 1
                stats.forward_seconds += own

    @property
    def profiled_seconds(self) -> float:
        """Time inside ops and the engine (the sum of all rows)."""
        return sum(s.total_seconds for s in self.ops.values())

    def table(self, limit: int | None = None) -> str:
        """The rows sorted by total self time, as an aligned text table (times in ms)."""
        rows = sorted(self.ops.items(), key=lambda item: item[1].total_seconds, reverse=True)
        total = self.profiled_seconds or 1.0
        width = max([len("op"), *(len(name) for name in self.ops)])
        header = (
            f"{'op':<{width}}  {'fwd calls':>9}  {'fwd ms':>9}  {'bwd calls':>9}  "
            f"{'bwd ms':>9}  {'total ms':>9}  {'share':>6}"
        )
        lines = [header, "-" * len(header)]
        for name, s in rows[:limit]:
            lines.append(
                f"{name:<{width}}  {s.forward_calls:>9}  {1e3 * s.forward_seconds:>9.2f}  "
                f"{s.backward_calls:>9}  {1e3 * s.backward_seconds:>9.2f}  "
                f"{1e3 * s.total_seconds:>9.2f}  {s.total_seconds / total:>6.1%}"
            )
        outside = self.wall_seconds - self.profiled_seconds
        lines.append(
            f"ops and engine: {1e3 * self.profiled_seconds:.2f} ms of {1e3 * self.wall_seconds:.2f}"
            f" ms wall time ({1e3 * outside:.2f} ms elsewhere: Python code between ops, "
            "optimiser, data)"
        )
        return "\n".join(lines)


_active: Profile | None = None


def active() -> Profile | None:
    """The profile being collected, if a :func:`profile` block is open."""
    return _active


@contextlib.contextmanager
def profile() -> Iterator[Profile]:
    """Collect per-op timings for the code inside the ``with`` block.

    Blocks cannot be nested, and only the thread that opened the block should run ops in it.
    """
    global _active
    if _active is not None:
        raise RuntimeError("profile() blocks cannot be nested")
    result = Profile()
    _active = result
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.wall_seconds = time.perf_counter() - start
        _active = None


def profiled(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for op functions: time each call under ``name`` while profiling."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            prof = _active
            if prof is None:
                return fn(*args, **kwargs)
            return prof.timed(name, False, fn, *args, **kwargs)

        return wrapper

    return decorate
