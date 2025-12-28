"""The per-op profiler: call counts, self-time accounting and the report."""

from __future__ import annotations

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import nn, profiler


def mlp_step(model: nn.Module, x: tg.Tensor, y: np.ndarray) -> None:
    tg.cross_entropy(model(x), y).backward()


def test_counts_forward_and_backward_calls_per_op() -> None:
    model = nn.MLP([3, 8, 8, 2])
    x, y = tg.randn((5, 3)), np.array([0, 1, 1, 0, 1])
    with profiler.profile() as prof:
        for _ in range(2):
            mlp_step(model, x, y)
    ops = prof.ops
    assert ops["linear"].forward_calls == 6 and ops["linear"].backward_calls == 6
    assert ops["relu"].forward_calls == 4 and ops["relu"].backward_calls == 4
    assert ops["cross_entropy"].forward_calls == 2
    assert ops[profiler.ENGINE].backward_calls == 2
    assert ops[profiler.ENGINE].forward_calls == 0


def test_self_time_excludes_nested_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter(range(100))  # a clock that advances by one second per reading
    monkeypatch.setattr(profiler.time, "perf_counter", lambda: float(next(ticks)))
    prof = profiler.Profile()

    def outer() -> int:
        return prof.timed("inner", False, lambda: 1) + 1

    assert prof.timed("outer", True, outer) == 2
    assert prof.ops["inner"].forward_seconds == 1.0  # read at t = 1 and t = 2
    assert prof.ops["outer"].backward_seconds == 2.0  # t = 0 to t = 3, minus the inner call
    assert prof.profiled_seconds == 3.0


def test_composite_ops_record_their_parts() -> None:
    a = tg.randn((256, 256), requires_grad=True)
    with profiler.profile() as prof:
        tg.mse_loss(a, np.zeros((256, 256), dtype=np.float32)).backward()
    assert {"mse_loss", "sub", "pow", "mean"} <= set(prof.ops)
    assert prof.ops["mse_loss"].backward_calls == 0  # a composite has no node of its own
    assert 0.0 < prof.profiled_seconds <= prof.wall_seconds
    assert all(s.forward_seconds >= 0 and s.backward_seconds >= 0 for s in prof.ops.values())


def test_create_graph_backward_charges_nested_ops() -> None:
    x = tg.Tensor(np.linspace(-1.0, 1.0, 5), requires_grad=True)
    with profiler.profile() as prof:
        (g,) = tg.autograd.grad(tg.sigmoid(x).sum(), [x], create_graph=True)
        g.sum().backward()
    # The differentiable VJP of sigmoid is built from mul/sub ops, timed as forward calls.
    assert prof.ops["sigmoid"].backward_calls >= 1
    assert prof.ops["mul"].forward_calls >= 2
    assert prof.ops["mul"].backward_calls >= 1  # ...and differentiated by the second pass
    assert prof.ops[profiler.ENGINE].backward_calls == 2


def test_nothing_is_recorded_outside_the_block_and_blocks_do_not_nest() -> None:
    with profiler.profile() as prof:
        assert profiler.active() is prof
        with pytest.raises(RuntimeError, match="nested"), profiler.profile():
            pass
    assert profiler.active() is None
    before = dict(prof.ops)
    tg.randn((2, 2)) + 1.0
    assert prof.ops == before


def test_an_exception_inside_an_op_keeps_the_accounting_consistent() -> None:
    with profiler.profile() as prof:
        with pytest.raises(ValueError, match="does not match"):
            tg.linear(tg.zeros((2, 3)), tg.zeros((4, 5)))
        tg.zeros(3) + 1.0
    assert prof._children == []
    assert prof.ops["linear"].forward_calls == 1
    assert prof.ops["add"].forward_calls == 1


def test_table_lists_ops_by_total_time() -> None:
    model = nn.MLP([3, 16, 2])
    with profiler.profile() as prof:
        mlp_step(model, tg.randn((8, 3)), np.zeros(8, dtype=np.int64))
    text = prof.table()
    header, rule, *rows, footer = text.splitlines()
    assert header.split()[:3] == ["op", "fwd", "calls"]
    assert set(rule) == {"-"}
    totals = [float(row.split()[-2]) for row in rows]
    assert totals == sorted(totals, reverse=True)
    assert footer.startswith("ops and engine:")
    assert len(prof.table(limit=2).splitlines()) == 5
