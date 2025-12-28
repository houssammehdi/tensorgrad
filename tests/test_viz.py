"""Graph drawing: Mermaid and Graphviz DOT source for recorded graphs."""

from __future__ import annotations

import re

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import viz


def small_graph() -> tuple[tg.Tensor, dict[str, tg.Tensor]]:
    w = tg.Tensor(np.ones((2, 3)), requires_grad=True)
    x = tg.Tensor(np.ones(3))
    y = tg.tanh(w @ x)
    return (y * y).sum(), {"w": w, "x": x}


def test_mermaid_source_of_a_small_graph() -> None:
    loss, names = small_graph()
    assert viz.to_mermaid(loss, names=names, direction="LR") == "\n".join(
        [
            "flowchart LR",
            '    n0["w<br/>(2, 3)"]:::param',
            '    n1["x<br/>(3,)"]:::constant',
            '    n2("matmul<br/>(2,)"):::op',
            '    n3("tanh<br/>(2,)"):::op',
            '    n4("mul<br/>(2,)"):::op',
            '    n5(["sum<br/>scalar"]):::op',
            "    n0 --> n2",
            "    n1 --> n2",
            "    n2 --> n3",
            "    n3 --> n4",
            "    n4 --> n5",
            "    classDef param fill:#dfeafa,stroke:#898781,color:#0b0b0b",
            "    classDef constant fill:none,stroke:#898781,stroke-dasharray:4 3,color:#898781",
            "    classDef op fill:#f3f2ee,stroke:#898781,color:#0b0b0b",
            "",
        ]
    )


def test_dot_source_is_consistent() -> None:
    loss, names = small_graph()
    dot = viz.to_dot(loss, names=names)
    assert dot.startswith("digraph tensorgrad {\n  rankdir=TB;")
    declared = set(re.findall(r"^  (n\d+) \[label=", dot, re.M))
    edges = re.findall(r"^  (n\d+) -> (n\d+);", dot, re.M)
    assert declared == {f"n{i}" for i in range(6)}
    assert {a for a, _ in edges} | {b for _, b in edges} <= declared
    assert len(edges) == 5  # the reused tanh output feeds mul through one drawn edge
    assert 'label="sum\\nscalar"' in dot and "penwidth=2" in dot
    assert 'label="x\\n(3,)", shape=box, style="dashed"' in dot


def test_names_label_intermediate_tensors_and_quotes_are_escaped() -> None:
    a = tg.Tensor([1.0, 2.0], requires_grad=True)
    hidden = tg.relu(a)
    out = hidden.sum()
    names = {'say "a"': a, "hidden": hidden}
    assert "say #quot;a#quot;<br/>(2,)" in viz.to_mermaid(out, names=names)
    assert "hidden = relu\\n(2,)" in viz.to_dot(out, names=names)
    assert 'label="say \\"a\\"\\n(2,)"' in viz.to_dot(out, names=names)


def test_several_outputs_share_one_graph() -> None:
    a = tg.Tensor([1.0, 2.0], requires_grad=True)
    b = tg.exp(a)
    text = viz.to_mermaid(b.sum(), b.mean())
    assert text.count("exp<br/>") == 1
    assert text.count('(["') == 2  # both outputs drawn as stadiums
    with pytest.raises(ValueError, match="at least one"):
        viz.to_dot()
