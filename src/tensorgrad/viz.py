"""Draw the recorded graph behind tensors as Graphviz DOT or Mermaid flowchart source.

    names = {"x": x, **dict(model.named_parameters())}
    print(tensorgrad.viz.to_mermaid(loss, names=names))

Leaves that require grad (parameters, inputs) are shaded, other leaves (constants such as
targets or masks) are dashed, every op is a rounded box labelled with its name and output
shape, and the outputs have a heavier outline. Mermaid renders on GitHub inside a
``mermaid`` code block; DOT renders with Graphviz (``dot -Tsvg graph.dot``).

A backward pass without ``retain_graph`` frees the graph it walks, so draw it first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from tensorgrad.tensor import Tensor

__all__ = ["to_dot", "to_mermaid"]

Direction = Literal["TB", "LR"]

_PARAM_FILL, _OP_FILL, _EDGE, _INK, _MUTED = "#dfeafa", "#f3f2ee", "#898781", "#0b0b0b", "#898781"


@dataclass
class _Vertex:
    id: str
    title: str
    shape: str
    kind: Literal["param", "constant", "op"]
    output: bool


def _shape_text(t: Tensor) -> str:
    return "scalar" if t.ndim == 0 else str(t.shape)


def _collect(
    roots: Iterable[Tensor], names: Mapping[str, Tensor] | None
) -> tuple[list[_Vertex], list[tuple[str, str]]]:
    """Vertices (inputs first) and edges ``(parent id, child id)`` of the graph."""
    roots = list(roots)
    if not roots:
        raise ValueError("pass at least one tensor to draw")
    label_of = {id(t): name for name, t in (names or {}).items()}
    output_ids = {id(r) for r in roots}
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(r, False) for r in reversed(roots)]
    while stack:  # iterative post-order DFS through every recorded input
        tensor, expanded = stack.pop()
        if expanded:
            order.append(tensor)
            continue
        if id(tensor) in visited:
            continue
        visited.add(id(tensor))
        stack.append((tensor, True))
        if tensor._node is not None:
            stack.extend((p, False) for p in reversed(tensor._node.parents))
    ids = {id(t): f"n{i}" for i, t in enumerate(order)}
    vertices: list[_Vertex] = []
    edges: list[tuple[str, str]] = []
    for t in order:
        name = label_of.get(id(t))
        if t._node is None:
            kind: Literal["param", "constant", "op"] = "param" if t.requires_grad else "constant"
            title = name or ("input" if t.requires_grad else "constant")
        else:
            kind = "op"
            title = t._node.op if name is None else f"{name} = {t._node.op}"
            for parent in dict.fromkeys(ids[id(p)] for p in t._node.parents):  # x * x: one edge
                edges.append((parent, ids[id(t)]))
        vertices.append(_Vertex(ids[id(t)], title, _shape_text(t), kind, id(t) in output_ids))
    return vertices, edges


def to_dot(
    *tensors: Tensor, names: Mapping[str, Tensor] | None = None, direction: Direction = "TB"
) -> str:
    """Graphviz DOT source of the graph that produced ``tensors``.

    Args:
        tensors: The outputs to draw (e.g. a loss).
        names: Labels for tensors, e.g. ``dict(model.named_parameters())``.
        direction: ``"TB"`` (top to bottom) or ``"LR"`` (left to right).
    """
    vertices, edges = _collect(tensors, names)
    lines = [
        "digraph tensorgrad {",
        f"  rankdir={direction};",
        f'  node [fontname="Helvetica", fontsize=10, color="{_EDGE}", fontcolor="{_INK}"];',
        f'  edge [color="{_EDGE}", arrowsize=0.6];',
    ]
    for v in vertices:
        label = f"{v.title}\\n{v.shape}".replace('"', '\\"')
        if v.kind == "param":
            style = f'shape=box, style="filled", fillcolor="{_PARAM_FILL}"'
        elif v.kind == "constant":
            style = f'shape=box, style="dashed", fontcolor="{_MUTED}"'
        else:
            style = f'shape=box, style="rounded,filled", fillcolor="{_OP_FILL}"'
        if v.output:
            style += ", penwidth=2"
        lines.append(f'  {v.id} [label="{label}", {style}];')
    lines.extend(f"  {a} -> {b};" for a, b in edges)
    lines.append("}")
    return "\n".join(lines) + "\n"


def to_mermaid(
    *tensors: Tensor, names: Mapping[str, Tensor] | None = None, direction: Direction = "TB"
) -> str:
    """Mermaid flowchart source of the graph that produced ``tensors``.

    Takes the same arguments as :func:`to_dot`. Paste the result into a ``mermaid`` code
    block to render it on GitHub.
    """
    vertices, edges = _collect(tensors, names)
    lines = [f"flowchart {direction}"]
    for v in vertices:
        label = f"{v.title}<br/>{v.shape}".replace('"', "#quot;")
        if v.output:
            opening, closing = "([", "])"  # stadium
        elif v.kind == "op":
            opening, closing = "(", ")"  # rounded box
        else:
            opening, closing = "[", "]"
        lines.append(f'    {v.id}{opening}"{label}"{closing}:::{v.kind}')
    lines.extend(f"    {a} --> {b}" for a, b in edges)
    lines += [
        f"    classDef param fill:{_PARAM_FILL},stroke:{_EDGE},color:{_INK}",
        f"    classDef constant fill:none,stroke:{_EDGE},stroke-dasharray:4 3,color:{_MUTED}",
        f"    classDef op fill:{_OP_FILL},stroke:{_EDGE},color:{_INK}",
    ]
    return "\n".join(lines) + "\n"
