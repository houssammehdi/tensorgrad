"""Execute the Python snippets in README.md so the documentation cannot drift from the code."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def run_blocks() -> dict[str, object]:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert blocks, "README has no python examples"
    namespace: dict[str, object] = {"__name__": "readme"}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"README block {i}", "exec"), namespace)
        except Exception as exc:
            raise AssertionError(f"README python block {i} failed:\n{block}") from exc
    return namespace


def test_readme_python_blocks_run() -> None:
    run_blocks()


def test_readme_graph_diagram_is_what_viz_draws() -> None:
    # The README's snippet stores tg.viz.to_mermaid(...) in `diagram`; the mermaid block
    # shown under it must be exactly that output.
    diagram = run_blocks()["diagram"]
    shown = re.findall(r"```mermaid\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert diagram in shown
