"""Execute the Python snippets in README.md so the documentation cannot drift from the code."""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_python_blocks_run() -> None:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert blocks, "README has no python examples"
    namespace: dict[str, object] = {"__name__": "readme"}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"README block {i}", "exec"), namespace)
        except Exception as exc:
            raise AssertionError(f"README python block {i} failed:\n{block}") from exc
