"""Shared figure styling for the example scripts (light surface, validated palette).

Categorical colours are used in their fixed slot order; text is always ink, never a series
colour; gridlines are solid hairlines.
"""

from __future__ import annotations

from typing import Any

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
#: Categorical slots in their fixed order (blue, orange, aqua, yellow, magenta, green, violet,
#: red); the first three are distinguishable pairwise, including under colour-vision
#: deficiencies.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def style_axes(ax: Any, title: str | None = None, grid: bool = True) -> None:
    """Recessive axes: hairline grid, no top/right spines, secondary-ink ticks."""
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if grid:
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def import_pyplot() -> Any:
    """Return ``matplotlib.pyplot`` with the non-interactive Agg backend, or ``None``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return None
    return plt
