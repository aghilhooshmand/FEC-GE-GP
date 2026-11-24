"""One look for every figure of the paper: font, sizes, colours, and a reproducible PDF.

Every figure function calls `use()` before drawing and `save(fig, path)` to write. The font is DejaVu Serif,
which ships with matplotlib, so the same code gives the same figure on any machine. Colours come from one
validated categorical palette (checked for colour-blind separation); a second series also gets its own line
style or marker, so no figure depends on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2a78d6"        # first series (GE, or the only series of a panel)
ORANGE = "#eb6834"      # second series (GP, or the second dataset); drawn dashed
GREEN = "#1baf7a"       # third series
INK = "#0b0b0b"         # text
INK_SECONDARY = "#52514e"   # axes, ticks, notes
GRID = "#e6e5e1"
FILL = "#eef2f7"        # diagram boxes
FILL_ACCENT = "#dde8f6"
FILL_HIT = "#dff3ea"    # diagram: cache hit (green tint)
FILL_MISS = "#fde6da"   # diagram: cache miss (orange tint)

BASE_SIZE = 8           # points, at the width the figure is printed (the text width)
FONT = "DejaVu Serif"


def use() -> None:
    """Set the shared style; call before creating a figure."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [FONT], "mathtext.fontset": "dejavuserif",
        "font.size": BASE_SIZE, "axes.titlesize": BASE_SIZE + 1, "axes.labelsize": BASE_SIZE,
        "xtick.labelsize": BASE_SIZE - 0.5, "ytick.labelsize": BASE_SIZE - 0.5, "legend.fontsize": BASE_SIZE - 0.5,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def title(ax, name: str, note: str | None = None) -> None:
    """Panel heading: the panel's name on the left, an optional statistic in grey on the right."""
    ax.set_title(name, loc="left", fontsize=BASE_SIZE + 1, color=INK, fontweight="bold")
    if note:
        ax.set_title(note, loc="right", fontsize=BASE_SIZE - 1, color=INK_SECONDARY)


def save(fig, path: Path) -> Path:
    """Write a PDF without dates or tool names, so a rebuild gives the same file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return path
