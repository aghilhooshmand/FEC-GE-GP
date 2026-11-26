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

# Size. MDPI's text width is 394.4 pt = 5.46 in. Plots are drawn at exactly that width (size()), so text set at
# BASE_SIZE prints at BASE_SIZE; the two diagrams (diagrams.py) are wider and printed across the full page width
# (text width plus the left margin, MDPI's \extralength), again at about their drawn size.
TEXT_WIDTH = 394.35 / 72.27     # inches
DRAWN_WIDTH = 6.3               # the width the plots were laid out for; heights are given for this width
BASE_SIZE = 8           # points, as printed
FONT = "DejaVu Serif"
DPI = 600               # for any raster part of a vector figure (e.g. a colour bar); MDPI asks for >= 300 dpi


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


def title(ax, name: str, note: str | None = None, stacked: bool = False) -> None:
    """Panel heading: the panel's name on the left, an optional statistic in grey on the right; with `stacked`
    (narrow panels), the statistic goes on its own line(s) under the name instead."""
    if note and stacked:
        lines = note.count("\n") + 1
        ax.set_title(name, loc="left", fontsize=BASE_SIZE + 1, color=INK, fontweight="bold",
                     pad=5 + lines * (BASE_SIZE + 1.5))
        ax.annotate(note, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=BASE_SIZE - 1, color=INK_SECONDARY, linespacing=1.2,
                    annotation_clip=False)
        return
    ax.set_title(name, loc="left", fontsize=BASE_SIZE + 1, color=INK, fontweight="bold")
    if note:
        ax.set_title(note, loc="right", fontsize=BASE_SIZE - 1, color=INK_SECONDARY)


def size(height: float) -> tuple[float, float]:
    """figsize for a plot printed at the text width; `height` is in inches for a DRAWN_WIDTH-wide layout and is
    reduced a little less than the width, so panels keep room for their labels."""
    return TEXT_WIDTH, round(height * 0.92, 2)


def panel_axes(fig) -> list:
    """The data panels of a figure, top to bottom, then left to right: visible axes that are not colour bars and
    not a twin of an earlier panel (same position). An axes can opt out with ax.no_panel_label = True."""
    out, seen = [], set()
    for ax in fig.axes:
        if (not ax.axison or ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar")
                or getattr(ax, "no_panel_label", False)):
            continue
        b = ax.get_position()
        key = (round(b.x0, 3), round(b.y0, 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(ax)
    return sorted(out, key=lambda a: (-round(a.get_position().y1, 2), round(a.get_position().x0, 2)))


def label_panels(fig) -> None:
    """MDPI style: (a), (b), ... in bold at the top left of each panel, when a figure has more than one."""
    axes = panel_axes(fig)
    if len(axes) < 2:
        return
    for i, ax in enumerate(axes):
        tag = f"({chr(97 + i)})"
        # a panel title gets the letter in front of it (style kept), unless a note on the right shares its line
        for text in (() if ax._right_title.get_text() else (ax._left_title, ax.title)):
            if text.get_text():
                bold = text.get_fontweight() in ("bold", 700)
                text.set_text((tag if bold else r"$\mathbf{" + tag + "}$") + " " + text.get_text())
                break
        else:                                    # no title: the letter above the top left corner
            ax.annotate(tag, xy=(0, 1), xycoords="axes fraction", xytext=(-3, 3),
                        textcoords="offset points", ha="right", va="bottom", fontsize=BASE_SIZE + 1,
                        fontweight="bold", color=INK, annotation_clip=False)


def save(fig, path: Path, labels: bool = True) -> Path:
    """Write a PDF without dates or tool names, so a rebuild gives the same file; label the panels first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if labels:
        label_panels(fig)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=DPI,
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return path
