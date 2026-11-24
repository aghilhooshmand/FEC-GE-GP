"""The two explanatory diagrams of the paper, drawn from code as vector PDFs.

    fig_ge_gp.pdf         Figure 1: how GE (left) and strongly typed GP (right) build and score one program
    fig_fec_workflow.pdf  Figure 2: fitness evaluation caching around the fitness evaluation

The GE example in Figure 1 is real: the genome maps to the phenotype shown under grammars/electricity_cat.bnf
with GRAPE's lazy mapper, and `ge_gp()` checks this before drawing. The GP tree is the same program.

    python diagrams.py <out_dir>        # writes both PDFs (paper_tables.py also calls this)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle

import figstyle as fs

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- shared drawing helpers


def _box(ax, x, y, w, h, text, fill=fs.FILL, size=fs.BASE_SIZE, weight="normal", mono=False, edge=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fill, ec=edge or fs.INK_SECONDARY, lw=0.8, zorder=2,
                                mutation_aspect=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=fs.INK, weight=weight, zorder=3,
            family="monospace" if mono else "serif", linespacing=1.25)


def _arrow(ax, points, color=fs.INK_SECONDARY, label=None, label_at=None, dashed=False, both=False):
    """A polyline arrow through `points` (data coordinates); the head is on the last point."""
    xs, ys = zip(*points)
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=1.0, zorder=1, ls="--" if dashed else "-",
                solid_capstyle="butt")
    ax.annotate("", xy=points[-1], xytext=points[-2], zorder=1,
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=color, lw=1.0,
                                linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0, mutation_scale=9))
    if label:
        x, y, ha = label_at
        ax.text(x, y, label, ha=ha, va="center", fontsize=fs.BASE_SIZE - 0.5, color=color, style="italic",
                zorder=3, bbox=dict(fc="white", ec="none", pad=0.6))


def _canvas(width, height, xmax, ymax):
    fs.use()
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, xmax), ax.set_ylim(0, ymax), ax.set_aspect("equal"), ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


# ---------------------------------------------------------------- Figure 2: the FEC workflow

def fec_workflow(out_dir: Path) -> Path:
    """FEC around the fitness evaluation: a hit reuses a stored fitness; a miss evaluates on all training rows
    and stores the result. Every stored fitness therefore comes from a full evaluation."""
    fig, ax = _canvas(7.0, 3.05, 116, 50.5)
    W, H = 19, 8.5
    ink, hit, miss = fs.INK_SECONDARY, "#13805a", "#b9461a"

    # once per run (left, bottom): the training rows and the fingerprint rows drawn from them
    ax.add_patch(FancyBboxPatch((1.5, 1.2), 46, 13.5, boxstyle="round,pad=0,rounding_size=1.2", fc="none",
                                ec=fs.INK_SECONDARY, lw=0.7, ls=(0, (3, 2)), zorder=0))
    ax.text(2.6, 13.4, "once per run", fontsize=fs.BASE_SIZE - 1, color=fs.INK_SECONDARY, style="italic",
            va="center")
    _box(ax, 12.5, 7, W, H, "training rows ($N$);\ntest rows unused")
    _box(ax, 36, 7, W, H, "$m=\\rho N$ fingerprint\nrows, at random")
    _arrow(ax, [(22, 7), (26.5, 7)])

    # per candidate (middle row)
    _box(ax, 12.5, 26, W, H, "new program $h$\n(GE or GP)")
    _box(ax, 36, 26, W, H, "run $h$ on the $m$\nfingerprint rows")
    _box(ax, 59.5, 26, W, H, "key = hash of\nthe $m$ outputs")
    _arrow(ax, [(22, 26), (26.5, 26)])
    _arrow(ax, [(45.5, 26), (50, 26)])
    _arrow(ax, [(36, 11.25), (36, 21.75)])                    # fingerprint rows feed the key step

    # decision and cache
    dx, dy, r = 80.5, 26, 6.2
    ax.add_patch(Polygon([(dx - r * 1.35, dy), (dx, dy + r), (dx + r * 1.35, dy), (dx, dy - r)], closed=True,
                         fc="white", ec=fs.INK_SECONDARY, lw=0.8, zorder=2))
    ax.text(dx, dy, "key in\ncache?", ha="center", va="center", fontsize=fs.BASE_SIZE, color=fs.INK, zorder=3)
    _arrow(ax, [(69, 26), (dx - r * 1.35, 26)])
    cx, cy, cw, ch = 105, 26, 17, 9
    ax.add_patch(Rectangle((cx - cw / 2, cy - ch / 2), cw, ch, fc=fs.FILL_ACCENT, ec="none", zorder=2))
    for yy, z in ((cy - ch / 2, 2), (cy + ch / 2, 3)):
        ax.add_patch(Ellipse((cx, yy), cw, 3.2, fc=fs.FILL_ACCENT, ec=fs.INK_SECONDARY, lw=0.8, zorder=z))
    ax.plot([cx - cw / 2] * 2, [cy - ch / 2, cy + ch / 2], color=fs.INK_SECONDARY, lw=0.8, zorder=3)
    ax.plot([cx + cw / 2] * 2, [cy - ch / 2, cy + ch / 2], color=fs.INK_SECONDARY, lw=0.8, zorder=3)
    ax.text(cx, cy - 1.0, "cache\nkey $\\rightarrow$ fitness", ha="center", va="center", fontsize=fs.BASE_SIZE,
            color=fs.INK, zorder=4, linespacing=1.3)
    _arrow(ax, [(dx + r * 1.35, 26), (cx - cw / 2, 26)], both=True)

    # hit (up) and miss (down)
    _box(ax, dx, 43, 21, H, "hit: return the\nstored fitness", fill=fs.FILL_HIT, edge=hit)
    _arrow(ax, [(dx, dy + r), (dx, 38.75)], color=hit, label="yes", label_at=(dx + 1.2, 35.3, "left"))
    _box(ax, dx, 7, 21, H, "miss: evaluate $h$ on\nall $N$ training rows", fill=fs.FILL_MISS, edge=miss)
    _arrow(ax, [(dx, dy - r), (dx, 11.25)], color=miss, label="no", label_at=(dx + 1.2, 16.6, "left"))
    _box(ax, cx, 7, cw, H, "store the fitness\nunder the key", fill=fs.FILL_MISS, edge=miss)
    _arrow(ax, [(dx + 10.5, 7), (cx - cw / 2, 7)], color=miss)
    _arrow(ax, [(cx, 11.25), (cx, cy - ch / 2 - 1.6)], color=miss)

    # both results go back to selection
    _box(ax, 12.5, 43, W, H, "selection and\nvariation")
    _arrow(ax, [(dx - 10.5, 43), (22, 43)], color=hit, label="fitness", label_at=(46, 43, "center"))
    _arrow(ax, [(cx + cw / 2, 7), (114.8, 7), (114.8, 49.6), (16.5, 49.6), (16.5, 47.25)], color=miss,
           label="fitness", label_at=(46, 49.6, "center"))
    _arrow(ax, [(12.5, 38.75), (12.5, 30.25)])
    return fs.save(fig, Path(out_dir) / "fig_fec_workflow.pdf")


# ---------------------------------------------------------------- Figure 1: GE and GP

GENOME = [13, 35, 24, 8, 42, 9, 16, 22, 59, 77, 31, 18, 53, 75, 140, 66]
PHENOTYPE = "greater_than_or_equal(add(x[0],x[8]),+0.35)"
STEPS = [  # (codon, rule, choices, chosen production) for the first steps of the derivation
    (13, "<bool>", 6, "greater_than_or_equal(<num>,<num>)"),
    (35, "<num>", 5, "add(<num>,<num>)"),
    (24, "<num>", 5, "<terminal>"),
    (8, "<terminal>", 2, "<num_feature>"),
    (42, "<num_feature>", 7, "x[0]"),
]


def check_mapping(grammar: Path = ROOT / "grammars" / "electricity_cat.bnf") -> None:
    """The GE panel is a real derivation: fail loudly if the genome no longer maps to PHENOTYPE."""
    from grape import grape
    ind = grape.Individual(GENOME, grape.Grammar(str(grammar)), 35, "lazy")
    if ind.invalid or ind.phenotype != PHENOTYPE:
        raise SystemExit(f"genome maps to {ind.phenotype!r}, not {PHENOTYPE!r}")


def ge_gp(out_dir: Path) -> Path:
    check_mapping()
    fs.use()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2))
    for ax in axes:
        ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.axis("off")

    def box(ax, y, h, text, size=fs.BASE_SIZE, mono=False, fill=fs.FILL, weight="normal"):
        ax.add_patch(FancyBboxPatch((4, y - h / 2), 92, h, boxstyle="round,pad=0,rounding_size=2", fc=fill,
                                    ec=fs.INK_SECONDARY, lw=0.8))
        ax.text(50, y, text, ha="center", va="center", fontsize=size, color=fs.INK, weight=weight,
                family="monospace" if mono else "serif", linespacing=1.3)

    def down(ax, y0, y1):
        ax.annotate("", (50, y1), (50, y0), arrowprops=dict(arrowstyle="-|>", color=fs.INK_SECONDARY, lw=1,
                                                            shrinkA=0, shrinkB=0, mutation_scale=9))

    def evaluation(ax):
        down(ax, 33.5, 27.5)
        box(ax, 20.5, 13, "run on every training row $i = 1,\\dots,N$\n"
                          "True $\\rightarrow$ class 1, False $\\rightarrow$ class 0")
        down(ax, 14, 10)
        box(ax, 5.5, 8, "fitness = training error rate (minimised)", fill=fs.FILL_ACCENT, weight="bold")

    # (a) GE
    ax = axes[0]
    ax.text(0, 99, "(a) Grammatical Evolution", fontsize=fs.BASE_SIZE + 1.5, color=fs.INK, weight="bold", va="top")
    ax.text(4, 91, "genome: integer codons", fontsize=fs.BASE_SIZE, color=fs.INK_SECONDARY, va="bottom")
    cells = GENOME[:14]
    w = 92 / len(cells)
    for k, c in enumerate(cells):
        x = 4 + w * (k + 0.5)
        ax.add_patch(Rectangle((x - w / 2, 82), w, 7.5, fc=fs.FILL_ACCENT if k < 5 else fs.FILL,
                               ec=fs.INK_SECONDARY, lw=0.6))
        ax.text(x, 85.75, str(c), ha="center", va="center", fontsize=fs.BASE_SIZE - 1, family="monospace",
                color=fs.INK)
    down(ax, 81.5, 77.5)
    ax.add_patch(FancyBboxPatch((4, 47.5), 92, 30, boxstyle="round,pad=0,rounding_size=2", fc=fs.FILL,
                                ec=fs.INK_SECONDARY, lw=0.8))
    ax.text(50, 73.5, "codon mod (number of choices) picks a production", ha="center", va="center",
            fontsize=fs.BASE_SIZE - 0.5, color=fs.INK)
    lines = [f"{c:>2} % {n} = {c % n}   {r} -> {p}" for c, r, n, p in STEPS]
    ax.text(7, 59.5, "\n".join(lines + ["..."]), ha="left", va="center", fontsize=fs.BASE_SIZE - 2.3,
            family="monospace", color=fs.INK, linespacing=1.4)
    down(ax, 47, 43.5)
    box(ax, 38.5, 9.5, "program (phenotype):\n" + PHENOTYPE, size=fs.BASE_SIZE - 1.3, mono=True)
    evaluation(ax)

    # (b) GP
    ax = axes[1]
    ax.text(0, 99, "(b) Strongly typed Genetic Programming", fontsize=fs.BASE_SIZE + 1.5, color=fs.INK,
            weight="bold", va="top")
    ax.text(4, 91, "genome: the program tree itself", fontsize=fs.BASE_SIZE, color=fs.INK_SECONDARY,
            va="bottom")
    nodes = {"ge": (50, 84, "$\\geq$", "bool"), "add": (26, 68, "add", "num"), "c": (70, 68, "+0.35", "num"),
             "x0": (14, 52.5, "x[0]", "num"), "x8": (38, 52.5, "x[8]", "num")}
    for a, b in [("ge", "add"), ("ge", "c"), ("add", "x0"), ("add", "x8")]:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=fs.INK_SECONDARY, lw=0.9, zorder=1)
    for x, y, label, kind in nodes.values():
        ax.add_patch(FancyBboxPatch((x - 7, y - 3.5), 14, 7, boxstyle="round,pad=0,rounding_size=2.5",
                                    fc=fs.FILL_ACCENT if kind == "bool" else fs.FILL, ec=fs.INK_SECONDARY, lw=0.8,
                                    zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=fs.BASE_SIZE + 0.5, color=fs.INK, zorder=3)
    for y, label, fc in [(57, "returns a boolean", fs.FILL_ACCENT), (51, "returns a number", fs.FILL)]:
        ax.add_patch(Rectangle((60, y - 1.6), 4, 3.2, fc=fc, ec=fs.INK_SECONDARY, lw=0.6))
        ax.text(66, y, label, fontsize=fs.BASE_SIZE - 1, color=fs.INK_SECONDARY, va="center")
    down(ax, 47, 43.5)
    box(ax, 38.5, 9.5, "program:\n" + PHENOTYPE, size=fs.BASE_SIZE - 1.3, mono=True)
    evaluation(ax)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08)
    return fs.save(fig, Path(out_dir) / "fig_ge_gp.pdf")


def build(out_dir: Path) -> list[Path]:
    return [ge_gp(out_dir), fec_workflow(out_dir)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python diagrams.py <out_dir>")
    for p in build(Path(sys.argv[1])):
        print(p)
