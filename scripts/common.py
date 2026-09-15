"""Shared configuration for the analysis scripts.

Figure styling follows the requirements of the Nature portfolio journals: no
titles inside the figure files, panel labels as lowercase bold letters, vector
output, and sans serif text at sizes that stay legible at a single column
width of 89 mm.
"""

from __future__ import annotations

import os

# Cap every numerical library at four threads before NumPy or SciPy is
# imported. The linear algebra here is small and the workload is a large
# number of independent solves, so extra BLAS threads add contention without
# adding speed, and they would otherwise take every core on the machine.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "4")
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIG_DIR = os.path.join(ROOT, "figures")
TAB_DIR = os.path.join(ROOT, "results", "tables")
DATA_DIR = os.path.join(ROOT, "results")

for _d in (FIG_DIR, TAB_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

# Column widths of the Nature portfolio layout, in inches.
SINGLE = 89.0 / 25.4
DOUBLE = 183.0 / 25.4

#: Colour sequence chosen to remain distinguishable in greyscale and under the
#: common forms of colour vision deficiency. Exposed measures are warm, so that
#: the shift of a robust portfolio from warm to cool is legible at a glance.
COLOURS = {
    "afforestation": "#8c3b1f",
    "forest_management": "#c0632c",
    "soil_carbon": "#dc9a4a",
    "wood_products": "#9a8bb5",
    "biochar": "#5f7fa8",
    "beccs": "#2d5f7f",
    "daccs": "#12354d",
    "deterministic": "#5a5a5a",
    "emission": "#5a5a5a",
    "net": "#111111",
    "fire": "#b03a3a",
    "accent": "#b03a3a",
    "exposed": "#c0632c",
    "protected": "#2d5f7f",
    "western_us": "#b5622c",
    "boreal_canada": "#2d6f8f",
    "grid": "#d9d9d9",
}

#: Plotting order and colour for the measures of :mod:`wrcp.regions`, most
#: exposed first, so a stacked composition reads as a risk gradient.
MEASURE_ORDER = [
    "afforestation",
    "forest_management",
    "soil_carbon",
    "wood_products",
    "biochar",
    "beccs",
    "daccs",
]

SEQUENCE = [COLOURS[k] for k in MEASURE_ORDER]


def measure_colours(names):
    """Colours for a list of measure names, in the order given."""
    return [COLOURS.get(n, COLOURS["net"]) for n in names]


def configure() -> None:
    """Apply the shared style. Called by every figure script."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.1,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.5,
            "legend.labelspacing": 0.3,
            "legend.borderpad": 0.2,
            "figure.dpi": 200,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.04) -> None:
    """Place a lowercase bold panel label, per the journal requirement."""
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def light_grid(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=COLOURS["grid"], linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str) -> str:
    """Write a figure as vector PDF, the preferred journal format."""
    path = os.path.join(FIG_DIR, name + ".pdf")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print("  figure -> figures/{}.pdf".format(name))
    return path


def save_table(df, name: str, float_format: str = "%.6g") -> str:
    """Write a results table as comma separated values."""
    path = os.path.join(TAB_DIR, name + ".csv")
    df.to_csv(path, index=False, float_format=float_format)
    print("  table  -> results/tables/{}.csv".format(name))
    return path


def write_values(values: Dict[str, object], name: str = "key_values") -> str:
    """Record scalar results so the manuscript can quote them verbatim.

    Every number stated in the manuscript is written here by the script that
    computes it, which keeps the text and the code in step.
    """
    path = os.path.join(DATA_DIR, name + ".txt")
    existing: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if "=" in line:
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
    for k, v in values.items():
        existing[str(k)] = (
            "{:.6g}".format(v) if isinstance(v, (int, float, np.floating)) else str(v)
        )
    with open(path, "w") as fh:
        for k in sorted(existing):
            fh.write("{} = {}\n".format(k, existing[k]))
    return path


# ---------------------------------------------------------------------------
# Visual enhancements.
#
# These helpers add depth to the plots without changing what is plotted: a
# curve keeps its data and gains a soft halo so it stays legible where several
# curves cross; a filled region gains a vertical gradient so the eye reads its
# extent before its boundary. Everything below is vector output and survives
# the conversion to the print formats.
# ---------------------------------------------------------------------------
import matplotlib.patheffects as _pe
from matplotlib.colors import LinearSegmentedColormap as _LSC
from matplotlib.colors import to_rgb as _to_rgb


def halo(width: float = 1.7, colour: str = "white", alpha: float = 0.85):
    """Path effects giving a line a soft outline, for crossing curves."""
    return [
        _pe.Stroke(linewidth=width, foreground=colour, alpha=alpha),
        _pe.Normal(),
    ]


def glow(colour: str, n: int = 4, base: float = 1.1, spread: float = 0.9):
    """Path effects giving a line a coloured glow, for emphasis."""
    out = []
    for i in range(n, 0, -1):
        out.append(
            _pe.Stroke(
                linewidth=base + spread * i,
                foreground=colour,
                alpha=0.055 * i / n * n,
            )
        )
    out.append(_pe.Normal())
    return out


def fade_cmap(colour: str, name: str = "fade"):
    """A colormap running from transparent to ``colour``."""
    r, g, b = _to_rgb(colour)
    return _LSC.from_list(name, [(r, g, b, 0.0), (r, g, b, 0.85)])


def gradient_fill(ax, x, y, colour: str, alpha: float = 0.55, zorder: int = 1):
    """Fill the area under ``y`` with a vertical gradient in ``colour``.

    The gradient is drawn as an image clipped to the curve, which keeps the
    output vector and avoids the flat wash that a plain fill produces.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(min(0.0, np.nanmin(y)))
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    im = ax.imshow(
        grad,
        extent=(float(x.min()), float(x.max()), y0, float(np.nanmax(y))),
        origin="lower",
        aspect="auto",
        cmap=fade_cmap(colour),
        alpha=alpha,
        zorder=zorder,
    )
    verts = np.column_stack([np.r_[x, x[::-1]], np.r_[y, np.full_like(y, y0)]])
    clip = plt.Polygon(verts, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    return im


def band(ax, x, lo, hi, colour: str, alpha: float = 0.20, zorder: int = 1):
    """A soft band between two curves, for ranges and envelopes."""
    return ax.fill_between(
        x, lo, hi, color=colour, alpha=alpha, lw=0, zorder=zorder
    )


def spine_style(ax, colour: str = "#3a3a3a"):
    """Thin, slightly recessive spines, so the data carries the contrast."""
    for s in ax.spines.values():
        s.set_color(colour)
        s.set_linewidth(0.55)
    ax.tick_params(colors=colour, width=0.55)
