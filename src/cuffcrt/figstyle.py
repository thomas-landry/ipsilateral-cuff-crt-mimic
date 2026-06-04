"""Single shared matplotlib style for the manuscript figures.

One source of truth for typography, palette, axis treatment, panel labels, and
output (vector PDF plus high-DPI PNG). Import this module from the figure
builder so the whole set reads as one consistent figure family.

Design intent for a plain feasibility/prevalence paper aimed at a physician
reader: a calm neutral base, colorblind-safe accents (Okabe-Ito derived),
clean despined axes with a faint horizontal grid, and editable vector text so a
typesetter can recolor or relabel. No internal jargon appears anywhere in this
module; figure semantics (excluded versus usable, occlusion versus
reperfusion) carry the color meaning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm

# --- Palette (Okabe-Ito colorblind-safe accents plus quiet neutrals) ---------
# A single saturated accent marks the usable signal; excluded/non-usable states
# read in the muted vermillion. Phase shading uses soft washes.
COLOR_USABLE = "#0072B2"      # blue: the usable occlusion-reperfusion signal
COLOR_EXCLUDED = "#D55E00"    # vermillion: excluded / non-usable
COLOR_NEUTRAL = "#009E73"     # green: a present-but-uninformative trace

INK = "#1A1A1A"               # near-black text / strong lines
GRAPHITE = "#4D4D4D"          # secondary text and axis edges
SLATE = "#7A7A7A"             # muted reference lines
MIST = "#B8C2CC"              # hairline grid / faint fills
PANEL_BG = "#F4F6F8"          # neutral box fill (funnel stages)

# Soft phase washes for the clean-candidate panel.
WASH_OCCLUSION = "#FBE6DC"    # occlusion (deep dip) band
WASH_REPERFUSION = "#DEEBF5"  # reperfusion (recovery) envelope band

# Neutral on-canvas display label for the locally hosted multimodal model.
# The manuscript prose calls it "the language model"; figures use the same
# wording so prose and canvases stay in lockstep and a rename is one edit.
# (Internal data keys, CSV column names, and file paths keep their original
# identifiers; only what is *drawn on a canvas* uses this constant.)
MODEL_DISPLAY = "Language model"


def _pick_sans() -> str:
    """Prefer a Helvetica/Arial-family sans; fall back to DejaVu Sans.

    Returns
    -------
    str
        The name of the first available preferred sans-serif family.
    """
    have = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Helvetica", "Arial", "Helvetica Neue", "TeX Gyre Heros", "DejaVu Sans"):
        if candidate in have:
            return candidate
    return "DejaVu Sans"


SANS = _pick_sans()


def apply_style() -> None:
    """Apply the shared rcParams. Call once before building any figure."""
    mpl.rcParams.update(
        {
            # typography
            "font.family": "sans-serif",
            "font.sans-serif": [SANS, "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "figure.titlesize": 12.5,
            "figure.titleweight": "bold",
            # text and line color
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRAPHITE,
            "xtick.color": GRAPHITE,
            "ytick.color": GRAPHITE,
            # axes furniture: despined, faint horizontal grid only
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": MIST,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            # ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            # legend
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": MIST,
            "legend.facecolor": "white",
            "legend.borderpad": 0.5,
            # figure / saving
            "figure.dpi": 130,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.08,
            # editable vector text
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # lines
            "lines.solid_capstyle": "round",
            "lines.antialiased": True,
            "patch.linewidth": 0.8,
        }
    )


def panel_label(ax: mpl.axes.Axes, letter: str, *, dx: float = -0.04, dy: float = 0.03) -> None:
    """Stamp a bold panel letter (A/B/C) in axes-fraction coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    letter : str
        Panel letter to draw.
    dx, dy : float
        Offset from the top-left corner, in axes fraction.
    """
    ax.text(
        dx,
        1.0 + dy,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13.0,
        fontweight="bold",
        color=INK,
    )


def save(fig: mpl.figure.Figure, out_dir: Path, slug: str) -> tuple[Path, Path]:
    """Save a figure as both vector PDF and high-DPI PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to write.
    out_dir : pathlib.Path
        Output directory (created if absent).
    slug : str
        File stem (no extension).

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        The written ``(png_path, pdf_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{slug}.png"
    pdf = out_dir / f"{slug}.pdf"
    fig.savefig(png, dpi=400)
    fig.savefig(pdf)
    return png, pdf
