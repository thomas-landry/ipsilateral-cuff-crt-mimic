"""Three-way concordance overview: reader, detector, MedGemma (step 55).

Renders a parallel-categories (parallel-sets) diagram across three vertical
axes, in left-to-right order: blinded human reader, rule-based detector, and
MedGemma. Each of the 568 gallery cards contributes one ribbon that threads
from its reader category, through its detector category, to its MedGemma
category. Ribbons are colored by the reader call only, so the dominant pattern,
cards the reader judged to carry no occlusion-reperfusion signature that the
two machines nonetheless mark present, reads as a wide colored band fanning
from the reader "absent" stack into the machine "present" stacks (machine
over-calling).

Three categories are first-class on every axis where they exist: present,
absent, and indeterminate. The detector emits only present or absent (it has no
indeterminate state), so its axis shows two stacks; the reader and MedGemma
axes show their occupied stacks. Stack heights and ribbon widths are exact card
counts; the total over all ribbons is 568.

A parallel-categories diagram was chosen over an alluvial/Sankey because the
three axes here are unordered nominal categories rather than a flow of a
conserved quantity, and because matplotlib has no native alluvial primitive;
hand-built stacked bars plus cubic-bezier ribbons give a clean, fully editable
vector result with exact, auditable widths. The cell counts that set every
ribbon width are written alongside the figure so the diagram is auditable.

Inputs
------
``--reader_csv``
    ``results/gallery/reader_form_blinded.csv`` (blinded reader call per card).
``--detector_csv``
    ``results/gallery/gallery_manifest.csv`` (``is_occlusion_signature``).
``--medgemma_csv``
    ``results/medgemma_galleryrender/gallery_render_calls_cardkeyed.csv``.

Outputs
-------
``--out_dir/fig_concordance_alluvial.png`` and ``.pdf`` (via
:func:`cuffcrt.figstyle.save`), plus
``--out_dir/fig_concordance_counts.csv`` (the reader x detector x MedGemma cell
counts that define the ribbon widths).

Examples
--------
::

    uv run python scripts/55_concordance_overview.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger

from cuffcrt import figstyle

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_READER = DEFAULT_REPO / "results/gallery/reader_form_blinded.csv"
DEFAULT_DETECTOR = DEFAULT_REPO / "results/gallery/gallery_manifest.csv"
DEFAULT_MEDGEMMA = (
    DEFAULT_REPO / "results/medgemma_galleryrender/gallery_render_calls_cardkeyed.csv"
)
DEFAULT_OUT = DEFAULT_REPO / "figures"

N_EXPECTED = 568

# Canonical three-class vocabulary used on every axis.
PRESENT = "present"
ABSENT = "absent"
INDETERMINATE = "indeterminate"
CLASS_ORDER = (PRESENT, INDETERMINATE, ABSENT)  # top-to-bottom on each axis
CLASS_LABEL = {
    PRESENT: "Present",
    INDETERMINATE: "Indeterminate",
    ABSENT: "Absent",
}

# Raw-call vocabulary -> canonical three-class labels.
CALL_TO_CLASS = {
    "occlusion_signature_present": PRESENT,
    "no_occlusion_signature": ABSENT,
    "indeterminate": INDETERMINATE,
}

AXES_ORDER = ("reader", "detector", "medgemma")
AXIS_LABEL = {
    "reader": "Human reader\n(blinded)",
    "detector": "Rule-based\ndetector",
    "medgemma": figstyle.MODEL_DISPLAY,
}

# Ribbons are colored by the READER call (the reference frame for over- and
# under-calling). Three colorblind-safe classes only; no rainbow.
READER_COLOR = {
    PRESENT: figstyle.COLOR_USABLE,    # blue: reader saw a signature
    ABSENT: figstyle.COLOR_EXCLUDED,   # vermillion: reader saw none
    INDETERMINATE: figstyle.SLATE,     # neutral gray: reader undecided
}

# A distinct hatch per reader class so the three ribbon families stay
# distinguishable in pure grayscale (forward diagonals, dots, back diagonals),
# not only by the Okabe-Ito color. The hatch lines are drawn in a single dark
# ink regardless of fill, so the pattern, not the tone, carries the distinction
# in black-and-white.
READER_HATCH = {
    PRESENT: "////",
    ABSENT: "\\\\\\\\",
    INDETERMINATE: "....",
}
_HATCH_EDGE = figstyle.INK         # hatch line color (grayscale-safe)
_HATCH_LINEWIDTH = 0.5             # hatch stroke weight (set via rcParam)
_HATCH_ALPHA = 0.85                # hatch overlay opacity (darker than the fill)

# Geometry of the three vertical axes (figure-data coordinates 0..1 vertical).
_X = {"reader": 0.06, "detector": 0.50, "medgemma": 0.94}
_BAR_HALFWIDTH = 0.018          # half-width of each category bar
_GAP_FRAC = 0.045               # vertical gap between stacked categories
_RIBBON_ALPHA = 0.55


def load_three_way(
    reader_csv: Path,
    detector_csv: Path,
    medgemma_csv: Path,
    *,
    expected_total: int = N_EXPECTED,
) -> pl.DataFrame:
    """Join the three sources on ``card_id`` and map to canonical classes.

    Parameters
    ----------
    reader_csv, detector_csv, medgemma_csv : pathlib.Path
        The blinded reader form, the gallery manifest (detector), and the
        card-keyed MedGemma calls.
    expected_total : int, optional
        The number of jointly present cards the inner join must yield. Defaults
        to the production value (568); tests pass a smaller fixture size.

    Returns
    -------
    polars.DataFrame
        One row per card with columns ``card_id``, ``reader``, ``detector``,
        ``medgemma``, each a canonical class string.

    Raises
    ------
    ValueError
        If the inner join does not yield exactly ``expected_total`` rows, if any
        class is null after mapping, or if an unexpected raw call is seen.
    """
    reader = (
        pl.read_csv(reader_csv)
        .select("card_id", "call")
        .with_columns(pl.col("call").replace_strict(CALL_TO_CLASS).alias("reader"))
        .select("card_id", "reader")
    )
    detector = (
        pl.read_csv(detector_csv)
        .select("card_id", "is_occlusion_signature")
        .with_columns(
            pl.when(pl.col("is_occlusion_signature"))
            .then(pl.lit(PRESENT))
            .otherwise(pl.lit(ABSENT))
            .alias("detector")
        )
        .select("card_id", "detector")
    )
    medgemma = (
        pl.read_csv(medgemma_csv)
        .select("card_id", "call")
        .with_columns(pl.col("call").replace_strict(CALL_TO_CLASS).alias("medgemma"))
        .select("card_id", "medgemma")
    )

    merged = reader.join(detector, on="card_id", how="inner").join(
        medgemma, on="card_id", how="inner"
    )
    if merged.height != expected_total:
        raise ValueError(
            f"expected {expected_total} jointly present cards, got {merged.height}"
        )
    nulls = merged.null_count().sum_horizontal().item()
    if nulls:
        raise ValueError(f"{nulls} null class values after mapping; check raw calls")
    return merged


def cell_counts(
    merged: pl.DataFrame, *, expected_total: int = N_EXPECTED
) -> pl.DataFrame:
    """Return the reader x detector x MedGemma cell counts.

    Parameters
    ----------
    merged : polars.DataFrame
        Output of :func:`load_three_way`.
    expected_total : int, optional
        The total the cell counts must sum to (the ribbon-width invariant).
        Defaults to the production value (568); tests pass a smaller size.

    Returns
    -------
    polars.DataFrame
        Columns ``reader``, ``detector``, ``medgemma``, ``n`` sorted by the
        canonical class order, with ``n`` summing to ``expected_total``.
    """
    counts = (
        merged.group_by("reader", "detector", "medgemma")
        .len(name="n")
        .with_columns(
            pl.col("reader").replace_strict(
                {c: i for i, c in enumerate(CLASS_ORDER)}
            ).alias("_r"),
            pl.col("detector").replace_strict(
                {c: i for i, c in enumerate(CLASS_ORDER)}
            ).alias("_d"),
            pl.col("medgemma").replace_strict(
                {c: i for i, c in enumerate(CLASS_ORDER)}
            ).alias("_m"),
        )
        .sort("_r", "_d", "_m")
        .drop("_r", "_d", "_m")
    )
    total = int(counts["n"].sum())
    if total != expected_total:
        raise ValueError(f"cell counts sum to {total}, expected {expected_total}")
    return counts


def _stack_layout(
    merged: pl.DataFrame,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compute the vertical extent (y0, y1) of every category bar per axis.

    Each axis fills the unit vertical span, with present at top and absent at
    the bottom, separated by fixed gaps. Bar heights are proportional to the
    card count in that category on that axis.

    Returns
    -------
    dict
        ``layout[axis][class] = (y0, y1)`` in data coordinates (0..1).
    """
    layout: dict[str, dict[str, tuple[float, float]]] = {}
    for axis in AXES_ORDER:
        col = merged[axis]
        present_classes = [c for c in CLASS_ORDER if (col == c).sum() > 0]
        counts = {c: int((col == c).sum()) for c in present_classes}
        n_gaps = max(len(present_classes) - 1, 0)
        usable = 1.0 - n_gaps * _GAP_FRAC
        total = sum(counts.values())
        y_top = 1.0
        spans: dict[str, tuple[float, float]] = {}
        for c in present_classes:
            h = usable * counts[c] / total
            spans[c] = (y_top - h, y_top)
            y_top = y_top - h - _GAP_FRAC
        layout[axis] = spans
    return layout


def _ribbon(
    ax: plt.Axes,
    x_left: float,
    x_right: float,
    yl0: float,
    yl1: float,
    yr0: float,
    yr1: float,
    color: str,
    hatch: str | None = None,
) -> None:
    """Draw one cubic-bezier ribbon between two axis bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x_left, x_right : float
        Inner edges of the left and right bars.
    yl0, yl1 : float
        Bottom and top of the ribbon footprint on the left bar.
    yr0, yr1 : float
        Bottom and top of the ribbon footprint on the right bar.
    color : str
        Fill color (reader-call keyed).
    hatch : str or None
        Hatch pattern (reader-call keyed) drawn over the fill so the ribbon
        class survives grayscale printing. ``None`` leaves the ribbon unhatched.
    """
    cx = (x_left + x_right) / 2.0
    verts = [
        (x_left, yl0),
        (cx, yl0),
        (cx, yr0),
        (x_right, yr0),
        (x_right, yr1),
        (cx, yr1),
        (cx, yl1),
        (x_left, yl1),
        (x_left, yl0),
    ]
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CLOSEPOLY,
    ]
    ribbon_path = mpath.Path(verts, codes)
    # Translucent color fill (the at-a-glance reader-color encoding).
    ax.add_patch(
        mpatches.PathPatch(
            ribbon_path,
            facecolor=color,
            edgecolor="none",
            alpha=_RIBBON_ALPHA,
            zorder=1,
        )
    )
    # Separate hatch-only overlay at higher opacity so the per-class pattern
    # reads in pure grayscale without darkening the color fill. The overlay has
    # no fill (only the hatch lines, in dark ink) and no solid border.
    if hatch:
        ax.add_patch(
            mpatches.PathPatch(
                ribbon_path,
                facecolor="none",
                edgecolor=_HATCH_EDGE,
                linewidth=0.0,
                hatch=hatch,
                alpha=_HATCH_ALPHA,
                zorder=1,
            )
        )


def build_figure(merged: pl.DataFrame) -> plt.Figure:
    """Build the three-axis parallel-categories concordance figure.

    Parameters
    ----------
    merged : polars.DataFrame
        Output of :func:`load_three_way`.

    Returns
    -------
    matplotlib.figure.Figure
        The parallel-categories figure (single axes, no canvas title).
    """
    figstyle.apply_style()
    # Keep hatch strokes fine so the pattern reads without crowding the ribbons.
    plt.rcParams["hatch.linewidth"] = _HATCH_LINEWIDTH
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 1.10)
    ax.axis("off")

    layout = _stack_layout(merged)
    rows = merged.to_dicts()

    # Every ribbon is keyed by the full (reader, detector, medgemma) triple so
    # each segment carries exactly one reader color. Within every category bar,
    # ribbon footprints are stacked top-down in canonical reader order first,
    # then by the canonical class order of the other endpoint. Using the same
    # global ordering at both ends of a transition keeps the two stacks
    # consistent (so a triple's left and right footprints line up) and groups
    # each reader color into a contiguous band, which is what makes reader
    # over-calling read at a glance.
    triple_counts: dict[tuple[str, str, str], int] = {}
    for r in rows:
        key = (r["reader"], r["detector"], r["medgemma"])
        triple_counts[key] = triple_counts.get(key, 0) + 1

    for left_axis, right_axis in (("reader", "detector"), ("detector", "medgemma")):
        x_left = _X[left_axis] + _BAR_HALFWIDTH
        x_right = _X[right_axis] - _BAR_HALFWIDTH

        # Cursors: fraction of each bar already consumed, from the top down.
        left_used: dict[str, float] = {c: 0.0 for c in layout[left_axis]}
        right_used: dict[str, float] = {c: 0.0 for c in layout[right_axis]}

        # Deterministic global order over triples: reader, then detector, then
        # medgemma in canonical class order. This same order drives both the
        # left and right cursors, so the stacks stay aligned.
        for rd in CLASS_ORDER:
            for dt in CLASS_ORDER:
                for mg in CLASS_ORDER:
                    n = triple_counts.get((rd, dt, mg), 0)
                    if n == 0:
                        continue
                    lc = rd if left_axis == "reader" else dt
                    rc = dt if left_axis == "reader" else mg
                    if lc not in layout[left_axis] or rc not in layout[right_axis]:
                        continue

                    l_y0, l_y1 = layout[left_axis][lc]
                    l_height = l_y1 - l_y0
                    l_total = max(int((merged[left_axis] == lc).sum()), 1)
                    frac_l = n / l_total
                    yl_top = l_y1 - left_used[lc]
                    yl_bot = yl_top - frac_l * l_height
                    left_used[lc] += frac_l * l_height

                    r_y0, r_y1 = layout[right_axis][rc]
                    r_height = r_y1 - r_y0
                    r_total = max(int((merged[right_axis] == rc).sum()), 1)
                    frac_r = n / r_total
                    yr_top = r_y1 - right_used[rc]
                    yr_bot = yr_top - frac_r * r_height
                    right_used[rc] += frac_r * r_height

                    _ribbon(
                        ax,
                        x_left,
                        x_right,
                        yl_bot,
                        yl_top,
                        yr_bot,
                        yr_top,
                        READER_COLOR[rd],
                        hatch=READER_HATCH[rd],
                    )

    # Draw the category bars on top, labeled with class + count.
    for axis in AXES_ORDER:
        x = _X[axis]
        for c in CLASS_ORDER:
            if c not in layout[axis]:
                continue
            y0, y1 = layout[axis][c]
            n = int((merged[axis] == c).sum())
            ax.add_patch(
                mpatches.Rectangle(
                    (x - _BAR_HALFWIDTH, y0),
                    2 * _BAR_HALFWIDTH,
                    y1 - y0,
                    facecolor=figstyle.INK,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
            )
            # Class label + count just outside the bar (right axis labels left;
            # others labeled on the side with more room).
            ymid = (y0 + y1) / 2.0
            label_bbox = {
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            }
            if axis == "medgemma":
                ax.text(
                    x + _BAR_HALFWIDTH + 0.012,
                    ymid,
                    f"{CLASS_LABEL[c]}\n{n}",
                    ha="left",
                    va="center",
                    fontsize=8.0,
                    color=figstyle.INK,
                    zorder=4,
                    bbox=label_bbox,
                )
            else:
                ax.text(
                    x - _BAR_HALFWIDTH - 0.012,
                    ymid,
                    f"{CLASS_LABEL[c]}\n{n}",
                    ha="right",
                    va="center",
                    fontsize=8.0,
                    color=figstyle.INK,
                    zorder=4,
                    bbox=label_bbox,
                )

    # Axis titles along the top.
    for axis in AXES_ORDER:
        ax.text(
            _X[axis],
            1.075,
            AXIS_LABEL[axis],
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color=figstyle.INK,
        )

    # Legend: ribbons keyed by reader call, by BOTH color and hatch, so the
    # legend doubles as the grayscale key. Each swatch carries the same fill
    # color and the same hatch pattern as its ribbon family.
    handles = [
        mpatches.Patch(
            facecolor=READER_COLOR[c],
            edgecolor=_HATCH_EDGE,
            linewidth=0.0,
            hatch=READER_HATCH[c],
            alpha=_RIBBON_ALPHA,
            label=f"Reader: {CLASS_LABEL[c].lower()}",
        )
        for c in (PRESENT, INDETERMINATE, ABSENT)
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=True,
        title="Ribbon color and hatch = blinded reader call",
        title_fontsize=8.0,
    )
    fig.tight_layout()
    return fig


def run(
    *,
    reader_csv: Path,
    detector_csv: Path,
    medgemma_csv: Path,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Load, build, and save the concordance figure and its cell-count CSV.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        ``(png, pdf, counts_csv)``.
    """
    merged = load_three_way(reader_csv, detector_csv, medgemma_csv)
    counts = cell_counts(merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_csv = out_dir / "fig_concordance_counts.csv"
    counts.write_csv(counts_csv)

    fig = build_figure(merged)
    png, pdf = figstyle.save(fig, out_dir, "fig_concordance_alluvial")
    plt.close(fig)
    logger.info("wrote figure {} and {}", png, pdf)
    logger.info(
        "wrote cell counts {} ({} cells, sum={})",
        counts_csv,
        counts.height,
        int(counts["n"].sum()),
    )
    return png, pdf, counts_csv


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Render the three-way (reader / detector / MedGemma) concordance "
            "parallel-categories figure and write its cell counts."
        )
    )
    p.add_argument("--reader_csv", type=Path, default=DEFAULT_READER)
    p.add_argument("--detector_csv", type=Path, default=DEFAULT_DETECTOR)
    p.add_argument("--medgemma_csv", type=Path, default=DEFAULT_MEDGEMMA)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_argparser().parse_args(argv)
    run(
        reader_csv=args.reader_csv,
        detector_csv=args.detector_csv,
        medgemma_csv=args.medgemma_csv,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
