"""Precision/recall/specificity figure: gallery vs population (step 47).

Renders a point-with-CI (forest-style) comparison of the rule-based detector and
MedGemma against the blinded reader reference, showing both the raw enriched
gallery estimate and the population (inverse-probability reweighted) estimate
for precision, recall, and specificity.

Input
-----
``--summary_csv``
    ``results/precision_recall_population/precision_recall_population_summary.csv``
    (written by ``scripts/46_population_reweight.py``). Long format with columns
    ``predictor, metric, estimate_kind, point_estimate, ci_low, ci_high``.

Output
------
``--out_dir/fig_precision_recall.png`` and ``.pdf`` (via
:func:`cuffcrt.figstyle.save`).

Examples
--------
::

    uv run python scripts/47_fig_precision_recall.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger

from cuffcrt import figstyle

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    DEFAULT_REPO / "results/precision_recall_population/precision_recall_population_summary.csv"
)
DEFAULT_OUT = DEFAULT_REPO / "figures"

METRIC_ORDER = ("precision", "recall", "specificity")
METRIC_LABEL = {
    "precision": "Precision\nP(reader+ | machine+)",
    "recall": "Recall (sensitivity)\nP(machine+ | reader+)",
    "specificity": "Specificity\nP(machine- | reader-)",
}
PREDICTOR_ORDER = ("detector", "medgemma")
PREDICTOR_LABEL = {
    "detector": "Rule-based detector",
    "medgemma": figstyle.MODEL_DISPLAY,
}
# Gallery (enriched sample) vs population (reweighted) within each predictor.
KIND_ORDER = ("gallery", "population")
KIND_LABEL = {
    "gallery": "Enriched gallery sample",
    "population": "Population (reweighted)",
}
KIND_COLOR = {
    "gallery": figstyle.SLATE,
    "population": figstyle.COLOR_USABLE,
}
KIND_MARKER = {"gallery": "o", "population": "D"}

# Population recall for both predictors rests on only two reader-positive cards
# in the heavily weighted detector-negative random stratum, so the point and its
# CI are uninformative. Flag both population-recall markers with this glyph; the
# caption explains what the asterisk means (no on-canvas footnote).
FRAGILE_RECALL_GLYPH = "*"


def _lookup(df: pl.DataFrame, predictor: str, metric: str, kind: str) -> tuple[float, float, float]:
    """Return ``(point, ci_low, ci_high)`` for one cell, or NaNs if absent."""
    sub = df.filter(
        (pl.col("predictor") == predictor)
        & (pl.col("metric") == metric)
        & (pl.col("estimate_kind") == kind)
    )
    if sub.height == 0:
        return float("nan"), float("nan"), float("nan")
    row = sub.row(0, named=True)
    return (
        float(row["point_estimate"]),
        float(row["ci_low"]),
        float(row["ci_high"]),
    )


def build_figure(df: pl.DataFrame) -> plt.Figure:
    """Build the gallery-vs-population precision/recall figure.

    Parameters
    ----------
    df : polars.DataFrame
        The population summary (long format).

    Returns
    -------
    matplotlib.figure.Figure
        A 3-panel figure (one panel per metric), each panel a horizontal
        point-with-CI plot for both predictors and both estimate kinds.
    """
    figstyle.apply_style()
    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(10.5, 4.4), sharex=True)

    # Row layout within each panel: two predictors (detector above MedGemma),
    # each split into a gallery sub-row and a population sub-row offset slightly
    # so the gallery-to-population shift reads as a vertical pair. A thin
    # connector joins the two sub-row points within a predictor.
    base_y = {p: i for i, p in enumerate(PREDICTOR_ORDER)}
    offset = {"gallery": +0.16, "population": -0.16}

    for ax, metric in zip(axes, METRIC_ORDER, strict=True):
        for predictor in PREDICTOR_ORDER:
            cell = {
                kind: _lookup(df, predictor, metric, kind) for kind in KIND_ORDER
            }
            # Thin connector showing the gallery -> population shift direction.
            gx = cell["gallery"][0]
            px = cell["population"][0]
            gy = base_y[predictor] + offset["gallery"]
            py = base_y[predictor] + offset["population"]
            ax.plot(
                [gx, px],
                [gy, py],
                color=figstyle.MIST,
                linewidth=1.0,
                zorder=2,
                solid_capstyle="round",
            )
            for kind in KIND_ORDER:
                point, lo, hi = cell[kind]
                y = base_y[predictor] + offset[kind]
                ax.errorbar(
                    point,
                    y,
                    xerr=[[max(point - lo, 0.0)], [max(hi - point, 0.0)]],
                    fmt=KIND_MARKER[kind],
                    color=KIND_COLOR[kind],
                    markersize=6.5,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    elinewidth=1.6,
                    capsize=3.0,
                    capthick=1.2,
                    zorder=3,
                )
                # Flag the fragile population-recall points with a glyph.
                if metric == "recall" and kind == "population":
                    ax.annotate(
                        FRAGILE_RECALL_GLYPH,
                        xy=(point, y),
                        xytext=(8.0, 4.0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=12.0,
                        fontweight="bold",
                        color=figstyle.INK,
                        zorder=4,
                    )

        ax.set_title(METRIC_LABEL[metric], fontsize=9.5, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(-0.6, len(PREDICTOR_ORDER) - 0.4)
        ax.set_yticks(list(base_y.values()))
        ax.set_yticklabels(
            [PREDICTOR_LABEL[p] for p in PREDICTOR_ORDER]
            if ax is axes[0]
            else [""] * len(PREDICTOR_ORDER)
        )
        ax.invert_yaxis()
        ax.grid(axis="x", color=figstyle.MIST, linewidth=0.6, alpha=0.7)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Estimate (proportion)")

    # No on-canvas interpretation: the population precision value and its
    # interpretation, and the meaning of the fragile-recall asterisk, are
    # carried in the manuscript caption. The asterisk glyph stays on the two
    # population-recall markers above.

    # Shared legend for the two estimate kinds.
    handles = [
        plt.Line2D(
            [],
            [],
            marker=KIND_MARKER[k],
            color=KIND_COLOR[k],
            linestyle="none",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=KIND_LABEL[k],
        )
        for k in KIND_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    # No suptitle: the figure title lives in the manuscript caption.
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.99))
    return fig


def run(*, summary_csv: Path, out_dir: Path) -> tuple[Path, Path]:
    """Load the summary, build, and save the figure. Returns ``(png, pdf)``."""
    df = pl.read_csv(summary_csv, infer_schema_length=20000)
    fig = build_figure(df)
    png, pdf = figstyle.save(fig, out_dir, "fig_precision_recall")
    plt.close(fig)
    logger.info("wrote figure {} and {}", png, pdf)
    return png, pdf


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Render the gallery-vs-population precision/recall/specificity "
            "comparison figure for the detector and MedGemma."
        )
    )
    p.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_argparser().parse_args(argv)
    run(summary_csv=args.summary_csv, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
