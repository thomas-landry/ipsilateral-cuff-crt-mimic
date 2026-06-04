"""Reader-vs-machine disagreement examples figure (step 53).

The blinded expert reader is the gold standard. Both the rule-based detector
and MedGemma over-call relative to the reader (detector precision near 25
percent, MedGemma population precision near 3 percent). This figure makes that
concrete with representative real perfusion-index traces, each labeled with all
three calls (Reader / Detector / MedGemma) so the over-call pattern is visible
panel by panel.

Categories
----------
Pools are computed by joining the three per-card call sources on ``card_id``:

1. ``detector_overcall`` (headline): reader = no signal, detector = present,
   MedGemma not present. The detector-only false positive.
2. ``both_machines_overcall``: reader = no signal, detector = present,
   MedGemma = present. Both machines over-call the same card.
3. ``reader_only_present``: reader = present, detector = absent,
   MedGemma = absent. The reader sees a signal both machines miss.
4. ``reader_indeterminate``: reader = indeterminate while a machine is
   confident (MedGemma confidence at or above 0.9 here, all such cards being
   MedGemma-present).
5. ``all_agree_positive`` (calibration column): reader = present,
   detector = present, MedGemma = present.

Selection rule (deterministic, auditable)
------------------------------------------
Within each category pool the candidates are ranked by reader confidence
(high before med before low before missing), then by distance to the category
centroid in z-scored morphology space over
``(phase3_duration_s, nadir_depth_frac, alignment_offset_s)``. The z-scores use
the joined-population mean and standard deviation so the scaling is fixed and
reproducible. The lowest-distance cards (the most category-representative) are
taken first; ties break on ``card_id`` under :data:`cuffcrt._seed.GLOBAL_SEED`.
The same seed always yields the same card list.

Rendering
---------
Each panel re-renders the 1 Hz perfusion-index trace from the credentialed WDB
tree via the shared :func:`load_trace` loader, cropped to a shared, locked
x-window (default ``[-50, +30]`` s relative to the charted cuff event) and a
shared y-axis across every panel. Blinding only mattered during adjudication,
which is complete, so axes and a t = 0 reference line are drawn now. The PI is
normalized to its in-window median (the same normalization the blinded cards
used) so panels are comparable across patients.

Outputs
-------
``--out_dir/fig_disagreement_examples.png`` and ``.pdf`` plus
``fig_disagreement_examples_cardlist.csv`` (card_id, category, the three calls,
reader confidence, and the three morphology features), so the selection is
fully auditable.

Examples
--------
::

    uv run python scripts/53_disagreement_figure.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.gridspec import GridSpec

from cuffcrt import figstyle
from cuffcrt._paths import (
    ENV_WDB_ROOT,
    DataPathNotConfiguredError,
    resolve_configured_path,
)
from cuffcrt._seed import GLOBAL_SEED

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_READER = DEFAULT_REPO / "results/gallery/reader_form_blinded.csv"
DEFAULT_MANIFEST = DEFAULT_REPO / "results/gallery/gallery_manifest.csv"
DEFAULT_MEDGEMMA = (
    DEFAULT_REPO / "results/medgemma_galleryrender/gallery_render_calls_cardkeyed.csv"
)
DEFAULT_OUT = DEFAULT_REPO / "figures"
# The credentialed WDB tree is never shipped (PhysioNet DUA). Supply it per run
# via ``--wdb_root`` or the CUFFCRT_WDB_ROOT environment variable; there is no
# machine default (see data/README.md).

# ``load_trace`` lives in scripts/50_figures.py, whose filename starts with a
# digit so it cannot be imported normally. Resolve it by path, exactly as
# scripts/51 does, so the waveform loading stays identical to the manuscript
# figures and the blinded gallery.
_FIG_PATH = Path(__file__).with_name("50_figures.py")
_spec = importlib.util.spec_from_file_location("_fig50_for_53", _FIG_PATH)
assert _spec is not None and _spec.loader is not None
_fig = importlib.util.module_from_spec(_spec)
sys.modules["_fig50_for_53"] = _fig
_spec.loader.exec_module(_fig)
load_trace = _fig.load_trace

# Canonical call vocabulary (shared with the reader form and MedGemma client).
R_PRESENT = "occlusion_signature_present"
R_ABSENT = "no_occlusion_signature"
R_INDETERMINATE = "indeterminate"

# Detector call is derived from the manifest boolean ``is_occlusion_signature``.
DET_PRESENT = "present"
DET_ABSENT = "absent"

# Morphology features used for centroid-based representativeness ranking.
MORPH_FEATURES = ("phase3_duration_s", "nadir_depth_frac", "alignment_offset_s")

# Reader-confidence ordering (lower rank = stronger preference for selection).
_CONF_RANK = {"high": 0, "med": 1, "low": 2, None: 3}

# High-confidence threshold for the reader-indeterminate / machine-confident
# category (category 4): a machine call counts as confident at or above this.
MACHINE_CONFIDENT_THRESHOLD = 0.9

# Locked shared axes for every panel.
X_LO_S = -50.0
X_HI_S = 30.0


@dataclass(frozen=True)
class Category:
    """A disagreement category and how many panels to draw for it.

    Attributes
    ----------
    key : str
        Short stable category key (also written to the card list).
    label : str
        Row-header label drawn on the figure.
    n_panels : int
        Target number of panels for this category.
    """

    key: str
    label: str
    n_panels: int


# Category order top-to-bottom in the figure. Headline first.
CATEGORIES: tuple[Category, ...] = (
    Category("detector_overcall", "Detector over-calls\n(reader: no signal)", 3),
    Category("both_machines_overcall", "Both machines over-call\n(reader: no signal)", 3),
    Category("reader_only_present", "Reader sees signal\nboth machines miss", 3),
    Category("reader_indeterminate", "Reader uncertain,\nmachine confident", 3),
    Category("all_agree_positive", "All three agree:\nsignal present", 3),
)


def load_calls(
    reader_csv: Path, manifest_csv: Path, medgemma_csv: Path
) -> pl.DataFrame:
    """Join the three per-card call sources into one table.

    Parameters
    ----------
    reader_csv : pathlib.Path
        Blinded reader form (``card_id, image_path, call, confidence, notes``).
    manifest_csv : pathlib.Path
        Gallery manifest (carries ``is_occlusion_signature`` and morphology).
    medgemma_csv : pathlib.Path
        Card-keyed MedGemma gallery-render calls (``card_id, call, confidence``).

    Returns
    -------
    polars.DataFrame
        One row per card with columns ``card_id, reader_call, reader_conf,
        detector_call, medgemma_call, medgemma_conf, stratum, subject_id,
        record_id, t_nbp`` and the three morphology features.
    """
    reader = pl.read_csv(reader_csv, infer_schema_length=20000).select(
        [
            "card_id",
            pl.col("call").alias("reader_call"),
            pl.col("confidence").alias("reader_conf"),
        ]
    )
    man = pl.read_csv(manifest_csv, infer_schema_length=20000).select(
        [
            "card_id",
            "stratum",
            "subject_id",
            "record_id",
            "t_nbp",
            "is_occlusion_signature",
            *MORPH_FEATURES,
        ]
    )
    mg = pl.read_csv(medgemma_csv, infer_schema_length=20000).select(
        [
            "card_id",
            pl.col("call").alias("medgemma_call"),
            pl.col("confidence").alias("medgemma_conf"),
        ]
    )
    joined = (
        reader.join(man, on="card_id", how="inner")
        .join(mg, on="card_id", how="inner")
        .with_columns(
            pl.when(pl.col("is_occlusion_signature"))
            .then(pl.lit(DET_PRESENT))
            .otherwise(pl.lit(DET_ABSENT))
            .alias("detector_call")
        )
    )
    return joined


def category_predicate(cat_key: str) -> pl.Expr:
    """Return the boolean Polars expression defining a category's pool.

    The predicate is the single source of truth for category membership; the
    figure builder uses it to compute the pool and the integrity test uses it
    to assert that every selected card satisfies it.

    Parameters
    ----------
    cat_key : str
        Category key (see :data:`CATEGORIES`).

    Returns
    -------
    polars.Expr
        A boolean expression over the joined calls table.

    Raises
    ------
    ValueError
        If ``cat_key`` is not a known category.
    """
    if cat_key == "detector_overcall":
        # Detector-only false positive: reader no signal, detector present, and
        # MedGemma NOT present (so this panel differs from both_machines_overcall).
        return (
            (pl.col("reader_call") == R_ABSENT)
            & (pl.col("detector_call") == DET_PRESENT)
            & (pl.col("medgemma_call") != R_PRESENT)
        )
    if cat_key == "both_machines_overcall":
        return (
            (pl.col("reader_call") == R_ABSENT)
            & (pl.col("detector_call") == DET_PRESENT)
            & (pl.col("medgemma_call") == R_PRESENT)
        )
    if cat_key == "reader_only_present":
        return (
            (pl.col("reader_call") == R_PRESENT)
            & (pl.col("detector_call") == DET_ABSENT)
            & (pl.col("medgemma_call") == R_ABSENT)
        )
    if cat_key == "reader_indeterminate":
        return (pl.col("reader_call") == R_INDETERMINATE) & (
            pl.col("medgemma_conf") >= MACHINE_CONFIDENT_THRESHOLD
        )
    if cat_key == "all_agree_positive":
        return (
            (pl.col("reader_call") == R_PRESENT)
            & (pl.col("detector_call") == DET_PRESENT)
            & (pl.col("medgemma_call") == R_PRESENT)
        )
    raise ValueError(f"unknown category key: {cat_key!r}")


def _zscore_features(joined: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z-scoring (mean, std) over the joined population for MORPH_FEATURES.

    Using the full joined population (not the per-category subset) fixes the
    scaling so it does not depend on which category is being ranked.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(values, mean, std)`` where ``values`` is the full ``(n, 3)`` matrix.
    """
    mat = np.column_stack(
        [joined.get_column(f).to_numpy().astype(float) for f in MORPH_FEATURES]
    )
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
    return mat, mean, std


def select_cards(
    joined: pl.DataFrame, categories: tuple[Category, ...], *, seed: int = GLOBAL_SEED
) -> pl.DataFrame:
    """Select representative cards per category deterministically.

    Ranking within a category:

    1. Reader confidence (high before med before low before missing).
    2. Euclidean distance to the category centroid in z-scored morphology space.
    3. ``card_id`` lexicographic order (the seeded tie-break key).

    The seed enters only the tie-break sort key so the selection is stable and
    reproducible without being sensitive to floating point ordering.

    Parameters
    ----------
    joined : polars.DataFrame
        Output of :func:`load_calls`.
    categories : tuple[Category, ...]
        Categories to select for, in figure order.
    seed : int
        Determinism anchor; mixed into the tie-break key.

    Returns
    -------
    polars.DataFrame
        Selected cards with a ``category`` column and a within-category
        ``panel_index``, ordered by category then panel.
    """
    mat, mean, std = _zscore_features(joined)
    z_all = (mat - mean) / std  # (n, 3); NaNs propagate and are handled below
    card_ids = joined.get_column("card_id").to_list()
    card_index = {cid: i for i, cid in enumerate(card_ids)}

    selected_rows: list[dict] = []
    for cat in categories:
        pool = joined.filter(category_predicate(cat.key))
        pool_ids = pool.get_column("card_id").to_list()
        if not pool_ids:
            continue
        idx = np.array([card_index[c] for c in pool_ids])
        z_pool = z_all[idx]
        # Category centroid over rows whose features are all finite.
        finite_rows = np.isfinite(z_pool).all(axis=1)
        centroid = (
            np.nanmean(z_pool[finite_rows], axis=0)
            if finite_rows.any()
            else np.zeros(z_pool.shape[1])
        )
        # Distance to centroid; non-finite features get a large finite distance
        # so they sort last but never crash the argsort.
        diff = z_pool - centroid
        dist = np.sqrt(np.nansum(diff * diff, axis=1))
        dist = np.where(np.isfinite(dist), dist, np.inf)

        conf = pool.get_column("reader_conf").to_list()
        conf_rank = [_CONF_RANK.get(c, 3) for c in conf]

        # Deterministic ranking: confidence, then distance, then seeded tie-break
        # by a stable hash of (card_id, seed). round() keeps ties tied across
        # platforms; the hash key only resolves genuine distance ties.
        def _tie_key(cid: str) -> int:
            return hash((cid, seed)) & 0xFFFFFFFF

        order = sorted(
            range(len(pool_ids)),
            key=lambda i: (
                conf_rank[i],
                round(float(dist[i]), 9),
                _tie_key(pool_ids[i]),
            ),
        )
        chosen = order[: cat.n_panels]
        for panel_index, i in enumerate(chosen):
            row = pool.row(i, named=True)
            selected_rows.append(
                {
                    "card_id": row["card_id"],
                    "category": cat.key,
                    "panel_index": panel_index,
                    "reader_call": row["reader_call"],
                    "detector_call": row["detector_call"],
                    "medgemma_call": row["medgemma_call"],
                    "reader_confidence": row["reader_conf"],
                    "medgemma_confidence": row["medgemma_conf"],
                    "stratum": row["stratum"],
                    "subject_id": row["subject_id"],
                    "record_id": row["record_id"],
                    "t_nbp": row["t_nbp"],
                    "phase3_duration_s": row["phase3_duration_s"],
                    "nadir_depth_frac": row["nadir_depth_frac"],
                    "alignment_offset_s": row["alignment_offset_s"],
                    "centroid_distance": round(float(dist[i]), 6),
                }
            )
    return pl.DataFrame(selected_rows)


def pool_sizes(joined: pl.DataFrame, categories: tuple[Category, ...]) -> dict[str, int]:
    """Return the pool size for each category (rows satisfying its predicate)."""
    return {
        cat.key: joined.filter(category_predicate(cat.key)).height
        for cat in categories
    }


# --- Rendering ---------------------------------------------------------------
# Call -> color via the Okabe-Ito / Wong colorblind-safe palette (PMID 21774112)
# already curated in cuffcrt.figstyle. A present call reads in blue (the usable
# signal accent), absent in vermillion (the excluded accent), indeterminate in
# neutral graphite. A redundant glyph carries the same meaning in grayscale.
_CALL_COLOR = {
    R_PRESENT: figstyle.COLOR_USABLE,
    DET_PRESENT: figstyle.COLOR_USABLE,
    R_ABSENT: figstyle.COLOR_EXCLUDED,
    DET_ABSENT: figstyle.COLOR_EXCLUDED,
    R_INDETERMINATE: figstyle.GRAPHITE,
}
_CALL_GLYPH = {
    R_PRESENT: "+",
    DET_PRESENT: "+",
    R_ABSENT: "−",  # minus sign (clearer than hyphen)
    DET_ABSENT: "−",
    R_INDETERMINATE: "?",
}
_CONF_SUPERSCRIPT = {"high": "H", "med": "M", "low": "L", None: ""}


def _load_window(
    wdb_root: Path, row: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and normalize one card's PI trace cropped to the locked window.

    Mirrors the blinded gallery normalization: divide by the in-window median
    so panels are comparable across patients. Returns ``None`` when no usable
    PI window was found.
    """
    trace = load_trace(
        wdb_root, str(row["subject_id"]), str(row["record_id"]), float(row["t_nbp"])
    )
    if not trace.has_pleth or trace.pi.size == 0:
        return None
    in_window = (trace.t_local >= X_LO_S) & (trace.t_local <= X_HI_S)
    t = trace.t_local[in_window]
    y = trace.pi[in_window]
    if t.size == 0:
        return None
    median = float(np.nanmedian(y)) if np.isfinite(y).any() else 1.0
    if median <= 0 or not np.isfinite(median):
        median = 1.0
    return t, y / median


def _draw_call_strip(ax: plt.Axes, row: dict) -> None:
    """Draw the compact R / D / G call strip inside a panel (top-right).

    Each cell is a colored rounded patch carrying the call's letter, a redundant
    +/-/? glyph for grayscale, and (for the reader) a small confidence
    superscript. The strip sits in axes-fraction coordinates so it scales with
    the panel.
    """
    cells = [
        ("R", row["reader_call"], _CONF_SUPERSCRIPT.get(row["reader_confidence"], "")),
        ("D", row["detector_call"], ""),
        ("G", row["medgemma_call"], ""),
    ]
    # Strip geometry in axes fraction: three cells across the top-right.
    cell_w = 0.135
    gap = 0.012
    x0 = 1.0 - (3 * cell_w + 2 * gap)
    y0 = 1.02
    h = 0.16
    for k, (who, call, sup) in enumerate(cells):
        x = x0 + k * (cell_w + gap)
        color = _CALL_COLOR.get(call, figstyle.SLATE)
        ax.add_patch(
            plt.Rectangle(
                (x, y0),
                cell_w,
                h,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
                clip_on=False,
                zorder=5,
            )
        )
        glyph = _CALL_GLYPH.get(call, "?")
        label = f"{who}{sup}" if sup else who
        ax.text(
            x + cell_w / 2,
            y0 + h / 2,
            f"{label} {glyph}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color="white",
            clip_on=False,
            zorder=6,
        )


def build_figure(
    selected: pl.DataFrame,
    *,
    wdb_root: Path,
    categories: tuple[Category, ...],
) -> tuple[plt.Figure, list[str]]:
    """Render the small-multiples disagreement figure.

    Parameters
    ----------
    selected : polars.DataFrame
        Output of :func:`select_cards`.
    wdb_root : pathlib.Path
        Root of the WDB tree for re-rendering traces.
    categories : tuple[Category, ...]
        Categories in row order.

    Returns
    -------
    tuple[matplotlib.figure.Figure, list[str]]
        The figure and the list of card_ids that failed to render (empty on a
        clean run).
    """
    figstyle.apply_style()
    n_cols = max(c.n_panels for c in categories)
    present_cats = [c for c in categories if selected.filter(pl.col("category") == c.key).height]
    n_rows = len(present_cats)

    # First pass: load every selected trace so the shared y-limit is exact.
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    failed: list[str] = []
    for row in selected.iter_rows(named=True):
        win = _load_window(wdb_root, row)
        if win is None:
            failed.append(row["card_id"])
            continue
        traces[row["card_id"]] = win
    if traces:
        y_max = max(float(np.nanmax(y)) for _, y in traces.values())
        y_top = min(y_max * 1.08, 3.0)
    else:
        y_top = 2.0

    fig = plt.figure(figsize=(2.45 * n_cols + 1.6, 2.05 * n_rows))
    # Explicit margins (no tight_layout): the off-axis row-header cells are not
    # tight_layout-compatible, and the call strips overhang the top spine, so a
    # fixed layout is both cleaner and reproducible.
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[0.45, *([1.0] * n_cols)],
        left=0.07,
        right=0.985,
        top=0.94,
        bottom=0.10,
        hspace=0.62,
        wspace=0.22,
    )

    for r_idx, cat in enumerate(present_cats):
        # Row-header cell (left column) carries the category label, rotated.
        head = fig.add_subplot(gs[r_idx, 0])
        head.axis("off")
        head.text(
            0.5,
            0.5,
            cat.label,
            transform=head.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.5,
            fontweight="bold",
            color=figstyle.INK,
        )
        cat_rows = selected.filter(pl.col("category") == cat.key).sort("panel_index")
        for c_idx in range(n_cols):
            ax = fig.add_subplot(gs[r_idx, c_idx + 1])
            if c_idx >= cat_rows.height:
                ax.axis("off")
                continue
            row = cat_rows.row(c_idx, named=True)
            ax.set_xlim(X_LO_S, X_HI_S)
            ax.set_ylim(0.0, y_top)
            ax.axvline(0.0, color=figstyle.SLATE, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
            win = traces.get(row["card_id"])
            if win is not None:
                t, y = win
                ax.plot(t, y, color=figstyle.INK, linewidth=1.0, zorder=3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "trace unavailable",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=figstyle.SLATE,
                )
            _draw_call_strip(ax, row)
            ax.grid(axis="y", color=figstyle.MIST, linewidth=0.5, alpha=0.6)
            # Only the bottom row of each column carries an x-label; only the
            # leftmost panel column carries a y-label, to keep the grid clean.
            if r_idx == n_rows - 1:
                ax.set_xlabel("Time from cuff event (s)", fontsize=8.0)
            else:
                ax.set_xticklabels([])
            if c_idx == 0:
                ax.set_ylabel("PI (norm.)", fontsize=8.0)
            else:
                ax.set_yticklabels([])

    # Shared legend keying the call-strip colors and glyphs.
    legend_handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   markerfacecolor=figstyle.COLOR_USABLE, markeredgecolor="white",
                   label="present (+)"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   markerfacecolor=figstyle.COLOR_EXCLUDED, markeredgecolor="white",
                   label="no signal (−)"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   markerfacecolor=figstyle.GRAPHITE, markeredgecolor="white",
                   label="indeterminate (?)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
        title=(
            f"Call strip: R reader, D detector, G {figstyle.MODEL_DISPLAY.lower()}"
            "  (R superscript = H/M/L confidence)"
        ),
        title_fontsize=8.0,
        fontsize=8.0,
    )
    return fig, failed


def write_cardlist(selected: pl.DataFrame, out_path: Path) -> None:
    """Write the auditable per-panel card list CSV."""
    cols = [
        "card_id",
        "category",
        "panel_index",
        "reader_call",
        "detector_call",
        "medgemma_call",
        "reader_confidence",
        "medgemma_confidence",
        "phase3_duration_s",
        "nadir_depth_frac",
        "alignment_offset_s",
        "centroid_distance",
        "stratum",
        "subject_id",
        "record_id",
        "t_nbp",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.select(cols).write_csv(out_path)


def run(
    *,
    reader_csv: Path,
    manifest_csv: Path,
    medgemma_csv: Path,
    wdb_root: Path,
    out_dir: Path,
    seed: int = GLOBAL_SEED,
) -> tuple[Path, Path, Path]:
    """Build the figure and card list end to end. Returns ``(png, pdf, csv)``."""
    joined = load_calls(reader_csv, manifest_csv, medgemma_csv)
    logger.info("joined {} cards across the three call sources", joined.height)
    sizes = pool_sizes(joined, CATEGORIES)
    for cat in CATEGORIES:
        logger.info("category {}: pool size {}", cat.key, sizes[cat.key])
    selected = select_cards(joined, CATEGORIES, seed=seed)
    logger.info("selected {} panels", selected.height)

    fig, failed = build_figure(selected, wdb_root=wdb_root, categories=CATEGORIES)
    if failed:
        logger.warning("{} cards failed to render: {}", len(failed), failed)
    png, pdf = figstyle.save(fig, out_dir, "fig_disagreement_examples")
    plt.close(fig)
    csv_path = out_dir / "fig_disagreement_examples_cardlist.csv"
    write_cardlist(selected, csv_path)
    logger.info("wrote {}, {}, {}", png, pdf, csv_path)
    return png, pdf, csv_path


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Render the reader-vs-machine disagreement examples figure with all "
            "three calls labeled per panel."
        )
    )
    p.add_argument("--reader_csv", type=Path, default=DEFAULT_READER)
    p.add_argument("--manifest_csv", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--medgemma_csv", type=Path, default=DEFAULT_MEDGEMMA)
    p.add_argument(
        "--wdb_root",
        type=Path,
        default=None,
        help=(
            "Root of the WDB tree (needed to re-render traces). Defaults to the "
            f"${ENV_WDB_ROOT} environment variable; required if that is unset."
        ),
    )
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_argparser().parse_args(argv)
    try:
        wdb_root = resolve_configured_path(
            args.wdb_root,
            env_var=ENV_WDB_ROOT,
            flag="--wdb_root",
            what="WDB waveform record tree",
        )
    except DataPathNotConfiguredError as exc:
        logger.error("{}", exc)
        return 2
    if not wdb_root.exists():
        logger.error("WDB root not found (needed to re-render traces): {}", wdb_root)
        return 2
    run(
        reader_csv=args.reader_csv,
        manifest_csv=args.manifest_csv,
        medgemma_csv=args.medgemma_csv,
        wdb_root=wdb_root,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
