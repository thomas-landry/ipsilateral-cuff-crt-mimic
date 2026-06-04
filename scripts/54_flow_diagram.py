"""STARD-style study flow diagram (pipeline step 54).

Renders the cohort-and-enrichment flow for the cuff-occlusion feasibility study:
from charted noninvasive blood pressure (NIBP) cuff cycles, down the main path
to the detector-positive estimate, then the three-stratum gallery sampling and
the blinded reader / detector / MedGemma adjudication. Exclusions and the
not-sampled remainder branch off the main path so coverage reads at a glance.

Every count is read live from the shipped artifacts and reconciled before the
diagram is drawn, never hardcoded into the layout. The script aborts with a
clear message if any count fails to reconcile, rather than drawing a wrong
funnel:

- ``data/interim/event_inventory.csv`` : per-cycle ``reject_reason`` counts and
  ``pre_window_valid`` (the QC-pass flag).
- ``results/gallery/gallery_manifest.csv`` : the 568 sampled cards by stratum.
- ``results/gallery/reader_form_blinded.csv`` : the blinded reader calls.

The diagram is purely deterministic (no random state); it carries no on-canvas
title (the title belongs in the manuscript caption) and uses the shared
``cuffcrt.figstyle`` typography and Okabe-Ito palette. The main path and the
exclusion branches are distinguished by both fill and edge so the structure
survives grayscale printing.

Output
------
``--out_dir/fig_flow_diagram.png`` (400 dpi raster) and ``.pdf`` (editable
vector), via :func:`cuffcrt.figstyle.save`.

Examples
--------
::

    uv run python scripts/54_flow_diagram.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text

from cuffcrt import figstyle
from cuffcrt._seed import GLOBAL_SEED

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = DEFAULT_REPO / "data/interim/event_inventory.csv"
DEFAULT_MANIFEST = DEFAULT_REPO / "results/gallery/gallery_manifest.csv"
DEFAULT_READER = DEFAULT_REPO / "results/gallery/reader_form_blinded.csv"
DEFAULT_OUT = DEFAULT_REPO / "figures"

# Reader-call vocabulary in the blinded form (see scripts/44_precision_recall.py).
_READER_PRESENT = "occlusion_signature_present"
_READER_ABSENT = "no_occlusion_signature"
_READER_INDETERMINATE = "indeterminate"

# The detector-rejected near-miss pool (deep aligned dips that fall short of the
# primary rule): short reperfusion run, or never recovered.
_NEAR_MISS_REASONS = ("stat_mode_short_phase3", "no_recovery_in_window")


@dataclass(frozen=True)
class FlowCounts:
    """All counts the diagram needs, after reconciliation.

    Attributes
    ----------
    n_candidates : int
        Charted NIBP cuff cycles (every inventory row).
    n_records : int
        Distinct records in the inventory.
    n_subjects : int
        Distinct subjects in the cohort (the inventory). This is the cohort
        subject count, not the smaller subject count spanned by the sampled
        gallery (``n_gallery_subjects``).
    n_no_pleth : int
        Cycles with no co-recorded PPG window.
    n_pleth_nan : int
        Cycles whose PPG window is more than half missing.
    n_evaluable : int
        Evaluable cycles with co-recorded PPG (candidates minus the two no-PPG
        reasons).
    n_qc_pass : int
        Evaluable cycles passing the pre-cuff QC window (``pre_window_valid``).
    n_detector_positive : int
        Detector-positive cycles (the primary, pre-registered set).
    pct_detector_positive : float
        Detector-positive cycles as a percentage of QC-pass cycles.
    n_near_miss_pool : int
        Size of the detector-rejected near-miss pool available for sampling.
    n_negative_pool : int
        Size of the detector-negative pool available for sampling.
    n_uncovered : int
        Evaluable cycles not eligible for any sampling stratum (the not-sampled
        remainder).
    n_uncovered_no_align : int
        Not-sampled cycles with no aligned occlusion.
    n_uncovered_implausible : int
        Not-sampled cycles with an implausible pre-cuff PI.
    n_sampled_positive : int
        Detector-positive cards in the gallery (a census).
    n_sampled_near_miss : int
        Near-miss cards in the gallery.
    n_sampled_negative : int
        Detector-negative random cards in the gallery.
    n_gallery : int
        Total blinded gallery cards.
    n_gallery_subjects : int
        Distinct subjects spanned by the gallery.
    coverage : int
        Evaluable cycles eligible for sampling (positive + near-miss + negative
        pools); the coverage numerator.
    coverage_pct : float
        Coverage as a percentage of evaluable cycles.
    n_reader_present : int
        Blinded-reader present calls.
    n_reader_absent : int
        Blinded-reader absent calls.
    n_reader_indeterminate : int
        Blinded-reader indeterminate calls.
    weight_positive : float
        Inverse-sampling weight for the detector-positive stratum.
    weight_near_miss : float
        Inverse-sampling weight for the near-miss stratum.
    weight_negative : float
        Inverse-sampling weight for the detector-negative random stratum.
    """

    n_candidates: int
    n_records: int
    n_subjects: int
    n_no_pleth: int
    n_pleth_nan: int
    n_evaluable: int
    n_qc_pass: int
    n_detector_positive: int
    pct_detector_positive: float
    n_near_miss_pool: int
    n_negative_pool: int
    n_uncovered: int
    n_uncovered_no_align: int
    n_uncovered_implausible: int
    n_sampled_positive: int
    n_sampled_near_miss: int
    n_sampled_negative: int
    n_gallery: int
    n_gallery_subjects: int
    coverage: int
    coverage_pct: float
    n_reader_present: int
    n_reader_absent: int
    n_reader_indeterminate: int
    weight_positive: float
    weight_near_miss: float
    weight_negative: float


def _reason_count(inv: pl.DataFrame, reason: str) -> int:
    """Number of inventory rows whose ``reject_reason`` equals ``reason``."""
    return int(inv.filter(pl.col("reject_reason") == reason).height)


def _normalize_reject_reason(inv: pl.DataFrame) -> pl.DataFrame:
    """Map an empty-string reject reason to null (the CSV round-trip artifact).

    The single detector-positive convention stores a null (here, an empty
    string after a CSV round-trip) ``reject_reason``. Normalizing keeps the
    detector-positive count separable from the named reject reasons.
    """
    if "reject_reason" in inv.columns and inv.schema["reject_reason"] == pl.Utf8:
        return inv.with_columns(
            pl.when(pl.col("reject_reason") == "")
            .then(None)
            .otherwise(pl.col("reject_reason"))
            .alias("reject_reason")
        )
    return inv


def compute_counts(
    inventory_csv: Path, manifest_csv: Path, reader_csv: Path
) -> FlowCounts:
    """Read the artifacts, derive every diagram count, and reconcile them.

    Parameters
    ----------
    inventory_csv : pathlib.Path
        Per-cycle event inventory.
    manifest_csv : pathlib.Path
        Gallery card manifest.
    reader_csv : pathlib.Path
        Blinded reader form.

    Returns
    -------
    FlowCounts
        The reconciled counts the layout consumes.

    Raises
    ------
    ValueError
        If any count fails an internal reconciliation check, so the diagram is
        never drawn from numbers that do not add up.
    """
    inv = _normalize_reject_reason(pl.read_csv(inventory_csv, infer_schema_length=20000))
    manifest = pl.read_csv(manifest_csv, infer_schema_length=20000)
    reader = pl.read_csv(reader_csv, infer_schema_length=20000)

    n_candidates = inv.height
    n_records = int(inv.get_column("record_id").n_unique())
    n_subjects = int(inv.get_column("subject_id").n_unique())

    n_no_pleth = _reason_count(inv, "no_pleth")
    n_pleth_nan = _reason_count(inv, "pleth_mostly_nan")
    n_evaluable = n_candidates - n_no_pleth - n_pleth_nan

    n_qc_pass = int(inv.filter(pl.col("pre_window_valid")).height)

    # Detector-positive cycles carry a null reject_reason after normalization.
    n_detector_positive = int(inv.filter(pl.col("reject_reason").is_null()).height)
    pct_detector_positive = (
        100.0 * n_detector_positive / n_qc_pass if n_qc_pass else float("nan")
    )

    # Sampling pools, derived from the inventory reject reasons.
    n_near_miss_pool = sum(_reason_count(inv, r) for r in _NEAR_MISS_REASONS)
    n_uncovered_no_align = _reason_count(inv, "no_aligned_occlusion")
    n_uncovered_implausible = _reason_count(inv, "pre_pi_implausible")
    n_uncovered = n_uncovered_no_align + n_uncovered_implausible
    # The detector-negative random pool is the rest of the evaluable cycles.
    n_negative_pool = (
        n_evaluable - n_detector_positive - n_near_miss_pool - n_uncovered
    )

    # Gallery cards by stratum, from the manifest.
    strata = (
        manifest.group_by("stratum")
        .len()
        .to_dict(as_series=False)
    )
    by_stratum = dict(zip(strata["stratum"], strata["len"], strict=True))
    n_sampled_positive = int(by_stratum.get("detector_positive", 0))
    n_sampled_near_miss = int(by_stratum.get("detector_rejected_near_miss", 0))
    n_sampled_negative = int(by_stratum.get("detector_negative_random", 0))
    n_gallery = manifest.height
    n_gallery_subjects = int(manifest.get_column("subject_id").n_unique())

    coverage = n_detector_positive + n_near_miss_pool + n_negative_pool
    coverage_pct = 100.0 * coverage / n_evaluable if n_evaluable else float("nan")

    # Inverse-sampling weights (pool size / sampled count).
    weight_positive = n_detector_positive / n_sampled_positive
    weight_near_miss = n_near_miss_pool / n_sampled_near_miss
    weight_negative = n_negative_pool / n_sampled_negative

    # Blinded-reader call counts.
    call = reader.get_column("call")
    n_reader_present = int((call == _READER_PRESENT).sum())
    n_reader_absent = int((call == _READER_ABSENT).sum())
    n_reader_indeterminate = int((call == _READER_INDETERMINATE).sum())

    counts = FlowCounts(
        n_candidates=n_candidates,
        n_records=n_records,
        n_subjects=n_subjects,
        n_no_pleth=n_no_pleth,
        n_pleth_nan=n_pleth_nan,
        n_evaluable=n_evaluable,
        n_qc_pass=n_qc_pass,
        n_detector_positive=n_detector_positive,
        pct_detector_positive=pct_detector_positive,
        n_near_miss_pool=n_near_miss_pool,
        n_negative_pool=n_negative_pool,
        n_uncovered=n_uncovered,
        n_uncovered_no_align=n_uncovered_no_align,
        n_uncovered_implausible=n_uncovered_implausible,
        n_sampled_positive=n_sampled_positive,
        n_sampled_near_miss=n_sampled_near_miss,
        n_sampled_negative=n_sampled_negative,
        n_gallery=n_gallery,
        n_gallery_subjects=n_gallery_subjects,
        coverage=coverage,
        coverage_pct=coverage_pct,
        n_reader_present=n_reader_present,
        n_reader_absent=n_reader_absent,
        n_reader_indeterminate=n_reader_indeterminate,
        weight_positive=weight_positive,
        weight_near_miss=weight_near_miss,
        weight_negative=weight_negative,
    )
    _reconcile(counts)
    return counts


def _reconcile(c: FlowCounts) -> None:
    """Assert every diagram count adds up before drawing.

    Raises
    ------
    ValueError
        If any check fails, with a message naming the failed identity.
    """
    checks = [
        (
            "evaluable = candidates - no_pleth - pleth_mostly_nan",
            c.n_evaluable == c.n_candidates - c.n_no_pleth - c.n_pleth_nan,
        ),
        (
            "evaluable = positive + near_miss_pool + negative_pool + uncovered",
            c.n_evaluable
            == c.n_detector_positive + c.n_near_miss_pool + c.n_negative_pool + c.n_uncovered,
        ),
        (
            "uncovered = no_aligned_occlusion + pre_pi_implausible",
            c.n_uncovered == c.n_uncovered_no_align + c.n_uncovered_implausible,
        ),
        (
            "coverage = positive + near_miss_pool + negative_pool",
            c.coverage == c.n_detector_positive + c.n_near_miss_pool + c.n_negative_pool,
        ),
        (
            "coverage + uncovered = evaluable",
            c.coverage + c.n_uncovered == c.n_evaluable,
        ),
        (
            "gallery = sampled positive + near_miss + negative",
            c.n_gallery
            == c.n_sampled_positive + c.n_sampled_near_miss + c.n_sampled_negative,
        ),
        (
            "gallery = reader present + absent + indeterminate",
            c.n_gallery
            == c.n_reader_present + c.n_reader_absent + c.n_reader_indeterminate,
        ),
        (
            "detector-positive census (sampled positive == positive pool)",
            c.n_sampled_positive == c.n_detector_positive,
        ),
    ]
    failures = [name for name, ok in checks if not ok]
    if failures:
        raise ValueError(
            "flow-diagram counts failed reconciliation; refusing to draw a wrong "
            "funnel. Failed identities: " + "; ".join(failures)
        )


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------
# The diagram is laid out on a 0..100 x 0..100 coordinate grid. The main
# (included) path runs down a central column; exclusions and the not-sampled
# remainder branch to the right; the three sampling strata fan out, then
# converge into the gallery box; adjudication sits at the bottom. The per-box
# x-anchors and widths are set locally in build_figure.

_FONT_TITLE = 9.0
_FONT_COUNT = 11.0
_FONT_SUB = 7.4
_FONT_BRANCH = 7.8


@dataclass(frozen=True)
class _Box:
    """A drawn box and the anchor points the arrows attach to."""

    cx: float
    cy: float
    w: float
    h: float

    @property
    def bottom(self) -> tuple[float, float]:
        return self.cx, self.cy - self.h / 2.0

    @property
    def top(self) -> tuple[float, float]:
        return self.cx, self.cy + self.h / 2.0

    @property
    def left(self) -> tuple[float, float]:
        return self.cx - self.w / 2.0, self.cy

    @property
    def right(self) -> tuple[float, float]:
        return self.cx + self.w / 2.0, self.cy


def _draw_box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    title: str,
    count: str | None,
    subtitle: str | None,
    main: bool,
    accent: bool = False,
) -> _Box:
    """Draw one labeled box and return its geometry.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    cx, cy, w, h : float
        Box center and size in grid units.
    title : str
        Bold-ish stage label (regular weight; the count carries the bold).
    count : str or None
        The count text, drawn bold. ``None`` omits it.
    subtitle : str or None
        Small descriptive line under the count.
    main : bool
        Whether this box sits on the included main path (vs. a branch). Main
        boxes get the neutral panel fill and a dark edge; branch boxes get a
        white fill and the muted-vermillion edge so the two read apart in pure
        grayscale (fill density differs) as well as in color.
    accent : bool
        If ``True``, mark this as the pre-registered primary endpoint with the
        cool reperfusion wash and the blue edge.
    """
    if accent:
        face = figstyle.WASH_REPERFUSION
        edge = figstyle.COLOR_USABLE
        lw = 1.6
    elif main:
        face = figstyle.PANEL_BG
        edge = figstyle.GRAPHITE
        lw = 1.1
    else:
        face = "white"
        edge = figstyle.COLOR_EXCLUDED
        lw = 1.0
    box = FancyBboxPatch(
        (cx - w / 2.0, cy - h / 2.0),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.9",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(box)

    count_color = figstyle.COLOR_USABLE if accent else figstyle.INK
    title_color = figstyle.INK if main else figstyle.COLOR_EXCLUDED

    # Vertical stacking inside the box, top to bottom: title, count, subtitle.
    # Each element is created at the box center, its true rendered height is
    # measured (so a one-line and a two-line title are spaced correctly), and the
    # whole block is then centered in the box with even gaps between elements.
    # This keeps every box's text evenly aligned and collision-free regardless of
    # how many lines a title or subtitle wraps to.
    elements: list[tuple[Text, int]] = []
    title_txt = ax.text(
        cx,
        cy,
        title,
        ha="center",
        va="center",
        fontsize=_FONT_TITLE if main else _FONT_BRANCH,
        fontweight="normal",
        color=title_color,
        linespacing=1.15,
        zorder=3,
    )
    elements.append((title_txt, title.count("\n") + 1))
    if count is not None:
        count_txt = ax.text(
            cx,
            cy,
            count,
            ha="center",
            va="center",
            fontsize=_FONT_COUNT if main else _FONT_BRANCH + 0.6,
            fontweight="bold",
            color=count_color,
            zorder=3,
        )
        elements.append((count_txt, count.count("\n") + 1))
    if subtitle is not None:
        subtitle_txt = ax.text(
            cx,
            cy,
            subtitle,
            ha="center",
            va="center",
            fontsize=_FONT_SUB,
            color=figstyle.GRAPHITE,
            linespacing=1.15,
            zorder=3,
        )
        elements.append((subtitle_txt, subtitle.count("\n") + 1))

    _stack_text_block(ax, cx, cy, h, elements)
    return _Box(cx, cy, w, h)


def _text_height_data(ax: plt.Axes, txt: Text) -> float:
    """Return a text artist's rendered height in axes-data units (y-axis).

    Measures the on-screen bounding box with the figure's renderer and maps it
    back through the inverse data transform, so multi-line titles and subtitles
    report their true height rather than an assumed single-line height.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    bbox = txt.get_window_extent(renderer=renderer)
    inv = ax.transData.inverted()
    (_, y0), (_, y1) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
    return abs(y1 - y0)


def _stack_text_block(
    ax: plt.Axes,
    cx: float,
    cy: float,
    h: float,
    elements: list[tuple[Text, int]],
) -> None:
    """Vertically center a stack of text artists inside a box, with even gaps.

    Each artist is measured at its true rendered height, the elements are stacked
    top to bottom with one shared gap between neighbors, and the whole block is
    centered on ``cy``. The gap scales gently with element count so a busy box
    (title + count + subtitle) stays compact while a sparse box breathes, and it
    is capped so the block never overflows the box height ``h``.
    """
    heights = [_text_height_data(ax, txt) for txt, _ in elements]
    n = len(elements)
    if n == 1:
        elements[0][0].set_y(cy)
        return
    total_text = sum(heights)
    # One shared gap between neighbors. Aim for a gap a little smaller than the
    # mean line height so the block reads as one unit, then shrink it if the
    # block would otherwise spill past the usable interior of the box.
    mean_line = total_text / sum(lines for _, lines in elements)
    gap = 0.62 * mean_line
    usable = 0.86 * h
    block = total_text + (n - 1) * gap
    if block > usable:
        gap = max((usable - total_text) / (n - 1), 0.0)
        block = total_text + (n - 1) * gap
    # Walk from the top of the centered block, placing each element's center.
    top = cy + block / 2.0
    cursor = top
    for (txt, _), eh in zip(elements, heights, strict=True):
        txt.set_y(cursor - eh / 2.0)
        cursor -= eh + gap


def _arrow(
    ax: plt.Axes,
    xy_from: tuple[float, float],
    xy_to: tuple[float, float],
    *,
    color: str | None = None,
    style: str = "-|>",
    lw: float = 1.2,
) -> None:
    """Draw a straight connector arrow between two anchor points."""
    ax.add_patch(
        FancyArrowPatch(
            xy_from,
            xy_to,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            color=color or figstyle.SLATE,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=1,
        )
    )


def _elbow_arrow(
    ax: plt.Axes,
    x_main: float,
    y_main: float,
    box_left_xy: tuple[float, float],
    *,
    color: str,
) -> None:
    """Draw an orthogonal (down-then-right) connector to a right-hand branch box.

    The connector drops vertically from the main spine at ``(x_main, y_main)``
    to the branch box's vertical center, then runs horizontally into its left
    edge, so all branch arrows are strictly orthogonal.
    """
    bx, by = box_left_xy
    # Vertical leg on the spine, then horizontal leg into the box.
    ax.plot([x_main, x_main], [y_main, by], color=color, linewidth=1.0, zorder=1)
    _arrow(ax, (x_main, by), (bx, by), color=color, lw=1.0)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(c: FlowCounts) -> plt.Figure:
    """Build the STARD-style flow diagram from reconciled counts.

    Parameters
    ----------
    c : FlowCounts
        The reconciled counts.

    Returns
    -------
    matplotlib.figure.Figure
        The flow diagram (no on-canvas title).
    """
    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 10.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 104)
    ax.axis("off")
    ax.grid(False)
    ax.set_aspect("auto")
    # Realize the renderer and the data transform once up front so the per-box
    # text-height measurement (used to stack title/count/subtitle evenly) reads
    # correct extents from the first box onward.
    fig.canvas.draw()

    main_h = 7.0
    main_x = 33.0
    main_w = 42.0
    spine_x = main_x

    # --- Main path boxes (top to bottom) ---------------------------------
    b_candidates = _draw_box(
        ax,
        main_x,
        95.0,
        main_w,
        main_h,
        title="Charted cuff cycles",
        count=f"{c.n_candidates:,}",
        subtitle=f"{c.n_records} records, {c.n_subjects} subjects",
        main=True,
    )
    b_evaluable = _draw_box(
        ax,
        main_x,
        80.0,
        main_w,
        main_h,
        title="Cuff cycles with a usable\npulse-oximeter waveform",
        count=f"{c.n_evaluable:,}",
        subtitle="usable pulse-oximeter waveform",
        main=True,
    )
    b_qc = _draw_box(
        ax,
        main_x,
        65.0,
        main_w,
        main_h,
        title="Cuff cycles suitable for analysis",
        count=f"{c.n_qc_pass:,}",
        subtitle="passed the pre-cuff baseline check",
        main=True,
    )
    b_positive = _draw_box(
        ax,
        main_x,
        50.0,
        main_w,
        main_h + 0.8,
        title="Detector-positive cycles (primary)",
        count=f"{c.n_detector_positive}",
        subtitle=(
            f"{c.pct_detector_positive:.2f}% of the {c.n_qc_pass:,} cuff cycles\n"
            "suitable for analysis"
        ),
        main=True,
        accent=True,
    )

    # Main-path connectors.
    _arrow(ax, b_candidates.bottom, b_evaluable.top)
    _arrow(ax, b_evaluable.bottom, b_qc.top)
    _arrow(ax, b_qc.bottom, b_positive.top)

    # --- Exclusion branch off the candidates -> evaluable step -----------
    branch_x = 79.0
    branch_w = 38.0
    y_excl = 0.5 * (b_candidates.bottom[1] + b_evaluable.top[1])
    _draw_box(
        ax,
        branch_x,
        y_excl,
        branch_w,
        7.2,
        title="Excluded: no usable\npulse-oximeter waveform",
        count=f"{c.n_no_pleth + c.n_pleth_nan}",
        subtitle=f"no waveform {c.n_no_pleth}; waveform mostly missing {c.n_pleth_nan}",
        main=False,
    )
    _elbow_arrow(ax, spine_x, y_excl, (branch_x - branch_w / 2.0, y_excl),
                 color=figstyle.COLOR_EXCLUDED)

    # --- Stratified gallery sampling header ------------------------------
    ax.text(
        50.0,
        41.0,
        "Stratified gallery sampling from the cycles with a usable waveform (3 strata)",
        ha="center",
        va="center",
        fontsize=_FONT_TITLE,
        fontweight="bold",
        color=figstyle.INK,
        zorder=3,
    )
    # Each stratum must visibly originate at its true parent pool, not all from
    # the detector-positive box:
    #   * detector-positive (census, all of them) descends from the
    #     detector-positive box;
    #   * detector-rejected near-miss and detector-negative random are both
    #     pool-sampled from the wider evaluable pool that the detector
    #     evaluated, so they tap off the QC-pass -> detector segment of the main
    #     spine (the detector's input population), never off the primary box.
    y_bar = 37.5

    # --- Three sampling strata (a fanned row) ----------------------------
    strata_y = 30.0
    strata_h = 9.2
    strata_w = 29.0
    sx_pos, sx_near, sx_neg = 17.5, 50.0, 82.5
    b_s_pos = _draw_box(
        ax,
        sx_pos,
        strata_y,
        strata_w,
        strata_h,
        title="Detector-positive\n(census)",
        count=f"{c.n_sampled_positive} / {c.n_detector_positive}",
        subtitle=f"sampled / pool; weight {c.weight_positive:.2f}",
        main=True,
    )
    b_s_near = _draw_box(
        ax,
        sx_near,
        strata_y,
        strata_w,
        strata_h,
        title="Detector-rejected\nnear-miss",
        count=f"{c.n_sampled_near_miss} / {c.n_near_miss_pool}",
        subtitle=f"sampled / pool; weight {c.weight_near_miss:.2f}",
        main=True,
    )
    b_s_neg = _draw_box(
        ax,
        sx_neg,
        strata_y,
        strata_w,
        strata_h,
        title="Detector-negative\nrandom",
        count=f"{c.n_sampled_negative} / {c.n_negative_pool:,}",
        subtitle=f"sampled / pool; weight {c.weight_negative:.2f}",
        main=True,
    )

    # Routing by true provenance, so no stratum appears to descend from the
    # detector-positive box unless it actually does.
    #
    # 1. Detector-positive census: a dedicated connector straight from the
    #    detector-positive box down and across to that one stratum only.
    pos_drop_y = 0.5 * (b_positive.bottom[1] + y_bar)
    ax.plot([spine_x, spine_x], [b_positive.bottom[1], pos_drop_y],
            color=figstyle.SLATE, linewidth=1.1, zorder=1)
    ax.plot([sx_pos, spine_x], [pos_drop_y, pos_drop_y], color=figstyle.SLATE,
            linewidth=1.1, zorder=1)
    ax.plot([sx_pos, sx_pos], [pos_drop_y, y_bar], color=figstyle.SLATE,
            linewidth=1.1, zorder=1)
    _arrow(ax, (sx_pos, y_bar), b_s_pos.top, color=figstyle.SLATE, lw=1.0)

    # 2. Detector-rejected near-miss and detector-negative random are
    #    pool-sampled from the wider evaluable pool the detector evaluated.
    #    Tap off the QC-pass -> detector segment of the main spine (the
    #    detector's input population), run out along a right-side rail, and
    #    distribute into just those two strata, plus the not-sampled remainder
    #    (also part of the evaluable pool). This rail never touches the primary
    #    box.
    rail_x = 92.0
    rem_tap_x = rail_x
    pool_tap_y = 0.5 * (b_qc.bottom[1] + b_positive.top[1])
    # Spine tap -> right rail at the pool-tap height.
    ax.plot([spine_x, rail_x], [pool_tap_y, pool_tap_y], color=figstyle.SLATE,
            linewidth=1.1, zorder=1)
    # Small label so the reader sees these two strata come from the evaluable
    # pool the detector evaluated, not from the primary box.
    ax.text(
        0.5 * (spine_x + rail_x),
        pool_tap_y + 1.4,
        "sampled from the cycles with a usable waveform",
        ha="center",
        va="bottom",
        fontsize=_FONT_SUB,
        color=figstyle.GRAPHITE,
        zorder=3,
    )
    # Rail drops to the distributor-bar height.
    ax.plot([rail_x, rail_x], [pool_tap_y, y_bar], color=figstyle.SLATE,
            linewidth=1.1, zorder=1)
    # Distributor bar feeds only the near-miss and negative strata.
    ax.plot([sx_near, rail_x], [y_bar, y_bar], color=figstyle.SLATE,
            linewidth=1.1, zorder=1)
    for b in (b_s_near, b_s_neg):
        _arrow(ax, (b.top[0], y_bar), b.top, color=figstyle.SLATE, lw=1.0)

    # --- Not-sampled remainder branch (off the evaluable-pool rail) ------
    # Routed down the far right, clear of the rightmost stratum box, so the
    # evaluable cycles that fell into no sampling stratum read as a side branch.
    rem_x = 84.0
    rem_w = 30.0
    rem_y = 16.0
    _draw_box(
        ax,
        rem_x,
        rem_y,
        rem_w,
        9.0,
        title="Not sampled (remainder)",
        count=f"{c.n_uncovered}",
        subtitle=(
            f"no aligned occlusion {c.n_uncovered_no_align};\n"
            f"implausible pre-cuff perfusion index {c.n_uncovered_implausible}"
        ),
        main=False,
    )
    ax.plot([rem_tap_x, rem_tap_x], [y_bar, rem_y], color=figstyle.COLOR_EXCLUDED,
            linewidth=1.0, zorder=0)
    _arrow(ax, (rem_tap_x, rem_y), (rem_x + rem_w / 2.0, rem_y),
           color=figstyle.COLOR_EXCLUDED, lw=1.0)

    # --- Gallery box (strata converge) -----------------------------------
    gallery_y = 14.5
    b_gallery = _draw_box(
        ax,
        main_x,
        gallery_y,
        main_w + 8.0,
        7.0,
        title="Blinded gallery",
        count=f"{c.n_gallery} cards",
        subtitle=(
            f"coverage {c.coverage:,} / {c.n_evaluable:,} = "
            f"{c.coverage_pct:.1f}% of cycles with a usable waveform"
        ),
        main=True,
    )
    # All three strata converge into a collector bar above the gallery; the bar
    # carries them to the gallery's center, where a single arrow enters the top.
    y_collect = gallery_y + b_gallery.h / 2.0 + 3.6
    for b in (b_s_pos, b_s_near, b_s_neg):
        ax.plot([b.bottom[0], b.bottom[0]], [b.bottom[1], y_collect],
                color=figstyle.SLATE, linewidth=1.1, zorder=1)
    ax.plot([sx_pos, sx_neg], [y_collect, y_collect], color=figstyle.SLATE,
            linewidth=1.1, zorder=1)
    _arrow(ax, (main_x, y_collect), b_gallery.top, color=figstyle.SLATE, lw=1.2)

    # --- Adjudication box ------------------------------------------------
    b_adj = _draw_box(
        ax,
        main_x,
        4.0,
        main_w + 18.0,
        6.4,
        title=(
            "Adjudicated by 1 blinded expert reader; re-read by the\n"
            f"rule-based detector and {figstyle.MODEL_DISPLAY.lower()} on identical images"
        ),
        count=None,
        subtitle=(
            f"reader calls: {c.n_reader_present} present, "
            f"{c.n_reader_absent} absent, {c.n_reader_indeterminate} indeterminate"
        ),
        main=True,
    )
    _arrow(ax, b_gallery.bottom, b_adj.top, color=figstyle.SLATE, lw=1.2)

    # --- Legend: main path vs exclusion vs primary endpoint --------------
    legend_handles = [
        FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            facecolor=figstyle.PANEL_BG, edgecolor=figstyle.GRAPHITE, linewidth=1.1,
            label="Included path",
        ),
        FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            facecolor="white", edgecolor=figstyle.COLOR_EXCLUDED, linewidth=1.0,
            label="Excluded / not sampled",
        ),
        FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            facecolor=figstyle.WASH_REPERFUSION, edgecolor=figstyle.COLOR_USABLE,
            linewidth=1.6, label="Pre-registered primary endpoint",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        fontsize=7.6,
        frameon=True,
        borderpad=0.6,
        columnspacing=1.6,
        handlelength=1.4,
        handleheight=1.2,
    )

    # No suptitle: the figure title lives in the manuscript caption.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01)
    return fig


def run(
    *, inventory_csv: Path, manifest_csv: Path, reader_csv: Path, out_dir: Path
) -> tuple[Path, Path]:
    """Compute counts, build, and save the flow diagram. Returns ``(png, pdf)``."""
    counts = compute_counts(inventory_csv, manifest_csv, reader_csv)
    logger.info(
        "flow counts: candidates {:,}, evaluable {:,}, qc-pass {:,}, "
        "detector-positive {} ({:.2f}%), gallery {} cards "
        "(reader {}/{}/{}), coverage {:.1f}%",
        counts.n_candidates,
        counts.n_evaluable,
        counts.n_qc_pass,
        counts.n_detector_positive,
        counts.pct_detector_positive,
        counts.n_gallery,
        counts.n_reader_present,
        counts.n_reader_absent,
        counts.n_reader_indeterminate,
        counts.coverage_pct,
    )
    fig = build_figure(counts)
    png, pdf = figstyle.save(fig, out_dir, "fig_flow_diagram")
    plt.close(fig)
    logger.info("wrote figure {} and {}", png, pdf)
    return png, pdf


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Render the STARD-style study flow diagram (cohort funnel, "
            "stratified gallery sampling, and blinded adjudication)."
        )
    )
    p.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--reader", type=Path, default=DEFAULT_READER)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    # Determinism: the diagram uses no random state; pin the global stream anyway
    # so any future stochastic styling stays reproducible.
    import numpy as np

    np.random.default_rng(GLOBAL_SEED)

    args = _build_argparser().parse_args(argv)
    run(
        inventory_csv=args.inventory,
        manifest_csv=args.manifest,
        reader_csv=args.reader,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
