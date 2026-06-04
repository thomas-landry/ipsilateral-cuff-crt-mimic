"""Render the three manuscript figures (pipeline step 50).

All three IMRAD figures are built from real data, deterministically, and saved
as both a high-DPI PNG and an editable vector PDF:

- **Figure 1 (funnel).** The feasibility funnel from candidate cuff cycles down
  to the morphology-based occlusion-signature estimate, with the 15 s primary
  count and the 10 s sensitivity stratum. Counts are computed from the event
  inventory via :func:`cuffcrt.analysis.funnel.aggregate_funnel`, never
  hardcoded.
- **Figure 2 (non-usable examples).** Representative cuff cycles that fail the
  usability check: no co-recorded photoplethysmogram, an unstable pre-cuff
  perfusion-index window, and a stable trace with no occlusion dip. Each panel
  plots the real 1 Hz perfusion index (or its absence) so the reason for
  exclusion is visible.
- **Figure 3 (one clean candidate).** The real perfusion-index trace across a
  cuff cycle for one of the two records that meet the 15 s rule
  (de-identified pseudo-ID ``pXXXXXXXX``), with the occlusion and reperfusion
  phases shaded.

The call is inferred from perfusion-index morphology; there is no ground-truth
cuff laterality in MIMIC-IV-WDB. Every count is a morphology-based estimate,
not a confirmed laterality.

The script reads waveforms in place and writes only derived plots (perfusion
index over time); no raw waveform samples or note text leave the pipeline. It
never overwrites its inputs.

Examples
--------
All three figures from the credentialed data::

    uv run python scripts/50_figures.py --which all \\
        --wdb-root /path/to/mimic-iv-wdb/0.1.0 \\
        --inventory /path/to/event_inventory.csv

Demo mode (no credentialing). The open MIMIC-IV-Demo carries clinical tables
only, with no waveform records, so only Figure 1 (built from the inventory CSV)
renders; the waveform figures fail clean with a data/README.md pointer::

    uv run python scripts/50_figures.py --which 1 --demo --inventory <demo_inventory.csv>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import wfdb
from loguru import logger
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text

from cuffcrt import figstyle
from cuffcrt._paths import (
    ENV_INVENTORY,
    ENV_WDB_ROOT,
    DataNotAvailableError,
    DataPathNotConfiguredError,
    env_path,
    require_path,
    resolve_configured_path,
    resolve_wdb_root,
)
from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.analysis.funnel import FunnelResult, aggregate_funnel
from cuffcrt.signal.cuff_event_detector import compute_pi_1hz, detect_cuff_event

# The credentialed WDB tree and the event inventory live outside this
# repository and are never copied in (PhysioNet DUA). They are supplied per run
# via --wdb-root / --inventory or the CUFFCRT_WDB_ROOT / CUFFCRT_INVENTORY
# environment variables (see data/README.md). There is no machine default.

# Window half-widths around the charted NBP timestamp, matching step 20 so the
# perfusion-index axis aligns with the detector's view.
PRE_WINDOW_S = 200.0
POST_WINDOW_S = 200.0

# de-identified MIMIC-IV-WDB pseudo-IDs, used under the PhysioNet DUA
# The two records that meet the 15 s primary rule. Figure 3 must use one of
# these; the earlier qualifying record (pXXXXXXXX) no longer qualifies at 15 s.
PRIMARY_SURVIVORS = ("p10014354", "p10079700")

# Hero cycle for Figure 3: the cleanest four-phase exemplar among the 15 s
# primary survivors (deep nadir, reperfusion in band, occlusion dip essentially
# at the cuff marker, reader-confirmed present). Pinned by subject and charted
# timestamp so the figure renders the same cycle every time; the subject is one
# of PRIMARY_SURVIVORS, so this stays inside the primary-survivor framing.
# A fallback cycle (the next-cleanest survivor) is tried if the primary target
# is not present in the inventory, before the generic first-survivor pick.
# de-identified MIMIC-IV-WDB pseudo-IDs, used under the PhysioNet DUA
_FIG3_TARGET = ("p10079700", 5864.980591460243)
_FIG3_FALLBACK = ("p10079700", 2258.9139221257356)
# How close (seconds) a candidate timestamp must be to a target to count as a
# match, absorbing float round-trips between the inventory and the manifest.
_FIG3_T_TOL_S = 1.0


def _rolling_median_1hz(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered rolling median over a 1 Hz trace (edges shrink the window).

    Matches the smoothing the detector applies to the search-window PI before
    threshold checks, so the smoothed overlay in Figure 3 reflects the envelope
    the detector actually measured.

    Parameters
    ----------
    values : numpy.ndarray
        1 Hz samples.
    window : int
        Window length in samples (seconds at 1 Hz).

    Returns
    -------
    numpy.ndarray
        The centered rolling median.
    """
    n = values.size
    if n == 0 or window <= 1:
        return values.copy()
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(values[max(0, i - half) : min(n, i + half + 1)])
    return out


@dataclass(frozen=True)
class RecordTrace:
    """A loaded perfusion-index trace for one charted NBP event.

    Attributes
    ----------
    subject_id : str
        Subject directory name (de-identified per PhysioNet).
    record_id : str
        Record directory name.
    t_nbp_s : float
        Charted NBP timestamp in seconds from record start.
    t_local : numpy.ndarray
        Time axis of the perfusion index, seconds relative to the NBP event
        (0 at the charted timestamp). Empty when no PPG was available.
    pi : numpy.ndarray
        1 Hz perfusion index over ``t_local``. Empty when no PPG was available.
    has_pleth : bool
        Whether a usable co-recorded photoplethysmogram was found.
    """

    subject_id: str
    record_id: str
    t_nbp_s: float
    t_local: np.ndarray
    pi: np.ndarray
    has_pleth: bool


# ---------------------------------------------------------------------------
# Waveform loading (read in place; mirrors scripts/20_extract_cuff_events.py)
# ---------------------------------------------------------------------------
def _resolve_record_dir(wdb_root: Path, subject_id: str, record_id: str) -> Path:
    """Locate a record directory under the WDB tree by subject and record id.

    Parameters
    ----------
    wdb_root : pathlib.Path
        Root of the WDB record tree (the directory containing ``RECORDS``).
    subject_id : str
        Subject directory name, for example ``pXXXXXXXX``.
    record_id : str
        Record directory name, for example ``XXXXXXXX``.

    Returns
    -------
    pathlib.Path
        The record directory.

    Raises
    ------
    DataNotAvailableError
        If the record directory cannot be found.
    """
    # The WDB layout nests subject dirs two levels deep (waves/pXXX/pXXXXXXXX),
    # but RECORDS lists the relative path; resolve via glob to stay layout-agnostic.
    candidates = sorted(wdb_root.glob(f"**/{subject_id}/{record_id}"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise DataNotAvailableError(
            f"record dir for {subject_id}/{record_id} not found under {wdb_root}."
        )
    return candidates[0]


def _parse_master_fs(master_hea: Path) -> float:
    """Pull the master frame rate from the first data line of a header."""
    with master_hea.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"unexpected master header line: {line!r}")
            return float(parts[2].split("/")[0])
    raise ValueError(f"no data line in master header {master_hea}")


def _load_pleth_window(
    record_basename: Path, master_fs: float, t_center_s: float
) -> tuple[np.ndarray, float] | None:
    """Read a PLETH window around ``t_center_s`` at the channel-native rate.

    ``smooth_frames=False`` keeps PLETH at its native rate so the cardiac
    component survives (the master-rate averaging path cancels it). Mirrors the
    extraction step.

    Returns
    -------
    tuple[numpy.ndarray, float] or None
        ``(pleth, fs_native)`` or ``None`` if the channel is unavailable.
    """
    sampfrom = max(0, int((t_center_s - PRE_WINDOW_S) * master_fs))
    sampto = int((t_center_s + POST_WINDOW_S) * master_fs)
    try:
        record = wfdb.rdrecord(
            str(record_basename),
            sampfrom=sampfrom,
            sampto=sampto,
            channel_names=["Pleth"],
            smooth_frames=False,
            return_res=32,
        )
    except Exception as exc:  # noqa: BLE001 - wfdb raises a wide variety
        logger.debug("rdrecord failed at t={:.0f}s: {}", t_center_s, exc)
        return None
    # wfdb's typeshed unions rdrecord's return as Record | MultiRecord, but at
    # runtime it is always a Record (single-segment read). The e_p_signal /
    # samps_per_frame attribute-access and the float() of the resulting value
    # are stub-only false positives.
    if record.e_p_signal is None or len(record.e_p_signal) == 0:  # pyright: ignore[reportAttributeAccessIssue]
        return None
    pleth = np.asarray(record.e_p_signal[0])  # pyright: ignore[reportAttributeAccessIssue]
    if pleth.size == 0 or not np.isfinite(pleth).any():
        return None
    samps_per_frame = record.samps_per_frame[0] if record.samps_per_frame is not None else 1  # pyright: ignore[reportAttributeAccessIssue]
    fs_native = float(record.fs) * float(samps_per_frame)  # pyright: ignore[reportArgumentType]
    return pleth, fs_native


def load_trace(wdb_root: Path, subject_id: str, record_id: str, t_nbp_s: float) -> RecordTrace:
    """Load the 1 Hz perfusion-index trace for one charted NBP event.

    Parameters
    ----------
    wdb_root : pathlib.Path
        Root of the WDB record tree.
    subject_id, record_id : str
        Identifiers locating the record directory.
    t_nbp_s : float
        Charted NBP timestamp in seconds from record start.

    Returns
    -------
    RecordTrace
        The trace; ``has_pleth`` is ``False`` and the arrays are empty when no
        usable photoplethysmogram was available for the window.
    """
    record_dir = _resolve_record_dir(wdb_root, subject_id, record_id)
    master_hea = record_dir / f"{record_id}.hea"
    master_fs = _parse_master_fs(master_hea)

    windowed = _load_pleth_window(record_dir / record_id, master_fs, t_nbp_s)
    if windowed is None:
        return RecordTrace(
            subject_id=subject_id,
            record_id=record_id,
            t_nbp_s=t_nbp_s,
            t_local=np.array([]),
            pi=np.array([]),
            has_pleth=False,
        )
    pleth, fs_native = windowed
    finite = np.isfinite(pleth)
    if finite.mean() < 0.5:
        return RecordTrace(
            subject_id=subject_id,
            record_id=record_id,
            t_nbp_s=t_nbp_s,
            t_local=np.array([]),
            pi=np.array([]),
            has_pleth=False,
        )
    pleth_clean = np.where(finite, pleth, np.nanmedian(pleth))
    t_pi, pi = compute_pi_1hz(pleth_clean, fs_native)
    # The window starts PRE_WINDOW_S before the NBP event; recenter time on it.
    t_local = t_pi - PRE_WINDOW_S
    return RecordTrace(
        subject_id=subject_id,
        record_id=record_id,
        t_nbp_s=t_nbp_s,
        t_local=t_local,
        pi=pi,
        has_pleth=True,
    )


# ---------------------------------------------------------------------------
# Inventory access
# ---------------------------------------------------------------------------
def load_inventory(inventory_csv: Path) -> pl.DataFrame:
    """Read the per-event inventory CSV used to build the funnel.

    Parameters
    ----------
    inventory_csv : pathlib.Path
        Concatenated per-event inventory (one row per charted NBP timestamp).

    Returns
    -------
    polars.DataFrame
        The inventory, with empty-string reject reasons normalized to null.
    """
    inv = pl.read_csv(inventory_csv, infer_schema_length=10000)
    if "reject_reason" in inv.columns and inv.schema["reject_reason"] == pl.Utf8:
        inv = inv.with_columns(
            pl.when(pl.col("reject_reason") == "")
            .then(None)
            .otherwise(pl.col("reject_reason"))
            .alias("reject_reason")
        )
    return inv


def _most_represented_record(inv: pl.DataFrame, reason: str) -> tuple[str, int] | None:
    """Return the ``(subject_id, record_id)`` with the most events of ``reason``.

    Ties break on ``(subject_id, record_id)`` so the choice is deterministic.
    Returns ``None`` if no event has this reason.
    """
    grp = (
        inv.filter(pl.col("reject_reason") == reason)
        .group_by(["subject_id", "record_id"])
        .len()
        .sort(["len", "subject_id", "record_id"], descending=[True, False, False])
    )
    if grp.is_empty():
        return None
    row = grp.row(0, named=True)
    return row["subject_id"], int(row["record_id"])


def _pick_event(
    inv: pl.DataFrame,
    *,
    reason: str | None,
    subject_id: str | None = None,
    record_id: int | None = None,
    require_valid: bool = False,
    sort_by: str | None = None,
    sort_descending: bool = False,
) -> dict:
    """Return one inventory row matching the given constraints, deterministically.

    Parameters
    ----------
    inv : polars.DataFrame
        The event inventory.
    reason : str or None
        Reject reason to match (``None`` matches a full occlusion-signature event).
    subject_id, record_id : optional
        Narrow the match to a specific record.
    require_valid : bool
        If ``True``, also require ``pre_window_valid`` (PPG present and stable).
    sort_by : str or None
        Column to sort the candidates by before taking the first; defaults to a
        deterministic ``(subject_id, record_id, nbp_timestamp_s)`` order.
    """
    pred = (
        pl.col("reject_reason").is_null()
        if reason is None
        else (pl.col("reject_reason") == reason)
    )
    if subject_id is not None:
        pred = pred & (pl.col("subject_id") == subject_id)
    if record_id is not None:
        pred = pred & (pl.col("record_id") == record_id)
    if require_valid:
        pred = pred & pl.col("pre_window_valid")
    rows = inv.filter(pred)
    if rows.is_empty():
        raise ValueError(
            f"no inventory row for reason={reason!r} subject={subject_id!r} "
            f"record={record_id!r} require_valid={require_valid}"
        )
    if sort_by:
        return rows.sort(sort_by, descending=sort_descending).row(0, named=True)
    return rows.sort(["subject_id", "record_id", "nbp_timestamp_s"]).row(0, named=True)


# ---------------------------------------------------------------------------
# Figure 1: funnel
# ---------------------------------------------------------------------------
def _funnel_row(stage: str, result: FunnelResult) -> int:
    """Return the ``events`` count for a funnel stage from the result table."""
    sub = result.funnel.filter(pl.col("stage") == stage)
    return int(sub.get_column("events")[0])


def _text_height_data(ax: plt.Axes, txt: Text) -> float:
    """Return a text artist's rendered height in axes-data units (y-axis).

    Measures the on-screen bounding box with the figure's renderer and maps it
    back through the inverse data transform, so single-line and wrapped labels
    report their true height. Used to stack a box's title/count/subtitle evenly.
    """
    renderer = ax.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    bbox = txt.get_window_extent(renderer=renderer)
    inv = ax.transData.inverted()
    (_, y0), (_, y1) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
    return abs(y1 - y0)


def _stack_text_block(
    ax: plt.Axes, cy: float, box_h: float, elements: list[Text]
) -> None:
    """Vertically center a stack of text artists in a box, with even gaps.

    Each artist is measured at its true rendered height, stacked top to bottom
    with one shared gap between neighbors, and the whole block is centered on
    ``cy``. The gap is sized from the mean line height and capped so the block
    never spills past the usable interior of the box.
    """
    n = len(elements)
    if n == 1:
        elements[0].set_y(cy)
        return
    heights = [_text_height_data(ax, t) for t in elements]
    total_text = sum(heights)
    line_counts = [t.get_text().count("\n") + 1 for t in elements]
    mean_line = total_text / sum(line_counts)
    gap = 0.55 * mean_line
    usable = 0.92 * box_h
    block = total_text + (n - 1) * gap
    if block > usable:
        gap = max((usable - total_text) / (n - 1), 0.0)
        block = total_text + (n - 1) * gap
    cursor = cy + block / 2.0
    for txt, eh in zip(elements, heights, strict=True):
        txt.set_y(cursor - eh / 2.0)
        cursor -= eh + gap


def build_figure_1(result: FunnelResult, out_dir: Path) -> tuple[Path, Path]:
    """Render the feasibility funnel (Figure 1).

    Stage counts are taken from ``result`` (computed from the inventory), never
    hardcoded. The percentages on the final box are expressed against the
    suitable-for-analysis denominator (the pre-cuff baseline pass count), which
    is the manuscript's primary denominator, not against all charted cycles.

    Parameters
    ----------
    result : FunnelResult
        Output of :func:`cuffcrt.analysis.funnel.aggregate_funnel`.
    out_dir : pathlib.Path
        Directory for the rendered files.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        The written ``(png, pdf)`` paths.
    """
    figstyle.apply_style()

    n_candidates = _funnel_row("candidate_cuff_cycles", result)
    n_no_pleth = _funnel_row("excluded_no_pleth", result)
    n_pleth_nan = _funnel_row("excluded_pleth_mostly_nan", result)
    n_evaluable = _funnel_row("evaluable_with_pleth", result)
    n_qc_pass = _funnel_row("qc_pass_pre_window", result)
    qc_patients = int(
        result.funnel.filter(pl.col("stage") == "qc_pass_pre_window").get_column("patients")[0]
    )
    primary = result.primary
    sensitivity = result.sensitivity

    # The manuscript reports the occlusion-signature yield against the
    # suitable-for-analysis denominator (cuff cycles that passed the pre-cuff
    # baseline check), not against every charted cycle. Compute both percentages
    # from the funnel counts so the canvas matches the prose exactly.
    pct_primary_suitable = 100.0 * primary.n_events / n_qc_pass
    pct_sensitivity_suitable = 100.0 * sensitivity.n_events / n_qc_pass

    # The excluded count on the second box folds together the cycles with no
    # usable pulse-oximeter waveform and the cycles whose waveform was mostly
    # missing, so the on-canvas arithmetic (candidates - excluded = evaluable)
    # closes exactly.
    n_excluded_waveform = n_no_pleth + n_pleth_nan

    # Funnel stages, top to bottom. Each is (title, count text, subtitle).
    # Abbreviations are spelled out on the canvas; the caption carries the rest.
    stages = [
        (
            "Charted cuff cycles",
            f"{n_candidates:,}",
            f"charted noninvasive blood-pressure timestamps, {result.n_records} records",
        ),
        (
            "Cuff cycles with a usable pulse-oximeter waveform",
            f"{n_evaluable:,}",
            f"after excluding {n_excluded_waveform:,} "
            f"({n_no_pleth:,} with no usable waveform plus "
            f"{n_pleth_nan:,} with a mostly-missing waveform)",
        ),
        (
            "Cuff cycles suitable for analysis",
            f"{n_qc_pass:,}",
            f"stable pre-cuff baseline ({qc_patients} patients)",
        ),
        (
            "Occlusion-signature estimate, primary (15 s rule)",
            f"{primary.n_events} cycles / {primary.n_patients} patients",
            f"{pct_primary_suitable:.2f}% of the {n_qc_pass:,} "
            "cuff cycles suitable for analysis",
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)
    # Realize the renderer and the data transform once up front so the per-box
    # text-height measurement (used to stack title/count/subtitle evenly) reads
    # correct extents from the first box onward.
    fig.canvas.draw()

    n = len(stages)
    box_h = 1.35
    gap = (10.0 - n * box_h) / (n + 1)
    # Boxes taper in width to read as a funnel.
    widths = np.linspace(8.4, 4.6, n)
    centers_y = []
    for i, ((title, count, subtitle), width) in enumerate(zip(stages, widths, strict=True)):
        y_top = 10.0 - (gap + i * (box_h + gap))
        y_bot = y_top - box_h
        y_mid = 0.5 * (y_top + y_bot)
        centers_y.append((y_mid, width))
        x_left = 5.0 - width / 2.0
        is_last = i == n - 1
        face = figstyle.WASH_REPERFUSION if is_last else figstyle.PANEL_BG
        edge = figstyle.COLOR_USABLE if is_last else figstyle.GRAPHITE
        box = FancyBboxPatch(
            (x_left, y_bot),
            width,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.4 if is_last else 1.0,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        # Draw title, count, and subtitle at the box center, then stack them as
        # one vertically centered block with even gaps. Measuring each line's
        # true height keeps every box's text evenly aligned regardless of the
        # title length, so no box reads top-heavy or cramped.
        title_txt = ax.text(
            5.0,
            y_mid,
            title,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=figstyle.INK,
        )
        count_txt = ax.text(
            5.0,
            y_mid,
            count,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color=edge if is_last else figstyle.COLOR_USABLE,
        )
        subtitle_txt = ax.text(
            5.0,
            y_mid,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.8,
            color=figstyle.GRAPHITE,
        )
        _stack_text_block(ax, y_mid, box_h, [title_txt, count_txt, subtitle_txt])

    # Connector arrows between stages.
    for (y_mid_a, _), (y_mid_b, _) in zip(centers_y[:-1], centers_y[1:], strict=True):
        arrow = FancyArrowPatch(
            (5.0, y_mid_a - box_h / 2.0 - 0.04),
            (5.0, y_mid_b + box_h / 2.0 + 0.04),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color=figstyle.SLATE,
        )
        ax.add_patch(arrow)

    # Sensitivity-stratum side annotation on the last box, against the same
    # suitable-for-analysis denominator as the primary box.
    y_last, w_last = centers_y[-1]
    ax.annotate(
        f"10 s sensitivity stratum:\n{sensitivity.n_events} cycles / "
        f"{sensitivity.n_patients} patients "
        f"({pct_sensitivity_suitable:.2f}%)",
        xy=(5.0 + w_last / 2.0, y_last),
        xytext=(9.9, y_last),
        ha="right",
        va="center",
        fontsize=7.8,
        color=figstyle.COLOR_EXCLUDED,
        arrowprops=dict(arrowstyle="-", color=figstyle.COLOR_EXCLUDED, linewidth=0.9),
    )

    # No on-canvas title or footnote: the figure title and any interpretation
    # live only in the manuscript caption, matching the newer builders
    # (47/53/54/55) and the project house rule.
    fig.tight_layout()
    return figstyle.save(fig, out_dir, "fig01_funnel")


# ---------------------------------------------------------------------------
# Figure 2: non-usable examples
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NonUsablePanel:
    """One Figure 2 panel: a chosen record/event plus how to caption it."""

    reason: str
    subject_id: str
    record_id: int
    title: str
    caption: str
    require_valid: bool
    sort_by: str | None
    sort_descending: bool


def _select_nonusable_panels(inv: pl.DataFrame) -> list[NonUsablePanel]:
    """Choose three representative non-usable cases from the inventory.

    For each reject reason the most-represented record is used (the record that
    contributes the most events of that reason), so each panel is a typical
    rather than a corner case, and the three panels come from different records
    where possible. Selection prefers the dominant exclusion (no co-recorded
    PPG), an unstable pre-cuff window, and a stable trace with no occlusion dip;
    it falls back to whatever reasons are present so the figure still renders on
    smaller inventories (for example demo mode).
    """
    # (reason, title, caption, require_valid, sort_by, sort_descending)
    wanted = [
        (
            "no_pleth",
            "No co-recorded PPG",
            "No usable photoplethysmogram at the cuff event",
            False,
            None,
            False,
        ),
        (
            "pre_window_unstable",
            "Unstable pre-cuff PI",
            "Pre-cuff PI too variable to set a baseline",
            False,
            "pre_window_quality",  # most variable first: the clearest example
            True,
        ),
        (
            "no_phase2",
            "No occlusion dip",
            "PPG present and stable, but no deep PI dip",
            True,
            "pre_window_quality",  # stablest first: the clean contralateral-like case
            False,
        ),
        (
            "pleth_mostly_nan",
            "PPG mostly missing",
            "More than half of the PPG window is missing",
            False,
            None,
            False,
        ),
        (
            "pre_pi_implausible",
            "Implausible pre-cuff PI",
            "Pre-cuff PI outside the plausible range",
            False,
            None,
            False,
        ),
    ]
    panels: list[NonUsablePanel] = []
    for reason, title, caption, require_valid, sort_by, sort_desc in wanted:
        rec = _most_represented_record(inv, reason)
        if rec is None:
            continue
        subject_id, record_id = rec
        panels.append(
            NonUsablePanel(
                reason=reason,
                subject_id=subject_id,
                record_id=record_id,
                title=title,
                caption=caption,
                require_valid=require_valid,
                sort_by=sort_by,
                sort_descending=sort_desc,
            )
        )
        if len(panels) == 3:
            break
    if not panels:
        raise ValueError("inventory has no non-usable reject reasons to plot")
    return panels


def build_figure_2(
    inv: pl.DataFrame, wdb_root: Path, out_dir: Path
) -> tuple[Path, Path, list[dict]]:
    """Render representative non-usable cuff cycles (Figure 2).

    Parameters
    ----------
    inv : polars.DataFrame
        The event inventory.
    wdb_root : pathlib.Path
        Root of the WDB record tree (for loading the real PI traces).
    out_dir : pathlib.Path
        Directory for the rendered files.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, list[dict]]
        ``(png, pdf, used)`` where ``used`` records which event backs each panel.
    """
    figstyle.apply_style()
    panels = _select_nonusable_panels(inv)

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    used: list[dict] = []
    for ax, panel, letter in zip(axes, panels, "ABCDE", strict=False):
        event = _pick_event(
            inv,
            reason=panel.reason,
            subject_id=panel.subject_id,
            record_id=panel.record_id,
            require_valid=panel.require_valid,
            sort_by=panel.sort_by,
            sort_descending=panel.sort_descending,
        )
        subject_id = event["subject_id"]
        record_id = str(event["record_id"])
        t_nbp = float(event["nbp_timestamp_s"])
        trace = load_trace(wdb_root, subject_id, record_id, t_nbp)

        # Only the panel letter, the plotted data, and spelled-out axis labels
        # appear on the canvas. The per-panel title and the explanation of each
        # exclusion reason live in the manuscript caption, not on the figure.
        figstyle.panel_label(ax, letter)
        ax.set_xlabel("time relative to charted cuff timestamp (s)")
        ax.set_ylabel("perfusion index (%)")
        ax.axvline(0.0, color=figstyle.SLATE, linewidth=0.9, linestyle="--", zorder=1)

        if not trace.has_pleth or trace.pi.size == 0:
            # No co-recorded PPG: there is no curve to draw. Show the absence.
            ax.set_xlim(-PRE_WINDOW_S, POST_WINDOW_S)
            ax.set_ylim(0, 1)
            ax.text(
                0.5,
                0.5,
                "no co-recorded\nphotoplethysmogram",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.0,
                color=figstyle.COLOR_EXCLUDED,
                fontweight="bold",
            )
        else:
            line_color = (
                figstyle.COLOR_NEUTRAL
                if panel.reason == "no_phase2"
                else figstyle.COLOR_EXCLUDED
            )
            ax.plot(trace.t_local, trace.pi, color=line_color, linewidth=1.1)
            ax.set_xlim(trace.t_local.min(), trace.t_local.max())
            top = float(np.nanpercentile(trace.pi, 99)) * 1.15
            ax.set_ylim(0, max(top, 1.0))

        # The reason each panel is non-usable (panel.caption) is intentionally
        # not drawn on the canvas; it is carried in the manuscript caption.
        used.append(
            dict(
                panel=letter,
                reason=panel.reason,
                subject_id=subject_id,
                record_id=record_id,
                nbp_timestamp_s=t_nbp,
                has_pleth=trace.has_pleth,
            )
        )

    # No on-canvas title and no per-panel explanatory text: the figure title and
    # the reason each panel is non-usable live only in the manuscript caption.
    fig.tight_layout()
    png, pdf = figstyle.save(fig, out_dir, "fig02_nonusable")
    return png, pdf, used


# ---------------------------------------------------------------------------
# Figure 3: one clean candidate target
# ---------------------------------------------------------------------------
def _survivor_at(
    inv: pl.DataFrame, subject: str, t_nbp: float, *, tol_s: float = _FIG3_T_TOL_S
) -> dict | None:
    """Return the 15 s survivor cycle for ``subject`` nearest ``t_nbp``, if any.

    A cycle qualifies as a 15 s survivor when it has no reject reason and a
    reperfusion (phase-3) envelope of at least 15 s. Among such cycles for the
    subject, the one whose charted timestamp is closest to ``t_nbp`` is returned,
    provided it lies within ``tol_s`` seconds (so a float round-trip between the
    inventory and other artifacts still matches). Returns ``None`` if the subject
    has no qualifying cycle within tolerance.
    """
    rows = inv.filter(
        (pl.col("subject_id") == subject)
        & pl.col("reject_reason").is_null()
        & (pl.col("phase3_duration_s") >= 15.0)
    )
    if rows.is_empty():
        return None
    rows = rows.with_columns(
        (pl.col("nbp_timestamp_s") - t_nbp).abs().alias("_dt")
    ).sort(["_dt", "nbp_timestamp_s"])
    best = rows.row(0, named=True)
    if best["_dt"] > tol_s:
        return None
    return {k: v for k, v in best.items() if k != "_dt"}


def _select_fig3_event(inv: pl.DataFrame, *, subject_id: str | None = None) -> dict:
    """Choose the Figure 3 cycle deterministically.

    Preference order, all inside the 15 s primary-survivor framing:

    1. The pinned hero cycle :data:`_FIG3_TARGET` (the cleanest four-phase
       exemplar: deep nadir, reperfusion in band, dip at the cuff marker,
       reader-confirmed present).
    2. The pinned fallback :data:`_FIG3_FALLBACK` (the next-cleanest survivor).
    3. The first 15 s survivor by charted timestamp for the requested subject,
       or for each of :data:`PRIMARY_SURVIVORS` in turn.

    When ``subject_id`` is given, the pins are only honored if they belong to
    that subject; otherwise selection falls straight to the generic pick for the
    requested subject.

    Raises
    ------
    ValueError
        If no 15 s survivor cycle can be found at all.
    """
    if subject_id is None:
        for subj, t_nbp in (_FIG3_TARGET, _FIG3_FALLBACK):
            event = _survivor_at(inv, subj, t_nbp)
            if event is not None:
                return event
        candidates = list(PRIMARY_SURVIVORS)
    else:
        candidates = [subject_id]

    for cand in candidates:
        rows = inv.filter(
            (pl.col("subject_id") == cand)
            & pl.col("reject_reason").is_null()
            & (pl.col("phase3_duration_s") >= 15.0)
        ).sort("nbp_timestamp_s")
        if not rows.is_empty():
            return rows.row(0, named=True)

    raise ValueError(
        f"no 15 s survivor event found for {candidates} in the inventory; "
        "Figure 3 needs the credentialed survivors."
    )


def build_figure_3(
    inv: pl.DataFrame, wdb_root: Path, out_dir: Path, *, subject_id: str | None = None
) -> tuple[Path, Path, dict]:
    """Render one clean candidate cuff cycle (Figure 3).

    By default the figure renders the pinned hero cycle (:data:`_FIG3_TARGET`),
    one of the two 15 s survivors (de-identified pseudo-ID ``pXXXXXXXX``). The real
    1 Hz perfusion-index trace is plotted with the occlusion (deep dip) and
    reperfusion (recovery) phases shaded, recomputing the phase anchors with the
    detector so the shading matches the loaded trace exactly. No subject or
    record identifier is drawn on the canvas; the figure title and any
    identifiers live only in the manuscript caption.

    Parameters
    ----------
    inv : polars.DataFrame
        The event inventory (used to locate the survivor event).
    wdb_root : pathlib.Path
        Root of the WDB record tree.
    out_dir : pathlib.Path
        Directory for the rendered files.
    subject_id : str or None
        Restrict the figure to a specific survivor; defaults to the pinned hero
        cycle, then the pinned fallback, then the first available survivor of
        :data:`PRIMARY_SURVIVORS`.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, dict]
        ``(png, pdf, used)`` where ``used`` records the backing event.

    Raises
    ------
    ValueError
        If no 15 s survivor cycle is present in the inventory.
    """
    figstyle.apply_style()

    event = _select_fig3_event(inv, subject_id=subject_id)

    subj = event["subject_id"]
    record_id = str(event["record_id"])
    t_nbp = float(event["nbp_timestamp_s"])
    trace = load_trace(wdb_root, subj, record_id, t_nbp)
    if not trace.has_pleth or trace.pi.size == 0:
        raise ValueError(f"could not load a usable PI trace for {subj}/{record_id}")

    # Recompute the detector anchors on the loaded trace so the shaded phases are
    # exactly consistent with what is plotted. The detector consumes raw PPG, so
    # reload the window and run it directly. Detector time is window-local (NBP at
    # PRE_WINDOW_S) and converts to the recentered axis (NBP at 0) below.
    record_dir = _resolve_record_dir(wdb_root, subj, record_id)
    master_fs = _parse_master_fs(record_dir / f"{record_id}.hea")
    windowed = _load_pleth_window(record_dir / record_id, master_fs, t_nbp)
    if windowed is None:
        raise ValueError(f"could not reload PPG window for {subj}/{record_id}")
    pleth, fs_native = windowed
    finite = np.isfinite(pleth)
    pleth_clean = np.where(finite, pleth, np.nanmedian(pleth))
    result = detect_cuff_event(pleth_clean, fs_native, PRE_WINDOW_S)

    occ_start = result.t_occlusion_start_s - PRE_WINDOW_S
    deflate_start = result.t_deflate_start_s - PRE_WINDOW_S
    release = result.t_release_s - PRE_WINDOW_S
    pre_mean = result.pre_event_pi_mean

    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    # Zoom to the cuff cycle: a margin around the occlusion-reperfusion window.
    x_lo = occ_start - 60.0
    x_hi = release + 60.0
    view = (trace.t_local >= x_lo) & (trace.t_local <= x_hi)

    # Phase shading: occlusion (deep dip) then reperfusion (recovery to release).
    ax.axvspan(
        occ_start,
        deflate_start,
        color=figstyle.WASH_OCCLUSION,
        zorder=0,
        label="occlusion (deep dip)",
    )
    ax.axvspan(
        deflate_start,
        release,
        color=figstyle.WASH_REPERFUSION,
        zorder=0,
        label="reperfusion (recovery)",
    )

    # Raw 1 Hz PI (faint) plus the 5 s rolling median (prominent), the latter
    # being the envelope the detector thresholds on.
    pi_smooth = _rolling_median_1hz(trace.pi)
    ax.plot(
        trace.t_local[view],
        trace.pi[view],
        color=figstyle.COLOR_USABLE,
        linewidth=0.8,
        alpha=0.35,
        zorder=2,
        label="perfusion index (1 Hz)",
    )
    ax.plot(
        trace.t_local[view],
        pi_smooth[view],
        color=figstyle.COLOR_USABLE,
        linewidth=2.0,
        zorder=3,
        label="perfusion index (5 s median)",
    )

    y_top = max(float(np.nanpercentile(trace.pi[view], 99)) * 1.2, pre_mean * 1.2)

    # Pre-cuff baseline reference.
    ax.axhline(pre_mean, color=figstyle.GRAPHITE, linewidth=0.9, linestyle=":", zorder=2)
    ax.text(
        x_lo + 2.0,
        pre_mean + 0.01 * y_top,
        "pre-cuff baseline",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=figstyle.GRAPHITE,
    )
    ax.axvline(0.0, color=figstyle.SLATE, linewidth=0.9, linestyle="--", zorder=2)
    ax.text(
        1.0,
        0.02 * y_top,
        "charted cuff timestamp",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=figstyle.SLATE,
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("time relative to charted cuff timestamp (s)")
    ax.set_ylabel("perfusion index (%)")
    # No on-canvas title and no subject/record identifier on the canvas: the
    # figure title and any identifiers live only in the manuscript caption,
    # matching the newer builders and the project house rule.
    ax.legend(loc="upper right", fontsize=7.8)

    # No on-canvas subtitle, title, or interpretive prose: the morphology
    # description and the laterality caveat live only in the manuscript caption.
    fig.tight_layout()
    png, pdf = figstyle.save(fig, out_dir, "fig03_clean_candidate")
    used = dict(
        subject_id=subj,
        record_id=record_id,
        nbp_timestamp_s=t_nbp,
        phase3_duration_s=float(result.phase3_duration_s),
        phase2_duration_s=float(result.phase2_duration_s),
        pre_event_pi_mean=float(pre_mean),
    )
    return png, pdf, used


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--which",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which figure(s) to render (default: all).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Use the open MIMIC-IV-Demo layout (no credentialing). The demo has no "
            "waveform records, so only Figure 1 renders; Figures 2 and 3 fail clean."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root of the data/ tree, used to derive the demo WDB root (default: data).",
    )
    parser.add_argument(
        "--wdb-root",
        type=Path,
        default=None,
        help=(
            "WDB record-tree root (the directory containing RECORDS). "
            "Defaults to the demo layout under --data-root in demo mode, "
            f"otherwise to the ${ENV_WDB_ROOT} environment variable (required "
            "for the waveform figures when not in demo mode)."
        ),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help=(
            "Per-event inventory CSV used to compute the funnel and locate "
            f"records. Defaults to the ${ENV_INVENTORY} environment variable; "
            "required if that is unset."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Output directory for the rendered figures (default: figures).",
    )
    parser.add_argument(
        "--survivor",
        choices=list(PRIMARY_SURVIVORS),
        default=None,
        help="Which 15 s survivor to plot in Figure 3 (default: first available).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Render the requested figures from real data.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on missing data).
    """
    args = _parse_args(argv)
    # Determinism: figures use no random state, but pin the global stream anyway
    # so any future stochastic styling stays reproducible.
    np.random.default_rng(GLOBAL_SEED)

    want = {"1", "2", "3"} if args.which == "all" else {args.which}
    needs_waveforms = bool(want & {"2", "3"})

    # The WDB root is only consulted for the waveform figures. Resolve it from
    # the explicit flag, then the demo layout under --data-root, then the
    # CUFFCRT_WDB_ROOT environment variable. There is no machine default; if
    # waveforms are needed and none of these is set, fail loud below.
    if args.wdb_root is not None:
        wdb_root: Path | None = args.wdb_root
    elif args.demo:
        wdb_root = resolve_wdb_root(args.data_root, demo=True)
    else:
        wdb_root = env_path(ENV_WDB_ROOT)

    try:
        inventory = resolve_configured_path(
            args.inventory,
            env_var=ENV_INVENTORY,
            flag="--inventory",
            what="event inventory CSV",
        )
    except DataPathNotConfiguredError as exc:
        logger.error("{}", exc)
        return 2

    logger.info("which={} demo={}", args.which, args.demo)
    logger.info("wdb_root={}", wdb_root)
    logger.info("inventory={}", inventory)
    logger.info("out={}", args.out)

    if wdb_root is not None and args.out.resolve() == wdb_root.resolve():
        logger.error("--out must differ from the WDB root; refusing to overwrite input.")
        return 2

    # Figure 1 needs only the inventory CSV. Figures 2 and 3 need the WDB
    # waveforms. The open MIMIC-IV-Demo carries clinical tables only (no
    # waveform records), so the waveform figures cannot be built in demo mode;
    # fail clean with the same data/README.md pointer the rest of the pipeline
    # uses.
    try:
        require_path(inventory, what="event inventory CSV")
    except DataNotAvailableError as exc:
        logger.error("{}", exc)
        return 2

    inv = load_inventory(inventory)
    result = aggregate_funnel(inv)
    p, s = result.primary, result.sensitivity
    logger.info(
        "funnel: {} candidates, no_pleth {}, primary {}/{} ({}), sensitivity {}/{} ({})",
        result.n_candidates,
        _funnel_row("excluded_no_pleth", result),
        p.n_events,
        p.n_patients,
        p.subjects,
        s.n_events,
        s.n_patients,
        s.subjects,
    )

    if "1" in want:
        png, pdf = build_figure_1(result, args.out)
        logger.info("Figure 1 -> {} , {}", png, pdf)

    if needs_waveforms:
        if wdb_root is None:
            logger.error(
                "WDB waveform record tree is not configured; pass --wdb-root or "
                "set the {} environment variable. Figures 2 and 3 require the "
                "credentialed MIMIC-IV-WDB; Figure 1 renders from the inventory "
                "CSV alone.",
                ENV_WDB_ROOT,
            )
            return 2
        try:
            require_path(wdb_root, what="WDB waveform record tree")
        except DataNotAvailableError as exc:
            extra = (
                " The open MIMIC-IV-Demo provides clinical tables only, with no waveform "
                "records, so Figures 2 and 3 require the credentialed MIMIC-IV-WDB. Figure 1 "
                "renders from the inventory CSV alone."
                if args.demo
                else ""
            )
            logger.error("{}{}", exc, extra)
            return 2

    if "2" in want:
        # The needs_waveforms guard above returns early when wdb_root is None,
        # so it is a definite Path here. Restate that for the type checker; this
        # never fires for any reachable input.
        assert wdb_root is not None
        png, pdf, used = build_figure_2(inv, wdb_root, args.out)
        logger.info("Figure 2 -> {} , {}", png, pdf)
        for u in used:
            logger.info(
                "  panel {}: {} {}/{} t={:.0f}s",
                u["panel"],
                u["reason"],
                u["subject_id"],
                u["record_id"],
                u["nbp_timestamp_s"],
            )

    if "3" in want:
        # Guaranteed non-None by the needs_waveforms guard above (see Figure 2).
        assert wdb_root is not None
        try:
            png, pdf, used = build_figure_3(inv, wdb_root, args.out, subject_id=args.survivor)
            logger.info("Figure 3 -> {} , {}", png, pdf)
            logger.info(
                "  clean candidate: {}/{} envelope {:.0f}s",
                used["subject_id"],
                used["record_id"],
                used["phase3_duration_s"],
            )
        except ValueError as exc:
            logger.error("Figure 3 failed: {}", exc)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
