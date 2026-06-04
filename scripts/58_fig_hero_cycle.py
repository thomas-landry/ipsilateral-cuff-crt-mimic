"""Hero figure: one representative occlusion-reperfusion perfusion-index cycle (step 58).

A single full-width annotated time series of one real charted cuff inflation, drawn
from the cycles that carry the occlusion-reperfusion signature (a morphology-based
laterality estimate; there is no ground-truth limb information in the source data, so
this remains an estimate). The trace shows a stable perfusion baseline, a collapse
toward zero while the cuff occludes the limb, then a climb back after release: the
continuous analog of capillary refill, captured because the pulse oximeter probe sits
on the same limb as the cuff.

How the trace is recovered (honest about its data)
--------------------------------------------------
The raw 1 Hz perfusion-index samples are not redistributed in this repository
(PhysioNet Data Use Agreement). The only on-disk artifact that carries the actual
trace of a single cycle is the per-cycle card image rendered by an earlier step, drawn
over a fixed ``[-60, +90] s`` window around the charted blood-pressure timestamp and
normalized to its in-window median. This script:

1. Selects the representative cycle deterministically from the per-cycle feature table
   (nearest to the cohort median on standardized nadir depth and recovery duration),
   with a documented legibility guard (below).
2. Recovers the trace by digitizing that real card image. The x-axis transform is
   recovered from the card's own evenly spaced tick marks against the renderer's fixed
   ``[-60, +90] s`` window; the y-axis is re-normalized to the trace's own median,
   reproducing the renderer's normalization. The y-axis is therefore the perfusion
   index relative to its own baseline (dimensionless), not a fabricated physical scale.
3. Re-plots the recovered samples as a clean vector line and overlays phase washes and
   vertical guides positioned by *that cycle's own measured landmarks* (occlusion
   onset, nadir, release), in seconds, read straight from the feature table. Every
   annotated number is a measured value; none are invented.

Legibility guard
----------------
A small number of signature cycles end with an extreme reperfusion overshoot. The
per-cycle median normalization then compresses the entire occlusion and recovery into
a flat floor under one tall spike, hiding the very morphology this figure exists to
show. The selection therefore walks the median-nearest ranking and skips any candidate
whose recovered trace peaks above a fixed multiple of baseline (:data:`PEAK_GUARD_X`),
choosing the nearest *legible* cycle. The guard is fixed and disclosed; it changes
which representative cycle is shown, never the data itself.

Determinism
-----------
The selection is fully determined by the feature table. ``GLOBAL_SEED`` is used only as
a defensive tie-break should two cycles sit at an identical median distance.

De-identification
-----------------
No subject pseudo-id, record id, or absolute clock timestamp is drawn on the canvas.
The recovered card image carries none of these on its own canvas, and the provenance
footnote refers to the representative cycle without any identifier.

Examples
--------
::

    uv run python scripts/58_fig_hero_cycle.py

Outputs ``figures/fig_hero_cycle.{pdf,png}`` (PNG at 400 dpi).
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
from loguru import logger
from PIL import Image

from cuffcrt import figstyle
from cuffcrt._seed import GLOBAL_SEED

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_DIR = DEFAULT_REPO / "data" / "interim" / "events"
DEFAULT_MANIFEST_CSV = DEFAULT_REPO / "results" / "gallery" / "gallery_manifest.csv"
DEFAULT_OUT = DEFAULT_REPO / "figures"
SLUG = "fig_hero_cycle"

# --- Renderer constants (must match the per-cycle card renderer) --------------
# The per-cycle card was rendered over this fixed window, relative to the charted
# blood-pressure timestamp (t = 0). The x-axis transform is recovered against it.
RENDER_WINDOW_LO_S = -60.0
RENDER_WINDOW_HI_S = 90.0
# The card x-axis carries ticks at these second values (matplotlib AutoLocator on a
# [-60, 90] axis). Used to fit the pixel->second transform from the detected tick
# marks. Over-determined: a least-squares line through all matched ticks.
EXPECTED_X_TICKS_S = (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0)

# --- Selection constants -----------------------------------------------------
# Legibility guard: skip cycles whose recovered trace peaks above this multiple of
# baseline (their median-normalized render hides the morphology under one spike).
PEAK_GUARD_X = 6.0

# --- Visual constants (Okabe-Ito derived washes from the house style) --------
INK = figstyle.INK
GRAPHITE = figstyle.GRAPHITE
SLATE = figstyle.SLATE
WASH_OCCLUSION = figstyle.WASH_OCCLUSION
WASH_REPERFUSION = figstyle.WASH_REPERFUSION


@dataclass(frozen=True)
class Cycle:
    """One representative cuff cycle: its identifiers, card image, and landmarks.

    Attributes
    ----------
    card_id : str
        Per-cycle card identifier (used only in logs, never drawn on the canvas).
    subject_id, record_id : str
        De-identified subject pseudo-id and waveform record id (used only in logs
        for provenance; never drawn on the canvas).
    image_path : pathlib.Path
        Absolute path to the rendered per-cycle card image (the trace source).
    t_nbp_s : float
        Charted blood-pressure timestamp (record-relative seconds); the card's local
        time axis is centered here (t = 0).
    occ_local_s, nadir_local_s, release_local_s : float
        Occlusion onset, nadir, and release, in seconds relative to ``t_nbp_s``.
    phase2_s, phase3_s : float
        Descent duration (onset to nadir) and the event-defining sub-occlusion run
        length, in seconds.
    nadir_depth_frac : float
        Nadir perfusion index as a fraction of the pre-event baseline.
    dist : float
        Standardized Euclidean distance to the cohort median (selection metric).
    """

    card_id: str
    subject_id: str
    record_id: str
    image_path: Path
    t_nbp_s: float
    occ_local_s: float
    nadir_local_s: float
    release_local_s: float
    phase2_s: float
    phase3_s: float
    nadir_depth_frac: float
    dist: float


def _load_signature_cycles(events_dir: Path, manifest_csv: Path) -> pl.DataFrame:
    """Load every signature cycle with the landmarks needed to place annotations.

    Reads all per-subject event parquets, keeps cycles flagged with the
    occlusion-reperfusion signature, and joins the card manifest to attach each
    cycle's rendered card image path.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record ``events_p*.parquet`` files.
    manifest_csv : pathlib.Path
        The per-cycle card manifest CSV.

    Returns
    -------
    polars.DataFrame
        One row per signature cycle, carrying landmark scalars, the card id, and the
        card image path.
    """
    paths = sorted(events_dir.glob("events_p*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no event parquets under {events_dir}")
    events = pl.concat([pl.read_parquet(p) for p in paths])
    sig = events.filter(pl.col("is_occlusion_signature"))

    manifest = pl.read_csv(manifest_csv, infer_schema_length=20000)
    cards = manifest.filter(
        (pl.col("stratum") == "detector_positive")
        & (pl.col("is_occlusion_signature"))
        & (pl.col("image_path").is_not_null())
    ).select(["card_id", "subject_id", "record_id", "t_nbp", "image_path"])

    # Join on the natural (subject, record, charttime) triple, rounding the float
    # timestamp to milliseconds to absorb formatting noise without colliding distinct
    # cycles. Cast the id columns to a common string type; the manifest CSV infers
    # record_id as an integer.
    sig = sig.with_columns(
        subject_id=pl.col("subject_id").cast(pl.Utf8),
        record_id=pl.col("record_id").cast(pl.Utf8),
        _t_key=pl.col("nbp_timestamp_s").round(3),
    )
    cards = cards.with_columns(
        subject_id=pl.col("subject_id").cast(pl.Utf8),
        record_id=pl.col("record_id").cast(pl.Utf8),
        _t_key=pl.col("t_nbp").round(3),
    )
    joined = sig.join(cards, on=["subject_id", "record_id", "_t_key"], how="inner")
    if joined.height == 0:
        raise RuntimeError("no signature cycle joined to a rendered card image")
    return joined


def _rank_by_median_distance(cycles: pl.DataFrame) -> pl.DataFrame:
    """Rank cycles by standardized Euclidean distance to the cohort median.

    The two selection features are nadir depth fraction and the event-defining
    recovery (phase-3) duration, standardized to zero mean and unit standard
    deviation across the signature cohort. The target is the per-feature median. Ties
    (rare) are broken by a stable seed-derived jitter so the result is reproducible.

    Parameters
    ----------
    cycles : polars.DataFrame
        Signature cycles with ``nadir_depth_frac`` and ``phase3_duration_s``.

    Returns
    -------
    polars.DataFrame
        ``cycles`` with a ``_dist`` column, sorted ascending (nearest first).
    """
    nadir = cycles["nadir_depth_frac"].to_numpy()
    phase3 = cycles["phase3_duration_s"].to_numpy()

    def _z(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        mu = float(x.mean())
        sd = float(x.std(ddof=0))
        sd = sd if sd > 0 else 1.0
        return (x - mu) / sd, mu, sd

    zn, mun, sdn = _z(nadir)
    zp, mup, sdp = _z(phase3)
    tn = (float(np.median(nadir)) - mun) / sdn
    tp = (float(np.median(phase3)) - mup) / sdp
    dist = np.hypot(zn - tn, zp - tp)

    # Defensive, deterministic tie-break: a tiny seed-derived jitter, far below the
    # smallest real distance gap, so identical distances order reproducibly.
    rng = np.random.default_rng(GLOBAL_SEED)
    jitter = rng.uniform(0.0, 1e-9, size=dist.size)
    dist = dist + jitter

    return cycles.with_columns(_dist=pl.Series(dist)).sort("_dist")


def _detect_axes_box(gray: np.ndarray) -> tuple[int, int]:
    """Locate the left and bottom axis spines in a card image.

    The left spine is the darkest near-vertical column in the left margin; the bottom
    spine is the longest dark row. Both are robust across cards because the renderer
    draws the same despined left/bottom axes.

    Parameters
    ----------
    gray : numpy.ndarray
        Grayscale card image (mean of RGB), shape ``(H, W)``.

    Returns
    -------
    tuple[int, int]
        ``(left_col, bottom_row)`` pixel indices of the two spines.
    """
    dark = gray < 90
    left_col = int(np.argmax(dark[:, :200].sum(axis=0)))
    bottom_row = int(np.argmax(dark.sum(axis=1)))
    return left_col, bottom_row


def _detect_x_ticks(gray: np.ndarray, left_col: int, bottom_row: int) -> list[float]:
    """Detect x-axis tick centers (pixel columns) below the bottom spine.

    Tick marks point downward just below the bottom spine; this clusters the dark
    columns in that thin band into tick centers, keeping only those at or right of the
    left spine (dropping any leftover axis-label glyph pixels).

    Parameters
    ----------
    gray : numpy.ndarray
        Grayscale card image.
    left_col, bottom_row : int
        Spine pixel positions from :func:`_detect_axes_box`.

    Returns
    -------
    list[float]
        Sorted tick-center pixel columns.
    """
    dark = gray < 90
    band = dark[bottom_row + 2 : bottom_row + 6, :].sum(axis=0)
    cols = [c for c in range(gray.shape[1]) if band[c] >= 2]
    if not cols:
        raise RuntimeError("no x-axis tick marks detected")
    centers: list[float] = []
    run = [cols[0]]
    for c in cols[1:]:
        if c - run[-1] <= 2:
            run.append(c)
        else:
            centers.append(float(np.mean(run)))
            run = [c]
    centers.append(float(np.mean(run)))
    return [c for c in centers if c >= left_col - 3]


def _fit_x_transform(tick_px: list[float]) -> tuple[float, float]:
    """Fit the pixel-column to local-second transform from detected x ticks.

    Matches the detected tick centers to :data:`EXPECTED_X_TICKS_S` (one-to-one, in
    order) and fits ``px = slope * seconds + intercept`` by least squares, so the
    transform is over-determined and robust to one noisy tick.

    Parameters
    ----------
    tick_px : list[float]
        Detected x-tick centers in pixels.

    Returns
    -------
    tuple[float, float]
        ``(slope, intercept)`` for ``px = slope * seconds + intercept``.

    Raises
    ------
    RuntimeError
        If the detected tick count does not match the expected count.
    """
    if len(tick_px) != len(EXPECTED_X_TICKS_S):
        raise RuntimeError(
            f"expected {len(EXPECTED_X_TICKS_S)} x ticks, detected {len(tick_px)}"
        )
    secs = np.asarray(EXPECTED_X_TICKS_S, dtype=float)
    px = np.asarray(sorted(tick_px), dtype=float)
    slope, intercept = np.polyfit(secs, px, 1)
    return float(slope), float(intercept)


def _digitize_card(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Digitize the black trace of a card image into ``(seconds, pi_norm)``.

    Recovers the x transform from the card's own tick marks (against the fixed render
    window) and the y values as a baseline-normalized perfusion index by dividing
    pixel height above the bottom spine by the trace median, reproducing the
    renderer's in-window median normalization without needing the absolute axis scale.

    Parameters
    ----------
    image_path : pathlib.Path
        Absolute path to the rendered per-cycle card image.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(seconds, pi_norm)``, the digitized trace sorted by time, where ``seconds``
        is local time relative to the charted blood-pressure event and ``pi_norm`` is
        the perfusion index relative to its own median baseline.
    """
    arr = np.asarray(Image.open(image_path).convert("RGB"))
    gray = arr.mean(axis=2)
    height, width = gray.shape
    left_col, bottom_row = _detect_axes_box(gray)
    slope, intercept = _fit_x_transform(_detect_x_ticks(gray, left_col, bottom_row))

    # Trace pixels: near-black, strictly inside the plot box (excludes the spines and
    # the axis-label glyphs in the margins).
    interior = np.zeros_like(gray, dtype=bool)
    interior[5:bottom_row, left_col + 2 : width - 5] = True
    trace = (gray < 80) & interior
    cols = np.where(trace.any(axis=0))[0]
    if cols.size == 0:
        raise RuntimeError(f"no trace pixels found in {image_path}")

    x_px: list[float] = []
    y_px: list[float] = []
    for c in range(int(cols.min()), int(cols.max()) + 1):
        rows = np.where(trace[:, c])[0]
        if rows.size:
            x_px.append(float(c))
            y_px.append(float(rows.mean()))  # centroid of the (antialiased) line

    seconds = (np.asarray(x_px) - intercept) / slope
    # Pixel height above the bottom spine is proportional to PI; normalize to the
    # trace median so baseline reads ~1.0, matching the renderer's normalization.
    height_px = bottom_row - np.asarray(y_px)
    pi_norm = height_px / float(np.median(height_px))

    order = np.argsort(seconds, kind="stable")
    return seconds[order], pi_norm[order]


def _select_cycle(events_dir: Path, manifest_csv: Path) -> tuple[Cycle, np.ndarray, np.ndarray]:
    """Choose the representative legible cycle and return it with its trace.

    Ranks the signature cohort by distance to the cohort median, then walks the
    ranking and returns the first cycle whose recovered trace peaks at or below
    :data:`PEAK_GUARD_X` (so the morphology is legible), digitizing candidates lazily
    until one passes.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record ``events_p*.parquet`` files.
    manifest_csv : pathlib.Path
        The per-cycle card manifest CSV.

    Returns
    -------
    tuple[Cycle, numpy.ndarray, numpy.ndarray]
        The chosen :class:`Cycle` and its digitized ``(seconds, pi_norm)`` trace.
    """
    ranked = _rank_by_median_distance(_load_signature_cycles(events_dir, manifest_csv))
    for row in ranked.iter_rows(named=True):
        # Manifest image paths are stored repository-relative; resolve against the
        # repo root.
        image_path = DEFAULT_REPO / row["image_path"]
        if not image_path.exists():
            logger.warning("card image missing, skipping: {}", image_path)
            continue
        seconds, pi_norm = _digitize_card(image_path)
        peak = float(pi_norm.max())
        if peak > PEAK_GUARD_X:
            logger.info(
                "skip {} (median-normalized peak {:.0f}x baseline exceeds guard)",
                row["card_id"],
                peak,
            )
            continue
        t_nbp = float(row["nbp_timestamp_s"])
        cycle = Cycle(
            card_id=row["card_id"],
            subject_id=row["subject_id"],
            record_id=row["record_id"],
            image_path=image_path,
            t_nbp_s=t_nbp,
            occ_local_s=float(row["t_occlusion_start_s"]) - t_nbp,
            nadir_local_s=float(row["t_nadir_s"]) - t_nbp,
            release_local_s=float(row["t_release_s"]) - t_nbp,
            phase2_s=float(row["phase2_duration_s"]),
            phase3_s=float(row["phase3_duration_s"]),
            nadir_depth_frac=float(row["nadir_depth_frac"]),
            dist=float(row["_dist"]),
        )
        logger.info(
            "selected {} (subject {}, record {}): nadir_frac {:.3f}, phase3 {:.0f}s, "
            "peak {:.1f}x baseline, median-distance {:.3f}",
            cycle.card_id,
            cycle.subject_id,
            cycle.record_id,
            cycle.nadir_depth_frac,
            cycle.phase3_s,
            peak,
            cycle.dist,
        )
        return cycle, seconds, pi_norm
    raise RuntimeError("no legible signature cycle found under the peak guard")


def _span_bracket(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    tick: float,
    label: str,
    color: str,
    *,
    label_x: float,
    ha: str,
) -> None:
    """Draw a downward span bracket with a label anchored at ``label_x``.

    The bracket rail sits at ``y`` with end ticks pointing downward toward the trace.
    Because adjacent phases can be narrow, the label is placed at an explicit
    ``label_x`` (not forced to the bracket center) so neighboring labels do not
    collide.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x0, x1 : float
        Bracket extent in data x (seconds).
    y : float
        Bracket rail position in data y.
    tick : float
        End-tick length in data y.
    label : str
        Text drawn above the rail.
    color : str
        Bracket and text color.
    label_x : float
        Data-x anchor for the label.
    ha : str
        Horizontal alignment of the label at ``label_x``.
    """
    ax.plot(
        [x0, x0, x1, x1],
        [y - tick, y, y, y - tick],
        color=color,
        lw=1.0,
        clip_on=False,
        solid_capstyle="round",
    )
    ax.annotate(
        label,
        xy=(label_x, y),
        xytext=(0, 3),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=8.0,
        color=color,
        clip_on=False,
    )


def build_figure(*, events_dir: Path, manifest_csv: Path, out_dir: Path) -> tuple[Path, Path]:
    """Build and save the hero occlusion-reperfusion cycle figure.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record ``events_p*.parquet`` files.
    manifest_csv : pathlib.Path
        The per-cycle card manifest CSV.
    out_dir : pathlib.Path
        Output directory for the PDF and PNG.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        The written ``(png_path, pdf_path)``.
    """
    figstyle.apply_style()
    cycle, seconds, pi_norm = _select_cycle(events_dir, manifest_csv)

    # Restrict to a tight window around the event for a legible hero panel.
    view_lo, view_hi = -42.0, 68.0
    keep = (seconds >= view_lo) & (seconds <= view_hi)
    t = seconds[keep]
    y = pi_norm[keep]
    # A calm y-top: round the in-view peak up to a tenth, with headroom above for a
    # caption-and-bracket band. All real samples remain visible; nothing is clipped.
    y_data_top = float(np.ceil(y.max() * 10) / 10)
    y_top = y_data_top + 0.95

    fig, ax = plt.subplots(figsize=(7.09, 3.74))  # ~180 x 95 mm
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.165)

    # Phase washes: occlusion (onset -> nadir) and reperfusion (nadir -> release).
    ax.axvspan(cycle.occ_local_s, cycle.nadir_local_s, color=WASH_OCCLUSION, zorder=0, lw=0)
    ax.axvspan(
        cycle.nadir_local_s, cycle.release_local_s, color=WASH_REPERFUSION, zorder=0, lw=0
    )

    # Faint half-baseline reference line (the occlusion threshold is half baseline).
    ax.axhline(0.5, color=SLATE, lw=0.7, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.annotate(
        "half of baseline",
        xy=(view_hi, 0.5),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=SLATE,
    )
    # Baseline reference line at 1.0 (the perfusion baseline the trace returns to).
    ax.axhline(1.0, color=SLATE, lw=0.7, ls=(0, (1, 2)), alpha=0.45, zorder=1)

    # Headroom band geometry (above the data): a row for the phase-duration brackets,
    # then a row for the plain-language landmark captions on top.
    bracket_y = y_data_top + 0.30
    bracket_tick = 0.10
    caption_y = y_data_top + 0.72

    # The recovered trace.
    ax.plot(t, y, color=INK, lw=1.5, solid_joinstyle="round", zorder=3)

    # Slim vertical guides at the three landmarks, reaching up into the bracket band.
    for x_landmark in (cycle.occ_local_s, cycle.nadir_local_s, cycle.release_local_s):
        ax.plot(
            [x_landmark, x_landmark],
            [0.0, bracket_y + bracket_tick],
            color=SLATE,
            lw=0.8,
            alpha=0.8,
            zorder=2,
        )

    # Plain-language landmark captions, horizontal, at the top of the headroom band.
    for x_landmark, text, ha, dx in (
        (cycle.occ_local_s, "cuff occludes limb", "right", -3),
        (cycle.nadir_local_s, "deepest point", "center", 0),
        (cycle.release_local_s, "recovered to baseline", "left", 3),
    ):
        ax.annotate(
            text,
            xy=(x_landmark, caption_y),
            xytext=(dx, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.0,
            color=GRAPHITE,
        )

    # Nadir marker on the trace.
    ax.scatter(
        [cycle.nadir_local_s],
        [cycle.nadir_depth_frac],
        s=24,
        color=INK,
        zorder=4,
        clip_on=False,
    )
    # Nadir depth annotation (measured nadir fraction of baseline, one decimal so the
    # value is not rounded into a deceptively tidy number).
    ax.annotate(
        f"nadir {cycle.nadir_depth_frac * 100:.1f}% of baseline",
        xy=(cycle.nadir_local_s, cycle.nadir_depth_frac),
        xytext=(13, 30),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=GRAPHITE, lw=0.8, shrinkA=0, shrinkB=4),
    )

    # Perfusion-baseline annotation near the left, on the 1.0 reference line.
    ax.annotate(
        "perfusion baseline",
        xy=(view_lo + 2, 1.0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=GRAPHITE,
    )

    # Duration brackets in the headroom band: each carries that cycle's measured
    # seconds. The phases are narrow, so labels are anchored to the outer edge of each
    # phase and aligned outward to avoid colliding with each other.
    _span_bracket(
        ax,
        cycle.occ_local_s,
        cycle.nadir_local_s,
        bracket_y,
        bracket_tick,
        f"occlusion descent  {cycle.phase2_s:.0f} s",
        GRAPHITE,
        label_x=cycle.occ_local_s - 1.0,
        ha="right",
    )
    _span_bracket(
        ax,
        cycle.nadir_local_s,
        cycle.release_local_s,
        bracket_y,
        bracket_tick,
        f"reperfusion recovery  {cycle.phase3_s:.0f} s",
        GRAPHITE,
        label_x=cycle.release_local_s + 1.0,
        ha="left",
    )

    # Axes furniture.
    ax.set_xlim(view_lo, view_hi)
    ax.set_ylim(0.0, y_top)
    ax.set_xlabel("time relative to cuff inflation (s)")
    ax.set_ylabel("perfusion index (relative to baseline)")
    ax.set_xticks([-40, -20, 0, 20, 40, 60])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.grid(axis="y", which="major")
    ax.margins(x=0)

    # De-identified provenance footnote. No subject pseudo-id, record id, or absolute
    # clock timestamp is drawn: the cycle is described only by its relation to the
    # cohort medians and its self-relative normalization.
    fig.text(
        0.085,
        0.018,
        "Representative cycle, nearest the cohort medians for nadir depth and "
        "recovery duration. Perfusion index normalized to its own baseline.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=SLATE,
    )

    png, pdf = figstyle.save(fig, out_dir, SLUG)
    plt.close(fig)
    logger.info("wrote {} and {}", png, pdf)
    return png, pdf


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Render the hero occlusion-reperfusion perfusion-index cycle figure "
            "(one representative real charted cuff inflation)."
        )
    )
    p.add_argument("--events_dir", type=Path, default=DEFAULT_EVENTS_DIR)
    p.add_argument("--manifest_csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_argparser().parse_args(argv)
    build_figure(
        events_dir=args.events_dir,
        manifest_csv=args.manifest_csv,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
