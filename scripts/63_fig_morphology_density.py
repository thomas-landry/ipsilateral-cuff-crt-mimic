"""Across-cycle density of the signature perfusion-index morphology (step 63).

Each of the 268 same-limb occlusion-reperfusion signature cycles is reconstructed on a
common time axis (t = 0 at occlusion start), baseline normalized to 1.0, from its
measured per-cycle landmark scalars. Stacking the 268 reconstructed trajectories into a
2D density image yields a single picture that shows both the canonical shape (descent
to an occlusion nadir, then a reperfusion climb toward baseline) and its cycle-to-cycle
spread. A pointwise median curve and a 25th-to-75th percentile band are overlaid.

Honesty notes (read before reusing this figure):
- Laterality is a MORPHOLOGY-BASED ESTIMATE. There is no ground-truth limb assignment
  in the data; "same-limb signature" means a cuff cycle whose perfusion-index window
  has the occlusion-reperfusion morphology, nothing more.
- The trajectories are RECONSTRUCTED from per-cycle scalars (descent duration, nadir
  depth, recovery duration, recovery fraction at window end), NOT resampled from raw
  oximeter waveform arrays. The events parquet holds landmark scalars, not sample
  arrays. Each cycle is drawn as a piecewise-linear curve through its measured
  landmarks. The figure therefore depicts the distribution of the measured landmarks,
  rendered as time courses; it is not a pixel-faithful waveform overlay. The on-canvas
  axis label and the caption state this.
- The recovery (right) leg of the median curve reflects the fixed analysis window used
  to define the event (a 15-second reperfusion-run floor), not a measured reperfusion
  latency: each cycle is drawn only out to its measured window end, so the right leg
  describes how the window is read, not how fast perfusion physically returns. The
  caption states this explicitly.
- No outcome, biomarker, causal, or accuracy claim is made or implied.

Data provenance (all real, all on disk):
- data/interim/events/events_p*.parquet : 19 per-record files, 9,224 charted cuff
  cycles total; the 268 rows with is_occlusion_signature == True are the signature
  cohort plotted here (15 of 19 subjects). Columns used: phase2_duration_s,
  nadir_depth_frac, phase3_duration_s, recovery_fraction_at_window_end (all 268
  non-null, verified).

De-identification
-----------------
No subject pseudo-id, record id, or absolute clock timestamp is drawn on the canvas;
the figure shows an aggregate density and summary curves over the cohort only.

Examples
--------
::

    uv run python scripts/63_fig_morphology_density.py

Outputs (>=400 dpi PNG plus vector PDF):

    figures/fig_morphology_density.png
    figures/fig_morphology_density.pdf

Deterministic: no stochastic step is required, but GLOBAL_SEED is pinned for
provenance and would seed any future jitter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from cuffcrt import figstyle
from cuffcrt._seed import GLOBAL_SEED

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_DIR = DEFAULT_REPO / "data" / "interim" / "events"
DEFAULT_OUT = DEFAULT_REPO / "figures"
SLUG = "fig_morphology_density"

# --- Reconstruction grid -----------------------------------------------------
# Common time axis, t = 0 at occlusion start. A short pre-occlusion baseline plateau
# anchors the eye at 1.0 before the descent begins.
T_MIN = -10.0
T_MAX = 60.0
N_T = 561          # 0.125 s resolution across [-10, 60]
T_GRID = np.linspace(T_MIN, T_MAX, N_T)

# Density image vertical extent. Recovery overshoot has a long upper tail (a handful of
# cycles climb well above baseline); the central core and the median/IQR carry the
# shape, so the displayed band is capped for legibility while the median and IQR are
# computed from the full, uncapped trajectories.
Y_MIN = 0.70
Y_MAX = 1.70
N_Y = 200
Y_EDGES = np.linspace(Y_MIN, Y_MAX, N_Y + 1)
T_EDGES = np.linspace(T_MIN, T_MAX, N_T + 1)


def load_signature_cycles(events_dir: Path) -> pl.DataFrame:
    """Load the 268 signature cycles with the four landmark columns.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record ``events_p*.parquet`` files.

    Returns
    -------
    polars.DataFrame
        One row per signature cycle, columns: phase2_duration_s, nadir_depth_frac,
        phase3_duration_s, recovery_fraction_at_window_end, subject_id.
    """
    files = sorted(events_dir.glob("events_p*.parquet"))
    if not files:
        raise FileNotFoundError(f"No event parquet files under {events_dir}")
    allev = pl.concat([pl.read_parquet(f) for f in files])
    cols = [
        "phase2_duration_s",
        "nadir_depth_frac",
        "phase3_duration_s",
        "recovery_fraction_at_window_end",
        "subject_id",
    ]
    sig = allev.filter(pl.col("is_occlusion_signature")).select(cols)
    n = sig.height
    if n != 268:
        raise ValueError(f"Expected 268 signature cycles, found {n}")
    nulls = sig.null_count()
    for c in cols[:4]:
        if nulls[c][0] != 0:
            raise ValueError(f"Unexpected nulls in {c}")
    return sig


def reconstruct_one(
    phase2: float, nadir_depth: float, phase3: float, recovery_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct one cycle as a piecewise-linear perfusion-index trajectory.

    Baseline = 1.0. Descent over ``phase2`` seconds from 1.0 to the nadir level
    ``1 - nadir_depth``; reperfusion over ``phase3`` seconds from the nadir to
    ``recovery_frac``. Pre-occlusion (t < 0) is baseline. The cycle is defined only out
    to its measured window end (t = phase2 + phase3); beyond that there is no landmark,
    so those grid points are marked invalid and contribute neither to the density nor
    to the summary curves (honest: no flat-hold extrapolation past the last measured
    point).

    Parameters
    ----------
    phase2 : float
        Descent (occlusion) duration in seconds.
    nadir_depth : float
        Nadir depth as a fraction of baseline (nadir level = 1 - nadir_depth).
    phase3 : float
        Reperfusion (recovery) duration in seconds.
    recovery_frac : float
        Perfusion index at window end as a fraction of baseline.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(values, valid)`` on ``T_GRID``. ``values`` is the reconstructed perfusion
        index; ``valid`` is a boolean mask of grid points the cycle actually spans.
    """
    nadir_level = 1.0 - nadir_depth
    # Guard against zero-length segments (phase2 can be 0 in a few cycles).
    p2 = max(phase2, 1e-6)
    p3 = max(phase3, 1e-6)

    t_descent_end = p2
    t_recovery_end = p2 + p3

    t_pts = np.array([T_MIN, 0.0, t_descent_end, t_recovery_end])
    y_pts = np.array([1.0, 1.0, nadir_level, recovery_frac])

    values = np.interp(T_GRID, t_pts, y_pts)
    valid = T_GRID <= t_recovery_end
    return values, valid


def build_trajectories(sig: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct all 268 trajectories into ``(values, valid)`` (268, N_T)."""
    p2 = sig["phase2_duration_s"].to_numpy()
    nd = sig["nadir_depth_frac"].to_numpy()
    p3 = sig["phase3_duration_s"].to_numpy()
    rf = sig["recovery_fraction_at_window_end"].to_numpy()
    traj = np.empty((sig.height, N_T), dtype=float)
    valid = np.empty((sig.height, N_T), dtype=bool)
    for i in range(sig.height):
        v, m = reconstruct_one(float(p2[i]), float(nd[i]), float(p3[i]), float(rf[i]))
        traj[i] = v
        valid[i] = m
    return traj, valid


def density_image(traj: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Stack trajectories into a smoothed, globally-normalized 2D density image.

    Each (time, perfusion-index) grid point a cycle passes through is counted; only
    grid points within a cycle's measured span contribute. The raw count image is
    lightly Gaussian-smoothed for a continuous read and normalized to its global
    maximum. Returns shape (N_Y, N_T).
    """
    dens = np.zeros((N_Y, N_T), dtype=float)
    # Bin each cycle's valid (time, value) samples into the image.
    y_idx_all = np.clip(np.searchsorted(Y_EDGES, traj, side="right") - 1, 0, N_Y - 1)
    for i in range(traj.shape[0]):
        cols = np.where(valid[i])[0]
        rows = y_idx_all[i, cols]
        np.add.at(dens, (rows, cols), 1.0)
    # Light smoothing: a touch more along y than along time so the band reads as a
    # continuous ridge, not stacked pixels or wisps.
    dens = gaussian_filter(dens, sigma=(2.6, 2.0))
    peak = dens.max()
    if peak > 0:
        dens /= peak
    return dens


def make_colormap() -> LinearSegmentedColormap:
    """White-to-ink continuous density ramp with a hint of the usable-signal hue.

    Light end is pure white (so empty regions vanish into the page); the ramp passes
    through a pale blue toward the saturated usable-signal color and ends near ink,
    giving a calm light-to-dark density read.
    """
    return LinearSegmentedColormap.from_list(
        "fingerprint_density",
        [
            (0.00, "#FFFFFF"),
            (0.18, "#E7EEF4"),
            (0.45, "#9DC3DE"),
            (0.72, figstyle.COLOR_USABLE),
            (1.00, figstyle.INK),
        ],
    )


def build_figure(*, events_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Build and save the across-cycle morphology density figure.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record ``events_p*.parquet`` files.
    out_dir : pathlib.Path
        Output directory for the PDF and PNG.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        The written ``(png_path, pdf_path)``.
    """
    # Pinned for provenance; no stochastic step is used.
    np.random.default_rng(GLOBAL_SEED)
    figstyle.apply_style()

    sig = load_signature_cycles(events_dir)
    n_cycles = sig.height
    n_subjects = sig["subject_id"].n_unique()
    traj, valid = build_trajectories(sig)

    # Pointwise summary curves over cycles that actually span each instant. (Masked
    # percentiles; only time points with a useful number of contributing cycles are
    # drawn, so the curves do not trail off into a thin tail.)
    masked = np.where(valid, traj, np.nan)
    with np.errstate(invalid="ignore"):
        median_curve = np.nanmedian(masked, axis=0)
        q25_curve = np.nanpercentile(masked, 25, axis=0)
        q75_curve = np.nanpercentile(masked, 75, axis=0)
    n_contrib = valid.sum(axis=0)
    draw_summary = n_contrib >= 0.40 * n_cycles  # at least ~40% of cycles span it

    dens = density_image(traj, valid)

    # Cohort landmark medians for honest annotation (exact, from the scalars).
    med_nadir_depth = float(sig["nadir_depth_frac"].median())  # pyright: ignore[reportArgumentType]
    med_nadir_level = 1.0 - med_nadir_depth
    med_descent = float(sig["phase2_duration_s"].median())  # pyright: ignore[reportArgumentType]
    med_recovery = float(sig["phase3_duration_s"].median())  # pyright: ignore[reportArgumentType]

    # --- Figure ---------------------------------------------------------------
    # ~120 x 90 mm.
    fig_w_in = 120.0 / 25.4
    fig_h_in = 90.0 / 25.4
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    fig.subplots_adjust(left=0.115, right=0.965, top=0.945, bottom=0.135)

    cmap = make_colormap()

    # Mask near-empty cells so the page shows through (avoids a flat gray field). The
    # image is already normalized to its global peak (max = 1.0); clip the display
    # ceiling a touch below the peak so the dense descent/nadir core saturates to ink
    # and the spread reads as a continuous wash.
    floor = 0.022
    dens_masked = np.ma.masked_where(dens <= floor, dens)
    vmax = 0.85

    im = ax.imshow(
        dens_masked,
        origin="lower",
        aspect="auto",
        extent=(T_MIN, T_MAX, Y_MIN, Y_MAX),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
        zorder=1,
    )

    # Soft phase washes under the median curve: occlusion (descent) and reperfusion
    # (recovery), keyed to the cohort-median descent/recovery span.
    occ_end = med_descent
    rep_end = med_descent + med_recovery
    ax.axvspan(0.0, occ_end, color=figstyle.WASH_OCCLUSION, alpha=0.55, lw=0, zorder=0)
    ax.axvspan(occ_end, rep_end, color=figstyle.WASH_REPERFUSION, alpha=0.55, lw=0, zorder=0)

    # Baseline reference at 1.0 and the t = 0 hairline.
    ax.axhline(1.0, color=figstyle.SLATE, lw=0.9, ls=(0, (5, 3)), zorder=2)
    ax.axvline(0.0, color=figstyle.SLATE, lw=0.8, ls=(0, (1, 2)), zorder=2)

    # IQR band and median curve, drawn only where enough cycles span the instant. The
    # three percentile curves are lightly Gaussian-smoothed for DISPLAY only
    # (sigma ~0.4 s) to remove sub-second kinks from differing per-cycle breakpoints;
    # the exact (unsmoothed) median nadir is logged and used for the reported value, so
    # no reported number depends on smoothing.
    disp_sigma = 3.0  # grid points (~0.375 s at 0.125 s resolution)
    median_disp = gaussian_filter(median_curve, sigma=disp_sigma)
    q25_disp = gaussian_filter(q25_curve, sigma=disp_sigma)
    q75_disp = gaussian_filter(q75_curve, sigma=disp_sigma)

    t_draw = T_GRID[draw_summary]
    ax.fill_between(
        t_draw,
        q25_disp[draw_summary],
        q75_disp[draw_summary],
        color=figstyle.COLOR_USABLE,
        alpha=0.16,
        lw=0,
        zorder=3,
        label="25th to 75th percentile",
    )
    ax.plot(
        t_draw,
        median_disp[draw_summary],
        color=figstyle.COLOR_USABLE,
        lw=2.2,
        solid_capstyle="round",
        zorder=4,
        label="median across cycles",
    )

    # No separate nadir marker is drawn. The cohort-median nadir depth is a median of
    # per-cycle nadir depths, whereas the median curve is a pointwise median across
    # cycles at each instant; because cycles reach their nadir at different times, the
    # two summaries are not the same value and should not be superimposed. The pointwise
    # median curve carries the ensemble shape; the cohort-median nadir depth is logged
    # (and belongs in the caption), not stamped on the canvas.

    # --- Axes furniture -------------------------------------------------------
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("time from occlusion start (s)")
    ax.set_ylabel("perfusion index (baseline = 1.0)")
    ax.set_xticks([-10, 0, 10, 20, 30, 40, 50, 60])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    # The density underlay replaces the default horizontal grid.
    ax.grid(False)
    ax.set_axisbelow(True)

    # Neutral phase labels inside the washes (no internal jargon, no title).
    y_lab = Y_MAX - 0.055
    ax.text(
        occ_end / 2.0,
        y_lab,
        "occlusion",
        ha="center",
        va="top",
        fontsize=8.0,
        color=figstyle.GRAPHITE,
        zorder=6,
    )
    ax.text(
        (occ_end + rep_end) / 2.0,
        y_lab,
        "reperfusion",
        ha="center",
        va="top",
        fontsize=8.0,
        color=figstyle.GRAPHITE,
        zorder=6,
    )

    # Cohort-size tag (counts, not a claim).
    ax.text(
        0.985,
        0.045,
        f"{n_cycles} cycles, {n_subjects} of 19 patients",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=figstyle.GRAPHITE,
        zorder=6,
    )

    # Honest reconstruction note on-canvas (small, neutral).
    ax.text(
        0.015,
        0.045,
        "reconstructed from per-cycle landmarks",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        style="italic",
        color=figstyle.SLATE,
        zorder=6,
    )

    # Legend for the overlaid summary curves.
    leg = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.10),
        fontsize=7.6,
        handlelength=1.6,
        borderpad=0.45,
        labelspacing=0.35,
    )
    leg.get_frame().set_linewidth(0.6)

    # Slim colorbar for the density underlay.
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, aspect=24)
    cbar.set_label("relative density of cycles", fontsize=7.8, color=figstyle.GRAPHITE)
    cbar.outline.set_linewidth(0.6)  # pyright: ignore[reportCallIssue]
    cbar.outline.set_edgecolor(figstyle.MIST)  # pyright: ignore[reportCallIssue]
    cbar.set_ticks([0.0, vmax])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=7.2, length=0)

    png, pdf = figstyle.save(fig, out_dir, SLUG)
    plt.close(fig)

    # --- Provenance to the log ------------------------------------------------
    logger.info("GLOBAL_SEED={}", GLOBAL_SEED)
    logger.info("signature cycles plotted: {} (subjects: {} of 19)", n_cycles, n_subjects)
    logger.info(
        "cohort-median landmarks: nadir_depth={:.3f} (level {:.3f}), descent={:.1f}s, "
        "recovery={:.1f}s",
        med_nadir_depth,
        med_nadir_level,
        med_descent,
        med_recovery,
    )
    logger.info("wrote {} and {}", png, pdf)
    return png, pdf


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Render the across-cycle morphology density figure: 268 reconstructed "
            "signature trajectories stacked into a 2D density with a median curve and "
            "interquartile band."
        )
    )
    p.add_argument("--events_dir", type=Path, default=DEFAULT_EVENTS_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_argparser().parse_args(argv)
    build_figure(events_dir=args.events_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
