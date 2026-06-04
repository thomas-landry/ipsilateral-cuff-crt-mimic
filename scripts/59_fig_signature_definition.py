"""Four measured quantities that define the signature, shown against their cuts (step 59).

A transparent picture of the operational rule that selects the occlusion-reperfusion
signature. The signature is not a special artifact: it is a cut on a continuum of four
per-cycle perfusion-index quantities. This panel draws the full distribution of each
quantity for the cycles on which it is actually measured, overlays the signature
subset, and marks the pre-set decision threshold, so a reader can see how close the
near-miss cycles sit to each line. This follows the TRIPOD-AI and STARD spirit: report
the rule and its operating point against the real distribution.

What is plotted
---------------
Two sets of charted cuff cycles, both real and both read from disk.

- Background ("cycles with a measured deep dip", n = 588). These are the cycles whose
  perfusion-index trace produced a deep, aligned dip with a recovery, so that all four
  quantities below are genuinely measured rather than undefined. This set is the union
  of the signature cycles and the recovered near-miss cycles; it coincides with the
  10-second sensitivity tier in the analysis. The remaining quality-control-pass cycles
  (6,236 in total) never produced a qualifying deep dip, so these four quantities are
  undefined for them and they cannot honestly be drawn on these axes; 6,236 is
  annotated only as the upstream denominator.
- Foreground ("signature cycles", n = 268). Cycles meeting the full signature
  definition: nadir below 0.20 of baseline, reperfusion run at least 15 s, and recovery
  to at least 0.85 of baseline, with the nadir aligned to the charted blood-pressure
  timestamp. Seen in 15 of the 19 records.

The four quantities and their pre-set thresholds (detector defaults):

- Nadir depth (nadir perfusion index / baseline). Cut: below 0.20 (a deeper dip is a
  smaller fraction). This cut is the entry gate that defines the measured set, so every
  background cycle already sits below it; the line is drawn to make that gate explicit.
- Descent duration (occlusion onset to nadir), in seconds. Descriptive context; it
  carries no separate threshold, so no cut line is drawn.
- Reperfusion run length (the event-defining duration), in seconds. Primary cut: at
  least 15 s (yields the 268 signature cycles in combination with the other rules). A
  lighter reference line marks the 10-second sensitivity tier (yields 588). This is the
  discriminating quantity.
- Recovery fraction (peak post-nadir perfusion index / baseline). Cut: at least 0.85 of
  baseline.

Honesty notes
-------------
- Laterality is never asserted. The signature label is a morphology-based estimate;
  there is no ground-truth cuff laterality in the data.
- No outcome, biomarker, or causal claim is implied. This is a description of an
  operational selection rule against the real distribution.
- The background is restricted to cycles on which these quantities are actually
  measured (n = 588). Drawing the 6,236 quality-control-pass cycles on these axes would
  be misleading because the quantities are undefined for the cycles that never produced
  a qualifying dip. That denominator is annotated as context only.
- The 15-second line is the reperfusion-run threshold alone; the 268 foreground
  reflects the full conjunction of all rules, so slightly more cycles clear the
  15-second line in isolation than carry the complete signature.
- Every value is computed from the real per-cycle features on disk. Nothing is
  simulated or illustrative-only.

De-identification
-----------------
No subject pseudo-id, record id, or absolute clock timestamp is drawn on the canvas;
the panels show distributions and thresholds only.

Examples
--------
::

    uv run python scripts/59_fig_signature_definition.py

Outputs (>=400 dpi PNG plus vector PDF):

    figures/fig_signature_definition.png
    figures/fig_signature_definition.pdf
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cuffcrt import figstyle
from cuffcrt._seed import GLOBAL_SEED

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_DIR = DEFAULT_REPO / "data" / "interim" / "events"
DEFAULT_OUT = DEFAULT_REPO / "figures"
SLUG = "fig_signature_definition"

# Detector defaults (single source of truth: cuffcrt.signal.cuff_event_detector).
NADIR_DEPTH_CUT = 0.20      # nadir must fall below this fraction of baseline
PRIMARY_MIN_S = 15.0        # primary reperfusion-run threshold (-> 268 cycles)
SENSITIVITY_MIN_S = 10.0    # sensitivity reperfusion-run threshold (-> 588)
RECOVERY_CUT = 0.85         # recovery must reach this fraction of baseline

# Canonical anchors (kept exact, cross-checked against the data at runtime).
N_QC_PASS = 6236            # quality-control-pass cycles (upstream denominator)
N_MEASURED = 588            # cycles with a measured deep dip (= 10 s tier)
N_SIGNATURE = 268           # signature cycles (primary)

# House palette (Okabe-Ito derived, colorblind-safe).
COLOR_BG = figstyle.SLATE             # muted gray: the measured background
COLOR_FG = figstyle.COLOR_USABLE      # blue: the signature subset
COLOR_CUT = figstyle.GRAPHITE         # graphite: primary threshold line
COLOR_REF = figstyle.SLATE            # lighter dotted 10 s reference line


def load_sets(events_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the measured-background and signature-foreground cycle tables.

    The background is the set of cycles on which the four quantities are genuinely
    measured (a finite, non-missing ``nadir_depth_frac``); the foreground is the
    signature subset of that background.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record ``events_p*.parquet`` files.

    Returns
    -------
    tuple of polars.DataFrame
        ``(background, foreground)``.

    Raises
    ------
    FileNotFoundError
        If no per-record event parquet files are present.
    """
    files = sorted(events_dir.glob("events_p*.parquet"))
    if not files:
        raise FileNotFoundError(f"No event parquet files under {events_dir}")
    df = pl.concat([pl.read_parquet(f) for f in files])
    logger.info("Loaded {} charted cycles from {} records", df.height, len(files))

    # Measured = the four quantities are finite (the deep-dip detector ran and produced
    # values). Polars stores the absent cases as float NaN, not null.
    background = df.filter(pl.col("nadir_depth_frac").is_not_nan())
    foreground = background.filter(pl.col("is_occlusion_signature"))
    return background, foreground


def _verify(background: pl.DataFrame, foreground: pl.DataFrame) -> None:
    """Cross-check the loaded sets against the canonical anchors.

    Raises
    ------
    AssertionError
        If a count drifts from its canonical value.
    """
    assert background.height == N_MEASURED, (
        f"measured background = {background.height}, expected {N_MEASURED}"
    )
    assert foreground.height == N_SIGNATURE, (
        f"signature foreground = {foreground.height}, expected {N_SIGNATURE}"
    )
    # Sanity: every background cycle already sits below the nadir entry gate.
    below_gate = background.filter(pl.col("nadir_depth_frac") < NADIR_DEPTH_CUT).height
    assert below_gate == N_MEASURED, (
        f"{below_gate} of {N_MEASURED} background cycles below the nadir gate"
    )
    # Sanity: the 10 s tier is the whole measured set; 15 s is a strict subset.
    n10 = background.filter(pl.col("phase3_duration_s") >= SENSITIVITY_MIN_S).height
    n15 = background.filter(pl.col("phase3_duration_s") >= PRIMARY_MIN_S).height
    assert n10 == N_MEASURED, f"phase3 >= 10 s = {n10}, expected {N_MEASURED}"
    assert n15 >= N_SIGNATURE, f"phase3 >= 15 s = {n15}, below signature count"
    logger.info(
        "Verified: background={}, signature={}, phase3>=15s={} (>= signature)",
        background.height,
        foreground.height,
        n15,
    )


# Per-panel specification: column, axis label, x-range, bin count, threshold(s).
# ``cut`` is the primary dashed line; ``ref`` the lighter dotted reference line;
# ``cut_side`` records which side of the cut the signature lies on, for the note.
PANELS = (
    {
        "col": "nadir_depth_frac",
        "label": "Nadir depth  (perfusion index / baseline)",
        "xmax": 0.20,
        "xmin": 0.0,
        "bins": 28,
        "cut": NADIR_DEPTH_CUT,
        "cut_text": "entry gate 0.20",
        "ref": None,
        "note": "below 0.20\n(deeper dip)",
        "note_x": 0.04,
    },
    {
        "col": "phase2_duration_s",
        "label": "Descent duration  (s)",
        "xmax": 35.0,
        "xmin": 0.0,
        "bins": 30,
        "cut": None,
        "cut_text": None,
        "ref": None,
        "note": "descriptive\n(no threshold)",
        "note_x": 22.0,
    },
    {
        "col": "phase3_duration_s",
        "label": "Reperfusion run length  (s)",
        "xmax": 70.0,
        "xmin": 8.0,
        "bins": 31,
        "cut": PRIMARY_MIN_S,
        "cut_text": "15 s primary",
        "ref": SENSITIVITY_MIN_S,
        "ref_text": "10 s sensitivity",
        "note": None,
        "note_x": None,
    },
    {
        "col": "recovery_fraction_at_window_end",
        "label": "Recovery fraction  (peak / baseline)",
        "xmax": 3.5,
        "xmin": 0.0,
        "bins": 30,
        "cut": RECOVERY_CUT,
        "cut_text": "0.85 recovery",
        "ref": None,
        "note": "at or above 0.85",
        "note_x": 2.05,
    },
)


def _draw_panel(ax: plt.Axes, spec: dict, bg: np.ndarray, fg: np.ndarray) -> None:
    """Draw one overlaid-histogram panel with its threshold line(s).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    spec : dict
        Panel specification (column, label, range, bins, thresholds).
    bg, fg : numpy.ndarray
        Background (measured) and foreground (signature) values for this column.
    """
    edges = np.linspace(spec["xmin"], spec["xmax"], spec["bins"] + 1)

    # Background: filled muted histogram, the full measured distribution.
    ax.hist(
        np.clip(bg, spec["xmin"], spec["xmax"]),
        bins=edges,
        color=COLOR_BG,
        alpha=0.30,
        edgecolor=COLOR_BG,
        linewidth=0.0,
        zorder=1,
    )
    # Foreground: signature subset, saturated blue, drawn over the background.
    ax.hist(
        np.clip(fg, spec["xmin"], spec["xmax"]),
        bins=edges,
        color=COLOR_FG,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )

    has_ref = spec.get("ref") is not None

    # Primary threshold: dashed graphite line. When a nearby reference line shares the
    # panel (reperfusion), the primary label is nudged right so the two labels do not
    # collide; otherwise it is centered on the line.
    if spec["cut"] is not None:
        ax.axvline(
            spec["cut"],
            color=COLOR_CUT,
            linestyle=(0, (5, 2)),
            linewidth=1.4,
            zorder=3,
        )
        ax.text(
            spec["cut"],
            1.018,
            spec["cut_text"],
            transform=ax.get_xaxis_transform(),
            ha="left" if has_ref else "center",
            va="bottom",
            fontsize=7.2,
            color=COLOR_CUT,
            fontweight="bold",
        )
    # Lighter dotted 10 s reference (reperfusion panel only). Its label sits on a lower
    # line, left-aligned at the panel edge, so it clears both the primary label above
    # and the left spine.
    if has_ref:
        ax.axvline(
            spec["ref"],
            color=COLOR_REF,
            linestyle=(0, (1, 2)),
            linewidth=1.1,
            zorder=3,
        )
        ax.text(
            0.035,
            0.96,
            spec["ref_text"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=figstyle.GRAPHITE,
        )

    # Soft in-panel orientation note (kept neutral; no jargon).
    if spec.get("note"):
        ax.text(
            spec["note_x"],
            0.93,
            spec["note"],
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=6.8,
            color=figstyle.SLATE,
            style="italic",
            linespacing=1.15,
        )

    ax.set_xlabel(spec["label"])
    ax.set_xlim(spec["xmin"], spec["xmax"])
    ax.margins(y=0.0)
    # Tidy spines / ticks come from the shared style; keep y light.
    ax.tick_params(axis="y", length=2.5)


def build_figure(*, events_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Build and save the four-quantity definition panel.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record ``events_p*.parquet`` files.
    out_dir : pathlib.Path
        Output directory for the PDF and PNG.

    Returns
    -------
    tuple of pathlib.Path
        The written ``(png_path, pdf_path)``.
    """
    # Policy seed; the figure is deterministic (fixed bins, no sampling), so no
    # randomness actually enters, but the generator is pinned regardless.
    np.random.default_rng(GLOBAL_SEED)
    figstyle.apply_style()

    background, foreground = load_sets(events_dir)
    _verify(background, foreground)

    # Spot-check echo: median nadir depth of the signature subset, against source.
    sig_nadir_med = float(np.median(foreground["nadir_depth_frac"].to_numpy()))
    logger.info("Signature nadir-depth median = {:.3f} (source spot-check)", sig_nadir_med)

    fig, axes = plt.subplots(2, 2, figsize=(150 / 25.4, 138 / 25.4))
    panel_letters = ("A", "B", "C", "D")

    for ax, spec, letter in zip(axes.flat, PANELS, panel_letters, strict=True):
        bg = background[spec["col"]].to_numpy()
        fg = foreground[spec["col"]].to_numpy()
        bg = bg[np.isfinite(bg)]
        fg = fg[np.isfinite(fg)]
        _draw_panel(ax, spec, bg, fg)
        figstyle.panel_label(ax, letter)

    # Shared y-axis label on the left column.
    for ax in axes[:, 0]:
        ax.set_ylabel("Cuff cycles")

    # Figure-level legend (semantics only; no on-canvas title).
    legend_handles = [
        Patch(facecolor=COLOR_BG, alpha=0.30, edgecolor="none",
              label=f"Cycles with a measured deep dip  (n = {N_MEASURED:,})"),
        Patch(facecolor=COLOR_FG, alpha=0.78, edgecolor="white",
              label=f"Signature cycles  (n = {N_SIGNATURE:,})"),
        Line2D([0], [0], color=COLOR_CUT, linestyle=(0, (5, 2)), linewidth=1.4,
               label="Primary threshold"),
        Line2D([0], [0], color=COLOR_REF, linestyle=(0, (1, 2)), linewidth=1.1,
               label="10 s sensitivity reference"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.004),
        frameon=True,
        fontsize=7.6,
        handlelength=1.8,
        columnspacing=1.6,
    )

    # Quiet denominator note (the upstream quality-control-pass count, as context).
    fig.text(
        0.5,
        0.972,
        f"Drawn from {N_QC_PASS:,} quality-control-pass cuff cycles; the four "
        f"quantities are defined for the {N_MEASURED} that produced a deep dip.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=figstyle.GRAPHITE,
    )

    fig.subplots_adjust(
        left=0.10, right=0.955, top=0.905, bottom=0.165, hspace=0.70, wspace=0.27
    )

    png, pdf = figstyle.save(fig, out_dir, SLUG)
    plt.close(fig)
    logger.info("Wrote {} and {}", png, pdf)
    return png, pdf


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Render the signature-definition figure: four per-cycle perfusion-index "
            "quantities with the measured background, the signature subset, and the "
            "pre-set thresholds."
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
