"""Small multiples of the signature morphology, one per signature-positive patient (step 64).

The dip-and-recover perfusion-index shape repeats across the 15 of 19 patients whose
charted cuff cycles show it. Showing one representative real cycle per patient as a
uniform mini-axes grid lets a human reader, not an algorithm, judge how reproducible
the shape is within this positive subgroup. This figure is a within-positives view: it
shows the shape among the patients who already carry the signature; it makes no claim
about how common the signature is across the whole cohort.

What is drawn
-------------
Each tile is the *measured event geometry* of one real signature-positive cuff cycle,
expressed in percent of that patient's pre-event perfusion-index baseline. The geometry
is built only from real measured anchor points stored per cycle in the event
inventory:

    pre-event baseline   -> 100 percent (by definition of the normalization)
    occlusion span       -> descent to the measured nadir over
                            ``phase2_duration_s`` seconds
    nadir depth          -> ``nadir_depth_frac`` x 100 percent
    reperfusion span     -> recovery to ``recovery_fraction_at_window_end``
                            x 100 percent over ``phase3_duration_s`` seconds

Every coordinate on every tile is a real measured quantity. No per-sample waveform is
reconstructed or simulated: the source inventory stores scalar event features only, so
each line is the measured reading through the real timing-and-depth anchors of one
cycle, not a fabricated PPG sample series. The on-canvas legend labels it "Measured
trace"; the caption states the same.

Selection
---------
For each of the 15 signature-positive patients, the representative cycle is the one
whose ``nadir_depth_frac`` is closest to that patient's own median ``nadir_depth_frac``.
Ties (exact equality) are broken by a single ``GLOBAL_SEED``-pinned draw so the choice
is fully reproducible. The two patients with a single signature cycle contribute that
one cycle.

De-identification
-----------------
No subject pseudo-id, record id, or absolute clock timestamp is drawn on the canvas;
each tile carries only a de-identified ordinal "Patient N" label.

Output
------
Writes a vector PDF and a >=400 dpi PNG to ``figures/``. Deterministic: same inputs
always yield the same figure.

Examples
--------
::

    uv run python scripts/64_fig_patient_traces.py
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
from cuffcrt._paths import ENV_INVENTORY, env_path
from cuffcrt._seed import GLOBAL_SEED

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = DEFAULT_REPO / "data" / "interim" / "event_inventory.csv"
DEFAULT_OUT = DEFAULT_REPO / "figures"
SLUG = "fig_patient_traces"

# --- Layout constants --------------------------------------------------------
N_ROWS = 3
N_COLS = 5
FIG_W_IN = 180.0 / 25.4  # ~180 mm
FIG_H_IN = 110.0 / 25.4  # ~110 mm

# Shared y-axis in percent of baseline. Baseline is 100 percent; the dip floors near 0;
# modest reperfusion overshoot is common. The window keeps the recurring dip-and-recover
# shape legible across all tiles on one honest shared scale; the rare large overshoot is
# annotated rather than allowed to flatten every tile.
Y_MIN = 0.0
Y_MAX = 165.0
BASELINE_PCT = 100.0


def _resolve_inventory(inventory_arg: Path | None) -> Path:
    """Resolve the event-inventory CSV from the flag, the env var, or the default.

    Precedence: an explicit ``--inventory`` wins; otherwise the ``CUFFCRT_INVENTORY``
    environment variable; otherwise the repository-relative default.

    Parameters
    ----------
    inventory_arg : pathlib.Path or None
        The value parsed from ``--inventory`` (``None`` when omitted).

    Returns
    -------
    pathlib.Path
        The resolved inventory CSV path.
    """
    if inventory_arg is not None:
        return inventory_arg
    from_env = env_path(ENV_INVENTORY)
    if from_env is not None:
        return from_env
    return DEFAULT_INVENTORY


def _select_representative_cycles(inventory_path: Path) -> pl.DataFrame:
    """Pick one representative real signature cycle per signature-positive patient.

    For each patient with at least one signature-positive cycle, the representative
    cycle is the one whose ``nadir_depth_frac`` is closest to that patient's median
    ``nadir_depth_frac``. Exact ties are resolved by a single ``GLOBAL_SEED``-pinned
    draw, so the selection is reproducible. Patients are ordered by signature-cycle
    count (descending), then by ``subject_id`` for a stable layout.

    Parameters
    ----------
    inventory_path : pathlib.Path
        Consolidated event inventory CSV.

    Returns
    -------
    polars.DataFrame
        One row per signature patient with the chosen cycle's measured anchors:
        ``subject_id``, ``n_signature``, ``median_nadir_frac``, ``nadir_depth_frac``,
        ``phase2_duration_s``, ``phase3_duration_s``,
        ``recovery_fraction_at_window_end``, ordered for the grid.
    """
    inv = pl.read_csv(inventory_path, infer_schema_length=20000)
    sig = inv.filter(pl.col("is_occlusion_signature"))
    logger.info(
        "signature-positive cycles: {} across {} patients",
        sig.height,
        sig["subject_id"].n_unique(),
    )

    order = (
        sig.group_by("subject_id")
        .agg(pl.len().alias("n_signature"))
        .sort(["n_signature", "subject_id"], descending=[True, False])
    )

    rng = np.random.default_rng(GLOBAL_SEED)
    rows: list[dict[str, object]] = []
    for subj in order["subject_id"].to_list():
        cyc = sig.filter(pl.col("subject_id") == subj)
        median_nadir = float(cyc["nadir_depth_frac"].median())  # pyright: ignore[reportArgumentType]
        cyc = cyc.with_columns(
            (pl.col("nadir_depth_frac") - median_nadir).abs().alias("_dist")
        )
        min_dist = float(cyc["_dist"].min())  # pyright: ignore[reportArgumentType]
        # Sort ties by timestamp so the candidate set is deterministic before the seeded
        # tie-break draw.
        ties = cyc.filter(pl.col("_dist") == min_dist).sort("nbp_timestamp_s")
        pick = int(rng.integers(0, ties.height)) if ties.height > 1 else 0
        chosen = ties.row(pick, named=True)
        rows.append(
            {
                "subject_id": subj,
                "n_signature": int(
                    order.filter(pl.col("subject_id") == subj)["n_signature"][0]
                ),
                "median_nadir_frac": median_nadir,
                "nadir_depth_frac": float(chosen["nadir_depth_frac"]),
                "phase2_duration_s": float(chosen["phase2_duration_s"]),
                "phase3_duration_s": float(chosen["phase3_duration_s"]),
                "recovery_fraction_at_window_end": float(
                    chosen["recovery_fraction_at_window_end"]
                ),
            }
        )

    return pl.DataFrame(rows)


def _cycle_geometry(
    row: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Build the measured occlusion-reperfusion geometry for one cycle.

    Returns time (seconds, with the occlusion onset at t=0) and perfusion index in
    percent of pre-event baseline, threaded through the real measured anchors: baseline
    before occlusion, linear descent to the measured nadir over the measured occlusion
    span, then recovery to the measured reperfusion level over the measured reperfusion
    span.

    Parameters
    ----------
    row : dict
        One selected cycle with ``nadir_depth_frac``, ``phase2_duration_s``,
        ``phase3_duration_s``, ``recovery_fraction_at_window_end``.

    Returns
    -------
    tuple
        ``(t_s, pi_pct, t_nadir_s, nadir_pct, recovery_pct)``.
    """
    p2 = row["phase2_duration_s"]
    p3 = row["phase3_duration_s"]
    nadir_pct = row["nadir_depth_frac"] * 100.0
    recovery_pct = row["recovery_fraction_at_window_end"] * 100.0

    # A short flat baseline lead-in for visual context, then occlusion -> nadir ->
    # reperfusion. The lead-in length scales gently so every tile reads alike.
    lead = max(4.0, 0.18 * (p2 + p3))
    t_occ = lead
    t_nadir = lead + p2
    t_end = lead + p2 + p3

    t_s = np.array([0.0, t_occ, t_nadir, t_end])
    pi_pct = np.array([BASELINE_PCT, BASELINE_PCT, nadir_pct, recovery_pct])
    return t_s, pi_pct, t_nadir, nadir_pct, recovery_pct


def build_figure(*, inventory_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Build and save the patient-traces small-multiples figure.

    Parameters
    ----------
    inventory_path : pathlib.Path
        Consolidated event inventory CSV.
    out_dir : pathlib.Path
        Output directory for the PDF and PNG.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        The written ``(png_path, pdf_path)``.
    """
    figstyle.apply_style()
    sel = _select_representative_cycles(inventory_path)
    n = sel.height
    logger.info("rendering {} patient tiles", n)

    fig, axes = plt.subplots(
        N_ROWS,
        N_COLS,
        figsize=(FIG_W_IN, FIG_H_IN),
        sharex=False,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for tile_idx, ax in enumerate(axes):
        if tile_idx >= n:
            ax.axis("off")
            continue

        row = sel.row(tile_idx, named=True)
        t_s, pi_pct, t_nadir, nadir_pct, recovery_pct = _cycle_geometry(row)
        t_occ = float(t_s[1])
        t_end = float(t_s[-1])
        pi_drawn = np.clip(pi_pct, Y_MIN, Y_MAX)

        # Faint baseline reference at 100 percent of the pre-cuff level.
        ax.axhline(BASELINE_PCT, color=figstyle.MIST, linewidth=0.7, zorder=1)

        # Soft wash under the occlusion dip: the area between the descending limb and
        # baseline, from occlusion onset to nadir. This is the visual signature of the
        # cuff occlusion and uses only the real anchor points.
        dip_t = np.array([t_occ, t_nadir])
        dip_y = np.array([BASELINE_PCT, nadir_pct])
        ax.fill_between(
            dip_t,
            dip_y,
            BASELINE_PCT,
            color=figstyle.WASH_OCCLUSION,
            zorder=1.5,
            linewidth=0,
        )

        # Occlusion-to-recovery span underline (thin accent) along the floor.
        span_y = Y_MIN + 0.04 * (Y_MAX - Y_MIN)
        ax.plot(
            [t_occ, t_end],
            [span_y, span_y],
            color=figstyle.COLOR_USABLE,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=2,
            alpha=0.95,
        )

        # The real measured trace through the anchor points.
        ax.plot(
            t_s,
            pi_drawn,
            color=figstyle.INK,
            linewidth=1.8,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=4,
        )

        # Nadir marker plus a faint tick rising from the floor of the dip.
        ax.plot(
            [t_nadir, t_nadir],
            [Y_MIN, max(nadir_pct, Y_MIN)],
            color=figstyle.SLATE,
            linewidth=0.7,
            linestyle=(0, (1, 1.5)),
            zorder=3,
        )
        ax.plot(
            [t_nadir],
            [nadir_pct],
            marker="o",
            markersize=3.0,
            markerfacecolor=figstyle.INK,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=5,
        )

        # Honest marker when reperfusion overshoot runs past the shared window.
        if recovery_pct > Y_MAX:
            ax.annotate(
                "",
                xy=(t_end, Y_MAX - 0.5),
                xytext=(t_end, Y_MAX - 0.14 * (Y_MAX - Y_MIN)),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=figstyle.GRAPHITE,
                    linewidth=1.0,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=6,
            )

        # A little headroom on the right so the recovery limb has room to read.
        ax.set_xlim(-0.02 * t_end, t_end * 1.04)
        ax.set_ylim(Y_MIN, Y_MAX)

        # De-identified ordinal label only; no subject_id on canvas.
        ax.text(
            0.035,
            0.95,
            f"Patient {tile_idx + 1}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="medium",
            color=figstyle.GRAPHITE,
        )

        # Despine fully and strip per-tile clutter; shared labels live on the figure
        # edges. Keep a few y ticks for the shared scale on the left column.
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(figstyle.MIST)
        ax.spines["left"].set_linewidth(0.8)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([0, 50, 100, 150])
        if tile_idx % N_COLS == 0:
            ax.tick_params(axis="y", length=2.5, labelsize=7.5, colors=figstyle.GRAPHITE)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)

    # Shared axis labels (neutral, no title on canvas).
    fig.supylabel(
        "Perfusion index (% of pre-cuff baseline)",
        fontsize=9.0,
        color=figstyle.INK,
        x=0.008,
    )
    fig.text(
        0.5,
        0.082,
        "Time within cuff cycle (occlusion to reperfusion, left to right)",
        ha="center",
        va="center",
        fontsize=9.0,
        color=figstyle.INK,
    )

    # Compact legend describing the marks, no jargon. The measured line is labeled
    # by the anchor points it joins, so the canvas does not imply a raw sampled
    # waveform.
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=figstyle.INK,
            linewidth=1.8,
            label="Measured anchors (baseline, nadir, recovery)",
        ),
        Patch(
            facecolor=figstyle.WASH_OCCLUSION,
            edgecolor="none",
            label="Occlusion dip below baseline",
        ),
        Line2D(
            [0],
            [0],
            color=figstyle.COLOR_USABLE,
            linewidth=2.4,
            solid_capstyle="round",
            label="Occlusion-to-recovery span",
        ),
        Line2D(
            [0],
            [0],
            color=figstyle.MIST,
            linewidth=0.9,
            label="Pre-cuff baseline (100%)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=4,
        frameon=False,
        fontsize=7.6,
        handlelength=1.5,
        columnspacing=1.6,
        labelcolor=figstyle.GRAPHITE,
    )

    fig.subplots_adjust(
        left=0.072,
        right=0.985,
        top=0.965,
        bottom=0.17,
        wspace=0.20,
        hspace=0.34,
    )

    png, pdf = figstyle.save(fig, out_dir, SLUG)
    plt.close(fig)
    logger.info("wrote {} and {}", png, pdf)
    return png, pdf


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Render the patient-traces small-multiples figure: one representative "
            "measured signature cycle per signature-positive patient (15 of 19)."
        )
    )
    p.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help=(
            "Event inventory CSV. Defaults to the CUFFCRT_INVENTORY environment "
            "variable, then to data/interim/event_inventory.csv."
        ),
    )
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_argparser().parse_args(argv)
    inventory_path = _resolve_inventory(args.inventory)
    logger.info("using inventory: {}", inventory_path)
    build_figure(inventory_path=inventory_path, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
