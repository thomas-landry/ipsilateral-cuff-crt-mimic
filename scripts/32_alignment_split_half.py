"""Subject-clustered split-half alignment-window calibration (step 32).

Implements the pre-registered subject-split alignment-window check committed
in ``findings/preregistration_detector.md``. Randomly partitions the subjects
into a calibration half and a held-out half (with the project's global seed),
measures the nadir-versus-charted-BP offset distribution on the calibration
half restricted to qualifying runs, derives two transparent data-driven
windows (R-95 = [p2.5, p97.5] and R-90 = [p5, p95]), and then applies all
three windows (R-95, R-90, pre-registered [-50, +30] s) to the held-out half.
For each window the held-out primary event count, patient count, primary rate
per evaluable QC-pass cycle, and a subject-clustered percentile bootstrap 95%
CI are reported.

"Qualifying run" follows the pre-registration: a sub-occlusion run with
length at least :data:`SENSITIVITY_MIN_S` seconds whose smoothed nadir is
below :data:`NADIR_DEPTH` of baseline. In the canonical events parquets
each row stores the deepest such run found in the search window; rows whose
``alignment_offset_s`` is finite are the qualifying rows visible to this
calibration. The detector's stored alignment window is the pre-registered
[-50, +30] s, so this script's calibration is a within-window analysis: the
recommended narrower windows are derived inside that outer envelope. This
is documented in the script's log output and stamped onto every artifact.

Inputs
------
- ``--events-dir`` : directory of per-record event parquets from step 20
  (default ``data/interim/events``).

Outputs
-------
- ``<out>/split_half_assignments.csv`` : one row per subject with the split.
- ``<out>/alignment_split_half_results.csv`` : one row per window with the
  calibration summary statistics and the held-out yield with its bootstrap
  CI.

Examples
--------
::

    uv run python scripts/32_alignment_split_half.py
    uv run python scripts/32_alignment_split_half.py \\
        --events-dir data/interim/events \\
        --out results/alignment_split_half
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.analysis import cluster_bootstrap_ci
from cuffcrt.signal.cuff_event_detector import (
    ALIGN_HI_S,
    ALIGN_LO_S,
    NADIR_DEPTH,
    PRIMARY_MIN_S,
    SENSITIVITY_MIN_S,
)

# Pre-registered alignment window (carried forward for the three-window comparison).
PREREG_ALIGN_LO_S = ALIGN_LO_S  # -50
PREREG_ALIGN_HI_S = ALIGN_HI_S  # +30

# Calibration percentile rules.
R95_LOWER_PCT = 2.5
R95_UPPER_PCT = 97.5
R90_LOWER_PCT = 5.0
R90_UPPER_PCT = 95.0

# Percentiles reported on the calibration offset distribution.
REPORTED_PCTS = (2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 97.5)


@dataclass(frozen=True)
class Window:
    """One named alignment window, in seconds (nadir minus charted BP)."""

    name: str
    lower_s: float
    upper_s: float


@dataclass(frozen=True)
class WindowResult:
    """Held-out yield under one alignment window."""

    window: Window
    n_events_primary: int
    n_patients_primary: int
    n_evaluable_qc_pass: int
    primary_rate: float
    ci_low: float
    ci_high: float


# ---------------------------------------------------------------------------
# I/O.
# ---------------------------------------------------------------------------


def load_inventory(events_dir: Path) -> pl.DataFrame:
    """Concatenate every ``events_*.parquet`` under ``events_dir``.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record event parquets from step 20.

    Returns
    -------
    polars.DataFrame
        The concatenated inventory.

    Raises
    ------
    FileNotFoundError
        If no parquet files are found.
    """
    parquets = sorted(events_dir.glob("events_*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"no events_*.parquet found under {events_dir}; run step 20 first."
        )
    logger.info("found {} per-record parquets", len(parquets))
    frames = [pl.read_parquet(p) for p in parquets]
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        raise FileNotFoundError(f"all parquets under {events_dir} were empty.")
    inv = pl.concat(frames, how="diagonal_relaxed")
    # CSV-roundtripped empty strings can sneak in as reject_reason; normalize.
    if "reject_reason" in inv.columns and inv.schema["reject_reason"] == pl.Utf8:
        inv = inv.with_columns(
            pl.when(pl.col("reject_reason") == "")
            .then(None)
            .otherwise(pl.col("reject_reason"))
            .alias("reject_reason")
        )
    return inv


# ---------------------------------------------------------------------------
# Helpers (pure; covered by unit tests on small fixtures).
# ---------------------------------------------------------------------------


def split_subjects(
    subject_ids: list[str],
    *,
    seed: int = GLOBAL_SEED,
) -> tuple[list[str], list[str]]:
    """Randomly partition unique subjects into calibration and held-out halves.

    The list is sorted before shuffling so the partition is reproducible for a
    given set of subjects regardless of input order. When the count is odd the
    extra subject is placed in the held-out half. At least one subject is
    required in each half.

    Parameters
    ----------
    subject_ids : list of str
        Distinct subject identifiers. Duplicates are removed.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Defaults to
        :data:`cuffcrt._seed.GLOBAL_SEED`.

    Returns
    -------
    calibration, held_out : list of str
        Sorted subject ids in each half.

    Raises
    ------
    ValueError
        If fewer than two distinct subjects are available.
    """
    unique = sorted(set(subject_ids))
    if len(unique) < 2:
        raise ValueError(
            f"need at least 2 distinct subjects to split; got {len(unique)}."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    half = len(unique) // 2  # extra goes to held_out on odd counts
    calibration = sorted(unique[i] for i in order[:half])
    held_out = sorted(unique[i] for i in order[half:])
    return calibration, held_out


def qualifying_offsets(
    inventory: pl.DataFrame,
    *,
    sensitivity_min_s: float = SENSITIVITY_MIN_S,
    nadir_depth: float = NADIR_DEPTH,
) -> np.ndarray:
    """Offsets ``t_nadir - t_nbp`` (s) for qualifying runs in ``inventory``.

    A qualifying run has run length at least ``sensitivity_min_s`` seconds,
    smoothed nadir below ``nadir_depth`` of baseline, and a finite alignment
    offset (which it has if and only if a deepest-nadir run was selected
    inside the detector's stored alignment window).
    """
    required = {"phase3_duration_s", "nadir_depth_frac", "alignment_offset_s"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"inventory missing required columns: {sorted(missing)}")
    df = inventory.filter(
        pl.col("phase3_duration_s") >= sensitivity_min_s,
        pl.col("nadir_depth_frac") < nadir_depth,
        pl.col("alignment_offset_s").is_finite(),
    )
    return df.get_column("alignment_offset_s").to_numpy()


def derive_windows(offsets: np.ndarray) -> tuple[Window, Window, Window]:
    """Build the three windows: R-95, R-90, and the pre-registered baseline.

    Returns the windows in the canonical reporting order.
    """
    if offsets.size == 0:
        raise ValueError("no qualifying offsets in the calibration half.")
    r95 = Window(
        name="R-95",
        lower_s=float(np.percentile(offsets, R95_LOWER_PCT)),
        upper_s=float(np.percentile(offsets, R95_UPPER_PCT)),
    )
    r90 = Window(
        name="R-90",
        lower_s=float(np.percentile(offsets, R90_LOWER_PCT)),
        upper_s=float(np.percentile(offsets, R90_UPPER_PCT)),
    )
    prereg = Window(name="prereg", lower_s=PREREG_ALIGN_LO_S, upper_s=PREREG_ALIGN_HI_S)
    return r95, r90, prereg


def calibration_summary(offsets: np.ndarray) -> dict[str, float]:
    """Median, IQR, and reported percentiles of the calibration offsets."""
    if offsets.size == 0:
        raise ValueError("no qualifying offsets to summarize.")
    summary: dict[str, float] = {
        "n_qualifying": int(offsets.size),
        "median_s": float(np.median(offsets)),
        "iqr_s": float(np.percentile(offsets, 75) - np.percentile(offsets, 25)),
        "q25_s": float(np.percentile(offsets, 25)),
        "q75_s": float(np.percentile(offsets, 75)),
    }
    for pct in REPORTED_PCTS:
        summary[f"p{pct:g}_s"] = float(np.percentile(offsets, pct))
    return summary


def primary_indicator_under_window(
    inventory: pl.DataFrame,
    window: Window,
    *,
    primary_min_s: float = PRIMARY_MIN_S,
    nadir_depth: float = NADIR_DEPTH,
) -> pl.DataFrame:
    """Per-row primary indicator under ``window``, restricted to evaluable QC-pass rows.

    Evaluable QC-pass means the row was not excluded for missing or bad PPG
    and the pre-window quality criterion passed (``pre_window_valid``). An
    event is primary under ``window`` when, in addition to the standard
    primary criteria (recovered + deep + run >= primary_min_s), the stored
    ``alignment_offset_s`` falls inside ``[window.lower_s, window.upper_s]``.
    Recovery is encoded by ``reject_reason is null`` in the canonical events
    parquets; that is, a primary event under the pre-registered window has
    ``reject_reason`` null and ``recovered`` true.

    Returns
    -------
    polars.DataFrame
        Columns ``subject_id`` and ``is_primary`` (0/1). One row per
        evaluable QC-pass cycle.
    """
    required = {
        "subject_id",
        "phase3_duration_s",
        "nadir_depth_frac",
        "alignment_offset_s",
        "pre_window_valid",
        "reject_reason",
        "recovered",
    }
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"inventory missing required columns: {sorted(missing)}")

    evaluable = inventory.filter(pl.col("pre_window_valid"))
    indicator = (
        pl.col("phase3_duration_s").is_finite()
        & (pl.col("phase3_duration_s") >= primary_min_s)
        & (pl.col("nadir_depth_frac") < nadir_depth)
        & pl.col("alignment_offset_s").is_finite()
        & (pl.col("alignment_offset_s") >= window.lower_s)
        & (pl.col("alignment_offset_s") <= window.upper_s)
        & pl.col("recovered")
    ).cast(pl.Int64)
    return evaluable.select("subject_id", indicator.alias("is_primary"))


def held_out_window_result(
    inventory: pl.DataFrame,
    window: Window,
    *,
    primary_min_s: float = PRIMARY_MIN_S,
    nadir_depth: float = NADIR_DEPTH,
    n_resamples: int = 10_000,
    seed: int = GLOBAL_SEED,
) -> WindowResult:
    """Compute the held-out yield and a clustered bootstrap CI for one window."""
    per_cycle = primary_indicator_under_window(
        inventory, window, primary_min_s=primary_min_s, nadir_depth=nadir_depth
    )
    n_evaluable = per_cycle.height
    if n_evaluable == 0:
        return WindowResult(
            window=window,
            n_events_primary=0,
            n_patients_primary=0,
            n_evaluable_qc_pass=0,
            primary_rate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
        )
    values = per_cycle.get_column("is_primary").to_numpy().astype(np.float64)
    clusters = per_cycle.get_column("subject_id").to_numpy()
    n_events = int(values.sum())
    positive_subjects = (
        per_cycle.filter(pl.col("is_primary") == 1).get_column("subject_id").n_unique()
    )
    boot = cluster_bootstrap_ci(
        values=values,
        clusters=clusters,
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=0.95,
        seed=seed,
    )
    return WindowResult(
        window=window,
        n_events_primary=n_events,
        n_patients_primary=int(positive_subjects),
        n_evaluable_qc_pass=n_evaluable,
        primary_rate=float(boot.point),
        ci_low=float(boot.ci_low),
        ci_high=float(boot.ci_high),
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _atomic_write_csv(df: pl.DataFrame, output_path: Path) -> None:
    """Write ``df`` to CSV via a tempfile plus rename for atomicity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    df.write_csv(tmp)
    tmp.replace(output_path)


def _assignments_table(calibration: list[str], held_out: list[str]) -> pl.DataFrame:
    rows = [{"subject_id": s, "split": "calibration"} for s in calibration]
    rows += [{"subject_id": s, "split": "held_out"} for s in held_out]
    return pl.DataFrame(rows).sort("subject_id")


def _results_table(
    cal_summary: dict[str, float],
    results: list[WindowResult],
) -> pl.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "window_name": r.window.name,
                "lower_s": r.window.lower_s,
                "upper_s": r.window.upper_s,
                "calibration_median_s": cal_summary["median_s"],
                "calibration_iqr_s": cal_summary["iqr_s"],
                "n_events_primary_heldout": r.n_events_primary,
                "n_patients_primary_heldout": r.n_patients_primary,
                "n_evaluable_qc_pass_heldout": r.n_evaluable_qc_pass,
                "primary_rate_per_evaluable_heldout": r.primary_rate,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
            }
        )
    return pl.DataFrame(rows)


def _interpretation(
    cal_summary: dict[str, float],
    results: list[WindowResult],
) -> str:
    """Plain-English interpretation paragraph, one sentence per fact."""
    by_name = {r.window.name: r for r in results}
    prereg = by_name["prereg"]
    r95 = by_name["R-95"]
    r90 = by_name["R-90"]
    median = cal_summary["median_s"]
    q25 = cal_summary["q25_s"]
    q75 = cal_summary["q75_s"]
    prereg_contains_iqr = (
        prereg.window.lower_s <= q25 and prereg.window.upper_s >= q75
    )
    prereg_contains_median = (
        prereg.window.lower_s <= median <= prereg.window.upper_s
    )

    def _inside(rate: float, other: WindowResult) -> str:
        if not np.isfinite(rate) or not np.isfinite(other.ci_low):
            return "undetermined (insufficient data)"
        inside = other.ci_low <= rate <= other.ci_high
        return "inside" if inside else "outside"

    inside_r95 = _inside(prereg.primary_rate, r95)
    inside_r90 = _inside(prereg.primary_rate, r90)

    paragraph = (
        "On the calibration half, the qualifying-run nadir-versus-charted-BP "
        f"offset has median {median:.1f} s and IQR [{q25:.1f}, {q75:.1f}] s; "
        f"the pre-registered window [{prereg.window.lower_s:.0f}, "
        f"{prereg.window.upper_s:.0f}] s "
        f"{'contains' if prereg_contains_median else 'does not contain'} the "
        f"median and {'contains' if prereg_contains_iqr else 'does not contain'} "
        "the IQR. Data-driven windows from the calibration half are R-95 = "
        f"[{r95.window.lower_s:.1f}, {r95.window.upper_s:.1f}] s and R-90 = "
        f"[{r90.window.lower_s:.1f}, {r90.window.upper_s:.1f}] s. On the held-out "
        f"half the pre-registered window yields {prereg.n_events_primary} primary "
        f"events / {prereg.n_patients_primary} patients = "
        f"{100 * prereg.primary_rate:.2f}% of "
        f"{prereg.n_evaluable_qc_pass} evaluable QC-pass cycles "
        f"(95% CI {100 * prereg.ci_low:.2f}-{100 * prereg.ci_high:.2f}%); "
        f"R-95 yields {r95.n_events_primary} / {r95.n_patients_primary} "
        f"({100 * r95.primary_rate:.2f}%, CI {100 * r95.ci_low:.2f}-"
        f"{100 * r95.ci_high:.2f}%); R-90 yields {r90.n_events_primary} / "
        f"{r90.n_patients_primary} ({100 * r90.primary_rate:.2f}%, CI "
        f"{100 * r90.ci_low:.2f}-{100 * r90.ci_high:.2f}%). The pre-registered "
        f"held-out yield falls {inside_r95} the R-95 bootstrap CI and {inside_r90} "
        "the R-90 bootstrap CI, indicating whether tightening the alignment "
        "criterion on calibration data materially changes the held-out yield."
    )
    return paragraph


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("data/interim/events"),
        help="Directory of per-record event parquets (default data/interim/events).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/alignment_split_half"),
        help="Output directory (default results/alignment_split_half).",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=10_000,
        help="Bootstrap resamples per window (default 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"Seed for the split and the bootstrap (default {GLOBAL_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the subject-clustered split-half alignment-window calibration.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on missing input).
    """
    args = _parse_args(argv)
    logger.info("events_dir={}", args.events_dir)
    logger.info("out={}", args.out)
    logger.info("seed={}", args.seed)

    if args.out.resolve() == args.events_dir.resolve():
        logger.error("--out must differ from --events-dir; refusing to overwrite inputs.")
        return 2

    try:
        inventory = load_inventory(args.events_dir)
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        return 2

    subject_ids = sorted(inventory.get_column("subject_id").unique().to_list())
    logger.info("loaded {} subjects, {} rows", len(subject_ids), inventory.height)

    calibration, held_out = split_subjects(subject_ids, seed=args.seed)
    logger.info(
        "split: {} calibration, {} held_out", len(calibration), len(held_out)
    )
    logger.info("calibration subjects: {}", calibration)
    logger.info("held_out subjects: {}", held_out)

    cal_inv = inventory.filter(pl.col("subject_id").is_in(calibration))
    held_inv = inventory.filter(pl.col("subject_id").is_in(held_out))

    offsets = qualifying_offsets(cal_inv)
    cal_summary = calibration_summary(offsets)
    logger.info(
        "calibration: {} qualifying offsets, median={:.2f}s, IQR=[{:.2f}, {:.2f}]s",
        cal_summary["n_qualifying"],
        cal_summary["median_s"],
        cal_summary["q25_s"],
        cal_summary["q75_s"],
    )
    pct_msg = ", ".join(
        f"p{p:g}={cal_summary[f'p{p:g}_s']:.2f}s" for p in REPORTED_PCTS
    )
    logger.info("calibration percentiles: {}", pct_msg)

    r95, r90, prereg = derive_windows(offsets)
    logger.info("R-95 = [{:.2f}, {:.2f}] s", r95.lower_s, r95.upper_s)
    logger.info("R-90 = [{:.2f}, {:.2f}] s", r90.lower_s, r90.upper_s)
    logger.info(
        "prereg = [{:.2f}, {:.2f}] s (locked in pre-registration)",
        prereg.lower_s,
        prereg.upper_s,
    )
    logger.warning(
        "calibration offsets are drawn from rows whose nadir already fell "
        "inside the detector's stored alignment window {} s; R-95 and R-90 "
        "are therefore narrower windows derived inside that outer envelope.",
        f"[{PREREG_ALIGN_LO_S:g}, {PREREG_ALIGN_HI_S:g}]",
    )

    results = [
        held_out_window_result(
            held_inv, w, n_resamples=args.n_resamples, seed=args.seed
        )
        for w in (r95, r90, prereg)
    ]
    for r in results:
        logger.info(
            "held-out {}: {} events / {} patients = {:.3f}% of {} evaluable "
            "QC-pass cycles (95% CI {:.3f}-{:.3f}%)",
            r.window.name,
            r.n_events_primary,
            r.n_patients_primary,
            100 * r.primary_rate,
            r.n_evaluable_qc_pass,
            100 * r.ci_low,
            100 * r.ci_high,
        )

    assignments = _assignments_table(calibration, held_out)
    results_table = _results_table(cal_summary, results)
    _atomic_write_csv(assignments, args.out / "split_half_assignments.csv")
    _atomic_write_csv(results_table, args.out / "alignment_split_half_results.csv")
    logger.info("wrote {}", args.out / "split_half_assignments.csv")
    logger.info("wrote {}", args.out / "alignment_split_half_results.csv")

    paragraph = _interpretation(cal_summary, results)
    logger.info("interpretation: {}", paragraph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
