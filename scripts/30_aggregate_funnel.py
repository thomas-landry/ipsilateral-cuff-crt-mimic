"""Aggregate per-record cuff-event tables into the feasibility funnel (step 30).

Reads every ``events_*.parquet`` produced by step 20, concatenates them, and
computes the feasibility funnel plus a per-patient summary via
:func:`cuffcrt.analysis.funnel.aggregate_funnel`. The occlusion-signature yield
is reported at two reperfusion-envelope thresholds: the primary 15 s rule and a
10 s sensitivity stratum, both derived from ``phase3_duration_s``.

All paths come from CLI arguments; no paths are hardcoded. Outputs are written
to the ``--out`` directory and the script never overwrites its inputs.

Outputs
-------
- ``<out>/funnel.csv`` : stage-by-stage counts.
- ``<out>/per_patient_summary.csv`` : per-subject QC and yield counts.

Examples
--------
::

    uv run python scripts/30_aggregate_funnel.py \\
        --events-dir data/interim/events --out results/feasibility

Open demo data::

    uv run python scripts/30_aggregate_funnel.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

from cuffcrt.analysis.funnel import (
    PRIMARY_PHASE3_S,
    SENSITIVITY_PHASE3_S,
    FunnelResult,
    aggregate_funnel,
)


def load_inventory(events_dir: Path) -> pl.DataFrame:
    """Concatenate every ``events_*.parquet`` under ``events_dir``.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record event parquets from step 20.

    Returns
    -------
    polars.DataFrame
        The concatenated inventory, sorted by ``(subject_id, nbp_timestamp_s)``.

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
    inventory = pl.concat(frames, how="diagonal_relaxed")
    # Normalize an empty-string reject_reason (a CSV round-trip artifact) to a
    # true null so it is treated as a full occlusion-signature event downstream.
    if "reject_reason" in inventory.columns and inventory.schema["reject_reason"] == pl.Utf8:
        inventory = inventory.with_columns(
            pl.when(pl.col("reject_reason") == "")
            .then(None)
            .otherwise(pl.col("reject_reason"))
            .alias("reject_reason")
        )
    sort_cols = [c for c in ("subject_id", "nbp_timestamp_s") if c in inventory.columns]
    return inventory.sort(sort_cols) if sort_cols else inventory


def _atomic_write_csv(df: pl.DataFrame, output_path: Path) -> None:
    """Write ``df`` to CSV via a tempfile plus rename for atomicity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    df.write_csv(tmp)
    tmp.replace(output_path)


def _log_summary(result: FunnelResult) -> None:
    """Log the headline funnel numbers."""
    logger.info("candidate cuff cycles: {} ({} records)", result.n_candidates, result.n_records)
    p, s = result.primary, result.sensitivity
    logger.info(
        "primary ({:g} s): {} events / {} patients = {:.3f}% {}",
        p.phase3_min_s,
        p.n_events,
        p.n_patients,
        p.pct_of_candidates,
        p.subjects,
    )
    logger.info(
        "sensitivity ({:g} s): {} events / {} patients = {:.3f}% {}",
        s.phase3_min_s,
        s.n_events,
        s.n_patients,
        s.pct_of_candidates,
        s.subjects,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the demo events/output directory layout under data/.",
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=None,
        help="Directory of per-record event parquets (default derived from --demo).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the funnel and per-patient CSVs.",
    )
    parser.add_argument(
        "--primary-phase3-s",
        type=float,
        default=PRIMARY_PHASE3_S,
        help=f"Primary reperfusion-envelope threshold in seconds (default {PRIMARY_PHASE3_S:g}).",
    )
    parser.add_argument(
        "--sensitivity-phase3-s",
        type=float,
        default=SENSITIVITY_PHASE3_S,
        help=(
            "Sensitivity reperfusion-envelope threshold in seconds "
            f"(default {SENSITIVITY_PHASE3_S:g})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Aggregate the funnel and write the output CSVs.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on missing input or unsafe paths).
    """
    args = _parse_args(argv)

    events_dir = args.events_dir or (
        Path("data/interim/events_demo") if args.demo else Path("data/interim/events")
    )
    out_dir = args.out or (
        Path("results/feasibility_demo") if args.demo else Path("results/feasibility")
    )

    logger.info("events_dir={}", events_dir)
    logger.info("out_dir={}", out_dir)

    if out_dir.resolve() == events_dir.resolve():
        logger.error("--out must differ from --events-dir; refusing to overwrite inputs.")
        return 2

    try:
        inventory = load_inventory(events_dir)
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        return 2

    result = aggregate_funnel(
        inventory,
        primary_phase3_s=args.primary_phase3_s,
        sensitivity_phase3_s=args.sensitivity_phase3_s,
    )

    _atomic_write_csv(result.funnel, out_dir / "funnel.csv")
    _atomic_write_csv(result.per_patient, out_dir / "per_patient_summary.csv")
    logger.info("wrote {}", out_dir / "funnel.csv")
    logger.info("wrote {}", out_dir / "per_patient_summary.csv")

    _log_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
