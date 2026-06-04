"""Consolidate per-record event parquets into one inventory CSV (pipeline step 22).

Pipeline step 20 (:mod:`scripts.20_extract_cuff_events`) emits one parquet of
derived per-event fields per WDB record under ``data/interim/events/``.
Downstream steps (the funnel aggregator in step 30, the local-LLM harness in
step 41) expect a single consolidated table. This script concatenates the
per-record parquets into ``data/interim/event_inventory.csv`` with a
deterministic, stable row order, closing the reproducibility gap: the
consolidated inventory was previously assembled inline and was not regenerable
from a clean checkout.

Rows are sorted by ``(subject_id, record_id, nbp_timestamp_s)``, so the output
is byte-reproducible run to run. Every derived field, including the detector
``reject_reason`` vocabulary, passes through unchanged; this is a pure
consolidation, not a re-classification. The output (CSV) differs from the inputs
(parquets); the script additionally refuses to write into the events directory.

Examples
--------
Consolidate the credentialed-data inventory::

    uv run python scripts/22_consolidate_inventory.py \\
        --events-dir data/interim/events \\
        --out data/interim/event_inventory.csv

Open demo data (no credentialing)::

    uv run python scripts/22_consolidate_inventory.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

from cuffcrt._paths import DataNotAvailableError, require_path
from cuffcrt.analysis.inventory import consolidate_events, find_event_parquets

# Demo events live under the interim tree, written by step 20 in --demo mode.
DEMO_EVENTS_SUBPATH = Path("interim/demo/events")
DEMO_OUT_SUBPATH = Path("interim/demo/event_inventory.csv")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the open MIMIC-IV-Demo interim layout (no credentialing).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root of the data/ tree, used to resolve --demo paths (default: data).",
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("data/interim/events"),
        help="Directory of per-record event parquets (default: data/interim/events).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/event_inventory.csv"),
        help="Output consolidated inventory CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args(argv)


def _log_reject_summary(inventory: pl.DataFrame) -> None:
    """Log the consolidated row count and reject_reason distribution."""
    logger.info("consolidated {} rows", inventory.height)
    n_subjects = inventory.get_column("subject_id").n_unique()
    n_records = inventory.get_column("record_id").n_unique()
    logger.info("{} subjects, {} records", n_subjects, n_records)
    summary = (
        inventory.get_column("reject_reason")
        .value_counts(sort=True)
        .sort("count", descending=True)
    )
    for reason, count in summary.iter_rows():
        label = reason if reason is not None else "None (detector-positive)"
        logger.info("  reject_reason={}: {}", label, count)


def main(argv: list[str] | None = None) -> int:
    """Consolidate per-record event parquets into one inventory CSV.

    Returns
    -------
    int
        Process exit code (0 on success, 2 when data is unavailable or the
        output would overwrite an input or an existing file without ``--force``).
    """
    args = _parse_args(argv)

    events_dir = args.events_dir
    out_path = args.out
    if args.demo:
        events_dir = args.data_root / DEMO_EVENTS_SUBPATH
        out_path = args.data_root / DEMO_OUT_SUBPATH

    logger.info("demo={}", args.demo)
    logger.info("events_dir={}", events_dir)
    logger.info("out={}", out_path)

    try:
        require_path(events_dir, what="per-record event parquets directory")
    except DataNotAvailableError as exc:
        logger.error("{}", exc)
        return 2

    # Never write into the input directory; the inventory must not clobber a parquet.
    if out_path.resolve().parent == events_dir.resolve():
        logger.error(
            "--out must not live inside --events-dir; refusing to write the "
            "consolidated inventory next to its input parquets."
        )
        return 2

    if out_path.exists() and not args.force:
        logger.error(
            "{} already exists; pass --force to overwrite.", out_path
        )
        return 2

    try:
        parquets = find_event_parquets(events_dir)
    except FileNotFoundError as exc:
        logger.error("{}\nConfirm step 20 has run; see data/README.md.", exc)
        return 2

    logger.info("found {} per-record parquets", len(parquets))
    inventory = consolidate_events(parquets)
    _log_reject_summary(inventory)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    inventory.write_csv(tmp)
    tmp.replace(out_path)
    logger.info("wrote {}", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
