"""Link MIMIC-IV-WDB records to ICU stays via wall-clock overlap (pipeline step 10).

Reads the WDB record tree and the clinical ``icu/icustays.csv.gz`` table,
computes per-record by per-stay overlap windows via
:func:`cuffcrt.cohort.wdb_linkage.link_wdb_to_icustays`, and writes the result
to a parquet file. The script is idempotent and writes via a tempfile-then-
rename for atomicity; it never overwrites its inputs.

Examples
--------
Full (credentialed) data, explicit paths::

    uv run python scripts/10_link_wdb_to_icustay.py \\
        --wdb-root data/raw/mimic-iv-wdb/0.1.0/waves \\
        --icustays-csv data/raw/mimic-iv-clinical/3.1/icu/icustays.csv.gz \\
        --output-parquet data/interim/wdb_to_icustay.parquet

Open demo data (no credentialing)::

    uv run python scripts/10_link_wdb_to_icustay.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

from cuffcrt._paths import (
    DataNotAvailableError,
    require_path,
    resolve_icustays_csv,
    resolve_wdb_root,
)
from cuffcrt.cohort.wdb_linkage import link_wdb_to_icustays


def _read_icustays(icustays_csv: Path) -> pl.DataFrame:
    """Read the gzip-compressed icustays CSV into a polars DataFrame.

    Coerces ``intime`` / ``outtime`` to ``Datetime`` and the IDs to ``Int64``.
    """
    df = pl.read_csv(
        icustays_csv,
        try_parse_dates=True,
        schema_overrides={
            "subject_id": pl.Int64,
            "hadm_id": pl.Int64,
            "stay_id": pl.Int64,
        },
    )
    for col in ("intime", "outtime"):
        if col in df.columns and df.schema[col] != pl.Datetime:
            df = df.with_columns(
                pl.col(col).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            )
    return df


def _atomic_write_parquet(df: pl.DataFrame, output_path: Path) -> None:
    """Write ``df`` to ``output_path`` via a tempfile plus rename for atomicity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(output_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the open MIMIC-IV-Demo dataset layout (no credentialing).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root of the data/ tree (default: data).",
    )
    parser.add_argument(
        "--wdb-root",
        type=Path,
        default=None,
        help="Override the WDB record-tree root (otherwise derived from --data-root).",
    )
    parser.add_argument(
        "--icustays-csv",
        type=Path,
        default=None,
        help="Override the icustays.csv.gz path (otherwise derived from --data-root).",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("data/interim/wdb_to_icustay.parquet"),
        help="Output parquet path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the WDB-to-icustays linkage and write a parquet.

    Returns
    -------
    int
        Process exit code (0 on success, 2 when data is unavailable).
    """
    args = _parse_args(argv)

    wdb_root = args.wdb_root or resolve_wdb_root(args.data_root, demo=args.demo)
    icustays_csv = args.icustays_csv or resolve_icustays_csv(args.data_root, demo=args.demo)

    logger.info("demo={}", args.demo)
    logger.info("wdb_root={}", wdb_root)
    logger.info("icustays_csv={}", icustays_csv)
    logger.info("output_parquet={}", args.output_parquet)

    try:
        require_path(wdb_root, what="WDB record tree")
        require_path(icustays_csv, what="icustays CSV")
    except DataNotAvailableError as exc:
        logger.error("{}", exc)
        return 2

    logger.info("reading icustays...")
    icustays = _read_icustays(icustays_csv)
    logger.info("loaded {} icustays rows", len(icustays))

    logger.info("walking WDB tree and computing overlaps...")
    linkage = link_wdb_to_icustays(wdb_root, icustays)
    n_modal = (
        int(linkage.select(pl.col("is_modal").sum()).item()) if not linkage.is_empty() else 0
    )
    logger.info("linkage rows: {} (modal: {})", len(linkage), n_modal)

    _atomic_write_parquet(linkage, args.output_parquet)
    logger.info("wrote {} ({} rows)", args.output_parquet, len(linkage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
