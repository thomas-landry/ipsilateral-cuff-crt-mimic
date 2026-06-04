"""Consolidate per-record cuff-event parquets into one inventory frame.

Pipeline step 20 (:mod:`scripts.20_extract_cuff_events`) emits one parquet of
derived per-event fields per WDB record under ``data/interim/events/``. Several
downstream steps (the funnel aggregator, the local-LLM harness) expect a single
consolidated table. This module concatenates the per-record parquets into one
DataFrame with a deterministic, stable row order so the resulting inventory is
byte-reproducible from a clean checkout.

The logic is separated from the CLI script (:mod:`scripts.22_consolidate_inventory`)
so it can be unit-tested without data on disk.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Glob for the per-record event parquets emitted by pipeline step 20.
EVENT_PARQUET_GLOB = "events_*.parquet"

# Stable sort keys. The detector emits rows in ascending NBP timestamp within a
# record, and subject_id sorts identically to the parquet filenames, so sorting
# on these three keys reproduces the historical concatenation order while being
# robust to any filesystem glob-order differences on a clean checkout.
SORT_KEYS = ("subject_id", "record_id", "nbp_timestamp_s")


def find_event_parquets(events_dir: Path) -> list[Path]:
    """Return the per-record event parquets in a directory, filename-sorted.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory holding the per-record ``events_*.parquet`` files emitted by
        pipeline step 20.

    Returns
    -------
    list[pathlib.Path]
        Matching parquet paths, sorted by filename for a stable iteration order.

    Raises
    ------
    FileNotFoundError
        If ``events_dir`` does not exist or contains no matching parquet.
    """
    if not events_dir.is_dir():
        raise FileNotFoundError(f"events directory not found at {events_dir}")
    parquets = sorted(events_dir.glob(EVENT_PARQUET_GLOB))
    if not parquets:
        raise FileNotFoundError(
            f"no {EVENT_PARQUET_GLOB} files found in {events_dir}"
        )
    return parquets


def consolidate_events(parquets: list[Path]) -> pl.DataFrame:
    """Concatenate per-record event parquets into one deterministically ordered frame.

    Reject reasons and every other derived field pass through unchanged; this is
    a pure consolidation, not a re-classification. Rows are sorted by
    :data:`SORT_KEYS` so the output is byte-reproducible run to run regardless of
    the order in which the parquets are supplied.

    Parameters
    ----------
    parquets : list[pathlib.Path]
        Per-record event parquets to combine. Must be non-empty and share the
        step-20 schema.

    Returns
    -------
    polars.DataFrame
        The combined inventory, one row per charted NBP cycle, sorted by
        ``(subject_id, record_id, nbp_timestamp_s)``.

    Raises
    ------
    ValueError
        If ``parquets`` is empty.
    """
    if not parquets:
        raise ValueError("no parquet files supplied to consolidate")
    frames = [pl.read_parquet(p) for p in parquets]
    combined = pl.concat(frames, how="vertical")
    return combined.sort(SORT_KEYS, maintain_order=True)
