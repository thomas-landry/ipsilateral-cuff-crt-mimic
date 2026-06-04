"""Tests for per-record event consolidation (:mod:`cuffcrt.analysis.inventory`)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cuffcrt.analysis.inventory import (
    SORT_KEYS,
    consolidate_events,
    find_event_parquets,
)


def _make_record(subject_id: str, record_id: str, rows: list[dict]) -> pl.DataFrame:
    """Build a minimal step-20-shaped frame for one record.

    Only the columns the consolidation reasons about are populated with varied
    values; the rest are filled so the frame round-trips through parquet cleanly.
    """
    return pl.DataFrame(
        [
            dict(
                subject_id=subject_id,
                record_id=record_id,
                nbp_timestamp_s=float(r["nbp_timestamp_s"]),
                is_occlusion_signature=bool(r.get("is_occlusion_signature", False)),
                reject_reason=r["reject_reason"],
            )
            for r in rows
        ],
        schema={
            "subject_id": pl.String,
            "record_id": pl.String,
            "nbp_timestamp_s": pl.Float64,
            "is_occlusion_signature": pl.Boolean,
            "reject_reason": pl.String,
        },
    )


@pytest.fixture
def two_record_events(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Write two synthetic per-record parquets to a tmp events dir.

    Returns the events dir and the parquet paths. Rows within each record are
    written out of timestamp order so the consolidation's sort is exercised.
    """
    events_dir = tmp_path / "events"
    events_dir.mkdir()

    rec_a = _make_record(
        "p00000002",
        "r2",
        [
            {"nbp_timestamp_s": 30.0, "reject_reason": "no_phase2"},
            {"nbp_timestamp_s": 10.0, "reject_reason": "no_pleth"},
            {"nbp_timestamp_s": 20.0, "reject_reason": None,
             "is_occlusion_signature": True},
        ],
    )
    rec_b = _make_record(
        "p00000001",
        "r1",
        [
            {"nbp_timestamp_s": 5.0, "reject_reason": "pleth_mostly_nan"},
            {"nbp_timestamp_s": 1.0, "reject_reason": "pre_window_unstable"},
        ],
    )
    path_a = events_dir / "events_p00000002.parquet"
    path_b = events_dir / "events_p00000001.parquet"
    rec_a.write_parquet(path_a)
    rec_b.write_parquet(path_b)
    return events_dir, [path_a, path_b]


def test_consolidate_combines_all_rows_and_preserves_reject_reasons(
    two_record_events: tuple[Path, list[Path]],
) -> None:
    """Row count equals the sum of inputs and reject reasons pass through verbatim."""
    _, parquets = two_record_events
    combined = consolidate_events(parquets)

    assert combined.height == 5

    # reject_reason distribution is preserved exactly, including the null
    # detector-positive convention (one clean positive row).
    counts = {
        row[0]: row[1]
        for row in combined.get_column("reject_reason")
        .value_counts(sort=True)
        .iter_rows()
    }
    assert counts == {
        "no_phase2": 1,
        "no_pleth": 1,
        "pleth_mostly_nan": 1,
        "pre_window_unstable": 1,
        None: 1,
    }
    # The single detector-positive row keeps a null reject_reason.
    positives = combined.filter(pl.col("is_occlusion_signature"))
    assert positives.height == 1
    assert positives.get_column("reject_reason").to_list() == [None]


def test_consolidate_is_deterministically_sorted_regardless_of_input_order(
    two_record_events: tuple[Path, list[Path]],
) -> None:
    """Output order is stable on the sort keys, independent of parquet input order."""
    _, parquets = two_record_events

    forward = consolidate_events(parquets)
    reverse = consolidate_events(list(reversed(parquets)))

    # Same bytes regardless of the order parquets were supplied in.
    assert forward.equals(reverse)

    # Rows are sorted by (subject_id, record_id, nbp_timestamp_s).
    expected = forward.sort(list(SORT_KEYS), maintain_order=True)
    assert forward.equals(expected)

    # Spot-check the explicit expected order across both records.
    got = forward.select(["subject_id", "record_id", "nbp_timestamp_s"]).rows()
    assert got == [
        ("p00000001", "r1", 1.0),
        ("p00000001", "r1", 5.0),
        ("p00000002", "r2", 10.0),
        ("p00000002", "r2", 20.0),
        ("p00000002", "r2", 30.0),
    ]


def test_find_event_parquets_filename_sorted(
    two_record_events: tuple[Path, list[Path]],
) -> None:
    """Discovery returns matching parquets sorted by filename."""
    events_dir, _ = two_record_events
    found = find_event_parquets(events_dir)
    assert [p.name for p in found] == [
        "events_p00000001.parquet",
        "events_p00000002.parquet",
    ]


def test_find_event_parquets_raises_on_empty_dir(tmp_path: Path) -> None:
    """An events dir with no matching parquet raises FileNotFoundError."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        find_event_parquets(empty)


def test_consolidate_empty_list_raises() -> None:
    """Consolidating an empty parquet list is a programming error."""
    with pytest.raises(ValueError, match="no parquet"):
        consolidate_events([])
