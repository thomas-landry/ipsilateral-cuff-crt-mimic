"""Tests for cuffcrt.analysis.card_bridge.

These exercise the ``card_id -> row_id`` bridge against small in-memory CSV
fixtures written to a tmp path, so the test reproduces the canonical row_id
construction without touching the credentialed inventory.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cuffcrt.analysis import build_card_to_rowid


def _write_inventory(path: Path) -> None:
    """Write a tiny inventory whose row order fixes the canonical row_id idx.

    row_id is ``"{subject_id}_{record_id}_{idx}"`` where idx is the 0-based row
    index after ``with_row_index``, so the second row (idx 1) of subject p01 /
    record r1 has row_id ``p01_r1_1``.
    """
    pl.DataFrame(
        {
            "subject_id": ["p01", "p01", "p02"],
            "record_id": ["r1", "r1", "r9"],
            "nbp_timestamp_s": [100.0001, 200.0002, 300.0],
            "is_occlusion_signature": [True, False, True],
        }
    ).write_csv(path)


def _write_manifest(path: Path) -> None:
    """Write a manifest whose triples point at inventory rows idx 1 and 0."""
    pl.DataFrame(
        {
            "card_id": ["A-card2", "B-card1"],
            "stratum": ["detector_positive", "detector_negative"],
            "subject_id": ["p01", "p01"],
            "record_id": ["r1", "r1"],
            # 200.0002 rounds to 200.0 -> matches inventory idx 1 (row_id p01_r1_1).
            # 100.0001 rounds to 100.0 -> matches inventory idx 0 (row_id p01_r1_0).
            "t_nbp": [200.0002, 100.0001],
        }
    ).write_csv(path)


def test_build_card_to_rowid_reproduces_known_mapping(tmp_path: Path) -> None:
    inventory = tmp_path / "event_inventory.csv"
    manifest = tmp_path / "gallery_manifest.csv"
    _write_inventory(inventory)
    _write_manifest(manifest)

    bridge = build_card_to_rowid(inventory, manifest)

    assert bridge.columns == [
        "card_id",
        "stratum",
        "row_id",
        "subject_id",
        "record_id",
        "t_nbp",
    ]
    assert bridge.height == 2
    mapping = dict(
        zip(bridge["card_id"].to_list(), bridge["row_id"].to_list(), strict=True)
    )
    assert mapping["A-card2"] == "p01_r1_1"
    assert mapping["B-card1"] == "p01_r1_0"
    # Every card resolved 1:1 to a unique row_id.
    assert bridge["row_id"].null_count() == 0
    assert bridge["row_id"].n_unique() == 2


def test_build_card_to_rowid_card_absent_from_inventory(tmp_path: Path) -> None:
    inventory = tmp_path / "event_inventory.csv"
    manifest = tmp_path / "gallery_manifest.csv"
    _write_inventory(inventory)
    # A card whose triple has no inventory match (record r_missing) keeps its
    # row but carries a null row_id rather than being dropped or erroring.
    pl.DataFrame(
        {
            "card_id": ["A-good", "C-orphan"],
            "stratum": ["detector_positive", "detector_positive"],
            "subject_id": ["p01", "p01"],
            "record_id": ["r1", "r_missing"],
            "t_nbp": [100.0, 999.0],
        }
    ).write_csv(manifest)

    bridge = build_card_to_rowid(inventory, manifest)

    assert bridge.height == 2
    mapping = dict(
        zip(bridge["card_id"].to_list(), bridge["row_id"].to_list(), strict=True)
    )
    assert mapping["A-good"] == "p01_r1_0"
    assert mapping["C-orphan"] is None


def test_build_card_to_rowid_missing_files_raise(tmp_path: Path) -> None:
    manifest = tmp_path / "gallery_manifest.csv"
    _write_manifest(manifest)
    with pytest.raises(FileNotFoundError):
        build_card_to_rowid(tmp_path / "nope.csv", manifest)

    inventory = tmp_path / "event_inventory.csv"
    _write_inventory(inventory)
    with pytest.raises(FileNotFoundError):
        build_card_to_rowid(inventory, tmp_path / "nope.csv")
