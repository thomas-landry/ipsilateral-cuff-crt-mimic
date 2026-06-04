"""Tests for the reproducibility manifest builder."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cuffcrt.analysis.funnel import aggregate_funnel
from cuffcrt.analysis.manifest import (
    build_manifest,
    headline_from_result,
    sha256_file,
    write_manifest,
)


def _toy_inventory() -> pl.DataFrame:
    """A tiny inventory exercising no_pleth, QC-pass, and one primary event."""
    return pl.DataFrame(
        {
            "subject_id": ["pA", "pA", "pB", "pB"],
            "record_id": ["r1", "r1", "r2", "r2"],
            "phase3_duration_s": [float("nan"), 17.0, float("nan"), 5.0],
            "pre_window_valid": [False, True, False, True],
            "reject_reason": ["no_pleth", None, "no_pleth", "no_phase2"],
        }
    )


def test_sha256_file_stable(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    target.write_text("hello\n")
    first = sha256_file(target)
    assert first == sha256_file(target)
    assert len(first) == 64


def test_headline_counts_match_funnel() -> None:
    result = aggregate_funnel(_toy_inventory())
    headline = headline_from_result(result)
    assert headline.n_candidates == 4
    assert headline.excluded_no_pleth == 2
    assert headline.primary_events == 1
    assert headline.primary_patients == 1


def test_build_manifest_hashes_listed_files(tmp_path: Path) -> None:
    result = aggregate_funnel(_toy_inventory())
    funnel_csv = tmp_path / "funnel.csv"
    per_patient_csv = tmp_path / "per_patient_summary.csv"
    result.funnel.write_csv(funnel_csv)
    result.per_patient.write_csv(per_patient_csv)

    manifest = build_manifest(
        result, [funnel_csv, per_patient_csv], inventory_source="toy"
    )
    assert manifest["manifest_schema"] == "cuffcrt/feasibility-manifest/1"
    assert manifest["inventory_source"] == "toy"
    assert {f["name"] for f in manifest["files"]} == {
        "funnel.csv",
        "per_patient_summary.csv",
    }
    for entry in manifest["files"]:
        assert len(entry["sha256"]) == 64


def test_build_manifest_missing_file_raises(tmp_path: Path) -> None:
    result = aggregate_funnel(_toy_inventory())
    with pytest.raises(FileNotFoundError):
        build_manifest(result, [tmp_path / "does_not_exist.csv"])


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    import json

    result = aggregate_funnel(_toy_inventory())
    funnel_csv = tmp_path / "funnel.csv"
    result.funnel.write_csv(funnel_csv)
    manifest = build_manifest(result, [funnel_csv])
    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)
    loaded = json.loads(out.read_text())
    assert loaded["headline_counts"]["n_candidates"] == 4
