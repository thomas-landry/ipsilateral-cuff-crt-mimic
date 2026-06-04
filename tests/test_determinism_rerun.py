"""Tests for the MedGemma determinism re-run helper (script 43, D6).

These exercise the script offline: no oMLX server, no network. The headline-run
input is fabricated in-memory; the re-render path is monkeypatched to return a
canned row so the tests do not require WDB tree access. The focus is on the
parts the re-run helper actually owns: deterministic subsampling, paired-join
semantics, agreement metric correctness, the 3 x 3 confusion structure,
parse-failure rates, and CLI parsing.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

from cuffcrt.llm.medgemma import VALID_CALLS

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_rerun_module():
    """Import scripts/43_determinism_rerun.py as a module."""
    path = SCRIPTS_DIR / "43_determinism_rerun.py"
    spec = importlib.util.spec_from_file_location("determinism_rerun_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["determinism_rerun_cli"] = module
    spec.loader.exec_module(module)
    return module


def _first_run_df(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal first-run DataFrame with the columns we validate against."""
    base_cols = {"row_id": [], "subject_id": [], "record_id": [], "call": [], "parsed_ok": []}
    for row in rows:
        base_cols["row_id"].append(row["row_id"])
        base_cols["subject_id"].append(row.get("subject_id", row["row_id"].split("_")[0]))
        base_cols["record_id"].append(row.get("record_id", row["row_id"].split("_")[1]))
        base_cols["call"].append(row["call"])
        base_cols["parsed_ok"].append(row.get("parsed_ok", row["call"] is not None))
    return pl.DataFrame(base_cols)


def _rerun_df(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal re-run DataFrame with row_id and call."""
    return pl.DataFrame(
        {
            "row_id": [r["row_id"] for r in rows],
            "call": [r["call"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# Deterministic subsampling
# --------------------------------------------------------------------------- #


def test_subsample_is_reproducible_across_invocations():
    module = _load_rerun_module()
    rows = [{"row_id": f"p{i:04d}_r{i:04d}_0", "call": "indeterminate"} for i in range(300)]
    first_run = _first_run_df(rows)
    pick_a = module.subsample_row_ids(first_run, n_rows=100, seed=20260426)
    pick_b = module.subsample_row_ids(first_run, n_rows=100, seed=20260426)
    assert pick_a == pick_b
    assert len(pick_a) == 100
    # All picks are real row_ids from the input universe.
    assert set(pick_a).issubset({r["row_id"] for r in rows})


def test_subsample_changes_with_seed():
    module = _load_rerun_module()
    rows = [{"row_id": f"p{i:04d}_r{i:04d}_0", "call": "indeterminate"} for i in range(300)]
    first_run = _first_run_df(rows)
    pick_a = module.subsample_row_ids(first_run, n_rows=100, seed=20260426)
    pick_b = module.subsample_row_ids(first_run, n_rows=100, seed=999)
    assert pick_a != pick_b


def test_subsample_falls_back_to_universe_when_n_exceeds_input():
    module = _load_rerun_module()
    rows = [{"row_id": f"p{i:04d}_r{i:04d}_0", "call": "indeterminate"} for i in range(50)]
    first_run = _first_run_df(rows)
    pick = module.subsample_row_ids(first_run, n_rows=100, seed=20260426)
    assert len(pick) == 50
    assert pick == [r["row_id"] for r in rows]


def test_subsample_rejects_missing_row_id_column():
    module = _load_rerun_module()
    bad = pl.DataFrame({"foo": ["a", "b"]})
    with pytest.raises(ValueError, match="row_id"):
        module.subsample_row_ids(bad, n_rows=10, seed=20260426)


# --------------------------------------------------------------------------- #
# Paired-join semantics: card_id (row_id) mismatches must be visible
# --------------------------------------------------------------------------- #


def test_paired_join_reports_missing_row_ids_explicitly():
    module = _load_rerun_module()
    first = _first_run_df(
        [
            {"row_id": "p0001_r0001_0", "call": "occlusion_signature_present"},
            {"row_id": "p0002_r0002_0", "call": "no_occlusion_signature"},
            {"row_id": "p0003_r0003_0", "call": "indeterminate"},
        ]
    )
    rerun = _rerun_df(
        [
            {"row_id": "p0001_r0001_0", "call": "occlusion_signature_present"},
            # p0002 missing from re-run
            {"row_id": "p0003_r0003_0", "call": "indeterminate"},
            {"row_id": "p9999_r9999_0", "call": "indeterminate"},  # extra in re-run
        ]
    )
    agreement = module.compute_agreement(first, rerun)
    assert agreement["missing_from_rerun"] == ["p0002_r0002_0"]
    assert agreement["missing_from_first"] == ["p9999_r9999_0"]
    # The paired count covers only rows present in both runs.
    assert agreement["n_paired"] == 2
    # Both paired rows agree.
    assert agreement["n_agree"] == 2
    assert agreement["overall_agreement_pct"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Agreement metric correctness: 90 matching + 10 mismatched -> 90%
# --------------------------------------------------------------------------- #


def test_overall_agreement_at_90_percent_from_synthetic_pair():
    module = _load_rerun_module()
    rows_first = []
    rows_rerun = []
    for i in range(90):
        rid = f"p{i:04d}_r{i:04d}_0"
        rows_first.append({"row_id": rid, "call": "occlusion_signature_present"})
        rows_rerun.append({"row_id": rid, "call": "occlusion_signature_present"})
    for i in range(90, 100):
        rid = f"p{i:04d}_r{i:04d}_0"
        rows_first.append({"row_id": rid, "call": "no_occlusion_signature"})
        rows_rerun.append({"row_id": rid, "call": "indeterminate"})
    first = _first_run_df(rows_first)
    rerun = _rerun_df(rows_rerun)
    agreement = module.compute_agreement(first, rerun)
    assert agreement["n_paired"] == 100
    assert agreement["n_agree"] == 90
    assert agreement["overall_agreement_pct"] == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# Confusion matrix: 3 x 3 over the canonical vocabulary
# --------------------------------------------------------------------------- #


def test_confusion_matrix_is_three_by_three_with_all_call_values():
    module = _load_rerun_module()
    first = _first_run_df(
        [
            {"row_id": "p0001_r0001_0", "call": "occlusion_signature_present"},
            {"row_id": "p0002_r0002_0", "call": "no_occlusion_signature"},
            {"row_id": "p0003_r0003_0", "call": "indeterminate"},
            {"row_id": "p0004_r0004_0", "call": "occlusion_signature_present"},
        ]
    )
    rerun = _rerun_df(
        [
            {"row_id": "p0001_r0001_0", "call": "occlusion_signature_present"},
            {"row_id": "p0002_r0002_0", "call": "indeterminate"},
            {"row_id": "p0003_r0003_0", "call": "indeterminate"},
            {"row_id": "p0004_r0004_0", "call": "no_occlusion_signature"},
        ]
    )
    confusion = module.compute_agreement(first, rerun)["confusion"]
    assert set(confusion.keys()) == set(VALID_CALLS)
    for first_call in VALID_CALLS:
        assert set(confusion[first_call].keys()) == set(VALID_CALLS)
    # Spot-check expected cells.
    assert confusion["occlusion_signature_present"]["occlusion_signature_present"] == 1
    assert confusion["occlusion_signature_present"]["no_occlusion_signature"] == 1
    assert confusion["no_occlusion_signature"]["indeterminate"] == 1
    assert confusion["indeterminate"]["indeterminate"] == 1


# --------------------------------------------------------------------------- #
# Parse-failure rate handles null/invalid calls without contaminating confusion
# --------------------------------------------------------------------------- #


def test_parse_failure_rates_when_either_run_has_null_calls():
    module = _load_rerun_module()
    first = _first_run_df(
        [
            {"row_id": "p0001_r0001_0", "call": "occlusion_signature_present"},
            {"row_id": "p0002_r0002_0", "call": None, "parsed_ok": False},
            {"row_id": "p0003_r0003_0", "call": "indeterminate"},
            {"row_id": "p0004_r0004_0", "call": "no_occlusion_signature"},
        ]
    )
    rerun = _rerun_df(
        [
            {"row_id": "p0001_r0001_0", "call": "occlusion_signature_present"},
            {"row_id": "p0002_r0002_0", "call": "indeterminate"},
            {"row_id": "p0003_r0003_0", "call": None},
            {"row_id": "p0004_r0004_0", "call": "no_occlusion_signature"},
        ]
    )
    agreement = module.compute_agreement(first, rerun)
    assert agreement["parse_failure_rate_first_pct"] == pytest.approx(25.0)
    assert agreement["parse_failure_rate_rerun_pct"] == pytest.approx(25.0)
    # A null call on either side keeps the pair out of the confusion matrix.
    # Only p0001 (present -> present) and p0004 (absent -> absent) survive.
    confusion = agreement["confusion"]
    total = sum(confusion[a][b] for a in VALID_CALLS for b in VALID_CALLS)
    assert total == 2
    assert confusion["occlusion_signature_present"]["occlusion_signature_present"] == 1
    assert confusion["no_occlusion_signature"]["no_occlusion_signature"] == 1


# --------------------------------------------------------------------------- #
# Agreement summary CSV: writes scalar metrics + every confusion cell
# --------------------------------------------------------------------------- #


def test_agreement_summary_csv_round_trip(tmp_path: Path):
    module = _load_rerun_module()
    first = _first_run_df(
        [
            {"row_id": "p0001_r0001_0", "call": "occlusion_signature_present"},
            {"row_id": "p0002_r0002_0", "call": "indeterminate"},
        ]
    )
    rerun = _rerun_df(
        [
            {"row_id": "p0001_r0001_0", "call": "occlusion_signature_present"},
            {"row_id": "p0002_r0002_0", "call": "indeterminate"},
        ]
    )
    agreement = module.compute_agreement(first, rerun)
    summary_path = module.write_agreement_summary(agreement, tmp_path)
    assert summary_path.exists()
    with summary_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    metrics = {row["metric"]: row["value"] for row in rows}
    # Scalars are present.
    assert "overall_agreement_pct" in metrics
    assert "parse_failure_rate_first_pct" in metrics
    assert "parse_failure_rate_rerun_pct" in metrics
    # All nine confusion cells are present.
    for first_call in VALID_CALLS:
        for rerun_call in VALID_CALLS:
            assert f"confusion[{first_call}->{rerun_call}]" in metrics


# --------------------------------------------------------------------------- #
# CLI parses correctly
# --------------------------------------------------------------------------- #


def test_cli_parses_all_required_and_optional_args():
    module = _load_rerun_module()
    args = module._parse_args(
        [
            "--first_run_csv",
            "results/medgemma/run.csv",
            "--inventory",
            "results/feasibility_audit/event_inventory.csv",
            "--wdb-root",
            "data/raw/mimic-iv-wdb/0.1.0",
            "--n_rows",
            "100",
            "--seed",
            "20260426",
            "--out_dir",
            "results/medgemma_determinism/",
            "--port",
            "8001",
        ]
    )
    assert args.first_run_csv == Path("results/medgemma/run.csv")
    assert args.n_rows == 100
    assert args.seed == 20260426
    assert args.port == 8001
    assert args.out_dir == Path("results/medgemma_determinism/")
    assert args.dry_run is False


def test_cli_defaults_are_sane():
    module = _load_rerun_module()
    args = module._parse_args(
        [
            "--first_run_csv",
            "x.csv",
            "--inventory",
            "y.csv",
            "--wdb-root",
            "z/",
        ]
    )
    assert args.n_rows == module.DEFAULT_N_ROWS
    assert args.port == module.DEFAULT_PORT
    # The constructed base URL exposes the chosen port.
    assert module._base_url_from_port(args.port) == "http://localhost:8001/v1"


# --------------------------------------------------------------------------- #
# load_first_run validates the canonical schema
# --------------------------------------------------------------------------- #


def test_load_first_run_rejects_missing_columns(tmp_path: Path):
    module = _load_rerun_module()
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("row_id,call\np0001_r0001_0,indeterminate\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        module.load_first_run(bad_csv)


def test_load_first_run_accepts_canonical_schema(tmp_path: Path):
    module = _load_rerun_module()
    good_csv = tmp_path / "first.csv"
    good_csv.write_text(
        "row_id,subject_id,record_id,call,parsed_ok\n"
        "p0001_r0001_0,p0001,r0001,indeterminate,true\n",
        encoding="utf-8",
    )
    df = module.load_first_run(good_csv)
    assert df.height == 1
    assert df["call"].to_list() == ["indeterminate"]
