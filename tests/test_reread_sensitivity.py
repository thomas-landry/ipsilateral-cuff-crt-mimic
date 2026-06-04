"""Tests for ``scripts/62_reread_sensitivity.py``.

Synthetic fixtures only; the script never touches real data here. The script
filename starts with a digit, so it is loaded by path through ``importlib.util``
exactly the way the existing precision/recall test loads scripts/44.

Covers the two-reference metric computation:

* Precision / recall / specificity point estimates under the pass-1 vs pass-2
  reference on a hand-checkable fixture where TP, FP, TN, FN are known by hand
  for each pass and each index test.
* Reference-indeterminate rows drop out of the binary denominator (matching
  scripts/44), and the per-pass uncallable rate counts them.
* The pass1 -> pass2 delta equals pass2_point - pass1_point.
* The clustered bootstrap is deterministic under a fixed seed.
* ``load_sample`` attaches subject_id from the manifest and fails loud when a
  sampled card has no subject.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "62_reread_sensitivity.py"
)
_spec = importlib.util.spec_from_file_location("_rs62", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_rs = importlib.util.module_from_spec(_spec)
sys.modules["_rs62"] = _rs
_spec.loader.exec_module(_rs)

OSP = _rs.OCCLUSION_SIGNATURE_PRESENT
NOS = _rs.NO_OCCLUSION_SIGNATURE
IND = _rs.INDETERMINATE


def _sample(rows: list[tuple[str, str, str, str, str, str]]) -> pl.DataFrame:
    """Build a sample-shaped frame already carrying subject_id and all calls.

    Each tuple is
    ``(card_id, subject_id, detector_call, language_model_call, pass1, pass2)``.
    """
    return pl.DataFrame(
        {
            "card_id": [r[0] for r in rows],
            "subject_id": [r[1] for r in rows],
            "detector_call": [r[2] for r in rows],
            "language_model_call": [r[3] for r in rows],
            "pass1_call": [r[4] for r in rows],
            "pass2_call": [r[5] for r in rows],
        }
    )


def _twelve_row_fixture() -> pl.DataFrame:
    """A fixture where each pass's confusion cells are known by hand.

    The detector call is the index test under test. Rows:

    Index (detector) positives: rows 1-6 (det = OSP). Negatives: rows 7-12.

    Pass 1 reference (one reader-indeterminate row that must drop out):
      det OSP & p1 OSP -> TP        rows 1, 2, 3
      det OSP & p1 NOS -> FP        rows 4, 5
      det OSP & p1 IND -> dropped   row 6
      det NOS & p1 OSP -> FN        row 7
      det NOS & p1 NOS -> TN        rows 8, 9, 10, 11, 12
      Pass-1 eligible n = 11. TP=3, FP=2, FN=1, TN=5.
        precision = 3/(3+2) = 0.600
        recall    = 3/(3+1) = 0.750
        specificity = 5/(5+2) = 0.714286

    Pass 2 reference (the corrected re-read; row 6 now callable as OSP, and
    rows 4, 5 flip NOS->OSP, i.e. the reader "undercalled" present on pass 1):
      det OSP & p2 OSP -> TP        rows 1, 2, 3, 4, 5, 6
      det NOS & p2 OSP -> FN        row 7
      det NOS & p2 NOS -> TN        rows 8, 9, 10, 11, 12
      Pass-2 eligible n = 12. TP=6, FP=0, FN=1, TN=5.
        precision = 6/(6+0) = 1.000
        recall    = 6/(6+1) = 0.857143
        specificity = 5/(5+0) = 1.000
    """
    rows = [
        # card, subj, detector, lm, pass1, pass2
        ("c01", "s1", OSP, OSP, OSP, OSP),
        ("c02", "s1", OSP, OSP, OSP, OSP),
        ("c03", "s2", OSP, OSP, OSP, OSP),
        ("c04", "s2", OSP, NOS, NOS, OSP),
        ("c05", "s3", OSP, NOS, NOS, OSP),
        ("c06", "s3", OSP, OSP, IND, OSP),
        ("c07", "s4", NOS, OSP, OSP, OSP),
        ("c08", "s4", NOS, NOS, NOS, NOS),
        ("c09", "s5", NOS, NOS, NOS, NOS),
        ("c10", "s5", NOS, NOS, NOS, NOS),
        ("c11", "s6", NOS, NOS, NOS, NOS),
        ("c12", "s6", NOS, NOS, NOS, NOS),
    ]
    return _sample(rows)


def _detector_point(summary: pl.DataFrame, metric: str, reference: str) -> float:
    sub = summary.filter(
        (pl.col("index_test") == "detector")
        & (pl.col("metric") == metric)
        & (pl.col("reference") == reference)
    )
    return float(sub.get_column("point_estimate")[0])


def _detector_n(summary: pl.DataFrame, metric: str, reference: str) -> int:
    sub = summary.filter(
        (pl.col("index_test") == "detector")
        & (pl.col("metric") == metric)
        & (pl.col("reference") == reference)
    )
    return int(sub.get_column("n_used_for_metric")[0])


def test_detector_points_under_both_references() -> None:
    fixture = _twelve_row_fixture()
    summary = _rs.compute_two_reference_metrics(
        fixture, n_bootstrap=200, seed=20260426
    )

    # Pass 1 (hand-derived above).
    assert _detector_point(summary, "precision", "pass1") == pytest.approx(0.600)
    assert _detector_point(summary, "recall", "pass1") == pytest.approx(0.750)
    assert _detector_point(summary, "specificity", "pass1") == pytest.approx(5.0 / 7.0)

    # Pass 2 (the corrected reference).
    assert _detector_point(summary, "precision", "pass2") == pytest.approx(1.000)
    assert _detector_point(summary, "recall", "pass2") == pytest.approx(6.0 / 7.0)
    assert _detector_point(summary, "specificity", "pass2") == pytest.approx(1.000)


def test_indeterminate_reference_row_drops_from_denominator() -> None:
    fixture = _twelve_row_fixture()
    summary = _rs.compute_two_reference_metrics(
        fixture, n_bootstrap=200, seed=20260426
    )
    # Pass 1 has one reader-indeterminate row (c06) among the detector
    # positives, so the precision denominator is 5 not 6, and the total
    # eligible n is 11. Pass 2 makes c06 callable: precision denominator 6.
    assert _detector_n(summary, "precision", "pass1") == 5
    assert _detector_n(summary, "precision", "pass2") == 6
    # Recall denominator = reference-positives. Pass 1 OSP among eligible: rows
    # 1,2,3 (det-pos) + 7 (det-neg) = 4. Pass 2 OSP: rows 1-6 + 7 = 7.
    assert _detector_n(summary, "recall", "pass1") == 4
    assert _detector_n(summary, "recall", "pass2") == 7


def test_uncallable_rate_counts_reference_indeterminate() -> None:
    fixture = _twelve_row_fixture()
    rates = _rs.compute_uncallable_rates(fixture, n_bootstrap=200, seed=20260426)
    p1 = rates.filter(pl.col("reference") == "pass1")
    p2 = rates.filter(pl.col("reference") == "pass2")
    assert int(p1.get_column("indeterminate_count")[0]) == 1  # c06
    assert int(p2.get_column("indeterminate_count")[0]) == 0
    assert int(p1.get_column("total_n")[0]) == 12
    assert float(p1.get_column("indeterminate_rate")[0]) == pytest.approx(1.0 / 12.0)


def test_delta_equals_pass2_minus_pass1() -> None:
    fixture = _twelve_row_fixture()
    summary = _rs.compute_two_reference_metrics(
        fixture, n_bootstrap=200, seed=20260426
    )
    delta = _rs.compute_delta_table(summary)
    det_prec = delta.filter(
        (pl.col("index_test") == "detector") & (pl.col("metric") == "precision")
    )
    p1 = float(det_prec.get_column("pass1_point")[0])
    p2 = float(det_prec.get_column("pass2_point")[0])
    d = float(det_prec.get_column("delta")[0])
    assert d == pytest.approx(p2 - p1)
    # Correcting the undercalls raises detector precision here.
    assert d > 0


def test_bootstrap_reproducible_under_same_seed() -> None:
    fixture = _twelve_row_fixture()
    t1 = _rs.compute_two_reference_metrics(fixture, n_bootstrap=500, seed=20260426)
    t2 = _rs.compute_two_reference_metrics(fixture, n_bootstrap=500, seed=20260426)
    cols = ["index_test", "metric", "reference", "point_estimate", "ci_low", "ci_high"]
    assert t1.select(cols).equals(t2.select(cols))


def test_load_sample_attaches_subject_and_fails_on_missing(tmp_path: Path) -> None:
    sample = pl.DataFrame(
        {
            "card_id": ["A-1", "B-1"],
            "stratum": ["detector_positive", "detector_rejected_near_miss"],
            "detector_call": [OSP, NOS],
            "language_model_call": [OSP, NOS],
            "pass1_call": [OSP, NOS],
            "pass2_call": [OSP, OSP],
        }
    )
    sample_csv = tmp_path / "sample.csv"
    sample.write_csv(sample_csv)

    good_manifest = pl.DataFrame(
        {"card_id": ["A-1", "B-1"], "subject_id": ["s1", "s2"]}
    )
    man_csv = tmp_path / "manifest.csv"
    good_manifest.write_csv(man_csv)
    loaded = _rs.load_sample(sample_csv, man_csv)
    assert loaded.height == 2
    assert set(loaded.columns) == {
        "card_id",
        "subject_id",
        "detector_call",
        "language_model_call",
        "pass1_call",
        "pass2_call",
    }
    assert loaded.get_column("subject_id").to_list() == ["s1", "s2"]

    bad_manifest = pl.DataFrame({"card_id": ["A-1"], "subject_id": ["s1"]})
    bad_csv = tmp_path / "bad_manifest.csv"
    bad_manifest.write_csv(bad_csv)
    with pytest.raises(ValueError, match="subject_id"):
        _rs.load_sample(sample_csv, bad_csv)


def test_run_writes_all_five_artifacts(tmp_path: Path) -> None:
    sample = pl.DataFrame(
        {
            "card_id": [r[0] for r in _FIXTURE_ROWS],
            "stratum": ["detector_positive"] * len(_FIXTURE_ROWS),
            "detector_call": [r[2] for r in _FIXTURE_ROWS],
            "language_model_call": [r[3] for r in _FIXTURE_ROWS],
            "pass1_call": [r[4] for r in _FIXTURE_ROWS],
            "pass2_call": [r[5] for r in _FIXTURE_ROWS],
        }
    )
    sample_csv = tmp_path / "sample.csv"
    sample.write_csv(sample_csv)
    manifest = pl.DataFrame(
        {
            "card_id": [r[0] for r in _FIXTURE_ROWS],
            "subject_id": [r[1] for r in _FIXTURE_ROWS],
        }
    )
    man_csv = tmp_path / "manifest.csv"
    manifest.write_csv(man_csv)
    out_dir = tmp_path / "out"

    _rs.run(
        sample_csv=sample_csv,
        gallery_manifest=man_csv,
        out_dir=out_dir,
        seed=20260426,
        n_bootstrap=200,
    )
    for name in (
        "reread_sensitivity_summary.csv",
        "reread_sensitivity_delta.csv",
        "reread_uncallable_rates.csv",
        "reread_sensitivity_summary.md",
        "run_metadata.json",
    ):
        assert (out_dir / name).exists(), name

    import json

    meta = json.loads((out_dir / "run_metadata.json").read_text())
    assert meta["seed"] == 20260426
    assert meta["n_cards"] == len(_FIXTURE_ROWS)
    assert "sha256" in meta["inputs"]["sample_csv"]
    # No em-dash in the rendered Markdown summary.
    md = (out_dir / "reread_sensitivity_summary.md").read_text()
    assert "—" not in md


_FIXTURE_ROWS = [
    ("c01", "s1", OSP, OSP, OSP, OSP),
    ("c02", "s1", OSP, OSP, OSP, OSP),
    ("c03", "s2", OSP, OSP, OSP, OSP),
    ("c04", "s2", OSP, NOS, NOS, OSP),
    ("c05", "s3", OSP, NOS, NOS, OSP),
    ("c06", "s3", OSP, OSP, IND, OSP),
    ("c07", "s4", NOS, OSP, OSP, OSP),
    ("c08", "s4", NOS, NOS, NOS, NOS),
    ("c09", "s5", NOS, NOS, NOS, NOS),
    ("c10", "s5", NOS, NOS, NOS, NOS),
    ("c11", "s6", NOS, NOS, NOS, NOS),
    ("c12", "s6", NOS, NOS, NOS, NOS),
]
