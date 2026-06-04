"""Tests for ``scripts/44_precision_recall.py``.

Synthetic fixtures only; the script is never executed against real data here.
The script filename starts with a digit so it is loaded by path through
``importlib.util`` exactly the same way the sensitivity-sweep script imports
the extractor.

Covers:

* Inner join semantics (only ``card_id``s present in all three inputs survive;
  drop counts reported per source).
* Confusion-matrix correctness on a hand-checkable 10-row fixture.
* Binary precision / recall / specificity formulas on a fixture where TP, FP,
  TN, FN are known by hand.
* Indeterminate-exclusion logic: rows where reader OR predictor is
  indeterminate are excluded from binary metrics but counted in
  ``indeterminate_rates``.
* Parse-failure handling: counted separately from indeterminate in
  ``indeterminate_rates`` and as its own confusion-matrix column for MedGemma.
* Cluster-bootstrap determinism under a fixed seed.
* Cluster-vs-row sanity: concentrating positives in a single subject widens
  the CI relative to a row-level (1-cluster-per-row) bootstrap on identical
  marginal counts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Load scripts/44_precision_recall.py by path (digit-prefixed filename).
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "44_precision_recall.py"
)
_spec = importlib.util.spec_from_file_location("_pr44", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_pr = importlib.util.module_from_spec(_spec)
sys.modules["_pr44"] = _pr
_spec.loader.exec_module(_pr)

OSP = _pr.OCCLUSION_SIGNATURE_PRESENT
NOS = _pr.NO_OCCLUSION_SIGNATURE
IND = _pr.INDETERMINATE
PF = _pr.PARSE_FAILURE


def _reader(rows: list[tuple[str, str]]) -> pl.DataFrame:
    """Build a reader-style frame the loader will accept after dropping empty calls."""
    return pl.DataFrame(
        {
            "card_id": [r[0] for r in rows],
            "call": [r[1] for r in rows],
        }
    )


def _medgemma(rows: list[tuple[str, str | None, bool]]) -> pl.DataFrame:
    """Build a MedGemma-style frame with ``card_id, call, parsed_ok``."""
    return pl.DataFrame(
        {
            "card_id": [r[0] for r in rows],
            "call": [r[1] for r in rows],
            "parsed_ok": [r[2] for r in rows],
        }
    )


def _manifest(rows: list[tuple[str, str, bool]]) -> pl.DataFrame:
    """Build a manifest-style frame; ``is_occlusion_signature`` is True for A-cards."""
    return pl.DataFrame(
        {
            "card_id": [r[0] for r in rows],
            "subject_id": [r[1] for r in rows],
            "is_occlusion_signature": [r[2] for r in rows],
        }
    )


def _write_and_load_reader(tmp_path: Path, df: pl.DataFrame) -> pl.DataFrame:
    path = tmp_path / "reader.csv"
    df.write_csv(path)
    return _pr.load_reader(path)


def _write_and_load_medgemma(tmp_path: Path, df: pl.DataFrame) -> pl.DataFrame:
    path = tmp_path / "medgemma.csv"
    df.write_csv(path)
    return _pr.load_medgemma(path)


def _write_and_load_manifest(tmp_path: Path, df: pl.DataFrame) -> pl.DataFrame:
    path = tmp_path / "manifest.csv"
    df.write_csv(path)
    return _pr.load_manifest(path)


# ---------------------------------------------------------------------------
# Join semantics
# ---------------------------------------------------------------------------


def test_inner_join_drops_only_unmatched_card_ids(tmp_path: Path) -> None:
    """Cards missing in any one source are dropped and counted per-source."""
    reader = _write_and_load_reader(
        tmp_path,
        _reader(
            [
                ("A-1", OSP),
                ("A-2", NOS),
                ("A-3", OSP),  # not in medgemma -> dropped, blamed on medgemma
                ("A-4", OSP),  # not in manifest -> dropped, blamed on manifest
            ]
        ),
    )
    medgemma = _write_and_load_medgemma(
        tmp_path,
        _medgemma(
            [
                ("A-1", OSP, True),
                ("A-2", NOS, True),
                ("A-4", OSP, True),
                ("A-5", OSP, True),  # not in reader -> dropped, blamed on reader
            ]
        ),
    )
    manifest = _write_and_load_manifest(
        tmp_path,
        _manifest(
            [
                ("A-1", "s1", True),
                ("A-2", "s1", False),
                ("A-3", "s2", True),
                ("A-5", "s2", True),
            ]
        ),
    )

    res = _pr.join_sources(reader, medgemma, manifest)
    assert res.n_joined == 2
    assert set(res.joined.get_column("card_id").to_list()) == {"A-1", "A-2"}
    # A-3 in reader+manifest but not medgemma.
    assert res.n_dropped_medgemma == 1
    # A-4 in reader+medgemma but not manifest.
    assert res.n_dropped_manifest == 1
    # A-5 in medgemma+manifest but not reader.
    assert res.n_dropped_reader == 1


def test_reader_empty_call_rows_are_dropped(tmp_path: Path) -> None:
    """Reader rows whose ``call`` is the empty string are treated as unrated."""
    df = pl.DataFrame(
        {
            "card_id": ["A-1", "A-2", "A-3"],
            "call": [OSP, "", "   "],
        }
    )
    loaded = _write_and_load_reader(tmp_path, df)
    assert loaded.height == 1
    assert loaded.get_column("card_id").to_list() == ["A-1"]


# ---------------------------------------------------------------------------
# Confusion matrix correctness on a hand-checkable 10-row fixture
# ---------------------------------------------------------------------------


def _ten_row_joined() -> pl.DataFrame:
    """Hand-built joined frame: 10 cards, reader vs predictors known cell-by-cell.

    Reader: 5 OSP, 3 NOS, 1 IND, 1 OSP (= 6 OSP total, 3 NOS, 1 IND).

    MedGemma calls (parsed_ok all True):
      OSP-OSP=3, OSP-NOS=2, NOS-OSP=1, NOS-NOS=2, OSP-IND=1, IND-NOS=1.
      So 3 TP, 1 FP (NOS-OSP), 2 TN, 2 FN (OSP-NOS), plus IND cells (1 OSP-IND
      and 1 IND-NOS) that drop out of the binary metric.
    """
    rows = [
        # card_id, subject, reader, medgemma, parsed_ok, detector_bool
        ("A-1", "s1", OSP, OSP, True, True),
        ("A-2", "s1", OSP, OSP, True, True),
        ("A-3", "s2", OSP, OSP, True, True),
        ("A-4", "s2", OSP, NOS, True, True),
        ("A-5", "s3", OSP, NOS, True, True),
        ("A-6", "s3", OSP, IND, True, True),  # OSP reader, IND medgemma
        ("A-7", "s4", NOS, OSP, True, False),
        ("A-8", "s4", NOS, NOS, True, False),
        ("A-9", "s5", NOS, NOS, True, False),
        ("A-10", "s5", IND, NOS, True, False),  # IND reader
    ]
    return pl.DataFrame(
        {
            "card_id": [r[0] for r in rows],
            "subject_id": [r[1] for r in rows],
            "reader_call": [r[2] for r in rows],
            "medgemma_call": [r[3] for r in rows],
            "medgemma_parsed_ok": [r[4] for r in rows],
            "detector_call": [OSP if r[5] else NOS for r in rows],
        }
    )


def test_confusion_matrix_hand_checked_counts() -> None:
    joined = _ten_row_joined()
    cm = _pr.compute_confusion_matrices(joined)
    medgemma_cm = cm.filter(pl.col("predictor") == "medgemma")

    def cell(ref: str, pred: str) -> int:
        sub = medgemma_cm.filter(
            (pl.col("reference_value") == ref) & (pl.col("predictor_value") == pred)
        )
        if sub.height == 0:
            return 0
        return int(sub.get_column("count")[0])

    assert cell(OSP, OSP) == 3
    assert cell(OSP, NOS) == 2
    assert cell(OSP, IND) == 1
    assert cell(NOS, OSP) == 1
    assert cell(NOS, NOS) == 2
    assert cell(NOS, IND) == 0
    assert cell(IND, OSP) == 0
    assert cell(IND, NOS) == 1
    assert cell(IND, IND) == 0
    # No parse failures in this fixture.
    assert cell(OSP, PF) == 0
    assert cell(NOS, PF) == 0
    assert cell(IND, PF) == 0


# ---------------------------------------------------------------------------
# Precision / recall / specificity formulas
# ---------------------------------------------------------------------------


def test_binary_metric_formulas_on_known_fixture() -> None:
    joined = _ten_row_joined()
    # Hand-derived after dropping rows with reader IND or predictor IND:
    # MedGemma denominator = 8 rows. TP=3, FP=1, FN=2, TN=2.
    # precision = 3/(3+1) = 0.75
    # recall    = 3/(3+2) = 0.60
    # specificity = 2/(2+1) = 0.6667
    table = _pr.compute_metric_table(joined, n_bootstrap=200, seed=20260426)
    mg = table.filter(pl.col("predictor") == "medgemma").sort("metric")

    def point(metric: str) -> float:
        sub = mg.filter(pl.col("metric") == metric)
        return float(sub.get_column("point_estimate")[0])

    def n_used(metric: str) -> int:
        sub = mg.filter(pl.col("metric") == metric)
        return int(sub.get_column("n_used_for_metric")[0])

    assert point("precision") == pytest.approx(0.75)
    assert point("recall") == pytest.approx(0.60)
    assert point("specificity") == pytest.approx(2.0 / 3.0)
    # Denominators: precision uses predictor-positives (TP+FP=4); recall uses
    # reference-positives (TP+FN=5); specificity uses reference-negatives
    # (TN+FP=3).
    assert n_used("precision") == 4
    assert n_used("recall") == 5
    assert n_used("specificity") == 3


# ---------------------------------------------------------------------------
# Indeterminate / parse-failure handling
# ---------------------------------------------------------------------------


def test_indeterminate_rows_excluded_from_binary_but_counted_in_rates() -> None:
    joined = _ten_row_joined()
    rates = _pr.compute_indeterminate_rates(joined, n_bootstrap=200, seed=20260426)
    mg = rates.filter(pl.col("predictor") == "medgemma")
    assert int(mg.get_column("indeterminate_count")[0]) == 1  # A-6
    assert int(mg.get_column("parse_failure_count")[0]) == 0
    assert int(mg.get_column("total_n")[0]) == 10
    # 1 IND + 0 PF over 10 = 0.10. (Note: the reader IND row, A-10, is NOT
    # counted in the MedGemma uncallable rate; only MedGemma's own
    # indeterminate or parse-failure counts in the MedGemma rate.)
    assert float(mg.get_column("indeterminate_rate")[0]) == pytest.approx(0.1)

    det = rates.filter(pl.col("predictor") == "detector")
    assert int(det.get_column("indeterminate_count")[0]) == 0
    assert int(det.get_column("parse_failure_count")[0]) == 0


def test_parse_failure_counted_separately_from_indeterminate() -> None:
    """A parse-failure row goes into the PARSE_FAILURE column, not IND."""
    rows = [
        ("A-1", "s1", OSP, None, False, True),
        ("A-2", "s1", OSP, IND, True, True),
        ("A-3", "s2", NOS, OSP, True, False),
        ("A-4", "s2", NOS, NOS, True, False),
    ]
    joined = pl.DataFrame(
        {
            "card_id": [r[0] for r in rows],
            "subject_id": [r[1] for r in rows],
            "reader_call": [r[2] for r in rows],
            "medgemma_call": [r[3] for r in rows],
            "medgemma_parsed_ok": [r[4] for r in rows],
            "detector_call": [OSP if r[5] else NOS for r in rows],
        }
    )

    cm = _pr.compute_confusion_matrices(joined)
    medgemma_cm = cm.filter(pl.col("predictor") == "medgemma")
    pf_row = medgemma_cm.filter(
        (pl.col("reference_value") == OSP)
        & (pl.col("predictor_value") == PF)
    )
    assert int(pf_row.get_column("count")[0]) == 1

    rates = _pr.compute_indeterminate_rates(joined, n_bootstrap=200, seed=20260426)
    mg = rates.filter(pl.col("predictor") == "medgemma")
    assert int(mg.get_column("indeterminate_count")[0]) == 1
    assert int(mg.get_column("parse_failure_count")[0]) == 1


# ---------------------------------------------------------------------------
# Bootstrap determinism and cluster effect
# ---------------------------------------------------------------------------


def test_bootstrap_ci_reproducible_under_same_seed() -> None:
    joined = _ten_row_joined()
    t1 = _pr.compute_metric_table(joined, n_bootstrap=500, seed=20260426)
    t2 = _pr.compute_metric_table(joined, n_bootstrap=500, seed=20260426)
    cols = ["predictor", "metric", "point_estimate", "ci_low", "ci_high"]
    assert t1.select(cols).equals(t2.select(cols))


def test_subject_clustering_widens_ci_vs_row_level() -> None:
    """Concentrating all positives in one subject widens the CI versus the
    same marginal counts spread one-per-cluster (row-level bootstrap).

    Construction: 20 rows, marginal positive = 10. In the clustered variant
    all 10 positives belong to a single subject; the row-level variant gives
    each row its own subject id. The clustered CI must be wider.
    """
    n = 20
    values = np.array([1.0] * 10 + [0.0] * 10)
    one_cluster = np.array(["s1"] * 10 + ["s2"] * 10)
    row_level = np.array([f"s{i}" for i in range(n)])

    # Use the script's _ratio_ci so we hit the same code path. Treat the
    # statistic as a mean: num = values, den = 1 for every row.
    num = values
    den = np.ones_like(values)

    p_c, lo_c, hi_c, _ = _pr._ratio_ci(
        num, den, one_cluster, n_bootstrap=2000, seed=20260426
    )
    p_r, lo_r, hi_r, _ = _pr._ratio_ci(
        num, den, row_level, n_bootstrap=2000, seed=20260426
    )
    width_c = hi_c - lo_c
    width_r = hi_r - lo_r
    # Marginal proportion identical by construction.
    assert p_c == pytest.approx(p_r)
    # Clustered CI strictly wider than the row-level CI.
    assert width_c > width_r


def test_detector_call_derivation_from_manifest(tmp_path: Path) -> None:
    """A-stratum cards (is_occlusion_signature=True) map to OSP; B/C to NOS."""
    df = pl.DataFrame(
        {
            "card_id": ["A-1", "B-1", "C-1"],
            "subject_id": ["s1", "s1", "s2"],
            "is_occlusion_signature": [True, False, False],
        }
    )
    loaded = _write_and_load_manifest(tmp_path, df)
    calls = dict(
        zip(
            loaded.get_column("card_id").to_list(),
            loaded.get_column("detector_call").to_list(),
            strict=True,
        )
    )
    assert calls == {"A-1": OSP, "B-1": NOS, "C-1": NOS}


def test_medgemma_loader_errors_when_card_id_missing(tmp_path: Path) -> None:
    """The harness today writes 'row_id'; the loader must refuse to guess."""
    bad = pl.DataFrame(
        {
            "row_id": ["s1_r1_0", "s1_r1_1"],
            "call": [OSP, NOS],
            "parsed_ok": [True, True],
        }
    )
    path = tmp_path / "bad_medgemma.csv"
    bad.write_csv(path)
    with pytest.raises(ValueError, match="card_id"):
        _pr.load_medgemma(path)


def test_manifest_loader_errors_without_subject_id(tmp_path: Path) -> None:
    """subject_id is required for the cluster-bootstrap; absence is fatal."""
    bad = pl.DataFrame(
        {
            "card_id": ["A-1"],
            "is_occlusion_signature": [True],
        }
    )
    path = tmp_path / "bad_manifest.csv"
    bad.write_csv(path)
    with pytest.raises(ValueError, match="subject_id"):
        _pr.load_manifest(path)


# ---------------------------------------------------------------------------
# End-to-end smoke test through the run() pipeline
# ---------------------------------------------------------------------------


def test_run_end_to_end_writes_all_four_files(tmp_path: Path) -> None:
    reader_csv = tmp_path / "reader.csv"
    medgemma_csv = tmp_path / "medgemma.csv"
    manifest_csv = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"

    _reader(
        [
            ("A-1", OSP),
            ("A-2", OSP),
            ("B-1", NOS),
            ("B-2", NOS),
        ]
    ).write_csv(reader_csv)
    _medgemma(
        [
            ("A-1", OSP, True),
            ("A-2", NOS, True),
            ("B-1", NOS, True),
            ("B-2", OSP, True),
        ]
    ).write_csv(medgemma_csv)
    _manifest(
        [
            ("A-1", "s1", True),
            ("A-2", "s1", True),
            ("B-1", "s2", False),
            ("B-2", "s2", False),
        ]
    ).write_csv(manifest_csv)

    _pr.run(
        reader_csv=reader_csv,
        medgemma_csv=medgemma_csv,
        gallery_manifest=manifest_csv,
        out_dir=out_dir,
        seed=20260426,
        n_bootstrap=200,
    )

    assert (out_dir / "precision_recall_summary.csv").exists()
    assert (out_dir / "confusion_matrices.csv").exists()
    assert (out_dir / "indeterminate_rates.csv").exists()
    assert (out_dir / "run_metadata.json").exists()

    import json

    meta = json.loads((out_dir / "run_metadata.json").read_text())
    assert meta["seed"] == 20260426
    assert meta["n_bootstrap"] == 200
    assert meta["join"]["n_joined"] == 4
    assert "sha256" in meta["inputs"]["reader_csv"]
