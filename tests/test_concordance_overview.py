"""Tests for ``scripts/55_concordance_overview.py``.

Synthetic fixtures plus a thin integration check against the real repo inputs.
The digit-prefixed script is loaded by path through ``importlib.util`` exactly
the way ``tests/test_precision_recall.py`` loads its target.

Covers:

* Inner join on ``card_id`` and canonical three-class mapping.
* The reader x detector x MedGemma cell counts sum to the expected 568, and a
  hand-checkable subset of marginals.
* Ribbon widths (= cell counts) sum to 568 (the load-bearing invariant the
  dispatch requires asserted).
* A bad join (a card missing from one source) is rejected loudly.
* Real-data integration: against the committed inputs the totals reproduce the
  verified reader marginals (102 present / 387 absent / 79 indeterminate) and
  the grand total is 568.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO / "scripts" / "55_concordance_overview.py"
_spec = importlib.util.spec_from_file_location("_conc55", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_conc = importlib.util.module_from_spec(_spec)
sys.modules["_conc55"] = _conc
_spec.loader.exec_module(_conc)


def _write_csvs(
    tmp_path: Path,
    reader_rows: list[tuple[str, str]],
    detector_rows: list[tuple[str, bool]],
    medgemma_rows: list[tuple[str, str]],
) -> tuple[Path, Path, Path]:
    """Write minimal reader / detector / MedGemma CSVs and return their paths."""
    reader_csv = tmp_path / "reader.csv"
    detector_csv = tmp_path / "detector.csv"
    medgemma_csv = tmp_path / "medgemma.csv"
    pl.DataFrame(
        {"card_id": [r[0] for r in reader_rows], "call": [r[1] for r in reader_rows]}
    ).write_csv(reader_csv)
    pl.DataFrame(
        {
            "card_id": [r[0] for r in detector_rows],
            "is_occlusion_signature": [r[1] for r in detector_rows],
        }
    ).write_csv(detector_csv)
    pl.DataFrame(
        {
            "card_id": [r[0] for r in medgemma_rows],
            "call": [r[1] for r in medgemma_rows],
        }
    ).write_csv(medgemma_csv)
    return reader_csv, detector_csv, medgemma_csv


def test_join_and_mapping_to_canonical_classes(tmp_path: Path) -> None:
    """Inner join maps raw calls to present/absent/indeterminate."""
    reader_csv, detector_csv, medgemma_csv = _write_csvs(
        tmp_path,
        reader_rows=[
            ("c1", "occlusion_signature_present"),
            ("c2", "no_occlusion_signature"),
            ("c3", "indeterminate"),
        ],
        detector_rows=[("c1", True), ("c2", False), ("c3", True)],
        medgemma_rows=[
            ("c1", "no_occlusion_signature"),
            ("c2", "no_occlusion_signature"),
            ("c3", "occlusion_signature_present"),
        ],
    )
    merged = _conc.load_three_way(
        reader_csv, detector_csv, medgemma_csv, expected_total=3
    )

    assert merged.height == 3
    by_card = {r["card_id"]: r for r in merged.to_dicts()}
    assert by_card["c1"]["reader"] == "present"
    assert by_card["c1"]["detector"] == "present"
    assert by_card["c1"]["medgemma"] == "absent"
    assert by_card["c2"]["reader"] == "absent"
    assert by_card["c2"]["detector"] == "absent"
    assert by_card["c3"]["reader"] == "indeterminate"
    assert by_card["c3"]["detector"] == "present"
    assert by_card["c3"]["medgemma"] == "present"


def test_cell_counts_sum_and_are_hand_checkable(tmp_path: Path) -> None:
    """Cell counts sum to the fixture total and reproduce a known cell."""
    reader_csv, detector_csv, medgemma_csv = _write_csvs(
        tmp_path,
        reader_rows=[
            ("c1", "no_occlusion_signature"),
            ("c2", "no_occlusion_signature"),
            ("c3", "occlusion_signature_present"),
            ("c4", "no_occlusion_signature"),
        ],
        detector_rows=[("c1", True), ("c2", True), ("c3", False), ("c4", True)],
        medgemma_rows=[
            ("c1", "occlusion_signature_present"),
            ("c2", "occlusion_signature_present"),
            ("c3", "occlusion_signature_present"),
            ("c4", "occlusion_signature_present"),
        ],
    )
    merged = _conc.load_three_way(
        reader_csv, detector_csv, medgemma_csv, expected_total=4
    )
    counts = _conc.cell_counts(merged, expected_total=4)

    assert int(counts["n"].sum()) == 4
    # Three cards are reader-absent / detector-present / medgemma-present.
    cell = counts.filter(
        (pl.col("reader") == "absent")
        & (pl.col("detector") == "present")
        & (pl.col("medgemma") == "present")
    )
    assert cell.height == 1
    assert int(cell["n"][0]) == 3


def test_missing_card_in_one_source_is_rejected(tmp_path: Path) -> None:
    """A card absent from one source breaks the expected-total guard loudly."""
    reader_csv, detector_csv, medgemma_csv = _write_csvs(
        tmp_path,
        reader_rows=[("c1", "no_occlusion_signature"), ("c2", "no_occlusion_signature")],
        detector_rows=[("c1", True)],  # c2 missing from detector
        medgemma_rows=[
            ("c1", "no_occlusion_signature"),
            ("c2", "no_occlusion_signature"),
        ],
    )
    with pytest.raises(ValueError, match="jointly present"):
        _conc.load_three_way(reader_csv, detector_csv, medgemma_csv)


def test_unexpected_raw_call_is_rejected(tmp_path: Path) -> None:
    """An out-of-vocabulary reader call raises rather than mapping to null."""
    reader_csv, detector_csv, medgemma_csv = _write_csvs(
        tmp_path,
        reader_rows=[("c1", "totally_unknown_call")],
        detector_rows=[("c1", True)],
        medgemma_rows=[("c1", "no_occlusion_signature")],
    )
    # polars `replace_strict` raises on an unseen mapping key.
    with pytest.raises(pl.exceptions.PolarsError):
        _conc.load_three_way(
            reader_csv, detector_csv, medgemma_csv, expected_total=1
        )


def test_real_inputs_reproduce_verified_marginals_and_total() -> None:
    """Against committed inputs: 568 cards, verified reader marginals."""
    reader_csv = _REPO / "results/gallery/reader_form_blinded.csv"
    detector_csv = _REPO / "results/gallery/gallery_manifest.csv"
    medgemma_csv = (
        _REPO / "results/medgemma_galleryrender/gallery_render_calls_cardkeyed.csv"
    )
    if not (reader_csv.exists() and detector_csv.exists() and medgemma_csv.exists()):
        pytest.skip("repo result CSVs not present in this checkout")

    merged = _conc.load_three_way(reader_csv, detector_csv, medgemma_csv)
    assert merged.height == 568

    reader_counts = {
        r["reader"]: r["len"]
        for r in merged.group_by("reader").len().to_dicts()
    }
    assert reader_counts["present"] == 102
    assert reader_counts["absent"] == 387
    assert reader_counts["indeterminate"] == 79

    counts = _conc.cell_counts(merged)
    # The load-bearing invariant: ribbon widths (cell counts) sum to 568.
    assert int(counts["n"].sum()) == 568
    # Detector has no indeterminate state.
    assert (
        merged.filter(pl.col("detector") == "indeterminate").height == 0
    )
    # MedGemma in the gallery-render run emits only present/absent.
    assert (
        merged.filter(pl.col("medgemma") == "indeterminate").height == 0
    )
