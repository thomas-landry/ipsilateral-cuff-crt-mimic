"""Integrity tests for ``scripts/53_disagreement_figure.py``.

The figure's scientific claim is that each panel's three calls
(Reader / Detector / MedGemma) match the source data for its disagreement
category. These tests enforce that by (1) asserting every selected card
satisfies its category's call predicate against the joined source tables, and
(2) checking that selection is deterministic under a fixed seed.

Selection runs against the real (read-only) joined call sources; rendering,
which needs the credentialed WDB tree, is not exercised here. The script
filename starts with a digit so it is loaded by path through
``importlib.util``, the same convention the other script tests use.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = REPO / "scripts" / "53_disagreement_figure.py"
_spec = importlib.util.spec_from_file_location("_disagreement53", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["_disagreement53"] = mod
_spec.loader.exec_module(mod)

READER_CSV = REPO / "results/gallery/reader_form_blinded.csv"
MANIFEST_CSV = REPO / "results/gallery/gallery_manifest.csv"
MEDGEMMA_CSV = REPO / "results/medgemma_galleryrender/gallery_render_calls_cardkeyed.csv"

_HAVE_DATA = READER_CSV.exists() and MANIFEST_CSV.exists() and MEDGEMMA_CSV.exists()
needs_data = pytest.mark.skipif(not _HAVE_DATA, reason="gallery call sources not present")


def _toy_joined() -> pl.DataFrame:
    """A small hand-built joined table covering all five categories.

    Each row is engineered so its category membership is obvious by eye, so the
    predicate logic can be tested without the real (large) data present.
    """
    R_PRES = mod.R_PRESENT
    R_ABS = mod.R_ABSENT
    R_IND = mod.R_INDETERMINATE
    rows = [
        # detector_overcall: reader absent, detector present, medgemma NOT present
        dict(card_id="A-det1", reader_call=R_ABS, reader_conf="high",
             is_occlusion_signature=True, medgemma_call=R_ABS, medgemma_conf=0.8),
        dict(card_id="A-det2", reader_call=R_ABS, reader_conf="high",
             is_occlusion_signature=True, medgemma_call=R_IND, medgemma_conf=0.5),
        # both_machines_overcall: reader absent, detector present, medgemma present
        dict(card_id="A-both1", reader_call=R_ABS, reader_conf="high",
             is_occlusion_signature=True, medgemma_call=R_PRES, medgemma_conf=0.95),
        dict(card_id="A-both2", reader_call=R_ABS, reader_conf="med",
             is_occlusion_signature=True, medgemma_call=R_PRES, medgemma_conf=0.9),
        # reader_only_present: reader present, detector absent, medgemma absent
        dict(card_id="B-ro1", reader_call=R_PRES, reader_conf="high",
             is_occlusion_signature=False, medgemma_call=R_ABS, medgemma_conf=0.7),
        # reader_indeterminate: reader indeterminate, medgemma confident
        dict(card_id="C-ind1", reader_call=R_IND, reader_conf="med",
             is_occlusion_signature=False, medgemma_call=R_PRES, medgemma_conf=0.95),
        dict(card_id="C-ind2", reader_call=R_IND, reader_conf="low",
             is_occlusion_signature=True, medgemma_call=R_PRES, medgemma_conf=0.92),
        # all_agree_positive: reader present, detector present, medgemma present
        dict(card_id="A-all1", reader_call=R_PRES, reader_conf="high",
             is_occlusion_signature=True, medgemma_call=R_PRES, medgemma_conf=0.9),
        # a non-member control: reader present but medgemma absent and detector present
        dict(card_id="A-ctrl", reader_call=R_PRES, reader_conf="med",
             is_occlusion_signature=True, medgemma_call=R_ABS, medgemma_conf=0.6),
    ]
    # Add synthetic morphology + ids so select_cards' z-scoring runs.
    for i, r in enumerate(rows):
        r["stratum"] = "detector_positive"
        r["subject_id"] = f"p{i:08d}"
        r["record_id"] = f"r{i:08d}"
        r["t_nbp"] = 100.0 + i
        r["phase3_duration_s"] = 12.0 + i
        r["nadir_depth_frac"] = 0.05 + 0.01 * i
        r["alignment_offset_s"] = -5.0 + i
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.when(pl.col("is_occlusion_signature"))
        .then(pl.lit(mod.DET_PRESENT))
        .otherwise(pl.lit(mod.DET_ABSENT))
        .alias("detector_call")
    )


def _assert_card_matches_category(row: dict, cat_key: str) -> None:
    """Assert a single selected row's three calls match its category definition."""
    R_PRES, R_ABS, R_IND = mod.R_PRESENT, mod.R_ABSENT, mod.R_INDETERMINATE
    if cat_key == "detector_overcall":
        assert row["reader_call"] == R_ABS
        assert row["detector_call"] == mod.DET_PRESENT
        assert row["medgemma_call"] != R_PRES
    elif cat_key == "both_machines_overcall":
        assert row["reader_call"] == R_ABS
        assert row["detector_call"] == mod.DET_PRESENT
        assert row["medgemma_call"] == R_PRES
    elif cat_key == "reader_only_present":
        assert row["reader_call"] == R_PRES
        assert row["detector_call"] == mod.DET_ABSENT
        assert row["medgemma_call"] == R_ABS
    elif cat_key == "reader_indeterminate":
        assert row["reader_call"] == R_IND
        assert float(row["medgemma_confidence"]) >= mod.MACHINE_CONFIDENT_THRESHOLD
    elif cat_key == "all_agree_positive":
        assert row["reader_call"] == R_PRES
        assert row["detector_call"] == mod.DET_PRESENT
        assert row["medgemma_call"] == R_PRES
    else:
        raise AssertionError(f"unknown category {cat_key!r}")


# --- Synthetic-data tests (always run) --------------------------------------
def test_predicates_partition_toy_data_as_expected() -> None:
    """Each toy card lands in exactly the categories it was engineered for."""
    j = _toy_joined()
    membership = {
        "detector_overcall": {"A-det1", "A-det2"},
        "both_machines_overcall": {"A-both1", "A-both2"},
        "reader_only_present": {"B-ro1"},
        "reader_indeterminate": {"C-ind1", "C-ind2"},
        "all_agree_positive": {"A-all1"},
    }
    for cat_key, expected in membership.items():
        got = set(
            j.filter(mod.category_predicate(cat_key)).get_column("card_id").to_list()
        )
        assert got == expected, f"{cat_key}: {got} != {expected}"


def test_selected_cards_satisfy_category_predicate_toy() -> None:
    """Every selected card's calls match its category definition (toy data)."""
    j = _toy_joined()
    selected = mod.select_cards(j, mod.CATEGORIES, seed=mod.GLOBAL_SEED)
    assert selected.height > 0
    for row in selected.iter_rows(named=True):
        _assert_card_matches_category(row, row["category"])


def test_selection_is_deterministic_toy() -> None:
    """Same seed yields an identical ordered card list (toy data)."""
    j = _toy_joined()
    a = mod.select_cards(j, mod.CATEGORIES, seed=mod.GLOBAL_SEED)
    b = mod.select_cards(j, mod.CATEGORIES, seed=mod.GLOBAL_SEED)
    assert a.select(["category", "panel_index", "card_id"]).to_dicts() == b.select(
        ["category", "panel_index", "card_id"]
    ).to_dicts()


def test_unknown_category_predicate_raises() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        mod.category_predicate("not_a_category")


# --- Real-data tests (skip if the gallery sources are absent) ---------------
@needs_data
def test_real_selected_cards_satisfy_category_predicate() -> None:
    """Every panel selected from the REAL data matches its category definition.

    This is the integrity guarantee: it ties each printed panel label back to
    the joined reader / detector / MedGemma source rows.
    """
    joined = mod.load_calls(READER_CSV, MANIFEST_CSV, MEDGEMMA_CSV)
    selected = mod.select_cards(joined, mod.CATEGORIES, seed=mod.GLOBAL_SEED)
    assert selected.height > 0
    # Cross-check each selected card against the freshly joined source row, not
    # just the cached selection columns, so a mismatch in the join would fail.
    src = {r["card_id"]: r for r in joined.iter_rows(named=True)}
    for row in selected.iter_rows(named=True):
        cid = row["card_id"]
        assert cid in src, f"selected card {cid} absent from joined source"
        s = src[cid]
        merged = {
            "reader_call": s["reader_call"],
            "detector_call": s["detector_call"],
            "medgemma_call": s["medgemma_call"],
            "medgemma_confidence": s["medgemma_conf"],
        }
        _assert_card_matches_category(merged, row["category"])
        # The card list's stored calls must equal the source calls.
        assert row["reader_call"] == s["reader_call"]
        assert row["detector_call"] == s["detector_call"]
        assert row["medgemma_call"] == s["medgemma_call"]


@needs_data
def test_real_selection_is_deterministic() -> None:
    """Same seed yields an identical ordered card list (real data)."""
    joined = mod.load_calls(READER_CSV, MANIFEST_CSV, MEDGEMMA_CSV)
    a = mod.select_cards(joined, mod.CATEGORIES, seed=mod.GLOBAL_SEED)
    b = mod.select_cards(joined, mod.CATEGORIES, seed=mod.GLOBAL_SEED)
    assert a.select(["category", "panel_index", "card_id"]).to_dicts() == b.select(
        ["category", "panel_index", "card_id"]
    ).to_dicts()


@needs_data
def test_real_pools_nonempty_and_reported() -> None:
    """Every category has a nonempty pool (so panels are not padded)."""
    joined = mod.load_calls(READER_CSV, MANIFEST_CSV, MEDGEMMA_CSV)
    sizes = mod.pool_sizes(joined, mod.CATEGORIES)
    for cat in mod.CATEGORIES:
        assert sizes[cat.key] > 0, f"empty pool for {cat.key}"
