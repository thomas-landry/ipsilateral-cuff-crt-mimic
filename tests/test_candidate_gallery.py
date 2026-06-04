"""Tests for the candidate-gallery sampling and manifest pipeline.

These tests skip the actual PNG render (which needs the credentialed WDB
tree) and exercise the deterministic, in-memory pieces: card-id stability,
stratum predicates, deterministic sampling, overlap subsetting, and the
no-render (validate-only) main path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

from cuffcrt._seed import GLOBAL_SEED

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_module():
    """Import scripts/51_candidate_gallery.py as a module (numeric filename)."""
    path = SCRIPTS_DIR / "51_candidate_gallery.py"
    spec = importlib.util.spec_from_file_location("candidate_gallery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_gallery"] = module
    spec.loader.exec_module(module)
    return module


def _toy_inventory() -> pl.DataFrame:
    """Build a small in-memory event table covering each stratum."""
    rows = [
        # Stratum A: detector positive (is_occlusion_signature == True)
        {
            "subject_id": "p0001",
            "record_id": "r0001",
            "nbp_timestamp_s": 100.0,
            "is_occlusion_signature": True,
            "phase3_duration_s": 20.0,
            "nadir_depth_frac": 0.05,
            "alignment_offset_s": -10.0,
            "reject_reason": None,
            "pleth_valid_fraction": 0.99,
        },
        {
            "subject_id": "p0002",
            "record_id": "r0002",
            "nbp_timestamp_s": 200.0,
            "is_occlusion_signature": True,
            "phase3_duration_s": 18.0,
            "nadir_depth_frac": 0.10,
            "alignment_offset_s": -5.0,
            "reject_reason": None,
            "pleth_valid_fraction": 0.95,
        },
        # Stratum B: near-miss (phase3 >= 10 AND nadir < 0.2 AND failed recovery
        # or stat_mode_short_phase3)
        {
            "subject_id": "p0003",
            "record_id": "r0003",
            "nbp_timestamp_s": 300.0,
            "is_occlusion_signature": False,
            "phase3_duration_s": 12.0,
            "nadir_depth_frac": 0.15,
            "alignment_offset_s": -8.0,
            "reject_reason": "stat_mode_short_phase3",
            "pleth_valid_fraction": 0.92,
        },
        {
            "subject_id": "p0004",
            "record_id": "r0004",
            "nbp_timestamp_s": 400.0,
            "is_occlusion_signature": False,
            "phase3_duration_s": 25.0,
            "nadir_depth_frac": 0.18,
            "alignment_offset_s": -15.0,
            "reject_reason": "no_recovery_in_window",
            "pleth_valid_fraction": 0.88,
        },
        # Stratum C: detector negative (no envelope) + renderable
        {
            "subject_id": "p0005",
            "record_id": "r0005",
            "nbp_timestamp_s": 500.0,
            "is_occlusion_signature": False,
            "phase3_duration_s": 0.0,
            "nadir_depth_frac": 0.9,
            "alignment_offset_s": float("nan"),
            "reject_reason": "no_phase2",
            "pleth_valid_fraction": 0.95,
        },
        {
            "subject_id": "p0006",
            "record_id": "r0006",
            "nbp_timestamp_s": 600.0,
            "is_occlusion_signature": False,
            "phase3_duration_s": 0.0,
            "nadir_depth_frac": 0.85,
            "alignment_offset_s": float("nan"),
            "reject_reason": "pre_window_unstable",
            "pleth_valid_fraction": 0.80,
        },
        # Excluded: no PPG, not renderable
        {
            "subject_id": "p0007",
            "record_id": "r0007",
            "nbp_timestamp_s": 700.0,
            "is_occlusion_signature": False,
            "phase3_duration_s": 0.0,
            "nadir_depth_frac": 0.7,
            "alignment_offset_s": float("nan"),
            "reject_reason": "no_pleth",
            "pleth_valid_fraction": 0.10,
        },
        # Excluded from stratum B: passes thresholds but rejected for the
        # alignment-window reason (which lacks measured phase3/nadir in real
        # events).  Here the predicate falls through because the reject_reason
        # is not in the near-miss list.
        {
            "subject_id": "p0008",
            "record_id": "r0008",
            "nbp_timestamp_s": 800.0,
            "is_occlusion_signature": False,
            "phase3_duration_s": 11.0,
            "nadir_depth_frac": 0.15,
            "alignment_offset_s": -70.0,
            "reject_reason": "no_aligned_occlusion",
            "pleth_valid_fraction": 0.93,
        },
    ]
    return pl.DataFrame(rows)


def test_define_strata_predicates_match_expected_rows():
    module = _load_module()
    inv = _toy_inventory()
    pools = module.define_strata(inv)
    assert pools[module.STRATUM_A].get_column("subject_id").to_list() == ["p0001", "p0002"]
    assert pools[module.STRATUM_B].get_column("subject_id").to_list() == ["p0003", "p0004"]
    assert pools[module.STRATUM_C].get_column("subject_id").to_list() == ["p0005", "p0006"]


def test_sample_stratum_returns_all_when_under_target():
    module = _load_module()
    pool = _toy_inventory().head(2)
    sampled, shortfall = module.sample_stratum(pool, target=10, seed=GLOBAL_SEED)
    assert sampled.height == 2
    assert shortfall == 8


def test_sample_stratum_is_deterministic_at_target():
    module = _load_module()
    rng = np.random.default_rng(0)
    pool = pl.DataFrame(
        {
            "subject_id": [f"p{i:04d}" for i in range(20)],
            "record_id": [f"r{i:04d}" for i in range(20)],
            "nbp_timestamp_s": rng.uniform(0, 1000, size=20).tolist(),
            "is_occlusion_signature": [False] * 20,
            "phase3_duration_s": [0.0] * 20,
            "nadir_depth_frac": [0.9] * 20,
            "alignment_offset_s": [float("nan")] * 20,
            "reject_reason": ["no_phase2"] * 20,
            "pleth_valid_fraction": [0.9] * 20,
        }
    )
    a, _ = module.sample_stratum(pool, target=5, seed=GLOBAL_SEED)
    b, _ = module.sample_stratum(pool, target=5, seed=GLOBAL_SEED)
    assert a.get_column("subject_id").to_list() == b.get_column("subject_id").to_list()


def test_compute_card_id_is_deterministic_and_stratum_prefixed():
    module = _load_module()
    cid_a = module.compute_card_id("p0001", 12.0, module.STRATUM_A, GLOBAL_SEED)
    cid_a_again = module.compute_card_id("p0001", 12.0, module.STRATUM_A, GLOBAL_SEED)
    cid_b = module.compute_card_id("p0001", 12.0, module.STRATUM_B, GLOBAL_SEED)
    assert cid_a == cid_a_again
    assert cid_a.startswith("A-")
    assert cid_b.startswith("B-")
    assert cid_a != cid_b


def test_compute_card_id_changes_with_inputs():
    module = _load_module()
    cid1 = module.compute_card_id("p0001", 12.0, "detector_positive", GLOBAL_SEED)
    cid2 = module.compute_card_id("p0001", 13.0, "detector_positive", GLOBAL_SEED)
    cid3 = module.compute_card_id("p0002", 12.0, "detector_positive", GLOBAL_SEED)
    assert cid1 != cid2 != cid3


def test_validate_only_main_writes_manifest_and_forms(tmp_path: Path):
    """Without --smoke or --full-render the pipeline writes manifests but no PNGs."""
    module = _load_module()
    # Write the toy inventory as a single events_p0000.parquet under events_dir.
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    _toy_inventory().write_parquet(events_dir / "events_p0000.parquet")

    out_dir = tmp_path / "gallery"
    code = module.main(
        [
            "--events-dir",
            str(events_dir),
            "--out",
            str(out_dir),
            "--overlap-size",
            "3",
        ]
    )
    assert code == 0
    manifest_path = out_dir / "gallery_manifest.csv"
    blinded_path = out_dir / "reader_form_blinded.csv"
    overlap_path = out_dir / "reader_form_overlap.csv"
    sampling_log = out_dir / "sampling_log.md"
    assert manifest_path.exists()
    assert blinded_path.exists()
    assert overlap_path.exists()
    assert sampling_log.exists()

    manifest_df = pl.read_csv(manifest_path)
    # 2 (A) + 2 (B) + 2 (C) = 6 manifest rows, all unrendered (no SHA).
    assert manifest_df.height == 6
    # All image_sha256 are null in validate-only mode.
    assert manifest_df.get_column("image_sha256").null_count() == 6
    # Reader form columns match the spec.
    reader_df = pl.read_csv(blinded_path)
    assert reader_df.columns == ["card_id", "image_path", "call", "notes"]
    # Overlap is bounded by manifest height.
    overlap_df = pl.read_csv(overlap_path)
    assert overlap_df.height == 3


def test_validate_only_fails_when_events_dir_missing(tmp_path: Path):
    module = _load_module()
    code = module.main(
        [
            "--events-dir",
            str(tmp_path / "does_not_exist"),
            "--out",
            str(tmp_path / "gallery"),
        ]
    )
    assert code == 2


def test_sampling_log_is_append_only(tmp_path: Path):
    """Running validate-only twice appends a second block to the sampling log."""
    module = _load_module()
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    _toy_inventory().write_parquet(events_dir / "events_p0000.parquet")
    out_dir = tmp_path / "gallery"
    for _ in range(2):
        code = module.main(
            [
                "--events-dir",
                str(events_dir),
                "--out",
                str(out_dir),
                "--overlap-size",
                "2",
            ]
        )
        assert code == 0
    text = (out_dir / "sampling_log.md").read_text(encoding="utf-8")
    assert text.count("## Run at ") == 2
