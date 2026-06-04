"""Tests for scripts/45_prompt_sensitivity_concordance core logic.

The script is loaded as a module (it lives under ``scripts/``, not a package) so
its importable functions can be tested without shelling out or calling a model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_concordance_module():
    """Load scripts/45 as a module for direct function testing."""
    filename = "45_prompt_sensitivity_concordance.py"
    name = filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_discover_variant_logs_finds_finalized_parquets(tmp_path: Path) -> None:
    mod = _load_concordance_module()
    # scripts/42 finalizes each variant as ``<variant>.parquet`` and leaves
    # provenance/intermediate files in the same dir. Only the finalized parquet
    # stems are variants; the underscore-prefixed manifest/fingerprint JSONs,
    # the per-variant checkpoint CSV, and the finalized CSV must all be ignored.
    (tmp_path / "v_compact.parquet").touch()
    (tmp_path / "v_explicit.parquet").touch()
    (tmp_path / "v_compact.csv").touch()
    (tmp_path / "v_explicit.csv").touch()
    (tmp_path / "v_compact_checkpoint.csv").touch()
    (tmp_path / "_run_manifest_v_compact_20260530T010000Z.json").touch()
    (tmp_path / "_model_fingerprint_20260530T010000Z.json").touch()
    # An underscore-prefixed parquet (defensive guard) must also be skipped.
    (tmp_path / "_scratch.parquet").touch()

    found = mod.discover_variant_logs(tmp_path)

    assert set(found) == {"v_compact", "v_explicit"}
    assert found["v_compact"].name == "v_compact.parquet"
    assert found["v_explicit"].name == "v_explicit.parquet"


def test_discover_variant_logs_prefers_plain_over_timestamped(tmp_path: Path) -> None:
    mod = _load_concordance_module()
    # If a stray timestamped sibling lingers next to the finalized form, the
    # plain ``<variant>.parquet`` is preferred.
    (tmp_path / "v_compact.parquet").touch()
    (tmp_path / "v_compact_20260531T020000Z.parquet").touch()

    found = mod.discover_variant_logs(tmp_path)

    assert set(found) == {"v_compact", "v_compact_20260531T020000Z"}
    assert found["v_compact"].name == "v_compact.parquet"


def test_discover_variant_logs_empty_when_absent(tmp_path: Path) -> None:
    mod = _load_concordance_module()
    assert mod.discover_variant_logs(tmp_path / "missing") == {}
    assert mod.discover_variant_logs(tmp_path) == {}


# Full scripts/42 finalized run-log column order (its LOG_COLUMNS_WITH_IDS):
# subject_id, record_id, card_id, stratum, then RUN_LOG_COLUMNS, then
# model_weights_sha256. The concordance reader only consumes card_id, call, and
# parsed_ok, but the on-disk fixtures carry the full schema so the test exercises
# the real <variant>.parquet shape rather than a three-column fiction.
_SCRIPT42_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "record_id",
    "card_id",
    "stratum",
    "row_id",
    "mode",
    "model",
    "base_url",
    "temperature",
    "max_tokens",
    "seed",
    "prompt_sha256",
    "run_utc",
    "parsed_ok",
    "schema_complete",
    "parse_error",
    "phenotype",
    "vasopressor_use",
    "shock_state",
    "notes",
    "image_path",
    "image_sha256",
    "observed",
    "call",
    "confidence",
    "rationale",
    "raw_response",
    "model_weights_sha256",
)


def _script42_variant_frame(card_ids: list[str], calls: list[str]) -> pl.DataFrame:
    """Build a frame matching scripts/42's finalized <variant>.parquet schema."""
    n = len(card_ids)
    data: dict[str, list] = {
        "subject_id": [f"s{i}" for i in range(n)],
        "record_id": [f"rec{i}" for i in range(n)],
        "card_id": card_ids,
        "stratum": ["detector_positive"] * n,
        "row_id": card_ids,  # scripts/42 sets row_id = card_id for adjudicate rows
        "mode": ["adjudicate"] * n,
        "model": ["medgemma-1.5-4b-it-bf16"] * n,
        "base_url": ["http://localhost:8000/v1"] * n,
        "temperature": [0.0] * n,
        "max_tokens": [1536] * n,
        "seed": [20260426] * n,
        "prompt_sha256": ["deadbeef"] * n,
        "run_utc": ["2026-05-30T00:00:00+00:00"] * n,
        "parsed_ok": [True] * n,
        "schema_complete": [True] * n,
        "parse_error": [None] * n,
        "phenotype": [None] * n,
        "vasopressor_use": [None] * n,
        "shock_state": [None] * n,
        "notes": [None] * n,
        "image_path": [f"results/gallery/{c}.png" for c in card_ids],
        "image_sha256": ["cafe"] * n,
        "observed": ["a dip and recovery"] * n,
        "call": calls,
        "confidence": [0.9] * n,
        "rationale": ["looks like an occlusion"] * n,
        "raw_response": ["{}"] * n,
        "model_weights_sha256": ["abc123"] * n,
    }
    return pl.DataFrame(data).select(list(_SCRIPT42_COLUMNS))


def test_discover_and_score_script42_finalized_output(tmp_path: Path) -> None:
    mod = _load_concordance_module()
    # Two finalized scripts/42 variant parquets plus a decoy manifest JSON, an
    # underscore-prefixed parquet, and the per-variant checkpoint CSV. Discovery
    # must return exactly the two variants, and the discovered parquets must read
    # back with the columns compute_concordance needs (card_id, call, parsed_ok).
    _script42_variant_frame(
        ["c1", "c2", "c3", "c4"],
        [
            "occlusion_signature_present",
            "no_occlusion_signature",
            "no_occlusion_signature",
            "no_occlusion_signature",
        ],
    ).write_parquet(tmp_path / "v_compact.parquet")
    _script42_variant_frame(
        ["c1", "c2", "c3", "c4"],
        [
            "occlusion_signature_present",
            "no_occlusion_signature",
            "occlusion_signature_present",
            "no_occlusion_signature",
        ],
    ).write_parquet(tmp_path / "v_explicit.parquet")
    (tmp_path / "_run_manifest_v_compact_20260530T010000Z.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "_scratch.parquet").touch()
    (tmp_path / "v_compact_checkpoint.csv").touch()

    found = mod.discover_variant_logs(tmp_path)
    assert set(found) == {"v_compact", "v_explicit"}

    variant_frames = {v: pl.read_parquet(p) for v, p in found.items()}
    for frame in variant_frames.values():
        assert {"card_id", "call", "parsed_ok"}.issubset(frame.columns)

    result = mod.compute_concordance(variant_frames, _canonical(), _bridge())

    by_variant = {e["variant"]: e for e in result["per_variant"]}
    assert set(by_variant) == {"v_compact", "v_explicit"}
    # v_compact flips c3 (present -> absent): 3/4 match.
    assert by_variant["v_compact"]["n_matched"] == 3
    assert by_variant["v_compact"]["concordance_pct"] == 75.0
    # v_explicit agrees with the canonical run on all four: 4/4 match.
    assert by_variant["v_explicit"]["n_matched"] == 4
    assert by_variant["v_explicit"]["concordance_pct"] == 100.0


def test_main_exits_nonzero_without_variant_outputs(tmp_path: Path) -> None:
    mod = _load_concordance_module()
    rc = mod.main(
        [
            "--variant-dir",
            str(tmp_path / "no_variants"),
            "--canonical",
            str(tmp_path / "canon.parquet"),
        ]
    )
    assert rc == 2


def _bridge() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "card_id": ["c1", "c2", "c3", "c4"],
            "stratum": [
                "detector_positive",
                "detector_positive",
                "detector_negative",
                "detector_negative",
            ],
            "row_id": ["r1", "r2", "r3", "r4"],
        }
    )


def _canonical() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row_id": ["r1", "r2", "r3", "r4"],
            "call": [
                "occlusion_signature_present",
                "no_occlusion_signature",
                "occlusion_signature_present",
                "no_occlusion_signature",
            ],
            "parsed_ok": [True, True, True, True],
        }
    )


def test_compute_concordance_counts_matches() -> None:
    mod = _load_concordance_module()
    # Variant agrees on c1, c2, c4 and flips c3 (present -> absent): 3/4 match.
    variant = pl.DataFrame(
        {
            "card_id": ["c1", "c2", "c3", "c4"],
            "call": [
                "occlusion_signature_present",
                "no_occlusion_signature",
                "no_occlusion_signature",
                "no_occlusion_signature",
            ],
            "parsed_ok": [True, True, True, True],
        }
    )
    result = mod.compute_concordance({"v_x": variant}, _canonical(), _bridge())

    assert len(result["per_variant"]) == 1
    entry = result["per_variant"][0]
    assert entry["variant"] == "v_x"
    assert entry["n_compare"] == 4
    assert entry["n_matched"] == 3
    assert entry["concordance_pct"] == 75.0
    # c3 is variant-negative where canonical is positive.
    assert entry["var_neg_canon_pos"] == 1
    assert entry["var_pos_canon_neg"] == 0
    assert entry["other_mismatch"] == 0

    # Per-stratum: positive stratum (r1,r2) all match -> 2/2; negative stratum
    # (r3,r4) has the c3 flip -> 1/2.
    strata = {(r["variant"], r["stratum"]): r for r in result["per_stratum"]}
    assert strata[("v_x", "detector_positive")]["matched"] == 2
    assert strata[("v_x", "detector_negative")]["matched"] == 1

    # Canonical subsample distribution covers all four bridged cards.
    sub = result["canonical_in_subsample"]
    assert sub["n"] == 4
    assert sub["present"] == 2
    assert sub["absent"] == 2


def test_compute_concordance_excludes_unbridged_cards() -> None:
    mod = _load_concordance_module()
    # A variant card not present in the bridge must drop out of the comparison.
    variant = pl.DataFrame(
        {
            "card_id": ["c1", "c_orphan"],
            "call": ["occlusion_signature_present", "occlusion_signature_present"],
            "parsed_ok": [True, True],
        }
    )
    result = mod.compute_concordance({"v_x": variant}, _canonical(), _bridge())
    entry = result["per_variant"][0]
    # Only c1 is in both the variant frame and the bridge.
    assert entry["n_total"] == 1
    assert entry["n_compare"] == 1
    assert entry["n_matched"] == 1
