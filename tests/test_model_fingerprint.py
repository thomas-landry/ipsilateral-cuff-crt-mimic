"""Tests for the model-fingerprint loader.

The fingerprint JSON is produced out-of-band by
``scripts/compute_model_sha.sh``; the harness only reads it. These tests cover
the missing-directory, missing-files, malformed-JSON, and happy-path branches
without invoking the shell script.
"""

from __future__ import annotations

import json
from pathlib import Path

from cuffcrt.llm.model_fingerprint import (
    FINGERPRINT_GLOB,
    ModelFingerprint,
    load_latest_fingerprint,
)


def _write_fingerprint(directory: Path, stamp: str, **overrides) -> Path:
    """Write a minimal fingerprint JSON to ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": overrides.get("model_id", "medgemma-1.5-4b-it-bf16"),
        "model_dir": overrides.get("model_dir", "/tmp/weights"),
        "model_weights_sha256": overrides.get("model_weights_sha256", "f" * 64),
        "files_hashed": overrides.get(
            "files_hashed",
            [{"path": "weights.safetensors", "size_bytes": 1, "sha256": "f" * 64}],
        ),
        "computed_utc": overrides.get("computed_utc", "2026-05-22T00:00:00+00:00"),
    }
    path = directory / f"_model_fingerprint_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_missing_directory_returns_none(tmp_path: Path):
    fingerprint = load_latest_fingerprint(tmp_path / "nope")
    assert fingerprint is None


def test_directory_with_no_fingerprints_returns_none(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    assert load_latest_fingerprint(tmp_path) is None


def test_loads_latest_when_multiple_present(tmp_path: Path):
    _write_fingerprint(tmp_path, "20260101T000000Z", model_weights_sha256="a" * 64)
    _write_fingerprint(tmp_path, "20260522T000000Z", model_weights_sha256="b" * 64)
    fingerprint = load_latest_fingerprint(tmp_path)
    assert isinstance(fingerprint, ModelFingerprint)
    assert fingerprint.model_weights_sha256 == "b" * 64
    assert fingerprint.files_hashed == 1


def test_malformed_json_returns_none(tmp_path: Path):
    path = tmp_path / "_model_fingerprint_20260522T000000Z.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_latest_fingerprint(tmp_path) is None


def test_fingerprint_glob_constant():
    """The glob must match files written by the shell helper."""
    assert FINGERPRINT_GLOB == "_model_fingerprint_*.json"


def test_fingerprint_with_no_files_hashed_field(tmp_path: Path):
    """A fingerprint missing ``files_hashed`` still loads with files_hashed=0."""
    path = tmp_path / "_model_fingerprint_20260522T000000Z.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "m",
                "model_dir": "/tmp",
                "model_weights_sha256": "c" * 64,
                "computed_utc": "2026-05-22T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fingerprint = load_latest_fingerprint(tmp_path)
    assert fingerprint is not None
    assert fingerprint.files_hashed == 0
    assert fingerprint.model_weights_sha256 == "c" * 64
