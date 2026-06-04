"""Tests for the scripts/41 ``--gallery-dir`` pre-rendered-PNG read path.

The ``--gallery-dir`` read path makes the MedGemma harness
read the same pre-rendered gallery PNGs the human reader adjudicated, so reader
and model view pixel-identical images. These tests exercise the pure helpers
that resolve a gallery PNG for an inventory row and the missing-PNG policy,
plus the backward-compatibility guarantee that no ``--gallery-dir`` flag leaves
the on-the-fly render path untouched.

The script filename begins with a digit, so it is imported via importlib from
its path rather than a normal module import. Nothing here shells out or touches
a live model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "41_run_medgemma_adjudication.py"
)
_spec = importlib.util.spec_from_file_location("_script41_gallery", _SCRIPT_PATH)
assert _spec is not None
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


# Two tiny PNG payloads with distinct bytes so a byte-for-byte assertion is
# meaningful. These are not valid renders; the resolver only moves bytes.
_PNG_A = b"\x89PNG\r\n\x1a\n-card-A-distinct-bytes"
_PNG_B = b"\x89PNG\r\n\x1a\n-card-B-different-bytes"


def _make_gallery(tmp_path: Path) -> Path:
    """Build a minimal on-disk gallery: two PNGs plus a gallery_manifest.csv.

    The manifest schema mirrors the real ``gallery_manifest.csv`` so the
    resolver is exercised against the production join columns
    ``(subject_id, record_id, t_nbp)`` and the per-stratum PNG layout.
    """
    gallery = tmp_path / "gallery"
    (gallery / "detector_positive").mkdir(parents=True)
    (gallery / "detector_negative_random").mkdir(parents=True)

    png_a = gallery / "detector_positive" / "A-aaaa000011112222.png"
    png_b = gallery / "detector_negative_random" / "C-cccc000011112222.png"
    png_a.write_bytes(_PNG_A)
    png_b.write_bytes(_PNG_B)

    manifest = pl.DataFrame(
        {
            "card_id": ["A-aaaa000011112222", "C-cccc000011112222"],
            "stratum": ["detector_positive", "detector_negative_random"],
            "subject_id": ["p10014354", "p10014354"],
            "record_id": ["81739927", "81739927"],
            "t_nbp": [1234.5, 5678.25],
            "image_path": [
                "results/gallery/detector_positive/A-aaaa000011112222.png",
                "results/gallery/detector_negative_random/C-cccc000011112222.png",
            ],
            "image_sha256": ["deadbeef", "feedface"],
        }
    )
    manifest.write_csv(gallery / "gallery_manifest.csv")
    return gallery


def test_build_gallery_lookup_keys_on_triple(tmp_path: Path) -> None:
    """The lookup maps the (subject, record, t_nbp) triple to an on-disk PNG."""
    gallery = _make_gallery(tmp_path)
    lookup = _mod.build_gallery_lookup(gallery)
    assert len(lookup) == 2
    path_a = _mod.resolve_gallery_png(lookup, "p10014354", "81739927", 1234.5)
    assert path_a is not None
    assert path_a.read_bytes() == _PNG_A
    path_b = _mod.resolve_gallery_png(lookup, "p10014354", "81739927", 5678.25)
    assert path_b is not None
    assert path_b.read_bytes() == _PNG_B


def test_resolve_gallery_png_tolerates_float_formatting(tmp_path: Path) -> None:
    """A t_nbp that differs only in trailing-float noise still resolves."""
    gallery = _make_gallery(tmp_path)
    lookup = _mod.build_gallery_lookup(gallery)
    # 1234.5 stored; query 1234.5000004 (sub-millisecond) must still match.
    path = _mod.resolve_gallery_png(lookup, "p10014354", "81739927", 1234.5000004)
    assert path is not None
    assert path.read_bytes() == _PNG_A


def test_resolve_gallery_png_returns_none_when_absent(tmp_path: Path) -> None:
    """A triple with no pre-rendered card resolves to None (caller decides)."""
    gallery = _make_gallery(tmp_path)
    lookup = _mod.build_gallery_lookup(gallery)
    assert _mod.resolve_gallery_png(lookup, "p99999999", "00000000", 1.0) is None


def test_build_gallery_lookup_missing_manifest_raises(tmp_path: Path) -> None:
    """An empty directory with no manifest is a clear, early error."""
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _mod.build_gallery_lookup(empty)


def test_parse_args_gallery_flags_default_off() -> None:
    """Without the flag, gallery_dir is None and the policy default is recorded.

    This is the backward-compatibility guarantee: the canonical run used no
    ``--gallery-dir``, so its absence must leave the namespace in the
    on-the-fly-render state.
    """
    args = _mod._parse_args([])
    assert args.gallery_dir is None
    assert args.gallery_missing == "fallback"


def test_parse_args_gallery_flags_parsed(tmp_path: Path) -> None:
    """The flag and its policy parse into the namespace."""
    args = _mod._parse_args(
        ["--gallery-dir", str(tmp_path), "--gallery-missing", "skip"]
    )
    assert args.gallery_dir == tmp_path
    assert args.gallery_missing == "skip"


def _stub_client():
    """Return an in-process stub client (no server, no network)."""
    from cuffcrt.llm.client import StubClient

    return StubClient(model="medgemma-1.5-4b-it-bf16", base_url="http://localhost:8000/v1")


def test_adjudicate_one_reads_exact_gallery_png_bytes(tmp_path: Path) -> None:
    """A row with a pre-rendered card is adjudicated on those exact PNG bytes.

    The run-log row must record the gallery PNG path and the SHA-256 of the
    pre-rendered bytes, never a freshly rendered scratch image. This is the
    core gallery-read guarantee: model and reader see the same pixels.
    """
    import hashlib

    gallery = _make_gallery(tmp_path)
    lookup = _mod.build_gallery_lookup(gallery)
    args = _mod._parse_args(["--gallery-dir", str(gallery)])
    row = {
        "subject_id": "p10014354",
        "record_id": "81739927",
        "row_id": "p10014354_81739927_0",
        "nbp_timestamp_s": 1234.5,
    }
    log_row = _mod._adjudicate_one(
        _stub_client(),
        row,
        wdb_root=tmp_path / "nonexistent_wdb",  # must never be touched
        scratch_dir=tmp_path / "scratch",
        base_url="http://localhost:8000/v1",
        args=args,
        fingerprint=None,
        gallery_lookup=lookup,
    )
    assert log_row["parsed_ok"] is True
    assert log_row["image_sha256"] == hashlib.sha256(_PNG_A).hexdigest()
    assert "A-aaaa000011112222.png" in log_row["image_path"]
    # No scratch PNG was rendered (the WDB path was never read).
    assert not (tmp_path / "scratch").exists()


def test_adjudicate_one_skip_policy_records_uncallable(tmp_path: Path) -> None:
    """With --gallery-missing skip, a row lacking a card is uncallable."""
    gallery = _make_gallery(tmp_path)
    lookup = _mod.build_gallery_lookup(gallery)
    args = _mod._parse_args(["--gallery-dir", str(gallery), "--gallery-missing", "skip"])
    row = {
        "subject_id": "p99999999",
        "record_id": "00000000",
        "row_id": "p99999999_00000000_0",
        "nbp_timestamp_s": 42.0,
    }
    log_row = _mod._adjudicate_one(
        _stub_client(),
        row,
        wdb_root=tmp_path / "nonexistent_wdb",
        scratch_dir=tmp_path / "scratch",
        base_url="http://localhost:8000/v1",
        args=args,
        fingerprint=None,
        gallery_lookup=lookup,
    )
    assert log_row["parsed_ok"] is False
    assert log_row["call"] is None
    assert log_row["parse_error"] == "gallery png missing (skipped)"


def test_adjudicate_one_no_gallery_lookup_is_backward_compatible(tmp_path: Path) -> None:
    """Without a gallery lookup the on-the-fly path runs (and fails cleanly here).

    With no WDB tree present, the on-the-fly path returns a parse-failed row
    citing the missing master header. The point is that the gallery branch is
    never entered, so the canonical-run behavior is byte-for-byte preserved.
    """
    args = _mod._parse_args([])
    assert args.gallery_dir is None
    row = {
        "subject_id": "p10014354",
        "record_id": "81739927",
        "row_id": "p10014354_81739927_0",
        "nbp_timestamp_s": 1234.5,
    }
    log_row = _mod._adjudicate_one(
        _stub_client(),
        row,
        wdb_root=tmp_path / "nonexistent_wdb",
        scratch_dir=tmp_path / "scratch",
        base_url="http://localhost:8000/v1",
        args=args,
        fingerprint=None,
        gallery_lookup=None,
    )
    assert log_row["parsed_ok"] is False
    assert "master header missing" in log_row["parse_error"]
