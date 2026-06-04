"""Tests for the resumable MedGemma adjudication batch driver."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_driver_module():
    """Import scripts/41_run_medgemma_adjudication.py as a module."""
    path = SCRIPTS_DIR / "41_run_medgemma_adjudication.py"
    spec = importlib.util.spec_from_file_location("medgemma_adjudication_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["medgemma_adjudication_driver"] = module
    spec.loader.exec_module(module)
    return module


def _row(module, row_id: str, *, model: str | None = None) -> dict:
    """Build a complete run-log row matching the driver's checkpoint schema."""
    return {col: None for col in module.LOG_COLUMNS_WITH_IDS} | {
        "subject_id": row_id.split("_")[0],
        "record_id": row_id.split("_")[1],
        "row_id": row_id,
        "mode": "adjudicate",
        "model": model or module.SERVED_MODEL_ID,
        "base_url": "http://localhost:8000/v1",
        "temperature": module.DEFAULT_TEMPERATURE,
        "max_tokens": module.MAX_TOKENS,
        "seed": module.GLOBAL_SEED,
        "prompt_sha256": "a" * 64,
        "run_utc": "2026-05-22T00:00:00+00:00",
        "parsed_ok": True,
        "schema_complete": True,
        "image_path": f"{row_id}.png",
        "image_sha256": "b" * 64,
        "observed": "stub trace",
        "call": "indeterminate",
        "confidence": 0.5,
        "rationale": "stubbed test row",
        "raw_response": '{"call":"indeterminate","confidence":0.5}',
    }


def test_checkpoint_append_and_read_roundtrip(tmp_path: Path):
    module = _load_driver_module()
    checkpoint = tmp_path / "checkpoint.csv"
    first = _row(module, "p0001_r0001_0")
    second = _row(module, "p0002_r0002_1")

    module._append_checkpoint_row(first, checkpoint)
    module._append_checkpoint_row(second, checkpoint)

    rows = module._read_checkpoint_rows(checkpoint)
    assert [row["row_id"] for row in rows] == ["p0001_r0001_0", "p0002_r0002_1"]
    assert rows[0]["raw_response"] == '{"call":"indeterminate","confidence":0.5}'


def test_checkpoint_compatibility_refuses_different_model(tmp_path: Path):
    module = _load_driver_module()
    checkpoint = tmp_path / "checkpoint.csv"
    module._append_checkpoint_row(_row(module, "p0001_r0001_0", model="old-model"), checkpoint)
    args = module._parse_args(["--stage", "full"])

    rows = module._read_checkpoint_rows(checkpoint)
    try:
        module._validate_checkpoint_compatible(
            rows,
            checkpoint_path=checkpoint,
            args=args,
            base_url="http://localhost:8000/v1",
        )
    except ValueError as exc:
        assert "old-model" in str(exc)
        assert "--no-resume" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("expected incompatible checkpoint to raise")


def test_main_resumes_from_checkpoint_and_skips_completed_rows(tmp_path: Path, monkeypatch):
    module = _load_driver_module()
    inventory = tmp_path / "inventory.csv"
    inventory.write_text(
        "\n".join(
            [
                "subject_id,record_id,reject_reason,is_occlusion_signature,"
                "phase3_duration_s,nbp_timestamp_s",
                "p0001,r0001,,false,0,10",
                "p0002,r0002,,false,0,20",
            ]
        ),
        encoding="utf-8",
    )
    wdb_root = tmp_path / "wdb"
    wdb_root.mkdir()
    out_dir = tmp_path / "out"
    checkpoint = out_dir / "checkpoint.csv"
    module._append_checkpoint_row(_row(module, "p0001_r0001_0"), checkpoint)

    calls: list[str] = []

    class _Client:
        model = module.SERVED_MODEL_ID

    def _fake_build_client(_args):
        del _args
        return _Client()

    def _fake_adjudicate_one(
        client,
        record,
        *,
        wdb_root,
        scratch_dir,
        base_url,
        args,
        fingerprint,
        gallery_lookup=None,
    ):
        del client, wdb_root, scratch_dir, base_url, args, fingerprint, gallery_lookup
        calls.append(record["row_id"])
        return _row(module, record["row_id"])

    monkeypatch.setattr(module, "_build_client", _fake_build_client)
    monkeypatch.setattr(module, "_adjudicate_one", _fake_adjudicate_one)

    code = module.main(
        [
            "--stage",
            "full",
            "--inventory",
            str(inventory),
            "--wdb-root",
            str(wdb_root),
            "--out",
            str(out_dir),
            "--checkpoint-csv",
            str(checkpoint),
        ]
    )

    assert code == 0
    assert calls == ["p0002_r0002_1"]
    checkpoint_df = pl.read_csv(checkpoint)
    assert checkpoint_df["row_id"].to_list() == ["p0001_r0001_0", "p0002_r0002_1"]
    final_df = pl.read_csv(next(out_dir.glob("medgemma_adjudication_full_*.csv")))
    assert final_df["row_id"].to_list() == ["p0001_r0001_0", "p0002_r0002_1"]
    # Per-run manifest is written as well.
    manifests = list(out_dir.glob("_run_manifest_*.json"))
    assert len(manifests) == 1


def test_write_run_log_handles_late_string_in_mostly_none_column(tmp_path: Path):
    """Regression: polars must not lock parse_error to Null dtype on first 100 rows.

    In the canonical full run on 2026-05-29, ``_write_run_log`` crashed after
    8,914 successful adjudications because polars inferred the
    ``parse_error`` column dtype from only the first 100 rows (all None) and
    then could not append the string ``"pleth mostly nan"`` from a later
    parse-failure row. The fix is ``infer_schema_length=None`` so the full
    column is scanned for type inference. This test reproduces that pattern:
    150 rows with ``parse_error=None`` followed by 1 row with a string
    ``parse_error``, then more None rows. Without the patch the
    ``pl.DataFrame(rows)`` call raises ``ComputeError``.
    """
    module = _load_driver_module()
    rows: list[dict] = []
    # 150 successful adjudications.
    for i in range(150):
        rows.append(_row(module, f"p{i:04d}_r{i:04d}_{i}"))
    # One parse-failure row whose parse_error is a string. This is the row
    # that crashed the canonical run; reproduce the exact value here.
    bad = _row(module, "p0150_r0150_150")
    bad["parsed_ok"] = False
    bad["schema_complete"] = False
    bad["call"] = None
    bad["confidence"] = None
    bad["rationale"] = None
    bad["observed"] = None
    bad["raw_response"] = ""
    bad["parse_error"] = "pleth mostly nan"
    rows.append(bad)
    # 49 more successful rows after the failure to stress the builder.
    for i in range(151, 200):
        rows.append(_row(module, f"p{i:04d}_r{i:04d}_{i}"))

    csv_path, parquet_path = module._write_run_log(
        rows, tmp_path, stage="test", stamp="20260530T000000Z"
    )

    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert parquet_path.exists() and parquet_path.stat().st_size > 0

    csv_df = pl.read_csv(csv_path, infer_schema_length=20000)
    parquet_df = pl.read_parquet(parquet_path)
    assert csv_df.height == 200
    assert parquet_df.height == 200

    csv_parse_error = csv_df["parse_error"].to_list()
    assert csv_parse_error[150] == "pleth mostly nan"
    other_indices = [i for i in range(200) if i != 150]
    assert all(csv_parse_error[i] is None for i in other_indices)

    parquet_parse_error = parquet_df["parse_error"].to_list()
    assert parquet_parse_error[150] == "pleth mostly nan"
    assert all(parquet_parse_error[i] is None for i in other_indices)


def test_resume_only_manifest_uses_checkpoint_timestamps(
    tmp_path: Path, monkeypatch
):
    """Regression: a resume-only invocation must report the original run window.

    In the canonical 2026-05-29 run, ``_write_run_log`` crashed but the
    checkpoint was complete. A recovery invocation (resume from an
    already-drained pool) previously stamped the manifest with the recovery
    wall clock at start and end. For a public-bound, provenance-sensitive
    manifest that is a lie. ``_resume_only_manifest_window`` now derives the
    window from the checkpoint rows' ``run_utc`` values; this test verifies
    the manifest reflects the synthetic checkpoint window, not the recovery
    invocation time.
    """
    module = _load_driver_module()

    # Build a synthetic checkpoint spanning a known window.
    checkpoint = tmp_path / "out" / "checkpoint.csv"
    expected_start = "2026-05-29T01:34:02+00:00"
    expected_end = "2026-05-30T03:09:56+00:00"
    # 10 rows whose run_utc values are interpolated between start and end so
    # min/max recover the bounds exactly.
    start_dt = dt.datetime.fromisoformat(expected_start)
    end_dt = dt.datetime.fromisoformat(expected_end)
    n_rows = 10
    span_seconds = int((end_dt - start_dt).total_seconds())
    # Distribute the rows across the window, with the first row exactly at
    # ``expected_start`` and the last row exactly at ``expected_end``. The
    # driver writes second-precision ISO strings; match that here so the
    # manifest's min/max recover the bounds character-for-character.
    for i in range(n_rows):
        row = _row(module, f"p{i:04d}_r{i:04d}_{i}")
        offset_s = round(span_seconds * i / (n_rows - 1))
        row["run_utc"] = (
            (start_dt + dt.timedelta(seconds=offset_s))
            .replace(microsecond=0)
            .isoformat()
        )
        module._append_checkpoint_row(row, checkpoint)

    # Inventory whose row_ids match the checkpoint, so select_pool returns
    # exactly the rows already in the checkpoint and the live pool is drained.
    inventory = tmp_path / "inventory.csv"
    inventory_lines = [
        "subject_id,record_id,reject_reason,is_occlusion_signature,"
        "phase3_duration_s,nbp_timestamp_s"
    ]
    for i in range(n_rows):
        inventory_lines.append(f"p{i:04d},r{i:04d},,false,0,{10 + i}")
    inventory.write_text("\n".join(inventory_lines), encoding="utf-8")

    wdb_root = tmp_path / "wdb"
    wdb_root.mkdir()
    out_dir = tmp_path / "out"

    # The resume-only path must NOT contact the client or render anything.
    # Guard against accidental calls by injecting sentinels that raise.
    def _forbidden_build_client(_args):
        del _args
        raise AssertionError("resume-only path must not build a client")

    def _forbidden_adjudicate_one(*_args, **_kwargs):
        del _args, _kwargs
        raise AssertionError("resume-only path must not call _adjudicate_one")

    monkeypatch.setattr(module, "_build_client", _forbidden_build_client)
    monkeypatch.setattr(module, "_adjudicate_one", _forbidden_adjudicate_one)

    code = module.main(
        [
            "--stage",
            "full",
            "--inventory",
            str(inventory),
            "--wdb-root",
            str(wdb_root),
            "--out",
            str(out_dir),
            "--checkpoint-csv",
            str(checkpoint),
        ]
    )

    assert code == 0
    manifests = list(out_dir.glob("_run_manifest_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["run_utc_start"] == expected_start
    assert manifest["run_utc_end"] == expected_end
    assert manifest["n_rows_total"] == n_rows


def test_resume_only_manifest_window_helper_falls_back_when_run_utc_missing():
    """If a row lacks run_utc, fall back to the invocation window with a warning.

    Defensive: the canonical checkpoint always carries run_utc, but the
    fallback path must still produce a valid window rather than crash on a
    malformed checkpoint.
    """
    module = _load_driver_module()
    rows = [
        {"run_utc": "2026-05-29T01:34:02+00:00"},
        {"run_utc": None},
    ]
    fallback_start = "2026-05-31T00:00:00+00:00"
    start, end = module._resume_only_manifest_window(
        rows, fallback_start=fallback_start
    )
    assert start == fallback_start
    # End should be a valid ISO-8601 UTC string (the invocation wall clock).
    assert dt.datetime.fromisoformat(end).tzinfo is not None
