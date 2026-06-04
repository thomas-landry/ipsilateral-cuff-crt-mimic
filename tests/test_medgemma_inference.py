"""Tests for the MedGemma (oMLX) inference harness.

These exercise the harness with no network and no server: response parsing
(including malformed and fenced output), prompt-SHA stability, and the
``--dry-run`` run log shape and contents. The in-process
:class:`~cuffcrt.llm.client.StubClient` stands in for the model everywhere.

Central guarantees under test:

- No raw note text ever lands in the run log (DUA). The note appears only via
  its prompt SHA-256.
- The canonical adjudication vocabulary is ``occlusion_signature_present``,
  ``no_occlusion_signature``, ``indeterminate``. The legacy values
  ``ipsilateral`` and ``not_ipsilateral`` are rejected at parse time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.llm.client import StubClient
from cuffcrt.llm.medgemma import (
    DEFAULT_MAX_TOKENS,
    RUN_LOG_COLUMNS,
    adjudicate_image,
    build_adjudicate_messages,
    build_extract_messages,
    extract_text,
    parse_adjudication_json,
    parse_extraction_json,
    prompt_sha256,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

# A 1x1 PNG, enough bytes for the image-path code without a plotting dependency.
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f6e0000000049454e44ae42"
    "6082"
)


def _load_script_module():
    """Import scripts/40_medgemma_inference.py as a module (numeric filename)."""
    path = SCRIPTS_DIR / "40_medgemma_inference.py"
    spec = importlib.util.spec_from_file_location("medgemma_inference_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["medgemma_inference_cli"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# JSON parsing: extraction
# --------------------------------------------------------------------------- #


def test_parse_extraction_clean_json():
    raw = (
        '{"phenotype": "septic shock", "vasopressor_use": true, '
        '"shock_state": "vasodilatory", "notes": "on norepinephrine"}'
    )
    result = parse_extraction_json(raw)
    assert result.parsed_ok is True
    assert result.phenotype == "septic shock"
    assert result.vasopressor_use is True
    assert result.shock_state == "vasodilatory"
    assert result.parse_error is None


def test_parse_extraction_code_fenced():
    raw = '```json\n{"phenotype": "stable", "vasopressor_use": false}\n```'
    result = parse_extraction_json(raw)
    assert result.parsed_ok is True
    assert result.phenotype == "stable"
    assert result.vasopressor_use is False
    # Missing keys default to None, not an error.
    assert result.shock_state is None


def test_parse_extraction_prose_around_object():
    raw = 'Sure, here is the JSON: {"phenotype": "x"} hope that helps!'
    result = parse_extraction_json(raw)
    assert result.parsed_ok is True
    assert result.phenotype == "x"


def test_parse_extraction_null_like_tokens_become_none():
    raw = '{"phenotype": "none", "vasopressor_use": "unknown", "shock_state": ""}'
    result = parse_extraction_json(raw)
    assert result.parsed_ok is True
    assert result.phenotype is None
    assert result.vasopressor_use is None
    assert result.shock_state is None


def test_parse_extraction_malformed_returns_not_ok():
    raw = "the model refused and wrote a paragraph with no json"
    result = parse_extraction_json(raw)
    assert result.parsed_ok is False
    assert result.parse_error is not None
    assert result.raw_response == raw


def test_parse_extraction_empty_string():
    result = parse_extraction_json("")
    assert result.parsed_ok is False
    assert result.raw_response == ""


# --------------------------------------------------------------------------- #
# JSON parsing: adjudication
# --------------------------------------------------------------------------- #


def test_parse_adjudication_clean_json():
    raw = (
        '{"observed": "deep dip then graded recovery", '
        '"call": "occlusion_signature_present", '
        '"confidence": 0.82, "rationale": "clear dip and recovery"}'
    )
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is True
    assert result.schema_complete is True
    assert result.call == "occlusion_signature_present"
    assert result.observed == "deep dip then graded recovery"
    assert result.confidence == 0.82
    assert result.rationale == "clear dip and recovery"


def test_parse_adjudication_normalizes_case_and_clamps_confidence():
    raw = (
        '{"observed": "flat trace", "call": "No_Occlusion_Signature", '
        '"confidence": 1.7, "rationale": "flat"}'
    )
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is True
    assert result.call == "no_occlusion_signature"
    assert result.confidence == 1.0  # clamped into [0, 1]


def test_parse_adjudication_invalid_call_is_not_ok():
    raw = '{"call": "maybe", "confidence": 0.5}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is False
    assert result.schema_complete is False
    assert result.call is None
    assert "invalid or missing call" in result.parse_error


def test_parse_adjudication_legacy_ipsilateral_is_refused():
    """A stale ``ipsilateral`` call from a v1 prompt must not parse silently."""
    raw = '{"call": "ipsilateral", "confidence": 0.9, "rationale": "looks ipsi"}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is False
    assert result.schema_complete is False
    assert result.call is None
    assert "legacy_call_value" in result.parse_error
    assert "ipsilateral" in result.parse_error


def test_parse_adjudication_legacy_not_ipsilateral_is_refused():
    raw = '{"call": "not_ipsilateral", "confidence": 0.4}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is False
    assert "legacy_call_value" in result.parse_error
    assert "not_ipsilateral" in result.parse_error


def test_parse_adjudication_non_numeric_confidence_is_not_ok():
    raw = '{"call": "indeterminate", "confidence": "high"}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is False
    assert result.call == "indeterminate"
    assert "non-numeric confidence" in result.parse_error


def test_parse_adjudication_fenced_and_prose():
    raw = (
        'Here is my call:\n```\n{"call": "indeterminate", "confidence": 0.3}\n```\nDone.'
    )
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is True
    assert result.call == "indeterminate"


def test_parse_adjudication_schema_complete_is_false_when_keys_missing():
    """A call parses without all four keys, but ``schema_complete`` is False."""
    raw = '{"call": "occlusion_signature_present", "confidence": 0.7}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is True
    assert result.schema_complete is False
    assert result.observed is None
    assert result.rationale is None


def test_parse_adjudication_missing_confidence_is_not_ok():
    # confidence key absent -> obj.get returns None -> guarded, no crash.
    raw = '{"call": "occlusion_signature_present", "rationale": "clear dip"}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is False
    assert result.confidence is None
    assert "non-numeric confidence" in result.parse_error


def test_parse_adjudication_explicit_null_confidence_is_not_ok():
    raw = '{"call": "occlusion_signature_present", "confidence": null}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is False
    assert result.confidence is None
    assert "non-numeric confidence" in result.parse_error


def test_parse_adjudication_bool_confidence_is_not_ok():
    # float(True) would silently be 1.0; a boolean is not a confidence.
    raw = '{"call": "occlusion_signature_present", "confidence": true}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is False
    assert result.confidence is None


def test_parse_adjudication_nan_confidence_is_not_ok():
    raw = '{"call": "occlusion_signature_present", "confidence": NaN}'
    result = parse_adjudication_json(raw)
    assert result.parsed_ok is False
    assert result.confidence is None
    assert "non-finite confidence" in result.parse_error


# --------------------------------------------------------------------------- #
# Prompt-SHA stability
# --------------------------------------------------------------------------- #


def test_extract_prompt_sha_is_stable():
    a = prompt_sha256(build_extract_messages("patient on norepinephrine, lactate 4.2"))
    b = prompt_sha256(build_extract_messages("patient on norepinephrine, lactate 4.2"))
    assert a == b
    assert len(a) == 64


def test_extract_prompt_sha_changes_with_text():
    a = prompt_sha256(build_extract_messages("text one"))
    b = prompt_sha256(build_extract_messages("text two"))
    assert a != b


def test_adjudicate_prompt_sha_is_stable_for_same_image():
    a = prompt_sha256(build_adjudicate_messages(_PNG_BYTES))
    b = prompt_sha256(build_adjudicate_messages(_PNG_BYTES))
    assert a == b


def test_adjudicate_prompt_sha_changes_with_image():
    a = prompt_sha256(build_adjudicate_messages(_PNG_BYTES))
    b = prompt_sha256(build_adjudicate_messages(_PNG_BYTES + b"\x00"))
    assert a != b


# --------------------------------------------------------------------------- #
# Stub client + harness wiring
# --------------------------------------------------------------------------- #


def test_stub_extract_returns_parseable_result():
    client = StubClient()
    result, sha = extract_text(client, "any context", seed=GLOBAL_SEED)
    assert result.parsed_ok is True
    assert len(sha) == 64


def test_stub_adjudicate_returns_parseable_result():
    client = StubClient()
    result, sha = adjudicate_image(client, _PNG_BYTES, seed=GLOBAL_SEED)
    assert result.parsed_ok is True
    assert result.call in (
        "occlusion_signature_present",
        "no_occlusion_signature",
        "indeterminate",
    )
    assert len(sha) == 64


# --------------------------------------------------------------------------- #
# Determinism: decoding params actually reach the client (reproducibility)
# --------------------------------------------------------------------------- #


class _SpyClient:
    """Records the kwargs of the last ``complete`` call. Returns canned JSON.

    Stands in for the live client to assert that ``temperature``, ``max_tokens``
    and ``seed`` are forwarded unchanged. This is what protects the
    reproducibility claim: the run log records the decoding params, so they must
    be the params actually passed to the model call.
    """

    model = "spy-model"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, *, temperature, max_tokens, seed) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
        return '{"phenotype": "x", "vasopressor_use": null, "shock_state": null, "notes": "ok"}'


def test_extract_forwards_decoding_params_to_client():
    spy = _SpyClient()
    extract_text(spy, "context", temperature=0.0, max_tokens=256, seed=12345)
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 256
    assert call["seed"] == 12345


def test_adjudicate_forwards_decoding_params_to_client():
    spy = _SpyClient()
    # Canned extraction JSON is not a valid adjudication, but parsing happens
    # after the call; we only assert the kwargs the client received.
    adjudicate_image(spy, _PNG_BYTES, temperature=0.7, max_tokens=99, seed=777)
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 99
    assert call["seed"] == 777


def test_extract_uses_deterministic_defaults_when_unspecified():
    spy = _SpyClient()
    extract_text(spy, "context", seed=GLOBAL_SEED)
    call = spy.calls[0]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call["seed"] == GLOBAL_SEED


def test_omlx_client_passes_params_to_openai_create(monkeypatch):
    """The live OMLXClient must forward all three decoding params to openai.

    Uses a fake ``openai`` module (no network) and captures the kwargs handed to
    ``chat.completions.create``. Guards against silently dropping a param, which
    would make the run log misreport the decoding actually used.
    """
    import sys
    import types

    captured: dict = {}

    class _FakeMessage:
        content = '{"call": "indeterminate", "confidence": 0.5, "rationale": "ok"}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            self.base_url = base_url
            self.chat = _FakeChat()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI
    # Provide the submodule the client imports for typing-only narrowing.
    fake_types = types.ModuleType("openai.types")
    fake_chat = types.ModuleType("openai.types.chat")
    fake_chat.ChatCompletionMessageParam = dict
    fake_types.chat = fake_chat
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setitem(sys.modules, "openai.types", fake_types)
    monkeypatch.setitem(sys.modules, "openai.types.chat", fake_chat)

    from cuffcrt.llm.client import OMLXClient

    client = OMLXClient(model="m", base_url="http://localhost:9999/v1")
    out = client.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=512,
        seed=20260426,
    )
    assert out  # non-empty content returned
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 512
    assert captured["seed"] == 20260426
    assert captured["model"] == "m"


# --------------------------------------------------------------------------- #
# Dry-run CLI: run-log shape and DUA (no raw note text)
# --------------------------------------------------------------------------- #


def test_dry_run_extract_writes_well_formed_runlog(tmp_path: Path):
    cli = _load_script_module()
    secret_note = "SECRET_NOTE_TOKEN patient with septic shock on pressors"
    input_dir = tmp_path / "notes"
    input_dir.mkdir()
    (input_dir / "rec_001.txt").write_text(secret_note, encoding="utf-8")
    out_dir = tmp_path / "out"

    code = cli.main(
        [
            "extract",
            "--input-dir",
            str(input_dir),
            "--out",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert code == 0

    csv_files = list(out_dir.glob("medgemma_runlog_*.csv"))
    parquet_files = list(out_dir.glob("medgemma_runlog_*.parquet"))
    assert len(csv_files) == 1
    assert len(parquet_files) == 1

    df = pl.read_csv(csv_files[0])
    # All required columns present, in canonical order.
    assert df.columns == list(RUN_LOG_COLUMNS)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["mode"] == "extract"
    assert row["seed"] == GLOBAL_SEED
    assert row["temperature"] == 0.0
    assert row["parsed_ok"] is True
    assert len(row["prompt_sha256"]) == 64

    # DUA: the raw note text must not appear anywhere in any output artifact.
    csv_text = csv_files[0].read_text(encoding="utf-8")
    assert "SECRET_NOTE_TOKEN" not in csv_text
    parquet_text = parquet_files[0].read_bytes()
    assert b"SECRET_NOTE_TOKEN" not in parquet_text


def test_dry_run_adjudicate_writes_well_formed_runlog(tmp_path: Path):
    cli = _load_script_module()
    input_dir = tmp_path / "plots"
    input_dir.mkdir()
    (input_dir / "plot_p10014354.png").write_bytes(_PNG_BYTES)
    out_dir = tmp_path / "out"

    code = cli.main(
        [
            "adjudicate",
            "--input-dir",
            str(input_dir),
            "--out",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert code == 0

    df = pl.read_csv(next(iter(out_dir.glob("medgemma_runlog_*.csv"))))
    assert df.columns == list(RUN_LOG_COLUMNS)
    row = df.row(0, named=True)
    assert row["mode"] == "adjudicate"
    assert row["call"] in (
        "occlusion_signature_present",
        "no_occlusion_signature",
        "indeterminate",
    )
    assert row["image_path"] == "plot_p10014354.png"
    assert row["image_sha256"] is not None and len(row["image_sha256"]) == 64
    assert row["parsed_ok"] is True


def test_dry_run_refuses_to_overwrite_input_dir(tmp_path: Path):
    cli = _load_script_module()
    input_dir = tmp_path / "notes"
    input_dir.mkdir()
    (input_dir / "rec.txt").write_text("context", encoding="utf-8")

    code = cli.main(
        [
            "extract",
            "--input-dir",
            str(input_dir),
            "--out",
            str(input_dir),  # same as input dir -> must refuse
            "--dry-run",
        ]
    )
    assert code == 2


def test_dry_run_no_inputs_is_clean_noop(tmp_path: Path):
    cli = _load_script_module()
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    code = cli.main(
        [
            "extract",
            "--input-dir",
            str(input_dir),
            "--out",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not list(out_dir.glob("*")) if out_dir.exists() else True


# --------------------------------------------------------------------------- #
# --demo wiring: resolve via _paths, fail clean when absent
# --------------------------------------------------------------------------- #


def test_demo_missing_inputs_fails_clean(tmp_path: Path):
    cli = _load_script_module()
    # data_root with no demo inputs built yet -> must exit 2, not crash.
    data_root = tmp_path / "data"
    data_root.mkdir()
    out_dir = tmp_path / "out"
    code = cli.main(
        [
            "extract",
            "--demo",
            "--data-root",
            str(data_root),
            "--out",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert code == 2


def test_demo_resolves_input_dir_when_present(tmp_path: Path):
    cli = _load_script_module()
    data_root = tmp_path / "data"
    # Mirror DEMO_ADJUDICATE_PLOTS_SUBPATH = interim/demo/plots.
    plots_dir = data_root / "interim" / "demo" / "plots"
    plots_dir.mkdir(parents=True)
    (plots_dir / "plot_demo.png").write_bytes(_PNG_BYTES)
    out_dir = tmp_path / "out"

    code = cli.main(
        [
            "adjudicate",
            "--demo",
            "--data-root",
            str(data_root),
            "--out",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert code == 0
    df = pl.read_csv(next(iter(out_dir.glob("medgemma_runlog_*.csv"))))
    assert df.height == 1
    assert df.row(0, named=True)["image_path"] == "plot_demo.png"
