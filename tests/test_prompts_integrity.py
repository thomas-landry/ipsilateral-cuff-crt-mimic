"""Frozen-prompt integrity tests.

These guard the prompt files under ``prompts/`` against silent drift. The
stamped SHA-256 header must agree with the recomputed digest of the body, the
canonical vocabulary must appear (and the legacy vocabulary must not), and the
loader must fail clean when either condition is violated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cuffcrt.llm.prompts import (
    ADJUDICATE_SYSTEM_PROMPT_FILE,
    ADJUDICATE_USER_PROMPT_FILE,
    PromptIntegrityError,
    load_adjudicate_prompts,
    load_prompt,
)

ALL_PROMPTS = (ADJUDICATE_SYSTEM_PROMPT_FILE, ADJUDICATE_USER_PROMPT_FILE)


def test_prompt_files_exist():
    for path in ALL_PROMPTS:
        assert path.exists(), f"missing prompt file: {path}"


@pytest.mark.parametrize("path", ALL_PROMPTS)
def test_prompt_sha_header_matches_body(path: Path):
    """Each prompt file begins with ``# sha256: <hex>`` over the body bytes."""
    raw = path.read_bytes()
    first_newline = raw.find(b"\n")
    assert first_newline != -1, f"prompt file {path} has no header line"
    header = raw[:first_newline].decode("utf-8")
    body = raw[first_newline + 1 :]
    assert header.startswith("# sha256: ")
    stamped = header[len("# sha256: ") :].strip()
    assert len(stamped) == 64
    assert hashlib.sha256(body).hexdigest() == stamped


def test_load_adjudicate_prompts_returns_canonical_vocabulary():
    """Loaded system prompt names the canonical call vocabulary verbatim."""
    system, user = load_adjudicate_prompts()
    assert "occlusion_signature_present" in system.text
    assert "no_occlusion_signature" in system.text
    assert "indeterminate" in system.text
    # User text is the short instruction; only the system prompt names the calls.
    assert user.body_bytes > 0
    # The legacy laterality vocabulary must not appear anywhere.
    for legacy in ("ipsilateral", "not_ipsilateral"):
        assert legacy not in system.text
        assert legacy not in user.text


def test_load_adjudicate_prompts_returns_verified_sha(tmp_path: Path):
    """The returned ``sha256`` is the SHA-256 of the body alone."""
    system, _ = load_adjudicate_prompts()
    body_bytes = system.text.encode("utf-8")
    assert hashlib.sha256(body_bytes).hexdigest() == system.sha256
    assert len(system.sha256) == 64


def test_load_prompt_rejects_missing_header(tmp_path: Path):
    """A prompt file without a SHA header raises ``PromptIntegrityError``."""
    path = tmp_path / "no_header.txt"
    path.write_text("plain body, no sha header\n", encoding="utf-8")
    with pytest.raises(PromptIntegrityError, match="missing the leading"):
        load_prompt(path)


def test_load_prompt_rejects_mismatched_sha(tmp_path: Path):
    """A prompt file whose stamped SHA does not match raises cleanly."""
    body = b"This is the body.\n"
    # Intentionally wrong stamp.
    wrong = "0" * 64
    payload = f"# sha256: {wrong}\n".encode() + body
    path = tmp_path / "wrong.txt"
    path.write_bytes(payload)
    with pytest.raises(PromptIntegrityError, match="sha mismatch"):
        load_prompt(path)


def test_load_prompt_rejects_malformed_sha(tmp_path: Path):
    """A prompt file with a non-hex SHA stamp fails cleanly."""
    path = tmp_path / "bad_stamp.txt"
    path.write_bytes(b"# sha256: nothex\nbody\n")
    with pytest.raises(PromptIntegrityError, match="malformed sha header"):
        load_prompt(path)


def test_load_prompt_round_trip(tmp_path: Path):
    """A correctly-stamped prompt loads with body text and verified digest."""
    body = b"body content with a # internal hash mark\n"
    sha = hashlib.sha256(body).hexdigest()
    path = tmp_path / "ok.txt"
    path.write_bytes(f"# sha256: {sha}\n".encode() + body)
    loaded = load_prompt(path)
    assert loaded.sha256 == sha
    assert loaded.text == body.decode("utf-8")
    assert loaded.body_bytes == len(body)


def test_load_prompt_missing_file_raises():
    """A missing prompt file raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_prompt(Path("/nonexistent/path/to/prompt.txt"))
