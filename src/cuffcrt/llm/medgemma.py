"""Prompt construction, response parsing, and the run-log schema for MedGemma.

This module is pure and deterministic apart from the model call itself. It
builds the chat-message payloads for the two inference modes, hashes the prompt
so identical inputs produce an identical SHA-256, parses the model's JSON reply
defensively (tolerating code fences and leading/trailing prose), and defines
the run-log columns that make every row reproducible.

Two modes:

``extract``
    Text in, structured JSON out. A de-identified note or set of structured
    fields is summarized into a small phenotype object. The note text itself is
    never written to an output artifact; only its contribution to the prompt
    SHA-256 is retained.

``adjudicate``
    Image in, structured JSON out. An unannotated perfusion-index-versus-time
    plot is shown to the model, which returns
    ``{observed, call, confidence, rationale}`` where ``call`` is one of
    ``{"occlusion_signature_present", "no_occlusion_signature", "indeterminate"}``.
    The reader is blinded: the detector's own verdict is never placed in the
    prompt, so the model's call is independent of the rule-based label. The
    system and user prompts are loaded from frozen files under ``prompts/`` and
    integrity-checked at load time; see :mod:`cuffcrt.llm.prompts`.

"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from cuffcrt.llm.client import ChatClient
from cuffcrt.llm.prompts import load_adjudicate_prompts

# Deterministic decoding defaults. Temperature 0 plus a fixed seed make the
# served model reproducible to the extent the runtime is deterministic; the
# run log records all three so any drift is detectable.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512

# Extraction system prompt remains in-module because it has no public on-disk
# counterpart; it is short, terse, and JSON-only. The parser tolerates fences.
EXTRACT_SYSTEM_PROMPT = (
    "You are a careful clinical data abstractor. You are given de-identified "
    "ICU context (a short note excerpt or structured fields). Return ONLY a "
    "single JSON object, no prose and no code fences, with these keys: "
    '"phenotype" (short string), "vasopressor_use" (true, false, or null), '
    '"shock_state" (short string or null), "notes" (short string). '
    "If a field is not stated, use null. Do not invent values."
)

# Canonical adjudication vocabulary. Anything else (including the legacy
# values listed below) is treated as a parse failure.
VALID_CALLS = ("occlusion_signature_present", "no_occlusion_signature", "indeterminate")
LEGACY_CALLS = ("ipsilateral", "not_ipsilateral")

# Required keys in a well-formed adjudication response. ``schema_complete`` on
# the result is True iff all four keys are present with the right types.
ADJUDICATE_REQUIRED_KEYS = ("observed", "call", "confidence", "rationale")

# Run-log columns (order is the on-disk column order). Outputs hold derived
# fields only: the raw note never appears, only prompt_sha256.
RUN_LOG_COLUMNS: tuple[str, ...] = (
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
    # extract-mode parsed fields
    "phenotype",
    "vasopressor_use",
    "shock_state",
    "notes",
    # adjudicate-mode parsed fields
    "image_path",
    "image_sha256",
    "observed",
    "call",
    "confidence",
    "rationale",
    # the verbatim model output (derived; contains no raw note text)
    "raw_response",
)


@dataclass(frozen=True)
class ExtractionResult:
    """Parsed result of one ``extract`` call.

    ``parsed_ok`` is ``False`` when the model output could not be parsed into
    the expected object; the partial fields are then ``None`` and
    ``parse_error`` explains why. ``raw_response`` is always retained.
    """

    parsed_ok: bool
    phenotype: str | None
    vasopressor_use: bool | None
    shock_state: str | None
    notes: str | None
    raw_response: str
    parse_error: str | None


@dataclass(frozen=True)
class AdjudicationResult:
    """Parsed result of one ``adjudicate`` call.

    ``call`` is constrained to :data:`VALID_CALLS`; an out-of-vocabulary call
    (including the legacy ``"ipsilateral"`` / ``"not_ipsilateral"`` values) is
    treated as a parse failure (``parsed_ok=False``) so it cannot silently
    pollute downstream tallies.

    ``schema_complete`` is True only when the response contained all four
    required keys (:data:`ADJUDICATE_REQUIRED_KEYS`) with the expected types.
    A response can be ``parsed_ok=True`` with ``schema_complete=False`` when
    the call and confidence parse cleanly but ``observed`` or ``rationale`` is
    missing (the call is still usable, but the schema is incomplete).
    """

    parsed_ok: bool
    schema_complete: bool
    observed: str | None
    call: str | None
    confidence: float | None
    rationale: str | None
    raw_response: str
    parse_error: str | None


def prompt_sha256(messages: list[dict]) -> str:
    """Return a stable SHA-256 hex digest of a chat-message payload.

    The payload is serialized with sorted keys and no insignificant whitespace,
    so identical logical inputs always hash identically regardless of dict
    construction order.

    Parameters
    ----------
    messages : list[dict]
        The chat-message payload (system and user turns, including any image
        parts encoded as data URLs).

    Returns
    -------
    str
        64-character hexadecimal SHA-256 digest.
    """
    serialized = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_extract_messages(context_text: str) -> list[dict]:
    """Build the chat payload for text extraction.

    Parameters
    ----------
    context_text : str
        De-identified note excerpt or structured-field text. This text is
        embedded in the prompt but is never written to any output artifact.

    Returns
    -------
    list[dict]
        System and user messages for a chat-completions call.
    """
    return [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": context_text},
    ]


def _encode_image_data_url(image_bytes: bytes, *, media_type: str = "image/png") -> str:
    """Return a base64 ``data:`` URL for raw image bytes."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def build_adjudicate_messages(image_bytes: bytes, *, media_type: str = "image/png") -> list[dict]:
    """Build the chat payload for image adjudication.

    The detector's verdict is intentionally absent from the payload: the
    reviewer is blinded to the rule-based label. The system and user prompts
    are loaded from the frozen files under ``prompts/`` (see
    :mod:`cuffcrt.llm.prompts`); any drift in those files triggers a clean
    integrity error.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the unannotated PI(t) plot.
    media_type : str, optional
        MIME type for the data URL (default ``"image/png"``).

    Returns
    -------
    list[dict]
        System and user messages, the latter carrying a text part and an
        image part.
    """
    system_prompt, user_prompt = load_adjudicate_prompts()
    data_url = _encode_image_data_url(image_bytes, media_type=media_type)
    return [
        {"role": "system", "content": system_prompt.text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt.text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def _extract_json_object(raw: str | None) -> dict[str, Any]:
    """Pull the first balanced JSON object out of a model response.

    Tolerates the common failure modes: Markdown code fences, a language tag
    after the fence, and leading or trailing prose around the object. A
    ``None`` response (e.g. a server returning no content) is treated the same
    as an empty one. Raises ``ValueError`` if no parseable object is found.
    """
    if raw is None:
        raise ValueError("empty response")
    text = raw.strip()
    if not text:
        raise ValueError("empty response")

    # Strip a fenced block if present (```json ... ``` or ``` ... ```).
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Fast path: the whole thing is JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: scan for the first balanced {...} span and parse it.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)

    raise ValueError("no JSON object found in response")


def _coerce_bool_or_none(value: Any) -> bool | None:
    """Coerce common truthy/falsy model encodings to ``bool`` or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "y", "1"}:
            return True
        if low in {"false", "no", "n", "0"}:
            return False
        if low in {"null", "none", "unknown", ""}:
            return None
    return None


def _coerce_str_or_none(value: Any) -> str | None:
    """Coerce a value to a trimmed string, mapping null-like tokens to ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return text


def parse_extraction_json(raw: str) -> ExtractionResult:
    """Parse an ``extract`` response into an :class:`ExtractionResult`.

    Never raises: malformed output yields ``parsed_ok=False`` with the original
    text preserved in ``raw_response``.
    """
    try:
        obj = _extract_json_object(raw)
    except ValueError as exc:
        return ExtractionResult(
            parsed_ok=False,
            phenotype=None,
            vasopressor_use=None,
            shock_state=None,
            notes=None,
            raw_response=raw if raw is not None else "",
            parse_error=str(exc),
        )
    return ExtractionResult(
        parsed_ok=True,
        phenotype=_coerce_str_or_none(obj.get("phenotype")),
        vasopressor_use=_coerce_bool_or_none(obj.get("vasopressor_use")),
        shock_state=_coerce_str_or_none(obj.get("shock_state")),
        notes=_coerce_str_or_none(obj.get("notes")),
        raw_response=raw,
        parse_error=None,
    )


def _adjudicate_failed(
    *,
    raw: str,
    parse_error: str,
    observed: str | None = None,
    call: str | None = None,
    confidence: float | None = None,
    rationale: str | None = None,
) -> AdjudicationResult:
    """Build a failure :class:`AdjudicationResult` with ``schema_complete=False``."""
    return AdjudicationResult(
        parsed_ok=False,
        schema_complete=False,
        observed=observed,
        call=call,
        confidence=confidence,
        rationale=rationale,
        raw_response=raw if raw is not None else "",
        parse_error=parse_error,
    )


def parse_adjudication_json(raw: str) -> AdjudicationResult:
    """Parse an ``adjudicate`` response into an :class:`AdjudicationResult`.

    Never raises. An unparseable response, a missing/invalid ``call``, a
    legacy (pre-canonicalization) ``call`` value, or a non-numeric
    ``confidence`` yields ``parsed_ok=False`` with the raw text preserved.

    The ``schema_complete`` flag on the returned result reflects whether all
    four required keys (:data:`ADJUDICATE_REQUIRED_KEYS`) were present with
    the expected types in the model's response.
    """
    try:
        obj = _extract_json_object(raw)
    except ValueError as exc:
        return _adjudicate_failed(raw=raw, parse_error=str(exc))

    raw_call = obj.get("call")
    call = raw_call.strip().lower() if isinstance(raw_call, str) else None
    observed = _coerce_str_or_none(obj.get("observed"))
    rationale = _coerce_str_or_none(obj.get("rationale"))

    if call in LEGACY_CALLS:
        return _adjudicate_failed(
            raw=raw,
            parse_error=f"legacy_call_value: {call!r} indicates a stale prompt",
            observed=observed,
            rationale=rationale,
        )
    if call not in VALID_CALLS:
        return _adjudicate_failed(
            raw=raw,
            parse_error=f"invalid or missing call: {raw_call!r}",
            observed=observed,
            rationale=rationale,
        )

    confidence_raw = obj.get("confidence")
    # ``float(None)`` raises TypeError and ``float("high")`` raises ValueError;
    # both are caught here so a missing or non-numeric confidence never crashes
    # parsing. ``bool`` is rejected explicitly (``float(True)`` would silently
    # become 1.0). NaN/inf pass ``float()`` but are not valid confidences, so a
    # finiteness check guards the clamp below.
    if confidence_raw is None or isinstance(confidence_raw, bool):
        return _adjudicate_failed(
            raw=raw,
            parse_error=f"non-numeric confidence: {confidence_raw!r}",
            observed=observed,
            call=call,
            rationale=rationale,
        )
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        return _adjudicate_failed(
            raw=raw,
            parse_error=f"non-numeric confidence: {confidence_raw!r}",
            observed=observed,
            call=call,
            rationale=rationale,
        )
    if not math.isfinite(confidence):
        return _adjudicate_failed(
            raw=raw,
            parse_error=f"non-finite confidence: {confidence_raw!r}",
            observed=observed,
            call=call,
            rationale=rationale,
        )
    confidence = max(0.0, min(1.0, confidence))

    schema_complete = (
        isinstance(raw_call, str)
        and observed is not None
        and rationale is not None
        and confidence_raw is not None
        and not isinstance(confidence_raw, bool)
    )

    return AdjudicationResult(
        parsed_ok=True,
        schema_complete=schema_complete,
        observed=observed,
        call=call,
        confidence=confidence,
        rationale=rationale,
        raw_response=raw,
        parse_error=None,
    )


def extraction_log_row(
    *,
    row_id: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    prompt_sha: str,
    run_utc: str,
    result: ExtractionResult,
) -> dict[str, Any]:
    """Assemble one run-log row for an ``extract`` call.

    The returned dict has exactly the keys in :data:`RUN_LOG_COLUMNS`.
    Adjudicate-only fields are ``None``. No raw note text is included; the note
    is represented only by ``prompt_sha256``.
    """
    return {
        "row_id": row_id,
        "mode": "extract",
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "prompt_sha256": prompt_sha,
        "run_utc": run_utc,
        "parsed_ok": result.parsed_ok,
        "schema_complete": None,
        "parse_error": result.parse_error,
        "phenotype": result.phenotype,
        "vasopressor_use": result.vasopressor_use,
        "shock_state": result.shock_state,
        "notes": result.notes,
        "image_path": None,
        "image_sha256": None,
        "observed": None,
        "call": None,
        "confidence": None,
        "rationale": None,
        "raw_response": result.raw_response,
    }


def adjudication_log_row(
    *,
    row_id: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    prompt_sha: str,
    run_utc: str,
    image_path: str,
    image_sha256: str | None,
    result: AdjudicationResult,
) -> dict[str, Any]:
    """Assemble one run-log row for an ``adjudicate`` call.

    The returned dict has exactly the keys in :data:`RUN_LOG_COLUMNS`.
    Extract-only fields are ``None``.
    """
    return {
        "row_id": row_id,
        "mode": "adjudicate",
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "prompt_sha256": prompt_sha,
        "run_utc": run_utc,
        "parsed_ok": result.parsed_ok,
        "schema_complete": result.schema_complete,
        "parse_error": result.parse_error,
        "phenotype": None,
        "vasopressor_use": None,
        "shock_state": None,
        "notes": None,
        "image_path": image_path,
        "image_sha256": image_sha256,
        "observed": result.observed,
        "call": result.call,
        "confidence": result.confidence,
        "rationale": result.rationale,
        "raw_response": result.raw_response,
    }


def extract_text(
    client: ChatClient,
    context_text: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int,
) -> tuple[ExtractionResult, str]:
    """Run one ``extract`` call and return ``(result, prompt_sha256)``.

    Parameters
    ----------
    client : ChatClient
        A live :class:`~cuffcrt.llm.client.OMLXClient` or
        :class:`~cuffcrt.llm.client.StubClient`.
    context_text : str
        De-identified context (never written to outputs).
    temperature, max_tokens, seed
        Decoding parameters, logged per row.

    Returns
    -------
    tuple[ExtractionResult, str]
        The parsed result and the prompt SHA-256.
    """
    messages = build_extract_messages(context_text)
    sha = prompt_sha256(messages)
    raw = client.complete(
        messages, temperature=temperature, max_tokens=max_tokens, seed=seed
    )
    return parse_extraction_json(raw), sha


def adjudicate_image(
    client: ChatClient,
    image_bytes: bytes,
    *,
    media_type: str = "image/png",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int,
) -> tuple[AdjudicationResult, str]:
    """Run one blinded ``adjudicate`` call and return ``(result, prompt_sha256)``.

    Parameters
    ----------
    client : ChatClient
        A live or stub chat client.
    image_bytes : bytes
        Raw bytes of the unannotated PI(t) plot.
    media_type : str, optional
        Image MIME type (default ``"image/png"``).
    temperature, max_tokens, seed
        Decoding parameters, logged per row.

    Returns
    -------
    tuple[AdjudicationResult, str]
        The parsed result and the prompt SHA-256.
    """
    messages = build_adjudicate_messages(image_bytes, media_type=media_type)
    sha = prompt_sha256(messages)
    raw = client.complete(
        messages, temperature=temperature, max_tokens=max_tokens, seed=seed
    )
    return parse_adjudication_json(raw), sha
