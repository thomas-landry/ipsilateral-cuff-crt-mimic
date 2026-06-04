"""Local-LLM inference harness for the MedGemma (oMLX) step.

This subpackage is a plain client to a local, OpenAI-compatible server (oMLX
serving MedGemma). It carries no model weights. The
result-generating path is a single chat-completions call with deterministic
decoding, plus a per-row run log capturing model id, prompt hash, and decoding
parameters so any output is reproducible and auditable.

Outputs hold derived fields only. Raw de-identified note text supplied to the
``extract`` mode is never written to any output artifact (only its SHA-256 via
the prompt hash).
"""

from cuffcrt.llm.medgemma import (
    RUN_LOG_COLUMNS,
    AdjudicationResult,
    ExtractionResult,
    adjudicate_image,
    adjudication_log_row,
    build_adjudicate_messages,
    build_extract_messages,
    extract_text,
    extraction_log_row,
    parse_adjudication_json,
    parse_extraction_json,
    prompt_sha256,
)

__all__ = [
    "RUN_LOG_COLUMNS",
    "AdjudicationResult",
    "ExtractionResult",
    "adjudicate_image",
    "adjudication_log_row",
    "build_adjudicate_messages",
    "build_extract_messages",
    "extract_text",
    "extraction_log_row",
    "parse_adjudication_json",
    "parse_extraction_json",
    "prompt_sha256",
]
