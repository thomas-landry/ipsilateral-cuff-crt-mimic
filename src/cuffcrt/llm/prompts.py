"""Frozen on-disk prompt loader for the MedGemma adjudication step.

The blinded adjudication prompts live as plain text under ``prompts/`` at the
repo root so they can be reviewed by any reader (no hidden Python constants).
Each file begins with a ``# sha256: <hex>`` line that records the SHA-256 of
the prompt body (everything after that header line). At load time the harness
recomputes the SHA-256 of the body and asserts it matches the stamped value;
any silent drift in the prompt text causes a clean ``PromptIntegrityError``
rather than a subtle change in MedGemma behavior.

The loader returns the body text (with the SHA header stripped) and the
verified digest, so callers can both feed the model the verbatim prompt and
record the digest in the per-row run log.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Directory layout. The prompts directory is a sibling of ``src/``; resolve it
# from this file so the loader works regardless of the caller's cwd.
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

ADJUDICATE_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "adjudicate_system.txt"
ADJUDICATE_USER_PROMPT_FILE = PROMPTS_DIR / "adjudicate_user.txt"

_SHA_HEADER_PREFIX = "# sha256:"


class PromptIntegrityError(RuntimeError):
    """Raised when a prompt file's stamped SHA disagrees with its body bytes."""


@dataclass(frozen=True)
class LoadedPrompt:
    """A loaded prompt with its verified SHA-256 digest.

    Attributes
    ----------
    path : pathlib.Path
        Absolute path to the prompt file on disk.
    text : str
        The prompt body that should be sent to the model. The leading
        ``# sha256:`` header line has been stripped.
    sha256 : str
        Lowercase 64-character hexadecimal SHA-256 of the body bytes (UTF-8).
    body_bytes : int
        Length in bytes of the body (after stripping the SHA header).
    """

    path: Path
    text: str
    sha256: str
    body_bytes: int


def _split_sha_header(raw: bytes) -> tuple[str | None, bytes]:
    """Split a stamped SHA header off the front of ``raw``.

    Returns ``(stamped_sha, body_bytes)``. If the file does not begin with a
    ``# sha256: <hex>`` line, ``stamped_sha`` is ``None`` and the body is the
    whole file.

    Notes
    -----
    The header consumes exactly one line including its trailing newline. Bodies
    that themselves contain ``#`` comment lines are preserved verbatim because
    only the first line is inspected.
    """
    newline_index = raw.find(b"\n")
    if newline_index == -1:
        first_line = raw.decode("utf-8", errors="replace").strip()
        remainder = b""
    else:
        first_line = raw[:newline_index].decode("utf-8", errors="replace").strip()
        remainder = raw[newline_index + 1 :]
    if not first_line.startswith(_SHA_HEADER_PREFIX):
        return None, raw
    stamped = first_line[len(_SHA_HEADER_PREFIX) :].strip().lower()
    return stamped, remainder


def load_prompt(path: Path) -> LoadedPrompt:
    """Load a prompt file and verify its stamped SHA-256 against its body.

    Parameters
    ----------
    path : pathlib.Path
        Absolute path to a prompt file (UTF-8 text).

    Returns
    -------
    LoadedPrompt
        The body text plus the recomputed SHA-256. The header is stripped from
        ``text`` so callers feed the model the body verbatim.

    Raises
    ------
    FileNotFoundError
        When ``path`` does not exist.
    PromptIntegrityError
        When the file lacks a stamped SHA header, when the stamped value is
        not 64 hexadecimal characters, or when the recomputed SHA disagrees
        with the stamped value.
    """
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    raw = path.read_bytes()
    stamped, body = _split_sha_header(raw)
    if stamped is None:
        raise PromptIntegrityError(
            f"prompt file {path} is missing the leading '# sha256: <hex>' header"
        )
    if len(stamped) != 64 or any(ch not in "0123456789abcdef" for ch in stamped):
        raise PromptIntegrityError(
            f"prompt file {path} has a malformed sha header: {stamped!r}"
        )
    computed = hashlib.sha256(body).hexdigest()
    if computed != stamped:
        raise PromptIntegrityError(
            f"prompt file {path} sha mismatch: stamped {stamped} but body hashes to {computed}"
        )
    return LoadedPrompt(
        path=path,
        text=body.decode("utf-8"),
        sha256=computed,
        body_bytes=len(body),
    )


def load_adjudicate_prompts() -> tuple[LoadedPrompt, LoadedPrompt]:
    """Load and verify both adjudication prompts.

    Returns
    -------
    tuple[LoadedPrompt, LoadedPrompt]
        ``(system_prompt, user_prompt)``.
    """
    return load_prompt(ADJUDICATE_SYSTEM_PROMPT_FILE), load_prompt(ADJUDICATE_USER_PROMPT_FILE)
