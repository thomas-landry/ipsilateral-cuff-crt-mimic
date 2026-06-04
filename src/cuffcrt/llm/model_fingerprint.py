"""Read and report the served-model SHA-256 fingerprint for the next run.

The fingerprint is produced out-of-band by ``scripts/compute_model_sha.sh``
which writes ``results/medgemma/_model_fingerprint_<utc_iso>.json``. The
MedGemma harness scripts call :func:`load_latest_fingerprint` at run start so
each per-row log entry and the per-run manifest can record what weights were
served. If no fingerprint file is present the harness still runs but logs a
loguru warning; this keeps the reproducibility claim honest (we report what
we know, not what we guess).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FINGERPRINT_DIR = Path("results/medgemma")
FINGERPRINT_GLOB = "_model_fingerprint_*.json"


@dataclass(frozen=True)
class ModelFingerprint:
    """The contents of one ``_model_fingerprint_<utc_iso>.json`` file.

    Attributes
    ----------
    path : pathlib.Path
        Absolute path to the fingerprint JSON on disk.
    model_id : str
        The reported model id (``OMLX_MODEL_ID`` at fingerprint time or
        ``"unknown"``).
    model_dir : str
        Absolute path that was hashed.
    model_weights_sha256 : str
        Composite SHA-256 of the sorted weight files.
    computed_utc : str
        ISO-8601 timestamp at which the fingerprint was produced.
    files_hashed : int
        Number of weight files included in the composite SHA-256.
    """

    path: Path
    model_id: str
    model_dir: str
    model_weights_sha256: str
    computed_utc: str
    files_hashed: int


def load_latest_fingerprint(
    fingerprint_dir: Path = DEFAULT_FINGERPRINT_DIR,
) -> ModelFingerprint | None:
    """Return the most recent fingerprint under ``fingerprint_dir``, or None.

    Parameters
    ----------
    fingerprint_dir : pathlib.Path, optional
        Directory holding the ``_model_fingerprint_<utc_iso>.json`` files
        (default :data:`DEFAULT_FINGERPRINT_DIR`).

    Returns
    -------
    ModelFingerprint or None
        ``None`` when the directory is missing, no matching files exist, or
        the newest matching file does not parse.
    """
    if not fingerprint_dir.exists() or not fingerprint_dir.is_dir():
        return None
    candidates = sorted(fingerprint_dir.glob(FINGERPRINT_GLOB))
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    files = payload.get("files_hashed") or []
    return ModelFingerprint(
        path=latest,
        model_id=str(payload.get("model_id", "unknown")),
        model_dir=str(payload.get("model_dir", "")),
        model_weights_sha256=str(payload.get("model_weights_sha256", "")),
        computed_utc=str(payload.get("computed_utc", "")),
        files_hashed=len(files) if isinstance(files, list) else 0,
    )
