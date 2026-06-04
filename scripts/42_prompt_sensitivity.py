"""Prompt-sensitivity helper for the MedGemma adjudication step (D5; step 42).

This batch driver re-runs MedGemma on a small seed-fixed subsample of the
blinded gallery under two or three minor rewordings of the canonical system
prompt. It produces one per-variant run log (CSV plus parquet) in the schema of
script 41, so a downstream concordance analysis can join variant calls against
the headline run on ``card_id``.

What it does
------------
1. Reads the blinded gallery manifest written by ``scripts/51_candidate_gallery.py``.
2. Draws a stratified random subsample (default 100 cards) across the three
   detector tiers (``detector_positive``, ``detector_rejected_near_miss``,
   ``detector_negative_random``), preserving each stratum's share of the
   manifest universe and using ``numpy.random.default_rng(seed)`` so the same
   seed always selects the same card ids.
3. For each requested variant (a frozen system-prompt file under ``prompts/``):

   * loads the variant system prompt through the SHA-verified loader
     :mod:`cuffcrt.llm.prompts`;
   * pairs it with the unchanged user prompt;
   * renders each card's PNG bytes from disk (the gallery already has
     pre-rendered PNGs, so no WDB access is required here);
   * runs the local oMLX MedGemma server with the standard decoding parameters
     and the standard parser;
   * writes a checkpoint-resumable run log under
     ``<out_dir>/<variant>_checkpoint.csv`` and finalizes as
     ``<out_dir>/<variant>.csv`` plus ``<out_dir>/<variant>.parquet``.

The per-row schema matches :data:`cuffcrt.llm.medgemma.RUN_LOG_COLUMNS` plus
``subject_id``, ``record_id``, ``card_id``, ``stratum``, and
``model_weights_sha256``, so the variant outputs can be joined against the
headline adjudication run on ``card_id``.

Vocabulary and parsing
----------------------
The variants reword the system prompt but preserve the JSON schema and the
three call values (``occlusion_signature_present``,
``no_occlusion_signature``, ``indeterminate``). 

Provenance and reproducibility
------------------------------
* Decoding is ``temperature=0`` with a fixed ``seed`` and a generous
  ``max_tokens``, matching script 41 so any variant-vs-headline difference is
  attributable to the prompt text alone.
* The variant prompt's stamped SHA-256 is verified at load time; drift causes a
  clean :class:`~cuffcrt.llm.prompts.PromptIntegrityError` before any model is
  called.
* Per row, the log records the served model id, the variant prompt SHA-256
  (via ``prompt_sha256`` over the chat-message payload), the gallery PNG's
  ``image_sha256``, base URL, temperature, seed, ``max_tokens``, and a UTC
  timestamp.
* The most recent ``_model_fingerprint_<utc_iso>.json`` under ``--out_dir`` (or
  its parent) is loaded and stamped onto each row as
  ``model_weights_sha256``. When no fingerprint is present the run still
  proceeds and the field is null, the same posture as script 41.

Inputs (read-only)
------------------
* ``--gallery_manifest`` (default ``results/gallery/gallery_manifest.csv``): the
  manifest that script 51 writes, carrying ``card_id``, ``stratum``,
  ``subject_id``, ``record_id``, ``image_path``, ``image_sha256``.
* The variant system prompts under ``prompts/adjudicate_system_v_*.txt`` and the
  unchanged user prompt ``prompts/adjudicate_user.txt``.

Outputs
-------
Under ``--out_dir`` (default ``results/medgemma_prompt_sensitivity/``):

* ``<variant>_checkpoint.csv`` (stable, append-after-each-row, fsynced).
* ``<variant>.csv`` and ``<variant>.parquet`` (finalized at successful end).
* ``_run_manifest_<variant>_<utc_stamp>.json`` (counts and provenance).

Example
-------
::

    uv run python scripts/42_prompt_sensitivity.py \\
        --variants v_compact,v_explicit \\
        --n_rows 100 \\
        --seed 20260426 \\
        --gallery_manifest results/gallery/gallery_manifest.csv \\
        --out_dir results/medgemma_prompt_sensitivity/ \\
        --port 8000
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.llm.client import ChatClient, OMLXClient, StubClient, resolve_base_url
from cuffcrt.llm.medgemma import (
    DEFAULT_TEMPERATURE,
    RUN_LOG_COLUMNS,
    AdjudicationResult,
    _encode_image_data_url,
    adjudication_log_row,
    parse_adjudication_json,
    prompt_sha256,
)
from cuffcrt.llm.model_fingerprint import ModelFingerprint, load_latest_fingerprint
from cuffcrt.llm.prompts import (
    ADJUDICATE_USER_PROMPT_FILE,
    PROMPTS_DIR,
    LoadedPrompt,
    load_prompt,
)

# Served model id on the local oMLX server (matches script 41).
SERVED_MODEL_ID = "medgemma-1.5-4b-it-bf16"

# Generous token budget; mirrors script 41 so the variants are otherwise
# decoding-identical to the headline run.
MAX_TOKENS = 1536

DEFAULT_GALLERY_MANIFEST = Path("results/gallery/gallery_manifest.csv")
DEFAULT_OUT_DIR = Path("results/medgemma_prompt_sensitivity")
DEFAULT_N_ROWS = 100

# Tiers (strata) used by the gallery; preserved by the stratified sampler.
GALLERY_STRATA: tuple[str, ...] = (
    "detector_positive",
    "detector_rejected_near_miss",
    "detector_negative_random",
)

# Identifier columns prepended to the canonical run-log columns. The schema is
# a superset of script 41's LOG_COLUMNS_WITH_IDS so a downstream join on
# ``card_id`` (or ``subject_id, record_id``) is trivial.
EXTRA_ID_COLUMNS: tuple[str, ...] = ("subject_id", "record_id", "card_id", "stratum")
EXTRA_TRAILING_COLUMNS: tuple[str, ...] = ("model_weights_sha256",)
LOG_COLUMNS_WITH_IDS: tuple[str, ...] = (
    *EXTRA_ID_COLUMNS,
    *RUN_LOG_COLUMNS,
    *EXTRA_TRAILING_COLUMNS,
)

# Filename convention for variant system prompts. The CLI accepts short
# labels like ``v_compact`` and resolves them to ``prompts/adjudicate_system_v_compact.txt``.
VARIANT_FILENAME_TEMPLATE = "adjudicate_system_{label}.txt"


def _load_dotenv_if_present() -> None:
    """Load a repo ``.env`` into the environment if python-dotenv is installed.

    The ``OMLX_API_KEY`` is read by the SDK client only; it is never logged.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision)."""
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0).isoformat()


def _utc_now_stamp() -> str:
    """Return the current UTC time as a filename-safe stamp (YYYYMMDDTHHMMSSZ)."""
    return dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_variant_prompt_path(label: str, prompts_dir: Path = PROMPTS_DIR) -> Path:
    """Resolve a short variant label to the on-disk prompt file path.

    Parameters
    ----------
    label : str
        Short variant label, for example ``"v_compact"``. The label is
        slotted into :data:`VARIANT_FILENAME_TEMPLATE`.
    prompts_dir : pathlib.Path, optional
        Directory holding the prompt files (default :data:`PROMPTS_DIR`).

    Returns
    -------
    pathlib.Path
        Absolute path to ``prompts/adjudicate_system_<label>.txt``.
    """
    return prompts_dir / VARIANT_FILENAME_TEMPLATE.format(label=label)


def load_variant_prompt(label: str, prompts_dir: Path = PROMPTS_DIR) -> LoadedPrompt:
    """Load and SHA-verify one variant system prompt.

    Parameters
    ----------
    label : str
        Short variant label such as ``"v_compact"``.
    prompts_dir : pathlib.Path, optional
        Directory holding the prompt files (default :data:`PROMPTS_DIR`).

    Returns
    -------
    LoadedPrompt
        Body text and verified SHA-256.
    """
    return load_prompt(resolve_variant_prompt_path(label, prompts_dir))


def stratified_subsample(
    manifest: pl.DataFrame,
    *,
    n_rows: int,
    seed: int,
    strata: tuple[str, ...] = GALLERY_STRATA,
) -> pl.DataFrame:
    """Draw a stratified random subsample preserving stratum proportions.

    For each stratum, the per-stratum quota is ``round(stratum_share * n_rows)``
    where ``stratum_share`` is the stratum's fraction of the full manifest. A
    final ``+/- 1`` adjustment is applied to the largest stratum if rounding
    makes the totals miss ``n_rows`` exactly. Within each stratum the rows are
    drawn with ``numpy.random.default_rng(seed)`` so the same ``(manifest, n_rows,
    seed)`` triple always yields the same card ids.

    Parameters
    ----------
    manifest : polars.DataFrame
        Gallery manifest with at least ``card_id`` and ``stratum`` columns.
    n_rows : int
        Target sample size. Must be at least one and at most ``manifest.height``.
    seed : int
        Seed for ``numpy.random.default_rng``.
    strata : tuple[str, ...], optional
        Stratum labels in canonical order (default :data:`GALLERY_STRATA`).

    Returns
    -------
    polars.DataFrame
        The subsample, sorted by ``card_id`` for stability.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if n_rows > manifest.height:
        raise ValueError(
            f"n_rows={n_rows} exceeds manifest size {manifest.height}"
        )
    if "stratum" not in manifest.columns or "card_id" not in manifest.columns:
        raise ValueError("manifest must have 'stratum' and 'card_id' columns")

    total = manifest.height
    rng = np.random.default_rng(seed)

    # Compute per-stratum quotas via largest-remainder rounding so the totals
    # sum to exactly ``n_rows`` while preserving stratum proportions.
    counts: dict[str, int] = {}
    raw_quotas: dict[str, float] = {}
    for stratum in strata:
        stratum_size = manifest.filter(pl.col("stratum") == stratum).height
        raw_quotas[stratum] = stratum_size / total * n_rows
        counts[stratum] = int(np.floor(raw_quotas[stratum]))
    deficit = n_rows - sum(counts.values())
    if deficit != 0:
        remainders = sorted(
            ((label, raw_quotas[label] - counts[label]) for label in strata),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for idx in range(abs(deficit)):
            label = remainders[idx % len(remainders)][0]
            counts[label] += 1 if deficit > 0 else -1

    drawn_frames: list[pl.DataFrame] = []
    for stratum in strata:
        pool = manifest.filter(pl.col("stratum") == stratum).sort("card_id")
        quota = counts[stratum]
        if quota <= 0:
            continue
        if quota > pool.height:
            raise ValueError(
                f"stratum {stratum!r} has {pool.height} rows but quota is {quota}"
            )
        # ``rng.choice`` over the integer indices, without replacement, then
        # gather. This is deterministic given ``seed`` and the stratum order.
        indices = rng.choice(pool.height, size=quota, replace=False)
        drawn_frames.append(pool[sorted(int(i) for i in indices)])

    drawn = pl.concat(drawn_frames) if drawn_frames else manifest.head(0)
    return drawn.sort("card_id")


def _read_image_bytes_for_row(row: dict, gallery_root: Path) -> tuple[bytes, str]:
    """Read the pre-rendered gallery PNG for one manifest row.

    The manifest's ``image_path`` is stored relative to the repository root
    (matches what script 51 writes). When ``gallery_root`` is provided, the
    path is resolved against it; an absolute ``image_path`` is honored as-is.

    Parameters
    ----------
    row : dict
        Manifest row (as returned by ``DataFrame.iter_rows(named=True)``).
    gallery_root : pathlib.Path
        Repository root the relative ``image_path`` is anchored to.

    Returns
    -------
    tuple[bytes, str]
        Raw PNG bytes plus the recomputed SHA-256 hex digest.
    """
    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = gallery_root / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"gallery image missing: {image_path}")
    image_bytes = image_path.read_bytes()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    return image_bytes, image_sha


def build_variant_messages(
    system_prompt: LoadedPrompt,
    user_prompt: LoadedPrompt,
    image_bytes: bytes,
    *,
    media_type: str = "image/png",
) -> list[dict]:
    """Build the chat payload for one variant adjudication call.

    Differs from :func:`cuffcrt.llm.medgemma.build_adjudicate_messages` only in
    that the system prompt is the variant's body rather than the canonical
    one. The user prompt is unchanged.

    Parameters
    ----------
    system_prompt : LoadedPrompt
        Verified variant system prompt.
    user_prompt : LoadedPrompt
        Verified canonical user prompt.
    image_bytes : bytes
        Raw bytes of the pre-rendered PI(t) PNG.
    media_type : str, optional
        Image MIME type (default ``"image/png"``).

    Returns
    -------
    list[dict]
        System and user messages, the latter carrying a text part and an
        image part.
    """
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


def _build_client(args: argparse.Namespace, base_url: str) -> ChatClient:
    """Construct the stub (dry-run) or live oMLX client with the served model id."""
    if args.dry_run:
        logger.info("dry-run: in-process stub client (no server, no network)")
        return StubClient(model=args.model, base_url=base_url)
    logger.info("live: oMLX client model={} base_url={}", args.model, base_url)
    return OMLXClient(model=args.model, base_url=base_url)


def _failed_row(
    *,
    row: dict,
    parse_error: str,
    client_model: str,
    base_url: str,
    args: argparse.Namespace,
    fingerprint: ModelFingerprint | None,
    image_path: str = "",
    image_sha256: str | None = None,
) -> dict:
    """Build a parse-failed run-log row preserving identifier columns."""
    result = AdjudicationResult(
        parsed_ok=False,
        schema_complete=False,
        observed=None,
        call=None,
        confidence=None,
        rationale=None,
        raw_response="",
        parse_error=parse_error,
    )
    base = adjudication_log_row(
        row_id=str(row["card_id"]),
        model=client_model,
        base_url=base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prompt_sha="",
        run_utc=_utc_now_iso(),
        image_path=image_path,
        image_sha256=image_sha256,
        result=result,
    )
    base["subject_id"] = str(row["subject_id"])
    base["record_id"] = str(row["record_id"])
    base["card_id"] = str(row["card_id"])
    base["stratum"] = str(row["stratum"])
    base["model_weights_sha256"] = (
        fingerprint.model_weights_sha256 if fingerprint else None
    )
    return base


def adjudicate_one_card(
    client: ChatClient,
    row: dict,
    *,
    system_prompt: LoadedPrompt,
    user_prompt: LoadedPrompt,
    gallery_root: Path,
    base_url: str,
    args: argparse.Namespace,
    fingerprint: ModelFingerprint | None,
) -> dict:
    """Adjudicate one gallery card under one variant prompt and return a log row.

    Parameters
    ----------
    client : ChatClient
        Live or stub chat client.
    row : dict
        One manifest row (named tuple from ``iter_rows(named=True)``).
    system_prompt, user_prompt : LoadedPrompt
        Verified prompts; the system prompt is the variant under test.
    gallery_root : pathlib.Path
        Repository root that anchors relative ``image_path`` values.
    base_url : str
        Server base URL (logged only).
    args : argparse.Namespace
        Parsed CLI args (carries decoding parameters).
    fingerprint : ModelFingerprint or None
        Latest model fingerprint, used to stamp ``model_weights_sha256``.

    Returns
    -------
    dict
        One run-log row with the keys in :data:`LOG_COLUMNS_WITH_IDS`.
    """
    try:
        image_bytes, image_sha = _read_image_bytes_for_row(row, gallery_root)
    except FileNotFoundError as exc:
        logger.warning("{}: {}", row["card_id"], exc)
        return _failed_row(
            row=row,
            parse_error=str(exc),
            client_model=client.model,
            base_url=base_url,
            args=args,
            fingerprint=fingerprint,
        )

    messages = build_variant_messages(system_prompt, user_prompt, image_bytes)
    sha = prompt_sha256(messages)
    raw = client.complete(
        messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    result = parse_adjudication_json(raw)
    log_row = adjudication_log_row(
        row_id=str(row["card_id"]),
        model=client.model,
        base_url=base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prompt_sha=sha,
        run_utc=_utc_now_iso(),
        image_path=str(row["image_path"]),
        image_sha256=image_sha,
        result=result,
    )
    log_row["subject_id"] = str(row["subject_id"])
    log_row["record_id"] = str(row["record_id"])
    log_row["card_id"] = str(row["card_id"])
    log_row["stratum"] = str(row["stratum"])
    log_row["model_weights_sha256"] = (
        fingerprint.model_weights_sha256 if fingerprint else None
    )
    logger.info(
        "{}: parsed_ok={} schema_complete={} call={} conf={}",
        row["card_id"],
        result.parsed_ok,
        result.schema_complete,
        result.call,
        result.confidence,
    )
    return log_row


def _checkpoint_path_for(out_dir: Path, variant: str) -> Path:
    """Return the stable per-variant checkpoint CSV path."""
    return out_dir / f"{variant}_checkpoint.csv"


def _append_checkpoint_row(row: dict, checkpoint_path: Path) -> None:
    """Append one completed row to the checkpoint CSV and fsync it."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0
    with checkpoint_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(LOG_COLUMNS_WITH_IDS), extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in LOG_COLUMNS_WITH_IDS})
        fh.flush()
        os.fsync(fh.fileno())


def _read_checkpoint_rows(checkpoint_path: Path) -> list[dict]:
    """Read an existing checkpoint CSV; refuse a malformed one."""
    if not checkpoint_path.exists():
        return []
    if checkpoint_path.stat().st_size == 0:
        return []
    try:
        df = pl.read_csv(checkpoint_path, infer_schema_length=20000)
    except Exception as exc:  # noqa: BLE001 - clear error trumps obscure trace
        raise ValueError(f"could not read checkpoint {checkpoint_path}: {exc}") from exc
    missing = [col for col in LOG_COLUMNS_WITH_IDS if col not in df.columns]
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint_path} is missing columns: {missing}"
        )
    return df.select(list(LOG_COLUMNS_WITH_IDS)).to_dicts()


def _write_variant_outputs(
    rows: list[dict], out_dir: Path, *, variant: str
) -> tuple[Path, Path]:
    """Write the finalized per-variant CSV and parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).select(list(LOG_COLUMNS_WITH_IDS))
    csv_path = out_dir / f"{variant}.csv"
    parquet_path = out_dir / f"{variant}.parquet"
    df.write_csv(csv_path)
    df.write_parquet(parquet_path)
    return csv_path, parquet_path


def _write_variant_manifest(
    *,
    rows: list[dict],
    out_dir: Path,
    variant: str,
    variant_prompt: LoadedPrompt,
    user_prompt: LoadedPrompt,
    stamp: str,
    run_utc_start: str,
    run_utc_end: str,
    base_url: str,
    args: argparse.Namespace,
    fingerprint: ModelFingerprint | None,
) -> Path:
    """Write a per-variant manifest JSON capturing counts and provenance."""
    n_total = len(rows)
    n_parsed = sum(1 for r in rows if r["parsed_ok"])
    n_schema = sum(1 for r in rows if r.get("schema_complete"))
    calls: dict[str, int] = {}
    for r in rows:
        if r["parsed_ok"] and r["call"] is not None:
            calls[r["call"]] = calls.get(r["call"], 0) + 1
    payload = {
        "variant_label": variant,
        "n_rows_total": n_total,
        "n_parsed": n_parsed,
        "n_schema_complete": n_schema,
        "calls": calls,
        "run_utc_start": run_utc_start,
        "run_utc_end": run_utc_end,
        "model_id": args.model,
        "model_weights_sha256": (
            fingerprint.model_weights_sha256 if fingerprint else None
        ),
        "model_fingerprint_path": (
            str(fingerprint.path) if fingerprint else None
        ),
        "prompt_variant_path": str(variant_prompt.path),
        "prompt_variant_sha256": variant_prompt.sha256,
        "prompt_user_path": str(user_prompt.path),
        "prompt_user_sha256": user_prompt.sha256,
        "decoding": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
        "server_endpoint": base_url,
        "host": socket.gethostname(),
        "dry_run": bool(args.dry_run),
        "n_rows_requested": args.n_rows,
        "sample_seed": args.seed,
    }
    manifest_path = out_dir / f"_run_manifest_{variant}_{stamp}.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("wrote variant manifest -> {}", manifest_path)
    return manifest_path


def _parse_variants(arg: str) -> list[str]:
    """Parse a comma-separated variants string into a deduplicated list."""
    labels = [token.strip() for token in arg.split(",") if token.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("--variants must list at least one label")
    if len(labels) > 3:
        raise argparse.ArgumentTypeError(
            f"--variants accepts at most 3 labels, got {len(labels)}"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        if label in seen:
            raise argparse.ArgumentTypeError(f"duplicate variant label: {label!r}")
        seen.add(label)
        ordered.append(label)
    return ordered


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variants",
        type=_parse_variants,
        default=["v_compact", "v_explicit"],
        help=(
            "Comma-separated list of 2 or 3 variant labels. Each label resolves "
            "to prompts/adjudicate_system_<label>.txt. Default: v_compact,v_explicit."
        ),
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=DEFAULT_N_ROWS,
        help=f"Subsample size (default: {DEFAULT_N_ROWS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"Seed for the stratified subsample RNG (default: GLOBAL_SEED={GLOBAL_SEED}).",
    )
    parser.add_argument(
        "--gallery_manifest",
        type=Path,
        default=DEFAULT_GALLERY_MANIFEST,
        help="Path to results/gallery/gallery_manifest.csv.",
    )
    parser.add_argument(
        "--gallery_root",
        type=Path,
        default=Path("."),
        help=(
            "Directory the manifest's relative image_path values are anchored to "
            "(default: current working directory, which is the repo root)."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for per-variant CSV/parquet and manifest JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=SERVED_MODEL_ID,
        help=f"Served model id (default: {SERVED_MODEL_ID}).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Server base URL (default: $OMLX_BASE_URL or http://localhost:<port>/v1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help=(
            "Port to assemble a default base URL when --base_url is not set "
            "and $OMLX_BASE_URL is unset (default: 8000)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Decoding temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Maximum new tokens (default: {MAX_TOKENS}).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use an in-process stub client (no server, no network).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume each variant from its checkpoint file (default).",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Ignore any existing checkpoints and rerun the subsample from scratch.",
    )
    return parser.parse_args(argv)


def _resolve_base_url(args: argparse.Namespace) -> str:
    """Resolve the server base URL honoring --base_url, $OMLX_BASE_URL, or --port."""
    if args.base_url:
        return args.base_url
    env_url = os.environ.get("OMLX_BASE_URL")
    if env_url:
        return env_url
    return f"http://localhost:{args.port}/v1"


def _run_one_variant(
    *,
    label: str,
    subsample: pl.DataFrame,
    args: argparse.Namespace,
    base_url: str,
    fingerprint: ModelFingerprint | None,
    user_prompt: LoadedPrompt,
) -> None:
    """Run all subsample rows under one variant prompt and write outputs."""
    variant_prompt = load_variant_prompt(label)
    logger.info(
        "variant={} system_sha256={} user_sha256={} body_bytes={}",
        label,
        variant_prompt.sha256,
        user_prompt.sha256,
        variant_prompt.body_bytes,
    )

    checkpoint_path = _checkpoint_path_for(args.out_dir, label)
    if args.resume:
        existing_rows = _read_checkpoint_rows(checkpoint_path)
    else:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.warning(
                "removed existing checkpoint for fresh --no_resume run: {}",
                checkpoint_path,
            )
        existing_rows = []

    completed_card_ids = {
        str(r["card_id"]) for r in existing_rows if r.get("card_id") is not None
    }
    if completed_card_ids:
        before = subsample.height
        remaining = subsample.filter(~pl.col("card_id").is_in(completed_card_ids))
        logger.info(
            "variant={} resume: loaded {} completed rows, skipping {}, remaining {}",
            label,
            len(existing_rows),
            before - remaining.height,
            remaining.height,
        )
    else:
        remaining = subsample

    rows: list[dict] = list(existing_rows)
    run_utc_start = _utc_now_iso()
    stamp = _utc_now_stamp()

    if remaining.height == 0:
        if not rows:
            logger.warning("variant={}: nothing to do (no subsample rows)", label)
            return
        logger.info("variant={}: all subsample rows already checkpointed", label)
    else:
        client = _build_client(args, base_url)
        for record in remaining.iter_rows(named=True):
            row = adjudicate_one_card(
                client,
                record,
                system_prompt=variant_prompt,
                user_prompt=user_prompt,
                gallery_root=args.gallery_root,
                base_url=base_url,
                args=args,
                fingerprint=fingerprint,
            )
            rows.append(row)
            _append_checkpoint_row(row, checkpoint_path)

    csv_path, parquet_path = _write_variant_outputs(rows, args.out_dir, variant=label)
    logger.info("wrote variant outputs -> {} and {}", csv_path, parquet_path)
    _write_variant_manifest(
        rows=rows,
        out_dir=args.out_dir,
        variant=label,
        variant_prompt=variant_prompt,
        user_prompt=user_prompt,
        stamp=stamp,
        run_utc_start=run_utc_start,
        run_utc_end=_utc_now_iso(),
        base_url=base_url,
        args=args,
        fingerprint=fingerprint,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the prompt-sensitivity sweep over the requested variants.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on input errors).
    """
    args = _parse_args(argv)
    _load_dotenv_if_present()

    if not args.gallery_manifest.exists():
        logger.error("gallery manifest not found: {}", args.gallery_manifest)
        return 2

    # Load the unchanged user prompt and SHA-verify each variant up front, so a
    # malformed variant fails before any model is called.
    user_prompt = load_prompt(ADJUDICATE_USER_PROMPT_FILE)
    logger.info(
        "user prompt loaded: path={} sha256={}",
        user_prompt.path,
        user_prompt.sha256,
    )
    for label in args.variants:
        path = resolve_variant_prompt_path(label)
        if not path.exists():
            logger.error("variant prompt missing: {}", path)
            return 2
        load_prompt(path)  # Raises PromptIntegrityError on drift.

    base_url = resolve_base_url(_resolve_base_url(args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = load_latest_fingerprint(args.out_dir)
    if fingerprint is None:
        # Fall back to the canonical medgemma results dir, where script 41
        # places fingerprints.
        fingerprint = load_latest_fingerprint(Path("results/medgemma"))
    if fingerprint is None:
        logger.warning(
            "no model fingerprint recorded; run scripts/compute_model_sha.sh "
            "before this run"
        )
    else:
        logger.info(
            "model fingerprint: id={} sha256={} computed_utc={}",
            fingerprint.model_id,
            fingerprint.model_weights_sha256,
            fingerprint.computed_utc,
        )

    manifest = pl.read_csv(args.gallery_manifest, infer_schema_length=20000)
    try:
        subsample = stratified_subsample(
            manifest, n_rows=args.n_rows, seed=args.seed
        )
    except ValueError as exc:
        logger.error("subsample failed: {}", exc)
        return 2
    logger.info(
        "subsampled {} cards (seed={}) from manifest of {} cards",
        subsample.height,
        args.seed,
        manifest.height,
    )

    for label in args.variants:
        _run_one_variant(
            label=label,
            subsample=subsample,
            args=args,
            base_url=base_url,
            fingerprint=fingerprint,
            user_prompt=user_prompt,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
