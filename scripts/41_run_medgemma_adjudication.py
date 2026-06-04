"""Blinded MedGemma adjudication over candidate cuff-occlusion waveforms (step 41).

This batch driver renders an unannotated perfusion-index-versus-time plot for
each *evaluable* candidate event (those that have a co-recorded PPG channel) and
asks a local MedGemma model, blinded to the deterministic detector's verdict,
whether the trace shows the occlusion signature: a single sustained deep drop
in PI followed by a graded reperfusion recovery. MedGemma's verdict is a
disclosed, secondary AI cross-read; the pre-registered detector remains the
primary classifier.

The model call is a plain client to a local OpenAI-compatible server (oMLX);
the result-generating path is a single deterministic chat-completions call with
no autonomous tooling around it. The image
shown to the model and the prompt around it contain no detector verdict, no
phase duration, no laterality word, and no axis annotation that could leak the
answer; blinding is enforced at render time.

Vocabulary
----------
Calls are drawn from
``{"occlusion_signature_present", "no_occlusion_signature", "indeterminate"}``.
The legacy ``"ipsilateral"`` / ``"not_ipsilateral"`` vocabulary is rejected at
parse time so a stale prompt cannot silently pollute the canonical tally.

Provenance and reproducibility
------------------------------
* Decoding is ``temperature=0`` with a fixed ``seed`` and a generous
  ``max_tokens`` so MedGemma's Gemma-style reasoning preamble does not get
  truncated before the JSON answer.
* The system and user prompts are loaded from frozen files under ``prompts/``
  whose first line stamps a SHA-256 of the body; a drift between stamped and
  recomputed digest stops the run before a model is called.
* Per row, the log records the served model id, prompt SHA-256, image
  SHA-256, base URL, temperature, seed, and a UTC timestamp.
* At the end of each run a manifest JSON is written summarizing counts,
  prompt and model fingerprints, decoding parameters, and the server endpoint.
  The served-model fingerprint comes from the most recent
  ``_model_fingerprint_<utc_iso>.json`` under ``--out``, produced out-of-band
  by ``scripts/compute_model_sha.sh``; when absent, the run still proceeds and
  the manifest records that fact rather than silently substituting a guess.

Inputs (read-only)
------------------
The candidate inventory and the WDB waveform tree live outside this repository
and are never copied in (PhysioNet DUA). Point ``--inventory`` and
``--wdb-root`` at the credentialed copies. The evaluable pool excludes events
whose ``reject_reason`` is ``no_pleth`` (no PPG to look at).

Outputs
-------
A stable checkpoint CSV under ``--out`` is appended and fsynced after each row,
then a timestamped run log (CSV + parquet) and a per-run manifest JSON are
written at successful completion. All outputs contain derived fields only; no
raw waveform samples and no note text are written. Working plots are rendered
to a gitignored scratch directory and are PI-derived, not raw signal.

Examples
--------
Pilot (two known 15 s survivors plus a handful of negative-but-evaluable
controls)::

    uv run python scripts/41_run_medgemma_adjudication.py --stage pilot

Full evaluable pool::

    uv run python scripts/41_run_medgemma_adjudication.py --stage full

Resume is on by default and skips row IDs already present in the checkpoint::

    uv run python scripts/41_run_medgemma_adjudication.py --stage full --resume
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger  # pyright: ignore[reportMissingImports]

from cuffcrt import figstyle
from cuffcrt._paths import (
    ENV_INVENTORY,
    ENV_WDB_ROOT,
    DataPathNotConfiguredError,
    resolve_configured_path,
)
from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.llm.client import ChatClient, OMLXClient, StubClient, resolve_base_url
from cuffcrt.llm.medgemma import (
    DEFAULT_TEMPERATURE,
    RUN_LOG_COLUMNS,
    AdjudicationResult,
    adjudication_log_row,
    build_adjudicate_messages,
    parse_adjudication_json,
    prompt_sha256,
)
from cuffcrt.llm.model_fingerprint import ModelFingerprint, load_latest_fingerprint
from cuffcrt.llm.prompts import (
    ADJUDICATE_SYSTEM_PROMPT_FILE,
    ADJUDICATE_USER_PROMPT_FILE,
    load_adjudicate_prompts,
)
from cuffcrt.signal.cuff_event_detector import compute_pi_1hz

# Served model id on the local oMLX server. oMLX drops the ``mlx-community/``
# prefix that the harness default carries, so the call must name the served id
# exactly or the server returns 404.
SERVED_MODEL_ID = "medgemma-1.5-4b-it-bf16"

# Generous token budget: MedGemma emits a long Gemma-style reasoning preamble
# (``<unused94>thought ...``) before the JSON answer. The pilot showed that 768
# tokens occasionally truncates the JSON mid-object after a verbose preamble, so
# the budget is set well above the observed worst case (~3k chars of preamble).
MAX_TOKENS = 1536

# WDB on-disk layout: waves/<first4 of subject>/<subject>/<record>/<record>.*
WDB_WAVES_SUBDIR = "waves"

# Pre/post window around the charted NBP timestamp, matching step 20 so the
# rendered PI(t) is the same trace the detector classified.
PRE_WINDOW_S = 200.0
POST_WINDOW_S = 200.0

# Default scratch directory for working plots (gitignored under data/).
DEFAULT_SCRATCH = Path("data/scratch_medgemma_plots")
DEFAULT_OUT = Path("results/medgemma")
LOG_COLUMNS_WITH_IDS = ("subject_id", "record_id", *RUN_LOG_COLUMNS)

# Negative/excluded-but-evaluable controls for the pilot. One per reject family
# present in the evaluable pool, chosen for variety, none of them survivors.
PILOT_CONTROL_REASONS = (
    "pre_window_unstable",
    "no_phase2",
    "pre_pi_implausible",
    "phase2_misaligned",
    "pleth_mostly_nan",
    "stat_mode_short_phase3",
)


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
    """Return the current UTC time as a filename-safe stamp (Y M D T H M S Z)."""
    return dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_record_dir(wdb_root: Path, subject_id: str, record_id: str) -> Path:
    """Resolve the on-disk record directory for a subject/record.

    Parameters
    ----------
    wdb_root : pathlib.Path
        Root of the WDB tree (the directory containing ``waves/``).
    subject_id : str
        Subject directory name, for example ``pXXXXXXXX``.
    record_id : str
        Record (study) id, for example ``XXXXXXXX``.

    Returns
    -------
    pathlib.Path
        Directory containing the record's master header and numerics CSV.
    """
    return wdb_root / WDB_WAVES_SUBDIR / subject_id[:4] / subject_id / record_id


def parse_master_fs(master_hea: Path) -> float:
    """Pull the master frame rate from the first data line of a master header."""
    with master_hea.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Unexpected master header line: {line!r}")
            return float(parts[2].split("/")[0])
    raise ValueError(f"No data line in master header {master_hea}")


def slice_pleth_window(
    record_basename: Path,
    master_fs: float,
    t_center_s: float,
) -> tuple[np.ndarray, float] | None:
    """Read a PLETH window around ``t_center_s`` at the channel-native rate.

    ``smooth_frames=False`` keeps PLETH at its native rate so the cardiac
    component survives (averaging to the master rate cancels it).

    Returns
    -------
    tuple[numpy.ndarray, float] or None
        ``(pleth_signal, native_fs)`` or ``None`` if unavailable.
    """
    import wfdb  # pyright: ignore[reportMissingImports]

    sampfrom = max(0, int((t_center_s - PRE_WINDOW_S) * master_fs))
    sampto = int((t_center_s + POST_WINDOW_S) * master_fs)
    try:
        record = wfdb.rdrecord(
            str(record_basename),
            sampfrom=sampfrom,
            sampto=sampto,
            channel_names=["Pleth"],
            smooth_frames=False,
            return_res=32,
        )
    except Exception as exc:  # noqa: BLE001 - wfdb raises a wide variety
        logger.debug("rdrecord failed at t={:.0f}s: {}", t_center_s, exc)
        return None
    # wfdb stubs union rdrecord's return as Record | MultiRecord; our call shape
    # (single channel, return_res=32) always yields Record at runtime.
    if record.e_p_signal is None or len(record.e_p_signal) == 0:  # pyright: ignore[reportAttributeAccessIssue]
        return None
    pleth = np.asarray(record.e_p_signal[0])  # pyright: ignore[reportAttributeAccessIssue]
    if pleth.size == 0 or not np.isfinite(pleth).any():
        return None
    samps_per_frame = record.samps_per_frame[0] if record.samps_per_frame is not None else 1  # pyright: ignore[reportAttributeAccessIssue]
    fs_native = float(record.fs) * float(samps_per_frame)  # pyright: ignore[reportArgumentType]
    return pleth, fs_native


def render_blinded_pi_plot(
    pi: np.ndarray,
    t_pi: np.ndarray,
    out_path: Path,
) -> None:
    """Render a blinded PI(t) plot to ``out_path``.

    The plot carries no detector verdict, no phase band, no event marker, no
    laterality word, and no leading title: only a generic perfusion-index trace
    against time. This is what the model sees.

    Parameters
    ----------
    pi : numpy.ndarray
        Perfusion index at 1 Hz.
    t_pi : numpy.ndarray
        Integer-second time axis for ``pi``.
    out_path : pathlib.Path
        Destination PNG path (parent created if absent).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(t_pi, pi, color=figstyle.INK, linewidth=1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Perfusion index (%)")
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# Filename of the per-card gallery manifest written by scripts/51. It carries
# the (subject_id, record_id, t_nbp) triple and the per-card PNG path, which is
# how a pre-rendered card is matched back to an inventory row.
GALLERY_MANIFEST_NAME = "gallery_manifest.csv"

# Rounding for the t_nbp join key. The manifest stores t_nbp as a float and the
# inventory carries nbp_timestamp_s as a float; rounding to milliseconds makes
# the triple key robust to trailing float-formatting noise without colliding
# distinct charted cycles (NIBP cycles are minutes apart, never sub-second).
_GALLERY_T_NBP_DECIMALS = 3


def _gallery_triple_key(subject_id: str, record_id: str, t_nbp: float) -> tuple[str, str, float]:
    """Return the ``(subject_id, record_id, rounded_t_nbp)`` gallery join key."""
    return (str(subject_id), str(record_id), round(float(t_nbp), _GALLERY_T_NBP_DECIMALS))


def build_gallery_lookup(gallery_dir: Path) -> dict[tuple[str, str, float], Path]:
    """Map each pre-rendered gallery card to its on-disk PNG by source triple.

    The gallery manifest written by ``scripts/51`` keys cards by ``card_id`` (a
    stratum-prefixed hash) and by the natural triple ``(subject_id, record_id,
    t_nbp)``. The MedGemma run log keys rows by ``row_id``
    (``subject_id_record_id_idx``), so the triple is the only stable join
    between an inventory row and a pre-rendered card (this is the same key
    ``scripts/44`` uses). The PNG is resolved as
    ``<gallery_dir>/<stratum>/<card_id>.png`` so the lookup is independent of
    where the harness is invoked from, and the on-disk PNG must exist to be
    included (cards whose render was skipped have a null ``image_sha256`` and no
    file).

    Parameters
    ----------
    gallery_dir : pathlib.Path
        Directory holding ``gallery_manifest.csv`` and the per-stratum PNG
        subdirectories (for example ``results/gallery``).

    Returns
    -------
    dict[tuple[str, str, float], pathlib.Path]
        Mapping from ``(subject_id, record_id, rounded_t_nbp)`` to the absolute
        PNG path. Only cards with a PNG present on disk are included.

    Raises
    ------
    FileNotFoundError
        If ``gallery_dir/gallery_manifest.csv`` is absent.
    """
    manifest_path = gallery_dir / GALLERY_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"gallery manifest not found: {manifest_path}")
    manifest = pl.read_csv(manifest_path, infer_schema_length=20000)
    lookup: dict[tuple[str, str, float], Path] = {}
    n_missing_png = 0
    for card in manifest.iter_rows(named=True):
        png_path = gallery_dir / str(card["stratum"]) / f"{card['card_id']}.png"
        if not png_path.exists():
            n_missing_png += 1
            continue
        key = _gallery_triple_key(card["subject_id"], card["record_id"], card["t_nbp"])
        lookup[key] = png_path
    logger.info(
        "gallery lookup: {} cards in manifest, {} resolved to on-disk PNGs "
        "({} manifest rows had no rendered PNG)",
        manifest.height,
        len(lookup),
        n_missing_png,
    )
    return lookup


def resolve_gallery_png(
    lookup: dict[tuple[str, str, float], Path],
    subject_id: str,
    record_id: str,
    t_nbp: float,
) -> Path | None:
    """Return the pre-rendered PNG for one inventory row, or ``None`` if absent.

    Parameters
    ----------
    lookup : dict
        Triple-keyed mapping from :func:`build_gallery_lookup`.
    subject_id, record_id : str
        Identifiers for the inventory row.
    t_nbp : float
        Charted NBP timestamp in seconds from record start.

    Returns
    -------
    pathlib.Path or None
        The on-disk PNG path when a pre-rendered card matches the row's triple,
        otherwise ``None``.
    """
    return lookup.get(_gallery_triple_key(subject_id, record_id, t_nbp))


def build_event_frame(inventory_path: Path) -> pl.DataFrame:
    """Read the inventory and add a stable per-row id and an evaluable flag.

    Parameters
    ----------
    inventory_path : pathlib.Path
        Candidate event inventory CSV.

    Returns
    -------
    polars.DataFrame
        The inventory with ``row_id`` and ``evaluable`` columns added.
    """
    df = pl.read_csv(inventory_path, infer_schema_length=20000)
    df = df.with_row_index(name="_idx")
    df = df.with_columns(
        row_id=pl.format(
            "{}_{}_{}",
            pl.col("subject_id"),
            pl.col("record_id"),
            pl.col("_idx"),
        ),
        # A null reject_reason marks a clean survivor (it has co-recorded PPG),
        # so it is evaluable. Comparing null with != yields null under Polars'
        # three-valued logic, which would silently drop those rows; fill the
        # null comparison result with True so survivors stay in the pool.
        evaluable=(pl.col("reject_reason") != "no_pleth").fill_null(True),
    )
    return df


def select_pool(df: pl.DataFrame, *, stage: str) -> pl.DataFrame:
    """Select the events to adjudicate for the requested stage.

    ``full`` returns the whole evaluable pool. ``pilot`` returns the two known
    15 s survivors plus a small set of negative/excluded-but-evaluable controls
    (one per reject family present), all blinded identically.

    Parameters
    ----------
    df : polars.DataFrame
        Inventory frame from :func:`build_event_frame`.
    stage : str
        ``"pilot"`` or ``"full"``.

    Returns
    -------
    polars.DataFrame
        Selected events.
    """
    evaluable = df.filter(pl.col("evaluable"))
    if stage == "full":
        return evaluable

    # ``is_occlusion_signature`` is the canonical detector-positive column on
    # the per-event parquets (renamed from ``ipsilateral`` during the
    # 2026-05-22 canonicalization pass). Fall back to a tolerant lookup so the
    # pilot still runs against test fixtures that may omit the column.
    # de-identified MIMIC-IV-WDB pseudo-IDs, used under the PhysioNet DUA
    survivor_mask = pl.col("subject_id").is_in(["p10014354", "p10079700"]) & (
        pl.col("phase3_duration_s") >= 15.0
    )
    if "is_occlusion_signature" in evaluable.columns:
        survivor_mask = survivor_mask & pl.col("is_occlusion_signature")
    elif "ipsilateral" in evaluable.columns:
        survivor_mask = survivor_mask & pl.col("ipsilateral")

    survivors = evaluable.filter(survivor_mask)
    controls: list[pl.DataFrame] = []
    for reason in PILOT_CONTROL_REASONS:
        hit = evaluable.filter(pl.col("reject_reason") == reason).head(2)
        if hit.height:
            controls.append(hit)
    control_df = pl.concat(controls) if controls else evaluable.head(0)
    selected = pl.concat([survivors, control_df]).unique(subset=["row_id"], keep="first")
    return selected.sort("row_id")


def _build_client(args: argparse.Namespace):
    """Construct the stub (dry-run) or live oMLX client with the served model id."""
    base_url = resolve_base_url(args.base_url)
    if args.dry_run:
        logger.info("dry-run: in-process stub client (no server, no network)")
        return StubClient(model=args.model, base_url=base_url)
    logger.info("live: oMLX client model={} base_url={}", args.model, base_url)
    return OMLXClient(model=args.model, base_url=base_url)


def _log_fingerprint(fingerprint: ModelFingerprint | None) -> None:
    """Log the loaded model fingerprint, or warn that none is present."""
    if fingerprint is None:
        logger.warning(
            "no model fingerprint recorded; run scripts/compute_model_sha.sh "
            "to fingerprint the served weights before this run"
        )
        return
    logger.info(
        "model fingerprint: id={} sha256={} computed_utc={} (files={})",
        fingerprint.model_id,
        fingerprint.model_weights_sha256,
        fingerprint.computed_utc,
        fingerprint.files_hashed,
    )


def _failed_row_for(
    row_id: str,
    *,
    subject_id: str,
    record_id: str,
    base_url: str,
    args: argparse.Namespace,
    fingerprint: ModelFingerprint | None,
    parse_error: str,
    client_model: str | None = None,
    image_path: str = "",
    image_sha256: str | None = None,
) -> dict:
    """Build an uncallable (parse-failed) run-log row with full provenance.

    A failed row keeps the pool tally complete and records why a cycle could
    not be adjudicated (missing header, no PPG window, mostly-NaN pleth, or, on
    the ``--gallery-dir`` skip path, an absent pre-rendered card).

    Parameters
    ----------
    client_model : str or None
        The served model id. ``None`` when no client was built (the gallery
        skip path can fail a row before the live client is constructed); the
        run log then records the requested model id instead.
    """
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
        row_id=row_id,
        model=client_model if client_model is not None else args.model,
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
    base["subject_id"] = subject_id
    base["record_id"] = record_id
    base["model_weights_sha256"] = (
        fingerprint.model_weights_sha256 if fingerprint else None
    )
    return base


def _adjudicate_image_bytes(
    image_bytes: bytes,
    *,
    client: ChatClient,
    row_id: str,
    subject_id: str,
    record_id: str,
    image_path_name: str,
    base_url: str,
    args: argparse.Namespace,
    fingerprint: ModelFingerprint | None,
) -> dict:
    """Adjudicate one already-encoded PNG and return a run-log row.

    This is the shared tail of the on-the-fly-render path and the
    pre-rendered-gallery path: both arrive here with PNG bytes in hand and from
    here the prompt assembly, model call, parse, and log-row build are
    identical.

    Parameters
    ----------
    image_bytes : bytes
        Encoded PNG shown to the model.
    image_path_name : str
        The image path recorded in the run log. For an on-the-fly render this
        is the scratch filename; for a pre-rendered card it is the gallery PNG
        path so provenance is auditable.
    """
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    messages = build_adjudicate_messages(image_bytes)
    sha = prompt_sha256(messages)
    raw = client.complete(
        messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    result = parse_adjudication_json(raw)
    log_row = adjudication_log_row(
        row_id=row_id,
        model=client.model,
        base_url=base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prompt_sha=sha,
        run_utc=_utc_now_iso(),
        image_path=image_path_name,
        image_sha256=image_sha,
        result=result,
    )
    log_row["subject_id"] = subject_id
    log_row["record_id"] = record_id
    log_row["model_weights_sha256"] = (
        fingerprint.model_weights_sha256 if fingerprint else None
    )
    logger.info(
        "{}: parsed_ok={} schema_complete={} call={} conf={}",
        row_id,
        result.parsed_ok,
        result.schema_complete,
        result.call,
        result.confidence,
    )
    return log_row


def _adjudicate_one(
    client: ChatClient,
    row: dict,
    *,
    wdb_root: Path | None,
    scratch_dir: Path,
    base_url: str,
    args: argparse.Namespace,
    fingerprint: ModelFingerprint | None,
    gallery_lookup: dict[tuple[str, str, float], Path] | None = None,
) -> dict:
    """Render one blinded plot, adjudicate it, and return a run-log row.

    On any failure to obtain or render the PPG, a parse-failed row is returned
    so the pool tally stays complete and the reason is auditable.

    When ``gallery_lookup`` is supplied (the ``--gallery-dir`` path), the
    pre-rendered gallery PNG for this row is read and shown to the model
    verbatim so the reader and the model view pixel-identical images. Rows with
    no pre-rendered card follow ``args.gallery_missing``: ``"fallback"`` renders
    on the fly exactly as the default path does (logged), and ``"skip"`` returns
    an uncallable row recording the absence so the pool tally stays complete.
    Without ``gallery_lookup`` the on-the-fly render path is unchanged, which is
    what the canonical run used.
    """
    subject_id = str(row["subject_id"])
    record_id = str(row["record_id"])
    row_id = str(row["row_id"])
    t_nbp = float(row["nbp_timestamp_s"])

    if gallery_lookup is not None:
        gallery_png = resolve_gallery_png(gallery_lookup, subject_id, record_id, t_nbp)
        if gallery_png is not None:
            image_bytes = gallery_png.read_bytes()
            return _adjudicate_image_bytes(
                image_bytes,
                client=client,
                row_id=row_id,
                subject_id=subject_id,
                record_id=record_id,
                image_path_name=str(gallery_png),
                base_url=base_url,
                args=args,
                fingerprint=fingerprint,
            )
        if args.gallery_missing == "skip":
            logger.warning("{}: no pre-rendered gallery PNG; skipping", row_id)
            return _failed_row_for(
                row_id,
                subject_id=subject_id,
                record_id=record_id,
                base_url=base_url,
                args=args,
                fingerprint=fingerprint,
                parse_error="gallery png missing (skipped)",
            )
        logger.info("{}: no pre-rendered gallery PNG; rendering on the fly", row_id)

    if wdb_root is None:  # unreachable in skip mode; guard the on-the-fly path
        raise DataPathNotConfiguredError(
            "WDB waveform record tree is required to render a plot on the fly; "
            f"pass --wdb-root or set the {ENV_WDB_ROOT} environment variable."
        )
    record_dir = resolve_record_dir(wdb_root, subject_id, record_id)
    master_hea = record_dir / f"{record_id}.hea"
    record_basename = record_dir / record_id

    def _failed_row(
        parse_error: str, *, image_path: str = "", image_sha256: str | None = None
    ) -> dict:
        return _failed_row_for(
            row_id,
            subject_id=subject_id,
            record_id=record_id,
            base_url=base_url,
            args=args,
            fingerprint=fingerprint,
            client_model=client.model,
            parse_error=parse_error,
            image_path=image_path,
            image_sha256=image_sha256,
        )

    if not master_hea.exists():
        logger.warning("{}: master header missing at {}", row_id, master_hea)
        return _failed_row(f"master header missing: {master_hea}")

    try:
        master_fs = parse_master_fs(master_hea)
        windowed = slice_pleth_window(record_basename, master_fs, t_nbp)
    except Exception as exc:  # noqa: BLE001 - render path must never abort the run
        logger.warning("{}: slice failed: {}", row_id, exc)
        return _failed_row(f"slice failed: {exc}")

    if windowed is None:
        return _failed_row("no pleth window")
    pleth, fs_native = windowed
    finite_mask = np.isfinite(pleth)
    if finite_mask.mean() < 0.5:
        return _failed_row("pleth mostly nan")
    pleth_clean = np.where(finite_mask, pleth, np.nanmedian(pleth))

    t_pi, pi = compute_pi_1hz(pleth_clean, fs_native)
    if pi.size == 0:
        return _failed_row("empty PI")

    image_path = scratch_dir / f"{row_id}.png"
    try:
        render_blinded_pi_plot(pi, t_pi, image_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("{}: render failed: {}", row_id, exc)
        return _failed_row(f"render failed: {exc}")

    image_bytes = image_path.read_bytes()
    return _adjudicate_image_bytes(
        image_bytes,
        client=client,
        row_id=row_id,
        subject_id=subject_id,
        record_id=record_id,
        image_path_name=image_path.name,
        base_url=base_url,
        args=args,
        fingerprint=fingerprint,
    )


def default_checkpoint_path(out_dir: Path, *, stage: str) -> Path:
    """Return the stable per-stage checkpoint path used for resumable runs."""
    return out_dir / f"medgemma_adjudication_{stage}_checkpoint.csv"


def _read_checkpoint_rows(checkpoint_path: Path) -> list[dict]:
    """Read an existing checkpoint CSV, validating the expected run-log columns."""
    if not checkpoint_path.exists():
        return []
    if checkpoint_path.stat().st_size == 0:
        return []
    try:
        df = pl.read_csv(checkpoint_path, infer_schema_length=20000)
    except Exception as exc:  # noqa: BLE001 - malformed checkpoints should fail clearly
        raise ValueError(f"could not read checkpoint {checkpoint_path}: {exc}") from exc
    missing = [col for col in LOG_COLUMNS_WITH_IDS if col not in df.columns]
    if missing:
        raise ValueError(f"checkpoint {checkpoint_path} is missing columns: {missing}")
    return df.select(list(LOG_COLUMNS_WITH_IDS)).to_dicts()


def _validate_checkpoint_compatible(
    rows: list[dict],
    *,
    checkpoint_path: Path,
    args: argparse.Namespace,
    base_url: str,
) -> None:
    """Refuse to resume from a checkpoint created with incompatible run params."""
    if not rows:
        return
    expected = {
        "model": args.model,
        "base_url": base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    for row in rows:
        for key, value in expected.items():
            observed = row.get(key)
            if observed is not None and str(observed) != str(value):
                raise ValueError(
                    f"checkpoint {checkpoint_path} has {key}={observed!r}, "
                    f"but this run requested {value!r}; use --no-resume or a different "
                    "--checkpoint-csv to start a fresh run"
                )


def _append_checkpoint_row(row: dict, checkpoint_path: Path) -> None:
    """Append one completed row to the checkpoint CSV and fsync it to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0
    with checkpoint_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LOG_COLUMNS_WITH_IDS), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in LOG_COLUMNS_WITH_IDS})
        f.flush()
        os.fsync(f.fileno())


def _write_run_log(rows: list[dict], out_dir: Path, *, stage: str, stamp: str) -> tuple[Path, Path]:
    """Write the run log as CSV and parquet with subject/record id columns first.

    Polars defaults to ``infer_schema_length=100`` when constructing a frame
    from a list of dicts, which sniffs only the first 100 rows for type
    inference. In a long adjudication run the first ~150 rows of
    ``parse_error`` are ``None`` (parses succeed), so Polars locks that column
    in as the ``Null`` dtype. A later rare parse-failure row whose
    ``parse_error`` is a string (for example ``"pleth mostly nan"``) then
    crashes the builder. Passing ``infer_schema_length=None`` forces a full
    scan so any late-appearing string is observed and typed correctly. The
    cost is well under 100 ms for the ~9k-row full pool.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows, infer_schema_length=None).select(list(LOG_COLUMNS_WITH_IDS))
    csv_path = out_dir / f"medgemma_adjudication_{stage}_{stamp}.csv"
    parquet_path = out_dir / f"medgemma_adjudication_{stage}_{stamp}.parquet"
    df.write_csv(csv_path)
    df.write_parquet(parquet_path)
    return csv_path, parquet_path


def _summarize(rows: list[dict]) -> dict[str, int]:
    """Log a compact tally and return the per-call counts for the manifest."""
    n = len(rows)
    n_ok = sum(1 for r in rows if r["parsed_ok"])
    n_schema = sum(1 for r in rows if r.get("schema_complete"))
    calls: dict[str, int] = {}
    for r in rows:
        if r["parsed_ok"] and r["call"] is not None:
            calls[r["call"]] = calls.get(r["call"], 0) + 1
    rate = (n_ok / n * 100.0) if n else 0.0
    logger.info(
        "rows={} parsed_ok={} ({:.1f}%) schema_complete={}", n, n_ok, rate, n_schema
    )
    logger.info("call distribution: {}", calls)
    return calls


def _resume_only_manifest_window(
    rows: list[dict],
    *,
    fallback_start: str,
) -> tuple[str, str]:
    """Derive ``(run_utc_start, run_utc_end)`` for a resume-only manifest.

    Used when the live pool is empty because every selected row is already
    present in the checkpoint. The honest provenance window for the manifest
    is the time spanned by the checkpoint rows' ``run_utc`` values, not the
    recovery invocation's wall clock.

    Parameters
    ----------
    rows : list of dict
        Checkpoint rows in the run-log column schema. Each row is expected
        to carry a non-null ISO-8601 ``run_utc`` string.
    fallback_start : str
        ISO-8601 UTC timestamp to fall back to if any row is missing its
        ``run_utc``. Paired with :func:`_utc_now_iso` for ``run_utc_end`` so
        the manifest still writes (with a warning) instead of failing.

    Returns
    -------
    tuple of (str, str)
        ``(run_utc_start, run_utc_end)`` ISO-8601 UTC timestamps. ISO-8601
        strings sort lexicographically by time at the same offset, so
        ``min``/``max`` over the row strings yields the true window.
    """
    row_utcs = [r["run_utc"] for r in rows if r.get("run_utc")]
    if row_utcs and len(row_utcs) == len(rows):
        return min(row_utcs), max(row_utcs)
    logger.warning(
        "resume-only manifest: {} of {} rows missing run_utc; "
        "falling back to invocation timestamps",
        len(rows) - len(row_utcs),
        len(rows),
    )
    return fallback_start, _utc_now_iso()


def _write_run_manifest(
    *,
    rows: list[dict],
    out_dir: Path,
    stage: str,
    stamp: str,
    run_utc_start: str,
    run_utc_end: str,
    base_url: str,
    args: argparse.Namespace,
    fingerprint: ModelFingerprint | None,
    system_prompt_sha: str,
    user_prompt_sha: str,
) -> Path:
    """Write a per-run manifest JSON capturing counts, prompts, model, host."""
    n_total = len(rows)
    n_parsed = sum(1 for r in rows if r["parsed_ok"])
    n_schema_complete = sum(1 for r in rows if r.get("schema_complete"))
    n_present = sum(
        1 for r in rows if r["parsed_ok"] and r["call"] == "occlusion_signature_present"
    )
    n_absent = sum(
        1 for r in rows if r["parsed_ok"] and r["call"] == "no_occlusion_signature"
    )
    n_indeterminate = sum(
        1 for r in rows if r["parsed_ok"] and r["call"] == "indeterminate"
    )
    n_parse_failure = n_total - n_parsed
    n_uncallable = n_indeterminate + n_parse_failure
    n_callable = n_present + n_absent

    payload = {
        "stage": stage,
        "run_utc_start": run_utc_start,
        "run_utc_end": run_utc_end,
        "n_rows_total": n_total,
        "n_parsed": n_parsed,
        "n_schema_complete": n_schema_complete,
        "n_callable": n_callable,
        "n_uncallable": n_uncallable,
        "n_occlusion_signature_present": n_present,
        "n_no_occlusion_signature": n_absent,
        "n_indeterminate": n_indeterminate,
        "n_parse_failure": n_parse_failure,
        "model_id": args.model,
        "model_weights_sha256": (
            fingerprint.model_weights_sha256 if fingerprint else None
        ),
        "model_fingerprint_path": (
            str(fingerprint.path) if fingerprint else None
        ),
        "model_fingerprint_computed_utc": (
            fingerprint.computed_utc if fingerprint else None
        ),
        "prompt_system_path": str(ADJUDICATE_SYSTEM_PROMPT_FILE),
        "prompt_system_sha256": system_prompt_sha,
        "prompt_user_path": str(ADJUDICATE_USER_PROMPT_FILE),
        "prompt_user_sha256": user_prompt_sha,
        "decoding": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
        "server_endpoint": base_url,
        "host": socket.gethostname(),
        "dry_run": bool(args.dry_run),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"_run_manifest_{stamp}.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("wrote run manifest -> {}", manifest_path)
    return manifest_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=("pilot", "full"),
        default="pilot",
        help="Pilot (survivors + controls) or full evaluable pool.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help=(
            "Candidate event inventory CSV (read-only, outside the repo). "
            f"Defaults to the ${ENV_INVENTORY} environment variable; required "
            "if that is unset."
        ),
    )
    parser.add_argument(
        "--wdb-root",
        type=Path,
        default=None,
        help=(
            "Root of the WDB tree (the directory containing waves/). "
            f"Defaults to the ${ENV_WDB_ROOT} environment variable; required "
            "for any on-the-fly render if that is unset."
        ),
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_SCRATCH,
        help="Gitignored directory for working PI(t) plots.",
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory holding a pre-rendered gallery "
            "(gallery_manifest.csv + per-stratum PNG subdirs, e.g. "
            "results/gallery). When given, each row that has a pre-rendered "
            "card is shown that exact PNG instead of an on-the-fly render so "
            "the human reader and the model view pixel-identical images. "
            "Absent: the on-the-fly render path (the canonical-run default)."
        ),
    )
    parser.add_argument(
        "--gallery-missing",
        choices=("fallback", "skip"),
        default="fallback",
        help=(
            "Behavior for rows with no pre-rendered gallery card (only "
            "meaningful with --gallery-dir). 'fallback' (default) renders on "
            "the fly and logs it; 'skip' records an uncallable row. Ignored "
            "without --gallery-dir."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for the run log (CSV + parquet) and manifest.",
    )
    parser.add_argument(
        "--checkpoint-csv",
        type=Path,
        default=None,
        help=(
            "Stable CSV checkpoint to append after each completed row. "
            "Default: <out>/medgemma_adjudication_<stage>_checkpoint.csv."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=SERVED_MODEL_ID,
        help=f"Served model id (default: {SERVED_MODEL_ID}).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base URL (default: $OMLX_BASE_URL or http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Decoding temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Maximum new tokens (default: {MAX_TOKENS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"Decoding seed (default: GLOBAL_SEED={GLOBAL_SEED}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of events (debugging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use an in-process stub client (no server, no network).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from --checkpoint-csv by skipping completed row_id values (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore any existing checkpoint and rerun the selected pool from scratch.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Render blinded plots, adjudicate them with MedGemma, and write the run log.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on input errors).
    """
    args = _parse_args(argv)
    _load_dotenv_if_present()
    figstyle.apply_style()

    # Verify the frozen prompts before anything else. Drift here would change
    # the model's behavior; better to stop now than after a long run.
    system_prompt, user_prompt = load_adjudicate_prompts()
    logger.info(
        "prompts loaded: system_sha256={} user_sha256={}",
        system_prompt.sha256,
        user_prompt.sha256,
    )

    try:
        inventory = resolve_configured_path(
            args.inventory,
            env_var=ENV_INVENTORY,
            flag="--inventory",
            what="event inventory CSV",
        )
    except DataPathNotConfiguredError as exc:
        logger.error("{}", exc)
        return 2
    if not inventory.exists():
        logger.error("inventory not found: {}", inventory)
        return 2
    # The WDB tree is only read when a plot is rendered on the fly. In pure
    # gallery-dir mode with the skip policy no on-the-fly render ever happens,
    # so the (large, credentialed) WDB tree need not be mounted; require it in
    # every other case, including gallery-dir + fallback.
    wdb_needed = args.gallery_dir is None or args.gallery_missing == "fallback"
    wdb_root: Path | None = None
    if wdb_needed:
        try:
            wdb_root = resolve_configured_path(
                args.wdb_root,
                env_var=ENV_WDB_ROOT,
                flag="--wdb-root",
                what="WDB waveform record tree",
            )
        except DataPathNotConfiguredError as exc:
            logger.error("{}", exc)
            return 2
        if not wdb_root.exists():
            logger.error("WDB root not found: {}", wdb_root)
            return 2

    fingerprint = load_latest_fingerprint(args.out)
    _log_fingerprint(fingerprint)

    logger.info("stage={} inventory={} wdb_root={}", args.stage, inventory, wdb_root)
    df = build_event_frame(inventory)
    pool = select_pool(df, stage=args.stage)
    if args.limit is not None:
        pool = pool.head(args.limit)
    logger.info("selected {} events for stage={}", pool.height, args.stage)
    if pool.height == 0:
        logger.warning("no events selected; nothing to do")
        return 0

    base_url = resolve_base_url(args.base_url)
    checkpoint_path = args.checkpoint_csv or default_checkpoint_path(args.out, stage=args.stage)
    checkpoint_rows: list[dict] = []
    if args.resume:
        try:
            checkpoint_rows = _read_checkpoint_rows(checkpoint_path)
            _validate_checkpoint_compatible(
                checkpoint_rows,
                checkpoint_path=checkpoint_path,
                args=args,
                base_url=base_url,
            )
        except ValueError as exc:
            logger.error("{}", exc)
            return 2
        completed_row_ids = {
            str(row["row_id"]) for row in checkpoint_rows if row.get("row_id") is not None
        }
        if completed_row_ids:
            before = pool.height
            pool = pool.filter(~pl.col("row_id").is_in(completed_row_ids))
            logger.info(
                "resume checkpoint={} loaded_rows={} skipped={} remaining={}",
                checkpoint_path,
                len(checkpoint_rows),
                before - pool.height,
                pool.height,
            )
        else:
            logger.info("checkpoint={} has no completed rows yet", checkpoint_path)
    else:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.warning(
                "removed existing checkpoint for fresh --no-resume run: {}",
                checkpoint_path,
            )
        logger.info("resume disabled; checkpoint will be rewritten at {}", checkpoint_path)

    run_utc_start = _utc_now_iso()
    rows: list[dict] = list(checkpoint_rows) if args.resume else []
    stamp = _utc_now_stamp()
    if pool.height == 0:
        if rows:
            # Resume-only recovery: the pool was fully drained by the
            # checkpoint, so no MedGemma calls happen this invocation. Using
            # the invocation wall clock for ``run_utc_start`` /
            # ``run_utc_end`` would mis-attribute the provenance window to the
            # recovery moment instead of the original adjudication run.
            # Derive the honest window from the checkpoint rows' ``run_utc``
            # values; fall back to the invocation timestamps only if the
            # checkpoint is missing ``run_utc`` (defensive).
            manifest_run_utc_start, manifest_run_utc_end = (
                _resume_only_manifest_window(
                    rows, fallback_start=run_utc_start
                )
            )
            csv_path, parquet_path = _write_run_log(
                rows, args.out, stage=args.stage, stamp=stamp
            )
            _summarize(rows)
            _write_run_manifest(
                rows=rows,
                out_dir=args.out,
                stage=args.stage,
                stamp=stamp,
                run_utc_start=manifest_run_utc_start,
                run_utc_end=manifest_run_utc_end,
                base_url=base_url,
                args=args,
                fingerprint=fingerprint,
                system_prompt_sha=system_prompt.sha256,
                user_prompt_sha=user_prompt.sha256,
            )
            logger.info(
                "all selected rows already checkpointed; wrote -> {} and {}",
                csv_path,
                parquet_path,
            )
        else:
            logger.warning("no uncheckpointed events selected; nothing to do")
        return 0

    gallery_lookup: dict[tuple[str, str, float], Path] | None = None
    if args.gallery_dir is not None:
        try:
            gallery_lookup = build_gallery_lookup(args.gallery_dir)
        except FileNotFoundError as exc:
            logger.error("{}", exc)
            return 2
        logger.info(
            "gallery-dir mode: reading pre-rendered PNGs from {} "
            "(missing-card policy: {})",
            args.gallery_dir,
            args.gallery_missing,
        )

    client = _build_client(args)
    scratch_dir = args.scratch_dir / args.stage

    for record in pool.iter_rows(named=True):
        row = _adjudicate_one(
            client,
            record,
            wdb_root=wdb_root,
            scratch_dir=scratch_dir,
            base_url=base_url,
            args=args,
            fingerprint=fingerprint,
            gallery_lookup=gallery_lookup,
        )
        rows.append(row)
        _append_checkpoint_row(row, checkpoint_path)

    csv_path, parquet_path = _write_run_log(
        rows, args.out, stage=args.stage, stamp=stamp
    )
    _summarize(rows)
    run_utc_end = _utc_now_iso()
    _write_run_manifest(
        rows=rows,
        out_dir=args.out,
        stage=args.stage,
        stamp=stamp,
        run_utc_start=run_utc_start,
        run_utc_end=run_utc_end,
        base_url=base_url,
        args=args,
        fingerprint=fingerprint,
        system_prompt_sha=system_prompt.sha256,
        user_prompt_sha=user_prompt.sha256,
    )
    logger.info("wrote run log -> {} and {}", csv_path, parquet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
