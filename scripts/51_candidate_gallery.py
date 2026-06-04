"""Build the blinded reader-adjudication gallery for the human reference task.

The pre-registered detector is the primary classifier; this gallery produces
the blinded human reference against which both the detector and MedGemma are
later scored (precision/recall, agreement). Each card is an unannotated 1 Hz
perfusion-index trace around a charted noninvasive blood-pressure timestamp,
sampled across three strata:

A. **detector_positive**: every event the canonical detector flagged with
   ``is_occlusion_signature == True``. The pre-registration asked for at least
   200; the current event parquets contain 268, so the gallery adjudicates all
   268 (sampling not needed).

B. **detector_rejected_near_miss**: events that passed the sub-0.50 envelope
   for at least 10 s AND had a nadir reaching below 0.20 of the pre-cuff
   baseline, but which the detector then rejected for the recovery or
   short-phase-3 criterion. These are the "would-be" candidates and form the
   recall stratum: did the detector wrongly throw them out?

C. **detector_negative_random**: events that did not qualify on the sub-0.50
   envelope criterion at all (clearly-no-event QC-pass cycles). Specificity
   check.

Anchor-free rendering
---------------------
Every card plot is blinded: no detector phase markers, no charted-BP marker,
no occlusion or release annotation, no subject id anywhere on the image. The
axes are labeled only ``Time (s)`` and ``PI (norm)``. The window is
``[-60, +90] s`` around the charted BP timestamp. Matplotlib rcParams are set
explicitly so the rendering is bit-reproducible.

Determinism and provenance
--------------------------
* ``cuffcrt._seed.GLOBAL_SEED`` seeds the per-stratum sampler so the same
  inputs always produce the same gallery.
* Each card's ``card_id`` is a deterministic short hash of
  ``(subject_id, t_nbp, stratum, GLOBAL_SEED)`` so the reader form joins
  unambiguously back to the source event.
* ``gallery_manifest.csv`` carries the detector internals for later
  precision/recall computation; these columns are **not** shown to the reader.
* ``reader_form_blinded.csv`` is what the reader fills in; only ``card_id``,
  ``image_path``, blank ``call``, blank ``notes``.
* ``reader_form_overlap.csv`` is a 50-card overlap subset for inter-reader
  agreement, drawn under ``GLOBAL_SEED + 1`` so it is independent of the
  stratum-sample RNG state.

Run modes
---------
The default mode validates the pipeline by computing all manifests and the
sampling log but does NOT render PNGs. ``--smoke`` renders 5 cards per stratum
(15 total) and writes the corresponding manifest + reader-form rows so the end
to end path can be confirmed. ``--full-render`` is the manual switch that
Thomas triggers when the WDB tree is mounted and rendering time is acceptable.

Examples
--------
Validate the sampling + manifest pipeline only (no PNGs)::

    uv run python scripts/51_candidate_gallery.py

End-to-end smoke render (15 PNGs)::

    uv run python scripts/51_candidate_gallery.py --smoke

Full render::

    uv run python scripts/51_candidate_gallery.py --full-render
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

from cuffcrt._paths import (
    ENV_WDB_ROOT,
    DataPathNotConfiguredError,
    resolve_configured_path,
)
from cuffcrt._seed import GLOBAL_SEED

# ``load_trace`` lives in scripts/50_figures.py, whose filename starts with a
# digit so it cannot be imported normally. Resolve it by path. The figures
# module is the canonical waveform loader; reusing it keeps the gallery
# semantically identical to the manuscript figures.
_FIG_PATH = Path(__file__).with_name("50_figures.py")
_spec = importlib.util.spec_from_file_location("_fig50", _FIG_PATH)
_fig = importlib.util.module_from_spec(_spec)  # pyright: ignore[reportArgumentType]
sys.modules["_fig50"] = _fig  # required so dataclasses in the module resolve
_spec.loader.exec_module(_fig)  # type: ignore[union-attr]
load_trace = _fig.load_trace

# The credentialed WDB tree is not included (PhysioNet DUA). Supply it per run
# via ``--wdb-root`` or the CUFFCRT_WDB_ROOT environment variable; there is no
# machine default (see data/README.md).

# Stratum names (also used as subdirectory names under
# ``results/gallery/`` and as values in the manifests).
STRATUM_A = "detector_positive"
STRATUM_B = "detector_rejected_near_miss"
STRATUM_C = "detector_negative_random"

# Stratum targets
TARGET_A = 268
TARGET_B = 200
TARGET_C = 100

# Render window relative to the charted BP timestamp (seconds).
RENDER_WINDOW_LO_S = -60.0
RENDER_WINDOW_HI_S = 90.0

# Smoke-render cap per stratum.
SMOKE_PER_STRATUM = 5

# Detector internals exposed on the manifest (for post-hoc precision/recall),
# never on the blinded reader form.
DETECTOR_INTERNALS = (
    "is_occlusion_signature",
    "phase3_duration_s",
    "nadir_depth_frac",
    "alignment_offset_s",
    "reject_reason",
)

# The reader-facing call vocabulary; must stay in sync with
# ``cuffcrt.llm.medgemma.VALID_CALLS``.
READER_CALL_VOCAB = (
    "occlusion_signature_present",
    "no_occlusion_signature",
    "indeterminate",
)


@dataclass(frozen=True)
class StratumSpec:
    """Sampling spec for one stratum."""

    name: str
    target: int
    pool: pl.DataFrame
    seed_offset: int


def _events_glob_paths(events_dir: Path) -> list[Path]:
    """Return sorted per-record event parquets under ``events_dir``."""
    return [Path(p) for p in sorted(glob.glob(str(events_dir / "events_*.parquet")))]


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(events_dir: Path) -> tuple[pl.DataFrame, list[tuple[Path, str]]]:
    """Concatenate the per-record event parquets and compute their SHA-256s.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory containing ``events_*.parquet`` (the canonical per-record
        event tables).

    Returns
    -------
    tuple[polars.DataFrame, list[tuple[pathlib.Path, str]]]
        ``(inventory_dataframe, [(path, sha256), ...])``. The list element
        order matches sorted file order so the sampling log is reproducible.
    """
    paths = _events_glob_paths(events_dir)
    if not paths:
        raise FileNotFoundError(f"no events_*.parquet under {events_dir}")
    parquet_shas: list[tuple[Path, str]] = []
    frames: list[pl.DataFrame] = []
    for p in paths:
        parquet_shas.append((p, _file_sha256(p)))
        frames.append(pl.read_parquet(p))
    inventory = pl.concat(frames, how="vertical")
    return inventory, parquet_shas


def define_strata(inventory: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Compute the candidate pool for each stratum.

    The near-miss predicate (stratum B) is intentionally narrow: a row is a
    near miss only if it passes the sub-0.50 envelope length gate
    (``phase3_duration_s >= 10``) AND the nadir gate
    (``nadir_depth_frac < 0.20``) but is then rejected by the detector for a
    non-quality reason (``no_recovery_in_window`` or ``stat_mode_short_phase3``,
    the only reject_reasons that retain measured phase3 and nadir values).
    ``no_aligned_occlusion`` rows are excluded because the detector did not
    record phase3 or nadir for them.

    Parameters
    ----------
    inventory : polars.DataFrame
        Concatenated per-record event table from :func:`load_inventory`.

    Returns
    -------
    dict[str, polars.DataFrame]
        Mapping from stratum name to the candidate pool DataFrame.
    """
    pool_a = inventory.filter(pl.col("is_occlusion_signature"))

    near_miss_reasons = ["no_recovery_in_window", "stat_mode_short_phase3"]
    pool_b = inventory.filter(
        (pl.col("phase3_duration_s") >= 10.0)
        & (pl.col("nadir_depth_frac") < 0.20)
        & pl.col("reject_reason").is_in(near_miss_reasons)
    )

    # Stratum C: detector negative on the envelope criterion AND renderable
    # (i.e. has co-recorded PPG; reject_reason != no_pleth).
    no_envelope_reasons = ["no_phase2", "pre_window_unstable"]
    pool_c = inventory.filter(
        pl.col("reject_reason").is_in(no_envelope_reasons)
        & (pl.col("pleth_valid_fraction") >= 0.5)
    )

    return {STRATUM_A: pool_a, STRATUM_B: pool_b, STRATUM_C: pool_c}


def sample_stratum(pool: pl.DataFrame, target: int, seed: int) -> tuple[pl.DataFrame, int]:
    """Sample up to ``target`` rows from ``pool`` deterministically.

    When ``pool.height <= target`` the entire pool is returned (no sampling
    needed). The shortfall is the difference between target and returned size.

    Parameters
    ----------
    pool : polars.DataFrame
        Candidate pool for the stratum.
    target : int
        Desired sample size.
    seed : int
        Per-stratum RNG seed (typically ``GLOBAL_SEED + offset``).

    Returns
    -------
    tuple[polars.DataFrame, int]
        ``(sampled_dataframe, shortfall)``.
    """
    n = pool.height
    if n <= target:
        return pool, max(0, target - n)
    # Deterministic permutation via numpy's seeded generator.
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    chosen = sorted(indices[:target].tolist())
    return pool[chosen], 0


def compute_card_id(subject_id: str, t_nbp_s: float, stratum: str, seed: int) -> str:
    """Return a short deterministic card id for one event.

    The hash is over a string built from ``(subject_id, t_nbp_s, stratum,
    seed)`` so the same event always lands on the same id. The first 16 hex
    characters of SHA-256 are enough for ~10^9 collision-free rows in
    practice; the manifest is the ground-truth join key in any case.

    Parameters
    ----------
    subject_id, stratum : str
        Identifiers.
    t_nbp_s : float
        Charted BP timestamp in seconds from record start.
    seed : int
        Determinism anchor (typically :data:`cuffcrt._seed.GLOBAL_SEED`).

    Returns
    -------
    str
        16-character lowercase hex string prefixed with the stratum's letter.
    """
    prefix = {STRATUM_A: "A", STRATUM_B: "B", STRATUM_C: "C"}.get(stratum, "X")
    key = f"{subject_id}|{t_nbp_s:.6f}|{stratum}|{seed}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


# Matplotlib rcParams for the blinded card. Kept inline (not via figstyle) so
# the rendering is decoupled from any future figstyle change; reproducibility
# beats style consistency here.
_RC_PARAMS_BLINDED: dict[str, object] = {
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 0,  # no title
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#B8C2CC",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "lines.solid_capstyle": "round",
    "lines.antialiased": True,
}


def _render_anchor_free_png(t_local: np.ndarray, pi: np.ndarray, out_path: Path) -> None:
    """Render one blinded card to ``out_path``.

    The plot carries no annotations beyond the axis labels: no subject id,
    no BP marker, no detector phase shading, no laterality word. The window is
    cropped to ``[RENDER_WINDOW_LO_S, RENDER_WINDOW_HI_S]`` so the reader sees
    only the region the detector reasons over.

    The PI signal is normalized to its in-window median before plotting so the
    y axis reads consistently across patients and the reader is not biased by
    absolute PI magnitude (which depends on probe and skin).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    in_window = (t_local >= RENDER_WINDOW_LO_S) & (t_local <= RENDER_WINDOW_HI_S)
    t = t_local[in_window]
    y = pi[in_window]
    if t.size == 0:
        raise ValueError("no PI samples in render window")
    # Normalize to in-window median; protects the axis from one-record swings.
    median = float(np.nanmedian(y)) if np.isfinite(y).any() else 1.0
    if median <= 0 or not np.isfinite(median):
        median = 1.0
    y_norm = y / median

    with matplotlib.rc_context(_RC_PARAMS_BLINDED):
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.plot(t, y_norm, color="#1A1A1A", linewidth=1.1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PI (norm)")
        ax.set_xlim(RENDER_WINDOW_LO_S, RENDER_WINDOW_HI_S)
        ax.set_ylim(bottom=0.0)
        ax.margins(x=0.0)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def _sample_overlap(
    cards_df: pl.DataFrame, size: int, seed: int
) -> pl.DataFrame:
    """Draw a deterministic overlap subset across all strata.

    Parameters
    ----------
    cards_df : polars.DataFrame
        The full per-card manifest (must include ``card_id``, ``stratum``,
        ``image_path``).
    size : int
        Number of cards to draw.
    seed : int
        RNG seed.

    Returns
    -------
    polars.DataFrame
        The overlap subset, sorted by ``card_id`` for stability.
    """
    if cards_df.height <= size:
        return cards_df.sort("card_id")
    rng = np.random.default_rng(seed)
    indices = np.arange(cards_df.height)
    rng.shuffle(indices)
    chosen = sorted(indices[:size].tolist())
    return cards_df[chosen].sort("card_id")


def _write_sampling_log(
    *,
    out_path: Path,
    stratum_stats: list[dict],
    parquet_shas: list[tuple[Path, str]],
    seed: int,
    timestamp_iso: str,
    smoke_mode: bool,
    full_render: bool,
) -> None:
    """Append one block to the sampling log.

    The log is append-only so multiple invocations on the same gallery
    directory leave a chronological record of what was sampled and rendered.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    if write_header:
        lines.append("# Gallery sampling log\n")
        lines.append(
            "Append-only record of each candidate-gallery sampling run. Each "
            "block captures the per-stratum counts (target, available, "
            "sampled, shortfall), the GLOBAL_SEED used, the input event "
            "parquet SHA-256s, and the rendering mode.\n"
        )
    mode = "full-render" if full_render else ("smoke" if smoke_mode else "validate-only")
    lines.append(f"\n## Run at {timestamp_iso} (mode={mode}, seed={seed})\n")
    lines.append("### Stratum counts\n")
    lines.append("| stratum | target | available | sampled | shortfall |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for s in stratum_stats:
        lines.append(
            f"| {s['stratum']} | {s['target']} | {s['available']} | "
            f"{s['sampled']} | {s['shortfall']} |\n"
        )
    lines.append("\n### Input event parquets (SHA-256)\n")
    for path, sha in parquet_shas:
        lines.append(f"- `{path.name}`: `{sha}`\n")
    with out_path.open("a", encoding="utf-8") as f:
        f.writelines(lines)


def _build_manifest_row(
    *, card_id: str, stratum: str, event: dict, image_path: Path, image_sha256: str | None
) -> dict:
    """Compose one row of ``gallery_manifest.csv`` from a sampled event."""
    return {
        "card_id": card_id,
        "stratum": stratum,
        "subject_id": str(event["subject_id"]),
        "record_id": str(event["record_id"]),
        "t_nbp": float(event["nbp_timestamp_s"]),
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "is_occlusion_signature": bool(event.get("is_occlusion_signature", False)),
        "phase3_duration_s": float(event.get("phase3_duration_s", float("nan"))),
        "nadir_depth_frac": float(event.get("nadir_depth_frac", float("nan"))),
        "alignment_offset_s": float(event.get("alignment_offset_s", float("nan"))),
        "reject_reason": event.get("reject_reason"),
    }


def _render_event(
    *,
    wdb_root: Path,
    event: dict,
    card_id: str,
    stratum: str,
    gallery_root: Path,
) -> tuple[Path, str | None]:
    """Render one event's PNG; return ``(image_path, image_sha256)``.

    Returns ``(image_path, None)`` when no usable PI window was available, so
    the caller can record the manifest row with an empty SHA. The PNG path is
    relative to ``gallery_root`` so the manifest stays portable.
    """
    image_path = gallery_root / stratum / f"{card_id}.png"
    trace = load_trace(
        wdb_root,
        str(event["subject_id"]),
        str(event["record_id"]),
        float(event["nbp_timestamp_s"]),
    )
    if not trace.has_pleth or trace.pi.size == 0:
        return image_path, None
    try:
        _render_anchor_free_png(trace.t_local, trace.pi, image_path)
    except ValueError as exc:
        logger.warning("{}: render skipped ({})", card_id, exc)
        return image_path, None
    return image_path, _file_sha256(image_path)


def _manifest_schema() -> dict[str, type]:
    """Return the on-disk Polars schema for ``gallery_manifest.csv``."""
    return {
        "card_id": pl.Utf8,
        "stratum": pl.Utf8,
        "subject_id": pl.Utf8,
        "record_id": pl.Utf8,
        "t_nbp": pl.Float64,
        "image_path": pl.Utf8,
        "image_sha256": pl.Utf8,
        "is_occlusion_signature": pl.Boolean,
        "phase3_duration_s": pl.Float64,
        "nadir_depth_frac": pl.Float64,
        "alignment_offset_s": pl.Float64,
        "reject_reason": pl.Utf8,
    }


def _write_manifest_csv(rows: list[dict], out_path: Path) -> None:
    """Write the per-card manifest as CSV with a stable column order.

    The explicit schema is necessary because the first N stratum-A rows have
    ``reject_reason=None`` (a clean detector positive), which would otherwise
    cause Polars to infer ``Null`` for that column and then fail when
    stratum-C string values arrive.
    """
    schema = _manifest_schema()
    df = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)


def _write_reader_form(
    manifest_df: pl.DataFrame, out_path: Path, *, gallery_root: Path
) -> None:
    """Write the blinded reader form (no detector internals)."""
    if manifest_df.height == 0:
        pl.DataFrame(
            schema={
                "card_id": pl.Utf8,
                "image_path": pl.Utf8,
                "call": pl.Utf8,
                "notes": pl.Utf8,
            }
        ).write_csv(out_path)
        return
    rel_paths = [
        str(Path(p).relative_to(gallery_root))
        for p in manifest_df.get_column("image_path").to_list()
    ]
    form = pl.DataFrame(
        {
            "card_id": manifest_df.get_column("card_id").to_list(),
            "image_path": rel_paths,
            "call": [""] * manifest_df.height,
            "notes": [""] * manifest_df.height,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    form.write_csv(out_path)


def build_gallery(
    *,
    events_dir: Path,
    wdb_root: Path,
    out_dir: Path,
    smoke: bool,
    full_render: bool,
    seed: int,
    overlap_size: int,
) -> int:
    """Build the gallery: sample strata, optionally render, write manifests.

    Parameters
    ----------
    events_dir : pathlib.Path
        Directory of per-record canonical event parquets.
    wdb_root : pathlib.Path
        Root of the WDB tree (used only when ``smoke`` or ``full_render``).
    out_dir : pathlib.Path
        Output directory for the gallery (PNGs land under ``out_dir/<stratum>/``).
    smoke : bool
        Render :data:`SMOKE_PER_STRATUM` cards per stratum.
    full_render : bool
        Render every sampled card. Overrides ``smoke``.
    seed : int
        RNG seed for stratum sampling.
    overlap_size : int
        Overlap-subset size for ``reader_form_overlap.csv``.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on input errors).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory, parquet_shas = load_inventory(events_dir)
    logger.info("loaded {} events from {}", inventory.height, events_dir)

    pools = define_strata(inventory)
    specs = [
        StratumSpec(STRATUM_A, TARGET_A, pools[STRATUM_A], seed_offset=2),
        StratumSpec(STRATUM_B, TARGET_B, pools[STRATUM_B], seed_offset=3),
        StratumSpec(STRATUM_C, TARGET_C, pools[STRATUM_C], seed_offset=4),
    ]

    stratum_stats: list[dict] = []
    sampled: dict[str, pl.DataFrame] = {}
    for spec in specs:
        chosen, shortfall = sample_stratum(spec.pool, spec.target, seed + spec.seed_offset)
        sampled[spec.name] = chosen
        stratum_stats.append(
            {
                "stratum": spec.name,
                "target": spec.target,
                "available": spec.pool.height,
                "sampled": chosen.height,
                "shortfall": shortfall,
            }
        )
        logger.info(
            "{}: target={} available={} sampled={} shortfall={}",
            spec.name,
            spec.target,
            spec.pool.height,
            chosen.height,
            shortfall,
        )

    render_caps = {
        spec.name: (
            sampled[spec.name].height
            if full_render
            else (min(SMOKE_PER_STRATUM, sampled[spec.name].height) if smoke else 0)
        )
        for spec in specs
    }
    logger.info("render caps per stratum: {}", render_caps)

    manifest_rows: list[dict] = []
    for spec in specs:
        cap = render_caps[spec.name]
        chosen = sampled[spec.name]
        rendered = 0
        for event in chosen.iter_rows(named=True):
            card_id = compute_card_id(
                str(event["subject_id"]),
                float(event["nbp_timestamp_s"]),
                spec.name,
                seed,
            )
            image_path = out_dir / spec.name / f"{card_id}.png"
            image_sha: str | None = None
            if rendered < cap:
                image_path, image_sha = _render_event(
                    wdb_root=wdb_root,
                    event=event,
                    card_id=card_id,
                    stratum=spec.name,
                    gallery_root=out_dir,
                )
                if image_sha is not None:
                    rendered += 1
            manifest_rows.append(
                _build_manifest_row(
                    card_id=card_id,
                    stratum=spec.name,
                    event=event,
                    image_path=image_path,
                    image_sha256=image_sha,
                )
            )
        logger.info("{}: rendered {} PNGs", spec.name, rendered)

    manifest_path = out_dir / "gallery_manifest.csv"
    _write_manifest_csv(manifest_rows, manifest_path)
    logger.info("wrote {} ({} rows)", manifest_path, len(manifest_rows))

    manifest_df = pl.read_csv(manifest_path, infer_schema_length=20000)

    blinded_form_path = out_dir / "reader_form_blinded.csv"
    _write_reader_form(manifest_df, blinded_form_path, gallery_root=out_dir)
    logger.info("wrote {} ({} rows)", blinded_form_path, manifest_df.height)

    overlap_df = _sample_overlap(manifest_df, overlap_size, seed=seed + 1)
    overlap_form_path = out_dir / "reader_form_overlap.csv"
    _write_reader_form(overlap_df, overlap_form_path, gallery_root=out_dir)
    logger.info("wrote {} ({} rows)", overlap_form_path, overlap_df.height)

    sampling_log_path = out_dir / "sampling_log.md"
    timestamp_iso = dt.datetime.now(tz=dt.UTC).replace(microsecond=0).isoformat()
    _write_sampling_log(
        out_path=sampling_log_path,
        stratum_stats=stratum_stats,
        parquet_shas=parquet_shas,
        seed=seed,
        timestamp_iso=timestamp_iso,
        smoke_mode=smoke,
        full_render=full_render,
    )
    logger.info("appended sampling log -> {}", sampling_log_path)

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("data/interim/events"),
        help="Directory containing canonical events_*.parquet.",
    )
    parser.add_argument(
        "--wdb-root",
        type=Path,
        default=None,
        help=(
            "Root of the WDB tree (only needed for --smoke / --full-render). "
            f"Defaults to the ${ENV_WDB_ROOT} environment variable; required "
            "for rendering if that is unset."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/gallery"),
        help="Gallery output directory (PNGs land under <out>/<stratum>/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"RNG seed for stratum sampling (default: GLOBAL_SEED={GLOBAL_SEED}).",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=50,
        help="Number of cards in reader_form_overlap.csv (default: 50).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Render only {SMOKE_PER_STRATUM} PNGs per stratum to validate the pipeline.",
    )
    parser.add_argument(
        "--full-render",
        action="store_true",
        help="Render every sampled card (production mode).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on input errors).
    """
    args = _parse_args(argv)
    if not args.events_dir.exists():
        logger.error("events dir not found: {}", args.events_dir)
        return 2
    wdb_root = args.wdb_root
    if args.smoke or args.full_render:
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
            logger.error("WDB root not found (needed for rendering): {}", wdb_root)
            return 2
    return build_gallery(
        events_dir=args.events_dir,
        wdb_root=wdb_root,
        out_dir=args.out,
        smoke=args.smoke,
        full_render=args.full_render,
        seed=args.seed,
        overlap_size=args.overlap_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
