"""Recompute + intra-rater reliability for the second-pass re-read (step 61).

Given the pass-2 export produced by the blinded re-read UI
(``reread_pass2_export.csv``, columns ``blind_id, call, confidence, notes,
utc``), this harness does everything needed to compare the second pass against
the first and to regenerate the downstream analyses on the corrected calls,
without ever overwriting a pass-1 artifact.

Steps
-----
1. De-blind: join ``blind_id`` -> ``card_id`` via the hidden
   ``_blind_map.csv`` from staging, then ``card_id`` -> ``row_id`` via
   :func:`cuffcrt.analysis.build_card_to_rowid`. Both joins are validated to
   resolve all cards (568/568, 0 unmatched); the harness fails loud otherwise.
2. Build the PASS-2 reference reader form at a NEW path
   ``results/gallery/reader_form_blinded_pass2.csv`` with the same schema as the
   pass-1 form (``card_id, image_path, call, confidence, notes``). Pass 1 is
   never touched.
3. Intra-rater reliability (pass 1 vs pass 2): Cohen's kappa for the 3-class
   call AND for a collapsed present-vs-(absent + indeterminate) binary, percent
   agreement, and a subject-clustered bootstrap 95% CI on agreement
   (:func:`cuffcrt.analysis.cluster_bootstrap_ci`, seed 20260426). The
   Landis-Koch band is reported verbally.
4. Change log ``results/gallery/reread_change_log.csv``: per card the stratum,
   the machine calls (detector + language model, pulled from the manifest /
   gallery-render run for the audit trail only), pass-1 call, pass-2 call,
   pass-2 confidence, a ``changed`` boolean, and the change ``direction``.
   Net change is summarized by direction and by stratum.
5. Recompute downstream on the PASS-2 reference WITHOUT clobbering canonical
   pass-1 outputs, by invoking the existing analysis/figure scripts with their
   ``--reader``/``--reader_csv`` flags pointed at the pass-2 form and their
   ``--out_dir`` pointed at NEW directories:
       results/precision_recall_readjud/             (scripts/44 logic)
       results/precision_recall_population_readjud/   (scripts/46 logic)
       figures/readjud/                               (scripts/47/53/54/55)
   The existing scripts are already parameterized by reader form and output
   directory, so no analysis logic is duplicated here (DRY).
6. Print a clear before/after summary: pass-1 vs pass-2 reader marginals,
   intra-rater kappa + CI, and detector & language-model precision / recall /
   specificity (gallery + population) under pass 2 versus pass 1.

Read-only / no-clobber contract
-------------------------------
The harness reads pass-1 artifacts (``reader_form_blinded.csv``, the manifest,
the gallery-render run log, the pass-1 ``results/precision_recall*`` summaries)
and never writes to any of them. Every new output goes to a ``*_readjud`` path
or to ``figures/readjud/``.

Examples
--------
::

    uv run python scripts/61_reread_recompute.py \\
        --pass2-export ~/Downloads/reread_pass2_export.csv

Measure-only mode (sample reliability)
--------------------------------------
When the second pass covers only a SAMPLE of cards (staged with
``scripts/60 --sample N``) and the sole goal is to MEASURE intra-rater
reliability, use ``--measure-only``. In that mode the harness:

* de-blinds and bridges only the cards present in the export (it requires that
  every exported ``blind_id`` resolves, NOT that all 568 cards are present),
* computes intra-rater agreement of those sampled pass-2 calls against pass 1,
* writes a concise reliability report
  (``results/gallery/reread_reliability_sample.md`` + ``.csv``) and a per-card
  change log for the sample (``results/gallery/reread_change_log_sample.csv``),
* and writes NOTHING else: it does not emit a pass-2 reference reader form,
  does not recompute scripts 44/46, and does not touch any figure.

Pass 1 (``reader_form_blinded.csv``) stays the canonical reference, untouched::

    uv run python scripts/61_reread_recompute.py --measure-only \\
        --pass2-export ~/Downloads/reread_pass2_export.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.analysis import (
    agreement_summary,
    build_card_to_rowid,
    cluster_bootstrap_ci,
    cohen_kappa,
    landis_koch_band,
)

# Call vocabulary, identical to the rest of the pipeline.
CALL_PRESENT = "occlusion_signature_present"
CALL_ABSENT = "no_occlusion_signature"
CALL_INDETERMINATE = "indeterminate"
CALLABLE = (CALL_PRESENT, CALL_ABSENT)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default input paths (canonical pass-1 artifacts; read-only here).
DEFAULT_BLIND_MAP = REPO_ROOT / "results/gallery_readjud_blind/_blind_map.csv"
DEFAULT_MANIFEST = REPO_ROOT / "results/gallery/gallery_manifest.csv"
DEFAULT_INVENTORY = REPO_ROOT / "data/interim/event_inventory.csv"
DEFAULT_PASS1 = REPO_ROOT / "results/gallery/reader_form_blinded.csv"
DEFAULT_MEDGEMMA = (
    REPO_ROOT / "results/medgemma_galleryrender/gallery_render_calls_cardkeyed.csv"
)

# New (non-clobbering) output paths.
DEFAULT_PASS2_FORM = REPO_ROOT / "results/gallery/reader_form_blinded_pass2.csv"
DEFAULT_CHANGE_LOG = REPO_ROOT / "results/gallery/reread_change_log.csv"
DEFAULT_PR_READJUD = REPO_ROOT / "results/precision_recall_readjud"
DEFAULT_PR_POP_READJUD = REPO_ROOT / "results/precision_recall_population_readjud"
DEFAULT_FIG_READJUD = REPO_ROOT / "figures/readjud"

# Measure-only (sample reliability) output paths. Separate filenames so a
# sample measurement never collides with a full-recompute change log.
DEFAULT_RELIABILITY_MD = REPO_ROOT / "results/gallery/reread_reliability_sample.md"
DEFAULT_RELIABILITY_CSV = REPO_ROOT / "results/gallery/reread_reliability_sample.csv"
DEFAULT_CHANGE_LOG_SAMPLE = (
    REPO_ROOT / "results/gallery/reread_change_log_sample.csv"
)

# Pass-1 precision/recall summaries, for the before/after comparison (read-only).
PASS1_PR_GALLERY = REPO_ROOT / "results/precision_recall/precision_recall_summary.csv"
PASS1_PR_POP = (
    REPO_ROOT
    / "results/precision_recall_population/precision_recall_population_summary.csv"
)

DEFAULT_N_BOOTSTRAP = 5000


@dataclass(frozen=True)
class DeblindResult:
    """De-blinded and bridged pass-2 calls.

    Attributes
    ----------
    frame : polars.DataFrame
        One row per card with ``blind_id, card_id, stratum, row_id,
        subject_id, image_path, call, confidence, notes, utc``.
    n_export : int
        Rows in the pass-2 export (post empty-call drop).
    n_mapped : int
        Rows that resolved blind_id -> card_id.
    n_bridged : int
        Rows that resolved card_id -> row_id.
    """

    frame: pl.DataFrame
    n_export: int
    n_mapped: int
    n_bridged: int


def load_pass2_export(path: Path) -> pl.DataFrame:
    """Load the re-read UI export and normalize the call/confidence strings.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``reread_pass2_export.csv`` (``blind_id, call, confidence,
        notes, utc``).

    Returns
    -------
    polars.DataFrame
        Columns ``blind_id, call, confidence, notes, utc`` with empty-call
        rows dropped and ``call`` lowercased/trimmed.

    Raises
    ------
    FileNotFoundError
        If the export is absent.
    ValueError
        If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"pass-2 export not found: {path}")
    df = pl.read_csv(path, infer_schema_length=20000)
    required = {"blind_id", "call"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"pass-2 export missing required columns: {missing}")
    for col in ("confidence", "notes", "utc"):
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    df = df.with_columns(
        pl.col("call")
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        .alias("call")
    )
    df = df.filter(pl.col("call").is_not_null() & (pl.col("call") != ""))
    return df.select(["blind_id", "call", "confidence", "notes", "utc"])


def deblind_and_bridge(
    *,
    pass2_export: Path,
    blind_map: Path,
    manifest: Path,
    inventory: Path,
) -> DeblindResult:
    """De-blind the pass-2 export and attach card_id, row_id, and stratum.

    Joins ``blind_id`` -> ``card_id`` (and ``stratum``) via the hidden staging
    map, attaches ``image_path`` and ``subject_id`` from the manifest, and
    attaches the canonical ``row_id`` via the card bridge. Validates that every
    rated card resolves through both joins; raises on any gap.

    Parameters
    ----------
    pass2_export : pathlib.Path
        The re-read UI export.
    blind_map : pathlib.Path
        Hidden de-blinding key from ``scripts/60`` (``blind_id, card_id,
        stratum``).
    manifest : pathlib.Path
        Gallery manifest (for ``image_path`` and ``subject_id``).
    inventory : pathlib.Path
        Consolidated event inventory (for the card -> row_id bridge).

    Returns
    -------
    DeblindResult
        The joined frame and the per-stage row counts.

    Raises
    ------
    FileNotFoundError
        If any input is absent.
    RuntimeError
        If any rated card fails to resolve blind_id -> card_id or
        card_id -> row_id.
    """
    if not blind_map.exists():
        raise FileNotFoundError(f"blind map not found: {blind_map}")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    export = load_pass2_export(pass2_export)
    n_export = export.height

    bmap = pl.read_csv(blind_map, infer_schema_length=20000)
    bmap_required = {"blind_id", "card_id", "stratum"}
    missing = sorted(bmap_required - set(bmap.columns))
    if missing:
        raise RuntimeError(f"_blind_map.csv missing columns: {missing}")

    mapped = export.join(bmap, on="blind_id", how="left")
    n_unmapped = mapped.filter(pl.col("card_id").is_null()).height
    if n_unmapped:
        bad = mapped.filter(pl.col("card_id").is_null()).get_column("blind_id").to_list()
        raise RuntimeError(
            f"{n_unmapped} pass-2 blind_id values did not resolve to a card_id "
            f"via {blind_map.name}; first few: {bad[:5]}"
        )
    n_mapped = mapped.height

    man = pl.read_csv(manifest, infer_schema_length=20000).select(
        ["card_id", "subject_id", "image_path"]
    )
    mapped = mapped.join(man, on="card_id", how="left")
    if mapped.filter(pl.col("subject_id").is_null()).height:
        raise RuntimeError("some card_ids are absent from the gallery manifest")

    bridge = build_card_to_rowid(inventory, manifest).select(["card_id", "row_id"])
    bridged = mapped.join(bridge, on="card_id", how="left")
    n_unbridged = bridged.filter(pl.col("row_id").is_null()).height
    if n_unbridged:
        bad = bridged.filter(pl.col("row_id").is_null()).get_column("card_id").to_list()
        raise RuntimeError(
            f"{n_unbridged} card_ids did not bridge to a canonical row_id; "
            f"first few: {bad[:5]}"
        )
    n_bridged = bridged.filter(pl.col("row_id").is_not_null()).height

    frame = bridged.select(
        [
            "blind_id",
            "card_id",
            "stratum",
            "row_id",
            "subject_id",
            "image_path",
            "call",
            "confidence",
            "notes",
            "utc",
        ]
    ).sort("card_id")
    logger.info(
        "de-blind + bridge: export={} mapped={} bridged={}",
        n_export,
        n_mapped,
        n_bridged,
    )
    return DeblindResult(
        frame=frame, n_export=n_export, n_mapped=n_mapped, n_bridged=n_bridged
    )


def write_pass2_form(frame: pl.DataFrame, out_path: Path) -> int:
    """Write the pass-2 reference reader form (pass-1 schema + confidence).

    Parameters
    ----------
    frame : polars.DataFrame
        The de-blinded pass-2 frame from :func:`deblind_and_bridge`.
    out_path : pathlib.Path
        Destination CSV (``reader_form_blinded_pass2.csv``).

    Returns
    -------
    int
        Number of rows written.
    """
    form = frame.select(["card_id", "image_path", "call", "confidence", "notes"]).sort(
        "card_id"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    form.write_csv(out_path)
    logger.info("wrote pass-2 reader form {} ({} rows)", out_path, form.height)
    return form.height


def load_pass1(path: Path) -> pl.DataFrame:
    """Load the pass-1 reader form, keeping ``card_id`` and a normalized call.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``reader_form_blinded.csv``.

    Returns
    -------
    polars.DataFrame
        Columns ``card_id, pass1_call`` (empty-call rows dropped).
    """
    df = pl.read_csv(path, infer_schema_length=20000)
    df = df.with_columns(
        pl.col("call")
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        .alias("call")
    )
    df = df.filter(pl.col("call").is_not_null() & (pl.col("call") != ""))
    return df.select(["card_id", pl.col("call").alias("pass1_call")])


def _collapse_binary(call: str) -> str:
    """Collapse a 3-class call to present vs not-present.

    ``occlusion_signature_present`` maps to itself; everything else
    (``no_occlusion_signature``, ``indeterminate``) maps to
    ``no_occlusion_signature``.
    """
    return CALL_PRESENT if call == CALL_PRESENT else CALL_ABSENT


@dataclass(frozen=True)
class ReliabilityResult:
    """Intra-rater reliability of pass 1 vs pass 2.

    Attributes
    ----------
    n : int
        Number of cards rated in both passes.
    kappa_3class : float
        Cohen's kappa over the 3-class call.
    kappa_binary : float
        Cohen's kappa over the collapsed present-vs-rest binary.
    percent_agreement : float
        Fraction of cards with an identical 3-class call across passes.
    agreement_ci_low : float
        Lower 95% bound on agreement (subject-clustered bootstrap).
    agreement_ci_high : float
        Upper 95% bound on agreement (subject-clustered bootstrap).
    band_3class : str
        Landis-Koch band for ``kappa_3class``.
    band_binary : str
        Landis-Koch band for ``kappa_binary``.
    """

    n: int
    kappa_3class: float
    kappa_binary: float
    percent_agreement: float
    agreement_ci_low: float
    agreement_ci_high: float
    band_3class: str
    band_binary: str


def compute_reliability(
    paired: pl.DataFrame, *, n_bootstrap: int, seed: int
) -> ReliabilityResult:
    """Compute intra-rater kappa, percent agreement, and a clustered CI.

    Parameters
    ----------
    paired : polars.DataFrame
        One row per card rated in both passes, with columns ``pass1_call,
        pass2_call, subject_id``.
    n_bootstrap : int
        Bootstrap resamples for the agreement CI.
    seed : int
        Bootstrap seed.

    Returns
    -------
    ReliabilityResult
        Kappas (3-class and binary), percent agreement, agreement CI, and the
        Landis-Koch bands.
    """
    p1 = paired.get_column("pass1_call").to_list()
    p2 = paired.get_column("pass2_call").to_list()

    summ = agreement_summary(p1, p2)
    b1 = [_collapse_binary(c) for c in p1]
    b2 = [_collapse_binary(c) for c in p2]
    kappa_binary = cohen_kappa(b1, b2)

    agree_indicator = np.array(
        [1.0 if x == y else 0.0 for x, y in zip(p1, p2, strict=True)],
        dtype=np.float64,
    )
    clusters = np.asarray(paired.get_column("subject_id").to_list())
    ci = cluster_bootstrap_ci(
        values=agree_indicator,
        clusters=clusters,
        n_resamples=n_bootstrap,
        seed=seed,
    )
    return ReliabilityResult(
        n=summ.n,
        kappa_3class=summ.cohen_kappa,
        kappa_binary=kappa_binary,
        percent_agreement=summ.percent_agreement,
        agreement_ci_low=ci.ci_low,
        agreement_ci_high=ci.ci_high,
        band_3class=landis_koch_band(summ.cohen_kappa),
        band_binary=landis_koch_band(kappa_binary),
    )


def _change_direction(pass1_call: str, pass2_call: str) -> str:
    """Classify the direction of a pass-1 -> pass-2 call change.

    Returns one of ``unchanged``, ``to_present``, ``from_present``,
    ``to_indeterminate``, ``from_indeterminate``, or ``other`` for the
    residual absent<->absent-class moves (which collapse to ``unchanged`` since
    both map to ``no_occlusion_signature``).
    """
    if pass1_call == pass2_call:
        return "unchanged"
    if pass2_call == CALL_PRESENT:
        return "to_present"
    if pass1_call == CALL_PRESENT:
        return "from_present"
    if pass2_call == CALL_INDETERMINATE:
        return "to_indeterminate"
    if pass1_call == CALL_INDETERMINATE:
        return "from_indeterminate"
    return "other"


def build_change_log(
    frame: pl.DataFrame,
    pass1: pl.DataFrame,
    manifest: pl.DataFrame,
    medgemma: pl.DataFrame,
) -> pl.DataFrame:
    """Build the per-card change log with machine calls for the audit trail.

    Machine calls (detector and language model) are attached for the audit
    trail ONLY; they are not used in any reliability computation and the reader
    never saw them. The detector call is derived from the manifest
    ``is_occlusion_signature`` boolean; the language-model call is the
    gallery-render run's per-card call.

    Parameters
    ----------
    frame : polars.DataFrame
        The de-blinded pass-2 frame (``card_id, stratum, call, confidence``).
    pass1 : polars.DataFrame
        Pass-1 calls (``card_id, pass1_call``).
    manifest : polars.DataFrame
        Gallery manifest (``card_id, is_occlusion_signature``).
    medgemma : polars.DataFrame
        Gallery-render language-model calls (``card_id, call``).

    Returns
    -------
    polars.DataFrame
        Columns ``card_id, stratum, detector_call, language_model_call,
        pass1_call, pass2_call, pass2_confidence, changed, direction``.
    """
    det = manifest.select(
        [
            "card_id",
            pl.when(_is_true(pl.col("is_occlusion_signature")))
            .then(pl.lit(CALL_PRESENT))
            .otherwise(pl.lit(CALL_ABSENT))
            .alias("detector_call"),
        ]
    )
    lm = medgemma.select(
        ["card_id", pl.col("call").str.to_lowercase().alias("language_model_call")]
    )

    log = (
        frame.select(
            [
                "card_id",
                "stratum",
                pl.col("call").alias("pass2_call"),
                pl.col("confidence").alias("pass2_confidence"),
            ]
        )
        .join(pass1, on="card_id", how="left")
        .join(det, on="card_id", how="left")
        .join(lm, on="card_id", how="left")
    )
    p1 = log.get_column("pass1_call").to_list()
    p2 = log.get_column("pass2_call").to_list()
    changed = [a != b for a, b in zip(p1, p2, strict=True)]
    direction = [
        _change_direction(str(a) if a is not None else "", str(b))
        for a, b in zip(p1, p2, strict=True)
    ]
    log = log.with_columns(
        pl.Series("changed", changed),
        pl.Series("direction", direction),
    )
    return log.select(
        [
            "card_id",
            "stratum",
            "detector_call",
            "language_model_call",
            "pass1_call",
            "pass2_call",
            "pass2_confidence",
            "changed",
            "direction",
        ]
    ).sort("card_id")


def _is_true(col: pl.Expr) -> pl.Expr:
    """Coerce a manifest truthy column (bool or ``"true"``/``"false"``) to bool."""
    return (
        col.cast(pl.Utf8, strict=False)
        .str.to_lowercase()
        .is_in(["true", "1", "t", "yes"])
    )


def _run_script(script: str, args: list[str], *, label: str) -> bool:
    """Invoke a pipeline script as a subprocess; log and continue on failure.

    The downstream analysis and figure scripts are already parameterized by
    reader form and output directory, so the harness drives them rather than
    duplicating their logic. A non-zero exit (for example a figure script that
    needs waveform data absent in this checkout) is logged as a warning and
    does not abort the harness, so the reliability + recompute artifacts that
    do not need waveforms still land.

    Parameters
    ----------
    script : str
        Script filename under ``scripts/``.
    args : list[str]
        CLI arguments to pass.
    label : str
        Human-readable label for the log line.

    Returns
    -------
    bool
        True if the subprocess exited 0.
    """
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *args]
    logger.info("[{}] running: {}", label, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(
            "[{}] exited {} (skipping). stderr tail:\n{}",
            label,
            proc.returncode,
            "\n".join(proc.stderr.strip().splitlines()[-8:]),
        )
        return False
    logger.info("[{}] ok", label)
    return True


def recompute_downstream(
    *,
    pass2_form: Path,
    manifest: Path,
    medgemma: Path,
    inventory: Path,
    pr_dir: Path,
    pr_pop_dir: Path,
    fig_dir: Path,
    seed: int,
    n_bootstrap: int,
    skip_figures: bool,
) -> dict[str, bool]:
    """Re-run scripts 44/46/47/53/54/55 on the pass-2 form into new dirs.

    Returns
    -------
    dict[str, bool]
        Per-step success flags.
    """
    results: dict[str, bool] = {}

    results["precision_recall"] = _run_script(
        "44_precision_recall.py",
        [
            "--reader_csv", str(pass2_form),
            "--medgemma_csv", str(medgemma),
            "--gallery_manifest", str(manifest),
            "--out_dir", str(pr_dir),
            "--seed", str(seed),
            "--n_bootstrap", str(n_bootstrap),
        ],
        label="44 precision_recall",
    )

    gallery_pr = pr_dir / "precision_recall_summary.csv"
    results["population"] = _run_script(
        "46_population_reweight.py",
        [
            "--reader_csv", str(pass2_form),
            "--medgemma_csv", str(medgemma),
            "--gallery_manifest", str(manifest),
            "--gallery_pr_csv", str(gallery_pr),
            "--out_dir", str(pr_pop_dir),
            "--seed", str(seed),
            "--n_bootstrap", str(n_bootstrap),
        ],
        label="46 population",
    )

    if skip_figures:
        logger.info("--skip-figures set; not re-rendering figures 47/53/54/55")
        return results

    pop_summary = pr_pop_dir / "precision_recall_population_summary.csv"
    results["fig_precision_recall"] = _run_script(
        "47_fig_precision_recall.py",
        ["--summary_csv", str(pop_summary), "--out_dir", str(fig_dir)],
        label="47 fig_precision_recall",
    )
    results["fig_disagreement"] = _run_script(
        "53_disagreement_figure.py",
        [
            "--reader_csv", str(pass2_form),
            "--manifest_csv", str(manifest),
            "--medgemma_csv", str(medgemma),
            "--out_dir", str(fig_dir),
            "--seed", str(seed),
        ],
        label="53 fig_disagreement",
    )
    results["fig_flow"] = _run_script(
        "54_flow_diagram.py",
        [
            "--inventory", str(inventory),
            "--manifest", str(manifest),
            "--reader", str(pass2_form),
            "--out_dir", str(fig_dir),
        ],
        label="54 fig_flow",
    )
    results["fig_alluvial"] = _run_script(
        "55_concordance_overview.py",
        [
            "--reader_csv", str(pass2_form),
            "--detector_csv", str(manifest),
            "--medgemma_csv", str(medgemma),
            "--out_dir", str(fig_dir),
        ],
        label="55 fig_alluvial",
    )
    return results


def _marginals(calls: list[str]) -> dict[str, int]:
    """Count calls into the three classes."""
    return {
        CALL_PRESENT: sum(1 for c in calls if c == CALL_PRESENT),
        CALL_ABSENT: sum(1 for c in calls if c == CALL_ABSENT),
        CALL_INDETERMINATE: sum(1 for c in calls if c == CALL_INDETERMINATE),
    }


def _read_pr_summary(path: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Read a precision/recall summary into ``(predictor, metric) -> (pt, lo, hi)``."""
    if not path.exists():
        return {}
    df = pl.read_csv(path, infer_schema_length=20000)
    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    pt_col = "point_estimate" if "point_estimate" in df.columns else "point"
    for row in df.iter_rows(named=True):
        key = (str(row["predictor"]), str(row["metric"]))
        out[key] = (
            float(row.get(pt_col, float("nan"))),
            float(row.get("ci_low", float("nan"))),
            float(row.get("ci_high", float("nan"))),
        )
    return out


def _fmt_pr(d: dict[tuple[str, str], tuple[float, float, float]], pred: str, metric: str) -> str:
    """Format one precision/recall cell as ``pt (lo-hi)`` or ``n/a``."""
    if (pred, metric) not in d:
        return "n/a"
    pt, lo, hi = d[(pred, metric)]
    return f"{pt:.3f} ({lo:.3f}-{hi:.3f})"


def print_summary(
    *,
    pass1_marg: dict[str, int],
    pass2_marg: dict[str, int],
    reliability: ReliabilityResult,
    change_log: pl.DataFrame,
    pr_dir: Path,
    pr_pop_dir: Path,
    seed: int,
    n_bootstrap: int,
) -> None:
    """Print the human-readable before/after summary to stdout.

    This is the only place the harness uses ``print`` rather than loguru, by
    design: this summary is the human-readable deliverable of the script.
    """
    line = "=" * 72

    def pct(part: int, whole: int) -> str:
        return f"{(100.0 * part / whole):.1f}%" if whole else "n/a"

    n1 = sum(pass1_marg.values())
    n2 = sum(pass2_marg.values())

    print(line)
    print("SECOND-PASS RE-READ: before/after summary")
    print(f"seed={seed}  n_bootstrap={n_bootstrap}")
    print(line)

    print("\nReader marginals (3-class call):")
    print(f"  {'class':<32}{'pass 1':>14}{'pass 2':>14}")
    for cls in (CALL_PRESENT, CALL_ABSENT, CALL_INDETERMINATE):
        c1, c2 = pass1_marg[cls], pass2_marg[cls]
        print(
            f"  {cls:<32}{c1:>6} ({pct(c1, n1):>6}){c2:>6} ({pct(c2, n2):>6})"
        )
    print(f"  {'total':<32}{n1:>14}{n2:>14}")

    print("\nIntra-rater reliability (pass 1 vs pass 2):")
    print(f"  cards rated in both passes : {reliability.n}")
    print(
        f"  percent agreement          : {reliability.percent_agreement:.3f} "
        f"(95% CI {reliability.agreement_ci_low:.3f}-{reliability.agreement_ci_high:.3f}, "
        f"subject-clustered)"
    )
    print(
        f"  Cohen kappa (3-class)      : {reliability.kappa_3class:.3f} "
        f"({reliability.band_3class})"
    )
    print(
        f"  Cohen kappa (present vs rest): {reliability.kappa_binary:.3f} "
        f"({reliability.band_binary})"
    )

    print("\nChange log summary:")
    n_changed = int(change_log.get_column("changed").sum())
    print(f"  changed cards: {n_changed} of {change_log.height} "
          f"({pct(n_changed, change_log.height)})")
    by_dir = (
        change_log.group_by("direction")
        .agg(pl.len().alias("n"))
        .sort("direction")
    )
    print("  by direction:")
    for row in by_dir.iter_rows(named=True):
        print(f"    {row['direction']:<22}{row['n']:>5}")
    by_stratum = (
        change_log.filter(pl.col("changed"))
        .group_by("stratum")
        .agg(pl.len().alias("n_changed"))
        .sort("stratum")
    )
    print("  changed cards by stratum:")
    for row in by_stratum.iter_rows(named=True):
        print(f"    {row['stratum']:<32}{row['n_changed']:>5}")

    pr1 = _read_pr_summary(PASS1_PR_GALLERY)
    pr2 = _read_pr_summary(pr_dir / "precision_recall_summary.csv")
    pop1 = _read_pr_summary(PASS1_PR_POP)
    pop2 = _read_pr_summary(pr_pop_dir / "precision_recall_population_summary.csv")

    print("\nGallery precision/recall/specificity (pass 1 -> pass 2):")
    for pred in ("detector", "medgemma"):
        print(f"  {pred}:")
        for metric in ("precision", "recall", "specificity"):
            print(
                f"    {metric:<12} {_fmt_pr(pr1, pred, metric):<24} -> "
                f"{_fmt_pr(pr2, pred, metric)}"
            )

    print("\nPopulation (IPW) precision/recall/specificity (pass 1 -> pass 2):")
    for pred in ("detector", "medgemma"):
        print(f"  {pred}:")
        for metric in ("precision", "recall", "specificity"):
            print(
                f"    {metric:<12} {_fmt_pr(pop1, pred, metric):<24} -> "
                f"{_fmt_pr(pop2, pred, metric)}"
            )
    print(line)


def _net_change_summary(change_log: pl.DataFrame) -> dict[str, int]:
    """Summarize the net direction of change for the sample.

    Counts moves that increase the present-call tally (``to_present``) versus
    moves that decrease it (``from_present``), plus the residual moves that do
    not cross the present boundary (absent <-> indeterminate). The net present
    delta is ``to_present - from_present``.

    Parameters
    ----------
    change_log : polars.DataFrame
        The per-card change log (needs a ``direction`` column).

    Returns
    -------
    dict[str, int]
        Keys ``to_present``, ``from_present``, ``other_changed``, ``unchanged``,
        and ``net_present_delta``.
    """
    counts = {
        row["direction"]: int(row["n"])
        for row in change_log.group_by("direction").agg(pl.len().alias("n")).iter_rows(
            named=True
        )
    }
    to_present = counts.get("to_present", 0)
    from_present = counts.get("from_present", 0)
    unchanged = counts.get("unchanged", 0)
    other_changed = sum(
        v for k, v in counts.items() if k not in ("to_present", "from_present", "unchanged")
    )
    return {
        "to_present": to_present,
        "from_present": from_present,
        "other_changed": other_changed,
        "unchanged": unchanged,
        "net_present_delta": to_present - from_present,
    }


def _net_direction_phrase(net_present_delta: int) -> str:
    """Render the net change direction as a short plain-language phrase."""
    if net_present_delta > 0:
        return (
            f"net shift toward present (+{net_present_delta} present calls in pass 2)"
        )
    if net_present_delta < 0:
        return (
            f"net shift away from present ({net_present_delta} present calls in pass 2)"
        )
    return "no net shift in present calls"


def write_reliability_csv(
    reliability: ReliabilityResult,
    net: dict[str, int],
    out_path: Path,
    *,
    seed: int,
    n_bootstrap: int,
) -> None:
    """Write the machine-readable one-row reliability summary CSV.

    Parameters
    ----------
    reliability : ReliabilityResult
        The computed intra-rater reliability for the sample.
    net : dict[str, int]
        The net-change summary from :func:`_net_change_summary`.
    out_path : pathlib.Path
        Destination CSV (``reread_reliability_sample.csv``).
    seed : int
        Bootstrap seed used for the agreement CI.
    n_bootstrap : int
        Number of bootstrap resamples used for the agreement CI.
    """
    row = {
        "n_cards": reliability.n,
        "percent_agreement": reliability.percent_agreement,
        "agreement_ci_low": reliability.agreement_ci_low,
        "agreement_ci_high": reliability.agreement_ci_high,
        "cohen_kappa_3class": reliability.kappa_3class,
        "landis_koch_3class": reliability.band_3class,
        "cohen_kappa_binary": reliability.kappa_binary,
        "landis_koch_binary": reliability.band_binary,
        "to_present": net["to_present"],
        "from_present": net["from_present"],
        "other_changed": net["other_changed"],
        "unchanged": net["unchanged"],
        "net_present_delta": net["net_present_delta"],
        "bootstrap_seed": seed,
        "n_bootstrap": n_bootstrap,
        "utc_timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_csv(out_path)
    logger.info("wrote reliability CSV {}", out_path)


def write_reliability_report(
    reliability: ReliabilityResult,
    net: dict[str, int],
    pass1_marg: dict[str, int],
    pass2_marg: dict[str, int],
    by_stratum: pl.DataFrame,
    out_path: Path,
    *,
    seed: int,
    n_bootstrap: int,
) -> None:
    """Write the concise human-readable reliability report (Markdown).

    The report is feasibility-framed, contains no patient identifiers and no
    home paths, uses American English, and avoids em-dashes.

    Parameters
    ----------
    reliability : ReliabilityResult
        The computed intra-rater reliability for the sample.
    net : dict[str, int]
        The net-change summary from :func:`_net_change_summary`.
    pass1_marg, pass2_marg : dict[str, int]
        Three-class call marginals on the sampled cards for pass 1 and pass 2.
    by_stratum : polars.DataFrame
        Per-stratum counts of sampled and changed cards (``stratum, n_sampled,
        n_changed``).
    out_path : pathlib.Path
        Destination Markdown file (``reread_reliability_sample.md``).
    seed : int
        Bootstrap seed used for the agreement CI.
    n_bootstrap : int
        Number of bootstrap resamples used for the agreement CI.
    """
    n = reliability.n
    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    def pct(part: int, whole: int) -> str:
        return f"{(100.0 * part / whole):.1f}%" if whole else "n/a"

    lines: list[str] = []
    lines.append("# Intra-rater reliability: sampled second-pass re-read")
    lines.append("")
    lines.append(
        "Feasibility/prevalence study. This report measures how consistently the "
        "principal reader re-applies the perfusion-index morphology call on a "
        "blinded re-read of a stratified sample of gallery cards. It is a "
        "reliability measurement only. Pass 1 "
        "(`results/gallery/reader_form_blinded.csv`) remains the canonical "
        "reference and is not revised by this report."
    )
    lines.append("")
    lines.append(f"- Cards re-read in the sample and paired with pass 1: {n}")
    lines.append(f"- Bootstrap seed: {seed}; resamples: {n_bootstrap}")
    lines.append(f"- Generated (UTC): {stamp}")
    lines.append("")
    lines.append("## Agreement")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(
        f"| percent agreement (3-class) | {reliability.percent_agreement:.3f} "
        f"(95% CI {reliability.agreement_ci_low:.3f} to "
        f"{reliability.agreement_ci_high:.3f}, subject-clustered) |"
    )
    lines.append(
        f"| Cohen kappa (3-class) | {reliability.kappa_3class:.3f} "
        f"({reliability.band_3class}) |"
    )
    lines.append(
        f"| Cohen kappa (present vs rest) | {reliability.kappa_binary:.3f} "
        f"({reliability.band_binary}) |"
    )
    lines.append("")
    lines.append(
        "Landis-Koch bands are reported for orientation only; the kappa point "
        "estimate and the agreement confidence interval are the primary numbers."
    )
    lines.append("")
    lines.append("## Call marginals on the sampled cards")
    lines.append("")
    lines.append("| call | pass 1 | pass 2 |")
    lines.append("| --- | --- | --- |")
    n1 = sum(pass1_marg.values())
    n2 = sum(pass2_marg.values())
    label = {
        CALL_PRESENT: "occlusion signature present",
        CALL_ABSENT: "no occlusion signature",
        CALL_INDETERMINATE: "indeterminate",
    }
    for cls in (CALL_PRESENT, CALL_ABSENT, CALL_INDETERMINATE):
        c1, c2 = pass1_marg[cls], pass2_marg[cls]
        lines.append(
            f"| {label[cls]} | {c1} ({pct(c1, n1)}) | {c2} ({pct(c2, n2)}) |"
        )
    lines.append("")
    lines.append("## Net change direction")
    lines.append("")
    lines.append(f"- {_net_direction_phrase(net['net_present_delta'])}.")
    lines.append(
        f"- Moves to present: {net['to_present']}; moves from present: "
        f"{net['from_present']}; other changes (absent vs indeterminate): "
        f"{net['other_changed']}; unchanged: {net['unchanged']}."
    )
    lines.append("")
    lines.append("## Changed cards by stratum")
    lines.append("")
    lines.append("| stratum | sampled | changed | changed % |")
    lines.append("| --- | --- | --- | --- |")
    for row in by_stratum.iter_rows(named=True):
        ns = int(row["n_sampled"])
        nc = int(row["n_changed"])
        lines.append(f"| {row['stratum']} | {ns} | {nc} | {pct(nc, ns)} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote reliability report {}", out_path)


def print_measure_summary(
    reliability: ReliabilityResult,
    net: dict[str, int],
    pass1_marg: dict[str, int],
    pass2_marg: dict[str, int],
    *,
    seed: int,
    n_bootstrap: int,
) -> None:
    """Print the measure-only kappa summary to stdout.

    Prints n cards, the per-class marginal breakdown, kappa with its confidence
    interval and Landis-Koch band, and the net change direction. This is the
    human-readable deliverable of the script, so it uses ``print`` rather
    than loguru by design.
    """
    line = "=" * 72
    n1 = sum(pass1_marg.values())
    n2 = sum(pass2_marg.values())

    def pct(part: int, whole: int) -> str:
        return f"{(100.0 * part / whole):.1f}%" if whole else "n/a"

    print(line)
    print("SAMPLED SECOND-PASS RE-READ: intra-rater reliability (measure-only)")
    print(f"seed={seed}  n_bootstrap={n_bootstrap}")
    print(line)
    print(f"\ncards re-read and paired with pass 1: {reliability.n}")

    print("\nPer-class call breakdown on the sampled cards:")
    print(f"  {'class':<32}{'pass 1':>14}{'pass 2':>14}")
    for cls in (CALL_PRESENT, CALL_ABSENT, CALL_INDETERMINATE):
        c1, c2 = pass1_marg[cls], pass2_marg[cls]
        print(f"  {cls:<32}{c1:>6} ({pct(c1, n1):>6}){c2:>6} ({pct(c2, n2):>6})")
    print(f"  {'total':<32}{n1:>14}{n2:>14}")

    print("\nAgreement (pass 1 vs pass 2 on the sample):")
    print(
        f"  percent agreement            : {reliability.percent_agreement:.3f} "
        f"(95% CI {reliability.agreement_ci_low:.3f}-{reliability.agreement_ci_high:.3f}, "
        f"subject-clustered)"
    )
    print(
        f"  Cohen kappa (3-class)        : {reliability.kappa_3class:.3f} "
        f"({reliability.band_3class})"
    )
    print(
        f"  Cohen kappa (present vs rest): {reliability.kappa_binary:.3f} "
        f"({reliability.band_binary})"
    )

    print("\nNet change direction:")
    print(f"  {_net_direction_phrase(net['net_present_delta'])}")
    print(
        f"  to_present={net['to_present']}  from_present={net['from_present']}  "
        f"other_changed={net['other_changed']}  unchanged={net['unchanged']}"
    )
    print(line)


def run_measure_only(
    *,
    pass2_export: Path,
    blind_map: Path,
    manifest_path: Path,
    inventory: Path,
    pass1_form: Path,
    medgemma_csv: Path,
    reliability_md_out: Path,
    reliability_csv_out: Path,
    change_log_out: Path,
    seed: int,
    n_bootstrap: int,
) -> int:
    """Measure-only intra-rater reliability on the sampled re-read.

    Joins the sampled pass-2 export to pass 1 over only the cards present in the
    export (every exported ``blind_id`` must resolve, but the export need not
    cover all 568 cards), computes intra-rater agreement, and writes the sample
    reliability report (`.md` + `.csv`) and the per-card sample change log.
    Writes no pass-2 reference form, recomputes nothing downstream, and touches
    no figure.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    deb = deblind_and_bridge(
        pass2_export=pass2_export,
        blind_map=blind_map,
        manifest=manifest_path,
        inventory=inventory,
    )
    logger.info(
        "measure-only: {} cards in export, all resolved through both joins "
        "(not requiring full 568-card coverage)",
        deb.n_export,
    )

    pass1 = load_pass1(pass1_form)
    paired = (
        deb.frame.select(
            ["card_id", "subject_id", pl.col("call").alias("pass2_call")]
        )
        .join(pass1, on="card_id", how="inner")
        .sort("card_id")
    )
    n_export = deb.frame.height
    if paired.height != n_export:
        missing = n_export - paired.height
        raise RuntimeError(
            f"{missing} of {n_export} sampled cards have no pass-1 call to pair "
            "against; cannot measure reliability on those cards"
        )
    logger.info("paired (sampled cards rated in both passes): {} cards", paired.height)
    reliability = compute_reliability(paired, n_bootstrap=n_bootstrap, seed=seed)

    manifest = pl.read_csv(manifest_path, infer_schema_length=20000).select(
        ["card_id", "is_occlusion_signature"]
    )
    medgemma = pl.read_csv(medgemma_csv, infer_schema_length=20000).select(
        ["card_id", "call"]
    )
    change_log = build_change_log(deb.frame, pass1, manifest, medgemma)
    change_log_out.parent.mkdir(parents=True, exist_ok=True)
    change_log.write_csv(change_log_out)
    logger.info(
        "wrote sample change log {} ({} rows)", change_log_out, change_log.height
    )

    net = _net_change_summary(change_log)
    by_stratum = (
        change_log.group_by("stratum")
        .agg(
            pl.len().alias("n_sampled"),
            pl.col("changed").sum().alias("n_changed"),
        )
        .sort("stratum")
    )

    pass1_marg = _marginals(paired.get_column("pass1_call").to_list())
    pass2_marg = _marginals(paired.get_column("pass2_call").to_list())

    write_reliability_csv(
        reliability, net, reliability_csv_out, seed=seed, n_bootstrap=n_bootstrap
    )
    write_reliability_report(
        reliability,
        net,
        pass1_marg,
        pass2_marg,
        by_stratum,
        reliability_md_out,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    print_measure_summary(
        reliability,
        net,
        pass1_marg,
        pass2_marg,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    logger.info(
        "measure-only complete; pass-1 reference untouched, no downstream recompute"
    )
    return 0


def run(
    *,
    pass2_export: Path,
    blind_map: Path,
    manifest_path: Path,
    inventory: Path,
    pass1_form: Path,
    medgemma_csv: Path,
    pass2_form_out: Path,
    change_log_out: Path,
    pr_dir: Path,
    pr_pop_dir: Path,
    fig_dir: Path,
    seed: int,
    n_bootstrap: int,
    skip_figures: bool,
) -> int:
    """End-to-end recompute + reliability pipeline. Returns an exit code."""
    deb = deblind_and_bridge(
        pass2_export=pass2_export,
        blind_map=blind_map,
        manifest=manifest_path,
        inventory=inventory,
    )

    write_pass2_form(deb.frame, pass2_form_out)

    pass1 = load_pass1(pass1_form)
    paired = (
        deb.frame.select(["card_id", "subject_id", pl.col("call").alias("pass2_call")])
        .join(pass1, on="card_id", how="inner")
        .sort("card_id")
    )
    logger.info("paired (rated both passes): {} cards", paired.height)
    reliability = compute_reliability(paired, n_bootstrap=n_bootstrap, seed=seed)

    manifest = pl.read_csv(manifest_path, infer_schema_length=20000).select(
        ["card_id", "is_occlusion_signature"]
    )
    medgemma = pl.read_csv(medgemma_csv, infer_schema_length=20000).select(
        ["card_id", "call"]
    )
    change_log = build_change_log(deb.frame, pass1, manifest, medgemma)
    change_log_out.parent.mkdir(parents=True, exist_ok=True)
    change_log.write_csv(change_log_out)
    logger.info("wrote change log {} ({} rows)", change_log_out, change_log.height)

    recompute_downstream(
        pass2_form=pass2_form_out,
        manifest=manifest_path,
        medgemma=medgemma_csv,
        inventory=inventory,
        pr_dir=pr_dir,
        pr_pop_dir=pr_pop_dir,
        fig_dir=fig_dir,
        seed=seed,
        n_bootstrap=n_bootstrap,
        skip_figures=skip_figures,
    )

    pass1_marg = _marginals(pass1.get_column("pass1_call").to_list())
    pass2_marg = _marginals(deb.frame.get_column("call").to_list())
    print_summary(
        pass1_marg=pass1_marg,
        pass2_marg=pass2_marg,
        reliability=reliability,
        change_log=change_log,
        pr_dir=pr_dir,
        pr_pop_dir=pr_pop_dir,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )

    # Reliability summary artifact for the audit trail.
    rel_path = pr_dir / "intra_rater_reliability.json"
    rel_path.parent.mkdir(parents=True, exist_ok=True)
    with rel_path.open("w") as fh:
        json.dump(
            {
                "n_paired": reliability.n,
                "percent_agreement": reliability.percent_agreement,
                "agreement_ci_low": reliability.agreement_ci_low,
                "agreement_ci_high": reliability.agreement_ci_high,
                "cohen_kappa_3class": reliability.kappa_3class,
                "cohen_kappa_binary": reliability.kappa_binary,
                "landis_koch_3class": reliability.band_3class,
                "landis_koch_binary": reliability.band_binary,
                "seed": seed,
                "n_bootstrap": n_bootstrap,
                "utc_timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            },
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")
    logger.info("wrote {}", rel_path)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pass2-export",
        type=Path,
        required=True,
        help="The reread_pass2_export.csv produced by the blinded re-read UI.",
    )
    p.add_argument("--blind-map", type=Path, default=DEFAULT_BLIND_MAP)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    p.add_argument("--pass1-form", type=Path, default=DEFAULT_PASS1)
    p.add_argument("--medgemma-csv", type=Path, default=DEFAULT_MEDGEMMA)
    p.add_argument("--pass2-form-out", type=Path, default=DEFAULT_PASS2_FORM)
    p.add_argument("--change-log-out", type=Path, default=DEFAULT_CHANGE_LOG)
    p.add_argument("--pr-dir", type=Path, default=DEFAULT_PR_READJUD)
    p.add_argument("--pr-pop-dir", type=Path, default=DEFAULT_PR_POP_READJUD)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_READJUD)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip re-rendering figures 47/53/54/55 (analysis CSVs still recompute).",
    )
    p.add_argument(
        "--measure-only",
        action="store_true",
        help=(
            "Measure intra-rater reliability on the sampled re-read ONLY. Joins "
            "over just the cards in the export (every exported blind_id must "
            "resolve), writes reread_reliability_sample.md/.csv and "
            "reread_change_log_sample.csv, and writes no pass-2 reference form, "
            "recomputes no analysis, and touches no figure. Pass 1 stays "
            "canonical."
        ),
    )
    p.add_argument(
        "--reliability-md-out", type=Path, default=DEFAULT_RELIABILITY_MD
    )
    p.add_argument(
        "--reliability-csv-out", type=Path, default=DEFAULT_RELIABILITY_CSV
    )
    p.add_argument(
        "--change-log-sample-out", type=Path, default=DEFAULT_CHANGE_LOG_SAMPLE
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on missing pass-2 export).
    """
    args = _parse_args(argv)
    if not args.pass2_export.exists():
        logger.error("pass-2 export not found: {}", args.pass2_export)
        return 2
    if args.measure_only:
        return run_measure_only(
            pass2_export=args.pass2_export,
            blind_map=args.blind_map,
            manifest_path=args.manifest,
            inventory=args.inventory,
            pass1_form=args.pass1_form,
            medgemma_csv=args.medgemma_csv,
            reliability_md_out=args.reliability_md_out,
            reliability_csv_out=args.reliability_csv_out,
            change_log_out=args.change_log_sample_out,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
        )
    return run(
        pass2_export=args.pass2_export,
        blind_map=args.blind_map,
        manifest_path=args.manifest,
        inventory=args.inventory,
        pass1_form=args.pass1_form,
        medgemma_csv=args.medgemma_csv,
        pass2_form_out=args.pass2_form_out,
        change_log_out=args.change_log_out,
        pr_dir=args.pr_dir,
        pr_pop_dir=args.pr_pop_dir,
        fig_dir=args.fig_dir,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        skip_figures=args.skip_figures,
    )


if __name__ == "__main__":
    raise SystemExit(main())
