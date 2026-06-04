"""Precision/recall vs reader reference for detector and MedGemma (step 44).

Joins three call sources on ``card_id`` and computes precision, recall, and
specificity for both the rule-based detector (pre-registered primary
classifier) and MedGemma (AI-assisted secondary analysis) against the blinded
human reader reference. All point estimates carry a subject-clustered
percentile bootstrap 95% CI (seed 20260426) via
:func:`cuffcrt.analysis.bootstrap.cluster_bootstrap_ci`. Indeterminate and
parse-failure rows are excluded from the binary metric denominators and
reported as separate uncallable rates, consistent with the D2 uncallable
class.

Inputs
------
``--reader_csv``
    ``results/gallery/reader_form_blinded.csv``. Columns ``card_id, image_path,
    call, notes``. Empty ``call`` strings mean the card has not been
    adjudicated yet and are dropped (with a count reported).
``--medgemma_csv``
    A MedGemma adjudication run log. Must contain at least ``card_id, call,
    parsed_ok`` and ideally ``parse_error``. If the harness wrote ``row_id``
    instead of ``card_id`` the script errors out with a clear message rather
    than silently joining on the wrong key.
``--gallery_manifest``
    ``results/gallery/gallery_manifest.csv``. Columns include ``card_id,
    stratum, subject_id, is_occlusion_signature``. The detector call comes
    directly from ``is_occlusion_signature`` (a boolean): ``true`` ->
    ``occlusion_signature_present``, ``false`` -> ``no_occlusion_signature``.
    The detector never produces ``indeterminate`` or parse failure. We also
    confirmed by inspecting card_ids that the stratum prefix ``A-`` aligns
    with ``is_occlusion_signature=true`` (detector_positive, 268 cards),
    ``B-`` with ``false`` (detector_rejected_near_miss, 200 cards), and
    ``C-`` with ``false`` (detector_negative_random, 100 cards).

Indeterminate-handling decision (documented, D2-consistent)
-----------------------------------------------------------
The cleanest unambiguous denominator for a binary precision/recall is the set
of cards where BOTH the reference (reader) and the predictor produced one of
the two callable values (``occlusion_signature_present`` or
``no_occlusion_signature``). Rows where reader OR predictor returned
``indeterminate`` or where the predictor parse-failed are excluded from the
binary metric and counted separately in ``indeterminate_rates.csv``. The 3x3
confusion matrix retains all three call values per predictor so the
indeterminate cell counts are visible.

Outputs
-------
``<out_dir>/precision_recall_summary.csv``
    Long format. Columns: ``predictor`` (``detector`` or ``medgemma``),
    ``metric`` (``precision``, ``recall``, ``specificity``),
    ``point_estimate``, ``ci_low``, ``ci_high``, ``n_used_for_metric``.
``<out_dir>/confusion_matrices.csv``
    Long format. Columns: ``predictor``, ``reference_value``,
    ``predictor_value``, ``count``. Includes all combinations of the three
    call values plus a ``parse_failure`` predictor_value for MedGemma.
``<out_dir>/indeterminate_rates.csv``
    Per predictor: ``indeterminate_count``, ``parse_failure_count``,
    ``total_n``, ``indeterminate_rate``, ``ci_low``, ``ci_high``. The CI is a
    subject-clustered bootstrap on the union (indeterminate or parse failure).
``<out_dir>/run_metadata.json``
    SHA-256 of each input CSV, seed, n_bootstrap, n_joined and n_dropped per
    source, and a UTC timestamp.

Examples
--------
::

    uv run python scripts/44_precision_recall.py \\
        --reader_csv results/gallery/reader_form_blinded.csv \\
        --medgemma_csv results/medgemma/canonical_run_log.csv \\
        --gallery_manifest results/gallery/gallery_manifest.csv \\
        --out_dir results/precision_recall/ \\
        --seed 20260426 \\
        --n_bootstrap 5000
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.analysis.bootstrap import cluster_bootstrap_ci

OCCLUSION_SIGNATURE_PRESENT = "occlusion_signature_present"
NO_OCCLUSION_SIGNATURE = "no_occlusion_signature"
INDETERMINATE = "indeterminate"
PARSE_FAILURE = "parse_failure"

CALLABLE_VALUES = (OCCLUSION_SIGNATURE_PRESENT, NO_OCCLUSION_SIGNATURE)
ALL_CALL_VALUES = (OCCLUSION_SIGNATURE_PRESENT, NO_OCCLUSION_SIGNATURE, INDETERMINATE)

DEFAULT_N_BOOTSTRAP = 5000


@dataclass(frozen=True)
class JoinResult:
    """Result of joining reader, MedGemma, and detector calls on ``card_id``.

    Attributes
    ----------
    joined : polars.DataFrame
        One row per ``card_id`` present in all three sources, with columns
        ``card_id, subject_id, reader_call, medgemma_call, medgemma_parsed_ok,
        detector_call``.
    n_reader : int
        Rows in the reader source with a non-empty ``call``.
    n_medgemma : int
        Rows in the MedGemma source.
    n_manifest : int
        Rows in the gallery manifest.
    n_joined : int
        Rows surviving the inner three-way join.
    n_dropped_reader : int
        Cards present in MedGemma + manifest but missing or unrated in reader.
    n_dropped_medgemma : int
        Cards present in reader + manifest but missing in MedGemma.
    n_dropped_manifest : int
        Cards present in reader + MedGemma but missing in manifest.
    """

    joined: pl.DataFrame
    n_reader: int
    n_medgemma: int
    n_manifest: int
    n_joined: int
    n_dropped_reader: int
    n_dropped_medgemma: int
    n_dropped_manifest: int


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _detector_call_from_manifest(is_occlusion_signature: pl.Series) -> pl.Series:
    """Map manifest boolean ``is_occlusion_signature`` to the call vocabulary.

    Polars may read the manifest CSV column as either ``Boolean`` or ``Utf8``
    (``"true"``/``"false"``) depending on the writer; coerce both into the
    string call values used by the reader and MedGemma.
    """
    if is_occlusion_signature.dtype == pl.Boolean:
        as_bool = is_occlusion_signature
    else:
        as_bool = (
            is_occlusion_signature.cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .is_in(["true", "1", "t", "yes"])
        )
    return pl.Series(
        name="detector_call",
        values=[
            OCCLUSION_SIGNATURE_PRESENT if v else NO_OCCLUSION_SIGNATURE
            for v in as_bool.to_list()
        ],
        dtype=pl.Utf8,
    )


def load_reader(path: Path) -> pl.DataFrame:
    """Load the blinded reader form.

    Drops rows whose ``call`` is empty (still unrated). Keeps ``card_id`` and
    a normalized ``reader_call`` string.
    """
    df = pl.read_csv(path, infer_schema_length=20000)
    required = {"card_id", "call"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"reader_csv missing required columns: {missing}")
    df = df.with_columns(pl.col("call").cast(pl.Utf8, strict=False).alias("call"))
    df = df.filter(pl.col("call").is_not_null() & (pl.col("call").str.strip_chars() != ""))
    df = df.with_columns(
        pl.col("call").str.strip_chars().str.to_lowercase().alias("reader_call")
    )
    return df.select(["card_id", "reader_call"])


def load_medgemma(path: Path) -> pl.DataFrame:
    """Load a MedGemma run log.

    Requires ``card_id`` (preferred) so the inner join uses the same key the
    reader and manifest use. The current harness emits ``row_id`` instead; if
    only ``row_id`` is present we error out rather than guess a mapping.
    """
    df = pl.read_csv(path, infer_schema_length=20000)
    if "card_id" not in df.columns:
        raise ValueError(
            "medgemma_csv has no 'card_id' column. The current harness writes "
            "'row_id' (subject_record_idx); re-run the gallery-adjudication "
            "branch that stamps 'card_id', or precompute a card_id column "
            "before invoking this script."
        )
    required = {"card_id", "call", "parsed_ok"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"medgemma_csv missing required columns: {missing}")
    df = df.with_columns(
        pl.col("call").cast(pl.Utf8, strict=False).str.to_lowercase().alias("medgemma_call"),
        pl.col("parsed_ok")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .alias("medgemma_parsed_ok"),
    )
    return df.select(["card_id", "medgemma_call", "medgemma_parsed_ok"])


def load_manifest(path: Path) -> pl.DataFrame:
    """Load the gallery manifest and synthesize the detector call column.

    Requires ``card_id, subject_id, is_occlusion_signature``. Errors out if
    ``subject_id`` is missing because the cluster-bootstrap unit is the
    subject.
    """
    df = pl.read_csv(path, infer_schema_length=20000)
    required = {"card_id", "subject_id", "is_occlusion_signature"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"gallery_manifest missing required columns: {missing}. "
            "subject_id is required for the cluster-bootstrap CIs."
        )
    detector = _detector_call_from_manifest(df.get_column("is_occlusion_signature"))
    df = df.with_columns(detector)
    return df.select(["card_id", "subject_id", "detector_call"])


def join_sources(
    reader: pl.DataFrame,
    medgemma: pl.DataFrame,
    manifest: pl.DataFrame,
) -> JoinResult:
    """Inner-join reader x medgemma x manifest on ``card_id``.

    Reports the per-source drop counts so the caller can record them in the
    run metadata. No NaN-filling: a card_id missing from any source is
    dropped.
    """
    n_reader = reader.height
    n_medgemma = medgemma.height
    n_manifest = manifest.height

    reader_ids = set(reader.get_column("card_id").to_list())
    medgemma_ids = set(medgemma.get_column("card_id").to_list())
    manifest_ids = set(manifest.get_column("card_id").to_list())
    common_ids = reader_ids & medgemma_ids & manifest_ids

    joined = (
        manifest.join(reader, on="card_id", how="inner")
        .join(medgemma, on="card_id", how="inner")
        .sort("card_id")
    )

    n_joined = joined.height
    n_dropped_reader = len((medgemma_ids & manifest_ids) - reader_ids)
    n_dropped_medgemma = len((reader_ids & manifest_ids) - medgemma_ids)
    n_dropped_manifest = len((reader_ids & medgemma_ids) - manifest_ids)

    logger.info(
        "joined card_ids: {} (reader={} medgemma={} manifest={} common={})",
        n_joined,
        n_reader,
        n_medgemma,
        n_manifest,
        len(common_ids),
    )
    return JoinResult(
        joined=joined,
        n_reader=n_reader,
        n_medgemma=n_medgemma,
        n_manifest=n_manifest,
        n_joined=n_joined,
        n_dropped_reader=n_dropped_reader,
        n_dropped_medgemma=n_dropped_medgemma,
        n_dropped_manifest=n_dropped_manifest,
    )


def _binary_eligible(
    reader_call: pl.Series,
    predictor_call: pl.Series,
    predictor_parsed_ok: pl.Series | None,
) -> pl.Series:
    """Mask of rows usable for the binary precision/recall computation.

    A row is eligible iff the reader call is callable AND the predictor call
    is callable AND (for MedGemma) ``parsed_ok`` is True. ``parsed_ok`` is
    ignored when ``predictor_parsed_ok`` is None (the detector branch).
    """
    reader_callable = reader_call.is_in(list(CALLABLE_VALUES))
    predictor_callable = predictor_call.is_in(list(CALLABLE_VALUES))
    if predictor_parsed_ok is None:
        return reader_callable & predictor_callable
    return reader_callable & predictor_callable & predictor_parsed_ok


def _metric_indicators(
    reader_calls: list[str], predictor_calls: list[str], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (numerator_per_row, denominator_per_row) 0/1 arrays for ``metric``.

    Treating ``occlusion_signature_present`` as the positive class:

    - precision: denom = predictor positive; num = predictor positive AND
      reference positive.
    - recall (sensitivity): denom = reference positive; num = reference
      positive AND predictor positive.
    - specificity: denom = reference negative; num = reference negative AND
      predictor negative.

    Bootstrapping the per-row numerator and denominator side by side, then
    taking ``sum(num)/sum(denom)`` on each resample, gives a CI on the ratio
    of two random sums that respects cluster correlation. We expose this by
    running a single ``cluster_bootstrap_ci`` on a combined value vector and
    using a ratio statistic in the caller.
    """
    n = len(reader_calls)
    num = np.zeros(n, dtype=np.float64)
    den = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = reader_calls[i]
        p = predictor_calls[i]
        if metric == "precision":
            if p == OCCLUSION_SIGNATURE_PRESENT:
                den[i] = 1.0
                if r == OCCLUSION_SIGNATURE_PRESENT:
                    num[i] = 1.0
        elif metric == "recall":
            if r == OCCLUSION_SIGNATURE_PRESENT:
                den[i] = 1.0
                if p == OCCLUSION_SIGNATURE_PRESENT:
                    num[i] = 1.0
        elif metric == "specificity":
            if r == NO_OCCLUSION_SIGNATURE:
                den[i] = 1.0
                if p == NO_OCCLUSION_SIGNATURE:
                    num[i] = 1.0
        else:
            raise ValueError(f"unknown metric: {metric}")
    return num, den


def _ratio_ci(
    num: np.ndarray,
    den: np.ndarray,
    clusters: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Subject-clustered CI for ``sum(num)/sum(den)``.

    Implementation: pack ``num`` and ``den`` into a 2-column value array,
    resample clusters with replacement via ``cluster_bootstrap_ci`` using a
    ratio statistic. Falls back to ``(point, point, point, denom)`` when
    every observation belongs to a single cluster (the bootstrap collapses)
    or when the denominator is zero on the full sample.
    """
    total_den = float(den.sum())
    total_num = float(num.sum())
    n_used = int(total_den)
    if total_den <= 0:
        return float("nan"), float("nan"), float("nan"), 0

    point = total_num / total_den

    # Encode (num, den) per row as a complex number so the bootstrap value
    # vector is 1-D; cluster_bootstrap_ci requires 1-D values. Real part
    # carries num, imag part carries den; the statistic unpacks them.
    packed = num.astype(np.complex128) + 1j * den.astype(np.complex128)

    def ratio_stat(arr: np.ndarray) -> float:
        n_sum = float(np.real(arr).sum())
        d_sum = float(np.imag(arr).sum())
        if d_sum <= 0:
            return float("nan")
        return n_sum / d_sum

    # cluster_bootstrap_ci internally calls ``statistic`` on a float64 view of
    # values, which would discard the imag part. Cast packed to float64 by
    # interleaving and run our own minimal cluster bootstrap here.
    rng = np.random.default_rng(seed)
    cluster_to_idx: dict[object, list[int]] = {}
    for i, c in enumerate(clusters.tolist()):
        cluster_to_idx.setdefault(c, []).append(i)
    unique = list(cluster_to_idx.keys())
    n_clusters = len(unique)
    if n_clusters <= 1:
        return point, point, point, n_used
    idx_lists = [np.asarray(cluster_to_idx[c], dtype=np.int64) for c in unique]

    replicates = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        pooled = np.concatenate([idx_lists[p] for p in picks])
        replicates[b] = ratio_stat(packed[pooled])

    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return point, float("nan"), float("nan"), n_used
    ci_low = float(np.percentile(finite, 2.5))
    ci_high = float(np.percentile(finite, 97.5))
    return point, ci_low, ci_high, n_used


def compute_metric_table(
    joined: pl.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pl.DataFrame:
    """Compute precision, recall, specificity for both predictors with CIs."""
    rows: list[dict[str, object]] = []
    for predictor, call_col, parsed_col in [
        ("detector", "detector_call", None),
        ("medgemma", "medgemma_call", "medgemma_parsed_ok"),
    ]:
        parsed_series = (
            joined.get_column(parsed_col) if parsed_col is not None else None
        )
        eligible_mask = _binary_eligible(
            joined.get_column("reader_call"),
            joined.get_column(call_col),
            parsed_series,
        )
        eligible = joined.filter(eligible_mask)
        if eligible.height == 0:
            for metric in ("precision", "recall", "specificity"):
                rows.append(
                    {
                        "predictor": predictor,
                        "metric": metric,
                        "point_estimate": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n_used_for_metric": 0,
                    }
                )
            continue

        reader_calls = eligible.get_column("reader_call").to_list()
        predictor_calls = eligible.get_column(call_col).to_list()
        clusters = np.asarray(eligible.get_column("subject_id").to_list())

        for metric in ("precision", "recall", "specificity"):
            num, den = _metric_indicators(reader_calls, predictor_calls, metric)
            point, ci_low, ci_high, n_used = _ratio_ci(
                num, den, clusters, n_bootstrap=n_bootstrap, seed=seed
            )
            rows.append(
                {
                    "predictor": predictor,
                    "metric": metric,
                    "point_estimate": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_used_for_metric": n_used,
                }
            )
    return pl.DataFrame(rows)


def compute_confusion_matrices(joined: pl.DataFrame) -> pl.DataFrame:
    """Build 3x3 (plus parse_failure for MedGemma) confusion matrices."""
    rows: list[dict[str, object]] = []

    for predictor, call_col, parsed_col in [
        ("detector", "detector_call", None),
        ("medgemma", "medgemma_call", "medgemma_parsed_ok"),
    ]:
        reader_calls = joined.get_column("reader_call").to_list()
        predictor_calls = joined.get_column(call_col).to_list()
        parsed_oks = (
            joined.get_column(parsed_col).to_list()
            if parsed_col is not None
            else [True] * joined.height
        )

        possible_predictor_values = list(ALL_CALL_VALUES)
        if predictor == "medgemma":
            possible_predictor_values = [*ALL_CALL_VALUES, PARSE_FAILURE]

        # Initialize the full table of zero cells so missing combinations
        # still appear as ``count=0`` (clearer than an absent row).
        counts: dict[tuple[str, str], int] = {
            (r, p): 0
            for r in ALL_CALL_VALUES
            for p in possible_predictor_values
        }
        for r, p, ok in zip(reader_calls, predictor_calls, parsed_oks, strict=True):
            if r not in ALL_CALL_VALUES:
                continue
            if predictor == "medgemma" and not ok:
                key = (r, PARSE_FAILURE)
            elif p in ALL_CALL_VALUES:
                key = (r, p)
            else:
                # Unknown predictor value for the detector (should not happen
                # since the detector vocab is bool); skip silently.
                continue
            counts[key] = counts.get(key, 0) + 1

        for (r, p), c in counts.items():
            rows.append(
                {
                    "predictor": predictor,
                    "reference_value": r,
                    "predictor_value": p,
                    "count": c,
                }
            )
    return pl.DataFrame(rows).sort(["predictor", "reference_value", "predictor_value"])


def compute_indeterminate_rates(
    joined: pl.DataFrame, *, n_bootstrap: int, seed: int
) -> pl.DataFrame:
    """Per-predictor indeterminate and parse-failure counts plus a CI.

    The CI is a subject-clustered bootstrap on the union indicator (the row is
    indeterminate OR a parse failure). For the detector this is structurally
    zero, but we still emit a zero row for symmetry.
    """
    rows: list[dict[str, object]] = []
    clusters = np.asarray(joined.get_column("subject_id").to_list())

    for predictor, call_col, parsed_col in [
        ("detector", "detector_call", None),
        ("medgemma", "medgemma_call", "medgemma_parsed_ok"),
    ]:
        n = joined.height
        if parsed_col is None:
            parsed = np.ones(n, dtype=bool)
        else:
            parsed = np.asarray(
                joined.get_column(parsed_col).to_list(), dtype=bool
            )
        calls = np.asarray(joined.get_column(call_col).to_list())
        is_indet = parsed & (calls == INDETERMINATE)
        is_parse_fail = ~parsed
        is_uncallable = is_indet | is_parse_fail

        n_indet = int(is_indet.sum())
        n_pf = int(is_parse_fail.sum())
        rate = float(is_uncallable.mean()) if n > 0 else float("nan")

        if n == 0:
            ci_low = ci_high = float("nan")
        else:
            res = cluster_bootstrap_ci(
                values=is_uncallable.astype(np.float64),
                clusters=clusters,
                n_resamples=n_bootstrap,
                seed=seed,
            )
            ci_low = res.ci_low
            ci_high = res.ci_high

        rows.append(
            {
                "predictor": predictor,
                "indeterminate_count": n_indet,
                "parse_failure_count": n_pf,
                "total_n": n,
                "indeterminate_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pl.DataFrame(rows)


def write_outputs(
    out_dir: Path,
    *,
    summary_df: pl.DataFrame,
    confusion_df: pl.DataFrame,
    indeterminate_df: pl.DataFrame,
    metadata: dict[str, object],
) -> None:
    """Write the four canonical artifacts to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.write_csv(out_dir / "precision_recall_summary.csv")
    confusion_df.write_csv(out_dir / "confusion_matrices.csv")
    indeterminate_df.write_csv(out_dir / "indeterminate_rates.csv")
    with (out_dir / "run_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")


def run(
    *,
    reader_csv: Path,
    medgemma_csv: Path,
    gallery_manifest: Path,
    out_dir: Path,
    seed: int,
    n_bootstrap: int,
) -> None:
    """End-to-end pipeline: load, join, compute, write.

    Side effects: writes four files under ``out_dir``. Logs a one-line summary
    of join counts and metric denominators via loguru.
    """
    logger.info(
        "loading reader={} medgemma={} manifest={}",
        reader_csv,
        medgemma_csv,
        gallery_manifest,
    )
    reader = load_reader(reader_csv)
    medgemma = load_medgemma(medgemma_csv)
    manifest = load_manifest(gallery_manifest)

    join = join_sources(reader, medgemma, manifest)
    logger.info(
        "join: n_joined={} dropped_reader={} dropped_medgemma={} dropped_manifest={}",
        join.n_joined,
        join.n_dropped_reader,
        join.n_dropped_medgemma,
        join.n_dropped_manifest,
    )

    summary_df = compute_metric_table(join.joined, n_bootstrap=n_bootstrap, seed=seed)
    confusion_df = compute_confusion_matrices(join.joined)
    indeterminate_df = compute_indeterminate_rates(
        join.joined, n_bootstrap=n_bootstrap, seed=seed
    )

    metadata: dict[str, object] = {
        "utc_timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "inputs": {
            "reader_csv": {
                "path": str(reader_csv),
                "sha256": _sha256_of_file(reader_csv),
                "n_rows": join.n_reader,
            },
            "medgemma_csv": {
                "path": str(medgemma_csv),
                "sha256": _sha256_of_file(medgemma_csv),
                "n_rows": join.n_medgemma,
            },
            "gallery_manifest": {
                "path": str(gallery_manifest),
                "sha256": _sha256_of_file(gallery_manifest),
                "n_rows": join.n_manifest,
            },
        },
        "join": {
            "n_joined": join.n_joined,
            "n_dropped_reader": join.n_dropped_reader,
            "n_dropped_medgemma": join.n_dropped_medgemma,
            "n_dropped_manifest": join.n_dropped_manifest,
        },
        "indeterminate_handling": (
            "D2-consistent: rows with reader OR predictor in "
            "{indeterminate, parse_failure} are excluded from the binary "
            "precision/recall denominator and reported separately in "
            "indeterminate_rates.csv. Confusion matrix retains all classes."
        ),
        "cluster_unit": "subject_id (from gallery_manifest)",
    }

    write_outputs(
        out_dir,
        summary_df=summary_df,
        confusion_df=confusion_df,
        indeterminate_df=indeterminate_df,
        metadata=metadata,
    )
    logger.info("wrote precision/recall artifacts to {}", out_dir)


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Precision/recall vs reader reference for the rule-based detector "
            "and MedGemma, with subject-clustered bootstrap 95% CIs."
        )
    )
    p.add_argument("--reader_csv", type=Path, required=True)
    p.add_argument("--medgemma_csv", type=Path, required=True)
    p.add_argument("--gallery_manifest", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_argparser().parse_args(argv)
    run(
        reader_csv=args.reader_csv,
        medgemma_csv=args.medgemma_csv,
        gallery_manifest=args.gallery_manifest,
        out_dir=args.out_dir,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
