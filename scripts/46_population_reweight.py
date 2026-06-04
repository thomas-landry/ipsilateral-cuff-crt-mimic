"""Population (inverse-probability) reweighted precision/recall (step 46).

Step 44 computed precision, recall, and specificity on the 568-card enriched
adjudication gallery. The gallery oversampled detector-positive cycles, so the
raw gallery recall and specificity do not reflect the full evaluable
population. This script reweights the per-card outcomes by the known
per-stratum inverse sampling fraction (Horvitz-Thompson weighting) and reports
population precision/recall/specificity with subject-clustered percentile
bootstrap 95% CIs, side by side with the raw gallery values for contrast.

It reuses the importable logic in :mod:`cuffcrt.analysis.reweight` (so unit
tests do not shell out) and the existing subject-clustered bootstrap seed
(:data:`cuffcrt._seed.GLOBAL_SEED`).

Partition reconciliation
------------------------
The three sampling strata tile a subset of the evaluable-with-pleth population
(strata membership read from the gallery sampler in
``scripts/51_candidate_gallery.py`` and matched against
``data/interim/event_inventory.csv``):

* detector_positive: reject_reason empty.
  universe 268, sampled 268, weight 1.0000.
* detector_rejected_near_miss: no_recovery_in_window + stat_mode_short_phase3.
  universe 320, sampled 200, weight 1.6000.
* detector_negative_random: no_phase2 + pre_window_unstable.
  universe 8107, sampled 100, weight 81.0700.

268 + 320 + 8107 = 8695, while evaluable-with-pleth = 8909. The 214-card
remainder is composed of detector-negative cycles in the reject-reason
categories ``no_aligned_occlusion`` (136) and ``pre_pi_implausible`` (78), which
were not part of any sampling stratum. They were never rendered or adjudicated,
so they carry no reader or machine label. We therefore report the population
estimate over the *covered* universe (8695 of 8909 evaluable cycles, 97.6%) and
state the coverage fraction explicitly rather than imputing labels for the
uncovered remainder. 

Inputs (defaults point at the canonical on-disk artifacts)
----------------------------------------------------------
``--reader_csv``
    ``results/gallery/reader_form_blinded.csv`` (gold-standard reader calls).
``--medgemma_csv``
    ``results/medgemma_galleryrender/gallery_render_calls_cardkeyed.csv``
    (pixel-matched gallery-render MedGemma calls, card-keyed).
``--gallery_manifest``
    ``results/gallery/gallery_manifest.csv`` (stratum, subject_id, detector
    call via ``is_occlusion_signature``).
``--gallery_pr_csv``
    ``results/precision_recall/precision_recall_summary.csv`` (the raw gallery
    point/CI carried through for side-by-side contrast).

Outputs (written under ``--out_dir``, default
``results/precision_recall_population/``)
----------------------------------------------------------------------------
``precision_recall_population_summary.csv``
    Long format: ``predictor, metric, estimate_kind`` (``gallery`` or
    ``population``), ``point_estimate, ci_low, ci_high, n_eligible_cards,
    weighted_denominator``.
``partition_reconciliation.csv``
    Per stratum: ``stratum, universe, sampled, weight``; plus an
    ``uncovered_remainder`` row and a ``coverage`` summary.
``run_metadata.json``
    Input SHA-256s, seed, n_bootstrap, estimand notes, partition coverage,
    join counts, indeterminate / parse-failure handling.

Examples
--------
::

    uv run python scripts/46_population_reweight.py
    uv run python scripts/46_population_reweight.py --n_bootstrap 10000
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import polars as pl
from loguru import logger

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.analysis.reweight import (
    NEGATIVE_CALL,
    POSITIVE_CALL,
    StratumSpec,
    assign_card_weights,
    reconcile_partition,
    weighted_metric_with_ci,
)

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_READER = DEFAULT_REPO / "results/gallery/reader_form_blinded.csv"
DEFAULT_MEDGEMMA = (
    DEFAULT_REPO / "results/medgemma_galleryrender/gallery_render_calls_cardkeyed.csv"
)
DEFAULT_MANIFEST = DEFAULT_REPO / "results/gallery/gallery_manifest.csv"
DEFAULT_GALLERY_PR = DEFAULT_REPO / "results/precision_recall/precision_recall_summary.csv"
DEFAULT_OUT = DEFAULT_REPO / "results/precision_recall_population"

DEFAULT_N_BOOTSTRAP = 5000

# Canonical sampling strata (see sampling_log.md + event_inventory.csv).
DEFAULT_STRATA = (
    StratumSpec("detector_positive", universe=268, sampled=268),
    StratumSpec("detector_rejected_near_miss", universe=320, sampled=200),
    StratumSpec("detector_negative_random", universe=8107, sampled=100),
)
# Evaluable-with-pleth population the estimate targets.
DEFAULT_EVALUABLE_POPULATION = 8909
UNCOVERED_LABEL = (
    "detector-negative cycles in reject categories no_aligned_occlusion (136) "
    "+ pre_pi_implausible (78) = 214; never sampled, unlabeled"
)

METRICS = ("precision", "recall", "specificity")


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reader(path: Path) -> pl.DataFrame:
    """Load the blinded reader form, dropping unrated (empty-call) rows.

    Returns ``card_id`` and a normalized ``reader_call`` string.
    """
    df = pl.read_csv(path, infer_schema_length=20000)
    missing = sorted({"card_id", "call"} - set(df.columns))
    if missing:
        raise ValueError(f"reader_csv missing columns: {missing}")
    df = df.with_columns(pl.col("call").cast(pl.Utf8, strict=False))
    df = df.filter(pl.col("call").is_not_null() & (pl.col("call").str.strip_chars() != ""))
    return df.with_columns(
        pl.col("call").str.strip_chars().str.to_lowercase().alias("reader_call")
    ).select(["card_id", "reader_call"])


def load_medgemma(path: Path) -> pl.DataFrame:
    """Load card-keyed MedGemma calls.

    Requires ``card_id, call, parsed_ok``. Returns ``card_id, medgemma_call,
    medgemma_parsed_ok``.
    """
    df = pl.read_csv(path, infer_schema_length=20000)
    if "card_id" not in df.columns:
        raise ValueError("medgemma_csv has no 'card_id' column; cannot join.")
    missing = sorted({"card_id", "call", "parsed_ok"} - set(df.columns))
    if missing:
        raise ValueError(f"medgemma_csv missing columns: {missing}")
    return df.with_columns(
        pl.col("call").cast(pl.Utf8, strict=False).str.to_lowercase().alias("medgemma_call"),
        pl.col("parsed_ok")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .alias("medgemma_parsed_ok"),
    ).select(["card_id", "medgemma_call", "medgemma_parsed_ok"])


def load_manifest(path: Path) -> pl.DataFrame:
    """Load the gallery manifest, synthesizing the detector call.

    Returns ``card_id, subject_id, stratum, detector_call``. The detector call
    comes from ``is_occlusion_signature`` (true -> present, false -> absent).
    """
    df = pl.read_csv(path, infer_schema_length=20000)
    missing = sorted(
        {"card_id", "subject_id", "stratum", "is_occlusion_signature"} - set(df.columns)
    )
    if missing:
        raise ValueError(f"gallery_manifest missing columns: {missing}")
    is_occ = df.get_column("is_occlusion_signature")
    if is_occ.dtype == pl.Boolean:
        as_bool = is_occ
    else:
        as_bool = (
            is_occ.cast(pl.Utf8, strict=False).str.to_lowercase().is_in(["true", "1", "t", "yes"])
        )
    detector = pl.Series(
        name="detector_call",
        values=[POSITIVE_CALL if v else NEGATIVE_CALL for v in as_bool.to_list()],
        dtype=pl.Utf8,
    )
    return df.with_columns(detector).select(["card_id", "subject_id", "stratum", "detector_call"])


def join_sources(
    reader: pl.DataFrame, medgemma: pl.DataFrame, manifest: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Inner-join the three sources on ``card_id``; report drop counts.

    Returns the joined frame (sorted by ``card_id``) and a dict of join counts.
    """
    ri = set(reader.get_column("card_id").to_list())
    gi = set(medgemma.get_column("card_id").to_list())
    mi = set(manifest.get_column("card_id").to_list())
    joined = (
        manifest.join(reader, on="card_id", how="inner")
        .join(medgemma, on="card_id", how="inner")
        .sort("card_id")
    )
    counts = {
        "n_reader": reader.height,
        "n_medgemma": medgemma.height,
        "n_manifest": manifest.height,
        "n_joined": joined.height,
        "n_dropped_reader": len((gi & mi) - ri),
        "n_dropped_medgemma": len((ri & mi) - gi),
        "n_dropped_manifest": len((ri & gi) - mi),
    }
    logger.info("join counts: {}", counts)
    return joined, counts


def load_gallery_pr(path: Path) -> pl.DataFrame:
    """Load the step-44 raw gallery precision/recall summary for contrast.

    Returns it normalized to the population output schema with
    ``estimate_kind='gallery'``.
    """
    df = pl.read_csv(path, infer_schema_length=20000)
    return df.select(
        pl.col("predictor"),
        pl.col("metric"),
        pl.lit("gallery").alias("estimate_kind"),
        pl.col("point_estimate"),
        pl.col("ci_low"),
        pl.col("ci_high"),
        pl.col("n_used_for_metric").alias("n_eligible_cards"),
        pl.lit(None, dtype=pl.Float64).alias("weighted_denominator"),
    )


def compute_population_table(
    joined: pl.DataFrame,
    strata: tuple[StratumSpec, ...],
    *,
    n_bootstrap: int,
    seed: int,
) -> pl.DataFrame:
    """Compute weighted population metrics for both predictors with CIs.

    Returns the long-format population rows (``estimate_kind='population'``).
    """
    strat_per_card = joined.get_column("stratum").to_list()
    weights = assign_card_weights(strat_per_card, strata)
    clusters = joined.get_column("subject_id").to_list()
    reader_calls = joined.get_column("reader_call").to_list()

    rows: list[dict[str, object]] = []
    for predictor, call_col, parsed_col in [
        ("detector", "detector_call", None),
        ("medgemma", "medgemma_call", "medgemma_parsed_ok"),
    ]:
        machine_calls = list(joined.get_column(call_col).to_list())
        if parsed_col is not None:
            parsed = joined.get_column(parsed_col).to_list()
            # A parse failure is treated as an uncallable machine value so it
            # drops out of binary denominators (D2-consistent).
            machine_calls = [
                m if ok else "parse_failure" for m, ok in zip(machine_calls, parsed, strict=True)
            ]
        for metric in METRICS:
            res = weighted_metric_with_ci(
                reader_calls,
                machine_calls,
                weights,
                clusters,
                metric,
                n_resamples=n_bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "predictor": predictor,
                    "metric": metric,
                    "estimate_kind": "population",
                    "point_estimate": res.point,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "n_eligible_cards": res.n_eligible,
                    "weighted_denominator": res.weighted_den,
                }
            )
    return pl.DataFrame(rows)


def build_partition_table(
    strata: tuple[StratumSpec, ...], evaluable_population: int
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Build the partition-reconciliation table and a summary dict."""
    rec = reconcile_partition(strata, evaluable_population, uncovered_label=UNCOVERED_LABEL)
    rows: list[dict[str, object]] = []
    for s in strata:
        rows.append(
            {
                "row_kind": "stratum",
                "name": s.name,
                "universe": s.universe,
                "sampled": s.sampled,
                "weight": s.weight,
            }
        )
    rows.append(
        {
            "row_kind": "uncovered_remainder",
            "name": rec.uncovered_label,
            "universe": rec.uncovered,
            "sampled": 0,
            "weight": float("nan"),
        }
    )
    rows.append(
        {
            "row_kind": "coverage",
            "name": "covered / evaluable_with_pleth",
            "universe": rec.covered_universe,
            "sampled": rec.evaluable_population,
            "weight": rec.coverage_fraction,
        }
    )
    summary = {
        "evaluable_population": rec.evaluable_population,
        "covered_universe": rec.covered_universe,
        "uncovered": rec.uncovered,
        "uncovered_label": rec.uncovered_label,
        "coverage_fraction": rec.coverage_fraction,
        "strata": [
            {
                "name": s.name,
                "universe": s.universe,
                "sampled": s.sampled,
                "weight": s.weight,
            }
            for s in strata
        ],
    }
    return pl.DataFrame(rows), summary


def write_outputs(
    out_dir: Path,
    *,
    summary_df: pl.DataFrame,
    partition_df: pl.DataFrame,
    metadata: dict[str, object],
) -> None:
    """Write the three population artifacts to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.write_csv(out_dir / "precision_recall_population_summary.csv")
    partition_df.write_csv(out_dir / "partition_reconciliation.csv")
    with (out_dir / "run_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")


def run(
    *,
    reader_csv: Path,
    medgemma_csv: Path,
    gallery_manifest: Path,
    gallery_pr_csv: Path,
    out_dir: Path,
    seed: int,
    n_bootstrap: int,
    strata: tuple[StratumSpec, ...] = DEFAULT_STRATA,
    evaluable_population: int = DEFAULT_EVALUABLE_POPULATION,
) -> None:
    """End-to-end: load, join, reweight, contrast with gallery, write."""
    reader = load_reader(reader_csv)
    medgemma = load_medgemma(medgemma_csv)
    manifest = load_manifest(gallery_manifest)
    joined, join_counts = join_sources(reader, medgemma, manifest)

    # Guard: every joined card must reference a known stratum.
    known = {s.name for s in strata}
    seen = set(joined.get_column("stratum").to_list())
    unknown = seen - known
    if unknown:
        raise ValueError(
            f"joined cards reference unknown strata {sorted(unknown)}; "
            f"known strata: {sorted(known)}."
        )

    pop_df = compute_population_table(joined, strata, n_bootstrap=n_bootstrap, seed=seed)
    gallery_df = load_gallery_pr(gallery_pr_csv)
    summary_df = pl.concat([gallery_df, pop_df], how="vertical_relaxed").sort(
        ["predictor", "metric", "estimate_kind"]
    )

    partition_df, partition_summary = build_partition_table(strata, evaluable_population)

    metadata: dict[str, object] = {
        "utc_timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "cluster_unit": "subject_id (from gallery_manifest)",
        "n_subjects": int(joined.get_column("subject_id").n_unique()),
        "estimand": {
            "positive_class": POSITIVE_CALL,
            "gold_standard": "blinded human reader",
            "precision": "P(reader+ | machine+)",
            "recall": "P(machine+ | reader+)",
            "specificity": "P(machine- | reader-)",
            "weighting": (
                "Horvitz-Thompson: per-card weight = stratum universe / stratum "
                "sampled; metric = sum(w*num)/sum(w*den). Precision's "
                "denominator mix shifts across strata under reweighting; recall "
                "is within the reader-positive class and specificity within the "
                "reader-negative class."
            ),
            "indeterminate_handling": (
                "D2-consistent: reader-indeterminate and MedGemma "
                "parse_failure/indeterminate cards are excluded from the binary "
                "denominator for the affected metric (num=den=0) and so carry "
                "no weighted mass into that ratio."
            ),
        },
        "partition_reconciliation": partition_summary,
        "inputs": {
            "reader_csv": {
                "path": str(reader_csv),
                "sha256": _sha256_of_file(reader_csv),
            },
            "medgemma_csv": {
                "path": str(medgemma_csv),
                "sha256": _sha256_of_file(medgemma_csv),
            },
            "gallery_manifest": {
                "path": str(gallery_manifest),
                "sha256": _sha256_of_file(gallery_manifest),
            },
            "gallery_pr_csv": {
                "path": str(gallery_pr_csv),
                "sha256": _sha256_of_file(gallery_pr_csv),
            },
        },
        "join": join_counts,
    }

    write_outputs(
        out_dir,
        summary_df=summary_df,
        partition_df=partition_df,
        metadata=metadata,
    )
    logger.info("wrote population reweight artifacts to {}", out_dir)


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Population (inverse-probability) reweighted precision/recall for "
            "the detector and MedGemma, with subject-clustered bootstrap 95% "
            "CIs, contrasted with the raw gallery values."
        )
    )
    p.add_argument("--reader_csv", type=Path, default=DEFAULT_READER)
    p.add_argument("--medgemma_csv", type=Path, default=DEFAULT_MEDGEMMA)
    p.add_argument("--gallery_manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--gallery_pr_csv", type=Path, default=DEFAULT_GALLERY_PR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
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
        gallery_pr_csv=args.gallery_pr_csv,
        out_dir=args.out_dir,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
