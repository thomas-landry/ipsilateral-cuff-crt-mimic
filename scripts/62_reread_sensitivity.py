"""Reference-correction sensitivity analysis on the 150-card re-read sample (step 62).

The blinded 568-card reader form (``results/gallery/reader_form_blinded.csv``)
stays the pre-specified primary reference for the paper. A blinded second-pass
re-read of a stratified 150-card subsample showed the principal reader
systematically undercalled on the first pass (moderate intra-rater agreement;
net shift toward more present calls on the re-read). This script quantifies
*how much* the index tests' agreement metrics move when, on those SAME 150
cards, the reference is switched from the pre-specified pass-1 calls to the
corrected pass-2 calls. Holding the card set fixed isolates the
reference-correction effect from any change in the sampled cards.

For BOTH the rule-based detector (the pre-registered primary classifier) and
the language model (an AI-assisted secondary analysis) it computes precision
(PPV), recall (sensitivity), and specificity under (i) the pass-1 reference and
(ii) the pass-2 reference, each with a subject-clustered nonparametric
percentile bootstrap 95% CI (seed 20260426), and reports the pass1 -> pass2
delta per metric and per index test. All counts are morphology-based estimates:
neither pass has access to ground-truth cuff laterality.

The metric definitions, the indeterminate/parse-failure handling, and the
clustered ratio bootstrap are taken verbatim from ``scripts/44_precision_recall.py``
(imported, not re-implemented), so a pass-1 cell here matches the corresponding
cell scripts/44 would produce on the same subset. Indeterminate reader calls
are excluded from the binary present/absent denominator on whichever pass is
acting as the reference, exactly as scripts/44 excludes them; they are reported
separately as the per-reference uncallable rate.

Inputs (read-only; never modified)
-----------------------------------
``--sample_csv``
    ``results/gallery/reread_change_log_sample.csv``. The 150 sampled cards
    with ``card_id, stratum, detector_call, language_model_call, pass1_call,
    pass2_call``. The two machine-call columns are the audit-trail copy already
    joined by ``scripts/61`` (detector from the manifest
    ``is_occlusion_signature`` boolean; language model from the gallery-render
    run), so no re-join is needed here.
``--gallery_manifest``
    ``results/gallery/gallery_manifest.csv``. Supplies the ``subject_id`` per
    ``card_id`` for the cluster-bootstrap unit. ``subject_id`` is the
    ``{subject}`` token of the canonical ``row_id`` the
    ``cuffcrt.analysis.build_card_to_rowid`` bridge would assign to the same
    card, so clustering on it is equivalent to clustering on the bridge's
    subject while keeping the committed manifest as the only required input.

Outputs (written to a NEW directory only)
------------------------------------------
``<out_dir>/reread_sensitivity_summary.csv``
    Long format. One row per (index_test x metric x reference). Columns:
    ``index_test`` (``detector`` / ``language_model``), ``metric``
    (``precision`` / ``recall`` / ``specificity``), ``reference`` (``pass1`` /
    ``pass2``), ``point_estimate``, ``ci_low``, ``ci_high``,
    ``n_used_for_metric``.
``<out_dir>/reread_sensitivity_delta.csv``
    Wide-ish format. One row per (index_test x metric). Columns:
    ``index_test``, ``metric``, ``pass1_point``, ``pass1_ci_low``,
    ``pass1_ci_high``, ``pass2_point``, ``pass2_ci_low``, ``pass2_ci_high``,
    ``delta`` (= ``pass2_point - pass1_point``).
``<out_dir>/reread_uncallable_rates.csv``
    Per reference: indeterminate count, total n, and the indeterminate rate
    with a subject-clustered bootstrap CI. (The detector and language model
    have no indeterminate or parse-failure calls in this sample, so the only
    uncallable rows come from the reference itself.)
``<out_dir>/reread_sensitivity_summary.md``
    Short human-readable summary table plus the one-line precision headline for
    each index test.
``<out_dir>/run_metadata.json``
    SHA-256 of each input, seed, n_bootstrap, n cards, n subjects, the pass-1
    and pass-2 reader marginals on the sample, and a UTC timestamp.

Examples
--------
::

    uv run python scripts/62_reread_sensitivity.py \\
        --sample_csv results/gallery/reread_change_log_sample.csv \\
        --gallery_manifest results/gallery/gallery_manifest.csv \\
        --out_dir results/precision_recall_reread_sensitivity/ \\
        --seed 20260426 \\
        --n_bootstrap 5000
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import polars as pl
from loguru import logger  # pyright: ignore[reportMissingImports]

from cuffcrt._seed import GLOBAL_SEED

DEFAULT_N_BOOTSTRAP = 5000

METRICS = ("precision", "recall", "specificity")
INDEX_TESTS = (
    ("detector", "detector_call"),
    ("language_model", "language_model_call"),
)
REFERENCES = ("pass1", "pass2")
_REFERENCE_COLUMN = {"pass1": "pass1_call", "pass2": "pass2_call"}


def _load_pr44() -> ModuleType:
    """Import ``scripts/44_precision_recall.py`` by path.

    The filename starts with a digit, so it cannot be imported by name. Loading
    it here keeps the metric definitions, the indeterminate handling, and the
    clustered ratio bootstrap identical to the primary precision/recall script
    instead of re-deriving them. The same path-import pattern is used by the
    existing test suite.

    Returns
    -------
    types.ModuleType
        The loaded ``44_precision_recall`` module.
    """
    script_path = Path(__file__).resolve().parent / "44_precision_recall.py"
    spec = importlib.util.spec_from_file_location("_pr44_for_62", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_pr44_for_62"] = module
    spec.loader.exec_module(module)
    return module


_PR44 = _load_pr44()

# Call-value constants and metric primitives reused verbatim from scripts/44.
OCCLUSION_SIGNATURE_PRESENT: str = _PR44.OCCLUSION_SIGNATURE_PRESENT
NO_OCCLUSION_SIGNATURE: str = _PR44.NO_OCCLUSION_SIGNATURE
INDETERMINATE: str = _PR44.INDETERMINATE
CALLABLE_VALUES: tuple[str, ...] = tuple(_PR44.CALLABLE_VALUES)
_metric_indicators = _PR44._metric_indicators
_ratio_ci = _PR44._ratio_ci

# Imported here so the uncallable-rate CI uses the same primitive as scripts/44.
from cuffcrt.analysis.bootstrap import cluster_bootstrap_ci  # noqa: E402


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_call(col: str) -> pl.Expr:
    """Lower-case, strip, and null-coalesce a call column to a clean string."""
    return (
        pl.col(col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        .alias(col)
    )


def load_sample(sample_csv: Path, manifest_csv: Path) -> pl.DataFrame:
    """Load the 150-card re-read sample and attach the clustering ``subject_id``.

    The sample already carries both machine-call columns and both reader passes;
    this only validates the schema, normalizes the call strings, and joins the
    ``subject_id`` from the gallery manifest (the cluster unit).

    Parameters
    ----------
    sample_csv : pathlib.Path
        ``results/gallery/reread_change_log_sample.csv``.
    manifest_csv : pathlib.Path
        ``results/gallery/gallery_manifest.csv`` (for ``subject_id``).

    Returns
    -------
    polars.DataFrame
        Columns ``card_id, subject_id, detector_call, language_model_call,
        pass1_call, pass2_call`` with normalized call strings.

    Raises
    ------
    ValueError
        If a required column is missing from either input, or if any sampled
        card lacks a ``subject_id`` in the manifest.
    """
    sample = pl.read_csv(sample_csv, infer_schema_length=20000)
    required = {
        "card_id",
        "detector_call",
        "language_model_call",
        "pass1_call",
        "pass2_call",
    }
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ValueError(f"sample_csv missing required columns: {missing}")
    sample = sample.with_columns(
        _normalize_call("detector_call"),
        _normalize_call("language_model_call"),
        _normalize_call("pass1_call"),
        _normalize_call("pass2_call"),
    )

    manifest = pl.read_csv(manifest_csv, infer_schema_length=20000)
    man_missing = sorted({"card_id", "subject_id"} - set(manifest.columns))
    if man_missing:
        raise ValueError(
            f"gallery_manifest missing required columns: {man_missing}. "
            "subject_id is required for the cluster-bootstrap CIs."
        )
    manifest = manifest.select(["card_id", "subject_id"]).unique(subset=["card_id"])

    joined = sample.join(manifest, on="card_id", how="left")
    n_no_subject = joined.filter(pl.col("subject_id").is_null()).height
    if n_no_subject:
        raise ValueError(
            f"{n_no_subject} sampled cards have no subject_id in the manifest; "
            "cannot form cluster-bootstrap units."
        )
    return joined.select(
        [
            "card_id",
            "subject_id",
            "detector_call",
            "language_model_call",
            "pass1_call",
            "pass2_call",
        ]
    ).sort("card_id")


def compute_two_reference_metrics(
    sample: pl.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pl.DataFrame:
    """Precision/recall/specificity for each index test under both references.

    For each index test (detector, language model) and each reference (pass 1,
    pass 2), a row is eligible for the binary metric iff both the reference call
    and the index-test call are one of the two callable values. The numerator
    and denominator per row, and the subject-clustered ratio CI, are computed
    with the scripts/44 primitives so the definitions match the primary
    analysis exactly.

    Parameters
    ----------
    sample : polars.DataFrame
        Output of :func:`load_sample`.
    n_bootstrap : int
        Number of cluster-bootstrap resamples for each CI.
    seed : int
        Seed for ``numpy.random.default_rng`` inside the ratio bootstrap.

    Returns
    -------
    polars.DataFrame
        Long format: one row per (``index_test``, ``metric``, ``reference``)
        with ``point_estimate``, ``ci_low``, ``ci_high``, ``n_used_for_metric``.
    """
    rows: list[dict[str, object]] = []
    callable_set = list(CALLABLE_VALUES)

    for index_test, call_col in INDEX_TESTS:
        predictor_calls_all = sample.get_column(call_col).to_list()
        for reference in REFERENCES:
            ref_col = _REFERENCE_COLUMN[reference]
            reference_calls_all = sample.get_column(ref_col).to_list()
            subjects_all = sample.get_column("subject_id").to_list()

            # Eligible rows: reference AND index-test call both callable. This
            # drops reference-indeterminate rows from the binary denominator,
            # matching scripts/44's _binary_eligible (the index tests have no
            # indeterminate/parse-failure calls in this sample).
            keep = [
                (r in callable_set) and (p in callable_set)
                for r, p in zip(reference_calls_all, predictor_calls_all, strict=True)
            ]
            ref_calls = [r for r, k in zip(reference_calls_all, keep, strict=True) if k]
            pred_calls = [p for p, k in zip(predictor_calls_all, keep, strict=True) if k]
            clusters = np.asarray(
                [s for s, k in zip(subjects_all, keep, strict=True) if k]
            )

            for metric in METRICS:
                if not ref_calls:
                    rows.append(
                        {
                            "index_test": index_test,
                            "metric": metric,
                            "reference": reference,
                            "point_estimate": float("nan"),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                            "n_used_for_metric": 0,
                        }
                    )
                    continue
                num, den = _metric_indicators(ref_calls, pred_calls, metric)
                point, ci_low, ci_high, n_used = _ratio_ci(
                    num, den, clusters, n_bootstrap=n_bootstrap, seed=seed
                )
                rows.append(
                    {
                        "index_test": index_test,
                        "metric": metric,
                        "reference": reference,
                        "point_estimate": point,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_used_for_metric": n_used,
                    }
                )
    return pl.DataFrame(rows)


def compute_delta_table(summary: pl.DataFrame) -> pl.DataFrame:
    """Pivot the long summary into one pass1-vs-pass2 row per (index_test, metric).

    Parameters
    ----------
    summary : polars.DataFrame
        Output of :func:`compute_two_reference_metrics`.

    Returns
    -------
    polars.DataFrame
        One row per (``index_test``, ``metric``) with the pass-1 and pass-2
        point/CI columns and ``delta`` = ``pass2_point - pass1_point``.
    """
    rows: list[dict[str, object]] = []
    for index_test, _ in INDEX_TESTS:
        for metric in METRICS:
            sub = summary.filter(
                (pl.col("index_test") == index_test) & (pl.col("metric") == metric)
            )

            def _cell(reference: str, field: str, _sub: pl.DataFrame = sub) -> float:
                r = _sub.filter(pl.col("reference") == reference)
                return float(r.get_column(field)[0]) if r.height else float("nan")

            p1 = _cell("pass1", "point_estimate")
            p2 = _cell("pass2", "point_estimate")
            rows.append(
                {
                    "index_test": index_test,
                    "metric": metric,
                    "pass1_point": p1,
                    "pass1_ci_low": _cell("pass1", "ci_low"),
                    "pass1_ci_high": _cell("pass1", "ci_high"),
                    "pass2_point": p2,
                    "pass2_ci_low": _cell("pass2", "ci_low"),
                    "pass2_ci_high": _cell("pass2", "ci_high"),
                    "delta": p2 - p1,
                }
            )
    return pl.DataFrame(rows)


def compute_uncallable_rates(
    sample: pl.DataFrame, *, n_bootstrap: int, seed: int
) -> pl.DataFrame:
    """Per-reference indeterminate count and rate with a clustered bootstrap CI.

    Only the reference (reader) produces indeterminate calls in this sample; the
    detector and language model have none. The rate is the fraction of the 150
    cards the reference left uncallable on that pass.

    Parameters
    ----------
    sample : polars.DataFrame
        Output of :func:`load_sample`.
    n_bootstrap : int
        Number of cluster-bootstrap resamples.
    seed : int
        Seed for the bootstrap.

    Returns
    -------
    polars.DataFrame
        One row per reference: ``reference``, ``indeterminate_count``,
        ``total_n``, ``indeterminate_rate``, ``ci_low``, ``ci_high``.
    """
    rows: list[dict[str, object]] = []
    clusters = np.asarray(sample.get_column("subject_id").to_list())
    n = sample.height
    for reference in REFERENCES:
        ref_col = _REFERENCE_COLUMN[reference]
        calls = np.asarray(sample.get_column(ref_col).to_list())
        is_indet = calls == INDETERMINATE
        n_indet = int(is_indet.sum())
        rate = float(is_indet.mean()) if n else float("nan")
        if n:
            res = cluster_bootstrap_ci(
                values=is_indet.astype(np.float64),
                clusters=clusters,
                n_resamples=n_bootstrap,
                seed=seed,
            )
            ci_low, ci_high = res.ci_low, res.ci_high
        else:
            ci_low = ci_high = float("nan")
        rows.append(
            {
                "reference": reference,
                "indeterminate_count": n_indet,
                "total_n": n,
                "indeterminate_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pl.DataFrame(rows)


def _reader_marginals(sample: pl.DataFrame, ref_col: str) -> dict[str, int]:
    """Count reference calls by value for the run-metadata record."""
    counts = (
        sample.group_by(ref_col)
        .agg(pl.len().alias("n"))
        .sort(ref_col)
    )
    return {
        str(v): int(c)
        for v, c in zip(
            counts.get_column(ref_col).to_list(),
            counts.get_column("n").to_list(),
            strict=True,
        )
    }


def _fmt_ci(point: float, lo: float, hi: float) -> str:
    """Format a point estimate with its CI as ``0.250 (0.120 to 0.390)``."""
    if not np.isfinite(point):
        return "n/a"
    return f"{point:.3f} ({lo:.3f} to {hi:.3f})"


def render_summary_md(
    delta: pl.DataFrame,
    uncallable: pl.DataFrame,
    *,
    n_cards: int,
    n_subjects: int,
    seed: int,
    n_bootstrap: int,
    timestamp: str,
) -> str:
    """Render the short Markdown summary, including the precision headline.

    Parameters
    ----------
    delta : polars.DataFrame
        Output of :func:`compute_delta_table`.
    uncallable : polars.DataFrame
        Output of :func:`compute_uncallable_rates`.
    n_cards : int
        Number of cards in the sample.
    n_subjects : int
        Number of distinct clustering subjects.
    seed : int
        Bootstrap seed.
    n_bootstrap : int
        Number of bootstrap resamples.
    timestamp : str
        UTC ISO timestamp string.

    Returns
    -------
    str
        Markdown document text (no em-dashes, American English).
    """
    pretty = {"detector": "rule-based detector", "language_model": "language model"}
    lines: list[str] = []
    lines.append("# Reference-correction sensitivity on the 150-card re-read sample")
    lines.append("")
    lines.append(
        "Feasibility/prevalence study. Pass 1 "
        "(`results/gallery/reader_form_blinded.csv`) remains the pre-specified "
        "primary reference for the paper and is not revised here. This report is "
        "a sensitivity analysis only: on the same 150 sampled cards it switches "
        "the reference from the pass-1 calls to the corrected pass-2 calls and "
        "reports how far the index tests' agreement metrics move. All values are "
        "morphology-based estimates; neither pass observes ground-truth cuff "
        "laterality."
    )
    lines.append("")
    lines.append(f"- Cards in the sample: {n_cards}")
    lines.append(f"- Clustering subjects: {n_subjects}")
    lines.append(f"- Bootstrap seed: {seed}; resamples: {n_bootstrap}")
    lines.append(f"- Generated (UTC): {timestamp}")
    lines.append("")
    lines.append("## Metrics under each reference, with pass1 -> pass2 delta")
    lines.append("")
    lines.append(
        "| index test | metric | pass 1 (95% CI) | pass 2 (95% CI) | delta |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for index_test, _ in INDEX_TESTS:
        for metric in METRICS:
            r = delta.filter(
                (pl.col("index_test") == index_test) & (pl.col("metric") == metric)
            )
            p1 = float(r.get_column("pass1_point")[0])
            p2 = float(r.get_column("pass2_point")[0])
            p1s = _fmt_ci(
                p1,
                float(r.get_column("pass1_ci_low")[0]),
                float(r.get_column("pass1_ci_high")[0]),
            )
            p2s = _fmt_ci(
                p2,
                float(r.get_column("pass2_ci_low")[0]),
                float(r.get_column("pass2_ci_high")[0]),
            )
            d = float(r.get_column("delta")[0])
            dstr = f"{d:+.3f}" if np.isfinite(d) else "n/a"
            lines.append(
                f"| {pretty[index_test]} | {metric} | {p1s} | {p2s} | {dstr} |"
            )
    lines.append("")
    lines.append("## Reference uncallable (indeterminate) rate per pass")
    lines.append("")
    lines.append("| reference | indeterminate | total | rate (95% CI) |")
    lines.append("| --- | --- | --- | --- |")
    for reference in REFERENCES:
        r = uncallable.filter(pl.col("reference") == reference)
        n_indet = int(r.get_column("indeterminate_count")[0])
        total = int(r.get_column("total_n")[0])
        rate = _fmt_ci(
            float(r.get_column("indeterminate_rate")[0]),
            float(r.get_column("ci_low")[0]),
            float(r.get_column("ci_high")[0]),
        )
        lines.append(f"| {reference} | {n_indet} | {total} | {rate} |")
    lines.append("")
    lines.append("## Precision headline")
    lines.append("")
    for index_test, _ in INDEX_TESTS:
        r = delta.filter(
            (pl.col("index_test") == index_test) & (pl.col("metric") == "precision")
        )
        p1 = float(r.get_column("pass1_point")[0])
        p2 = float(r.get_column("pass2_point")[0])
        d = p2 - p1
        verb = "rises" if d > 0 else ("falls" if d < 0 else "is unchanged")
        lines.append(
            f"- Correcting the undercalls moves {pretty[index_test]} precision "
            f"from {p1:.3f} (pass 1) to {p2:.3f} (pass 2); it {verb} by "
            f"{abs(d):.3f} on the 150-card sample."
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    *,
    summary: pl.DataFrame,
    delta: pl.DataFrame,
    uncallable: pl.DataFrame,
    summary_md: str,
    metadata: dict[str, object],
) -> None:
    """Write the five artifacts to ``out_dir`` (created if absent)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(out_dir / "reread_sensitivity_summary.csv")
    delta.write_csv(out_dir / "reread_sensitivity_delta.csv")
    uncallable.write_csv(out_dir / "reread_uncallable_rates.csv")
    (out_dir / "reread_sensitivity_summary.md").write_text(summary_md)
    with (out_dir / "run_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")


def run(
    *,
    sample_csv: Path,
    gallery_manifest: Path,
    out_dir: Path,
    seed: int,
    n_bootstrap: int,
) -> None:
    """End-to-end pipeline: load, compute under both references, write.

    Side effects: writes five files under ``out_dir`` and logs a one-line
    summary of the precision deltas via loguru. Never modifies any input.
    """
    logger.info(
        "loading sample={} manifest={}", sample_csv, gallery_manifest
    )
    sample = load_sample(sample_csv, gallery_manifest)
    n_cards = sample.height
    n_subjects = sample.get_column("subject_id").n_unique()
    logger.info("sample: {} cards across {} subjects", n_cards, n_subjects)

    summary = compute_two_reference_metrics(
        sample, n_bootstrap=n_bootstrap, seed=seed
    )
    delta = compute_delta_table(summary)
    uncallable = compute_uncallable_rates(
        sample, n_bootstrap=n_bootstrap, seed=seed
    )

    timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    summary_md = render_summary_md(
        delta,
        uncallable,
        n_cards=n_cards,
        n_subjects=n_subjects,
        seed=seed,
        n_bootstrap=n_bootstrap,
        timestamp=timestamp,
    )

    metadata: dict[str, object] = {
        "utc_timestamp": timestamp,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "n_cards": n_cards,
        "n_subjects": n_subjects,
        "inputs": {
            "sample_csv": {
                "path": str(sample_csv),
                "sha256": _sha256_of_file(sample_csv),
            },
            "gallery_manifest": {
                "path": str(gallery_manifest),
                "sha256": _sha256_of_file(gallery_manifest),
            },
        },
        "reader_marginals": {
            "pass1": _reader_marginals(sample, "pass1_call"),
            "pass2": _reader_marginals(sample, "pass2_call"),
        },
        "reference_note": (
            "Pass 1 (reader_form_blinded.csv) is the pre-specified primary "
            "reference and is unchanged. Pass 2 is the corrected re-read on the "
            "same 150 cards. This is a reference-correction sensitivity analysis; "
            "all metrics are morphology-based estimates with no ground-truth cuff "
            "laterality."
        ),
        "indeterminate_handling": (
            "Cards where the acting reference is indeterminate are excluded from "
            "the binary precision/recall/specificity denominator, matching "
            "scripts/44. The detector and language model produce no indeterminate "
            "or parse-failure calls on this sample, so the only uncallable cards "
            "come from the reference itself."
        ),
        "cluster_unit": "subject_id (from gallery_manifest)",
    }

    write_outputs(
        out_dir,
        summary=summary,
        delta=delta,
        uncallable=uncallable,
        summary_md=summary_md,
        metadata=metadata,
    )

    for index_test, _ in INDEX_TESTS:
        r = delta.filter(
            (pl.col("index_test") == index_test) & (pl.col("metric") == "precision")
        )
        logger.info(
            "{} precision pass1={:.3f} -> pass2={:.3f} (delta {:+.3f})",
            index_test,
            float(r.get_column("pass1_point")[0]),
            float(r.get_column("pass2_point")[0]),
            float(r.get_column("delta")[0]),
        )
    logger.info("wrote reread-sensitivity artifacts to {}", out_dir)


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Reference-correction sensitivity analysis on the 150-card re-read "
            "sample: detector and language-model precision/recall/specificity "
            "under the pass-1 vs pass-2 reference, with subject-clustered "
            "bootstrap 95% CIs and pass1->pass2 deltas."
        )
    )
    p.add_argument(
        "--sample_csv",
        type=Path,
        default=Path("results/gallery/reread_change_log_sample.csv"),
    )
    p.add_argument(
        "--gallery_manifest",
        type=Path,
        default=Path("results/gallery/gallery_manifest.csv"),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/precision_recall_reread_sensitivity/"),
    )
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_argparser().parse_args(argv)
    run(
        sample_csv=args.sample_csv,
        gallery_manifest=args.gallery_manifest,
        out_dir=args.out_dir,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
