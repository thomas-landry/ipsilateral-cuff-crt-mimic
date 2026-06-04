"""Per-variant concordance: prompt-sensitivity run vs canonical run (step 45).

D5 asks whether the headline positivity rate survives paraphrasing the
adjudication prompt. ``scripts/42`` re-runs the gallery subsample under several
prompt wordings and writes one finalized run log per variant, keyed by
``card_id``. The canonical headline run (``scripts/41``) is keyed by ``row_id``.
This script joins the two through the validated ``card_id -> row_id`` bridge
(:func:`cuffcrt.analysis.build_card_to_rowid`) and reports, per variant, how
often the variant's call matches the canonical call, overall and per stratum.

This script never calls a model. It only joins files already on disk: the
variant run logs from ``scripts/42``, the canonical run log from ``scripts/41``,
and the bridge built from the inventory and the gallery manifest. If the variant
outputs are absent (``scripts/42`` has not been run yet), it prints a clear
message and exits non-zero rather than crashing.

Variant output discovery
------------------------
``scripts/42`` writes its finalized per-variant run log as
``<variant>.parquet`` (alongside ``<variant>.csv``) under ``--variant-dir``,
where ``<variant>`` is the prompt-variant label (for example ``v_compact``). It
also leaves intermediate and provenance files in the same directory:
``<variant>_checkpoint.csv``, ``_run_manifest_<variant>_<utc>.json``, and
``_model_fingerprint_<utc>.json``. This script discovers variants by taking the
stem of every ``*.parquet`` file in ``--variant-dir`` that does not start with
``_``, so the set of variants scored is whatever finalized logs are present and
adding a variant in ``scripts/42`` needs no change here.

Outputs
-------
A per-variant CSV (``concordance_summary.csv``) and a JSON
(``concordance.json``) carrying the per-variant summary, the per-stratum
breakdown, and the canonical call distribution restricted to the scored
subsample. All outputs hold derived counts only.

Usage
-----
Once ``scripts/42`` has produced its variant outputs::

    uv run python scripts/45_prompt_sensitivity_concordance.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from loguru import logger  # pyright: ignore[reportMissingImports]

from cuffcrt.analysis.card_bridge import build_card_to_rowid

# The three call values MedGemma can emit; anything else is uncallable.
_PRESENT = "occlusion_signature_present"
_ABSENT = "no_occlusion_signature"
_INDETERMINATE = "indeterminate"

# Default I/O locations. The variant dir and canonical run are the only inputs
# that change between runs; the inventory and manifest are fixed repo paths.
DEFAULT_VARIANT_DIR = Path("results/medgemma_prompt_sensitivity")
DEFAULT_CANONICAL = Path(
    "results/medgemma/medgemma_adjudication_full_20260530T051617Z.parquet"
)
DEFAULT_INVENTORY = Path("data/interim/event_inventory.csv")
DEFAULT_MANIFEST = Path("results/gallery/gallery_manifest.csv")
DEFAULT_OUT_DIR = Path("results/medgemma_prompt_sensitivity")

# Filenames written by scripts/42 that are NOT finalized per-variant run logs:
# the per-variant manifest and the model fingerprint both start with ``_``. The
# concordance summary this script itself writes is a CSV (``concordance_summary``)
# and a JSON (``concordance``), not a parquet, so it cannot be misread as a
# variant log even when ``--out-dir`` and ``--variant-dir`` coincide.
_NON_VARIANT_PREFIX = "_"


def discover_variant_logs(variant_dir: Path) -> dict[str, Path]:
    """Find the finalized run-log parquet for each prompt variant.

    ``scripts/42`` finalizes each variant as ``<variant>.parquet`` in
    ``variant_dir``; the variant id is the file stem. Files whose name starts
    with ``_`` are provenance artifacts (``_run_manifest_*``,
    ``_model_fingerprint_*``) and are skipped, and the intermediate
    ``<variant>_checkpoint.csv`` is ignored because only ``*.parquet`` files are
    considered. The three labels are not hardcoded: any finalized variant
    parquet is discovered.

    Parameters
    ----------
    variant_dir : pathlib.Path
        Directory holding ``<variant>.parquet`` files written by ``scripts/42``.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from variant id (the parquet stem) to its finalized parquet
        path. Empty when the directory is absent or holds no finalized logs.
    """
    if not variant_dir.exists():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(variant_dir.glob("*.parquet")):
        if path.name.startswith(_NON_VARIANT_PREFIX):
            continue
        variant = path.stem
        # Prefer the plain finalized ``<variant>.parquet`` form if a stray
        # timestamped sibling (e.g. ``<variant>_<utc>.parquet``) also exists.
        existing = found.get(variant)
        if existing is None or len(path.name) < len(existing.name):
            found[variant] = path
    return found


def _safe_pct(numerator: int, denominator: int, *, digits: int = 2) -> float | None:
    """Return ``100 * numerator / denominator`` rounded, or ``None`` if empty."""
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, digits)


def compute_concordance(
    variant_frames: dict[str, pl.DataFrame],
    canonical: pl.DataFrame,
    bridge: pl.DataFrame,
) -> dict:
    """Compute per-variant concordance against the canonical run.

    For each variant, the variant calls (keyed by ``card_id``) are joined to the
    canonical calls (keyed by ``row_id``) through the bridge, and the fraction of
    cards whose variant call equals the canonical call is reported, overall and
    per stratum.

    Parameters
    ----------
    variant_frames : dict[str, polars.DataFrame]
        Mapping from variant id to that variant's run log. Each frame must carry
        ``card_id``, ``call``, and ``parsed_ok``.
    canonical : polars.DataFrame
        Canonical run log carrying ``row_id``, ``call``, and ``parsed_ok``.
    bridge : polars.DataFrame
        ``card_id -> row_id`` map from :func:`build_card_to_rowid`, carrying
        ``card_id``, ``stratum``, and ``row_id``.

    Returns
    -------
    dict
        ``{"per_variant": [...], "per_stratum": [...],
        "canonical_in_subsample": {...}}`` with counts and percentages.
    """
    canon = canonical.select(["row_id", "call", "parsed_ok"]).rename(
        {"call": "canonical_call", "parsed_ok": "canon_parsed_ok"}
    )
    card_canon = bridge.select(["card_id", "stratum", "row_id"]).join(
        canon, on="row_id", how="left"
    )

    summary: list[dict] = []
    per_stratum: list[dict] = []
    for variant in sorted(variant_frames):
        vdf = variant_frames[variant].select(["card_id", "call", "parsed_ok"]).rename(
            {"call": "variant_call", "parsed_ok": "variant_parsed_ok"}
        )
        joined = vdf.join(card_canon, on="card_id", how="inner")

        n_present = joined.filter(pl.col("variant_call") == _PRESENT).height
        n_absent = joined.filter(pl.col("variant_call") == _ABSENT).height
        n_indeterminate = joined.filter(pl.col("variant_call") == _INDETERMINATE).height
        n_var_parse_fail = joined.filter(~pl.col("variant_parsed_ok")).height
        n_canon_parse_fail = joined.filter(~pl.col("canon_parsed_ok")).height
        n_callable = n_present + n_absent + n_indeterminate
        positive_rate = _safe_pct(n_present, n_callable)

        both = joined.filter(
            pl.col("variant_call").is_not_null()
            & pl.col("canonical_call").is_not_null()
        )
        n_compare = both.height
        n_matched = both.filter(
            pl.col("variant_call") == pl.col("canonical_call")
        ).height
        concordance = _safe_pct(n_matched, n_compare)
        var_pos_canon_neg = both.filter(
            (pl.col("variant_call") == _PRESENT)
            & (pl.col("canonical_call") == _ABSENT)
        ).height
        var_neg_canon_pos = both.filter(
            (pl.col("variant_call") == _ABSENT)
            & (pl.col("canonical_call") == _PRESENT)
        ).height
        other_mismatch = n_compare - n_matched - var_pos_canon_neg - var_neg_canon_pos

        summary.append(
            {
                "variant": variant,
                "n_total": joined.height,
                "n_compare": n_compare,
                "n_matched": n_matched,
                "concordance_pct": concordance,
                "var_present": n_present,
                "var_absent": n_absent,
                "var_indeterminate": n_indeterminate,
                "var_parse_failure": n_var_parse_fail,
                "canon_parse_fail_in_sub": n_canon_parse_fail,
                "var_positive_rate_pct": positive_rate,
                "var_pos_canon_neg": var_pos_canon_neg,
                "var_neg_canon_pos": var_neg_canon_pos,
                "other_mismatch": other_mismatch,
            }
        )

        grouped = (
            both.with_columns(
                (pl.col("variant_call") == pl.col("canonical_call")).alias("_match")
            )
            .group_by("stratum")
            .agg(pl.len().alias("n"), pl.col("_match").sum().alias("matched"))
            .sort("stratum")
        )
        for row in grouped.iter_rows(named=True):
            per_stratum.append(
                {
                    "variant": variant,
                    "stratum": row["stratum"],
                    "n": row["n"],
                    "matched": row["matched"],
                    "conc_pct": _safe_pct(row["matched"], row["n"], digits=1),
                }
            )

    sub = card_canon.filter(pl.col("canonical_call").is_not_null())
    n_sub = sub.height
    n_sub_present = sub.filter(pl.col("canonical_call") == _PRESENT).height
    canonical_in_subsample = {
        "n": n_sub,
        "present": n_sub_present,
        "absent": sub.filter(pl.col("canonical_call") == _ABSENT).height,
        "indeterminate": sub.filter(pl.col("canonical_call") == _INDETERMINATE).height,
        "positive_rate_pct": _safe_pct(n_sub_present, n_sub),
    }

    return {
        "per_variant": summary,
        "per_stratum": per_stratum,
        "canonical_in_subsample": canonical_in_subsample,
    }


def _write_outputs(result: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write the concordance summary CSV and the full JSON, return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "concordance_summary.csv"
    json_path = out_dir / "concordance.json"
    pl.DataFrame(result["per_variant"]).write_csv(csv_path)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variant-dir",
        type=Path,
        default=DEFAULT_VARIANT_DIR,
        help="Directory of scripts/42 finalized variant run logs (<variant>.parquet).",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=DEFAULT_CANONICAL,
        help="Canonical headline run log from scripts/41 (parquet).",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
        help="Consolidated event inventory CSV (for the card->row_id bridge).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Gallery manifest CSV (for the card->row_id bridge).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for concordance_summary.csv and concordance.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Join the variant logs to the canonical run and write concordance.

    Returns
    -------
    int
        Process exit code: 0 on success, 2 when inputs are missing (including
        the expected "variant outputs not present yet" case).
    """
    args = _parse_args(argv)

    variant_logs = discover_variant_logs(args.variant_dir)
    if not variant_logs:
        logger.error(
            "no variant run logs found under {}; run scripts/42 first "
            "(uv run python scripts/42_prompt_sensitivity.py)",
            args.variant_dir,
        )
        return 2
    logger.info(
        "found {} variant log(s): {}",
        len(variant_logs),
        ", ".join(f"{v}={p.name}" for v, p in sorted(variant_logs.items())),
    )

    if not args.canonical.exists():
        logger.error("canonical run log not found: {}", args.canonical)
        return 2

    try:
        bridge = build_card_to_rowid(args.inventory, args.manifest)
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        return 2

    n_unmatched = bridge.filter(pl.col("row_id").is_null()).height
    if n_unmatched:
        logger.warning(
            "{} of {} cards did not resolve to a row_id; "
            "those cards drop out of the concordance",
            n_unmatched,
            bridge.height,
        )

    canonical = pl.read_parquet(args.canonical)
    variant_frames = {v: pl.read_parquet(p) for v, p in variant_logs.items()}

    result = compute_concordance(variant_frames, canonical, bridge)

    csv_path, json_path = _write_outputs(result, args.out_dir)
    for entry in result["per_variant"]:
        logger.info(
            "{}: concordance={}% (matched {}/{}), positive_rate={}%",
            entry["variant"],
            entry["concordance_pct"],
            entry["n_matched"],
            entry["n_compare"],
            entry["var_positive_rate_pct"],
        )
    logger.info("wrote concordance -> {} and {}", csv_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
