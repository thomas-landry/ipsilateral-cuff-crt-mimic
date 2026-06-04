"""Determinism re-run helper for the MedGemma adjudication (step 43, D6).

Re-adjudicates a deterministic 100-row subsample of a prior MedGemma run in a
**fresh oMLX server process**, then reports row-level paired agreement against
the original run. This is the D6 "determinism re-run" companion to the headline
adjudication (script 41); together with the prompt-sensitivity helper (script
42) it characterizes how stable the model's calls are under nominally identical
decoding.

The "fresh process" matters: oMLX-served vision-language models are not
guaranteed bit-identical across server restarts (CPU/GPU scheduling, BLAS
kernel selection, JIT caching). Re-running inside the same long-lived server
process would systematically understate that drift. To enforce fresh-process
semantics, this script does **not** start the oMLX server itself: the operator
must start a second oMLX process on a port distinct from the headline run's
port (default ``--port 8001`` here so the headline 8000 is not reused), point
this script at it via ``--base-url`` (constructed automatically from
``--port``), and let the script issue calls. The harness picks up the new
server-start ``_model_fingerprint_*.json`` from ``--out_dir`` the same way 41
does, so the run log records whatever weights the new server reports.

Inputs (read-only)
------------------
- ``--first_run_csv``: a canonical adjudication CSV produced by script 41
  (canonical run log columns plus ``subject_id`` and ``record_id``). The
  subsample is drawn from this file, **not** from the gallery manifest, so the
  re-run hits exactly the row_ids the headline run touched. Rows with no
  ``call`` set (parse failures in the headline run) are still eligible: the
  determinism question is about reproducibility of the model, not of a
  successful parse.
- ``--inventory``: the same candidate event inventory that 41 used. Required
  because the first-run CSV does not carry ``nbp_timestamp_s``, which the
  render path needs to re-cut the PI(t) window.
- ``--wdb-root``: same WDB tree 41 read.

Outputs
-------
- ``<out_dir>/determinism_rerun_<UTC>.parquet`` and ``.csv``: the re-run log
  for the 100-row subsample, schema-compatible with 41's adjudication output.
- ``<out_dir>/agreement_summary.csv``: point estimates for overall row-level
  agreement (%), per-call confusion (3 x 3 over the canonical call vocabulary),
  and parse-failure rates for both runs. No confidence intervals here; the
  precision/recall analysis (script 44) is where bootstraps live.

This script must be run as a plain detached process, not wrapped in an
interactive runtime.

Example
-------
::

    # Start a fresh oMLX server on port 8001 in a separate terminal first.
    OMLX_MODEL_DIR=/path/to/oMLX/cache/mlx-community--medgemma-1.5-4b-it-bf16 \\
    OMLX_MODEL_ID=mlx-community/medgemma-1.5-4b-it-bf16 \\
    ./scripts/compute_model_sha.sh  # writes a fingerprint under --out_dir

    uv run python scripts/43_determinism_rerun.py \\
        --first_run_csv results/medgemma/medgemma_adjudication_full_<UTC>.csv \\
        --n_rows 100 \\
        --seed 20260426 \\
        --out_dir results/medgemma_determinism/ \\
        --port 8001
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
import socket
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import polars as pl
from loguru import logger

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.llm.client import resolve_base_url
from cuffcrt.llm.medgemma import (
    DEFAULT_TEMPERATURE,
    RUN_LOG_COLUMNS,
    VALID_CALLS,
)
from cuffcrt.llm.model_fingerprint import load_latest_fingerprint
from cuffcrt.llm.prompts import load_adjudicate_prompts

# Default output location for D6 artifacts.
DEFAULT_OUT_DIR = Path("results/medgemma_determinism")

# Distinct port to encourage a separate oMLX process from the headline run.
DEFAULT_PORT = 8001

# Number of rows in the deterministic subsample (D6 spec).
DEFAULT_N_ROWS = 100

# Reuse step 41's harness functions rather than re-implementing them. The file
# starts with a numeric prefix so a plain ``import`` does not work; load it by
# spec the same way the existing tests do.
SCRIPTS_DIR = Path(__file__).resolve().parent
DRIVER_PATH = SCRIPTS_DIR / "41_run_medgemma_adjudication.py"

# Columns added by the harness on top of the canonical run-log columns.
LOG_COLUMNS_WITH_IDS = (
    "subject_id",
    "record_id",
    *RUN_LOG_COLUMNS,
    "model_weights_sha256",
)


def _load_driver_module() -> ModuleType:
    """Load ``scripts/41_run_medgemma_adjudication.py`` as a Python module.

    Returns
    -------
    types.ModuleType
        The imported driver module. Reused for ``_build_client``,
        ``_adjudicate_one``, ``_utc_now_iso``, ``_utc_now_stamp``, and the
        underlying row-render path so this script is a thin wrapper on
        top of the canonical harness.
    """
    spec = importlib.util.spec_from_file_location(
        "medgemma_adjudication_driver", DRIVER_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import wiring
        raise RuntimeError(f"could not load driver module at {DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["medgemma_adjudication_driver"] = module
    spec.loader.exec_module(module)
    return module


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision)."""
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0).isoformat()


def _utc_now_stamp() -> str:
    """Return the current UTC time as a filename-safe stamp."""
    return dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _base_url_from_port(port: int) -> str:
    """Build a localhost oMLX base URL from a port (``http://localhost:<port>/v1``)."""
    return f"http://localhost:{port}/v1"


def subsample_row_ids(
    first_run_df: pl.DataFrame,
    *,
    n_rows: int,
    seed: int,
) -> list[str]:
    """Deterministically choose ``n_rows`` row_ids from a first-run CSV.

    The selection uses :class:`numpy.random.Generator` (``default_rng(seed)``)
    so two invocations with the same seed and the same first-run input yield
    the same row_ids in the same order. When the input has fewer than
    ``n_rows`` rows the whole input is returned in its original order, so the
    re-run still covers the available universe.

    Parameters
    ----------
    first_run_df : polars.DataFrame
        First-run log; must include a ``row_id`` column.
    n_rows : int
        Target subsample size.
    seed : int
        RNG seed (typically :data:`cuffcrt._seed.GLOBAL_SEED`).

    Returns
    -------
    list[str]
        Chosen row_ids in the deterministic draw order.
    """
    if "row_id" not in first_run_df.columns:
        raise ValueError("first_run_csv is missing a 'row_id' column")
    universe = first_run_df.get_column("row_id").to_list()
    if not universe:
        return []
    if n_rows <= 0:
        return []
    if n_rows >= len(universe):
        logger.warning(
            "first-run input has {} rows (< requested {}); using all rows",
            len(universe),
            n_rows,
        )
        return list(universe)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(universe), size=n_rows, replace=False)
    indices.sort()
    return [universe[int(i)] for i in indices]


def load_first_run(first_run_csv: Path) -> pl.DataFrame:
    """Load the first-run adjudication CSV and validate the canonical schema.

    Parameters
    ----------
    first_run_csv : pathlib.Path
        Path to the first-run adjudication CSV (output of script 41).

    Returns
    -------
    polars.DataFrame
        The first-run frame.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if not first_run_csv.exists():
        raise FileNotFoundError(f"first_run_csv not found: {first_run_csv}")
    df = pl.read_csv(first_run_csv, infer_schema_length=20000)
    required = {"row_id", "subject_id", "record_id", "call", "parsed_ok"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"first_run_csv {first_run_csv} is missing required columns: "
            f"{sorted(missing)}"
        )
    return df


def join_inventory(
    selected_row_ids: list[str],
    inventory_path: Path,
    driver: ModuleType,
) -> pl.DataFrame:
    """Look up subject/record/timestamp for each selected row_id.

    Reuses the driver's :func:`build_event_frame` so row_id construction is
    bit-identical to the headline run. Any row_id that does not appear in the
    inventory is dropped with a loguru warning; the surviving frame is what
    drives the re-render path.

    Parameters
    ----------
    selected_row_ids : list[str]
        Subsample picked by :func:`subsample_row_ids`.
    inventory_path : pathlib.Path
        Same inventory CSV the headline run consumed.
    driver : types.ModuleType
        The loaded step-41 driver module.

    Returns
    -------
    polars.DataFrame
        Rows from the inventory in the order of ``selected_row_ids``.
    """
    if not selected_row_ids:
        return pl.DataFrame(
            {
                "subject_id": [],
                "record_id": [],
                "row_id": [],
                "nbp_timestamp_s": [],
            }
        )
    inventory = driver.build_event_frame(inventory_path)
    selected_set = set(selected_row_ids)
    matched = inventory.filter(pl.col("row_id").is_in(selected_set))
    missing = selected_set - set(matched.get_column("row_id").to_list())
    if missing:
        logger.warning(
            "{} selected row_ids missing from inventory and dropped: {}",
            len(missing),
            sorted(missing)[:5],
        )
    # Preserve the deterministic draw order so the run log is reproducible.
    order = pl.DataFrame(
        {"row_id": selected_row_ids, "_order": list(range(len(selected_row_ids)))}
    )
    matched = matched.join(order, on="row_id", how="inner").sort("_order").drop("_order")
    return matched


def _append_checkpoint_row(row: dict, checkpoint_path: Path) -> None:
    """Append one completed row to a CSV checkpoint and fsync it to disk.

    Mirrors :func:`scripts.41_run_medgemma_adjudication._append_checkpoint_row`
    so the re-run is resumable in the same shape as the headline run.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0
    )
    with checkpoint_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(LOG_COLUMNS_WITH_IDS), extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in LOG_COLUMNS_WITH_IDS})
        f.flush()
        os.fsync(f.fileno())


def _write_run_log(
    rows: list[dict], out_dir: Path, *, stamp: str
) -> tuple[Path, Path]:
    """Write the re-run log as CSV and parquet with deterministic column order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).select(list(LOG_COLUMNS_WITH_IDS))
    csv_path = out_dir / f"determinism_rerun_{stamp}.csv"
    parquet_path = out_dir / f"determinism_rerun_{stamp}.parquet"
    df.write_csv(csv_path)
    df.write_parquet(parquet_path)
    return csv_path, parquet_path


def compute_agreement(
    first_run_df: pl.DataFrame, rerun_df: pl.DataFrame
) -> dict[str, object]:
    """Compute paired agreement metrics for the determinism re-run.

    The join is an inner join on ``row_id``; row_ids that appear in only one
    frame are surfaced explicitly under ``missing_from_first`` /
    ``missing_from_rerun`` rather than silently turning into NaNs.

    Parameters
    ----------
    first_run_df : polars.DataFrame
        Headline-run log, must contain ``row_id`` and ``call``.
    rerun_df : polars.DataFrame
        Determinism re-run log, must contain ``row_id`` and ``call``.

    Returns
    -------
    dict
        Keys: ``n_first``, ``n_rerun``, ``n_paired``, ``n_agree``,
        ``overall_agreement_pct``, ``confusion`` (3 x 3 dict-of-dicts over
        :data:`VALID_CALLS`, only over rows where both runs produced a valid
        call), ``parse_failure_rate_first``, ``parse_failure_rate_rerun``,
        ``missing_from_first``, ``missing_from_rerun``.
    """
    first_ids = set(first_run_df.get_column("row_id").to_list())
    rerun_ids = set(rerun_df.get_column("row_id").to_list())
    missing_from_first = sorted(rerun_ids - first_ids)
    missing_from_rerun = sorted(first_ids - rerun_ids)

    first_small = first_run_df.select(["row_id", "call"]).rename({"call": "call_first"})
    rerun_small = rerun_df.select(["row_id", "call"]).rename({"call": "call_rerun"})
    paired = first_small.join(rerun_small, on="row_id", how="inner")
    n_paired = paired.height

    n_agree = int(
        paired.filter(
            (pl.col("call_first") == pl.col("call_rerun"))
            & pl.col("call_first").is_not_null()
        ).height
    )
    overall_agreement_pct = (n_agree / n_paired * 100.0) if n_paired else 0.0

    # 3 x 3 confusion over the canonical vocabulary, restricted to rows where
    # both runs produced a valid call. Rows with a null call on either side
    # contribute to the parse-failure tallies, not to the matrix.
    confusion: dict[str, dict[str, int]] = {
        call: {c: 0 for c in VALID_CALLS} for call in VALID_CALLS
    }
    counts: Counter[tuple[str, str]] = Counter()
    for row in paired.iter_rows(named=True):
        first = row["call_first"]
        rerun = row["call_rerun"]
        if first in VALID_CALLS and rerun in VALID_CALLS:
            counts[(first, rerun)] += 1
    for (first_call, rerun_call), n in counts.items():
        confusion[first_call][rerun_call] = int(n)

    def _parse_failure_rate(df: pl.DataFrame) -> float:
        n_total = df.height
        if n_total == 0:
            return 0.0
        n_null = int(df.filter(pl.col("call").is_null()).height)
        return n_null / n_total * 100.0

    return {
        "n_first": int(first_run_df.height),
        "n_rerun": int(rerun_df.height),
        "n_paired": int(n_paired),
        "n_agree": int(n_agree),
        "overall_agreement_pct": float(overall_agreement_pct),
        "confusion": confusion,
        "parse_failure_rate_first_pct": _parse_failure_rate(first_run_df),
        "parse_failure_rate_rerun_pct": _parse_failure_rate(rerun_df),
        "missing_from_first": missing_from_first,
        "missing_from_rerun": missing_from_rerun,
    }


def write_agreement_summary(
    agreement: dict[str, object], out_dir: Path
) -> Path:
    """Write a tidy ``agreement_summary.csv`` of the agreement metrics.

    Two row groups: scalar metrics (one ``metric, value`` row each) followed by
    the 3 x 3 confusion matrix (one ``confusion[first][rerun], count`` row per
    cell). This keeps the file plain-text greppable and Polars-readable while
    avoiding JSON-in-CSV nesting.

    Parameters
    ----------
    agreement : dict
        Output of :func:`compute_agreement`.
    out_dir : pathlib.Path
        Output directory (created if absent).

    Returns
    -------
    pathlib.Path
        The written summary CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "agreement_summary.csv"
    rows: list[dict[str, object]] = []
    scalar_keys = (
        "n_first",
        "n_rerun",
        "n_paired",
        "n_agree",
        "overall_agreement_pct",
        "parse_failure_rate_first_pct",
        "parse_failure_rate_rerun_pct",
    )
    for key in scalar_keys:
        rows.append({"metric": key, "value": agreement[key]})
    missing_from_first = cast(list[str], agreement["missing_from_first"])
    missing_from_rerun = cast(list[str], agreement["missing_from_rerun"])
    rows.append({"metric": "n_missing_from_first", "value": len(missing_from_first)})
    rows.append({"metric": "n_missing_from_rerun", "value": len(missing_from_rerun)})
    confusion = cast(dict[str, dict[str, int]], agreement["confusion"])
    for first_call in VALID_CALLS:
        for rerun_call in VALID_CALLS:
            rows.append(
                {
                    "metric": f"confusion[{first_call}->{rerun_call}]",
                    "value": confusion[first_call][rerun_call],
                }
            )
    pl.DataFrame(rows).write_csv(summary_path)
    return summary_path


def _write_run_manifest(
    *,
    out_dir: Path,
    stamp: str,
    agreement: dict[str, object],
    args: argparse.Namespace,
    base_url: str,
    fingerprint,
    selected_row_ids: list[str],
    system_prompt_sha: str,
    user_prompt_sha: str,
    run_utc_start: str,
    run_utc_end: str,
) -> Path:
    """Write a per-run manifest JSON capturing the re-run provenance."""
    payload = {
        "kind": "determinism_rerun",
        "first_run_csv": str(args.first_run_csv),
        "inventory": str(args.inventory),
        "wdb_root": str(args.wdb_root),
        "n_rows_requested": int(args.n_rows),
        "n_rows_selected": len(selected_row_ids),
        "seed": int(args.seed),
        "port": int(args.port),
        "server_endpoint": base_url,
        "host": socket.gethostname(),
        "run_utc_start": run_utc_start,
        "run_utc_end": run_utc_end,
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
        "prompt_system_sha256": system_prompt_sha,
        "prompt_user_sha256": user_prompt_sha,
        "decoding": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
        "agreement_overall_pct": agreement["overall_agreement_pct"],
        "n_paired": agreement["n_paired"],
        "n_agree": agreement["n_agree"],
        "parse_failure_rate_first_pct": agreement["parse_failure_rate_first_pct"],
        "parse_failure_rate_rerun_pct": agreement["parse_failure_rate_rerun_pct"],
        "dry_run": bool(args.dry_run),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"_run_manifest_determinism_{stamp}.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the determinism re-run CLI options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--first_run_csv",
        type=Path,
        required=True,
        help="First-run MedGemma adjudication CSV to re-adjudicate from.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help=(
            "Candidate event inventory CSV (same one step 41 read). Needed for "
            "nbp_timestamp_s when re-rendering the PI(t) window."
        ),
    )
    parser.add_argument(
        "--wdb-root",
        type=Path,
        required=True,
        help="Root of the WDB tree (directory containing waves/).",
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
        help=f"RNG seed for the subsample draw (default: GLOBAL_SEED={GLOBAL_SEED}).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=(
            "Port of the fresh oMLX server process the operator must have "
            f"started in a separate terminal (default: {DEFAULT_PORT}; chosen "
            "distinct from the headline 8000 to encourage a separate process)."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "Override the oMLX base URL. Default is constructed from --port "
            "as http://localhost:<port>/v1."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medgemma-1.5-4b-it-bf16",
        help="Served model id on the fresh oMLX server.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("data/scratch_medgemma_plots") / "determinism",
        help="Gitignored directory for re-rendered PI(t) plots.",
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
        default=1536,
        help="Maximum new tokens (default: 1536, matches step 41).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use an in-process stub client (no server, no network).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Re-adjudicate a 100-row subsample and write the paired-agreement summary.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on input errors).
    """
    args = _parse_args(argv)
    driver = _load_driver_module()
    driver._load_dotenv_if_present()
    from cuffcrt import figstyle  # noqa: PLC0415 - lazy; only needed if we run

    figstyle.apply_style()

    # Verify the frozen prompts before any network calls. Drift here would
    # change the model's behavior; better to stop now than after a long run.
    system_prompt, user_prompt = load_adjudicate_prompts()
    logger.info(
        "prompts loaded: system_sha256={} user_sha256={}",
        system_prompt.sha256,
        user_prompt.sha256,
    )

    if not args.first_run_csv.exists():
        logger.error("first_run_csv not found: {}", args.first_run_csv)
        return 2
    if not args.inventory.exists():
        logger.error("inventory not found: {}", args.inventory)
        return 2
    if not args.wdb_root.exists():
        logger.error("WDB root not found: {}", args.wdb_root)
        return 2
    if args.n_rows <= 0:
        logger.error("--n_rows must be positive, got {}", args.n_rows)
        return 2

    base_url = args.base_url or _base_url_from_port(args.port)
    # Honor an explicit override; otherwise the constructed URL above wins.
    resolved_base_url = resolve_base_url(base_url)
    logger.info(
        "determinism re-run starting; port={} base_url={} (must be a FRESH oMLX server)",
        args.port,
        resolved_base_url,
    )

    fingerprint = load_latest_fingerprint(args.out_dir)
    driver._log_fingerprint(fingerprint)

    first_run_df = load_first_run(args.first_run_csv)
    logger.info(
        "first-run input: {} rows from {}",
        first_run_df.height,
        args.first_run_csv,
    )

    selected_row_ids = subsample_row_ids(
        first_run_df, n_rows=args.n_rows, seed=args.seed
    )
    logger.info("selected {} row_ids (seed={})", len(selected_row_ids), args.seed)

    pool = join_inventory(selected_row_ids, args.inventory, driver)
    if pool.height == 0:
        logger.error("no selected row_ids survived the inventory join; aborting")
        return 2

    # Build a fake args namespace the driver's _adjudicate_one expects.
    driver_args = argparse.Namespace(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        model=args.model,
        base_url=resolved_base_url,
        dry_run=args.dry_run,
    )
    client = driver._build_client(driver_args)
    scratch_dir = args.scratch_dir

    stamp = _utc_now_stamp()
    checkpoint_path = args.out_dir / f"determinism_rerun_{stamp}_checkpoint.csv"
    run_utc_start = _utc_now_iso()
    rerun_rows: list[dict] = []
    for record in pool.iter_rows(named=True):
        row = driver._adjudicate_one(
            client,
            record,
            wdb_root=args.wdb_root,
            scratch_dir=scratch_dir,
            base_url=resolved_base_url,
            args=driver_args,
            fingerprint=fingerprint,
        )
        rerun_rows.append(row)
        _append_checkpoint_row(row, checkpoint_path)

    csv_path, parquet_path = _write_run_log(rerun_rows, args.out_dir, stamp=stamp)
    logger.info("wrote re-run log -> {} and {}", csv_path, parquet_path)

    rerun_df = pl.DataFrame(rerun_rows).select(list(LOG_COLUMNS_WITH_IDS))
    agreement = compute_agreement(first_run_df, rerun_df)
    summary_path = write_agreement_summary(agreement, args.out_dir)
    logger.info(
        "agreement: paired={} agree={} overall={:.2f}% (parse-fail first={:.2f}% rerun={:.2f}%)",
        agreement["n_paired"],
        agreement["n_agree"],
        agreement["overall_agreement_pct"],
        agreement["parse_failure_rate_first_pct"],
        agreement["parse_failure_rate_rerun_pct"],
    )
    logger.info("wrote agreement summary -> {}", summary_path)

    run_utc_end = _utc_now_iso()
    manifest_path = _write_run_manifest(
        out_dir=args.out_dir,
        stamp=stamp,
        agreement=agreement,
        args=args,
        base_url=resolved_base_url,
        fingerprint=fingerprint,
        selected_row_ids=selected_row_ids,
        system_prompt_sha=system_prompt.sha256,
        user_prompt_sha=user_prompt.sha256,
        run_utc_start=run_utc_start,
        run_utc_end=run_utc_end,
    )
    logger.info("wrote run manifest -> {}", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
