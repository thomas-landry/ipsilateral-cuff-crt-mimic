"""Local MedGemma (oMLX) inference harness (pipeline step 40).

This is a plain client to a local, OpenAI-compatible server. Start the server
separately, for example::

    omlx serve mlx-community/medgemma-1.5-4b-it-bf16 --port 8000

then point this script at it. The path is a single deterministic chat-completions
call with no autonomous tooling around it.

Two subcommands:

``extract``
    TEXT mode. Read de-identified note / structured-field text (one record per
    input file, or a directory of ``*.txt``) and emit a small structured
    phenotype object per record. The note text is NEVER written to the run log;
    only its SHA-256 (via the prompt hash) is retained.

``adjudicate``
    IMAGE mode. Read unannotated perfusion-index-versus-time plots (``*.png``)
    and emit ``{observed, call, confidence, rationale}`` per plot, where
    ``call`` is one of ``{"occlusion_signature_present",
    "no_occlusion_signature", "indeterminate"}``. The reviewer is blinded to
    the detector's verdict (it is never placed in the prompt).

Determinism and provenance. Decoding is ``temperature=0`` with a fixed
``max_tokens`` and ``seed=GLOBAL_SEED``. Every row of the run log records the
model id, prompt SHA-256, base URL, temperature, seed, and UTC timestamp, so
any output is reproducible and auditable. The run log is written as both CSV
and parquet under ``--out``.

Flags
-----
``--demo``
    Resolve the demo input directory for the chosen mode under ``--data-root``
    (no credentialing), via :mod:`cuffcrt._paths`. The demo inputs are derived
    artifacts produced by the upstream steps run in demo mode (text notes for
    ``extract``, perfusion plots for ``adjudicate``); ``--demo`` only locates
    that directory and fails clean with a pointer to ``data/README.md`` when it
    is absent. The LLM call itself is data-agnostic. An explicit ``--input-dir``
    or ``--input-file`` always overrides ``--demo``.
``--dry-run``
    Use an in-process stub client that returns canned, well-formed JSON. No
    server and no network are contacted. Useful for CI and for verifying the
    run-log shape.
``--out``
    Output directory for the run log (created if absent). The script refuses to
    write the run log into its own input directory.

Examples
--------
Dry run over a directory of plots (no server)::

    uv run python scripts/40_medgemma_inference.py adjudicate \\
        --input-dir data/interim/plots --out data/interim/llm --dry-run

Live text extraction against a local oMLX server::

    OMLX_BASE_URL=http://localhost:8000/v1 OMLX_API_KEY=... \\
    uv run python scripts/40_medgemma_inference.py extract \\
        --input-dir data/interim/notes --out data/interim/llm
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import polars as pl
from loguru import logger

from cuffcrt._paths import require_path, resolve_demo_llm_input_dir
from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.llm.client import DEFAULT_MODEL, OMLXClient, StubClient, resolve_base_url
from cuffcrt.llm.medgemma import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    RUN_LOG_COLUMNS,
    adjudicate_image,
    adjudication_log_row,
    extract_text,
    extraction_log_row,
)


def _load_dotenv_if_present() -> None:
    """Load a repo ``.env`` into the environment if python-dotenv is installed.

    Sourcing ``.env`` is optional: the harness reads ``OMLX_*`` from the
    environment regardless. The key is never logged.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision)."""
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0).isoformat()


def _build_client(args: argparse.Namespace):
    """Construct the stub (dry-run) or live oMLX client."""
    base_url = resolve_base_url(args.base_url)
    if args.dry_run:
        logger.info("dry-run: using in-process stub client (no server, no network)")
        return StubClient(model=args.model, base_url=base_url)
    logger.info("live: oMLX client model={} base_url={}", args.model, base_url)
    return OMLXClient(model=args.model, base_url=base_url)


def _resolve_input_dir(args: argparse.Namespace, *, mode: str) -> Path:
    """Resolve the directory to glob for inputs.

    An explicit ``--input-dir`` always wins. Otherwise, with ``--demo`` set, the
    demo default for the mode is resolved via :mod:`cuffcrt._paths` and required
    to exist (failing clean with a pointer to ``data/README.md`` when absent).
    With neither given, the caller must supply an input.
    """
    if args.input_dir is not None:
        return args.input_dir
    if args.demo:
        demo_dir = resolve_demo_llm_input_dir(args.data_root, mode=mode)
        # Fail clean (not with a confusing empty-glob no-op) when the demo
        # inputs have not been produced yet by the upstream steps.
        require_path(demo_dir, what=f"demo {mode} inputs")
        return demo_dir
    raise FileNotFoundError("provide --input-file or --input-dir (or use --demo)")


def _collect_inputs(args: argparse.Namespace, *, mode: str, suffix: str) -> list[Path]:
    """Return sorted input files for the chosen mode.

    A single ``--input-file`` takes precedence; otherwise the input directory
    (explicit ``--input-dir`` or the resolved ``--demo`` default) is globbed for
    ``*{suffix}`` files.
    """
    if args.input_file is not None:
        return [args.input_file]
    input_dir = _resolve_input_dir(args, mode=mode)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    return sorted(input_dir.glob(f"*{suffix}"))


def _guard_output_dir(out_dir: Path, inputs: list[Path]) -> None:
    """Refuse to write outputs into the input directory (no self-overwrite)."""
    out_resolved = out_dir.resolve()
    input_parents = {p.resolve().parent for p in inputs}
    if out_resolved in input_parents:
        raise ValueError(
            f"--out ({out_dir}) must differ from the input directory; "
            "refusing to write outputs alongside inputs."
        )


def _write_run_log(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """Write the run log as CSV and parquet with the canonical column order.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        ``(csv_path, parquet_path)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).select(list(RUN_LOG_COLUMNS))
    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"medgemma_runlog_{stamp}.csv"
    parquet_path = out_dir / f"medgemma_runlog_{stamp}.parquet"
    df.write_csv(csv_path)
    df.write_parquet(parquet_path)
    return csv_path, parquet_path


def _run_extract(args: argparse.Namespace, client) -> list[dict]:
    """Run text extraction over the collected inputs; return run-log rows."""
    inputs = _collect_inputs(args, mode="extract", suffix=".txt")
    _guard_output_dir(args.out, inputs)
    base_url = resolve_base_url(args.base_url)
    rows: list[dict] = []
    for path in inputs:
        context_text = path.read_text(encoding="utf-8")
        result, sha = extract_text(
            client,
            context_text,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        rows.append(
            extraction_log_row(
                row_id=path.stem,
                model=client.model,
                base_url=base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
                prompt_sha=sha,
                run_utc=_utc_now_iso(),
                result=result,
            )
        )
        logger.info("extract {}: parsed_ok={}", path.name, result.parsed_ok)
    return rows


def _run_adjudicate(args: argparse.Namespace, client) -> list[dict]:
    """Run blinded image adjudication over the collected plots; return rows."""
    import hashlib

    inputs = _collect_inputs(args, mode="adjudicate", suffix=".png")
    _guard_output_dir(args.out, inputs)
    base_url = resolve_base_url(args.base_url)
    rows: list[dict] = []
    for path in inputs:
        image_bytes = path.read_bytes()
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        result, sha = adjudicate_image(
            client,
            image_bytes,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        rows.append(
            adjudication_log_row(
                row_id=path.stem,
                model=client.model,
                base_url=base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
                prompt_sha=sha,
                run_utc=_utc_now_iso(),
                image_path=path.name,
                image_sha256=image_sha,
                result=result,
            )
        )
        logger.info(
            "adjudicate {}: parsed_ok={} call={}", path.name, result.parsed_ok, result.call
        )
    return rows


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of input files (*.txt for extract, *.png for adjudicate).",
    )
    sub.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="A single input file (takes precedence over --input-dir).",
    )
    sub.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/llm"),
        help="Output directory for the run log (CSV + parquet).",
    )
    sub.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Resolve the demo input directory for this mode under --data-root "
            "(no credentialing); fails clean if those inputs are not present."
        ),
    )
    sub.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root of the data/ tree, used to resolve --demo inputs (default: data).",
    )
    sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Use an in-process stub client (no server, no network).",
    )
    sub.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Served model id (default: {DEFAULT_MODEL}).",
    )
    sub.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base URL (default: $OMLX_BASE_URL or http://localhost:8000/v1).",
    )
    sub.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Decoding temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    sub.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum new tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    sub.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"Decoding seed (default: GLOBAL_SEED={GLOBAL_SEED}).",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser(
        "extract", help="TEXT mode: de-identified context -> structured JSON."
    )
    _add_common_args(extract_parser)
    adjudicate_parser = subparsers.add_parser(
        "adjudicate", help="IMAGE mode: PI(t) plot -> {call, confidence, rationale}."
    )
    _add_common_args(adjudicate_parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the selected inference mode and write the run log.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on input/output errors).
    """
    args = _parse_args(argv)
    _load_dotenv_if_present()

    logger.info("command={} demo={} dry_run={}", args.command, args.demo, args.dry_run)

    try:
        client = _build_client(args)
    except ImportError as exc:
        logger.error("{}", exc)
        return 2

    try:
        if args.command == "extract":
            rows = _run_extract(args, client)
        else:
            rows = _run_adjudicate(args, client)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("{}", exc)
        return 2

    if not rows:
        logger.warning("no inputs matched; nothing written")
        return 0

    csv_path, parquet_path = _write_run_log(rows, args.out)
    n_ok = sum(1 for r in rows if r["parsed_ok"])
    logger.info(
        "wrote {} rows ({} parsed_ok) -> {} and {}", len(rows), n_ok, csv_path, parquet_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
