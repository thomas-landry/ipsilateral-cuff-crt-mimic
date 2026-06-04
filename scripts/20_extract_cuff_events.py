"""Per-record cuff-event extraction (pipeline step 20).

For each WDB record:

1. Read the master header to get the sampling rate and segment layout.
2. Parse the numerics CSV (gzipped) for rows with a charted noninvasive
   systolic blood pressure (NBP).
3. For each NBP timestamp, slice the PLETH channel and apply
   :func:`cuffcrt.signal.cuff_event_detector.detect_cuff_event`.
4. Write one parquet of derived per-event fields per record.

Outputs hold derived fields only (timing anchors, durations, classification,
quality flags); no raw waveform samples or note text are written. The script is
idempotent: existing per-record parquet files are skipped unless ``--force``.
It never overwrites its inputs.

Examples
--------
Full (credentialed) data::

    uv run python scripts/20_extract_cuff_events.py \\
        --data-root data --output-dir data/interim/events --n-records 20

Open demo data (no credentialing)::

    uv run python scripts/20_extract_cuff_events.py --demo
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import polars as pl
import wfdb
from loguru import logger

from cuffcrt._paths import DataNotAvailableError, require_path, resolve_wdb_root
from cuffcrt.signal.cuff_event_detector import detect_cuff_event

PRE_WINDOW_S = 200.0
POST_WINDOW_S = 200.0


def find_records(wdb_root: Path, n: int) -> list[tuple[str, Path]]:
    """Return up to ``n`` ``(subject_id, record_dir)`` tuples from RECORDS.

    Skips subject directories that contain no record subdirectory or no
    numerics CSV.

    Parameters
    ----------
    wdb_root : pathlib.Path
        Root of the WDB record tree (the directory containing ``RECORDS``).
    n : int
        Maximum number of records to return.

    Returns
    -------
    list[tuple[str, pathlib.Path]]
        ``(subject_id, record_dir)`` pairs.
    """
    records_file = wdb_root / "RECORDS"
    if not records_file.exists():
        raise FileNotFoundError(f"RECORDS file not found at {records_file}")

    found: list[tuple[str, Path]] = []
    for raw_line in records_file.read_text().splitlines():
        if len(found) >= n:
            break
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        subject_dir = wdb_root / line.rstrip("/")
        if not subject_dir.is_dir():
            logger.debug("missing subject dir: {}", subject_dir)
            continue
        record_dirs = [p for p in subject_dir.iterdir() if p.is_dir()]
        if not record_dirs:
            continue
        record_dir = sorted(record_dirs)[0]
        numerics_csv = record_dir / f"{record_dir.name}n.csv.gz"
        master_hea = record_dir / f"{record_dir.name}.hea"
        if not numerics_csv.exists() or not master_hea.exists():
            logger.debug("missing numerics or master header in {}", record_dir)
            continue
        found.append((subject_dir.name, record_dir))
    return found


def _master_freq_field(master_hea: Path) -> list[str]:
    """Return the ``frame_fs[/counter_freq]`` field split on ``/``."""
    with master_hea.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Unexpected master header line: {line!r}")
            return parts[2].split("/")
    raise ValueError(f"No data line in master header {master_hea}")


def parse_master_fs(master_hea: Path) -> float:
    """Pull the waveform frame rate from the first data line of a master header."""
    return float(_master_freq_field(master_hea)[0])


def parse_master_counter_freq(master_hea: Path) -> float:
    """Return the counter (base) frequency from a master header.

    The header frequency field is ``frame_fs/counter_freq`` (for example
    ``62.4725/999.56``). The numerics CSV ``time`` column is expressed in counter
    ticks, not waveform frames, so NBP timestamps must be divided by the counter
    frequency, not the frame rate, to recover seconds. Falls back to the frame
    rate when no counter field is present.
    """
    fields = _master_freq_field(master_hea)
    if len(fields) > 1 and fields[1]:
        return float(fields[1])
    return float(fields[0])


def load_nbp_timestamps(numerics_csv: Path, counter_freq: float) -> np.ndarray:
    """Return charted NBP times in seconds from record start.

    The numerics file is a sparse CSV. Rows where the systolic NBP column
    (``NBPs [mmHg]``) is non-null and physiologically plausible (40 to 260
    mmHg) are extracted.

    Parameters
    ----------
    numerics_csv : pathlib.Path
        Gzipped numerics CSV for the record.
    counter_freq : float
        Counter (base) frequency from the master header. The numerics ``time``
        column is expressed in counter ticks, so dividing by this frequency
        (NOT the waveform frame rate) recovers seconds from record start.

    Returns
    -------
    numpy.ndarray
        NBP timestamps in seconds from record start.
    """
    target_col = "NBPs [mmHg]"
    timestamps: list[float] = []
    with gzip.open(numerics_csv, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
        header = [h.strip().strip('"') for h in header]
        if "time" not in header or target_col not in header:
            raise ValueError(
                f"{numerics_csv} missing 'time' or '{target_col}' columns; got: {header}"
            )
        time_idx = header.index("time")
        nbps_idx = header.index(target_col)
        for raw_line in f:
            cols = raw_line.rstrip("\n").split(",")
            if len(cols) <= max(time_idx, nbps_idx):
                continue
            raw = cols[nbps_idx].strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if not (40.0 <= value <= 260.0):
                continue
            try:
                t_sample = float(cols[time_idx])
            except ValueError:
                continue
            timestamps.append(t_sample / counter_freq)
    return np.asarray(timestamps, dtype=float)


def slice_pleth_window(
    record_basename: Path,
    master_fs: float,
    t_center_s: float,
    pre_s: float = PRE_WINDOW_S,
    post_s: float = POST_WINDOW_S,
) -> tuple[np.ndarray, float] | None:
    """Read a window of PLETH around ``t_center_s`` at the channel-native rate.

    ``smooth_frames=False`` forces wfdb to return PLETH at its native rate
    (typically 125 Hz) rather than averaging adjacent samples down to the
    master frame rate (typically 62.5 Hz). The averaging path destroys the
    cardiac component via cancellation across adjacent waveform peaks.

    Returns
    -------
    tuple[numpy.ndarray, float] or None
        ``(pleth_signal, fs)`` or ``None`` if the channel is unavailable for
        the requested window.
    """
    sampfrom = max(0, int((t_center_s - pre_s) * master_fs))
    sampto = int((t_center_s + post_s) * master_fs)
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
    # wfdb's typeshed unions rdrecord's return as Record | MultiRecord, but at
    # runtime it is always a Record (single-segment read). The e_p_signal /
    # samps_per_frame attribute-access and the float() of the resulting value
    # are stub-only false positives.
    if record.e_p_signal is None or len(record.e_p_signal) == 0:  # pyright: ignore[reportAttributeAccessIssue]
        return None
    pleth = np.asarray(record.e_p_signal[0])  # pyright: ignore[reportAttributeAccessIssue]
    if pleth.size == 0 or not np.isfinite(pleth).any():
        return None
    samps_per_frame = record.samps_per_frame[0] if record.samps_per_frame is not None else 1  # pyright: ignore[reportAttributeAccessIssue]
    fs_native = float(record.fs) * float(samps_per_frame)  # pyright: ignore[reportArgumentType]
    return pleth, fs_native


def extract_record(subject_id: str, record_dir: Path) -> pl.DataFrame:
    """Run per-record extraction and return a derived-fields DataFrame.

    Parameters
    ----------
    subject_id : str
        Subject directory name (for example ``pXXXXXXXX``).
    record_dir : pathlib.Path
        Record directory containing the master header and numerics CSV.

    Returns
    -------
    polars.DataFrame
        One row per charted NBP timestamp, derived fields only.
    """
    record_basename = record_dir / record_dir.name
    master_hea = record_dir / f"{record_dir.name}.hea"
    numerics_csv = record_dir / f"{record_dir.name}n.csv.gz"

    master_fs = parse_master_fs(master_hea)
    counter_freq = parse_master_counter_freq(master_hea)
    nbp_times = load_nbp_timestamps(numerics_csv, counter_freq)
    logger.info(
        "{}/{}: master_fs={:.4f} Hz, counter_freq={:.4f} Hz, n_nbp={}",
        subject_id,
        record_dir.name,
        master_fs,
        counter_freq,
        len(nbp_times),
    )

    rows: list[dict] = []
    for t_nbp in nbp_times:
        windowed = slice_pleth_window(record_basename, master_fs, t_nbp)
        if windowed is None:
            rows.append(_empty_row(subject_id, record_dir.name, t_nbp, reason="no_pleth"))
            continue
        pleth, fs = windowed
        finite_mask = np.isfinite(pleth)
        if finite_mask.mean() < 0.5:
            rows.append(
                _empty_row(subject_id, record_dir.name, t_nbp, reason="pleth_mostly_nan")
            )
            continue
        pleth_clean = np.where(finite_mask, pleth, np.nanmedian(pleth))

        local_t_nbp = PRE_WINDOW_S
        result = detect_cuff_event(pleth_clean, fs, local_t_nbp)
        rows.append(
            dict(
                subject_id=subject_id,
                record_id=record_dir.name,
                nbp_timestamp_s=float(t_nbp),
                is_occlusion_signature=bool(result.is_occlusion_signature),
                stat_mode_candidate=bool(result.stat_mode_candidate),
                recovered=bool(result.recovered),
                ambiguous_multi_dip=bool(result.ambiguous_multi_dip),
                pre_event_pi_mean=float(result.pre_event_pi_mean),
                pre_window_quality=float(result.pre_window_quality),
                pre_window_valid=bool(result.pre_window_valid),
                t_occlusion_start_s=_to_record_time(
                    result.t_occlusion_start_s, local_t_nbp, t_nbp
                ),
                t_deflate_start_s=_to_record_time(result.t_deflate_start_s, local_t_nbp, t_nbp),
                t_nadir_s=_to_record_time(result.t_nadir_s, local_t_nbp, t_nbp),
                t_release_s=_to_record_time(result.t_release_s, local_t_nbp, t_nbp),
                phase2_duration_s=float(result.phase2_duration_s),
                phase3_duration_s=float(result.phase3_duration_s),
                nadir_depth_frac=float(result.nadir_depth_frac),
                recovery_fraction_at_window_end=float(result.recovery_fraction_at_window_end),
                alignment_offset_s=float(result.alignment_offset_s),
                pleth_fs=float(fs),
                pleth_valid_fraction=float(finite_mask.mean()),
                reject_reason=result.reject_reason,
            )
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def _to_record_time(local_value: float, local_t_nbp: float, t_nbp: float) -> float:
    """Convert a time within the local PPG window to seconds from record start."""
    if local_value is None or not np.isfinite(local_value):
        return float("nan")
    return float(local_value - local_t_nbp + t_nbp)


def _empty_row(subject_id: str, record_id: str, t_nbp: float, reason: str) -> dict:
    nan = float("nan")
    return dict(
        subject_id=subject_id,
        record_id=record_id,
        nbp_timestamp_s=float(t_nbp),
        is_occlusion_signature=False,
        stat_mode_candidate=False,
        recovered=False,
        ambiguous_multi_dip=False,
        pre_event_pi_mean=nan,
        pre_window_quality=nan,
        pre_window_valid=False,
        t_occlusion_start_s=nan,
        t_deflate_start_s=nan,
        t_nadir_s=nan,
        t_release_s=nan,
        phase2_duration_s=nan,
        phase3_duration_s=nan,
        nadir_depth_frac=nan,
        recovery_fraction_at_window_end=nan,
        alignment_offset_s=nan,
        pleth_fs=nan,
        pleth_valid_fraction=0.0,
        reject_reason=reason,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the open MIMIC-IV-Demo dataset layout (no credentialing).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root of the data/ tree (default: data).",
    )
    parser.add_argument(
        "--wdb-root",
        type=Path,
        default=None,
        help="Override the WDB record-tree root (otherwise derived from --data-root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/events"),
        help="Directory for per-record event parquets.",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=20,
        help="Maximum number of records to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if an output parquet already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Extract cuff events for each selected record.

    Returns
    -------
    int
        Process exit code (0 on success, 2 when data is unavailable).
    """
    args = _parse_args(argv)
    wdb_root = args.wdb_root or resolve_wdb_root(args.data_root, demo=args.demo)

    logger.info("demo={}", args.demo)
    logger.info("wdb_root={}", wdb_root)
    logger.info("output_dir={}", args.output_dir)

    try:
        require_path(wdb_root, what="WDB record tree")
    except DataNotAvailableError as exc:
        logger.error("{}", exc)
        return 2

    if args.output_dir.resolve() == wdb_root.resolve():
        logger.error("output_dir must differ from the input WDB root; refusing to overwrite input.")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        records = find_records(wdb_root, args.n_records)
    except FileNotFoundError as exc:
        logger.error("{}\n{}", exc, "Confirm the WDB layout in data/README.md.")
        return 2

    logger.info("selected {} records", len(records))
    if len(records) < args.n_records:
        logger.warning(
            "only {} records satisfy the inclusion filter (requested {})",
            len(records),
            args.n_records,
        )

    for subject_id, record_dir in records:
        out_path = args.output_dir / f"events_{subject_id}.parquet"
        if out_path.exists() and not args.force:
            logger.info("skip {} (cached at {})", subject_id, out_path)
            continue
        try:
            df = extract_record(subject_id, record_dir)
        except Exception:
            logger.exception("extract failed for {}", subject_id)
            continue
        if df.is_empty():
            logger.warning("{}: no events written", subject_id)
            continue
        df.write_parquet(out_path)
        n_signature = int(df.get_column("is_occlusion_signature").sum())
        logger.info(
            "{}: wrote {} events ({} occlusion-signature) -> {}",
            subject_id,
            len(df),
            n_signature,
            out_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
