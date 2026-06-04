"""Detector sensitivity sweep over the locked parameters (step 31).

Holds all detector parameters at their pre-registered defaults and varies one
at a time, tabulating the primary and sensitivity event and patient counts for
each setting. The 1 Hz perfusion-index trace for each charted NBP window is
computed once and cached, then re-scored across every parameter value via
:func:`cuffcrt.signal.cuff_event_detector.detect_cuff_event_on_pi`. This keeps
the sweep fast and fully deterministic: no random state, no raw waveforms held
in memory beyond the single window being reduced to PI.

The sweep never writes into the canonical extraction or funnel paths; its
output goes to a separate directory. It does not promote anything to canonical
results/.

Examples
--------
::

    uv run python scripts/31_sensitivity_sweep.py \\
        --wdb-root /path/to/mimic-iv-wdb/0.1.0 \\
        --out results/sensitivity --n-records 19
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from cuffcrt.signal.cuff_event_detector import (
    ALIGN_HI_S,
    ALIGN_LO_S,
    NADIR_DEPTH,
    OCCLUSION_FRACTION,
    PRIMARY_MIN_S,
    RECOVERY_FRACTION,
    RECOVERY_HOLD_S,
    SENSITIVITY_MIN_S,
    compute_pi_1hz,
    detect_cuff_event_on_pi,
)

# Reuse the extractor's record discovery and window slicing. Its filename starts
# with a digit, so it cannot be imported normally; load it by path.
_EXTRACT_PATH = Path(__file__).with_name("20_extract_cuff_events.py")
_spec = importlib.util.spec_from_file_location("_extract20", _EXTRACT_PATH)
_extract = importlib.util.module_from_spec(_spec)
sys.modules["_extract20"] = _extract
_spec.loader.exec_module(_extract)  # type: ignore[union-attr]

PRE_WINDOW_S = _extract.PRE_WINDOW_S


# Sweep grids (the pre-registered sensitivity analyses). The locked default in
# each grid is the headline value.
NADIR_DEPTH_GRID = [0.10, 0.15, 0.20, 0.25]
RECOVERY_FRACTION_GRID = [0.70, 0.80, 0.85, 0.90, 0.95]
ALIGNMENT_GRID: list[tuple[str, float, float]] = [
    ("+/-30", -30.0, 30.0),
    ("+/-45", -45.0, 45.0),
    ("+/-60", -60.0, 60.0),
    ("[-50,+30]", -50.0, 30.0),
]
PRIMARY_MIN_GRID = [10.0, 15.0, 20.0]


def _cache_pi_windows(
    records: list[tuple[str, Path]],
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """Compute and cache the 1 Hz PI trace for every charted NBP window.

    Returns one tuple ``(subject_id, t_pi, pi, nbp_local_s)`` per evaluable
    window. Raw waveforms are released immediately after the PI reduction, so
    only the small 1 Hz arrays persist.
    """
    cached: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for subject_id, record_dir in records:
        record_basename = record_dir / record_dir.name
        master_hea = record_dir / f"{record_dir.name}.hea"
        numerics_csv = record_dir / f"{record_dir.name}n.csv.gz"
        master_fs = _extract.parse_master_fs(master_hea)
        counter_freq = _extract.parse_master_counter_freq(master_hea)
        nbp_times = _extract.load_nbp_timestamps(numerics_csv, counter_freq)
        logger.info("{}: caching PI for {} NBP windows", subject_id, len(nbp_times))
        for t_nbp in nbp_times:
            windowed = _extract.slice_pleth_window(record_basename, master_fs, t_nbp)
            if windowed is None:
                continue
            pleth, fs = windowed
            finite = np.isfinite(pleth)
            if finite.mean() < 0.5:
                continue
            pleth_clean = np.where(finite, pleth, np.nanmedian(pleth))
            t_pi, pi = compute_pi_1hz(pleth_clean, fs)
            if pi.size == 0:
                continue
            cached.append((subject_id, t_pi, pi, PRE_WINDOW_S))
    logger.info("cached PI for {} evaluable windows", len(cached))
    return cached


def _score(
    cached: list[tuple[str, np.ndarray, np.ndarray, float]],
    *,
    nadir_depth: float,
    occlusion_fraction: float,
    recovery_fraction: float,
    recovery_hold_s: float,
    align_lo: float,
    align_hi: float,
    primary_min_s: float,
    sensitivity_min_s: float,
) -> dict[str, int]:
    """Score every cached window under one parameter set; return event/patient counts."""
    primary_subjects: set[str] = set()
    sensitivity_subjects: set[str] = set()
    n_primary = 0
    n_sensitivity = 0
    for subject_id, t_pi, pi, nbp_local in cached:
        r = detect_cuff_event_on_pi(
            t_pi,
            pi,
            nbp_local,
            nadir_depth=nadir_depth,
            occlusion_fraction=occlusion_fraction,
            recovery_fraction=recovery_fraction,
            recovery_hold_s=recovery_hold_s,
            align_lo=align_lo,
            align_hi=align_hi,
            primary_min_s=primary_min_s,
            sensitivity_min_s=sensitivity_min_s,
        )
        # Primary: a recovered, aligned, deep event with run >= primary_min_s.
        if r.reject_reason is None and np.isfinite(r.phase3_duration_s):
            n_primary += 1
            primary_subjects.add(subject_id)
        # Sensitivity: deep, aligned dip (recovered or not) with run >= sensitivity_min_s.
        sensitivity_reasons = (None, "stat_mode_short_phase3", "no_recovery_in_window")
        if (
            r.reject_reason in sensitivity_reasons
            and np.isfinite(r.phase3_duration_s)
            and r.phase3_duration_s >= sensitivity_min_s
        ):
            n_sensitivity += 1
            sensitivity_subjects.add(subject_id)
    return {
        "n_primary_events": n_primary,
        "n_primary_patients": len(primary_subjects),
        "n_sensitivity_events": n_sensitivity,
        "n_sensitivity_patients": len(sensitivity_subjects),
    }


def run_sweep(cached: list[tuple[str, np.ndarray, np.ndarray, float]]) -> pl.DataFrame:
    """Vary one parameter at a time around the locked defaults; tabulate counts."""
    defaults = dict(
        nadir_depth=NADIR_DEPTH,
        occlusion_fraction=OCCLUSION_FRACTION,
        recovery_fraction=RECOVERY_FRACTION,
        recovery_hold_s=RECOVERY_HOLD_S,
        align_lo=ALIGN_LO_S,
        align_hi=ALIGN_HI_S,
        primary_min_s=PRIMARY_MIN_S,
        sensitivity_min_s=SENSITIVITY_MIN_S,
    )
    rows: list[dict] = []

    for value in NADIR_DEPTH_GRID:
        params = {**defaults, "nadir_depth": value}
        counts = _score(cached, **params)
        rows.append({"parameter": "nadir_depth", "value": f"{value:g}", **counts})
    for value in RECOVERY_FRACTION_GRID:
        params = {**defaults, "recovery_fraction": value}
        counts = _score(cached, **params)
        rows.append({"parameter": "recovery_fraction", "value": f"{value:g}", **counts})
    for label, lo, hi in ALIGNMENT_GRID:
        params = {**defaults, "align_lo": lo, "align_hi": hi}
        counts = _score(cached, **params)
        rows.append({"parameter": "alignment", "value": label, **counts})
    for value in PRIMARY_MIN_GRID:
        params = {**defaults, "primary_min_s": value}
        counts = _score(cached, **params)
        rows.append({"parameter": "primary_min_s", "value": f"{value:g}", **counts})

    return pl.DataFrame(rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--wdb-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/sensitivity"))
    p.add_argument("--n-records", type=int, default=19)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the sensitivity sweep and write the table.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on missing input).
    """
    args = _parse_args(argv)
    if not args.wdb_root.exists():
        logger.error("WDB root not found: {}", args.wdb_root)
        return 2
    records = _extract.find_records(args.wdb_root, args.n_records)
    logger.info("sweeping over {} records", len(records))
    cached = _cache_pi_windows(records)
    if not cached:
        logger.error("no evaluable PI windows; nothing to sweep")
        return 2
    table = run_sweep(cached)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "sensitivity_sweep.csv"
    table.write_csv(out_path)
    logger.info("wrote {}", out_path)
    with pl.Config(tbl_rows=50):
        logger.info("\n{}", table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
