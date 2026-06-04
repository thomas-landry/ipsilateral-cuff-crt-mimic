"""Unit tests for the split-half alignment-window calibration helpers.

Exercises the pure helper functions in ``scripts/32_alignment_split_half.py``
on a small synthetic 6-subject toy inventory. No real MIMIC data are needed.
The script lives in ``scripts/`` and its filename starts with a digit, so it
must be loaded by path rather than imported normally (mirrors how
``scripts/31_sensitivity_sweep.py`` imports ``20_extract_cuff_events.py``).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Load the step-32 script by path.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "32_alignment_split_half.py"
)
_spec = importlib.util.spec_from_file_location("_step32", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_step32 = importlib.util.module_from_spec(_spec)
sys.modules["_step32"] = _step32
_spec.loader.exec_module(_step32)


NAN = float("nan")


def _row(
    subject_id: str,
    *,
    phase3: float,
    nadir_frac: float,
    offset: float,
    pre_window_valid: bool = True,
    recovered: bool = True,
    reject_reason: str | None = None,
    record_id: str = "rec",
) -> dict:
    return {
        "subject_id": subject_id,
        "record_id": f"{subject_id}_{record_id}",
        "phase3_duration_s": phase3,
        "nadir_depth_frac": nadir_frac,
        "alignment_offset_s": offset,
        "pre_window_valid": pre_window_valid,
        "recovered": recovered,
        "reject_reason": reject_reason,
    }


def _toy_inventory() -> pl.DataFrame:
    """Six subjects spanning the recoverable mix of cases.

    Per-subject pattern (each row is one charted NBP cycle):

    - s1: 3 cycles, all primary-eligible (run >= 15, nadir < 0.20), offsets
      span the whole [-50, +30] envelope.
    - s2: 2 cycles, both primary-eligible, offsets near zero.
    - s3: 2 cycles, primary-eligible at offset -45 and +25 (edges).
    - s4: 1 cycle, qualifying-but-short (run 12 s) -> sensitivity only, not
      primary at any window.
    - s5: 2 cycles, primary-eligible at offsets -10 and +5.
    - s6: 1 cycle that failed QC -> excluded from evaluable.
    """
    rows = [
        _row("s1", phase3=20.0, nadir_frac=0.10, offset=-40.0),
        _row("s1", phase3=22.0, nadir_frac=0.08, offset=-5.0),
        _row("s1", phase3=18.0, nadir_frac=0.12, offset=25.0),
        _row("s2", phase3=16.0, nadir_frac=0.09, offset=-2.0),
        _row("s2", phase3=17.0, nadir_frac=0.11, offset=3.0),
        _row("s3", phase3=15.0, nadir_frac=0.10, offset=-45.0),
        _row("s3", phase3=15.0, nadir_frac=0.10, offset=25.0),
        _row(
            "s4",
            phase3=12.0,
            nadir_frac=0.15,
            offset=-1.0,
            recovered=True,
            reject_reason="stat_mode_short_phase3",
        ),
        _row("s5", phase3=20.0, nadir_frac=0.10, offset=-10.0),
        _row("s5", phase3=20.0, nadir_frac=0.10, offset=5.0),
        _row(
            "s6",
            phase3=NAN,
            nadir_frac=NAN,
            offset=NAN,
            pre_window_valid=False,
            recovered=False,
            reject_reason="pre_window_unstable",
        ),
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# split_subjects
# ---------------------------------------------------------------------------


def test_split_subjects_is_deterministic():
    """Same seed and subject set produces the same split across runs."""
    subjects = [f"s{i}" for i in range(1, 7)]
    cal_a, held_a = _step32.split_subjects(subjects, seed=20260426)
    cal_b, held_b = _step32.split_subjects(subjects, seed=20260426)
    assert cal_a == cal_b
    assert held_a == held_b


def test_split_subjects_partition_is_disjoint_and_complete():
    """Every subject ends up in exactly one half; halves are disjoint."""
    subjects = [f"s{i}" for i in range(1, 7)]
    cal, held = _step32.split_subjects(subjects, seed=20260426)
    assert set(cal).isdisjoint(set(held))
    assert set(cal) | set(held) == set(subjects)


def test_split_subjects_odd_count_extra_goes_to_held_out():
    """Odd subject count: held_out has one more subject than calibration."""
    subjects = [f"s{i}" for i in range(1, 8)]  # 7 subjects
    cal, held = _step32.split_subjects(subjects, seed=20260426)
    assert len(cal) == 3
    assert len(held) == 4


def test_split_subjects_requires_at_least_two_subjects():
    """One subject cannot be split."""
    with pytest.raises(ValueError, match="at least 2"):
        _step32.split_subjects(["only_one"])


def test_split_subjects_deduplicates_input():
    """Duplicate subject ids are collapsed before splitting."""
    cal, held = _step32.split_subjects(["s1", "s1", "s2", "s3"], seed=20260426)
    assert sorted(cal + held) == ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# qualifying_offsets and calibration_summary
# ---------------------------------------------------------------------------


def test_qualifying_offsets_filters_short_and_shallow():
    """Excludes runs below 10 s, shallow nadirs, and non-finite offsets.

    Qualifying means run >= sensitivity_min_s (10 s) and nadir < nadir_depth
    (0.20). The toy has 10 such rows: 3 from s1, 2 from s2, 2 from s3,
    s4's 12 s run (sensitivity-only at the funnel level but qualifying for
    the calibration distribution), and 2 from s5. s6's row is QC-fail with
    NaN offset.
    """
    inv = _toy_inventory()
    offsets = _step32.qualifying_offsets(inv)
    assert offsets.size == 10
    assert np.all(np.isfinite(offsets))


def test_qualifying_offsets_drops_shallow_and_too_short():
    """Below-floor and shallow rows are excluded from the calibration set."""
    rows = [
        # Qualifying.
        _row("a", phase3=15.0, nadir_frac=0.10, offset=0.0),
        # Too short.
        _row("b", phase3=8.0, nadir_frac=0.10, offset=0.0),
        # Too shallow.
        _row("c", phase3=15.0, nadir_frac=0.25, offset=0.0),
        # Non-finite offset.
        _row("d", phase3=15.0, nadir_frac=0.10, offset=NAN),
    ]
    inv = pl.DataFrame(rows)
    offsets = _step32.qualifying_offsets(inv)
    assert offsets.size == 1


def test_calibration_summary_reports_required_keys():
    """Summary exposes median, IQR, and the named percentiles."""
    offsets = np.array([-30.0, -10.0, 0.0, 5.0, 10.0, 20.0])
    s = _step32.calibration_summary(offsets)
    assert s["n_qualifying"] == 6
    assert math.isclose(s["median_s"], np.median(offsets))
    for pct in (2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 97.5):
        assert f"p{pct:g}_s" in s


# ---------------------------------------------------------------------------
# derive_windows
# ---------------------------------------------------------------------------


def test_derive_windows_r95_is_wider_than_r90():
    """R-95 (2.5/97.5) is at least as wide as R-90 (5/95)."""
    rng = np.random.default_rng(20260426)
    offsets = rng.uniform(-50.0, 30.0, size=200)
    r95, r90, prereg = _step32.derive_windows(offsets)
    assert r95.name == "R-95"
    assert r90.name == "R-90"
    assert prereg.name == "prereg"
    width_r95 = r95.upper_s - r95.lower_s
    width_r90 = r90.upper_s - r90.lower_s
    assert width_r95 >= width_r90


def test_derive_windows_prereg_is_locked_constants():
    """The third window equals the pre-registered [-50, +30] s constants."""
    offsets = np.array([-30.0, -10.0, 0.0, 5.0, 10.0, 20.0])
    _, _, prereg = _step32.derive_windows(offsets)
    assert prereg.lower_s == _step32.PREREG_ALIGN_LO_S
    assert prereg.upper_s == _step32.PREREG_ALIGN_HI_S


# ---------------------------------------------------------------------------
# primary_indicator_under_window
# ---------------------------------------------------------------------------


def test_primary_indicator_window_drops_out_of_window_events():
    """A tight window excludes events whose offset lies outside it."""
    inv = _toy_inventory()
    tight = _step32.Window(name="tight", lower_s=-6.0, upper_s=6.0)
    per_cycle = _step32.primary_indicator_under_window(inv, tight)
    # Evaluable QC-pass rows = all rows except s6 (QC-fail) = 10.
    assert per_cycle.height == 10
    # Inside [-6, 6]: s1 row at -5, s2 rows at -2 and 3, s5 row at 5 -> 4 events.
    assert int(per_cycle.get_column("is_primary").sum()) == 4


def test_primary_indicator_full_window_counts_all_primary_eligible():
    """The pre-registered window picks up every primary-eligible cycle."""
    inv = _toy_inventory()
    full = _step32.Window(
        name="prereg",
        lower_s=_step32.PREREG_ALIGN_LO_S,
        upper_s=_step32.PREREG_ALIGN_HI_S,
    )
    per_cycle = _step32.primary_indicator_under_window(inv, full)
    # All 9 deep-and-long rows are inside [-50, +30] and recovered -> 9 events.
    assert int(per_cycle.get_column("is_primary").sum()) == 9


def test_primary_indicator_requires_recovery():
    """A non-recovered qualifying row is not counted as primary."""
    rows = [
        _row(
            "sA",
            phase3=20.0,
            nadir_frac=0.10,
            offset=0.0,
            recovered=False,
            reject_reason="no_recovery_in_window",
        ),
        _row("sA", phase3=20.0, nadir_frac=0.10, offset=0.0),
    ]
    inv = pl.DataFrame(rows)
    full = _step32.Window(
        name="prereg",
        lower_s=_step32.PREREG_ALIGN_LO_S,
        upper_s=_step32.PREREG_ALIGN_HI_S,
    )
    per_cycle = _step32.primary_indicator_under_window(inv, full)
    assert per_cycle.height == 2
    assert int(per_cycle.get_column("is_primary").sum()) == 1


# ---------------------------------------------------------------------------
# held_out_window_result (integration of indicator + bootstrap)
# ---------------------------------------------------------------------------


def test_held_out_window_result_bounds_and_counts():
    """Reports event count, patient count, point estimate, and a bracketing CI."""
    inv = _toy_inventory()
    full = _step32.Window(
        name="prereg",
        lower_s=_step32.PREREG_ALIGN_LO_S,
        upper_s=_step32.PREREG_ALIGN_HI_S,
    )
    r = _step32.held_out_window_result(inv, full, n_resamples=500, seed=20260426)
    assert r.n_evaluable_qc_pass == 10
    assert r.n_events_primary == 9
    # 4 subjects contribute primary events: s1, s2, s3, s5.
    assert r.n_patients_primary == 4
    assert math.isclose(r.primary_rate, 9.0 / 10.0)
    assert r.ci_low <= r.primary_rate <= r.ci_high
