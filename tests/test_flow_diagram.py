"""Tests for ``scripts/54_flow_diagram.py``.

The flow-diagram builder reads three shipped artifacts (the event inventory, the
gallery manifest, and the blinded reader form), derives every box count, and
reconciles them before drawing. These tests confirm:

* The derived counts reconcile (the funnel adds up, coverage and the
  not-sampled remainder partition the evaluable pool, the gallery strata sum to
  568, and the reader calls sum to the gallery).
* The reconciliation guard raises rather than drawing a wrong funnel when a
  count is perturbed.
* The figure builds and carries no on-canvas title (caption-only convention).

The synthetic fixtures mirror the real artifacts' structure. A real-data smoke
check runs only when the shipped inventory/manifest/reader files are present, so
the suite stays green in environments without them.

The script filename starts with a digit, so it is loaded by path through
``importlib.util`` the same way the other digit-prefixed scripts are tested.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import polars as pl
import pytest

# Load scripts/54_flow_diagram.py by path (digit-prefixed filename).
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "54_flow_diagram.py"
_spec = importlib.util.spec_from_file_location("_flow54", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_flow = importlib.util.module_from_spec(_spec)
sys.modules["_flow54"] = _flow
_spec.loader.exec_module(_flow)

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "data/interim/event_inventory.csv"
MANIFEST = REPO / "results/gallery/gallery_manifest.csv"
READER = REPO / "results/gallery/reader_form_blinded.csv"


def _inventory_fixture() -> pl.DataFrame:
    """A small inventory whose reject-reason mix mirrors the real one's shape.

    Counts are chosen so the reconciliation identities are easy to check by
    hand:

    * 9 no_pleth + 1 pleth_mostly_nan = 10 non-evaluable.
    * 4 detector-positive (null reject_reason).
    * near-miss pool = 5 stat_mode_short_phase3 + 1 no_recovery_in_window = 6.
    * not-sampled remainder = 2 no_aligned_occlusion + 1 pre_pi_implausible = 3.
    * the rest (no_phase2, pre_window_unstable) form the negative pool.

    ``pre_window_valid`` is set on a subset to give a QC-pass count distinct
    from the evaluable count, matching the real artifact where QC-pass is a
    parallel flag rather than a strict serial subtraction.
    """
    rows: list[dict] = []

    def add(n: int, *, reason: str | None, valid: bool, subj_prefix: str) -> None:
        for i in range(n):
            rows.append(
                {
                    "subject_id": f"{subj_prefix}{i % 3}",
                    "record_id": f"{subj_prefix}{i % 3}",
                    "nbp_timestamp_s": 100.0 + i,
                    "pre_window_valid": valid,
                    "reject_reason": reason,
                }
            )

    add(9, reason="no_pleth", valid=False, subj_prefix="a")
    add(1, reason="pleth_mostly_nan", valid=False, subj_prefix="a")
    add(4, reason=None, valid=True, subj_prefix="p")  # detector-positive
    add(5, reason="stat_mode_short_phase3", valid=True, subj_prefix="n")
    add(1, reason="no_recovery_in_window", valid=True, subj_prefix="n")
    add(2, reason="no_aligned_occlusion", valid=False, subj_prefix="x")
    add(1, reason="pre_pi_implausible", valid=False, subj_prefix="x")
    # Negative pool: no_phase2 + pre_window_unstable.
    add(12, reason="no_phase2", valid=True, subj_prefix="g")
    add(6, reason="pre_window_unstable", valid=False, subj_prefix="g")
    return pl.DataFrame(rows)


def _manifest_fixture() -> pl.DataFrame:
    """A gallery manifest: census of the 4 positives, 3 near-miss, 5 negative."""
    rows: list[dict] = []

    def add(prefix: str, stratum: str, n: int) -> None:
        for i in range(n):
            rows.append(
                {"card_id": f"{prefix}-{i}", "stratum": stratum, "subject_id": f"{prefix}{i % 3}"}
            )

    add("A", "detector_positive", 4)
    add("B", "detector_rejected_near_miss", 3)
    add("C", "detector_negative_random", 5)
    return pl.DataFrame(rows)


def _reader_fixture() -> pl.DataFrame:
    """A reader form whose calls sum to the 12 gallery cards (3/7/2)."""
    calls = (
        [_flow._READER_PRESENT] * 3
        + [_flow._READER_ABSENT] * 7
        + [_flow._READER_INDETERMINATE] * 2
    )
    return pl.DataFrame({"card_id": [f"R-{i}" for i in range(12)], "call": calls})


def _counts_from_fixtures(tmp_path: Path):
    """Build reconciled counts from the fixtures written under ``tmp_path``."""
    inv = tmp_path / "inv.csv"
    man = tmp_path / "man.csv"
    rdr = tmp_path / "rdr.csv"
    _inventory_fixture().write_csv(inv)
    _manifest_fixture().write_csv(man)
    _reader_fixture().write_csv(rdr)
    return _flow.compute_counts(inv, man, rdr)


def test_fixture_counts_reconcile(tmp_path: Path) -> None:
    """The hand-checkable fixture reconciles end to end."""
    c = _counts_from_fixtures(tmp_path)

    assert c.n_candidates == 41
    assert c.n_no_pleth == 9
    assert c.n_pleth_nan == 1
    # evaluable = candidates - no_pleth - pleth_mostly_nan
    assert c.n_evaluable == c.n_candidates - c.n_no_pleth - c.n_pleth_nan == 31
    assert c.n_detector_positive == 4
    assert c.n_near_miss_pool == 6
    assert c.n_uncovered_no_align == 2
    assert c.n_uncovered_implausible == 1
    assert c.n_uncovered == c.n_uncovered_no_align + c.n_uncovered_implausible == 3
    # negative pool = evaluable - positive - near_miss_pool - uncovered
    assert c.n_negative_pool == 31 - 4 - 6 - 3 == 18

    # The evaluable pool partitions into the three pools plus the remainder.
    assert (
        c.n_evaluable
        == c.n_detector_positive + c.n_near_miss_pool + c.n_negative_pool + c.n_uncovered
    )
    # Coverage is the three sampling pools; coverage + remainder == evaluable.
    assert c.coverage == c.n_detector_positive + c.n_near_miss_pool + c.n_negative_pool
    assert c.coverage + c.n_uncovered == c.n_evaluable

    # Gallery strata sum to the gallery; reader calls sum to the gallery.
    assert c.n_gallery == c.n_sampled_positive + c.n_sampled_near_miss + c.n_sampled_negative == 12
    assert (
        c.n_gallery
        == c.n_reader_present + c.n_reader_absent + c.n_reader_indeterminate
    )
    # Detector-positive is a census: sampled equals pool.
    assert c.n_sampled_positive == c.n_detector_positive

    # Weights are pool / sampled.
    assert c.weight_positive == pytest.approx(4 / 4)
    assert c.weight_near_miss == pytest.approx(6 / 3)
    assert c.weight_negative == pytest.approx(18 / 5)


def test_reconcile_guard_rejects_bad_counts() -> None:
    """A perturbed count trips the guard rather than drawing a wrong funnel."""
    good = _flow.FlowCounts(
        n_candidates=100,
        n_records=2,
        n_subjects=8,
        n_no_pleth=10,
        n_pleth_nan=0,
        n_evaluable=90,
        n_qc_pass=60,
        n_detector_positive=4,
        pct_detector_positive=6.67,
        n_near_miss_pool=6,
        n_negative_pool=77,
        n_uncovered=3,
        n_uncovered_no_align=2,
        n_uncovered_implausible=1,
        n_sampled_positive=4,
        n_sampled_near_miss=3,
        n_sampled_negative=5,
        n_gallery=12,
        n_gallery_subjects=6,
        coverage=87,
        coverage_pct=96.7,
        n_reader_present=3,
        n_reader_absent=7,
        n_reader_indeterminate=2,
        weight_positive=1.0,
        weight_near_miss=2.0,
        weight_negative=15.4,
    )
    # The good record reconciles.
    _flow._reconcile(good)

    # Break the evaluable identity: it should now raise.
    import dataclasses

    bad = dataclasses.replace(good, n_evaluable=91)
    with pytest.raises(ValueError, match="reconciliation"):
        _flow._reconcile(bad)

    # Break the gallery sum: it should also raise.
    bad2 = dataclasses.replace(good, n_sampled_negative=6)
    with pytest.raises(ValueError, match="reconciliation"):
        _flow._reconcile(bad2)


def test_figure_builds_without_title(tmp_path: Path) -> None:
    """The figure renders and carries no on-canvas title (caption-only rule)."""
    c = _counts_from_fixtures(tmp_path)
    fig = _flow.build_figure(c)
    assert fig._suptitle is None
    # No Axes title text either.
    for ax in fig.axes:
        assert ax.get_title() == ""


@pytest.mark.skipif(
    not (INVENTORY.exists() and MANIFEST.exists() and READER.exists()),
    reason="shipped inventory/manifest/reader artifacts not present",
)
def test_real_artifacts_match_canonical_funnel() -> None:
    """When the shipped artifacts are present, the canonical numbers reconcile.

    This pins the headline counts the manuscript reports: 9,224 candidates;
    8,909 evaluable; 6,236 QC-pass; 268 detector-positive (4.30%); 568 gallery
    cards (102/387/79); 214 not-sampled remainder; 97.6% coverage.
    """
    c = _flow.compute_counts(INVENTORY, MANIFEST, READER)
    assert c.n_candidates == 9224
    assert c.n_records == 19
    assert c.n_gallery_subjects == 16
    assert c.n_no_pleth == 310
    assert c.n_pleth_nan == 5
    assert c.n_evaluable == 8909
    assert c.n_qc_pass == 6236
    assert c.n_detector_positive == 268
    assert round(c.pct_detector_positive, 2) == 4.30
    assert c.n_near_miss_pool == 320
    assert c.n_negative_pool == 8107
    assert c.n_uncovered == 214
    assert c.n_uncovered_no_align == 136
    assert c.n_uncovered_implausible == 78
    assert c.n_sampled_positive == 268
    assert c.n_sampled_near_miss == 200
    assert c.n_sampled_negative == 100
    assert c.n_gallery == 568
    assert c.coverage == 8695
    assert round(c.coverage_pct, 1) == 97.6
    assert c.n_reader_present == 102
    assert c.n_reader_absent == 387
    assert c.n_reader_indeterminate == 79
    assert c.weight_positive == pytest.approx(1.0)
    assert c.weight_near_miss == pytest.approx(1.6)
    assert c.weight_negative == pytest.approx(81.07)
