"""Tests for the feasibility-funnel aggregation on synthetic fixture inventories.

These exercise :func:`cuffcrt.analysis.funnel.aggregate_funnel` without any real
data. The fixture is constructed to mirror the structure of the real event
inventory: a bulk of ``no_pleth`` rows, some pre-window rejects, ``no_phase2``
rejects, short-envelope stat-mode candidates, and a handful of dip rows whose
reperfusion envelopes span the 10 s and 15 s thresholds. The per-patient
summary columns use the canonical ``has_occlusion_signature_*`` names.
"""

import math

import polars as pl

from cuffcrt.analysis.funnel import aggregate_funnel

NAN = float("nan")


def _row(
    subject_id: str,
    *,
    reject_reason: str | None,
    phase3: float = NAN,
    pre_window_valid: bool = False,
    record_id: str = "rec",
):
    return {
        "subject_id": subject_id,
        "record_id": f"{subject_id}_{record_id}",
        "nbp_timestamp_s": 100.0,
        "phase3_duration_s": phase3,
        "pre_window_valid": pre_window_valid,
        "reject_reason": reject_reason,
    }


def _fixture_inventory() -> pl.DataFrame:
    rows = []
    # 10 candidates with no co-recorded PPG.
    rows += [_row("p1", reject_reason="no_pleth") for _ in range(10)]
    # 1 mostly-NaN PPG.
    rows.append(_row("p1", reject_reason="pleth_mostly_nan"))
    # 2 pre-window rejects (evaluable, QC fail).
    rows += [_row("p2", reject_reason="pre_window_unstable") for _ in range(2)]
    # 3 no_phase2 (QC pass, no dip).
    rows += [_row("p2", reject_reason="no_phase2", pre_window_valid=True) for _ in range(3)]
    # 1 misaligned dip (QC pass).
    rows.append(_row("p3", reject_reason="phase2_misaligned", pre_window_valid=True))
    # 2 stat-mode short envelopes (QC pass, dip found, phase3 < 10).
    rows.append(
        _row("p3", reject_reason="stat_mode_short_phase3", phase3=4.0, pre_window_valid=True)
    )
    rows.append(
        _row("p4", reject_reason="stat_mode_short_phase3", phase3=5.0, pre_window_valid=True)
    )
    # Dip rows (reject_reason None) at varying envelope lengths:
    #   p3: 17 s -> qualifies at both 10 and 15
    #   p4: 18 s -> qualifies at both 10 and 15
    #   p5: 12 s -> qualifies at 10 only
    #   p6: 14 s, 10 s -> qualify at 10 only
    rows.append(_row("p3", reject_reason=None, phase3=17.0, pre_window_valid=True))
    rows.append(_row("p4", reject_reason=None, phase3=18.0, pre_window_valid=True))
    rows.append(_row("p5", reject_reason=None, phase3=12.0, pre_window_valid=True))
    rows.append(_row("p6", reject_reason=None, phase3=14.0, pre_window_valid=True))
    rows.append(_row("p6", reject_reason=None, phase3=10.0, pre_window_valid=True))
    return pl.DataFrame(rows)


def test_funnel_primary_and_sensitivity_yields():
    """Primary (15 s) keeps only >=15 s envelopes; sensitivity (10 s) keeps >=10 s."""
    inv = _fixture_inventory()
    result = aggregate_funnel(inv)

    # 15 s: p3 (17) and p4 (18) -> 2 events / 2 patients.
    assert result.primary.n_events == 2
    assert result.primary.n_patients == 2
    assert result.primary.subjects == ["p3", "p4"]

    # 10 s: p3 (17), p4 (18), p5 (12), p6 (14 and 10) -> 5 events / 4 patients.
    assert result.sensitivity.n_events == 5
    assert result.sensitivity.n_patients == 4
    assert result.sensitivity.subjects == ["p3", "p4", "p5", "p6"]


def test_funnel_renamed_stage_rows_present():
    """The canonical funnel exposes occlusion-signature stage row names."""
    inv = _fixture_inventory()
    result = aggregate_funnel(inv)
    stages = set(result.funnel.get_column("stage").to_list())
    assert "occlusion_signature_primary_15s" in stages
    assert "occlusion_signature_sensitivity_10s" in stages


def test_funnel_candidate_and_exclusion_counts():
    """Funnel stages reflect the constructed fixture."""
    inv = _fixture_inventory()
    result = aggregate_funnel(inv)
    funnel = {r["stage"]: r["events"] for r in result.funnel.to_dicts()}

    assert result.n_candidates == inv.height
    assert funnel["candidate_cuff_cycles"] == inv.height
    assert funnel["excluded_no_pleth"] == 10
    assert funnel["excluded_pleth_mostly_nan"] == 1
    assert funnel["evaluable_with_pleth"] == inv.height - 11
    assert funnel["rejected_no_phase2"] == 3
    assert funnel["rejected_misaligned"] == 1
    assert funnel["stat_mode_short_envelope"] == 2


def test_funnel_percent_of_candidates():
    """The primary yield percentage is events / candidates * 100."""
    inv = _fixture_inventory()
    result = aggregate_funnel(inv)
    expected = 100.0 * result.primary.n_events / result.n_candidates
    assert math.isclose(result.primary.pct_of_candidates, expected, rel_tol=1e-9)


def test_per_patient_summary_yield_columns():
    """Per-patient summary marks only patients with qualifying primary events."""
    inv = _fixture_inventory()
    result = aggregate_funnel(inv)
    pp = {r["subject_id"]: r for r in result.per_patient.to_dicts()}

    assert pp["p3"]["has_occlusion_signature_primary"] is True
    assert pp["p4"]["has_occlusion_signature_primary"] is True
    assert pp["p5"]["has_occlusion_signature_primary"] is False
    assert pp["p6"]["has_occlusion_signature_primary"] is False
    # p5 and p6 do qualify at the sensitivity threshold.
    assert pp["p5"]["has_occlusion_signature_sensitivity"] is True
    assert pp["p6"]["has_occlusion_signature_sensitivity"] is True


def test_funnel_empty_subject_with_only_no_pleth():
    """A patient with only no_pleth rows contributes no occlusion-signature events."""
    inv = pl.DataFrame([_row("pX", reject_reason="no_pleth") for _ in range(5)])
    result = aggregate_funnel(inv)
    assert result.primary.n_events == 0
    assert result.sensitivity.n_events == 0
    assert result.n_candidates == 5
