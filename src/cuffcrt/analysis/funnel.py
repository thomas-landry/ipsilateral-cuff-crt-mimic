"""Aggregate per-record cuff-event tables into the feasibility funnel.

This module is the deterministic core behind ``scripts/30_aggregate_funnel.py``.
It takes the concatenated per-event inventory (one row per charted NBP
timestamp, as written by step 20) and produces:

- a stage-by-stage funnel count (candidate cycles down to occlusion-signature
  events),
- a per-patient summary,
- the occlusion-signature yield under two reperfusion-envelope thresholds.

Both yields are derived from ``phase3_duration_s`` rather than from the stored
``is_occlusion_signature`` column, so a single inventory supports both the
primary analysis (15 s run, conservative) and the sensitivity stratum (10 s
run). ``phase3_duration_s`` is the sub-occlusion run length (the event-defining
duration), not the recovery duration.

Recovery handling differs by stratum, matching the pre-registration:

- A **primary** event must have recovered to baseline. It counts at the primary
  threshold when ``reject_reason`` is null (a full, recovered, aligned event)
  and ``phase3_duration_s >= primary_threshold``.
- A **sensitivity** event need not have recovered. It counts at the sensitivity
  threshold when ``reject_reason`` is null, ``stat_mode_short_phase3`` (deep,
  aligned, recovered, but run shorter than the primary floor), or
  ``no_recovery_in_window`` (deep, aligned, run long enough, but never returned
  to baseline) and ``phase3_duration_s >= sensitivity_threshold``.

The asymmetry (``no_recovery_in_window`` counts only at sensitivity) enforces
the rule that an unrecovered event can never be primary regardless of its run
length.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# Reperfusion-envelope thresholds (seconds).
PRIMARY_PHASE3_S = 15.0
SENSITIVITY_PHASE3_S = 10.0

# reject_reason values that indicate a deep, aligned dip was found and a run
# duration was measured (so phase3_duration_s is meaningful). A primary event
# additionally requires recovery, so non-recovered dips are excluded from the
# primary set but kept in the sensitivity set.
_PRIMARY_DIP_REASONS = (None, "stat_mode_short_phase3")
_SENSITIVITY_DIP_REASONS = (None, "stat_mode_short_phase3", "no_recovery_in_window")

FUNNEL_SCHEMA: dict[str, type[pl.DataType]] = {
    "stage": pl.Utf8,
    "events": pl.Int64,
    "patients": pl.Int64,
    "note": pl.Utf8,
}


@dataclass(frozen=True)
class YieldStratum:
    """Occlusion-signature yield under one reperfusion-envelope threshold.

    Attributes
    ----------
    phase3_min_s : float
        The reperfusion-envelope threshold applied (seconds).
    n_events : int
        Number of qualifying occlusion-signature events.
    n_patients : int
        Number of distinct subjects with at least one qualifying event.
    pct_of_candidates : float
        Events as a percentage of all candidate cuff cycles.
    subjects : list[str]
        Sorted subject ids contributing at least one qualifying event.
    """

    phase3_min_s: float
    n_events: int
    n_patients: int
    pct_of_candidates: float
    subjects: list[str]


@dataclass(frozen=True)
class FunnelResult:
    """Full feasibility-funnel result.

    Attributes
    ----------
    funnel : polars.DataFrame
        Stage-by-stage counts (see :data:`FUNNEL_SCHEMA`).
    per_patient : polars.DataFrame
        One row per subject with QC and yield counts.
    primary : YieldStratum
        Occlusion-signature yield at the primary (15 s) threshold.
    sensitivity : YieldStratum
        Occlusion-signature yield at the sensitivity (10 s) threshold.
    n_candidates : int
        Total candidate cuff cycles (rows in the inventory).
    n_records : int
        Distinct record ids in the inventory.
    """

    funnel: pl.DataFrame
    per_patient: pl.DataFrame
    primary: YieldStratum
    sensitivity: YieldStratum
    n_candidates: int
    n_records: int


def _occlusion_signature_mask(
    df: pl.DataFrame,
    phase3_min_s: float,
    dip_reasons: tuple[str | None, ...],
) -> pl.Series:
    """Boolean mask: rows that carry the occlusion signature at ``phase3_min_s``.

    A qualifying row had a deep, aligned dip whose run length was measured: its
    ``reject_reason`` is in ``dip_reasons`` and its ``phase3_duration_s`` meets
    the threshold. ``reject_reason`` may be null, which ``Series.is_in`` does not
    match, so the null case is handled explicitly. The primary stratum passes
    the recovery-requiring reason set; the sensitivity stratum passes the
    relaxed set that also admits non-recovered dips.
    """
    reason = df.get_column("reject_reason")
    non_null_reasons = [r for r in dip_reasons if r is not None]
    dip_found = reason.is_in(non_null_reasons)
    if None in dip_reasons:
        dip_found = reason.is_null() | dip_found
    phase3 = df.get_column("phase3_duration_s")
    return (
        dip_found
        & phase3.is_not_null()
        & phase3.is_not_nan()
        & (phase3 >= phase3_min_s)
    )


def _yield_stratum(
    df: pl.DataFrame,
    phase3_min_s: float,
    n_candidates: int,
    dip_reasons: tuple[str | None, ...],
) -> YieldStratum:
    """Compute the occlusion-signature yield at one threshold."""
    qualifying = df.filter(_occlusion_signature_mask(df, phase3_min_s, dip_reasons))
    subjects = sorted(qualifying.get_column("subject_id").unique().to_list())
    n_events = qualifying.height
    pct = (100.0 * n_events / n_candidates) if n_candidates > 0 else float("nan")
    return YieldStratum(
        phase3_min_s=phase3_min_s,
        n_events=n_events,
        n_patients=len(subjects),
        pct_of_candidates=pct,
        subjects=subjects,
    )


def _count_reason(df: pl.DataFrame, reason: str) -> int:
    return int(df.filter(pl.col("reject_reason") == reason).height)


def aggregate_funnel(
    inventory: pl.DataFrame,
    *,
    primary_phase3_s: float = PRIMARY_PHASE3_S,
    sensitivity_phase3_s: float = SENSITIVITY_PHASE3_S,
) -> FunnelResult:
    """Build the feasibility funnel from a concatenated event inventory.

    Parameters
    ----------
    inventory : polars.DataFrame
        Concatenated per-event rows (the union of all ``events_*.parquet``).
        Must contain at least ``subject_id``, ``record_id``,
        ``phase3_duration_s``, ``pre_window_valid``, and ``reject_reason``.
    primary_phase3_s : float
        Primary reperfusion-envelope threshold (default 15 s).
    sensitivity_phase3_s : float
        Sensitivity reperfusion-envelope threshold (default 10 s).

    Returns
    -------
    FunnelResult
        Funnel counts, per-patient summary, and both yield strata.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {
        "subject_id",
        "record_id",
        "phase3_duration_s",
        "pre_window_valid",
        "reject_reason",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"inventory missing required columns: {missing}")

    n_candidates = inventory.height
    n_records = int(inventory.get_column("record_id").n_unique())
    n_subjects = int(inventory.get_column("subject_id").n_unique())

    n_no_pleth = _count_reason(inventory, "no_pleth")
    n_pleth_nan = _count_reason(inventory, "pleth_mostly_nan")
    n_evaluable = n_candidates - n_no_pleth - n_pleth_nan

    qc_pass = inventory.filter(pl.col("pre_window_valid"))
    n_qc_pass = qc_pass.height
    n_qc_patients = int(qc_pass.get_column("subject_id").n_unique())

    n_no_phase2 = _count_reason(inventory, "no_phase2")
    # Alignment reject; keep the legacy reason name as a fallback for old inventories.
    n_misaligned = _count_reason(inventory, "no_aligned_occlusion") + _count_reason(
        inventory, "phase2_misaligned"
    )
    n_stat_mode = _count_reason(inventory, "stat_mode_short_phase3")
    n_no_recovery = _count_reason(inventory, "no_recovery_in_window")

    primary = _yield_stratum(
        inventory, primary_phase3_s, n_candidates, _PRIMARY_DIP_REASONS
    )
    sensitivity = _yield_stratum(
        inventory, sensitivity_phase3_s, n_candidates, _SENSITIVITY_DIP_REASONS
    )

    funnel = pl.DataFrame(
        [
            {
                "stage": "candidate_cuff_cycles",
                "events": n_candidates,
                "patients": n_subjects,
                "note": "charted NBP timestamps across all records",
            },
            {
                "stage": "excluded_no_pleth",
                "events": n_no_pleth,
                "patients": 0,
                "note": "no co-recorded PPG window available",
            },
            {
                "stage": "excluded_pleth_mostly_nan",
                "events": n_pleth_nan,
                "patients": 0,
                "note": "PPG window more than half missing",
            },
            {
                "stage": "evaluable_with_pleth",
                "events": n_evaluable,
                "patients": 0,
                "note": "candidates - no_pleth - pleth_mostly_nan",
            },
            {
                "stage": "qc_pass_pre_window",
                "events": n_qc_pass,
                "patients": n_qc_patients,
                "note": "stable, plausible pre-cuff PI window",
            },
            {
                "stage": "rejected_no_phase2",
                "events": n_no_phase2,
                "patients": 0,
                "note": "no qualifying deep dip in the search window",
            },
            {
                "stage": "rejected_misaligned",
                "events": n_misaligned,
                "patients": 0,
                "note": "deep dip present but its nadir not aligned with the NBP timestamp",
            },
            {
                "stage": "stat_mode_short_envelope",
                "events": n_stat_mode,
                "patients": 0,
                "note": "deep aligned dip recovered but sub-occlusion run < primary threshold",
            },
            {
                "stage": "rejected_no_recovery",
                "events": n_no_recovery,
                "patients": 0,
                "note": "deep aligned dip that never returned to baseline (sensitivity only)",
            },
            {
                "stage": f"occlusion_signature_primary_{primary_phase3_s:g}s",
                "events": primary.n_events,
                "patients": primary.n_patients,
                "note": f"reperfusion envelope >= {primary_phase3_s:g} s (primary)",
            },
            {
                "stage": f"occlusion_signature_sensitivity_{sensitivity_phase3_s:g}s",
                "events": sensitivity.n_events,
                "patients": sensitivity.n_patients,
                "note": f"reperfusion envelope >= {sensitivity_phase3_s:g} s (sensitivity)",
            },
        ],
        schema=FUNNEL_SCHEMA,
    )

    per_patient = _per_patient_summary(inventory, primary_phase3_s, sensitivity_phase3_s)

    return FunnelResult(
        funnel=funnel,
        per_patient=per_patient,
        primary=primary,
        sensitivity=sensitivity,
        n_candidates=n_candidates,
        n_records=n_records,
    )


def _per_patient_summary(
    inventory: pl.DataFrame,
    primary_phase3_s: float,
    sensitivity_phase3_s: float,
) -> pl.DataFrame:
    """Per-subject QC and yield counts."""
    df = inventory.with_columns(
        _occlusion_signature_mask(
            inventory, primary_phase3_s, _PRIMARY_DIP_REASONS
        ).alias("_occ_primary"),
        _occlusion_signature_mask(
            inventory, sensitivity_phase3_s, _SENSITIVITY_DIP_REASONS
        ).alias("_occ_sensitivity"),
        (pl.col("reject_reason") == "no_pleth").alias("_no_pleth"),
        (pl.col("reject_reason") == "pleth_mostly_nan").alias("_pleth_nan"),
        (pl.col("reject_reason") == "stat_mode_short_phase3").alias("_stat_mode"),
    )
    summary = (
        df.group_by("subject_id")
        .agg(
            pl.len().alias("n_total"),
            (pl.len() - pl.col("_no_pleth").sum() - pl.col("_pleth_nan").sum()).alias(
                "n_evaluable"
            ),
            pl.col("_no_pleth").sum().alias("n_no_pleth"),
            pl.col("pre_window_valid").sum().alias("n_qc_pass"),
            pl.col("_stat_mode").sum().alias("n_stat_mode"),
            pl.col("_occ_primary").sum().alias("n_occlusion_signature_primary"),
            pl.col("_occ_sensitivity").sum().alias("n_occlusion_signature_sensitivity"),
        )
        .with_columns(
            (pl.col("n_occlusion_signature_primary") >= 1).alias(
                "has_occlusion_signature_primary"
            ),
            (pl.col("n_occlusion_signature_sensitivity") >= 1).alias(
                "has_occlusion_signature_sensitivity"
            ),
        )
        .sort("subject_id")
    )
    return summary
