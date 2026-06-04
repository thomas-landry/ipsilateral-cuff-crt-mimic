Canonicalized 2026-05-22; v2 suffix dropped; detector parameters unchanged.

# Pre-registration: cuff-occlusion detector parameters

Determined before recomputing any feasibility yield, to avoid tuning-on and
reporting-on the same events. Derived from convergent occlusion-reperfusion physiology and from
detector-independent measurement of the 265 v1 events, informed by prior signal-characterization
work in this project. 

## Parameters
1. PI: 1 Hz perfusion index; 5 s rolling-median smoothing before threshold logic.
2. Baseline: **median** PI over the pre-cuff window [t_nbp-120, t_nbp-60] s. Pre-window QC
   (plausibility, relative-SD stability).
3. Occlusion phase: a contiguous run with smoothed **PI < 0.50 x baseline** that contains a
   **nadir (min PI) < 0.20 x baseline**. Onset = start of that run; `t_nadir` = argmin; descent =
   onset to nadir.
4. Reperfusion end (`t_release`): first second at or after `t_nadir` where PI returns to
   **>= 0.85 x baseline and stays >= 0.85 for >= 2 s**. Reperfusion duration = `t_nadir` to
   `t_release`. If 0.85 is not reached within the window, set release = NaN and record
   `recovery_fraction_at_window_end`; the event is flagged non-recovered (cannot be primary).
5. Alignment: the nadir must fall within **[-50, +30] s of the charted BP** (asymmetric; dips lead
   the BP).
6. Dip selection (wrong-dip fix): among qualifying runs (sub-0.50 duration >= 10 s, nadir < 0.20)
   whose nadir lies in the alignment window, select the run with the **deepest nadir**; tie-break by
   proximity to the BP. Set `ambiguous_multi_dip = True` when >= 2 qualifying runs exist.
7. Event classes:
  - **Primary**: qualifying run with sub-0.50 duration **>= 15 s** AND nadir < 0.20 AND nadir
     aligned AND recovered to >= 0.85.
  - **Sensitivity**: sub-0.50 duration **>= 10 s** AND nadir < 0.20 AND nadir aligned (recovery not
     required). Absorbs v1 `stat_mode_candidate`.
8. Determinism: no random state in the detector. `GLOBAL_SEED` only for sampling the adjudication
   gallery and any audit subset.

## Validation commitments (pre-specified)
- Synthetic-truth tests choose/confirm the operating point, not the 265 real events.
- Sensitivity analyses to report: nadir depth {0.10-0.25}, recovery endpoint {0.70-0.95},
  alignment {+/-30/45/60 and [-50,+30]}, duration floors. Headline = estimate + range.
- Primary estimand: proportion of EVALUABLE charted cuff cycles (co-recorded PPG passing pre-cuff
  QC) that are detector-positive; also report per-patient and per-all-charted-cycles, with
  subject-clustered bootstrap CIs (seed 20260426).
- Expected direction (pre-stated): a modest decrease in count vs v1's 265, shifted toward higher
  specificity. No specific number asserted before the run.
