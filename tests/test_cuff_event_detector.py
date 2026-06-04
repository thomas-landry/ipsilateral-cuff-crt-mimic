"""Synthetic-truth validation of the cuff-occlusion detector.

These tests pin the pre-registered operating point
(``findings/preregistration_detector.md``) against signals with known ground
truth from :mod:`cuffcrt.signal.synthetic`. All tests use a deterministic
seeded RNG so results are bit-stable and require no real data or network.

The detector is parameterized: each criterion (nadir depth, occlusion fraction,
recovery fraction and hold, alignment bounds, duration floors) is a function
argument whose default equals the locked value. The detector remains fully
deterministic with no random state.

Scenarios covered (per the build dispatch):
- a deep, recovered, aligned standard event is primary;
- a shallow-nadir dip (does not reach 0.20 baseline) is rejected;
- a short occlusion (sub-0.50 run < 10 s) is rejected;
- a 10 to 15 s occlusion is sensitivity but not primary;
- fast vs slow recovery both reach release when they cross 0.85 sustained;
- a recovery that never reaches 0.85 is non-recovered (release NaN, cannot be
  primary, records recovery_fraction_at_window_end);
- a nadir outside the [-50, +30] s alignment window is rejected;
- a two-dip distractor where the deeper dip must win, with ambiguous_multi_dip
  set;
- a no-event motion-only signal is rejected.
"""

import numpy as np
import pytest

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.signal.cuff_event_detector import detect_cuff_event
from cuffcrt.signal.synthetic import (
    CuffEvent,
    _apply_cuff_envelope,
    add_motion_artifact,
    synthetic_ppg,
)

FS = 125.0


@pytest.fixture
def rng():
    return np.random.default_rng(GLOBAL_SEED)


def test_locked_defaults_match_preregistration():
    """The detector defaults must equal the locked pre-registration parameters."""
    import inspect

    sig = inspect.signature(detect_cuff_event)
    empty = inspect.Parameter.empty
    d = {k: v.default for k, v in sig.parameters.items() if v.default is not empty}
    assert d["nadir_depth"] == pytest.approx(0.20)
    assert d["occlusion_fraction"] == pytest.approx(0.50)
    assert d["recovery_fraction"] == pytest.approx(0.85)
    assert d["recovery_hold_s"] == pytest.approx(2.0)
    assert d["align_lo"] == pytest.approx(-50.0)
    assert d["align_hi"] == pytest.approx(30.0)
    assert d["primary_min_s"] == pytest.approx(15.0)
    assert d["sensitivity_min_s"] == pytest.approx(10.0)


def test_standard_deep_recovered_aligned_event_is_primary(rng):
    """A deep (floor 0.05), well-recovered, aligned cuff cycle is a primary event."""
    cuff = CuffEvent(
        t_inflation_start_s=240.0,
        inflation_duration_s=7.0,
        hold_duration_s=10.0,
        deflate_duration_s=30.0,
        floor=0.05,
    )
    # Charted BP near cycle end; nadir leads it, inside [-50, +30].
    nbp = cuff.t_deflate_start_s + 5.0
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=cuff, rng=rng)
    r = detect_cuff_event(ppg, FS, nbp)

    assert r.is_occlusion_signature, f"expected primary; reject={r.reject_reason}"
    assert r.reject_reason is None
    assert r.nadir_depth_frac < 0.20
    assert r.phase3_duration_s >= 15.0  # sub-0.50 run length is the event duration
    assert np.isfinite(r.t_release_s)
    assert np.isfinite(r.t_nadir_s)
    # alias contract: t_deflate_start_s == t_nadir_s
    assert r.t_deflate_start_s == pytest.approx(r.t_nadir_s)
    assert -50.0 <= r.alignment_offset_s <= 30.0
    assert r.recovered
    assert not r.ambiguous_multi_dip


def test_shallow_nadir_is_rejected(rng):
    """A dip that never reaches 0.20 baseline is not an occlusion."""
    cuff = CuffEvent(
        t_inflation_start_s=240.0,
        inflation_duration_s=7.0,
        hold_duration_s=12.0,
        deflate_duration_s=30.0,
        floor=0.30,  # nadir ~0.30 baseline, above the 0.20 depth floor
    )
    nbp = cuff.t_deflate_start_s + 5.0
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=cuff, rng=rng)
    r = detect_cuff_event(ppg, FS, nbp)
    assert not r.is_occlusion_signature
    assert not r.recovered or r.reject_reason is not None


def test_short_occlusion_below_sensitivity_floor_is_rejected(rng):
    """A sub-0.50 run shorter than 10 s does not qualify even though deep."""
    cuff = CuffEvent(
        t_inflation_start_s=240.0,
        inflation_duration_s=2.0,
        hold_duration_s=2.0,
        deflate_duration_s=5.0,
        n_deflate_steps=4,
        floor=0.05,
    )
    nbp = cuff.t_deflate_start_s
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=cuff, rng=rng)
    r = detect_cuff_event(ppg, FS, nbp)
    assert not r.is_occlusion_signature
    # sub-0.50 run length, if any dip found, is below the sensitivity floor
    if np.isfinite(r.phase3_duration_s):
        assert r.phase3_duration_s < 10.0


def test_medium_occlusion_is_sensitivity_not_primary(rng):
    """A 10 to 15 s sub-0.50 run that recovers is sensitivity, not primary."""
    # A short hold plus quick deflate gives a sub-0.50 excursion in the 10 to 15 s band.
    cuff = CuffEvent(
        t_inflation_start_s=240.0,
        inflation_duration_s=4.0,
        hold_duration_s=5.0,
        deflate_duration_s=12.0,
        n_deflate_steps=8,
        floor=0.05,
    )
    nbp = cuff.t_deflate_start_s + 3.0
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=cuff, rng=rng)
    r = detect_cuff_event(ppg, FS, nbp)
    # Qualifies for sensitivity (>=10 s, deep, aligned) but not primary (<15 s).
    assert r.nadir_depth_frac < 0.20
    assert 10.0 <= r.phase3_duration_s < 15.0
    assert not r.is_occlusion_signature, "a sub-15 s run must not be flagged primary"


def test_fast_and_slow_recovery_both_reach_release(rng):
    """Release is the first 0.85-sustained second; both recovery speeds find it."""
    fast = CuffEvent(
        t_inflation_start_s=240.0,
        hold_duration_s=10.0,
        deflate_duration_s=20.0,
        floor=0.05,
    )
    slow = CuffEvent(
        t_inflation_start_s=240.0,
        hold_duration_s=10.0,
        deflate_duration_s=45.0,
        floor=0.05,
    )
    for cuff in (fast, slow):
        nbp = cuff.t_deflate_start_s + 5.0
        _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=cuff, rng=rng)
        r = detect_cuff_event(ppg, FS, nbp)
        assert r.recovered, f"expected recovery for deflate={cuff.deflate_duration_s}"
        assert np.isfinite(r.t_release_s)
        assert r.t_release_s >= r.t_nadir_s


def test_non_recovered_event_has_nan_release_and_cannot_be_primary(rng):
    """A deep dip that never returns to 0.85 baseline is non-recovered."""
    # Hold to the end of the window so PI never recovers within the search window.
    cuff = CuffEvent(
        t_inflation_start_s=240.0,
        inflation_duration_s=7.0,
        hold_duration_s=120.0,  # stays occluded past the search window
        deflate_duration_s=10.0,
        floor=0.05,
    )
    nbp = 255.0  # inside the hold; nadir aligned
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=cuff, rng=rng)
    r = detect_cuff_event(ppg, FS, nbp)
    assert not r.recovered
    assert not np.isfinite(r.t_release_s)
    assert not r.is_occlusion_signature, "non-recovered cannot be primary"
    assert np.isfinite(r.recovery_fraction_at_window_end)
    assert r.recovery_fraction_at_window_end < 0.85


def test_nadir_outside_alignment_window_is_rejected(rng):
    """A genuine deep dip whose nadir is far from the BP is rejected for alignment."""
    cuff = CuffEvent(
        t_inflation_start_s=120.0,
        hold_duration_s=10.0,
        deflate_duration_s=30.0,
        floor=0.05,
    )
    nbp_far = 300.0  # nadir is ~170 s before this; outside [-50, +30]
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=cuff, rng=rng)
    r = detect_cuff_event(ppg, FS, nbp_far)
    assert not r.is_occlusion_signature


def test_two_dip_distractor_deeper_dip_wins_and_flags_ambiguous(rng):
    """Given two qualifying dips, the deeper one is selected and ambiguity flagged."""
    # Shallower dip nearer the BP, deeper dip slightly further but still in window.
    shallow = CuffEvent(
        t_inflation_start_s=240.0,
        inflation_duration_s=5.0,
        hold_duration_s=12.0,
        deflate_duration_s=25.0,
        floor=0.15,
    )
    deep = CuffEvent(
        t_inflation_start_s=280.0,
        inflation_duration_s=5.0,
        hold_duration_s=12.0,
        deflate_duration_s=25.0,
        floor=0.03,
    )
    # Charted BP between the two nadirs; both nadirs inside [-50, +30].
    nbp = 290.0
    # Build one PPG whose AC envelope carries both dips: the elementwise product
    # of the two single-event envelopes (each dip multiplies the shared cardiac
    # AC). This places two distinct sub-occlusion runs in one trace.
    n = int(600.0 * FS)
    t = np.arange(n) / FS
    cardiac = 0.6 * np.sin(2 * np.pi * 1.0 * t) + 0.4 * np.sin(4 * np.pi * 1.0 * t)
    env = _apply_cuff_envelope(t, shallow) * _apply_cuff_envelope(t, deep)
    base = 1.0 + 0.1 * cardiac * env + 0.005 * rng.standard_normal(n)
    r = detect_cuff_event(base, FS, nbp)
    # Both nadirs lie inside [-50, +30]; the deeper (0.03) dip must be selected
    # over the shallower (0.15) one even though the shallow nadir is comparable
    # in proximity to the BP.
    assert r.nadir_depth_frac < 0.10, f"deeper dip should win, got {r.nadir_depth_frac}"
    assert r.ambiguous_multi_dip


def test_motion_only_signal_is_rejected(rng):
    """A motion-artifact burst with no cuff event is not flagged ipsilateral."""
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=None, rng=rng)
    ppg_motion = add_motion_artifact(
        ppg,
        sampling_rate=FS,
        onset_s=270.0,
        duration_s=10.0,
        drift_amplitude=0.4,
        burst_amplitude=0.08,
        rng=rng,
    )
    r = detect_cuff_event(ppg_motion, FS, nbp_timestamp_s=275.0)
    assert not r.is_occlusion_signature
    assert not r.recovered or r.reject_reason is not None


def test_clean_ppg_no_event_is_rejected(rng):
    """No injected event must yield no detection at several timestamps."""
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=None, rng=rng)
    for nbp in (200.0, 300.0, 400.0):
        r = detect_cuff_event(ppg, FS, nbp)
        assert not r.is_occlusion_signature


def test_parameter_sweep_changes_outcome(rng):
    """Tightening nadir_depth must be able to flip a borderline event to rejected."""
    cuff = CuffEvent(
        t_inflation_start_s=240.0,
        hold_duration_s=12.0,
        deflate_duration_s=30.0,
        floor=0.17,  # nadir ~0.17 baseline: passes at 0.20, fails at 0.15
    )
    nbp = cuff.t_deflate_start_s + 5.0
    _, ppg = synthetic_ppg(duration_s=600.0, sampling_rate=FS, cuff_event=cuff, rng=rng)
    r_loose = detect_cuff_event(ppg, FS, nbp, nadir_depth=0.20)
    r_tight = detect_cuff_event(ppg, FS, nbp, nadir_depth=0.15)
    assert r_loose.is_occlusion_signature
    assert not r_tight.is_occlusion_signature
