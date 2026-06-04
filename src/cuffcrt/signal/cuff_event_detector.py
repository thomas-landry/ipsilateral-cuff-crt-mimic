"""Detect the cuff-occlusion signature in a photoplethysmogram.

When a noninvasive blood-pressure cuff and the pulse-oximeter probe sit on the
same limb, an automated cuff cycle transiently occludes arterial inflow to the
probe. The perfusion index (PI) then traces a stereotyped occlusion-reperfusion
shape: a near-total collapse to a deep nadir, a sustained floor, and a graded
recovery back toward baseline.

This module reduces a raw PPG window to a 1 Hz PI trace, smooths it with a 5 s
rolling median, then classifies each charted noninvasive blood-pressure (NBP)
timestamp by whether the occlusion-reperfusion signature is present and
aligned. The detector parameters are pre-registered in
``findings/preregistration_detector.md``. Detection is deterministic and uses
no random state.

Phase definitions:

1. Baseline is the **median** smoothed PI over the pre-cuff window
   ``[t_nbp - 120, t_nbp - 60]`` s.
2. The occlusion is a contiguous run with smoothed ``PI < occlusion_fraction *
   baseline`` (default 0.50) that contains a nadir below ``nadir_depth *
   baseline`` (default 0.20). Onset is the start of that run; ``t_nadir`` is the
   argmin inside it.
3. Reperfusion end (``t_release``) is the first second at or after ``t_nadir``
   where smoothed PI reaches ``recovery_fraction * baseline`` (default 0.85) and
   stays there for ``recovery_hold_s`` (default 2 s). If recovery never sustains,
   ``t_release`` is NaN, ``recovered`` is False, and
   ``recovery_fraction_at_window_end`` records how close the trace came.
4. Alignment: the nadir must fall within ``[align_lo, align_hi]`` s of the
   charted BP (default ``[-50, +30]``); dips lead the BP.
5. Dip selection: among qualifying sub-occlusion runs (length
   ``>= sensitivity_min_s``, nadir ``< nadir_depth``) whose nadir lies in the
   alignment window, the run with the deepest nadir is chosen; ties break by
   proximity to the charted BP. ``ambiguous_multi_dip`` is set when two or more
   qualifying runs exist.

Backward-compatible field mapping for downstream scripts (30/50/51):

    t_occlusion_start_s  onset of the selected sub-occlusion run
    t_deflate_start_s    alias for t_nadir_s (the full-occlusion instant)
    t_release_s          reperfusion end (0.85-baseline sustained), NaN if none
    phase2_duration_s    descent duration (occlusion onset to nadir)
    phase3_duration_s    sub-occlusion run length (the event-defining duration
                         the funnel thresholds on)
    alignment_offset_s   t_nadir minus the charted BP timestamp

Additional fields: ``t_nadir_s``, ``nadir_depth_frac``,
``recovery_fraction_at_window_end``, ``ambiguous_multi_dip``, ``recovered``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

# ---------------------------------------------------------------------------
# Locked detector parameters (pre-registration). Exposed as function-argument
# defaults so a sensitivity sweep can vary them without editing the detector.
# ---------------------------------------------------------------------------
NADIR_DEPTH = 0.20  # nadir must reach below this fraction of baseline
OCCLUSION_FRACTION = 0.50  # sub-occlusion run threshold (fraction of baseline)
RECOVERY_FRACTION = 0.85  # reperfusion end threshold (fraction of baseline)
RECOVERY_HOLD_S = 2.0  # seconds the recovered band must be sustained
ALIGN_LO_S = -50.0  # alignment window lower bound (nadir minus BP)
ALIGN_HI_S = 30.0  # alignment window upper bound (nadir minus BP)
PRIMARY_MIN_S = 15.0  # sub-occlusion run length for a primary event
SENSITIVITY_MIN_S = 10.0  # sub-occlusion run length for a sensitivity event

# Search windows (seconds, relative to t_nbp).
PRE_WINDOW_LO_S = 120.0  # pre-window starts here before t_nbp
PRE_WINDOW_HI_S = 60.0   # pre-window ends here before t_nbp
SEARCH_LO_S = 60.0       # search starts this many seconds before t_nbp
SEARCH_HI_S = 90.0       # search ends this many seconds after t_nbp

# Signal-quality gates on the pre-window.
PRE_PI_MIN = 1.0
PRE_PI_MAX = 100.0
PRE_PI_REL_SD_MAX = 0.30  # rolling SD divided by the median

# Tie-break epsilon for equal-depth nadirs (fraction of baseline).
NADIR_TIE_EPS = 0.01

# Smoothing window for the 1 Hz PI trace (seconds).
SMOOTH_WINDOW_S = 5


@dataclass(frozen=True)
class CuffEventResult:
    """Classification of one charted NBP timestamp under the detector.

    All time fields are in seconds from the start of the supplied PPG window
    (the same axis as the input ``nbp_timestamp_s``).

    A primary event has ``is_occlusion_signature = True`` and
    ``reject_reason = None``: a sub-occlusion run of at least ``primary_min_s``
    whose nadir is below ``nadir_depth`` and inside the alignment window, with a
    defined ``t_release_s`` (recovered). Shorter (``>= sensitivity_min_s``) or
    non-recovered qualifying events carry a descriptive ``reject_reason`` but
    still populate ``phase3_duration_s`` so the funnel can count them at the
    sensitivity threshold.

    Attributes
    ----------
    nbp_timestamp_s : float
        Charted NBP timestamp (window-local seconds).
    is_occlusion_signature : bool
        True only for a primary event (recovered, aligned, deep, run >= primary).
        This is a morphology-based call and does not assert cuff laterality;
        MIMIC-IV-WDB carries no ground-truth cuff-arm label.
    stat_mode_candidate : bool
        Retained for backward compatibility: True when a qualifying dip was found
        but the sub-occlusion run is below ``primary_min_s`` (the sensitivity
        stratum / former rapid-cycle branch).
    recovered : bool
        True when reperfusion reached and sustained the recovery fraction.
    ambiguous_multi_dip : bool
        True when two or more qualifying sub-occlusion runs were present.
    pre_event_pi_mean : float
        Pre-cuff baseline (median smoothed PI over the pre-window). Named for
        backward compatibility; it is a median, not a mean.
    pre_window_quality : float
        Relative SD of the pre-window (SD divided by median).
    pre_window_valid : bool
        Whether the pre-window passed quality gates.
    t_occlusion_start_s : float
        Onset of the selected sub-occlusion run.
    t_deflate_start_s : float
        Alias for ``t_nadir_s`` (the full-occlusion instant).
    t_nadir_s : float
        Second of minimum smoothed PI inside the selected run.
    t_release_s : float
        Reperfusion end (sustained recovery fraction), NaN if non-recovered.
    phase2_duration_s : float
        Descent duration (occlusion onset to nadir).
    phase3_duration_s : float
        Sub-occlusion run length (the event-defining duration).
    nadir_depth_frac : float
        Nadir PI divided by baseline (the depth fraction reached).
    recovery_fraction_at_window_end : float
        Max smoothed PI after the nadir divided by baseline (how close recovery
        came when ``recovered`` is False).
    alignment_offset_s : float
        ``t_nadir`` minus the charted BP timestamp.
    reject_reason : str or None
        None for a primary event; otherwise the rejection or downgrade reason.
    """

    nbp_timestamp_s: float
    is_occlusion_signature: bool
    stat_mode_candidate: bool
    recovered: bool
    ambiguous_multi_dip: bool
    pre_event_pi_mean: float
    pre_window_quality: float
    pre_window_valid: bool
    t_occlusion_start_s: float
    t_deflate_start_s: float
    t_nadir_s: float
    t_release_s: float
    phase2_duration_s: float
    phase3_duration_s: float
    nadir_depth_frac: float
    recovery_fraction_at_window_end: float
    alignment_offset_s: float
    reject_reason: str | None


def compute_pi_1hz(ppg: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute perfusion index at 1 Hz from a raw PPG signal.

    PI is approximated as ``100 * AC_envelope / DC``, where AC is the
    bandpass-filtered cardiac component (0.5 to 8 Hz) and the AC envelope is a
    lowpass of ``|AC|``. Values are binned to 1 Hz by mean.

    Parameters
    ----------
    ppg : numpy.ndarray
        Raw photoplethysmogram samples.
    sampling_rate : float
        PPG sampling rate in Hz.

    Returns
    -------
    t_1hz : numpy.ndarray
        Integer-second time axis aligned to the start of ``ppg``.
    pi : numpy.ndarray
        Perfusion index, percentage. Empty when the signal is too short.
    """
    if sampling_rate <= 0 or ppg.size == 0:
        return np.array([]), np.array([])

    nyq = sampling_rate / 2.0
    sos_ac = sps.butter(4, [0.5 / nyq, 8.0 / nyq], btype="band", output="sos")
    sos_dc = sps.butter(2, 0.4 / nyq, btype="low", output="sos")
    sos_envelope = sps.butter(2, 1.0 / nyq, btype="low", output="sos")

    ac = sps.sosfiltfilt(sos_ac, ppg)
    dc = sps.sosfiltfilt(sos_dc, ppg)
    ac_envelope = sps.sosfiltfilt(sos_envelope, np.abs(ac))
    pi_raw = 100.0 * ac_envelope / np.maximum(np.abs(dc), 1e-9)

    duration_s = len(ppg) / sampling_rate
    n_bins = int(np.floor(duration_s))
    if n_bins == 0:
        return np.array([]), np.array([])
    samples_per_bin = int(sampling_rate)
    usable = n_bins * samples_per_bin
    pi_1hz = pi_raw[:usable].reshape(n_bins, samples_per_bin).mean(axis=1)
    t_1hz = np.arange(n_bins, dtype=float)
    return t_1hz, pi_1hz


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median; edges use shrinking windows."""
    n = len(values)
    if n == 0 or window <= 1:
        return values.copy()
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def _runs_below(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """All contiguous (start, end) inclusive index pairs where values < threshold."""
    below = values < threshold
    if not below.any():
        return []
    edges = np.diff(below.astype(np.int8))
    starts = (np.where(edges == 1)[0] + 1).tolist()
    ends = np.where(edges == -1)[0].tolist()
    if below[0]:
        starts.insert(0, 0)
    if below[-1]:
        ends.append(len(below) - 1)
    return list(zip(starts, ends, strict=True))


def _check_pre_window_quality(pre_pi: np.ndarray) -> tuple[float, float, str | None]:
    """Return ``(baseline, rel_sd, reject_reason)`` for the pre-cuff window.

    ``baseline`` is the **median** of the smoothed pre-window PI; the median is
    robust to the unstable upper tail of the trace. ``rel_sd`` is the window
    standard deviation divided by that median.
    """
    if pre_pi.size == 0 or not np.isfinite(pre_pi).any():
        return float("nan"), float("nan"), "no_pre_window"
    finite = pre_pi[np.isfinite(pre_pi)]
    if finite.size < 30:
        return float("nan"), float("nan"), "no_pre_window"
    baseline = float(np.median(finite))
    sd = float(np.std(finite))
    rel = sd / baseline if baseline > 0 else float("nan")
    if not np.isfinite(baseline) or not (PRE_PI_MIN <= baseline <= PRE_PI_MAX):
        return baseline, rel, "pre_pi_implausible"
    if not np.isfinite(rel) or rel > PRE_PI_REL_SD_MAX:
        return baseline, rel, "pre_window_unstable"
    return baseline, rel, None


@dataclass(frozen=True)
class _QualifyingRun:
    """A sub-occlusion run that meets the depth and duration floors."""

    start_idx: int
    end_idx: int
    nadir_idx: int
    nadir_frac: float
    duration_s: float
    t_nadir: float
    offset: float  # t_nadir minus the charted BP


def _enumerate_qualifying_runs(
    search_pi: np.ndarray,
    search_t: np.ndarray,
    baseline: float,
    nbp_timestamp_s: float,
    occlusion_fraction: float,
    nadir_depth: float,
    sensitivity_min_s: float,
) -> list[_QualifyingRun]:
    """Enumerate sub-occlusion runs meeting depth and duration floors.

    A qualifying run has smoothed ``PI < occlusion_fraction * baseline`` for a
    span of at least ``sensitivity_min_s`` seconds and contains a nadir below
    ``nadir_depth * baseline``.
    """
    threshold_occ = occlusion_fraction * baseline
    threshold_depth = nadir_depth * baseline
    runs = _runs_below(search_pi, threshold_occ)
    out: list[_QualifyingRun] = []
    for start, end in runs:
        duration_s = float(search_t[end] - search_t[start])
        if duration_s < sensitivity_min_s:
            continue
        segment = search_pi[start : end + 1]
        local_min = int(np.argmin(segment))
        nadir_idx = start + local_min
        nadir_pi = float(search_pi[nadir_idx])
        if nadir_pi >= threshold_depth:
            continue
        t_nadir = float(search_t[nadir_idx])
        out.append(
            _QualifyingRun(
                start_idx=start,
                end_idx=end,
                nadir_idx=nadir_idx,
                nadir_frac=nadir_pi / baseline,
                duration_s=duration_s,
                t_nadir=t_nadir,
                offset=t_nadir - nbp_timestamp_s,
            )
        )
    return out


def _select_run(
    runs: list[_QualifyingRun],
    align_lo: float,
    align_hi: float,
) -> _QualifyingRun | None:
    """Pick the deepest-nadir run whose nadir is inside the alignment window.

    Ties on depth (within :data:`NADIR_TIE_EPS` of baseline) break by proximity
    of the nadir to the charted BP. Deterministic.
    """
    aligned = [r for r in runs if align_lo <= r.offset <= align_hi]
    if not aligned:
        return None
    deepest = min(r.nadir_frac for r in aligned)
    near_deepest = [r for r in aligned if r.nadir_frac <= deepest + NADIR_TIE_EPS]
    return min(near_deepest, key=lambda r: (abs(r.offset), r.start_idx))


def _find_release(
    search_pi: np.ndarray,
    search_t: np.ndarray,
    nadir_idx: int,
    baseline: float,
    recovery_fraction: float,
    recovery_hold_s: float,
) -> tuple[float, bool, float]:
    """Find the reperfusion-end time after the nadir.

    Returns ``(t_release, recovered, recovery_fraction_at_window_end)``.
    ``t_release`` is the first second at or after the nadir where smoothed PI is
    at or above ``recovery_fraction * baseline`` and stays there for at least
    ``recovery_hold_s`` seconds. If never sustained, ``t_release`` is NaN,
    ``recovered`` is False, and the third value reports the max post-nadir PI as
    a fraction of baseline.
    """
    threshold = recovery_fraction * baseline
    post = search_pi[nadir_idx:]
    post_t = search_t[nadir_idx:]
    hold = max(1, int(round(recovery_hold_s)))
    above = post >= threshold
    n = len(post)
    for i in range(n):
        if not above[i]:
            continue
        end = min(n, i + hold)
        if above[i:end].all() and (end - i) >= hold:
            return float(post_t[i]), True, float(np.max(post) / baseline)
    frac_end = float(np.max(post) / baseline) if post.size else float("nan")
    return float("nan"), False, frac_end


def detect_cuff_event(
    ppg: np.ndarray,
    sampling_rate: float,
    nbp_timestamp_s: float,
    *,
    nadir_depth: float = NADIR_DEPTH,
    occlusion_fraction: float = OCCLUSION_FRACTION,
    recovery_fraction: float = RECOVERY_FRACTION,
    recovery_hold_s: float = RECOVERY_HOLD_S,
    align_lo: float = ALIGN_LO_S,
    align_hi: float = ALIGN_HI_S,
    primary_min_s: float = PRIMARY_MIN_S,
    sensitivity_min_s: float = SENSITIVITY_MIN_S,
) -> CuffEventResult:
    """Classify one charted NBP timestamp by the cuff-occlusion signature.

    Parameters
    ----------
    ppg : numpy.ndarray
        Continuous PPG signal centered around the NBP event. Should cover at
        least ``[t_nbp - 120, t_nbp + 90]`` s.
    sampling_rate : float
        PPG sampling rate in Hz (use the channel's native rate, not the master
        record's frame rate).
    nbp_timestamp_s : float
        Charted NBP timestamp in seconds from the start of ``ppg``.
    nadir_depth : float
        Nadir must reach below this fraction of baseline (default 0.20).
    occlusion_fraction : float
        Sub-occlusion run threshold as a fraction of baseline (default 0.50).
    recovery_fraction : float
        Reperfusion-end threshold as a fraction of baseline (default 0.85).
    recovery_hold_s : float
        Seconds the recovered band must be sustained (default 2).
    align_lo, align_hi : float
        Alignment window bounds on ``t_nadir - t_nbp`` (default -50, +30).
    primary_min_s : float
        Sub-occlusion run length for a primary event (default 15 s).
    sensitivity_min_s : float
        Sub-occlusion run length for a sensitivity event (default 10 s).

    Returns
    -------
    CuffEventResult
        The classification and timing anchors for this timestamp.
    """
    t_pi, pi = compute_pi_1hz(ppg, sampling_rate)
    if pi.size == 0:
        return _empty_result("no_pi", nbp_timestamp_s)
    return detect_cuff_event_on_pi(
        t_pi,
        pi,
        nbp_timestamp_s,
        nadir_depth=nadir_depth,
        occlusion_fraction=occlusion_fraction,
        recovery_fraction=recovery_fraction,
        recovery_hold_s=recovery_hold_s,
        align_lo=align_lo,
        align_hi=align_hi,
        primary_min_s=primary_min_s,
        sensitivity_min_s=sensitivity_min_s,
    )


def _empty_result(
    reason: str,
    nbp_timestamp_s: float,
    *,
    pre_event_pi_mean: float = float("nan"),
    pre_window_quality: float = float("nan"),
    pre_window_valid: bool = False,
    t_occlusion_start_s: float = float("nan"),
    t_nadir_s: float = float("nan"),
    phase2_duration_s: float = float("nan"),
    phase3_duration_s: float = float("nan"),
    nadir_depth_frac: float = float("nan"),
    recovery_fraction_at_window_end: float = float("nan"),
    alignment_offset_s: float = float("nan"),
    ambiguous_multi_dip: bool = False,
    stat_mode_candidate: bool = False,
    recovered: bool = False,
) -> CuffEventResult:
    """Build a non-detection ``CuffEventResult`` with the given reason."""
    nan = float("nan")
    return CuffEventResult(
        nbp_timestamp_s=nbp_timestamp_s,
        is_occlusion_signature=False,
        stat_mode_candidate=stat_mode_candidate,
        recovered=recovered,
        ambiguous_multi_dip=ambiguous_multi_dip,
        pre_event_pi_mean=pre_event_pi_mean,
        pre_window_quality=pre_window_quality,
        pre_window_valid=pre_window_valid,
        t_occlusion_start_s=t_occlusion_start_s,
        t_deflate_start_s=t_nadir_s,
        t_nadir_s=t_nadir_s,
        t_release_s=nan,
        phase2_duration_s=phase2_duration_s,
        phase3_duration_s=phase3_duration_s,
        nadir_depth_frac=nadir_depth_frac,
        recovery_fraction_at_window_end=recovery_fraction_at_window_end,
        alignment_offset_s=alignment_offset_s,
        reject_reason=reason,
    )


def detect_cuff_event_on_pi(
    t_pi: np.ndarray,
    pi: np.ndarray,
    nbp_timestamp_s: float,
    *,
    nadir_depth: float = NADIR_DEPTH,
    occlusion_fraction: float = OCCLUSION_FRACTION,
    recovery_fraction: float = RECOVERY_FRACTION,
    recovery_hold_s: float = RECOVERY_HOLD_S,
    align_lo: float = ALIGN_LO_S,
    align_hi: float = ALIGN_HI_S,
    primary_min_s: float = PRIMARY_MIN_S,
    sensitivity_min_s: float = SENSITIVITY_MIN_S,
) -> CuffEventResult:
    """Classify one NBP timestamp from a precomputed 1 Hz PI trace.

    This is the parameter-varying core shared by :func:`detect_cuff_event` and
    the sensitivity sweep. Computing the 1 Hz PI once and reusing it across many
    parameter sets keeps the sweep fast and fully deterministic. ``t_pi`` and
    ``pi`` are the raw (unsmoothed) 1 Hz outputs of :func:`compute_pi_1hz`; the
    5 s rolling-median smoothing is applied here so it is identical across all
    callers.

    Parameters
    ----------
    t_pi : numpy.ndarray
        Integer-second time axis from :func:`compute_pi_1hz`.
    pi : numpy.ndarray
        Raw 1 Hz perfusion index from :func:`compute_pi_1hz`.
    nbp_timestamp_s : float
        Charted NBP timestamp in seconds from the start of the window.

    Returns
    -------
    CuffEventResult
        The classification and timing anchors for this timestamp.
    """
    nan = float("nan")

    def _empty(
        reason: str,
        *,
        pre_event_pi_mean: float = nan,
        pre_window_quality: float = nan,
        pre_window_valid: bool = False,
        t_occlusion_start_s: float = nan,
        t_nadir_s: float = nan,
        phase2_duration_s: float = nan,
        phase3_duration_s: float = nan,
        nadir_depth_frac: float = nan,
        recovery_fraction_at_window_end: float = nan,
        alignment_offset_s: float = nan,
        ambiguous_multi_dip: bool = False,
        stat_mode_candidate: bool = False,
        recovered: bool = False,
    ) -> CuffEventResult:
        return CuffEventResult(
            nbp_timestamp_s=nbp_timestamp_s,
            is_occlusion_signature=False,
            stat_mode_candidate=stat_mode_candidate,
            recovered=recovered,
            ambiguous_multi_dip=ambiguous_multi_dip,
            pre_event_pi_mean=pre_event_pi_mean,
            pre_window_quality=pre_window_quality,
            pre_window_valid=pre_window_valid,
            t_occlusion_start_s=t_occlusion_start_s,
            t_deflate_start_s=t_nadir_s,
            t_nadir_s=t_nadir_s,
            t_release_s=nan,
            phase2_duration_s=phase2_duration_s,
            phase3_duration_s=phase3_duration_s,
            nadir_depth_frac=nadir_depth_frac,
            recovery_fraction_at_window_end=recovery_fraction_at_window_end,
            alignment_offset_s=alignment_offset_s,
            reject_reason=reason,
        )

    if pi.size == 0:
        return _empty("no_pi")

    # Smooth the whole 1 Hz trace once with the 5 s rolling median so the
    # baseline and the search-window threshold logic operate on the same
    # envelope. Per-second PI is noisy because beat-to-beat amplitude varies;
    # the underlying cuff envelope is much smoother than the raw trace.
    pi_smooth = _rolling_median(pi, window=SMOOTH_WINDOW_S)

    pre_mask = (t_pi >= nbp_timestamp_s - PRE_WINDOW_LO_S) & (
        t_pi < nbp_timestamp_s - PRE_WINDOW_HI_S
    )
    if not pre_mask.any():
        return _empty("no_pre_window")
    pre_pi = pi_smooth[pre_mask]
    baseline, rel_sd, quality_reason = _check_pre_window_quality(pre_pi)
    if quality_reason is not None:
        return _empty(
            quality_reason,
            pre_event_pi_mean=baseline,
            pre_window_quality=rel_sd,
            pre_window_valid=False,
        )

    search_mask = (t_pi >= nbp_timestamp_s - SEARCH_LO_S) & (
        t_pi < nbp_timestamp_s + SEARCH_HI_S
    )
    search_pi = pi_smooth[search_mask]
    search_t = t_pi[search_mask]
    if search_pi.size == 0:
        return _empty(
            "no_search_window",
            pre_event_pi_mean=baseline,
            pre_window_quality=rel_sd,
            pre_window_valid=True,
        )

    runs = _enumerate_qualifying_runs(
        search_pi,
        search_t,
        baseline,
        nbp_timestamp_s,
        occlusion_fraction,
        nadir_depth,
        sensitivity_min_s,
    )
    ambiguous = len(runs) >= 2
    if not runs:
        return _empty(
            "no_phase2",
            pre_event_pi_mean=baseline,
            pre_window_quality=rel_sd,
            pre_window_valid=True,
        )

    selected = _select_run(runs, align_lo, align_hi)
    if selected is None:
        # A qualifying dip exists but none aligns with the charted BP.
        return _empty(
            "no_aligned_occlusion",
            pre_event_pi_mean=baseline,
            pre_window_quality=rel_sd,
            pre_window_valid=True,
            ambiguous_multi_dip=ambiguous,
        )

    t_occlusion_start = float(search_t[selected.start_idx])
    t_nadir = selected.t_nadir
    phase2_duration = t_nadir - t_occlusion_start
    run_duration = selected.duration_s

    t_release, recovered, frac_end = _find_release(
        search_pi,
        search_t,
        selected.nadir_idx,
        baseline,
        recovery_fraction,
        recovery_hold_s,
    )

    # Classification.
    #  - primary: run >= primary_min_s, deep, aligned, recovered.
    #  - sensitivity (stat_mode_candidate): qualifying (>= sensitivity_min_s,
    #    deep, aligned) but either run < primary_min_s or not recovered.
    is_primary = run_duration >= primary_min_s and recovered
    reject_reason: str | None
    if is_primary:
        reject_reason = None
    elif not recovered:
        reject_reason = "no_recovery_in_window"
    else:
        reject_reason = "stat_mode_short_phase3"

    return CuffEventResult(
        nbp_timestamp_s=nbp_timestamp_s,
        is_occlusion_signature=is_primary,
        stat_mode_candidate=not is_primary,
        recovered=recovered,
        ambiguous_multi_dip=ambiguous,
        pre_event_pi_mean=baseline,
        pre_window_quality=rel_sd,
        pre_window_valid=True,
        t_occlusion_start_s=t_occlusion_start,
        t_deflate_start_s=t_nadir,
        t_nadir_s=t_nadir,
        t_release_s=t_release,
        phase2_duration_s=phase2_duration,
        phase3_duration_s=run_duration,
        nadir_depth_frac=selected.nadir_frac,
        recovery_fraction_at_window_end=frac_end,
        alignment_offset_s=selected.offset,
        reject_reason=reject_reason,
    )
