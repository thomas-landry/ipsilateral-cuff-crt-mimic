"""Synthetic PPG signal generators for unit-testing the cuff-event detector.

Models the four-phase oscillometric cuff cycle (inflation, above-systolic
hold, stepwise deflate, and rapid dump). Used to validate
:mod:`cuffcrt.signal.cuff_event_detector` without touching real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cuffcrt._seed import RNG


@dataclass(frozen=True)
class CuffEvent:
    """Ground-truth oscillometric cuff cycle injected into a synthetic signal.

    Parameters
    ----------
    t_inflation_start_s : float
        Time at which cuff inflation begins (phase 1).
    inflation_duration_s : float
        Duration of the inflation ramp (phase 1).
    hold_duration_s : float
        Duration of the above-systolic hold (phase 2).
    deflate_duration_s : float
        Duration of the stepwise deflate (phase 3).
    n_deflate_steps : int
        Number of discrete pressure steps during deflate (typically 8 to 16).
    floor : float
        AC envelope amplitude during the hold phase, as a fraction of baseline.
        Real PPG floors are ~0.02 to 0.08 of the unblocked amplitude.
    """

    t_inflation_start_s: float
    inflation_duration_s: float = 7.0
    hold_duration_s: float = 8.0
    deflate_duration_s: float = 30.0
    n_deflate_steps: int = 12
    floor: float = 0.05

    @property
    def t_hold_start_s(self) -> float:
        return self.t_inflation_start_s + self.inflation_duration_s

    @property
    def t_deflate_start_s(self) -> float:
        return self.t_hold_start_s + self.hold_duration_s

    @property
    def t_release_s(self) -> float:
        return self.t_deflate_start_s + self.deflate_duration_s


def synthetic_ppg(
    duration_s: float = 600.0,
    sampling_rate: float = 125.0,
    heart_rate_hz: float = 1.0,
    ac_amplitude: float = 0.1,
    dc_level: float = 1.0,
    noise_sd: float = 0.005,
    cuff_event: CuffEvent | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic PPG signal optionally containing a cuff event.

    The baseline PPG is a sinusoidal cardiac waveform (fundamental plus second
    harmonic) on a fixed DC offset, plus white noise. A cuff event modulates
    the AC amplitude through the four-phase oscillometric profile.

    Parameters
    ----------
    duration_s : float
        Total signal duration in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    heart_rate_hz : float
        Cardiac fundamental frequency in Hz.
    ac_amplitude : float
        Amplitude of the pulsatile (AC) component.
    dc_level : float
        Constant (DC) offset.
    noise_sd : float
        Standard deviation of additive white noise.
    cuff_event : CuffEvent or None
        Optional cuff cycle to inject.
    rng : numpy.random.Generator or None
        Random generator; defaults to the project :data:`~cuffcrt._seed.RNG`.

    Returns
    -------
    t : numpy.ndarray
        Time axis in seconds, shape ``(n,)``.
    ppg : numpy.ndarray
        PPG signal, shape ``(n,)``.
    """
    if rng is None:
        rng = RNG

    n = int(duration_s * sampling_rate)
    t = np.arange(n) / sampling_rate

    cardiac = 0.6 * np.sin(2 * np.pi * heart_rate_hz * t) + 0.4 * np.sin(
        4 * np.pi * heart_rate_hz * t
    )

    envelope = np.ones_like(t)
    if cuff_event is not None:
        envelope = _apply_cuff_envelope(t, cuff_event)

    ac = ac_amplitude * cardiac * envelope
    dc = dc_level * np.ones_like(t)
    noise = noise_sd * rng.standard_normal(n)
    ppg = dc + ac + noise
    return t, ppg


def _apply_cuff_envelope(t: np.ndarray, cuff: CuffEvent) -> np.ndarray:
    """Build the four-phase AC-envelope multiplier for one cuff cycle."""
    env = np.ones_like(t)
    floor = cuff.floor

    # Phase 1: inflation ramp (linear 1.0 -> floor)
    p1 = (t >= cuff.t_inflation_start_s) & (t < cuff.t_hold_start_s)
    env[p1] = 1.0 - (1.0 - floor) * (
        (t[p1] - cuff.t_inflation_start_s) / cuff.inflation_duration_s
    )

    # Phase 2: above-systolic hold at floor
    p2 = (t >= cuff.t_hold_start_s) & (t < cuff.t_deflate_start_s)
    env[p2] = floor

    # Phase 3: stepwise deflate (discrete steps from floor up to ~0.95)
    p3 = (t >= cuff.t_deflate_start_s) & (t < cuff.t_release_s)
    if p3.any():
        rel_t = t[p3] - cuff.t_deflate_start_s
        step_idx = np.minimum(
            (rel_t / cuff.deflate_duration_s * cuff.n_deflate_steps).astype(int),
            cuff.n_deflate_steps - 1,
        )
        step_levels = floor + (0.95 - floor) * (
            np.arange(cuff.n_deflate_steps) / max(1, cuff.n_deflate_steps - 1)
        )
        env[p3] = step_levels[step_idx]

    # Phase 4: rapid dump plus post-cycle recovery (exponential rejoin)
    p4 = t >= cuff.t_release_s
    if p4.any():
        rec_t = t[p4] - cuff.t_release_s
        env[p4] = 1.0 - (1.0 - 0.95) * np.exp(-rec_t / 3.0)

    return env


def add_motion_artifact(
    ppg: np.ndarray,
    sampling_rate: float,
    onset_s: float,
    duration_s: float,
    drift_amplitude: float = 0.3,
    burst_amplitude: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Superimpose a low-frequency drift plus high-frequency burst on a PPG.

    Mimics motion artifact: the slow component shifts the DC offset over a few
    seconds, the fast component injects broadband noise that may transiently
    deflate the AC envelope without producing the stereotyped two-part cuff
    signature.

    Parameters
    ----------
    ppg : numpy.ndarray
        Input PPG signal.
    sampling_rate : float
        Sampling rate in Hz.
    onset_s : float
        Artifact onset in seconds.
    duration_s : float
        Artifact duration in seconds.
    drift_amplitude : float
        Amplitude of the slow drift component.
    burst_amplitude : float
        Standard deviation of the high-frequency burst.
    rng : numpy.random.Generator or None
        Random generator; defaults to the project :data:`~cuffcrt._seed.RNG`.

    Returns
    -------
    numpy.ndarray
        A copy of ``ppg`` with the artifact superimposed.
    """
    if rng is None:
        rng = RNG
    out = ppg.copy()
    n = len(ppg)
    t = np.arange(n) / sampling_rate
    mask = (t >= onset_s) & (t < onset_s + duration_s)
    n_mask = int(mask.sum())
    drift = drift_amplitude * np.sin(2 * np.pi * 0.3 * (t[mask] - onset_s))
    burst = burst_amplitude * rng.standard_normal(n_mask)
    out[mask] = out[mask] + drift + burst
    return out
