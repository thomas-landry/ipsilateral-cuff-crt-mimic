"""Population (inverse-probability) reweighting of gallery precision/recall.

The 568-card adjudication gallery was a stratified, detector-enriched sample of
the full evaluable cuff-cycle population. It oversampled detector-positive
cycles so that raw (unweighted) recall and specificity computed on the gallery
do not reflect the full evaluable population. This module reweights the
per-card classification outcomes by the known per-stratum inverse sampling
fraction (Horvitz-Thompson weighting) so that precision, recall, and
specificity estimate population quantities. Confidence intervals come from the
existing subject-clustered percentile bootstrap
(:func:`cuffcrt.analysis.bootstrap.cluster_bootstrap_ci`); here we resample
subjects, recompute the weighted ratio metrics per resample, and take percentile
intervals.

Estimand
--------
Treating ``occlusion_signature_present`` as the positive class and the blinded
human reader as the gold standard:

* precision   = P(reader+ | machine+)
* recall      = P(machine+ | reader+)         (sensitivity)
* specificity = P(machine- | reader-)

Under stratified oversampling each metric is a ratio of two population totals,
each estimated by a Horvitz-Thompson weighted sum over the sampled cards:

    metric = (sum_i w_i * num_i) / (sum_i w_i * den_i)

where ``w_i`` is the per-card population weight (stratum universe divided by
stratum sampled count), ``num_i`` is the per-card numerator indicator and
``den_i`` the per-card denominator indicator for that metric (the same 0/1
indicators used in the unweighted step-44 computation). The weighting changes
precision's denominator mix across strata (machine-positive cards live in all
three strata at very different sampling rates), while recall is computed within
the reader-positive class and specificity within the reader-negative class.

Indeterminate handling
--------------------------------------
Cards where the reader returned ``indeterminate`` or where the machine returned
``indeterminate`` / ``parse_failure`` are excluded from the binary denominator
for the affected metric, exactly as in step 44, and so carry no weighted mass
into that ratio. They are not silently dropped from the population: their weight
still describes population mass, the card is simply not callable.

The single public entry points are :func:`assign_card_weights`,
:func:`reconcile_partition`, :func:`metric_indicators`, :func:`weighted_metric`,
and :func:`weighted_metric_with_ci`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from cuffcrt._seed import GLOBAL_SEED

__all__ = [
    "POSITIVE_CALL",
    "NEGATIVE_CALL",
    "INDETERMINATE_CALL",
    "PARSE_FAILURE_CALL",
    "StratumSpec",
    "PartitionReconciliation",
    "WeightedMetricResult",
    "assign_card_weights",
    "metric_indicators",
    "reconcile_partition",
    "weighted_metric",
    "weighted_metric_with_ci",
]

POSITIVE_CALL = "occlusion_signature_present"
NEGATIVE_CALL = "no_occlusion_signature"
INDETERMINATE_CALL = "indeterminate"
PARSE_FAILURE_CALL = "parse_failure"

CALLABLE_CALLS = (POSITIVE_CALL, NEGATIVE_CALL)
_VALID_METRICS = ("precision", "recall", "specificity")


@dataclass(frozen=True)
class StratumSpec:
    """Sampling specification for one gallery stratum.

    Attributes
    ----------
    name : str
        Stratum identifier (matches the ``stratum`` column of the gallery
        manifest), for example ``"detector_positive"``.
    universe : int
        Number of cards of this stratum that existed in the evaluable
        population sampling frame (the count from which the gallery sample was
        drawn).
    sampled : int
        Number of cards of this stratum actually drawn into the gallery.

    Notes
    -----
    The per-card inverse-probability (population) weight for every card in this
    stratum is ``universe / sampled``. ``universe`` must be at least
    ``sampled`` and ``sampled`` must be positive.
    """

    name: str
    universe: int
    sampled: int

    @property
    def weight(self) -> float:
        """Per-card population weight ``universe / sampled``."""
        if self.sampled <= 0:
            raise ValueError(
                f"stratum {self.name!r} has sampled={self.sampled}; must be a positive integer."
            )
        if self.universe < self.sampled:
            raise ValueError(
                f"stratum {self.name!r} has universe={self.universe} < "
                f"sampled={self.sampled}; a stratum cannot sample more cards "
                "than exist in its universe."
            )
        return float(self.universe) / float(self.sampled)


@dataclass(frozen=True)
class PartitionReconciliation:
    """Reconciliation of the sampling strata against the evaluable population.

    Attributes
    ----------
    strata : tuple[StratumSpec, ...]
        The sampling strata used to draw the gallery.
    evaluable_population : int
        Total cards in the evaluable population the estimate targets (for
        example evaluable-with-pleth cycles).
    covered_universe : int
        Sum of the strata universes (the part of the population the sampling
        frame tiled).
    uncovered : int
        ``evaluable_population - covered_universe``. Cards in the evaluable
        population that fell outside every sampling stratum (never sampled,
        unlabeled).
    uncovered_label : str
        Plain-language description of which reject-reason categories make up the
        uncovered remainder.
    coverage_fraction : float
        ``covered_universe / evaluable_population``.

    Notes
    -----
    The uncovered remainder carries no reader or machine label. It is
    detector-negative by construction (it sits outside the detector-positive
    stratum), so for population denominators it belongs to the detector-negative
    universe; but because it was never adjudicated it cannot contribute callable
    outcomes. We therefore report population metrics over the *covered*
    universe and state the coverage fraction explicitly rather than imputing
    labels for the uncovered remainder.
    """

    strata: tuple[StratumSpec, ...]
    evaluable_population: int
    covered_universe: int
    uncovered: int
    uncovered_label: str
    coverage_fraction: float


@dataclass(frozen=True)
class WeightedMetricResult:
    """Point and percentile-CI for one weighted ratio metric.

    Attributes
    ----------
    metric : str
        ``"precision"``, ``"recall"``, or ``"specificity"``.
    point : float
        Weighted point estimate ``sum(w*num)/sum(w*den)`` on the full sample.
    ci_low, ci_high : float
        Lower / upper percentile bootstrap bounds at ``confidence_level``.
    n_eligible : int
        Number of cards entering the binary denominator (callable on both
        reader and machine) for this metric, before weighting.
    weighted_den : float
        Weighted denominator total (population-scale).
    n_resamples : int
        Number of subject-clustered bootstrap replicates.
    confidence_level : float
        Nominal CI coverage.
    seed : int
        Bootstrap seed.
    n_clusters : int
        Number of distinct subject clusters among the eligible cards.
    """

    metric: str
    point: float
    ci_low: float
    ci_high: float
    n_eligible: int
    weighted_den: float
    n_resamples: int
    confidence_level: float
    seed: int
    n_clusters: int


def reconcile_partition(
    strata: Sequence[StratumSpec],
    evaluable_population: int,
    uncovered_label: str = "",
) -> PartitionReconciliation:
    """Reconcile the sampling strata universes against the evaluable population.

    Parameters
    ----------
    strata : Sequence[StratumSpec]
        The sampling strata. Their ``universe`` counts should tile a subset of
        the evaluable population.
    evaluable_population : int
        Total cards in the population the estimate targets.
    uncovered_label : str, optional
        Plain-language description of the uncovered remainder's composition.

    Returns
    -------
    PartitionReconciliation
        Covered/uncovered counts and the coverage fraction.

    Raises
    ------
    ValueError
        If ``evaluable_population`` is not positive or if the strata universes
        exceed the evaluable population (which would mean the frame double
        counts cards).
    """
    if evaluable_population <= 0:
        raise ValueError(f"evaluable_population must be positive; got {evaluable_population}.")
    covered = int(sum(s.universe for s in strata))
    if covered > evaluable_population:
        raise ValueError(
            f"strata universes sum to {covered}, which exceeds the evaluable "
            f"population {evaluable_population}; strata must not double count."
        )
    uncovered = evaluable_population - covered
    return PartitionReconciliation(
        strata=tuple(strata),
        evaluable_population=int(evaluable_population),
        covered_universe=covered,
        uncovered=uncovered,
        uncovered_label=uncovered_label,
        coverage_fraction=covered / evaluable_population,
    )


def assign_card_weights(
    stratum_per_card: Sequence[str],
    strata: Sequence[StratumSpec],
) -> np.ndarray:
    """Map each card's stratum name to its per-card population weight.

    Parameters
    ----------
    stratum_per_card : Sequence[str]
        Stratum name for each card (length ``n_cards``).
    strata : Sequence[StratumSpec]
        Sampling specifications; one per distinct stratum name.

    Returns
    -------
    numpy.ndarray of shape (n_cards,)
        The population weight ``universe / sampled`` for each card.

    Raises
    ------
    ValueError
        If a card references a stratum name that has no ``StratumSpec``, or if
        any ``StratumSpec`` is degenerate (see :attr:`StratumSpec.weight`).
    """
    weight_by_name = {s.name: s.weight for s in strata}
    out = np.empty(len(stratum_per_card), dtype=np.float64)
    for i, name in enumerate(stratum_per_card):
        if name not in weight_by_name:
            raise ValueError(
                f"card {i} references stratum {name!r} with no StratumSpec; "
                f"known strata: {sorted(weight_by_name)}."
            )
        out[i] = weight_by_name[name]
    return out


def metric_indicators(
    reader_calls: Sequence[str],
    machine_calls: Sequence[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-card 0/1 numerator and denominator indicators for ``metric``.

    Mirrors the step-44 binary definition with
    ``occlusion_signature_present`` as the positive class. A card contributes
    to the denominator only when both the reader and machine calls are callable
    for that metric's class; indeterminate / parse-failure cards yield zeros in
    both arrays and so drop out of the weighted ratio.

    Parameters
    ----------
    reader_calls, machine_calls : Sequence[str]
        Per-card reader (gold-standard) and machine calls.
    metric : str
        One of ``"precision"``, ``"recall"``, ``"specificity"``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(num, den)`` float64 arrays of shape ``(n_cards,)``.

    Raises
    ------
    ValueError
        If ``metric`` is not recognized or the two call sequences differ in
        length.
    """
    if metric not in _VALID_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {_VALID_METRICS}.")
    if len(reader_calls) != len(machine_calls):
        raise ValueError(
            "reader_calls and machine_calls must have equal length; "
            f"got {len(reader_calls)} and {len(machine_calls)}."
        )
    n = len(reader_calls)
    num = np.zeros(n, dtype=np.float64)
    den = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = reader_calls[i]
        m = machine_calls[i]
        if metric == "precision":
            if m == POSITIVE_CALL and r in CALLABLE_CALLS:
                den[i] = 1.0
                if r == POSITIVE_CALL:
                    num[i] = 1.0
        elif metric == "recall":
            if r == POSITIVE_CALL and m in CALLABLE_CALLS:
                den[i] = 1.0
                if m == POSITIVE_CALL:
                    num[i] = 1.0
        else:  # specificity
            if r == NEGATIVE_CALL and m in CALLABLE_CALLS:
                den[i] = 1.0
                if m == NEGATIVE_CALL:
                    num[i] = 1.0
    return num, den


def weighted_metric(
    num: ArrayLike,
    den: ArrayLike,
    weights: ArrayLike,
) -> float:
    """Horvitz-Thompson weighted ratio ``sum(w*num)/sum(w*den)``.

    Parameters
    ----------
    num, den : array-like
        Per-card numerator and denominator 0/1 indicators (same length).
    weights : array-like
        Per-card population weights (same length).

    Returns
    -------
    float
        The weighted ratio, or NaN if the weighted denominator is zero.

    Raises
    ------
    ValueError
        If the three arrays differ in length.
    """
    num_a = np.asarray(num, dtype=np.float64)
    den_a = np.asarray(den, dtype=np.float64)
    w_a = np.asarray(weights, dtype=np.float64)
    if not (num_a.shape == den_a.shape == w_a.shape):
        raise ValueError(
            "num, den, weights must have identical shapes; got "
            f"{num_a.shape}, {den_a.shape}, {w_a.shape}."
        )
    wd = float(np.sum(w_a * den_a))
    if wd <= 0.0:
        return float("nan")
    wn = float(np.sum(w_a * num_a))
    return wn / wd


def weighted_metric_with_ci(
    reader_calls: Sequence[str],
    machine_calls: Sequence[str],
    weights: ArrayLike,
    clusters: Sequence[object],
    metric: str,
    *,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = GLOBAL_SEED,
) -> WeightedMetricResult:
    """Weighted point estimate + subject-clustered percentile bootstrap CI.

    Resamples subjects (clusters) with replacement, recomputes the weighted
    ratio metric on the pooled cards each resample, and reports percentile
    bounds. The point estimate is computed once on the full sample (not the
    bootstrap mean), matching the convention in
    :func:`cuffcrt.analysis.bootstrap.cluster_bootstrap_ci`.

    Parameters
    ----------
    reader_calls, machine_calls : Sequence[str]
        Per-card reader and machine calls.
    weights : array-like
        Per-card population weights.
    clusters : Sequence[object]
        Per-card subject id (the cluster axis).
    metric : str
        ``"precision"``, ``"recall"``, or ``"specificity"``.
    n_resamples : int, optional
        Number of cluster-bootstrap replicates. Default 5000.
    confidence_level : float, optional
        Nominal CI coverage in (0, 1). Default 0.95.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Default
        :data:`cuffcrt._seed.GLOBAL_SEED`.

    Returns
    -------
    WeightedMetricResult
        Point, CI bounds, and bookkeeping.

    Raises
    ------
    ValueError
        On mismatched lengths, invalid ``metric``, non-positive
        ``n_resamples``, or out-of-range ``confidence_level``.
    """
    n = len(reader_calls)
    w_a = np.asarray(weights, dtype=np.float64)
    clusters_a = np.asarray(list(clusters))
    if not (len(machine_calls) == w_a.shape[0] == clusters_a.shape[0] == n):
        raise ValueError(
            "reader_calls, machine_calls, weights, clusters must share length; "
            f"got {n}, {len(machine_calls)}, {w_a.shape[0]}, {clusters_a.shape[0]}."
        )
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive int; got {n_resamples!r}.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level!r}.")

    num, den = metric_indicators(reader_calls, machine_calls, metric)
    point = weighted_metric(num, den, w_a)
    n_eligible = int(np.count_nonzero(den))
    weighted_den = float(np.sum(w_a * den))

    # Group card indices by subject cluster, preserving first-seen order.
    cluster_to_idx: dict[object, list[int]] = {}
    for i, c in enumerate(clusters_a.tolist()):
        cluster_to_idx.setdefault(c, []).append(i)
    unique = list(cluster_to_idx.keys())
    n_clusters = len(unique)
    idx_lists = [np.asarray(cluster_to_idx[c], dtype=np.int64) for c in unique]

    if n_clusters <= 1:
        return WeightedMetricResult(
            metric=metric,
            point=point,
            ci_low=point,
            ci_high=point,
            n_eligible=n_eligible,
            weighted_den=weighted_den,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed,
            n_clusters=n_clusters,
        )

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        pooled = np.concatenate([idx_lists[p] for p in picks])
        replicates[b] = weighted_metric(num[pooled], den[pooled], w_a[pooled])

    finite = replicates[np.isfinite(replicates)]
    alpha = 1.0 - confidence_level
    if finite.size == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.percentile(finite, 100.0 * (alpha / 2.0)))
        ci_high = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))

    return WeightedMetricResult(
        metric=metric,
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_eligible=n_eligible,
        weighted_den=weighted_den,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        n_clusters=n_clusters,
    )
