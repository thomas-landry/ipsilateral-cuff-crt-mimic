"""Subject-clustered bootstrap confidence intervals.

Pre-registered inferential procedure for the feasibility paper. The cluster
axis is the patient/subject id: clusters (subjects) are resampled with
replacement, then all observations belonging to the resampled clusters are
included for the statistic. This preserves the within-subject correlation
structure of repeated cuff cycles per patient. Percentile intervals are used
(2.5th / 97.5th percentile by default).

The single public entry point is :func:`cluster_bootstrap_ci`. Randomness uses
``numpy.random.default_rng(seed)`` with the global default seed taken from
:mod:`cuffcrt._seed`; no module-level RNG state is created here.

Examples
--------
>>> import numpy as np
>>> from cuffcrt.analysis import cluster_bootstrap_ci
>>> values = np.array([0, 1, 1, 0, 1, 1, 0, 0])
>>> clusters = np.array(["s1", "s1", "s1", "s2", "s2", "s2", "s3", "s3"])
>>> res = cluster_bootstrap_ci(values, clusters, n_resamples=2000)
>>> 0.0 <= res.ci_low <= res.point <= res.ci_high <= 1.0
True
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from cuffcrt._seed import GLOBAL_SEED

__all__ = ["BootstrapResult", "cluster_bootstrap_ci"]


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a subject-clustered percentile bootstrap.

    Attributes
    ----------
    point : float
        The statistic computed on the full observed data (no resampling).
    ci_low : float
        Lower percentile of the bootstrap distribution at ``confidence_level``.
    ci_high : float
        Upper percentile of the bootstrap distribution at ``confidence_level``.
    n_resamples : int
        Number of cluster-bootstrap replicates drawn.
    confidence_level : float
        Nominal coverage of the interval (for example 0.95).
    seed : int
        Seed passed to ``numpy.random.default_rng``.
    n_clusters : int
        Number of distinct clusters in the input.
    n_obs : int
        Number of observations in the input.
    """

    point: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float
    seed: int
    n_clusters: int
    n_obs: int


def _as_1d_array(name: str, values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional; got shape {arr.shape}.")
    return arr


def cluster_bootstrap_ci(
    values: ArrayLike,
    clusters: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = GLOBAL_SEED,
) -> BootstrapResult:
    """Compute a subject-clustered percentile bootstrap CI.

    Resamples clusters (typically patient/subject ids) with replacement, then
    pools all observations belonging to the resampled clusters and applies
    ``statistic`` to the pooled vector. The reported CI uses the lower and
    upper percentiles of the resulting bootstrap distribution at
    ``confidence_level`` (for the default 0.95, percentiles 2.5 and 97.5).

    Parameters
    ----------
    values : array-like of shape (n_obs,)
        Observation-level values (one per cuff cycle, for example). Numeric.
    clusters : array-like of shape (n_obs,)
        Cluster id for each observation; must be the same length as ``values``.
        Any hashable type is accepted.
    statistic : Callable[[np.ndarray], float], optional
        Reduction applied to the pooled observation vector on each resample
        and once to the full input for the point estimate. Defaults to
        :func:`numpy.mean`, which yields a proportion when ``values`` is 0/1.
    n_resamples : int, optional
        Number of cluster-bootstrap replicates. Must be at least 1. The
        pre-registered default is 10000.
    confidence_level : float, optional
        Nominal CI coverage in (0, 1). Default 0.95.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Defaults to
        :data:`cuffcrt._seed.GLOBAL_SEED`.

    Returns
    -------
    BootstrapResult
        Point estimate, lower and upper percentile CI bounds, and bookkeeping.

    Raises
    ------
    ValueError
        If ``values`` and ``clusters`` have different lengths, if either is
        empty, if ``n_resamples`` is not a positive integer, or if
        ``confidence_level`` is not strictly between 0 and 1.

    Notes
    -----
    The point estimate is computed on the original observations (not the
    bootstrap mean), so it equals what a non-bootstrap reporter would compute
    by hand. When every observation belongs to one cluster the CI collapses
    to the point value because the resample always returns the same data.
    """
    values_arr = _as_1d_array("values", values).astype(np.float64, copy=False)
    clusters_arr = _as_1d_array("clusters", clusters)

    if values_arr.shape[0] != clusters_arr.shape[0]:
        raise ValueError(
            "values and clusters must have the same length; "
            f"got {values_arr.shape[0]} and {clusters_arr.shape[0]}."
        )
    if values_arr.size == 0:
        raise ValueError("values must contain at least one observation.")
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive int; got {n_resamples!r}.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1); got {confidence_level!r}."
        )

    # Group observation indices by cluster id. Preserve first-seen order so the
    # mapping is deterministic for a given input.
    cluster_to_indices: dict[Hashable, list[int]] = {}
    for i, c in enumerate(clusters_arr.tolist()):
        cluster_to_indices.setdefault(c, []).append(i)

    unique_clusters = list(cluster_to_indices.keys())
    n_clusters = len(unique_clusters)
    n_obs = int(values_arr.shape[0])

    # Pre-build index arrays per cluster for fast pooling on each resample.
    index_lists: list[np.ndarray] = [
        np.asarray(cluster_to_indices[c], dtype=np.int64) for c in unique_clusters
    ]

    point = float(statistic(values_arr))

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        pooled = np.concatenate([index_lists[p] for p in picks])
        replicates[b] = float(statistic(values_arr[pooled]))

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low = float(np.percentile(replicates, lower_pct))
    ci_high = float(np.percentile(replicates, upper_pct))

    return BootstrapResult(
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        n_clusters=n_clusters,
        n_obs=n_obs,
    )
