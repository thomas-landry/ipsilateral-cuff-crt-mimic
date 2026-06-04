"""Tests for the subject-clustered percentile bootstrap.

Covers: determinism under fixed seed, single-cluster degeneracy, proportion
recovery on a 0/1 array, the qualitative cluster-size sanity check
(concentrated mass widens the CI), and basic API input validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.analysis import BootstrapResult, cluster_bootstrap_ci


def test_determinism_same_seed_same_ci():
    """Identical seed and inputs produce identical CIs across runs."""
    rng = np.random.default_rng(0)
    n = 200
    values = rng.integers(0, 2, size=n)
    clusters = rng.integers(0, 25, size=n)

    res1 = cluster_bootstrap_ci(values, clusters, n_resamples=2000, seed=GLOBAL_SEED)
    res2 = cluster_bootstrap_ci(values, clusters, n_resamples=2000, seed=GLOBAL_SEED)

    assert isinstance(res1, BootstrapResult)
    assert res1.point == res2.point
    assert res1.ci_low == res2.ci_low
    assert res1.ci_high == res2.ci_high


def test_determinism_different_seed_different_ci():
    """Different seeds yield a different bootstrap distribution."""
    rng = np.random.default_rng(1)
    values = rng.integers(0, 2, size=200)
    clusters = rng.integers(0, 25, size=200)

    res_a = cluster_bootstrap_ci(values, clusters, n_resamples=2000, seed=1)
    res_b = cluster_bootstrap_ci(values, clusters, n_resamples=2000, seed=2)

    # Point is computed from the data, not the bootstrap, so it is identical.
    assert res_a.point == res_b.point
    # The percentile bounds should differ under different seeds for this size.
    assert (res_a.ci_low, res_a.ci_high) != (res_b.ci_low, res_b.ci_high)


def test_single_cluster_ci_collapses_to_point():
    """One-cluster input: every resample returns the same data, CI = point."""
    values = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    clusters = np.array(["only"] * 5)
    res = cluster_bootstrap_ci(values, clusters, n_resamples=500)

    assert res.n_clusters == 1
    assert res.n_obs == 5
    assert math.isclose(res.point, np.mean(values))
    assert math.isclose(res.ci_low, res.point)
    assert math.isclose(res.ci_high, res.point)


def test_proportion_statistic_point_equals_observed_mean():
    """The point estimate from statistic=np.mean on 0/1 equals the proportion."""
    values = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    clusters = np.array(["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"])
    res = cluster_bootstrap_ci(values, clusters, n_resamples=1000)
    assert math.isclose(res.point, float(np.mean(values)))
    assert res.ci_low <= res.point <= res.ci_high


def test_cluster_size_concentration_widens_ci():
    """One giant cluster + singletons -> wider CI than evenly-spread clusters.

    Both datasets share the same observation vector (so the point estimate is
    identical), but the concentrated layout has fewer effective independent
    clusters, so the resample distribution is wider.
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    n_obs = 100
    values = rng.integers(0, 2, size=n_obs).astype(np.float64)

    # Layout A: one giant cluster holds the first 80 obs; remaining 20 are
    # singletons. Resampling the giant cluster reuses 80 correlated obs at a
    # time, so the bootstrap distribution is wide.
    big_id = "big"
    clusters_concentrated = np.array(
        [big_id] * 80 + [f"single_{i}" for i in range(20)]
    )

    # Layout B: 20 evenly-sized clusters of 5 obs each. Each resample mixes
    # many independent groups, so the bootstrap distribution is narrower.
    clusters_even = np.array([f"c{i // 5}" for i in range(n_obs)])

    res_conc = cluster_bootstrap_ci(
        values, clusters_concentrated, n_resamples=4000, seed=GLOBAL_SEED
    )
    res_even = cluster_bootstrap_ci(
        values, clusters_even, n_resamples=4000, seed=GLOBAL_SEED
    )

    width_conc = res_conc.ci_high - res_conc.ci_low
    width_even = res_even.ci_high - res_even.ci_low
    assert math.isclose(res_conc.point, res_even.point)
    assert width_conc > width_even


def test_rejects_mismatched_lengths():
    """API: values and clusters must have the same length."""
    with pytest.raises(ValueError, match="same length"):
        cluster_bootstrap_ci(
            np.array([0, 1, 1]), np.array(["a", "b"]), n_resamples=10
        )


def test_rejects_empty_input():
    """API: empty inputs are rejected."""
    with pytest.raises(ValueError, match="at least one observation"):
        cluster_bootstrap_ci(np.array([]), np.array([]), n_resamples=10)


def test_rejects_bad_n_resamples():
    """API: n_resamples must be a positive int."""
    with pytest.raises(ValueError, match="n_resamples"):
        cluster_bootstrap_ci(
            np.array([0, 1]), np.array(["a", "b"]), n_resamples=0
        )


def test_rejects_bad_confidence_level():
    """API: confidence_level must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="confidence_level"):
        cluster_bootstrap_ci(
            np.array([0, 1]),
            np.array(["a", "b"]),
            n_resamples=10,
            confidence_level=1.5,
        )


def test_accepts_list_inputs():
    """API: plain lists also work, not just numpy arrays."""
    res = cluster_bootstrap_ci(
        [0, 1, 1, 0, 1, 0],
        ["s1", "s1", "s2", "s2", "s3", "s3"],
        n_resamples=500,
    )
    assert res.n_clusters == 3
    assert res.n_obs == 6
