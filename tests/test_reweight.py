"""Tests for population inverse-probability reweighting of precision/recall.

The reweighting math is checked against hand-computed values on a tiny synthetic
fixture with known stratum weights, and the partition-coverage accounting is
checked independently. These tests do not shell out; they import the importable
logic from :mod:`cuffcrt.analysis.reweight`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.analysis.reweight import (
    INDETERMINATE_CALL,
    NEGATIVE_CALL,
    PARSE_FAILURE_CALL,
    POSITIVE_CALL,
    StratumSpec,
    assign_card_weights,
    metric_indicators,
    reconcile_partition,
    weighted_metric,
    weighted_metric_with_ci,
)

POS = POSITIVE_CALL
NEG = NEGATIVE_CALL
IND = INDETERMINATE_CALL
PF = PARSE_FAILURE_CALL


# --------------------------------------------------------------------------- #
# StratumSpec weights
# --------------------------------------------------------------------------- #
def test_stratum_weight_is_universe_over_sampled() -> None:
    assert StratumSpec("pos", universe=268, sampled=268).weight == pytest.approx(1.0)
    assert StratumSpec("nm", universe=320, sampled=200).weight == pytest.approx(1.6)
    assert StratumSpec("neg", universe=8107, sampled=100).weight == pytest.approx(81.07)


def test_stratum_weight_rejects_oversampling() -> None:
    with pytest.raises(ValueError, match="cannot sample more"):
        _ = StratumSpec("bad", universe=5, sampled=10).weight


def test_stratum_weight_rejects_zero_sampled() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        _ = StratumSpec("bad", universe=5, sampled=0).weight


# --------------------------------------------------------------------------- #
# assign_card_weights
# --------------------------------------------------------------------------- #
def test_assign_card_weights_maps_each_card() -> None:
    strata = [
        StratumSpec("pos", 268, 268),
        StratumSpec("nm", 320, 200),
        StratumSpec("neg", 8107, 100),
    ]
    cards = ["pos", "neg", "nm", "pos"]
    w = assign_card_weights(cards, strata)
    np.testing.assert_allclose(w, [1.0, 81.07, 1.6, 1.0])


def test_assign_card_weights_unknown_stratum_errors() -> None:
    with pytest.raises(ValueError, match="no StratumSpec"):
        assign_card_weights(["ghost"], [StratumSpec("pos", 1, 1)])


# --------------------------------------------------------------------------- #
# metric_indicators
# --------------------------------------------------------------------------- #
def test_metric_indicators_precision() -> None:
    reader = [POS, NEG, POS, NEG, IND]
    machine = [POS, POS, NEG, NEG, POS]
    num, den = metric_indicators(reader, machine, "precision")
    # precision denom = machine positive AND reader callable: rows 0,1
    # (row 4 has reader indeterminate so it drops).
    np.testing.assert_allclose(den, [1, 1, 0, 0, 0])
    # num = those where reader positive: row 0 only.
    np.testing.assert_allclose(num, [1, 0, 0, 0, 0])


def test_metric_indicators_recall() -> None:
    reader = [POS, POS, POS, NEG, IND]
    machine = [POS, NEG, IND, POS, POS]
    num, den = metric_indicators(reader, machine, "recall")
    # recall denom = reader positive AND machine callable: rows 0,1
    # (row 2 machine indeterminate drops).
    np.testing.assert_allclose(den, [1, 1, 0, 0, 0])
    np.testing.assert_allclose(num, [1, 0, 0, 0, 0])


def test_metric_indicators_specificity() -> None:
    reader = [NEG, NEG, NEG, POS, NEG]
    machine = [NEG, POS, PF, NEG, IND]
    num, den = metric_indicators(reader, machine, "specificity")
    # spec denom = reader negative AND machine callable: rows 0,1
    # (row 2 parse failure, row 4 indeterminate both drop).
    np.testing.assert_allclose(den, [1, 1, 0, 0, 0])
    np.testing.assert_allclose(num, [1, 0, 0, 0, 0])


def test_metric_indicators_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        metric_indicators([POS], [POS], "f1")


def test_metric_indicators_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        metric_indicators([POS, NEG], [POS], "recall")


# --------------------------------------------------------------------------- #
# weighted_metric: hand-computed reference values
# --------------------------------------------------------------------------- #
def test_weighted_metric_equals_unweighted_when_all_weights_one() -> None:
    num = np.array([1.0, 0.0, 1.0, 0.0])
    den = np.array([1.0, 1.0, 1.0, 1.0])
    w = np.ones(4)
    assert weighted_metric(num, den, w) == pytest.approx(0.5)


def test_weighted_metric_hand_computed() -> None:
    # Two strata. Stratum A weight 1 (3 cards), stratum B weight 10 (2 cards).
    # cards:        A1   A2   A3   B1   B2
    num = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    den = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    w = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
    # weighted num = 1+0+1 + 10 + 0 = 12 ; weighted den = 1+1+1+10+10 = 23
    assert weighted_metric(num, den, w) == pytest.approx(12.0 / 23.0)
    # contrast: unweighted = 3/5 = 0.6
    assert weighted_metric(num, den, np.ones(5)) == pytest.approx(3.0 / 5.0)


def test_weighted_metric_zero_denominator_is_nan() -> None:
    assert math.isnan(weighted_metric([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]))


def test_weighted_metric_length_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        weighted_metric([1.0], [1.0, 1.0], [1.0])


def test_weighted_metric_recall_full_pipeline_hand_computed() -> None:
    # Realistic mini-gallery, three strata, recall (machine+ | reader+).
    # Stratum weights: pos=1, near=2, neg=20.
    # reader-positive cards (the recall denominator universe):
    #   pos stratum:  2 cards, machine calls [POS, NEG]      -> 1 TP
    #   near stratum: 2 cards, machine calls [POS, POS]      -> 2 TP
    #   neg stratum:  1 card,  machine call  [NEG]           -> 0 TP
    reader = [POS, POS, POS, POS, POS]
    machine = [POS, NEG, POS, POS, NEG]
    strat = ["pos", "pos", "near", "near", "neg"]
    strata = [
        StratumSpec("pos", 4, 4),  # weight 1
        StratumSpec("near", 8, 4),  # weight 2
        StratumSpec("neg", 100, 5),  # weight 20
    ]
    w = assign_card_weights(strat, strata)
    np.testing.assert_allclose(w, [1, 1, 2, 2, 20])
    num, den = metric_indicators(reader, machine, "recall")
    np.testing.assert_allclose(den, [1, 1, 1, 1, 1])
    np.testing.assert_allclose(num, [1, 0, 1, 1, 0])
    # weighted num = 1*1 + 1*0 + 2*1 + 2*1 + 20*0 = 5
    # weighted den = 1+1+2+2+20 = 26
    assert weighted_metric(num, den, w) == pytest.approx(5.0 / 26.0)
    # raw recall = 3/5 = 0.6 ; reweighted is much lower because the neg-stratum
    # miss is upweighted 20x.
    assert weighted_metric(num, den, np.ones(5)) == pytest.approx(3.0 / 5.0)


# --------------------------------------------------------------------------- #
# weighted_metric_with_ci
# --------------------------------------------------------------------------- #
def test_weighted_metric_with_ci_point_matches_weighted_metric() -> None:
    reader = [POS, POS, POS, POS]
    machine = [POS, NEG, POS, POS]
    strat = ["a", "a", "b", "b"]
    strata = [StratumSpec("a", 2, 2), StratumSpec("b", 10, 2)]
    w = assign_card_weights(strat, strata)
    clusters = ["s1", "s1", "s2", "s3"]
    res = weighted_metric_with_ci(
        reader, machine, w, clusters, "recall", n_resamples=500, seed=GLOBAL_SEED
    )
    num, den = metric_indicators(reader, machine, "recall")
    assert res.point == pytest.approx(weighted_metric(num, den, w))
    assert res.ci_low <= res.point <= res.ci_high
    assert res.n_eligible == 4
    assert res.n_clusters == 3


def test_weighted_metric_with_ci_single_cluster_collapses() -> None:
    reader = [POS, POS]
    machine = [POS, NEG]
    w = np.array([1.0, 1.0])
    clusters = ["s1", "s1"]
    res = weighted_metric_with_ci(reader, machine, w, clusters, "recall", n_resamples=100)
    assert res.ci_low == res.point == res.ci_high
    assert res.n_clusters == 1


def test_weighted_metric_with_ci_is_deterministic() -> None:
    reader = [POS, NEG, POS, NEG, POS, NEG]
    machine = [POS, POS, NEG, NEG, POS, NEG]
    w = np.array([1.0, 5.0, 1.0, 5.0, 1.0, 5.0])
    clusters = ["s1", "s1", "s2", "s2", "s3", "s3"]
    r1 = weighted_metric_with_ci(
        reader, machine, w, clusters, "specificity", n_resamples=300, seed=GLOBAL_SEED
    )
    r2 = weighted_metric_with_ci(
        reader, machine, w, clusters, "specificity", n_resamples=300, seed=GLOBAL_SEED
    )
    assert (r1.ci_low, r1.ci_high, r1.point) == (r2.ci_low, r2.ci_high, r2.point)


def test_weighted_metric_with_ci_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="positive int"):
        weighted_metric_with_ci([POS], [POS], [1.0], ["s1"], "recall", n_resamples=0)
    with pytest.raises(ValueError, match="confidence_level"):
        weighted_metric_with_ci([POS], [POS], [1.0], ["s1"], "recall", confidence_level=1.5)
    with pytest.raises(ValueError, match="share length"):
        weighted_metric_with_ci([POS, NEG], [POS], [1.0], ["s1"], "recall")


# --------------------------------------------------------------------------- #
# Partition reconciliation
# --------------------------------------------------------------------------- #
def test_reconcile_partition_covers_and_reports_gap() -> None:
    # The canonical strata for this study.
    strata = [
        StratumSpec("detector_positive", 268, 268),
        StratumSpec("detector_rejected_near_miss", 320, 200),
        StratumSpec("detector_negative_random", 8107, 100),
    ]
    rec = reconcile_partition(
        strata,
        evaluable_population=8909,
        uncovered_label="no_aligned_occlusion + pre_pi_implausible",
    )
    assert rec.covered_universe == 268 + 320 + 8107  # 8695
    assert rec.uncovered == 8909 - 8695  # 214
    assert rec.coverage_fraction == pytest.approx(8695 / 8909)
    assert "no_aligned_occlusion" in rec.uncovered_label


def test_reconcile_partition_exact_tiling_no_gap() -> None:
    strata = [StratumSpec("a", 60, 30), StratumSpec("b", 40, 40)]
    rec = reconcile_partition(strata, evaluable_population=100)
    assert rec.uncovered == 0
    assert rec.coverage_fraction == pytest.approx(1.0)


def test_reconcile_partition_rejects_overcount() -> None:
    strata = [StratumSpec("a", 60, 30), StratumSpec("b", 60, 30)]
    with pytest.raises(ValueError, match="exceeds the evaluable population"):
        reconcile_partition(strata, evaluable_population=100)


def test_reconcile_partition_rejects_nonpositive_population() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        reconcile_partition([StratumSpec("a", 1, 1)], evaluable_population=0)


# --------------------------------------------------------------------------- #
# Integration-flavored check on the real canonical strata arithmetic
# --------------------------------------------------------------------------- #
def test_canonical_strata_weights_match_expected() -> None:
    strata = {
        s.name: s.weight
        for s in (
            StratumSpec("detector_positive", 268, 268),
            StratumSpec("detector_rejected_near_miss", 320, 200),
            StratumSpec("detector_negative_random", 8107, 100),
        )
    }
    assert strata["detector_positive"] == pytest.approx(1.0)
    assert strata["detector_rejected_near_miss"] == pytest.approx(1.6)
    assert strata["detector_negative_random"] == pytest.approx(81.07)
