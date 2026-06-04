"""Tests for the MedGemma run-log aggregator.

Covers the four named cases (pure-callable, all-uncallable, mixed, empty) and
asserts the two separately-reported rates (callable rate, headline rate per
callable) match the documented definitions.
"""

from __future__ import annotations

import polars as pl

from cuffcrt.analysis.medgemma_aggregate import (
    AggregateResult,
    aggregate_medgemma_calls,
)


def _df(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal run-log DataFrame from a list of rows."""
    return pl.DataFrame(
        rows,
        schema={"parsed_ok": pl.Boolean, "call": pl.Utf8},
    )


def test_aggregate_empty_dataframe_returns_none_rates():
    df = pl.DataFrame(
        {"parsed_ok": [], "call": []},
        schema={"parsed_ok": pl.Boolean, "call": pl.Utf8},
    )
    result = aggregate_medgemma_calls(df)
    assert isinstance(result, AggregateResult)
    assert result.n_total == 0
    assert result.n_callable == 0
    assert result.n_uncallable == 0
    assert result.n_occlusion_signature_present == 0
    assert result.callable_rate is None
    assert result.uncallable_rate is None
    assert result.headline_rate_per_callable is None


def test_aggregate_pure_callable_no_uncallable():
    """All rows are callable; 3/4 are positives."""
    rows = [
        {"parsed_ok": True, "call": "occlusion_signature_present"},
        {"parsed_ok": True, "call": "occlusion_signature_present"},
        {"parsed_ok": True, "call": "occlusion_signature_present"},
        {"parsed_ok": True, "call": "no_occlusion_signature"},
    ]
    result = aggregate_medgemma_calls(_df(rows))
    assert result.n_total == 4
    assert result.n_callable == 4
    assert result.n_uncallable == 0
    assert result.n_occlusion_signature_present == 3
    assert result.callable_rate == 1.0
    assert result.uncallable_rate == 0.0
    assert result.headline_rate_per_callable == 0.75


def test_aggregate_all_uncallable():
    """Every row is either indeterminate or a parse failure."""
    rows = [
        {"parsed_ok": True, "call": "indeterminate"},
        {"parsed_ok": True, "call": "indeterminate"},
        {"parsed_ok": False, "call": None},
        {"parsed_ok": False, "call": None},
    ]
    result = aggregate_medgemma_calls(_df(rows))
    assert result.n_total == 4
    assert result.n_callable == 0
    assert result.n_uncallable == 4
    assert result.n_occlusion_signature_present == 0
    assert result.callable_rate == 0.0
    assert result.uncallable_rate == 1.0
    # Zero denominator on the headline rate -> None, not a ZeroDivisionError.
    assert result.headline_rate_per_callable is None


def test_aggregate_mixed_case():
    """Mixed: 2 present, 3 absent, 2 indeterminate, 1 parse failure (n=8)."""
    rows = [
        {"parsed_ok": True, "call": "occlusion_signature_present"},
        {"parsed_ok": True, "call": "occlusion_signature_present"},
        {"parsed_ok": True, "call": "no_occlusion_signature"},
        {"parsed_ok": True, "call": "no_occlusion_signature"},
        {"parsed_ok": True, "call": "no_occlusion_signature"},
        {"parsed_ok": True, "call": "indeterminate"},
        {"parsed_ok": True, "call": "indeterminate"},
        {"parsed_ok": False, "call": None},
    ]
    result = aggregate_medgemma_calls(_df(rows))
    assert result.n_total == 8
    assert result.n_parsed == 7
    assert result.n_callable == 5  # 2 + 3
    assert result.n_uncallable == 3  # 2 indeterminate + 1 parse failure
    assert result.n_occlusion_signature_present == 2
    assert result.callable_rate == 5 / 8
    assert result.uncallable_rate == 3 / 8
    assert result.headline_rate_per_callable == 2 / 5


def test_aggregate_legacy_calls_are_not_counted_as_present():
    """A row carrying the pre-canonicalization vocabulary is not callable.

    In practice parsing already rejects legacy values, but the aggregator must
    not treat them as positives even if a stale row somehow appears.
    """
    rows = [
        {"parsed_ok": True, "call": "ipsilateral"},  # legacy vocab, not present
        {"parsed_ok": True, "call": "occlusion_signature_present"},
    ]
    result = aggregate_medgemma_calls(_df(rows))
    assert result.n_occlusion_signature_present == 1
    assert result.n_callable == 1


def test_aggregate_parsed_ok_coerced_from_int_column():
    """``parsed_ok`` may arrive from CSV as 0/1; the aggregator coerces it."""
    df = pl.DataFrame(
        {
            "parsed_ok": [1, 1, 0],
            "call": ["occlusion_signature_present", "no_occlusion_signature", None],
        }
    )
    result = aggregate_medgemma_calls(df)
    assert result.n_callable == 2
    assert result.n_uncallable == 1
    assert result.n_occlusion_signature_present == 1
