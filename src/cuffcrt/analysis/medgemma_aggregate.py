"""Aggregate a MedGemma adjudication run log into the reportable rates.

The MedGemma run log is one row per candidate event, with a ``call`` in
``{"occlusion_signature_present", "no_occlusion_signature", "indeterminate"}``
when parsing succeeded and a ``parse_error`` otherwise. The downstream
manuscript reports two rates separately:

- ``headline_rate_per_callable`` =
  ``n_occlusion_signature_present / n_callable`` where ``n_callable`` is the
  count of rows with ``call in {"occlusion_signature_present",
  "no_occlusion_signature"}``. This is the rate the secondary analysis
  actually measures: among traces MedGemma could call, how often did it agree
  the occlusion signature is present.

- ``uncallable_rate`` = ``n_uncallable / n_total`` where ``n_uncallable`` =
  ``n_indeterminate + n_parse_failure``. This is the rate that quantifies how
  often the model could not deliver a usable call at all. Reporting it
  separately keeps the headline honest: a headline that quietly drops the
  uncallable rows can drift far from the underlying signal.

Both rates round to None when the relevant denominator is zero (no callable
rows, or no rows at all) so the call site can render "n/a" rather than
crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# The headline call value (case-sensitive). Stays in sync with
# ``cuffcrt.llm.medgemma.VALID_CALLS``.
OCCLUSION_SIGNATURE_PRESENT = "occlusion_signature_present"
NO_OCCLUSION_SIGNATURE = "no_occlusion_signature"
INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class AggregateResult:
    """Counts and rates from one MedGemma adjudication run log.

    Attributes
    ----------
    n_total : int
        Rows in the input log.
    n_parsed : int
        Rows with ``parsed_ok=True``.
    n_uncallable : int
        ``n_indeterminate + n_parse_failure`` (rows MedGemma could not place
        into a binary call).
    n_callable : int
        Rows with ``call`` in ``{"occlusion_signature_present",
        "no_occlusion_signature"}``.
    n_occlusion_signature_present : int
        Subset of ``n_callable`` where MedGemma agreed the signature is
        present.
    callable_rate : float or None
        ``n_callable / n_total``; ``None`` when ``n_total`` is zero.
    uncallable_rate : float or None
        ``n_uncallable / n_total``; ``None`` when ``n_total`` is zero.
    headline_rate_per_callable : float or None
        ``n_occlusion_signature_present / n_callable``; ``None`` when
        ``n_callable`` is zero.
    """

    n_total: int
    n_parsed: int
    n_uncallable: int
    n_callable: int
    n_occlusion_signature_present: int
    callable_rate: float | None
    uncallable_rate: float | None
    headline_rate_per_callable: float | None


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return ``numerator / denominator`` or ``None`` for a zero denominator."""
    if denominator <= 0:
        return None
    return numerator / denominator


def aggregate_medgemma_calls(run_log_df: pl.DataFrame) -> AggregateResult:
    """Reduce a MedGemma run-log DataFrame to the manuscript-reportable counts.

    Parameters
    ----------
    run_log_df : polars.DataFrame
        A frame with at least the columns ``parsed_ok`` and ``call``. The
        ``call`` column may contain nulls for rows that failed to parse.

    Returns
    -------
    AggregateResult
        Counts and the two separately-reported rates (callable rate and
        headline rate per callable). The uncallable class is the union of
        rows with ``call == "indeterminate"`` and rows with
        ``parsed_ok=False``; the callable class is the union of
        ``occlusion_signature_present`` and ``no_occlusion_signature``.

    Notes
    -----
    Counts come from the ``call`` and ``parsed_ok`` columns only; the function
    does not look at ``schema_complete`` or ``parse_error``. The headline
    denominator deliberately excludes ``indeterminate`` and parse-failed rows
    so the rate is interpretable as MedGemma's positive-call rate among
    interpretable traces.
    """
    n_total = int(run_log_df.height)
    if n_total == 0:
        return AggregateResult(
            n_total=0,
            n_parsed=0,
            n_uncallable=0,
            n_callable=0,
            n_occlusion_signature_present=0,
            callable_rate=None,
            uncallable_rate=None,
            headline_rate_per_callable=None,
        )

    # ``parsed_ok`` is the authoritative flag for parse success. Some downstream
    # readers may have written it as bool, others as a 0/1 int; coerce with
    # ``pl.col("parsed_ok").cast(pl.Boolean)`` to be robust to either.
    parsed_mask = pl.col("parsed_ok").cast(pl.Boolean, strict=False).fill_null(False)
    call_col = pl.col("call")

    n_parsed = int(run_log_df.filter(parsed_mask).height)
    n_parse_failure = n_total - n_parsed

    n_present = int(
        run_log_df.filter(parsed_mask & (call_col == OCCLUSION_SIGNATURE_PRESENT)).height
    )
    n_absent = int(
        run_log_df.filter(parsed_mask & (call_col == NO_OCCLUSION_SIGNATURE)).height
    )
    n_indeterminate = int(
        run_log_df.filter(parsed_mask & (call_col == INDETERMINATE)).height
    )

    n_callable = n_present + n_absent
    n_uncallable = n_indeterminate + n_parse_failure

    return AggregateResult(
        n_total=n_total,
        n_parsed=n_parsed,
        n_uncallable=n_uncallable,
        n_callable=n_callable,
        n_occlusion_signature_present=n_present,
        callable_rate=_safe_rate(n_callable, n_total),
        uncallable_rate=_safe_rate(n_uncallable, n_total),
        headline_rate_per_callable=_safe_rate(n_present, n_callable),
    )
