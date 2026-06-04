"""Inter / intra-rater agreement statistics.

Pure-NumPy implementations of percent agreement and Cohen's kappa for paired
categorical calls, plus a Landis-Koch verbal band for a kappa value. Used for
the second-pass intra-rater reliability analysis (pass 1 vs pass 2 of the same
reader on the same cards). No external stats dependency; the formulas are
standard and unit-tested against hand-checked fixtures.

Cohen's kappa
-------------
For two raters assigning each item to one of ``k`` categories, with observed
agreement ``p_o`` and chance agreement ``p_e`` (from the product of the two
raters' marginals),

    kappa = (p_o - p_e) / (1 - p_e).

Kappa is 1.0 for perfect agreement, 0.0 for agreement at the chance level, and
negative for systematic disagreement. When ``p_e == 1`` (both raters used a
single category for every item) kappa is undefined; this module returns ``1.0``
if the two raters' calls are identical in that degenerate case and ``nan``
otherwise, and the caller should report percent agreement alongside.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AgreementResult",
    "cohen_kappa",
    "landis_koch_band",
    "percent_agreement",
]


@dataclass(frozen=True)
class AgreementResult:
    """Paired-rating agreement summary.

    Attributes
    ----------
    n : int
        Number of paired items.
    percent_agreement : float
        Fraction of items where the two raters gave the same label, in [0, 1].
    cohen_kappa : float
        Cohen's kappa. ``nan`` when chance agreement is 1 and the raters are
        not identical.
    categories : tuple[str, ...]
        Sorted union of category labels observed across both raters.
    """

    n: int
    percent_agreement: float
    cohen_kappa: float
    categories: tuple[str, ...]


def _validate_pair(rater_a: Sequence[str], rater_b: Sequence[str]) -> tuple[list[str], list[str]]:
    """Coerce inputs to equal-length string lists and validate.

    Parameters
    ----------
    rater_a, rater_b : Sequence[str]
        Per-item categorical calls from each pass/rater.

    Returns
    -------
    tuple[list[str], list[str]]
        The two inputs as string lists.

    Raises
    ------
    ValueError
        If the inputs differ in length or are empty.
    """
    a = [str(x) for x in rater_a]
    b = [str(x) for x in rater_b]
    if len(a) != len(b):
        raise ValueError(f"raters must have equal length; got {len(a)} and {len(b)}.")
    if not a:
        raise ValueError("raters must contain at least one paired item.")
    return a, b


def percent_agreement(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Fraction of items where the two raters agree.

    Parameters
    ----------
    rater_a, rater_b : Sequence[str]
        Equal-length per-item categorical calls.

    Returns
    -------
    float
        Observed agreement in [0, 1].
    """
    a, b = _validate_pair(rater_a, rater_b)
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    return agree / len(a)


def cohen_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Cohen's kappa for two paired categorical raters.

    Categories are the sorted union of labels seen in either input, so a label
    used by only one rater still contributes to the chance term.

    Parameters
    ----------
    rater_a, rater_b : Sequence[str]
        Equal-length per-item categorical calls.

    Returns
    -------
    float
        Cohen's kappa. Returns ``1.0`` in the degenerate single-category case
        when the raters are identical, otherwise ``nan`` when chance agreement
        is exactly 1.
    """
    a, b = _validate_pair(rater_a, rater_b)
    cats = sorted(set(a) | set(b))
    index = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(a)

    conf = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(a, b, strict=True):
        conf[index[x], index[y]] += 1.0

    p_o = float(np.trace(conf)) / n
    row_marg = conf.sum(axis=1) / n
    col_marg = conf.sum(axis=0) / n
    p_e = float(np.dot(row_marg, col_marg))

    if np.isclose(p_e, 1.0):
        return 1.0 if a == b else float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def landis_koch_band(kappa: float) -> str:
    """Return the Landis-Koch verbal strength label for a kappa value.

    Bands (Landis and Koch, 1977): < 0 poor, 0.00-0.20 slight,
    0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial,
    0.81-1.00 almost perfect.

    Parameters
    ----------
    kappa : float
        A Cohen's kappa value.

    Returns
    -------
    str
        The verbal band, or ``"undefined"`` for a ``nan`` input.
    """
    if not np.isfinite(kappa):
        return "undefined"
    if kappa < 0.0:
        return "poor"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


def agreement_summary(rater_a: Sequence[str], rater_b: Sequence[str]) -> AgreementResult:
    """Compute percent agreement and Cohen's kappa for a paired rating.

    Parameters
    ----------
    rater_a, rater_b : Sequence[str]
        Equal-length per-item categorical calls.

    Returns
    -------
    AgreementResult
        Bundled n, percent agreement, kappa, and the observed categories.
    """
    a, b = _validate_pair(rater_a, rater_b)
    cats = tuple(sorted(set(a) | set(b)))
    return AgreementResult(
        n=len(a),
        percent_agreement=percent_agreement(a, b),
        cohen_kappa=cohen_kappa(a, b),
        categories=cats,
    )
