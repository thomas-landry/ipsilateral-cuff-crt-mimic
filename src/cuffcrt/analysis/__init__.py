"""Aggregation and feasibility-funnel analysis."""

from cuffcrt.analysis.agreement import (
    AgreementResult,
    agreement_summary,
    cohen_kappa,
    landis_koch_band,
    percent_agreement,
)
from cuffcrt.analysis.bootstrap import BootstrapResult, cluster_bootstrap_ci
from cuffcrt.analysis.card_bridge import build_card_to_rowid

__all__ = [
    "AgreementResult",
    "BootstrapResult",
    "agreement_summary",
    "build_card_to_rowid",
    "cluster_bootstrap_ci",
    "cohen_kappa",
    "landis_koch_band",
    "percent_agreement",
]
