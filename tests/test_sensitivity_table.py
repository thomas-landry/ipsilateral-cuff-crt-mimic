"""Tests for ``scripts/56_sensitivity_table.py``.

The script's whole purpose is to refuse to render unless the result artifacts
still carry the recorded numbers, so the tests exercise that contract: the
loaders accept the real (verified) artifacts and reproduce the headline values,
and they raise when an artifact is perturbed away from its anchor. The
digit-prefixed script is loaded by path through ``importlib.util``.

Verified anchors (from the build workbooks and the source CSV/JSON files):

* Determinism: 100% paired agreement over 100 cards.
* Prompt sensitivity (gallery-render reference): concordance 66.0 / 69.5 / 52.0;
  positive rate 84 / 70.5 / 79.
* Render sensitivity (independent-render reference): concordance 38 / 42 / 44.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO / "scripts" / "56_sensitivity_table.py"
_spec = importlib.util.spec_from_file_location("_sens56", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_sens = importlib.util.module_from_spec(_spec)
sys.modules["_sens56"] = _sens
_spec.loader.exec_module(_sens)

_DETERMINISM_CSV = _REPO / "results/medgemma_determinism/agreement_summary.csv"
_PROMPT_DIR = _REPO / "results/medgemma_prompt_sensitivity"

_have_real = (
    _DETERMINISM_CSV.exists()
    and (_PROMPT_DIR / "concordance_vs_galleryrender" / "concordance_summary.csv").exists()
    and (_PROMPT_DIR / "concordance_vs_headline" / "concordance_summary.csv").exists()
)
real_only = pytest.mark.skipif(not _have_real, reason="result artifacts not in checkout")


@real_only
def test_determinism_matches_source_and_anchor() -> None:
    """The D6 summary loads and reports 100% agreement over 100 paired cards."""
    det = _sens.load_determinism(_DETERMINISM_CSV)
    assert det["agreement_pct"] == 100.0
    assert int(det["n_paired"]) == 100
    assert int(det["n_agree"]) == 100
    assert det["parse_fail_rerun_pct"] == 0.0


@real_only
def test_prompt_and_render_match_source_and_anchors() -> None:
    """Prompt + render concordance reproduce the verified per-variant numbers."""
    prompt_bundle, render_bundle = _sens.load_prompt_and_render(_PROMPT_DIR)

    pv = prompt_bundle.by_variant
    assert round(float(pv["v_compact"]["concordance_pct"]), 1) == 66.0
    assert round(float(pv["v_explicit"]["concordance_pct"]), 1) == 69.5
    assert round(float(pv["v_terse_criteria"]["concordance_pct"]), 1) == 52.0
    assert round(float(pv["v_compact"]["var_positive_rate_pct"]), 0) == 84.0
    assert round(float(pv["v_explicit"]["var_positive_rate_pct"]), 1) == 70.5
    assert round(float(pv["v_terse_criteria"]["var_positive_rate_pct"]), 0) == 79.0
    # v_explicit parsed 95 of 100 (5 parse failures).
    assert int(pv["v_explicit"]["n_compare"]) == 95
    assert int(pv["v_explicit"]["var_parse_failure"]) == 5

    rv = render_bundle.by_variant
    assert round(float(rv["v_compact"]["concordance_pct"]), 0) == 38.0
    assert round(float(rv["v_explicit"]["concordance_pct"]), 0) == 42.0
    assert round(float(rv["v_terse_criteria"]["concordance_pct"]), 0) == 44.0

    # Role assignment confirmed via reference positive rate.
    assert abs(prompt_bundle.ref_posrate - 48.42) < 1.0
    assert abs(render_bundle.ref_posrate - 28.35) < 1.0


@real_only
def test_rendered_rows_carry_the_verified_strings() -> None:
    """The assembled table rows embed the verified numbers as displayed."""
    det = _sens.load_determinism(_DETERMINISM_CSV)
    prompt_bundle, render_bundle = _sens.load_prompt_and_render(_PROMPT_DIR)
    rows = _sens.build_rows(det, prompt_bundle, render_bundle)

    by_analysis = {r.analysis.split("\n")[0]: r for r in rows}
    assert "100% paired agreement (100/100)" in by_analysis["Determinism"].result

    prompt_result = by_analysis["Prompt sensitivity"].result
    assert "compact 66.0%" in prompt_result
    assert "explicit 69.5%" in prompt_result
    assert "terse 52.0%" in prompt_result
    assert "compact 84%" in prompt_result

    render_result = by_analysis["Render sensitivity"].result
    assert "compact 38.0%" in render_result
    assert "explicit 42.1%" in render_result
    assert "terse 44.0%" in render_result


def test_drifted_determinism_value_is_rejected(tmp_path: Path) -> None:
    """A perturbed agreement value fails the anchor check loudly."""
    bad = tmp_path / "agreement_summary.csv"
    pl.DataFrame(
        {
            "metric": [
                "overall_agreement_pct",
                "n_paired",
                "n_agree",
                "parse_failure_rate_rerun_pct",
            ],
            "value": [88.0, 100.0, 88.0, 0.0],
        }
    ).write_csv(bad)
    with pytest.raises(ValueError, match="determinism agreement"):
        _sens.load_determinism(bad)


def test_drifted_prompt_concordance_is_rejected(tmp_path: Path) -> None:
    """A perturbed prompt concordance value fails the anchor check loudly."""
    gdir = tmp_path / "concordance_vs_galleryrender"
    hdir = tmp_path / "concordance_vs_headline"
    gdir.mkdir()
    hdir.mkdir()

    def _summary(conc: dict[str, float], posrate: dict[str, float]) -> pl.DataFrame:
        variants = list(conc)
        return pl.DataFrame(
            {
                "variant": variants,
                "n_total": [100] * len(variants),
                "n_compare": [100, 95, 100],
                "n_matched": [int(conc[v]) for v in variants],
                "concordance_pct": [conc[v] for v in variants],
                "var_positive_rate_pct": [posrate[v] for v in variants],
                "var_parse_failure": [0, 5, 0],
            }
        )

    variants = ["v_compact", "v_explicit", "v_terse_criteria"]
    # Gallery-render dir with a DRIFTED v_compact concordance (66 -> 50).
    _summary(
        {"v_compact": 50.0, "v_explicit": 69.5, "v_terse_criteria": 52.0},
        {"v_compact": 84.0, "v_explicit": 70.5, "v_terse_criteria": 79.0},
    ).write_csv(gdir / "concordance_summary.csv")
    (gdir / "concordance.json").write_text(
        json.dumps({"canonical_in_subsample": {"positive_rate_pct": 48.42}})
    )
    _summary(
        {"v_compact": 38.0, "v_explicit": 42.0, "v_terse_criteria": 44.0},
        {"v_compact": 84.0, "v_explicit": 70.5, "v_terse_criteria": 79.0},
    ).write_csv(hdir / "concordance_summary.csv")
    (hdir / "concordance.json").write_text(
        json.dumps({"canonical_in_subsample": {"positive_rate_pct": 28.35}})
    )

    assert variants  # silence unused in some linters
    with pytest.raises(ValueError, match="prompt concordance"):
        _sens.load_prompt_and_render(tmp_path)


def test_swapped_reference_posrate_is_rejected(tmp_path: Path) -> None:
    """If the reference positive rates do not identify the roles, refuse."""
    gdir = tmp_path / "concordance_vs_galleryrender"
    hdir = tmp_path / "concordance_vs_headline"
    gdir.mkdir()
    hdir.mkdir()
    sane = pl.DataFrame(
        {
            "variant": ["v_compact", "v_explicit", "v_terse_criteria"],
            "n_total": [100, 100, 100],
            "n_compare": [100, 95, 100],
            "n_matched": [66, 66, 52],
            "concordance_pct": [66.0, 69.5, 52.0],
            "var_positive_rate_pct": [84.0, 70.5, 79.0],
            "var_parse_failure": [0, 5, 0],
        }
    )
    sane.write_csv(gdir / "concordance_summary.csv")
    sane.write_csv(hdir / "concordance_summary.csv")
    # Both directories report the WRONG reference positive rate.
    (gdir / "concordance.json").write_text(
        json.dumps({"canonical_in_subsample": {"positive_rate_pct": 10.0}})
    )
    (hdir / "concordance.json").write_text(
        json.dumps({"canonical_in_subsample": {"positive_rate_pct": 10.0}})
    )
    with pytest.raises(ValueError, match="role assignment unsafe"):
        _sens.load_prompt_and_render(tmp_path)
