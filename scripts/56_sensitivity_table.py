"""MedGemma robustness summary table (step 56).

Assembles a compact, summary of the three MedGemma robustness analyses: 
determinism (D6), prompt sensitivity (D5), and render sensitivity. The script 
reads them from the result artifacts
(`results/medgemma_determinism/agreement_summary.csv` and the two
`results/medgemma_prompt_sensitivity/concordance_vs_*/concordance_summary.csv`
files), verifies each value against an expected anchor drawn from the build
workbooks, and fails loudly if an artifact has drifted from the recorded result.

It writes both a machine-readable CSV (`figures/table_sensitivity_summary.csv`)
and a clean vector + high-DPI table image (`.pdf` / `.png`) rendered with the
shared `cuffcrt.figstyle` typography.

A note on the two concordance directories. Both compare the three prompt
variants (run on the pre-rendered gallery PNGs) against a MedGemma reference.
`concordance_vs_galleryrender` uses the gallery-render reference, which was
produced from the SAME pre-rendered PNGs under the canonical prompt, so it
isolates the effect of prompt wording (the images are held constant). The
`concordance_vs_headline` directory uses the independent render-on-the-fly
reference, so it mixes the prompt change with a different render path and is read
here as the render-sensitivity row. The script confirms this assignment from the
reference positive rate recorded in each ``concordance.json``.

Inputs
------
``--determinism_csv``
    ``results/medgemma_determinism/agreement_summary.csv``.
``--prompt_dir``
    ``results/medgemma_prompt_sensitivity`` (contains the two
    ``concordance_vs_*`` subdirectories).

Outputs
-------
``--out_dir/table_sensitivity_summary.csv`` and a rendered
``table_sensitivity_summary.png`` / ``.pdf`` (via :func:`cuffcrt.figstyle.save`).

Examples
--------
::

    uv run python scripts/56_sensitivity_table.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger

from cuffcrt import figstyle

DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_DETERMINISM = DEFAULT_REPO / "results/medgemma_determinism/agreement_summary.csv"
DEFAULT_PROMPT_DIR = DEFAULT_REPO / "results/medgemma_prompt_sensitivity"
DEFAULT_OUT = DEFAULT_REPO / "figures"

VARIANT_ORDER = ("v_compact", "v_explicit", "v_terse_criteria")
VARIANT_LABEL = {
    "v_compact": "compact",
    "v_explicit": "explicit",
    "v_terse_criteria": "terse",
}

# Anchors recorded in the build workbooks; the script verifies the artifacts
# against these and refuses to render if a value has drifted.
_TOL = 0.6  # percentage-point tolerance for rounding differences

EXPECTED_DETERMINISM_AGREEMENT_PCT = 100.0
EXPECTED_DETERMINISM_N = 100
# Prompt sensitivity is the gallery-render reference (same images, prompt varies).
EXPECTED_PROMPT_CONCORDANCE = {"v_compact": 66.0, "v_explicit": 69.5, "v_terse_criteria": 52.0}
EXPECTED_PROMPT_POSRATE = {"v_compact": 84.0, "v_explicit": 70.5, "v_terse_criteria": 79.0}
# Render sensitivity is the independent-render (headline) reference.
EXPECTED_RENDER_CONCORDANCE_RANGE = (38.0, 44.0)
# Reference positive rates (over the full 568) that identify which directory is
# which: the gallery-render reference sits near 48%, the headline reference near
# 28%. Used only to confirm the directory-to-role assignment.
_GALLERYRENDER_REF_POSRATE = 48.42
_HEADLINE_REF_POSRATE = 28.35


@dataclass(frozen=True)
class SensitivityRow:
    """One row of the rendered summary table."""

    analysis: str
    n_cards: str
    setup: str
    result: str
    takeaway: str


@dataclass(frozen=True)
class ConcordanceBundle:
    """One concordance directory: per-variant rows plus reference positive rate."""

    by_variant: dict[str, dict[str, object]]
    ref_posrate: float


def _approx(a: float, b: float, tol: float = _TOL) -> bool:
    """Return True if ``a`` and ``b`` agree within ``tol`` percentage points."""
    return math.isclose(a, b, abs_tol=tol)


def _cell(row: dict[str, object], key: str) -> float:
    """Return one concordance-summary cell as a float.

    ``polars.DataFrame.to_dicts`` yields values typed ``object``; this narrows a
    single numeric cell so call sites stay type-clean.
    """
    return float(row[key])  # type: ignore[arg-type]


def load_determinism(determinism_csv: Path) -> dict[str, float]:
    """Read the D6 agreement summary and verify the headline scalars.

    Parameters
    ----------
    determinism_csv : pathlib.Path
        ``agreement_summary.csv`` from ``scripts/43``.

    Returns
    -------
    dict[str, float]
        ``{"agreement_pct", "n_paired", "n_agree", "parse_fail_rerun_pct"}``.

    Raises
    ------
    ValueError
        If the recorded agreement or paired count disagrees with the workbook
        anchor.
    """
    df = pl.read_csv(determinism_csv)
    vals = {r["metric"]: float(r["value"]) for r in df.to_dicts()}
    agreement = vals["overall_agreement_pct"]
    n_paired = vals["n_paired"]
    n_agree = vals["n_agree"]
    parse_fail = vals["parse_failure_rate_rerun_pct"]
    if not _approx(agreement, EXPECTED_DETERMINISM_AGREEMENT_PCT):
        raise ValueError(
            f"determinism agreement {agreement} != expected "
            f"{EXPECTED_DETERMINISM_AGREEMENT_PCT}"
        )
    if int(n_paired) != EXPECTED_DETERMINISM_N:
        raise ValueError(
            f"determinism n_paired {int(n_paired)} != expected {EXPECTED_DETERMINISM_N}"
        )
    return {
        "agreement_pct": agreement,
        "n_paired": n_paired,
        "n_agree": n_agree,
        "parse_fail_rerun_pct": parse_fail,
    }


def _load_concordance(summary_csv: Path, json_path: Path) -> ConcordanceBundle:
    """Load one concordance directory: per-variant rows + reference positive rate.

    Returns
    -------
    ConcordanceBundle
        The per-variant concordance rows keyed by variant id and the reference
        positive rate read from ``concordance.json``.
    """
    df = pl.read_csv(summary_csv)
    by_variant = {str(r["variant"]): r for r in df.to_dicts()}
    ref_posrate = float("nan")
    if json_path.exists():
        meta = json.loads(json_path.read_text())
        ref = meta.get("canonical_in_subsample", {})
        ref_posrate = float(ref.get("positive_rate_pct", float("nan")))
    return ConcordanceBundle(by_variant=by_variant, ref_posrate=ref_posrate)


def load_prompt_and_render(
    prompt_dir: Path,
) -> tuple[ConcordanceBundle, ConcordanceBundle]:
    """Load the prompt-sensitivity and render-sensitivity concordance bundles.

    The two ``concordance_vs_*`` directories are assigned to roles by their
    reference positive rate (recorded in ``concordance.json``): the higher rate
    (~48%) is the gallery-render reference and drives the prompt-sensitivity row;
    the lower rate (~28%) is the independent-render reference and drives the
    render-sensitivity row. Each role's headline concordance figures are then
    verified against the workbook anchors.

    Parameters
    ----------
    prompt_dir : pathlib.Path
        ``results/medgemma_prompt_sensitivity``.

    Returns
    -------
    tuple[ConcordanceBundle, ConcordanceBundle]
        ``(prompt_bundle, render_bundle)``.

    Raises
    ------
    ValueError
        If a directory is missing, the role assignment is ambiguous, or a
        verified concordance value drifts from its anchor.
    """
    gallery = _load_concordance(
        prompt_dir / "concordance_vs_galleryrender" / "concordance_summary.csv",
        prompt_dir / "concordance_vs_galleryrender" / "concordance.json",
    )
    headline = _load_concordance(
        prompt_dir / "concordance_vs_headline" / "concordance_summary.csv",
        prompt_dir / "concordance_vs_headline" / "concordance.json",
    )

    # Confirm the role assignment from the reference positive rates.
    if not _approx(gallery.ref_posrate, _GALLERYRENDER_REF_POSRATE, tol=1.0):
        raise ValueError(
            "concordance_vs_galleryrender reference positive rate "
            f"{gallery.ref_posrate} does not match the gallery-render "
            f"reference (~{_GALLERYRENDER_REF_POSRATE}); role assignment unsafe"
        )
    if not _approx(headline.ref_posrate, _HEADLINE_REF_POSRATE, tol=1.0):
        raise ValueError(
            "concordance_vs_headline reference positive rate "
            f"{headline.ref_posrate} does not match the headline reference "
            f"(~{_HEADLINE_REF_POSRATE}); role assignment unsafe"
        )

    prompt_bundle = gallery
    render_bundle = headline

    # Verify prompt-sensitivity concordance + positive rates against anchors.
    pv = prompt_bundle.by_variant
    for variant, expected in EXPECTED_PROMPT_CONCORDANCE.items():
        got = _cell(pv[variant], "concordance_pct")
        if not _approx(got, expected):
            raise ValueError(
                f"prompt concordance {variant}={got} != expected {expected}"
            )
    for variant, expected in EXPECTED_PROMPT_POSRATE.items():
        got = _cell(pv[variant], "var_positive_rate_pct")
        if not _approx(got, expected):
            raise ValueError(
                f"prompt positive rate {variant}={got} != expected {expected}"
            )

    # Verify render-sensitivity concordance falls in the recorded band.
    rv = render_bundle.by_variant
    lo, hi = EXPECTED_RENDER_CONCORDANCE_RANGE
    for variant in VARIANT_ORDER:
        got = _cell(rv[variant], "concordance_pct")
        if not (lo - _TOL <= got <= hi + _TOL):
            raise ValueError(
                f"render concordance {variant}={got} outside expected band "
                f"[{lo}, {hi}]"
            )

    return prompt_bundle, render_bundle


def build_rows(
    determinism: dict[str, float],
    prompt_bundle: ConcordanceBundle,
    render_bundle: ConcordanceBundle,
) -> list[SensitivityRow]:
    """Assemble the table rows from the verified artifacts.

    Parameters
    ----------
    determinism : dict
        Output of :func:`load_determinism`.
    prompt_bundle, render_bundle : dict
        Outputs of :func:`load_prompt_and_render`.

    Returns
    -------
    list[SensitivityRow]
        Three rows: determinism, prompt sensitivity, render sensitivity.
    """
    pv = prompt_bundle.by_variant
    rv = render_bundle.by_variant

    n_agree = int(determinism["n_agree"])
    n_paired = int(determinism["n_paired"])

    prompt_conc = ", ".join(
        f"{VARIANT_LABEL[v]} {_cell(pv[v], 'concordance_pct'):.1f}%"
        for v in VARIANT_ORDER
    )
    prompt_pos = ", ".join(
        f"{VARIANT_LABEL[v]} {_cell(pv[v], 'var_positive_rate_pct'):.0f}%"
        for v in VARIANT_ORDER
    )
    render_conc = ", ".join(
        f"{VARIANT_LABEL[v]} {_cell(rv[v], 'concordance_pct'):.1f}%"
        for v in VARIANT_ORDER
    )

    determinism_row = SensitivityRow(
        analysis="Determinism\n(fresh-process re-run)",
        n_cards="100",
        setup=(
            "Re-run on a restarted server; greedy decoding\n"
            "(temperature 0, fixed seed). Genuine ~5 s per card\n"
            "decodes, not a cache replay."
        ),
        result=f"{determinism['agreement_pct']:.0f}% paired agreement ({n_agree}/{n_paired})",
        takeaway="Calls reproduce exactly\nacross a restart.",
    )
    prompt_row = SensitivityRow(
        analysis="Prompt sensitivity\n(3 reworded prompts)",
        n_cards="100",
        setup=(
            "Three reworded system prompts on the same\n"
            "gallery images, vs the canonical-prompt reference\n"
            "on those images."
        ),
        result=(
            f"Concordance: {prompt_conc}.\n"
            f"Positive rate: {prompt_pos}\n"
            f"(reference {_galleryref_posrate_str(prompt_bundle)})."
        ),
        takeaway="Calls shift with wording;\nvariants over-call.",
    )
    render_row = SensitivityRow(
        analysis="Render sensitivity\n(prompt + render path)",
        n_cards="100",
        setup=(
            "Same three variants on gallery images, vs the\n"
            "independent render-on-the-fly reference\n"
            "(prompt and render path both differ)."
        ),
        result=f"Concordance: {render_conc}.",
        takeaway="Render path adds a second\nlarge source of disagreement.",
    )
    return [determinism_row, prompt_row, render_row]


def _galleryref_posrate_str(prompt_bundle: ConcordanceBundle) -> str:
    """Format the gallery-render reference positive rate for the table cell."""
    return f"{prompt_bundle.ref_posrate:.0f}% positive over 568"


_TABLE_COLUMNS = ("Analysis", "Cards", "Setup", "Result", "Takeaway")
_COL_WIDTHS = (0.16, 0.05, 0.29, 0.275, 0.225)


def build_figure(rows: list[SensitivityRow]) -> plt.Figure:
    """Render the summary table as a clean matplotlib table figure.

    Parameters
    ----------
    rows : list[SensitivityRow]
        The verified table rows.

    Returns
    -------
    matplotlib.figure.Figure
        A single-axes table figure (no canvas title).
    """
    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(12.0, 3.7))
    ax.axis("off")

    cell_text = [
        [r.analysis, r.n_cards, r.setup, r.result, r.takeaway] for r in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=_TABLE_COLUMNS,
        colWidths=list(_COL_WIDTHS),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 2.3)

    n_rows = len(rows)
    for (ri, ci), cell in table.get_celld().items():
        cell.set_edgecolor(figstyle.MIST)
        cell.set_linewidth(0.7)
        cell.PAD = 0.03
        if ri == 0:
            # Header row.
            cell.set_facecolor(figstyle.INK)
            cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)
            cell.set_height(cell.get_height() * 0.7)
        else:
            cell.set_facecolor("white" if ri % 2 == 1 else figstyle.PANEL_BG)
            cell.set_text_props(color=figstyle.INK, va="center")
            if ci == 0:
                cell.set_text_props(
                    color=figstyle.INK, fontweight="bold", va="center"
                )
            if ci == 1:
                cell.set_text_props(color=figstyle.INK, ha="center", va="center")
        if ri == n_rows:  # last data row: nothing special, kept for clarity
            pass

    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
    return fig


def run(
    *,
    determinism_csv: Path,
    prompt_dir: Path,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Verify, assemble, and write the CSV plus the rendered table image.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        ``(csv_path, png, pdf)``.
    """
    determinism = load_determinism(determinism_csv)
    prompt_bundle, render_bundle = load_prompt_and_render(prompt_dir)
    rows = build_rows(determinism, prompt_bundle, render_bundle)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table_sensitivity_summary.csv"
    # Flatten newlines for the machine-readable CSV.
    pl.DataFrame(
        {
            "analysis": [r.analysis.replace("\n", " ") for r in rows],
            "n_cards": [r.n_cards for r in rows],
            "setup": [r.setup.replace("\n", " ") for r in rows],
            "result": [r.result.replace("\n", " ") for r in rows],
            "takeaway": [r.takeaway.replace("\n", " ") for r in rows],
        }
    ).write_csv(csv_path)

    fig = build_figure(rows)
    png, pdf = figstyle.save(fig, out_dir, "table_sensitivity_summary")
    plt.close(fig)
    logger.info("wrote table CSV {}", csv_path)
    logger.info("wrote table image {} and {}", png, pdf)
    return csv_path, png, pdf


def _build_argparser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Assemble and render the MedGemma robustness summary table "
            "(determinism, prompt sensitivity, render sensitivity)."
        )
    )
    p.add_argument("--determinism_csv", type=Path, default=DEFAULT_DETERMINISM)
    p.add_argument("--prompt_dir", type=Path, default=DEFAULT_PROMPT_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_argparser().parse_args(argv)
    run(
        determinism_csv=args.determinism_csv,
        prompt_dir=args.prompt_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
