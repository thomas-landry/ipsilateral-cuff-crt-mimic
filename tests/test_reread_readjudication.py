"""Tests for the blinded second-pass re-read staging + recompute harness.

Covers:

* Staging mapping integrity (``scripts/60_stage_blinded_reread.py``): the
  interleave shuffle is a permutation, deterministic per seed, distinct from
  pass 1, and spreads strata so no two adjacent cards share a stratum on the
  real 268/200/100 split; full stage on a synthetic gallery produces 1:1
  ``blind_id`` <-> ``card_id``, no ``card_id`` collisions, byte-identical PNG
  copies, a hidden ``_blind_map.csv``, and a ``reread.html`` that leaks no
  card_id / stratum and never references the map.
* The blind_id -> card_id -> row_id join (``scripts/61_reread_recompute.py``):
  a synthetic export de-blinds through the staging map and bridges to row_ids,
  and a missing/extra blind_id fails loud.
* Cohen's kappa on a hand-checked synthetic fixture
  (``cuffcrt.analysis.agreement``), plus the collapsed binary and the change
  direction classifier.

All fixtures are synthetic; nothing here touches the real gallery PNGs or the
canonical reader form.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.analysis import agreement_summary, cohen_kappa, landis_koch_band

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(stem: str, module_name: str):
    """Load a digit-prefixed script by path (same pattern as the other tests)."""
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS / f"{stem}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


stage60 = _load("60_stage_blinded_reread", "_stage60")
recompute61 = _load("61_reread_recompute", "_recompute61")


# ---------------------------------------------------------------------------
# Synthetic gallery fixture: 6 positive + 4 near-miss + 2 negative = 12 cards.
# ---------------------------------------------------------------------------
_STRATA_PLAN = (
    ["detector_positive"] * 6 + ["detector_rejected_near_miss"] * 4 + ["neg"] * 2
)
_PREFIX = {
    "detector_positive": "A-",
    "detector_rejected_near_miss": "B-",
    "neg": "C-",
}


@pytest.fixture
def synthetic_gallery(tmp_path: Path) -> dict[str, object]:
    """Write a tiny synthetic gallery (manifest + per-stratum PNGs)."""
    gallery_root = tmp_path / "results" / "gallery"
    rows: list[dict[str, object]] = []
    for i, stratum in enumerate(_STRATA_PLAN):
        cid = f"{_PREFIX[stratum]}{i:016x}"
        stratum_dir = gallery_root / stratum
        stratum_dir.mkdir(parents=True, exist_ok=True)
        png = stratum_dir / f"{cid}.png"
        # Distinct bytes per card so byte-identity checks are meaningful.
        png.write_bytes(b"PNGDATA-" + cid.encode("ascii"))
        rows.append(
            {
                "card_id": cid,
                "stratum": stratum,
                "subject_id": f"p{i % 3:08d}",
                "record_id": 81000000 + i,
                "t_nbp": 1000.0 + i,
                "image_path": f"results/gallery/{stratum}/{cid}.png",
                "image_sha256": "x" * 64,
                "is_occlusion_signature": stratum == "detector_positive",
                "phase3_duration_s": 20.0,
                "nadir_depth_frac": 0.1,
                "alignment_offset_s": -5.0,
                "reject_reason": "" if stratum == "detector_positive" else "no_phase2",
            }
        )
    manifest = gallery_root / "gallery_manifest.csv"
    pl.DataFrame(rows).write_csv(manifest)
    return {
        "repo_root": tmp_path,
        "gallery_root": gallery_root,
        "manifest": manifest,
        "card_ids": [r["card_id"] for r in rows],
        "strata": {r["card_id"]: r["stratum"] for r in rows},
    }


# ---------------------------------------------------------------------------
# Staging: interleave shuffle.
# ---------------------------------------------------------------------------
def test_interleave_is_permutation_and_deterministic() -> None:
    card_ids = [f"c{i}" for i in range(12)]
    strata = list(_STRATA_PLAN)
    order_a = stage60.interleave_strata(card_ids, strata, seed=stage60.REREAD_SEED)
    order_b = stage60.interleave_strata(card_ids, strata, seed=stage60.REREAD_SEED)
    assert order_a == order_b
    assert sorted(order_a) == list(range(12))


def test_interleave_differs_from_pass1_seed() -> None:
    card_ids = [f"c{i}" for i in range(12)]
    strata = list(_STRATA_PLAN)
    reread = stage60.interleave_strata(card_ids, strata, seed=stage60.REREAD_SEED)
    pass1_seed = stage60.interleave_strata(card_ids, strata, seed=GLOBAL_SEED)
    assert reread != pass1_seed
    assert stage60.REREAD_SEED != GLOBAL_SEED


def test_interleave_no_adjacent_same_stratum_on_realistic_split() -> None:
    # 268/200/100 like the real gallery: no stratum is a strict majority, so
    # the greedy spread should achieve zero same-stratum adjacencies.
    strata = (
        ["detector_positive"] * 268
        + ["detector_rejected_near_miss"] * 200
        + ["neg"] * 100
    )
    card_ids = [f"c{i}" for i in range(len(strata))]
    order = stage60.interleave_strata(card_ids, strata, seed=stage60.REREAD_SEED)
    seq = [strata[i] for i in order]
    adjacencies = sum(1 for a, b in zip(seq, seq[1:], strict=False) if a == b)
    assert adjacencies == 0


def test_interleave_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        stage60.interleave_strata(["a", "b"], ["s"], seed=1)


# ---------------------------------------------------------------------------
# Staging: full stage + verification + HTML blinding.
# ---------------------------------------------------------------------------
def test_stage_reread_full(synthetic_gallery: dict[str, object]) -> None:
    out_dir = Path(synthetic_gallery["repo_root"]) / "results" / "gallery_readjud_blind"  # type: ignore[arg-type]
    rc = stage60.stage_reread(
        manifest_path=synthetic_gallery["manifest"],  # type: ignore[arg-type]
        gallery_root=synthetic_gallery["gallery_root"],  # type: ignore[arg-type]
        out_dir=out_dir,
        seed=stage60.REREAD_SEED,
    )
    assert rc == 0

    pngs = sorted(out_dir.glob("blind_*.png"))
    assert len(pngs) == 12
    # Contiguous blind_0001..blind_0012.
    assert [p.stem for p in pngs] == [f"blind_{i:04d}" for i in range(1, 13)]

    bmap = pl.read_csv(out_dir / "_blind_map.csv")
    assert bmap.columns == ["blind_id", "card_id", "stratum"]
    assert bmap.height == 12
    # 1:1 blind_id <-> card_id, no collisions.
    assert bmap.get_column("blind_id").n_unique() == 12
    assert bmap.get_column("card_id").n_unique() == 12
    assert set(bmap.get_column("card_id").to_list()) == set(
        synthetic_gallery["card_ids"]  # type: ignore[arg-type]
    )

    # Byte-identical copies: each staged PNG matches its source card bytes.
    id_to_stratum = synthetic_gallery["strata"]
    for row in bmap.iter_rows(named=True):
        staged_bytes = (out_dir / f"{row['blind_id']}.png").read_bytes()
        expected = b"PNGDATA-" + str(row["card_id"]).encode("ascii")
        assert staged_bytes == expected
        assert id_to_stratum[row["card_id"]] == row["stratum"]  # type: ignore[index]


def test_reread_html_blinding(synthetic_gallery: dict[str, object]) -> None:
    out_dir = Path(synthetic_gallery["repo_root"]) / "results" / "gallery_readjud_blind"  # type: ignore[arg-type]
    stage60.stage_reread(
        manifest_path=synthetic_gallery["manifest"],  # type: ignore[arg-type]
        gallery_root=synthetic_gallery["gallery_root"],  # type: ignore[arg-type]
        out_dir=out_dir,
        seed=stage60.REREAD_SEED,
    )
    html = (out_dir / "reread.html").read_text(encoding="utf-8")

    # No card_id, no stratum prefix, no stratum label anywhere in the document.
    for cid in synthetic_gallery["card_ids"]:  # type: ignore[union-attr]
        assert cid not in html
    for stratum in ("detector_positive", "detector_rejected_near_miss"):
        assert stratum not in html
    # The hidden map is never referenced.
    assert "_blind_map" not in html
    # Images are loaded by relative blind_id only; no stratum dir in any src.
    assert 'elImg.src = bid + ".png"' in html
    assert "results/gallery" not in html
    # Export columns are exactly the blinded schema.
    assert "blind_id,call,confidence,notes,utc" in html
    assert "reread_pass2_export.csv" in html
    # No machine-call or pass-1 vocabulary leaks.
    assert "is_occlusion_signature" not in html
    assert "medgemma" not in html.lower()


def test_stratified_sample_proportional_and_deterministic() -> None:
    # Realistic 268/200/100 split (total 568). A 150-card proportional sample
    # should land near 70.8 / 52.8 / 26.4 and sum to exactly 150.
    strata = (
        ["detector_positive"] * 268
        + ["detector_rejected_near_miss"] * 200
        + ["detector_negative_random"] * 100
    )
    card_ids = [f"c{i}" for i in range(len(strata))]
    keep_a = stage60.stratified_sample(card_ids, strata, 150, stage60.SAMPLE_SEED)
    keep_b = stage60.stratified_sample(card_ids, strata, 150, stage60.SAMPLE_SEED)
    # Deterministic and a valid subset of distinct indices.
    assert keep_a == keep_b
    assert keep_a == sorted(keep_a)
    assert len(keep_a) == len(set(keep_a)) == 150
    assert all(0 <= i < len(card_ids) for i in keep_a)

    per = {s: 0 for s in set(strata)}
    for i in keep_a:
        per[strata[i]] += 1
    # Realized total is exact; each stratum is within 1 of its proportional target.
    assert sum(per.values()) == 150
    targets = {
        "detector_positive": 150 * 268 / 568,
        "detector_rejected_near_miss": 150 * 200 / 568,
        "detector_negative_random": 150 * 100 / 568,
    }
    for s, t in targets.items():
        assert abs(per[s] - t) <= 1.0

    # A different sample seed selects a different set of cards.
    keep_other = stage60.stratified_sample(card_ids, strata, 150, stage60.SAMPLE_SEED + 1)
    assert keep_other != keep_a


def test_stratified_sample_out_of_range_raises() -> None:
    card_ids = [f"c{i}" for i in range(5)]
    strata = ["a"] * 5
    with pytest.raises(ValueError):
        stage60.stratified_sample(card_ids, strata, 0, 1)
    with pytest.raises(ValueError):
        stage60.stratified_sample(card_ids, strata, 6, 1)


def test_stage_reread_sample_integrity(synthetic_gallery: dict[str, object]) -> None:
    # Sample 6 of the 12 synthetic cards (6 pos / 4 near-miss / 2 neg). The
    # proportional 6-card draw is 3 / 2 / 1 and must stage exactly those.
    out_dir = Path(synthetic_gallery["repo_root"]) / "results" / "gallery_readjud_blind"  # type: ignore[arg-type]
    rc = stage60.stage_reread(
        manifest_path=synthetic_gallery["manifest"],  # type: ignore[arg-type]
        gallery_root=synthetic_gallery["gallery_root"],  # type: ignore[arg-type]
        out_dir=out_dir,
        seed=stage60.REREAD_SEED,
        sample_size=6,
        sample_seed=stage60.SAMPLE_SEED,
    )
    assert rc == 0

    pngs = sorted(out_dir.glob("blind_*.png"))
    assert len(pngs) == 6
    # Contiguous blind_0001..blind_0006 even though only a subset was staged.
    assert [p.stem for p in pngs] == [f"blind_{i:04d}" for i in range(1, 7)]

    bmap = pl.read_csv(out_dir / "_blind_map.csv")
    assert bmap.columns == ["blind_id", "card_id", "stratum"]
    assert bmap.height == 6
    # 1:1 blind_id <-> card_id, no collisions, all from the real gallery.
    assert bmap.get_column("blind_id").n_unique() == 6
    assert bmap.get_column("card_id").n_unique() == 6
    assert set(bmap.get_column("card_id").to_list()).issubset(
        set(synthetic_gallery["card_ids"])  # type: ignore[arg-type]
    )

    # Proportional 6-of-12: 3 detector_positive, 2 near-miss, 1 neg.
    counts = {
        row["stratum"]: row["n"]
        for row in bmap.group_by("stratum").len(name="n").iter_rows(named=True)
    }
    assert counts == {
        "detector_positive": 3,
        "detector_rejected_near_miss": 2,
        "neg": 1,
    }

    # Byte-identical copies for every staged card.
    for row in bmap.iter_rows(named=True):
        staged_bytes = (out_dir / f"{row['blind_id']}.png").read_bytes()
        expected = b"PNGDATA-" + str(row["card_id"]).encode("ascii")
        assert staged_bytes == expected

    # The HTML still leaks nothing and references only blind ids.
    html = (out_dir / "reread.html").read_text(encoding="utf-8")
    assert "card 1 of 6" in html
    assert "_blind_map" not in html
    for cid in bmap.get_column("card_id").to_list():
        assert cid not in html


def test_stage_reread_idempotent(synthetic_gallery: dict[str, object]) -> None:
    out_dir = Path(synthetic_gallery["repo_root"]) / "results" / "gallery_readjud_blind"  # type: ignore[arg-type]
    kwargs = dict(
        manifest_path=synthetic_gallery["manifest"],
        gallery_root=synthetic_gallery["gallery_root"],
        out_dir=out_dir,
        seed=stage60.REREAD_SEED,
    )
    stage60.stage_reread(**kwargs)  # type: ignore[arg-type]
    first = pl.read_csv(out_dir / "_blind_map.csv")
    # A stale file that a re-run must remove.
    (out_dir / "blind_9999.png").write_bytes(b"stale")
    stage60.stage_reread(**kwargs)  # type: ignore[arg-type]
    second = pl.read_csv(out_dir / "_blind_map.csv")
    assert first.equals(second)
    assert not (out_dir / "blind_9999.png").exists()


# ---------------------------------------------------------------------------
# Recompute: blind_id -> card_id -> row_id join.
# ---------------------------------------------------------------------------
@pytest.fixture
def staged_with_inventory(synthetic_gallery: dict[str, object]) -> dict[str, object]:
    """Stage the synthetic gallery and build a matching inventory for bridging."""
    repo_root = Path(synthetic_gallery["repo_root"])  # type: ignore[arg-type]
    out_dir = repo_root / "results" / "gallery_readjud_blind"
    stage60.stage_reread(
        manifest_path=synthetic_gallery["manifest"],  # type: ignore[arg-type]
        gallery_root=synthetic_gallery["gallery_root"],  # type: ignore[arg-type]
        out_dir=out_dir,
        seed=stage60.REREAD_SEED,
    )
    # Build an inventory whose (subject_id, record_id, nbp_timestamp_s) triple
    # matches the manifest so build_card_to_rowid resolves every card.
    man = pl.read_csv(synthetic_gallery["manifest"])  # type: ignore[arg-type]
    inv = man.select(
        pl.col("subject_id"),
        pl.col("record_id"),
        pl.col("t_nbp").alias("nbp_timestamp_s"),
    )
    inv_path = repo_root / "data" / "interim" / "event_inventory.csv"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv.write_csv(inv_path)
    return {
        "out_dir": out_dir,
        "manifest": synthetic_gallery["manifest"],
        "inventory": inv_path,
        "blind_map": out_dir / "_blind_map.csv",
        "card_ids": synthetic_gallery["card_ids"],
    }


def test_deblind_and_bridge_resolves_all(
    staged_with_inventory: dict[str, object], tmp_path: Path
) -> None:
    bmap = pl.read_csv(staged_with_inventory["blind_map"])  # type: ignore[arg-type]
    # Synthesize a full pass-2 export keyed by blind_id.
    export = bmap.select(
        pl.col("blind_id"),
        pl.lit("no_occlusion_signature").alias("call"),
        pl.lit("high").alias("confidence"),
        pl.lit("").alias("notes"),
        pl.lit("2026-06-01T00:00:00.000Z").alias("utc"),
    )
    export_path = tmp_path / "reread_pass2_export.csv"
    export.write_csv(export_path)

    res = recompute61.deblind_and_bridge(
        pass2_export=export_path,
        blind_map=staged_with_inventory["blind_map"],  # type: ignore[arg-type]
        manifest=staged_with_inventory["manifest"],  # type: ignore[arg-type]
        inventory=staged_with_inventory["inventory"],  # type: ignore[arg-type]
    )
    assert res.n_export == 12
    assert res.n_mapped == 12
    assert res.n_bridged == 12
    assert res.frame.get_column("row_id").null_count() == 0
    assert set(res.frame.get_column("card_id").to_list()) == set(
        staged_with_inventory["card_ids"]  # type: ignore[arg-type]
    )


def test_deblind_unknown_blind_id_fails_loud(
    staged_with_inventory: dict[str, object], tmp_path: Path
) -> None:
    export = pl.DataFrame(
        {
            "blind_id": ["blind_9999"],
            "call": ["occlusion_signature_present"],
            "confidence": ["high"],
            "notes": [""],
            "utc": ["2026-06-01T00:00:00.000Z"],
        }
    )
    export_path = tmp_path / "bad_export.csv"
    export.write_csv(export_path)
    with pytest.raises(RuntimeError, match="did not resolve to a card_id"):
        recompute61.deblind_and_bridge(
            pass2_export=export_path,
            blind_map=staged_with_inventory["blind_map"],  # type: ignore[arg-type]
            manifest=staged_with_inventory["manifest"],  # type: ignore[arg-type]
            inventory=staged_with_inventory["inventory"],  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Measure-only: sampled intra-rater reliability, including the partial-export
# path (export covers only a subset of cards, not all 12).
# ---------------------------------------------------------------------------
def test_measure_only_partial_export(
    staged_with_inventory: dict[str, object], tmp_path: Path
) -> None:
    bmap = pl.read_csv(staged_with_inventory["blind_map"])  # type: ignore[arg-type]

    # Pass-1 reader form over ALL 12 cards: call them all "no_occlusion_signature".
    man = pl.read_csv(staged_with_inventory["manifest"])  # type: ignore[arg-type]
    pass1 = man.select(
        pl.col("card_id"),
        pl.col("image_path"),
        pl.lit("no_occlusion_signature").alias("call"),
        pl.lit("high").alias("confidence"),
        pl.lit("").alias("notes"),
    )
    pass1_path = tmp_path / "reader_form_blinded.csv"
    pass1.write_csv(pass1_path)
    pass1_before = pass1_path.read_bytes()

    # A medgemma calls CSV (machine call, audit-trail only).
    medgemma = man.select(
        pl.col("card_id"),
        pl.lit("no_occlusion_signature").alias("call"),
    )
    medgemma_path = tmp_path / "medgemma.csv"
    medgemma.write_csv(medgemma_path)

    # PARTIAL export: only the first 5 blind ids are rated. Two of them flip to
    # "occlusion_signature_present" (a real pass1->pass2 change); three stay.
    sample = bmap.head(5).with_columns(
        pl.Series(
            "call",
            [
                "occlusion_signature_present",
                "occlusion_signature_present",
                "no_occlusion_signature",
                "no_occlusion_signature",
                "no_occlusion_signature",
            ],
        ),
        pl.lit("high").alias("confidence"),
        pl.lit("").alias("notes"),
        pl.lit("2026-06-02T00:00:00.000Z").alias("utc"),
    ).select(["blind_id", "call", "confidence", "notes", "utc"])
    export_path = tmp_path / "reread_pass2_export.csv"
    sample.write_csv(export_path)

    md_out = tmp_path / "reread_reliability_sample.md"
    csv_out = tmp_path / "reread_reliability_sample.csv"
    change_log_out = tmp_path / "reread_change_log_sample.csv"
    # Paths that measure-only must NEVER write.
    forbidden_form = tmp_path / "reader_form_blinded_pass2.csv"
    forbidden_pr = tmp_path / "precision_recall_readjud"

    rc = recompute61.run_measure_only(
        pass2_export=export_path,
        blind_map=staged_with_inventory["blind_map"],  # type: ignore[arg-type]
        manifest_path=staged_with_inventory["manifest"],  # type: ignore[arg-type]
        inventory=staged_with_inventory["inventory"],  # type: ignore[arg-type]
        pass1_form=pass1_path,
        medgemma_csv=medgemma_path,
        reliability_md_out=md_out,
        reliability_csv_out=csv_out,
        change_log_out=change_log_out,
        seed=GLOBAL_SEED,
        n_bootstrap=200,
    )
    assert rc == 0

    # Outputs written.
    assert md_out.exists()
    assert csv_out.exists()
    assert change_log_out.exists()
    # Forbidden outputs NOT written.
    assert not forbidden_form.exists()
    assert not forbidden_pr.exists()
    # Pass-1 reference is byte-for-byte untouched.
    assert pass1_path.read_bytes() == pass1_before

    # Change log covers only the 5 exported cards.
    cl = pl.read_csv(change_log_out)
    assert cl.height == 5
    assert int(cl.get_column("changed").sum()) == 2

    # Reliability CSV: 5 paired cards, percent agreement 3/5 = 0.6.
    rel = pl.read_csv(csv_out)
    assert rel.height == 1
    assert rel.get_column("n_cards")[0] == 5
    assert rel.get_column("percent_agreement")[0] == pytest.approx(0.6)
    # Two cards moved to present, none away -> net present delta +2.
    assert rel.get_column("to_present")[0] == 2
    assert rel.get_column("from_present")[0] == 0
    assert rel.get_column("net_present_delta")[0] == 2

    # Report is feasibility-framed and clean of em-dashes / home paths.
    report = md_out.read_text(encoding="utf-8")
    assert "intra-rater reliability" in report.lower()
    assert "—" not in report  # no em-dash
    assert str(Path.home()) not in report


def test_measure_only_unknown_blind_id_fails_loud(
    staged_with_inventory: dict[str, object], tmp_path: Path
) -> None:
    # An export with a blind id that is not in the staging map must fail loud,
    # even in measure-only mode (every exported blind_id must resolve).
    man = pl.read_csv(staged_with_inventory["manifest"])  # type: ignore[arg-type]
    pass1 = man.select(
        pl.col("card_id"),
        pl.col("image_path"),
        pl.lit("no_occlusion_signature").alias("call"),
        pl.lit("high").alias("confidence"),
        pl.lit("").alias("notes"),
    )
    pass1_path = tmp_path / "reader_form_blinded.csv"
    pass1.write_csv(pass1_path)
    medgemma_path = tmp_path / "medgemma.csv"
    man.select(
        pl.col("card_id"), pl.lit("no_occlusion_signature").alias("call")
    ).write_csv(medgemma_path)

    export = pl.DataFrame(
        {
            "blind_id": ["blind_0001", "blind_9999"],
            "call": ["no_occlusion_signature", "occlusion_signature_present"],
            "confidence": ["high", "high"],
            "notes": ["", ""],
            "utc": ["2026-06-02T00:00:00.000Z", "2026-06-02T00:00:00.000Z"],
        }
    )
    export_path = tmp_path / "bad_export.csv"
    export.write_csv(export_path)

    with pytest.raises(RuntimeError, match="did not resolve to a card_id"):
        recompute61.run_measure_only(
            pass2_export=export_path,
            blind_map=staged_with_inventory["blind_map"],  # type: ignore[arg-type]
            manifest_path=staged_with_inventory["manifest"],  # type: ignore[arg-type]
            inventory=staged_with_inventory["inventory"],  # type: ignore[arg-type]
            pass1_form=pass1_path,
            medgemma_csv=medgemma_path,
            reliability_md_out=tmp_path / "r.md",
            reliability_csv_out=tmp_path / "r.csv",
            change_log_out=tmp_path / "cl.csv",
            seed=GLOBAL_SEED,
            n_bootstrap=50,
        )


# ---------------------------------------------------------------------------
# Kappa on a hand-checked synthetic fixture.
# ---------------------------------------------------------------------------
def test_cohen_kappa_perfect_agreement() -> None:
    a = ["x", "y", "z", "x", "y"]
    assert cohen_kappa(a, a) == pytest.approx(1.0)


def test_cohen_kappa_hand_checked() -> None:
    # 2x2 hand-checkable case. 10 items.
    # rater A: 6 present, 4 absent. rater B: 5 present, 5 absent.
    # Confusion: present/present=4, present/absent=2, absent/present=1, absent/absent=3.
    a = (
        ["present"] * 4 + ["present"] * 2 + ["absent"] * 1 + ["absent"] * 3
    )
    b = (
        ["present"] * 4 + ["absent"] * 2 + ["present"] * 1 + ["absent"] * 3
    )
    # p_o = (4 + 3) / 10 = 0.7
    # marg A: present 6/10, absent 4/10 ; marg B: present 5/10, absent 5/10
    # p_e = 0.6*0.5 + 0.4*0.5 = 0.5 ; kappa = (0.7-0.5)/(1-0.5) = 0.4
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_agreement_summary_matches_components() -> None:
    a = ["present", "absent", "indeterminate", "present", "absent"]
    b = ["present", "absent", "present", "absent", "absent"]
    summ = agreement_summary(a, b)
    assert summ.n == 5
    # agree on items 0,1,4 -> 3/5
    assert summ.percent_agreement == pytest.approx(0.6)
    assert summ.cohen_kappa == pytest.approx(cohen_kappa(a, b))
    assert set(summ.categories) == {"present", "absent", "indeterminate"}


def test_landis_koch_bands() -> None:
    assert landis_koch_band(-0.1) == "poor"
    assert landis_koch_band(0.1) == "slight"
    assert landis_koch_band(0.3) == "fair"
    assert landis_koch_band(0.5) == "moderate"
    assert landis_koch_band(0.7) == "substantial"
    assert landis_koch_band(0.9) == "almost perfect"
    assert landis_koch_band(float("nan")) == "undefined"


def test_change_direction_classifier() -> None:
    cd = recompute61._change_direction
    assert cd("no_occlusion_signature", "no_occlusion_signature") == "unchanged"
    assert cd("no_occlusion_signature", "occlusion_signature_present") == "to_present"
    assert cd("occlusion_signature_present", "no_occlusion_signature") == "from_present"
    assert cd("no_occlusion_signature", "indeterminate") == "to_indeterminate"
    assert cd("indeterminate", "no_occlusion_signature") == "from_indeterminate"


def test_collapse_binary() -> None:
    cb = recompute61._collapse_binary
    assert cb("occlusion_signature_present") == "occlusion_signature_present"
    assert cb("no_occlusion_signature") == "no_occlusion_signature"
    assert cb("indeterminate") == "no_occlusion_signature"
