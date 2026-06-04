"""Tests for the blinded reader-adjudication web app builder (scripts/52).

These exercise the importable, in-memory pieces: deterministic blind shuffle,
card embedding, the blind_index -> card_id round trip, blinding (no card_id or
stratum leaks into the visible card section), the four synthetic reference
examples (present and distinct from gallery PNGs), and that the assembled HTML
is well formed enough to load.

The tests build a small synthetic gallery (a handful of card PNGs) so they do
not depend on the credentialed waveform tree, then also run a fast assertion
against the real manifest count when it is present.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from cuffcrt._seed import GLOBAL_SEED

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """Import scripts/52_build_adjudication_app.py (numeric filename)."""
    path = SCRIPTS_DIR / "52_build_adjudication_app.py"
    spec = importlib.util.spec_from_file_location("adjudication_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["adjudication_app"] = module
    spec.loader.exec_module(module)
    return module


def _make_card_png(path: Path, color: str) -> None:
    """Write a tiny distinct PNG so each card has a unique base64 payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.0, 1.0))
    ax.plot([0, 1, 2], [0, 1, 0], color=color)
    fig.savefig(path)
    plt.close(fig)


def _toy_gallery(tmp_path: Path, n_per_stratum: int = 3) -> tuple[Path, Path]:
    """Build a toy gallery (manifest + PNGs) and return (manifest, gallery_root).

    Mirrors the real layout: ``<gallery_root>/<stratum>/<card_id>.png`` with the
    manifest ``image_path`` repo-relative to the gallery root's parent-of-parent
    (here, tmp_path), matching how the canonical manifest stores paths like
    ``results/gallery/detector_positive/A-....png``.
    """
    gallery_root = tmp_path / "results" / "gallery"
    strata = {
        "detector_positive": "A",
        "detector_rejected_near_miss": "B",
        "detector_negative_random": "C",
    }
    rows: list[dict[str, object]] = []
    palette = ["#111", "#222", "#333", "#444", "#555", "#666", "#777", "#888", "#999"]
    p = 0
    for stratum, prefix in strata.items():
        for k in range(n_per_stratum):
            card_id = f"{prefix}-{stratum[:4]}{k:04d}"
            rel = f"results/gallery/{stratum}/{card_id}.png"
            abs_png = tmp_path / rel
            _make_card_png(abs_png, palette[p % len(palette)])
            p += 1
            rows.append(
                {
                    "card_id": card_id,
                    "stratum": stratum,
                    "subject_id": f"p{p:05d}",
                    "record_id": f"r{p:05d}",
                    "t_nbp": float(100 * p),
                    "image_path": rel,
                    "image_sha256": hashlib.sha256(abs_png.read_bytes()).hexdigest(),
                    "is_occlusion_signature": stratum == "detector_positive",
                }
            )
    manifest = gallery_root / "gallery_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(manifest)
    return manifest, gallery_root


def test_blind_order_is_deterministic_for_seed():
    module = _load_module()
    ids = [f"id{i:03d}" for i in range(50)]
    a = module.blind_order(ids, GLOBAL_SEED)
    b = module.blind_order(ids, GLOBAL_SEED)
    assert a == b
    # It actually permutes (not the identity) and preserves the multiset.
    assert a != ids
    assert sorted(a) == sorted(ids)


def test_blind_order_changes_with_seed():
    module = _load_module()
    ids = [f"id{i:03d}" for i in range(50)]
    assert module.blind_order(ids, GLOBAL_SEED) != module.blind_order(ids, GLOBAL_SEED + 1)


def test_load_cards_assigns_sequential_blind_index(tmp_path: Path):
    module = _load_module()
    manifest, gallery_root = _toy_gallery(tmp_path, n_per_stratum=3)
    cards = module.load_cards(manifest, gallery_root, GLOBAL_SEED)
    assert len(cards) == 9
    assert [c.blind_index for c in cards] == list(range(1, 10))
    # Every card has a non-empty base64 payload.
    assert all(len(c.image_b64) > 0 for c in cards)


def test_blind_index_to_card_id_round_trips(tmp_path: Path):
    module = _load_module()
    manifest, gallery_root = _toy_gallery(tmp_path, n_per_stratum=4)
    cards = module.load_cards(manifest, gallery_root, GLOBAL_SEED)
    df = pl.read_csv(manifest)
    manifest_ids = set(df.get_column("card_id").to_list())
    mapped_ids = {c.card_id for c in cards}
    # 1:1 onto every manifest card_id, no extras, no drops.
    assert mapped_ids == manifest_ids
    assert len({c.card_id for c in cards}) == len(cards)
    assert len({c.blind_index for c in cards}) == len(cards)


def test_two_builds_same_seed_identical_blind_order(tmp_path: Path):
    """Deterministic shuffle: same seed -> identical blind_index -> card_id map."""
    module = _load_module()
    manifest, gallery_root = _toy_gallery(tmp_path, n_per_stratum=5)
    cards1 = module.load_cards(manifest, gallery_root, GLOBAL_SEED)
    cards2 = module.load_cards(manifest, gallery_root, GLOBAL_SEED)
    map1 = {c.blind_index: c.card_id for c in cards1}
    map2 = {c.blind_index: c.card_id for c in cards2}
    assert map1 == map2


def test_reference_examples_are_four_and_synthetic(tmp_path: Path):
    module = _load_module()
    examples = module.build_reference_examples(GLOBAL_SEED)
    assert len(examples) == module.N_REFERENCE_EXAMPLES == 4
    # Each has a caption and a base64 image.
    assert all(ex.caption and ex.image_b64 for ex in examples)
    # The four images are distinct from one another.
    shas = {hashlib.sha256(base64.b64decode(ex.image_b64)).hexdigest() for ex in examples}
    assert len(shas) == 4

    # And distinct from every real gallery PNG (synthetic, not unblinding).
    manifest, gallery_root = _toy_gallery(tmp_path, n_per_stratum=3)
    gallery_shas = set()
    for png in gallery_root.rglob("*.png"):
        gallery_shas.add(hashlib.sha256(png.read_bytes()).hexdigest())
    assert shas.isdisjoint(gallery_shas)


def test_html_embeds_all_cards_and_examples(tmp_path: Path):
    module = _load_module()
    manifest, gallery_root = _toy_gallery(tmp_path, n_per_stratum=3)
    out = tmp_path / "adjudication.html"
    code = module.build_app(
        manifest_path=manifest,
        gallery_root=gallery_root,
        prompt_path=REPO_ROOT / "prompts" / "adjudicate_system.txt",
        out_path=out,
        seed=GLOBAL_SEED,
    )
    assert code == 0
    doc = out.read_text(encoding="utf-8")

    # 4 reference examples are inlined as data-URI <img> tags.
    n_data_uri_imgs = doc.count('src="data:image/png;base64,')
    assert n_data_uri_imgs == 4

    # The 9 card payloads are embedded in the cards JSON blob (one "img" key each).
    m = re.search(r'<script id="cardsData"[^>]*>(.*?)</script>', doc, re.DOTALL)
    assert m is not None
    cards_blob = m.group(1)
    assert cards_blob.count('"img":') == 9


def test_html_does_not_leak_card_id_or_stratum_into_visible_text(tmp_path: Path):
    """Blinding: the visible card section must not contain card_id or stratum."""
    module = _load_module()
    manifest, gallery_root = _toy_gallery(tmp_path, n_per_stratum=3)
    out = tmp_path / "adjudication.html"
    module.build_app(
        manifest_path=manifest,
        gallery_root=gallery_root,
        prompt_path=REPO_ROOT / "prompts" / "adjudicate_system.txt",
        out_path=out,
        seed=GLOBAL_SEED,
    )
    doc = out.read_text(encoding="utf-8")

    # Strip out the embedded JSON blobs (the de-blinding map legitimately holds
    # card_ids there for export); what remains is the visible DOM + CSS/JS.
    visible = re.sub(
        r'<script id="(cardsData|blindMap|appConfig)"[^>]*>.*?</script>',
        "",
        doc,
        flags=re.DOTALL,
    )

    df = pl.read_csv(manifest)
    for cid in df.get_column("card_id").to_list():
        assert cid not in visible, f"card_id {cid} leaked into visible text"
    # No stratum names anywhere in the visible markup.
    for stratum in (
        "detector_positive",
        "detector_rejected_near_miss",
        "detector_negative_random",
    ):
        assert stratum not in visible, f"stratum {stratum} leaked into visible text"
    # The blind index label is present.
    assert "card 1 of 9" in visible


def test_html_has_no_external_resource_references(tmp_path: Path):
    """Self-contained: no http(s) URLs and no non-data src/href to fetch."""
    module = _load_module()
    manifest, gallery_root = _toy_gallery(tmp_path, n_per_stratum=2)
    out = tmp_path / "adjudication.html"
    module.build_app(
        manifest_path=manifest,
        gallery_root=gallery_root,
        prompt_path=REPO_ROOT / "prompts" / "adjudicate_system.txt",
        out_path=out,
        seed=GLOBAL_SEED,
    )
    doc = out.read_text(encoding="utf-8")
    assert "http://" not in doc
    assert "https://" not in doc
    # Every src= is a base64 data URI (the card image is set in JS, not markup).
    for m in re.finditer(r'src="([^"]*)"', doc):
        assert m.group(1).startswith("data:image/png;base64,"), m.group(1)
    # No external stylesheet or script links.
    assert "<link" not in doc
    assert "src=\"http" not in doc


def test_html_is_well_formed(tmp_path: Path):
    """The document parses cleanly and has matched key structural tags."""
    module = _load_module()
    manifest, gallery_root = _toy_gallery(tmp_path, n_per_stratum=2)
    out = tmp_path / "adjudication.html"
    module.build_app(
        manifest_path=manifest,
        gallery_root=gallery_root,
        prompt_path=REPO_ROOT / "prompts" / "adjudicate_system.txt",
        out_path=out,
        seed=GLOBAL_SEED,
    )
    doc = out.read_text(encoding="utf-8")

    class _Counter(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.starts = 0
            self.ends = 0

        def handle_starttag(self, tag: str, attrs: object) -> None:
            self.starts += 1

        def handle_endtag(self, tag: str) -> None:
            self.ends += 1

    parser = _Counter()
    parser.feed(doc)  # raises if grossly malformed
    assert parser.starts > 0 and parser.ends > 0
    assert doc.strip().startswith("<!DOCTYPE html>")
    assert doc.rstrip().endswith("</html>")
    # Required interactive scaffolding is present.
    for token in ("callGroup", "confGroup", "exportCsvBtn", "progressFill", "notesBox"):
        assert token in doc


def test_real_manifest_has_568_cards_if_present():
    """Smoke check against the canonical manifest count (skips if absent)."""
    manifest = REPO_ROOT / "results" / "gallery" / "gallery_manifest.csv"
    if not manifest.exists():
        return  # nothing to assert in environments without the gallery
    df = pl.read_csv(manifest)
    assert df.height == 568
    assert df.get_column("card_id").n_unique() == 568
