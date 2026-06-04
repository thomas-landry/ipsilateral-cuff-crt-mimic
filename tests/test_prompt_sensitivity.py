"""Tests for ``scripts/42_prompt_sensitivity.py`` and the variant prompt files.

These cover the deterministic, in-memory pieces of the prompt-sensitivity
helper without standing up an oMLX server or touching credentialed data:

* every variant system prompt under ``prompts/adjudicate_system_v_*.txt`` loads
  through the SHA-verified loader and preserves the canonical call vocabulary;
* the stratified subsample is deterministic across invocations with the same
  seed and preserves the per-stratum proportions within plus or minus one row;
* the CLI parser accepts the documented invocation and rejects malformed
  ``--variants`` strings;
* a full end-to-end ``main()`` run with the stub client (no network) writes
  one checkpoint, one CSV, one parquet, and one manifest JSON per variant.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.llm.prompts import PROMPTS_DIR, load_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

VARIANT_PROMPT_LABELS = ("v_compact", "v_explicit", "v_terse_criteria")
CANONICAL_CALLS = (
    "occlusion_signature_present",
    "no_occlusion_signature",
    "indeterminate",
)
LEGACY_CALLS = ("ipsilateral", "not_ipsilateral")


def _load_script_module():
    """Import ``scripts/42_prompt_sensitivity.py`` as a module (numeric filename)."""
    path = SCRIPTS_DIR / "42_prompt_sensitivity.py"
    spec = importlib.util.spec_from_file_location("prompt_sensitivity_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prompt_sensitivity_driver"] = module
    spec.loader.exec_module(module)
    return module


def _toy_gallery_manifest() -> pl.DataFrame:
    """Build a small in-memory gallery manifest preserving the 568-card schema.

    Proportions are kept close to the real 268/200/100 split so the stratified
    sampler exercise sees a realistic class imbalance.
    """
    rows: list[dict] = []
    for i in range(268):
        rows.append(
            {
                "card_id": f"A-pos-{i:04d}",
                "stratum": "detector_positive",
                "subject_id": f"p{i:07d}",
                "record_id": f"r{i:07d}",
                "t_nbp": float(i),
                "image_path": f"results/gallery/detector_positive/A-pos-{i:04d}.png",
                "image_sha256": "a" * 64,
                "is_occlusion_signature": True,
                "phase3_duration_s": 20.0,
                "nadir_depth_frac": 0.1,
                "alignment_offset_s": 0.0,
                "reject_reason": "",
            }
        )
    for i in range(200):
        rows.append(
            {
                "card_id": f"B-nm-{i:04d}",
                "stratum": "detector_rejected_near_miss",
                "subject_id": f"p{1000 + i:07d}",
                "record_id": f"r{1000 + i:07d}",
                "t_nbp": float(i),
                "image_path": (
                    f"results/gallery/detector_rejected_near_miss/B-nm-{i:04d}.png"
                ),
                "image_sha256": "b" * 64,
                "is_occlusion_signature": False,
                "phase3_duration_s": 12.0,
                "nadir_depth_frac": 0.18,
                "alignment_offset_s": 0.0,
                "reject_reason": "no_phase2",
            }
        )
    for i in range(100):
        rows.append(
            {
                "card_id": f"C-neg-{i:04d}",
                "stratum": "detector_negative_random",
                "subject_id": f"p{2000 + i:07d}",
                "record_id": f"r{2000 + i:07d}",
                "t_nbp": float(i),
                "image_path": (
                    f"results/gallery/detector_negative_random/C-neg-{i:04d}.png"
                ),
                "image_sha256": "c" * 64,
                "is_occlusion_signature": False,
                "phase3_duration_s": 0.0,
                "nadir_depth_frac": 0.6,
                "alignment_offset_s": 0.0,
                "reject_reason": "no_phase2",
            }
        )
    return pl.DataFrame(rows)


# -- Variant prompt files -------------------------------------------------------


@pytest.mark.parametrize("label", VARIANT_PROMPT_LABELS)
def test_variant_prompt_file_exists(label: str) -> None:
    """Each variant system prompt file must be present on disk."""
    path = PROMPTS_DIR / f"adjudicate_system_{label}.txt"
    assert path.exists(), f"missing variant prompt: {path}"


@pytest.mark.parametrize("label", VARIANT_PROMPT_LABELS)
def test_variant_prompt_loads_with_sha_verification(label: str) -> None:
    """Each variant must load cleanly through the SHA-verified loader."""
    path = PROMPTS_DIR / f"adjudicate_system_{label}.txt"
    loaded = load_prompt(path)
    assert loaded.body_bytes > 0
    assert len(loaded.sha256) == 64


@pytest.mark.parametrize("label", VARIANT_PROMPT_LABELS)
def test_variant_prompt_preserves_call_vocabulary(label: str) -> None:
    """Each variant must keep the canonical call vocabulary and refuse the legacy one."""
    path = PROMPTS_DIR / f"adjudicate_system_{label}.txt"
    loaded = load_prompt(path)
    for call in CANONICAL_CALLS:
        assert call in loaded.text, f"variant {label!r} lost canonical call {call!r}"
    for legacy in LEGACY_CALLS:
        assert legacy not in loaded.text, (
            f"variant {label!r} carries legacy call {legacy!r}"
        )


@pytest.mark.parametrize("label", VARIANT_PROMPT_LABELS)
def test_variant_prompt_keeps_required_json_keys(label: str) -> None:
    """Each variant must still name all four required output JSON keys."""
    path = PROMPTS_DIR / f"adjudicate_system_{label}.txt"
    text = load_prompt(path).text
    for key in ("observed", "call", "confidence", "rationale"):
        assert f'"{key}"' in text, f"variant {label!r} dropped key {key!r}"


def test_variant_loader_helper_returns_loaded_prompt() -> None:
    """The script's ``load_variant_prompt`` helper returns a verified prompt."""
    module = _load_script_module()
    loaded = module.load_variant_prompt("v_compact")
    assert loaded.body_bytes > 0
    assert len(loaded.sha256) == 64


# -- Stratified subsample -------------------------------------------------------


def test_stratified_subsample_is_deterministic() -> None:
    """The same ``(manifest, n_rows, seed)`` always selects the same card ids."""
    module = _load_script_module()
    manifest = _toy_gallery_manifest()
    first = module.stratified_subsample(manifest, n_rows=100, seed=GLOBAL_SEED)
    second = module.stratified_subsample(manifest, n_rows=100, seed=GLOBAL_SEED)
    assert first["card_id"].to_list() == second["card_id"].to_list()


def test_stratified_subsample_preserves_tier_proportions() -> None:
    """The subsample's stratum counts match the proportional quotas within +/- 1."""
    module = _load_script_module()
    manifest = _toy_gallery_manifest()
    n_rows = 100
    subsample = module.stratified_subsample(manifest, n_rows=n_rows, seed=GLOBAL_SEED)
    assert subsample.height == n_rows

    total = manifest.height
    counts = (
        subsample.group_by("stratum")
        .len()
        .rename({"len": "drawn"})
        .to_dicts()
    )
    counts_by_stratum = {row["stratum"]: row["drawn"] for row in counts}
    for stratum in module.GALLERY_STRATA:
        size = manifest.filter(pl.col("stratum") == stratum).height
        expected = size / total * n_rows
        observed = counts_by_stratum.get(stratum, 0)
        assert abs(observed - expected) <= 1, (
            f"stratum {stratum!r}: drew {observed}, expected ~{expected:.2f}"
        )


def test_stratified_subsample_changes_with_different_seed() -> None:
    """A different seed (almost) always selects a different set of card ids."""
    module = _load_script_module()
    manifest = _toy_gallery_manifest()
    seed_a = GLOBAL_SEED
    seed_b = GLOBAL_SEED + 1
    a = set(
        module.stratified_subsample(manifest, n_rows=100, seed=seed_a)["card_id"].to_list()
    )
    b = set(
        module.stratified_subsample(manifest, n_rows=100, seed=seed_b)["card_id"].to_list()
    )
    assert a != b


def test_stratified_subsample_rejects_oversized_request() -> None:
    """Requesting more rows than the manifest holds raises a clear error."""
    module = _load_script_module()
    manifest = _toy_gallery_manifest()
    with pytest.raises(ValueError, match="exceeds manifest size"):
        module.stratified_subsample(manifest, n_rows=manifest.height + 1, seed=GLOBAL_SEED)


def test_stratified_subsample_rejects_non_positive_n_rows() -> None:
    """A non-positive ``n_rows`` raises a clear error."""
    module = _load_script_module()
    manifest = _toy_gallery_manifest()
    with pytest.raises(ValueError, match="must be positive"):
        module.stratified_subsample(manifest, n_rows=0, seed=GLOBAL_SEED)


# -- CLI parsing ----------------------------------------------------------------


def test_cli_parses_full_documented_invocation(tmp_path: Path) -> None:
    """The example invocation in the script docstring parses end-to-end."""
    module = _load_script_module()
    manifest = tmp_path / "gallery_manifest.csv"
    manifest.write_text("card_id,stratum\n", encoding="utf-8")
    args = module._parse_args(
        [
            "--variants",
            "v_compact,v_explicit",
            "--n_rows",
            "100",
            "--seed",
            "20260426",
            "--gallery_manifest",
            str(manifest),
            "--out_dir",
            str(tmp_path / "out"),
            "--port",
            "8000",
        ]
    )
    assert args.variants == ["v_compact", "v_explicit"]
    assert args.n_rows == 100
    assert args.seed == 20260426
    assert args.port == 8000
    assert args.gallery_manifest == manifest
    assert args.out_dir == tmp_path / "out"
    # Defaults are stable and match script 41's decoding parameters.
    assert args.temperature == module.DEFAULT_TEMPERATURE
    assert args.max_tokens == module.MAX_TOKENS
    assert args.model == module.SERVED_MODEL_ID
    assert args.resume is True


def test_cli_rejects_too_many_variants() -> None:
    """The CLI accepts at most three variants."""
    module = _load_script_module()
    with pytest.raises(SystemExit):
        module._parse_args(["--variants", "v_a,v_b,v_c,v_d"])


def test_cli_rejects_duplicate_variants() -> None:
    """Duplicate variant labels are a CLI error."""
    module = _load_script_module()
    with pytest.raises(SystemExit):
        module._parse_args(["--variants", "v_compact,v_compact"])


def test_cli_rejects_empty_variants() -> None:
    """An empty ``--variants`` string is a CLI error."""
    module = _load_script_module()
    with pytest.raises(SystemExit):
        module._parse_args(["--variants", ""])


def test_resolve_variant_prompt_path_uses_template() -> None:
    """The label-to-path resolver slots the label into the canonical template."""
    module = _load_script_module()
    resolved = module.resolve_variant_prompt_path("v_compact")
    assert resolved.name == "adjudicate_system_v_compact.txt"
    assert resolved.parent == PROMPTS_DIR


def test_resolve_base_url_prefers_explicit_then_env_then_port(monkeypatch) -> None:
    """Server URL precedence: --base_url, then $OMLX_BASE_URL, then --port."""
    module = _load_script_module()
    monkeypatch.delenv("OMLX_BASE_URL", raising=False)

    args_explicit = module._parse_args(
        ["--base_url", "http://explicit:9000/v1", "--port", "9999"]
    )
    assert module._resolve_base_url(args_explicit) == "http://explicit:9000/v1"

    monkeypatch.setenv("OMLX_BASE_URL", "http://envhost:9100/v1")
    args_env = module._parse_args(["--port", "9999"])
    assert module._resolve_base_url(args_env) == "http://envhost:9100/v1"

    monkeypatch.delenv("OMLX_BASE_URL", raising=False)
    args_port = module._parse_args(["--port", "9999"])
    assert module._resolve_base_url(args_port) == "http://localhost:9999/v1"


# -- End-to-end main run --------------------------------------------------------


def _write_toy_manifest_and_pngs(tmp_path: Path) -> tuple[Path, Path]:
    """Materialize a tiny manifest plus matching PNGs under ``tmp_path``.

    The manifest mirrors the real schema but holds only nine cards (three per
    stratum) so a request for nine total cards trivially preserves proportions.
    """
    gallery_root = tmp_path / "repo"
    gallery_root.mkdir()
    image_dir = gallery_root / "gallery"
    image_dir.mkdir()

    rows: list[dict] = []
    png_header = (
        b"\x89PNG\r\n\x1a\n"  # minimal valid-looking PNG signature
    )
    for stratum, prefix in (
        ("detector_positive", "A"),
        ("detector_rejected_near_miss", "B"),
        ("detector_negative_random", "C"),
    ):
        for i in range(3):
            card_id = f"{prefix}-{stratum}-{i:02d}"
            rel_path = f"gallery/{card_id}.png"
            abs_path = gallery_root / rel_path
            abs_path.write_bytes(png_header + bytes([i]) * 16)
            rows.append(
                {
                    "card_id": card_id,
                    "stratum": stratum,
                    "subject_id": f"p{i:07d}",
                    "record_id": f"r{i:07d}",
                    "image_path": rel_path,
                    "image_sha256": "deadbeef" * 8,
                }
            )
    manifest_path = gallery_root / "gallery_manifest.csv"
    pl.DataFrame(rows).write_csv(manifest_path)
    return manifest_path, gallery_root


def test_main_writes_per_variant_outputs_with_stub_client(tmp_path: Path) -> None:
    """A full ``main()`` run with ``--dry_run`` writes outputs for every variant."""
    module = _load_script_module()
    manifest_path, gallery_root = _write_toy_manifest_and_pngs(tmp_path)
    out_dir = tmp_path / "out"

    code = module.main(
        [
            "--variants",
            "v_compact,v_explicit",
            "--n_rows",
            "9",
            "--seed",
            str(GLOBAL_SEED),
            "--gallery_manifest",
            str(manifest_path),
            "--gallery_root",
            str(gallery_root),
            "--out_dir",
            str(out_dir),
            "--dry_run",
        ]
    )
    assert code == 0

    for variant in ("v_compact", "v_explicit"):
        csv_path = out_dir / f"{variant}.csv"
        parquet_path = out_dir / f"{variant}.parquet"
        checkpoint_path = out_dir / f"{variant}_checkpoint.csv"
        manifests = list(out_dir.glob(f"_run_manifest_{variant}_*.json"))

        assert csv_path.exists(), f"missing csv for {variant}"
        assert parquet_path.exists(), f"missing parquet for {variant}"
        assert checkpoint_path.exists(), f"missing checkpoint for {variant}"
        assert len(manifests) == 1, f"expected one manifest for {variant}"

        df = pl.read_csv(csv_path)
        assert df.height == 9
        # Schema is a superset of the canonical adjudication run-log.
        for col in module.LOG_COLUMNS_WITH_IDS:
            assert col in df.columns, f"variant {variant} missing column {col!r}"
        # Stub client returns an indeterminate call; downstream concordance
        # work joins on card_id, which must be present.
        assert df["card_id"].n_unique() == 9
        assert set(df["call"].to_list()) <= set(CANONICAL_CALLS) | {None}

        payload = json.loads(manifests[0].read_text())
        assert payload["variant_label"] == variant
        assert payload["n_rows_total"] == 9
        assert payload["dry_run"] is True


def test_main_returns_2_when_manifest_missing(tmp_path: Path) -> None:
    """A missing gallery manifest path is a clean exit-2 error."""
    module = _load_script_module()
    code = module.main(
        [
            "--variants",
            "v_compact",
            "--n_rows",
            "5",
            "--gallery_manifest",
            str(tmp_path / "does_not_exist.csv"),
            "--out_dir",
            str(tmp_path / "out"),
            "--dry_run",
        ]
    )
    assert code == 2


def test_main_returns_2_when_variant_prompt_missing(tmp_path: Path) -> None:
    """An unknown variant label is a clean exit-2 error."""
    module = _load_script_module()
    manifest_path, gallery_root = _write_toy_manifest_and_pngs(tmp_path)
    code = module.main(
        [
            "--variants",
            "v_does_not_exist",
            "--n_rows",
            "9",
            "--gallery_manifest",
            str(manifest_path),
            "--gallery_root",
            str(gallery_root),
            "--out_dir",
            str(tmp_path / "out"),
            "--dry_run",
        ]
    )
    assert code == 2


def test_global_seed_is_the_documented_constant() -> None:
    """The seed default propagates ``cuffcrt._seed.GLOBAL_SEED`` (20260426)."""
    module = _load_script_module()
    args = module._parse_args([])
    assert args.seed == GLOBAL_SEED
    # Sanity check on the underlying RNG: the same seed gives the same first draw.
    a = np.random.default_rng(GLOBAL_SEED).integers(0, 10_000)
    b = np.random.default_rng(GLOBAL_SEED).integers(0, 10_000)
    assert a == b
