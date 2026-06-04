"""Build the single-file blinded reader-adjudication web app (step 52).

The candidate gallery (step 51) renders 568 anchor-free perfusion-index (PI)
cards across three detector strata. This script packages every card into ONE
self-contained ``adjudication.html`` that the principal reader opens by double
click: all PNGs are base64-embedded, all CSS and JavaScript are inline, and
there are no external assets, no network calls, and no server. The reader works
through the cards offline, one large trace per screen, and exports a CSV that is
later merged back into ``reader_form_blinded.csv``.

Blinding
--------
The reference standard must be blinded so the reader judges morphology alone,
never the detector's stratum. The app therefore shows ONLY a blind sequential
index ("card 237 of 568"). It never renders ``card_id``, the ``A-``/``B-``/
``C-`` stratum prefix, the ``stratum`` value, ``is_occlusion_signature``, the
detector verdict, the filename, or any folder path. The 568 cards are shuffled
deterministically with ``numpy.random.default_rng(GLOBAL_SEED)`` so the
presentation order is identical across rebuilds (required for multi-session
resume). The blind-index -> ``card_id`` map is embedded in the page JavaScript
only so the CSV export can be de-blinded; it is never displayed on screen.

Reference panel
---------------
A collapsible "what to look for" panel at the top shows four SYNTHETIC
illustrative traces (generated here from the cuff-event envelope physiology in
:mod:`cuffcrt.signal.synthetic`, never real gallery PNGs, which would unblind
the reader). They are rendered in the same anchor-free style as the cards.

Reproducibility
---------------
Re-running this script against the same manifest and seed produces the same
blind order and the same embedded images. The build embeds the manifest
SHA-256 and seed in the page so a rebuild is auditable. This script reads only;
it never writes the canonical reader form, manifest, prompts, or PNGs, and it
never calls a model.

Examples
--------
Build against the canonical gallery::

    uv run python scripts/52_build_adjudication_app.py

Custom manifest / output::

    uv run python scripts/52_build_adjudication_app.py \\
        --manifest results/gallery/gallery_manifest.csv \\
        --gallery-root results/gallery \\
        --out results/gallery/adjudication.html
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib.util
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

from cuffcrt._seed import GLOBAL_SEED
from cuffcrt.signal.synthetic import CuffEvent, _apply_cuff_envelope

# ---------------------------------------------------------------------------
# Reuse the gallery's exact anchor-free render style so the synthetic reference
# examples look like the real cards. ``51_candidate_gallery.py`` starts with a
# digit, so import it by path.
# ---------------------------------------------------------------------------
_GALLERY_PATH = Path(__file__).with_name("51_candidate_gallery.py")
_spec = importlib.util.spec_from_file_location("_gallery51", _GALLERY_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - import guard
    raise ImportError(f"cannot load gallery module from {_GALLERY_PATH}")
_gallery = importlib.util.module_from_spec(_spec)
sys.modules["_gallery51"] = _gallery
_spec.loader.exec_module(_gallery)

_RC_PARAMS_BLINDED = _gallery._RC_PARAMS_BLINDED
RENDER_WINDOW_LO_S: float = _gallery.RENDER_WINDOW_LO_S
RENDER_WINDOW_HI_S: float = _gallery.RENDER_WINDOW_HI_S

# Reader-facing call vocabulary; must stay in sync with scripts/44 and MedGemma.
CALL_PRESENT = "occlusion_signature_present"
CALL_ABSENT = "no_occlusion_signature"
CALL_INDETERMINATE = "indeterminate"
CALL_VOCAB = (CALL_PRESENT, CALL_ABSENT, CALL_INDETERMINATE)

# Confidence vocabulary (per card).
CONFIDENCE_VOCAB = ("low", "med", "high")

# localStorage namespace; bumping it invalidates any saved progress.
LOCALSTORAGE_KEY = "cuffcrt_adjudication_v1"

# How many synthetic reference examples appear in the "what to look for" panel.
N_REFERENCE_EXAMPLES = 4


@dataclass(frozen=True)
class Card:
    """One blinded adjudication card after deterministic shuffling.

    Parameters
    ----------
    blind_index : int
        1-based position the reader sees ("card N of M"). Stable across
        rebuilds for a fixed seed and manifest.
    card_id : str
        The canonical gallery id (e.g. ``A-b1769fc4ee128967``). Embedded only
        in the de-blinding map used at export; never shown on screen.
    image_b64 : str
        The card PNG as a base64 ASCII string (no data-URI prefix).
    """

    blind_index: int
    card_id: str
    image_b64: str


@dataclass(frozen=True)
class ReferenceExample:
    """One synthetic illustrative trace for the reference panel.

    Parameters
    ----------
    key : str
        Stable identifier for the example (used in the DOM id).
    title : str
        Short heading shown above the example image.
    caption : str
        Verbatim "what to look for" caption from the adjudication criteria.
    image_b64 : str
        The synthetic trace PNG as base64 ASCII (no data-URI prefix).
    """

    key: str
    title: str
    caption: str
    image_b64: str


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file on disk.

    Parameters
    ----------
    path : pathlib.Path
        File to digest.

    Returns
    -------
    str
        Lowercase hex SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blind_order(card_ids: list[str], seed: int) -> list[str]:
    """Return ``card_ids`` permuted deterministically for the given seed.

    The permutation is generated with ``numpy.random.default_rng(seed)`` and a
    Fisher-Yates style ``shuffle`` so the same seed and the same input order
    always yield the same presentation order. The caller is responsible for
    passing ``card_ids`` in a stable source order (manifest row order).

    Parameters
    ----------
    card_ids : list[str]
        Source-ordered card ids.
    seed : int
        RNG seed; the project uses :data:`cuffcrt._seed.GLOBAL_SEED`.

    Returns
    -------
    list[str]
        A new list containing the same ids in shuffled order.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(card_ids))
    rng.shuffle(idx)
    return [card_ids[i] for i in idx.tolist()]


def load_cards(manifest_path: Path, gallery_root: Path, seed: int) -> list[Card]:
    """Read the manifest, embed every PNG, and return shuffled blinded cards.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Path to ``gallery_manifest.csv``. Must contain ``card_id`` and
        ``image_path`` (repo-relative).
    gallery_root : pathlib.Path
        Root the manifest ``image_path`` values resolve against. Repo-relative
        ``image_path`` values already include the gallery prefix, so they are
        resolved against the repo root (the parent of ``gallery_root``); paths
        that already exist as given are used directly.
    seed : int
        Shuffle seed.

    Returns
    -------
    list[Card]
        One :class:`Card` per manifest row, in blind (shuffled) order, with
        ``blind_index`` assigned 1..N over the shuffled sequence.

    Raises
    ------
    FileNotFoundError
        If the manifest or any referenced PNG is missing.
    ValueError
        If required columns are absent or card ids are not unique.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    df = pl.read_csv(manifest_path, infer_schema_length=20000)
    required = {"card_id", "image_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    card_ids = [str(c) for c in df.get_column("card_id").to_list()]
    image_paths = [str(p) for p in df.get_column("image_path").to_list()]
    if len(set(card_ids)) != len(card_ids):
        raise ValueError("manifest card_id column is not unique")

    id_to_path = dict(zip(card_ids, image_paths, strict=True))
    id_to_b64: dict[str, str] = {}
    repo_root = gallery_root.resolve().parent.parent
    for cid, rel in id_to_path.items():
        png = Path(rel)
        if not png.is_absolute() and not png.exists():
            png = repo_root / rel
        if not png.exists():
            raise FileNotFoundError(f"card PNG not found for {cid}: {rel}")
        id_to_b64[cid] = base64.b64encode(png.read_bytes()).decode("ascii")

    ordered = blind_order(card_ids, seed)
    return [
        Card(blind_index=i + 1, card_id=cid, image_b64=id_to_b64[cid])
        for i, cid in enumerate(ordered)
    ]


def _synthetic_pi_traces(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build four synthetic 1 Hz PI(t) traces for the reference panel.

    The traces are composed from the cuff-event envelope physiology in
    :mod:`cuffcrt.signal.synthetic` (the positive shape reuses
    ``_apply_cuff_envelope``), so they are genuinely synthetic and tied to the
    detector model, never copied from a real card. Each is returned on the same
    ``[RENDER_WINDOW_LO_S, RENDER_WINDOW_HI_S]`` time axis the cards use, at
    1 Hz, already median-normalized-friendly (the renderer normalizes again).

    Parameters
    ----------
    seed : int
        Seed for the additive noise so the examples are reproducible.

    Returns
    -------
    dict[str, tuple[numpy.ndarray, numpy.ndarray]]
        Mapping example key -> ``(t_local, pi)``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(RENDER_WINDOW_LO_S, RENDER_WINDOW_HI_S + 1.0, 1.0)
    base = 1.0
    noise_sd = 0.03

    # 1. Positive: one sustained deep drop to near zero then one graded
    #    monotonic recovery. Derived from the four-phase cuff envelope at the
    #    PI (envelope) level, centered near t=0 like the real cards.
    cuff = CuffEvent(
        t_inflation_start_s=-8.0,
        inflation_duration_s=6.0,
        hold_duration_s=10.0,
        deflate_duration_s=24.0,
        n_deflate_steps=10,
        floor=0.04,
    )
    pos = _apply_cuff_envelope(t.astype(float), cuff) * base
    pos = pos + noise_sd * rng.standard_normal(t.size)
    pos = np.clip(pos, 0.0, None)

    # 2. Negative, flat/spiky noise: no sustained deep drop.
    flat = base + 0.05 * rng.standard_normal(t.size)
    spike_idx = rng.choice(t.size, size=4, replace=False)
    flat[spike_idx] += rng.uniform(0.1, 0.25, size=4)
    flat = np.clip(flat, 0.0, None)

    # 3. Negative, many comparable dips: multiple similar dips, no single
    #    dominant deep drop with a clean recovery.
    many = base * np.ones(t.size)
    for _ in range(6):
        center = rng.uniform(RENDER_WINDOW_LO_S + 10, RENDER_WINDOW_HI_S - 10)
        width = rng.uniform(3.0, 6.0)
        depth = rng.uniform(0.35, 0.55)
        many = many - depth * np.exp(-0.5 * ((t - center) / width) ** 2)
    many = many + 0.02 * rng.standard_normal(t.size)
    many = np.clip(many, 0.0, None)

    # 4. Indeterminate: ambiguous / low quality. A shallow, partial drop that
    #    neither reaches near zero nor recovers cleanly, buried in noise.
    indet = base * np.ones(t.size)
    drop = (t >= 0) & (t <= 25)
    indet[drop] = indet[drop] - 0.4 * np.sin(np.pi * (t[drop]) / 25.0) ** 0.7
    indet = indet + 0.12 * rng.standard_normal(t.size)
    indet = np.clip(indet, 0.0, None)

    return {
        "present": (t, pos),
        "absent_flat": (t, flat),
        "absent_many": (t, many),
        "indeterminate": (t, indet),
    }


def _render_trace_b64(t_local: np.ndarray, pi: np.ndarray) -> str:
    """Render one anchor-free PI trace to base64 PNG, gallery style.

    Mirrors ``51_candidate_gallery._render_anchor_free_png`` (in-window median
    normalization, identical rcParams, ``[lo, hi]`` x-window, ``ylim`` bottom
    at 0, no title/marker/laterality word) but returns base64 bytes instead of
    writing a file, so the synthetic reference examples match the real cards.

    Parameters
    ----------
    t_local : numpy.ndarray
        Time axis in seconds (event-centered).
    pi : numpy.ndarray
        Perfusion-index samples aligned to ``t_local``.

    Returns
    -------
    str
        Base64 ASCII of the PNG bytes (no data-URI prefix).
    """
    in_window = (t_local >= RENDER_WINDOW_LO_S) & (t_local <= RENDER_WINDOW_HI_S)
    t = t_local[in_window]
    y = pi[in_window]
    if t.size == 0:
        raise ValueError("no PI samples in render window")
    median = float(np.nanmedian(y)) if np.isfinite(y).any() else 1.0
    if median <= 0 or not np.isfinite(median):
        median = 1.0
    y_norm = y / median

    buf = io.BytesIO()
    with matplotlib.rc_context(_RC_PARAMS_BLINDED):
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.plot(t, y_norm, color="#1A1A1A", linewidth=1.1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PI (norm)")
        ax.set_xlim(RENDER_WINDOW_LO_S, RENDER_WINDOW_HI_S)
        ax.set_ylim(bottom=0.0)
        ax.margins(x=0.0)
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_reference_examples(seed: int) -> list[ReferenceExample]:
    """Render the four synthetic "what to look for" examples.

    Captions are taken verbatim from the adjudication criteria
    (``prompts/adjudicate_system.txt``) so the reader judges the same thing
    MedGemma was asked to judge.

    Parameters
    ----------
    seed : int
        Seed for the synthetic-trace noise.

    Returns
    -------
    list[ReferenceExample]
        Exactly :data:`N_REFERENCE_EXAMPLES` examples.
    """
    traces = _synthetic_pi_traces(seed)
    specs = [
        (
            "present",
            "occlusion_signature_present",
            "A single sustained deep drop in PI to near zero, lasting several "
            "seconds, immediately followed by one graded, roughly monotonic "
            "recovery back toward baseline.",
        ),
        (
            "absent_flat",
            "no_occlusion_signature (flat / spiky noise)",
            "Flat or pure spiky noise, with no sustained deep drop.",
        ),
        (
            "absent_many",
            "no_occlusion_signature (many comparable dips)",
            "Multiple similar dips, with no single dominant deep drop followed "
            "by a clean recovery.",
        ),
        (
            "indeterminate",
            "indeterminate",
            "Too ambiguous or low quality to decide.",
        ),
    ]
    examples: list[ReferenceExample] = []
    for key, title, caption in specs:
        t, pi = traces[key]
        examples.append(
            ReferenceExample(
                key=key,
                title=title,
                caption=caption,
                image_b64=_render_trace_b64(t, pi),
            )
        )
    return examples


def _criteria_text(prompt_path: Path) -> str:
    """Return the adjudication-criteria body for the help panel.

    Reads ``prompts/adjudicate_system.txt`` and strips the leading ``# sha256``
    provenance line so the reader sees the criteria prose only. Falls back to a
    short built-in summary if the prompt file is absent (keeps the build robust
    for ``--demo``-style invocations without the prompt).

    Parameters
    ----------
    prompt_path : pathlib.Path
        Path to the adjudication system prompt.

    Returns
    -------
    str
        Criteria text (no leading sha line).
    """
    if not prompt_path.exists():
        logger.warning("prompt not found, using fallback criteria text: {}", prompt_path)
        return (
            "Classify each trace into exactly one of three categories. The "
            "pattern of interest is a single sustained deep drop in PI to near "
            "zero lasting several seconds, immediately followed by one graded, "
            "roughly monotonic recovery toward baseline."
        )
    lines = prompt_path.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if not ln.strip().lower().startswith("# sha256")]
    return "\n".join(body).strip()


def _build_html(
    cards: list[Card],
    examples: list[ReferenceExample],
    *,
    criteria_text: str,
    manifest_sha256: str,
    seed: int,
) -> str:
    """Assemble the complete self-contained HTML document.

    The blind-index -> ``card_id`` map and the base64 card images are embedded
    as JSON in a ``<script>`` block. No card-identifying text (card_id,
    stratum, filename) appears in any visible DOM node; the card screen renders
    only "card N of M" plus the image.

    Parameters
    ----------
    cards : list[Card]
        Shuffled blinded cards.
    examples : list[ReferenceExample]
        Synthetic reference examples.
    criteria_text : str
        Verbatim adjudication criteria.
    manifest_sha256 : str
        SHA-256 of the source manifest (provenance; shown in the footer).
    seed : int
        Shuffle seed (provenance; shown in the footer).

    Returns
    -------
    str
        The full HTML document as a single string.
    """
    n = len(cards)
    # Visible payload: only blind_index + image. NO card_id, NO stratum.
    cards_payload = [{"i": c.blind_index, "img": c.image_b64} for c in cards]
    # De-blinding map, embedded for export only; never rendered on screen.
    blind_to_card = {str(c.blind_index): c.card_id for c in cards}

    examples_html = "\n".join(
        f"""
        <figure class="ref-example">
          <figcaption><span class="ref-title">{html.escape(ex.title)}</span></figcaption>
          <img alt="synthetic reference example" src="data:image/png;base64,{ex.image_b64}" />
          <p class="ref-caption">{html.escape(ex.caption)}</p>
        </figure>"""
        for ex in examples
    )

    cards_json = json.dumps(cards_payload, separators=(",", ":"))
    blind_map_json = json.dumps(blind_to_card, separators=(",", ":"))
    criteria_html = html.escape(criteria_text).replace("\n", "<br>")

    config_json = json.dumps(
        {
            "storageKey": LOCALSTORAGE_KEY,
            "nCards": n,
            "callVocab": list(CALL_VOCAB),
            "confidenceVocab": list(CONFIDENCE_VOCAB),
            "manifestSha256": manifest_sha256,
            "seed": seed,
        },
        separators=(",", ":"),
    )

    return _HTML_TEMPLATE.format(
        n=n,
        examples_html=examples_html,
        criteria_html=criteria_html,
        cards_json=cards_json,
        blind_map_json=blind_map_json,
        config_json=config_json,
        manifest_sha256=html.escape(manifest_sha256),
        seed=seed,
    )


def build_app(
    *,
    manifest_path: Path,
    gallery_root: Path,
    prompt_path: Path,
    out_path: Path,
    seed: int,
) -> int:
    """Build the self-contained adjudication HTML and write it to ``out_path``.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Path to ``gallery_manifest.csv``.
    gallery_root : pathlib.Path
        Gallery root used to resolve repo-relative ``image_path`` values.
    prompt_path : pathlib.Path
        Path to the adjudication system prompt (for the criteria panel).
    out_path : pathlib.Path
        Destination HTML file.
    seed : int
        Deterministic shuffle / synthetic-trace seed.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    logger.info("loading cards from {} (seed={})", manifest_path, seed)
    cards = load_cards(manifest_path, gallery_root, seed)
    logger.info("embedded {} card PNGs", len(cards))

    examples = build_reference_examples(seed)
    logger.info("rendered {} synthetic reference examples", len(examples))

    criteria_text = _criteria_text(prompt_path)
    manifest_sha256 = file_sha256(manifest_path)

    doc = _build_html(
        cards,
        examples,
        criteria_text=criteria_text,
        manifest_sha256=manifest_sha256,
        seed=seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("wrote {} ({:.1f} MB, {} cards)", out_path, size_mb, len(cards))
    return 0


# ---------------------------------------------------------------------------
# HTML template. Kept as a module constant so _build_html stays readable. Curly
# braces that are literal CSS/JS are doubled so str.format leaves them intact;
# the only single-brace substitutions are the named fields below.
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PI trace adjudication</title>
<style>
  :root {{
    --bg: #f4f6f8; --panel: #ffffff; --ink: #1a1a1a; --muted: #5a6672;
    --line: #d3dae1; --accent: #1f6feb; --present: #1a7f37; --absent: #8a8f98;
    --indet: #9a6700; --good: #1a7f37;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 15px; line-height: 1.45;
  }}
  header {{
    position: sticky; top: 0; z-index: 10; background: var(--panel);
    border-bottom: 1px solid var(--line); padding: 10px 18px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }}
  header h1 {{ font-size: 16px; margin: 0; font-weight: 600; }}
  .progress-wrap {{ flex: 1 1 320px; min-width: 220px; }}
  .progress-bar {{
    height: 10px; background: #e6eaee; border-radius: 6px; overflow: hidden;
  }}
  .progress-fill {{ height: 100%; width: 0%; background: var(--good); transition: width .15s; }}
  .progress-text {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}
  .saved {{
    font-size: 12px; color: var(--good); opacity: 0; transition: opacity .3s;
  }}
  .saved.show {{ opacity: 1; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 16px 18px 80px; }}
  details.refpanel {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    margin-bottom: 18px; padding: 4px 14px;
  }}
  details.refpanel > summary {{
    cursor: pointer; font-weight: 600; padding: 8px 0; outline: none;
  }}
  .criteria {{
    font-size: 13.5px; color: var(--muted); background: #fafbfc;
    border-left: 3px solid var(--accent); padding: 10px 12px; border-radius: 4px;
    margin: 8px 0 14px;
  }}
  .ref-grid {{
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px;
  }}
  figure.ref-example {{
    margin: 0; background: #fff; border: 1px solid var(--line);
    border-radius: 8px; padding: 8px;
  }}
  figure.ref-example img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
  .ref-title {{ font-weight: 600; font-size: 13px; }}
  figure.ref-example figcaption {{ margin-bottom: 6px; }}
  .ref-caption {{ font-size: 12.5px; color: var(--muted); margin: 6px 2px 2px; }}
  .card-panel {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }}
  .card-index {{
    font-size: 14px; color: var(--muted); margin-bottom: 8px; font-weight: 600;
    letter-spacing: .3px;
  }}
  .card-img-wrap {{ background: #fff; border-radius: 8px; }}
  .card-img-wrap img {{ width: 100%; height: auto; display: block; }}
  .controls {{ margin-top: 16px; display: flex; flex-direction: column; gap: 14px; }}
  .control-row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .control-label {{
    font-size: 12px; color: var(--muted); width: 100px; flex: 0 0 100px;
    text-transform: uppercase; letter-spacing: .5px;
  }}
  .btngroup {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  button.call, button.conf {{
    font: inherit; cursor: pointer; border: 1px solid var(--line);
    background: #fff; color: var(--ink); padding: 8px 14px; border-radius: 8px;
    transition: all .1s;
  }}
  button.call .key, button.conf .key {{
    display: inline-block; font-size: 11px; color: var(--muted);
    border: 1px solid var(--line); border-radius: 4px; padding: 0 5px;
    margin-left: 7px;
  }}
  button.call:hover, button.conf:hover {{ border-color: var(--accent); }}
  button.call[data-active="1"] {{ color: #fff; border-color: transparent; }}
  button.call[data-call="occlusion_signature_present"][data-active="1"] {{ background: var(--present); }}
  button.call[data-call="no_occlusion_signature"][data-active="1"] {{ background: var(--absent); }}
  button.call[data-call="indeterminate"][data-active="1"] {{ background: var(--indet); }}
  button.call[data-active="1"] .key {{ color: #fff; border-color: rgba(255,255,255,.5); }}
  button.conf[data-active="1"] {{ background: var(--accent); color: #fff; border-color: transparent; }}
  button.conf[data-active="1"] .key {{ color: #fff; border-color: rgba(255,255,255,.5); }}
  textarea.notes {{
    font: inherit; width: 100%; min-height: 54px; resize: vertical;
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px;
  }}
  .nav {{ display: flex; align-items: center; gap: 10px; margin-top: 18px; flex-wrap: wrap; }}
  .nav button {{
    font: inherit; cursor: pointer; border: 1px solid var(--line);
    background: #fff; padding: 8px 14px; border-radius: 8px;
  }}
  .nav button:hover {{ border-color: var(--accent); }}
  .nav .spacer {{ flex: 1; }}
  .toggle {{ font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 6px; }}
  .jumpbox {{ font: inherit; width: 70px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; }}
  .export-row {{
    margin-top: 22px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  }}
  .export-row button {{
    font: inherit; cursor: pointer; border: 1px solid var(--accent);
    background: var(--accent); color: #fff; padding: 9px 16px; border-radius: 8px;
  }}
  .export-row button.secondary {{ background: #fff; color: var(--accent); }}
  .banner {{
    display: none; background: #fff8c5; border: 1px solid #d4a72c;
    color: #6b5900; padding: 10px 14px; border-radius: 8px; margin-top: 12px;
    font-size: 13.5px;
  }}
  .banner.show {{ display: block; }}
  footer {{
    max-width: 980px; margin: 0 auto; padding: 0 18px 40px; color: var(--muted);
    font-size: 11.5px;
  }}
  kbd {{
    font-family: ui-monospace, Menlo, monospace; font-size: 11px;
    border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; background: #fafbfc;
  }}
</style>
</head>
<body>
<header>
  <h1>PI trace adjudication</h1>
  <div class="progress-wrap">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <div class="progress-text" id="progressText">0 / {n} rated</div>
  </div>
  <span class="saved" id="savedIndicator">saved</span>
</header>

<main>
  <details class="refpanel" id="refpanel">
    <summary>What to look for (reference examples and criteria)</summary>
    <div class="criteria">{criteria_html}</div>
    <div class="ref-grid">
      {examples_html}
    </div>
  </details>

  <div class="card-panel">
    <div class="card-index" id="cardIndex">card 1 of {n}</div>
    <div class="card-img-wrap"><img id="cardImg" alt="perfusion-index trace" /></div>

    <div class="controls">
      <div class="control-row">
        <span class="control-label">Call</span>
        <div class="btngroup" id="callGroup">
          <button class="call" data-call="occlusion_signature_present">occlusion signature present<span class="key">P</span></button>
          <button class="call" data-call="no_occlusion_signature">no occlusion signature<span class="key">N</span></button>
          <button class="call" data-call="indeterminate">indeterminate<span class="key">I</span></button>
        </div>
      </div>
      <div class="control-row">
        <span class="control-label">Confidence</span>
        <div class="btngroup" id="confGroup">
          <button class="conf" data-conf="low">low<span class="key">1</span></button>
          <button class="conf" data-conf="med">med<span class="key">2</span></button>
          <button class="conf" data-conf="high">high<span class="key">3</span></button>
        </div>
      </div>
      <div class="control-row">
        <span class="control-label">Notes</span>
        <textarea class="notes" id="notesBox" placeholder="optional comments"></textarea>
      </div>
    </div>

    <div class="nav">
      <button id="prevBtn">&larr; Prev</button>
      <button id="nextBtn">Next &rarr;</button>
      <button id="nextUnratedBtn">Jump to next unrated</button>
      <span class="spacer"></span>
      <label class="toggle"><input type="checkbox" id="autoAdvance" checked /> auto-advance on call</label>
      <input class="jumpbox" id="jumpBox" type="number" min="1" max="{n}" placeholder="go to #" />
    </div>
  </div>

  <div class="banner" id="unratedBanner"></div>

  <div class="export-row">
    <button id="exportCsvBtn">Export CSV</button>
    <button id="exportJsonBtn" class="secondary">Export JSON (backup)</button>
  </div>
</main>

<footer>
  Blinded reference task. Presentation order is fixed by seed {seed}; manifest
  SHA-256 <code>{manifest_sha256}</code>. Keyboard: <kbd>P</kbd>/<kbd>N</kbd>/<kbd>I</kbd> call,
  <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> confidence, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate.
  Progress autosaves to this browser.
</footer>

<script id="cardsData" type="application/json">{cards_json}</script>
<script id="blindMap" type="application/json">{blind_map_json}</script>
<script id="appConfig" type="application/json">{config_json}</script>
<script>
(function () {{
  "use strict";
  var CARDS = JSON.parse(document.getElementById("cardsData").textContent);
  var BLIND_MAP = JSON.parse(document.getElementById("blindMap").textContent);
  var CFG = JSON.parse(document.getElementById("appConfig").textContent);
  var N = CFG.nCards;
  var KEY = CFG.storageKey;

  // state: blind_index (string) -> {{ call, confidence, notes }}
  var state = {{}};
  try {{
    var raw = localStorage.getItem(KEY);
    if (raw) state = JSON.parse(raw) || {{}};
  }} catch (e) {{ state = {{}}; }}

  var pos = 0; // 0-based index into CARDS
  // Resume at first unrated card on load (falls back to 0 if all rated).
  (function () {{
    for (var k = 0; k < CARDS.length; k++) {{
      var rec = state[String(CARDS[k].i)];
      if (!rec || !rec.call) {{ pos = k; return; }}
    }}
    pos = 0;
  }})();

  var elImg = document.getElementById("cardImg");
  var elIndex = document.getElementById("cardIndex");
  var elNotes = document.getElementById("notesBox");
  var elFill = document.getElementById("progressFill");
  var elPText = document.getElementById("progressText");
  var elSaved = document.getElementById("savedIndicator");
  var elBanner = document.getElementById("unratedBanner");
  var elAuto = document.getElementById("autoAdvance");
  var callBtns = Array.prototype.slice.call(document.querySelectorAll("button.call"));
  var confBtns = Array.prototype.slice.call(document.querySelectorAll("button.conf"));

  var savedTimer = null;
  function flashSaved() {{
    elSaved.classList.add("show");
    if (savedTimer) clearTimeout(savedTimer);
    savedTimer = setTimeout(function () {{ elSaved.classList.remove("show"); }}, 900);
  }}

  function persist() {{
    try {{ localStorage.setItem(KEY, JSON.stringify(state)); flashSaved(); }} catch (e) {{}}
  }}

  function ratedCount() {{
    var c = 0;
    for (var k = 0; k < CARDS.length; k++) {{
      var rec = state[String(CARDS[k].i)];
      if (rec && rec.call) c++;
    }}
    return c;
  }}

  function updateProgress() {{
    var done = ratedCount();
    var pct = N ? Math.round((done / N) * 100) : 0;
    elFill.style.width = pct + "%";
    elPText.textContent = done + " / " + N + " rated, " + (N - done) + " unrated (" + pct + "%)";
  }}

  function rec(idx) {{
    var bi = String(CARDS[idx].i);
    if (!state[bi]) state[bi] = {{ call: "", confidence: "", notes: "" }};
    return state[bi];
  }}

  function renderCard() {{
    var card = CARDS[pos];
    elIndex.textContent = "card " + card.i + " of " + N;
    elImg.src = "data:image/png;base64," + card.img;
    var r = rec(pos);
    callBtns.forEach(function (b) {{
      b.setAttribute("data-active", b.getAttribute("data-call") === r.call ? "1" : "0");
    }});
    confBtns.forEach(function (b) {{
      b.setAttribute("data-active", b.getAttribute("data-conf") === r.confidence ? "1" : "0");
    }});
    elNotes.value = r.notes || "";
    updateProgress();
  }}

  function go(delta) {{
    pos = Math.min(CARDS.length - 1, Math.max(0, pos + delta));
    renderCard();
  }}

  function goTo(idx1) {{
    var idx = idx1 - 1;
    if (idx >= 0 && idx < CARDS.length) {{ pos = idx; renderCard(); }}
  }}

  function nextUnrated() {{
    for (var step = 1; step <= CARDS.length; step++) {{
      var k = (pos + step) % CARDS.length;
      var rr = state[String(CARDS[k].i)];
      if (!rr || !rr.call) {{ pos = k; renderCard(); return; }}
    }}
  }}

  function setCall(call) {{
    rec(pos).call = call;
    persist();
    renderCard();
    if (elAuto.checked) {{ setTimeout(function () {{ go(1); }}, 120); }}
  }}

  function setConf(conf) {{
    rec(pos).confidence = conf;
    persist();
    renderCard();
  }}

  callBtns.forEach(function (b) {{
    b.addEventListener("click", function () {{ setCall(b.getAttribute("data-call")); }});
  }});
  confBtns.forEach(function (b) {{
    b.addEventListener("click", function () {{ setConf(b.getAttribute("data-conf")); }});
  }});

  elNotes.addEventListener("input", function () {{ rec(pos).notes = elNotes.value; }});
  elNotes.addEventListener("change", function () {{ rec(pos).notes = elNotes.value; persist(); }});

  document.getElementById("prevBtn").addEventListener("click", function () {{ go(-1); }});
  document.getElementById("nextBtn").addEventListener("click", function () {{ go(1); }});
  document.getElementById("nextUnratedBtn").addEventListener("click", nextUnrated);
  document.getElementById("jumpBox").addEventListener("change", function (e) {{
    var v = parseInt(e.target.value, 10);
    if (!isNaN(v)) goTo(v);
  }});

  var CALL_KEYS = {{ p: "occlusion_signature_present", n: "no_occlusion_signature", i: "indeterminate" }};
  var CONF_KEYS = {{ "1": "low", "2": "med", "3": "high" }};
  document.addEventListener("keydown", function (e) {{
    if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT")) return;
    var k = e.key.toLowerCase();
    if (CALL_KEYS[k]) {{ e.preventDefault(); setCall(CALL_KEYS[k]); }}
    else if (CONF_KEYS[e.key]) {{ e.preventDefault(); setConf(CONF_KEYS[e.key]); }}
    else if (e.key === "ArrowLeft") {{ e.preventDefault(); go(-1); }}
    else if (e.key === "ArrowRight") {{ e.preventDefault(); go(1); }}
  }});

  function csvEscape(s) {{
    s = (s == null) ? "" : String(s);
    if (/[",\\n\\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }}

  function buildCsv() {{
    var lines = ["card_id,call,confidence,notes"];
    // Emit in blind-index order for a stable, reviewable file.
    var indices = CARDS.map(function (c) {{ return c.i; }}).sort(function (a, b) {{ return a - b; }});
    indices.forEach(function (bi) {{
      var cid = BLIND_MAP[String(bi)];
      var r = state[String(bi)] || {{}};
      lines.push([csvEscape(cid), csvEscape(r.call || ""), csvEscape(r.confidence || ""), csvEscape(r.notes || "")].join(","));
    }});
    return lines.join("\\n") + "\\n";
  }}

  function download(filename, text, mime) {{
    var blob = new Blob([text], {{ type: mime }});
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    setTimeout(function () {{ URL.revokeObjectURL(url); }}, 1000);
  }}

  function checkUnratedBanner() {{
    var done = ratedCount();
    var miss = N - done;
    if (miss > 0) {{
      elBanner.textContent = "Heads up: " + miss + " of " + N + " cards are still unrated. The export includes them with an empty call.";
      elBanner.classList.add("show");
    }} else {{
      elBanner.classList.remove("show");
    }}
  }}

  document.getElementById("exportCsvBtn").addEventListener("click", function () {{
    checkUnratedBanner();
    download("reader_adjudication_export.csv", buildCsv(), "text/csv");
  }});
  document.getElementById("exportJsonBtn").addEventListener("click", function () {{
    checkUnratedBanner();
    var payload = {{ config: CFG, blindMap: BLIND_MAP, state: state }};
    download("reader_adjudication_state.json", JSON.stringify(payload, null, 2), "application/json");
  }});

  renderCard();
}})();
</script>
</body>
</html>
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/gallery/gallery_manifest.csv"),
        help="Path to gallery_manifest.csv.",
    )
    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=Path("results/gallery"),
        help="Gallery root for resolving repo-relative image_path values.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("prompts/adjudicate_system.txt"),
        help="Adjudication system prompt (criteria panel source).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/gallery/adjudication.html"),
        help="Destination HTML file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"Deterministic shuffle seed (default: GLOBAL_SEED={GLOBAL_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on input errors).
    """
    args = _parse_args(argv)
    if not args.manifest.exists():
        logger.error("manifest not found: {}", args.manifest)
        return 2
    return build_app(
        manifest_path=args.manifest,
        gallery_root=args.gallery_root,
        prompt_path=args.prompt,
        out_path=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
