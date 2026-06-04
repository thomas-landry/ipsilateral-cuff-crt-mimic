"""Stage the blinded second-pass re-read of the 568-card gallery (step 60).

This is the staging half of the second-pass reader re-adjudication. The
principal reader has already adjudicated all 568 gallery cards once (pass 1,
``results/gallery/reader_form_blinded.csv``). To quantify intra-rater
reliability, the reader re-reads cards cold under a fresh blinding. This script
prepares that re-read so the blinding holds even if the reader inspects the
staged files.

Two modes
---------
* Full mode (default, ``--sample`` omitted): stage all 568 cards. This is the
  original behavior and is unchanged.
* Sample mode (``--sample N``): stage a stratified random sample of ``N`` cards
  drawn proportionally across the gallery strata, for a lighter re-read whose
  only purpose is to MEASURE intra-rater reliability. Pass 1 stays the canonical
  reference and is never touched. The sample is deterministic for a given
  ``--sample`` and ``--sample-seed``.

What it does
------------
1. Copies the selected card PNGs from ``results/gallery/<stratum>/<card_id>.png``
   into a FLAT directory ``results/gallery_readjud_blind/`` renamed to opaque
   identifiers ``blind_0001.png`` .. ``blind_NNNN.png`` (NNNN = 0568 in full
   mode, or the realized sample size in sample mode).
2. Assigns the opaque ids in a re-read-seeded shuffled order that interleaves
   the strata so that no two consecutive cards share a stratum (anti-clumping).
   The shuffle uses a re-read seed (default 20260601) DISTINCT from the
   pass-1 / project ``GLOBAL_SEED`` (20260426), so the display order differs
   from pass 1 yet is fully reproducible. In sample mode the proportional
   per-stratum draw uses a separate documented sample seed (default 20260602).
3. Writes a HIDDEN de-blinding key ``_blind_map.csv`` (``blind_id, card_id,
   stratum``) into the same directory. The re-read HTML never loads this file;
   it is the only link from an opaque ``blind_id`` back to its ``card_id`` and
   stratum, and it is what the recompute harness (``scripts/61``) uses to
   de-blind the pass-2 export.
4. Verifies the planned count copied, a strict 1:1 ``blind_id`` <-> ``card_id``
   mapping, no ``card_id`` collisions, and that every staged PNG is
   byte-identical to its gallery source. Logs the full verification.

Blinding rationale
-------------------
The pass-1 tool kept the de-blinding map inside the HTML (in a JSON blob). A
determined reader could read that blob. This staging design removes the key
from the reader-facing surface entirely: the opaque PNG filenames carry no
stratum prefix (the gallery ``A-``/``B-``/``C-`` prefix is dropped), and the
``card_id`` -> ``stratum`` key lives only in ``_blind_map.csv``, which the HTML
never references. Even with browser dev tools open, the reader sees only
``blind_####.png``.

Reproducibility
---------------
Deterministic: a fixed re-read seed and a fixed manifest produce the same
opaque order and the same files on every run. The script is idempotent: it
clears and rebuilds the blind directory each run, and it never modifies any
canonical pass-1 artifact (the manifest, the reader form, or the gallery PNGs).
No new randomness is introduced beyond the single seeded shuffle.

Examples
--------
Stage all 568 cards against the canonical gallery::

    uv run python scripts/60_stage_blinded_reread.py

Stage a stratified 150-card sample for a reliability re-read::

    uv run python scripts/60_stage_blinded_reread.py --sample 150

Custom source / output / seed::

    uv run python scripts/60_stage_blinded_reread.py \\
        --manifest results/gallery/gallery_manifest.csv \\
        --gallery-root results/gallery \\
        --out-dir results/gallery_readjud_blind \\
        --seed 20260601
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

# Re-read shuffle seed. DISTINCT from the project GLOBAL_SEED (20260426) so the
# second-pass display order differs from pass 1, while staying reproducible.
REREAD_SEED = 20260601

# Sample-draw seed for sample mode (--sample N). DISTINCT from both REREAD_SEED
# and the project GLOBAL_SEED so the stratified card selection is its own
# reproducible stream, independent of the display-order shuffle.
SAMPLE_SEED = 20260602

# Width of the zero-padded opaque blind id (blind_0001 .. blind_0568).
_BLIND_ID_WIDTH = 4

# Columns of the hidden de-blinding key, in a stable order.
_BLIND_MAP_COLUMNS = ("blind_id", "card_id", "stratum")


@dataclass(frozen=True)
class StagedCard:
    """One card after blinded staging.

    Attributes
    ----------
    blind_id : str
        Opaque reader-facing id, e.g. ``blind_0001``. Carries no stratum or
        ``card_id`` information.
    card_id : str
        The canonical gallery id (e.g. ``A-b1769fc4ee128967``). Stored only in
        the hidden ``_blind_map.csv``; never exposed to the reader.
    stratum : str
        The detector stratum (e.g. ``detector_positive``). Hidden key only.
    src_path : pathlib.Path
        Absolute path to the source gallery PNG.
    """

    blind_id: str
    card_id: str
    stratum: str
    src_path: Path


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Parameters
    ----------
    path : pathlib.Path
        File to digest.

    Returns
    -------
    str
        Lowercase hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_png(image_path: str, repo_root: Path) -> Path:
    """Resolve a manifest ``image_path`` to an existing PNG on disk.

    Manifest ``image_path`` values are repo-relative and already include the
    ``results/gallery/`` prefix. Paths that exist as given are used directly;
    otherwise they are resolved against ``repo_root``.

    Parameters
    ----------
    image_path : str
        Repo-relative image path from the manifest.
    repo_root : pathlib.Path
        Repository root the relative path resolves against.

    Returns
    -------
    pathlib.Path
        An existing PNG path.

    Raises
    ------
    FileNotFoundError
        If neither the literal nor the repo-root-anchored path exists.
    """
    candidate = Path(image_path)
    if candidate.exists():
        return candidate
    anchored = repo_root / image_path
    if anchored.exists():
        return anchored
    raise FileNotFoundError(f"card PNG not found: {image_path}")


def interleave_strata(
    card_ids: list[str],
    strata: list[str],
    seed: int,
) -> list[int]:
    """Return a shuffled order of indices that spreads strata apart.

    The order is produced in two deterministic steps for the given ``seed``:

    1. Within each stratum, shuffle the member indices with
       ``numpy.random.default_rng(seed)``.
    2. Interleave the shuffled per-stratum queues by repeatedly drawing the
       next card from whichever stratum is currently the most over-represented
       relative to how many of its cards remain. This greedy spreading avoids
       long same-stratum runs without needing a second RNG stream.

    The result is a permutation of ``range(len(card_ids))``. For inputs with
    more than one stratum, no two adjacent positions share a stratum unless one
    stratum holds a strict majority of the remaining cards (in which case some
    adjacency is unavoidable; with this gallery's 268/200/100 split no such
    forced adjacency occurs).

    Parameters
    ----------
    card_ids : list[str]
        Source-ordered card ids (manifest row order).
    strata : list[str]
        Stratum label per card; same length and order as ``card_ids``.
    seed : int
        RNG seed for the within-stratum shuffle. Use the re-read seed.

    Returns
    -------
    list[int]
        A permutation of ``range(len(card_ids))`` in blinded display order.

    Raises
    ------
    ValueError
        If ``card_ids`` and ``strata`` differ in length or are empty.
    """
    if len(card_ids) != len(strata):
        raise ValueError(
            f"card_ids and strata must match in length; got {len(card_ids)} "
            f"and {len(strata)}."
        )
    if not card_ids:
        raise ValueError("card_ids must be non-empty.")

    rng = np.random.default_rng(seed)

    # Group source indices by stratum, preserving first-seen stratum order so
    # the result is deterministic for a given input order and seed.
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(strata):
        by_stratum[s].append(i)

    # Shuffle each stratum's indices independently, then treat each as a queue.
    queues: dict[str, list[int]] = {}
    for s, idxs in by_stratum.items():
        arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(arr)
        queues[s] = arr.tolist()

    total = len(card_ids)
    order: list[int] = []
    last_stratum: str | None = None

    # Greedy spreading: at each step pick, among strata that still have cards
    # AND differ from the previous pick, the one with the largest remaining
    # count. Fall back to the largest remaining stratum if the only option is
    # the previous one (a forced adjacency).
    for _ in range(total):
        candidates = [s for s, q in queues.items() if q]
        non_repeat = [s for s in candidates if s != last_stratum]
        pool = non_repeat if non_repeat else candidates
        # Largest remaining queue first; ties broken by stratum name for
        # determinism.
        chosen = max(pool, key=lambda s: (len(queues[s]), s))
        order.append(queues[chosen].pop(0))
        last_stratum = chosen

    return order


def stratified_sample(
    card_ids: list[str],
    strata: list[str],
    sample_size: int,
    sample_seed: int,
) -> list[int]:
    """Select a proportional stratified random sample of card indices.

    The per-stratum allotment is the largest-remainder (Hamilton) apportionment
    of ``sample_size`` across the strata in proportion to each stratum's share
    of the full set, so the realized total equals ``sample_size`` exactly while
    each stratum's count stays as close as possible to its proportional target.
    Ties in the remainder are broken by stratum name for determinism. Within
    each stratum the allotted number of member indices is drawn without
    replacement using ``numpy.random.default_rng(sample_seed)``.

    Parameters
    ----------
    card_ids : list[str]
        Source-ordered card ids (manifest row order).
    strata : list[str]
        Stratum label per card; same length and order as ``card_ids``.
    sample_size : int
        Desired number of cards to sample. Must be in ``[1, len(card_ids)]``.
    sample_seed : int
        RNG seed for the within-stratum draw. Use the sample seed.

    Returns
    -------
    list[int]
        Source indices of the sampled cards, sorted ascending. (Display order
        is decided later by :func:`interleave_strata`.)

    Raises
    ------
    ValueError
        If lengths mismatch, the inputs are empty, or ``sample_size`` is outside
        ``[1, len(card_ids)]``.
    """
    if len(card_ids) != len(strata):
        raise ValueError(
            f"card_ids and strata must match in length; got {len(card_ids)} "
            f"and {len(strata)}."
        )
    if not card_ids:
        raise ValueError("card_ids must be non-empty.")
    total = len(card_ids)
    if not 1 <= sample_size <= total:
        raise ValueError(
            f"sample_size must be in [1, {total}]; got {sample_size}."
        )

    # Group source indices by stratum in first-seen order for determinism.
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(strata):
        by_stratum[s].append(i)

    # Largest-remainder apportionment of sample_size across strata.
    names = list(by_stratum.keys())
    sizes = {s: len(by_stratum[s]) for s in names}
    raw = {s: sample_size * sizes[s] / total for s in names}
    floors = {s: int(np.floor(raw[s])) for s in names}
    allotted = sum(floors.values())
    leftover = sample_size - allotted
    # Distribute leftover seats to the largest fractional remainders; break ties
    # by stratum name so the result is fully deterministic.
    remainders = sorted(
        names, key=lambda s: (raw[s] - floors[s], s), reverse=True
    )
    alloc = dict(floors)
    for s in remainders[:leftover]:
        alloc[s] += 1
    # A stratum can never be allotted more than it holds (proportional shares
    # are bounded by the stratum size), but guard defensively.
    for s in names:
        alloc[s] = min(alloc[s], sizes[s])

    rng = np.random.default_rng(sample_seed)
    chosen: list[int] = []
    for s in names:
        idxs = np.asarray(by_stratum[s], dtype=np.int64)
        rng.shuffle(idxs)
        chosen.extend(idxs[: alloc[s]].tolist())
    return sorted(chosen)


def load_staging_plan(
    manifest_path: Path,
    gallery_root: Path,
    seed: int,
    *,
    sample_size: int | None = None,
    sample_seed: int = SAMPLE_SEED,
) -> list[StagedCard]:
    """Build the ordered list of cards to stage from the gallery manifest.

    Reads the manifest, resolves each PNG, optionally restricts to a stratified
    random sample of ``sample_size`` cards, computes the interleaved blinded
    order for ``seed`` over the selected cards, and assigns opaque ``blind_id``
    strings in that order.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Path to ``gallery_manifest.csv`` (needs ``card_id``, ``stratum``,
        ``image_path``).
    gallery_root : pathlib.Path
        Gallery root used to resolve repo-relative ``image_path`` values.
    seed : int
        Re-read display-order shuffle seed.
    sample_size : int or None, optional
        If given, stage only a proportional stratified random sample of this
        many cards (see :func:`stratified_sample`). ``None`` (default) stages
        every card.
    sample_seed : int, optional
        Seed for the stratified draw when ``sample_size`` is given. Defaults to
        :data:`SAMPLE_SEED`.

    Returns
    -------
    list[StagedCard]
        One :class:`StagedCard` per selected card, in blinded display order,
        with ``blind_id`` assigned ``blind_0001`` .. ``blind_NNNN``.

    Raises
    ------
    FileNotFoundError
        If the manifest or any referenced PNG is missing.
    ValueError
        If required columns are absent, ``card_id`` is not unique, or
        ``sample_size`` is out of range.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    df = pl.read_csv(manifest_path, infer_schema_length=20000)
    required = {"card_id", "stratum", "image_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    card_ids = [str(c) for c in df.get_column("card_id").to_list()]
    strata = [str(s) for s in df.get_column("stratum").to_list()]
    image_paths = [str(p) for p in df.get_column("image_path").to_list()]
    if len(set(card_ids)) != len(card_ids):
        raise ValueError("manifest card_id column is not unique")

    repo_root = gallery_root.resolve().parent.parent
    src_paths = [_resolve_png(p, repo_root) for p in image_paths]

    if sample_size is not None:
        keep = stratified_sample(card_ids, strata, sample_size, sample_seed)
        card_ids = [card_ids[i] for i in keep]
        strata = [strata[i] for i in keep]
        src_paths = [src_paths[i] for i in keep]
        per_stratum = {s: strata.count(s) for s in sorted(set(strata))}
        logger.info(
            "stratified sample: requested {} -> realized {} cards; "
            "per-stratum sizes {} (sample_seed={})",
            sample_size,
            len(card_ids),
            per_stratum,
            sample_seed,
        )

    order = interleave_strata(card_ids, strata, seed)

    staged: list[StagedCard] = []
    for position, src_idx in enumerate(order, start=1):
        blind_id = f"blind_{position:0{_BLIND_ID_WIDTH}d}"
        staged.append(
            StagedCard(
                blind_id=blind_id,
                card_id=card_ids[src_idx],
                stratum=strata[src_idx],
                src_path=src_paths[src_idx],
            )
        )
    return staged


def verify_staging(staged: list[StagedCard], out_dir: Path) -> None:
    """Verify the staged directory against the plan; raise on any defect.

    Checks performed and logged:

    * count of staged PNGs equals the plan length,
    * every ``blind_id`` is unique and matches ``blind_####`` numbering,
    * every ``card_id`` is unique (no collisions / no card staged twice),
    * each staged PNG is byte-identical (SHA-256) to its gallery source.

    Parameters
    ----------
    staged : list[StagedCard]
        The staging plan that was just executed.
    out_dir : pathlib.Path
        Directory the PNGs and ``_blind_map.csv`` were written to.

    Raises
    ------
    RuntimeError
        If any check fails.
    """
    n = len(staged)
    staged_pngs = sorted(out_dir.glob("blind_*.png"))
    if len(staged_pngs) != n:
        raise RuntimeError(
            f"staged PNG count {len(staged_pngs)} != plan count {n}"
        )

    blind_ids = [c.blind_id for c in staged]
    card_ids = [c.card_id for c in staged]
    if len(set(blind_ids)) != n:
        raise RuntimeError("blind_id values are not unique")
    if len(set(card_ids)) != n:
        raise RuntimeError("card_id collision: a card was staged more than once")
    expected_ids = {f"blind_{i:0{_BLIND_ID_WIDTH}d}" for i in range(1, n + 1)}
    if set(blind_ids) != expected_ids:
        raise RuntimeError("blind_id numbering is not a contiguous 1..N range")

    mismatches = 0
    for card in staged:
        dest = out_dir / f"{card.blind_id}.png"
        if not dest.exists():
            raise RuntimeError(f"missing staged PNG: {dest}")
        if _sha256_of_file(dest) != _sha256_of_file(card.src_path):
            mismatches += 1
    if mismatches:
        raise RuntimeError(f"{mismatches} staged PNGs are not byte-identical to source")

    logger.info(
        "verification passed: {} PNGs staged 1:1, blind_id and card_id unique, "
        "all bytes match source",
        n,
    )


def _adjacent_same_stratum(staged: list[StagedCard]) -> int:
    """Count consecutive positions sharing a stratum (interleaving quality).

    Parameters
    ----------
    staged : list[StagedCard]
        Staged cards in display order.

    Returns
    -------
    int
        Number of adjacent (i, i+1) pairs with the same stratum.
    """
    return sum(
        1
        for a, b in zip(staged, staged[1:], strict=False)
        if a.stratum == b.stratum
    )


def stage_reread(
    *,
    manifest_path: Path,
    gallery_root: Path,
    out_dir: Path,
    seed: int,
    sample_size: int | None = None,
    sample_seed: int = SAMPLE_SEED,
) -> int:
    """Stage the blinded re-read: copy PNGs, write the key, verify.

    Idempotent: any existing ``blind_*.png`` and ``_blind_map.csv`` in
    ``out_dir`` are removed before rebuilding so a re-run yields a clean,
    deterministic directory.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Path to ``gallery_manifest.csv``.
    gallery_root : pathlib.Path
        Gallery root for resolving repo-relative ``image_path`` values.
    out_dir : pathlib.Path
        Destination flat directory for the opaque PNGs and the hidden key.
    seed : int
        Re-read display-order shuffle seed.
    sample_size : int or None, optional
        If given, stage a proportional stratified random sample of this many
        cards instead of all cards. ``None`` (default) keeps the full-stage
        behavior.
    sample_seed : int, optional
        Seed for the stratified draw. Defaults to :data:`SAMPLE_SEED`.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    mode = "full (all cards)" if sample_size is None else f"sample ({sample_size} cards)"
    logger.info(
        "staging blinded re-read from {} in {} mode (seed={}, distinct from GLOBAL_SEED)",
        manifest_path,
        mode,
        seed,
    )
    staged = load_staging_plan(
        manifest_path,
        gallery_root,
        seed,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    logger.info("planned {} cards in interleaved blinded order", len(staged))

    out_dir.mkdir(parents=True, exist_ok=True)
    # Idempotency: clear prior staged outputs so re-runs are deterministic.
    for stale in out_dir.glob("blind_*.png"):
        stale.unlink()
    stale_map = out_dir / "_blind_map.csv"
    if stale_map.exists():
        stale_map.unlink()

    for card in staged:
        dest = out_dir / f"{card.blind_id}.png"
        shutil.copyfile(card.src_path, dest)

    # Hidden de-blinding key. The HTML never references this file.
    blind_map = pl.DataFrame(
        {
            "blind_id": [c.blind_id for c in staged],
            "card_id": [c.card_id for c in staged],
            "stratum": [c.stratum for c in staged],
        }
    ).select(list(_BLIND_MAP_COLUMNS))
    blind_map.write_csv(stale_map)
    logger.info("wrote hidden de-blinding key {} ({} rows)", stale_map, blind_map.height)

    verify_staging(staged, out_dir)

    # Write the reader-facing HTML. It references blind_####.png by relative
    # path only and never loads _blind_map.csv, so the map cannot leak.
    html_path = out_dir / "reread.html"
    html_path.write_text(_build_reread_html([c.blind_id for c in staged]), encoding="utf-8")
    logger.info("wrote blinded re-read UI {}", html_path)

    n_adj = _adjacent_same_stratum(staged)
    logger.info(
        "interleaving: {} of {} adjacent pairs share a stratum",
        n_adj,
        max(len(staged) - 1, 0),
    )
    logger.info("blinded re-read staged at {}", out_dir)
    return 0


def _build_reread_html(blind_ids: list[str]) -> str:
    """Assemble the self-contained blinded re-read HTML document.

    The page loads each card as ``blind_####.png`` by relative path. The only
    data embedded in the DOM is the ordered list of opaque ``blind_id`` strings;
    no ``card_id``, stratum, machine call, pass-1 call, or filename prefix is
    present. The de-blinding key (``_blind_map.csv``) is never referenced.

    Parameters
    ----------
    blind_ids : list[str]
        Opaque ids in display order (``blind_0001`` .. ``blind_NNNN``).

    Returns
    -------
    str
        The complete HTML document as a single string.
    """
    n = len(blind_ids)
    ids_json = json.dumps(blind_ids, separators=(",", ":"))
    return _REREAD_HTML_TEMPLATE.format(n=n, ids_json=ids_json, seed=REREAD_SEED)


# ---------------------------------------------------------------------------
# Re-read HTML template. Literal CSS/JS braces are doubled so str.format leaves
# them intact; the single-brace fields are n, ids_json, seed. The page loads
# blind_####.png by relative path and embeds ONLY the ordered blind_id list.
# It never embeds card_id, stratum, machine calls, or pass-1 calls.
# ---------------------------------------------------------------------------
_REREAD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PI trace re-read (pass 2)</title>
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
  .progress-bar {{ height: 10px; background: #e6eaee; border-radius: 6px; overflow: hidden; }}
  .progress-fill {{ height: 100%; width: 0%; background: var(--good); transition: width .15s; }}
  .progress-text {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}
  .saved {{ font-size: 12px; color: var(--good); opacity: 0; transition: opacity .3s; }}
  .saved.show {{ opacity: 1; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 16px 18px 80px; }}
  .reminder {{
    background: #eef4ff; border: 1px solid #c5d8ff; color: #1b3a6b;
    border-radius: 8px; padding: 9px 13px; margin-bottom: 16px; font-size: 13.5px;
  }}
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
    border: 1px solid var(--line); border-radius: 4px; padding: 0 5px; margin-left: 7px;
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
  .export-row {{ margin-top: 22px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .export-row button {{
    font: inherit; cursor: pointer; border: 1px solid var(--accent);
    background: var(--accent); color: #fff; padding: 9px 16px; border-radius: 8px;
  }}
  .export-row button.secondary {{ background: #fff; color: var(--accent); }}
  .banner {{
    display: none; background: #fff8c5; border: 1px solid #d4a72c;
    color: #6b5900; padding: 10px 14px; border-radius: 8px; margin-top: 12px; font-size: 13.5px;
  }}
  .banner.show {{ display: block; }}
  footer {{
    max-width: 980px; margin: 0 auto; padding: 0 18px 40px; color: var(--muted); font-size: 11.5px;
  }}
  kbd {{
    font-family: ui-monospace, Menlo, monospace; font-size: 11px;
    border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; background: #fafbfc;
  }}
</style>
</head>
<body>
<header>
  <h1>PI trace re-read (pass 2)</h1>
  <div class="progress-wrap">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <div class="progress-text" id="progressText">0 / {n} rated</div>
  </div>
  <span class="saved" id="savedIndicator">saved</span>
</header>

<main>
  <div class="reminder">
    Cold re-read. Judge each trace on morphology alone. You are not shown your
    earlier calls, any detector or model output, or which group a card came
    from. Look for one sustained deep drop in perfusion toward zero, lasting
    several seconds, followed by a single graded recovery back toward baseline.
  </div>

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
          <button class="conf" data-conf="medium">medium<span class="key">2</span></button>
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
  Blinded second-pass re-read. Presentation order is fixed by re-read seed {seed}.
  Keyboard: <kbd>P</kbd>/<kbd>N</kbd>/<kbd>I</kbd> call, <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd>
  confidence, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate. Progress autosaves to this browser.
</footer>

<script id="blindIds" type="application/json">{ids_json}</script>
<script>
(function () {{
  "use strict";
  var IDS = JSON.parse(document.getElementById("blindIds").textContent);
  var N = IDS.length;
  var KEY = "cuffcrt_reread_pass2_v1";

  // state: blind_id -> {{ call, confidence, notes }}
  var state = {{}};
  try {{
    var raw = localStorage.getItem(KEY);
    if (raw) state = JSON.parse(raw) || {{}};
  }} catch (e) {{ state = {{}}; }}

  var pos = 0; // 0-based index into IDS
  (function () {{
    for (var k = 0; k < IDS.length; k++) {{
      var rec0 = state[IDS[k]];
      if (!rec0 || !rec0.call) {{ pos = k; return; }}
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
    for (var k = 0; k < IDS.length; k++) {{
      var r = state[IDS[k]];
      if (r && r.call) c++;
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
    var bid = IDS[idx];
    if (!state[bid]) state[bid] = {{ call: "", confidence: "", notes: "" }};
    return state[bid];
  }}
  function renderCard() {{
    var bid = IDS[pos];
    elIndex.textContent = "card " + (pos + 1) + " of " + N;
    elImg.src = bid + ".png";
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
    pos = Math.min(IDS.length - 1, Math.max(0, pos + delta));
    renderCard();
  }}
  function goTo(idx1) {{
    var idx = idx1 - 1;
    if (idx >= 0 && idx < IDS.length) {{ pos = idx; renderCard(); }}
  }}
  function nextUnrated() {{
    for (var step = 1; step <= IDS.length; step++) {{
      var k = (pos + step) % IDS.length;
      var rr = state[IDS[k]];
      if (!rr || !rr.call) {{ pos = k; renderCard(); return; }}
    }}
  }}
  function setCall(call) {{
    var r = rec(pos);
    r.call = call;
    r.utc = new Date().toISOString();
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
  var CONF_KEYS = {{ "1": "low", "2": "medium", "3": "high" }};
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
  function isoUtc() {{ return new Date().toISOString(); }}
  function buildCsv() {{
    var lines = ["blind_id,call,confidence,notes,utc"];
    IDS.forEach(function (bid) {{
      var r = state[bid] || {{}};
      var utc = (r.call) ? (r.utc || isoUtc()) : "";
      lines.push([
        csvEscape(bid), csvEscape(r.call || ""), csvEscape(r.confidence || ""),
        csvEscape(r.notes || ""), csvEscape(utc)
      ].join(","));
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
    var miss = N - ratedCount();
    if (miss > 0) {{
      elBanner.textContent = "Heads up: " + miss + " of " + N + " cards are still unrated. The export includes them with an empty call.";
      elBanner.classList.add("show");
    }} else {{
      elBanner.classList.remove("show");
    }}
  }}

  document.getElementById("exportCsvBtn").addEventListener("click", function () {{
    checkUnratedBanner();
    download("reread_pass2_export.csv", buildCsv(), "text/csv");
  }});
  document.getElementById("exportJsonBtn").addEventListener("click", function () {{
    checkUnratedBanner();
    download("reread_pass2_state.json", JSON.stringify({{ state: state }}, null, 2), "application/json");
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
        "--out-dir",
        type=Path,
        default=Path("results/gallery_readjud_blind"),
        help="Destination flat directory for opaque PNGs and the hidden key.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=REREAD_SEED,
        help=f"Re-read display-order shuffle seed (default: REREAD_SEED={REREAD_SEED}).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "If given, stage a proportional stratified random sample of this "
            "many cards instead of all cards (default: stage all cards)."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=SAMPLE_SEED,
        help=f"Stratified-sample draw seed (default: SAMPLE_SEED={SAMPLE_SEED}).",
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
    if args.sample is not None and args.sample < 1:
        logger.error("--sample must be a positive integer; got {}", args.sample)
        return 2
    return stage_reread(
        manifest_path=args.manifest,
        gallery_root=args.gallery_root,
        out_dir=args.out_dir,
        seed=args.seed,
        sample_size=args.sample,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
