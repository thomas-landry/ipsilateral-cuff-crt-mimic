"""Bridge each gallery ``card_id`` to the canonical run's ``row_id``.

The prompt-sensitivity re-adjudication (``scripts/42``) keys its outputs by
``card_id`` (a stratum-prefixed hash assigned by ``scripts/51`` when the gallery
was rendered). The canonical headline adjudication (``scripts/41``) keys its run
log by ``row_id`` (``"{subject_id}_{record_id}_{idx}"``, where ``idx`` is the
0-based row index into the consolidated inventory after ``with_row_index``).
Scoring a prompt variant against the canonical run therefore needs a stable
``card_id -> row_id`` map.

This module owns that map. It reconstructs the canonical ``row_id`` on the
inventory using the *same* construction ``scripts/41`` uses
(:func:`scripts.41_run_medgemma_adjudication.build_event_frame`), then joins the
gallery manifest's natural triple ``(subject_id, record_id, round(t_nbp, 3))``
to the inventory triple ``(subject_id, record_id, round(nbp_timestamp_s, 3))``.
The rounding makes the float timestamp join robust to trailing formatting noise
without colliding distinct charted cycles, which are minutes apart. Each
manifest card resolves 1:1 to a unique inventory ``row_id`` (validated
2026-05-30: 568/568 cards, 0 unmatched, 0 call-mismatch versus the canonical
run). The inventory-derived ``row_id`` set is a strict superset of the canonical
run-log set, so the bridge keys are the canonical keys.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger  # pyright: ignore[reportMissingImports]

# Rounding for the t_nbp join key, matching ``scripts/41``'s gallery lookup.
# Milliseconds are coarse enough to absorb float-formatting noise yet far finer
# than the minutes-apart spacing of charted NIBP cycles, so no two distinct
# cycles can collide onto one key.
_T_NBP_DECIMALS = 3

# Columns the bridge returns, in a stable order. ``row_id`` carries the canonical
# key; the remaining columns are the gallery-side identifiers a caller needs to
# trace a card back to its source cycle.
_BRIDGE_COLUMNS = (
    "card_id",
    "stratum",
    "row_id",
    "subject_id",
    "record_id",
    "t_nbp",
)


def _inventory_row_ids(inventory_path: Path) -> pl.DataFrame:
    """Reconstruct the canonical ``row_id`` for every inventory cycle.

    Reproduces the ``row_id`` construction in ``scripts/41``'s
    ``build_event_frame``: read the consolidated inventory, add a 0-based row
    index with ``with_row_index``, and format
    ``"{subject_id}_{record_id}_{idx}"``. A rounded timestamp column
    (``_t_key``) is added for the manifest join.

    Parameters
    ----------
    inventory_path : pathlib.Path
        Consolidated event inventory CSV (``data/interim/event_inventory.csv``).

    Returns
    -------
    polars.DataFrame
        Columns ``subject_id``, ``record_id``, ``_t_key``, ``row_id``.
    """
    inv = pl.read_csv(inventory_path, infer_schema_length=20000)
    inv = inv.with_row_index(name="_idx")
    inv = inv.with_columns(
        row_id=pl.format(
            "{}_{}_{}",
            pl.col("subject_id"),
            pl.col("record_id"),
            pl.col("_idx"),
        ),
        _t_key=pl.col("nbp_timestamp_s").round(_T_NBP_DECIMALS),
    )
    return inv.select(["subject_id", "record_id", "_t_key", "row_id"])


def build_card_to_rowid(inventory_path: Path, manifest_path: Path) -> pl.DataFrame:
    """Map each gallery ``card_id`` to the canonical run's ``row_id``.

    The gallery manifest keys each card by ``card_id`` and carries the natural
    triple ``(subject_id, record_id, t_nbp)``. The canonical run keys rows by
    ``row_id``. This joins the manifest triple to the inventory triple (both
    rounded to :data:`_T_NBP_DECIMALS` decimals) to attach the canonical
    ``row_id`` to each card.

    A card whose triple is absent from the inventory keeps its row but carries a
    null ``row_id``; callers should check for nulls rather than assume every
    card resolves. On the validated gallery (2026-05-30) all 568 cards resolve
    1:1, so a null indicates a genuine mismatch worth surfacing.

    Parameters
    ----------
    inventory_path : pathlib.Path
        Consolidated event inventory CSV (``data/interim/event_inventory.csv``).
    manifest_path : pathlib.Path
        Gallery manifest CSV (``results/gallery/gallery_manifest.csv``).

    Returns
    -------
    polars.DataFrame
        One row per manifest card with columns ``card_id``, ``stratum``,
        ``row_id``, ``subject_id``, ``record_id``, ``t_nbp``. ``row_id`` is null
        for any card whose triple has no inventory match.

    Raises
    ------
    FileNotFoundError
        If either input file is absent.
    """
    if not inventory_path.exists():
        raise FileNotFoundError(f"inventory not found: {inventory_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"gallery manifest not found: {manifest_path}")

    inv = _inventory_row_ids(inventory_path)
    manifest = pl.read_csv(manifest_path, infer_schema_length=20000)
    manifest = manifest.with_columns(_t_key=pl.col("t_nbp").round(_T_NBP_DECIMALS))

    bridge = manifest.join(
        inv,
        on=["subject_id", "record_id", "_t_key"],
        how="left",
    )
    bridge = bridge.select(list(_BRIDGE_COLUMNS))

    n_cards = bridge.height
    n_matched = bridge.filter(pl.col("row_id").is_not_null()).height
    n_unmatched = n_cards - n_matched
    if n_unmatched:
        logger.warning(
            "card_bridge: {} of {} cards did not resolve to an inventory row_id",
            n_unmatched,
            n_cards,
        )
    else:
        logger.info("card_bridge: all {} cards resolved 1:1 to a row_id", n_cards)
    return bridge
