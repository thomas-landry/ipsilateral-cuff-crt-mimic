"""Link MIMIC-IV-WDB master records to ``icu/icustays`` rows by wall-clock overlap.

Given the WDB record tree (for example ``data/raw/mimic-iv-wdb/0.1.0/waves/``)
and the clinical ``icu/icustays.csv.gz`` table, this module emits a long-format
``(record_id, subject_id, stay_id, overlap_s, is_modal)`` frame keyed on every
WDB-record by ICU-stay pair whose ``[record_start, record_end]`` window
intersects the ``[intime, outtime]`` window. The "modal" stay is the
longest-overlap one per record (deterministic tie-break: earlier ``intime``).

Wall-clock window resolution
----------------------------
For each WDB master record (the ``<record_id>.hea`` at the subject's record
directory root, e.g. ``waves/pXX/pXXXXXXXX/XXXXXXXX/XXXXXXXX.hea``) we use
:func:`wfdb.rdheader` to read ``base_date``, ``base_time``, ``sig_len``, and
``fs`` (the multi-segment master frame rate, not the channel-native rate). The
record's wall-clock window is::

    record_start = combine(base_date, base_time)
    record_end   = record_start + timedelta(seconds=sig_len / fs)

Records lacking ``base_time`` and/or ``base_date`` are logged at warning level
and skipped.

Subject-id parsing
------------------
The WDB tree is laid out as ``waves/pXX/pXXXXXXXX/XXXXXXXX/...``. The
``subject_id`` is encoded in the second-level path component (``pXXXXXXXX``)
without the ``p`` prefix and parsed as ``int`` per :data:`SUBJECT_DIR_PATTERN`.

Modal-stay tie-break
--------------------
If two ICU stays have identical overlap with the same record, the one with the
earlier ``intime`` is marked modal. This preserves determinism when a record
straddles a same-day transfer.
"""

from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Final, Protocol

import polars as pl
from loguru import logger

# Subject directory pattern: ``p<digits>``. The numeric portion (no ``p``
# prefix) is the MIMIC-IV ``subject_id`` integer.
SUBJECT_DIR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^p(\d+)$")

# Output schema for :func:`link_wdb_to_icustays`.
OUTPUT_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "record_id": pl.Utf8,
    "subject_id": pl.Int64,
    "stay_id": pl.Int64,
    "record_start": pl.Datetime,
    "record_end": pl.Datetime,
    "stay_intime": pl.Datetime,
    "stay_outtime": pl.Datetime,
    "overlap_s": pl.Float64,
    "is_modal": pl.Boolean,
}


class _WfdbHeader(Protocol):
    """Subset of the :class:`wfdb.io.record.Record` interface we depend on."""

    base_date: object  # datetime.date | None
    base_time: object  # datetime.time | None
    sig_len: int
    fs: float


def _record_time_window(header: _WfdbHeader) -> tuple[datetime, datetime] | None:
    """Compute a record's wall-clock window from a wfdb-style header.

    Parameters
    ----------
    header : _WfdbHeader
        Object exposing ``base_date``, ``base_time``, ``sig_len`` and ``fs``.

    Returns
    -------
    tuple[datetime, datetime] or None
        ``(record_start, record_end)``; ``None`` when the header lacks
        ``base_date`` or ``base_time``, when ``fs`` is not strictly positive,
        or when ``sig_len`` is non-positive.
    """
    base_date = getattr(header, "base_date", None)
    base_time = getattr(header, "base_time", None)
    sig_len = getattr(header, "sig_len", None)
    fs = getattr(header, "fs", None)
    if base_date is None or base_time is None:
        return None
    if sig_len is None or fs is None:
        return None
    if not isinstance(base_time, time):
        return None
    try:
        sig_len_int = int(sig_len)
        fs_f = float(fs)
    except (TypeError, ValueError):
        return None
    if sig_len_int <= 0 or fs_f <= 0.0:
        return None
    record_start = datetime.combine(base_date, base_time)  # type: ignore[arg-type]
    duration_s = sig_len_int / fs_f
    record_end = record_start + timedelta(seconds=duration_s)
    return record_start, record_end


def _subject_id_from_path(record_path: Path) -> int | None:
    """Parse ``subject_id`` from a WDB record path.

    The path is expected to follow ``.../waves/pXX/pXXXXXXXX/<record_id>/...``;
    we look up the tree for the first ancestor matching
    :data:`SUBJECT_DIR_PATTERN`.
    """
    for parent in [record_path, *record_path.parents]:
        m = SUBJECT_DIR_PATTERN.match(parent.name)
        if m:
            return int(m.group(1))
    return None


def _master_record_basenames(wdb_root: Path) -> list[Path]:
    """Walk ``wdb_root`` and return master-record basenames (no ``.hea``).

    A master record header sits at the record-directory root and has the same
    stem as the directory: ``<record_dir>/<record_dir.name>.hea``.
    """
    if not wdb_root.exists():
        logger.warning("wdb_root {} does not exist; no records.", wdb_root)
        return []
    bases: list[Path] = []
    for hea in wdb_root.rglob("*.hea"):
        if hea.stem == hea.parent.name:
            bases.append(hea.with_suffix(""))
    return sorted(bases)


def _overlap_seconds(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> float:
    """Seconds of overlap between two intervals; 0 if disjoint."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return 0.0
    return (end - start).total_seconds()


def link_wdb_to_icustays(
    wdb_root: Path,
    icustays: pl.DataFrame,
) -> pl.DataFrame:
    """Link every WDB record under ``wdb_root`` to overlapping ICU stays.

    Parameters
    ----------
    wdb_root : pathlib.Path
        Directory tree containing the WDB records, e.g.
        ``data/raw/mimic-iv-wdb/0.1.0/waves``. Each subject lives under a
        ``p<bucket>/p<subject_id>/<record_id>/`` directory; the master header
        is at ``<record_id>/<record_id>.hea``.
    icustays : polars.DataFrame
        MIMIC-IV ``icu/icustays.csv.gz`` parsed into a polars DataFrame.
        Required columns: ``subject_id`` (int), ``stay_id`` (int), ``intime``
        (datetime), ``outtime`` (datetime). Extra columns are ignored.

    Returns
    -------
    polars.DataFrame
        Long-format frame with the columns in :data:`OUTPUT_SCHEMA`. One row
        per (record, stay_id) overlap. ``is_modal`` is ``True`` only on the
        longest-overlap row per record; ties on ``overlap_s`` are broken by the
        earlier ``intime``. Empty frame (with the schema) when no overlaps are
        found.

    Notes
    -----
    - Records missing ``base_time`` or ``base_date`` are skipped with a
      warning log.
    - Subject directories whose name does not match
      :data:`SUBJECT_DIR_PATTERN` are skipped with a warning log.
    - Output is sorted by ``(record_id, -overlap_s, stay_intime)`` so the
      modal-stay row appears first for each record.
    """
    # Lazy import so unit tests using a stubbed reader can import this module
    # even where the wfdb data tree is unavailable.
    import wfdb  # noqa: PLC0415

    if not _icustays_has_required_columns(icustays):
        missing = sorted({"subject_id", "stay_id", "intime", "outtime"} - set(icustays.columns))
        raise ValueError(f"icustays missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    bases = _master_record_basenames(wdb_root)
    if not bases:
        return _empty_output()

    icustays_by_subject = _index_icustays(icustays)

    for base in bases:
        record_id = base.name
        subject_id = _subject_id_from_path(base)
        if subject_id is None:
            logger.warning("Could not parse subject_id from path {}; skipping.", base)
            continue
        try:
            header = wfdb.rdheader(str(base))
        except Exception as exc:  # noqa: BLE001 - wfdb raises a wide variety
            logger.warning("wfdb.rdheader({}) failed: {}; skipping.", base, exc)
            continue
        # wfdb's typeshed unions rdheader's return as Record | MultiRecord; at
        # runtime it is always a Record, which structurally satisfies
        # _WfdbHeader. The MultiRecord branch is a stub-only false positive.
        window = _record_time_window(header)  # pyright: ignore[reportArgumentType]
        if window is None:
            logger.warning(
                "Record {} lacks base_date/base_time/sig_len/fs; skipping.",
                record_id,
            )
            continue
        record_start, record_end = window

        candidate_stays = icustays_by_subject.get(subject_id, [])
        for stay in candidate_stays:
            stay_id, intime, outtime = stay
            overlap_s = _overlap_seconds(record_start, record_end, intime, outtime)
            if overlap_s <= 0.0:
                continue
            rows.append(
                {
                    "record_id": record_id,
                    "subject_id": subject_id,
                    "stay_id": stay_id,
                    "record_start": record_start,
                    "record_end": record_end,
                    "stay_intime": intime,
                    "stay_outtime": outtime,
                    "overlap_s": overlap_s,
                    "is_modal": False,
                }
            )

    if not rows:
        return _empty_output()

    df = pl.DataFrame(rows, schema=OUTPUT_SCHEMA)
    df = _flag_modal(df)
    return df.sort(
        ["record_id", "overlap_s", "stay_intime"],
        descending=[False, True, False],
    )


def _icustays_has_required_columns(icustays: pl.DataFrame) -> bool:
    """True if ``icustays`` has the four columns we depend on."""
    required = {"subject_id", "stay_id", "intime", "outtime"}
    return required.issubset(icustays.columns)


def _index_icustays(
    icustays: pl.DataFrame,
) -> dict[int, list[tuple[int, datetime, datetime]]]:
    """Group ``icustays`` rows by ``subject_id`` for O(1) lookup."""
    out: dict[int, list[tuple[int, datetime, datetime]]] = {}
    cols = ["subject_id", "stay_id", "intime", "outtime"]
    for row in icustays.select(cols).iter_rows():
        subject_id, stay_id, intime, outtime = row
        if subject_id is None or stay_id is None or intime is None or outtime is None:
            continue
        if outtime <= intime:
            continue
        out.setdefault(int(subject_id), []).append((int(stay_id), intime, outtime))
    return out


def _flag_modal(df: pl.DataFrame) -> pl.DataFrame:
    """Set ``is_modal=True`` on the longest-overlap row per record.

    Tie-break: earlier ``stay_intime`` wins. The column is rebuilt rather than
    mutated in place (polars frames are immutable).
    """
    modal = (
        df.sort(["record_id", "overlap_s", "stay_intime"], descending=[False, True, False])
        .group_by("record_id", maintain_order=True)
        .agg(pl.col("stay_id").first().alias("modal_stay_id"))
    )
    df = df.join(modal, on="record_id", how="left")
    return df.with_columns((pl.col("stay_id") == pl.col("modal_stay_id")).alias("is_modal")).drop(
        "modal_stay_id"
    )


def _empty_output() -> pl.DataFrame:
    """Empty output with the canonical :data:`OUTPUT_SCHEMA`."""
    return pl.DataFrame({c: [] for c in OUTPUT_SCHEMA}, schema=OUTPUT_SCHEMA)


__all__ = [
    "OUTPUT_SCHEMA",
    "SUBJECT_DIR_PATTERN",
    "link_wdb_to_icustays",
]
