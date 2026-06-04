"""Reproducibility manifest for the deterministic feasibility outputs.

This module records sha256 content hashes of the *derived* outputs of the
feasibility pipeline (the funnel CSV and the per-patient summary CSV) together
with the headline counts, so that a reader who reruns the pipeline can confirm
they regenerated bit-identical files.

Only derived, aggregate outputs are hashed. Raw waveform data and the row-level
event inventory are never hashed or shipped (PhysioNet Data Use Agreement).

The manifest is deterministic given the same inputs: the funnel module is pure
NumPy/polars with no randomness in the aggregation path, so the hashes are
stable across machines for the same input inventory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cuffcrt.analysis.funnel import FunnelResult

# Number of bytes read per chunk when hashing a file.
_HASH_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the hex sha256 digest of a file, read in chunks.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hex-encoded sha256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HeadlineCounts:
    """The headline feasibility numbers a reader should be able to reproduce.

    Attributes
    ----------
    n_candidates : int
        Total candidate cuff cycles in the inventory.
    n_records : int
        Distinct WDB record ids.
    excluded_no_pleth : int
        Cycles with no co-recorded usable PPG window.
    qc_pass_pre_window_events : int
        Cycles passing the pre-cuff stability check (``pre_window_valid``).
    qc_pass_pre_window_patients : int
        Distinct subjects contributing at least one QC-pass cycle.
    primary_events : int
        Ipsilateral events at the primary (15 s) threshold.
    primary_patients : int
        Distinct subjects with at least one primary event.
    sensitivity_events : int
        Ipsilateral events at the sensitivity (10 s) threshold.
    sensitivity_patients : int
        Distinct subjects with at least one sensitivity event.
    """

    n_candidates: int
    n_records: int
    excluded_no_pleth: int
    qc_pass_pre_window_events: int
    qc_pass_pre_window_patients: int
    primary_events: int
    primary_patients: int
    sensitivity_events: int
    sensitivity_patients: int


def headline_from_result(result: FunnelResult) -> HeadlineCounts:
    """Extract the headline counts from a :class:`FunnelResult`.

    Parameters
    ----------
    result : FunnelResult
        Output of :func:`cuffcrt.analysis.funnel.aggregate_funnel`.

    Returns
    -------
    HeadlineCounts
        The numbers a reader should reproduce.
    """
    funnel = result.funnel
    no_pleth = int(funnel.filter(funnel["stage"] == "excluded_no_pleth")["events"][0])
    qc_row = funnel.filter(funnel["stage"] == "qc_pass_pre_window")
    return HeadlineCounts(
        n_candidates=result.n_candidates,
        n_records=result.n_records,
        excluded_no_pleth=no_pleth,
        qc_pass_pre_window_events=int(qc_row["events"][0]),
        qc_pass_pre_window_patients=int(qc_row["patients"][0]),
        primary_events=result.primary.n_events,
        primary_patients=result.primary.n_patients,
        sensitivity_events=result.sensitivity.n_events,
        sensitivity_patients=result.sensitivity.n_patients,
    )


def build_manifest(
    result: FunnelResult,
    output_files: Iterable[Path],
    *,
    inventory_source: str | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Build a reproducibility manifest dict for the funnel outputs.

    Parameters
    ----------
    result : FunnelResult
        The aggregated funnel result.
    output_files : Iterable[pathlib.Path]
        Derived output files to hash (for example ``funnel.csv`` and
        ``per_patient_summary.csv``). Each must already exist on disk.
    inventory_source : str or None
        A human-readable label for the inventory the funnel was built from
        (for example a relative path or a dataset version). Never an absolute
        home path.
    repo_root : pathlib.Path or None
        When supplied, the ``path`` field for each file is the POSIX path of
        the target relative to this root; otherwise ``path`` is the target's
        ``name`` (basename). Used to record canonical relative paths
        (``results/feasibility/funnel.csv``) without leaking absolute home
        paths into the manifest.

    Returns
    -------
    dict
        Manifest mapping with schema version, timestamp, headline counts,
        threshold parameters, and per-file sha256 digests.

    Raises
    ------
    FileNotFoundError
        If any listed output file does not exist.
    """
    files = []
    for path in sorted(set(output_files)):
        if not path.exists():
            raise FileNotFoundError(f"manifest target missing: {path}")
        if repo_root is not None:
            try:
                rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                rel = path.name
        else:
            rel = path.name
        files.append(
            {
                "name": path.name,
                "path": rel,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    headline = headline_from_result(result)
    return {
        "manifest_schema": "cuffcrt/feasibility-manifest/1",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "inventory_source": inventory_source,
        "thresholds": {
            "primary_phase3_s": result.primary.phase3_min_s,
            "sensitivity_phase3_s": result.sensitivity.phase3_min_s,
        },
        "headline_counts": headline.__dict__,
        "yield": {
            "primary": {
                "events": result.primary.n_events,
                "patients": result.primary.n_patients,
                "pct_of_candidates": result.primary.pct_of_candidates,
                "subjects": result.primary.subjects,
            },
            "sensitivity": {
                "events": result.sensitivity.n_events,
                "patients": result.sensitivity.n_patients,
                "pct_of_candidates": result.sensitivity.pct_of_candidates,
                "subjects": result.sensitivity.subjects,
            },
        },
        "files": files,
    }


def write_manifest(manifest: dict, output_path: Path) -> None:
    """Write a manifest to JSON via a tempfile plus rename for atomicity.

    Parameters
    ----------
    manifest : dict
        The manifest mapping from :func:`build_manifest`.
    output_path : pathlib.Path
        Destination path (typically ``results/manifest.json``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    tmp.replace(output_path)
