"""Shared data-path resolution for the pipeline scripts.

Centralizes the layout described in ``data/README.md`` so every script agrees
on where the credentialed (full) and open (demo) datasets live, and so the
``--demo`` flag fails with one consistent, actionable message when the demo
data is absent. No data is shipped with this repository; users obtain MIMIC
themselves under the PhysioNet Data Use Agreement.
"""

from __future__ import annotations

import os
from pathlib import Path

# Environment variables that point shipping scripts at credentialed inputs the
# repository never ships (PhysioNet Data Use Agreement). See data/README.md.
ENV_WDB_ROOT = "CUFFCRT_WDB_ROOT"
ENV_INVENTORY = "CUFFCRT_INVENTORY"

# Default on-disk layout, relative to a data root (see data/README.md).
FULL_WDB_SUBPATH = Path("raw/mimic-iv-wdb/0.1.0")
FULL_ICUSTAYS_SUBPATH = Path("raw/mimic-iv-clinical/3.1/icu/icustays.csv.gz")
DEMO_WDB_SUBPATH = Path("raw/mimic-iv-demo/2.2")
DEMO_ICUSTAYS_SUBPATH = Path("raw/mimic-iv-demo/2.2/icu/icustays.csv.gz")

# Derived inputs for the local-LLM harness (pipeline step 40). These are
# produced by upstream steps run in --demo mode (notes for extract, perfusion
# plots for adjudicate) and live under the interim tree. No dataset ships here;
# --demo resolves the directory and fails clean if it has not been built yet.
DEMO_EXTRACT_NOTES_SUBPATH = Path("interim/demo/notes")
DEMO_ADJUDICATE_PLOTS_SUBPATH = Path("interim/demo/plots")

_README_HINT = (
    "Obtain the data yourself and place it under the data/ tree as described "
    "in data/README.md. Nothing under data/ is shipped with this repository "
    "(PhysioNet Data Use Agreement)."
)


class DataNotAvailableError(FileNotFoundError):
    """Raised when an expected MIMIC path is missing on disk."""


class DataPathNotConfiguredError(RuntimeError):
    """Raised when a credentialed input is neither flagged nor in the env."""


def env_path(env_var: str) -> Path | None:
    """Return the path in ``env_var`` if set and non-empty, else ``None``.

    Parameters
    ----------
    env_var : str
        Name of the environment variable to read.

    Returns
    -------
    pathlib.Path or None
        The configured path, or ``None`` when the variable is unset or empty.
    """
    raw = os.environ.get(env_var, "").strip()
    return Path(raw) if raw else None


def resolve_configured_path(
    flag_value: Path | None,
    *,
    env_var: str,
    flag: str,
    what: str,
) -> Path:
    """Resolve a credentialed input from an explicit flag or an env var.

    Precedence: an explicit ``--flag`` value wins; otherwise the value of
    ``env_var``; if neither is present the function fails loud rather than
    falling back to a machine-specific default. The repository ships no data
    (PhysioNet Data Use Agreement), so there is no safe default to use here.

    Parameters
    ----------
    flag_value : pathlib.Path or None
        The value parsed from the command-line flag (``None`` when omitted).
    env_var : str
        Name of the environment variable consulted when the flag is omitted.
    flag : str
        The command-line flag name, used in the error message.
    what : str
        Human-readable name of the input, used in the error message.

    Returns
    -------
    pathlib.Path
        The resolved path. Existence is not checked here; pair with
        :func:`require_path` to fail clean when the path is absent.

    Raises
    ------
    DataPathNotConfiguredError
        If neither ``flag_value`` nor ``env_var`` provides a path.
    """
    if flag_value is not None:
        return flag_value
    from_env = env_path(env_var)
    if from_env is not None:
        return from_env
    raise DataPathNotConfiguredError(
        f"{what} is not configured. Pass {flag} or set the {env_var} "
        f"environment variable.\n{_README_HINT}"
    )


def require_path(path: Path, *, what: str) -> Path:
    """Return ``path`` if it exists, else raise an actionable error.

    Parameters
    ----------
    path : pathlib.Path
        The path that must exist.
    what : str
        Human-readable name of the dataset, used in the error message.

    Returns
    -------
    pathlib.Path
        The same ``path``, confirmed to exist.

    Raises
    ------
    DataNotAvailableError
        If ``path`` does not exist.
    """
    if not path.exists():
        raise DataNotAvailableError(f"{what} not found at {path}.\n{_README_HINT}")
    return path


def resolve_wdb_root(data_root: Path, *, demo: bool) -> Path:
    """Resolve the WDB record root for full or demo mode.

    Parameters
    ----------
    data_root : pathlib.Path
        The ``data/`` directory root.
    demo : bool
        If ``True``, resolve the open MIMIC-IV-Demo layout; otherwise the
        credentialed MIMIC-IV-WDB layout.

    Returns
    -------
    pathlib.Path
        The WDB record-tree root.
    """
    sub = DEMO_WDB_SUBPATH if demo else FULL_WDB_SUBPATH
    return data_root / sub


def resolve_icustays_csv(data_root: Path, *, demo: bool) -> Path:
    """Resolve the ``icustays.csv.gz`` path for full or demo mode."""
    sub = DEMO_ICUSTAYS_SUBPATH if demo else FULL_ICUSTAYS_SUBPATH
    return data_root / sub


def resolve_demo_llm_input_dir(data_root: Path, *, mode: str) -> Path:
    """Resolve the demo input directory for the local-LLM harness.

    Parameters
    ----------
    data_root : pathlib.Path
        The ``data/`` directory root.
    mode : str
        ``"extract"`` (text notes) or ``"adjudicate"`` (perfusion plots).

    Returns
    -------
    pathlib.Path
        The demo input directory for that mode. The directory is not guaranteed
        to exist; use :func:`require_path` to fail clean when it is absent.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"extract"`` or ``"adjudicate"``.
    """
    if mode == "extract":
        return data_root / DEMO_EXTRACT_NOTES_SUBPATH
    if mode == "adjudicate":
        return data_root / DEMO_ADJUDICATE_PLOTS_SUBPATH
    raise ValueError(f"unknown LLM harness mode: {mode!r}")
