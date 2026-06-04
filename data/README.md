# Data (not included)

This repository contains **code, not data**. MIMIC is credentialed; redistribution is prohibited
under the PhysioNet Data Use Agreement (DUA). Obtain the data yourself, then place it here.
See the top-level [README.md](./../README.md) for the pipeline that consumes these inputs.

## Required (full reproduction, credentialed)
- **MIMIC-IV-WDB v0.1.0** (waveforms): https://physionet.org/content/mimic4wdb/0.1.0/
  -> `data/raw/mimic-iv-wdb/0.1.0/`
- **MIMIC-IV v3.1** (clinical tables): https://physionet.org/content/mimiciv/3.1/
  -> `data/raw/mimic-iv-clinical/3.1/`
- **MIMIC-IV-Note** (optional, for the text-extraction step): https://physionet.org/content/mimic-iv-note/

## Open demo (no credentialing)
- **MIMIC-IV-Demo v2.2** (~100 patients, openly licensed): https://physionet.org/content/mimic-iv-demo/
  -> `data/raw/mimic-iv-demo/2.2/`
  Used by `--demo` mode so anyone can run the full pipeline end-to-end without credentialing.

## Environment variables

The credentialed waveform tree and the consolidated event-inventory CSV live
outside this repository and are never copied in (PhysioNet DUA). Scripts that
read them resolve their location from a command-line flag first, then from an
environment variable. There is no machine default; if neither is set, the script
fails with an actionable message rather than guessing.

- `CUFFCRT_WDB_ROOT` -> root of the MIMIC-IV-WDB record tree (the directory
  containing `waves/`). Used by `scripts/41_run_medgemma_adjudication.py`,
  `scripts/50_figures.py`, `scripts/51_candidate_gallery.py`, and
  `scripts/53_disagreement_figure.py` when `--wdb-root` (or `--wdb_root`) is not
  passed. The `--demo` path does not use this variable; it resolves the demo
  layout under `--data-root` instead.
- `CUFFCRT_INVENTORY` -> path to the consolidated per-event inventory CSV. Used
  by `scripts/41_run_medgemma_adjudication.py` and `scripts/50_figures.py` when
  `--inventory` is not passed.

Example:

```sh
export CUFFCRT_WDB_ROOT=/path/to/mimic-iv-wdb/0.1.0
export CUFFCRT_INVENTORY=/path/to/event_inventory.csv
```

A command-line flag, when supplied, always overrides the environment variable.

Everything under `data/` is gitignored.
