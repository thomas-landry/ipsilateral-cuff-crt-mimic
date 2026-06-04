# Rule-Based Detection and Multimodal Language-Model Adjudication of Opportunistic Capillary-Refill-Like Signals in ICU Photoplethysmography: A MIMIC-IV-WDB Feasibility Study

A capillary-refill-like signal can be read from an ICU photoplethysmogram (PPG) only when the
SpO2 probe and the noninvasive blood-pressure (NIBP) cuff sit on the **same arm**, so that each
automated cuff inflation acts as an unplanned occlusion-reperfusion event. Standard monitoring
practice places the probe and cuff on opposite arms. This project asks, in MIMIC-IV-WDB, how often
the same-arm (ipsilateral) geometry actually occurs, and whether a usable occlusion-reperfusion
signature appears when it does. This is a feasibility and prevalence study, not a biomarker. Because
MIMIC-IV-WDB does not record which arm the cuff is on, "ipsilateral" is inferred from the PPG
perfusion-index waveform morphology; every count is therefore a morphology-based estimate, not a
ground-truth laterality measurement. No outcome or mortality modeling is performed. For the numeric
results, see the accompanying manuscript/preprint rather than this README, so there is a single
source of truth.

The record and subject identifiers that appear in the code and figures are de-identified
MIMIC-IV-WDB pseudo-IDs, used here under the PhysioNet Data Use Agreement (DUA).

## Data access
This repository contains **code, not data**. MIMIC is credentialed and redistribution is prohibited
under the PhysioNet DUA, so no data is included or redistributed here. Obtain the datasets yourself
from PhysioNet and place them under `data/` as described in [README.md](./data/README.md).

A `--demo` path runs the pipeline end-to-end on the openly licensed **MIMIC-IV-Demo** with no
credentialing. The demo carries clinical tables only and no waveform records, so steps that need
waveforms (and the waveform figures) fail with an actionable message in demo mode; the cohort,
funnel, and inventory steps run fully.

## Install

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs the core dependencies. The local language-model step is optional and pulls in an
OpenAI-compatible client:

```bash
uv sync --extra llm
```

Linting uses ruff (`uv run ruff check .`); type checking is configured for pyright via
`pyrightconfig.json` and the `[tool.pyright]` block in `pyproject.toml`.

## Environment variables

The credentialed waveform tree and the consolidated event-inventory CSV live outside this
repository and are never copied in (PhysioNet DUA). Scripts that read them resolve the location from
a command-line flag first, then from an environment variable. There is no machine default: if
neither the flag nor the variable is set, the script fails with an actionable message rather than
guessing.

- `CUFFCRT_WDB_ROOT` points at the root of the MIMIC-IV-WDB record tree. Used by
  `scripts/41_run_medgemma_adjudication.py`, `scripts/50_figures.py`,
  `scripts/51_candidate_gallery.py`, and `scripts/53_disagreement_figure.py` when `--wdb-root` is
  not passed.
- `CUFFCRT_INVENTORY` points at the consolidated per-event inventory CSV. Used by
  `scripts/41_run_medgemma_adjudication.py` and `scripts/50_figures.py` when `--inventory` is not
  passed.

```bash
export CUFFCRT_WDB_ROOT=/path/to/mimic-iv-wdb/0.1.0
export CUFFCRT_INVENTORY=/path/to/event_inventory.csv
```

A command-line flag, when supplied, always overrides the environment variable. The `--demo` path
does not use these variables; it resolves the demo layout under `--data-root` instead.

## Reproduce

Scripts are numbered by pipeline stage: `10_` cohort/linkage, `20_` signal extraction (with `22_`
inventory consolidation), `30_` funnel/features, `40_`/`41_` language-model adjudication, and `50_`+
figures. Each script has a `--help` with its full flag set; the commands below are a representative
walkthrough rather than an exhaustive list. The credentialed run uses 20 waveform records by
default. All scripts write to a separate output path and never overwrite their inputs.

### 1. Cohort linkage (step 10)

Link WDB records to ICU stays by wall-clock overlap.

```bash
# Credentialed data:
uv run python scripts/10_link_wdb_to_icustay.py \
    --wdb-root data/raw/mimic-iv-wdb/0.1.0/waves \
    --icustays-csv data/raw/mimic-iv-clinical/3.1/icu/icustays.csv.gz \
    --output-parquet data/interim/wdb_to_icustay.parquet

# Open demo (no credentialing):
uv run python scripts/10_link_wdb_to_icustay.py --demo
```

### 2. Signal extraction and inventory (steps 20, 22)

For each record, slice the PLETH channel around every charted NIBP timestamp and run the
pre-registered cuff-event detector (see [preregistration_detector.md](./findings/preregistration_detector.md)).
Outputs hold derived per-event fields only (timing anchors, durations, classification, quality
flags); no raw waveform samples or note text are written. Step 22 consolidates the per-record
parquets into one inventory CSV with a deterministic row order.

```bash
uv run python scripts/20_extract_cuff_events.py \
    --data-root data --output-dir data/interim/events --n-records 20
uv run python scripts/22_consolidate_inventory.py \
    --events-dir data/interim/events --out data/interim/event_inventory.csv
```

### 3. Feasibility funnel (step 30)

Aggregate the per-record events into the feasibility funnel and a per-patient summary. The
occlusion-signature yield is reported at two reperfusion-envelope thresholds: a 15 s primary rule
and a 10 s sensitivity stratum. This step has a `--demo` path.

```bash
uv run python scripts/30_aggregate_funnel.py \
    --events-dir data/interim/events --out results/feasibility
# Open demo:
uv run python scripts/30_aggregate_funnel.py --demo
```

Step 31 sweeps the detector's pre-registered parameters one at a time as a sensitivity analysis.

### 4. Language-model adjudication (steps 40, 41)

A local **MedGemma** model provides a disclosed, secondary cross-read of the candidate traces; the
pre-registered detector remains the primary classifier. This step is a plain client to a local,
OpenAI-compatible server (oMLX). No record-level data leaves your machine, and the repository ships
the prompts and the client, not model weights. Start the server separately, then point the script
at it:

```bash
omlx serve mlx-community/medgemma-1.5-4b-it-bf16 --port 8000

# Single-purpose inference harness (text or image mode):
uv run python scripts/40_medgemma_inference.py adjudicate \
    --input-dir results/gallery/detector_positive \
    --base-url http://localhost:8000/v1

# Batch driver over the candidate pool (resolves inputs from the env vars above):
uv run python scripts/41_run_medgemma_adjudication.py --stage pilot
```

The image and prompt shown to the model carry no detector verdict, phase duration, laterality word,
or axis annotation, so the cross-read is blinded. Decoding is deterministic (`temperature=0`, fixed
seed). Every row of the run log records the served model id, prompt SHA-256, image SHA-256, base
URL, temperature, seed, and a UTC timestamp; a per-run manifest summarizes the prompt and model
fingerprints. The model-weights fingerprint is computed out of band by
`scripts/compute_model_sha.sh`. A `--dry-run` flag uses an in-process stub client (no server, no
network) for testing the run-log shape.

### 5. Figures (steps 50+)

Render the manuscript figures from the inventory and waveforms. All figures are built from live
counts (never hardcoded) and saved as both a high-DPI PNG and an editable vector PDF.

```bash
uv run python scripts/50_figures.py --which all \
    --wdb-root /path/to/mimic-iv-wdb/0.1.0 \
    --inventory /path/to/event_inventory.csv
# Figure 1 (the funnel) also renders in demo mode from the inventory CSV alone:
uv run python scripts/50_figures.py --which 1 --demo --inventory <demo_inventory.csv>
```

Additional figure scripts build the candidate gallery (`51_`), the study flow diagram (`54_`), and
related panels.

Four further figures describe the occlusion-reperfusion signature itself. Each reads the per-record
event parquets from step 20 (and, for the patient-traces figure, the consolidated inventory from
step 22), so run those steps first. Each writes a high-DPI PNG and a vector PDF to `figures/` and has
a `--help` with its full flag set; the defaults resolve to the repository-relative paths shown above,
so the bare command works after step 20 (and step 22) have run.

```bash
# Hero figure: one representative occlusion-reperfusion cycle.
uv run python scripts/58_fig_hero_cycle.py \
    --events_dir data/interim/events \
    --manifest_csv results/gallery/gallery_manifest.csv

# Signature definition: the four per-cycle quantities against their thresholds.
uv run python scripts/59_fig_signature_definition.py \
    --events_dir data/interim/events

# Morphology density: 268 reconstructed signature cycles as a 2D density.
uv run python scripts/63_fig_morphology_density.py \
    --events_dir data/interim/events

# Patient traces: one representative cycle per signature-positive patient.
# Resolves the inventory from --inventory, then CUFFCRT_INVENTORY, then the default.
uv run python scripts/64_fig_patient_traces.py \
    --inventory data/interim/event_inventory.csv
```

## Tests

```bash
uv run pytest
```

The suite (278 tests as of this writing) covers the detector, funnel, inventory, bootstrap,
language-model harness, and figure builders.

## Reproducibility notes

- The global random seed is pinned in `src/cuffcrt/_seed.py` (`GLOBAL_SEED`). The detector itself
  uses no random state; the seed governs only the adjudication-gallery sampling and audit subsets.
- Every script exposes a command-line interface; nothing in the result-generating path requires a
  notebook.
- No script overwrites its own input. Outputs are written to separate paths via a
  tempfile-then-rename for atomicity where applicable.
- Language-model outputs hold derived fields only, never raw note text, and log full provenance
  (model id, prompt SHA, seed, decoding parameters, run timestamp) per row.
- Reporting follows TRIPOD-AI (PMID 38626948) and TRIPOD-LLM (PMID 39779929).

## Citation

If you use this software, please cite it. Fill the placeholders at release:

```
Landry TC, Kim Y. Cuff-occlusion capillary-refill signals in MIMIC-IV-WDB.
Version <RELEASE_TAG>, commit <RELEASE_COMMIT_HASH>. Zenodo. https://doi.org/<ZENODO_DOI>
```

- Repository: `<PLACEHOLDER_REPOSITORY_URL>`
- Release commit: `<PLACEHOLDER_RELEASE_COMMIT_HASH>`
- Zenodo DOI: `<PLACEHOLDER_ZENODO_DOI>`

## Authors
- **Thomas C. Landry, MD** (corresponding). ORCID [0009-0009-1802-9673](https://orcid.org/0009-0009-1802-9673)
- **Youjin Kim, MD**. ORCID [0009-0007-0889-4669](https://orcid.org/0009-0007-0889-4669)

Department of Internal Medicine, Legacy Salmon Creek Medical Center, Vancouver, WA, USA.

## License

MIT. See [LICENSE](LICENSE).
