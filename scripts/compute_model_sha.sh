#!/usr/bin/env bash
# Record a SHA-256 fingerprint of the served MedGemma weights for the next run.
#
# Writes results/medgemma/_model_fingerprint_<utc_iso>.json with:
#   model_id              from $OMLX_MODEL_ID, or "unknown"
#   model_dir             absolute path that was hashed
#   model_weights_sha256  SHA-256 of the sorted concatenation of weight files
#   files_hashed          list of {path, size_bytes, sha256}
#   computed_utc          ISO-8601 UTC timestamp of this run
#
# Inputs are weight files matching *.safetensors, *.bin, or *.gguf under
# $OMLX_MODEL_DIR (recursive). The composite hash is built by streaming the
# sorted, relative-path-prefixed files into one SHA-256 so the digest is stable
# across machines with identical weights.
#
# Usage:
#   OMLX_MODEL_DIR=/path/to/weights OMLX_MODEL_ID=medgemma-1.5-4b-it-bf16 \
#     scripts/compute_model_sha.sh
#
# Exit codes:
#   0  fingerprint written
#   2  $OMLX_MODEL_DIR unset or missing, or no weight files found

set -euo pipefail

if [[ -z "${OMLX_MODEL_DIR:-}" ]]; then
  echo "error: OMLX_MODEL_DIR is not set" >&2
  echo "set it to the directory holding the served weights, e.g." >&2
  echo "  OMLX_MODEL_DIR=/path/to/weights scripts/compute_model_sha.sh" >&2
  exit 2
fi

MODEL_DIR="${OMLX_MODEL_DIR}"
MODEL_ID="${OMLX_MODEL_ID:-unknown}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "error: OMLX_MODEL_DIR is not a directory: ${MODEL_DIR}" >&2
  exit 2
fi

OUT_DIR="results/medgemma"
mkdir -p "${OUT_DIR}"

UTC_ISO="$(date -u +"%Y%m%dT%H%M%SZ")"
OUT_FILE="${OUT_DIR}/_model_fingerprint_${UTC_ISO}.json"

# Collect candidate weight files, sorted relative to MODEL_DIR for stability.
# Use a NUL-separated list so paths with spaces survive intact.
MAPFILE_TMP="$(mktemp)"
trap 'rm -f "${MAPFILE_TMP}"' EXIT
( cd "${MODEL_DIR}" && find . -type f \
    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.gguf' \) \
    -print0 ) | LC_ALL=C sort -z > "${MAPFILE_TMP}"

if [[ ! -s "${MAPFILE_TMP}" ]]; then
  echo "error: no *.safetensors, *.bin, or *.gguf files under ${MODEL_DIR}" >&2
  exit 2
fi

# Pick a SHA-256 binary that works on both macOS and Linux.
sha256_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256
  else
    echo "error: no sha256sum or shasum on PATH" >&2
    exit 2
  fi
}

# Composite SHA-256: stream "<relpath>\0" followed by file bytes for each file
# in sorted order. The path prefix makes the digest sensitive to file renames
# as well as content changes.
COMPOSITE_SHA="$(
  while IFS= read -r -d '' rel; do
    printf '%s\0' "${rel}"
    cat "${MODEL_DIR}/${rel}"
  done < "${MAPFILE_TMP}" | sha256_cmd | awk '{print $1}'
)"

# Build the JSON. Use python for safe escaping rather than printf-piping JSON.
python3 - "${MODEL_DIR}" "${MODEL_ID}" "${COMPOSITE_SHA}" "${OUT_FILE}" "${MAPFILE_TMP}" "${UTC_ISO}" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys

model_dir, model_id, composite_sha, out_path, listing_path, utc_iso = sys.argv[1:7]

# Convert the UTC stamp on the filename back into a proper ISO-8601 timestamp.
ts = dt.datetime.strptime(utc_iso, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC).isoformat()

with open(listing_path, "rb") as f:
    raw_entries = [p for p in f.read().split(b"\x00") if p]
files = []
for entry in raw_entries:
    rel = entry.decode("utf-8")
    abs_path = os.path.join(model_dir, rel)
    size = os.path.getsize(abs_path)
    digest = hashlib.sha256()
    with open(abs_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    files.append({
        "path": rel,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    })

payload = {
    "model_id": model_id,
    "model_dir": os.path.abspath(model_dir),
    "model_weights_sha256": composite_sha,
    "files_hashed": files,
    "computed_utc": ts,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"wrote {out_path}")
print(f"model_weights_sha256={composite_sha}")
print(f"files_hashed={len(files)}")
PY
