#!/usr/bin/env bash
# PDB_missing data pipeline — chains the data-prep modules in src/data/preprocessing/.
# Run from anywhere; the script cds to repo root automatically.
set -euo pipefail

# Repo root = two levels up from this script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Defaults
PYTHON_BIN="python"
IDS_PATH="data/pdb_entity_ids.txt"
RAW_PATH="data/final_cleaned_dataset/pdb_missing_raw.txt"
OUTPUT_PATH="data/final_cleaned_dataset/pdb_missing_caid4.txt"
FILTER_INPUT_DIR="data/final_cleaned_dataset"
FILTER_SPLITS="train val test"
FILTER_PREFIX="final_update_or_caid4"

SKIP_FETCH=0
SKIP_BUILD=0
SKIP_MERGE=0
SKIP_FILTER=0
SKIP_REDUCE=0

usage() {
  cat <<EOF
Usage: bash scripts/data/run_pipeline.sh [options]

Chains the data-prep modules:
  1. src.data.preprocessing.fetch_pdb_entity_ids       -> entity ID list
  2. src.data.preprocessing.create_pdb_missing_dataset -> clustered PDB_missing dataset
  3. src.data.preprocessing.create_dataset_new_caid4   -> DisProt+PDB_missing merge,
       CAID3 exact-filter, master cluster, 60/20/20 split -> *_unaltered_data.txt
  4. src.data.preprocessing.filter_phase2_dataset      -> Phase 2 filtered splits

Step skip flags (each step is opt-out):
  --skip-fetch              Use existing --ids file; skip step 1.
  --skip-build              Use existing --output file; skip step 2.
  --skip-merge              Don't run the DisProt merge + split (step 3).
  --skip-filter             Don't run the Phase 2 split filter (step 4).

Step 1/2 paths:
  --ids PATH                Entity IDs list             (default: $IDS_PATH)
  --raw PATH                Intermediate raw fetch file (default: $RAW_PATH)
  --output PATH             Final clustered dataset     (default: $OUTPUT_PATH)

Step 4 paths:
  --filter-input-dir DIR    Splits directory            (default: $FILTER_INPUT_DIR)
  --filter-splits "S1 S2"   Splits to process           (default: "$FILTER_SPLITS")
  --filter-prefix STR       Filename infix              (default: $FILTER_PREFIX)


Other:
  --python BIN              Python binary               (default: $PYTHON_BIN)
  -h, --help                Show this message.

Examples:
  # Full run with defaults
  bash scripts/data/run_pipeline.sh

  # Reuse existing IDs, just rebuild the dataset and refilter
  bash scripts/data/run_pipeline.sh --skip-fetch

  # Only run the Phase 2 filter on existing splits
  bash scripts/data/run_pipeline.sh --skip-fetch --skip-build

  # Use a specific conda env's python
  bash scripts/data/run_pipeline.sh --python /path/to/envs/idp/bin/python
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-fetch)        SKIP_FETCH=1; shift ;;
    --skip-build)        SKIP_BUILD=1; shift ;;
    --skip-merge)        SKIP_MERGE=1; shift ;;
    --skip-filter)       SKIP_FILTER=1; shift ;;
    --ids)               IDS_PATH="$2"; shift 2 ;;
    --raw)               RAW_PATH="$2"; shift 2 ;;
    --output)            OUTPUT_PATH="$2"; shift 2 ;;
    --filter-input-dir)  FILTER_INPUT_DIR="$2"; shift 2 ;;
    --filter-splits)     FILTER_SPLITS="$2"; shift 2 ;;
    --filter-prefix)     FILTER_PREFIX="$2"; shift 2 ;;
    --python)            PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# Pre-flight
if ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
  echo "ERROR: python binary not runnable: $PYTHON_BIN" >&2
  exit 1
fi
for s in src/data/preprocessing/fetch_pdb_entity_ids.py \
         src/data/preprocessing/create_pdb_missing_dataset.py \
         src/data/preprocessing/create_dataset_new_caid4.py \
         src/data/preprocessing/filter_phase2_dataset.py; do
  if [[ ! -f "$s" ]]; then
    echo "ERROR: missing pipeline script: $s" >&2
    exit 1
  fi
done

echo "Repo root      : $REPO_ROOT"
echo "Python         : $("$PYTHON_BIN" --version 2>&1)"
echo

# Step 1: fetch entity IDs
if [[ $SKIP_FETCH -eq 0 ]]; then
  echo "==> Step 1/4: fetch_pdb_entity_ids -> $IDS_PATH"
  "$PYTHON_BIN" -m src.data.preprocessing.fetch_pdb_entity_ids --output "$IDS_PATH"
else
  echo "==> Step 1/4: SKIPPED"
  # Only fail-fast if Step 2 will actually consume this file.
  if [[ $SKIP_BUILD -eq 0 && ! -s "$IDS_PATH" ]]; then
    echo "ERROR: --ids file missing or empty (needed by Step 2): $IDS_PATH" >&2
    exit 1
  fi
fi
echo

# Step 2: build PDB_missing dataset
if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "==> Step 2/4: create_pdb_missing_dataset -> $OUTPUT_PATH"
  mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$RAW_PATH")"
  "$PYTHON_BIN" -m src.data.preprocessing.create_pdb_missing_dataset \
    --ids "$IDS_PATH" \
    --raw "$RAW_PATH" \
    --output "$OUTPUT_PATH"
else
  echo "==> Step 2/4: SKIPPED"
fi
echo

# Step 3: merge DisProt + PDB_missing, filter CAID3 (exact-match),
#           cluster master pool, 60/20/20 split -> *_unaltered_data.txt
if [[ $SKIP_MERGE -eq 0 ]]; then
  echo "==> Step 3/4: src.data.preprocessing.create_dataset_new_caid4 (DisProt merge + master cluster + split)"
  "$PYTHON_BIN" -m src.data.preprocessing.create_dataset_new_caid4
else
  echo "==> Step 3/4: SKIPPED"
fi
echo

# Step 4: filter splits to Phase 2
if [[ $SKIP_FILTER -eq 0 ]]; then
  echo "==> Step 4/4: filter_phase2_dataset (input_dir=$FILTER_INPUT_DIR, prefix=$FILTER_PREFIX, splits=$FILTER_SPLITS)"
  # shellcheck disable=SC2086  # word-splitting is intentional for --splits
  "$PYTHON_BIN" -m src.data.preprocessing.filter_phase2_dataset \
    --input_dir "$FILTER_INPUT_DIR" \
    --prefix "$FILTER_PREFIX" \
    --splits $FILTER_SPLITS
else
  echo "==> Step 4/4: SKIPPED"
fi
echo


echo "Pipeline complete."
