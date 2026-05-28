#!/usr/bin/env bash
# Download TCGA-BRCA diagnostic slides and rename them to {case_id}.svs.
#
# GDC downloads each file into data/raw/<uuid>/<long_name>.svs.
# This script flattens and renames them to data/raw/<case_id>.svs
# so the preprocessing pipeline can find them.
#
# Prerequisites:
#   Install gdc-client: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
#   (macOS: brew install gdc-client  OR  download binary from the link above)
#
# Usage:
#   bash data/download.sh                              # uses defaults
#   bash data/download.sh data/manifest.txt data/raw/  # explicit paths

set -euo pipefail

MANIFEST="${1:-data/manifest.txt}"
OUTPUT_DIR="${2:-data/raw}"
N_PROCS="${3:-8}"

if ! command -v gdc-client &>/dev/null; then
    echo "Error: gdc-client not found. Install from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool"
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found: $MANIFEST"
    echo "Run: python scripts/prepare_labels.py --out_manifest $MANIFEST"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Downloading slides ==="
echo "  Manifest : $MANIFEST"
echo "  Output   : $OUTPUT_DIR"
echo "  Workers  : $N_PROCS"
echo ""

gdc-client download \
    --manifest "$MANIFEST" \
    --dir "$OUTPUT_DIR" \
    --n-processes "$N_PROCS" \
    --retry-amount 3 \
    --wait-time 30

echo ""
echo "=== Renaming files to {case_id}.svs ==="

# GDC layout:  $OUTPUT_DIR/<uuid>/TCGA-XX-XXXX-01Z-00-DX1.<uuid>.svs
# Target:      $OUTPUT_DIR/TCGA-XX-XXXX.svs
# case_id = first three hyphen-delimited fields of the TCGA barcode

n_renamed=0
n_skipped=0

while IFS= read -r -d '' svs_path; do
    filename=$(basename "$svs_path")
    stem="${filename%.svs}"

    # Extract TCGA-TSS-PATIENT (first 3 parts: TCGA-XX-XXXX)
    case_id=$(echo "$stem" | awk -F'-' '{print $1"-"$2"-"$3}')

    if [[ "$case_id" != TCGA-* ]]; then
        echo "  Skipping unexpected filename: $filename"
        (( n_skipped++ )) || true
        continue
    fi

    dest="$OUTPUT_DIR/${case_id}.svs"

    if [ -f "$dest" ]; then
        echo "  Already exists, skipping: ${case_id}.svs"
        (( n_skipped++ )) || true
    else
        mv "$svs_path" "$dest"
        echo "  $filename  →  ${case_id}.svs"
        (( n_renamed++ )) || true
    fi
done < <(find "$OUTPUT_DIR" -name "*.svs" -not -path "$OUTPUT_DIR/*.svs" -print0)

# Clean up empty uuid subdirectories
find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -empty -delete 2>/dev/null || true

echo ""
echo "Done. Renamed: $n_renamed  Skipped: $n_skipped"
echo "Slides ready in: $OUTPUT_DIR"
