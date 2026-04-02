#!/bin/bash
# Build fingerprint embedding cache for therapeutic_tools.
#
# Computes Morgan + FeatureMorgan fingerprints directly from raw TDC CSVs
# (no Intern-S1-recipe pickle dependency).
#
# Step 1: Build fingerprint npz files from data/tdc/raw/
# Step 2: Rebuild knn_metrics.json for fingerprint embeddings
#
# Usage:
#   bash build_fingerprint_cache.sh [--tasks TASK1 TASK2 ...] [--splits train val]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY=/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/python

echo "=== Step 1: Build fingerprint embeddings from raw TDC CSVs ==="
echo "Using Python: $PY"
$PY "$SCRIPT_DIR/build_fingerprint_embeddings.py" --overwrite "$@"

echo ""
echo "=== Step 2: Rebuild knn_metrics.json (fingerprint type) ==="
$PY "$SCRIPT_DIR/build_knn_metrics.py" --embedding-type fingerprint

echo ""
echo "Done. Fingerprint embeddings:"
ls "$SCRIPT_DIR/fingerprint/"
