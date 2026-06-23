#!/bin/bash
#SBATCH --job-name=fame3r-train
#SBATCH --partition=b200-mig45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=fame3r_train_%j.log

# FAME3R training is CPU-only (sklearn RandomForest), should be fast.
# Using b200-mig45 as requested.

set -euo pipefail

# Activate conda env
source /vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/activate 2>/dev/null || true
export PATH="/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${FAME3R_OUTPUT_DIR:-$PACKAGE_DIR/cache/fame3r_models}"

echo "=== FAME3R Training ==="
echo "Output dir: $OUTPUT_DIR"
echo "Start time: $(date)"

python "$SCRIPT_DIR/train_fame3r.py" \
    --output-dir "$OUTPUT_DIR"

echo "End time: $(date)"
echo "=== Done ==="
