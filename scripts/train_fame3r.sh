#!/bin/bash
#SBATCH --job-name=fame3r-train
#SBATCH --partition=b200-mig45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/vast/projects/myatskar/design-documents/hf_home/fame3r_models/train_%j.log

# FAME3R training is CPU-only (sklearn RandomForest), should be fast.
# Using b200-mig45 as requested.

set -euo pipefail

# Activate conda env
source /vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/activate 2>/dev/null || true
export PATH="/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin:$PATH"

OUTPUT_DIR="/vast/projects/myatskar/design-documents/hf_home/fame3r_models"

echo "=== FAME3R Training ==="
echo "Output dir: $OUTPUT_DIR"
echo "Start time: $(date)"

python /vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/tools/therapeutic_tools/scripts/train_fame3r.py \
    --output-dir "$OUTPUT_DIR"

echo "End time: $(date)"
echo "=== Done ==="
