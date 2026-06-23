#!/bin/bash
#SBATCH --job-name=attnsom-retrain
#SBATCH --partition=b200-mig45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=attnsom_retrain_%j.log

set -euo pipefail

source /vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/activate 2>/dev/null || true
export PATH="/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULT_DIR="${ATTNSOM_RESULT_DIR:-$PACKAGE_DIR/cache/attnsom/results}"
mkdir -p "$RESULT_DIR"

echo "=== ATTNSOM Retraining ==="
echo "Result dir: $RESULT_DIR"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd "${ATTNSOM_ROOT:-$PACKAGE_DIR/ATTNSOM}"

python retrain.py \
    --max_epochs 200 \
    --patience 30 \
    --inner_val_ratio 0.1 \
    --save_checkpoint \
    --exclude_tdc \
    --result_dir "$RESULT_DIR"

echo "End time: $(date)"
echo "=== Done ==="
