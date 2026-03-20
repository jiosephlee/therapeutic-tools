#!/bin/bash
#SBATCH --job-name=attnsom-train
#SBATCH --output=attnsom_train_%j.log
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

ATTNSOM_DIR="/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/tools/therapeutic_tools/ATTNSOM"
ATTNSOM_PY=/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/python
RESULT_DIR=/vast/projects/myatskar/design-documents/hf_home/attnsom_results

export PYTHONUNBUFFERED=1
echo "ATTNSOM_DIR: $ATTNSOM_DIR"
echo "RESULT_DIR: $RESULT_DIR"

cd "$ATTNSOM_DIR"

$ATTNSOM_PY main.py \
    --dataset_dir "$ATTNSOM_DIR/cyp_dataset" \
    --result_dir "$RESULT_DIR" \
    --exclude_tdc \
    --save_checkpoint \
    --max_epochs 50 \
    --hidden_size 256 \
    --num_layers 4 \
    --num_attn_heads 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --loss focal \
    --n_splits 10 \
    "$@"

echo "Done. Checkpoint at: $RESULT_DIR/attnsom_checkpoint.pt"
