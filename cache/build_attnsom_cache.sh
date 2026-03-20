#!/bin/bash
#SBATCH --job-name=attnsom-cache
#SBATCH --output=attnsom_cache_%j.log
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

export PYTHONUNBUFFERED=1
CACHE_DIR="/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/tools/therapeutic_tools/cache"
ATTNSOM_PY=/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/python

cd "$CACHE_DIR/.."

$ATTNSOM_PY "$CACHE_DIR/build_attnsom_cache.py" "$@"
