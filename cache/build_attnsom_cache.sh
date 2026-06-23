#!/bin/bash
#SBATCH --job-name=attnsom-cache
#SBATCH --output=attnsom_cache_%j.log
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$SCRIPT_DIR"
ATTNSOM_PY=/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/python

cd "$CACHE_DIR/.."

$ATTNSOM_PY "$CACHE_DIR/build_attnsom_cache.py" "$@"
