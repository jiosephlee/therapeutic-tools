#!/bin/bash
#SBATCH --job-name=gloryx-cache
#SBATCH --partition=genoa-std-mem
#SBATCH --time=6:00:00
#SBATCH --mem=4G
#SBATCH --output=/vast/home/j/jojolee/OpenRLHF-Tools/runs/gloryx_cache_%j.log

# No GPU needed — just API calls
cd /vast/home/j/jojolee/OpenRLHF-Tools

# Use the project conda env
PYTHON=/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/python

$PYTHON -u openrlhf/tools/therapeutic_tools/scripts/build_gloryx_cache.py \
    --batch-size 50 \
    --delay 10 \
    --poll-interval 10 \
    --job-timeout 600
