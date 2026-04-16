#!/bin/bash
#BSUB -J scene_detect
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o /work3/s225083/cs280/project/logs/scene_detect_%J.out
#BSUB -e /work3/s225083/cs280/project/logs/scene_detect_%J.err

source /work3/s225083/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /work3/s225083/cs280/project
mkdir -p logs

THRESHOLD=${1:-27.0}

echo "=== Running scene detection (threshold=${THRESHOLD}) ==="
python scene_detect.py \
    --frames_root frames \
    --out cuts.json \
    --threshold ${THRESHOLD}

echo "=== Done ==="
