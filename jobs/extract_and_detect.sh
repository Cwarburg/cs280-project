#!/bin/bash
#BSUB -J extract_detect
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 12:00
#BSUB -o /work3/s225083/cs280/project/logs/extract_%J.out
#BSUB -e /work3/s225083/cs280/project/logs/extract_%J.err

source /work3/s225083/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /work3/s225083/cs280/project

mkdir -p logs

echo "=== Extracting frames ==="
python extract_frames.py \
    --src SoccerNet/england_epl \
    --out frames \
    --fps 2

echo "=== Running scene detection ==="
python scene_detect.py \
    --frames_root frames \
    --out cuts.json \
    --threshold 0.35

echo "=== Done ==="
