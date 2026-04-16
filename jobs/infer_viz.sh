#!/bin/bash
#BSUB -J infer_viz
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o /work3/s225083/cs280/project/logs/infer_viz_%J.out
#BSUB -e /work3/s225083/cs280/project/logs/infer_viz_%J.err

source /work3/s225083/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /work3/s225083/cs280/project

mkdir -p logs viz_inference

echo "=== Inference + GradCAM visualization (resnet50_r1 best.pt) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python infer_viz.py \
    --n_samples 10 \
    --seq_len   12 \
    --seed      42 \
    --out       viz_inference

echo "=== Done. Output in viz_inference/ ==="
