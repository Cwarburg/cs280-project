#!/bin/bash
#BSUB -J train_temporal
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o /work3/s225083/cs280/project/logs/train_%J.out
#BSUB -e /work3/s225083/cs280/project/logs/train_%J.err

source /work3/s225083/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /work3/s225083/cs280/project

mkdir -p logs checkpoints

echo "=== Training pairwise temporal ordering model ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python train.py \
    --frames_root     frames \
    --cuts_json       cuts.json \
    --encoder         vit_small_patch16_224_dino \
    --embed_dim       256 \
    --hidden_dim      512 \
    --dropout         0.3 \
    --epochs          30 \
    --batch_size      128 \
    --lr              1e-4 \
    --cut_radius      5 \
    --val_fraction    0.1 \
    --num_workers     4 \
    --tau_n_seq       100 \
    --tau_seq_len     30 \
    --data_fraction   0.1 \
    --out             checkpoints/dino_vits16_cutwindow

echo "=== Done ==="
