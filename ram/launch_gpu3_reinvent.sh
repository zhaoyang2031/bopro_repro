#!/bin/bash
source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate reinvent4
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=4

cd /data/xk/zhaoyang/ram_repro/REINVENT4-main

echo "GPU3: REINVENT4 30 seeds (mpo2obj + mpo3obj + mpo4obj, seed 0-9)"

for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "[$(date)] MPO2OBJ seed=$seed"
    python run_offline.py --task mpo2obj --seed $seed --device cuda:0 --wandb online
    echo "[$(date)] MPO3OBJ seed=$seed"
    python run_offline.py --task mpo3obj --seed $seed --device cuda:0 --wandb online
    echo "[$(date)] MPO4OBJ seed=$seed"
    python run_offline.py --task mpo4obj --seed $seed --device cuda:0 --wandb online
done

echo "[$(date)] GPU3: All 30 seeds done"
