#!/bin/bash
# Saturn docking 50 runs on GPU2: 5 targets x 10 seeds
source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate saturn
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

cd /data/xk/zhaoyang/ram_repro/saturn-master

TARGETS=(parp1 fa7 jak2 braf 5ht1b)

echo "GPU2: Saturn docking 50 runs (5 targets x 10 seeds)"
echo "Start: $(date)"

for target in "${TARGETS[@]}"; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        echo "[$(date)] Saturn dock_${target} seed=$seed"
        python -u run_docking.py --task "dock_${target}" --seed $seed --device cuda:0 --wandb online
    done
done

echo "[$(date)] GPU2: All 50 Saturn docking seeds done"
