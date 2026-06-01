#!/bin/bash
# Saturn 30 seeds on GPU0: mpo2obj + mpo3obj + mpo4obj (seed 0-9)
source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate saturn
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

cd /data/xk/zhaoyang/ram_repro/saturn-master

echo "GPU0: Saturn 30 seeds (mpo2obj + mpo3obj + mpo4obj, seed 0-9)"
echo "Start: $(date)"

for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "[$(date)] MPO2OBJ seed=$seed"
    python -u saturn.py configs/mpo2obj_seed${seed}.json
    echo "[$(date)] MPO3OBJ seed=$seed"
    python -u saturn.py configs/mpo3obj_seed${seed}.json
    echo "[$(date)] MPO4OBJ seed=$seed"
    python -u saturn.py configs/mpo4obj_seed${seed}.json
done

echo "[$(date)] GPU0: All 30 Saturn seeds done"
