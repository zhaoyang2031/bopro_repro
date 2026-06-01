#!/bin/bash
# GFN 30 seeds on GPU1: mpo2obj + mpo3obj + mpo4obj (seed 0-9)
source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate genetic_gfn
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

cd /data/xk/zhaoyang/ram_repro/genetic_gfn-main/pmo

echo "GPU1: GFN 30 seeds (mpo2obj + mpo3obj + mpo4obj, seed 0-9)"
echo "Start: $(date)"

for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "[$(date)] MPO2OBJ seed=$seed"
    python -u run.py genetic_gfn --oracles jnk3 gsk3b --max_oracle_calls 10000 --seed $seed --wandb online --run_name GFN_MPO2obj_seed${seed}
    echo "[$(date)] MPO3OBJ seed=$seed"
    python -u run.py genetic_gfn --oracles jnk3 gsk3b qed --max_oracle_calls 10000 --seed $seed --wandb online --run_name GFN_MPO3obj_seed${seed}
    echo "[$(date)] MPO4OBJ seed=$seed"
    python -u run.py genetic_gfn --oracles jnk3 gsk3b qed sa --max_oracle_calls 10000 --seed $seed --wandb online --run_name GFN_MPO4obj_seed${seed}
done

echo "[$(date)] GPU1: All 30 GFN seeds done"
