#!/bin/bash
# launch_bopro_random.sh
# BOPRO MolOpt: random方法，8个蛋白质 × 3种子 = 24个实验
# GPU1和GPU3并行，各12个实验

source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate bopro
cd /data/xk/zhaoyang/bopro_repro/bopro
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/data/xk/zhaoyang/bopro_repro/bopro:$PYTHONPATH
export WANDB_MODE=online

OUT_DIR="/nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b"
LOG_DIR="/nas1/xk/zhaoyang/bopro_repro/logs"
mkdir -p $OUT_DIR $LOG_DIR

# 8个蛋白质 (去掉CDK1)
PROTEINS=(SRC EGFR ABL1 CDK2 AKT1 MAPK1 AKT2 KDR)
SEEDS=(0 1 2)

# 生成任务列表
TASKS=()
for protein in "${PROTEINS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_id="random_${protein}_seed${seed}"
        if [ -f "${OUT_DIR}/${run_id}/SRC/results.json" ]; then
            echo "[SKIP] ${run_id}"
            continue
        fi
        TASKS+=("${protein}|${seed}")
    done
done

TOTAL=${#TASKS[@]}
echo "=========================================="
echo "Random method: ${TOTAL} tasks"
echo "GPUs: 1 and 3"
echo "=========================================="

# 分配到GPU1和GPU3
GPU1_TASKS=()
GPU3_TASKS=()
for i in "${!TASKS[@]}"; do
    if [ $((i % 2)) -eq 0 ]; then
        GPU1_TASKS+=("${TASKS[$i]}")
    else
        GPU3_TASKS+=("${TASKS[$i]}")
    fi
done

echo "GPU1: ${#GPU1_TASKS[@]} tasks"
echo "GPU3: ${#GPU3_TASKS[@]} tasks"

# 运行函数
run_on_gpu() {
    local gpu=$1
    shift
    local tasks=("$@")
    export CUDA_VISIBLE_DEVICES=$gpu

    for task in "${tasks[@]}"; do
        IFS='|' read -r protein seed <<< "$task"
        run_id="random_${protein}_seed${seed}"
        echo "[$(date)] GPU${gpu}: ${run_id}"

        python -u src/molopt_bo.py \
            --gen_model=qwen-2-7b-instruct \
            --repr_model=molformer \
            --repr_prompt=target_based \
            --low_dim_strategy=off \
            --acquisition_fn=random \
            --out_dir=$OUT_DIR \
            --run_id=$run_id \
            --task_fpath=data/molopt/data.json \
            --target=$protein \
            --n_evaluations=200 \
            --n_seeds=1 \
            --seed=$seed \
            --llm_temperature=1.0 \
            --llm_tokens=512 \
            --llm_top_p=0.9 \
            --opt_batch_size=1 \
            --vec2text_batch_size=10 \
            --vec2text_n_parallel=10 \
            --vec2text_unique_retries=3 \
            --vec2text_demos=10 \
            --kernel_mean_prior_mean=0.4 \
            --kernel_mean_prior_std=0.01 \
            --kernel_lengthscale_prior_concentration=4 \
            --kernel_lengthscale_prior_rate=2 \
            --kernel_outputscale_prior_concentration=4 \
            --kernel_outputscale_prior_rate=2 \
            --gp_kernel=matern \
            --gp_noise_var=0.001 \
            --no-kernel_per_dim_lengthscale \
            --use_method_defaults \
            --no-arc_use_scores \
            --no-visualize_posterior \
            > "${LOG_DIR}/${run_id}.log" 2>&1

        echo "[$(date)] GPU${gpu}: Done ${run_id}"
    done
}

echo "Start: $(date)"

# GPU1和GPU3并行
run_on_gpu 1 "${GPU1_TASKS[@]}" &
pid1=$!
run_on_gpu 3 "${GPU3_TASKS[@]}" &
pid3=$!

echo "PIDs: GPU1=$pid1 GPU3=$pid3"

wait $pid1 $pid3

echo ""
echo "=========================================="
echo "Random experiments completed!"
echo "End: $(date)"
echo "=========================================="
