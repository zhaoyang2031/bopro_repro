#!/bin/bash
# launch_bopro_full.sh
# BOPRO MolOpt 完整实验：81个实验 (9蛋白质 × 3方法 × 3种子)
# 4个GPU并行，每个GPU跑20个实验
# 已完成: logEI_SRC_seed0，剩余80个

# ============================================================
# 环境设置
# ============================================================
source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate bopro
cd /data/xk/zhaoyang/bopro_repro/bopro
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/data/xk/zhaoyang/bopro_repro/bopro:$PYTHONPATH
export WANDB_MODE=online

# ============================================================
# 配置
# ============================================================
OUT_DIR="/nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b"
LOG_DIR="/nas1/xk/zhaoyang/bopro_repro/logs"
mkdir -p $OUT_DIR $LOG_DIR

PROTEINS=(SRC EGFR ABL1 CDK2 AKT1 CDK1 MAPK1 AKT2 KDR)
METHODS=(logEI OPRO RS)
SEEDS=(0 1 2)

# ============================================================
# 生成任务列表 (跳过已完成的)
# ============================================================
TASKS=()
for method in "${METHODS[@]}"; do
    for protein in "${PROTEINS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_id="${method}_${protein}_seed${seed}"
            # 跳过已完成的
            if [ -f "${OUT_DIR}/${run_id}/SRC/results.json" ]; then
                echo "[SKIP] ${run_id} (already done)"
                continue
            fi
            TASKS+=("${method}|${protein}|${seed}")
        done
    done
done

TOTAL=${#TASKS[@]}
echo "=========================================="
echo "Total tasks to run: $TOTAL"
echo "GPUs: 4 (0,1,2,3)"
echo "=========================================="

# ============================================================
# 每个GPU分配任务并顺序执行
# ============================================================
run_on_gpu() {
    local gpu=$1
    shift
    local tasks=("$@")

    export CUDA_VISIBLE_DEVICES=$gpu

    for task in "${tasks[@]}"; do
        IFS='|' read -r method protein seed <<< "$task"
        run_id="${method}_${protein}_seed${seed}"
        log="${LOG_DIR}/${run_id}.log"

        echo "[$(date)] GPU${gpu}: ${run_id}"
        python -u src/molopt_bo.py \
            --gen_model=qwen-2-7b-instruct \
            --repr_model=molformer \
            --repr_prompt=target_based \
            --low_dim_strategy=off \
            --acquisition_fn=$method \
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
            > "$log" 2>&1
        echo "[$(date)] GPU${gpu}: Done ${run_id}"
    done
}

# ============================================================
# 将任务均匀分配到4个GPU
# ============================================================
GPU0_TASKS=()
GPU1_TASKS=()
GPU2_TASKS=()
GPU3_TASKS=()

for i in "${!TASKS[@]}"; do
    case $((i % 4)) in
        0) GPU0_TASKS+=("${TASKS[$i]}") ;;
        1) GPU1_TASKS+=("${TASKS[$i]}") ;;
        2) GPU2_TASKS+=("${TASKS[$i]}") ;;
        3) GPU3_TASKS+=("${TASKS[$i]}") ;;
    esac
done

echo "GPU0: ${#GPU0_TASKS[@]} tasks"
echo "GPU1: ${#GPU1_TASKS[@]} tasks"
echo "GPU2: ${#GPU2_TASKS[@]} tasks"
echo "GPU3: ${#GPU3_TASKS[@]} tasks"
echo ""
echo "Start: $(date)"

# ============================================================
# 4个GPU并行执行
# ============================================================
run_on_gpu 0 "${GPU0_TASKS[@]}" &
pid0=$!
run_on_gpu 1 "${GPU1_TASKS[@]}" &
pid1=$!
run_on_gpu 2 "${GPU2_TASKS[@]}" &
pid2=$!
run_on_gpu 3 "${GPU3_TASKS[@]}" &
pid3=$!

echo "PIDs: GPU0=$pid0 GPU1=$pid1 GPU2=$pid2 GPU3=$pid3"

# ============================================================
# 等待所有GPU完成
# ============================================================
wait $pid0 $pid1 $pid2 $pid3

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End: $(date)"
echo "Results: $OUT_DIR"
echo "=========================================="
