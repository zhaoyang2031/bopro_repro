#!/bin/bash
# launch_qwen35_opro_bopro.sh
# 使用 Qwen3.5-4B (SGLang部署) 复现 OPRO 和 BOPRO(logEI)
# GPU 2: SGLang 服务 (Qwen3.5-4B)
# GPU 3: 实验代码 (molformer + BO)

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
GEN_MODEL="qwen-3.5-4b"
REPR_MODEL="molformer"
OUT_DIR="/nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen35-4b"
LOG_DIR="/nas1/xk/zhaoyang/bopro_repro/logs-qwen35-4b"
SGLANG_PORT=30000
mkdir -p $OUT_DIR $LOG_DIR

PROTEINS=(SRC EGFR ABL1 CDK2 AKT1 CDK1 MAPK1 AKT2 KDR)
METHODS=(OPRO logEI)  # OPRO 和 BOPRO(logEI)
SEEDS=(0 1 2)

# ============================================================
# 1. 启动 SGLang 服务 (GPU 2)
# ============================================================
echo "=========================================="
echo "Starting SGLang server on GPU 2..."
echo "=========================================="
export CUDA_VISIBLE_DEVICES=2

# 启动 SGLang 服务
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port $SGLANG_PORT \
    --reasoning-parser qwen3 \
    > $LOG_DIR/sglang_server.log 2>&1 &

SGLANG_PID=$!
echo "SGLang PID: $SGLANG_PID"

# 等待服务就绪
echo "Waiting for SGLang server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:$SGLANG_PORT/v1/models > /dev/null 2>&1; then
        echo "SGLang server is ready! (took ${i}s)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: SGLang server failed to start within 120s"
        cat $LOG_DIR/sglang_server.log
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ============================================================
# 2. 运行实验 (GPU 3)
# ============================================================
export CUDA_VISIBLE_DEVICES=3

# 生成任务列表 (跳过已完成的)
TASKS=()
for method in "${METHODS[@]}"; do
    for protein in "${PROTEINS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_id="${method}_${protein}_seed${seed}"
            if [ -f "${OUT_DIR}/${run_id}/${protein}/results.json" ]; then
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
echo "Methods: OPRO, logEI (BOPRO)"
echo "Gen model: $GEN_MODEL (SGLang)"
echo "Repr model: $REPR_MODEL"
echo "GPU: 3 (experiment), 2 (SGLang)"
echo "=========================================="
echo "Start: $(date)"

for task in "${TASKS[@]}"; do
    IFS='|' read -r method protein seed <<< "$task"
    run_id="${method}_${protein}_seed${seed}"
    log="${LOG_DIR}/${run_id}.log"

    echo "[$(date)] Running: ${run_id}"
    python -u src/molopt_bo.py \
        --gen_model=$GEN_MODEL \
        --repr_model=$REPR_MODEL \
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
    echo "[$(date)] Done: ${run_id}"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End: $(date)"
echo "Results: $OUT_DIR"
echo "=========================================="

# 关闭 SGLang 服务
echo "Shutting down SGLang server..."
kill $SGLANG_PID 2>/dev/null
