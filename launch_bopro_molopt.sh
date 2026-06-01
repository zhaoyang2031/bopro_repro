#!/bin/bash
# launch_bopro_molopt.sh
# BOPRO MolOpt (Dockstring) 复现脚本
# 使用 Qwen2-7B-Instruct 在 RTX 4090 上运行
# 参考论文: "Searching for Optimal Solutions with LLMs via Bayesian Optimization" (ICLR 2025)

# 环境设置
source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate ddom_gtg
cd /data/xk/zhaoyang/bopro_repro/bopro

# GPU设置 (不要用GPU 0，留给交互式调试)
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# 输出目录 (使用/nas1避免/data磁盘满)
OUT_DIR="/nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b"
mkdir -p $OUT_DIR

# ============================================================
# 实验配置 (严格参考论文 Section A.4.1)
# ============================================================

# 方法: BOPRO-LogEI, OPRO(greedy), RS(repeated sampling)
# 论文Section 6.1: 3个BOPRO变体 + OPRO + RS + Random
# 最小验证: 只跑3个核心方法
METHODS=("logEI" "OPRO" "RS")

# 种子数: 3个 (论文Section 7.2)
SEEDS=(0 1 2)

# 蛋白质靶点: 先测试5个，后续可扩展到58个
PROTEINS=(SRC EGFR ABL1 CDK2 AKT1)

# ============================================================
# 论文参数 (Section A.4.1)
# ============================================================
# --n_evaluations=200           # 解决方案生成数
# --opt_batch_size=1            # BO优化batch size
# --vec2text_batch_size=10      # 解码batch size
# --vec2text_unique_retries=3   # 重复重试次数
# --vec2text_demos=10           # 上下文示例数
# --llm_temperature=1.0         # LLM温度
# --llm_tokens=512              # 最大token数
# --llm_top_p=0.9               # Top-p采样
# ============================================================

echo "=========================================="
echo "BOPRO MolOpt with Qwen2-7B-Instruct"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Methods: ${METHODS[*]}"
echo "Proteins: ${PROTEINS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Output: $OUT_DIR"
echo "Start: $(date)"
echo "=========================================="

# 运行实验
TOTAL=$(( ${#METHODS[@]} * ${#PROTEINS[@]} * ${#SEEDS[@]} ))
COUNT=0

for method in "${METHODS[@]}"; do
    for protein in "${PROTEINS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))
            RUN_ID="${method}_${protein}_seed${seed}"
            echo ""
            echo "[$(date)] ($COUNT/$TOTAL) Running: $RUN_ID"

            python -u src/molopt_bo.py \
                --gen_model="qwen-2-7b-instruct" \
                --repr_model="molformer" \
                --repr_prompt="target_based" \
                --low_dim_strategy="off" \
                --acquisition_fn=$method \
                --out_dir=$OUT_DIR \
                --run_id=$RUN_ID \
                --task_fpath="data/molopt/data.json" \
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
                --gp_kernel="matern" \
                --gp_noise_var=0.001 \
                --no-kernel_per_dim_lengthscale \
                --use_method_defaults \
                --no-arc_use_scores \
                --no-visualize_posterior

            if [ $? -eq 0 ]; then
                echo "[$(date)] Completed: $RUN_ID"
            else
                echo "[$(date)] FAILED: $RUN_ID"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End: $(date)"
echo "Results saved to: $OUT_DIR"
echo "=========================================="
