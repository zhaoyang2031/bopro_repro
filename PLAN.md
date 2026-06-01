# BOPRO MolOpt (dockstring) 复现计划

## 目标
使用 Qwen2-7B-Instruct 复现 BOPRO 论文的 MolOpt (dockstring) benchmark。

## 项目信息
- GitHub: https://github.com/zhaoyang2031/bopro_repro
- 服务器: ssh xk@180.209.6.83 (4x RTX 4090, 24GB VRAM)
- 根目录: /data/xk/zhaoyang
- 代码目录: /data/xk/zhaoyang/bopro_repro/bopro
- 结果存储: /nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b
- Conda环境: ddom_gtg (已存在，扩展安装依赖)

## 技术分析

### 为什么 Qwen2-7B-Instruct 可行？
1. **已支持**: `qwen-2-7b-instruct` 已在 `HF_PIPELINE_MODELS` 字典中注册
2. **显存**: 模型约14GB (fp16)，RTX 4090 (24GB) 完全够用
3. **架构**: 代码使用 HuggingFace pipeline 加载本地模型，无需API
4. **服务器**: 4张RTX 4090，可并行运行4个实验

### 与原始论文的差异
| 项目 | 原始论文 | 我们的复现 |
|------|----------|------------|
| 生成模型 | mistral-large-2407 (70B+) | Qwen2-7B-Instruct (7B) |
| 推理方式 | Bedrock API | 本地HF pipeline |
| 预期效果 | 基准效果 | 可能略低 (模型更小) |

## 实验配置

### 实验参数 (严格参考论文 Section A.4.1)
```bash
# 实验方法 (论文Section 6.1)
# BOPRO变体: logEI, UCB, thompson_sampling
# Baselines: OPRO(greedy), RS(repeated sampling), random
# 最小验证: logEI vs OPRO vs RS (3个方法)

# 蛋白质靶点: 58个 (来自Dockstring benchmark)

# 核心参数 (论文Section A.4.1)
--gen_model="qwen-2-7b-instruct"      # 生成模型
--repr_model="molformer"               # 表示模型
--repr_prompt="target_based"           # 表示提示
--low_dim_strategy="off"               # 不使用降维 (论文默认)
--n_evaluations=200                    # 解决方案生成数 (论文: 200)
--opt_batch_size=1                     # BO优化batch size
--vec2text_batch_size=10               # 解码batch size (论文: 10)
--vec2text_n_parallel=10               # 并行解码数
--vec2text_unique_retries=3            # 重复重试次数 (论文: 3)
--vec2text_demos=10                    # 上下文示例数 (论文: 10)
--llm_temperature=1.0                  # LLM温度 (论文: 1.0)
--llm_tokens=512                       # 最大token数 (论文: 512)
--llm_top_p=0.9                        # Top-p采样 (论文: 0.9)

# GP超参数 (论文Section A.4.2)
--gp_kernel="matern"                   # Matérn 5/2 kernel
--gp_noise_var=0.001                   # 噪声方差
--kernel_lengthscale_prior_concentration=4  # 长度尺度先验
--kernel_lengthscale_prior_rate=2
--kernel_outputscale_prior_concentration=4  # 输出尺度先验
--kernel_outputscale_prior_rate=2
--kernel_mean_prior_mean=0.4           # 均值先验
--kernel_mean_prior_std=0.01

# 其他设置
--use_method_defaults                  # 使用方法默认值
--no-arc_use_scores                    # 不使用ARC分数
--no-visualize_posterior               # 不可视化后验
--no-kernel_per_dim_lengthscale        # 不使用每维长度尺度
```

### 实验规模 (论文Section 7.2)
- **目标**: 58个蛋白质靶点
- **方法**: 6个 (BOPRO-LogEI, BOPRO-UCB, BOPRO-TS, OPRO, RS, Random)
- **种子**: 3个
- **总实验数**: 58 × 6 × 3 = 1044个
- **每个实验**: 200次解决方案生成
- **最终指标**: 所有蛋白质的平均scalarized objective

### 最小验证方案 (推荐先跑)
- **方法**: 3个 (BOPRO-LogEI, OPRO, RS)
- **蛋白质**: 5-10个 (先测试)
- **种子**: 3个
- **总实验数**: 10 × 3 × 3 = 90个

## 实施步骤

### Phase 1: 环境准备 (服务器) ✅ 已完成
1. **SSH登录服务器**
   ```bash
   ssh xk@180.209.6.83
   ```

2. **使用现有conda环境** ✅
   ```bash
   source /data/xk/zhaoyang/miniconda3/etc/profile.d/conda.sh
   conda activate ddom_gtg
   # Python 3.9.23, torch 2.8.0, transformers 4.57.6
   ```

3. **安装缺失依赖** ✅ 已执行 (正在后台安装)
   ```bash
   conda install -c conda-forge -y sentence-transformers dockstring gensim
   ```

4. **上传代码** ✅ 已完成
   ```bash
   # 本地执行
   scp -r D:/BOPRO-ICLR-2025-main/bopro xk@180.209.6.83:/data/xk/zhaoyang/bopro_repro/
   scp D:/BOPRO-ICLR-2025-main/launch_bopro_molopt.sh xk@180.209.6.83:/data/xk/zhaoyang/bopro_repro/
   ```

5. **验证代码**
   ```bash
   ls -la /data/xk/zhaoyang/bopro_repro/bopro/
   ls -la /data/xk/zhaoyang/bopro_repro/bopro/src/
   ```

### Phase 2: 代码适配
1. **检查 MolFormer 模型路径**
   - 在 `utils/generation.py` 中，`HF_SFORMER_MODELS["molformer"]` 指向本地路径
   - 需要确认服务器上是否有该模型，或下载到正确位置
   - 可能需要修改为: `/data/xk/zhaoyang/.cache/huggingface/hub/models--sentence-transformers--ibm--MoLFormer-XL-both-10pct`

2. **调整批量参数**
   - 根据显存调整 `vec2text_n_parallel` (RTX 4090建议设为1)
   - 根据GPU数量调整并行实验数 (最多4个)

3. **检查 dockstring 依赖**
   - dockstring 需要 autodock-vina
   - 可能需要安装: `conda install -c conda-forge vina`

### Phase 3: 运行实验

#### 方案: 使用launch脚本 (参考launch_gpu0_gfn_dock.sh风格)

```bash
#!/bin/bash
# launch_bopro_molopt.sh
# 在单个GPU上运行BOPRO MolOpt实验
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bopro
cd /data/xk/zhaoyang/bopro_repro/bopro

# 设置GPU (不要用GPU 0)
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# 创建输出目录
OUT_DIR="/nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b"
mkdir -p $OUT_DIR

# 实验配置
METHODS=("logEI" "OPRO" "RS")  # 3个方法: BOPRO-LogEI, OPRO(greedy), RS(repeated sampling)
SEEDS=(0 1 2)                   # 3个种子

# 蛋白质靶点 (先测试5个)
PROTEINS=(SRC EGFR ABL1 CDK2 AKT1)

echo "BOPRO MolOpt with Qwen2-7B-Instruct"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Methods: ${METHODS[*]}"
echo "Proteins: ${PROTEINS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Start: $(date)"

# 运行实验
for method in "${METHODS[@]}"; do
    for protein in "${PROTEINS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN_ID="${method}_${protein}_seed${seed}"
            echo "[$(date)] Running: $RUN_ID"
            
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
            
            echo "[$(date)] Completed: $RUN_ID"
        done
    done
done

echo "[$(date)] All experiments completed!"
```

#### 使用方法
```bash
# 1. 上传脚本到服务器
scp launch_bopro_molopt.sh xk@180.209.6.83:/data/xk/zhaoyang/bopro_repro/

# 2. SSH登录并运行
ssh xk@180.209.6.83
cd /data/xk/zhaoyang/bopro_repro
chmod +x launch_bopro_molopt.sh
nohup ./launch_bopro_molopt.sh > logs/molopt_run.log 2>&1 &

# 3. 监控进度
tail -f logs/molopt_run.log
```

### Phase 4: 结果收集与分析
1. **结果目录结构**
   ```
   /nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b/
   ├── {run_id}/
   │   ├── {protein}/
   │   │   ├── seed-0.json
   │   │   ├── seed-1.json
   │   │   └── seed-2.json
   │   └── results.json  (聚合结果)
   ```

2. **关键指标** (在results.json中)
   - `avg_best_y`: 平均最优分数
   - `avg_opt_rate`: 达到最优 (score=1.0) 的成功率
   - `avg_steps_to_opt`: 达到最优的平均步数
   - `avg_gain_from_warmstart`: 相比warmstart的提升
   - `avg_best_y_qed`: 平均QED分数
   - `avg_best_y_vina`: 平均Vina分数

3. **与论文对比**
   - 论文 Table 1: MolOpt 结果
   - 对比不同BO方法 (logEI, UCB, TS, random, none) 的效果
   - 分析Qwen2-7B vs Mistral-70B的差异
   - 注意: 7B模型效果可能略低，但应能看到BO方法间的相对差异

4. **提取结果脚本**
   ```bash
   # 提取所有实验的最佳分数
   find /nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b -name "results.json" -exec grep -l "avg_best_y" {} \;
   
   # 汇总统计
   python -c "
   import json, glob
   results = []
   for f in glob.glob('/nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b/*/results.json'):
       with open(f) as fh:
           r = json.load(fh)
           results.append(r)
   # 按acquisition_fn分组统计
   from collections import defaultdict
   stats = defaultdict(list)
   for r in results:
       acq = r['args']['acquisition_fn']
       stats[acq].append(r['avg_best_y'])
   for acq, vals in stats.items():
       print(f'{acq}: {sum(vals)/len(vals):.4f} ± {np.std(vals):.4f}')
   "
   ```

## 注意事项

### 1. 模型加载
- Qwen2-7B-Instruct 需要约14GB显存 (fp16)
- RTX 4090 有24GB显存，足够运行
- 使用 `device_map="auto"` 自动分配到可用GPU
- 确保CUDA_VISIBLE_DEVICES设置正确
- **注意**: 不要同时在一张卡上加载多个模型

### 2. 并行策略
- **单个实验**: 1张RTX 4090足够 (Qwen2-7B约14GB)
- **多实验并行**: 可同时运行3个实验 (3张GPU，留1张给交互)
- **注意**: RTX 4090显存比A100小，需控制并发
- **建议**: 使用launch脚本顺序运行，简单可靠
- **GPU分配**: 不要使用GPU 0 (留给交互式调试)

### 3. 时间预估
- 单个实验 (200次评估): 约15-30分钟
- 最小验证 (90个实验): 约22-45小时 (串行)
- 全部1044个实验: 约260-520小时 (串行)
- 3GPU并行: 约87-173小时 (3.6-7.2天)
- **RTX 4090 vs A100**: 4090较慢，实际时间可能更长
- **建议**: 先跑最小验证，再决定是否全量

### 4. 检查点与恢复
- 代码支持 `--load_from_prev_run` 参数
- 可以从之前的运行继续
- 建议分批运行，每批10-20个蛋白质
- 队列系统支持 `--skip-completed` 自动跳过已完成

### 5. 已知问题
- dockstring 可能需要额外的系统依赖 (如 autodock-vina)
- MolFormer 模型路径可能需要调整
- 某些蛋白质对接可能失败 (会被跳过)
- RTX 4090 显存有限，注意监控使用情况

## 调试建议

### 快速验证
先在单个蛋白质上测试:
```bash
ssh xk@180.209.6.83
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bopro
cd /data/xk/zhaoyang/bopro_repro/bopro

export CUDA_VISIBLE_DEVICES=1
python -u src/molopt_bo.py \
    --gen_model="qwen-2-7b-instruct" \
    --repr_model="molformer" \
    --repr_prompt="target_based" \
    --low_dim_strategy="off" \
    --acquisition_fn="logEI" \
    --task_fpath="data/molopt/data.json" \
    --target="SRC" \
    --n_evaluations=10 \
    --n_seeds=1 \
    --seed=0 \
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
    --no-visualize_posterior \
    --verbose
```

### 性能监控
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控实验进度
tail -f /nas1/xk/zhaoyang/bopro_repro/logs/queue_molopt.log

# 查看具体实验日志
tail -f /nas1/xk/zhaoyang/bopro_repro/logs/full/bopro_molopt_*.log
```

## 参考资源
- 论文: https://openreview.net/pdf?id=aVfDrl7xDV
- 原始代码: D:\BOPRO-ICLR-2025-main\bopro
- 服务器指南: D:\BOPRO-ICLR-2025-main\bbo\repro\HOWTO.md
- GPU启动脚本: D:\BOPRO-ICLR-2025-main\ram\launch_gpu0_gfn_dock.sh
