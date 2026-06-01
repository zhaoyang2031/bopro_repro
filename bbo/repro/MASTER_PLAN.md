# BBO复现总计划

## 服务器配置

- **服务器**: xk@ta2 (4x RTX 4090)
- **项目根目录**: /data/xk/zhaoyang
- **结果存储**: /nas1/xk/zhaoyang/bbo_repro (避免/data磁盘满)
- **日志目录**: /nas1/xk/zhaoyang/bbo_repro/logs

## 已完成的实验

| 算法 | Benchmark | 维度/任务 | 重复次数 | 状态 |
|------|-----------|----------|---------|------|
| DiBO | Ackley | 200D | 4/4 seeds | ✅ 完成 |

## 实验执行顺序

### 🟥 第一优先级：DDOM (Diffusion Models for Black-Box Optimization)

**论文**: Diffusion Models for Black-Box Optimization

**Benchmark**: Design-Bench任务 (6个任务 × 3 seeds)
1. dkitty (DKittyMorphology-Exact-v0) - 3 seeds
2. ant (AntMorphology-Exact-v0) - 3 seeds
3. tf-bind-8 (TFBind8-Exact-v0) - 3 seeds
4. tf-bind-10 (TFBind10-Exact-v0) - 3 seeds
5. superconductor (Superconductor-RandomForest-v0) - 3 seeds
6. nas (CIFARNAS-Exact-v0) - 3 seeds

**总实验数**: 18个

**预计时间**: 
- 每个任务: ~2小时/seed
- 总计: ~36小时 (4卡并行约9小时)

**配置文件**: `ddom_full_experiments.json`

---

### 🟧 第二优先级：GTG (Guided Trajectory Generation)

**论文**: Guided Trajectory Generation with Diffusion Models for Offline Model-based Optimization

**Benchmark**: Design-Bench任务 (5个任务 × 3 seeds)
1. ant - 3 seeds
2. dkitty - 3 seeds
3. superconductor - 3 seeds
4. tfbind8 - 3 seeds
5. tfbind10 - 3 seeds

**总实验数**: 15个

**预计时间**: 待评估

**配置文件**: `gtg_full_experiments.json`

---

### 🟨 第三优先级：VSD (Variational Search Distributions)

**论文**: Variational Search Distributions

**Benchmark**: Ehrlich序列优化任务 (3个序列长度 × 2种proposal × 2 seeds)
- seqlen=15, 32, 64
- proposal: tfm (Transformer), mf (Multi-factorial)
- seeds: 42, 123

**总实验数**: 9个

**预计时间**: ~2-4小时/任务

**配置文件**: `vsd_full_experiments.json`

**依赖环境**: `vsd_genbo` (Python 3.10)

---

### 🟩 第四优先级：GenBO (Generative Bayesian Optimization)

**论文**: Generative Bayesian Optimization: Generative Models as Acquisition Functions

**Benchmark**: Ehrlich序列优化任务 (3个序列长度 × 2种proposal × 2 seeds)
- seqlen=15, 32, 64
- proposal: tfm (Transformer), mf (Multi-factorial)
- seeds: 42, 123

**总实验数**: 8个

**预计时间**: ~2-4小时/任务

**配置文件**: `genbo_full_experiments.json`

**依赖环境**: `vsd_genbo` (Python 3.10)

---

### 🟦 第五优先级：NF-BO (Latent Bayesian Optimization)

**论文**: Latent Bayesian Optimization via Autoregressive Normalizing Flows

**Benchmark**: Guacamol分子优化任务 (7个任务 × 5 repetitions)
1. adip - 5 repetitions
2. med2 - 5 repetitions
3. osmb - 5 repetitions
4. pdop - 5 repetitions
5. rano - 5 repetitions
6. zale - 5 repetitions
7. valt - 5 repetitions

**总实验数**: 35个

**预计时间**: 
- 每个任务: ~1小时/rep
- 总计: ~35小时 (4卡并行约9小时)

**配置文件**: `nfbo_500_experiments.json`

---

### 🟪 第六优先级：DiBO (Posterior Inference with Diffusion Models)

**论文**: Posterior Inference with Diffusion Models for High-Dimensional Black-box Optimization

**Benchmark**:
1. Ackley (200D, 400D) - 4 seeds each
2. Rastrigin (200D, 400D) - 4 seeds each
3. Rosenbrock (200D, 400D) - 4 seeds each
4. Levy (200D, 400D) - 4 seeds each

**总实验数**: 32个 (已完成4个)

**预计时间**: 
- 200D: ~2小时/seed
- 400D: ~4小时/seed
- 总计: ~88小时 (4卡并行约22小时)

**配置文件**: `dibo_full_experiments.json`

---

## 并行策略

### 按顺序串行执行 (推荐)

```
阶段1: DDOM全部实验 (GPU0-3并行)
  └─ 完成后自动进入阶段2

阶段2: GTG全部实验 (GPU0-3并行)
  └─ 完成后自动进入阶段3

阶段3: VSD Ehrlich (GPU0-3并行)
  └─ 完成后自动进入阶段4

阶段4: GenBO Ehrlich (GPU0-3并行)
  └─ 完成后自动进入阶段5

阶段5: NF-BO全部实验 (GPU0-3并行)
  └─ 完成后自动进入阶段6

阶段6: DiBO剩余实验 (GPU0-3并行)
```

**优点**: 简单可靠，不会混淆不同算法的结果
**缺点**: 总时间较长

---

## 执行步骤

### Step 1: DDOM实验 (18个)

```bash
# 启动DDOM队列
python run_queue.py \
  --experiments ddom_full_experiments.json \
  --suite full \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --skip-completed \
  --logdir /nas1/xk/zhaoyang/bbo_repro/logs
```

### Step 2: GTG实验 (15个)

```bash
# 启动GTG队列
python run_queue.py \
  --experiments gtg_full_experiments.json \
  --suite full \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --skip-completed \
  --logdir /nas1/xk/zhaoyang/bbo_repro/logs
```

### Step 3: VSD Ehrlich实验 (9个)

```bash
# 启动VSD队列
python run_queue.py \
  --experiments vsd_full_experiments.json \
  --suite full \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --skip-completed \
  --logdir /nas1/xk/zhaoyang/bbo_repro/logs
```

### Step 4: GenBO Ehrlich实验 (8个)

```bash
# 启动GenBO队列
python run_queue.py \
  --experiments genbo_full_experiments.json \
  --suite full \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --skip-completed \
  --logdir /nas1/xk/zhaoyang/bbo_repro/logs
```

### Step 5: NF-BO实验 (35个)

```bash
# 启动NF-BO队列
python run_queue.py \
  --experiments nfbo_500_experiments.json \
  --suite full \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --skip-completed \
  --logdir /nas1/xk/zhaoyang/bbo_repro/logs
```

### Step 6: DiBO剩余实验 (28个)

```bash
# 启动DiBO队列
python run_queue.py \
  --experiments dibo_full_experiments.json \
  --suite full \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --skip-completed \
  --logdir /nas1/xk/zhaoyang/bbo_repro/logs
```

---

## 预期输出

### 图表输出

每个benchmark生成一张图表：
- **横轴**: 函数评估次数
- **纵轴**: 最佳得分
- **阴影**: 标准误差 (Standard error over N trials)

### 文件结构

```
/nas1/xk/zhaoyang/bbo_repro/
├── logs/
│   ├── full/
│   │   ├── ddom_full_*.gpu*.log
│   │   ├── gtg_full_*.gpu*.log
│   │   ├── vsd_full_*.gpu*.log
│   │   ├── genbo_full_*.gpu*.log
│   │   ├── nfbo_*.gpu*.log
│   │   └── dibo_full_*.gpu*.log
│   └── status.jsonl
├── figures/
│   ├── ddom_*.png
│   ├── gtg_*.png
│   ├── vsd_*.png
│   ├── genbo_*.png
│   ├── nfbo_*.png
│   └── dibo_*.png
└── results/
    ├── ddom/
    ├── gtg/
    ├── vsd/
    ├── genbo/
    ├── nfbo/
    └── dibo/
```

---

## 关键特性

1. **自动跳过已完成实验**: `run_queue.py`会检查日志，跳过已完成的实验
2. **不停止运行**: 队列管理器自动分配下一个任务
3. **失败自动重试**: 失败的任务自动重试最多3次
4. **实时监控**: 使用`gpu_progress.sh`监控进度

---

## 注意事项

1. **磁盘空间**: /data已满99%，所有结果必须存储到/nas1
2. **GPU显存**: 400D实验可能需要较大显存，注意OOM
3. **日志管理**: 定期清理旧日志，避免占用过多空间
4. **监控**: 使用`gpu_progress.sh`实时监控实验进度
5. **Conda环境**: 各算法可能需要不同的conda环境 (ddom, gtg, vsd, genbo, nfbo, dibo)

---

## 当前阻塞问题

- **VSD/GenBO**: 需要进一步配置实验
- **DLLM4BBO**: ❌ 代码不完整，无法复现
