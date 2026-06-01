# MolStitch 复现计划 (更新于 2026-05-26)

## 实验协议（论文 Section 4.1 + Appendix L/N）

### 关键发现: Oracle预算的准确含义 (Appendix L.1)

**之前理解错误:**
- "5000+5000" 被误解为 "5000离线训练步 + 5000 online oracle query步"

**Paper L.1原文:**
> For the MPO task, the total number of oracle calls was limited to 10,000. We allocated 5,000 calls to construct the offline dataset and reserved the remaining 5,000 for evaluation. After completing offline optimization, the performance of the fine-tuned generative model was evaluated using the remaining 5,000 oracle calls on the molecules it newly generated.

**正确理解:**
```
前5000 oracle calls: 从ZINC随机采样5000个分子，用oracle一次性打分 → 构建offline dataset
训练全过程:        纯offline，只在offline dataset上训练，不消耗任何新oracle
后5000 oracle calls: 训练完成后，从模型采样新分子，用oracle评估 → 算最终HV/R2
```

Docking同理: 1500构建offline dataset + 1500评估 = 3000 total。

**这意味着所有backbone方法（REINVENT、GFN、Saturn）在训练期间都不能调用oracle!** 任何online GA loop、online oracle query都是错误的。

### 实验总览

| 参数 | MPO实验 | Docking实验 |
|------|---------|------------|
| 骨架 | REINVENT（Saturn换Mamba，GeneticGFN换GFlowNets） | 同左 |
| 离线数据 | ZINC 15随机采5000分子，用oracle打分（前5000次oracle call） | ZINC 15随机采1500分子，用QuickVina2打分（前1500次oracle call） |
| Oracle预算 | 10,000（5000离线 + 5000评估） | 3,000（1500离线 + 1500评估） |
| 目标 | JNK3, GSK3β, QED, SA（2/3/4目标） | parp1, fa7, jak2, braf, 5ht1b + QED + SA |
| 评估指标 | **HV(↑) + R2(↓)** | **HV(↑)** |
| 补充指标 | Top-100/10 | QED, SA, Top-100/10 |
| 重复 | 10 seeds (0-9)，均值±标准差 | 同左 |
| 场景 | **Full-offline only** | **Full-offline only** |

### Full-offline统一训练范式

所有backbone方法必须遵循相同的两阶段结构：

```
Phase 1 (Offline Training, 0次新oracle调用):
  - Load 5000个offline分子 (已打分, oracle calls=5000)
  - 在offline分子上训练model (TB loss / REINVENT loss / etc)
  - 训练步数: 5000步 (对应paper的offline_steps)

Phase 2 (Evaluation, 剩余5000次oracle调用):
  - 从训练好的model采样新分子
  - 用真实oracle打分 (消耗剩余5000次oracle call)
  - 合并offline分子 + 新分子 → 算HV/R2
  - 评估期间模型不再更新
```

**Paper L.3 适配说明**: "Since all of these generative models were originally designed for an online setting, we made necessary adjustments to the number of molecule updates and the experience replay to adapt them for our offline settings." — 适配的核心是保留原有训练机制（augmented memory、replay buffer、TB loss等），但把数据源从在线oracle换成离线数据集。不是把方法改造成完全不同的东西。

**Paper N 各方法描述**:
- **REINVENT**: "trained exclusively on the offline dataset without applying further offline MBO techniques" — 纯offline训练
- **AugMem (Saturn基础)**: "incorporating molecular data augmentation techniques and experience replay" — augmented memory保留
- **GeneticGFN**: "leverages domain-specific genetic operators" — offline设置下GA被移除，只保留TB loss训练
- **Saturn**: "builds upon REINVENT while replacing GRU with Mamba" — Mamba + augmented memory
- **BootGen**: "employing a bootstrapping technique to iteratively augmenting the offline dataset with self-generated data" — bootstrap迭代
- **RaM**: "advocates for a ranking-based model... employing learning-to-rank techniques" — 排序损失

### 为什么MPO用HV+R2，Docking只用HV

**MPO**: 4个目标各自需要独立的oracle调用，JNK3/GSK3β极难优化（Top1仅0.07/0.08），QED/SA相对容易。HV衡量Pareto前沿绝对体积，R2衡量最坏情况加权距离——HV看整体覆盖，R2暴露"某些目标被牺牲"的短板。两者互补。

**Docking**: QuickVina2一次调用同时出对接分数，QED/SA是瞬间计算的分子性质，三个目标评估成本相同，分布差异比MPO更均匀。HV作为单一综合指标已足够反映Pareto前沿质量，额外报告QED/SA均值供参考。

### 归一化公式（Appendix N）
- SA: `Normalized SA = (10 - raw_SA) / 9`（raw SA范围1-10，取反后越大越易合成）
- Docking: `Normalized DS = -raw_docking_score / 20`（Vina分数越负越好，取反后越大越好）
- QED, JNK3, GSK3β: 已在[0,1]区间，无需归一化

### 评估指标定义（Appendix O）
- **HV(↑)**: 超体积。参考点`(0,0,...,0)`。计算Pareto前沿支配区域的Lebesgue体积。越大越好。
- **R2(↓)**: R2指标。`R2(X,W) = max_{w∈W} min_{x∈X} Σ w_i·[f*_i - f_i(x)]`。使用自适应权重向量集W。越小越好。

### 附录N: 离线数据集统计（每seed 5000分子）
| 指标 | Min | Top1 | Top5 | Top10 | Top100 | Top1000 | Mean |
|------|-----|------|------|-------|--------|---------|------|
| QED | 0.0022 | 0.8604 | 0.8403 | 0.8265 | 0.7484 | 0.5809 | 0.3510 |
| SA | 0.0000 | 1.0000 | 0.9995 | 0.9980 | 0.9858 | 0.9293 | 0.6450 |
| JNK3 | 0.0000 | 0.0727 | 0.0559 | 0.0487 | 0.0231 | 0.0080 | 0.0029 |
| GSK3β | 0.0000 | 0.809 | 0.0550 | 0.0451 | 0.0192 | 0.0064 | 0.0021 |

### 对比方法（Table 1 & 2）
Full-offline: REINVENT, GeneticGFN, Saturn, BootGen, RaM, MolStitch（**全部报告Full-offline**）
Semi-offline: 仅MolStitch报告（Table 3, 4），**我们不跑半离线**

### MolStitch最优配置（Table 5, 7, 8）
StitchNet + Rank-based Proxy + IPO + 4 Proxies + Online Proxy Training
β=0.2 (IPO温度), 相似度阈值δ=0.8, α=1.0 (Dirichlet)

### 超参数（Table 17）
| | REINVENT | GFlowNets | Mamba |
|--|----------|-----------|-------|
| Batch size | 200 | 200 | 200 |
| Embedding dim | 128 | 128 | 256 |
| Hidden dim | 512 | 512 | 256 |
| Layers | 3 | 3 | 12 |
| Sigma | 500 | 500 | 500 |
| Experience replay | 300 | 300 | 300 |
| Augmentation round | 8 | 8 | 8 |
| Batch update | 2 | 2 | 2 |
| Learning rate | 5e-4 | 5e-4 | 5e-4 |
| logZ | — | 0.001 | — |

---

## 一、MolStitch + REINVENT基线

**代码**: `MolStitch-main/`
**环境**: `mol_stitch`（Python 3.8, PyTorch 1.12, rdkit, tdc, botorch, pymoo, pygmo）

### 1.1 MolStitch（完整方法）— 直接可跑

```bash
# MPO 4目标
python run.py offline_cluster --ga_mode gfn \
  --oracles "qed:1+sa:1+jnk3:1+gsk3b:1" \
  --max_oracle_calls 10000 --offline_batch_size 5000 \
  --seed 0 --device 0 --wandb disabled

# Docking (parp1为例)
python run.py offline_cluster --ga_mode gfn \
  --oracles "parp1:1+qed:1+sa:1" \
  --max_oracle_calls 3000 --offline_batch_size 1500 \
  --seed 0 --device 0 --wandb disabled
```

**已具备**:
- 10个离线数据集 `.pt`（`main/offline_cluster/data/offline_dataset/`）
- 预训练Prior.ckpt + StitchNet Prior_CO checkpoint
- TDC oracles（qed, sa, jnk3, gsk3b）
- QuickVina2 + 5靶点pdbqt（`evaluators/dock/`）
- HV/R2评估（`evaluators/hypervolume.py`）

### 1.2 REINVENT基线 — 用REINVENT4

论文的REINVENT基线用REINVENT loss（`loss = (prior_ll + σ·reward - agent_ll)²`），与MolStitch的GFN trajectory balance loss不同。用REINVENT4的staged learning实现。

**关键注意**: 论文Appendix N定义REINVENT基线是**仅用离线数据集训练，不查询oracle**（"trained exclusively on the offline dataset without applying further offline MBO techniques"）。REINVENT4的staged learning默认是online的（生成新分子并查询oracle），需要配置为offline模式：仅replay离线数据集中的分子进行训练，不生成新分子查询oracle。

**适配工作**:
1. 下载REINVENT prior模型（从Zenodo: https://doi.org/10.5281/zenodo.15641296）
2. 写TDC activity插件 `reinvent_plugins/components/comp_tdc_activity.py`（~40行，参照comp_chemprop.py接口）
3. 写MPO TOML config：offline模式（仅replay离线数据，不查询oracle），4个scoring components用于最终评估
4. 写Docking TOML config（DockStream + 5靶点pdbqt）
5. 后处理：从REINVENT4输出提取分子 → 统一评估脚本算HV/R2

### 1.3 所有方法共享离线数据集

公平对比要求所有方法使用相同的离线数据集。MolStitch的 `main/offline_cluster/data/offline_dataset/` 下有10个 `.pt` 文件（seed 0-9）。其他方法必须加载这些相同数据，不能各自重新采样。

---

## 二、GeneticGFN 🟡

**代码**: `genetic_gfn-main/`, 对比 pure: `genetic_gfn-main-pure/`
**环境**: `genetic_gfn`（Python 3.7, PyTorch 1.12, PyTDC==0.4.0, dgl, polyleven）
**核心算法**: GRU RNN + GFlowNet trajectory balance loss (penalty=pb)
**状态**: 🟡 已适配完成但效果差 — MPO4obj seed0 HV=0.106（REINVENT ~0.38, Saturn 0.40）

### 2.1 Full-Offline适配

**原始GeneticGFN（pure）**: 在线GA+GFN循环 — Agent采样 → oracle打分 → 遗传算子(crossover/mutation) → oracle再打分 → TB loss训练。全程在线，无Phase1/Phase2分离。

**Paper L.3适配说明**: "Since all of these generative models were originally designed for an online setting, we made necessary adjustments to the number of molecule updates and the experience replay to adapt them for our offline settings." — 保留原有训练机制，但把数据源从在线oracle换成离线数据集。

**Full-offline修改** (`run.py`):
```
Phase 1 (Offline TB Training, 5000步, 0次新oracle调用):
  - 加载共享offline dataset (MolStitch_offline_dataset[seed].pt)
  - Pre-fill oracle buffer + offline_seqs/offline_rewards
  - 5000步 TB loss训练 (penalty=pb):
    forward_flow = agent_ll + log_z
    backward_flow = prior_ll + sigma * batch_r
    loss = (forward_flow - backward_flow)^2
  - 每步8轮augmentation (rank_based_sample, exp_replay=300)
  - 梯度裁剪已移除（匹配pure代码）
  - 训练期间完全不接触oracle

Phase 2 (Evaluation, 剩余5000次oracle调用):
  - Agent.rnn.eval() → 纯采样 → WeightedMultiOracle打分
  - 不做任何训练、不做GA操作
  - n_calls达到10000时停止
```

### 2.2 已发现并修复的Bug

**Fix 1 — HV/R2 pareto guard** (`optimizer.py:145-146`):
- 问题: `len(pareto) > 1` 改为 `len(pareto) > 0`
- 原因: offline数据5000分子的Pareto前沿只有1个点

**Fix 2 — HV/R2计算对齐MolStitch** (`optimizer.py:139-150`):
- 已改为MolStitch `get_hypervolume(None, pareto, num_obj)`

**Fix 3 — penalty=pb 对齐REINVENT loss** (已实现):
- 从 `penalty=prior_kl` 改为 `penalty=pb`
- `backward_flow = prior_ll + sigma * batch_r`（加了prior_ll作为anchor）
- 与REINVENT loss `(agent_NLL - prior_NLL + sigma*reward)^2` 数学等价（当log_z≈0）

**Fix 4 — 移除梯度裁剪** (已实现):
- 原代码对RNN参数做 `clip_grad_norm_`，但不裁剪logZ
- 导致logZ吸收所有梯度持续增长
- 已移除，匹配pure代码

**Fix 5 — rank_based_sample恢复** (已实现):
- augmentation使用 `rank_coefficient=0.01` 的rank-based sampling
- 匹配pure代码 `1.0 / (0.01*N + rank)` 权重

### 2.3 ⚠️ 核心问题：logZ在offline-only场景下持续增长

**现象**: logZ从0.001线性增长到2.48（5000步），每步+0.0005
- 不是"爆炸"，是TB loss的固有行为
- sigma=500时，`sigma * r ≈ 200` 远大于 `agent_ll ≈ -30`
- logZ被迫增长来平衡forward/backward flow

**影响**: Phase 2用logZ=2.48的agent采样，agent分布可能偏离好的区域
- HV仅从0.090增长到0.106（Phase 2几乎无提升）
- 对比REINVENT ~0.38，差距巨大

**Paper L.3原文**: "we reduced the logZ value to 0.001 and aligned the learning rate with that of the generative model (from 0.1 to 0.0005). This adjustment resulted in more stable training and significantly improved performance."
- 论文确认了logZ=0.001/lr_z=5e-4大幅改善了性能
- 但论文没提到offline-only场景下logZ的持续增长问题

**纯代码对比**: pure GFN也是同样配置（sigma=500, lr_z=5e-4），但pure是online场景，logZ和agent协同演化，不会出现分布偏移

**可能的解释**: 论文的GFN结果可能在offline训练后，Phase 2评估时使用了某种我们没有的机制（如不同的采样策略或logZ处理），或者论文中GFN的HV本身就比REINVENT低（论文Table 1中GFN的HV确实低于REINVENT和Saturn）

### 2.4 与pure代码的关键差异

| 差异 | Pure (在线GA+GFN) | 修改后 (full-offline) | 影响 |
|------|-------------------|-------------------|------|
| log_z初值 | 5.0, lr_z=0.1 | 0.001, lr_z=5e-4 | Table 17对齐 ✓ |
| GA loop | 每步样本→GA crossover→oracle→训练 | 完全移除 | 失去exploration |
| 在线训练 | 每步都有TB loss回传 | Phase 2无训练 | Agent冻结 |
| batch_size | 64 | 200 | Table 17 ✓ |
| beta | 50 | 500 | Table 17 ✓ |
| exp_replay | 64 | 300 | Table 17 ✓ |
| penalty | prior_kl | pb | 对齐REINVENT ✓ |
| rank_based_sample | rank_coeff=0.01 | rank_coeff=0.01 | ✓ |
| 梯度裁剪 | 无 | 无 | ✓ |
| 离线数据 | 无 | 加载MolStitch 5000分子 | 论文要求 ✓ |

### 2.5 当前状态与下一步

- [x] penalty=pb 已实现
- [x] 梯度裁剪已移除
- [x] rank_based_sample已恢复
- [ ] **待验证**: MPO4obj seed0 HV=0.106，远低于REINVENT ~0.38
- [ ] **分析中**: logZ增长是否是根本原因，还是有其他因素
- [ ] **可能方向**: 对比论文Table 1中GFN的实际HV水平，确认GFN在此设置下本身是否就比REINVENT差

```bash
# MPO: 每个seed跑1次多目标
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python pmo/run.py genetic_gfn \
    --oracles jnk3 gsk3b qed sa \
    --seed $seed --max_oracle_calls 10000 \
    --wandb online --run_name GFN_MPO4obj_seed${seed}
done
```

### 2.6 Docking实验（后续）

同之前方案B：用MolStitch的QuickVina2模块接入GFN训练循环。

---

## 三、Saturn 🟢

**代码**: `saturn-master/`, 对比 pure: `saturn-master-pure/`
**环境**: `saturn`（Python 3.10, PyTorch 1.12, mamba-ssm==1.2.0, causal-conv1d==1.2.0, rdkit, tdc）
**核心算法**: Mamba SSM + REINVENT loss + Augmented Memory
**状态**: 🟢 MPO2obj seed0 HV=0.4032（与REINVENT ~0.38相当），核心机制已修复

### 3.1 Full-Offline适配

**原始Saturn**: Online RL循环 — Mamba agent采样 → oracle打分 → REINVENT loss → backprop → augmented memory (SMILES randomization) → hallucinated memory (GraphGA crossover/mutation) → beam enumeration → experience replay。

**Paper N适配说明**: "AugMem incorporates molecular data augmentation techniques and experience replay to enhance performance." — augmented memory 是Saturn的核心特性，在offline设置中保留。

**Full-offline修改** (`reinforcement_learning.py`):
```
Phase 1 (Offline Training, 5000步, 0次新oracle调用):
  - 加载MolStitch offline dataset → pre-fill replay buffer
  - 5000步 offline training:
    - _offline_train_step(): 从replay buffer采样 → REINVENT loss
    - _offline_aug_step() × augmentation_rounds(8):
      - sample_memory() → randomize_smiles_batch() → REINVENT loss
      - augmented_memory_replay() → 随机采样cap(200) → REINVENT loss
  - gradient clipping: clip_grad_norm_(max_norm=1.0)（sigma=500的安全措施）

Phase 2 (Evaluation, 剩余5000次oracle调用):
  - Agent采样 → oracle打分 → replay buffer更新
  - 无training/loss/backprop/augmented/hallucinated/beam
  - _validity_drift_guard: patience=10次零validity后reset到best_agent checkpoint
```

### 3.2 已修复的问题

**Fix 1 — Augmented Memory OOM** (已修复):
- 原因: `augmented_memory_replay()` 返回replay buffer中全部300分子，加上sample的200分子=500分子
- Mamba selective_scan kernel内存开销大，500分子导致OOM
- 修复: 在 `_offline_aug_step()` 中cap augmented batch在 `sample_size`(200)：
  ```python
  n = min(len(aug_smiles), self.replay_buffer.sample_size)
  indices = np.random.choice(len(aug_smiles), size=n, replace=False)
  ```

**Fix 2 — augmented_memory配置** (已修复):
- 所有30个mpo*.json配置文件的 `augmented_memory` 从 `false` 改为 `true`
- 这是Saturn的核心特性（SMILES randomization for sample efficiency）

**Fix 3 — _validity_drift_guard改为patience-based** (已修复):
- 原代码: 单次validity<0.05就reset到prior（过于激进，丢失训练进度）
- 改为: patience=10次连续零validity后reset到best_agent checkpoint
- 匹配pure Saturn代码行为

**Fix 4 — HV/R2计算对齐MolStitch** (已修复):
- 使用 `get_hypervolume(None, pareto, num_obj)`

### 3.3 测试结果

**MPO2obj seed0** (5月27日完成):
| Oracle Calls | HV | 说明 |
|---|---|---|
| 5,113 | 0.1425 | offline训练刚结束 |
| 7,525 | 0.1683 | Phase 2缓慢提升 |
| 8,514 | **0.4032** | 突然跳变 |
| 10,019 | 0.4032 | 稳定到结束 |

- avg_top1=0.580, avg_top10=0.444, avg_top100=0.312
- batch validity ~95%
- HV在8514 oracle calls处突然从0.168跳到0.403，可能是某个高质量分子被采样到

**其他seed** (5月27日，无augmented_memory修复):
- seed1: HV=0.2942, R2=1.276
- seed2: HV=0.2155, R2=1.296
- seed3: 未完成

**当前测试** (5月28日，有augmented_memory修复): seed0在GPU0上运行中

### 3.4 与pure代码的关键差异

| 差异 | Pure | 修改后 | 影响 |
|------|------|--------|------|
| compute_loss | 使用negated NLL | 直接使用NLL空间 | 数学等价 |
| backpropagate | 无gradient clipping | clip_grad_norm_(max_norm=1.0) | sigma=500安全措施 |
| _validity_drift_guard | patience=10次零validity后reset | patience=10次零validity后reset | ✓ 匹配 |
| augmented_memory | true | true | ✓ 已修复 |
| augmented batch cap | 无（全部返回） | cap at sample_size=200 | 防OOM |
| run() | 单phase在线RL | Phase1+Phase2分离 | 论文要求 ✓ |
| offline_data | 无 | 从MolStitch加载 | 论文要求 ✓ |

### 3.5 当前状态与下一步

- [x] augmented_memory=true 已配置
- [x] OOM修复（augmented batch cap）
- [x] _validity_drift_guard改为patience-based
- [x] MPO2obj seed0 测试通过 (HV=0.4032)
- [ ] **进行中**: MPO2obj seed0 带augmented_memory修复的重跑
- [ ] **待做**: 全部30 runs (3 MPO × 10 seeds)

```bash
# MPO: 每个seed跑1次多目标
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python saturn.py configs/mpo2obj_seed${seed}.json
done
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python saturn.py configs/mpo3obj_seed${seed}.json
done
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python saturn.py configs/mpo4obj_seed${seed}.json
done
```

---

## 四、REINVENT4

**代码**: `REINVENT4-main/`
**环境**: `reinvent4`（Python 3.10, PyTorch 2.9.1, rdkit>=2025）
**核心算法**: GRU/Transformer + Staged Learning (RL) + 多组件scoring

### 4.1 前置依赖
- 从Zenodo下载 `reinvent.prior` 模型
- 独立conda环境（Python版本/PyTorch版本与其他方法不兼容）

### 4.2 MPO实验

**现状**: 有QED和SA scoring components，**没有jnk3/gsk3b**

**方案**:
1. 新建 `reinvent_plugins/components/comp_tdc_activity.py`（~40行）
   - 用TDC Oracle计算jnk3/gsk3b分数
   - 注册为 `@add_tag("TDCActivity")`

2. 写TOML config `configs/mpo_staged_learning.toml`
   - 4个scoring components: QED, SAScore, TDCActivity(JNK3), TDCActivity(GSK3B)
   - aggregation: arithmetic_mean

### 4.3 Docking实验

**现状**: docking需通过DockStream（外部工具）

**方案**:
1. 安装DockStream
2. 从MolStitch复制5个pdbqt + qvina02
3. 为每个靶点写DockStream JSON配置
4. 在TOML config中配置comp_dockstream组件

**需新建/修改文件**:
- `reinvent_plugins/components/comp_tdc_activity.py` — 新建（~40行）
- `configs/mpo_staged_learning.toml` — 新建
- `configs/docking_staged_learning.toml` — 新建

---

## 五、BootGen — 后续适配

**代码**: `bootgen-main/`
**核心算法**: Score-conditioned LSTM generator (CondDecoder) + DropoutRegressor proxy + 迭代bootstrap
**状态**: ⏳ 未开始

### 5.1 原始代码分析

**BootGen并非天然适应offline MBO**，需要大量适配：

1. **数据格式不同**: BootGen用Design-Bench库的整数编码序列（RNA: 0-3映射U/G/C/A），不是SMILES分子
2. **任务不同**: 原始任务是生物序列（GFP/UTR/RNA），不是分子优化
3. **单目标**: proxy输出单个标量，generator conditioned on单个score=1.0，无多目标概念
4. **无HV/R2**: 评估只有Percentile 50/100和Diversity

**BootGen的核心流程**:
```
训练阶段 (1500 stages for non-GFP):
  每stage: rank_weighted_training() × 10 steps
    - 从offline dataset按rank加权采样batch(256)
    - CondDecoder: input=(token, score) → next_token prediction
    - loss = cross_entropy, grad_clip=0.5
  stage > 1250时，每5个stage做bootstrap:
    - generator生成1000个candidates (conditioned on score=1.0)
    - proxy(MC dropout MLP)打分
    - 选top-2非重复candidates加入训练集

评估阶段:
  - generator生成1280个candidates → proxy打分 → 取top-128
  - 计算Percentile 50/100, Diversity
```

### 5.2 适配方案

**方案A: 替换generator为REINVENT架构** (~200行改动)
- 用MolStitch的MultiGRU RNN替换CondDecoder
- 保留BootGen核心: proxy + bootstrap迭代增强
- 数据管道: 加载MolStitch `.pt` → SMILES → tokenize
- 多目标: weighted sum scalarization → 单score condition
- 评估: 加入HV/R2

**方案B: 保留CondDecoder，适配SMILES** (~300行改动)
- 修改tokenizer: 从RNA核苷酸改为SMILES字符
- 修改oracle wrapper: 从RNA binding改为TDC/QuickVina2
- 加入多目标scalarization
- 加入HV/R2评估

**估计工作量**: 200-300行（远不止100行）

---

## 六、Offline-RaM — 后续适配

**代码**: `Offline-RaM-main/`
**核心算法**: MLP + Learning-to-Rank losses (ListNet/RankNet等) + 梯度搜索
**状态**: ⏳ 未开始

### 6.1 原始代码分析

**RaM并非天然适应offline MBO**，需要大量适配：

1. **数据格式不同**: RaM用Design-Bench库的整数编码输入（AntMorphology/TFBind8等），不是SMILES分子
2. **模型不同**: SimpleMLP `[2048,2048]` → 单标量输出，不是序列生成器
3. **优化方式不同**: 梯度搜索（adam_search/grad_search）直接在输入空间优化，不是生成分子
4. **单目标**: 模型输出单标量，评估只有Normalized-Score percentiles
5. **无HV/R2**: 评估只有single-objective percentiles和Elite metrics

**RaM的核心流程**:
```
训练阶段 (100 epochs):
  - 创建listwise训练数据: 随机采样10000个子集，每个子集1000个molecules
  - SimpleMLP: input → [2048, 2048] → output_dim
  - Ranking loss (ListNet/RankNet/LambdaRank等) on listwise data
  - 验证: MSE, Spearman rank correlation, AUC-PR on held-out elite data

搜索阶段 (gradient-based):
  - 从top-128 designs初始化
  - Adam优化器在输入空间做梯度上升
  - discrete: lr=0.1, 100 steps
  - continuous: lr=0.001, 200 steps
```

### 6.2 适配方案

**方案: 替换整个优化管道** (~300行改动)
- 保留RaM核心: `losses/` 目录的排序损失函数
- 替换SimpleMLP为REINVENT generator
- 排序代理的输出作为reward signal驱动REINVENT训练
- 数据管道: 加载MolStitch `.pt` → SMILES → tokenize
- 多目标: weighted sum scalarization → 单score for ranking
- 评估: 加入HV/R2

**关键区别**: RaM原文的梯度搜索是在连续/离散输入空间直接优化，而分子SMILES是离散序列，不能直接做梯度搜索。需要用排序代理作为reward来训练生成器（类似REINVENT的RL训练），而不是直接优化输入。

**估计工作量**: ~300行（远不止100行）

---

## 七、统一评估脚本

**新建文件**: `eval_unified.py`（~50-80行，复用MolStitch的evaluators模块）

功能:
1. 读取各方法输出的分子列表（SMILES）
2. 用TDC同时给所有分子打4个目标分（jnk3, gsk3b, qed, sa）
3. 归一化到[0,1]（SA: `(10-raw)/9`, 其他已在[0,1]）
4. **直接调用MolStitch的 `evaluators/hypervolume.py`** 计算HV/R2（不要从零写）
5. 输出10个seed的均值±标准差

```bash
# MPO评估
python eval_unified.py \
  --input_dir results/molstitch/mpo/ \
  --experiment mpo \
  --seeds 0 1 2 3 4 5 6 7 8 9

# Docking评估
python eval_unified.py \
  --input_dir results/molstitch/docking/ \
  --experiment docking \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

---

## 八、文件清理方案

### MolStitch-main — 保留
```
run.py
main/optimizer.py
main/graph_ga/          (offline_cluster引用crossover/mutate，必须保留)
main/offline_cluster/   (核心算法)
main/utils/
evaluators/
```
可删: `main/offline_cluster/pretrain.py`, `pretrain_stitch.py`, `multiprocess.py`

### genetic_gfn-main — 清理
保留:
```
pmo/                    (run.py, data/, main/optimizer.py, main/utils/, main/genetic_gfn/, main/reinvent/, main/stoned/, main/graph_ga/)
sars_cov2/              (docking实验需要scoring_function.py + data/targets/)
requirements.txt
```
可删:
```
multi_objective/        (缺oracle/模块+语法错误)
sars_cov2/codes/        (重复代码)
sars_cov2/ckpt/         (预训练checkpoint，与我们无关)
pmo/main/genetic_gfn_selfies/
pmo/main/genetic_gfn_al/
pmo/main/gflownet/
pmo/main/gflownet_al/
pmo/main/smiles_ga/
pmo/main/selfies_ga/
pmo/main/smiles_lstm_hc/
pmo/main/selfies_lstm_hc/
pmo/main/mol_ga/
pmo/main/synnet/
pmo/main/gpbo/
pmo/main/gegl/
```

### saturn-master — 清理
保留:
```
saturn.py, setup.sh
models/
oracles/ (oracle.py, oracle_component.py, utils.py, dataclass.py, physchem/qed.py, physchem/sa_score.py, docking/, reward_aggregator/, reward_shaping/)
goal_directed_generation/
experience_replay/
diversity_filter/
beam_enumeration/
hallucinated_memory/
utils/
scoring/
experimental_reproduction/part_3/           (GEAM docking configs)
experimental_reproduction/checkpoint_models/zinc-250k-mamba-epoch-50.prior
```
可删:
```
experimental_reproduction/part_1/
experimental_reproduction/part_2/
experimental_reproduction/constrained_synthesizability/
experimental_reproduction/synthesizability/
experimental_reproduction/synthesizability_control/
experimental_reproduction/config_generator.ipynb
experimental_reproduction/checkpoint_models/ (除zinc-250k-mamba外全删)
enumeration/
oracles/similarity/
oracles/structural/
oracles/synthesizability/  (保留sa_score.py，其他删)
oracles/xtb/
oracles/docking/dockstream.py
oracles/docking/gnina.py
tests/
```

### REINVENT4-main — 保留
```
reinvent/
reinvent_plugins/
configs/
pyproject.toml, install.py
```
可删: `uv.lock`, `notebooks/`, `contrib/`, `.github/`, `support/`, `tests/`

### 根目录 — 保留所有PDF和.md文件

---

## 九、服务器部署

### 目录结构
```
/nas1/xk/zhaoyang/ram_repro/    (实际文件)
├── MolStitch-main/
├── genetic_gfn-main/
├── saturn-master/
├── REINVENT4-main/
├── bootgen-main/               (暂不动)
├── Offline-RaM-main/           (暂不动)
├── eval_unified.py
├── PLAN.md
├── experiments.md
└── logs/

/data/xk/zhaoyang/ram_repro/    (软链接→NAS)
```

### Conda环境
| 环境 | Python | 用途 | 关键依赖 |
|------|--------|------|----------|
| `mol_stitch` | 3.8 | MolStitch + 评估脚本 | torch 1.12, rdkit, tdc, botorch, pymoo, pygmo, pybel |
| `genetic_gfn` | 3.7 | GeneticGFN | torch 1.12, PyTDC==0.4.0, dgl, polyleven, selfies |
| `saturn` | 3.10 | Saturn | torch 1.12, mamba-ssm==1.2.0, causal-conv1d==1.2.0, rdkit, tdc |
| `reinvent4` | 3.10 | REINVENT4 | torch 2.9.1, rdkit>=2025, pandas, pydantic |

---

## 十、实用踩坑备忘

### QuickVina2
- 上传后立即 `chmod +x qvina02`（MolStitch的`evaluators/dock/`和Saturn的`oracles/docking/docking_grids/`各有一份）
- 只支持Linux x86_64
- 先跑一次单分子对接验证

### TDC Oracle
- JNK3/GSK3B首次运行自动下载~100MB模型，提前预热
- 用TDC 0.4.0，不要用0.5.x（分数可能不同）

### Saturn关键配置
- augmented_memory: 必须为true（Saturn核心特性，SMILES randomization for sample efficiency）
- augmented batch cap: 必须限制在sample_size(200)以内，否则Mamba selective_scan OOM
- validity guard: patience=10次零validity后reset到best_agent（不要单次<0.05就reset）
- gradient clipping: clip_grad_norm_(max_norm=1.0)（sigma=500的安全措施）

### Mamba安装（Saturn）
- 用预编译wheel: `pip install mamba-ssm==1.2.0 causal-conv1d==1.2.0 --no-build-isolation`
- 不要用pip直接装最新版，会编译失败

### 随机种子
所有方法必须同时固定numpy、torch、rdkit三个种子:
```python
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

### GeneticGFN超参数 (Table 17 + L.3)
- logZ: 0.001（从pure的5.0降低，Appendix L.3确认大幅改善性能）
- lr_z: 5e-4（从pure的0.1降低，与generative model对齐）
- penalty: pb（对齐REINVENT loss，backward_flow含prior_ll）
- 无梯度裁剪（匹配pure代码）
- rank_based_sample: rank_coefficient=0.01（匹配pure代码）

### 共享离线数据集
所有方法必须使用MolStitch的 `main/offline_cluster/data/offline_dataset/MolStitch_offline_dataset[0-9].pt`，不能各自重新采样。

---

## 十一、执行顺序

| Phase | 状态 | 内容 |
|-------|------|------|
| 0 | ✅ 完成 | 环境搭建，代码清理，共享离线数据集确认 |
| 1 | ✅ 完成 | MolStitch MPO + Docking |
| 2 | ✅ 完成 | REINVENT4 `run_offline.py` 验证 (full-offline正确) |
| 3 | ✅ 完成 | Fix 1/2/3: GFN HV/R2对齐MolStitch |
| 4 | ✅ 完成 | GFN + Saturn full-offline修改 + 测试 |
| 5 | 🔄 进行中 | Saturn MPO 30 seeds (GPU0) + GFN待定 |
| 6 | ⏳ 待做 | GFN + Saturn Docking实验 |
| 7 | ⏳ 待做 | BootGen + RaM 适配 (full-offline) |
| 8 | ⏳ 待做 | 汇总结果，生成Table 1 & 2 |

### 当前任务: Phase 5

**Saturn**: MPO2obj seed0测试通过(HV=0.4032)，准备启动30 runs
**GFN**: MPO4obj seed0测试完成(HV=0.106)，效果差，待分析是否是方法本身在此设置下的限制

### 当前运行状态
```
GPU0: Saturn MPO2obj seed0 测试中 (带augmented_memory=true修复)
GPU1: 空闲

待Saturn seed0确认结果后:
  GPU0+1: Saturn MPO (2obj/3obj/4obj × 10 seeds = 30 runs)
    python -u saturn.py configs/mpo2obj_seedN.json
    python -u saturn.py configs/mpo3obj_seedN.json
    python -u saturn.py configs/mpo4obj_seedN.json

GFN: 待分析 — MPO4obj seed0 HV=0.106 远低于预期，可能需要新方案
```
