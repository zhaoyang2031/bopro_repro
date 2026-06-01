# MolStitch 实验配置文档
> 对比方法：REINVENT、GeneticGFN、Saturn、BootGen、RaM、MolStitch(Ours)
> 基于论文 *Offline Model-based Optimization for Real-World Molecular Discovery* 精确提取
> 参考：paper_official.pdf Section 4.1, Appendix L/N/O, Table 17/18

---

## 1. 实验整体控制
### 1.1 实验任务
- **主任务1：分子性质优化(MPO)**：基于PMO基准，支持2/3/4目标多目标优化
- **主任务2：蛋白对接分数优化**：基于Lee et al. (2023)协议，优化5种靶蛋白的结合亲和力
### 1.2 实验场景
- **仅全离线(Full-offline)**：使用预构建的静态离线数据集，无额外Oracle调用
### 1.3 公平性控制
- 所有方法使用相同的离线数据集（MolStitch的 `offline_dataset/MolStitch_offline_dataset[seed].pt`）
- 每个实验重复**10次不同随机种子**(0-9)，报告`均值±标准差`
- 所有目标值统一归一化到`[0,1]`区间

---

## 2. 数据集与Oracle调用分配
### 2.1 基础数据集
- **ZINC 15数据集**(Sterling & Irwin, 2015)
- 离线数据集构建：从ZINC随机采样指定数量分子，执行Oracle调用获取真实目标分数，存储为`(SMILES, 目标分数向量)`格式
### 2.2 Oracle调用分配
| 实验任务 | 离线数据集大小 | Oracle预算 | 评估方式 |
|----------|---------------|-----------|---------|
| MPO任务 | 5,000 | 10,000总（含离线） | HV + R2 |
| 对接任务 | 1,500 | 3,000总（含离线） | HV |

### 2.3 附录N: 离线数据集统计（每seed 5000分子）
| 指标 | Min | Top1 | Top5 | Top10 | Top100 | Top1000 | Mean |
|------|-----|------|------|-------|--------|---------|------|
| QED | 0.0022 | 0.8604 | 0.8403 | 0.8265 | 0.7484 | 0.5809 | 0.3510 |
| SA | 0.0000 | 1.0000 | 0.9995 | 0.9980 | 0.9858 | 0.9293 | 0.6450 |
| JNK3 | 0.0000 | 0.0727 | 0.0559 | 0.0487 | 0.0231 | 0.0080 | 0.0029 |
| GSK3β | 0.0000 | 0.0809 | 0.0550 | 0.0451 | 0.0192 | 0.0064 | 0.0021 |

---

## 3. 优化目标定义（均为最大化）
### 3.1 MPO任务目标
1. **JNK3**：c-Jun N-末端激酶3抑制活性（TDC Oracle）
2. **GSK3β**：糖原合成酶激酶3β抑制活性（TDC Oracle）
3. **QED**：定量药物相似性评分(0-1)
4. **SA**：合成可及性评分
  - 归一化公式：`Normalized SA = (10 - raw_SA) / 9`（Appendix N）
  - raw SA范围1-10，越低越易合成，取反归一化后越大越好
### 3.2 对接任务目标
1. **5种靶蛋白对接分数**：parp1, fa7, jak2, braf, 5ht1b
  - 对接工具：QuickVina2
  - 归一化公式：`Normalized DS = -raw_docking_score / 20`
2. 同时优化**QED**和**SA**（同MPO任务定义）

---

## 4. 评估指标（Appendix O）
### 4.1 MPO实验
| 指标 | 方向 | 定义 |
|------|------|------|
| **HV (超体积)** | ↑ | Pareto前沿支配区域相对于参考点`(0,0,...,0)`的Lebesgue体积。衡量收敛性和多样性 |
| **R2** | ↓ | `R2(X,W) = max_{w∈W} min_{x∈X} Σ w_i·[f*_i - f_i(x)]`。使用自适应权重向量集W。衡量最坏情况下的加权距离 |

### 4.2 Docking实验
| 指标 | 方向 | 定义 |
|------|------|------|
| **HV (超体积)** | ↑ | 同上，3目标（docking score, QED, SA）的Pareto前沿体积 |
| 补充 | — | QED均值、SA均值、Top-100/10分子标量化得分 |

---

## 5. 各方法核心配置

### 5.1 共享骨架超参数（Table 17）

| 参数 | REINVENT | GFlowNets (GeneticGFN) | Mamba (Saturn) |
|------|----------|----------------------|----------------|
| Batch size | 200 | 200 | 200 |
| Embedding dim | 128 | 128 | 256 |
| Hidden dim | 512 | 512 | 256 |
| Number of layers | 3 | 3 | 12 |
| Sigma | 500 | 500 | 500 |
| Experience replay size | 300 | 300 | 300 |
| Augmentation round | 8 | 8 | 8 |
| Batch update | 2 | 2 | 2 |
| Learning rate | 5e-4 | 5e-4 | 5e-4 |
| logZ | — | **0.001** | — |

### 5.2 REINVENT（基线方法）
- **核心原理**：基于RL的自回归分子生成模型，GRU架构
- **离线适配**：仅用离线数据集训练，用目标分数作为reward（Appendix N）
- **Loss**: `(prior_log_likelihood + σ * reward - agent_log_likelihood)²`
- **代码**: REINVENT4-main/（需配置offline模式）
- **配置**: batch_size=200, embedding=128, hidden=512, layers=3, sigma=500, lr=5e-4

### 5.3 GeneticGFN
- **核心原理**：GFlowNets + 遗传算法（crossover/mutation）
- **关键改进**：遗传算法增强exploitation，GFlowNets提升种群多样性
- **离线适配**：GFN trajectory balance loss on offline data
- **超参数调整（Appendix L.3）**: logZ从5.0→0.001，lr从0.1→5e-4
- **MPO**: 修改pmo/run.py支持加权多目标（scalarization）
- **代码**: genetic_gfn-main/pmo/
- **pmo/run.py额外超参数**（来自hparams_default.yaml）:
| 超参数 | 值 |
|--------|-----|
| Batch size | 64 |
| num_keep | 1024 |
| experience_loop | 8 |
| experience_replay | 64 |
| mutation_rate | 0.01 |
| population_size | 64 |
| offspring_size | 8 |
| ga_generations | 2 |
| rank_coefficient | 0.01 |
| penalty | prior_kl |
| kl_coefficient | 0.01 |
| ga_method | graph_ga |

### 5.4 Saturn
- **核心原理**：Mamba架构替换GRU + REINVENT loss + Augmented Memory + Hallucinated Memory
- **关键改进**：用Mamba替换REINVENT的GRU，提升复杂分子结构建模能力
- **Docking**: GEAM oracle（QuickVina2 + QED + SA），需修改硬编码路径
- **MPO**: 需写TDC wrapper支持jnk3/gsk3b
- **代码**: saturn-master/
- **RL超参数**（来自ReinforcementLearningParameters dataclass）:
| 超参数 | 值 |
|--------|-----|
| Batch size | 16 |
| Learning rate | 0.0001 |
| Sigma | 128.0 |
| Augmented memory | True |
| Augmentation rounds | 10 |
| Selective memory purge | True |

### 5.5 BootGen
- **核心原理**：Score-conditioned LSTM generator + 代理模型bootstrap + 迭代增强
- **关键流程**：代理模型训练→生成合成样本→代理模型标注→高质量样本加入训练集→迭代
- **适配**: 从生物序列任务适配到SMILES分子任务（复用MolStitch组件）
- **代码**: bootgen-main/
- **配置**:
| 超参数 | 值 |
|--------|-----|
| 基础生成模型 | REINVENT |
| 代理模型 | MLP with MC Dropout |
| 自举迭代次数 | 5 |
| 每次迭代添加样本数 | 1000 |
| 样本筛选阈值 | Top 20% |

### 5.6 RaM (Offline-RaM)
- **核心原理**：排序代理（LTR）替代分数回归，优先考虑分子相对排序而非绝对分数
- **关键改进**：缓解代理模型过估计问题
- **适配**: 从Design-Bench适配到SMILES分子任务（复用MolStitch组件）
- **代码**: Offline-RaM-main/
- **配置**:
| 超参数 | 值 |
|--------|-----|
| 基础生成模型 | REINVENT |
| 代理模型 | 排序感知神经网络 |
| 损失函数 | LambdaRank |
| 优化器 | AdamW，lr=5e-4 |
| 训练轮次 | 50 |

### 5.7 MolStitch (Ours)
#### 整体框架
- 三个核心组件：StitchNet（分子拼接）、Rank-based Proxy（排序代理）、Preference Optimization（偏好优化）
- 三个训练阶段：StitchNet无监督预训练 → 代理模型+StitchNet自监督训练 → 离线模型基优化
#### 最优配置（Table 5, 7, 8）
- StitchNet + Rank-based Proxy + IPO + 4 Proxies + Online Proxy Training
#### StitchNet配置（Table 18）
| 超参数 | 值 |
|--------|-----|
| α (Dirichlet优先级采样) | 1.0 |
| Stitch轮次 | 16 |
| 每轮拼接分子数 | 250 |
| 缓冲池大小 | 1000 |
| IPO温度系数β | 0.2 |
| 相似度阈值δ | 0.8 |
#### Rank-based Proxy
- 成对排序损失（BCE），4个代理模型多数投票
- 训练轮次50，优化器AdamW，lr=5e-4
#### hparams_default.yaml
| 超参数 | 值 |
|--------|-----|
| learning_rate | 0.0005 |
| batch_size | 200 |
| sigma | 500 |
| experience_replay | 300 |
| max_buffer | 400 |
| replay迭代 | 8 |
| stitch_round | 16 |
| self_aug_round | 8 |
| num_proxy | 4 |
| div_filter | NoFilter |
#### 代码
- `MolStitch-main/` — `python run.py offline_cluster --ga_mode gfn`

---

## 6. 各方法运行命令

### MolStitch
```bash
# MPO 4目标
python run.py offline_cluster --ga_mode gfn \
  --oracles "qed:1+sa:1+jnk3:1+gsk3b:1" \
  --max_oracle_calls 10000 --offline_batch_size 5000 \
  --seed 0 --device 0 --wandb disabled

# Docking (以parp1为例)
python run.py offline_cluster --ga_mode gfn \
  --oracles "parp1:1+qed:1+sa:1" \
  --max_oracle_calls 3000 --offline_batch_size 1500 \
  --seed 0 --device 0 --wandb disabled
```

### REINVENT4（需适配）
```bash
# 下载prior模型 + 写TDC插件 + 写TOML config（offline模式）
reinvent configs/mpo_staged_learning.toml
```

### GeneticGFN（需改pmo/run.py）
```bash
# 改造后：加权多目标
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python pmo/run.py genetic_gfn --oracles qed jnk3 gsk3b sa \
    --seed $seed --max_oracle_calls 10000
done
```

### Saturn（需改路径+写TDC wrapper）
```bash
# Docking
python saturn.py config_docking_template.json --seed 0
# MPO: 需先写TDC oracle wrapper
python saturn.py config_mpo_template.json --seed 0
```

### BootGen / RaM（后续适配）

---

## 7. 服务器信息
- 服务器: `ssh xk@180.209.6.83` (4× RTX 4090)
- 代码目录: `/nas1/xk/zhaoyang/ram_repro/` (NAS)，`/data/xk/zhaoyang/ram_repro/` (软链接)
- 日志目录: `/data/xk/zhaoyang/ram_repro/logs/`
- 实验队列: 参照 `repro/HOWTO.md`，为每个方法写experiment JSON
- GPU 0留给交互式调试，实验用GPU 1,2,3
- 不要动别人的东西，只在zhaoyang目录下操作
- 大数据放NAS（`/nas1/xk/zhaoyang/`），避免吃满/和/data
