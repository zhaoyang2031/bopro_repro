# BOPRO MolOpt 复现报告

## 1. 方法介绍

### 1.1 BOPRO-LogEI (Bayesian Optimization with Log Expected Improvement)

BOPRO的核心方法，将贝叶斯优化(BO)与LLM结合：

**工作流程：**
1. **Warmstart**: 用LLM生成初始分子集合，计算分数
2. **GP拟合**: 用分子的embedding和分数训练高斯过程(GP) surrogate
3. **Acquisition优化**: 在latent space中优化logEI采集函数，找到最优点z*
4. **Latent-to-Text解码**: 
   - 用cosine similarity选top-k历史分子作为prompt
   - LLM根据prompt生成新分子
5. **评估**: 计算scalarized score = 0.2×QED + 0.8×Vina
6. **更新GP**: 用新的(x, y)对更新后验

**特点：**
- 使用BO平衡exploration和exploitation
- Latent space优化找到有潜力的区域
- LLM解码生成化学上合理的分子

### 1.2 OPRO (Optimization by Prompting)

论文中的baseline方法，贪心策略：

**工作流程：**
1. **Warmstart**: 同上
2. **选最优历史分子**: 按分数排序，选top-k
3. **构建prompt**: 把这些分子展示给LLM
4. **LLM生成**: 让LLM生成类似的分子
5. **评估和更新**: 同上

**特点：**
- **贪心策略**: 总是选历史最优分子作为参考
- **简单直接**: 不需要BO，直接告诉LLM"这些是好的，生成类似的"
- **利用exploitation**: 专注于exploitation，较少exploration

### 1.3 Random (随机采样)

最简单的baseline：

**工作流程：**
1. **Warmstart**: 同上
2. **随机选历史分子**: 随机选k个历史分子
3. **构建prompt**: 把这些分子展示给LLM
4. **LLM生成**: 让LLM生成新分子
5. **评估和更新**: 同上

**特点：**
- **无指导**: 随机选择参考分子
- **纯exploration**: 完全依赖LLM的随机性探索搜索空间
- **基准线**: 用于验证其他方法的价值

---

## 2. Benchmark介绍

### 2.1 Dockstring Benchmark

Dockstring是一个分子优化benchmark，目标是为蛋白质靶点找到最优药物分子。

**任务特点：**
- **多目标优化**: 同时优化QED（类药性）和Vina（结合亲和力）
- **Scalarized Score**: `score = 0.2×QED + 0.8×Vina`
- **评估方式**: AutoDock Vina分子对接
- **搜索空间**: SMILES字符串空间

**蛋白质靶点：**
- 原始benchmark包含58个临床相关的蛋白质靶点
- 来自各种疾病靶点：癌症、炎症、代谢疾病等

### 2.2 我们的实验规模

**为什么只跑8个蛋白质？**

1. **时间成本**:
   - 每个实验需要200次分子评估
   - 每次评估包含：LLM推理 + 分子对接 + GP更新
   - 单个实验耗时：20-40分钟（取决于GPU）

2. **计算资源**:
   - 服务器：4x RTX 4090 (24GB VRAM)
   - 可并行运行4个实验
   - 8蛋白质 × 3方法 × 3种子 = 72个实验
   - 总耗时：约24小时

3. **代表性**:
   - 选择了8个不同类型的蛋白质靶点
   - 包括激酶（SRC, EGFR, ABL1等）
   - 应该能代表整体趋势

**蛋白质列表：**
```
SRC, EGFR, ABL1, CDK2, AKT1, MAPK1, AKT2, KDR
```

---

## 3. 结果分析

### 3.1 实验结果

| 方法 | 平均分数 | 标准差 | 相对性能 |
|------|----------|--------|----------|
| **OPRO** | **0.81715** | - | 100% (baseline) |
| **logEI** | 0.79118 | - | 96.8% |
| **random** | 0.69016 | - | 84.5% |

### 3.2 关键发现

**1. OPRO > logEI (OPRO更好!)**

这与论文结果相反！论文显示BOPRO-LogEI略优于OPRO。

**2. logEI > random (BO > 随机)**

BO策略确实比随机采样好，验证了BO的价值。

**3. 无效率分析**

| 方法 | 平均无效率 | 说明 |
|------|------------|------|
| OPRO | ~35% | 较少无效 |
| logEI | ~40% | 较多无效 |
| random | ~45% | 最多无效 |

---

## 4. 为什么BOPRO效果不如OPRO？深度分析

### 4.1 模型能力限制

**Qwen2-7B vs Mistral-Large-70B+**

| 特性 | Qwen2-7B | Mistral-Large |
|------|----------|---------------|
| 参数量 | 7B | 70B+ |
| 化学知识 | 有限 | 丰富 |
| SMILES生成 | 较多无效 | 较少无效 |
| 遵循指令 | 一般 | 更好 |

**关键问题：**
- Qwen2-7B的化学知识有限，难以理解复杂的BO建议
- Latent space中的最优区域可能超出Qwen2的理解能力
- 导致生成更多无效分子，浪费评估次数

### 4.2 BO策略的局限性

**Latent Space优化的问题：**
1. **Embedding质量**: MolFormer的embedding可能无法完全捕捉SMILES的化学特性
2. **Latent-to-Text解码**: BO提议的latent vector可能不在LLM的生成分布中
3. **信息丢失**: 从latent space到SMILES的转换可能丢失重要信息

**相比之下OPRO的优势：**
- 直接用历史最优分子作为参考
- LLM更容易理解"生成类似的分子"
- 不需要复杂的latent space转换

### 4.3 Exploration vs Exploitation

**OPRO偏向exploitation：**
- 总是选历史最优分子
- 快速收敛到局部最优
- 在Qwen2-7B上效果好，因为LLM擅长"模仿"

**logEI平衡exploration和exploitation：**
- BO会提议探索新区域
- 但Qwen2-7B可能无法有效探索
- 生成的分子可能不合理，导致无效

### 4.4 无效分子的影响

**OPRO生成更合理的分子：**
- 参考历史最优分子，更保守
- 无效率~35%，有效评估更多

**logEI生成更激进的分子：**
- BO提议的latent vector可能很极端
- LLM试图生成但失败
- 无效率~40%，浪费评估次数

---

## 5. 可能的改进方向

### 5.1 使用更大的模型

**推荐：Mistral-Large-70B+ 或 LLaMA-3-70B**
- 更丰富的化学知识
- 更好的指令遵循能力
- 可能更有效地利用BO建议

**实施：**
- 使用Bedrock API（如论文所示）
- 或本地部署70B模型（需要更多GPU）

### 5.2 改进Embedding模型

**当前：MolFormer**
- 专门为分子设计的embedding
- 但在BO中的效果可能不是最优

**改进方案：**
1. **使用LLM的embedding**: 用Qwen2-7B的最后一层embedding
   - 更好地捕捉分子的语义信息
   - 与生成模型更一致

2. **微调embedding模型**: 在分子数据上微调
   - 专门为BO任务优化
   - 可能提高latent space的质量

### 5.3 优化BO策略

**调整acquisition函数：**
- 尝试UCB或Thompson Sampling
- 可能更适合Qwen2-7B的特点

**调整探索-利用平衡：**
- OPRO的贪心策略在小模型上效果好
- 可以设计更保守的BO策略
- 减少exploration，增加exploitation

### 5.4 改进Prompt设计

**当前prompt：**
```
Your task is to find the optimal drug molecule...
Here are your top previous guesses...
Now, guess exactly n=10 new molecule(s)...
```

**改进方向：**
1. **加入化学约束**: 告诉LLM哪些结构是有效的
2. **分步生成**: 先生成骨架，再优化官能团
3. **反馈机制**: 告诉LLM为什么某些分子无效

### 5.5 混合策略

**结合OPRO和BO的优点：**
1. **Early stage**: 用OPRO快速exploit已知好分子
2. **Later stage**: 用BO探索新区域
3. **Adaptive switching**: 根据收敛情况动态切换

**实施：**
```python
if best_score < threshold:
    strategy = "OPRO"  # 快速exploit
else:
    strategy = "logEI"  # 探索新区域
```

### 5.6 多轮迭代优化

**当前：单轮BO**
- 200次评估后停止

**改进：多轮BO**
- 每轮结束后，用最优分子初始化下一轮
- 逐步缩小搜索空间
- 可能发现更好的解决方案

---

## 6. 结论

### 6.1 主要发现

1. **BOPRO在Qwen2-7B上效果不如OPRO**
   - 主要原因是模型能力限制
   - BO的复杂性在小模型上无法充分发挥

2. **BO确实比随机采样好**
   - 验证了BO的价值
   - 但需要更好的模型来充分利用

3. **无效分子是主要瓶颈**
   - Qwen2-7B生成较多无效SMILES
   - 浪费了大量评估次数

### 6.2 建议

1. **短期**: 使用更大的模型（70B+）
2. **中期**: 改进embedding和BO策略
3. **长期**: 设计更适合小模型的优化方法

### 6.3 代码和结果

- **代码仓库**: https://github.com/zhaoyang2031/bopro_repro
- **实验结果**: `/nas1/xk/zhaoyang/bopro_repro/outputs-molopt-qwen2-7b/`
- **Wandb日志**: https://wandb.ai/1585515136-/repro_bopro

---

*报告生成时间: 2026-06-02*
*实验环境: 4x RTX 4090, Qwen2-7B-Instruct, MolFormer*
