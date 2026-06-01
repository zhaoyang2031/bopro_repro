# Table 17 & 18 (MolStitch Paper Appendix L.3): Hyperparameters

## Table 17: Generative model hyperparameters

| Parameter | REINVENT | GFlowNets | Mamba (Saturn) |
|-----------|----------|-----------|----------------|
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

## Table 18: StitchNet hyperparameters (MolStitch only)

| Parameter | Value |
|-----------|-------|
| α for priority sampling | 1.0 |
| Number of stitch rounds | 16 |
| Stitched molecules per round | 250 |
| Population pool | 1000 |
| Temperature β for IPO | 0.2 |

## Current State vs Table 17

### REINVENT4 (run_offline.py) — FIXED 2026-05-24
| Parameter | Current | Table 17 | Match? |
|-----------|---------|----------|--------|
| Batch size | 200 | 200 | ✓ |
| Sigma | 500 | 500 | ✓ |
| LR | 5e-4 | 5e-4 | ✓ |
| Exp replay | 300 | 300 | ✓ |
| Aug rounds | 8 | 8 | ✓ |
| Gradient clipping | 1.0 | — | Added for sigma=500 safety |

### Saturn (configs/mpo*_seed*.json) — FIXED 2026-05-24
| Parameter | Original | Table 17 | After fix |
|-----------|----------|----------|-----------|
| Batch size | 16 | 200 | 200 |
| Sigma | 128 | 500 | 500 |
| LR | 0.0001 | 5e-4 | 5e-4 |
| Exp replay (mem/sample) | 100/10 | 300 | 300/200 |
| Aug rounds | 10 | 8 | 8 |
| Gradient clipping | none | — | 1.0 (fixes validity crash) |

**Root cause**: sigma=500 → `(agent_ll - prior_ll - sigma*reward)²` ≈ 250,000 magnitude → gradients explode → Mamba weights distorted → validity 0-2%. Gradient clipping (max_norm=1.0) added to `backpropagate()`.

### GeneticGFN (hparams_default.yaml + run.py) — FIXED 2026-05-24
| Parameter | Current | Table 17 | Match? |
|-----------|---------|----------|--------|
| Batch size | 200 | 200 | ✓ |
| LR | 5e-4 | 5e-4 | ✓ |
| lr_z | 5e-4 | 5e-4 | ✓ |
| Exp replay | 300 | 300 | ✓ |
| Exp loop (=aug rounds) | 8 | 8 | ✓ |
| Beta (=sigma) | 500 | 500 | ✓ |
| logZ | 0.001 | 0.001 | ✓ |
| Gradient clipping | 1.0 | — | Added for beta=500 safety |

### MolStitch (run.py + run_gfn_proxy_pref.py) — VERIFIED 2026-05-24
| Parameter | Current | Table 17 | Match? |
|-----------|---------|----------|--------|
| self_aug_round | 8 | 8 | ✓ |
| offline_batch_size | 5000 | — | — |
| batch_size (hparams.yaml) | 200 | 200 | ✓ |
| sigma (hparams.yaml) | 500 | 500 | ✓ |
| lr (hparams.yaml) | 5e-4 | 5e-4 | ✓ |
| exp_replay (hparams.yaml) | 300 | 300 | ✓ |
| logZ (hardcoded) | 0.001 | 0.001 | ✓ |
| reward scale (hardcoded) | 500 | 500 | ✓ |
| **Table 18 params** | All match | — | ✓ |

## Saturn purge_memory() 说明

`purge_memory()` 在 online 场景下合理：replay buffer 只保留 reward 最高的 N 个分子（300），用于 experience replay。

但在 full-offline 场景下，加载完 5000 个分子后立即 purge，只保留 300 个用于训练：
- avg_top1 更高 (0.82 vs 0.60)：过拟合 top 300，学会最大化加权和
- HV 更低 (0.14 vs 0.40)：丢失多样性，Pareto 前沿覆盖差

修复：离线训练阶段不 purge，直接随机采样全部 5000 个分子。

## Saturn n_oracle wandb bug

`reinforcement_learning.py` line 441: wandb.log key 是空字符串 `""` 而非 `"n_oracle"`，导致 wandb 上该字段无法正常显示。已修复。

---

# 🔴 2026-05-27 深度诊断报告 (Updated 2026-05-27 最终版)

## 现状

| 方法 | HV (mpo2obj seed0) | 状态 |
|------|-------------------|------|
| REINVENT4 | ~0.3+ (正常运行) | ✅ 基准对齐 |
| GFN | 0.163 (final) | 🔴 penalty=prior_kl 太弱 |
| Saturn | 0.1425 (flat, 不变!) | 🔴 tdc_activity oracle 全返回0 |

---

## 问题1: 为什么GFN要对标REINVENT loss?

**Paper L.3 明确推荐使用 `penalty=pb` (sum of likelihood and logZ):**

> "during preference optimization, while both REINVENT and Mamba require only the generative model's likelihood as input, we recommend using the sum of likelihood and logZ for GFlowNets in order to further improve performance."

**数学等价性证明:**

GFN TB loss with `penalty=pb`:
```
loss = (agent_NLL + logZ - beta*reward - prior_NLL)²
     = (agent_NLL - prior_NLL + sigma*reward - logZ)²    (beta=sigma=500)
     ≈ (agent_NLL - prior_NLL + sigma*reward)²           (logZ=0.001 ≈ 0)
```
= **REINVENT loss** ✓

REINVENT loss 的核心是 `prior_NLL` 作为 anchor，防止 agent 偏离 prior 太远。当前 `penalty=prior_kl` + `kl_coefficient=0.01` 只给 prior regularization 加了 0.01 倍的权重，几乎无效 → agent drift → 效果差。

**结论: GFN 应对标 REINVENT loss，因为 paper 说要用 `penalty=pb`（就是 REINVENT loss）。** 修改: `hparams_default.yaml` 中 `penalty: prior_kl` → `penalty: pb`。

---

## 问题2: Saturn tdc_activity Oracle 根因分析

### Bug 1: 特征维度 mismatch (主要问题)

`tdc_activity.py` 自定义加载 sklearn 模型 + fpscores.pkl:
- `fpscores.pkl`: list of 3549 elements，每个是 `[score, bit_id_1, ...]` — 这是 **fragment-based feature selection**
- sklearn RandomForest 模型: expects **3549 features** (每个 fragment 一个 score)
- Morgan fingerprint: **2048 bits** (raw fingerprint)

当前代码逻辑 (BROKEN):
```python
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)  # 2048 bits
fp_arr = arr.reshape(1, -1)  # (1, 2048)
if fp_arr.shape[1] != len(self.feature_names):  # 2048 ≠ 3549
    fp_arr = np.zeros((1, 3549))  # (1, 3549) 全零
    ConvertToNumpyArray(fp, fp_arr[0])  # 只填前2048位，后1501位=0
return model.predict_proba(fp_arr)[0, 1]  # 模型看到垃圾输入 → 预测class 0
```

**问题**: 
1. fpscores.pkl 是 fragment 列表，不是 feature_names 列表 — `_load_model` 把整个 list 当作 `feature_names` 是错误的
2. 即使有 3549 个 features，模型期望的是 **每个 fragment 的 score**（通过计算该 fragment 对应 fingerprint bits 得到），不是 raw fingerprint bits
3. 2048 bits → 强行塞入 3549-dim 数组 → 后 1501 维全零 → 模型收到的是残缺输入

### Bug 2: `except Exception: return 0.0` 吞掉所有错误

即使 sklearn 版本不兼容、pickle 损坏、维度不匹配等，也静默返回 0.0，无法发现真正错误。

### 为什么 TDC 原版 (MolStitch/REINVENT 用的) 没问题?

MolStitch 和 REINVENT 使用 `from tdc import Oracle; Oracle('jnk3')`。TDC 内部正确实现了：
1. 加载 sklearn model + fpscores.pkl (fragment 列表)
2. 对每个 fragment，计算其对应 fingerprint bit 的分数
3. 构建正确的 3549-dim feature vector → model.predict_proba → 正确输出

### 修复方案

**推荐: 直接使用 TDC Oracle（对齐 MolStitch/REINVENT）**

```python
# 替换 tdc_activity.py 为:
from tdc import Oracle

class TDCActivity(OracleComponent):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.target = parameters.specific_parameters.get("target", "jnk3")
        self._oracle = Oracle(self.target)  # TDC's own implementation
        self.name = f"tdc_{self.target}"

    def __call__(self, mols):
        import numpy as np
        from rdkit import Chem
        smiles = [Chem.MolToSmiles(m) for m in mols]
        return np.array([self._oracle(s) for s in smiles])
```

这样可以 100% 保证与 MolStitch/REINVENT 使用的 oracle 一致。

---

## 问题3: Table 17 参数验证

### GFN (hparams_default.yaml) — 本地配置 ✅

| Parameter | Table 17 | Current | Match? |
|-----------|----------|---------|--------|
| Batch size | 200 | 200 | ✓ |
| Sigma (=beta) | 500 | 500 | ✓ |
| LR | 5e-4 | 0.0005 | ✓ |
| lr_z | 5e-4 | 0.0005 | ✓ |
| Exp replay | 300 | 300 | ✓ |
| Aug rounds | 8 | 8 | ✓ |
| logZ | 0.001 | 0.001 (hardcoded in run.py) | ✓ |
| Embedding dim | 128 | 128 (hardcoded in RNN init) | ✓ |
| Hidden dim | 512 | 512 (hardcoded in RNN init) | ✓ |
| Layers | 3 | 3 (hardcoded in RNN init) | ✓ |
| **penalty** | **pb** | **prior_kl** | 🔴 **不匹配!** |

### Saturn — 本地 dataclass 默认值 vs Table 17

| Parameter | Table 17 | dataclass.py 默认 | Config JSON | Match? |
|-----------|----------|-------------------|-------------|--------|
| Batch size | 200 | (no default) | 需查服务器 | ⚠️ |
| Sigma | 500 | **128.0** | 需查服务器 | 🔴 默认错 |
| LR | 5e-4 | **0.0001** | 需查服务器 | 🔴 默认错 |
| Aug rounds | 8 | **10** | 需查服务器 | 🔴 默认错 |
| Exp replay (mem/sample) | 300/200 | **100/10** | 需查服务器 | 🔴 默认错 |
| Embedding dim | 256 | 256 (MambaConfig) | N/A | ✓ |
| Hidden dim | 256 | 256 (MambaConfig d_model) | N/A | ✓ |
| Layers | 12 | 12 (MambaConfig n_layer) | N/A | ✓ |

**Saturn 本地 dataclass 默认值全部是旧论文的值，服务器部署的 JSON config 必须显式覆盖才能对齐 Table 17。** SSH 不可用，无法直接验证服务器 config。

### REINVENT4 (run_offline.py) — ✅ 全部匹配

### MolStitch — ✅ 全部匹配

---

## 修复优先级

1. **Saturn tdc_activity.py**: 替换为 TDC Oracle 封装 (unblock oracle, HV 将从 0.14 提升)
2. **GFN penalty**: `prior_kl` → `pb` (loss 对齐 REINVENT, HV 将从 0.16 提升)
3. **Saturn config JSONs**: 验证服务器上 mpo*_seed*.json 的参数覆盖了 dataclass 默认值

## L.3
=== PAGE 37 ===
Offline Model-based Optimization for Real-World Molecular Discovery
L.3. Hyperparameters and Implementation Details
Implementation of the generative model. We closely followed the architecture settings for REINVENT as described in the
PMO benchmark (Gao et al., 2022), while the settings for GFlowNets were based on GeneticGFN (Kim et al., 2024a), and
those for Mamba were taken from Saturn (Guo & Schwaller, 2024b). Since all of these generative models were originally
designed for an online setting, we made necessary adjustments to the number of molecule updates and the experience replay
to adapt them for our offline settings. The final hyperparameters for the generative models were primarily determined based
on the performance of REINVENT, which served as our backbone generative model, and are detailed in Table 17.
Stabilizing GFlowNets. During the training of GFlowNets, we encountered instability with the original setting of the
logZ parameter, which plays a crucial role in trajectory balancing and needs to be adjusted according to specific settings
(Malkin et al., 2022). To be more specific, it was initially set to a high value ( logZ = 5.0) with a learning rate of 0.1, as
specified in GeneticGFN. To stabilize the training process, we reduced the logZ value to 0.001 and aligned the learning
rate with that of the generative model (from 0.1 to 0.0005). This adjustment resulted in more stable training and significantly
improved performance. Additionally, during preference optimization, while both REINVENT and Mamba require only the
generative model’s likelihood as input, we recommend using the sum of likelihood and logZ for GFlowNets in order to
further improve performance.
Hyperparameters for StitchNet. Recall that StitchNet combines two parent molecules as input and generates stitched
molecules in an auto-regressive manner. Therefore, it operates by computing the hidden dimensions h1andh2of two parent
molecules m1andm2, respectively, and then averaging these hidden dimensions ash1+h2
2. StitchNet is built upon the
REINVENT architecture. During the self-supervised training process for StitchNet, we applied a similarity threshold δ= 0.8
between the original molecules and the stitched molecules. During the molecular stitching process, StitchNet combines
two parent molecules, each sampled with different weight configurations through priority sampling. The resulting stitched
molecules are stored in a buffer. Once the buffer is full, two molecules are randomly sampled to create non-overlapping pairs.
These pairs are then evaluated by the proxy model to identify the winning and losing molecules. Subsequently, the IPO-like
loss is applied to increase the likelihood of generating winning molecules while reducing the likelihood of generating losing
molecules. The hyperparameter settings for StitchNet are summarized in Table 18.