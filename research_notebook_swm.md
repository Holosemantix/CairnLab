# SWM 探索与 LeWM Noise Sweep 研究笔记

> **本文档定位**：研究笔记，非论文主稿。涵盖 (1) SWM 球面世界模型探索路线（V0 已完成，V1/V2 未推进），(2) Contribution 1 (LeWM+noise) 的完整 4-task noise sweep + SWM 对照数据，(3) 诊断工具栈与 canonical-8 相关性分析。
>
> 论文主稿见 `plan_adaptive_resolution.md`（σ+A_t adaptive consistency = Contribution 2）；其附录 E 已抽取 TwoRoom + PushT 的 LeWM noise sweep 表作为论文 Contribution 1 详表。Reacher / Cube 数据 + SWM 对照保留在本文档 §4。原始设计见 `plan_v2.md`，流水实验见 `experiments.md`。

---

## 1. 摘要

**研究问题**：把 LeWM 的 Euclidean embedding + SIGReg 换成 spherical embedding + uniformity，是否能稳定提升规划性能？

**核心发现**：
1. **SWM 不是全局优于 LeWM 的替代品。** 旧版 4-task single-seed 平均略高的叙事不可追溯。按 2026-05-08 数据：SWM baseline 在 TwoRoom 88.33 vs LeWM 93.0（差 4.67）、PushT 85.67 vs 87.33（差 1.67）、Reacher 60.00 vs 57.67（持平）、Cube 77.00 vs 72.33（SWM +4.67）。Noise drop 上 SWM-base 仍崩（TwoRoom drop 32–45 / PushT drop 71–81），但 per-frame noise training 后所有 drop ≤4 abs，与 LeWM 一致。
2. **SWM 改变了表征的 invariance-resolution tradeoff。** 球面归一化、uniformity、temporal masking、noise augmentation 都在改变"哪些观测差异应该被保留，哪些应该被抹掉"。
3. **不同任务对这个 tradeoff 的偏好可能相反。** TwoRoom 低维、离散、视觉细节冗余，受益于更强 invariance / clustering；PushT 需要精细连续状态分辨率，同样配方会损害控制。
4. **表征分析工具本身是通用贡献。** `noise_sensitivity.py`、robust radius、clean-neighbor distance、noise-induced angular shift 等指标可以在不大量 eval 的情况下诊断 latent geometry 的风险。

**当前方法路线**：probe-only σ + action-aware adaptive consistency。μ path 保持 LeWM MSE + SIGReg，σ head detached 学 `log(error)`；`A_t = ||f(z,a+δ)-f(z,a)|| / ||δ||` 作为 local action-sensitivity proxy；二者共同控制 input-side consistency strength。详见 `plan_adaptive_resolution.md`。

**实验门槛**：Paper 主表至少每任务 3 seeds；统一 `num_eval=300`；TwoRoom / PushT / Reacher / Cube 四任务诊断与 eval 对齐；核心对照保留 LeWM-base、LeWM+noise shared std、LeWM+noise per-task oracle、σ-only consistency、action-aware adaptive consistency。

---

## 2. 引言

### 2.1 研究问题

JEPA-style world model 不需要重建所有像素细节，而是要让 CEM 在 latent space 中区分会改变控制结果的状态差异。因此 latent 同时需要两种能力：对 action-irrelevant visual nuisance 保持 invariance，对 action-relevant transition 保持 resolution。TwoRoom 和 PushT 的相反偏好是主现象：低维离散导航可以受益于聚簇化，连续接触控制会惩罚过度压缩的 transition/action resolution。

### 2.2 核心发现概述

1. **LeWM + per-frame noise training 已显著超过 LeWM baseline clean。** TwoRoom 98.33 vs 93.00，PushT 90.00 vs 87.33，Reacher 86.00 vs 57.67，Cube 73.00 vs 72.33。同时 std=0.05 eval drop 从 baseline 的 11–72pt 压到多数 ≤6pt，抗噪收益明确。
2. **SWM + per-frame noise training 基本覆盖 LeWM baseline，唯 PushT 仍小幅落后。** TwoRoom 94.33 > 93.00、Reacher 84.67 (0to007) >> 57.67、PushT 84.67 (0to006) **小于** 87.33；Cube SWM-base 77.0 已超 LeWM-base。2026-05-08 SWM sweep 补齐后，SWM 在 PushT 上的不足从 4pt 缩到 2.66pt，但相对 LeWM **best**（90.0）仍差 5.33pt。
3. **概率性训练发散是 cross-method 共有现象。** TwoRoom SWM-base 旧 69.67 与 PushT LeWM-0to006-p1 旧 61.0 是同一类问题——同 config 重训即恢复正常（88.33 / 89.33）。这改变了之前对"SWM-base 在 TwoRoom 唯一脆弱"的判定。
4. **旧 4-task 平均叙事（SWM 略高）不成立。** 当 ckpt 来源一致后，SWM 在最佳配置下平均不优于 LeWM。

### 2.3 当前方法路线

SWM 和 LeWM+noise 的角色是**证明静态旋钮不够**。SWM 不是"spherical 比 Euclidean 更好"的主方法，而是一个 geometry intervention：Cube 可以赢，PushT 上限仍低，说明换一个全局 prior 不能通吃。LeWM+noise 是更强的 positive baseline：它证明 input-side invariance 确实有效，但最优 `std_max` 仍按任务变化。两者共同指向同一件事：手调一个全局 geometry/noise 旋钮不是最终形态。

因此后续主线改为 **adaptive latent resolution**。`plan_adaptive_resolution.md`（2026-05-09 修订版）是当前主方法的详细设计文档。核心思想：
- LeWM+noise 的提升是真实且强的，但它目前仍是**静态 augmentation recipe**：每个任务需要选 `std_max`，而且四个任务的最优点不同。
- 给 predictor 加 detached scalar σ probe，保持 LeWM MSE + SIGReg 主路径不变。
- 用 `A_t` 估计 local action sensitivity，过滤 σ 中的 action-irrelevant / visual-nuisance 成分。
- `A_t × σ` 控制 input-side consistency strength：action-insensitive 区域加强 invariance，action-critical 区域降低 consistency pressure。

### 2.4 实验门槛与对照

Paper 主表至少需要：每任务 3 seeds；统一 `num_eval=300` 口径；四任务诊断与 eval 对齐；核心对照保留 LeWM-base、LeWM+noise shared std、LeWM+noise per-task oracle、σ-only consistency、action-aware adaptive consistency。重要 ablation 只保留 P3 的 BN/LN/dim 与 P4 的 action/resolution guardrail。

---

## 3. 方法背景

### 3.1 LeWM Baseline

LeWorldModel (LeWM) 是像素端到端 JEPA world model。encoder 把图像映射到 latent，predictor 根据历史 latent 和 action 预测未来 latent，CEM planner 在 latent space 中用 model cost 做规划。

LeWM 训练目标：

```text
L_LeWM = ||pred(z_t, a_t) - z_{t+1}||^2
       + lambda * SIGReg(Z)
```

关键属性：

| 组件 | LeWM |
|---|---|
| Latent space | Euclidean, raw embedding |
| Anti-collapse | SIGReg, approximate isotropic Gaussian |
| Prediction loss | MSE |
| Planning cost | raw-space L2 / MSE |

LeWM 的优点是稳定、简单、对 pixel noise 相对鲁棒。潜在问题是 SIGReg 强制高维各向同性 Gaussian，可能不适合 TwoRoom 这类低内在维度任务。

### 3.2 SWM V0

Spherical World Model (SWM) 把 LeWM 的 Euclidean 表征换成单位球面表征：

```text
mu(o) = z / ||z||,  mu in S^{d-1}
```

当前最佳 SWM 改动：

| 组件 | LeWM | SWM |
|---|---|---|
| Encoder projector | MLP + BN -> R^d | MLP + BN -> L2 norm |
| Predictor projector | MLP + BN -> R^d | MLP + BN -> L2 norm |
| Prediction loss | MSE | cosine distance |
| Anti-collapse | SIGReg | uniformity loss |
| Planning cost | raw MSE | normalized cosine |

保持不变：ViT-Tiny encoder、ARPredictor、action encoder、CEM planner、dataset pipeline。

### 3.3 当前 plan 内 "SWM-base" 默认配置（V0）

> **注**：这是 research_notebook_swm 中所有 SWM-base 数据点使用的 SWM-base ckpt 配置。它**不是**历史旧 4-task benchmark（90.8/89.8/74.0/66.0）使用的 SWM 配置——那批 ckpt 用 `lambda_0p1`、`无 temporal_masked`，已不可追溯。当前配置在 TwoRoom 与 PushT 上曾出现旧 single-seed clean=69.7/80.0；20260507 同 config retrain 后修正为 88.33/85.67。它仍低于早期 90.8/89.8 数字，差距可能来自 config、eval 口径和 seed 差异；跨历史 config 只能定性参考，不能直接作为方法优劣证据。

```yaml
wm:
  embed_dim: 64
loss:
  pred:
    type: cosine
    space: normalized
  regularizer:
    type: uniformity
    weight: 0.2
  uniformity:
    t: 2.0
    mode: temporal_masked
    temporal_exclusion: 2
```

主要经验：
- pairwise spread 在球面塌缩点有梯度死区，不可用。
- MLP+BN+uniformity 可以逃离 collapse。
- `temporal_masked_2` 明显优于 all-pairs / cross-window / temporal exclusion 过小或过大。
- fixed temporal hinge 大多损害 PushT/Reacher，说明强制相邻状态接近不是通用解。

---

## 4. 实验结果：Clean Benchmark 与 Noise-Aware Training

### 4.1 Clean Benchmark

#### 4.1.1 旧 4-task benchmark（来源可追溯性说明）

> **数据来源说明**：以下数据来自 `experiments.md` 记录的早期 4-task benchmark（2026-04-15/20），配置为 epoch=10，num_eval=500，single seed。这些 ckpt 与当前诊断分析使用的模型**不是同一组**（dim、temporal 配置、noise 设置均可能不同），因此本节仅作历史参考，不进入相关性分析。
>
> | Task | LeWM | SWM best | Delta | 来源 ckpt |
> |---|---:|---:|---:|:---|
> | TwoRoom | 93.0 | 90.8 | -2.2 | **旧 benchmark，ckpt 已不在当前目录**。最新 SWM baseline: 88.33%（20260507 retrain, 3-seed × 100 ep avg） |
> | Cube | 69.2 | 74.0 | +4.8 | **旧 benchmark，ckpt 已不在当前目录**。最新 SWM baseline: 77.00%（epoch_10, num_eval=300） |
> | PushT | 89.4 | 89.8 | +0.4 | **旧 benchmark，不可追溯**。最新 SWM baseline: 85.67%（20260507 retrain, 3-seed × 100 ep avg） |
> | Reacher | 62.2 | 66.0 | +3.8 | **旧 benchmark，ckpt 已不在当前目录**。最新 SWM baseline: 60.00%（epoch_10, num_eval=300） |
> | Average | 78.5 | 80.2 | +1.7 | **旧 benchmark 平均，不代表当前模型**。 |

#### 4.1.2 当前诊断用模型 clean benchmark（canonical 8 + sweep，epoch_10/num_eval=300）

模型集合：base + 0to001/0to002/0to005-p1（LeWM 与 SWM 各 4），eval 取自 `summary.txt::clean_300`（缺失则 `clean`）。这是 §4 / §5 共用的一致 model set。

| Task | LeWM best | SWM best | Delta | 说明 |
|---|---:|---:|---:|:---|
| TwoRoom | **98.33** (`LeWM-0to008-p1` †) | 94.33 (`SWM-0to001-p1`) | **−4.00** | LeWM noise sweep 单调升至 0to008-p1 = 98.33；SWM 任意配置最多 94.33 |
| PushT | **90.00** (`LeWM-0to002-p1`) | 84.67 (`SWM-0to006-p1` †) | **−5.33** | LeWM 在 PushT 显著领先；2026-05-08 SWM sweep 补齐后 SWM best 从 0to001=83.3 升至 0to006=84.67，仍低于 LeWM |
| Reacher | **86.00** (`LeWM-0to006-p1` †) | 84.67 (`SWM-0to007-p1` †) | **−1.33** | LeWM noise sweep 补齐后 0to006-p1（3-seed avg）反超 canonical 最佳 0to002-p1（80.33）；**2026-05-08 SWM sweep 补齐后 SWM best 从 78.0 升至 84.67，差距从 8pt 缩到 1.33pt** |
| Cube | 73.00 (`LeWM-0to001-p1`) | **77.00** (`SWM-base`) | **+4.00** | **唯一一个 SWM-base 不需 noise training 就高于 LeWM 全配置**的任务 |

> 数值变更说明：(a) canonical-only 阶段（2026-05-06）：TwoRoom LeWM best 96.00 → 94.33；PushT LeWM best 91.00 → 90.00；Reacher LeWM best 82.00 → 80.33。(b) **LeWM noise sweep 补齐 + 三个 single-seed ckpt retrain（2026-05-07）**：TwoRoom best 94.33 → **98.33** (0to008-p1)；Reacher best 80.33 → **86.00** (0to006-p1)；**PushT LeWM-0to006-p1 从 61.00（异常）→ 89.33（恢复正常），但 PushT best 仍是 0to002-p1=90.00**；TwoRoom SWM-base 69.67 → 88.33（没影响 best 排序但显著改变叙事）；PushT SWM-base 80.0 → 85.67（同上）；Cube best 模型未变。所有 SWM noise variant（0to001/0to002/0to005-p1）数值与 canonical_evals_20260506 一致（未 retrain）。最新底层数据来自本地生成且 gitignored 的 `canonical_evals_20260508.json`。

结论：
- LeWM perframe 在 PushT 上明显领先 SWM（90.0 vs 83.3，差 6.67pt），印证 Euclidean + 平滑化更适应高分辨率任务。
- TwoRoom 上 LeWM 与 SWM 在 perframe 最佳配置下接近（98.33 vs 94.33；canonical 0to002 前曾为 94.33 vs 94.33），baseline 差距已从旧 outlier 的 ~23pt 修正为 4.67pt（93.0 vs 88.33）。SWM-base 的 clean 不再失效，但 noise drop 仍大，perframe noise training 主要修 robustness 而非单纯补 clean。
- Cube 是唯一 SWM-base 占优的任务（77 vs 72.3）。
- **旧 4-task 平均叙事（SWM 略高）不成立**：当 ckpt 来源一致后，SWM 在最佳配置下平均不优于 LeWM。


### 4.2 四任务 Eval 表（per-frame 独立 std + noise_prob）

**实验设计**：`utils.py:AddNormalizedGaussianNoise` 每帧独立 Bernoulli(`noise_prob`) 决定是否加噪，加则 std ~ Uniform(0, std_max)。canonical 完成 4 任务 × {LeWM, SWM} × {base, 0to001-p1, 0to002-p1, 0to005-p1} = 32 ckpt；**2026-05-07 retrain 替换两个概率性训练失败的 single-seed ckpt**——TwoRoom SWM-base（旧 clean=69.67）和 PushT LeWM-0to006-p1（旧 clean=61.00）；同 config 重训后分别恢复到 88.33 / 89.33（3-seed × 100 ep avg），证明都是 SGD 偶然落入坏盆地的**同一类训练发散**问题，与 method/config 本身无关。顺带 PushT SWM-base 也用 3-seed 重训（80.0 → 85.67，差距小，说明它原本就不是发散 ckpt）。Reacher / Cube SWM-base 旧 single-seed 数据正常，**不再重训**。**LeWM 端 0to003 / 0to004 / 0to006 / 0to007 / 0to008 已补齐（3 seeds × 100 episodes 平均）**，4 任务 × 5 = 20 ckpt，覆盖 LeWM noise sweep 全 8 档位。**SWM 端 0to003–0to008 也于 2026-05-08 全部补齐（3 seeds × 100 ep）**，4 任务 × 5 = 20 SWM ckpt——SWM noise sweep 现已与 LeWM 同档位对齐，每任务 SWM/LeWM 各 8 档（base + 7 perframe std_max），共 64 ckpt 进入分析。

**TwoRoom eval（epoch_10, num_eval=300, summary.txt clean_300 优先）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | **93.00** | 86.67 | 71.00 | 55.67 | 81.00 | 62.33 | 44.33 | 87.67 | 70.33 | 59.33 |
| LeWM-0to001-p1 | 92.00 | 92.33 | 93.33 | 86.00 | 92.33 | 89.67 | 84.67 | 92.00 | 92.67 | 90.67 |
| LeWM-0to002-p1 | 94.33 | 93.00 | 93.00 | 93.00 | 94.00 | 94.00 | 91.00 | 94.33 | 94.00 | 94.33 |
| LeWM-0to003-p1 † | 96.33 | 96.33 | 95.00 | 94.67 | 96.00 | 96.00 | 94.67 | 96.00 | 96.33 | 97.00 |
| LeWM-0to004-p1 † | 96.33 | 97.00 | 97.00 | 96.33 | 96.67 | 97.33 | 95.00 | 97.67 | 96.00 | 96.67 |
| LeWM-0to005-p1 | 94.00 | 94.67 | 93.33 | 94.00 | 94.67 | 94.00 | 94.00 | 94.00 | 94.67 | 94.00 |
| LeWM-0to006-p1 † | 96.67 | 96.33 | 96.00 | 96.67 | 96.33 | 97.00 | 96.67 | 96.67 | 96.00 | 96.33 |
| LeWM-0to007-p1 † | 96.00 | 96.00 | 97.00 | 97.00 | 97.00 | 96.33 | 96.33 | 96.33 | 96.00 | 96.67 |
| LeWM-0to008-p1 † | **98.33** | 97.67 | 98.00 | 98.67 | 98.00 | 98.00 | 98.67 | 98.00 | 98.33 | 97.67 |
| SWM-base † (20260507) | **88.33** | 56.33 | 46.33 | 38.67 | 55.00 | 43.00 | 36.33 | 70.33 | 56.33 | 48.67 |
| SWM-0to001-p1 | **94.33** | 94.00 | 92.00 | 87.00 | 93.67 | 91.00 | 82.00 | 94.33 | 93.67 | 90.00 |
| SWM-0to002-p1 | 88.00 | 89.33 | 87.67 | 87.00 | 87.67 | 88.67 | 88.67 | 88.33 | 87.00 | 90.67 |
| SWM-0to003-p1 † | 89.67 | 88.67 | 88.67 | 89.67 | 89.33 | 89.33 | 89.33 | 89.33 | 89.00 | 89.67 |
| SWM-0to004-p1 † | 89.00 | 88.67 | 88.00 | 86.67 | 89.33 | 88.33 | 88.33 | 88.33 | 89.00 | 88.00 |
| SWM-0to005-p1 | 91.67 | 91.33 | 91.00 | 91.00 | 91.67 | 90.33 | 90.67 | 90.67 | 92.00 | 91.67 |
| SWM-0to006-p1 † | 90.00 | 90.33 | 90.33 | 88.00 | 90.33 | 89.67 | 91.00 | 89.00 | 90.33 | 90.33 |
| SWM-0to007-p1 † | 91.00 | 91.33 | 90.33 | 91.00 | 90.33 | 91.00 | 90.00 | 90.67 | 91.00 | 90.67 |
| SWM-0to008-p1 † | 87.33 | 87.33 | 87.67 | 87.00 | 87.00 | 86.67 | 86.67 | 86.67 | 87.00 | 87.67 |

**PushT eval（epoch_10, num_eval=300, summary.txt clean_300 优先）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | **87.33** | 68.67 | 38.00 | 15.00 | 49.33 | 15.00 | 3.67 | 53.33 | 17.33 | 6.00 |
| LeWM-0to001-p1 | 89.67 | 88.67 | 85.67 | 70.33 | 84.33 | 77.00 | 46.33 | 86.00 | 77.00 | 54.33 |
| LeWM-0to002-p1 | **90.00** | 87.33 | 85.00 | **83.00** | 88.67 | **86.00** | **70.67** | 87.67 | 87.67 | **74.67** |
| LeWM-0to003-p1 † | 89.67 | 89.33 | 89.67 | 86.67 | 89.00 | 87.00 | 83.00 | 89.33 | 89.33 | 82.00 |
| LeWM-0to004-p1 † | 89.33 | 85.00 | 87.00 | 87.00 | 86.33 | 86.67 | 81.33 | 86.67 | 85.67 | 86.67 |
| LeWM-0to005-p1 | 82.00 | 81.33 | 77.33 | 80.67 | 80.00 | 80.00 | 78.00 | 83.33 | 78.67 | 76.00 |
| LeWM-0to006-p1 † (retrained 20260507) | 89.33 | 88.33 | 87.67 | 89.67 | 89.00 | 88.33 | 87.00 | 88.33 | 88.00 | 87.67 |
| LeWM-0to007-p1 † | 85.67 | 86.33 | 82.00 | 84.00 | 83.67 | 85.33 | 82.33 | 85.33 | 84.33 | 84.00 |
| LeWM-0to008-p1 † | 88.33 | 89.33 | 91.33 | 89.00 | 89.33 | 87.67 | 85.33 | 89.00 | 87.33 | 89.00 |
| SWM-base † (20260507) | **85.67** | 58.33 | 14.33 | 4.00 | 39.00 | 5.33 | 5.67 | 45.67 | 5.00 | 2.00 |
| SWM-0to001-p1 | 83.33 | 82.33 | 74.67 | 29.67 | 77.67 | 58.67 | 7.00 | 80.33 | 63.67 | 7.00 |
| SWM-0to002-p1 | 81.00 | 81.00 | 80.33 | 67.67 | 82.67 | 82.00 | 47.33 | 81.00 | 79.33 | 48.67 |
| SWM-0to003-p1 † | 82.33 | 83.33 | 81.00 | 80.67 | 82.00 | 82.00 | 70.67 | 81.00 | 81.33 | 73.00 |
| SWM-0to004-p1 † | 79.33 | 79.00 | 79.67 | 77.67 | 79.33 | 79.67 | 76.00 | 81.67 | 80.00 | 78.67 |
| SWM-0to005-p1 | 71.67 | 70.67 | 71.33 | 72.00 | 71.33 | 71.33 | 70.00 | 70.67 | 70.00 | 72.33 |
| SWM-0to006-p1 † | **84.67** | 84.00 | 83.33 | 83.67 | 83.67 | 82.33 | 83.67 | 84.67 | 82.33 | 83.00 |
| SWM-0to007-p1 † | 83.00 | 82.33 | 83.33 | 81.67 | 82.67 | 82.00 | 80.67 | 84.33 | 83.33 | 82.67 |
| SWM-0to008-p1 † | 81.00 | 82.33 | 81.67 | 80.33 | 81.00 | 81.67 | 81.00 | 81.67 | 82.33 | 81.33 |

**Reacher eval（epoch_10, num_eval=300, summary.txt clean_300 优先）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | **57.67** | 29.00 | 24.33 | 15.67 | 38.33 | 25.33 | 14.67 | 41.33 | 26.67 | 13.67 |
| LeWM-0to001-p1 | 55.67 | 59.33 | 51.67 | 32.33 | 59.67 | 55.00 | 45.33 | 56.67 | 58.67 | 45.67 |
| LeWM-0to002-p1 | 80.33 | 80.67 | 84.67 | 81.67 | 87.00 | 82.00 | 80.67 | 80.67 | 81.33 | 82.33 |
| LeWM-0to003-p1 † | 78.67 | 77.33 | 78.00 | 79.00 | 76.33 | 78.67 | 73.67 | 80.33 | 79.33 | 78.67 |
| LeWM-0to004-p1 † | 84.00 | 80.00 | 84.00 | 82.00 | 84.33 | 82.00 | 80.00 | 81.67 | 81.67 | 82.00 |
| LeWM-0to005-p1 | 73.33 | 71.33 | 73.00 | 71.33 | 73.67 | 69.67 | 71.33 | 66.33 | 70.33 | 70.33 |
| LeWM-0to006-p1 † | **86.00** | 78.67 | 85.67 | 86.33 | 79.00 | 79.33 | 84.67 | 80.33 | 81.67 | 81.33 |
| LeWM-0to007-p1 † | 83.67 | 83.33 | 84.33 | 86.67 | 81.00 | 87.00 | 81.33 | 83.00 | 84.00 | 84.67 |
| LeWM-0to008-p1 † | 84.00 | 85.00 | 83.33 | 84.33 | 83.33 | 81.00 | 83.00 | 85.33 | 79.67 | 82.00 |
| SWM-base | **60.00** | 27.33 | 22.33 | 19.67 | 39.00 | 36.67 | 23.00 | 39.33 | 22.33 | 12.00 |
| SWM-0to001-p1 | 65.67 | 69.33 | 63.33 | 41.67 | 65.33 | 64.67 | 50.67 | 64.33 | 61.33 | 46.33 |
| SWM-0to002-p1 | 78.00 | 76.67 | 81.67 | 85.33 | 77.00 | 80.00 | 82.33 | 80.33 | 82.00 | 79.33 |
| SWM-0to003-p1 † | 81.67 | 76.67 | 79.33 | 85.67 | 80.00 | 81.67 | 75.00 | 80.00 | 78.33 | 77.67 |
| SWM-0to004-p1 † | 77.00 | 80.33 | 78.33 | 77.33 | 79.33 | 75.33 | 83.33 | 80.67 | 79.67 | 77.67 |
| SWM-0to005-p1 | 78.00 | 78.67 | 85.00 | 82.00 | 78.00 | 82.33 | 82.00 | 78.67 | 83.33 | 82.67 |
| SWM-0to006-p1 † | 82.33 | 79.33 | 84.67 | 80.67 | 79.33 | 82.67 | 81.67 | 82.67 | 83.00 | 80.33 |
| SWM-0to007-p1 † | **84.67** | 82.33 | 84.33 | 82.67 | 81.67 | 85.00 | 80.00 | 84.00 | 83.00 | 86.00 |
| SWM-0to008-p1 † | 79.33 | 76.33 | 79.33 | 84.00 | 83.33 | 81.00 | 82.67 | 81.67 | 81.00 | 81.67 |

**Cube eval（epoch_10, num_eval=300, summary.txt clean_300 优先）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | **72.33** | 68.00 | 54.67 | 47.00 | 66.00 | 61.33 | 52.33 | 69.00 | 60.00 | 49.00 |
| LeWM-0to001-p1 | **73.00** | 73.00 | 67.33 | 53.67 | 70.00 | 66.33 | 53.33 | 72.00 | 69.33 | 54.00 |
| LeWM-0to002-p1 | 64.67 | 65.33 | 64.00 | 63.33 | 63.33 | 61.00 | 63.00 | 63.67 | 62.67 | 63.67 |
| LeWM-0to003-p1 † | 65.00 | 67.33 | 65.67 | 67.33 | 69.67 | 66.67 | 67.33 | 67.67 | 66.67 | 68.33 |
| LeWM-0to004-p1 † | 69.00 | 65.67 | 65.67 | 63.67 | 66.00 | 66.33 | 67.00 | 67.00 | 66.00 | 65.00 |
| LeWM-0to005-p1 | 61.33 | 63.33 | 61.67 | 63.00 | 62.33 | 61.33 | 60.67 | 62.00 | 62.67 | 62.00 |
| LeWM-0to006-p1 † | 66.67 | 67.33 | 66.00 | 66.33 | 66.33 | 64.33 | 65.00 | 65.33 | 66.00 | 66.67 |
| LeWM-0to007-p1 † | 67.67 | 68.33 | 65.00 | 69.67 | 68.00 | 67.67 | 68.00 | 67.33 | 69.00 | 67.33 |
| LeWM-0to008-p1 † | 62.33 | 62.00 | 61.67 | 63.00 | 61.00 | 61.00 | 60.33 | 60.67 | 61.67 | 62.00 |
| SWM-base | **77.00** | 62.67 | 49.00 | 47.67 | 69.33 | 56.00 | 52.67 | 64.00 | 48.67 | 50.00 |
| SWM-0to001-p1 | 72.33 | 74.33 | 66.67 | 47.33 | 74.67 | 71.00 | 56.67 | 73.33 | 70.33 | 49.67 |
| SWM-0to002-p1 | 71.00 | 74.00 | 70.67 | 71.33 | 72.67 | 70.67 | 72.67 | 72.00 | 70.33 | 72.33 |
| SWM-0to003-p1 † | 70.33 | 69.67 | 70.33 | 68.33 | 70.00 | 72.67 | 70.67 | 71.33 | 71.33 | 69.00 |
| SWM-0to004-p1 † | 74.33 | 72.33 | 75.00 | 70.67 | 71.67 | 71.33 | 70.33 | 72.67 | 74.00 | 71.33 |
| SWM-0to005-p1 | 62.67 | 63.00 | 62.67 | 63.33 | 64.33 | 64.00 | 63.33 | 64.33 | 63.67 | 63.67 |
| SWM-0to006-p1 † | 70.33 | 70.33 | 68.67 | 71.00 | 70.00 | 68.67 | 70.67 | 70.33 | 71.00 | 71.67 |
| SWM-0to007-p1 † | 72.00 | 72.67 | 71.33 | 73.33 | 71.67 | 70.67 | 70.33 | 70.00 | 71.33 | 71.33 |
| SWM-0to008-p1 † | 70.00 | 72.00 | 69.00 | 71.33 | 70.00 | 70.00 | 68.00 | 71.00 | 70.67 | 69.33 |

> **数据来源说明**：表中 † 行是 3-seed × 100 ep 平均（seed=42/43/44），从每个 ckpt `eval_results/summary.txt` 中 "eval metrics (aggregated across seeds)" 段抽取 `success_rate: mean=...`；其余无 † 行是 single-seed × 300 ep 直接读 `summary.txt::<cond>_300`。**TwoRoom SWM-base / PushT SWM-base / PushT LeWM-0to006-p1 三个 ckpt 已 3-seed × 100 ep retrain 替换**（路径带 `_20260507` 后缀，3-seed std 见本地生成的 `canonical_evals_20260508.json::<task>::<label>::evals::<cond>_std3seed`）；其中 TwoRoom SWM-base + PushT LeWM-0to006-p1 是为了修补**概率性训练发散**，PushT SWM-base 顺带升级到 3-seed。**2026-05-08 SWM noise sweep 补齐**：4 任务 × {0to003, 0to004, 0to006, 0to007, 0to008}-p1 = 20 ckpt 全部 3-seed × 100 ep；这批数据**尚未** ingested 到 `canonical_evals_20260508.json`（该 JSON 仅含 canonical 8）——SWM sweep 的数值直接来自每 ckpt `eval_results/summary.txt` 的 aggregated 段（reacher 系列因当前 summary 的 aggregated 段为空，按 per-seed raw 段重新求 mean，结果一致）。下次 `canonical_evals_*.json` 重生成时应把 SWM/LeWM sweep 全部纳入新口径。`eval_summary.csv` 之前因 `ast.literal_eval` 解析不了 numpy `array(...)` 字面量为空，已在 commit `31751d4` 修复（多行 brace-balanced 扫描 + array(...) 替换）。所有 ckpt 文件名均为 `*_epoch_10_object.ckpt`。Reacher/Cube SWM-base 不存在发散问题，旧 single-seed 数值保留。
>
> **新 LeWM noise sweep + 2026-05-07 retrain 观察**：
> - **概率性训练发散是 cross-method 共有现象**：TwoRoom SWM-base 旧 69.67 与 PushT LeWM-0to006-p1 旧 61.0 是同一类问题——同 config 重训即恢复正常（88.33 / 89.33）。**这条结论改变了之前对 SWM-base 在 TwoRoom "唯一脆弱" 的判定**——那是单 seed 训练运气问题，不是 SWM 方法或 temporal_masked config 的固有缺陷。
> - **TwoRoom**：LeWM noise sweep clean 单调升至 0to008-p1 = 98.33；SWM-base retrain 后 88.33 落入"接近但低于 LeWM-base 93"区间。
> - **PushT**：0to006-p1 retrain 后 89.33；SWM-base retrain 80→85.67（小幅升），但 noise drop @ std=0.05 仍 71/81/80——**clean 提升不带 noise robustness**，是 SWM 方法本身的特点（与训练发散不同的另一回事）。
> - **Reacher**：0to006-p1 = 86.00 反超 canonical 最佳 0to002-p1（80.33），LeWM noise sweep 在 Reacher 上有 ~6pt 上界提升空间。
> - **Cube**：5 个新档位 clean 集中在 62–69，没有突破 canonical 0to001-p1 的 73.0；Cube 上更高 noise 不再带来增益。
>
> 同 ckpt 不同 run 的 single-seed 漂移可达 ±5pt（如 Reacher SWM-0to002-p1 多次 run 在 78–83 之间）；多 seed 平均见各 ckpt 内 `clean_seed42/43/44` 字段。


### 4.3 Eval Drop 与 Noise Sensitivity

**Eval drop（clean − std=0.05；只列 baseline 与最大异常 perframe，其他 perframe drop 全部 |Δ|≤6）**

| 任务 | 模型 | clean | goal_drop | pix_drop | pix+goal_drop |
|---|---|---:|---:|---:|---:|
| TwoRoom | LeWM-base | 93.0 | **22.0** | **22.7** | **30.7** |
| TwoRoom | SWM-base † (20260507) | 88.3 | **42.0** | **32.0** | **45.3** |
| PushT | LeWM-base | 87.3 | **49.3** | **70.0** | **72.3** |
| PushT | SWM-base † (20260507) | 85.7 | **71.3** | **80.7** | **80.3** |
| PushT | SWM-0to001-p1 | 83.3 | 8.7 | **19.7** | **24.7** |
| Reacher | LeWM-base | 57.7 | **33.3** | **31.0** | **32.3** |
| Reacher | SWM-base | 60.0 | **37.7** | **37.7** | **23.3** |
| Cube | LeWM-base | 72.3 | **17.7** | **12.3** | **11.0** |
| Cube | SWM-base | 77.0 | **28.0** | **28.3** | **21.0** |

> **统一口径**：`drop = clean − std=0.05`；canonical perframe + LeWM sweep + 2026-05-08 SWM sweep（共 4 任务 × {LeWM 8 perframe + SWM 8 perframe} − 重复 = 60 行 perframe + 8 baseline = 68 ckpt），baseline 之外所有 perframe 三个 drop 列 |Δ|≤6（多数 ≤2），完整可由上方 eval 表减得；负值（如 Reacher SWM-0to005-p1=−7）是抽样波动。**新 sweep 中唯一 mild 越界**：Reacher LeWM-0to006-p1 † pix+goal_drop = 86.0−79.33 = 6.67，仍属抽样范围。**2026-05-07 retrain 观察**：TwoRoom SWM-base 新版 drop 32–45（相比旧版 35–45 略降但仍崩）；PushT SWM-base 新版 drop 71–81（旧版 73/77/75，几乎一样）——clean 提升但 encoder fragility 不变；PushT LeWM-0to006-p1 retrain 后 drop ≤2，与其它 LeWM perframe 一致。
> - per-frame 独立 std 显著修复 noise failure：所有 perframe drop 集中在 [−7, +6.7]。
> - 唯一例外是 PushT SWM-0to001-p1 的 pix/pix+goal drop 仍 19.7/24.7（noise 强度不够，0to002 已降至 ~1）。
> - clean 差异在 perframe 之间仍然保留：PushT LeWM-0to002 90.0 vs SWM-0to005 71.7，drop 都接近 0 但 clean 差 18pt。
> - **2026-05-08 SWM sweep 补齐后观察**：SWM perframe drop 同样全部 ≤4 abs（4 任务 × 5 新档位 = 20 行），证实 SWM 与 LeWM 一样能通过 perframe noise training 把 noise failure 几乎完全修复——SWM 的 noise robustness 落后于 LeWM 不是结构性的，而是只在 base 与 0to001（noise 强度不足）阶段表现出来。

**Noise sensitivity 对照（std=0.05, goal frame, normalized space；canonical 8 模型/任务）**

> 数据来自每个 ckpt 的 `eval_results/diagnostics/noise_sensitivity.csv`（std=0.05, frame_scope=goal, embedding_space=normalized）。`risk` 取自 CSV 同行字段。**所有 LeWM/SWM 8 模型均使用 epoch_10/num_eval=300 ckpt；TwoRoom/PushT SWM-base 使用 20260507 retrain 版，Reacher/Cube SWM-base 保留旧 single-seed 版。**

TwoRoom：

| 模型 | clean_nn_cos_dist | noise_angle_deg_median | noise_to_nn_cos_ratio | risk |
|---|---:|---:|---:|---|
| LeWM-base | 0.0449 | 5.51° | 0.1031 | low |
| LeWM-0to001-p1 | 0.0430 | 1.83° | 0.0119 | low |
| LeWM-0to002-p1 | 0.0413 | 1.05° | 0.041 | low |
| LeWM-0to005-p1 | 0.0356 | 0.45° | 0.0009 | low |
| SWM-base † (20260507) | 0.0490 | 8.62° | 0.2308 | low |
| SWM-0to001-p1 | 0.0566 | 1.58° | 0.067 | low |
| SWM-0to002-p1 | 0.0521 | 0.86° | 0.022 | low |
| SWM-0to005-p1 | 0.0475 | 0.41° | 0.0005 | low |

> **2026-05-07 retrain 重要修正**：旧 SWM-base 在此条件下 risk=high（angle=20.12°, ratio=1.70），是 52 ckpt 中**唯一** high-risk 标签——这是 §4.5 / §5 一直引用的"主因证据"。3-seed retrain 后 risk 降到 low（angle=8.62°, ratio=0.23）。**"SWM-base TwoRoom 唯一 fragile" 这一判断已不成立**——旧版是 single-seed unlucky outlier，retrain 后 SWM-base 在 std=0.05 处和其它 SWM 一样属 low risk 段。最大 std=0.1 处 noise_angle_slope 仍较高（1852°/std），geometry_flag 仍是 `fragile,high_angle_gain`，但严重程度已从"独特异常"降为"SWM 全任务的共有现象"。

PushT：

| 模型 | clean_nn_cos_dist | noise_angle_deg_median | noise_to_nn_cos_ratio | risk |
|---|---:|---:|---:|---|
| LeWM-base | 0.2360 | 1.33° | 0.011 | low |
| LeWM-0to001-p1 | 0.2242 | 0.61° | 0.0003 | low |
| LeWM-0to002-p1 | 0.2477 | 0.36° | 0.0001 | low |
| LeWM-0to005-p1 | 0.2226 | 0.23° | 0.0000 | low |
| SWM-base † (20260507) | 0.2711 | 1.56° | 0.014 | low |
| SWM-0to001-p1 | 0.2810 | 0.52° | 0.0001 | low |
| SWM-0to002-p1 | 0.2622 | 0.33° | 0.0001 | low |
| SWM-0to005-p1 | 0.2134 | 0.09° | 0.0000 | low |

Reacher：

| 模型 | clean_nn_cos_dist | noise_angle_deg_median | noise_to_nn_cos_ratio | risk |
|---|---:|---:|---:|---|
| LeWM-base | 0.0633 | 3.22° | 0.0249 | low |
| LeWM-0to001-p1 | 0.0670 | 0.80° | 0.014 | low |
| LeWM-0to002-p1 | 0.0696 | 0.09° | 0.0000 | low |
| LeWM-0to005-p1 | 0.0584 | 0.08° | 0.0000 | low |
| SWM-base | 0.0933 | 2.54° | 0.0105 | low |
| SWM-0to001-p1 | 0.0955 | 0.58° | 0.0005 | low |
| SWM-0to002-p1 | 0.0942 | 0.08° | 0.0000 | low |
| SWM-0to005-p1 | 0.0953 | 0.06° | 0.0000 | low |

Cube：

| 模型 | clean_nn_cos_dist | noise_angle_deg_median | noise_to_nn_cos_ratio | risk |
|---|---:|---:|---:|---|
| LeWM-base | 0.1856 | 1.40° | 0.016 | low |
| LeWM-0to001-p1 | 0.1879 | 0.72° | 0.0004 | low |
| LeWM-0to002-p1 | 0.1334 | 0.12° | 0.0000 | low |
| LeWM-0to005-p1 | 0.1176 | 0.08° | 0.0000 | low |
| SWM-base | 0.2596 | 2.85° | 0.048 | low |
| SWM-0to001-p1 | 0.2538 | 0.71° | 0.0003 | low |
| SWM-0to002-p1 | 0.2566 | 0.13° | 0.0000 | low |
| SWM-0to005-p1 | 0.1680 | 0.07° | 0.0000 | low |

**关键事实（数值锚点）**

- **TwoRoom SWM-base 的 `high` 标签已确认为旧 outlier**：旧 single-seed ckpt 在 std=0.05 跨过 ratio=1（1.6978）、noise_angle 20°，曾对应 goal_0.03 暴跌至 21.3；20260507 retrain 后同条件为 8.62°/0.23/risk=low，goal_0.03=56.33。当前可保留的结论是 SWM-base 的 high-std angular slope 仍偏高，但不再是唯一 high-risk 主因。
- **per-frame 训练把 noise_angle@0.05 拉到 <1°**：SWM 任意 perframe 配置（0to001/0to002/0to005-p1）的 angle ≤1.6°（TwoRoom 0to001 是个例外，1.58°）；LeWM 同样从 baseline 1–5° 降到 <0.5°（除 TwoRoom LeWM-base=5.51°）。
- **`clean_nn_cos_dist` 跨任务尺度对比**：SWM normalized space 上 PushT/Cube ≈ 0.21–0.28，TwoRoom/Reacher ≈ 0.04–0.10；LeWM raw 上 PushT≈0.22–0.25，TwoRoom/Reacher≈0.04–0.07。任务自身决定 latent 局部尺度，并非全部由 noise training 决定。
- **PushT noise sweet spot**：SWM 最优 0to001-p1（clean 83.3, goal_0.08=29.7 → fragile under heavy noise；0to002-p1 在 goal_0.08=67.7 更鲁棒但 clean 81.0 略低）；LeWM 最优 0to002-p1（clean 90.0, goal_0.08=83.0）。**即使在最优强度，SWM 仍明显落后 LeWM**——这是 SWM 在精细操作任务上的结构性劣势。
- **per-frame 修复 asymmetric（TwoRoom）**：SWM-0to005-p1 的 pix-only / goal-only 都接近 91，baseline 在 24–35% 崩溃。
- **Reacher/Cube 的 per-frame 收益**：Reacher baseline drop 31–38（goal/pix/pix+goal @ std=0.05），0to005-p1 后降至 −7.0–3.0；Cube baseline drop 11–28，0to005-p1 后降至 −1.3–0.0。

**Predictor rollout drift（history 加噪 @ max std=0.08；T8_l2 与 T8_angle）**

| 任务 | 模型 | T8_l2 (base→perframe) | 改善倍数 | T8_angle (base→best perframe) |
|---|---|---:|---:|---:|
| TwoRoom | LeWM | 18.62 → 0.97 (0to005-p1) | **19×** | 88.6° → 3.5° |
| TwoRoom | SWM  | 1.43 → 0.11 (0to005-p1) | **13×** | 91.4° → 4.3° |
| PushT   | LeWM | 18.65 → 3.56 (0to005-p1) | **5×** | — |
| PushT   | SWM  | 1.41 → 0.02 (0to005-p1) | **70×** | — |
| Reacher | LeWM | 15.17 → 0.21 (0to005-p1) | **73×** | 77.0° → 0.7° |
| Reacher | SWM  | 1.39 → 0.01 (0to005-p1) | **139×** | 85.5° → 0.6° |
| Cube    | LeWM | 20.20 → 0.19 (0to005-p1) | **106×** | 87.9° → 0.7° |
| Cube    | SWM  | 1.38 → 0.01 (0to005-p1) | **138×** | 87.7° → 0.6° |

> std=0.05 累积口径（T1→T8）确认 LeWM drift 在 T1 即接近 saturation（0.5–0.9 → 平台），SWM 同样 T1 即饱和——说明 single-step predictor error 主导，per-frame 训练把 Lipschitz 常数压到足够低使大噪声输入也不发散。
>
> **2026-05-07 LeWM noise sweep 扩展**：把 LeWM 0to006/0to007/0to008-p1 的 max std=0.08 history-scope T8_l2 接进来后，TwoRoom 单调降到 0to008 = 0.66（vs 0to005 = 0.97，**28×**）；PushT 单调降到 0to008 = 1.06（vs 0to005 = 3.56，**18×**）；Reacher 0to005 = 0.21 仍是最低（0to008 = 0.36），Cube 0to005 = 0.19 仍是最低（0to008 = 0.60）。即 LeWM 在 TwoRoom/PushT 的 predictor 平滑性可继续随 noise std 提升，Reacher/Cube 在 0to005 即已饱和。

**CKA(clean, noisy) @ max std=0.08**

| 任务 | LeWM-base | LeWM 任意 perframe | SWM-base | SWM 任意 perframe |
|---|---:|---:|---:|---:|
| TwoRoom | 0.27 | 0.98 | 0.39 | 0.99 |
| PushT   | 0.55 | 0.91–0.99 | 0.28 | 0.52–0.89 |
| Reacher | 0.31 | ≥0.999 | 0.23 | 0.43–1.00 |
| Cube    | 0.37 | ≥0.998 | 0.18 | 0.23–0.999 |

> baseline CKA 普遍 <0.6，noise training 后跃升 >0.87；PushT/Reacher/Cube SWM-perframe-0to001 仍相对偏低（0.23–0.58），低强度 noise 在 SWM 上未完全稳定 latent。`pred_target/nn_cos_ratio` 在 max std 下所有 ckpt 都 ≤0.0002（远低于 1.0 警戒线），单点级 target shift 不是任何 ckpt 的瓶颈。

**Task resolution & action predictability（per-task 跨配置范围）**

| 任务 | trans_res_cos 范围 (LeWM/SWM) | id_probe_r² 范围 | lidar_rank 范围 |
|---|---:|---:|---:|
| TwoRoom | 0.55–0.55 / 0.63–0.73 | 0.14–0.28 | 4.3–10.5 |
| PushT   | 0.07–0.09 / 0.11–0.12 | 0.67–0.77 | 13.9–37.3 |
| Reacher | 0.14–0.18 / 0.18–0.19 | 0.16–0.23 / 0.07–0.11 | 42.8–46.9 |
| Cube    | 0.24–0.36 / 0.28–0.51 | 0.56–0.67 / 0.48–0.61 | 37.8–66.3 |

> **关键现象**：
> 1. `trans_res_cos` 区分任务类型：TwoRoom（离散转移，cos 0.55–0.73）vs PushT/Reacher（连续控制，cos 0.07–0.19）vs Cube（中等离散，0.24–0.51）。
> 2. `id_probe_r²` PushT/Cube (0.48–0.77) >> TwoRoom/Reacher (0.07–0.28)：manipulation 任务 latent 天然保留更强 action 可预测性。
> 3. **SWM resolution 受损不是全局**：Reacher SWM 0.18 > LeWM 0.14，Cube SWM 0.30 > LeWM 0.24；只在 PushT 上 SWM 略高于 LeWM (0.11 vs 0.09) 但 eval 显著低。

**Latent-noise sensitivity（直接对 `z` 注入高斯噪声，跳过 encoder；@ std=0.08, history scope T8_l2 / cost_slope_goal）**

| 任务 | LeWM (T8/cost) | SWM (T8/cost) | LeWM 端 noise_geometry | SWM 端 noise_geometry |
|---|---:|---:|---|---|
| TwoRoom | 5.81 / 2.08 | 0.57 / 1.02 | ambient | tangent |
| PushT   | 12.05 / 3.63 | 0.77 / 1.39 | ambient | tangent |
| Reacher | 3.69 / 2.84 | 0.44 / 1.46 | ambient | tangent |
| Cube    | 4.14 / 2.88 | 0.44 / 1.44 | ambient | tangent |

> 范围中给出的是 baseline；perframe 端波动 ±10%，详见各 ckpt `latent_noise_sensitivity.json`。
>
> **三层归因核心结论**：
> 1. **SWM predictor 天生比 LeWM 对 latent perturbation 稳定 8–10×**（cosine/normalized predictor 内建尺度不变性）。
> 2. **LeWM cost surface 对 goal latent 扰动敏感约 2×**（L2 cost 在 Euclidean 空间斜率大）。
> 3. **per-frame pixel-noise training 不改善 predictor 端 latent-noise 鲁棒性**（LeWM-perframe 与 LeWM-base 的 latent T8 drift 几乎相同；SWM-perframe 甚至略升）——瓶颈在 **Layer 1 (encoder)**，noise training 收益集中在 pixel→latent 映射平滑化。
> 4. `robust_radius_z` (history scope, rollout-drift fallback)：TwoRoom 0.05–0.021，PushT 0.018–0.031，Reacher 0.024–0.043，Cube 0.047–0.065；与 eval 相关性 |ρ|≤0.62，不构成强预测信号。

**Planning signal probe（CEM cost 区分 expert vs random）**

所有 ckpt `expert_beats_best_random ≥ 0.844`，`expert_beats_random ≥ 0.984`——planning signal 在 canonical 32 个 ckpt 中都有效（LeWM sweep 20 + SWM sweep 20 = 40 个新 ckpt 未单独跑 planning probe，可由 `run_planning_action_probe.py` 重构后回填，但训练用的 dataset/predictor 已知没有 planning failure，预期一致）。

> **Cost 尺度差异**：LeWM 的 L2 margin ~257–366，SWM 的 cosine margin ~0.64–0.92（理论上界 2），但 normalized 后均工作。差异不在 signal 有无，而在 cost slope 对 latent perturbation 的敏感度（latent-noise 表里 SWM 1.0–1.8 vs LeWM 2.0–3.8）。

**Action effect probe**：已并入 `run_full_diagnostics`（新增 `tools/repr_analysis/action_effect.py`，sub-probe 与 `task_resolution` 平级），`run_trainer.sh` 后续训练自动产 `action_effect.{csv,json}`，并把 `mean_pred_shift_norm` / `action_perturb_pred_shift_corr` / `interpolation_monotonicity` 写入 `diagnostics_summary.json`。canonical 8 × 4 任务现有 ckpt **待回填**：用 `python -m tools.repr_analysis.run_full_diagnostics --skip-noise --skip-predictor --skip-resolution --skip-latent-noise --model <label>=<ckpt> --dataset <task> --save-dir <results_dir>/diagnostics` 单独刷一遍即可。旧独立入口 `run_planning_action_probe.py` 硬编码非 canonical 路径、`action.*` 全部 `KeyError: 'emb'`（commit 13dda0f 已修复 `encode_sequences` 但未重跑），可视为 deprecated。

结果保存：`dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht,reacher,cube}/repr_analysis/{p03_diagnostics,latent_noise_diagnostics}/`

### 4.4 几何形态：per-frame → 平滑化

**核心结论**：noise augmentation 的实现方式决定 latent geometry 形态，而 geometry 形态决定 task-specific eval 走向。这条结论同时排除了"noise 训练就是简单 Lipschitz smoothing"的简单假设。

| 实现 | Geometry | TwoRoom | PushT (asymmetric) | 适用前提 |
|---|---|---|---|---|
| per-frame 独立 std | smoothed（noise angle ≈ 0.4°，clean_nn 不压缩） | clean 不升、所有 noise 条件持平 | asymmetric 修复 | 需要分辨率保留的任务 |

完整证据见：
- §4.2 eval 表（per-frame 全 noise mode 持平）。
- §4.3 几何指标（`clean_nn_cos_dist`、`noise_angle_slope`、`geometry_flag`）量化形态差异。

为什么任务方向不同：
- TwoRoom 内在状态低维，视觉冗余，更强的 invariance 对 planner 有益。
- PushT 需要保留"再推一点 / 已经到位"的细粒度差异，过度平滑合并这些差异即损害 planning resolution。

### 4.5 机制归因：Cost Surface 与 Latent-Noise Probing

目标：把 SWM noise failure 拆成三层归因：

| 层 | 工具 | 问题 |
|---|---|---|
| Encoder | `noise_sensitivity.py` | pixel noise 是否先把 latent goal/history 编到错误区域 |
| Predictor | `predictor_sensitivity.py` / `latent_noise_sensitivity.py` | noisy history 或 noisy z 是否被 predictor 放大 |
| Cost | eval cost swap / `cost_surface_slope_z` | planning cost 是否在大角度或 latent perturbation 下饱和/失去梯度 |

#### 4.5.1 Eval-only cost swap

固定 SWM checkpoint，不重新训练，只在 eval 改 planning cost：

| 变体 | cost type | cost space | 目的 |
|---|---|---|---|
| A | cosine | normalized | 当前 SWM inference |
| B | mse | raw | 检验 L2 cost 是否能缓解 noisy-goal failure |

命令：

```bash
python eval.py --config-name=tworoom.yaml policy=<swm_ckpt> \
  eval.corruption.std=0.03 \
  eval.inference.cost_type=mse eval.inference.cost_space=raw
```

**当前结果（TwoRoom, std=0.03, pixels+goal, num_eval=50）**

| 变体 | cost type | cost space | score |
|---|---|---|---:|
| A | cosine | normalized | 36.0 |
| B | mse | raw | 42.0 |

结论：
- `raw + mse` 只带来小幅回升（+6），没有接近对应旧 ckpt clean SWM（69.67，epoch_10, num_eval=300）或 LeWM std=0.03（90）。该结果仍说明 eval-only cost swap 不足以修复旧 failure；当前 retrain clean=88.33 后需另做同口径 cost swap 才能定量比较。
- cost saturation 可能贡献了一部分损害，但不是主因。
- 主导失败仍然是 upstream encoder / noisy goal embedding corruption：目标 latent 已经偏到错误区域，eval-only cost swap 无法修复。

#### 4.5.2 Latent-noise probing

P5 原本单列为一个诊断实验；这里并入 P2，因为它和 cost swap 都是在做 encoder / predictor / cost 的机制解耦。区别是：P2.1 从 eval 端替换 cost，P2.2 直接在 encoded `z` 上加噪，跳过 encoder。

**当前结论**（按 §5.3 决策标准评估）：
- TwoRoom：`latent_predictor_rollout_T8_l2_history` 与 eval ρ=+0.738，与 input-space 端 `predictor_rollout_T8_l2`（ρ=+0.667）方向一致但更强 → 部分解耦，**predictor 端独立贡献存在**。
- PushT：`latent_predictor_rollout_T8_l2_history`（+0.627）与 input-space `predictor_rollout_T8_l2`（+0.636）几乎共线，单步信号 `predictor_target_to_nn_cos_ratio_at_max_std`（−0.791）为最强且接近 ≥0.7 → 主因仍是 encoder + 单步 predictor target shift，predictor 稳定性信号已显著增强（num_eval=150 重跑后）。
- 三层归因当前判定：**TwoRoom: encoder 主导 + predictor 独立辅助；PushT: encoder + 单步 predictor 主导**。cost surface（latent slope `z`）不是任一任务的主要解释变量。

| 注入位置 | 工具 | 度量的是 |
|---|---|---|
| pixels（全帧） | `noise_sensitivity.py` | encoder local Lipschitz + 几何尺度 |
| pixels（history-only） | `predictor_sensitivity.py` | encoder + predictor 多步累积漂移 |
| latent `z` | `latent_noise_sensitivity.py` | predictor + cost 对 z 的局部 smoothness，剔除 encoder amplification |

当前实现口径：

| 指标 | 当前定义 | 用途 |
|---|---|---|
| `predictor_target_shift_z` | predictor(`z+ε`, a) 与 predictor(`z`, a) 的 cosine / L2 距离 | latent-only single-step sensitivity |
| `predictor_rollout_drift_z(T)` | 从 noisy vs clean latent history 出发的多步 rollout 偏差 | predictor 下游放大 |
| `cost_surface_slope_z` | 固定 prediction，扰动 goal latent 后的 mean cost delta / std | cost 对 goal latent perturbation 的局部斜率 |
| `robust_radius_z` | `target_to_nn_cos_ratio` 跨过 1 时的 std | predictor target shift 相对 clean NN 尺度的 empirical latent radius |

注意：早期草稿把 `robust_radius_z` 写成 rollout drift crossing，这是概念上可选的定义，但**当前代码实现用的是 `target_to_nn_cos_ratio` crossing**。`latent_lipschitz_est` 还不是独立输出字段；不要在结果表里把它当成可用指标。

实现入口：`tools/repr_analysis/latent_noise_sensitivity.py`

- 噪声注入位置：encoder 输出 `z`，跳过 encoder。
- `frame_scopes ∈ {history, goal, all}`，与 input-space 工具镜像。
- `noise_geometry ∈ {auto, ambient, tangent}`；auto 默认对 SWM normalized space 使用 tangent，对 LeWM/raw space 使用 ambient。
- `std_mode ∈ {relative, absolute}`；默认 relative，按 per-token clean norm 缩放，跨 LeWM/SWM 可比。
- 已挂入 `run_full_diagnostics.py`，CLI 默认开启（`--skip-latent-noise` 可关），`diagnostics_summary.json` 多出 `latent_*` 字段。

决策标准：

| 结果 | 解释 | 行动 |
|---|---|---|
| latent diagnostic 与 eval drop \|ρ\| ≥ 0.7 且与 input-space 端解耦 | predictor / cost 是独立失败机制 | 论文贡献里保留 encoder-decoupled latent diagnostic |
| latent 端与 input 端跨模型几乎共线 | encoder amplification 已经吃满信息瓶颈 | 简化诊断，只报 input-space 版本 |
| latent 端弱、input 端强 | encoder 主导，predictor / cost 相对鲁棒 | 把 P3（encoder 拆解）提到主线 |


## 5. 诊断工具与表征分析

### 5.1 指标分层与工具栈

**诊断指标分层**

| 层 | 主指标 |
|---|---|
| Encoder shift | `noise_angle_deg_median/p90`, `noise_l2_median/p90` |
| Encoder geometry | `clean_nn_cos_dist`, `clean_pair_cos_dist`, `clean_norm_mean`, `effective_rank` |
| 派生比例 | `noise_to_nn_cos_ratio_median/p90`, `robust_radius_std`, `noise_angle_slope_deg_per_std` |
| 几何标签 | `geometry_flag` (`clustered/fragile/robust/balanced`) + `recommendation` |
| Predictor | `predictor_rollout_drift(T)`, `predictor_target_shift` |
| Task resolution | `transition_resolution_ratio`, `id_probe_r²`, `lidar_rank` |
| Latent-noise | `predictor_rollout_drift_z(T)`, `cost_surface_slope_z`, `robust_radius_z`, `latent_*_slope_per_std_z` |
| 目标变量 | `eval_drop_{pix+goal,goal,pix}` @ std∈{0.03,0.05,0.08} |

> **报告口径**：主表用 goal-scope median 指标（`robust_radius_std` / `noise_angle_slope` / `clean_nn_cos_dist` / `clean_eff_rank` / `geometry_flag`）；p90 / L2 / history scope 信息进附表（history scope 对应 pixels-only failure）；latent-noise 字段独立附表。

**工具栈**

| 模块 | 输出字段 |
|---|---|
| `noise_sensitivity.py` | encoder shift / geometry / NN ratio；`frame_scope ∈ {goal, history, all}`；含 `_linear_cka` |
| `predictor_sensitivity.py` | history 加噪后的 rollout drift T1..T_max + 单步 target shift |
| `task_resolution.py` | `transition_resolution_ratio` + linear probe ID + LiDAR rank |
| `latent_noise_sensitivity.py` | 直接对 `z` 注入噪声；`noise_geometry ∈ {ambient, tangent}`；输出 `*_z` 字段 |
| `run_full_diagnostics.py` | 统一调度，落 `diagnostics_summary.json` |
| `diagnostic_correlation.py` | 诊断 ↔ eval 自动相关性（Spearman + Pearson + bootstrap CI） |

### 5.2 数据矩阵：canonical 8

> **canonical 8 模型/任务**：base + 0to001-p1 + 0to002-p1 + 0to005-p1 (LeWM 与 SWM 各 4 个) = 8。所有数值取自每个 ckpt 的 `eval_results/diagnostics/diagnostics_summary.json`，goal scope，normalized space。`>0.08` 表示在 std 扫到 0.08 仍未跨过 ratio=1，即极鲁棒。

**TwoRoom（n=8）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | **0.0142** | >0.08 | 1085.8 | 0.0449 | 47.60 | balanced |
| LeWM-0to001-p1 | 0.0415 | >0.08 | 366.3 | 0.0430 | 47.09 | robust |
| LeWM-0to002-p1 | 0.0653 | >0.08 | 211.0 | 0.0413 | 45.54 | robust |
| LeWM-0to005-p1 | **>0.08** | >0.08 | 86.5 | 0.0356 | 40.86 | balanced |
| SWM-base † (20260507) | 0.0095 | >0.08 | 1852.3 | 0.0490 | 37.88 | fragile,high_angle_gain |
| SWM-0to001-p1 | 0.0476 | >0.08 | 325.5 | 0.0566 | 36.66 | robust |
| SWM-0to002-p1 | 0.0752 | >0.08 | 169.9 | 0.0521 | 37.32 | robust |
| SWM-0to005-p1 | **>0.08** | >0.08 | 80.1 | 0.0475 | 36.41 | balanced |

**PushT（n=8）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | 0.0537 | >0.08 | 284.8 | 0.2360 | 76.42 | **robust** |
| LeWM-0to001-p1 | >0.08 | >0.08 | 120.7 | 0.2242 | 78.59 | balanced |
| LeWM-0to002-p1 | >0.08 | >0.08 | 71.8 | 0.2477 | 77.41 | balanced |
| LeWM-0to005-p1 | >0.08 | >0.08 | 46.8 | 0.2226 | 78.32 | balanced |
| SWM-base | 0.0273 | >0.08 | 707.7 | 0.2582 | 52.94 | **robust** |
| SWM-0to001-p1 | 0.0717 | >0.08 | 103.7 | 0.2810 | 55.45 | **robust** |
| SWM-0to002-p1 | >0.08 | >0.08 | 68.6 | 0.2622 | 55.09 | balanced |
| SWM-0to005-p1 | >0.08 | >0.08 | 14.9 | 0.2134 | 51.98 | balanced |

**Reacher（n=8）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | 0.0142 | >0.08 | 831.7 | 0.0633 | 61.04 | balanced |
| LeWM-0to001-p1 | 0.0524 | >0.08 | 159.6 | 0.0670 | 56.16 | robust |
| LeWM-0to002-p1 | >0.08 | >0.08 | 16.6 | 0.0696 | 70.38 | balanced |
| LeWM-0to005-p1 | >0.08 | >0.08 | 15.2 | 0.0584 | 53.41 | balanced |
| SWM-base | 0.0201 | >0.08 | 651.1 | 0.0933 | 50.96 | robust |
| SWM-0to001-p1 | 0.0695 | >0.08 | 111.2 | 0.0955 | 52.64 | robust |
| SWM-0to002-p1 | >0.08 | >0.08 | 16.5 | 0.0942 | 50.64 | balanced |
| SWM-0to005-p1 | >0.08 | >0.08 | 11.0 | 0.0953 | 51.96 | balanced |

**Cube（n=8）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | 0.0356 | >0.08 | 327.0 | 0.1856 | 73.25 | robust |
| LeWM-0to001-p1 | 0.0621 | >0.08 | 150.7 | 0.1879 | 71.83 | robust |
| LeWM-0to002-p1 | >0.08 | >0.08 | 21.6 | 0.1335 | 73.13 | balanced |
| LeWM-0to005-p1 | >0.08 | >0.08 | 15.0 | 0.1176 | 67.51 | balanced |
| SWM-base | 0.0284 | >0.08 | 660.6 | 0.2596 | 53.69 | robust |
| SWM-0to001-p1 | 0.0537 | >0.08 | 152.2 | 0.2538 | 53.10 | robust |
| SWM-0to002-p1 | >0.08 | >0.08 | 26.2 | 0.2566 | 53.18 | balanced |
| SWM-0to005-p1 | >0.08 | >0.08 | 14.7 | 0.1680 | 51.38 | balanced |

**关键发现**

1. **TwoRoom SWM-base 不再是唯一 high-risk outlier**：20260507 retrain 后 `robust_radius=0.0095`、`noise_angle_slope=1852`、std=0.05 risk=low；旧 `0.029/3975/high` 是 single-seed 训练发散放大的诊断。它仍是 TwoRoom 里 slope 最高的 baseline，对应 §4.3 中 noise drop 32–45，说明 encoder fragility 仍在，但不能再作为"SWM-base 唯一崩溃"的主因证据。
2. **per-frame 训练把 noise_angle_slope 压到两位数**：TwoRoom LeWM-0to005-p1 从 1085→86.5 (12×)，SWM-base retrain 后从 1852→80.1 (23×)；PushT/Reacher/Cube 同向变化。`robust_radius` 全部从 0.01–0.05 升到 >0.08（censored）。
3. **Predictor 稳定性意外提升**：per-frame 训练的 rollout drift（T=8 L2）在 max std=0.08 下比 baseline 降低一个数量级。TwoRoom LeWM 18.62→0.97（**19×**）、SWM 1.43→0.11（**13×**）；PushT LeWM 18.65→3.56（**5×**）、SWM 1.41→0.02（**70×**）；Reacher LeWM 15.17→0.21（**73×**）、SWM 1.39→0.01（**139×**）；Cube LeWM 20.20→0.19（**106×**）、SWM 1.38→0.01（**138×**）。说明噪声训练同时改善了动力学预测的平滑性。
4. **clean_nn_cos_dist 不再是 TwoRoom 的强信号**：per-frame 训练的 LeWM 系列 nn 距离从 0.045→0.036 略降，SWM 系列从 0.036→0.057→0.052→0.048 先升后降；与 eval 的相关性在 n=8 canonical 上由旧 ρ=−0.91（含 fixed-std 异常点）回落到 ρ≈+0.04（详见 §5.3）。

**Tail risk（noise sensitivity @ std=0.08, p90 vs median, 多 scope）**

> 完整 32 行原始数据见各 ckpt `noise_sensitivity.csv`。这里只汇总跨任务规律：
>
> - **per-frame 模型 p90 与 median 差 <5°**，分布集中；**baseline p90 比 median 高 15–20°**（TwoRoom/PushT）或 **超过 100°**（Reacher/Cube），明显 tail risk。
> - **`nn_l2_ratio`**（noise_l2 / clean_nn_l2，p90/median 一致）：baseline ≥1.55（TwoRoom 5.12 / PushT 1.55 / Reacher 6.15 / Cube 2.22），per-frame ≤0.51；ratio<1 说明 noise 仍在邻域内。
> - **goal/history/all scope 三者基本一致**（差 <0.3°），pixels-only 与 goal-only failure 不能通过 frame_scope 区分。
> - **PushT SWM-0to001-p1 例外**：median 57.3° / p90 80.2°，说明该 noise 强度下 SWM 仍未稳定（对应 §4.2 该模型 noise drop 仍较大）。
> - **L2 与 cosine ratio 定性一致**：LeWM raw clean_nn_l2 跨任务差异大（TwoRoom 0.058 / PushT 9.47 / Reacher 4.84 / Cube 8.24），SWM normalized clean_nn_l2 更一致（0.27–0.75）。

### 5.3 相关性分析（canonical n=8）

> **2026-05-06 重算**：此前的相关性数据基于一个 8/11/10/11 模型集合，里面混入了已废弃的 `fixed-std` 和 `perframe-p05` 变体。本次重算改为只用 canonical 8 模型/任务（base + 0to001-p1 + 0to002-p1 + 0to005-p1，LeWM 与 SWM 各 4 个），eval 取自每个 ckpt `eval_results/summary.txt` 的 `clean_300` 列（缺失则 `clean`），诊断取自 `eval_results/diagnostics/diagnostics_summary.json`。所有 n=8。
>
> **重要变化**：旧 P0.4/P0.5 中 TwoRoom `clean_nn_cos_dist` ρ=−0.91 / `lidar_rank` ρ=−0.81 等"强信号"只是含 fixed-std 的 SWM-noise_std0_005=97.6 这一个外点拉出来的相关；canonical 8 上 TwoRoom 这两个指标都退化为弱相关。
>
> **2026-05-07 / 2026-05-08 数据扩展**：LeWM noise sweep 0to003 / 0to004 / 0to006 / 0to007 / 0to008-p1（4 任务 × 5 = 20 ckpt，2026-05-07 完成）+ SWM noise sweep 同 5 档位（4 任务 × 5 = 20 ckpt，2026-05-08 完成）共 40 个新 ckpt 已含完整 noise / predictor / resolution / latent_noise / geometry 诊断（`eval_results/diagnostics/`）。可作为**within-method n=8 sweep 单调性子分析**（每任务 LeWM 8 档 + SWM 8 档，分别独立跑一遍相关性），**还可在新 n=16 / 任务上重跑交叉检查**——这是补完后最可信的 within-method × across-method 对照集。
>
> **2026-05-08 canonical n=8 数据修正**：TwoRoom SWM-base / PushT SWM-base / PushT LeWM-0to006-p1 三个 ckpt 用 3-seed retrain 替换（详 §4.2 数据来源说明），eval 与诊断都重算了。下表中部分相关系数因此发生**实质变化**。完整 6-维度交叉检查的最新数值见 §5.4 表，底层为本地生成且 gitignored 的 `canonical_correlations_20260508.json`；本节主表保留作为快照，请优先参考 §5.4。

**核心：诊断指标的任务特异性（canonical n=8）**

| 指标 | TwoRoom (r / ρ) | PushT (r / ρ) | Reacher (r / ρ) | Cube (r / ρ) | 含义 |
|---|---:|---:|---:|---:|---|
| `clean_nn_cos_dist_median` ↔ eval | +0.39 / +0.04 | +0.16 / +0.12 | +0.21 / +0.32 | **+0.81 / +0.79** | Cube 强正；其余三任务弱 |
| `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | −0.96 / −0.43 | **−0.80 / −0.93** | −0.56 / −0.58 | +0.17 / +0.04 | PushT 最强信号；TwoRoom Pearson 强但 Spearman 弱（线性 vs 单调） |
| `predictor_rollout_T8_l2` ↔ eval | +0.22 / +0.23 | +0.68 / **+0.79** | **−0.71 / −0.83** | +0.41 / **+0.76** | **方向反转**：Reacher 强负，TwoRoom 弱正，PushT/Cube 中正 |
| `clean_effective_rank` ↔ eval | +0.49 / +0.44 | +0.77 / **+0.81** | +0.14 / −0.12 | −0.15 / +0.10 | PushT 强正（高维好），其余三任务弱 |
| `lidar_rank` ↔ eval | +0.52 / +0.44 | +0.72 / +0.74 | −0.41 / −0.28 | +0.00 / +0.34 | PushT 强正，TwoRoom/Reacher 中弱 |
| `noise_angle_slope_deg_per_std` ↔ eval | −0.93 / −0.16 | −0.01 / +0.31 | **−0.72 / −0.74** | **+0.77 / +0.90** | Cube 强正、Reacher 强负（**方向反转**） |
| `cka_linear_at_max_std` ↔ eval | +0.58 / +0.29 | −0.08 / −0.02 | **+0.92 / +0.68** | **−0.85 / −0.96** | **方向反转**：Reacher 正、Cube 负 |
| `latent_rollout_angle_slope_per_std_z` ↔ eval | −0.76 / −0.55 | +0.45 / +0.10 | +0.10 / −0.08 | **+0.69 / +0.62** | Cube 中正；TwoRoom 中负 |
| `latent_cost_surface_slope_z` ↔ eval | +0.47 / +0.61 | **+0.74 / +0.93** | −0.20 / −0.14 | −0.28 / −0.37 | PushT 强正；其余三任务弱/中等 |
| `latent_predictor_rollout_T8_l2_history` ↔ eval | +0.47 / +0.53 | +0.74 / **+0.86** | −0.22 / −0.42 | −0.14 / +0.25 | PushT 强正；TwoRoom 中正 |
| `id_probe_r2` ↔ eval | −0.05 / −0.25 | +0.79 / **+0.81** | +0.21 / +0.08 | +0.69 / **+0.83** | PushT/Cube 强正 |
| `transition_resolution_ratio_cos` ↔ eval | +0.14 / −0.01 | −0.76 / −0.60 | +0.39 / +0.42 | −0.64 / −0.68 | PushT 中负、Cube 中负（保留分辨率与 eval 反相关在这两任务上一致） |

结论（基于 canonical n=8）：
- **TwoRoom 瓶颈**：Spearman 上**没有**任何指标 |ρ|≥0.7。最强为 `latent_cost_surface_slope_z` (+0.61)。但 Pearson 上 `predictor_target_to_nn_cos_ratio_at_max_std` r=−0.96、`noise_angle_slope` r=−0.93 给出极强线性信号，说明 ρ 弱是因为 8 个数据点中两个 baseline 严重偏离主线（leverage points）。**TwoRoom 没有线性化也保留 monotonicity 的稳健指标**——这是 canonical 重算后最重要的负面发现。
- **PushT 瓶颈**：predictor target shift 控制是 |ρ|=0.93 的最强信号，其次是 `latent_cost_surface_slope_z`（+0.93）和 `latent_predictor_rollout_T8_l2_history`（+0.86）。三者都暗示 predictor + latent cost surface 的平滑性主导 PushT。
- **Reacher 瓶颈**：`predictor_rollout_T8_l2` ρ=−0.83 最强（drift 越小 eval 越高），其次 `noise_angle_slope` ρ=−0.74（角向增益越小越好）。**与 PushT 不同**：PushT 是 predictor drift 越大反而越好（混合 confounder），Reacher 单调与之相反。
- **Cube 瓶颈**：`cka_linear_at_max_std` ρ=−0.96（CKA 越低 eval 越高，SWM 天生低 CKA 但 eval 高）+ `noise_angle_slope` ρ=+0.90 + `id_probe_r2` ρ=+0.83 + `clean_nn_cos_dist` ρ=+0.79 + `predictor_rollout_T8_l2` ρ=+0.76 + `transition_resolution_ratio_cos` ρ=−0.68 形成一组同向信号群。
- **跨任务通用指标**：仍**无**。`predictor_rollout_T8_l2` 在 Reacher 上 ρ=−0.83，但在 PushT 上 ρ=+0.79（**完全相反**）；`noise_angle_slope` 在 Cube 上 ρ=+0.90 与 Reacher ρ=−0.74 **方向反转**。Paper 主指标必须按任务选择。
- **TwoRoom 退化**：旧版表里 `clean_nn_cos_dist` ρ=−0.91 在 canonical 上变成 +0.04，`lidar_rank` ρ=−0.81 → +0.44，`transition_resolution_ratio` ρ=−0.88 → −0.01。**这些"主指标"在干净模型集合上失效**。最简解释：旧 8-model 集里包含 `LeWM-fixed-std`(97.0)、`SWM-fixed-std`(97.6) 这两个高 eval 但 high-clustering 异常点，把相关性拉到强负——剔除后失效。

> **解释约束**：`predictor_rollout_T8_l2` 在 TwoRoom 弱正、PushT/Cube 中正、Reacher 强负——四种符号都出现说明该指标本身不是因果，更像跨模型混杂多个 confounder（latent 尺度、noise training 强度、task difficulty）。Paper 主指标应优先使用方向稳定且归一化明确的 `predictor_target_to_nn_cos_ratio_at_max_std`（PushT 强、TwoRoom Pearson 强）和 `latent_cost_surface_slope_z`（PushT 强）；rollout drift 只作为辅助或机制图，必须经 §5.5 holdout 验证。

clean eval 与 noise robustness 在 TwoRoom 不是简单正相关：SWM baseline 走"高角向增益 / noise fragile"路径，LeWM per-frame 走"平滑且 clean 不差"路径——两条路径必须用诊断指标分开归因（详 §4.4）。

**局限**：
1. n=8 单 seed，bootstrap CI 普遍宽（多数指标 CI 跨过 0）。`noise_robust_radius_std` 在 6 个 perframe 模型上 censored 为 >0.08，实际可比 n 仅 2–4。
2. `clean_300` 列的单次 single-seed 抽样波动 ±2pt，可能影响 |ρ| 0.05 量级。
3. 只有 TwoRoom/PushT SWM-base、PushT LeWM-0to006-p1 与 LeWM 0to003–0to008 sweep 有 3-seed 平均；canonical 8 中多数点仍是 single-seed。因此所有 ρ 仍应视为 checkpoint-level 点估计，不能当作 task-level 显著性结论。
4. SWM-base 当前 ckpt 是 `temporal_masked_2_dim64`，与历史 90/89 的 SWM-base 不同；TwoRoom/PushT 已用同 config retrain 修复 outlier，但仍不能和历史 `lambda_0p1` 无 temporal_masked 版本直接互证。

**图表**：`p0_correlation_{tworoom,pusht,reacher,cube}.png`、`predictor_drift_eval_correlation.png`、`noise_angle_curve_goal.png`、`noise_ratio_curve_goal.png`、`geometry_tradeoff_goal.png`。保存路径：`dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht,reacher,cube}/repr_analysis/p03_diagnostics/`。

![TwoRoom P0 诊断指标与 eval 相关性](assets/diagnostics/p0_correlation_tworoom.png)
![PushT P0 诊断指标与 eval 相关性](assets/diagnostics/p0_correlation_pusht.png)
![Predictor Drift 与 Eval 相关性（双任务）](assets/diagnostics/predictor_drift_eval_correlation.png)
![Noise Angle 曲线](assets/diagnostics/noise_angle_curve_goal.png)
![Noise Ratio 曲线](assets/diagnostics/noise_ratio_curve_goal.png)
![Geometry Tradeoff 散点](assets/diagnostics/geometry_tradeoff_goal.png)

**自动化相关性结果（canonical n=8，2026-05-06 重算）**

每任务 |ρ| top-5（强相关 ≥0.7 / 中等 0.4–0.7）：

| 任务 | Top-1 ρ | Top-2 ρ | Top-3 ρ | Top-4 ρ | Top-5 ρ |
|---|---|---|---|---|---|
| TwoRoom (无 ≥0.7) | latent_cost_surface_slope_z **+0.61** | latent_rollout_angle_slope_per_std_z −0.55 | latent_predictor_rollout_T8_l2_history +0.53 | id_probe_r2_min −0.47 | clean_effective_rank +0.44 |
| PushT (8 个 ≥0.7) | predictor_target_to_nn_cos_ratio **−0.93** | latent_cost_surface_slope_z **+0.93** | latent_predictor_rollout_T8_l2_history **+0.86** | clean_effective_rank **+0.81** | id_probe_r2 **+0.81** |
| Reacher (2 个 ≥0.7) | predictor_rollout_T8_l2 **−0.83** | noise_angle_slope **−0.74** | cka_linear +0.68 | predictor_target_to_nn_cos_ratio −0.58 | latent_rollout_l2_slope_per_std_z −0.47 |
| Cube (5 个 ≥0.7) | cka_linear_at_max_std **−0.96** | noise_angle_slope **+0.90** | id_probe_r2 **+0.83** | clean_nn_cos_dist **+0.79** | predictor_rollout_T8_l2 **+0.76** |

完整 16 行 ρ + Pearson r 表见 §5.3 master table（cross-task 对比）。每任务"几乎不相关"（|ρ|<0.1）的指标：
- TwoRoom：`clean_nn_cos_dist` (+0.04)、`transition_resolution_ratio_cos` (−0.01)
- PushT：`cka_linear_at_max_std` (−0.02)、`clean_nn_cos_dist` (+0.12)、`lidar_rank/latent_rollout_angle_slope` 边缘
- Reacher：`latent_robust_radius_z` (+0.02)、`id_probe_r2_min` (−0.06)、`latent_rollout_angle_slope` (−0.08)
- Cube：`predictor_target_to_nn_cos_ratio` (+0.04)、`clean_effective_rank` (+0.17)、`lidar_rank` (+0.34)

**Pearson 强而 Spearman 弱**（线性 vs 单调）：TwoRoom `predictor_target_to_nn_cos_ratio` (r=−0.96, ρ=−0.43)、`noise_angle_slope` (r=−0.93, ρ=−0.16)、`latent_robust_radius_z` (r=+0.85, ρ=+0.24)；Reacher `cka_linear` (r=+0.92, ρ=+0.68)。这些指标是 baseline 离群点（高 ratio/slope 集中在 SWM-base/LeWM-base）拉出来的线性，单调性不足。

> 旧 n=8/11/10/11 表（含 fixed-std 与 p05 变体）已废弃。`diagnostic_correlation.csv` 文件需重跑刷新到 canonical 8 集合。


### 5.4 交叉检查：不同验证角度是否一致

**设计**：同一 32 个模型（canonical 8 模型/任务），从 3 个不同"镜头"检验结论稳健性。

**方法 1：within-method 模型间排序**

| 任务 | LeWM-only 主指标 | LeWM-only 排序 | SWM-only 主指标 | SWM-only 排序 |
|---|---|---|---|---|
| TwoRoom | noise_angle_slope | base>0to001>0to002>0to005 ✅ | noise_angle_slope | base>0to001>0to002>0to005 ✅ |
| PushT | predictor_target_to_nn_cos_ratio | base>0to001>0to002>0to005 ✅ | predictor_target_to_nn_cos_ratio | base>0to001>0to002>0to005 ✅ |
| Reacher | noise_angle_slope | base>0to001>0to002<0to005 ⚠️ | noise_angle_slope | base>0to001>0to002>0to005 ✅ |
| Cube | noise_angle_slope | base>0to001>0to002>0to005 ✅ | noise_angle_slope | base>0to001>0to002>0to005 ✅ |

**方法 2：partial vs full eval 对照**

| 对照 | TwoRoom | PushT | Reacher | Cube |
|---|---|---|---|---|
| clean vs pix+goal 最低 | 一致 ✅ | 一致 ✅ | 一致 ✅ | 一致 ✅ |
| baseline vs sweep 范围 | 一致 ✅ | 一致 ✅ | 一致 ✅ | 一致 ✅ |

**方法 3：group contrast 分组统计**

按 `geometry_flag` 分组（canonical n=8）：

| 任务 | 组 | 模型数 | 中位数 eval | 范围 |
|---|---:|---:|---:|---:|
| TwoRoom | balanced | 2 | 90.7 | 88.3–93.0 |
| TwoRoom | robust | 4 | 96.7 | 93.0–100.0 |
| PushT | balanced | 4 | 88.3 | 87.0–90.0 |
| PushT | robust | 3 | 87.0 | 87.0–87.3 |
| Reacher | balanced | 3 | 60.0 | 57.7–62.0 |
| Reacher | robust | 2 | 60.0 | 60.0–60.0 |
| Cube | balanced | 3 | 75.7 | 71.0–77.0 |
| Cube | robust | 2 | 72.3 | 72.0–72.7 |

> **结论**：`geometry_flag` 分组在 canonical n=8 上几乎无 discriminant power：TwoRoom "balanced" 与 "robust" 组中位数只差 6pt（而且范围重叠），PushT/Cube/Reacher 两组几乎无差异。**旧版固定-std 与 p05 变体把 group contrast 拉出了差异；剔除后几何标签不再提供独立预测价值。** 这支持把 `geometry_flag` 从 paper 主叙事降为"图例装饰"或附录定性参考。

#### P0.5c n=18 sweep cross-check（2026-05-08，LeWM 9 + SWM 9）

> **动机**：P0.5b 仅在 canonical n=8 上做交叉检查；2026-05-08 SWM noise sweep 0to003–0to008 补齐后，每任务可用 LeWM 9 档 + SWM 9 档 = **18 ckpt**（noise std ∈ {0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}，每方法 9 档无重复）做更稳的 within-method × cross-method 联合分析。下方 `ρ_n18` = 联合 Spearman；`p|std_n18` / `p|meth_n18` 分别对 std_max 与 method dummy 做 partial Spearman；`LeWM_n9` / `SWM_n9` = within-method n=9 ρ。**严格门槛**仍是 |ρ_n18| ≥ 0.5 ∧ |p|std_n18| ≥ 0.5 ∧ |p|meth_n18| ≥ 0.5。底层数据见本地生成的 `cross_check_corr_n16_20260508.json`（命名沿用，实际 n=18）。
>
> **前置修复**：跑此分析前需先 `python3 -m tools.repr_analysis.regen_diagnostics_summary <每个 ckpt 的 diagnostics 目录>`——旧版 LeWM-base 与 LeWM/SWM noise sweep ckpt 的 `diagnostics_summary.json` 被后期 action_effect probe 覆盖只剩 5 字段，需用 per-probe JSON 重新合并。

**严格 n=18 通过**（同时满足 |ρ_n18| / |p|std| / |p|meth| ≥ 0.5）：

| 任务 | 指标 | ρ_n18 | LeWM_n9 | SWM_n9 | p|std_n18 | p|meth_n18 | n=8 ρ_all | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PushT | `predictor_target_to_nn_cos_ratio_at_max_std` | **−0.89** | −0.73 | −0.69 | **−0.70** | **−0.91** | −0.90 | n=8 主指标在 n=18 上仍最强 |
| PushT | `latent_cost_surface_slope_z` | **+0.80** | +0.75 | +0.20 | +0.45 | **+0.90** | +0.76 | 仍稳；p|std 较 n=8 (+0.90) 略降 |
| PushT | `predictor_rollout_T8_l2` | +0.74 | +0.24 | +0.58 | +0.34 | **+0.78** | +0.83 | within-LeWM_n9 仅 +0.24，主要靠 SWM/cluster；通过但需 caveat |
| PushT | `id_probe_r2` | +0.82 | +0.54 | +0.37 | +0.48 | +0.67 | +0.71 | p|std 边缘（0.48），但 LeWM_n9 单调性显著好转（n=4=+0.20 → n=9=+0.54） |
| PushT | `clean_effective_rank` | +0.70 | −0.01 | +0.20 | +0.11 | +0.73 | +0.62 | **未通过严格门槛**（p|std=+0.11；本质仍是 cluster 放大） |
| Cube | `cka_linear_at_max_std` | **−0.76** | −0.87 | −0.72 | **−0.80** | −0.65 | −0.96 | 仍稳，p|std 较 n=8 (−0.80) 维持 |
| Cube | `noise_angle_slope_deg_per_std` | +0.75 | +0.85 | +0.72 | **+0.81** | +0.13 | +0.90 | **重要修正**：n=8 p|std=+0.13（被判为"纯 noise-intensity 代理"应降级），**n=18 p|std=+0.81 反升**——sweep 补齐后 std≠eval 单调，within-method 在更长 sweep 上仍排序 eval；该指标**应从"降级"恢复为 Cube 主指标候选** |
| Cube | `clean_nn_cos_dist_median` | +0.71 | +0.58 | +0.49 | +0.60 | +0.66 | +0.79 | n=8 通过，n=18 仍稳 |

**n=18 上信号显著衰减或不通过**：

| 任务 | 指标 | ρ_n18 | n=8 ρ_all | p|std_n18 | p|meth_n18 | 解读 |
|---|---|---:|---:|---:|---:|---|
| Reacher | `predictor_rollout_T8_l2` | **−0.33** | −0.83 | −0.50 | −0.12 | **n=8 主指标在 n=18 上失效**：LeWM_n9=−0.36 / SWM_n9=−0.43 都弱；p|meth 仅 −0.12——之前 −0.83 主要由 method-axis cluster 拉出。**Reacher 应 demote 此主指标** |
| Reacher | `noise_angle_slope_deg_per_std` | −0.37 | −0.74 | −0.39 | −0.11 | 同上稀释，p|meth 完全失效 |
| Reacher | `predictor_target_to_nn_cos_ratio` | −0.41 | −0.57 | −0.53 | **−0.85** | p|std 仍 −0.53、p|meth −0.85 较强；但 ρ_n18 仅 −0.41，未达严格门槛 |
| TwoRoom | `latent_cost_surface_slope_z` | +0.73 | +0.56 | +0.12 | +0.61 | n=18 上 ρ 增强但 p|std 弱（+0.12）——主要由 std 共变 |
| TwoRoom | `latent_predictor_rollout_T8_l2_history` | +0.73 | +0.57 | +0.14 | +0.58 | 同上 |
| TwoRoom | `id_probe_r2` | −0.58 | −0.50 | −0.46 | −0.62 | p|std 边缘（−0.46），仅勉强不达严格 0.5 门槛——是 TwoRoom 4 任务中 candidate 最强者 |
| Cube | `predictor_rollout_T8_l2` | +0.31 | +0.75 | +0.04 | +0.93 | n=8 上 partial|std sign flip 已被识别；n=18 上 ρ_n18 也弱化，进一步证实 cluster 放大 |
| Cube | `id_probe_r2` | +0.04 | +0.83 | +0.39 | +0.31 | 大幅衰减——within-method n=9 都接近 0；n=8 上的 +0.83 被 cluster + std 共变完全解释 |

**结论与 paper 主指标更新（n=18 修订版，2026-05-08）**：

1. **PushT 主指标稳定**：`predictor_target_to_nn_cos_ratio_at_max_std`、`latent_cost_surface_slope_z` 在 n=18 上仍 6/6 通过严格门槛；`predictor_rollout_T8_l2` 与 `id_probe_r2` 在 n=18 上 ρ 仍强但 p|std 边缘（0.34 / 0.48），需 caveat 报告。`clean_effective_rank` 在 n=18 partial|std=+0.11 完全失效——确认是 cluster 放大产物，**降级**。
2. **Cube 主指标修正**：`cka_linear_at_max_std` 与 `clean_nn_cos_dist_median` 仍是主指标。**重要变化**：`noise_angle_slope_deg_per_std` 之前在 n=8 partial|std=+0.13 被判为"纯 noise-intensity 代理"降级；n=18 p|std=+0.81 表明 sweep 补齐后 within-method 在更长 std 范围内仍能排序 eval（即 std 与 eval 共变 ≠ std 唯一原因）——**该指标恢复为 Cube 主指标候选**。`predictor_rollout_T8_l2` 与 `id_probe_r2` 在 n=18 上失效，confirm 降级。
3. **Reacher 主指标全失效**：`predictor_rollout_T8_l2` n=8 ρ=−0.83 → n=18 ρ=−0.33；`noise_angle_slope` n=8 −0.74 → n=18 −0.37。两个 within-method n=9 都 |ρ|≤0.45，且 p|meth 都接近 0。**Reacher 在 n=18 上无任何严格通过的主指标**——之前 n=8 上的"强信号"是 LeWM 4 + SWM 4 的 method-axis 拉出来的，sweep 补齐后被稀释。Paper 应承认 Reacher 缺少 within-method label-free predictor，或单独报 P0.6 holdout 结果。
4. **TwoRoom 主指标弱化**：`id_probe_r2` 在 n=8 上严格通过（p|std=−0.62）；n=18 上 p|std=−0.46（边缘不通过）。`latent_cost_surface_slope_z` n=18 ρ=+0.73 但 p|std=+0.12 失败。**TwoRoom 在 n=18 上无 6/6 严格通过指标**——延续 n=8 的弱信号问题，但比 Reacher 略好（id_probe_r2 仍接近门槛）。
5. **总体**：sweep 补齐让 n 翻倍后**信号普遍稀释**——只有 PushT 的 2 个主指标和 Cube 的 3 个主指标在 n=18 上仍通过严格门槛；Reacher / TwoRoom 主指标全部跌出门槛或边缘。**这是 paper 写作的重要现实约束**：跨任务 label-free predictor 的承诺在更广 sweep 下变弱，必须按任务给条件叙事而非全局承诺。
6. **方法学补丁**：`cross_check_correlations.py` 已扩展支持 within-SWM_n9 + n=18 combined sweep + partial|std/method on n=18；`SWM_SWEEP_EXTRA` / `SWM_SWEEP_EVALS` 落库。

### 5.5 Holdout validation（n=8 sweep 子集 vs canonical 8）

**目的**：验证上述相关性结论是否随方法（LeWM/SWM）、数据集（clean vs dirty）、种子（1 vs 3-seed）改变而稳健。

**实验设计**：
1. **Split 1**：4 模型 held-out = {LeWM-base, SWM-base, LeWM-0to005-p1, SWM-0to005-p1}，4 模型 train = {0to001-p1, 0to002-p1}（4-4 留一/二分交叉验证）。
2. **Split 2**：3 模型 held-out = {base, 0to005-p1}（极端），3 模型 train = {0to001-p1, 0to002-p1}（中间）。
3. **Split 3**：5 模型 held-out = {SWM-0to001/0to002/0to005, LeWM-0to001/0to002}，3 模型 train = {LeWM-base, SWM-base, LeWM-0to005-p1}（method crossover）。
4. **Best-variance test**：用随机切分（bootstrap 100 次）看哪个指标在跨子集时 ρ 符号最稳定（sign consistency）。

**当前状态**：相关脚本（`diagnostic_correlation.py` 的 `cross_validate_subsets` + `bootstrap_sign_consistency`）已写但未跑完。计算量中等（每任务每 split 跑 10+ 指标 × 3 种验证设计 = ~120 次 Spearman 相关），**待执行**。

**预期最低门槛**：
- 至少 1 个指标在至少 2 个任务上（包括 PushT 和 Cube 之一）的 sign consistency ≥80%。
- 若没有任何指标 cross-task sign consistency >60%，结论降级为"task-specific diagnostic is required"。
- PushT 的 `predictor_target_to_nn_cos_ratio` 和 Cube 的 `noise_angle_slope` 是最有希望的不变信号，需验证。

### 5.6 P0.6：分片嵌入探针

**动机**：P0.5 发现评估级别的噪声敏感度（像素级 → 编码器层面）与噪声参数的空间分布（目标、历史、全图）几乎无法区分。同时，与预测器相关的结果（预测器漂移）对噪声参数也不敏感。这表明瓶颈在编码器前端，但需要从像素映射到潜在嵌入的"分片"视角，以揭示编码器内部的崩溃模式。

**假设**：噪声敏感度在分片嵌入层面（即时间帧、通道组或空间组）的分布，相比全局敏感度，能更好地解释任务特定的预测器失败。

**实验**：

1. **输入片段定义**：对于每个任务，将输入划分为**空间**组（例如，PushT 的分割掩码通道、分割嵌入）、**时间**组（上下文帧 vs. 目标帧）和**语义**组（例如，目标掩码 vs. 背景）。使用 `patchify` 后的令牌或 `emb_history/goal/ctxt`，以及可选的注意力掩码，作为探针特征。
2. **每片段噪声注入**：对每个片段独立注入高斯噪声，其它片段保持干净，使用标准噪声敏感框架（范围为 0.05–0.08 的标准差，预测器自举中的余弦/距离指标）。
3. **片段级预测器目标偏移指标**：将 `predictor_target_to_nn_cos_ratio` 和 `noise_angle` 指标从全局扩展到每个片段：记录每个片段单独受扰动时的错误率（或者受扰动与受保护部分的对比）。
4. **聚合**：生成按片段排序的敏感度条形图，按模型和任务聚类。目标是确认某些片段（例如，PushT 的目标掩码与机器人本体）是否主导了全局 `noise_angle` 信号。

**关键指标**：

| 指标 | 来源/计算公式 |
|---|---|
| `fragment_noise_angle_median[segment]` | 片段 `s` 在标准差=0.05 时的余弦噪声角度中位数 |
| `fragment_predictor_target_shift_median[segment]` | 片段 `s` 的目标偏移中位数 |
| `fragment_robust_radius_std[segment]` | 片段 `s` 的比率=1 交点 |
| `fragment_sensitivity_ranking` | 按 `fragment_noise_angle` 降序排列 |

**状态**：待实现。组件：
- `tools/repr_analysis/fragment_noise_sensitivity.py` — 需实现，基于 `noise_sensitivity.py`，增加片段遮罩注入和每片段指标收集；对时间片段（历史/目标）复用 `frame_scope`，对空间片段需支持按通道遮罩；空间遮罩实现和架构具体化强相关（不同任务的分割遮罩通道数不同，如 PushT 是 3/4 通道的分割掩码）。
- `run_full_diagnostics.py` — 添加 `--fragment-segments` 参数和片段级摘要写入。
- 替代/迭代方案：先实现最简单版本（时间片段——等同于已有的 `history/goal/all`，已做；空间片段——对图像输入的 patchify token 做空间分组，如 4 个象限或按语义通道分），再逐步精细化。

**优先级**：P2（机制归因）完成后，根据结论是否还需 encoder 内部分析决定。若 P2 证实 encoder 主导，P0.6 提供的是"哪部分 encoder 最脆弱"的归因信息，优先级升为高；若 P2 证实 predictor 或 cost 主导，P0.6 优先级降为低，可延后。

**已有数据但尚未合并进 P0.6**：
- PushT `goal_split_loss_mse.csv` 的各子目标 MSE（`goal_dist` / `goal_dist_32` / `goal_logit` / `logit32` / `mse` / `mse32`）在噪声实验下已存在，但尚未在分片框架下做聚合/排名分析；这些子目标可视为预定义语义片段的实例。
- 当前 P0.6 与已有空间片段数据的关系：已有数据是粗粒度（子目标级别，对应框架的最终输出头），P0.6 想要的是细粒度（编码器前端的 patch/token 级别）。粗粒度数据可提供验证基准，但两者不是同一回事。

## 6. 讨论

### 6.1 SWM vs LeWM 任务画像

| 维度 | LeWM | SWM | 当前证据 |
|---|---|---|---|
| **几何** | 各向异性（raw/不约束），latent 各向异性（分任务聚类/扩散） | 各向同性 + 强制单位球，latent 各向异性（但投影在球面上） | §4.3, §4.4 几何指标 |
| **敏感区域** | 小聚类时敏感，大聚类时鲁棒 | 低噪声时敏感，高噪声时鲁棒 | §4.2 噪声 drop 表 |
| **任务分辨率** | PushT 高 (0.07–0.09)、Cube 中 (0.24–0.36) | PushT 略高 (0.11–0.12)、Cube 略高 (0.28–0.51) | §4.3 task resolution 表 |
| **Action 可预测性** | PushT 0.67–0.77，Cube 0.56–0.67 | PushT 低，Cube 略低 | §4.3 |
| **预测器平滑度** | 高（T8_l2 绝对值大但改善空间也大） | 天生平滑（T8_l2 绝对值小） | §4.3 predictor drift 表 |
| **噪声训练收益** | 显著修复所有任务 | 显著修复 all but PushT（PushT 0to001 仍 mild） | §4.2 |
| **最优档位** | TwoRoom 0to008，PushT 0to002，Reacher 0to005/0to006，Cube 0to005 | 0to002–0to005 普遍最优 | §4.2 全 sweep 表 |
| **排序翻转** | PushT SWM 不能超 LeWM，Reacher 接近持平，Cube SWM>LeWM | Reacher/LeWM 接近持平，PushT 显著低于 | §4.2 |

### 6.2 SWM 不是 LeWM 的严格超集

旧口号"把 Euclidean 空间换成 spherical 自动解决 sigma collapse"在 4 任务测试中并不成立：

- **TwoRoom**：per-frame SWM-base retrain 后 clean 88.3 vs LeWM-base 93.0，0to005-p1 下 SWM 95.0 vs LeWM 98.0。**SWM 未能超越 LeWM 任何档位**。
- **PushT**：SWM 最优 0to002-p1 81.0 vs LeWM 最优 0to002-p1 90.0。**差距最大**。
- **Reacher**：SWM 0to005-p1 69.0 vs LeWM 0to005/0to006-p1 75.0–80.0；base 下 SWM 60.0 > LeWM 57.7。**接近但略落后**。
- **Cube**：SWM-base 77.0 vs LeWM-base 72.3；SWM 0to005-p1 75.7 vs LeWM 0to005-p1 75.7。**接近持平**。

**结论**：SWM 不是全局更好，也不是全局更差——它是一个在 Cube/Reacher 上接近持平、在 PushT 上落后于 LeWM、在 TwoRoom 上经过 noise 训练后差距缩小的替代设计。这与最初声称的"spherical 自动优于 Euclidean"不符。

### 6.3 P2 失败的 SWM 检查

P2（归因检查）结论：
- `SWM 的 noise failure = 小 noise 下高 angle gain → 目标编码器崩溃`
- `per-frame 训练 = encoder 学习 invariance + predictor 学习平滑 → 修复`
- `PushT failure = 过度平滑化降低 task resolution`

P2 的原始假设是：SWM noise failure 是一种可通过标准方法（噪声训练）系统性修复的编码器-level 问题。新数据（2026-05-07 retrain + 2026-05-08 SWM sweep）后：
- 假设在 TwoRoom/Reacher/Cube 上成立。
- **PushT 不成立**：SWM-0to001-p1 的 noise drop 仍 19.7/24.7，SWM-0to002-p1 clean 81.0（比 LeWM 90.0 差 9pt）。原因不是 noise failure，而是**SWM 的 cosine predictor 分辨率在 PushT 的连续控制任务上先天不足**。
- **SWM noise sweep 扩展后（0to003–0to008）**：PushT SWM-0to003-p1 的 noise drop 大幅改善至 ≤4（goal/pix/pix+goal），clean 72.0 → SWM-0to004-p1 75.7 → SWM-0to005-p1 77.3 → SWM-0to006-p1 78.7 → SWM-0to007-p1 81.3 → SWM-0to008-p1 80.0。**noise 训练把 robustness 修好了，但 clean 上限仍在 81.3，仍远低于 LeWM 的 90.0**。这支持把"clean resolution"与"noise robustness"在论文里明确分开讨论。

### 6.4 主研究线重构

**原主计划**：先 V0（spherical encoder/predictor + cosine loss + spread/uniformity），再 V1（vMF κ），再 V2（ball-cap OOD）。

**新主计划**：从 V0 SWM base 转向**两路并行**：

1. **方法路：自适应分辨率（P4）**（详见 `plan_adaptive_resolution.md`）。
   - Stage A：probe-only σ（已实现，canonical 8 × 4 任务 clean 与噪声评估稳定）。
   - Stage B：action-aware 门控（已实现，≤8.6% 一致性衰减，action gate 区分度 4–8×；2026-05-07 重跑后 SWM-0to006-p1 + α=0.1 关键数据待补充）。
   - Stage C：adaptive consistency loss（`L_total = L_cos + α(z_t)·L_MSE`；entry gate: TwoRoom probe+gate clean ≥ 92 + BN fix）。
   - 当前判定：
     - TwoRoom：LeWM 足够好（clean 98.33 @ 0to008，noise 修复后 95+），Stage C 对 TwoRoom 边际收益有限。
     - PushT：唯一真正需要 Stage C 的任务（SWM clean 上限 81.3 vs LeWM 90.0，MSE 分辨率对连续控制有天然优势）。
     - Reacher：LeWM 与 SWM 接近，Stage C 用于 Reacher 可能打破平局。
     - Cube：SWM 已足够好，Stage C 可选。
   - **BN 漂移 bug 正交于 Stage C**：必须在任何 `α_cons > 0` 前修复（见 §6.5）。

2. **表征路：诊断框架（P0/P1/P3）**。论文中必须以**任务维度**分开写，而不是一个统一"SWM 就是更好/更差"的全局叙事。每任务给出：
   - 对应的 SWM 瓶颈机制（TwoRoom: noise angle slope；PushT: predictor target shift + resolution；Reacher: predictor rollout drift；Cube: noise angle slope + CKA）。
   - 哪个 diagnostic 是最佳预测器。
   - 解决路径（P4 adaptive resolution / P3 encoder 拆解 / P0.6 分片探针）。

### 6.5 讨论：BN 漂移 bug 的正交性

> **背景**：Pilot-2A（probe-only σ + action-aware gate）中发现，开启 action_gate 的 eval 第一轮就导致 action logging 的 pred_emb / tgt_emb 出现 class-level 偏移（如 action=[−1,1] 对应 gate 值 0.71，action=[0,0] 对应 0.34）。经排查是 eval 模式的 BatchNorm 统计量（冻结 running mean/variance）与训练模式不同：训练时小 batch 统计量（噪声批次 B=32）与 eval 时累积统计量（预训练 100 epoch）差异过大，导致 eval 时 frozen BN 把输入映射到错误分布。
>
> **修复**：在 `compute_action_gate_metrics()` 中，perturb forward 前临时把 BN 层切到 `.eval()`（用 frozen running stats），perturb forward 完恢复 `.train()`。这个改动让 gate 值与 `scale_t` 的 Pearson 相关从 −0.02→0.81，TwoRoom 的 clean eval 从 89.33 恢复到 96.33。
>
> **结论**：
> 1. BN 漂移 bug 独立于 SWM/Spherical 几何——任何使用 BatchNorm 的模型在 action_gate eval 时都会遇到。
> 2. 它是"action-aware adaptive consistency 无法正确工作"的技术障碍，必须在 Stage C 前修复。
> 3. 修复代码已经在 `train.py` 本地（commit 前需要清理）。
> 4. 该 bug 对 TwoRoom 影响最大（clean drop 7pt），PushT/Cube 影响较小（1–2pt），Reacher 未测。

### 6.6 论文叙事建议

**原叙事**：Spherical JEPA 解决 LeWM 的 sigma collapse → 噪声训练进一步修复 → SWM 是 LeWM 的超集。

**新叙事**：

1. **章节 1**：引入问题——LeWM 在简单任务（TwoRoom）上接近完美，但在噪声/部分可观测条件下崩溃；我们假设 Euclidean space 的分布假设（isotropic Gaussian）在异质环境中过度约束。
2. **章节 2**：提出 SWM——spherical encoder/predictor + cosine loss + uniformity regularizer。用 4-task 展示它不是全局更好，而是**任务依赖**：SWM 在 Cube/Reacher 上持平/略优，在 PushT 上因分辨率损失落后，在 TwoRoom 上因噪声角度增益高而 fragility 略高。
3. **章节 3**：提出诊断框架（P0 系列）——5 个诊断指标 + 3 层归因（encoder/predictor/cost）。展示指标是任务特异的（PushT 的 predictor target shift vs Cube 的 noise angle slope），但 canonical n=8 上弱相关。引入噪声训练（per-frame）作为统一修复路径。
4. **章节 4**：噪声训练实验——全 sweep（0to001–0to008）覆盖 4 任务 × 2 方法 = 64 ckpt。证实 per-frame 独立 std 显著改善鲁棒性（drop |Δ|≤6.7），但 PushT SWM 的 clean 上限仍有差距。
5. **章节 5**：提出自适应分辨率（P4）——probe-only σ + action-aware gate + 自适应 consistency，指向 PushT clean 瓶颈的解决方案。给出 Stage A/B/C 的 roadmap 与 entry criteria。
6. **结论**：SWM 不是 LeWM 的替代，而是一个需要**任务适配诊断+自适应训练**才能发挥优势的框架。

**不能保留的旧叙事**：
- "SWM 在所有 4 任务上超过 LeWM"（不成立，PushT 落后）。
- "noise angle slope 是跨任务通用主指标"（不成立，Reacher 与 Cube 方向相反）。
- "clean_nn_cos_dist 负相关"（canonical n=8 上失效）。
- "SWM-base TwoRoom 唯一 fragile"（2026-05-07 retrain 后风险降至 low）。

## 7. 未来工作与路线图

### 7.1 自适应分辨率（P4）

**核心方法**：在 SWM spherical 框架内，用一个**独立的分辨率头**输出每观测浓度 κ，在需要高分辨率的区域（如 PushT 的接触/推移）保留信息，在冗余区域降低 κ（类似 attention / modulation）。

**三阶段 ladder**：
- **Stage A（Probe-only σ，已完成）**：detach σ 头，MSE 预测 loss 不变；验证 σ 头的校准性。见 `plan_adaptive_resolution.md` §2.1。
- **Stage B（Action-aware gate，已完成 ≤ 92%）**：action perturbation → 测量 action sensitivity → 用 scale_t 调制 σ；验证 action 区分度 ≥4×。见 `plan_adaptive_resolution.md` §2.2。
- **Stage C（Adaptive consistency loss，规划中）**：`L_total = L_cos + α(z_t)·L_MSE`，其中 `α(z_t) = sigmoid(κ_head(z_t))` 或 `α(z_t) = 1 − sigmoid(gate(z_t))`。
  - Entry gate: TwoRoom probe+gate clean ≥ 92 + BN fix 后稳定。
  - 预期效果：PushT SWM clean 从 ~81 提升到接近 LeWM 90；TwoRoom 边际增益有限但可接受。

完整细节与 Stage A/B 实验记录见 [`plan_adaptive_resolution.md`](plan_adaptive_resolution.md)。

### 7.2 表征分析路线图（P0 系列）

| 阶段 | 目标 | 产出 | 负责人 | 状态 |
|---|---|---|---|---|
| P0.1 | 诊断指标定义与命名规范 | 10 项指标 + 4 项衍生比例 + `geometry_flag` | — | ✅ 完成 |
| P0.2 | 自动化流水线 | `run_full_diagnostics.py` 产 `diagnostics_summary.json` + 单指标 `.csv` | — | ✅ 完成 |
| P0.3 | 数据矩阵（canonical 8 + sweep） | 4 任务 × 8 模型表（§5.2） | — | ✅ 完成 |
| P0.4 | 相关性分析 | 诊断 ↔ eval Spearman/Pearson 表（§5.3）；2026-05-08 用 n=16 重跑 | — | ✅ canonical n=8 完成；n=16 待跑 |
| P0.5 | 交叉检查 | 6 维度验证 + 5 种不一致检测（§5.4）；旧版结论已废弃 | — | ✅ canonical n=8 完成；n=16 验证待跑 |
| P0.6 | 分片嵌入探针 | `fragment_noise_sensitivity.py`（§5.6） | — | ⏳ 设计完成，代码待写 |
| P0.7 | 决策节点 | Task-specific 诊断 → 自动选择训练策略的决策树 | — | 📋 设计阶段 |

**P0.7 决策节点设计**：
- 输入：clean eval + noise drop + 3 个诊断指标（`noise_angle_slope`、`predictor_target_to_nn_cos_ratio`、`latent_cost_surface_slope_z`）。
- 输出：推荐训练策略（{base, perframe, mix}）和预期 clean/noise 提升范围。
- 验证：用已观测的 64 ckpt 做 leave-one-out 预测准确率评估。
- 状态：仅设计，代码未写。

### 7.3 SWM V1（vMF κ per-obs）

在 V0 SWM 的 L2-normalized 固定球面上，加入 per-observation κ 参数化：

- `z_i = μ_i · vMF(κ_i)`，其中 `κ_i = softplus(κ_head(enc_i))`
- 损失：log-likelihood under vMF + 正则化 `E[κ]` 不过高（避免过锐化）
- 目标：让模型在需要高分辨率的 PushT 上提高 κ，在冗余区域降低 κ
- 与 P4 Stage C 的关系：vMF 是 Stage C 的"概率版本"（连续 κ 调制 vs 离散 gate 切换）；可尝试先跑 vMF 看是否解决 clean 分辨率问题，再决定 Stage C 的必要性
- 状态：未实现。设计文档见 §7.1 `plan_adaptive_resolution.md` V1 章节。

### 7.4 SWM V2（Ball-cap OOD）

在 V1 基础上，增加可学习的 ball-cap（余弦 margin）约束：

- `sim(z, goal) > τ` 是 in-distribution，`sim < τ` 是 OOD
- `τ` 可学习（或用训练集分布校准）
- 应用：planning 时拒绝远离训练分布的 rollout，或在 encoder 训练时惩罚OOD 编码
- 状态：未实现。属于远期工作。

### 7.5 决策节点与论文撰写

| 决策 | 条件 | 状态 |
|---|---|---|
| P4 Stage C 是否值得做 | PushT SWM clean ≥ 85（>80 才值得做） | ⏳ 待 Stage B 重跑后判断 |
| P0.7 决策树是否值得做 | P0.4 n=16 重跑后仍有 ≥2 任务 sign consistency ≥70% | ⏳ 待 n=16 跑完 |
| 是否用 vMF 替代 action-gate | vMF 实验 PushT clean ≥ 85 且无需 per-action gate | ⏳ 未开始 |
| SWM 论文是否可写 | 4 任务 clean 至少 2 个任务 SWM > LeWM，或明确给出"SWM 是任务依赖的"叙事 | ⚠️ 当前 1 个（Cube），需 PushT 修复或接受"非全局"叙事 |

**论文撰写入口**：
- 先写 P0/P2/P4 的完整中文技术报告（作为 `research_notebook_swm.md` 与 `plan_adaptive_resolution.md` 的精炼版）。
- 再写 NeurIPS/ICML 风格的英文论文（引言 + 方法 + 实验 + 讨论）。
- 关键约束：不能把"旧 4 任务平均叙事"当作标题级结论；标题必须是"Task-Dependent Spherical World Models"或"Adaptive Spherical Representations for Visual Planning"。

---


## 附录 A：诊断指标速查

### A.1 指标定义与计算口径

| 指标 | 计算公式 | 口径 |
|---|---|---|
| `noise_angle_deg_median` | `arccos(sim(z_noisy, z_clean))·180/π`，取每帧中位数 | goal scope, normalized space, std=0.05 |
| `noise_l2_median` | `||z_noisy − z_clean||₂`，取中位数 | 同上 |
| `noise_to_nn_cos_ratio_median` | `noise_angle_deg / clean_nn_cos_deg`，取中位数 | `clean_nn_cos_deg = arccos(clean_nn_cos_dist)·180/π` |
| `robust_radius_std` | `noise_angle` 曲线跨过 1 时的 std（crossing） | >0.08 = 极鲁棒 |
| `noise_angle_slope_deg_per_std` | `noise_angle` 对 std 的 log-log 回归斜率 | 反映 angle 随 std 增长的幂律 |
| `geometry_flag` | 阈值判断 | `clustered`（nn<0.05, rank<15）/ `fragile`（nn>0.10, risk）/ `robust`（ratio<1）/ `balanced`（其余） |
| `predictor_rollout_drift_T(T)` | `||rollout_noisy − rollout_clean||₂`（或 cosine angle）在 T 步 | history scope, max std |
| `predictor_target_to_nn_cos_ratio_at_max_std` | 单步 target shift / clean NN 距离，取 max std | 跨任务可比 |
| `clean_effective_rank` | 协方差矩阵 top-10 奇异值能量占比的倒数 | 高维 vs 低维区分 |
| `lidar_rank` | LiDAR 框架下线性 probe 分类 rank | 信息保留度 |
| `transition_resolution_ratio_cos` | 相邻 clean token pair 的 cos dist / all-pair mean | 离散 vs 连续控制区分 |
| `cka_linear_at_max_std` | clean vs noisy latent 的 centered kernel alignment | max std |
| `latent_predictor_rollout_T8_l2_history` | 从 noisy `z` 出发的 rollout drift | latent noise, history scope |
| `latent_cost_surface_slope_z` | 固定 prediction，扰动 goal `z` 后的 cost delta/std | latent noise, goal scope |
| `id_probe_r2` | 线性 probe 拟合 action 的 R² | task resolution |

### A.2 阈值参考（canonical n=8 统计）

| 指标 | 低值（鲁棒） | 中值 | 高值（脆弱） | 说明 |
|---|---|---|---|---|
| `noise_angle_deg_median` | <2° | 2–8° | >10° | SWM baseline 旧版 20°，retrain 后 8.6° |
| `noise_to_nn_cos_ratio_median` | <0.1 | 0.1–1.0 | >1.0 | ratio>1 时 noise 已超出 NN 邻域 |
| `robust_radius_std` | >0.08 | 0.03–0.08 | <0.02 | crossing 越左越脆弱 |
| `noise_angle_slope_deg_per_std` | <100 | 100–500 | >1000 | slope 越高越 fragile |
| `clean_nn_cos_dist` | — | — | — | 无通用阈值；任务内对比用 |
| `predictor_rollout_T8_l2` | 低任务依赖 | — | — | TwoRoom 中值 0.5，PushT 中值 2.0，无统一阈值 |
| `cka_linear_at_max_std` | >0.9 | 0.5–0.9 | <0.5 | baseline 普遍 <0.6 |

> **旧版阈值表已废弃**。`fragile/high_angle_gain` 不再作为 SWM-base TwoRoom 的专属标签（retrain 后 risk=low）。`robust_radius=0.01` 以下仍可视为脆弱，但"唯一"一词已不适用。

### A.3 工具 CLI 参考

```bash
# 全量诊断（默认开启所有子模块）
python -m tools.repr_analysis.run_full_diagnostics \
  --model lewm_base_tworoom=<ckpt_path> \
  --model swm_base_tworoom=<ckpt_path> \
  --dataset tworoom \
  --save-dir <results_dir>/diagnostics \
  --num-eval 300

# 跳过子模块
python -m tools.repr_analysis.run_full_diagnostics \
  --model lewm_base_tworoom=<ckpt_path> \
  --dataset tworoom \
  --save-dir <results_dir>/diagnostics \
  --skip-noise --skip-predictor  # 只跑 resolution + latent-noise

# 单独跑噪声敏感度
python -m tools.repr_analysis.noise_sensitivity \
  --checkpoint <ckpt_path> --dataset <task> --output <dir>

# 单独跑 predictor 敏感度
python -m tools.repr_analysis.predictor_sensitivity \
  --checkpoint <ckpt_path> --dataset <task> --output <dir>

# 单独跑 latent-noise 敏感度
python -m tools.repr_analysis.latent_noise_sensitivity \
  --checkpoint <ckpt_path> --dataset <task> --output <dir> \
  --noise-geometry tangent  # SWM 推荐；LeWM 默认 ambient

# 相关性分析（canonical n=8）
python tools/repr_analysis/diagnostic_correlation.py \
  --results-dir <results_dir> \
  --output diagnostic_correlation.csv

# P0.5b cross-check（6 维度 × 5 种不一致检测）
python tools/repr_analysis/diagnostic_correlation.py \
  --results-dir <results_dir> \
  --cross-check --output cross_check_20260508.json
```

### A.4 文件结构

```
repr_analysis/
├── run_full_diagnostics.py          # 主调度（默认全开）
├── noise_sensitivity.py              # P0.1 encoder shift
├── predictor_sensitivity.py          # P0.2 predictor drift
├── task_resolution.py                # P0.3 task resolution
├── latent_noise_sensitivity.py       # P0.5 latent-noise
├── diagnostic_correlation.py         # P0.4 相关性 + P0.5b 交叉检查
├── action_effect.py                  # action effect probe（sub-probe）
└── fragment_noise_sensitivity.py     # P0.6 分片探针（待实现）
```

## 附录 B：CKPT 溯源表

### B.1 Canonical 8（所有任务）

> 下表列出了 canonical 8 模型（LeWM/SWM 各 base/0to001/0to002/0to005-p1）在各任务上的 ckpt 路径。所有路径相对于 `outputs/` 目录。
>
> **2026-05-08 数据修正**：TwoRoom SWM-base 和 PushT SWM-base 使用 20260507 3-seed retrain 版；PushT LeWM-0to006-p1 使用同 config retrain 版。Reacher/Cube SWM-base 保留旧 single-seed 版（待 3-seed retrain 替换）。LeWM 0to003–0to008 和 SWM 0to003–0to008 共 40 个新 ckpt 不在 canonical 8 表中，见 §B.2。

**TwoRoom**

| 模型 | 目录 | ckpt 文件 | eval 文件 | 备注 |
|---|---|---|---|---|
| LeWM-base | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to001-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to002-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to005-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-base † (20260507) | `outputs/2026-05-07/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | 3-seed retrain |
| SWM-0to001-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-0to002-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-0to005-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |

**PushT**

| 模型 | 目录 | ckpt 文件 | eval 文件 | 备注 |
|---|---|---|---|---|
| LeWM-base | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to001-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to002-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to005-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-base † (20260507) | `outputs/2026-05-07/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | 3-seed retrain |
| SWM-0to001-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-0to002-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-0to005-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |

**Reacher**

| 模型 | 目录 | ckpt 文件 | eval 文件 | 备注 |
|---|---|---|---|---|
| LeWM-base | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to001-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to002-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to005-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-base | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | 旧 single-seed，待 retrain |
| SWM-0to001-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-0to002-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-0to005-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |

**Cube**

| 模型 | 目录 | ckpt 文件 | eval 文件 | 备注 |
|---|---|---|---|---|
| LeWM-base | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to001-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to002-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| LeWM-0to005-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-base | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | 旧 single-seed，待 retrain |
| SWM-0to001-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-0to002-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |
| SWM-0to005-p1 | `outputs/2026-04-22/` | `epoch=10-step=...ckpt` | `eval_results/summary.txt` | canonical |

### B.2 LeWM Noise Sweep（0to003–0to008）与 SWM Noise Sweep（0to003–0to008）

> 2026-05-07 完成 LeWM 扩展 5 档（4 任务 × 5 = 20 ckpt）；2026-05-08 完成 SWM 扩展 5 档（4 任务 × 5 = 20 ckpt）。所有 ckpt 位于 `outputs/2026-05-07/`（LeWM）和 `outputs/2026-05-08/`（SWM）。
>
> **命名规则**：`<method>_noise_std0_<intensity>_p1_<task>_seed<X>/`。
>
> **关键数据**（简化表，完整见 §4.2）：

**TwoRoom LeWM Sweep**

| 档位 | 目录标识 | clean | drop (pix+goal) | predictor_T8_l2 |
|---|---|---:|---:|---:|
| 0to003 | `lewm_noise_std0_003_p1` | 97.67 | 0.0 | — |
| 0to004 | `lewm_noise_std0_004_p1` | 97.67 | 0.0 | — |
| 0to006 | `lewm_noise_std0_006_p1` | 97.33 | 0.0 | 0.66 |
| 0to007 | `lewm_noise_std0_007_p1` | 97.33 | 0.0 | — |
| 0to008 | `lewm_noise_std0_008_p1` | **98.33** | 0.0 | 0.66 |

**PushT LeWM Sweep**

| 档位 | 目录标识 | clean | drop (pix+goal) | predictor_T8_l2 |
|---|---|---:|---:|---:|
| 0to003 | `lewm_noise_std0_003_p1` | 86.0 | — | — |
| 0to004 | `lewm_noise_std0_004_p1` | 87.0 | — | — |
| 0to006 | `lewm_noise_std0_006_p1` | 86.7 | 1.0 | 1.06 |
| 0to007 | `lewm_noise_std0_007_p1` | 85.7 | — | — |
| 0to008 | `lewm_noise_std0_008_p1` | 85.0 | — | — |

**TwoRoom SWM Sweep**

| 档位 | 目录标识 | clean | drop (pix+goal) | predictor_T8_l2 |
|---|---|---:|---:|---:|
| 0to003 | `swm_noise_std0_003_p1` | 95.0 | 0.0 | — |
| 0to004 | `swm_noise_std0_004_p1` | 95.0 | 0.0 | — |
| 0to006 | `swm_noise_std0_006_p1` | 97.0 | 0.0 | — |
| 0to007 | `swm_noise_std0_007_p1` | 96.0 | 0.0 | — |
| 0to008 | `swm_noise_std0_008_p1` | 97.0 | 0.0 | — |

**PushT SWM Sweep**

| 档位 | 目录标识 | clean | drop (pix+goal) | predictor_T8_l2 |
|---|---|---:|---:|---:|
| 0to003 | `swm_noise_std0_003_p1` | 72.0 | ≤4 | — |
| 0to004 | `swm_noise_std0_004_p1` | 75.7 | ≤4 | — |
| 0to005 | `swm_noise_std0_005_p1` | 77.3 | ≤4 | — |
| 0to006 | `swm_noise_std0_006_p1` | 78.7 | ≤4 | — |
| 0to007 | `swm_noise_std0_007_p1` | **81.3** | ≤4 | — |
| 0to008 | `swm_noise_std0_008_p1` | 80.0 | ≤4 | — |

> PushT SWM sweep 完整数据：clean 从 0to003 的 72.0 单调上升到 0to007 的 81.3，0to008 略降到 80.0；noise drop 全部 ≤4（robustness 已修复）。但 81.3 仍远低于 LeWM-0to002-p1 的 90.0。

### B.3 废弃模型

| 模型 | 废弃原因 | 对应 ckpt |
|---|---|---|
| fixed-std | 实现错误（每 epoch 固定一个 std，不覆盖数据分布） | `outputs/2026-04-18/` 部分目录 |
| perframe-p05 | 与 p1 无显著差异，增加复杂度 | `outputs/2026-04-20/` 部分目录 |
| hetero-loss | 灾难性失败（Pilot-1B PushT 87→13），只保留 probe | `outputs/2026-04-20/` hetero-loss 目录 |
| action-gate w/o BN fix | 诊断 bug，非模型 bug；修复后保留 | 同上 |

### B.4 关键文件路径速查

| 文件 | 路径 |
|---|---|
| 原始流水账计划（仅历史参考） | `research_notebook_swm.md` 版本历史（git log） |
| 当前重组版 | `research_notebook_swm.md` |
| 自适应分辨率子计划 | `plan_adaptive_resolution.md` |
| 评估主脚本 | `eval.py` |
| 训练主脚本（LeWM） | `train.py` |
| 训练主脚本（SWM） | `train_swm.py` |
| JEPA 模型 | `jepa.py` |
| 共享模块 | `module.py` |
| LeWM 训练配置 | `config/train/lewm.yaml` |
| SWM 训练配置 | `config/train/swm.yaml` |
| 诊断工具目录 | `tools/repr_analysis/` |
| 评估结果目录 | `outputs/YYYY-MM-DD/<run>/eval_results/` |
| 诊断结果目录 | `outputs/YYYY-MM-DD/<run>/eval_results/diagnostics/` |

## 附录 C：撞车风险

> 本节记录已知的内部/外部撞车风险，供论文发表前核查。

### C.1 内部撞车

| 风险 | 详情 | 缓解 |
|---|---|---|
| P4 与 V1（vMF）重叠 | P4 Stage C 的自适应 consistency 和 V1 的 per-obs κ 都试图解决分辨率问题 | P4 先完成；vMF 若效果更好则合并为 V1'，放弃 gate 路线 |
| P0.6 与 P2 重叠 | 分片探针和机制归因都试图定位 encoder 内故障 | P2 先完成；P0.6 提供进一步细分，不替代 P2 |
| SWM noise sweep 与 LeWM noise sweep 数据重复 | 两 sweep 评估配置相同，只是 method 不同 | 用同一套 `eval.py` 和诊断脚本，数据格式统一，避免格式漂移 |

### C.2 外部撞车

| 论文/方法 | 重叠点 | 区分 |
|---|---|---|
| LeWM (original) | 基线 | SWM 是其 spherical 变体，已明确引用 |
| JEPA/I-JEPA | 自监督 encoder-predictor 架构 | SWM 是具体实例，非架构创新；创新点在 spherical + uniformity + adaptive resolution |
| SIGReg | LeWM 的 regularizer | SWM 用 uniformity 替代，已明确对比 |
| vMF 在表示学习中的使用 | V1 计划 | 我们的 vMF 是 per-observation 且用于 world model 动力学，与分类/聚类中的全局 vMF 不同 |
| Contrastive learning on sphere | Spherical representation | 我们用于视觉动力学预测 + planning，非对比学习 |

### C.3 数据可复现性

| 项 | 状态 | 行动 |
|---|---|---|
| 训练脚本版本 | `train.py` / `train_swm.py` 已 git 追踪 | 论文附 commit hash |
| 评估脚本版本 | `eval.py` 已 git 追踪 | 同上 |
| 配置版本 | `config/train/*.yaml` 已 git 追踪 | 同上 |
| 诊断脚本版本 | `tools/repr_analysis/*.py` 已 git 追踪 | 同上 |
| 种子 | canonical 8 多数 single-seed；sweep 部分 3-seed | 论文声明；3-seed 数据优先引用 |
| 硬件 | 单一 A100 80GB | 论文声明 |
| 依赖版本 | `requirements.txt` + `stable-pretraining` / `stable-worldmodel` | 论文声明 git commit |

## 附录 D：历史修正记录

> 记录本计划中已被推翻的结论及其原因，避免论文/报告中引用过时信息。

| 日期 | 旧结论 | 修正 | 原因 |
|---|---|---|---|
| 2026-04-20 | "SWM 在 4 任务上全局优于 LeWM" | 不成立；PushT SWM 落后于 LeWM | 旧数据只含 TwoRoom + PushT 的 baseline，未跑全 sweep；per-frame 后 SWM 在 PushT 仍差 9pt |
| 2026-04-22 | "SWM-base TwoRoom 唯一 fragile (high-risk)" | 不成立；2026-05-07 retrain 后 risk=low | 旧版是 single-seed training outlier；3-seed 平均后 fragility 仍存在但非"唯一" |
| 2026-04-22 | "clean_nn_cos_dist 负相关"（ρ=−0.91 TwoRoom） | canonical n=8 上 ρ=+0.04 | 旧 8-model 集含 fixed-std 异常点（高 eval + high clustering），剔除后失效 |
| 2026-04-22 | "lidar_rank 负相关"（ρ=−0.81 TwoRoom） | canonical n=8 上 ρ=+0.44 | 同上，fixed-std 异常点驱动 |
| 2026-04-22 | "transition_resolution_ratio 负相关"（ρ=−0.88 TwoRoom） | canonical n=8 上 ρ=−0.01 | 同上 |
| 2026-04-25 | "geometry_flag group contrast 有 discriminant power" | canonical n=8 上几乎无 | 旧版含 fixed-std/p05 变体，把 group 差异拉大；剔除后两组范围重叠 |
| 2026-04-28 | "hetero-loss 路径可行" | 废弃；只保留 probe | Pilot-1B PushT clean 87→13，灾难性失败 |
| 2026-05-02 | "action-gate 可直接用于 Stage C" | 需先修复 BN drift | eval 时 frozen BN 统计量导致 gate 值 class-level 偏移；修复代码已本地实现 |
| 2026-05-07 | "SWM-base PushT clean 85.7" | 修正为 85.7（retrain 后），但 noise drop 仍 71–81 | retrain 改善了 clean 但 encoder fragility 不变；结论不变 |
| 2026-05-08 | "SWM perframe noise 训练无法修复 PushT" | 修正为"SWM 0to003+ 可修复 robustness，但 clean 上限仍低" | SWM sweep 0to003–0to008 补齐后，drop 全部 ≤4，但 clean 最高 81.3 vs LeWM 90.0 |

> **原则**：所有被推翻的结论保留在此表中，不删除；论文/报告中引用时核对日期，避免使用已标注"不成立"的旧结论。

---

*文件结束*
