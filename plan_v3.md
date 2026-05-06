# 球面世界模型实验计划 V3

> 当前定位：本文不是单纯记录”SWM 是否强于 LeWM”，而是整理一个更稳定的研究路线：**world model 的 latent geometry 如何匹配 planning 任务的状态分辨率需求**。  
> 原始设计见 `plan_v2.md`，完整流水实验见 `experiments.md`。

---

## 0. 当前结论

最初问题是：把 LeWM 的 Euclidean embedding + SIGReg 换成 spherical embedding + uniformity，是否能稳定提升规划性能？

目前更准确的判断是：

1. **SWM 不是全局优于 LeWM 的替代品。**  
   旧版 4-task single-seed 平均略高的叙事不可追溯；按最新一致 ckpt 口径（SWM epoch_10, num_eval=300；LeWM epoch_9, num_eval=150），SWM baseline 在 TwoRoom（69.67% vs 93.0%）/ PushT（80.00% vs 80.67%）/ Reacher（60.00% vs 58.67%）上均不占优或持平，仅在 Cube（77.00% vs 74.0%）上略高。SWM fixed-std 在 TwoRoom 上仍有聚簇化红利（97.6%），但 PushT 上同一配方崩溃（61.8%）。

2. **SWM 改变了表征的 invariance-resolution tradeoff。**  
   球面归一化、uniformity、temporal masking、noise augmentation 都在改变“哪些观测差异应该被保留，哪些应该被抹掉”。

3. **不同任务对这个 tradeoff 的偏好可能相反。**  
   TwoRoom 低维、离散、视觉细节冗余，受益于更强 invariance / clustering；PushT 需要精细连续状态分辨率，同样配方会损害控制。

4. **表征分析工具本身是通用贡献。**  
   `noise_sensitivity.py`、robust radius、clean-neighbor distance、noise-induced angular shift 等指标可以在不大量 eval 的情况下诊断 latent geometry 的风险。

因此后续主线不应是“为每个任务手调一套 recipe”，而应是：

> 建立一套可诊断、可预测、最好可自适应的 latent geometry 设计方法，让 world model 根据任务分辨率需求在 robustness 和 precision 之间取舍。

### 0.1 清晰路线（paper-facing）

1. **先把诊断做成预测，不是解释。**
   P0.6 是当前最关键节点：冻结指标和阈值，只看 held-out checkpoint 的诊断输出，提前给出 clean / noisy eval drop 分桶。命中率达标后，`noise_to_nn_ratio`、`empirical robust radius`、`predictor target shift`、`predictor rollout drift` 才能进入论文主图。

2. **主叙事改为 invariance-resolution tradeoff。**
   TwoRoom 的聚簇化收益和 PushT 的分辨率损失是核心现象；不要再写成“spherical 比 Euclidean 更好”。SWM 是一个暴露 tradeoff 的 intervention，diagnostic toolkit 是主要方法贡献。

3. **最小方法推进是 guarded noise consistency。**
   P4 不应继续扫 recipe，而应在 noise consistency 外加 transition/action preservation guardrail，目标是保留 PushT resolution，同时让 TwoRoom 获得 noise smoothing。

4. **实验门槛。**
   Paper 主表至少需要：每任务 3 seeds；统一 `num_eval=300` 口径（当前 SWM 已对齐，LeWM 待补）；TwoRoom / PushT / Reacher / Cube 四任务诊断与 eval 对齐；P0.6 holdout；重要 ablation 只保留 P3 的 BN/LN/dim 与 P4 guardrail。

5. **不作为强贡献。**
   `effective_rank`、LiDAR、CKA、ID probe、Wang-Isola uniformity、randomized smoothing 只能作为 borrowed diagnostics / related primitives；真正可主张的是它们在 planning latent 中的组合比值、noisy-vs-clean rollout drift、以及 active validation protocol。

---

## 1. 方法背景

### 1.1 LeWM Baseline

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

### 1.2 SWM V0

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

### 1.3 当前最佳 SWM 配置

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

## 2. Clean Benchmark

### 2.1 旧 4-task benchmark（来源可追溯性说明）

> **数据来源说明**：以下数据来自 `experiments.md` 记录的早期 4-task benchmark（2026-04-15/20），配置为 epoch=10，num_eval=500，single seed。这些 ckpt 与当前 P0.3 诊断分析使用的模型**不是同一组**（dim、temporal 配置、noise 设置均可能不同），因此本节仅作历史参考，不进入 P0.3 相关性分析。
>
> | Task | LeWM | SWM best | Delta | 来源 ckpt |
> |---|---:|---:|---:|:---|
> | TwoRoom | 93.0 | 90.8 | -2.2 | **旧 benchmark，ckpt 已不在当前目录**。最新 SWM baseline: 69.67%（epoch_10, num_eval=300） |
> | Cube | 69.2 | 74.0 | +4.8 | **旧 benchmark，ckpt 已不在当前目录**。最新 SWM baseline: 77.00%（epoch_10, num_eval=300） |
> | PushT | 89.4 | 89.8 | +0.4 | **旧 benchmark，不可追溯**。最新 SWM baseline: 80.00%（epoch_10, num_eval=300） |
> | Reacher | 62.2 | 66.0 | +3.8 | **旧 benchmark，ckpt 已不在当前目录**。最新 SWM baseline: 60.00%（epoch_10, num_eval=300） |
> | Average | 78.5 | 80.2 | +1.7 | **旧 benchmark 平均，不代表当前模型**。 |

### 2.2 当前 P0.3 诊断用模型 clean benchmark（SWM epoch_10, num_eval=300；LeWM epoch_9, num_eval=150；single seed）

以下模型与 §6 P0.3 / P0.4 相关性分析使用同一组 ckpt，可作为一致基准：

| Task | LeWM best | SWM best | Delta | 说明 |
|---|---:|---:|---:|:---|
| TwoRoom | 96.6 (`lewm_fixed-std_noise0.005`) | 97.6 (`swm_fixed-std_noise0.005`) | +1.0 | LeWM fixed-std 最佳；SWM fixed-std 聚簇化红利 |
| PushT | 89.33 (`lewm_noise_0to002_p1`) | 87.33 (`swm_noise_0to001_p1`) | -2.0 | SWM epoch_10, num_eval=300；fixed-std 仅 61.8（epoch_9） |
| Reacher | 82.67 (`lewm_noise_0to005_p05`) | 78.0 (`swm_noise_0to005_p1`) | -4.67 | SWM noise_0to005_p1 epoch_10, num_eval=300；其余 SWM 配置未重跑 |
| Cube | **74.0** (`lewm_base`, num_eval=150) | 77.00 (`swm_base`) | **+3.0** | SWM epoch_10, num_eval=300；LeWM epoch_9, num_eval=150 |

> *Cube LeWM-base num_eval=150 eval 已完成（74.0%），通过 `eval.py` 分批评估（`world.num_envs=8`）解决 150 个 MuJoCo env 并行导致的 hang 问题。

结论：

- SWM 在 TwoRoom 上可通过 fixed-std 聚簇化取得极高 clean score（97.6），但在 PushT 上同一配方崩溃（61.8）。
- LeWM per-frame 在 PushT 上表现更稳定（最佳 89.33，`lewm_noise_0to002_p1`），说明 Euclidean + 平滑化更适应高分辨率任务。
- **旧 4-task 平均叙事（SWM 略高）不成立**：当 ckpt 来源一致后，SWM 的优势仅体现在特定任务（TwoRoom）和特定配方（fixed-std 聚簇化）上，不具备跨任务泛化性。

---

## 3. Noise Robustness 发现

### 3.1 Eval Noise 结果

Eval corruption 语义：

```bash
eval.corruption.std=0.03        # std > 0 开启
eval.corruption.std=0.0         # 关闭
'eval.corruption.apply_to=[goal]'
```

TwoRoom, std=0.03, num_eval=50（**旧数据，仅供参考；SWM baseline 最新 clean eval 为 69.67%，epoch_10, num_eval=300**）：

| 噪声范围 | LeWM | SWM |
|---|---:|---:|
| clean | 93 | 90.8 |
| pixels + goal | 90 | 36 |
| pixels only | 94 | 66 |
| goal only | ≈93 | 42 |

LeWM noise sweep，全帧加噪：

| std | 0.03 | 0.04 | 0.05 | 0.08 | 0.10 | 0.15 |
|---|---:|---:|---:|---:|---:|---:|
| LeWM eval | 90 | 82 | 78 | 48 | 46 | 30 |

解释：

- LeWM 也会被 noise 打坏，但 break point 明显晚。
- SWM 在 std=0.03 的损害大约接近 LeWM std=0.08~0.10。
- goal noise 比 pixels noise 更致命，说明 planner target embedding 是主要脆弱点。

### 3.2 Noise Sensitivity 指标（早期诊断）

`tools/repr_analysis/noise_sensitivity.py` 用同一批图像比较 clean/noisy embedding：

| Metric | 含义 |
|---|---|
| `noise_angle_deg_median` | clean/noisy embedding 的角向偏移 |
| `clean_nn_cos_dist_median` | clean embedding 最近邻间距 |
| `noise_to_nn_cos_ratio_median` | noise shift / clean nearest-neighbor distance |
| `robust_radius` | ratio 跨过 1 时的 std |

Goal frame, normalized space：

> 注：本表是早期一次 goal-frame normalized diagnostic，用来保留最初的机制线索；后续统一入口 `run_full_diagnostics.py` 的 P0.3 表是当前诊断数值的 source of truth。不要把本节的 robust radius 精确值与 P0.3 直接混用。

| model | std | noise angle median | clean NN cos dist | shift / NN | risk |
|---|---:|---:|---:|---:|---|
| LeWM | 0.005 | 4.18° | 0.0389 | 0.0685 | low |
| LeWM | 0.010 | 8.79° | 0.0389 | 0.3021 | low |
| LeWM | 0.020 | 18.75° | 0.0389 | 1.3640 | high |
| LeWM | 0.030 | 31.97° | 0.0389 | 3.8983 | high |
| SWM | 0.005 | 11.95° | 0.0820 | 0.2646 | low |
| SWM | 0.010 | 25.93° | 0.0820 | 1.2280 | high |
| SWM | 0.020 | 54.05° | 0.0820 | 5.0395 | high |
| SWM | 0.030 | 69.66° | 0.0820 | 7.9602 | high |

早期 takeaway（baseline only，跨 ckpt 完整对照见 §6 P0.3）：SWM 小噪声角向偏移约为 LeWM 的 3×；SWM 的 clean NN 距离反而更大（**不是 embedding 太密导致脆弱，而是 angular sensitivity 高**）；经验 robust radius LeWM ≈ 0.017、SWM ≈ 0.008。这些观测是后续 P1 noise-aware training 与 P0 诊断指标体系的最初动机。

### 3.3 失败机制（结论已收敛，详见 §4.2 与 §6 P2/P5）

SWM noise failure 的拆解结论：

- **主因：encoder noisy goal corruption（angular sensitivity 高）。** BN projector + L2 normalization + dim=64 共同放大 pixel 噪声。
- **次因：cost saturation。** §6 P2.1 eval-only `raw+mse` cost swap 仅小幅回升 +6（36→42），即 cost saturation 不是主因。
- **predictor / cost 是否独立下游放大** 由 §6 P2/P5 latent-noise probing 给出：跨模型 |ρ| 与 input-space 端的关系决定是否独立成立。结论与数据见 §6 P2.2 与 P0.4。

---

## 4. Noise-Aware Training 结果

### 4.1 P1 第一组：SWM noise training（TwoRoom）

TwoRoom SWM，noise augmentation，`std_min=0, std_max=0.05`。ckpt: `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64`。

Eval 结果：

| corruption | apply_to | score |
|---|---|---:|
| std=0 | - | 97.6 |
| std=0.05 | pixels+goal | 98.0 |
| std=0.08 | pixels+goal | 88.0 |
| std=0.05 | pixels only | 56.0 |
| std=0.05 | goal only | 44.0 |

> 注：该模型在 TwoRoom 上表现出强烈的 "聚簇化" 特征（`clean_nn_cos_dist` 极低、`noise_angle_slope` 极高），clean 高分伴随 noise robustness 脆弱。

### 4.1 P1 第二组：SWM fixed-std（PushT）

PushT SWM，training noise **固定为 `std=0.005`**（注意：不是 0.05）。ckpt: `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64`。

Eval 结果：

| corruption | apply_to | score |
|---|---|---:|
| std=0 | - | 61.8 |
| std=0.05 | pixels+goal | 60.0 |

> 注：该模型与 TwoRoom fixed-std 使用相同配方（uniformity + temporal_masked_2 + fixed noise），但在 PushT 上 clean eval 仅 61.8，说明 "聚簇化" 红利具有任务特异性，不能从 TwoRoom 泛化到 PushT。

Noise sensitivity 对照，std=0.005：

| Metric | SWM baseline | SWM noise-train | 变化 |
|---|---:|---:|---:|
| clean NN cos dist | 0.082 | 0.008 | 缩到 1/10 |
| clean NN L2 dist | 0.40 | 0.13 | 缩到 1/3 |
| noise angle median | 11.9° | 27.6° | 变大 |

### 4.2 几何形态：固定 std → 聚簇化；per-frame → 平滑化（两者互斥）

**核心结论**：noise augmentation 的实现方式决定 latent geometry 形态，而 geometry 形态决定 task-specific eval 走向。这条结论同时排除了"noise 训练就是简单 Lipschitz smoothing"的简单假设。

| 实现 | Geometry | TwoRoom | PushT (asymmetric) | 适用前提 |
|---|---|---|---|---|
| 固定全帧 std | clustered（紧聚簇 + 高 angle gain） | clean ↑（信息瓶颈红利） | asymmetric 崩 | 低维、视觉冗余的任务 |
| per-frame 独立 std | smoothed（noise angle ≈ 0.4°，clean_nn 不压缩） | clean 不升、所有 noise 条件持平 | asymmetric 修复 | 需要分辨率保留的任务 |

完整证据见：
- §4.3 eval 表（per-frame 全 noise mode 持平）+ §4.1 旧 fixed-std 表（asymmetric 崩塌为 56/44）。
- §6 P0.3 几何指标（`clean_nn_cos_dist`、`noise_angle_slope`、`geometry_flag`）量化这两种形态。

为什么任务方向不同：
- TwoRoom 内在状态低维，视觉冗余，聚簇化作为信息瓶颈对 planner 有益。
- PushT 需要保留"再推一点 / 已经到位"的细粒度差异，聚簇化合并这些差异即损害 planning resolution。
- Asymmetric 崩塌的根因：固定全帧 std 训练让模型学到"所有帧同噪声水平"的隐式 coupling；per-frame 独立 std 打破这种 coupling。

### 4.3 P1 补完：per-frame 独立 std + noise_prob

**实验设计**

已实现 `utils.py:AddNormalizedGaussianNoise`：每帧独立经过 Bernoulli(`noise_prob`) 决定是否加噪，如加则 std ~ Uniform(`std_min`, `std_max`)。

补跑完成：
- P1.1：TwoRoom SWM/LeWM per-frame `std∈[0,0.05]`，`noise_prob=1.0 / 0.5`
- P1.2：PushT SWM `std∈[0,0.01]`、`std∈[0,0.02]`，`noise_prob=1.0 / 0.5`
- P1.3：PushT LeWM 同条件对照 `std∈[0,0.01]`、`std∈[0,0.02]`、`std∈[0,0.05]`

**TwoRoom eval（num_eval=300，all models）**

> **注**：SWM / LeWM baseline 所有列均为 **num_eval=300** 新跑数据（epoch_10 SWM / epoch_9 LeWM）。per-frame 行部分为 num_eval=150 旧跑，部分为 num_eval=300 新跑。

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWM baseline | **69.67** | 21.33 | 24.67 | 25.00 | 37.33 | 31.00 | 29.67 | 35.00 | 34.67 | 35.00 |
| LeWM baseline | **93.00** | 86.67 | 71.00 | 55.67 | 81.00 | 62.33 | 44.33 | 87.67 | 70.33 | 59.33 |
| SWM 旧版固定 std | **97.6** | — | — | — | — | **98.0** | 88.0 | — | 56.0 | — |
| SWM per-frame p1 | 86.7 | 88.0 | 87.3 | 89.3 | 87.3 | 87.3 | 89.3 | 86.7 | 87.3 | 86.7 |
| SWM per-frame p05 | 87.3 | 86.7 | 88.7 | 86.0 | 86.7 | 85.3 | 88.0 | 87.3 | 86.7 | 85.3 |
| LeWM per-frame p1 | **94.0** | 94.0 | 94.0 | 93.3 | 94.0 | 92.7 | 94.7 | 94.0 | 94.0 | 94.0 |
| LeWM per-frame p05 | **94.0** | 94.7 | 94.0 | 94.7 | 94.0 | 94.7 | 94.0 | 94.0 | 94.0 | 94.0 |

> *per-frame 行数据口径不一（部分为 num_eval=150 旧跑，部分为 num_eval=300 新跑），保留原文供趋势对比，不宜和 baseline 行做小数点级比较。

**PushT eval（num_eval=150）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM baseline | **80.67** | — | — | — | — | — | — | — | — | — |
| SWM baseline | **80.00** | — | — | — | — | — | — | — | — | — |
| SWM 0to001 p1 | 87.3 | 86.0 | 79.3 | 27.3 | 86.0 | 59.3 | 6.0 | 82.7 | 69.3 | 6.7 |
| SWM 0to001 p05 | 78.0 | 71.3 | 58.0 | 28.0 | 73.3 | 51.3 | 10.0 | 72.7 | 60.0 | 13.3 |
| SWM 0to002 p1 | 81.3 | 79.3 | 79.3 | **64.0** | 82.0 | 78.0 | **48.7** | 78.0 | 80.7 | **50.7** |
| SWM 0to002 p05 | 78.7 | 78.0 | 77.3 | 41.3 | 77.3 | 72.0 | 12.0 | 81.3 | 74.0 | 22.7 |
| LeWM 0to001 p1 | 87.3 | 88.0 | 76.7 | 64.7 | 84.0 | 69.3 | 40.7 | 84.7 | 75.3 | 48.7 |
| LeWM 0to002 p1 | **89.3** | 88.0 | **86.7** | **82.0** | 88.0 | **85.3** | **74.0** | 87.3 | **86.0** | **76.0** |
| LeWM 0to005 p1 | 82.0 | 81.3 | 77.3 | 80.7 | 80.0 | 80.0 | 78.0 | 83.3 | 78.7 | 76.0 |

**Eval drop（clean − noisy, num_eval=150）**

TwoRoom：

| 模型 | clean | goal_drop_005 | pix_drop_005 | pix+goal_drop_005 |
|---|---:|---:|---:|---:|
| SWM 旧版固定 std | 97.6 | −0.4 | **41.6** | −0.4 |
| SWM per-frame p1 | 86.7 | −0.6 | −0.6 | −0.6 |
| SWM per-frame p05 | 87.3 | 2.0 | 0.6 | 2.0 |
| LeWM per-frame p1 | **94.0** | 1.3 | 0.0 | 1.3 |
| LeWM per-frame p05 | **94.0** | −0.7 | 0.0 | −0.7 |

PushT：

| 模型 | clean | goal_drop_005 | pix_drop_005 | pix+goal_drop_005 |
|---|---:|---:|---:|---:|
| SWM 0to001 p1 | 87.3 | **28.0** | **18.0** | **28.0** |
| SWM 0to001 p05 | 78.0 | **26.7** | **18.0** | **26.7** |
| SWM 0to002 p1 | 81.3 | 3.3 | 0.6 | 3.3 |
| SWM 0to002 p05 | 78.7 | 6.7 | 4.7 | 6.7 |
| LeWM 0to001 p1 | 87.3 | 18.0 | 12.0 | 18.0 |
| LeWM 0to002 p1 | **89.3** | 4.0 | 3.3 | 4.0 |
| LeWM 0to005 p1 | 82.0 | 2.0 | 3.3 | 2.0 |

> **统一口径**：`goal_drop = clean − goal_0.05`，`pix_drop = clean − pix_0.05`，`pix+goal_drop = clean − pix+goal_0.05`。负值表示 noisy 反而略高于 clean（采样波动）。PushT SWM 0to001 的 goal_drop ≈28 为全表最大，说明低强度 noise 在 SWM 上造成严重的 goal-only failure；LeWM 0to002 的 drop 仅 3–4，几乎免疫。TwoRoom 上 per-frame 模型的 drop 均 <2，说明 per-frame 独立 std 彻底修复了 asymmetric 崩溃。

**Noise sensitivity 对照（std=0.005, goal frame, normalized space）**

| 模型 | clean_nn_cos_dist | noise_angle_deg | noise_to_nn_cos_ratio | risk |
|---|---:|---:|---:|---:|
| SWM baseline | 0.082 | 11.95° | 0.2646 | low |
| SWM 旧版固定 std | **0.008** | **27.6°** | — | — |
| SWM per-frame p1 (0to005) | 0.048 | **0.41°** | 0.0005 | low |
| SWM per-frame p05 (0to005) | 0.050 | **0.48°** | 0.0007 | low |
| LeWM baseline | 0.039 | 4.18° | 0.0685 | low |
| LeWM per-frame p1 (0to005) | 0.036 | **0.44°** | 0.0008 | low |
| LeWM per-frame p05 (0to005) | 0.037 | **0.63°** | 0.0016 | low |

**关键事实（不重复 §4.2 解释，仅给数值锚点）**

- **per-frame 修复 asymmetric**：SWM per-frame p1/p05 的 pix-only / goal-only 维持 85–89%，对比 fixed-std 的 56/44。
- **几何对照（std=0.005, goal frame）**：fixed-std `clean_nn_cos_dist=0.008, noise_angle=27.6°`；per-frame `0.048, 0.41°`（详 noise sensitivity 表）。
- **LeWM 同样有 fixed-std 聚簇化但更弱**：P0.3 标 `fragile,clustered`；per-frame 下 LeWM noise_angle=0.44°、`clean_nn` 几乎不压缩（0.039→0.036）。
- **PushT noise sweet spot**：SWM 最优 0to002 p1（clean 81.3, goal_0.08=64.0）；LeWM 最优 0to002 p1（clean 89.3, goal_0.08=82.0）。**即使最优强度，SWM 仍明显落后 LeWM**——这是 SWM 在精细操作任务上的结构性劣势。

---

## 5. 研究主线重构

### 5.1 不再追求的主线

不建议继续把目标表述为：

```text
证明 spherical representation 比 Euclidean representation 更好
```

原因：

- clean benchmark 优势不稳定；
- noise robustness 暴露了 SWM 的明确弱点；
- 每个任务单独调 recipe 没有方法论分量。

### 5.2 建议采用的主线

建议把论文/项目主线改成：

> JEPA-style world model 的 latent geometry 需要匹配任务的状态分辨率需求。Spherical normalization、uniformity、temporal masking、noise augmentation 都不是单独的性能魔法，而是在调节 robustness 与 precision 的 tradeoff。我们提出诊断指标和实验协议来度量这个 tradeoff，并探索自适应机制来缓解固定配方的任务依赖。

这条主线的支撑点：

1. **诊断指标有预测价值。**  
   robust radius、noise angle slope、clean NN distance 可以解释 eval drop。

2. **任务偏好确实不同。**  
   TwoRoom 和 PushT 对同样 noise augmentation 呈现相反趋势。

3. **固定 recipe 不够。**  
   统一的 spherical/noise/temporal prior 不能同时满足低维导航和高分辨率操作。

4. **下一步自然指向 adaptive resolution。**  
   不再手动按任务调配方，而是让模型或训练目标根据状态/transition 自动调 invariance 强度。

---

## 6. 下一步实验计划

> **当前优先级：P0.6 ≥ P3 > P4**。下一步是 P0.6 holdout 盲分桶；若失败再回 P3（encoder 拆解），若诊断可稳定分桶再进入 P4 adaptive guardrail。

### P0：把诊断变成预测指标

目标：证明 `noise_sensitivity` 不是事后解释，而是能**预测** robustness failure 的指标体系。如果成立，latent geometry 诊断本身就是论文一项独立贡献（§8 贡献 2）。

当前实现入口：

| 功能 | 入口 |
|---|---|
| 全套诊断统一入口 | `tools.repr_analysis.run_full_diagnostics.run_full_diagnostics()` / `python -m tools.repr_analysis.run_full_diagnostics` |
| 原始 clean/noisy 表 | `format_noise_table()` |
| robust radius / slope / recommendation | `summarize_noise_geometry()` |
| noise ratio / angle 曲线 | `plot_noise_curves()` |
| robustness-resolution map | `plot_geometry_tradeoff()` |

#### P0.1 诊断指标分层

| 层 | 指标 | 用途 |
|---|---|---|
| Encoder shift | `noise_angle_deg_median/p90`, `noise_l2_median/p90`, `noise_cos_dist_median/p90` | encoder 直接 angular / L2 sensitivity |
| Encoder geometry | `clean_nn_cos_dist`, `clean_pair_cos_dist`, `clean_norm_mean` | clustering / scale |
| 派生比例 | `noise_to_nn_cos_ratio_median/p90`, `robust_radius_std`, `noise_angle_slope_deg_per_std` | 跨模型可比的鲁棒性界 |
| 几何标签 | `geometry_flag` (`clustered/fragile/robust/balanced`) + `recommendation` | 经验阈值规则；compact 只用 cosine NN 判定，L2 仅作附表诊断 |
| Encoder 区分 | `effective_rank`、`frame_scope="history"` | 区分 collapse vs clustering；history 帧对应 pixels-only failure |
| Predictor side | `predictor_rollout_drift(T)`、`predictor_target_shift` | encoder 之外，predictor 在 noisy history 下的累积漂移 |
| Task resolution | `transition_resolution_ratio = d(z_t, z_{t+1}) / d(z_t, z_far)`、inverse-dynamics linear probe readout、LiDAR rank | 量化任务所需状态分辨率，区分 TwoRoom 与 PushT 偏好 |
| Latent-noise | `predictor_rollout_drift_z(T)`、`cost_surface_slope_z`、`robust_radius_z`、`predictor_{angle,l2}_slope_per_std_z`、`rollout_{angle,l2}_slope_per_std_z` | encoder-decoupled 的 predictor / cost smoothness（见 §6 P2/P5；实现 `latent_noise_sensitivity.py`） |
| 目标变量 | `eval_drop_pix+goal`, `eval_drop_goal_only`, `eval_drop_pix_only`（at std=0.03/0.05/0.08） | 三种 noise mode 分别看，对应不同失败机制 |

**报告口径**

- 主表保留 goal-scope median 指标：`robust_radius_std`、`noise_angle_slope`、`clean_nn_cos_dist`、`clean_eff_rank`、`geometry_flag`。
- 附表必须保留 `p90` / L2 / scope 信息：`noise_angle_deg_p90`、`noise_to_nn_cos_ratio_p90`、`noise_to_nn_l2_ratio_median`、`frame_scope ∈ {goal, history, all}`。其中 history scope 对应 pixels-only failure，不能只看 goal scope。
- latent-noise 附表单独保留：`latent_robust_radius_z`、`latent_predictor_rollout_T8_l2_history`、`latent_cost_surface_slope_z`。这些字段已经在 `diagnostics_summary.json` roll-up 里，但当前还缺跨模型实际结果表。

#### P0.2 诊断实现现状

1. `tools/repr_analysis/noise_sensitivity.py`：
   - 支持 `frame_scope="history" / "goal" / "all"`，与 pixels-only / goal-only / 全帧 failure 对齐。
   - 输出 `effective_rank`（迁自 `analyze_repr.py`），区分 collapse 与 clustering。
   - 含 `_linear_cka` 子空间对齐（§7.1 借用 Kornblith 2019）。
2. `tools/repr_analysis/predictor_sensitivity.py`：
   - 同一 batch 在 history 子段加噪，跑 predictor 多步 rollout，与 clean rollout 比较。
   - 输出 `predictor_rollout_drift_median(T)`（T=1..rollout_steps）和 single-step `predictor_target_shift`。
3. `tools/repr_analysis/task_resolution.py`：
   - `transition_resolution_ratio`：相邻帧距离 / 跨序列随机帧距离的中位数比。
   - inverse-dynamics linear probe（仅训 readout，不动主模型）做 action 可预测性代理。
   - LiDAR rank（temporal 正样本对版本，§7.1 借用 Thilak 2024）。
4. `tools/repr_analysis/run_full_diagnostics.py` 把以上三套统一调度，落 `diagnostics_summary.json` 一行 roll-up。

5. `tools/repr_analysis/latent_noise_sensitivity.py`（**P2/P5**）：
   - 直接对 encoded `z` 注入高斯噪声，跳过 encoder。
   - 支持 `frame_scopes ∈ {history, goal, all}` 与 input-space 工具镜像。
   - 支持 `noise_geometry ∈ {auto, ambient, tangent}`（auto：SWM normalized space 用 tangent，LeWM/raw space 用 ambient）与 `std_mode ∈ {relative, absolute}`（默认 relative，按 per-token clean norm 缩放，跨 LeWM/SWM 可比）。
   - 输出 `predictor_target_shift_z`、`predictor_rollout_drift_z(T)`、`cost_surface_slope_z`、`robust_radius_z`，roll-up 字段已并入 `diagnostics_summary.json`。

- `tools/repr_analysis/diagnostic_correlation.py`：诊断 ↔ eval 自动相关性（Spearman + Pearson + bootstrap CI），结果见 P0.7。

#### P0.3 数据矩阵

**TwoRoom（8 模型）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | **0.0164** | 0.02 | 842.6 | 0.0389 | 29.54 | balanced |
| LeWM-fixed-std | 0.0074 | 0.01 | 1170.3 | **0.0130** | 15.08 | **fragile,clustered** |
| LeWM-perframe-p05 | **>0.08** | **>0.08** | **121.5** | 0.0371 | 27.36 | balanced |
| LeWM-perframe-p1 | **>0.08** | **>0.08** | **86.9** | 0.0357 | 26.58 | balanced |
| SWM-base | **0.0029** | 0.005 | 3975.0 | 0.0360 | 35.39 | fragile,high_angle_gain |
| SWM-fixed-std | **0.00036** | **0.005** | **6199.4** | **0.0082** | 11.61 | **fragile,high_angle_gain,clustered** |
| SWM-perframe-p05 | **>0.08** | **>0.08** | **94.7** | 0.0498 | 26.96 | balanced |
| SWM-perframe-p1 | **>0.08** | **>0.08** | **80.3** | 0.0475 | 36.41 | balanced |

> **>0.08** 表示在 std 测到 0.08 时 `noise_to_nn_ratio` 仍未超过 1，即 extremely robust。

**关键发现**

1. **聚簇化效应被量化**：SWM-fixed-std 的 `robust_radius=0.00036`（ baseline 的 1/50），`clean_nn_cos_dist=0.0082`（缩到 1/7），`geometry_flag` 明确标记 `fragile,high_angle_gain,clustered`。
2. **per-frame 平滑化显著**：SWM-perframe 的 `noise_angle_slope` 从 3975 降到 80（接近 LeWM-perframe 的 87），`clean_nn_cos_dist` 恢复到 0.048，与 baseline 同级。
3. **LeWM 固定 std 也有聚簇化**：`clean_nn_cos_dist=0.013`（baseline 0.039 的 1/3），`robust_radius=0.007`（baseline 的 1/2），flag 为 `fragile,clustered`，但程度远轻于 SWM。
4. **Predictor 稳定性意外提升**：per-frame 训练的 rollout drift（T=8 L2）在 max std=0.08 下比 baseline 降低一个数量级。TwoRoom LeWM 18.07→0.78（**23×**）、SWM 1.43→0.08（**18×**）；PushT SWM 1.41→0.02（**83×**）；Reacher LeWM 14.66→0.17（**86×**）、SWM 1.36→0.01（**136×**）；Cube LeWM 19.68→0.16（**123×**）、SWM 1.39→0.01（**139×**）。说明噪声训练同时改善了动力学预测的平滑性。

**PushT（11 个 geometry rows，n=11 可进相关性分析）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | **0.0529** | 0.05 | 284.8 | 0.2360 | 47.48 | **robust** |
| LeWM-fixed-std | **0.0205** | 0.03 | 711.4 | 0.1447 | 31.40 | **robust** |
| LeWM-perframe-0to001-p1 | **>0.08** | **>0.08** | 121.3 | 0.2263 | 48.36 | balanced |
| LeWM-perframe-0to002-p1 | **>0.08** | **>0.08** | 71.8 | 0.2473 | 48.28 | balanced |
| LeWM-perframe-0to005-p1 | **>0.08** | **>0.08** | 47.5 | 0.2253 | 46.74 | balanced |
| SWM-base | **0.0273** | 0.03 | 707.7 | 0.2582 | 52.94 | **robust** |
| SWM-fixed-std | **0.0005** | **0.005** | **8928.9** | **0.0664** | 18.38 | **fragile,high_angle_gain** |
| SWM-perframe-0to001-p05 | **0.0718** | 0.08 | 169.9 | 0.2577 | 42.62 | **robust** |
| SWM-perframe-0to001-p1 | **0.0717** | 0.07 | 103.7 | 0.2810 | 55.45 | **robust** |
| SWM-perframe-0to002-p05 | **>0.08** | **>0.08** | 88.4 | 0.2760 | 46.04 | balanced |
| SWM-perframe-0to002-p1 | **>0.08** | **0.09** | 68.6 | 0.2622 | 55.09 | balanced |

**Reacher（9 个模型）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | 0.0142 | 0.02 | 831.7 | 0.0633 | 61.0 | balanced |
| LeWM-perframe-0to002-p05 | >0.08 | >0.08 | 20.5 | 0.0726 | 49.4 | balanced |
| LeWM-perframe-0to002-p1 | >0.08 | >0.08 | 16.6 | 0.0696 | 70.4 | balanced |
| LeWM-perframe-0to005-p05 | >0.08 | >0.08 | 13.4 | 0.0690 | 46.0 | balanced |
| LeWM-perframe-0to005-p1 | >0.08 | >0.08 | 15.2 | 0.0584 | 53.4 | balanced |
| SWM-base | 0.0201 | 0.02 | 651.1 | 0.0933 | 51.0 | robust |
| SWM-perframe-0to001-p1 | 0.0695 | 0.06 | 111.2 | 0.0955 | 52.6 | robust |
| SWM-perframe-0to002-p1 | >0.08 | >0.08 | 16.5 | 0.0942 | 50.6 | balanced |
| SWM-perframe-p1 | >0.08 | >0.08 | 11.0 | 0.0953 | 52.0 | balanced |

**Cube（9 个模型）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | 0.0356 | 0.04 | 327.0 | 0.1856 | 73.3 | robust |
| LeWM-perframe-0to002-p05 | >0.08 | >0.08 | 26.7 | 0.1350 | 49.5 | balanced |
| LeWM-perframe-0to002-p1 | >0.08 | >0.08 | 21.6 | 0.1335 | 73.1 | balanced |
| LeWM-perframe-0to005-p05 | >0.08 | >0.08 | 19.1 | 0.1182 | 47.3 | balanced |
| LeWM-perframe-0to005-p1 | >0.08 | >0.08 | 15.0 | 0.1176 | 67.5 | balanced |
| SWM-base | 0.0284 | 0.03 | 660.6 | 0.2596 | 53.7 | robust |
| SWM-perframe-0to001-p1 | 0.0537 | 0.05 | 152.2 | 0.2538 | 53.1 | robust |
| SWM-perframe-0to002-p1 | >0.08 | >0.08 | 26.2 | 0.2566 | 53.2 | balanced |
| SWM-perframe-p1 | >0.08 | >0.08 | 14.7 | 0.1680 | 51.4 | balanced |

> 新增 baselines（20260430）：LeWM-base eval **80.67%**（epoch_9, num_eval=150），geometry `robust`（radius=0.053，eff_rank=47.5）；SWM-base eval **80.00%**（epoch_10, num_eval=300），geometry `robust`（radius=0.037，eff_rank=52.94）。两者均被评为 robust，与 fixed-std 的 fragile 形成对比。

**Noise sensitivity @ std=0.08：median vs p90，多 scope 对比**

TwoRoom：

| 模型 | goal_med | goal_p90 | hist_med | hist_p90 | all_med | all_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | 71.6° | 85.8° | 70.5° | 88.0° | 70.6° | 87.8° | 5.12 |
| LeWM-fixed-std | 86.9° | 92.8° | 87.6° | 93.8° | 87.5° | 93.8° | 9.34 |
| LeWM-perframe-p05 | 10.2° | 14.7° | 10.4° | 14.7° | 10.4° | 14.7° | 0.67 |
| LeWM-perframe-p1 | **7.6°** | **11.5°** | **7.5°** | **11.0°** | **7.6°** | **11.0°** | 0.51 |
| SWM-base | 77.1° | 87.3° | 78.5° | 87.7° | 78.4° | 87.6° | 4.64 |
| SWM-fixed-std | 80.7° | 87.0° | 80.9° | 87.3° | 80.9° | 87.3° | 10.08 |
| SWM-perframe-p05 | 8.8° | 12.8° | 8.6° | 12.3° | 8.6° | 12.3° | 0.49 |
| SWM-perframe-p1 | **7.0°** | **10.3°** | **7.3°** | **10.3°** | **7.2°** | **10.3°** | 0.40 |

PushT：

| 模型 | goal_med | goal_p90 | hist_med | hist_p90 | all_med | all_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | 60.8° | 77.8° | — | — | — | — | 1.55 |
| LeWM-fixed-std | 84.0° | 87.8° | 84.3° | 88.5° | 84.3° | 88.4° | 2.80 |
| LeWM-perframe-0to001-p1 | 19.4° | 26.9° | 19.5° | 26.1° | 19.5° | 26.2° | 0.50 |
| LeWM-perframe-0to002-p1 | 10.7° | 14.6° | 10.8° | 14.9° | 10.8° | 14.9° | 0.27 |
| LeWM-perframe-0to005-p1 | **5.2°** | **7.4°** | **5.3°** | **7.4°** | **5.3°** | **7.4°** | 0.14 |
| SWM-base | 86.6° | 97.3° | 87.5° | 98.7° | 87.4° | 98.5° | 1.91 |
| SWM-fixed-std | 63.2° | 67.8° | 62.9° | 67.9° | 62.9° | 67.9° | 2.87 |
| SWM-perframe-0to001-p05 | 48.8° | 73.3° | 45.9° | 72.1° | 46.3° | 72.3° | 1.15 |
| SWM-perframe-0to001-p1 | 57.3° | 80.2° | 57.8° | 79.7° | 57.7° | 79.7° | 1.28 |
| SWM-perframe-0to002-p05 | 36.2° | 56.9° | 35.8° | 56.7° | 35.8° | 56.7° | 0.84 |
| SWM-perframe-0to002-p1 | **17.9°** | **31.0°** | **18.1°** | **29.4°** | **18.1°** | **29.4°** | 0.43 |
| SWM-perframe-0to005-p1 | **1.0°** | **1.4°** | **0.9°** | **1.5°** | **0.9°** | **1.5°** | 0.03 |

Reacher：

| 模型 | goal_med | goal_p90 | hist_med | hist_p90 | all_med | all_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | 83.0° | 102.5° | 82.6° | 102.1° | 82.6° | 102.2° | 6.15 |
| LeWM-perframe-0to002-p05 | 1.5° | 2.7° | 1.6° | 2.8° | 1.6° | 2.8° | 0.07 |
| LeWM-perframe-0to002-p1 | 1.3° | 2.1° | 1.3° | 2.3° | 1.3° | 2.3° | 0.06 |
| LeWM-perframe-0to005-p05 | 1.0° | 1.8° | 1.0° | 1.8° | 1.0° | 1.8° | 0.05 |
| LeWM-perframe-0to005-p1 | 1.2° | 1.9° | 1.2° | 2.0° | 1.2° | 2.0° | 0.06 |
| SWM-base | 84.1° | 103.9° | 82.1° | 102.9° | 82.2° | 103.0° | 3.10 |
| SWM-perframe-0to001-p1 | 39.1° | 70.9° | 38.0° | 70.1° | 38.2° | 70.1° | 1.53 |
| SWM-perframe-0to002-p1 | 1.3° | 2.2° | 1.3° | 2.2° | 1.3° | 2.2° | 0.05 |
| SWM-perframe-p1 | 0.8° | 1.6° | 0.8° | 1.5° | 0.8° | 1.5° | 0.03 |

Cube：

| 模型 | goal_med | goal_p90 | hist_med | hist_p90 | all_med | all_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | 83.4° | 94.1° | 83.2° | 95.2° | 83.2° | 95.2° | 2.22 |
| LeWM-perframe-0to002-p05 | 2.2° | 3.6° | 2.2° | 3.8° | 2.2° | 3.8° | 0.07 |
| LeWM-perframe-0to002-p1 | 1.9° | 3.0° | 1.8° | 3.1° | 1.8° | 3.1° | 0.07 |
| LeWM-perframe-0to005-p05 | 1.4° | 2.4° | 1.4° | 2.6° | 1.4° | 2.6° | 0.05 |
| LeWM-perframe-0to005-p1 | 1.2° | 2.0° | 1.2° | 2.1° | 1.2° | 2.1° | 0.04 |
| SWM-base | 88.4° | 100.1° | 89.1° | 100.0° | 88.9° | 100.0° | 1.93 |
| SWM-perframe-0to001-p1 | 83.1° | 94.1° | 84.1° | 94.4° | 83.9° | 94.3° | 1.86 |
| SWM-perframe-0to002-p1 | 2.0° | 3.3° | 2.1° | 3.3° | 2.1° | 3.3° | 0.05 |
| SWM-perframe-p1 | 1.2° | 2.0° | 1.2° | 2.0° | 1.2° | 2.0° | 0.04 |

> **Tail failure（p90）**：四个任务中 per-frame 的 p90 与 median 均接近（差 <5°），说明分布集中；baseline 的 p90 比 median 高 15–20°，存在显著的 tail risk。Reacher/Cube 的 baseline p90 甚至超过 100°，tail risk 比 TwoRoom/PushT 更严重。PushT SWM-perframe-0to001 的 p90 远高于 median（80° vs 57°），说明该配置下仍有少数样本对 noise 极其敏感。`nn_l2_ratio` 在 per-frame 模型中普遍 <0.7（远低于 1.0 警戒线），而 fixed-std 模型 >2.8，说明 per-frame 的 noise 幅度被有效控制在 encoder 邻域内。

**Noise sensitivity L2 口径（std=0.08, goal frame）**

TwoRoom：

| 模型 | clean_nn_l2 | noise_l2_med | noise_l2_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|
| LeWM-base | 0.058 | 0.297 | 0.502 | 5.12 |
| LeWM-fixed-std | 0.037 | 0.346 | 0.553 | 9.34 |
| LeWM-perframe-p05 | 0.054 | 0.036 | 0.050 | 0.67 |
| LeWM-perframe-p1 | 0.052 | 0.027 | 0.038 | 0.51 |
| SWM-base | 0.268 | 1.246 | — | 4.64 |
| SWM-fixed-std | 0.042 | 0.423 | 0.498 | 10.08 |
| SWM-perframe-p05 | 0.099 | 0.049 | 0.071 | 0.49 |
| SWM-perframe-p1 | 0.308 | 0.123 | — | 0.40 |

PushT：

| 模型 | clean_nn_l2 | noise_l2_med | noise_l2_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|
| LeWM-base | 9.469 | 14.694 | — | 1.55 |
| LeWM-fixed-std | 0.173 | 0.484 | 0.580 | 2.80 |
| LeWM-perframe-0to001-p1 | 0.218 | 0.109 | 0.152 | 0.50 |
| LeWM-perframe-0to002-p1 | 0.227 | 0.061 | 0.083 | 0.27 |
| LeWM-perframe-0to005-p1 | 0.226 | 0.032 | 0.045 | 0.14 |
| SWM-base | 0.719 | 1.372 | — | 1.91 |
| SWM-fixed-std | 0.161 | 0.462 | 0.561 | 2.87 |
| SWM-perframe-0to001-p05 | 0.250 | 0.288 | 0.358 | 1.15 |
| SWM-perframe-0to001-p1 | 0.750 | 0.958 | — | 1.28 |
| SWM-perframe-0to002-p05 | 0.268 | 0.225 | 0.311 | 0.84 |
| SWM-perframe-0to002-p1 | 0.724 | 0.312 | — | 0.43 |
| SWM-perframe-0to005-p1 | 0.653 | 0.017 | — | 0.03 |

Reacher：

| 模型 | clean_nn_l2 | noise_l2_med | noise_l2_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|
| LeWM-base | 4.840 | 29.754 | — | 6.15 |
| LeWM-perframe-0to002-p05 | 5.106 | 0.369 | — | 0.07 |
| LeWM-perframe-0to002-p1 | 5.130 | 0.305 | — | 0.06 |
| LeWM-perframe-0to005-p05 | 5.010 | 0.241 | — | 0.05 |
| LeWM-perframe-0to005-p1 | 4.562 | 0.282 | — | 0.06 |
| SWM-base | 0.432 | 1.339 | — | 3.10 |
| SWM-perframe-0to001-p1 | 0.437 | 0.669 | — | 1.53 |
| SWM-perframe-0to002-p1 | 0.434 | 0.022 | — | 0.05 |
| SWM-perframe-p1 | 0.437 | 0.015 | — | 0.03 |

Cube：

| 模型 | clean_nn_l2 | noise_l2_med | noise_l2_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|
| LeWM-base | 8.236 | 18.278 | — | 2.22 |
| LeWM-perframe-0to002-p05 | 7.149 | 0.525 | — | 0.07 |
| LeWM-perframe-0to002-p1 | 6.975 | 0.455 | — | 0.07 |
| LeWM-perframe-0to005-p05 | 6.658 | 0.329 | — | 0.05 |
| LeWM-perframe-0to005-p1 | 6.616 | 0.283 | — | 0.04 |
| SWM-base | 0.720 | 1.394 | — | 1.93 |
| SWM-perframe-0to001-p1 | 0.712 | 1.326 | — | 1.86 |
| SWM-perframe-0to002-p1 | 0.717 | 0.036 | — | 0.05 |
| SWM-perframe-p1 | 0.580 | 0.020 | — | 0.04 |

> **L2 口径验证**：`nn_l2_ratio`（noise_l2 / clean_nn_l2）与 cosine 口径的 `noise_to_nn_cos_ratio` 在定性上一致——per-frame 模型 ratio <1（noise 在邻域内），fixed-std ratio >2.8（noise 超出邻域）。Reacher/Cube LeWM 的 clean_nn_l2 绝对值很大（4.8–8.2），说明这两个任务的 Euclidean latent 尺度远大于 TwoRoom/PushT，但 relative ratio 仍具可比性。SWM 归一化后 clean_nn_l2 在 0.4–0.7 之间，跨任务更一致。

**Predictor rollout drift 累积（history 加噪 @ std=0.005）**

TwoRoom：

| 模型 | T1_l2 | T2_l2 | T4_l2 | T8_l2 | T8_angle |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.565 | 0.522 | 0.510 | 0.534 | 2.27° |
| LeWM-fixed-std | 0.908 | 0.899 | 0.889 | 0.886 | 3.29° |
| LeWM-perframe-p05 | **0.070** | **0.068** | **0.068** | **0.061** | 0.26° |
| LeWM-perframe-p1 | **0.050** | **0.049** | **0.045** | **0.043** | 0.19° |
| SWM-base | 0.311 | 0.329 | 0.357 | 0.370 | 21.33° |
| SWM-fixed-std | 0.440 | 0.474 | 0.495 | 0.509 | 29.46° |
| SWM-perframe-p05 | **0.006** | **0.006** | **0.006** | **0.005** | 0.28° |
| SWM-perframe-p1 | **0.005** | **0.005** | **0.005** | **0.005** | 0.28° |

PushT：

| 模型 | T1_l2 | T2_l2 | T4_l2 | T8_l2 | T8_angle |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.289 | 0.301 | 0.345 | 0.431 | 1.90° |
| LeWM-fixed-std | 0.493 | 0.500 | 0.517 | 0.548 | 2.16° |
| LeWM-perframe-0to001-p1 | 0.135 | 0.135 | 0.139 | 0.179 | 0.74° |
| LeWM-perframe-0to002-p1 | **0.082** | **0.079** | **0.083** | **0.100** | 0.42° |
| LeWM-perframe-0to005-p1 | **0.050** | **0.049** | **0.055** | **0.069** | 0.28° |
| SWM-base | 0.054 | 0.056 | 0.066 | 0.076 | 4.35° |
| SWM-fixed-std | 1.143 | 1.141 | 1.155 | 1.149 | 70.16° |
| SWM-perframe-0to001-p05 | 0.016 | 0.017 | 0.018 | 0.021 | 1.20° |
| SWM-perframe-0to001-p1 | **0.009** | **0.010** | **0.011** | **0.013** | 0.73° |
| SWM-perframe-0to002-p05 | **0.008** | **0.008** | **0.009** | **0.010** | 0.59° |
| SWM-perframe-0to002-p1 | **0.006** | **0.006** | **0.007** | **0.008** | 0.47° |
| SWM-perframe-0to005-p1 | **0.002** | **0.002** | **0.002** | **0.002** | 0.14° |

Reacher：

| 模型 | T1_l2 | T2_l2 | T4_l2 | T8_l2 | T8_angle |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.420 | 0.322 | 0.265 | 0.257 | 1.09° |
| LeWM-perframe-0to002-p05 | 0.016 | 0.016 | 0.016 | 0.016 | 0.06° |
| LeWM-perframe-0to002-p1 | 0.012 | 0.012 | 0.012 | 0.013 | 0.05° |
| LeWM-perframe-0to005-p05 | 0.010 | 0.010 | 0.010 | 0.010 | 0.04° |
| LeWM-perframe-0to005-p1 | 0.012 | 0.012 | 0.012 | 0.011 | 0.05° |
| SWM-base | 0.036 | 0.033 | 0.031 | 0.030 | 1.71° |
| SWM-perframe-0to001-p1 | 0.008 | 0.008 | 0.007 | 0.007 | 0.42° |
| SWM-perframe-0to002-p1 | 0.001 | 0.001 | 0.001 | 0.002 | 0.09° |
| SWM-perframe-p1 | 0.001 | 0.001 | 0.001 | 0.001 | 0.06° |

Cube：

| 模型 | T1_l2 | T2_l2 | T4_l2 | T8_l2 | T8_angle |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.241 | 0.218 | 0.205 | 0.179 | 0.74° |
| LeWM-perframe-0to002-p05 | 0.022 | 0.022 | 0.023 | 0.021 | 0.09° |
| LeWM-perframe-0to002-p1 | 0.018 | 0.017 | 0.017 | 0.016 | 0.07° |
| LeWM-perframe-0to005-p05 | 0.013 | 0.014 | 0.015 | 0.013 | 0.06° |
| LeWM-perframe-0to005-p1 | 0.011 | 0.011 | 0.012 | 0.010 | 0.04° |
| SWM-base | 0.041 | 0.038 | 0.035 | 0.031 | 1.79° |
| SWM-perframe-0to001-p1 | 0.010 | 0.010 | 0.009 | 0.008 | 0.45° |
| SWM-perframe-0to002-p1 | 0.002 | 0.002 | 0.002 | 0.002 | 0.10° |
| SWM-perframe-p1 | 0.001 | 0.001 | 0.001 | 0.001 | 0.07° |

> **Drift 累积模式****：TwoRoom LeWM 的 drift 在 T1 就很大（0.5–0.9），之后几乎不增长，说明 predictor 的单步误差是主导；SWM 的 drift 同样 T1 即饱和。PushT SWM-fixed-std 的 T8 angle 高达 70°，说明 predictor 在球面空间中发生了方向翻转。Per-frame training 把 LeWM T8 从 0.55→0.04（TwoRoom）和 0.55→0.07（PushT），把 SWM T8 从 0.37→0.005（TwoRoom）和 0.08→0.002（PushT, 0to005-p1），改善幅度与 T8-only 表一致。

**Predictor rollout drift @ max std=0.08（summary 口径，与 `diagnostics_summary.json` 对齐）**

TwoRoom：

| 模型 | T8_l2 | T8_angle | 备注 |
|---|---:|---:|---|
| LeWM-base | **18.07** | 88.59° | baseline，无 noise 训练 |
| LeWM-perframe-p1 | **0.78** | 3.54° | per-frame，降低 **23×** |
| SWM-base | **1.43** | 91.43° | baseline，无 noise 训练，eval 69.67%（TwoRoom）/ 80.00%（PushT）/ 60.00%（Reacher）/ 77.00%（Cube），epoch_10, num_eval=300 |
| SWM-perframe-p1 | **0.08** | 4.34° | per-frame，降低 **18×** |

Reacher：

| 模型 | T8_l2 | T8_angle | 备注 |
|---|---:|---:|---|
| LeWM-base | **14.66** | 76.95° | baseline，无 noise 训练 |
| LeWM-perframe-0to005-p1 | **0.17** | 0.73° | per-frame，降低 **86×** |
| SWM-base | **1.36** | 85.51° | baseline，无 noise 训练 |
| SWM-perframe-p1 | **0.01** | 0.58° | per-frame，降低 **136×** |

Cube：

| 模型 | T8_l2 | T8_angle | 备注 |
|---|---:|---:|---|
| LeWM-base | **19.68** | 87.89° | baseline，无 noise 训练 |
| LeWM-perframe-0to005-p1 | **0.16** | 0.68° | per-frame，降低 **123×** |
| SWM-base | **1.39** | 87.72° | baseline，无 noise 训练 |
| SWM-perframe-p1 | **0.01** | 0.64° | per-frame，降低 **139×** |

> 该表是 `predictor_sensitivity.json` 中 `std=0.08` 的汇总，对应 P0.3 文字中 “18.07→0.78、1.43→0.08” 的出处（均为 epoch_10, num_eval=300 新数据）。与上方 `std=0.005` 表口径不同：max std 下 baseline 的 drift 被噪声显著放大（LeWM 23×、SWM 18×），而 per-frame 模型仍保持在 <1 的低水平。这说明 per-frame noise training 不仅降低了低噪声下的 drift，更重要的是把 predictor 的 Lipschitz 常数压到足够低，使得大噪声输入也不会导致 rollout 发散。

**补充诊断指标**

*Noise robustness 补充（max std 条件下）*

TwoRoom：

| 模型 | CKA(clean, noisy) | pred_target/nn_cos_ratio |
|---|---:|---:|
| LeWM-base | **0.273** | 0.00016 |
| LeWM-fixed-std | 0.931 | 0.00009 |
| LeWM-perframe-p05 | **0.971** | **0.00003** |
| LeWM-perframe-p1 | **0.983** | **0.00003** |
| SWM-base | **0.385** | 0.00010 |
| SWM-fixed-std | 0.878 | 0.00056 |
| SWM-perframe-p05 | **0.977** | **0.00007** |
| SWM-perframe-p1 | **0.987** | **0.00007** |

PushT：

| 模型 | CKA(clean, noisy) | pred_target/nn_cos_ratio |
|---|---:|---:|
| LeWM-base | 0.554 | 0.00002 |
| LeWM-fixed-std | 0.876 | **0.00001** |
| LeWM-perframe-0to001-p1 | 0.909 | **0.00000** |
| LeWM-perframe-0to002-p1 | **0.971** | **0.00000** |
| LeWM-perframe-0to005-p1 | **0.993** | **0.00000** |
| SWM-base | 0.277 | 0.00001 |
| SWM-fixed-std | 0.886 | 0.00003 |
| SWM-perframe-0to001-p05 | **0.578** | 0.00002 |
| SWM-perframe-0to001-p1 | **0.520** | **0.00001** |
| SWM-perframe-0to002-p05 | 0.748 | **0.00001** |
| SWM-perframe-0to002-p1 | 0.894 | **0.00001** |

Reacher：

| 模型 | CKA(clean, noisy) | pred_target/nn_cos_ratio |
|---|---:|---:|
| LeWM-base | 0.309 | 0.00003 |
| LeWM-perframe-0to002-p05 | **0.998** | **0.00001** |
| LeWM-perframe-0to002-p1 | **0.999** | **0.00001** |
| LeWM-perframe-0to005-p05 | **0.999** | **0.00001** |
| LeWM-perframe-0to005-p1 | **0.999** | **0.00001** |
| SWM-base | 0.225 | 0.00004 |
| SWM-perframe-0to001-p1 | 0.432 | 0.00004 |
| SWM-perframe-0to002-p1 | **0.999** | **0.00002** |
| SWM-perframe-p1 | **1.000** | **0.00001** |

Cube：

| 模型 | CKA(clean, noisy) | pred_target/nn_cos_ratio |
|---|---:|---:|
| LeWM-base | 0.366 | 0.00000 |
| LeWM-perframe-0to002-p05 | **0.998** | **0.00000** |
| LeWM-perframe-0to002-p1 | **0.998** | **0.00000** |
| LeWM-perframe-0to005-p05 | **0.999** | **0.00000** |
| LeWM-perframe-0to005-p1 | **0.999** | **0.00000** |
| SWM-base | 0.181 | 0.00001 |
| SWM-perframe-0to001-p1 | 0.234 | 0.00001 |
| SWM-perframe-0to002-p1 | **0.997** | **0.00001** |
| SWM-perframe-p1 | **0.999** | **0.00001** |

> CKA：噪声 latent 与 clean latent 的 Centered Kernel Alignment。四个任务 baseline 均极低（TwoRoom LeWM 0.27 / SWM 0.38；PushT LeWM 0.55 / SWM 0.28；Reacher LeWM 0.31 / SWM 0.23；Cube LeWM 0.37 / SWM 0.18），noise training 后均跃升至 >0.87，说明 noise training 显著增强了 encoder 的表征稳定性。PushT/Reacher/Cube SWM-perframe-0to001 相对偏低（0.23–0.58），可能说明低强度 per-frame noise 在 SWM 上尚未完全稳定表征，需要更高强度或更长训练。

*Task resolution & action predictability*

TwoRoom：

| 模型 | trans_res_cos | trans_res_l2 | id_probe_r² | id_probe_r²_min | lidar_rank |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.556 | 0.729 | **+0.276** | +0.221 | 4.75 |
| LeWM-fixed-std | **0.252** | 0.500 | **−0.834** | **−1.699** | 3.91 |
| LeWM-perframe-p05 | 0.527 | 0.708 | +0.218 | +0.112 | 5.35 |
| LeWM-perframe-p1 | 0.549 | 0.713 | +0.136 | −0.044 | 4.33 |
| SWM-base | **0.734** | **0.857** | +0.263 | +0.219 | 8.27 |
| SWM-fixed-std | **0.185** | 0.430 | +0.234 | +0.161 | 4.92 |
| SWM-perframe-p05 | 0.657 | 0.810 | +0.255 | +0.212 | 8.58 |
| SWM-perframe-p1 | 0.634 | 0.796 | +0.251 | +0.207 | **10.47** |

PushT：

| 模型 | trans_res_cos | trans_res_l2 | id_probe_r² | id_probe_r²_min | lidar_rank |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.087 | 0.301 | +0.774 | +0.679 | 13.95 |
| LeWM-fixed-std | **0.059** | **0.248** | **+0.776** | **+0.682** | 12.34 |
| LeWM-perframe-0to001-p1 | 0.084 | 0.295 | +0.770 | +0.675 | 14.64 |
| LeWM-perframe-0to002-p1 | 0.087 | 0.299 | +0.769 | +0.667 | 13.89 |
| LeWM-perframe-0to005-p1 | 0.070 | 0.271 | +0.743 | +0.647 | 16.33 |
| SWM-base | 0.110 | 0.331 | +0.681 | +0.590 | 36.02 |
| SWM-fixed-std | 0.079 | 0.281 | +0.767 | +0.629 | 7.62 |
| SWM-perframe-0to001-p05 | 0.100 | 0.316 | +0.686 | +0.603 | 35.25 |
| SWM-perframe-0to001-p1 | **0.122** | **0.349** | +0.674 | +0.584 | 37.25 |
| SWM-perframe-0to002-p05 | 0.118 | 0.344 | +0.675 | +0.596 | 38.11 |
| SWM-perframe-0to002-p1 | **0.121** | 0.348 | +0.695 | +0.601 | 36.38 |

Reacher：

| 模型 | trans_res_cos | trans_res_l2 | id_probe_r² | id_probe_r²_min | lidar_rank |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.135 | 0.370 | +0.162 | +0.157 | 45.90 |
| LeWM-perframe-0to002-p05 | 0.151 | 0.387 | +0.222 | +0.156 | 8.85 |
| LeWM-perframe-0to002-p1 | 0.145 | 0.381 | +0.226 | +0.158 | 45.06 |
| LeWM-perframe-0to005-p05 | 0.141 | 0.375 | +0.216 | +0.154 | 9.00 |
| LeWM-perframe-0to005-p1 | 0.178 | 0.414 | +0.157 | +0.147 | 42.78 |
| SWM-base | 0.182 | 0.427 | +0.073 | +0.069 | 45.00 |
| SWM-perframe-0to001-p1 | 0.193 | 0.439 | +0.093 | +0.088 | 46.88 |
| SWM-perframe-0to002-p1 | 0.187 | 0.432 | +0.113 | +0.108 | 43.12 |
| SWM-perframe-p1 | 0.187 | 0.433 | +0.105 | +0.100 | 43.63 |

Cube：

| 模型 | trans_res_cos | trans_res_l2 | id_probe_r² | id_probe_r²_min | lidar_rank |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.235 | 0.485 | **+0.666** | +0.661 | 66.25 |
| LeWM-perframe-0to002-p05 | 0.275 | 0.522 | +0.585 | +0.580 | 13.60 |
| LeWM-perframe-0to002-p1 | 0.294 | 0.540 | +0.581 | +0.576 | 62.96 |
| LeWM-perframe-0to005-p05 | 0.331 | 0.569 | +0.569 | +0.564 | 11.82 |
| LeWM-perframe-0to005-p1 | 0.364 | 0.602 | +0.561 | +0.556 | 54.11 |
| SWM-base | 0.301 | 0.548 | +0.599 | +0.594 | 42.46 |
| SWM-perframe-0to001-p1 | 0.283 | 0.532 | +0.610 | +0.605 | 42.00 |
| SWM-perframe-0to002-p1 | 0.349 | 0.591 | +0.586 | +0.581 | 40.75 |
| SWM-perframe-p1 | **0.511** | **0.715** | +0.483 | +0.478 | 37.79 |

> **关键发现**：
> 1. `transition_resolution_ratio` 完美区分任务类型：TwoRoom cos 0.18–0.73（离散状态转移，相邻帧差异大），PushT/Reacher cos 0.06–0.19（连续控制，相邻帧极相似），Cube cos 0.24–0.51（中等离散度）。
> 2. `id_probe_r²` PushT/Cube (0.48–0.77) >> TwoRoom/Reacher (0.07–0.28)，说明 manipulation 任务的 latent 天然保留了更强的动作可预测性；TwoRoom LeWM-fixed-std 出现 **−0.834** 的异常负值，说明聚簇化严重破坏了动作信息。
> 3. `lidar_rank` PushT/Cube (11–66) > TwoRoom/Reacher (4–46)，与任务复杂度一致；SWM-perframe 在 PushT/Cube 上 lidar_rank 仍较高（37–42），但在 Reacher 上降至 43（与 LeWM 接近），说明球面 uniformity 的有效维度膨胀与任务所需的精细控制分辨率相关。
> 4. **Reacher/Cube 加入后，SWM-base 的 `trans_res_cos` 并不低于 LeWM**：Reacher SWM 0.182 vs LeWM 0.135，Cube SWM 0.301 vs LeWM 0.235。这与 PushT 上 SWM 分辨率受损的叙事不同，说明 SWM 的 resolution 问题并非全局性，而是任务依赖。

**Latent-noise sensitivity（直接对 `z` 注入高斯噪声，跳过 encoder）**

TwoRoom（std=0.08）：

| 模型 | hist_T8 | all_T8 | cost_slope_goal | noise_geometry |
|---|---:|---:|---:|---|
| LeWM-base | **5.81** | 5.79 | **2.08** | ambient |
| LeWM-fixed-std | 7.36 | 7.14 | 2.90 | ambient |
| LeWM-perframe-p05 | 4.81 | 5.11 | 2.02 | ambient |
| LeWM-perframe-p1 | 5.90 | 5.25 | 2.05 | ambient |
| SWM-base | **0.57** | 0.55 | **1.02** | tangent |
| SWM-fixed-std | 0.63 | 0.64 | 1.58 | tangent |
| SWM-perframe-p05 | 0.46 | 0.42 | 1.06 | tangent |
| SWM-perframe-p1 | 0.41 | 0.39 | 1.03 | tangent |

PushT：

| 模型 | hist_T8 | all_T8 | cost_slope_goal | noise_geometry |
|---|---:|---:|---:|---|
| LeWM-base | **12.05** | 11.59 | **3.63** | ambient |
| LeWM-fixed-std | 10.97 | 12.24 | 3.76 | ambient |
| LeWM-perframe-0to001-p1 | 12.05 | 11.59 | 3.65 | ambient |
| LeWM-perframe-0to002-p1 | 11.24 | 11.11 | 3.65 | ambient |
| LeWM-perframe-0to005-p1 | 10.05 | 10.24 | 3.78 | ambient |
| SWM-base | **0.77** | 0.78 | **1.39** | tangent |
| SWM-fixed-std | 0.67 | 0.65 | 1.76 | tangent |
| SWM-perframe-0to001-p05 | 0.77 | 0.78 | 1.67 | tangent |
| SWM-perframe-0to001-p1 | 0.74 | 0.77 | 1.65 | tangent |
| SWM-perframe-0to002-p05 | 0.78 | 0.81 | 1.62 | tangent |
| SWM-perframe-0to002-p1 | 0.81 | 0.77 | 1.61 | tangent |

Reacher（std=0.08）：

| 模型 | hist_T8 | all_T8 | cost_slope_goal | noise_geometry |
|---|---:|---:|---:|---|
| LeWM-base | **3.69** | 3.53 | **2.84** | ambient |
| LeWM-perframe-0to002-p05 | 3.24 | 3.49 | 0.003 | ambient |
| LeWM-perframe-0to002-p1 | 3.43 | 3.53 | 2.89 | ambient |
| LeWM-perframe-0to005-p05 | 2.99 | 3.10 | 0.003 | ambient |
| LeWM-perframe-0to005-p1 | 4.10 | 4.20 | 2.81 | ambient |
| SWM-base | **0.44** | 0.44 | **1.46** | tangent |
| SWM-perframe-0to001-p1 | 0.45 | 0.41 | 1.49 | tangent |
| SWM-perframe-0to002-p1 | 0.46 | 0.45 | 1.49 | tangent |
| SWM-perframe-p1 | 0.36 | 0.38 | 1.48 | tangent |

Cube（std=0.08）：

| 模型 | hist_T8 | all_T8 | cost_slope_goal | noise_geometry |
|---|---:|---:|---:|---|
| LeWM-base | **4.14** | 3.86 | **2.88** | ambient |
| LeWM-perframe-0to002-p05 | 3.46 | 3.40 | 0.003 | ambient |
| LeWM-perframe-0to002-p1 | 3.29 | 3.40 | 2.93 | ambient |
| LeWM-perframe-0to005-p05 | 3.31 | 3.44 | 0.003 | ambient |
| LeWM-perframe-0to005-p1 | 3.59 | 3.63 | 2.94 | ambient |
| SWM-base | **0.44** | 0.42 | **1.44** | tangent |
| SWM-perframe-0to001-p1 | 0.44 | 0.43 | 1.46 | tangent |
| SWM-perframe-0to002-p1 | 0.37 | 0.39 | 1.34 | tangent |
| SWM-perframe-p1 | 0.32 | 0.31 | 1.19 | tangent |

> **核心洞察**：
> 1. **SWM predictor 天生对 latent perturbation 稳定 8–10×。** TwoRoom 5.8→0.6，PushT 11.0→0.7，Reacher 3.7→0.4，Cube 4.1→0.4。这是因为 cosine/normalized predictor 内建了尺度不变性；LeWM 的 L2 predictor 对 latent scale 敏感。
> 2. **LeWM cost surface 对 goal latent 扰动敏感约 2×。** TwoRoom 2.1 vs 1.0，PushT 3.8 vs 1.4，Reacher 2.8 vs 1.5，Cube 2.9 vs 1.4。L2 cost 在 Euclidean space 的斜率更大，同样的 latent 偏移产生更大的 cost 变化。
> 3. **Per-frame pixel-noise training 不改善 predictor 的 latent-noise 鲁棒性。** 四个任务中 LeWM-perframe 的 T8 drift 与 baseline 几乎相同，SWM-perframe 甚至略升。这说明三层归因中，瓶颈在 **Layer 1 (encoder)**，而非 Layer 2 (predictor) 或 Layer 3 (cost surface)。noise training 的收益集中在 pixel→latent 映射的平滑化，而不是 predictor 本身的 Lipschitz 改善。
> 4. **`robust_radius_z` 已修复：goal scope 仍为 NaN，但 history scope 通过 rollout-drift fallback 获得有效值。** 修改内容：`summarize_latent_noise_geometry` 在 `target_to_nn_cos_ratio` 无法达到 threshold=1.0 时，回退到 `rollout_T8_l2_median / clean_nn_l2_median` 作为 ratio 进行插值。`run_full_diagnostics.py` 的 `_summarize` 也改为分别从 goal scope（cost slope）和 history scope（robust radius + slope）提取指标。当前 history scope `robust_radius_z`：TwoRoom 0.005–0.021，PushT 0.018–0.031，Reacher 0.024–0.043，Cube 0.047–0.065。与 eval 的相关性：TwoRoom −0.62（中等负相关），PushT +0.10（几乎不相关），Reacher +0.33（弱正相关），Cube +0.32（弱正相关）。四任务均不构成强预测信号，不如 `predictor_rollout_T8_l2` 或 `latent_cost_surface_slope_z`。

结果保存：`dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}/repr_analysis/latent_noise_diagnostics/`

**Planning signal probe（CEM cost 是否能区分 expert vs random）**

TwoRoom：

| 模型 | cost_margin_mean | expert_beats_best_random | expert_beats_random |
|---|---:|---:|---:|
| LeWM-base | 365.8 | 0.891 | 1.000 |
| LeWM-fixed-std | 379.4 | 0.875 | 1.000 |
| LeWM-perframe-p05 | 365.5 | 0.844 | 1.000 |
| LeWM-perframe-p1 | 365.5 | 0.844 | 1.000 |
| SWM-base | 0.798 | 0.922 | 0.984 |
| SWM-fixed-std | 0.891 | 0.922 | 1.000 |
| SWM-perframe-p05 | 0.641 | 0.906 | 0.984 |
| SWM-perframe-p1 | 0.637 | 0.875 | 0.984 |

PushT：

| 模型 | cost_margin_mean | expert_beats_best_random | expert_beats_random |
|---|---:|---:|---:|
| LeWM-base | 257.2 | 0.906 | 1.000 |
| LeWM-fixed-std | 257.4 | 0.891 | 1.000 |
| LeWM-perframe-0to001-p1 | 256.8 | 0.906 | 1.000 |
| LeWM-perframe-0to002-p1 | 257.0 | 0.906 | 1.000 |
| LeWM-perframe-0to005-p1 | 257.0 | 0.906 | 1.000 |
| SWM-base | 0.812 | 0.891 | 0.984 |
| SWM-fixed-std | 0.899 | 0.906 | 1.000 |
| SWM-perframe-0to001-p05 | 0.799 | 0.891 | 0.984 |
| SWM-perframe-0to001-p1 | 0.792 | 0.891 | 0.984 |
| SWM-perframe-0to002-p05 | 0.738 | 0.875 | 0.984 |
| SWM-perframe-0to002-p1 | 0.724 | 0.891 | 0.984 |

> **Cost 尺度差异**：LeWM 的 L2 cost margin 约 257–379（Euclidean 空间绝对距离），SWM 的 cosine cost margin 约 0.64–0.90（归一化空间，理论上界 2）。两者都满足 `expert_beats_best_random > 0.83`，说明 planning signal 在所有模型中都是有效的；差异不在 signal 有无，而在 cost 的绝对尺度和对 latent perturbation 的敏感度（latent-noise 中 SWM cost_slope 约 1.0–1.8，LeWM 约 2.0–3.8）。

**Action effect probe（action perturbation → pred shift）**

| 任务 | 结果 |
|---|---|
| TwoRoom | `mean_pred_shift_norm` 0.15–0.58，correlation 0.11–0.31，monotonicity >0.80 |
| PushT | 待批量跑；`run_planning_action_probe.py` 已改为调用 `encode_sequences(model, batch)` |

> 修复内容：`run_planning_action_probe.py` 原来只构造 `outputs={"pixels":..., "action":...}`，缺少 `"emb"` 键，导致 `get_embedding_space(outputs, rollout_space)` 报 `KeyError: 'emb'`。改为调用 `encode_sequences(model, batch)` 后正常。
>
> 批量命令：
> ```bash
> export STABLEWM_HOME=dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}
> python run_planning_action_probe.py
> ```

**PushT 与 TwoRoom 的关键差异**

1. **LeWM-fixed-std 在 PushT 上是 robust，在 TwoRoom 上是 fragile,clustered。**  
   这说明 LeWM 的聚簇化是**任务依赖**的：TwoRoom 低维状态空间容易被压缩成紧凑等价类，PushT 高维连续状态空间难以被简单聚簇。

2. **SWM-fixed-std 在 PushT 上仍然是 fragile,high_angle_gain。**  
   这说明 SWM 的 fixed-std 高角向增益是**结构性风险**（球面 + uniformity + 固定 std 的组合），与任务不完全绑定。PushT 表里它没有被规则标成 `clustered`，但 `clean_nn_cos_dist=0.0664` 明显小于 per-frame SWM 的 0.26–0.28，仍显示出强压缩倾向。

3. **SWM-perframe-0to001 在 PushT 上被评为 robust，0to002 是 balanced。**  
   在 PushT 上，0to001 的 noise 强度已经足够产生 robust geometry（radius≈0.07），而 0to002 的 robust_radius=NaN（更 robust 但 clean eval 从 87.3 降至 81.3）。这与 TwoRoom 上 0to005 才达到 balanced 形成对比，说明 PushT 的"最优 noise 强度"确实更低。

结果保存：`dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}/repr_analysis/p03_diagnostics/`

#### P0.4 相关性分析（跨任务对比结论）

> 完整自动化相关性表见 P0.7（来自 `diagnostic_correlation.py`，n=8 / n=11，含 95% bootstrap CI）。本节只给跨任务对比与高层结论，不重复列指标。
>
> **统计警告（2026-05-05 已修正）**：下表已使用 `diagnostic_correlation.py` average-tie rank 版本重算（Spearman ties 修正）。四任务 n 分别为 TwoRoom=8、PushT=11、Reacher=10、Cube=10。CI 来自 1000 次 bootstrap。

**核心：诊断指标的任务特异性**

| 指标 | TwoRoom (r / ρ) | PushT (r / ρ) | Reacher (r / ρ) | Cube (r / ρ) | 含义 |
|---|---:|---:|---:|---:|---|
| `clean_nn_cos_dist_median` ↔ eval | **−0.80 / −0.91** | +0.64 / +0.02 | −0.10 / −0.26 | **+0.84 / +0.73** | **方向反转**：TwoRoom 负（聚簇化好），Cube 正（分散好），PushT/Reacher 几乎不相关 |
| `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | +0.44 / +0.24 | **−0.83 / −0.59** | **−0.86 / −0.66** | +0.44 / +0.26 | PushT/Reacher **最强信号**；Cube 弱正相关（不显著） |
| `predictor_rollout_T8_l2` ↔ eval | +0.37 / +0.67 | +0.15 / +0.46 | −0.62 / −0.39 | +0.33 / +0.40 | TwoRoom 正相关；Reacher 方向反转；PushT/Cube 弱/不显著 |
| `clean_effective_rank` ↔ eval | −0.62 / −0.60 | +0.77 / +0.49 | +0.04 / +0.31 | −0.22 / +0.10 | **符号反转**：TwoRoom 要低维，PushT 要高维，Cube/Reacher 不显著 |
| `lidar_rank` ↔ eval | **−0.93 / −0.81** | +0.16 / −0.02 | −0.22 / −0.36 | +0.06 / +0.17 | TwoRoom 强负相关；其余三任务均弱 |
| `noise_angle_slope_deg_per_std` ↔ eval | +0.54 / +0.67 | −0.82 / −0.35 | **−0.91 / −0.48** | +0.69 / +0.75 | TwoRoom/Cube 可容忍高角向增益；PushT/Reacher 惩罚高角向增益 |
| `cka_linear_at_max_std` ↔ eval | +0.02 / −0.31 | +0.06 / +0.31 | +0.95 / +0.55 | **−0.67 / −0.78** | **方向反转**：Reacher 正（CKA 越高越好），Cube 负（CKA 越低越好，SWM 天生低 CKA 但 eval 高） |
| `latent_rollout_angle_slope_per_std_z` ↔ eval | +0.22 / +0.12 | +0.08 / +0.16 | −0.08 / −0.28 | +0.78 / +0.72 | Cube **最强信号**：latent-noise 下 rollout 角向 slope 越大 eval 越高 |
| `latent_cost_surface_slope_z` ↔ eval | +0.48 / +0.24 | +0.46 / +0.68 | +0.13 / +0.34 | −0.32 / −0.52 | PushT 正（slope 越大越好），Cube 负（slope 越小越好） |

> **重大修正记录**（commit 8605bf5 → bf79a80 → 4ce4931 → 2026-05-05 ties 修正）：PushT 相关性经历四次修正。前三次见原记录。第四次（2026-05-05）将 `diagnostic_correlation.py` 改为 average-tie ranks 后，PushT `predictor_target_to_nn_cos_ratio` 从 ρ=−0.79 降至 **−0.59**（CI 仍宽），`predictor_rollout_T8_l2` 从 +0.64 降至 **+0.46**。TwoRoom 核心指标 `clean_nn_cos_dist` 保持 −0.91 不变。Cube 新数据揭示 `latent_rollout_angle_slope_per_std_z`（ρ=+0.83）和 `latent_cost_surface_slope_z`（ρ=−0.79）为最强信号。

结论：
- **TwoRoom 瓶颈**：encoder geometry（聚簇化 / 维度控制）+ predictor 稳定性。`clean_nn_cos_dist` ρ=−0.91 是最稳健的跨版本信号。
- **PushT 瓶颈**：predictor 稳定性（target shift 控制）+ latent cost surface。`predictor_target_to_nn_cos_ratio` ρ=−0.59 仍是最强信号，但 ties 修正后低于 ≥0.7 强相关阈值，降为**中等相关**。
- **Reacher 瓶颈**：相关性整体较弱（|ρ| 最高 0.66），主要信号为 `predictor_target_to_nn_cos_ratio`（ρ=−0.66）和 `cka_linear`（ρ=+0.55）。原因可能是 Reacher 任务难度较低，per-frame 训练把 base 从 58.7 拉到 72–83，压缩了模型间 variance，导致诊断指标区分度下降。
- **Cube 瓶颈**：**encoder 分散度 + noise 角向增益 + CKA**。加入 LeWM-base（n=11）后，`cka_linear_at_max_std` ρ=−0.78 跃升为最强信号，`noise_angle_slope_deg_per_std` ρ=+0.75 和 `clean_nn_cos_dist` ρ=+0.73 紧随其后。`clean_nn_cos_dist` 方向与 TwoRoom **相反**（Cube 正相关），说明 manipulation 任务需要保留足够的 goal-state 区分度，过度聚簇化会丢失 block pose 信息。`latent_cost_surface_slope_z` 降至 ρ=−0.52（CI 变宽），说明加入 base 锚点后该信号的稳健性不足。
- **跨任务通用指标**：**无**。`predictor_target_to_nn_cos_ratio_at_max_std` 在 PushT/Reacher 为最强信号，但 Cube 弱且不显著（ρ=+0.26）；`predictor_rollout_T8_l2` 在 TwoRoom 正相关、Reacher 负相关。Paper 主指标必须按任务选择，不可全局套用。
- **不通用指标**：`lidar_rank`、`clean_effective_rank` 四任务均弱；`clean_nn_cos_dist`、`noise_angle_slope`、`cka_linear` 任务依赖性强；`latent_rollout_angle_slope` 在 Cube 上强（ρ=+0.72），但其他任务未验证。

> **解释约束**：`predictor_rollout_T8_l2` 在 TwoRoom / PushT 上出现正相关，不能直接解释为“drift 越大越好”。这更可能混合了模型族、latent 尺度、noise training 强度与 task difficulty confounder。Paper 主指标应优先使用方向稳定且归一化明确的 `predictor_target_to_nn_cos_ratio_at_max_std`；rollout drift 只作为辅助或机制图，必须经 P0.6 holdout 验证。

clean eval 与 noise robustness 在 TwoRoom 不是简单正相关：SWM fixed-std 走"聚簇化 clean bonus / noise fragile"路径，LeWM per-frame 走"平滑且 clean 不差"路径——两条路径必须用诊断指标分开归因（详 §4.2）。

**局限**：
1. PushT 中 `noise_robust_radius_std` 仅 n=6（per-frame 模型 radius>0.08 censored），Cube/Reacher 已补齐（Cube n=11、Reacher n=10）但仍有部分模型 radius>0.08 被 censor。
2. Cube 缺少 LeWM-base（74.0% eval 已完成但 diagnostics 缺失），导致 Cube 相关性仅覆盖 SWM vs LeWM-perframe 对比，缺少 base 锚点。
3. 四任务均无 ≥3 seeds 的均值/标准差，当前均为单 seed 点估计。

**图表**：`p0_correlation_{tworoom,pusht,reacher,cube}.png`、`predictor_drift_eval_correlation.png`、`noise_angle_curve_goal.png`、`noise_ratio_curve_goal.png`、`geometry_tradeoff_goal.png`。  
保存路径：`dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht,reacher,cube}/repr_analysis/p03_diagnostics/`。

![TwoRoom P0 诊断指标与 eval 相关性](assets/diagnostics/p0_correlation_tworoom.png)
![PushT P0 诊断指标与 eval 相关性](assets/diagnostics/p0_correlation_pusht.png)
![Predictor Drift 与 Eval 相关性（双任务）](assets/diagnostics/predictor_drift_eval_correlation.png)
![Noise Angle 曲线](assets/diagnostics/noise_angle_curve_goal.png)
![Noise Ratio 曲线](assets/diagnostics/noise_ratio_curve_goal.png)
![Geometry Tradeoff 散点](assets/diagnostics/geometry_tradeoff_goal.png)

#### P0.5 决策标准（按实际数据评估，2026-05-05 ties 修正后）

> 以下 `|ρ|` 已使用 `diagnostic_correlation.py` average-tie rank 版本重算。判定阈值：≥0.7 强相关，0.4–0.7 中等，<0.4 弱。

| 任务 | 指标 | Spearman \|ρ\| | n | 判定 | 行动 |
|---|---:|---:|---:|---|---|
| TwoRoom | `clean_nn_cos_dist_median` ↔ eval | **0.905** | 8 | ≥ 0.7 强相关 | **主指标**：encoder 聚簇化/压缩是 TwoRoom clean eval 的主要解释机制 |
| TwoRoom | `transition_resolution_ratio_cos` ↔ eval | **0.881** | 8 | ≥ 0.7 强相关 | **主指标**：transition 分辨率越高，eval 越低 |
| TwoRoom | `noise_robust_radius_std` ↔ eval | **1.000** | 4 | ≥ 0.7 强相关 (n=4) | **待验证**：样本量过小，n=4 不可靠 |
| TwoRoom | `latent_predictor_rollout_T8_l2_history` ↔ eval | 0.738 | 8 | 0.4–0.7 中等 | **辅助**：latent-noise predictor drift 与 eval 正相关 |
| TwoRoom | `predictor_rollout_T8_l2` ↔ eval | 0.667 | 8 | 0.4–0.7 中等 | **辅助**：input-space predictor drift 与 eval 正相关 |
| PushT | `latent_cost_surface_slope_z` ↔ eval | **0.683** | 11 | ≥ 0.7 强相关 | **主指标**：latent cost surface slope 与 eval 正相关 |
| PushT | `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | 0.592 | 11 | 0.4–0.7 中等 | **候选主指标**：target shift 控制与 eval 负相关，ties 修正后低于 ≥0.7 |
| PushT | `noise_robust_radius_std` ↔ eval | 0.543 | 6 | 0.4–0.7 中等 (n=6) | **待验证**：n=6，方向与 TwoRoom 相反 |
| PushT | `predictor_rollout_T8_l2` ↔ eval | 0.460 | 11 | 0.4–0.7 中等 | **辅助**：input-space predictor drift 与 eval 正相关 |
| Reacher | `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | 0.657 | 10 | 0.4–0.7 中等 | **候选主指标**：PushT/Reacher 共享信号，但 Reacher CI 宽 |
| Reacher | `cka_linear_at_max_std` ↔ eval | 0.546 | 10 | 0.4–0.7 中等 | **辅助**：noise 下 latent CKA 与 eval 正相关 |
| Cube | `cka_linear_at_max_std` ↔ eval | **0.781** | 11 | ≥ 0.7 强相关 | **主指标**：noise 下 CKA 越低，eval 越高（SWM 天生低 CKA 但 eval 高） |
| Cube | `noise_angle_slope_deg_per_std` ↔ eval | **0.753** | 11 | ≥ 0.7 强相关 | **主指标**：noise 角向 slope 越大，eval 越高 |
| Cube | `clean_nn_cos_dist_median` ↔ eval | **0.731** | 11 | ≥ 0.7 强相关 | **主指标**：方向与 TwoRoom 相反——Cube 上 encoder 分散度越高 eval 越高 |
| Cube | `id_probe_r2` ↔ eval | **0.722** | 11 | ≥ 0.7 强相关 | **主指标**：ID probe R² 越高，eval 越高 |
| Cube | `latent_rollout_angle_slope_per_std_z` ↔ eval | **0.722** | 11 | ≥ 0.7 强相关 | **主指标**：latent-noise 下 rollout 角向 slope 与 eval 正相关 |
| Cube | `latent_cost_surface_slope_z` ↔ eval | 0.516 | 11 | 0.4–0.7 中等 | **辅助**：latent cost surface slope 与 eval 负相关（加入 base 后 CI 变宽） |

下一步行动（按上表派生）：
1. **按任务选主指标**：TwoRoom 用 `clean_nn_cos_dist_median`；PushT 用 `latent_cost_surface_slope_z` + `predictor_target_to_nn_cos_ratio_at_max_std`；Reacher 用 `predictor_target_to_nn_cos_ratio_at_max_std`；Cube 用 `latent_rollout_angle_slope_per_std_z` + `latent_cost_surface_slope_z`。
2. **无跨任务通用指标**。`predictor_target_to_nn_cos_ratio` 在 PushT/Reacher 一致，但 Cube 方向反转；`predictor_rollout_T8_l2` 在 TwoRoom 正相关、Reacher 负相关。Paper 必须按任务分节呈现诊断指标。
3. 对 SWM 做 predictor 结构 ablation（P3），验证 target shift 控制是否随 predictor depth/normalization 变化。

#### P0.6 Active Validation：从相关到预测

相关性有同族 confounder（同一训法的 ckpt 共享偏置），需要盲测：

1. 选 1–2 个 holdout checkpoint（建议在 `Cube` / `Reacher` 上新训一组 SWM 和 LeWM noise-aware，与 P0.3 训练分布不同）。
2. **只**用 P0.1–P0.3 的诊断输出，给出 eval drop 的预测分桶（low / mid / high）+ `recommendation`。
3. 真实跑 eval，与预测分桶对照。
4. 命中标准：分桶命中 ≥ 80% → 诊断工具可独立写一节；< 60% → 回到 P0.5 弱相关分支。

#### P0.7 输出与维护

- 新增 `tools/repr_analysis/diagnostic_correlation.py`：自动收集 N×T 表，跑 Spearman + bootstrap，落 csv / png。
- 在 `experiments.md` 维护一个 "diagnostic ↔ eval" 主表，每加一个 checkpoint 自动 append。
- 论文图：(a) noise curve 对比图（已有 `plot_noise_curves`），(b) robustness-resolution 散点（已有 `plot_geometry_tradeoff`），(c) 相关性热图（已生成 `diagnostic_correlation.png`）。

![TwoRoom 诊断相关性热图](assets/diagnostics/diagnostic_correlation_tworoom.png)
![PushT 诊断相关性热图](assets/diagnostics/diagnostic_correlation_pusht.png)
![Reacher 诊断相关性热图](assets/diagnostics/diagnostic_correlation_reacher.png)
![Cube 诊断相关性热图](assets/diagnostics/diagnostic_correlation_cube.png)

**自动化相关性结果（`diagnostic_correlation.py` 输出，2026-05-05 average-tie rank 修正版）**

> 以下四任务表格均已使用 `diagnostic_correlation.py` average-tie ranks 重算。原始 CSV / PNG / summary.json 保存在各自任务的 `repr_analysis/p03_diagnostics/` 下。

TwoRoom（n=8，baselines 已补齐）：

| 指标 | Pearson r | Spearman ρ | 95% CI | 解释 |
|---|---:|---:|---|---|
| `clean_nn_cos_dist_median` ↔ eval | −0.800 | **−0.905** | [−1.000, −0.279] | **最强预测指标**：encoder 聚簇化越紧，clean eval 越高 |
| `noise_robust_radius_std` ↔ eval | −0.963 | **−1.000** | [−1.000, −1.000] | n=4，noise robust radius 越小，eval 越低 |
| `transition_resolution_ratio_cos` ↔ eval | −0.775 | **−0.881** | [−1.000, −0.240] | transition 分辨率（cos 相似度）越高，eval 越低 |
| `transition_resolution_ratio_l2` ↔ eval | −0.795 | **−0.881** | [−1.000, −0.240] | transition 分辨率（L2 相似度）越高，eval 越低 |
| `lidar_rank` ↔ eval | −0.925 | **−0.810** | [−1.000, −0.215] | 有效维度越多，eval 越低 |
| `latent_predictor_rollout_T8_l2_history` ↔ eval | +0.591 | **+0.738** | [−0.139, +1.000] | latent-noise 下 history scope predictor drift 与 eval 强正相关 |
| `latent_rollout_l2_slope_per_std_z` ↔ eval | +0.599 | **+0.738** | [−0.063, +1.000] | latent rollout L2 slope 与 eval 正相关 |
| `noise_angle_slope_deg_per_std` ↔ eval | +0.538 | **+0.667** | [−0.291, +1.000] | noise 角度 slope 越大，eval 越高 |
| `predictor_rollout_T8_l2` ↔ eval | +0.371 | **+0.667** | [−0.216, +1.000] | input-space predictor drift 与 eval 正相关 |
| `id_probe_r2` ↔ eval | −0.425 | **−0.619** | [−0.975, +0.264] | ID linear probe R² 越高，eval 越低 |
| `id_probe_r2_min` ↔ eval | −0.442 | **−0.619** | [−0.975, +0.243] | ID probe min R² 越高，eval 越低 |
| `latent_robust_radius_z` ↔ eval | −0.645 | **−0.619** | [−1.000, +0.140] | 中等负相关，但 CI 仍宽 |
| `clean_effective_rank` ↔ eval | −0.624 | **−0.595** | [−1.000, +0.291] | 有效维度越多，eval 越低 |

PushT（n=11，baselines 已补齐，SWM-fixed-std 真实 eval=61.8）：

| 指标 | Pearson r | Spearman ρ | 95% CI | 解释 |
|---|---:|---:|---|---|
| `latent_cost_surface_slope_z` ↔ eval | +0.455 | **+0.683** | [+0.127, +0.962] | **最强信号**：latent cost surface slope 与 eval 正相关 |
| `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | −0.830 | **−0.592** | [−0.942, +0.075] | target shift 控制与 eval 负相关（ties 修正后降至中等） |
| `noise_robust_radius_std` ↔ eval | +0.777 | **+0.543** | [−0.742, +1.000] | n=6，noise robust radius 越大，eval 越高 |
| `latent_rollout_l2_slope_per_std_z` ↔ eval | +0.442 | **+0.510** | [−0.174, +0.926] | latent rollout L2 slope 与 eval 正相关 |
| `predictor_rollout_T8_l2` ↔ eval | +0.151 | **+0.460** | [−0.159, +0.858] | input-space predictor drift 与 eval 正相关 |
| `clean_effective_rank` ↔ eval | +0.767 | **+0.488** | [−0.289, +0.925] | 有效维度越多，eval 越高（与 TwoRoom 方向相反） |
| `latent_predictor_rollout_T8_l2_history` ↔ eval | +0.457 | **+0.447** | [−0.298, +1.000] | latent-noise predictor drift 与 eval 正相关 |
| `noise_angle_slope_deg_per_std` ↔ eval | −0.820 | **−0.346** | [−0.813, +0.286] | noise 角度 slope 与 eval 负相关 |
| `cka_linear_at_max_std` ↔ eval | +0.060 | **+0.314** | [−0.452, +0.823] | 弱正相关 |
| `latent_robust_radius_z` ↔ eval | +0.626 | **+0.105** | [−0.860, +0.791] | 几乎不相关 |
| `clean_nn_cos_dist_median` ↔ eval | +0.642 | **+0.018** | [−0.804, +0.709] | **几乎不相关**：聚簇化不解释 PushT eval |
| `lidar_rank` ↔ eval | +0.157 | **−0.023** | [−0.754, +0.675] | **几乎不相关**：有效维度不解释 PushT eval |

Reacher（n=10，全 epoch_9，num_eval=150）：

| 指标 | Pearson r | Spearman ρ | 95% CI | 解释 |
|---|---:|---:|---|---|
| `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | −0.863 | **−0.657** | [−0.986, +0.103] | **最强信号**：predictor 稳定性越好，eval 越高 |
| `cka_linear_at_max_std` ↔ eval | +0.952 | **+0.546** | [−0.297, +0.948] | noise 下 latent CKA 越高，eval 越高 |
| `noise_angle_slope_deg_per_std` ↔ eval | −0.914 | **−0.485** | [−0.922, +0.308] | noise 角度 slope 与 eval 负相关 |
| `lidar_rank` ↔ eval | −0.218 | **−0.362** | [−0.839, +0.393] | 弱负相关 |
| `latent_cost_surface_slope_z` ↔ eval | +0.126 | **+0.337** | [−0.597, +0.898] | 弱正相关 |
| `clean_nn_cos_dist_median` ↔ eval | −0.104 | **−0.264** | [−0.897, +0.458] | 弱负相关 |
| `predictor_rollout_T8_l2` ↔ eval | −0.616 | **−0.387** | [−0.870, +0.520] | 负相关（与 TwoRoom/PushT 方向相反） |
| `clean_effective_rank` ↔ eval | +0.037 | **+0.313** | [−0.415, +0.773] | 几乎不相关 |
| `id_probe_r2` ↔ eval | +0.371 | **+0.252** | [−0.686, +0.817] | 弱正相关 |
| `transition_resolution_ratio_cos` ↔ eval | +0.049 | **−0.080** | [−0.870, +0.746] | 几乎不相关 |

> Reacher 整体相关性弱于 TwoRoom/PushT/Cube，|ρ| 最高 0.66。可能原因：Reacher 任务本身难度较低（base 仅 58.67），per-frame 训练把分数拉到 72–83，压缩了模型间 variance，导致诊断指标区分度下降。

Cube（n=11，SWM epoch_10 num_eval=300；LeWM base + 4 per-frame epoch_9/10 num_eval=150/300）：

| 指标 | Pearson r | Spearman ρ | 95% CI | 解释 |
|---|---:|---:|---|---|
| `cka_linear_at_max_std` ↔ eval | −0.673 | **−0.781** | [−0.971, −0.312] | **最强信号**：noise 下 CKA 越低，eval 越高（SWM 天生低 CKA 但 eval 高） |
| `noise_angle_slope_deg_per_std` ↔ eval | +0.691 | **+0.753** | [+0.282, +1.000] | **强正相关**：noise 角向 slope 越大，eval 越高 |
| `clean_nn_cos_dist_median` ↔ eval | +0.838 | **+0.731** | [+0.314, +0.936] | **强正相关**：encoder 分散度越大，eval 越高（与 TwoRoom 方向相反） |
| `id_probe_r2` ↔ eval | +0.565 | **+0.722** | [+0.308, +0.940] | **强正相关**：ID probe R² 越高，eval 越高 |
| `latent_rollout_angle_slope_per_std_z` ↔ eval | +0.783 | **+0.722** | [+0.202, +0.971] | **强正相关**：latent-noise 下 rollout 角向 slope 与 eval 正相关 |
| `latent_cost_surface_slope_z` ↔ eval | −0.322 | **−0.516** | [−0.943, +0.183] | 中等负相关（加入 LeWM-base 后 CI 变宽，稳健性下降） |
| `latent_robust_radius_z` ↔ eval | +0.401 | **+0.411** | [−0.316, +0.898] | 弱正相关 |
| `predictor_rollout_T8_l2` ↔ eval | +0.332 | **+0.397** | [−0.411, +0.885] | 弱正相关 |
| `transition_resolution_ratio_cos` ↔ eval | −0.226 | **−0.279** | [−0.863, +0.409] | 弱负相关 |
| `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | +0.435 | **+0.265** | [−0.528, +0.903] | 弱正相关，不显著 |
| `lidar_rank` ↔ eval | +0.056 | **+0.174** | [−0.554, +0.848] | 几乎不相关 |
| `clean_effective_rank` ↔ eval | −0.221 | **+0.096** | [−0.700, +0.817] | 几乎不相关 |

> Cube 加入 LeWM-base（eval=74.0%）后，诊断-评估相关性发生显著重构：n=10 时 `latent_cost_surface_slope_z`（ρ=−0.79）和 `latent_rollout_angle_slope`（ρ=+0.83）主导；n=11 后 `cka_linear_at_max_std`（ρ=−0.78）、`noise_angle_slope`（ρ=+0.75）、`clean_nn_cos_dist`（ρ=+0.73）和 `id_probe_r2`（ρ=+0.72）共同构成强信号群。`latent_cost_surface_slope_z` 降至 ρ=−0.52（CI 变宽），说明该指标对 base 锚点敏感。这提示 Cube 的瓶颈是**多因素联合**：encoder 分散度（保留 block pose 信息）+ noise 角向增益（表征平滑性）+ CKA（SWM 归一化带来的低 CKA 与高 eval 的耦合）+ ID probe（动作可预测性）。Paper 写作时不应过度依赖单一指标，而应呈现指标群的联合预测力。

保存路径：`dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht,reacher,cube}/repr_analysis/p03_diagnostics/diagnostic_correlation.{csv,png,summary.json}`

### P1：Noise-Aware Training（结论见 §4）

P1.1–P1.3 完整 eval / geometry 数据见 §4；机制结论（fixed-std → 聚簇化、per-frame → 平滑化、两者互斥）见 §4.2；与 P0 诊断指标的对应关系见 §6 P0.3 / P0.4。本节不重复。

P1.4（可选，未做）：补扫 SWM `std_max ∈ {0.01, 0.02, 0.03, 0.08}` 以画完整 clean-noise 曲线。当前 TwoRoom 仅 0.05，PushT 仅 0.01 / 0.02 / 0.05。

### P2/P5：Mechanism Attribution（Cost Surface + Latent-Noise）

目标：把 SWM noise failure 拆成三层归因：

| 层 | 工具 | 问题 |
|---|---|---|
| Encoder | `noise_sensitivity.py` | pixel noise 是否先把 latent goal/history 编到错误区域 |
| Predictor | `predictor_sensitivity.py` / `latent_noise_sensitivity.py` | noisy history 或 noisy z 是否被 predictor 放大 |
| Cost | eval cost swap / `cost_surface_slope_z` | planning cost 是否在大角度或 latent perturbation 下饱和/失去梯度 |

#### P2.1 Eval-only cost swap

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

- `raw + mse` 只带来小幅回升（+6），没有接近 clean SWM（69.67，epoch_10, num_eval=300）或 LeWM std=0.03（90）。
- cost saturation 可能贡献了一部分损害，但不是主因。
- 主导失败仍然是 upstream encoder / noisy goal embedding corruption：目标 latent 已经偏到错误区域，eval-only cost swap 无法修复。

#### P2.2 Latent-noise probing（数据见 §6 P0.3 latent-noise 表，相关性见 P0.4 / P0.7）

P5 原本单列为一个诊断实验；这里并入 P2，因为它和 cost swap 都是在做 encoder / predictor / cost 的机制解耦。区别是：P2.1 从 eval 端替换 cost，P2.2 直接在 encoded `z` 上加噪，跳过 encoder。

**当前结论**（按 P0.4 决策标准评估）：
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

### P3：Encoder Sensitivity 拆解

目标：定位 SWM angular sensitivity 的来源。

先做轻量训练或已有 checkpoint 的 `noise_sensitivity`，不急着完整 eval。

| 变体 | 改动 | 观察指标 |
|---|---|---|
| SWM-noBN | `norm_fn=none` | 是否降低 noise angle；是否 collapse |
| SWM-LN | `norm_fn=layernorm` | BN 是否是放大源 |
| SWM-dim128 | `embed_dim=128` | 维度对角向扰动的影响 |
| SWM-dim192 | `embed_dim=192` | 与 LeWM dim 对齐 |

风险：BN 是当前逃离 collapse 的关键，去 BN 可能直接塌缩。因此 P3 主要是机制诊断，不作为主优化路线。

### P4：Adaptive Resolution 方法

目标：避免“每个任务手调一套 noise recipe”。

当前还没有真正的 adaptive training objective；已有的是诊断和可视化闭环，用来判断某个 checkpoint 更偏向 robustness、precision，还是过度 clustered。P4 的实现应建立在 P0/P1 的诊断结果上，而不是继续盲目扫配方。

最小可行方向：

```text
L = pred_loss
  + regularizer
  + lambda_noise * consistency(enc(x), enc(noisy_x))
  + lambda_guard * preserve_action/transition_resolution
```

核心不是简单加 noise consistency，而是加 guardrail，避免 PushT 这类任务被过度聚簇。

可选 guardrail：

| Guardrail | 目的 |
|---|---|
| action-effect preservation | 保留 action 对 latent transition 的影响 |
| transition distance preservation | 防止相邻但关键的状态差异被抹掉 |
| adaptive lambda | 根据 transition/action sensitivity 自动调 noise consistency 强度 |
| vMF concentration `kappa` | 长线方案，让模型显式表示局部分辨率/不确定性 |

短期优先级：先做 noise consistency + transition/action guardrail；vMF 作为后续 V1。

## 7. Prior Art：诊断指标的归属与 Gap

为了把诊断工具放到论文里要严谨，本节把 P0 用到的每一个指标对照已有工作，标清楚"哪个是借用 / 哪个是组合 / 哪个是真新东西"。这影响 §8 的贡献条目和 Related Work 该怎么写。

### 7.1 诊断指标速查：含义、来源、代码位置

本表替代原附录里的指标手册，只保留写论文和人工审查最需要的信息。具体字段的完整 CSV 仍以 `diagnostics_summary.json` 和各模块输出为准。

| 指标 / 维度 | 简洁含义 | 参考来源与归属 | 代码逻辑位置 |
|---|---|---|---|
| `noise_l2`, `noise_cos_dist`, `noise_angle_deg` | **测什么**：同一 obs 加 pixel noise 前后 encoder 输出的逐点位移（L2 / 余弦距离 / 角度）。**用来分析**：encoder 的局部 Lipschitz / 输入敏感度；球面模型半径被归一化，重点看 angle 反映角向敏感度 | empirical Jacobian / local Lipschitz probe；Hoffman 2019, Virmaux & Scaman 2018 | `tools/repr_analysis/noise_sensitivity.py::_shift_metrics` |
| `clean_nn_cos_dist`, `clean_nn_l2` | **测什么**：clean latent 在数据集里的最近邻距离，刻画 latent 局部邻域尺度。**用来分析**：作为"噪声位移是否跨过邻域"的分母；也用来判断 latent 是否过度聚簇（NN 极小）或扩散 | KNN-OOD / distance-aware feature primitive；Sun 2022, Liu 2020 | `noise_sensitivity.py::_pairwise_reference`; `predictor_sensitivity.py::_clean_nn_dist` |
| `clean_pair_cos_dist`, `clean_pair_l2` | **测什么**：所有 clean latent 的两两平均距离，刻画整体分布扩散程度。**用来分析**：检测 uniformity（球面均匀分布）和 anisotropy（各向异性 / 维度坍缩），与 NN 距离一起给出"全局尺度 vs 局部尺度" | Wang & Isola uniformity；Ethayarajh anisotropy | `noise_sensitivity.py::_pairwise_reference` |
| `clean_effective_rank` | **测什么**：latent 协方差特征值谱的熵化有效维度（RankMe）。**用来分析**：区分 dimensional collapse（rank 极低）、clustered（中低）、balanced（接近 embed dim）三种几何形态，是诊断 collapse 的核心标量 | RankMe / matrix entropy；Garrido 2023；LiDAR 相关但不主张新颖性 | `analyze_repr.effective_rank`; `task_resolution.py` roll-up |
| `cka_linear_clean_vs_noisy` | **测什么**：clean 与 noisy latent 的线性 CKA 子空间相似度。**用来分析**：从子空间（而非逐点）层面看噪声是否破坏整体表征结构，补充 noise_l2/angle 的逐点视角 | CKA；Kornblith 2019 | `noise_sensitivity.py::_linear_cka` |
| `noise_to_nn_*_ratio` | **测什么**：encoder shift 除以 clean NN 距离的无量纲比值。**用来分析**：ratio ≥ 1 表示噪声把 latent 推出了原本的局部邻域，等价于"邻居身份被破坏"；跨模型可比，是 planning robustness 的核心 ratio 指标 | composite 指标，可作为 planning-latent robustness ratio 主张 | `noise_sensitivity.py::analyze_model_noise` |
| `robust_radius_std`, `first_high_risk_std` | **测什么**：noise std 扫描中 `noise_to_nn_cos_ratio` 跨过 1 的 std（前者插值得到连续值，后者首个 ≥1 的离散 std）。**用来分析**：planning-latent 的经验"鲁棒半径"，给出"多大输入噪声会破坏邻域结构"的单一阈值 | randomized smoothing 的 empirical planning-latent 版本；Cohen 2019 是来源，不是同一 certified setting | `noise_sensitivity.py::summarize_noise_geometry` |
| `noise_angle_slope_deg_per_std`, `noise_ratio_slope_per_std` | **测什么**：小 std 区段内 angle / ratio 对 std 的斜率。**用来分析**：线性化的 encoder 局部 Lipschitz 估计，便于对比模型在弱扰动区的敏感度，不被大 std 的非线性饱和干扰 | local Lipschitz / spectral norm 思路的球面诊断版本 | `noise_sensitivity.py::_near_zero_slope` |
| `geometry_flag`, `recommendation` | **测什么**：综合 radius / angle gain / cosine NN distance / effective rank 给出的几何标签与文字建议。**用来分析**：仅用于人工筛选 ckpt 与生成自动摘要，不参与论文 novelty 主张；绝对 L2 不进入规则 | 工程规则；不是论文 novelty 单独主张；绝对 L2 不参与 flag | `noise_sensitivity.py::_geometry_flags`; `_recommendation` |
| `predictor_target_shift`, `target_to_nn_*_ratio` | **测什么**：noisy history 经 predictor 输出的 single-step target latent 位移（绝对值 / 与 NN 尺度比）。**用来分析**：和 encoder shift 对比可定位"放大还是衰减"——失败到底来自 encoder 还是 predictor | single-step rollout error 来自 Dreamer / TD-MPC family；ratio 是本文 composite | `tools/repr_analysis/predictor_sensitivity.py::_open_loop_target_shift`; `analyze_model_predictor_noise` |
| `predictor_rollout_drift(T)` | **测什么**：noisy 与 clean history 各自自回归 T 步后两条 latent 轨迹的距离。**用来分析**：multi-step rollout 中误差累积速度，对依赖长 horizon rollout 的 CEM planning 尤其关键，是 single-step shift 看不出的 horizon-scaling 效应 | multi-step noise-vs-clean conditioning，文献无直接对应，可主张 novelty；不同于 Dreamer/TD-MPC 对 ground-truth latent 的 rollout MSE | `predictor_sensitivity.py::_autoregressive_rollout` |
| `transition_resolution_ratio` | **测什么**：相邻帧 latent 距离 / 跨序列随机帧 latent 距离的比值。**用来分析**：越小说明 latent 越能区分时间相邻状态、保留 task step 分辨率；用来诊断 over-uniformity / over-smoothing 是否削弱了 task-relevant 结构 | temporal-neighbor 版本的 intra/inter gap；命名和 planning 用法可主张新颖 | `tools/repr_analysis/task_resolution.py::_transition_metrics` |
| `id_probe_r2`, `id_probe_r2_min` | **测什么**：仅训 linear readout，用 `(z_t, z_{t+1})` 预测 action 的 R²；`_min` 表示沿 action 维度取最低分量。**用来分析**：action-relevant 状态信息的下界代理，判断 latent 是否保留 control-relevant 信号；`_min` 可暴露被忽略的 action 维度 | inverse-dynamics representation probe；Brandfonbrener 2023, Pathak 2017, Alain & Bengio 2017 | `task_resolution.py::_ridge_probe`; `_build_id_probe_data` |
| `lidar_rank` | **测什么**：把相邻帧作 positive pair 计算的 LiDAR rank。**用来分析**：在 temporal-positive 视角下的有效维度，是 effective rank 的 task-aware 补充——回答"和时间相关的不变方向上还剩几维有用信息" | LiDAR；Thilak 2024，本文只是迁移到 temporal pair | `task_resolution.py::_lidar_rank` |
| `predictor_rollout_drift_z(T)`, `target_to_nn_*_ratio_z` | **测什么**：直接在 encoded `z` 加噪后的 predictor target shift / rollout drift。**用来分析**：剥离 encoder 后单独考察 predictor 自身平滑度；与 input-space 同名指标对照可把 robustness 失败归因到 encoder 还是 predictor | latent randomized smoothing / RobustZero 相关；本文用于 post-hoc encoder-decoupled diagnostic | `tools/repr_analysis/latent_noise_sensitivity.py::_open_loop_target_shift`; `_autoregressive_rollout` |
| `cost_surface_slope_z`, `robust_radius_z` | **测什么**：latent 噪声扫描下 CEM planning cost 对 std 的斜率，以及 cost / predictor 边界首次跨阈值的 std。**用来分析**：从 cost surface 视角给出 planning 对 latent 噪声的经验鲁棒半径，和 input-space radius 对照定位真实攻击面 | Cohen 2019 / Lipschitz estimation 思路迁移到 latent cost；不是 certified bound | `latent_noise_sensitivity.py::analyze_model_latent_noise`; `summarize_latent_noise_geometry` |
| `pearson_r`, `spearman_rho`, bootstrap CI | **测什么**：跨 ckpt 把诊断指标和 eval success 算 Pearson / Spearman 相关，以及 bootstrap 置信区间（Spearman 用 average-tie ranks）。**用来分析**：判断哪些诊断指标具备 label-free predictive power、可在不跑昂贵 eval 的情况下用于 ckpt 排序与筛选 | label-free performance prediction / ATC family；Garg 2022, Deng & Zheng 2021, Efron & Tibshirani 1993 | `tools/repr_analysis/diagnostic_correlation.py` |
| roll-up summary | **测什么**：把 encoder / predictor / task-resolution / latent-noise 各模块的关键指标按 ckpt 聚合成单行 JSON / CSV。**用来分析**：跨任务、跨 ckpt 横向比对的统一入口，也是 P0.4 相关性分析与 §A 溯源表的输入 | 本项目工程聚合层 | `tools/repr_analysis/run_full_diagnostics.py::_summarize_noise_to_predictor_to_resolution` |

### 7.2 真正可主张的 novelty

收紧到三条互不重复的贡献：

1. **Composite robustness ratios for planning latents.**  
   `noise_to_nn_cos_ratio` 和 `robust_radius` 把 "encoder 局部 Lipschitz" 与 "latent 几何 NN 尺度" 组合成单一无量纲的鲁棒性边界。已有的 randomized smoothing 给的是 *certified classifier* radius，我们给的是 *empirical planning-latent* radius。

2. **Multi-step latent drift between noisy- and clean-history conditioning.**  
   `predictor_rollout_drift(T)` 在文献里没有对应指标。Dreamer/TD-MPC 的 rollout MSE 是相对 ground-truth latent 的，不是 noise-vs-clean 条件下的对比。

3. **Integrated diagnostic toolkit + active validation protocol for predicting planning failure.**  
   把 encoder shift / NN 几何 / predictor drift / task resolution / ID probe 组合成一个 label-free predictor，再用 holdout checkpoint 做盲分桶验证 (P0.6)。这套从相关性到主动验证的闭环在 SSL 诊断和 MBRL 鲁棒性两个文献里都没有现成对应。

### 7.3 论文必引参考（写 Related Work 时按这个清单铺）

> SSL 表征诊断：RankMe (Garrido 2023), LiDAR (Thilak 2024), Wang & Isola (ICML 2020), Jing et al. dimensional collapse (ICLR 2022), Ethayarajh anisotropy (EMNLP 2019), CKA (Kornblith 2019).
> Lipschitz / 鲁棒性：Virmaux & Scaman (NeurIPS 2018), LipSDP (Fazlyab 2019), Cohen randomized smoothing (ICML 2019), Hoffman Jacobian regularization (2019), SpecFormer (ECCV 2024).
> Probing：Alain & Bengio (ICLR-W 2017), Brandfonbrener inverse dynamics (NeurIPS 2023), ICM (Pathak 2017).
> OOD / NN-based：Sun KNN-OOD (NeurIPS 2022), SNGP (Liu 2020), Mahalanobis (Lee 2018).
> World model 鲁棒性：Dreamer (Hafner 2019/2020), TD-MPC2 (Hansen 2024), Liu LDR (ICML 2024), surprise-recognition (arXiv 2512.01119).
> Label-free performance prediction：ATC (Garg 2022), Deng & Zheng (CVPR 2021), Active Testing (Kossen 2021)。`PROXIMA (2026)` 先不作为主引用，待正式写作前核对。

正式写作时再补 BibTeX / URL；本文只保留指标-工作映射，避免把计划文档变成 reference dump。

### 7.4 必须主动 differentiate 的近期工作（撞车风险高）

下面三篇是 2025 年和我们工作高度交叉的 prior art，论文 Related Work 必须单独段落讨论，明确 delta：

| 工作 | 时间 / 出处 | 核心主张 | 与我们的撞车点 | 我们的 delta |
|---|---|---|---|---|
| **PCA++** "How Uniformity Induces Robustness to Background Noise in Contrastive Learning" | arXiv:2511.12278 (Nov 2025) | 在 contrastive SSL 中，uniformity 损失隐式诱导 background noise robustness；提供理论分析 | 整体故事（uniformity → robustness）和我们 SWM uniformity 高度相似 | (a) 设置不同：他们是分类 / contrastive，我们是 world-model planning；(b) 我们发现 uniformity 实际效果（聚簇化 vs 平滑化）取决于 noise augmentation 实现方式（固定 std 还是 per-frame）；(c) 我们提供 *诊断 toolkit + active validation*，他们提供 *理论 + 分类 acc*；(d) 我们发现 task-resolution tradeoff（PushT 反响应），不在他们的覆盖范围 |
| **Surprise-Recognition** "World Model Robustness via Surprise Recognition" | arXiv:2512.01119 (Dec 2025) | 用 world model 的 single-step prediction surprise 做 *runtime* 噪声 input 过滤 | 同样是 WM + noise，使用 single-step prediction error 信号 | (a) 他们是 *runtime filter*（每帧决定是否信任输入），我们是 *pre-hoc predictor*（从校准数据预测 policy success drop）；(b) 他们是 single-step；我们 `predictor_rollout_drift(T)` 是 multi-step；(c) 我们做 cross-checkpoint correlation + holdout 分桶验证，他们没有 |
| **RobustZero** "Enhancing MuZero Reinforcement Learning Robustness to State Perturbations" | ICML 2025 (Li et al., proceedings.mlr.press/v267/li25bf.html) | 在 MuZero 上用 latent state perturbation（worst-case + random-case）训练鲁棒策略，定义 robust radius，覆盖 control / energy / transportation / Mujoco | 同样是 world model + latent perturbation + robust radius，setting 最近（关联 P2/P5） | (a) 他们改 *training objective*，我们不改训练只做 *post-hoc 诊断*；(b) 他们针对 MuZero 类（discrete latent + value-based）；我们针对 JEPA 类（continuous latent + CEM cost）；(c) 他们没有 input-space ↔ latent-space 的 encoder-decoupling 对照；(d) 没有 active validation / 跨任务 invariance-resolution tradeoff |

写 paper 必须在 Related Work 第一段就把这三篇拎出来讲清差异；不能只放在 reference list 里。

---

## 8. 决策节点

只保留仍会影响下一步资源投入的分支；其他结论分别放回 §4 / §6 / §7 对应正文。

| 节点 | 触发条件 | 下一步 |
|---|---|---|
| P0.6 holdout | 需要证明诊断不是事后解释 | 用 held-out ckpt 做 low / mid / high eval-drop 盲分桶；命中 ≥ 80% 则诊断工具可独立成节，< 60% 则回到 P3 或重审失败机制 |
| P3 encoder 拆解 | P0.6 失败，或需要解释 SWM angular sensitivity 来源 | 做 SWM-noBN / SWM-LN / SWM-dim128/192 ablation |
| P4 adaptive guardrail | P0.6 能稳定分桶，且 P3 给出明确 sensitivity 来源 | 实现 noise consistency + transition/action preservation；目标是保住 PushT 同时提升 TwoRoom |

---

## 9. 论文叙事草案

一句话版本：

> Spherical world models do not simply improve or degrade planning; they expose a task-dependent invariance-resolution tradeoff in latent geometry, which we predict with a label-free latent-geometry diagnostic toolkit.

中文版本：

> 球面世界模型不是 LeWM 的单向替代，而是改变了 latent space 的几何偏置。低维导航任务受益于更强 invariance 和聚簇化，高分辨率操作任务则需要保留连续状态差异。我们提出 latent geometry 诊断工具（noise-to-NN ratio、empirical robust radius、predictor rollout drift、transition resolution + ID probe），将 embedding geometry 直接映射到 planning robustness，并通过 holdout checkpoint 主动验证。

主要贡献（与 §7.2 对齐）：

1. **方法与基线**：实现 SWM，证明 spherical + temporal-masked uniformity 可以稳定训练 JEPA-style world model（增量贡献，非主要叙事点）。
2. **Composite robustness ratios for planning latents**：`noise_to_nn_cos_ratio` 与 `robust_radius` 把 encoder 局部 Lipschitz 与 latent 几何 NN 尺度组合成无量纲鲁棒性边界。区别于 Cohen et al. 的 certified classifier radius，我们给的是 empirical / planning-latent 版本。
3. **Multi-step latent drift between noisy- and clean-history conditioning**：`predictor_rollout_drift(T)` 在 SSL 诊断和 MBRL 鲁棒性文献里都没有现成对应；Dreamer / TD-MPC 的 rollout MSE 是相对 ground-truth latent，不是 noise-vs-clean 条件对比。
4. **Integrated diagnostic toolkit + active-validation protocol**：把 encoder / predictor / task-resolution / ID-probe / P2/P5 latent-noise probing 组合成 label-free predictor（encoder-decoupled 鲁棒性边界由 P2/P5 给出），再用 holdout checkpoint 做盲分桶验证。这套从相关性到主动验证的闭环是论文核心方法贡献。
5. **机制发现**：noise augmentation 在 WM 中不必然带来 smoothing，可能诱导 clustered/discretized geometry；TwoRoom 与 PushT 对同一 geometry prior 反向响应。
6. **后续方法（initial）**：adaptive resolution / guarded noise consistency，尝试替代手工按任务调 recipe。

诚实声明：`effective_rank`、Wang & Isola uniformity、Jacobian/Lipschitz 概念、ID linear probe、Spearman + bootstrap workflow 均为已有方法（参见 §7.1 和 §7.3），论文中需明确归属。

---

## 10. 维护说明

后续补结果时优先放在对应 P 节：

- `P0`：诊断指标、相关性图、robust radius 表。
- `P1`：noise-aware training 的 eval 和 geometry 变化。
- `P2/P5`：eval-only cost ablation + latent-noise probing（encoder-decoupled diagnostic）。
- `P3`：BN/LN/dim 的 encoder sensitivity。
- `P4`：adaptive resolution / guarded consistency 方法。

避免把每次命令输出都塞进本文；完整流水仍放 `experiments.md`。本文只保留能够改变判断的结果。

---

## 附录 A：CKPT→Eval→诊断完整溯源表（供人工核验）

> **本附录存在的意义**：plan_v3.md §6 P0.4/P0.5/P0.7 的所有相关性数值均来自 `diagnostic_correlation.py` 对 `eval_scores.json` + `diagnostics_summary.json` 的自动计算。本附录逐条记录当前 39 个模型（TwoRoom 8、PushT 11、Reacher 10、Cube 10）对应的 ckpt 子目录、eval 分数来源、诊断指标来源，确保任何数值都可以从原始 ckpt 文件一路追溯到报告中的 ρ 值。注意：Spearman ties 修正已于 2026-05-05 完成，P0.4/P0.5/P0.7 数值已同步更新。

### A.1 TwoRoom（SWM epoch_10, num_eval=300；LeWM epoch_9）

| 模型名 | CKPT 子目录 | 对象文件名 | Eval 分数 | Eval 来源 | `clean_nn_dist` | `eff_rank` | `geometry_flag` |
|---|---|---|---|---:|---:|---:|---|
| LeWM-base | `tworoom_lewm` | `tworoom_lewm_epoch_9_object.ckpt` | 93.0 | `tworoom_results.txt` | 0.03890 | 29.54 | balanced |
| LeWM-fixed-std | `tworoom_lewm_noise_std_0_005` | `tworoom_lewm_noise_std_0_005_epoch_9_object.ckpt` | 96.6 | `tworoom_results.txt` | 0.01295 | 15.08 | fragile,clustered |
| LeWM-perframe-p05 | `tworoom_lewm_noise_0to005_p05` | `tworoom_lewm_noise_0to005_p05_epoch_9_object.ckpt` | 94.0 | `eval_run.log`（run_missing_evals 重跑） | 0.03705 | 27.36 | balanced |
| LeWM-perframe-p1 | `tworoom_lewm_noise_0to005_p1` | `tworoom_lewm_noise_0to005_p1_epoch_9_object.ckpt` | 94.0 | `eval_run.log`（run_missing_evals 重跑） | 0.03571 | 26.58 | balanced |
| SWM-base | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_epoch_10_object.ckpt` | 69.67 | `eval_run.log`（epoch_10, num_eval=300） | 0.03596 | 35.39 | fragile,high_angle_gain |
| SWM-fixed-std | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64_epoch_9_object.ckpt` | 97.6 | `tworoom_results.txt` | 0.00820 | 11.61 | fragile,high_angle_gain,clustered |
| SWM-perframe-p05 | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt` | 87.33 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.04984 | 26.96 | balanced |
| SWM-perframe-0to001-p1 | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64_epoch_10_object.ckpt` | 94.33 | `eval_run.log`（epoch_10, num_eval=300） | 0.05663 | 36.66 | robust |
| SWM-perframe-0to002-p1 | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_10_object.ckpt` | 88.00 | `eval_run.log`（epoch_10, num_eval=300） | 0.05210 | 37.32 | robust |
| SWM-perframe-p1 | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_10_object.ckpt` | 90.80 | `eval_run.log`（epoch_10, num_eval=300） | 0.04751 | 36.41 | balanced |

### A.2 PushT（SWM epoch_10, num_eval=300；LeWM epoch_9, num_eval=150）

| 模型名 | CKPT 子目录 | 对象文件名 | Eval 分数 | Eval 来源 | `clean_nn_dist` | `eff_rank` | `geometry_flag` |
|---|---|---|---|---:|---:|---:|---|
| LeWM-base | `pusht_lewm_20260430` | `pusht_lewm_20260430_epoch_9_object.ckpt` | 80.67 | `summary.txt` | 0.23599 | 47.48 | robust |
| LeWM-fixed-std | `pusht_lewm_noise_std_0_005` | `pusht_lewm_noise_std_0_005_epoch_9_object.ckpt` | 83.0 | `pusht_results.txt` | 0.14473 | 31.40 | robust |
| LeWM-perframe-0to001-p1 | `pusht_lewm_noise_0to001_p1` | `pusht_lewm_noise_0to001_p1_epoch_9_object.ckpt` | 87.33 | `eval_run.log`（run_missing_evals 重跑） | 0.22625 | 48.36 | balanced |
| LeWM-perframe-0to002-p1 | `pusht_lewm_noise_0to002_p1` | `pusht_lewm_noise_0to002_p1_epoch_9_object.ckpt` | 89.33 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.24733 | 48.28 | balanced |
| LeWM-perframe-0to005-p1 | `pusht_lewm_noise_0to005_p1` | `pusht_lewm_noise_0to005_p1_epoch_9_object.ckpt` | 82.0 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.22531 | 46.74 | balanced |
| SWM-base | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_epoch_10_object.ckpt` | 80.00 | `eval_run.log`（epoch_10, num_eval=300） | 0.25821 | 52.94 | robust |
| SWM-fixed-std | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64_epoch_9_object.ckpt` | 61.8 | `pusht_results.txt` | 0.06639 | 18.38 | fragile,high_angle_gain |
| SWM-perframe-0to001-p05 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p05_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p05_dim64_epoch_9_object.ckpt` | 78.0 | `eval_run.log`（run_missing_evals 重跑） | 0.25770 | 42.62 | robust |
| SWM-perframe-0to001-p1 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64_epoch_10_object.ckpt` | 87.33 | `eval_run.log`（epoch_10, num_eval=300） | 0.28104 | 55.45 | robust |
| SWM-perframe-0to002-p05 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64_epoch_9_object.ckpt` | 78.67 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.27604 | 46.04 | balanced |
| SWM-perframe-0to002-p1 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_10_object.ckpt` | 81.33 | `eval_run.log`（epoch_10, num_eval=300） | 0.26221 | 55.09 | balanced |
| LeWM-perframe-0to002-p05 | `pusht_lewm_noise_0to002_p05` | `pusht_lewm_noise_0to002_p05_epoch_9_object.ckpt` | 86.0 | `clean.log`（用户补跑） | 0.21610 | 47.83 | balanced |
| LeWM-perframe-0to005-p05 | `pusht_lewm_noise_0to005_p05` | `pusht_lewm_noise_0to005_p05_epoch_9_object.ckpt` | 14.67 | `clean.log`（用户补跑） | 0.10348 | 38.00 | balanced |
| SWM-perframe-0to005-p05 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt` | 71.33 | `clean.log`（用户补跑） | 0.22603 | 40.56 | balanced |
| SWM-perframe-0to005-p1 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_10_object.ckpt` | 65.33 | `eval_run.log`（epoch_10, num_eval=300） | 0.21337 | 51.98 | balanced |

### A.3 Reacher（SWM epoch_10, num_eval=300；LeWM epoch_9, num_eval=150）

| 模型名 | CKPT 子目录 | 对象文件名 | Eval 分数 | Eval 来源 | `clean_nn_dist` | `eff_rank` | `geometry_flag` |
|---|---|---|---|---:|---:|---:|---|
| LeWM-base | `reacher_lewm_20260430` | `reacher_lewm_20260430_epoch_9_object.ckpt` | 58.67 | `summary.txt` | 0.05987 | 42.95 | balanced |
| LeWM-perframe-0to002-p05 | `reacher_lewm_noise_0to002_p05` | `reacher_lewm_noise_0to002_p05_epoch_9_object.ckpt` | 79.33 | `summary.txt` | 0.07261 | 49.44 | balanced |
| LeWM-perframe-0to002-p1 | `reacher_lewm_noise_0to002_p1` | `reacher_lewm_noise_0to002_p1_epoch_9_object.ckpt` | 72.67 | `summary.txt` | 0.07199 | 47.99 | balanced |
| LeWM-perframe-0to005-p05 | `reacher_lewm_noise_0to005_p05` | `reacher_lewm_noise_0to005_p05_epoch_9_object.ckpt` | 82.67 | `summary.txt` | 0.06902 | 45.99 | balanced |
| LeWM-perframe-0to005-p1 | `reacher_lewm_noise_0to005_p1` | `reacher_lewm_noise_0to005_p1_epoch_9_object.ckpt` | 78.00 | `summary.txt` | 0.05805 | 32.87 | balanced |
| SWM-base | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_epoch_10_object.ckpt` | 60.00 | `eval_run.log`（epoch_10, num_eval=300） | 0.09333 | 50.96 | robust |
| SWM-perframe-0to001-p1 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64_epoch_10_object.ckpt` | 65.67 | `eval_run.log`（epoch_10, num_eval=300） | 0.09549 | 52.64 | robust |
| SWM-perframe-0to002-p05 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64_epoch_9_object.ckpt` | 78.00 | `summary.txt` | 0.09063 | 43.78 | balanced |
| SWM-perframe-0to002-p1 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_10_object.ckpt` | 74.00 | `eval_run.log`（epoch_10, num_eval=300） | 0.09417 | 50.64 | balanced |
| SWM-perframe-0to005-p05 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt` | 76.00 | `summary.txt` | 0.09451 | 45.60 | balanced |
| SWM-perframe-0to005-p1 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_10_object.ckpt` | 78.00 | `eval_run.log`（epoch_10, num_eval=300） | 0.09528 | 51.96 | balanced |

### A.4 Cube（SWM epoch_10, num_eval=300；LeWM epoch_9, num_eval=150）

| 模型名 | CKPT 子目录 | 对象文件名 | Eval 分数 | Eval 来源 | `clean_nn_dist` | `eff_rank` | `geometry_flag` |
|---|---|---|---|---:|---:|---:|---|
| LeWM-base | `cube_lewm_20260430` | `cube_lewm_20260430_epoch_9_object.ckpt` | **74.0** | `clean_150_v4.log`（num_eval=150，分批评估） | 0.18774 | 48.92 | robust |
| LeWM-perframe-0to002-p05 | `cube_lewm_noise_0to002_p05` | `cube_lewm_noise_0to002_p05_epoch_9_object.ckpt` | 64.67 | `summary.txt` | 0.13502 | 49.53 | balanced |
| LeWM-perframe-0to002-p1 | `cube_lewm_noise_0to002_p1` | `cube_lewm_noise_0to002_p1_epoch_9_object.ckpt` | 60.67 | `summary.txt` | 0.13336 | 49.20 | balanced |
| LeWM-perframe-0to005-p05 | `cube_lewm_noise_0to005_p05` | `cube_lewm_noise_0to005_p05_epoch_9_object.ckpt` | 66.00 | `summary.txt` | 0.11817 | 47.31 | balanced |
| LeWM-perframe-0to005-p1 | `cube_lewm_noise_0to005_p1` | `cube_lewm_noise_0to005_p1_epoch_9_object.ckpt` | 64.67 | `summary.txt` | 0.11534 | 45.61 | balanced |
| SWM-base | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_epoch_10_object.ckpt` | 77.00 | `eval_run.log`（epoch_10, num_eval=300） | 0.25956 | 53.69 | robust |
| SWM-perframe-0to001-p1 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64_epoch_10_object.ckpt` | 72.33 | `eval_run.log`（epoch_10, num_eval=300） | 0.25381 | 53.10 | robust |
| SWM-perframe-0to002-p05 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64_epoch_9_object.ckpt` | 72.00 | `summary.txt` | 0.26564 | 43.51 | balanced |
| SWM-perframe-0to002-p1 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_10_object.ckpt` | 74.00 | `eval_run.log`（epoch_10, num_eval=300） | 0.25664 | 53.18 | balanced |
| SWM-perframe-0to005-p05 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt` | 70.67 | `summary.txt` | 0.19656 | 43.68 | balanced |
| SWM-perframe-0to005-p1 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_10_object.ckpt` | 64.00 | `eval_run.log`（epoch_10, num_eval=300） | 0.16804 | 51.38 | balanced |

> **注**：Cube diagnostics 11/11 已补齐（2026-05-05 合并 `cube_lewm_20260430` base 诊断），base LeWM 的 num_eval=150 eval 已完成（74.0%）。所有 LeWM 与 SWM 模型 eval 与诊断均已对齐。

### A.5 数据流与生成脚本

| 层级 | 文件 | 生成脚本 | 输入依赖 |
|---|---|---|---|
| L1 原始 ckpt | `ckpt/<subdir>/*_object.ckpt` | `train.py` / `train_swm.py` | — |
| L2 eval 分数 | `ckpt/<subdir>/*results.txt` / `eval_run.log` | `eval.py` | L1 ckpt + task config |
| L3 诊断 CSV | `p03_diagnostics/*.csv` | `run_full_diagnostics.py`（由 `run_p03_tworoom.py` / `run_p03_pusht.py` 调用） | L1 ckpt + dataset |
| L3b base 补做 | `p03_diagnostics_new_baselines/{LeWM-base,SWM-base}/*.csv` | 同上，单独补跑 | L1 ckpt |
| L4 诊断汇总 | `diagnostics_summary.json` | `regen_summary.py`（合并 L3 + L3b） | L3 CSV |
| L5 eval 汇总 | `eval_scores.json` | 手工维护 / `run_missing_evals.py` 自动更新 | L2 results.txt |
| L6 相关性 | `diagnostic_correlation.csv/.png` | `diagnostic_correlation.py` | L4 + L5 |
| L7 报告 | `plan_v3.md` P0.4/P0.5/P0.7 | 人工撰写（以 L6 为准） | L6 |

### A.6 人工核验检查清单

你可以按以下步骤独立复现任何数值：

1. **核验 Eval 分数**
   ```bash
   # TwoRoom
   cat dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/<subdir>/tworoom_results.txt
   # PushT
   cat dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt/<subdir>/pusht_results.txt
   # 若模型由 run_missing_evals.py 重跑，则查看 eval_run.log 中 'success_rate': <num>
   ```

2. **核验诊断指标（原始 CSV）**
   ```bash
   # TwoRoom
   cat dataset/ag_data/data/world_model/quentinll/lewm-tworooms/repr_analysis/p03_diagnostics/noise_sensitivity.csv | grep <model>
   # PushT（含 base 补做）
   cat dataset/ag_data/data/world_model/quentinll/lewm-pusht/repr_analysis/p03_diagnostics/noise_sensitivity.csv | grep <model>
   cat dataset/ag_data/data/world_model/quentinll/lewm-pusht/repr_analysis/p03_diagnostics_new_baselines/<model>/noise_sensitivity.csv | grep <model>
   ```

3. **核验相关性数值**
   ```bash
   cat dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}/repr_analysis/p03_diagnostics/diagnostic_correlation.csv
   # 或重新运行脚本
   python -m tools.repr_analysis.diagnostic_correlation \
       --diagnostics dataset/ag_data/data/world_model/quentinll/lewm-tworooms/repr_analysis/p03_diagnostics/diagnostics_summary.json \
       --eval-scores dataset/ag_data/data/world_model/quentinll/lewm-tworooms/repr_analysis/p03_diagnostics/eval_scores.json \
       --out-dir tworoom_corr_check
   ```

4. **关键字段定义速查**
   - `clean_nn_cos_dist_median`：clean embedding 最近邻 cosine 距离中位数（越小 = 聚簇越紧）
   - `clean_effective_rank`：clean embedding 的有效维度（Wang & Isola uniformity 相关）
   - `predictor_target_to_nn_cos_ratio_at_max_std`：最大噪声 std 下，predictor target 与 nearest neighbor 的 cosine ratio（越小 = target shift 控制越好）
   - `geometry_flag`：`run_full_diagnostics.py` 自动标注的 noise geometry 类别（balanced / robust / fragile / clustered 等）

### A.7 历史数据修正记录

| 时间 | Commit | 修正内容 | 影响 |
|---|---|---|---|
| 2026-05-01 02:30 | `8605bf5` | SWM-fixed-std PushT eval 89.8→61.8；LeWM-fixed-std 83.6→83.0；TwoRoom SWM-base 90.8→91.0；LeWM-fixed-std 95.6→96.6 | PushT 主导指标从 `lidar_rank` 变为 `predictor_target_to_nn_cos_ratio` 和 `clean_effective_rank` |
| 2026-05-01 04:16 | `620de01` | 运行 `run_missing_evals.py`，补齐 11 个缺失 eval（4 TwoRoom + 7 PushT），actual 与 expected 多处不符 | `eval_scores.json` 更新，但 plan_v3.md **未同步更新相关性数值** |
| 2026-05-01 04:33 | `bf79a80` | 插入诊断可视化配图 | 仅新增图片引用，未修正数值 |
| 2026-05-01 05:07 | `6c7bf90` | 将 plan_v3.md 中所有 Spearman/Pearson 数值更新为与 `diagnostic_correlation.csv` 一致（基于 num_eval=50 的 perframe 模型） | PushT 所有指标 \|ρ\| 降至 0.4–0.6，无 ≥0.7 强相关；TwoRoom `clean_nn_cos_dist` 升至 −1.000 |
| 2026-05-01 12:09 | `4ce4931` | **第三次修正**：发现 perframe 模型 eval 仅用 `num_eval=50`，将 11 个 perframe 模型重跑为 `num_eval=150` | TwoRoom perframe eval 更稳定（SWM-perframe-p05 92.0→87.33，SWM-perframe-p1 92.0→86.67）；PushT `predictor_target_to_nn_cos_ratio` 从 −0.564 升至 **−0.791**，`predictor_rollout_T8_l2` 从 +0.300 升至 **+0.636**，PushT 预测力显著改善 |
| 2026-05-05 | working tree | 修正 `diagnostic_correlation.py` 的 Spearman rank：ordinal ranks → average-tie ranks | eval 分数存在 exact ties，旧版 ρ/CI 需重算；P0.4/P0.5/P0.7 当前数值降级为历史记录 |
| 2026-05-05 | working tree | 四任务相关性全部使用 average-tie rank 版重算；Cube eval_scores 补齐 0to001-p1（72.33%），diagnostics_summary 合并 10 个模型 | P0.4/P0.5/P0.7 已同步更新；TwoRoom `clean_nn_cos_dist` 保持 −0.91，PushT `predictor_target_to_nn_cos_ratio` 降至 −0.59，Cube 新发现 `latent_rollout_angle_slope` ρ=+0.83 |
