# 球面世界模型实验计划 V3

> 当前定位：本文不是单纯记录”SWM 是否强于 LeWM”，而是整理一个更稳定的研究路线：**world model 的 latent geometry 如何匹配 planning 任务的状态分辨率需求**。  
> 原始设计见 `plan_v2.md`，完整流水实验见 `experiments.md`。

---

## 0. 当前结论

最初问题是：把 LeWM 的 Euclidean embedding + SIGReg 换成 spherical embedding + uniformity，是否能稳定提升规划性能？

目前更准确的判断是：

1. **SWM 不是全局优于 LeWM 的替代品。**  
   在 clean eval 上，当前最佳 SWM 与 LeWM 接近，4-task single-seed 平均略高，但没有形成压倒性优势。

2. **SWM 改变了表征的 invariance-resolution tradeoff。**  
   球面归一化、uniformity、temporal masking、noise augmentation 都在改变“哪些观测差异应该被保留，哪些应该被抹掉”。

3. **不同任务对这个 tradeoff 的偏好可能相反。**  
   TwoRoom 低维、离散、视觉细节冗余，受益于更强 invariance / clustering；PushT 需要精细连续状态分辨率，同样配方会损害控制。

4. **表征分析工具本身是通用贡献。**  
   `noise_sensitivity.py`、robust radius、clean-neighbor distance、noise-induced angular shift 等指标可以在不大量 eval 的情况下诊断 latent geometry 的风险。

因此后续主线不应是“为每个任务手调一套 recipe”，而应是：

> 建立一套可诊断、可预测、最好可自适应的 latent geometry 设计方法，让 world model 根据任务分辨率需求在 robustness 和 precision 之间取舍。

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

> ⚠️ **数据来源审计**：以下数据来自 `experiments.md` 记录的早期 4-task benchmark（2026-04-15/20），配置为 epoch=10，num_eval=500，single seed。这些 ckpt 与当前 P0.3 诊断分析使用的模型**不是同一组**（dim、temporal 配置、noise 设置均可能不同），因此本节仅作历史参考，不进入 P0.3 相关性分析。
>
> | Task | LeWM | SWM best | Delta | 来源 ckpt |
> |---|---:|---:|---:|:---|
> | TwoRoom | 93.0 | 90.8 | -2.2 | LeWM: `tworoom_lewm`；SWM: `tworoom_swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_2_dim_64_20260420` |
> | Cube | 69.2 | 74.0 | +4.8 | 旧 benchmark，ckpt 已不在当前目录，不可追溯 |
> | PushT | 89.4 | 89.8 | +0.4 | LeWM 来源不明（当前 `pusht_lewm` 仅 81.0）；SWM `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260415` **results.txt 已丢失**，不可追溯。89.8 并非 SWM-fixed-std（真实 61.8）。 |
> | Reacher | 62.2 | 66.0 | +3.8 | 旧 benchmark，ckpt 已不在当前目录，不可追溯 |
> | Average | 78.5 | 80.2 | +1.7 | — |

### 2.2 当前 P0.3 诊断用模型 clean benchmark（epoch_9，single seed；num_eval 以各模型 Eval 来源列标注为准）

以下模型与 §6 P0.3 / P0.4 相关性分析使用同一组 ckpt，可作为一致基准：

| Task | LeWM best | SWM best | Delta | 说明 |
|---|---:|---:|---:|:---|
| TwoRoom | 96.6 (`lewm_fixed-std_noise0.005`) | 97.6 (`swm_fixed-std_noise0.005`) | +1.0 | LeWM fixed-std 最佳；SWM fixed-std 聚簇化红利 |
| PushT | 89.33 (`lewm_noise_0to002_p1`) | 81.33 (`swm_noise_0to002_p1`) | -8.0 | 基于 epoch_9, num_eval=150；fixed-std 仅 61.8 |
| Reacher | 82.67 (`lewm_noise_0to005_p05`) | 78.0 (`swm_noise_0to002_p05` / `swm_noise_0to005_p1`) | -4.67 | LeWM per-frame 最佳；SWM 各配置接近 |
| Cube | 90.0 (`lewm_base`, num_eval=10*) | 78.0 (`swm_base`) | -12.0 | LeWM-base num_eval=10 近似；其余 9 模型 num_eval=150 |

> *Cube LeWM-base 的 num_eval=150 eval 因环境兼容性问题卡住，当前使用 num_eval=10 结果（90.0）作为近似值，待修复后补齐。

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

TwoRoom, std=0.03, num_eval=50：

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

**TwoRoom eval（per-frame 新跑为 num_eval=150；baseline 旧行为不同 eval budget）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWM baseline | 90.8 | **42*** | — | — | **36*** | — | — | **66*** | — | — |
| SWM 旧版固定 std | **97.6** | — | — | — | — | **98.0** | 88.0 | — | 56.0 | — |
| SWM per-frame p1 | 86.7 | 88.0 | 87.3 | 89.3 | 87.3 | 87.3 | 89.3 | 86.7 | 87.3 | 86.7 |
| SWM per-frame p05 | 87.3 | 86.7 | 88.7 | 86.0 | 86.7 | 85.3 | 88.0 | 87.3 | 86.7 | 85.3 |
| LeWM per-frame p1 | **94.0** | 94.0 | 94.0 | 93.3 | 94.0 | 92.7 | 94.7 | 94.0 | 94.0 | 94.0 |
| LeWM per-frame p05 | **94.0** | 94.7 | 94.0 | 94.7 | 94.0 | 94.7 | 94.0 | 94.0 | 94.0 | 94.0 |

> *baseline clean 来自 §2 的 4-task benchmark（num_eval=500）；baseline noisy 来自 §3.1 的 `std=0.03` 旧评估（num_eval=50）。它只用于锚定旧 failure，不应和 per-frame num_eval=150 行做小数点级比较。

**PushT eval（num_eval=150）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LeWM baseline | **86.7** | — | — | — | — | — | — | — | — | — |
| SWM baseline | 78.7 | — | — | — | — | — | — | — | — | — |
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
| 几何标签 | `geometry_flag` (`clustered/fragile/robust/balanced`) + `recommendation` | 经验阈值规则 |
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
   - 支持 `noise_geometry ∈ {ambient, tangent}`（tangent 用于 SWM 切空间扰动）与 `std_mode ∈ {relative, absolute}`（默认 relative，按 per-token clean norm 缩放，跨 LeWM/SWM 可比）。
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
| SWM-base | **0.0183** | 0.02 | 961.9 | 0.0594 | 29.04 | balanced |
| SWM-fixed-std | **0.00036** | **0.005** | **6199.4** | **0.0082** | 11.61 | **fragile,high_angle_gain,clustered** |
| SWM-perframe-p05 | **>0.08** | **>0.08** | **94.7** | 0.0498 | 26.96 | balanced |
| SWM-perframe-p1 | **>0.08** | **>0.08** | **80.3** | 0.0477 | 26.89 | balanced |

> **>0.08** 表示在 std 测到 0.08 时 `noise_to_nn_ratio` 仍未超过 1，即 extremely robust。

**关键发现**

1. **聚簇化效应被量化**：SWM-fixed-std 的 `robust_radius=0.00036`（ baseline 的 1/50），`clean_nn_cos_dist=0.0082`（缩到 1/7），`geometry_flag` 明确标记 `fragile,high_angle_gain,clustered`。
2. **per-frame 平滑化显著**：SWM-perframe 的 `noise_angle_slope` 从 962 降到 80（接近 LeWM-perframe 的 87），`clean_nn_cos_dist` 恢复到 0.048，与 baseline 同级。
3. **LeWM 固定 std 也有聚簇化**：`clean_nn_cos_dist=0.013`（baseline 0.039 的 1/3），`robust_radius=0.007`（baseline 的 1/2），flag 为 `fragile,clustered`，但程度远轻于 SWM。
4. **Predictor 稳定性意外提升**：per-frame 训练的 rollout drift（T=8 L2）在 max std=0.08 下比 baseline 降低一个数量级（TwoRoom LeWM: 18.07→0.78，降低 **23×**；SWM: 1.25→0.07，降低 **18×**；详见上方 `@ max std=0.08` 表），说明噪声训练同时改善了动力学预测的平滑性。

**PushT（11 个 geometry rows，n=11 可进相关性分析）**

| 模型 | robust_radius | first_risk_std | noise_angle_slope (°/std) | clean_nn_cos_dist | clean_eff_rank | geometry_flag |
|---|---:|---:|---:|---:|---:|---|
| LeWM-base | **0.0529** | 0.05 | 284.8 | 0.2360 | 47.48 | **robust** |
| LeWM-fixed-std | **0.0205** | 0.03 | 711.4 | 0.1447 | 31.40 | **robust** |
| LeWM-perframe-0to001-p1 | **>0.08** | **>0.08** | 121.3 | 0.2263 | 48.36 | balanced |
| LeWM-perframe-0to002-p1 | **>0.08** | **>0.08** | 71.8 | 0.2473 | 48.28 | balanced |
| LeWM-perframe-0to005-p1 | **>0.08** | **>0.08** | 47.5 | 0.2253 | 46.74 | balanced |
| SWM-base | **0.0372** | 0.05 | 399.7 | 0.2645 | 44.02 | **robust** |
| SWM-fixed-std | **0.0005** | **0.005** | **8928.9** | **0.0664** | 18.38 | **fragile,high_angle_gain** |
| SWM-perframe-0to001-p05 | **0.0718** | 0.08 | 169.9 | 0.2577 | 42.62 | **robust** |
| SWM-perframe-0to001-p1 | **0.0669** | 0.08 | 103.2 | 0.2845 | 45.70 | **robust** |
| SWM-perframe-0to002-p05 | **>0.08** | **>0.08** | 88.4 | 0.2760 | 46.04 | balanced |
| SWM-perframe-0to002-p1 | **>0.08** | **>0.08** | 69.5 | 0.2600 | 45.46 | balanced |

> 新增 baselines（20260430）：LeWM-base eval **86.7%**，geometry `robust`（radius=0.053，eff_rank=47.5）；SWM-base eval **78.7%**，geometry `robust`（radius=0.037，eff_rank=44.0）。两者均被评为 robust，与 fixed-std 的 fragile 形成对比。

**Noise sensitivity @ std=0.08：median vs p90，多 scope 对比**

TwoRoom：

| 模型 | goal_med | goal_p90 | hist_med | hist_p90 | all_med | all_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | 71.6° | 85.8° | 70.5° | 88.0° | 70.6° | 87.8° | 5.12 |
| LeWM-fixed-std | 86.9° | 92.8° | 87.6° | 93.8° | 87.5° | 93.8° | 9.34 |
| LeWM-perframe-p05 | 10.2° | 14.7° | 10.4° | 14.7° | 10.4° | 14.7° | 0.67 |
| LeWM-perframe-p1 | **7.6°** | **11.5°** | **7.5°** | **11.0°** | **7.6°** | **11.0°** | 0.51 |
| SWM-base | 69.9° | 89.4° | 71.0° | 91.1° | 70.8° | 90.9° | 3.32 |
| SWM-fixed-std | 80.7° | 87.0° | 80.9° | 87.3° | 80.9° | 87.3° | 10.08 |
| SWM-perframe-p05 | 8.8° | 12.8° | 8.6° | 12.3° | 8.6° | 12.3° | 0.49 |
| SWM-perframe-p1 | **7.3°** | **10.0°** | **7.3°** | **10.5°** | **7.3°** | **10.5°** | 0.41 |

PushT：

| 模型 | goal_med | goal_p90 | hist_med | hist_p90 | all_med | all_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| LeWM-base | 60.8° | 77.8° | — | — | — | — | 1.55 |
| LeWM-fixed-std | 84.0° | 87.8° | 84.3° | 88.5° | 84.3° | 88.4° | 2.80 |
| LeWM-perframe-0to001-p1 | 19.4° | 26.9° | 19.5° | 26.1° | 19.5° | 26.2° | 0.50 |
| LeWM-perframe-0to002-p1 | 10.7° | 14.6° | 10.8° | 14.9° | 10.8° | 14.9° | 0.27 |
| LeWM-perframe-0to005-p1 | **5.2°** | **7.4°** | **5.3°** | **7.4°** | **5.3°** | **7.4°** | 0.14 |
| SWM-base | 85.1° | 96.2° | — | — | — | — | 1.86 |
| SWM-fixed-std | 63.2° | 67.8° | 62.9° | 67.9° | 62.9° | 67.9° | 2.87 |
| SWM-perframe-0to001-p05 | 48.8° | 73.3° | 45.9° | 72.1° | 46.3° | 72.3° | 1.15 |
| SWM-perframe-0to001-p1 | 59.1° | 81.0° | 57.2° | 79.5° | 57.5° | 79.6° | 1.31 |
| SWM-perframe-0to002-p05 | 36.2° | 56.9° | 35.8° | 56.7° | 35.8° | 56.7° | 0.84 |
| SWM-perframe-0to002-p1 | **19.6°** | **33.8°** | **18.7°** | **30.6°** | **18.8°** | **31.0°** | 0.47 |

> **Tail failure（p90）**：TwoRoom 中 per-frame 的 p90 与 median 接近（差 <5°），说明分布集中；baseline 的 p90 比 median 高 15–20°，存在显著的 tail risk。PushT SWM-perframe-0to001 的 p90 远高于 median（73° vs 49°），说明该配置下仍有少数样本对 noise 极其敏感。`nn_l2_ratio` 在 per-frame 模型中普遍 <0.7（远低于 1.0 警戒线），而 fixed-std 模型 >2.8，说明 per-frame 的 noise 幅度被有效控制在 encoder 邻域内。

**Noise sensitivity L2 口径（std=0.08, goal frame）**

TwoRoom：

| 模型 | clean_nn_l2 | noise_l2_med | noise_l2_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|
| LeWM-base | 0.058 | 0.297 | 0.502 | 5.12 |
| LeWM-fixed-std | 0.037 | 0.346 | 0.553 | 9.34 |
| LeWM-perframe-p05 | 0.054 | 0.036 | 0.050 | 0.67 |
| LeWM-perframe-p1 | 0.052 | 0.027 | 0.038 | 0.51 |
| SWM-base | 0.112 | 0.371 | 0.452 | 3.32 |
| SWM-fixed-std | 0.042 | 0.423 | 0.498 | 10.08 |
| SWM-perframe-p05 | 0.099 | 0.049 | 0.071 | 0.49 |
| SWM-perframe-p1 | 0.101 | 0.041 | 0.056 | 0.41 |

PushT：

| 模型 | clean_nn_l2 | noise_l2_med | noise_l2_p90 | nn_l2_ratio |
|---|---:|---:|---:|---:|
| LeWM-base | 9.469 | 14.694 | — | 1.55 |
| LeWM-fixed-std | 0.173 | 0.484 | 0.580 | 2.80 |
| LeWM-perframe-0to001-p1 | 0.218 | 0.109 | 0.152 | 0.50 |
| LeWM-perframe-0to002-p1 | 0.227 | 0.061 | 0.083 | 0.27 |
| LeWM-perframe-0to005-p1 | 0.226 | 0.032 | 0.045 | 0.14 |
| SWM-base | 0.727 | 1.352 | — | 1.86 |
| SWM-fixed-std | 0.161 | 0.462 | 0.561 | 2.87 |
| SWM-perframe-0to001-p05 | 0.250 | 0.288 | 0.358 | 1.15 |
| SWM-perframe-0to001-p1 | 0.267 | 0.350 | 0.477 | 1.31 |
| SWM-perframe-0to002-p05 | 0.268 | 0.225 | 0.311 | 0.84 |
| SWM-perframe-0to002-p1 | 0.261 | 0.123 | 0.197 | 0.47 |

> **L2 口径验证**：`nn_l2_ratio`（noise_l2 / clean_nn_l2）与 cosine 口径的 `noise_to_nn_cos_ratio` 在定性上一致——per-frame 模型 ratio <1（noise 在邻域内），fixed-std ratio >2.8（noise 超出邻域）。PushT 的绝对 L2 值普遍大于 TwoRoom（clean_nn_l2 0.17–0.27 vs 0.04–0.11），说明 PushT latent 的 Euclidean 尺度更大，但 relative ratio 仍具可比性。

**Predictor rollout drift 累积（history 加噪 @ std=0.005）**

TwoRoom：

| 模型 | T1_l2 | T2_l2 | T4_l2 | T8_l2 | T8_angle |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.565 | 0.522 | 0.510 | 0.534 | 2.27° |
| LeWM-fixed-std | 0.908 | 0.899 | 0.889 | 0.886 | 3.29° |
| LeWM-perframe-p05 | **0.070** | **0.068** | **0.068** | **0.061** | 0.26° |
| LeWM-perframe-p1 | **0.050** | **0.049** | **0.045** | **0.043** | 0.19° |
| SWM-base | 0.071 | 0.069 | 0.070 | 0.075 | 4.31° |
| SWM-fixed-std | 0.440 | 0.474 | 0.495 | 0.509 | 29.46° |
| SWM-perframe-p05 | **0.006** | **0.006** | **0.006** | **0.005** | 0.28° |
| SWM-perframe-p1 | **0.005** | **0.005** | **0.005** | **0.005** | 0.29° |

PushT：

| 模型 | T1_l2 | T2_l2 | T4_l2 | T8_l2 | T8_angle |
|---|---:|---:|---:|---:|---:|
| LeWM-base | 0.289 | 0.301 | 0.345 | 0.431 | 1.90° |
| LeWM-fixed-std | 0.493 | 0.500 | 0.517 | 0.548 | 2.16° |
| LeWM-perframe-0to001-p1 | 0.135 | 0.135 | 0.139 | 0.179 | 0.74° |
| LeWM-perframe-0to002-p1 | **0.082** | **0.079** | **0.083** | **0.100** | 0.42° |
| LeWM-perframe-0to005-p1 | **0.050** | **0.049** | **0.055** | **0.069** | 0.28° |
| SWM-base | 0.038 | 0.039 | 0.041 | 0.048 | 2.74° |
| SWM-fixed-std | 1.143 | 1.141 | 1.155 | 1.149 | 70.16° |
| SWM-perframe-0to001-p05 | 0.016 | 0.017 | 0.018 | 0.021 | 1.20° |
| SWM-perframe-0to001-p1 | **0.009** | **0.010** | **0.011** | **0.013** | 0.74° |
| SWM-perframe-0to002-p05 | **0.008** | **0.008** | **0.009** | **0.010** | 0.59° |
| SWM-perframe-0to002-p1 | **0.006** | **0.006** | **0.007** | **0.008** | 0.45° |

> **Drift 累积模式**：TwoRoom LeWM 的 drift 在 T1 就很大（0.5–0.9），之后几乎不增长，说明 predictor 的单步误差是主导；SWM 的 drift 同样 T1 即饱和。PushT SWM-fixed-std 的 T8 angle 高达 70°，说明 predictor 在球面空间中发生了方向翻转。Per-frame training 把 LeWM T8 从 0.55→0.04（TwoRoom）和 0.55→0.07（PushT），把 SWM T8 从 0.51→0.005（TwoRoom）和 1.15→0.008（PushT），改善幅度与 T8-only 表一致。

**Predictor rollout drift @ max std=0.08（summary 口径，与 `diagnostics_summary.json` 对齐）**

TwoRoom：

| 模型 | T8_l2 | T8_angle | 备注 |
|---|---:|---:|---|
| LeWM-base | **18.07** | 88.59° | baseline，无 noise 训练 |
| LeWM-perframe-p1 | **0.78** | 3.54° | per-frame，降低 **23×** |
| SWM-base | **1.25** | 77.33° | baseline，无 noise 训练 |
| SWM-perframe-p1 | **0.07** | 4.18° | per-frame，降低 **18×** |

> 该表是 `predictor_sensitivity.json` 中 `std=0.08` 的汇总，对应 P0.3 文字中 “18→0.8、1.25→0.07” 的出处。与上方 `std=0.005` 表口径不同：max std 下 baseline 的 drift 被噪声显著放大（LeWM 18×、SWM 16×），而 per-frame 模型仍保持在 <1 的低水平。这说明 per-frame noise training 不仅降低了低噪声下的 drift，更重要的是把 predictor 的 Lipschitz 常数压到足够低，使得大噪声输入也不会导致 rollout 发散。

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

> CKA：噪声 latent 与 clean latent 的 Centered Kernel Alignment。baseline 极低（TwoRoom LeWM 0.27 / SWM 0.38），noise training 后均跃升至 >0.87，说明 noise training 显著增强了 encoder 的表征稳定性。PushT SWM-perframe-0to001 反常地低（0.52–0.58），可能说明该配置下的 latent 对噪声过于敏感或发生了表征切换。

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

> **关键发现**：
> 1. `transition_resolution_ratio` 完美区分任务类型：TwoRoom cos 0.18–0.73（离散状态转移，相邻帧差异大），PushT cos 0.06–0.12（连续控制，相邻帧极相似）。
> 2. `id_probe_r²` PushT (0.67–0.77) >> TwoRoom (0.14–0.28)，说明 PushT 的 latent 天然保留了更强的动作可预测性；但 TwoRoom LeWM-fixed-std 出现 **−0.834** 的异常负值，说明聚簇化严重破坏了动作信息。
> 3. `lidar_rank` PushT (7.6–38.1) > TwoRoom (3.9–10.5)，与任务复杂度一致；SWM-perframe 在 PushT 上 lidar_rank 飙升到 35–38（远高于 LeWM 的 13–16），可能暗示球面 uniformity 在高维连续控制任务中引入了额外的有效维度。

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

> **核心洞察**：
> 1. **SWM predictor 天生对 latent perturbation 稳定 10–16×。** TwoRoom 5.8→0.6，PushT 11.0→0.7。这是因为 cosine/normalized predictor 内建了尺度不变性；LeWM 的 L2 predictor 对 latent scale 敏感。
> 2. **LeWM cost surface 对 goal latent 扰动敏感约 2×。** TwoRoom 2.1 vs 1.0，PushT 3.8 vs 1.6。L2 cost 在 Euclidean space 的斜率更大，同样的 latent 偏移产生更大的 cost 变化。
> 3. **Per-frame pixel-noise training 不改善 predictor 的 latent-noise 鲁棒性。** LeWM-perframe 的 T8 drift 与 baseline 几乎相同（5.9 vs 5.8），SWM-perframe 甚至略升（0.81 vs 0.67）。这说明三层归因中，瓶颈在 **Layer 1 (encoder)**，而非 Layer 2 (predictor) 或 Layer 3 (cost surface)。noise training 的收益集中在 pixel→latent 映射的平滑化，而不是 predictor 本身的 Lipschitz 改善。
> 4. **`robust_radius_z` 已修复：goal scope 仍为 NaN，但 history scope 通过 rollout-drift fallback 获得有效值。** 修改内容：`summarize_latent_noise_geometry` 在 `target_to_nn_cos_ratio` 无法达到 threshold=1.0 时，回退到 `rollout_T8_l2_median / clean_nn_l2_median` 作为 ratio 进行插值。`run_full_diagnostics.py` 的 `_summarize` 也改为分别从 goal scope（cost slope）和 history scope（robust radius + slope）提取指标。当前 history scope `robust_radius_z`：TwoRoom 0.005–0.021，PushT 0.018–0.031。与 eval 的相关性：TwoRoom −0.54（弱/中等），PushT −0.08（弱），说明 latent robust radius 对 eval 的解释力有限，不如 `predictor_rollout_T8_l2`（两任务均中等正相关）。

结果保存：`/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}/repr_analysis/latent_noise_diagnostics/`

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
> export STABLEWM_HOME=/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}
> python run_planning_action_probe.py
> ```

**PushT 与 TwoRoom 的关键差异**

1. **LeWM-fixed-std 在 PushT 上是 robust，在 TwoRoom 上是 fragile,clustered。**  
   这说明 LeWM 的聚簇化是**任务依赖**的：TwoRoom 低维状态空间容易被压缩成紧凑等价类，PushT 高维连续状态空间难以被简单聚簇。

2. **SWM-fixed-std 在 PushT 上仍然是 fragile,high_angle_gain。**  
   这说明 SWM 的 fixed-std 高角向增益是**结构性风险**（球面 + uniformity + 固定 std 的组合），与任务不完全绑定。PushT 表里它没有被规则标成 `clustered`，但 `clean_nn_cos_dist=0.0664` 明显小于 per-frame SWM 的 0.26–0.28，仍显示出强压缩倾向。

3. **SWM-perframe-0to001 在 PushT 上被评为 robust，0to002 是 balanced。**  
   在 PushT 上，0to001 的 noise 强度已经足够产生 robust geometry（radius≈0.07），而 0to002 的 robust_radius=NaN（更 robust 但 clean eval 从 87.3 降至 81.3）。这与 TwoRoom 上 0to005 才达到 balanced 形成对比，说明 PushT 的"最优 noise 强度"确实更低。

结果保存：`/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}/repr_analysis/p03_diagnostics/`

#### P0.4 相关性分析（跨任务对比结论）

> 完整自动化相关性表见 P0.7（来自 `diagnostic_correlation.py`，n=8 / n=11，含 95% bootstrap CI）。本节只给跨任务对比与高层结论，不重复列指标。

**核心：诊断指标的任务特异性**

| 指标 | TwoRoom (r / ρ) | PushT (r / ρ) | Reacher (r / ρ) | 含义 |
|---|---:|---:|---:|---|
| `clean_nn_cos_dist_median` ↔ eval | **−0.80 / −0.91** | **+0.71 / +0.18** | −0.10 / −0.32 | TwoRoom 强负相关，PushT/Reacher **几乎不相关**；聚簇化不是高维连续控制任务的瓶颈 |
| `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | +0.52 / +0.14 | **−0.86 / −0.79** | **−0.86 / −0.68** | PushT/Reacher **最强信号**：predictor target shift 控制与 eval 负相关 |
| `predictor_rollout_T8_l2` ↔ eval | **+0.37 / +0.67** | **+0.28 / +0.64** | −0.62 / −0.36 | TwoRoom/PushT 强正相关；Reacher 方向反转（弱负相关） |
| `clean_effective_rank` ↔ eval | **−0.62 / −0.60** | **+0.78 / +0.66** | +0.04 / +0.27 | **符号反转**：TwoRoom 需要低维，PushT 需要高维；Reacher 几乎不相关 |
| `lidar_rank` ↔ eval | **−0.93 / −0.81** | **+0.25 / +0.13** | −0.22 / −0.45 | TwoRoom 强负相关，PushT/Reacher 弱相关 |
| `noise_angle_slope_deg_per_std` ↔ eval | **+0.54 / +0.67** | **−0.87 / −0.34** | **−0.91 / −0.52** | TwoRoom 可容忍高角向增益，PushT/Reacher 惩罚高角向增益 |

> **重大修正记录**（commit 8605bf5 → bf79a80 → 4ce4931）：PushT 相关性经历三次修正。第一次（8605bf5）将 SWM-fixed-std eval 从误标 89.8 修正为 61.8。第二次（bf79a80）补齐 11 个缺失 eval 后发现 \|ρ\| 均不足。第三次（4ce4931）将 perframe 模型 eval 从 `num_eval=50` 重跑为 `num_eval=150` 后，PushT `predictor_target_to_nn_cos_ratio` 升至 ρ=−0.791，`predictor_rollout_T8_l2` 升至 ρ=+0.636，**接近强相关阈值**。当前 PushT 预测力已显著改善，但仍需 P0.6 holdout 盲测验证。

结论：
- **TwoRoom 瓶颈**：encoder geometry（聚簇化 / 维度控制）+ predictor 稳定性。
- **PushT 瓶颈**：predictor 稳定性（target shift 控制、rollout drift）+ 有效维度（方向相反）。
- **Reacher 瓶颈**：相关性整体较弱（|ρ| 最高 0.69），主要信号为 `predictor_target_to_nn_cos_ratio`（ρ=−0.68）和 `cka_linear`（ρ=+0.58）。原因可能是 Reacher 任务难度较低，per-frame 训练把 base 从 58.7 拉到 72–83，压缩了模型间 variance，导致诊断指标区分度下降。
- **跨任务通用指标**：`predictor_target_to_nn_cos_ratio_at_max_std` 在 PushT（ρ=−0.79）和 Reacher（ρ=−0.68）均为最强信号，TwoRoom 上弱（ρ=+0.14）。`predictor_rollout_T8_l2` 在 TwoRoom（ρ=+0.67）和 PushT（ρ=+0.64）强正相关，但 Reacher 上方向反转（ρ=−0.36）。
- **不通用指标**：`lidar_rank`、`clean_nn_cos_dist`、`noise_angle_slope`、`clean_effective_rank` 任务依赖性强。
- **Cube**：base LeWM eval 待补齐（当前 num_eval=10 近似 90.0），diagnostics 10/10 模型已完成，但 base LeWM 的 num_eval=150 eval 仍报错，相关性分析待 base eval 补齐后进行。

clean eval 与 noise robustness 在 TwoRoom 不是简单正相关：SWM fixed-std 走"聚簇化 clean bonus / noise fragile"路径，LeWM per-frame 走"平滑且 clean 不差"路径——两条路径必须用诊断指标分开归因（详 §4.2）。

**局限**：PushT 中 `noise_robust_radius_std` 仅 n=6（per-frame 模型 radius>0.08 censored），需扩到 Cube / Reacher 才能评估稳定性。

**图表**：`p0_correlation_{tworoom,pusht}.png`、`predictor_drift_eval_correlation.png`、`noise_angle_curve_goal.png`、`noise_ratio_curve_goal.png`、`geometry_tradeoff_goal.png`。  
保存路径：`/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}/repr_analysis/p03_diagnostics/`。

![TwoRoom P0 诊断指标与 eval 相关性](assets/diagnostics/p0_correlation_tworoom.png)
![PushT P0 诊断指标与 eval 相关性](assets/diagnostics/p0_correlation_pusht.png)
![Predictor Drift 与 Eval 相关性（双任务）](assets/diagnostics/predictor_drift_eval_correlation.png)
![Noise Angle 曲线](assets/diagnostics/noise_angle_curve_goal.png)
![Noise Ratio 曲线](assets/diagnostics/noise_ratio_curve_goal.png)
![Geometry Tradeoff 散点](assets/diagnostics/geometry_tradeoff_goal.png)

#### P0.5 决策标准（按实际数据评估）

| 任务 | 指标 | Spearman \|ρ\| | 判定 | 行动 |
|---|---:|---:|---|---|
| TwoRoom | `clean_nn_cos_dist_median` ↔ eval | **−0.905** | ≥ 0.7 强相关 | **主指标**：encoder 聚簇化/压缩是 TwoRoom clean eval 的主要解释机制 |
| TwoRoom | `transition_resolution_ratio_cos` ↔ eval | **−0.881** | ≥ 0.7 强相关 | **主指标**：transition 分辨率越高，eval 越低 |
| TwoRoom | `noise_robust_radius_std` ↔ eval | **−1.000** (n=4) | ≥ 0.7 强相关 (n=4) | **待验证**：样本量过小，需扩到 Cube / Reacher |
| TwoRoom | `latent_predictor_rollout_T8_l2_history` ↔ eval | **+0.738** | 0.4–0.7 中等 | **辅助**：latent-noise predictor drift 与 eval 正相关 |
| TwoRoom | `predictor_rollout_T8_l2` ↔ eval | **+0.667** | 0.4–0.7 中等 | **辅助**：input-space predictor drift 与 eval 正相关 |
| PushT | `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | **−0.791** | 0.4–0.7 中等 | **候选主指标**：target shift 控制与 eval 负相关，接近 ≥0.7 |
| PushT | `clean_effective_rank` ↔ eval | **+0.664** | 0.4–0.7 中等 | **候选主指标**：有效维度与 eval 正相关，接近 ≥0.7 |
| PushT | `noise_robust_radius_std` ↔ eval | **+0.543** (n=6) | 0.4–0.7 中等 (n=6) | **待验证**：n=6，方向与 TwoRoom 相反 |
| PushT | `latent_predictor_rollout_T8_l2_history` ↔ eval | **+0.627** | 0.4–0.7 中等 | **辅助**：latent-noise predictor drift 与 eval 正相关 |
| PushT | `lidar_rank` ↔ eval | **0.127** | < 0.4 弱 | **不在覆盖范围** |
| PushT | `clean_nn_cos_dist` ↔ eval | **0.091** | < 0.4 弱 | **不在覆盖范围** |

下一步行动（按上表派生）：
1. 论文中以 `clean_nn_cos_dist_median` 作 TwoRoom 主指标，`predictor_target_to_nn_cos_ratio_at_max_std` 作 PushT 主指标，`predictor_rollout_T8_l2` 作跨任务通用辅助指标。
2. 对 SWM 做 predictor 结构 ablation（P3），验证 target shift 控制是否随 predictor depth/normalization 变化。
3. 扩到 Cube / Reacher 后重新评估 `predictor_rollout_T8_l2` 是否真的是通用信号。

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

**自动化相关性结果（`diagnostic_correlation.py` 输出）**

TwoRoom（n=8，baselines 已补齐）：

| 指标 | Pearson r | Spearman ρ | 95% CI | 解释 |
|---|---:|---:|---|---|
| `clean_nn_cos_dist_median` ↔ eval | −0.800 | **−0.905** | [−0.976, −0.191] | **最强预测指标**：encoder 聚簇化越紧，clean eval 越高 |
| `noise_robust_radius_std` ↔ eval | −0.963 | **−1.000** | [−1.000, −0.200] | n=4，noise robust radius 越小，eval 越低 |
| `transition_resolution_ratio_cos` ↔ eval | −0.775 | **−0.881** | [−0.952, −0.191] | transition 分辨率（cos 相似度）越高，eval 越低 |
| `transition_resolution_ratio_l2` ↔ eval | −0.795 | **−0.881** | [−0.952, −0.191] | transition 分辨率（L2 相似度）越高，eval 越低 |
| `lidar_rank` ↔ eval | −0.925 | **−0.810** | [−0.952, −0.095] | 有效维度越多，eval 越低（PushT 方向几乎不相关） |
| `latent_predictor_rollout_T8_l2_history` ↔ eval | +0.591 | **+0.738** | [−0.072, +1.000] | latent-noise 下 history scope predictor drift 与 eval 强正相关 |
| `latent_rollout_l2_slope_per_std_z` ↔ eval | +0.599 | **+0.738** | [−0.024, +0.976] | latent rollout L2 slope 与 eval 正相关 |
| `noise_angle_slope_deg_per_std` ↔ eval | +0.538 | **+0.667** | [−0.262, +1.000] | noise 角度 slope 越大，eval 越高 |
| `predictor_rollout_T8_l2` ↔ eval | +0.371 | **+0.667** | [−0.167, +1.000] | input-space predictor drift 与 eval 正相关 |
| `id_probe_r2` ↔ eval | −0.425 | **−0.619** | [−0.857, +0.286] | ID linear probe R² 越高，eval 越低 |
| `id_probe_r2_min` ↔ eval | −0.442 | **−0.619** | [−0.857, +0.262] | ID probe min R² 越高，eval 越低 |
| `latent_robust_radius_z` ↔ eval | −0.645 | **−0.619** | [−0.929, +0.192] | 中等负相关，但 CI 仍宽 |
| `clean_effective_rank` ↔ eval | −0.624 | **−0.595** | [−0.929, +0.310] | 有效维度越多，eval 越低（PushT 方向几乎不相关） |

Reacher（n=10，全 epoch_9，num_eval=150）：

| 指标 | Pearson r | Spearman ρ | 95% CI | 解释 |
|---|---:|---:|---|---|
| `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | −0.863 | **−0.685** | [−0.939, +0.128] | **最强信号**（与 PushT 一致）：predictor 稳定性越好，eval 越高 |
| `cka_linear_at_max_std` ↔ eval | +0.952 | **+0.576** | [−0.261, +0.952] | noise 下 latent CKA 越高，eval 越高 |
| `noise_angle_slope_deg_per_std` ↔ eval | −0.914 | **−0.515** | [−0.843, +0.309] | noise 角度 slope 与 eval 负相关 |
| `lidar_rank` ↔ eval | −0.218 | −0.455 | [−0.794, +0.431] | 弱负相关 |
| `latent_cost_surface_slope_z` ↔ eval | +0.126 | +0.382 | [−0.503, +0.879] | 弱正相关 |
| `clean_nn_cos_dist_median` ↔ eval | −0.104 | −0.321 | [−0.806, +0.503] | 弱负相关 |
| `predictor_rollout_T8_l2` ↔ eval | −0.616 | −0.358 | [−0.806, +0.516] | 负相关（与 TwoRoom/PushT 方向相反） |
| `clean_effective_rank` ↔ eval | +0.037 | +0.273 | [−0.382, +0.770] | 几乎不相关 |
| `id_probe_r2` ↔ eval | +0.371 | +0.285 | [−0.600, +0.830] | 弱正相关 |
| `transition_resolution_ratio_cos` ↔ eval | +0.049 | −0.127 | [−0.794, +0.746] | 几乎不相关 |

> Reacher 整体相关性弱于 TwoRoom/PushT，|ρ| 最高仅 0.69。可能原因：Reacher 任务本身难度较低（base 仅 58.67），per-frame 训练把分数拉到 72–83，压缩了模型间的 variance，导致诊断指标区分度下降。

PushT（n=11，baselines 已补齐，SWM-fixed-std 真实 eval=61.8）：

| 指标 | Pearson r | Spearman ρ | 95% CI | 解释 |
|---|---:|---:|---|---|
| `predictor_target_to_nn_cos_ratio_at_max_std` ↔ eval | **−0.863** | **−0.791** | [−0.918, −0.382] | **最强信号**：noise 下 target shift 相对 NN 越小，eval 越高 |
| `clean_effective_rank` ↔ eval | +0.784 | **+0.664** | [+0.009, +0.973] | 有效维度越多，eval 越高（与 TwoRoom 方向相反） |
| `noise_robust_radius_std` ↔ eval | +0.788 | **+0.543** | [−0.429, +1.000] | n=6，noise robust radius 越大，eval 越高（与 TwoRoom 相反） |
| `latent_predictor_rollout_T8_l2_history` ↔ eval | +0.534 | **+0.627** | [−0.109, +0.991] | latent-noise 下 history scope predictor drift 与 eval 强正相关 |
| `predictor_rollout_T8_l2` ↔ eval | +0.282 | **+0.636** | [+0.063, +0.882] | input-space predictor drift 与 eval 强正相关 |
| `latent_rollout_l2_slope_per_std_z` ↔ eval | +0.508 | **+0.582** | [−0.073, +0.973] | latent rollout L2 slope 与 eval 正相关 |
| `latent_cost_surface_slope_z` ↔ eval | +0.518 | **+0.673** | [+0.109, +0.936] | latent cost surface slope 与 eval 正相关 |
| `noise_angle_slope_deg_per_std` ↔ eval | −0.867 | **−0.336** | [−0.782, +0.364] | noise 角度 slope 越大，eval 越低（与 TwoRoom 方向相反） |
| `latent_robust_radius_z` ↔ eval | +0.635 | **+0.300** | [−0.545, +0.882] | 弱正相关，CI 宽 |
| `transition_resolution_ratio_l2` ↔ eval | +0.262 | **+0.236** | [−0.546, +0.782] | 几乎不相关 |
| `transition_resolution_ratio_cos` ↔ eval | +0.226 | **+0.227** | [−0.527, +0.791] | 几乎不相关 |
| `id_probe_r2` ↔ eval | +0.050 | **+0.355** | [−0.282, +0.945] | 弱正相关 |
| `id_probe_r2_min` ↔ eval | +0.298 | **+0.382** | [−0.300, +0.909] | 弱正相关 |
| `clean_nn_cos_dist_median` ↔ eval | +0.707 | **+0.182** | [−0.618, +0.800] | **几乎不相关**：聚簇化不解释 PushT eval |
| `lidar_rank` ↔ eval | +0.250 | **+0.127** | [−0.582, +0.855] | **几乎不相关**：有效维度不解释 PushT eval |

> 上面是按 |Spearman ρ| 排序的 top 字段。跨任务对比（含哪个指标在 TwoRoom / PushT 上是主导信号）见 P0.4 综合结论表，不再重复。

保存路径：`/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}/repr_analysis/p03_diagnostics/diagnostic_correlation.{csv,png,summary.json}`

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

- `raw + mse` 只带来小幅回升（+6），没有接近 clean SWM（90.8）或 LeWM std=0.03（90）。
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
- `noise_geometry ∈ {ambient, tangent}`；tangent 用于 SWM 切空间扰动。
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
| `noise_l2`, `noise_cos_dist`, `noise_angle_deg` | 同一 observation 加 pixel noise 前后的 encoder shift；球面模型重点看 angle | empirical Jacobian / local Lipschitz probe；Hoffman 2019, Virmaux & Scaman 2018 | `tools/repr_analysis/noise_sensitivity.py::_shift_metrics` |
| `clean_nn_cos_dist`, `clean_nn_l2` | clean latent 的最近邻尺度，作为“噪声是否推出局部邻域”的分母 | KNN-OOD / distance-aware feature primitive；Sun 2022, Liu 2020 | `noise_sensitivity.py::_pairwise_reference`; `predictor_sensitivity.py::_clean_nn_dist` |
| `clean_pair_cos_dist`, `clean_pair_l2` | clean latent 的全配对距离，用作 uniformity / anisotropy 参考 | Wang & Isola uniformity；Ethayarajh anisotropy | `noise_sensitivity.py::_pairwise_reference` |
| `clean_effective_rank` | latent 协方差谱的有效维度，用来区分 collapse、clustered、balanced | RankMe / matrix entropy；Garrido 2023；LiDAR 相关但不主张新颖性 | `analyze_repr.effective_rank`; `task_resolution.py` roll-up |
| `cka_linear_clean_vs_noisy` | clean/noisy 表征子空间对齐；补充 per-point shift | CKA；Kornblith 2019 | `noise_sensitivity.py::_linear_cka` |
| `noise_to_nn_*_ratio` | encoder shift 除以 clean NN 尺度；ratio ≥ 1 表示噪声跨过局部邻域 | composite 指标，可作为 planning-latent robustness ratio 主张 | `noise_sensitivity.py::analyze_model_noise` |
| `robust_radius_std`, `first_high_risk_std` | `noise_to_nn_cos_ratio` 跨过 1 的插值 / 首次离散 std | randomized smoothing 的 empirical planning-latent 版本；Cohen 2019 是来源，不是同一 certified setting | `noise_sensitivity.py::summarize_noise_geometry` |
| `noise_angle_slope_deg_per_std`, `noise_ratio_slope_per_std` | 小 std 附近的 angular gain / ratio gain | local Lipschitz / spectral norm 思路的球面诊断版本 | `noise_sensitivity.py::_near_zero_slope` |
| `geometry_flag`, `recommendation` | 按 radius、angle gain、NN distance、effective rank 给出的经验几何标签和建议 | 工程规则；不是论文 novelty 单独主张 | `noise_sensitivity.py::_geometry_flags`; `_recommendation` |
| `predictor_target_shift`, `target_to_nn_*_ratio` | noisy history 经过 predictor 后的 single-step target shift；ratio 版本跨模型可比 | single-step rollout error 来自 Dreamer / TD-MPC family；ratio 是本文 composite | `tools/repr_analysis/predictor_sensitivity.py::_open_loop_target_shift`; `analyze_model_predictor_noise` |
| `predictor_rollout_drift(T)` | noisy vs clean history 自回归 T 步后的 latent drift | multi-step noise-vs-clean conditioning，文献无直接对应，可主张 novelty；不同于 Dreamer/TD-MPC 对 ground-truth latent 的 rollout MSE | `predictor_sensitivity.py::_autoregressive_rollout` |
| `transition_resolution_ratio` | 相邻帧距离 / 跨序列随机帧距离；衡量 latent 是否保留任务分辨率 | temporal-neighbor 版本的 intra/inter gap；命名和 planning 用法可主张新颖 | `tools/repr_analysis/task_resolution.py::_transition_metrics` |
| `id_probe_r2`, `id_probe_r2_min` | 只训练 linear readout，用 `(z_t,z_{t+1})` 预测 action，代理 action-relevant state information | inverse-dynamics representation probe；Brandfonbrener 2023, Pathak 2017, Alain & Bengio 2017 | `task_resolution.py::_ridge_probe`; `_build_id_probe_data` |
| `lidar_rank` | 用相邻帧作 positive pair 的 LiDAR rank | LiDAR；Thilak 2024，本文只是迁移到 temporal pair | `task_resolution.py::_lidar_rank` |
| `predictor_rollout_drift_z(T)`, `target_to_nn_*_ratio_z` | 直接在 encoded `z` 加噪，剥离 encoder 后测 predictor smoothness | latent randomized smoothing / RobustZero 相关；本文用于 post-hoc encoder-decoupled diagnostic | `tools/repr_analysis/latent_noise_sensitivity.py::_open_loop_target_shift`; `_autoregressive_rollout` |
| `cost_surface_slope_z`, `robust_radius_z` | latent 噪声下 planning cost / predictor 边界的经验斜率与半径 | Cohen 2019 / Lipschitz estimation 思路迁移到 latent cost；不是 certified bound | `latent_noise_sensitivity.py::analyze_model_latent_noise`; `summarize_latent_noise_geometry` |
| `pearson_r`, `spearman_rho`, bootstrap CI | 诊断指标与 eval score 的相关性和置信区间 | label-free performance prediction / ATC family；Garg 2022, Deng & Zheng 2021, Efron & Tibshirani 1993 | `tools/repr_analysis/diagnostic_correlation.py` |
| roll-up summary | 把 encoder、predictor、task-resolution、latent-noise 指标汇成一行/ckpt | 本项目工程聚合层 | `tools/repr_analysis/run_full_diagnostics.py::_summarize_noise_to_predictor_to_resolution` |

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

> **本附录存在的意义**：plan_v3.md §6 P0.4/P0.5/P0.7 的所有相关性数值均来自 `diagnostic_correlation.py` 对 `eval_scores.json` + `diagnostics_summary.json` 的自动计算。本附录逐条记录这 19 个模型对应的 ckpt 子目录、eval 分数来源、诊断指标来源，确保任何数值都可以从原始 ckpt 文件一路追溯到报告中的 ρ 值。

### A.1 TwoRoom（8 模型，全 epoch_9）

| 模型名 | CKPT 子目录 | 对象文件名 | Eval 分数 | Eval 来源 | `clean_nn_dist` | `eff_rank` | `geometry_flag` |
|---|---|---|---|---:|---:|---:|---|
| LeWM-base | `tworoom_lewm` | `tworoom_lewm_epoch_9_object.ckpt` | 93.0 | `tworoom_results.txt` | 0.03890 | 29.54 | balanced |
| LeWM-fixed-std | `tworoom_lewm_noise_std_0_005` | `tworoom_lewm_noise_std_0_005_epoch_9_object.ckpt` | 96.6 | `tworoom_results.txt` | 0.01295 | 15.08 | fragile,clustered |
| LeWM-perframe-p05 | `tworoom_lewm_noise_0to005_p05` | `tworoom_lewm_noise_0to005_p05_epoch_9_object.ckpt` | 94.0 | `eval_run.log`（run_missing_evals 重跑） | 0.03705 | 27.36 | balanced |
| LeWM-perframe-p1 | `tworoom_lewm_noise_0to005_p1` | `tworoom_lewm_noise_0to005_p1_epoch_9_object.ckpt` | 94.0 | `eval_run.log`（run_missing_evals 重跑） | 0.03571 | 26.58 | balanced |
| SWM-base | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425_epoch_9_object.ckpt` | 91.0 | `tworoom_results.txt` | 0.05943 | 29.04 | balanced |
| SWM-fixed-std | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64_epoch_9_object.ckpt` | 97.6 | `tworoom_results.txt` | 0.00820 | 11.61 | fragile,high_angle_gain,clustered |
| SWM-perframe-p05 | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt` | 87.33 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.04984 | 26.96 | balanced |
| SWM-perframe-p1 | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64` | `tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_9_object.ckpt` | 86.67 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.04775 | 26.89 | balanced |

### A.2 PushT（11 模型，全 epoch_9）

| 模型名 | CKPT 子目录 | 对象文件名 | Eval 分数 | Eval 来源 | `clean_nn_dist` | `eff_rank` | `geometry_flag` |
|---|---|---|---|---:|---:|---:|---|
| LeWM-base | `pusht_lewm_20260430` | `pusht_lewm_20260430_epoch_9_object.ckpt` | 80.67 | `pusht_results.txt` | 0.23599 | 47.48 | robust |
| LeWM-fixed-std | `pusht_lewm_noise_std_0_005` | `pusht_lewm_noise_std_0_005_epoch_9_object.ckpt` | 83.0 | `pusht_results.txt` | 0.14473 | 31.40 | robust |
| LeWM-perframe-0to001-p1 | `pusht_lewm_noise_0to001_p1` | `pusht_lewm_noise_0to001_p1_epoch_9_object.ckpt` | 87.33 | `eval_run.log`（run_missing_evals 重跑） | 0.22625 | 48.36 | balanced |
| LeWM-perframe-0to002-p1 | `pusht_lewm_noise_0to002_p1` | `pusht_lewm_noise_0to002_p1_epoch_9_object.ckpt` | 89.33 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.24733 | 48.28 | balanced |
| LeWM-perframe-0to005-p1 | `pusht_lewm_noise_0to005_p1` | `pusht_lewm_noise_0to005_p1_epoch_9_object.ckpt` | 82.0 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.22531 | 46.74 | balanced |
| SWM-base | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260430` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260430_epoch_9_object.ckpt` | 77.33 | `pusht_results.txt` | 0.26449 | 44.02 | robust |
| SWM-fixed-std | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64_epoch_9_object.ckpt` | 61.8 | `pusht_results.txt` | 0.06639 | 18.38 | fragile,high_angle_gain |
| SWM-perframe-0to001-p05 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p05_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p05_dim64_epoch_9_object.ckpt` | 78.0 | `eval_run.log`（run_missing_evals 重跑） | 0.25770 | 42.62 | robust |
| SWM-perframe-0to001-p1 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64_epoch_9_object.ckpt` | 87.33 | `eval_run.log`（run_missing_evals 重跑） | 0.28448 | 45.70 | robust |
| SWM-perframe-0to002-p05 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64_epoch_9_object.ckpt` | 78.67 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.27604 | 46.04 | balanced |
| SWM-perframe-0to002-p1 | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64` | `pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_9_object.ckpt` | 81.33 | `eval_run.log`（run_missing_evals 重跑，num_eval=150） | 0.26000 | 45.46 | balanced |

### A.3 Reacher（10 模型，全 epoch_9，num_eval=150）

| 模型名 | CKPT 子目录 | 对象文件名 | Eval 分数 | Eval 来源 | `clean_nn_dist` | `eff_rank` | `geometry_flag` |
|---|---|---|---|---:|---:|---:|---|
| LeWM-base | `reacher_lewm_20260430` | `reacher_lewm_20260430_epoch_9_object.ckpt` | 58.67 | `summary.txt` | 0.05987 | 42.95 | balanced |
| LeWM-perframe-0to002-p05 | `reacher_lewm_noise_0to002_p05` | `reacher_lewm_noise_0to002_p05_epoch_9_object.ckpt` | 79.33 | `summary.txt` | 0.07261 | 49.44 | balanced |
| LeWM-perframe-0to002-p1 | `reacher_lewm_noise_0to002_p1` | `reacher_lewm_noise_0to002_p1_epoch_9_object.ckpt` | 72.67 | `summary.txt` | 0.07199 | 47.99 | balanced |
| LeWM-perframe-0to005-p05 | `reacher_lewm_noise_0to005_p05` | `reacher_lewm_noise_0to005_p05_epoch_9_object.ckpt` | 82.67 | `summary.txt` | 0.06902 | 45.99 | balanced |
| LeWM-perframe-0to005-p1 | `reacher_lewm_noise_0to005_p1` | `reacher_lewm_noise_0to005_p1_epoch_9_object.ckpt` | 78.00 | `summary.txt` | 0.05805 | 32.87 | balanced |
| SWM-base | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260430` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260430_epoch_9_object.ckpt` | 54.67 | `summary.txt` | 0.09940 | 44.17 | balanced |
| SWM-perframe-0to002-p05 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64_epoch_9_object.ckpt` | 78.00 | `summary.txt` | 0.09063 | 43.78 | balanced |
| SWM-perframe-0to002-p1 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_9_object.ckpt` | 74.00 | `summary.txt` | 0.09335 | 42.87 | balanced |
| SWM-perframe-0to005-p05 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt` | 76.00 | `summary.txt` | 0.09451 | 45.60 | balanced |
| SWM-perframe-0to005-p1 | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64` | `reacher_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_9_object.ckpt` | 78.00 | `summary.txt` | 0.09647 | 44.48 | balanced |

### A.4 Cube（10 模型，全 epoch_9；LeWM-base num_eval=10，其余 num_eval=150）

| 模型名 | CKPT 子目录 | 对象文件名 | Eval 分数 | Eval 来源 | `clean_nn_dist` | `eff_rank` | `geometry_flag` |
|---|---|---|---|---:|---:|---:|---|
| LeWM-base | `cube_lewm_20260430` | `cube_lewm_20260430_epoch_9_object.ckpt` | 90.0 | `clean_10.log`（num_eval=10 近似） | 0.18774 | 48.92 | robust |
| LeWM-perframe-0to002-p05 | `cube_lewm_noise_0to002_p05` | `cube_lewm_noise_0to002_p05_epoch_9_object.ckpt` | 64.67 | `summary.txt` | 0.13502 | 49.53 | balanced |
| LeWM-perframe-0to002-p1 | `cube_lewm_noise_0to002_p1` | `cube_lewm_noise_0to002_p1_epoch_9_object.ckpt` | 60.67 | `summary.txt` | 0.13336 | 49.20 | balanced |
| LeWM-perframe-0to005-p05 | `cube_lewm_noise_0to005_p05` | `cube_lewm_noise_0to005_p05_epoch_9_object.ckpt` | 66.00 | `summary.txt` | 0.11817 | 47.31 | balanced |
| LeWM-perframe-0to005-p1 | `cube_lewm_noise_0to005_p1` | `cube_lewm_noise_0to005_p1_epoch_9_object.ckpt` | 64.67 | `summary.txt` | 0.11534 | 45.61 | balanced |
| SWM-base | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260430` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260430_epoch_9_object.ckpt` | 78.00 | `summary.txt` | 0.24280 | 43.51 | robust |
| SWM-perframe-0to002-p05 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64_epoch_9_object.ckpt` | 72.00 | `summary.txt` | 0.26564 | 43.51 | balanced |
| SWM-perframe-0to002-p1 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_9_object.ckpt` | 74.00 | `summary.txt` | 0.25471 | 44.76 | balanced |
| SWM-perframe-0to005-p05 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt` | 70.67 | `summary.txt` | 0.19656 | 43.68 | balanced |
| SWM-perframe-0to005-p1 | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64` | `cube_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_9_object.ckpt` | 64.00 | `summary.txt` | 0.16893 | 42.79 | balanced |

> **注**：Cube diagnostics 10/10 已补齐（2026-05-02 重跑），base LeWM 的 num_eval=150 eval 仍待修复。

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

### A.4 人工核验检查清单

你可以按以下步骤独立复现任何数值：

1. **核验 Eval 分数**
   ```bash
   # TwoRoom
   cat /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/<subdir>/tworoom_results.txt
   # PushT
   cat /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt/<subdir>/pusht_results.txt
   # 若模型由 run_missing_evals.py 重跑，则查看 eval_run.log 中 'success_rate': <num>
   ```

2. **核验诊断指标（原始 CSV）**
   ```bash
   # TwoRoom
   cat /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/repr_analysis/p03_diagnostics/noise_sensitivity.csv | grep <model>
   # PushT（含 base 补做）
   cat /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht/repr_analysis/p03_diagnostics/noise_sensitivity.csv | grep <model>
   cat /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht/repr_analysis/p03_diagnostics_new_baselines/<model>/noise_sensitivity.csv | grep <model>
   ```

3. **核验相关性数值**
   ```bash
   cat /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-{tworooms,pusht}/repr_analysis/p03_diagnostics/diagnostic_correlation.csv
   # 或重新运行脚本
   python -m tools.repr_analysis.diagnostic_correlation \
       --diagnostics /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/repr_analysis/p03_diagnostics/diagnostics_summary.json \
       --eval-scores /opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/repr_analysis/p03_diagnostics/eval_scores.json \
       --out-dir /tmp/tworoom_corr_check
   ```

4. **关键字段定义速查**
   - `clean_nn_cos_dist_median`：clean embedding 最近邻 cosine 距离中位数（越小 = 聚簇越紧）
   - `clean_effective_rank`：clean embedding 的有效维度（Wang & Isola uniformity 相关）
   - `predictor_target_to_nn_cos_ratio_at_max_std`：最大噪声 std 下，predictor target 与 nearest neighbor 的 cosine ratio（越小 = target shift 控制越好）
   - `geometry_flag`：`run_full_diagnostics.py` 自动标注的 noise geometry 类别（balanced / robust / fragile / clustered 等）

### A.5 历史数据修正记录

| 时间 | Commit | 修正内容 | 影响 |
|---|---|---|---|
| 2026-05-01 02:30 | `8605bf5` | SWM-fixed-std PushT eval 89.8→61.8；LeWM-fixed-std 83.6→83.0；TwoRoom SWM-base 90.8→91.0；LeWM-fixed-std 95.6→96.6 | PushT 主导指标从 `lidar_rank` 变为 `predictor_target_to_nn_cos_ratio` 和 `clean_effective_rank` |
| 2026-05-01 04:16 | `620de01` | 运行 `run_missing_evals.py`，补齐 11 个缺失 eval（4 TwoRoom + 7 PushT），actual 与 expected 多处不符 | `eval_scores.json` 更新，但 plan_v3.md **未同步更新相关性数值** |
| 2026-05-01 04:33 | `bf79a80` | 插入诊断可视化配图 | 仅新增图片引用，未修正数值 |
| 2026-05-01 05:07 | `6c7bf90` | 将 plan_v3.md 中所有 Spearman/Pearson 数值更新为与 `diagnostic_correlation.csv` 一致（基于 num_eval=50 的 perframe 模型） | PushT 所有指标 \|ρ\| 降至 0.4–0.6，无 ≥0.7 强相关；TwoRoom `clean_nn_cos_dist` 升至 −1.000 |
| 2026-05-01 12:09 | `4ce4931` | **第三次修正**：发现 perframe 模型 eval 仅用 `num_eval=50`，将 11 个 perframe 模型重跑为 `num_eval=150` | TwoRoom perframe eval 更稳定（SWM-perframe-p05 92.0→87.33，SWM-perframe-p1 92.0→86.67）；PushT `predictor_target_to_nn_cos_ratio` 从 −0.564 升至 **−0.791**，`predictor_rollout_T8_l2` 从 +0.300 升至 **+0.636**，PushT 预测力显著改善 |
