# 球面世界模型实验计划 V3

> 当前定位：本文不是单纯记录“SWM 是否强于 LeWM”，而是整理一个更稳定的研究路线：**world model 的 latent geometry 如何匹配 planning 任务的状态分辨率需求**。  
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

## 2. Clean Benchmark 状态

4-task benchmark，epoch=10，num_eval=500，single seed：

| Task | LeWM | SWM best | Delta |
|---|---:|---:|---:|
| TwoRoom | 93.0 | 90.8 | -2.2 |
| Cube | 69.2 | 74.0 | +4.8 |
| PushT | 89.4 | 89.8 | +0.4 |
| Reacher | 62.2 | 66.0 | +3.8 |
| Average | 78.5 | 80.2 | +1.7 |

结论：

- SWM clean performance 不差，甚至在平均上略高。
- 但优势不够稳定，不能支撑“球面空间全局更好”的叙事。
- 更有价值的方向是分析为什么某些任务受益、某些任务不受益。

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

### 3.2 Noise Sensitivity 指标

`tools/repr_analysis/noise_sensitivity.py` 用同一批图像比较 clean/noisy embedding：

| Metric | 含义 |
|---|---|
| `noise_angle_deg_median` | clean/noisy embedding 的角向偏移 |
| `clean_nn_cos_dist_median` | clean embedding 最近邻间距 |
| `noise_to_nn_cos_ratio_median` | noise shift / clean nearest-neighbor distance |
| `robust_radius` | ratio 跨过 1 时的 std |

Goal frame, normalized space：

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

关键点：

- SWM 小噪声角向偏移约为 LeWM 的 3x。
- SWM 的 clean nearest-neighbor distance 反而更大，不是“embedding 太密”导致脆弱。
- 经验 robust radius：
  - LeWM 约 `std=0.017`
  - SWM 约 `std=0.008`

### 3.3 当前失败机制假设

SWM 的 noise failure 是两段串联：

1. **Encoder angular sensitivity 高。**  
   可疑因素：BN projector、L2 normalization 在小 norm 处放大、dim=64 比 dim=192 更容易产生方向扰动。

2. **Cosine planning cost 大角度下信息不足。**  
   当 noisy goal 已经偏到 70° 左右，`1 - cos` 接近饱和，planner 对错误 goal direction 的修正能力下降。

P2 会检验第二点：同一 checkpoint eval-only 改成 `raw + mse` cost，看 noisy score 是否回升。

---

## 4. Noise-Aware Training 结果

### 4.1 P1 第一组：SWM noise training

TwoRoom SWM，noise augmentation，`std_min=0, std_max=0.05`。

Eval 结果：

| Task | corruption | apply_to | score |
|---|---|---|---:|
| TwoRoom | std=0 | - | 97.6 |
| TwoRoom | std=0.05 | pixels+goal | 98.0 |
| TwoRoom | std=0.08 | pixels+goal | 88.0 |
| TwoRoom | std=0.05 | pixels only | 56.0 |
| TwoRoom | std=0.05 | goal only | 44.0 |
| PushT | std=0 | - | 61.8 |
| PushT | std=0.05 | pixels+goal | 60.0 |

Noise sensitivity 对照，std=0.005：

| Metric | SWM baseline | SWM noise-train | 变化 |
|---|---:|---:|---:|
| clean NN cos dist | 0.082 | 0.008 | 缩到 1/10 |
| clean NN L2 dist | 0.40 | 0.13 | 缩到 1/3 |
| noise angle median | 11.9° | 27.6° | 变大 |

### 4.2 解释：不是平滑化，而是聚簇化

简单的 Lipschitz hypothesis 被推翻。Noise training 没有降低 encoder 的局部角向 sensitivity；它把 clean 状态压进更紧的等价类。

更准确的解释：

```text
noise augmentation -> clustered / discretized geometry
                   -> 对低维、冗余视觉任务有利
                   -> 对高分辨率连续控制任务有害
```

这解释了三件事：

1. **TwoRoom 提升。**  
   TwoRoom 内在状态低维，视觉细节大多冗余。聚簇化像信息瓶颈，帮助 planner 忽略无关变化。

2. **Asymmetric noise 仍崩。**  
   如果训练 noise 是整段序列一致分布，模型可能学到“所有帧同噪声水平”的隐式 coupling。只给 goal 或 pixels 加噪会破坏这种分布。

3. **PushT 下降。**  
   PushT 需要保留“再推一点”和“已经到位”的细粒度差异。聚簇化合并了这些差异，降低 planning resolution。

### 4.3 P1 补完：per-frame 独立 std + noise_prob

**实验设计**

已实现 `utils.py:AddNormalizedGaussianNoise`：每帧独立经过 Bernoulli(`noise_prob`) 决定是否加噪，如加则 std ~ Uniform(`std_min`, `std_max`)。

补跑完成：
- P1.1：TwoRoom SWM/LeWM per-frame `std∈[0,0.05]`，`noise_prob=1.0 / 0.5`
- P1.2：PushT SWM `std∈[0,0.01]`、`std∈[0,0.02]`，`noise_prob=1.0 / 0.5`
- P1.3：PushT LeWM 同条件对照 `std∈[0,0.01]`、`std∈[0,0.02]`、`std∈[0,0.05]`

**TwoRoom eval（num_eval=150）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWM baseline | 90.8 | — | — | — | — | **36*** | — | — | **66*** | — |
| SWM 旧版固定 std | **97.6** | — | — | — | — | **98.0** | 88.0 | — | 56.0 | — |
| SWM per-frame p1 | 86.7 | 88.0 | 87.3 | 89.3 | 87.3 | 87.3 | 89.3 | 86.7 | 87.3 | 86.7 |
| SWM per-frame p05 | 87.3 | 86.7 | 88.7 | 86.0 | 86.7 | 85.3 | 88.0 | 87.3 | 86.7 | 85.3 |
| LeWM per-frame p1 | **94.0** | 94.0 | 94.0 | 93.3 | 94.0 | 92.7 | 94.7 | 94.0 | 94.0 | 94.0 |
| LeWM per-frame p05 | **94.0** | 94.7 | 94.0 | 94.7 | 94.0 | 94.7 | 94.0 | 94.0 | 94.0 | 94.0 |

> *baseline 为 plan_v3.md §3.1 的 std=0.03 数据。

**PushT eval（num_eval=150）**

| 模型 | clean | goal_0.03 | goal_0.05 | goal_0.08 | pix+goal_0.03 | pix+goal_0.05 | pix+goal_0.08 | pix_0.03 | pix_0.05 | pix_0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWM baseline | 89.8 | — | — | — | — | — | — | — | — | — |
| SWM 0to001 p1 | 87.3 | 86.0 | 79.3 | 27.3 | 86.0 | 59.3 | 6.0 | 82.7 | 69.3 | 6.7 |
| SWM 0to001 p05 | 78.0 | 71.3 | 58.0 | 28.0 | 73.3 | 51.3 | 10.0 | 72.7 | 60.0 | 13.3 |
| SWM 0to002 p1 | 81.3 | 79.3 | 79.3 | **64.0** | 82.0 | 78.0 | **48.7** | 78.0 | 80.7 | **50.7** |
| SWM 0to002 p05 | 78.7 | 78.0 | 77.3 | 41.3 | 77.3 | 72.0 | 12.0 | 81.3 | 74.0 | 22.7 |
| LeWM 0to001 p1 | 87.3 | 88.0 | 76.7 | 64.7 | 84.0 | 69.3 | 40.7 | 84.7 | 75.3 | 48.7 |
| LeWM 0to002 p1 | **89.3** | 88.0 | **86.7** | **82.0** | 88.0 | **85.3** | **74.0** | 87.3 | **86.0** | **76.0** |
| LeWM 0to005 p1 | 82.0 | 81.3 | 77.3 | 80.7 | 80.0 | 80.0 | 78.0 | 83.3 | 78.7 | 76.0 |

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

**更新后的解释**

1. **per-frame 独立 std 彻底修复 asymmetric 崩溃。**  
   SWM per-frame p1/p05 的 pixels-only / goal-only 均维持在 85–89%，相比旧版固定 std 的 56/44 是质的改善。帧间噪声分布不一致性是旧版崩溃的主因。

2. **固定 std → 聚簇化；per-frame → 平滑化。两者互斥。**  
   - 旧版固定 std：clean_nn_dist 缩到 1/10，noise angle 反而增大到 27.6°。TwoRoom clean 大幅提升（90.8→97.6），但 asymmetric 崩溃。
   - 新版 per-frame：noise angle 从 11.9° 降到 0.41°（与 LeWM 同级），asymmetric 修复，但失去 clean 提升（86.7，甚至略低于 baseline 90.8）。

   > 核心结论：noise augmentation 的实现方式决定几何形态。固定全帧噪声导致聚簇化（有益 clean、有害 asymmetric），per-frame 独立噪声导致平滑化（有益 asymmetric、无益 clean）。

3. **LeWM 近乎完美，无聚簇化副作用。**  
   LeWM per-frame 在 TwoRoom 上 clean=94.0，所有噪声条件保持 93–95%，noise angle 仅 0.44°，且 clean_nn_dist 几乎不压缩（0.039→0.036）。聚簇化是 SWM（球面+uniformity）特有的副作用。

4. **PushT 存在 noise sweet spot，但 SWM 仍不如 LeWM。**  
   SWM 最优为 0to002 p1（clean 81.3, goal_0.08=64.0）；LeWM 最优为 0to002 p1（clean **89.3**, goal_0.08=**82.0**）。即使最优强度下，SWM 的 clean 和 noise 鲁棒性均明显落后。

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

> **当前优先级：P0 > P3 > P4**。P1 已完成（§4.3），P2 已结论（§6 P2 末尾，cost saturation 非主因，不再投入）。P0 数据矩阵已经基本就位（§4.3 的 6+ checkpoint × 2 task），先做 P0 相关性分析，再按结论决定 P3 / P4 投入强度。

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

| 层 | 指标 | 状态 | 用途 |
|---|---|---|---|
| Encoder shift | `noise_angle_deg_median/p90`, `noise_l2_median/p90`, `noise_cos_dist_median/p90` | ✅ | encoder 直接 angular / L2 sensitivity |
| Encoder geometry | `clean_nn_cos_dist`, `clean_pair_cos_dist`, `clean_norm_mean` | ✅ | clustering / scale |
| 派生比例 | `noise_to_nn_cos_ratio_median/p90`, `robust_radius_std`, `noise_angle_slope_deg_per_std` | ✅ | 跨模型可比的鲁棒性界 |
| 几何标签 | `geometry_flag` (`clustered/fragile/robust/balanced`) + `recommendation` | ✅ | 经验阈值规则 |
| Encoder 区分 | `effective_rank`（已在 `analyze_repr.py`，未暴露给 `noise_sensitivity`）；`frame_scope="history"` | ⚠️ 待迁 | 区分 collapse vs clustering；history 帧对应 pixels-only failure |
| Predictor side | `predictor_rollout_drift(T)`、`predictor_target_shift` | ❌ 缺 | encoder 之外，predictor 在 noisy history 下的累积漂移 |
| Task resolution | `transition_resolution_ratio = d(z_t, z_{t+1}) / d(z_t, z_far)`、inverse-dynamics linear probe readout | ❌ 缺 | 量化任务所需状态分辨率，区分 TwoRoom 与 PushT 偏好 |
| 目标变量 | `eval_drop_pix+goal`, `eval_drop_goal_only`, `eval_drop_pix_only`（at std=0.03/0.05/0.08） | ✅ 已收 | 三种 noise mode 分别看，对应不同失败机制 |

#### P0.2 缺失诊断的补充实现

1. `tools/repr_analysis/noise_sensitivity.py` 扩展：
   - 增加 `frame_scope="history"`（取 `z[:, :-1]`），与 pixels-only failure 对齐。
   - 输出 `effective_rank`（迁自 `analyze_repr.py`），区分 collapse 与 clustering。
2. 新增 `tools/repr_analysis/predictor_sensitivity.py`：
   - 同一 batch 在 history 子段加噪，跑 predictor 多步 rollout，与 clean rollout 比较。
   - 输出 `predictor_rollout_drift_median(T)`（T = 1..rollout_steps）和 single-step `predictor_target_shift`。
3. 新增 `tools/repr_analysis/task_resolution.py`：
   - `transition_resolution_ratio`：相邻帧距离 / 跨序列随机帧距离的中位数比。
   - inverse-dynamics linear probe（仅训 readout，不动主模型）做 action 可预测性代理。
   - 复用现有 `loss.transition_distance` / `loss.inverse_dynamics` 的 head 结构，只切到 eval-mode probe。

#### P0.3 数据收集（用 §4.3 已有 ckpt）

固定 7 个 checkpoint（已有，不需重训）：

```
LeWM-base, LeWM-perframe-p1, LeWM-perframe-p05
SWM-base, SWM-fixed-std (聚簇典型), SWM-perframe-p1, SWM-perframe-p05
```

任务：`TwoRoom` + `PushT`（已有 eval matrix），可选扩 `Cube` + `Reacher`。

每个 (checkpoint × task) 收齐：
- `noise_sensitivity`（goal / all / history × std grid）
- `predictor_sensitivity`（T = 1, 2, 4, 8）
- `task_resolution`
- 对应 `eval_drop_{pix+goal, goal_only, pix_only}_std{0.03, 0.05, 0.08}`（已有）

panel = (checkpoint × task) ≥ 14 行（7 × 2），扩到 4 任务则 28 行。

#### P0.4 相关性分析

- 主统计：**Spearman ρ + 1000-bootstrap 95% CI**（鲁棒于非线性、抗 outlier）。
- 关键对：

```text
robust_radius_std         ↔ eval_drop_pix+goal
noise_angle_slope         ↔ eval_drop_goal_only      (encoder Lipschitz → goal embedding 错位)
clean_nn_cos_dist         ↔ clean_eval               (聚簇是否伤精度，PushT-like 任务尤甚)
predictor_rollout_drift   ↔ eval_drop_pix_only       (history noise 是 predictor 主输入)
transition_res_ratio      ↔ clean_eval on PushT      (任务分辨率 vs 性能)
effective_rank            ↔ eval_drop_pix+goal       (collapse 早期信号)
```

- 多变量：把上述指标作为特征，对 eval drop 跑 leave-one-checkpoint-out ridge / random forest，看 R²。

#### P0.5 决策标准

| Spearman \|ρ\| | 解释 | 行动 |
|---|---|---|
| ≥ 0.7 | 强相关，单指标即可解释 | 把对应指标作为论文主贡献 |
| 0.4 – 0.7 | 中等，单指标不足 | 用多变量组合预测；同时补 predictor / task-resolution 指标 |
| < 0.4 | 弱，机制不在覆盖范围 | 回到 P3（encoder 拆解）或审视 planner / CEM dynamics |

#### P0.6 Active Validation：从相关到预测

相关性有同族 confounder（同一训法的 ckpt 共享偏置），需要盲测：

1. 选 1–2 个 holdout checkpoint（建议在 `Cube` / `Reacher` 上新训一组 SWM 和 LeWM noise-aware，与 P0.3 训练分布不同）。
2. **只**用 P0.1–P0.3 的诊断输出，给出 eval drop 的预测分桶（low / mid / high）+ `recommendation`。
3. 真实跑 eval，与预测分桶对照。
4. 命中标准：分桶命中 ≥ 80% → 诊断工具可独立写一节；< 60% → 回到 P0.5 弱相关分支。

#### P0.7 输出与维护

- 新增 `tools/repr_analysis/diagnostic_correlation.py`：自动收集 N×T 表，跑 Spearman + bootstrap，落 csv / png。
- 在 `experiments.md` 维护一个 "diagnostic ↔ eval" 主表，每加一个 checkpoint 自动 append。
- 论文图：(a) noise curve 对比图（已有 `plot_noise_curves`），(b) robustness-resolution 散点（已有 `plot_geometry_tradeoff`），(c) 相关性热图（待加）。

### P1：Noise-Aware Training（已完成）

目标：确认 noise augmentation 的收益/损害是否由 task resolution 决定。

状态：P1.1–P1.3 已完成，结果见 §4.3。核心结论：

- per-frame 独立 std 修复 asymmetric noise 崩溃，但带来的是**平滑化**而非**聚簇化**。
- 固定全帧 std 导致聚簇化（TwoRoom clean 提升、asymmetric 崩）；per-frame 独立 std 导致平滑化（asymmetric 修复、clean 不升）。
- LeWM 在同等 noise augmentation 下表现近乎完美，无聚簇化副作用。
- PushT 上 SWM 和 LeWM 均存在 sweet spot（SWM 约 0to002，LeWM 约 0to002），但 SWM 最优仍明显落后于 LeWM。

整理结果时固定看三类输出：

| 输出 | 用途 |
|---|---|
| clean eval | 是否损害原任务 |
| noisy eval | 是否提升 robustness |
| noise_sensitivity | 几何变化是 smoothing 还是 clustering |

P1.4（可选）：若后续需要完整 clean-noise 曲线作图，可补扫 SWM `std_max ∈ {0.01, 0.02, 0.03, 0.08}`。当前 TwoRoom 仅有 0.05，PushT 仅有 0.01/0.02/0.05。

### P2：Cost Surface 解耦

目标：区分 encoder noise sensitivity 和 cosine cost saturation。

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

解释：

- B 明显回升：cost saturation 是重要因素。
- B 仍然低：encoder 已把 noisy goal 编到错误位置，cost swap 无法救。

**当前结果（TwoRoom, std=0.03, pixels+goal, num_eval=50）**

| 变体 | cost type | cost space | score |
|---|---|---|---:|
| A | cosine | normalized | 36.0 |
| B | mse | raw | 42.0 |

结论：

- `raw + mse` 只带来小幅回升（+6），没有接近 clean SWM（90.8）或 LeWM std=0.03（90）。
- cost saturation 可能贡献了一部分损害，但不是主因。
- 主导失败仍然是 upstream encoder / noisy goal embedding corruption：目标 latent 已经偏到错误区域，eval-only cost swap 无法修复。
- P2 因此不再作为主要修复方向；后续优先级应回到 P0/P1/P3/P4。

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

---

## 7. Prior Art：诊断指标的归属与 Gap

为了把诊断工具放到论文里要严谨，本节把 P0 用到的每一个指标对照已有工作，标清楚"哪个是借用 / 哪个是组合 / 哪个是真新东西"。这影响 §8 的贡献条目和 Related Work 该怎么写。

### 7.1 单指标的 prior art

| 我们的指标 | 已有最接近的命名 / 工作 | 关系 |
|---|---|---|
| `effective_rank` | **RankMe** (Garrido et al., ICML 2023)；LiDAR (Thilak et al., ICLR 2024)；matrix entropy / spectral entropy | 直接借用，用作 collapse 区分；引用 RankMe，不主张 novelty |
| `clean_pair_cos_dist` (median) | **Wang & Isola alignment/uniformity** (ICML 2020)；anisotropy (Ethayarajh, EMNLP 2019) | `L_uniform` 是聚合版；我们用作几何参考量，引用 Wang & Isola |
| `clean_nn_cos_dist` | NN-distance OOD (Sun et al., NeurIPS 2022)；SNGP distance-aware features (Liu et al., NeurIPS 2020) | 同样是 latent NN 距离原语；我们用作 noise shift 的归一化分母 |
| `noise_l2`, `noise_angle_deg` | empirical Jacobian probing (Hoffman et al., 2019)；randomized smoothing (Cohen et al., ICML 2019) | 概念等价于"沿数据流形的局部 Jacobian 范数"，引用为 inspiration |
| `noise_angle_slope` (deg per std) | 局部 Lipschitz / spectral norm (Virmaux & Scaman, NeurIPS 2018; LipSDP 2019) | 球面几何下的角度变体；与 certified bounds 不是同一目的 |
| `noise_to_nn_cos_ratio` | 无完全对应；概念上是 signal-to-noise / Mahalanobis-style ratio | **Composite，可主张为 novel** |
| `robust_radius` (interpolated crossing) | certified radius in randomized smoothing (Cohen et al., 2019) | label-free / planning-latent 的 empirical 版本，主张为 novel |
| `predictor_target_shift` (single-step) | rollout MSE in Dreamer / TD-MPC / TD-MPC2 | 标准做法，引用即可 |
| `predictor_rollout_drift(T)` | 文献里没有同名指标；最近 surprise-recognition (arXiv 2512.01119) 用 single-step 误差做 runtime filter | **多步 latent drift between noisy/clean conditioning，主张为 novel** |
| `transition_resolution_ratio` | 检索领域 intra/inter-class gap；ID gap (Brandfonbrener 2023) | 时间近邻版本；主张为 novel naming |
| Inverse-dynamics linear probe | **Brandfonbrener et al.** (NeurIPS 2023) "Inverse Dynamics Pretraining Learns Good Representations"；ICM (Pathak et al., ICML 2017) | 直接引用；novelty 在用 noise-induced drop 作 failure predictor |
| Spearman ρ + bootstrap CI workflow | **ATC** (Garg et al., ICLR 2022)；Deng & Zheng (CVPR 2021)；PROXIMA (2026) | 标准 label-free performance prediction 做法；引用为 method |
| Active validation on holdout checkpoints | active testing (Kossen et al., ICML 2021) | 框架不新；新颖在应用到 world-model robustness |

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
> Label-free performance prediction：ATC (Garg 2022), Deng & Zheng (CVPR 2021), Active Testing (Kossen 2021), PROXIMA (2026).

完整链接见 `experiments.md` 末尾的"参考文献"段；本文只保留指标-工作映射，避免链接腐烂。

### 7.4 必须主动 differentiate 的近期工作（撞车风险高）

下面两篇是 2025 年和我们工作高度交叉的 prior art，论文 Related Work 必须单独段落讨论，明确 delta：

| 工作 | 时间 / 出处 | 核心主张 | 与我们的撞车点 | 我们的 delta |
|---|---|---|---|---|
| **PCA++** "How Uniformity Induces Robustness to Background Noise in Contrastive Learning" | arXiv:2511.12278 (Nov 2025) | 在 contrastive SSL 中，uniformity 损失隐式诱导 background noise robustness；提供理论分析 | 整体故事（uniformity → robustness）和我们 SWM uniformity 高度相似 | (a) 设置不同：他们是分类 / contrastive，我们是 world-model planning；(b) 我们发现 uniformity 实际效果（聚簇化 vs 平滑化）取决于 noise augmentation 实现方式（固定 std 还是 per-frame）；(c) 我们提供 *诊断 toolkit + active validation*，他们提供 *理论 + 分类 acc*；(d) 我们发现 task-resolution tradeoff（PushT 反响应），不在他们的覆盖范围 |
| **Surprise-Recognition** "World Model Robustness via Surprise Recognition" | arXiv:2512.01119 (Dec 2025) | 用 world model 的 single-step prediction surprise 做 *runtime* 噪声 input 过滤 | 同样是 WM + noise，使用 single-step prediction error 信号 | (a) 他们是 *runtime filter*（每帧决定是否信任输入），我们是 *pre-hoc predictor*（从校准数据预测 policy success drop）；(b) 他们是 single-step；我们 `predictor_rollout_drift(T)` 是 multi-step；(c) 我们做 cross-checkpoint correlation + holdout 分桶验证，他们没有 |

写 paper 必须在 Related Work 第一段就把这两篇拎出来讲清差异；不能只放在 reference list 里。

### 7.5 附加可借鉴方法（已加入实现）

| 方法 | 来源 | 状态 |
|---|---|---|
| **CKA(clean, noisy)** | Kornblith et al., ICML 2019 | ✅ 已加 `noise_sensitivity.py:_linear_cka`，作为 per-point shift 的子空间对齐补充 |
| **LiDAR rank**（temporal 正样本对版本） | Thilak et al., ICLR 2024 | ✅ 已加 `task_resolution.py:_lidar_rank` |
| **Brandfonbrener ID 定理** | NeurIPS 2023 | ✅ 论文写作时引用，作为 ID linear probe 的正当性依据；docstring 已注 |
| **ATC 阈值-分桶框架** | Garg et al., ICLR 2022 | ⚠️ 待加入 `diagnostic_correlation.py`（P0.7）；P0.6 active validation 应按 ATC 流程描述 |

---

## 8. 决策节点

| 节点 | 如果结果是... | 下一步 |
|---|---|---|
| P0.4 | robust_radius / angle_slope 与 eval drop 强相关 (\|ρ\| ≥ 0.7) | 把诊断作为论文主贡献之一 |
| P0.4 | 单指标弱、多变量组合可预测 | 报组合指标，补 predictor / task-resolution 指标 |
| P0.4 | 诊断与 eval 全面不相关 | 优先查 planner / cost / action dynamics |
| P0.6 | holdout 分桶命中 ≥ 80% | 诊断工具独立写一节，可投 short paper |
| P0.6 | 命中 < 60% | 转 P3（encoder 拆解）或重审失败机制 |
| P1 | 固定 std 导致聚簇化（TwoRoom 升、PushT 降），per-frame 导致平滑化（asymmetric 修复、clean 不升） | 实现方式决定几何形态；task resolution tradeoff 成立 |
| P1 | LeWM 无聚簇化且 noise robustness 全面优于 SWM | 聚簇化不是通用现象；球面 + uniformity 是诱因 |
| P1 | PushT sweet spot 存在但 SWM 最优仍落后 LeWM | 球面表征的连续性 prior 与精细操作存在结构性冲突 |
| P2 | raw MSE 回升 | cost saturation 是重要失败环节 |
| P2 | raw MSE 不回升 | encoder noisy goal 已经主导失败 |
| P4 | adaptive guardrail 保住 PushT 且提升 TwoRoom | 形成真正方法贡献 |

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
4. **Integrated diagnostic toolkit + active-validation protocol**：把 encoder / predictor / task-resolution / ID-probe 组合成 label-free predictor，再用 holdout checkpoint 做盲分桶验证。这套从相关性到主动验证的闭环是论文核心方法贡献。
5. **机制发现**：noise augmentation 在 WM 中不必然带来 smoothing，可能诱导 clustered/discretized geometry；TwoRoom 与 PushT 对同一 geometry prior 反向响应。
6. **后续方法（initial）**：adaptive resolution / guarded noise consistency，尝试替代手工按任务调 recipe。

诚实声明：`effective_rank`、Wang & Isola uniformity、Jacobian/Lipschitz 概念、ID linear probe、Spearman + bootstrap workflow 均为已有方法（参见 §7.1 和 §7.3），论文中需明确归属。

---

## 10. 维护说明

后续补结果时优先放在对应 P 节：

- `P0`：诊断指标、相关性图、robust radius 表。
- `P1`：noise-aware training 的 eval 和 geometry 变化。
- `P2`：eval-only cost ablation。
- `P3`：BN/LN/dim 的 encoder sensitivity。
- `P4`：adaptive resolution / guarded consistency 方法。

避免把每次命令输出都塞进本文；完整流水仍放 `experiments.md`。本文只保留能够改变判断的结果。

---

## 参考文件

| 文件 | 内容 |
|---|---|
| `plan_v2.md` | 原始设计文档 |
| `experiments.md` | 完整实验记录 |
| `config/train/swm.yaml` | SWM 训练配置 |
| `config/train/lewm.yaml` | LeWM 训练配置 |
| `tools/repr_analysis/noise_sensitivity.py` | Noise sensitivity 诊断（含 history scope、effective_rank、CKA、collapse vs clustered 区分） |
| `tools/repr_analysis/predictor_sensitivity.py` | Predictor open-loop target shift + 自回归 rollout drift |
| `tools/repr_analysis/task_resolution.py` | transition resolution ratio + ID linear probe + LiDAR rank |
| `tools/repr_analysis/run_full_diagnostics.py` | 统一入口，跑全套并产出 `diagnostics_summary.json` 一行 roll-up |
| `tools/repr_analysis/diagnostic_correlation.py` | （P0.7 待加）诊断 ↔ eval 相关性自动化 + ATC 框架 |
| `run_trainer.sh` | 训练 → eval sweep → 全套诊断（自动调 run_full_diagnostics）→ summary 一站式 |
| `tools/repr_analysis/repr_compare_template.ipynb` | Notebook 对比模板 |
| `jepa.py` | JEPA + SphericalJEPA 实现 |
| `module.py` | loss 与共享模块 |
