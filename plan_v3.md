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

### P0：把诊断变成预测指标

目标：证明 `noise_sensitivity` 不是事后解释，而能预测 robustness failure。

要做：

1. 对 LeWM / SWM / SWM-noise-train 跑 4-task noise_sensitivity。
2. 对同一批 checkpoint 跑少量 noise eval。
3. 画相关性：

```text
robust_radius vs eval degradation slope
noise_angle_slope vs eval drop
clean_nn_distance vs clean performance
```

判断标准：

- 如果 robust_radius 和 eval drop 强相关，诊断工具就是独立贡献。
- 如果相关性弱，说明 planner/cost/action dynamics 还有额外因素，需要 P2/P3 补充。

### P1：Noise-Aware Training 补完

目标：确认 noise augmentation 的收益/损害是否由 task resolution 决定。

当前在补跑：

- P1.1：TwoRoom per-frame independent noise + `noise_prob`
- P1.2：PushT 小强度 noise
- P1.3：LeWM 同条件对照

建议整理结果时固定看三类输出：

| 输出 | 用途 |
|---|---|
| clean eval | 是否损害原任务 |
| noisy eval | 是否提升 robustness |
| noise_sensitivity | 几何变化是 smoothing 还是 clustering |

P1.4 可作为主图实验：SWM 上扫 `std_max ∈ {0.01, 0.02, 0.03, 0.05, 0.08}`，比较 TwoRoom / PushT clean eval 曲线。预期 TwoRoom 与 PushT 趋势相反。

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

## 7. 决策节点

| 节点 | 如果结果是... | 下一步 |
|---|---|---|
| P0 | robust_radius 预测 eval drop | 把诊断作为主贡献之一 |
| P0 | 诊断与 eval 不相关 | 优先查 planner/cost/action dynamics |
| P1 | TwoRoom 升、PushT 降趋势稳定 | 支撑 task resolution tradeoff 主线 |
| P1 | LeWM 也聚簇化 | noise augmentation effect 是通用 SSL/WM 现象 |
| P1 | 只有 SWM 聚簇化 | 球面 + uniformity 特别易产生等价类 |
| P2 | raw MSE 回升 | cost saturation 是重要失败环节 |
| P2 | raw MSE 不回升 | encoder noisy goal 已经主导失败 |
| P4 | adaptive guardrail 保住 PushT 且提升 TwoRoom | 形成真正方法贡献 |

---

## 8. 论文叙事草案

一句话版本：

> Spherical world models do not simply improve or degrade planning; they expose a task-dependent invariance-resolution tradeoff in latent geometry.

中文版本：

> 球面世界模型不是 LeWM 的单向替代，而是改变了 latent space 的几何偏置。低维导航任务受益于更强 invariance 和聚簇化，高分辨率操作任务则需要保留连续状态差异。我们提出 noise sensitivity / robust radius 等诊断工具，量化这种 tradeoff，并进一步探索自适应 resolution 的训练机制。

可能贡献：

1. **方法与基线**：实现 SWM，证明 spherical + temporal-masked uniformity 可以稳定训练 JEPA-style world model。
2. **诊断工具**：提出 latent noise sensitivity 和 empirical robust radius，连接 embedding geometry 与 planning robustness。
3. **机制发现**：noise augmentation 在 WM 中不必然带来 smoothness，而可能诱导 clustered/discretized geometry。
4. **任务规律**：TwoRoom 与 PushT 对同一 geometry prior 反向响应，说明 world model 表征需要匹配任务分辨率。
5. **后续方法**：adaptive resolution / guarded noise consistency，尝试替代手工按任务调 recipe。

---

## 9. 维护说明

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
| `tools/repr_analysis/noise_sensitivity.py` | Noise sensitivity 诊断工具 |
| `tools/repr_analysis/repr_compare_template.ipynb` | Notebook 对比模板 |
| `jepa.py` | JEPA + SphericalJEPA 实现 |
| `module.py` | loss 与共享模块 |
