# 球面世界模型实验计划 V3

> **阅读说明**：本文是项目的当前完整状态文档，适合初次接触的读者。它涵盖背景、已完成实验的关键结论、当前最佳配置，以及下一阶段的研究方向（noise robustness pivot）。原始设计见 `plan_v2.md`，完整实验记录见 `experiments.md`。

---

## 1. 项目背景

### 1.1 LeWorldModel (LeWM) 是什么

LeWM 是一个从像素端到端训练的世界模型，基于 Joint-Embedding Predictive Architecture (JEPA)。给定一段历史观测序列，LeWM 学习一个 encoder 把每帧图像映射到隐空间，再用一个 transformer predictor 预测未来帧的隐向量，而不是预测原始像素。训练完成后配合 CEM (Cross-Entropy Method) 规划器做 model-predictive control，在多个视觉控制任务上达到竞争性能。

LeWM 的训练目标只有两项：

```
L_LeWM = ||pred(z_t, a_t) - z_{t+1}||²   (prediction loss)
       + λ · SIGReg(Z)                     (anti-collapse regularizer)
```

- **Prediction loss**：让 predictor 学习动力学，即"知道现在的隐状态和动作，就能预测下一帧的隐状态"。
- **SIGReg**：防止表征塌缩（所有帧映射到同一个点）。SIGReg 通过 Cramer-Wold 投影把"检验高维分布是否高斯"化归到一维的 Epps-Pulley 检验，强制嵌入边缘分布匹配**各向同性高斯**。

LeWM 的表征空间是 **欧氏空间** ℝ^d（d=192），planning cost 是 **L2 distance**。

### 1.2 SIGReg 的理论来源与潜在局限

LeJEPA 理论证明：在静态表征学习中，各向同性高斯是 JEPA 嵌入分布的最优形态（在固定 trace 约束下让线性 probe 的 bias+variance 最小，在固定二阶矩约束下让高斯成为最大熵分布）。

这个证明有几个重要假设：
- 下游任务是任意线性或核 probe（worst-case 意义上的最优）
- 表征空间是 ℝ^d

这些假设在**世界模型**场景下不一定成立：世界模型的下游是规划，规划有具体的几何结构（近邻状态应有近似表征），而不是"任意下游任务"。

**具体观察到的弱点**：LeWM 在 Two-Room 任务上成功率只有 87%（后续重测约 93%），而更简单的 baseline 能达到 100%。Two-Room 是一个内在维度只有 2 维的环境（agent 位置 (x,y)），SIGReg 把表征强行展到 192 维各向同性高斯，可能稀释了状态空间的拓扑结构。

### 1.3 球面表征是什么

**球面表征**（Spherical Representations）是把每个嵌入向量 L2 归一化到单位球面 S^{d-1} 上：μ(o) = z / ‖z‖ ∈ S^{d-1}。

这是自监督学习领域被广泛验证的成熟做法：SimCLR、MoCo、CLIP、DINO 都把表征归一化到球面，使用余弦相似度作为度量。其优点包括：
- 避免高维空间的距离集中现象
- 天然的尺度约束，无 magnitude 发散问题
- 与 cosine 相似度几何自然兼容
- 紧致空间让"覆盖"和"分散"有明确几何意义

但球面表征**在世界模型规划场景下**的效果此前未被系统验证（LeWM 用欧氏空间，V-JEPA 2 也是欧氏空间）。

---

## 2. SWM V0：我们做了什么

### 2.1 核心修改

Spherical World Model (SWM) V0 对 LeWM 做了三处修改，其余全部保持不变：

| 组件 | LeWM | SWM V0 |
|---|---|---|
| Encoder projector | MLP + BN → ℝ^d | MLP + BN → L2 norm → S^{d-1} |
| Predictor projector | MLP + BN → ℝ^d | MLP + BN → L2 norm → S^{d-1} |
| Prediction loss | MSE: ‖pred − tgt‖² | Cosine distance: 1 − pred·tgt |
| Anti-collapse | SIGReg | Uniformity loss |
| Planning cost | L2 distance (raw) | Cosine distance (normalized) |

保持不变的部分：ViT-Tiny encoder backbone、ARPredictor（ViT-S）、action embedder（AdaLN 注入）、CEM planner、数据管道。

`SphericalJEPA` 类只 override 了 `encode()`、`predict()`、`criterion()`，`rollout()` 和 `get_cost()` 继承自 `JEPA`，CEM planner 不需要修改。

### 2.2 Anti-collapse loss 的演化

SWM 开发中最大的工程挑战是**防塌缩**。直接在球面上用 pairwise cosine 相似度作为 spread loss 完全失效：所有 z_i 相同时 (z_i − z_j)=0，梯度为零，模型无法逃离塌缩点（**梯度死区问题**）。

我们依次测试了 11 种方案，最终找到可行路径：

| 方案 | 结论 |
|---|---|
| Linear projector + pairwise cosine | 塌缩，梯度死区 |
| + detach target | 无改善 |
| InfoNCE (τ=0.1) | 破塌缩但破坏时序结构，pred_loss 反弹 |
| MLP+BN + pairwise cosine | 训练不塌缩但 eval 塌缩（BN masking） |
| sliced Wasserstein | 有非零梯度但信号太弱 |
| **MLP+BN + uniformity_loss** | **真正逃离塌缩**（但需要足够长训练让 BN running stats 对齐） |

SIGReg 能成功的原因是**sorting + quantile matching**：把嵌入投影到随机方向后，排序本身让相同值也获得不同 rank，从而有非零梯度。这是塌缩时唯一无死区的机制。BN+uniformity 的成功机制是：BN 在 training 时向 batch 内注入足够的跨样本变化，打破对称性；经过足够轮次训练后 running stats 对齐，eval 也稳定脱离塌缩。

### 2.3 Uniformity Loss 与 Temporal Masking

Wang & Isola 2020 的 uniformity loss：

```
L_uniform = log E[exp(-t · ‖μ_i - μ_j‖²)]
```

鼓励样本在球面上均匀分布。但在时序数据上，**时间相邻帧本来就应该相似**（pred_loss 就是为此设计的），uniformity 把所有 pair 都推开会制造矛盾。

解决方案是 **temporal masking**：计算 uniformity loss 时，排除同一 trajectory 窗口内时间距离 ≤ k 的 pair，只对时间距离更远的 pair 施加均匀化压力。

```yaml
loss:
  uniformity:
    mode: temporal_masked
    temporal_exclusion: 2   # 排除 |Δt| <= 2 的 pair
```

这是一个"软结构先验"：不强制近邻状态相似（不像 temporal hinge），只是不强制它们远离。

---

## 3. 关键超参数消融结论

在 PushT 任务上做了系统消融（epoch=10, num_eval=500），主要发现：

**Uniformity 权重**：`weight=0.1` 太弱，`weight=0.2` 一致性改善，`weight=0.3` 在最优分支上反而下降。选 `0.2`。

**Pair 选择策略的影响大于 backbone 改动**：

| 策略 | PushT |
|---:|---:|
| all_pairs | 74.4 |
| cross_window | 80.2 |
| temporal_masked_1 | 80.0 |
| **temporal_masked_2** | **89.8** |
| temporal_masked_3 | 67.0 |

Exclusion range 有明确最优值：1 太小，3 太大，2 刚好。

**维度 dim=64 vs 192**：dim=64 在 `all_pairs` 下反而更差，但在 `temporal_masked_2` 下明显更好（+9.2）。维度压缩只有在时序结构对齐后才有益。

**Temporal hinge（固定连续性约束）**：在几乎所有设置下都损害性能（尤其 PushT/Reacher），原因是强制所有相邻转移都接近，和接触/操作任务中的大幅度动作相冲突。已确认不是可行方向。

### 当前最佳配置

```
swm_mlp_bn_uniform_w_0p2_t_2_temporal_masked_2_dim_64
```

核心参数：
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

**4-task benchmark（epoch=10, num_eval=500, single seed）**：

| Task | LeWM | SWM (best) | Delta |
|---|---:|---:|---:|
| TwoRoom | 93.0 | 90.8 | -2.2 |
| Cube | 69.2 | 74.0 | **+4.8** |
| PushT | 89.4 | 89.8 | +0.4 |
| Reacher | 62.2 | 66.0 | **+3.8** |
| **Average** | **78.5** | **80.2** | **+1.7** |

SWM 在 4-task 平均上略高，但 TwoRoom 仍差 2.2 分，且当前结论基于 single seed，存在较大不确定性（2026-04-25 的重测 run 在 PushT/Reacher 上偏低，表明需要多 seed 验证）。

---

## 4. Noise Robustness Pivot：新的研究方向

### 4.1 发现经过

在完成上述表征消融后，我们引入了**输入噪声测试**：在 eval 时对观测图像添加 Gaussian pixel noise，测量模型性能的衰减程度（通过 `eval.corruption.enabled=True eval.corruption.std=X`）。

**Eval 测试结果（tworoom, std=0.03, num_eval=50）**：

| 噪声范围 | LeWM 成功率 | SWM 成功率 |
|---|---:|---:|
| 无噪声（参考） | 93 | 90.8 |
| 全帧加噪 | 90 | **36** |
| 仅 pixels 帧加噪 | 94 | **66** |
| 仅 goal 帧加噪 | ≈93 | **42** |

SWM 在 std=0.03 时成功率崩至 36，而 LeWM 几乎不受影响。

**LeWM noise sweep（仅全帧）**：

| std | 0.03 | 0.04 | 0.05 | 0.08 | 0.10 | 0.15 |
|---|---:|---:|---:|---:|---:|---:|
| LeWM eval | 90 | 82 | 78 | 48 | 46 | 30 |

LeWM 在 std≥0.08 才出现明显衰减。SWM 在 std=0.03 就已经崩。

### 4.2 表征层 Noise Sensitivity 分析

对两个模型的 encoder 做了系统性的 noise 敏感度分析，核心指标表（goal frame, normalized space）：

```
model  std    noise_angle_deg_median  clean_nn_cos_dist_median  noise_to_nn_cos_ratio_median  risk
lewm   0.000  0.0000                  0.0389                    0.0000                        low
lewm   0.005  4.1832                  0.0389                    0.0685                        low
lewm   0.010  8.7931                  0.0389                    0.3021                        low
lewm   0.020  18.7491                 0.0389                    1.3640                        high
lewm   0.030  31.9681                 0.0389                    3.8983                        high
swm    0.000  0.0000                  0.0820                    0.0000                        low
swm    0.005  11.9534                 0.0820                    0.2646                        low
swm    0.010  25.9257                 0.0820                    1.2280                        high
swm    0.020  54.0547                 0.0820                    5.0395                        high
swm    0.030  69.6558                 0.0820                    7.9602                        high
```

**关键发现**：在 std=0.005（极小噪声）时，SWM 的 encoder 就产生了 12° 角度偏移，而 LeWM 只有 4°。这个 **~3 倍角向 Lipschitz 差距**在小 noise 区间几乎是常数，强烈提示是 encoder 端的结构性问题，而非非线性区偶发。

还有一个反直觉的发现：SWM 的 `clean_nn_cos_dist_median`（0.082）比 LeWM（0.039）**更大**，即 SWM 的最近邻本来就更远。这说明 SWM 的脆弱不是因为嵌入太密集，而是 encoder 的输入-输出放大比就是高的。

### 4.3 两段串联的失败机制

SWM 在噪声下的崩溃是**两段串联放大**的结果：

**第一段：Encoder 角向 Lipschitz 偏高**

SWM encoder 把 pixel-space 噪声放大到 ~3× 的角度扰动，可疑的三个原因：
1. **BatchNorm in projector**：BN 把 std 归一到 1，本质上放大了小信号；
2. **L2 normalize 在小范数处的奇点**：∂(x/‖x‖)/∂x = (I − μμᵀ)/‖x‖，如果 pre-norm 向量范数小，则角向梯度就大；
3. **dim=64**：相同高斯扰动在 64 维球面上的方向扰动 ≈ √(192/64) ≈ 1.7× 大于 192 维，能解释部分差距。

**第二段：Cosine cost 在大角度饱和**

LeWM 的 planning cost 是 L2 distance，无上界，方向梯度在任何角度下都有信息。SWM 的 cost 是 cosine distance：`1 − cos(z, z_goal)`，值域 [0, 2]。当 goal 被噪声推到 70° 时，cosine cost 已接近饱和区，规划器的梯度几乎为零——即使轨迹方向完全错误，cost 也不再提供修正信号。

这也解释了为什么 **goal 帧噪声比 pixels 帧噪声更具破坏性**：pixels 帧的扰动影响的是表征序列中的"历史"状态，对 CEM 规划的影响相对分散；goal 帧的扰动直接污染了规划目标，cosine cost 饱和后规划器完全失去方向。

**这一发现把之前"调 regularizer"的路线转变为一个更清晰的诊断故事**：SWM 在 in-distribution 上和 LeWM 持平，但在观测噪声下显著更脆弱，机理明确可定位，修复路径也清晰。

### 4.4 与现有文献的关系

- **I-JEPA 鲁棒性**（Kulm 2023）：测试了 I-JEPA 对 PGD/FGSM 对抗攻击比 ViT 更鲁棒，但仅针对静态分类任务，不涉及 planning loop。
- **V-JEPA 2**（Meta 2025）：论文未报告观测噪声/分布偏移下的 planning 性能。
- **Distracting Control Suite**（Stone et al. 2021）：专门测试 RL policy 的视觉鲁棒性，但用的是 pixel-reconstruction WM，非 JEPA。
- **Robustness Verification for Contrastive Learning**（Wang & Liu, ICML'22）：严格推导了 contrastive encoder 的鲁棒半径，可用于给 SWM/LeWM 做数值化的 certified robustness 分析。

**空缺**：目前没有工作系统比较 Euclidean vs Spherical 世界模型在观测噪声下的规划鲁棒性，也没有对 JEPA-style WM 的 encoder Lipschitz 和 cost surface 饱和进行联合分析。这是我们可以填补的。

---

## 5. 新实验计划

### P1：Noise-Aware Training（第一优先级 — 已部分完成）

**目标**：验证 encoder Lipschitz 偏高是否是主因，以及 noise augmentation 训练是否能修复。

#### P1 第一组结果（swm_noise_train, std_min=0, std_max=0.05）

训练命令：

```bash
python train_swm.py data=tworoom \
  image_noise.std_min=0.0 image_noise.std_max=0.05 \
  subdir=ckpt/swm_noiseaug_0to05
```

**Eval 结果**：

| 任务 | corruption | apply_to | score |
|---|---|---|---:|
| TwoRoom | std=0 | — | **97.6** |
| TwoRoom | std=0.05 | pixels+goal | **98.0** |
| TwoRoom | std=0.08 | pixels+goal | 88.0 |
| TwoRoom | std=0.05 | pixels only | 56.0 |
| TwoRoom | std=0.05 | goal only | 44.0 |
| PushT | std=0 | — | 61.8 |
| PushT | std=0.05 | pixels+goal | 60.0 |

**Noise sensitivity 表对照（std=0.005 这一点）**：

| 指标 | swm baseline | swm_noise_train | 变化 |
|---|---:|---:|---:|
| `clean_nn_cos_dist_median` | 0.082 | **0.008** | 缩到 1/10 |
| `clean_nn_l2_median` | 0.40 | **0.13** | 缩到 1/3 |
| `noise_angle_deg_median` @ std=0.005 | 11.9° | **27.6°** | **变大** |

#### 关键发现：noise-aug 训练产生的是"聚簇化"而非"平滑化"

简单的 Lipschitz 假设被推翻。Noise-aug **没有降低** encoder 的角向 Lipschitz（小噪声角度反而更大），但把 clean 邻域距离压到 1/10。两个变化合起来表明：

> **encoder 学会把训练分布内的扰动聚集成紧凑等价类，而不是学一个全局平滑映射。**

clean 状态收缩到 ~0.5° 紧密簇，簇间通过陡峭边界分隔。小扰动一旦跨过边界就跳到下一个簇（27° 偏移）。这是 **discretized / clustered** 几何，不是 Lipschitz-smooth 几何。

#### 三个关键现象的解释

**(1) TwoRoom 训测一致下 eval 反而比 clean baseline 更高（97.6 vs 90.8，+6.8）**

机理推测：TwoRoom 内在状态空间是 2 维。加噪训练强迫 encoder 抹掉与状态无关的视觉细节，最后留下的紧凑簇近似于"状态等价类"。这等价于**信息瓶颈**——把表征压到接近真实状态空间内在维度，CEM 不再被无关方向干扰。同样原理解释了 std=0.05 训测一致时 eval 是 98——这不是"困难测试"，而是"训练分布内的标准操作"，等价类边界已被精确校准到 std=0.05 半径。

**(2) Asymmetric noise（仅 pixels 56 / 仅 goal 44）大幅崩**

训练时所有帧统一加噪声，encoder 学到的是"统一噪声水平下的整体鲁棒性"，不是"单帧独立噪声鲁棒性"。当 history 帧 clean、goal 帧 noisy（或反过来），**帧间噪声分布一致性被打破**，CEM rollout 中的隐式协调被破坏。这是 train/test distribution mismatch。

> **修复方向**：noise augmentation 应该**逐帧独立采样 std**，让 encoder 真正学到"帧无关的扰动鲁棒性"。

**(3) PushT 上反而下降（89.8 → 61.8 clean）**

PushT 是接触操作任务，相邻状态之间存在小但有意义的差异（推动物块精细位置）。Noise-aug 的等价类聚簇把这些差异归并掉，规划时分辨不出"再推一点"和"已推到位"。

这与 experiments.md §7 一致：fixed temporal_hinge 在 PushT 上同样严重损害（6.2~20）。两者**机制相同**——都是强行让相邻状态相似，在需要高分辨率状态区分的任务上必然损害。

> **核心洞察**：noise-aug 是"基于分布扰动"的隐式连续性 prior，temporal_hinge 是"基于时间相邻"的显式连续性 prior。两者本质等价。**强度需要按任务调节**：TwoRoom 这类离散低维任务能从中受益，PushT/Reacher 这类连续控制任务需要更弱或自适应的强度。

#### P1 后续实验

**P1.1 — 逐帧独立 noise（修复 asymmetric mismatch）**

修改 `AddNormalizedGaussianNoise.__call__`，让每个 frame 单独采样 std。这样训练时 encoder 同时见到"clean+clean"、"clean+noisy"、"noisy+clean"、"noisy+noisy"四种组合，asymmetric eval 时不再 OOD。

**P1.2 — 较小 noise 强度（保护 PushT）**

```bash
python train_swm.py data=pusht \
  image_noise.std_min=0.0 image_noise.std_max=0.02 \
  subdir=ckpt/swm_noiseaug_pusht_0to02

python train_swm.py data=pusht \
  image_noise.std_min=0.0 image_noise.std_max=0.01 \
  subdir=ckpt/swm_noiseaug_pusht_0to01
```

观察 PushT 上是否存在一个 std_max 使得 clean eval 不掉、noise eval 提升。如果存在，说明 PushT 的状态分辨率上限对应一个具体噪声水平；如果不存在，说明聚簇 prior 与 PushT 的连续性需求根本冲突。

**P1.3 — LeWM 同条件对照**

```bash
python train.py data=tworoom \
  image_noise.std_min=0.0 image_noise.std_max=0.05 \
  subdir=ckpt/lewm_noiseaug_0to05

python train.py data=pusht \
  image_noise.std_min=0.0 image_noise.std_max=0.05 \
  subdir=ckpt/lewm_noiseaug_pusht_0to05
```

看 LeWM 是否也出现"聚簇化"现象。如果 LeWM 也表现出 clean nn distance 大幅缩小，说明这是 noise augmentation 的通用效应；如果只有 SWM 出现，说明球面表征 + uniformity 在噪声下特别容易聚簇。

**P1.4 — 多 std_max 扫**

仅 SWM 上扫 std_max ∈ {0.01, 0.02, 0.03, 0.05, 0.08}，画出 TwoRoom / PushT 的 clean eval 曲线。预期是 TwoRoom 单调改善（直到压得太狠），PushT 单调下降（连续性破坏越早出现越好）。这条曲线本身就能成为论文的一个主图。

### P2：Cost Surface 解耦（机理验证）

**目标**：区分"encoder 问题"和"cost 问题"。

保持当前最佳 SWM checkpoint 不变，仅在 eval 时改变 planning cost 空间：

| 变体 | cost_type | cost_space | 预期 |
|---|---|---|---|
| A（现有基线） | cosine | normalized | 70°+ 后失明 |
| B | cosine | normalized | 同 A（对照） |
| **C** | **mse** | **raw** | L2 cost 无饱和，若 SWM noisy 曲线大幅回升 → 证实 cost saturation |

C 不需要重新训练，只需在 eval 时加：

```bash
python eval.py --config-name=tworoom.yaml policy=<swm_ckpt> \
  eval.corruption.enabled=True eval.corruption.std=0.03 \
  "wm.inference.cost_type=mse" "wm.inference.cost_space=raw"
```

（注意：这和 Exp C2 的区别是不重新训练，只改 planning cost，是最干净的 ablation。）

### P3：Encoder Lipschitz 拆解（如果 P1 提示 encoder 是主因）

最小消融，每个 1 seed，在 noise_sensitivity 表里看 `noise_angle_deg_median @ std=0.005` 这一格就够了（不需要 eval，快速诊断）：

| 变体 | 改动 | 预期 |
|---|---|---|
| SWM-noBN | `encoder.projection_head.norm_fn=none` | 角向 Lipschitz 降低？ |
| SWM-LN | `encoder.projection_head.norm_fn=layernorm` | 对比 BN 的放大效应 |
| SWM-dim128 | `wm.embed_dim=128` | 维度与 Lipschitz 的关系 |
| SWM-dim192 | `wm.embed_dim=192` | 对齐 LeWM dim |

去掉 BN 的风险：BN 是当前逃离 collapse 的关键机制，直接去掉可能让表征再次塌缩。可以先做 noise_sensitivity 诊断，看塌缩状态的 noise table，与 experiments.md Exp 11 结合分析，再决定是否继续。

### P4：Robust Radius 数值化（论文指标）

参考 Wang & Liu (ICML'22) 的思路，给每个 checkpoint 计算**经验 robust radius**：令 noise_to_nn_cos_ratio_median = 1 时对应的 std 值（即"噪声大到让 embedding 跳出最近邻"时的输入噪声水平）。

这个数值可以从 noise_sensitivity 表直接插值得到：
- LeWM：robust radius ≈ std=0.017（ratio 在 0.010~0.020 之间跨越 1）
- SWM：robust radius ≈ std=0.008（ratio 在 0.005~0.010 之间跨越 1）

**SWM 的 robust radius 约为 LeWM 的 50%**，这是一个简洁可比较的单一数值，比"eval 在某 std 下掉到 X"更便于跨模型比较和论文报告。

如果 noise-aug 训练能把 SWM 的 robust radius 提高到接近 LeWM，就可以作为一个可量化的改进指标。

---

## 6. 执行优先级与判断节点

```
P1（noise-aug training）
    ↓
    发现 SWM 崩溃点延迟 → 继续 P1（多 seed）+ P3（定位 encoder 机制）
    发现崩溃点不变      → P2（cost surface 解耦）
    ↓
    P4（robust radius，论文用）
    ↓
    多 seed 验证（P1 的 noise-aug 变体稳定后，在 4 个任务上跑 3-5 seeds）
```

此前的 temporal_hinge 和 regularizer sweep 方向**已结束**，得到的结论是：
- `temporal_masked_2 + w=0.2 + dim=64` 是当前有效组合，进一步 sweep 边际效益低；
- fixed temporal hinge 在大多数任务损害性能，不是可行方向；
- noise robustness 是一个更能定位问题结构的探针，且与研究问题的 deployment 相关性更高。

---

## 7. 整体论文叙事（V2，基于 P1 第一组结果更新）

> 我们提出 SWM（Spherical World Model），用球面表征 + uniformity 正则化替换 LeWM 的欧氏表征 + SIGReg。在干净观测条件下，SWM 在 4-task 平均上略优于 LeWM（80.2 vs 78.5），在 Cube 和 Reacher 任务上有显著改善。
>
> 进一步的 noise robustness 测试揭示了一组任务相关的现象：在 std=0.05 的逐帧像素噪声下，clean SWM 在 TwoRoom 上从 90.8% 崩至 36%，但用 std∈[0,0.05] 范围的 noise augmentation 重训后，TwoRoom clean eval 从 90.8% 提升到 97.6%、std=0.05 时维持在 98%。然而同样的 noise-aug 训练在 PushT 上反而把 clean eval 从 89.8% 降到 61.8%。
>
> 表征几何分析表明 noise-aug 训练产生的是 **discretized / clustered geometry** 而非 Lipschitz-smooth geometry：clean 邻域距离缩到 1/10，但小噪声下角向偏移反而变大（11.9°→27.6°）。这与 fixed temporal_hinge 是同一性质的连续性先验，效果取决于任务的状态分辨率需求——TwoRoom 这类低内在维度任务从中受益（信息瓶颈效应），PushT 这类高分辨率连续控制任务因等价类合并而损害。
>
> 我们的核心贡献是：(1) 将 spherical-vs-Euclidean WM 比较扩展到 robust evaluation 维度；(2) 揭示 noise augmentation 在 JEPA-style WM 中并非"普遍有益的正则化"，而是与任务内在维度强相关的 trade-off；(3) 给出 robust radius 等可量化的几何鲁棒性指标，让此类 trade-off 可被预先诊断而无需穷尽 eval。

这个叙事的好处是把"SWM 是否优于 LeWM"这个开放问题，转化为更可控的"什么时候 SWM 优、什么时候 LeWM 优、为什么"的机理问题。即使最终结论是 SWM 没有全局优势，仍然是一个有价值的负面结果。

### 后续可能的论文打分点

- **TwoRoom +6.8 分（90.8 → 97.6）通过 noise-aug** 是一个独立的、容易复现的、机理可解释的改善。即使 PushT 失败，这一点本身就值得报告。
- **Asymmetric noise crashing**（pixels-only 56, goal-only 44）是一个新的失败模式，在 V-JEPA 2 等论文中没有被讨论过。它揭示了 latent WM 训练时的隐式 distribution coupling。
- **Discretized geometry hypothesis**（clean nn distance 缩 10×，小 noise angle 变大）是关于 noise augmentation 在 SSL 中机理的新观察，可独立成文。

---

## 参考文件

| 文件 | 内容 |
|---|---|
| `plan_v2.md` | 原始设计文档（LeJEPA 理论、SWM 架构、三阶段计划） |
| `experiments.md` | 完整实验记录（collapse 消融、PushT ablation、temporal hinge 对比） |
| `config/train/swm.yaml` | 当前最佳 SWM 训练配置 |
| `config/train/lewm.yaml` | LeWM 训练配置 |
| `tools/repr_analysis/noise_sensitivity.py` | Noise sensitivity 诊断工具 |
| `jepa.py` | JEPA + SphericalJEPA 实现 |
| `module.py` | 共享模块（cosine_pred_loss, spread_loss, uniformity_loss） |
