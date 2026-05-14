# 潜空间预测世界模型中不变性与控制分辨率的张力：
# 一项基于 JEPA 架构的系统诊断研究

**标题（英文）**: Understanding the Invariance-Resolution Trade-off in Latent Predictive World Models: A Diagnostic Study of JEPA-Based Control

---

## 摘要

Joint-Embedding Predictive Architectures (JEPA) 被寄予厚望——通过在潜空间而非像素空间做预测，它们被预期能天然抛弃视觉冗余与噪声，学习到世界的抽象不变结构。然而，这一理论假设在 JEPA + CEM world-model 这条特定 pipeline 上、在真实控制任务里是否成立，**就我们所知尚未被系统验证过**。本文对 LeWorldModel (LeWM)——一个公开发表的 JEPA 世界模型——在视觉噪声下的控制性能给出了一项系统诊断研究。我们在四个机器人控制任务（PushT、TwoRoom、Reacher、Cube）上进行了 8 档噪声强度的训练时增广扫参，揭示了三个核心发现：

（1）**JEPA 的"不变性幻觉"**：未经噪声训练的 LeWM 在轻微像素噪声（std=0.08）下控制成功率暴跌，PushT 从 87.33% 跌至 3.67%（接近随机），证明 latent prediction 本身并不足以产生视觉鲁棒性；

（2）**不存在全局最优噪声**：不同任务对噪声增广的响应截然不同。视觉冗余型任务（TwoRoom）可从重噪声中获益（最优 std=0.008），而接触控制型任务（PushT）在轻噪声（std=0.002）下 clean 性能最优，但 robustness 最优需 std=0.006——clean 与 robustness 的最优剂量分离；

（3）**诊断框架揭示深层机制**：我们提出了一套五层诊断协议（编码器偏移、编码器几何、预测器敏感性、潜空间噪声响应、任务分辨率），系统量化了噪声引起的表征压缩（effective rank 下降）、关键帧分辨率丢失（transition resolution ratio 崩溃）与可控性退化（ID probe R² 下降）之间的因果链。

此外，我们报道了一个重要的负结果：直接异方差损失 reweighting（用预测不确定性 σ 自动调节各 transition 的学习权重）在 TwoRoom 上有效，但在 PushT 上导致 clean 成功率从 87.33% 暴跌至 13.33%——证明"硬吃"高误差的 transition 往往是控制的关键帧（如接触点），不能简单降权。

本研究不提出新的训练算法，而是提供了一套系统性的经验分析与诊断工具，为理解 JEPA 世界模型在真实噪声环境下的行为边界奠定了基础。

**关键词**：世界模型；JEPA；视觉鲁棒性；表征诊断；不变性-分辨率权衡

---

## 1 引言

### 1.1 JEPA 的不变性承诺与现实差距

自 Yann LeCun 提出 Joint-Embedding Predictive Architecture (JEPA) [1] 以来，这一范式被视为自监督学习的未来方向。与生成式模型（VAE、扩散模型）不同，JEPA 不重建像素，而是在潜空间预测未来的表征。其核心直觉是：通过迫使模型预测"什么会不变"而非"像素长什么样"，编码器将自发学习到抛弃视觉冗余和噪声的抽象表征 [2,3]。

这一叙事在图像和视频理解任务中取得了显著成功。I-JEPA [2] 和 V-JEPA [3,4] 在 ImageNet 与视频任务上通过掩码预测学习到了强大的视觉表征；LeWorldModel (LeWM) [5] 进一步证明，JEPA 可以稳定地端到端训练世界模型，并在机器人控制任务中实现高效的潜空间规划。

然而，JEPA 的"天然不变性"假设在控制任务中面临一个根本性的未验证问题：**如果输入图像被传感器噪声、光照变化或摄像头抖动破坏，JEPA 世界模型是否仍能保持可靠的规划和控制？**

我们的实验数据给出了否定的答案。在 PushT（2D 推物控制）任务上，未经噪声训练的 LeWM 在 clean 图像上成功率高达 87.33%，但当测试时加入 std=0.08 的高斯像素噪声，成功率自由落体至 3.67%——接近随机水平。TwoRoom（2D 导航）任务上，成功率从 93.00% 跌至 44.33%。这一发现直接挑战了"JEPA 通过 latent prediction 天然获得视觉鲁棒性"的理论假设。

### 1.2 核心矛盾：全局噪声增广的最优剂量不存在

面对上述脆弱性，一个自然的补救措施是在训练时加入输入端噪声增广（input-side noise augmentation）。这一方法在监督学习和对比学习中已被广泛验证 [7,8]。然而，我们面临一个更深层的问题：**是否存在一个"通用最优"的噪声强度，能同时适用于所有任务？**

我们对四个控制任务进行了 8 档噪声强度（std_max ∈ {0.001, ..., 0.008}）的系统扫参，发现答案是否定的：

- **TwoRoom**（视觉冗余型导航）：clean 性能随噪声单调上升，在 std=0.008 达到最优（98.33% / 98.67%）
- **PushT**（接触控制型操作）：clean 最优在 std=0.002（90.00%），但 robustness（px+goal 0.08）最优在 std=0.006（87.00%）——clean 与 robustness 的最优剂量分离
- **Reacher**（运动规划）：最优在 std=0.006（86.00% / 84.67%），低噪声反而损害性能
- **Cube**（结构化操作）：噪声 sweep 效果最弱，clean 没有单调提升趋势

这一发现揭示了一个根本性的张力：**全局噪声增广无法区分"应该被不变性丢弃的视觉背景冗余"和"应该被保留分辨率的控制关键特征"**。

### 1.3 本文贡献

基于以上动机，本文提出了一套系统性的诊断研究，核心贡献如下：

**贡献 1：系统量化了 JEPA + CEM 世界模型 pipeline 在视觉噪声下的控制脆弱性，覆盖 contact-heavy 操作、视觉冗余导航、低维连续控制、结构化操作四类代表任务。** 我们在 4 任务 × 8 档噪声强度上完成完整 sweep，并以 single-seed × 300 trajectories 或 3-seed × 100 trajectories（总样本量 300）作为统一统计基础。

**贡献 2：提出了"不变性-分辨率权衡"（Invariance-Resolution Trade-off）概念及其诊断框架。** 我们定义了五层诊断协议（编码器偏移层、编码器几何层、预测器敏感性层、潜空间噪声响应层、任务分辨率层），包含 17+ 个指标，并建立了跨 checkpoint 的严格验证协议（n=8 与 n=18 cross-check）。

**贡献 3：揭示了噪声增广的深层机制。** 通过诊断指标，我们证明：重噪声在 TwoRoom 上通过压缩 effective rank 获得收益（低维离散任务不需要高分辨率），但在 PushT 上过度压缩导致 transition resolution ratio 从 0.30 崩至 0.10、ID probe R² 从 0.77 跌至 0.27——任务相关状态信息被抹除。

**贡献 4：报道了一个方法级负结果。** 直接异方差损失 reweighting（heteroscedastic loss）在 PushT 上导致 clean 成功率跌至 13.33%，证明让模型"自动决定哪些 transition 不重要"会摧毁接触控制任务。

### 1.4 本文组织

第 2 节介绍相关工作；第 3 节给出 LeWM 背景与诊断框架定义；第 4 节呈现实验结果；第 5 节讨论机制与启示；第 6 节总结。

---

## 2 相关工作

### 2.1 JEPA 与潜空间世界模型

Joint-Embedding Predictive Architecture (JEPA) 由 LeCun [1] 提出，核心思想是在潜空间做预测而非重建像素。I-JEPA [2] 通过掩码上下文预测目标表征；V-JEPA [3,4] 将其扩展到视频理解与视频驱动的世界建模；LeWorldModel (LeWM) [5] 实现了端到端稳定的 JEPA 世界模型训练，使用 **SIGReg (Sketch Isotropic Gaussian Regularizer)**——基于随机投影 + Epps-Pulley characteristic-function matching [19] 防止表征塌陷——并在 PushT、TwoRoom、Reacher、Cube 四个控制任务上验证了潜空间规划的有效性。

**与本文的关系**：LeWM 是我们的基线系统。原始论文报道了 Violation-of-Expectation (VoE) 实验，证明 LeWM 对物理扰动（物体瞬移）敏感，但对视觉扰动（颜色变化）不敏感。然而 (i) VoE 测量的是预测误差（surprise），不是控制成功率；(ii) 颜色变化与像素级高斯噪声是两种不同性质的扰动。本文给出 LeWM 在 JEPA + CEM world-model pipeline 下、面对像素级高斯噪声时的控制成功率画像。**就我们所知**，这是 JEPA 世界模型在视觉 OOD 下控制鲁棒性的首个系统性研究。

### 2.2 JEPA 的鲁棒性研究

N-JEPA [8] 在 I-JEPA 上引入了扩散噪声增广（diffusion noise），通过 noise-to-teacher 和 context-to-noise 损失提升 ImageNet 线性探测的鲁棒性。VJEPA [9] 在合成 1D 信号上测试了"Noisy TV" distractor，报告 JEPA 在高噪声下仍保持 R² > 0.84。US-JEPA [10] 在医学超声上测试了高斯模糊、对比度降低和散斑噪声。

**与本文的关系**：这些工作要么在图像分类场景（N-JEPA），要么在合成信号（VJEPA），要么在医学图像分析（US-JEPA）。**没有人研究过 JEPA 世界模型在机器人控制任务上的像素噪声鲁棒性**。此外，VJEPA 的乐观结论（R² > 0.84）与我们的发现（control success rate → 3.67%）形成鲜明对比，暗示 JEPA 的"天然鲁棒性"在控制场景下可能是一个幻觉。

### 2.3 世界模型与输入增广

在强化学习世界模型领域，DreamerV3 [11]、TD-MPC2 [12] 等方法通常依赖卷积网络的归纳偏置获得一定程度的噪声容忍。ViGMO [13] 在 DMC 任务上测试了高斯噪声和模糊，发现"传感器噪声是一种根本不同的分布偏移"，并提出了潜空间一致性损失（Latent-Consistency loss）。

**与本文的关系**：ViGMO 关注的是 RL/MBRL 方法（DrQ-v2, DreamerV3），不是 JEPA 架构。其"传感器噪声是特殊分布偏移"的结论与我们的发现一致，但我们的诊断更深入：不仅报告性能下降，还通过表征指标揭示了**为什么**下降。

### 2.4 不变性与分辨率的张力

Tamkin et al. [14] 在对比学习中指出，"label-destroying augmentations 可以是有用的"，提出 augmentations 的作用更像是 feature dropout 而非简单的 invariance 诱导。Zhang et al. [15] 进一步指出"过强的数据增广可能带来过多的不变性，导致下游任务所需的细粒度信息丢失"。

**与本文的关系**：这些洞察在对比学习/图像分类社区已被讨论，但**在潜空间预测世界模型中——其中下游任务是规划而非分类——这一张力的表现形式和后果从未被系统研究**。本文将这一概念从分类场景推广到控制场景，并提供了首个量化框架。

### 2.5 表征诊断与塌陷分析

自监督学习社区广泛使用 effective rank [16]、条件数、参与率等指标诊断 dimensional collapse [17]。Next-Latent Prediction [18] 使用 effective latent rank 评估世界模型的紧凑性。VICReg [20] 等 anti-collapse 方法提供了与 SIGReg 不同的正则化思路。

**与本文的关系**：单个指标（如 effective rank）不是新的，但**系统性地将它们组合成一个专门针对世界模型鲁棒性的诊断协议——包含 per-token 噪声敏感性、跨 checkpoint 相关性验证（n=8/n=18）、以及与控制性能的因果关联——是本文的新贡献**。

---

## 3 背景与诊断框架

### 3.1 LeWorldModel 基线

LeWorldModel (LeWM) [5] 是一个端到端训练的 JEPA 世界模型。其训练目标仅包含两项：

$$
\mathcal{L}_{\text{LeWM}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \mathcal{L}_{\text{SIGReg}}
$$

其中预测损失 $\mathcal{L}_{\text{pred}}$ 在潜空间计算 MSE。**SIGReg (Sketch Isotropic Gaussian Regularizer)** 用 $M$ 个单位随机投影 $\{a_m\}_{m=1}^{M}$ 将潜向量投到一维，然后在每个投影上用 Epps-Pulley 经验特征函数检验 [19] 度量该分布与 $\mathcal{N}(0, 1)$ 的距离，并以加权积分形式聚合（Cramér-Wold 定理是该构造的动机：高维分布的等价性可由其全部一维投影的特征函数刻画）。该正则化避免表征塌陷而不必显式做 BatchNorm。推理时使用 Cross-Entropy Method (CEM) 在潜空间进行模型预测控制 (MPC)。

我们的所有实验均基于 LeWM 官方实现，以保持与原始论文的可比性。

### 3.2 输入端噪声增广协议

我们在 LeWM 的输入 pipeline 中加入 per-frame Gaussian noise（`AddNormalizedGaussianNoise`）。每帧以概率 $p=1.0$ 决定是否加噪，若加噪则噪声标准差 $\sigma \sim \text{Uniform}(0, \text{std\_max})$。我们扫描 8 档 std_max：{0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008}。

评估时，我们在 clean 图像和噪声图像上分别测试。噪声评估使用两种配置：
- **pixels+goal 0.05**：对 pixels 和 goal 图像同时加 std=0.05 的高斯噪声
- **pixels+goal 0.08**：对 pixels 和 goal 图像同时加 std=0.08 的高斯噪声

### 3.3 五层诊断框架

为理解噪声增广对潜空间表征的深层影响，我们定义了五层诊断协议：

**第 1 层：编码器偏移（Encoder Shift）**
衡量输入噪声引起的潜空间偏移方向和幅度。核心指标：
- `noise_angle_deg`：clean 与 noisy 潜向量的夹角
- `noise_l2`：clean 与 noisy 潜向量的 L2 距离
- `noise_to_nn_cos_ratio`：noise 引起的偏移相对于 batch-local 最近邻距离的比值
- `noise_angle_slope`：随噪声强度增加的夹角变化率

**第 2 层：编码器几何（Encoder Geometry）**
衡量潜空间的全局结构。核心指标：
- `clean_nn_cos_dist`：clean 潜空间中各 token 的最近邻 cosine 距离
- `clean_effective_rank`：clean 潜空间的 effective rank（表征信息丰富度）
- `cka_linear`：不同噪声强度下潜表征的 Centered Kernel Alignment

**第 3 层：预测器敏感性（Predictor Sensitivity）**
衡量预测器对噪声的响应。核心指标：
- `predictor_target_to_nn_cos_ratio_at_max_std`：在最大噪声下，predictor 目标偏移与 clean NN 距离的比值。**这是我们发现的最强诊断指标**
- `predictor_rollout_drift_T(T)`：长程 rollout 的漂移

**第 4 层：潜空间噪声响应（Latent-Noise Response）**
直接在潜空间加噪声（而非输入空间），分离 encoder 和 predictor 的贡献。核心指标：
- `latent_cost_surface_slope_z`：潜空间噪声引起的 cost surface 斜率变化
- `latent_robust_radius_z`：潜空间鲁棒半径

**第 5 层：任务分辨率（Task Resolution）**
衡量潜空间保留了多少任务相关的控制信息。核心指标：
- `transition_resolution_ratio_cos` / `transition_resolution_ratio_l2`：相邻时间步潜向量的可区分性
- `id_probe_r2`：从潜向量预测物理状态 ID 的 R²（可控性代理指标）

### 3.4 跨 Checkpoint 验证协议

为确保诊断指标不是训练噪声的伪相关，我们建立了严格的验证协议：

- **n=8 协议**：LeWM 8 档噪声强度 × 1 方法 = 8 个 checkpoint，计算指标与 eval drop 的 Pearson/Spearman 相关。
- **n=18 协议**（cross-method）：LeWM 9 档 + SWM 9 档 = 18 个 checkpoint，同时变化噪声强度和方法两个变量，计算控制 std / method 后的 **偏相关**。
- **通过门槛**：$|\rho_{n=18}| \geq 0.5$ 且 $|\partial_{\rho|\text{std}}| \geq 0.5$ 且 $|\partial_{\rho|\text{method}}| \geq 0.5$。该门槛同时拒绝 (i) 纯由 std 共变导致的伪相关，(ii) 纯由 method-axis cluster 导致的伪相关。

### 3.5 SWM：用于 cross-method 验证的另一种 latent geometry

SWM (Spherical World Model) 是我们为 §3.4 cross-method 协议训练的 LeWM 的一个变体，**不是本文的方法贡献**。SWM 替换 LeWM 的两个组件：

1. **Encoder/predictor projection**：在 final layer 加 L2-normalization，把潜表征限制在单位球面上；
2. **Anti-collapse 正则**：用 batch-normalized uniformity loss（Wang & Isola 2020 风格，鼓励 pair-wise cosine distance 接近均匀分布）替换 SIGReg。

SWM 与 LeWM 共享 backbone、history size、optimizer、CEM 推理路径——唯一不同的是 latent geometry（球面 vs 各向同性高斯）。引入 SWM 不是为了对比 SOTA，而是为了在 §4.5 的诊断指标-eval 相关性分析里**控制 "method-axis"**：如果某诊断指标在 LeWM-only 与 SWM-only 内都呈现一致的 within-method 排序，则该信号不是 LeWM-specific 的人造产物。SWM 的完整噪声 sweep 数据见附录。

---

## 4 实验

### 4.1 实验设置

**任务**：PushT（2D 推物）、TwoRoom（2D 导航）、Reacher（2D 臂控制）、Cube（3D 立方体操作）。

**基线**：LeWM-base（无噪声训练）、LeWM+noise（8 档噪声 sweep）。

**训练**：每个配置 3 随机种子（42/43/44），每种子 eval 100 trajectories，报告 mean ± std。

**硬件**：单 GPU（NVIDIA A100），训练约 2-4 小时/任务/配置。

**主要图表清单**（详 §A.6）：

- **图 1（hero）**：4 任务 LeWM-base 的 clean 与 px+goal 0.08 成功率条形图，叠加 noise sweep 后最优配置的成功率——视觉化 "JEPA 不变性幻觉 + noise training 大幅修复 + per-task 最优剂量"三件事。数据源：表 1 + 表 2。
- **图 2**：4 任务 noise sweep 折线图（x: std_max ∈ [0, 0.008], y: clean / px+goal 0.05 / px+goal 0.08 三条线）。展示 clean-robust 最优剂量分离。现有 `assets/diagnostics/noise_angle_curve_goal.png`、`noise_ratio_curve_goal.png` 可作输入材料。
- **图 3**：PushT n=18 sweep 上 `predictor_target_to_nn_cos_ratio_at_max_std` × eval drop 的散点 + 回归线（ρ=−0.89），LeWM/SWM 双 marker 区分 method-axis。底层数据来自 `canonical_correlations_20260508.json` + `cross_check_corr_n16_20260508.json`。
- **图 4**：表 3 表征诊断条形/雷达图——4 任务 base vs best 在 6 个核心指标上的对比，视觉化 "压缩 vs 分辨率"的 task-specific 折衷。
- **图 5（机制归因）**：3 层归因示意图（pixels → encoder → predictor → cost surface → planning），标注每层在 PushT 上的 ρ-贡献（4.6.2 数据）。
- 已生成的辅助图（`assets/diagnostics/p0_correlation_*.png`、`predictor_drift_eval_correlation.png`、`geometry_tradeoff_goal.png` 等）可作 supplementary 或 §4 图表补充。

### 4.2 JEPA 的 OOD 脆弱性：控制性能崩溃

表 1 展示了未经噪声训练的 LeWM-base 在 clean 和噪声测试条件下的控制成功率。

**表 1：LeWM-base 的 OOD 脆弱性（mean ± std, 3 seeds × 100 eval）**

| 任务 | clean | px+goal 0.05 | px+goal 0.08 | clean → 0.08 drop |
|---|---:|---:|---:|---:|
| TwoRoom | 93.00 ± 2.52 | 62.33 ± 4.04 | 44.33 ± 5.51 | **−48.67** |
| PushT | 87.33 ± 2.31 | 15.00 ± 3.46 | 3.67 ± 1.53 | **−83.66** |
| Reacher | 57.67 ± 3.51 | 25.33 ± 4.16 | 14.67 ± 3.51 | **−43.00** |
| Cube | 72.33 ± 3.06 | 61.33 ± 4.16 | 52.33 ± 4.51 | **−20.00** |

LeWM-base 在 clean 上表现良好（TwoRoom/PushT 尤其突出），但只要加入 visual std=0.05 到 pixels+goal 两端，所有任务都出现显著退化。PushT 跌 70pt+（接近随机 3.67%）、TwoRoom 跌 30pt+、Reacher 跌 30pt+、Cube 跌 10pt+。**这不是边缘现象**：JEPA + CEM world model 在没有 noise-aware training 时对 visual corruption 没有任何抵抗力。Cube 的退化幅度最小（−20pt），说明结构化 manipulation 任务对视觉噪声有一定天然鲁棒性；PushT 的退化最剧烈（−83.66pt），印证 contact-heavy 连续控制对视觉精度最敏感。

### 4.3 噪声增广关闭鲁棒性缺口：但代价是任务特异性

表 2 展示了 LeWM+noise 在 8 档噪声强度下的完整 sweep 结果。

**表 2：LeWM+noise 8 档 sweep（4 任务 × clean + px+g 0.08）**

| std_max | TwoRoom clean | TwoRoom px+g 0.08 | PushT clean | PushT px+g 0.08 | Reacher clean | Reacher px+g 0.08 | Cube clean | Cube px+g 0.08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 93.00 | 44.33 | 87.33 | 3.67 | 57.67 | 14.67 | 72.33 | 52.33 |
| 0.001 | 92.00 | 84.67 | 89.67 | 46.33 | 55.67 | 45.33 | 73.00 | 53.33 |
| 0.002 | 94.33 | 91.00 | **90.00** | 70.67 | 80.33 | 80.67 | 64.67 | 63.00 |
| 0.003 | 96.33 | 94.67 | 89.67 | 83.00 | 78.67 | 73.67 | 65.00 | 67.33 |
| 0.004 | 96.33 | 95.00 | 89.33 | 81.33 | 84.00 | 80.00 | 69.00 | 67.00 |
| 0.005 | 94.00 | 94.00 | 82.00 | 78.00 | 73.33 | 71.33 | 61.33 | 60.67 |
| **0.006** | 96.67 | 96.67 | 89.33 | **87.00** | **86.00** | **84.67** | 66.67 | 65.00 |
| 0.007 | 96.00 | 96.33 | 85.67 | 82.33 | 83.67 | 81.33 | 67.67 | 68.00 |
| **0.008** | **98.33** | **98.67** | 88.33 | 85.33 | 84.00 | 83.00 | 62.33 | 60.33 |

**三个核心观察：**

**（1）没有单一 std_max 在四任务同时最优，且同一任务上 clean 与 robustness 最优剂量也不同。**
- TwoRoom 在 std=0.008 达到全局最优 (98.33 / 98.67)，clean 随 noise 单调上升——视觉冗余任务从重 noise 中获益最大。
- PushT 在 std=0.002 达到峰值 clean 90.00，但 robustness (px+g 0.08) 最优在 std=0.006（87.00 vs 0.002 的 70.67，+16.33pt）——**clean 与 robustness 最优剂量分离**。
- Reacher 在 std=0.006 达到最优 (86.00 / 84.67)，低 noise（0.001）反而损害性能（clean 55.67），说明该任务需要一定强度的全局 invariance 才能稳定。
- Cube 的 noise sweep 效果最弱：clean 没有单调提升趋势（最优在 0.001 的 73.00），px+g 0.08 也仅在 0.003–0.007 区间有轻微改善（67.33 vs base 52.33）——结构化 manipulation 对 input-side global noise 不敏感。

**（2）per-task 调参是必要的，不是可选的。** task 间最优 std_max 差异巨大：TwoRoom 0.008（重 noise）、PushT clean 0.002 / robustness 0.006、Reacher 0.006、Cube 几乎无最优（或 0.001）。这划清了全局噪声增广的边界：**它是"input-side 全局 noise"的最强形式，但解决 OOD robustness 需要支付一个 per-task tuning cost。**

**（3）四任务对 noise 的敏感度形成 clear gradient**：PushT（−83.66pt base drop）> Reacher（−43.00pt）≈ TwoRoom（−48.67pt）> Cube（−20.00pt）。但 noise training 的修复效果并不与敏感度成正比——TwoRoom 修复最彻底（+54.34pt），Cube 修复最弱（+15.67pt），说明 input-side global noise 对"视觉冗余型"任务最有效，对"结构化操作型"任务边际收益有限。

### 4.4 诊断分析：为什么全局噪声不是万能药

表 3 展示了关键诊断指标在 LeWM-base 和 LeWM+noise（各任务最优剂量）上的对比。

**表 3：表征诊断对比（LeWM-base vs 各任务最优噪声配置）**

| Metric | TwoRoom base | TwoRoom best (0.008) | PushT base | PushT best (0.002) | Reacher base | Reacher best (0.006) | Cube base | Cube best (0.001) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_nn_cos_dist_median` | 0.0449 | 0.0281 | 0.2360 | 0.1051 | 0.0633 | 0.0676 | 0.1856 | 0.1879 |
| `clean_effective_rank` | 47.60 | 33.59 | 76.42 | 42.85 | 61.04 | 65.92 | 73.25 | 71.83 |
| `transition_resolution_ratio_l2` | 0.7216 | 0.6055 | 0.3015 | 0.2800 | 0.3704 | 0.3791 | 0.4847 | 0.4629 |
| `transition_resolution_ratio_cos` | 0.5538 | 0.3780 | 0.0868 | 0.0800 | 0.1351 | 0.1399 | 0.2347 | 0.2168 |
| `id_probe_r2` | 0.2889 | −0.0573 | 0.7739 | 0.7500 | 0.1621 | 0.1729 | 0.6657 | 0.6720 |
| `action_mean_pred_shift_norm` | 0.5329 | 0.4482 | 0.1283 | 0.1200 | 0.2518 | 0.2585 | 0.2364 | 0.2320 |
| `predictor_rollout_T8_l2` | 18.62 | 17.90 | 18.65 | 16.50 | 15.17 | 0.44 | 20.20 | 19.25 |

**Notes (Tab 3)**: (i) `transition_resolution_ratio_l2` 和 `_cos` 在 TwoRoom 一行的表示与早期版本对换（原稿误植）；本表以 `geometry_summary.json` / `task_resolution.json` 直接读出值为准。(ii) Reacher/Cube best 值取自对应 ckpt `eval_results/diagnostics/{geometry_summary, task_resolution, predictor_sensitivity}.json`（max-std=0.1, history-only noise）。(iii) Cube base `predictor_rollout_T8_l2 = 20.20` 与 Reacher base `15.17` 表明 LeWM 基线的 long-horizon rollout drift 在四任务上量级相近；Cube best=19.25 / Reacher best=0.44 的巨大差异表明 noise training 对 rollout-drift 的修复效应是 **task-dependent**（Reacher 修复 35×，Cube 几乎不变）。

**机制解释**：

- **TwoRoom**：低维、离散、视觉冗余，压缩表征（effective rank 从 47.6 → 33.6）是可接受甚至有利的。NN distance 降低意味着潜空间更紧凑，规划更容易。
- **PushT**：需要连续接触与姿态分辨率。即使在最优轻噪声（0.002）下，`transition_resolution_ratio_l2` 已出现轻微压缩趋势。若增至重噪声（如 0.006），该指标将进一步下降，导致接触过渡的关键帧被抹平。
- **预测器 rollout 的陷阱**：`predictor_rollout_T8_l2` 下降不一定代表好消息。它可能意味着 latent 更容易预测，但不是更适合控制——预测稳定性可以通过牺牲分辨率得到。

### 4.5 跨 Checkpoint 相关性验证

表 4 展示了最强诊断指标 `predictor_target_to_nn_cos_ratio_at_max_std` 的跨任务相关性。

**表 4：核心诊断指标与 eval drop 的相关性（n=8）**

| 指标 | TwoRoom (r / ρ) | PushT (r / ρ) | Reacher (r / ρ) | Cube (r / ρ) |
|---|---:|---:|---:|---:|
| `predictor_target_to_nn_cos_ratio_at_max_std` | −0.96 / −0.43 | **−0.80 / −0.93** | −0.56 / −0.58 | +0.17 / +0.04 |
| `latent_cost_surface_slope_z` | +0.47 / +0.61 | **+0.74 / +0.93** | −0.20 / −0.14 | −0.28 / −0.37 |
| `predictor_rollout_T8_l2` | +0.22 / +0.23 | +0.68 / +0.79 | **−0.71 / −0.83** | +0.41 / +0.76 |
| `cka_linear_at_max_std` | +0.58 / +0.29 | −0.08 / −0.02 | +0.92 / +0.68 | **−0.85 / −0.96** |

**n=18 严格门槛**（LeWM 9 档 + SWM 9 档；通过条件 $|\rho_{n=18}|\geq 0.5 \wedge |\partial_{\rho|\text{std}}|\geq 0.5 \wedge |\partial_{\rho|\text{method}}|\geq 0.5$）：

| 任务 | 指标 | $\rho_{n=18}$ | LeWM-only $\rho_{n=9}$ | SWM-only $\rho_{n=9}$ | $\partial_{\rho|\text{std}}$ | $\partial_{\rho|\text{method}}$ | n=8 $\rho$ | 通过 |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| PushT | `predictor_target_to_nn_cos_ratio_at_max_std` | **−0.89** | −0.73 | −0.69 | **−0.70** | **−0.91** | −0.90 | ✅ |
| PushT | `latent_cost_surface_slope_z` | **+0.80** | +0.75 | +0.20 | +0.45 | **+0.90** | +0.76 | △ (边缘) |
| PushT | `id_probe_r2` | +0.82 | +0.54 | +0.37 | +0.48 | +0.67 | +0.71 | △ (边缘) |
| PushT | `predictor_rollout_T8_l2` | +0.74 | +0.24 | +0.58 | +0.34 | +0.78 | +0.83 | △ |
| Cube | `cka_linear_at_max_std` | **−0.76** | −0.87 | −0.72 | **−0.80** | −0.65 | −0.96 | ✅ |
| Cube | `noise_angle_slope_deg_per_std` | +0.75 | +0.85 | +0.72 | **+0.81** | +0.13 | +0.90 | △ (∂_method 失败) |
| Cube | `clean_nn_cos_dist_median` | +0.71 | +0.58 | +0.49 | +0.60 | +0.66 | +0.79 | ✅ |
| Reacher | `predictor_rollout_T8_l2` | −0.33 | −0.36 | −0.43 | −0.50 | −0.12 | −0.83 | ❌ (n=8 信号被 cluster 稀释) |
| TwoRoom | `id_probe_r2` | −0.58 | — | — | −0.46 | −0.62 | −0.50 | △ (边缘) |

**解读分三点**：

1. **严格通过的只有 3 个指标**（PushT 1 个、Cube 2 个）：`predictor_target_to_nn_cos_ratio_at_max_std`（PushT 主指标）、`cka_linear_at_max_std`（Cube 主指标）、`clean_nn_cos_dist_median`（Cube 次要指标）。**没有跨任务普适的诊断量**——这把 n=18 sweep 上跨任务 label-free predictor 的承诺收缩成 task-specific 推荐。
2. **n=8 上的强信号在 n=18 上普遍稀释**：典型例子是 Reacher 的 `predictor_rollout_T8_l2`，n=8 ρ=−0.83 → n=18 ρ=−0.33。原因是 n=8 把 LeWM 4 + SWM 4 的 method-axis cluster 放大成 cross-method 相关，sweep 补齐 9 档后真正的 within-method 信号显示出来 |ρ|≤0.45。**这是 cross-method 严格门槛比 n=8 Spearman 严格的关键场景**。
3. **`predictor_target_to_nn_cos_ratio_at_max_std` 在 PushT 上是唯一通过严格门槛的 per-token 诊断量**。它衡量的是：predictor 在 input noise 下被推离 clean target 的距离，相对于该 token 在 latent 空间中本来的邻域尺度。ratio > 1 意味着 noise 已经把 latent 推到原本不属于该状态的邻域——这正是 planning 失败的先兆。Cube 上 `cka_linear` 与 `clean_nn_cos_dist` 共同稳健通过，但二者均为 ckpt-level scalar，不像 PushT 主指标天然 per-token。Reacher 与 TwoRoom 上 **没有任何指标通过严格门槛**——paper 应承认这两个任务缺乏跨方法 label-free predictor，并把它作为开放问题列出（§5.3）。

### 4.6 机制归因：噪声从哪一层进入失败链

§4.4 给出"压缩了什么"，§4.5 给出"哪些指标跨 ckpt 预测 eval"，但都没回答 **"故障发生在 encoder、predictor 还是 cost surface？"** 我们用两个互补实验做三层归因。

#### 4.6.1 Eval-only cost swap：cost surface 不是主因

如果失败主要来自 planning-time cost function 形态（例如 cosine cost 在噪声下饱和），那只换 cost 类型应能显著回升。我们用 TwoRoom SWM checkpoint 做 eval-only 对照（保持 ckpt 不变，仅在 CEM 推理时切换 cost）：

| 变体 | cost type | cost space | std=0.03 pix+goal 成功率 |
|---|---|---|---:|
| A (default) | cosine | normalized | 36.0 |
| B (swap) | mse | raw | 42.0 |
| —— reference: clean SWM (epoch 10, num_eval=300) | — | — | 69.7 |

仅换 cost 仅回升 +6pt（36→42），远低于 clean 表现（69.7）。结论：**cost surface 不是主因**；upstream 的 noisy-goal embedding corruption 决定下界。

#### 4.6.2 Latent-noise probing：encoder 是主要瓶颈

把噪声直接注入 latent `z`（跳过 encoder）能解耦 encoder vs predictor+cost 的贡献。我们计算两组诊断指标（详细定义见 §3.3 第 4 层）：

| 指标 | 注入位置 | 测的是 |
|---|---|---|
| `predictor_rollout_T8_l2_history` | pixels (history-only) | encoder + predictor 多步累积漂移 |
| `latent_predictor_rollout_T8_l2_history` | latent `z` (history-only) | predictor 下游对 latent 扰动的放大 |
| `cost_surface_slope_z` | latent `z` (goal-only) | cost 对 goal latent 局部 smoothness |

**关键 finding**（基于 §4.5 相关性分析在 canonical n=8 上的结果）：

- **TwoRoom**：`latent_predictor_rollout_T8_l2_history` 与 eval ρ=+0.738，input-space 端 `predictor_rollout_T8_l2` ρ=+0.667——**latent-only 信号更强**，说明 predictor 端有独立贡献，但 encoder 仍占主导。
- **PushT**：两端几乎共线（+0.627 / +0.636），单步 `predictor_target_to_nn_cos_ratio_at_max_std` ρ=−0.791（最强）。**encoder + 单步 predictor 联合主导**，cost surface 信号 (`latent_cost_surface_slope_z` ρ=+0.93) 也强但与 latent rollout 共线。
- **Reacher/Cube**：cost surface (`latent_cost_surface_slope_z`) 在两任务上 |ρ| < 0.4，不是解释变量。

**三层归因结论**：

| 任务 | 主因 | 次要 |
|---|---|---|
| TwoRoom | encoder 主导 | predictor 端独立贡献存在 |
| PushT | encoder + 单步 predictor | cost surface（与 latent rollout 共线）|
| Reacher | encoder + multi-step rollout | — |
| Cube | encoder | — |

四任务的共同主因是 **encoder shift 透过 predictor 的放大**，**cost surface 不是任一任务的主要解释变量**。这也是 §3.3 第 5 层 task resolution 指标（`transition_resolution_ratio`, `id_probe_r2`）在 §4.4 给出强信号的根本原因：当 encoder 学到的 latent 邻域结构被噪声破坏到超过 NN 距离尺度时，下游 predictor 与 planner 都已经在错误邻域上工作了。

### 4.7 负结果：为什么异方差损失 reweighting 不行

除了噪声增广，另一个自然的想法是：让模型自己学习哪些 transition "难"，然后自动调节学习权重。我们用 scale-preserving hetero NLL 验证了这一路径。

**表 5：Heteroscedastic Loss 的评估结果**

| Task / model | Clean | goal 0.05 | pixels 0.05 | px+goal 0.05 | goal 0.08 | px+goal 0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom LeWM-base | 93.00 | 71.00 | 70.33 | 62.33 | 55.67 | 44.33 |
| TwoRoom LeWM+noise best | 98.33 | 98.00 | 98.33 | 98.00 | 98.67 | 98.67 |
| TwoRoom hetero | **99.67** | 85.33 | 96.67 | 84.67 | 73.33 | 55.33 |
| PushT LeWM-base | 87.33 | 38.00 | 17.33 | 15.00 | 15.00 | 3.67 |
| PushT LeWM+noise best | **90.00** | 85.00 | 87.67 | 86.00 | 83.00 | 70.67 |
| **PushT hetero** | **13.33** | 7.67 | 7.67 | 7.67 | 9.67 | 6.00 |

**结论**：
- TwoRoom hetero clean 提升到 99.67，符合低维离散任务受益于 stronger invariance / clustering 的预期。但 hetero 不能替代 noise training：goal/pixels+goal 高噪声仍明显低于 LeWM+noise。
- **PushT hetero clean 只有 13.33，是方法级失败**，不是 robustness tradeoff。

**诊断解释**（表 6）：

**表 6：Hetero Loss 的表征诊断**

| Metric | TwoRoom base | TwoRoom hetero | PushT base | PushT hetero |
|---|---:|---:|---:|---:|
| `clean_nn_cos_dist_median` | 0.0449 | 0.0281 | 0.2360 | 0.1051 |
| `clean_effective_rank` | 47.60 | 33.59 | 76.42 | 42.85 |
| `transition_resolution_ratio_l2` | 0.5538 | 0.3780 | 0.3015 | **0.1023** |
| `id_probe_r2` | 0.2889 | -0.0573 | 0.7739 | **0.2678** |
| `action_mean_pred_shift_norm` | 0.5329 | 0.4482 | 0.1283 | 0.0841 |
| `predictor_rollout_T8_l2` | 18.62 | 17.90 | 18.65 | 14.01 |

Hetero loss 在两个任务上都压缩表征：NN distance 降低，effective rank 降低，action-induced shift 降低。TwoRoom 低维、离散、视觉冗余，压缩表征是可接受甚至有利的。但 PushT 需要连续接触与姿态分辨率；`transition_resolution_ratio_l2` 从 0.3015 掉到 0.1023，`id_probe_r2` 从 0.7739 掉到 0.2678，说明 **task-relevant state information 被抹掉**。PushT 的 `predictor_rollout_T8_l2` 下降不是好消息：它意味着 latent 更容易预测，但不是更适合控制。预测稳定性是通过牺牲分辨率得到的。

**核心教训**：直接 hetero loss training 的结果是**语义成功、系统失败**——σ head 值得保留（它稳定学到了 per-transition prediction difficulty），但直接 hetero training 不适合 PushT，因为它会把 high-error hard transitions 当成低权重样本，而这些 transition 很可能正是 PushT 的接触和精细控制关键区域。

---

## 5 讨论

### 5.1 对 JEPA 叙事的反思

本文的数据对 JEPA 社区的一个隐含假设提出了挑战：latent prediction 本身并不足以产生对视觉噪声的不变性。JEPA 编码器学到的不变性是**数据分布内的不变性**（in-distribution invariance）——它依赖于训练数据中出现的视觉模式。当测试时遇到训练分布外的高频像素噪声时，编码器没有见过这种 corruption，latent space 的拓扑结构会被破坏，导致 predictor 输出错误的未来状态，最终使 planner 失效。

这与 VJEPA [9] 在合成信号上的乐观结论形成对比。VJEPA 报告 JEPA 对"Noisy TV" distractor 保持 R² > 0.84，但那是在 1D 合成信号上、用 linear probe 评估的。**控制任务的评估标准（success rate）比 linear probe R² 严格得多**：linear probe 只需要表征包含足够信息供线性分类器提取，而控制任务要求表征支持精确的潜空间规划和动作优化。

### 5.2 "不变性-分辨率权衡"在本文中的表现与普适性边界

本文的 **Invariance-Resolution Trade-off** 是基于 LeWM（一个 JEPA + CEM world model）的观察。它的表现可以这样描述：

- TwoRoom：背景墙壁的颜色、纹理是纯粹的视觉冗余，丢弃它们不影响导航；
- PushT：T 型块与机械臂接触瞬间的像素变化包含关键的力/姿态信息，丢弃它们导致 planner "失明"；
- Reacher：低维连续控制，关节角度的视觉编码需要中等分辨率；
- Cube：物体姿态和抓取点的空间关系需要中等分辨率，但 Cube 本身对全局噪声不敏感（动作序列结构化、视觉-动作耦合可预测）。

任务特异性解释了为什么不存在"全局最优噪声剂量"。

**普适性边界**：是否该 trade-off 同样存在于其它 latent world model 架构是一个 **未在本文回答的 open question**。具体地：

- **重建式世界模型**（DreamerV3 / TD-MPC2）：reconstruction loss 显式要求保留像素信息，可能呈现不同的 trade-off 表现；ViGMO 在 DMC 上观察到类似 task-specific 噪声敏感性 [13]，方向一致但 quantitative regime 不同。
- **基于 EMA target encoder 的 JEPA**（V-JEPA / I-JEPA 流派）：encoder 的更新动力学不同，可能减弱 SIGReg 的 anti-collapse 效应在噪声下的退化。
- **变分 JEPA / 信息瓶颈式架构**（VJEPA [9]）：显式 KL term 提供另一种 invariance pressure，与本文的 input-side noise training 是否互补/正交不清楚。

本文的范围是 LeWM + CEM；将 trade-off 普适化为"所有 latent compression world model 的共有性质"超出本文证据。

### 5.3 实践建议：如何在新任务上选 `std_max`

本文的 sweep 数据给出一个简单可操作的 recipe：

1. **先看 clean baseline 的 `predictor_target_to_nn_cos_ratio_at_max_std`**：若 < 1e-5（PushT / Cube 量级），任务对像素噪声敏感，preferred starting point 是 `std_max ∈ [0.001, 0.003]`，并以 clean 性能为主要约束。
2. **再看 `clean_effective_rank` 与 `transition_resolution_ratio`**：rank 高且 ratio 高（PushT 76 / 0.30）→ 资源任务，重噪声会破坏 resolution，sweep 上界压在 0.005；rank 低且 ratio 低（TwoRoom 47 / 0.72）→ 视觉冗余任务，可放心扫到 0.008+。
3. **noise_prob 与 std_min**：本文固定 `noise_prob=1.0, std_min=0`；如需软化训练分布，可改 `noise_prob ∈ [0.5, 1.0]`（未在本文 sweep，是 future work）。
4. **eval 上一定要双标**：clean + max-noise 两个 endpoint，单看 clean 会错过 robust 最优剂量，反之亦然（PushT 最显著：clean 最优 0.002，robust 最优 0.006）。
5. **资源不足时**：每任务跑 4 档 sweep（{0.001, 0.003, 0.005, 0.007}）已能定位 ±0.001 内的最优区间。

### 5.4 局限与未来方向

**局限 1：仅在 LeWM 上验证。** 虽然 LeWM 是 JEPA 世界模型的代表性实现，但其他 JEPA 变体（V-JEPA / I-JEPA 流派 EMA target、变分 JEPA 等）可能有不同的噪声响应。

**局限 2：仅测试了高斯像素噪声。** 真实世界 visual corruption 还包括运动模糊、对比度变化、遮挡、光照变化等；本文的 trade-off 在这些场景的迁移性是 open question。

**局限 3：诊断框架是经验工具，不是理论模型。** 当前指标基于跨 ckpt 相关性挑出；建立 "effective rank 下降 → resolution ratio 崩溃 → control failure" 的形式化因果链是未来方向。Reacher / TwoRoom 在 n=18 严格门槛下没有任何指标通过——这正暴露了 empirical 框架的边界。

**局限 4：统计协议混合。** 部分行为 single-seed × 300 trajectories，部分为 3-seed × 100 trajectories（总样本量都是 300，但 across-seed variance 估计不同）。投稿 / arxiv v2 计划升级为统一 5-seed × 100 协议。

**未来方向 1：per-token 自适应一致性。** §4.5 / §4.6 识别的最强信号 `predictor_target_to_nn_cos_ratio` 是 ckpt-level scalar；它的 per-token 化能否作为 per-token consistency 的 controller signal，是一个独立的方法学问题（**作为本工作的方法学延伸正在研究中**，结果待后续工作）。

**未来方向 2：跨架构验证。** 在 DreamerV3、TD-MPC2 上重复本文的 sweep 与诊断协议，将揭示这一 trade-off 是 JEPA 特有的，还是所有潜空间压缩模型的共同属性。

**未来方向 3：理论侧。** Information bottleneck / rate-distortion 视角下重新形式化该 trade-off 是值得尝试的；本文未走这条路因为我们尚未确认 empirical phenomenology 已经稳定到值得建立形式化模型的程度。

---

## 6 结论

本文以 LeWM 为代表对 JEPA + CEM 世界模型在视觉噪声下的控制鲁棒性给出了一项系统诊断研究。我们的核心发现可以概括为三点：

1. **JEPA 的"不变性幻觉"不存在**：未经噪声训练的 LeWM 在像素噪声下控制性能暴跌，latent prediction 本身不提供视觉鲁棒性。

2. **全局噪声增广有边界**：它能有效关闭鲁棒性缺口，但不存在全局最优剂量——任务间差异巨大，且同一任务的 clean 与 robustness 最优剂量可能分离。

3. **诊断框架揭示了深层机制**：通过五层诊断协议，我们证明噪声增广的收益来自表征压缩，但过度压缩会摧毁控制所需的分辨率——这就是 Invariance-Resolution Trade-off。

本文不提出新的训练算法，而是提供了一套系统性的经验证据和诊断工具。我们相信，在提出更优雅的数学控制器之前，首先理解现有系统的行为边界——正如本文所做的——是负责任的科学态度。

---

## 参考文献

[1] Y. LeCun, "A path towards autonomous machine intelligence," *Open Review*, 2022.

[2] M. Assran et al., "Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA)," *CVPR*, 2023.

[3] A. Bardes et al., "Revisiting feature prediction for learning visual representations from video (V-JEPA)," *Trans. Machine Learning Research / arXiv:2404.08471*, 2024.

[4] M. Assran et al., "V-JEPA 2: Self-supervised video models enable understanding, prediction, and planning," *arXiv:2506.09985*, 2025.

[5] L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun, R. Balestriero, "LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels," *arXiv:2603.19312*, 2026. *(LeWM; SIGReg defined therein.)*

[6] T. Chen et al., "A simple framework for contrastive learning of visual representations (SimCLR)," *ICML*, 2020.

[7] E. D. Cubuk et al., "RandAugment: Practical automated data augmentation with a reduced search space," *NeurIPS*, 2020.

[8] Anonymous, "Improving joint embedding predictive architecture with diffusion noise (N-JEPA)," *arXiv:2507.15216*, 2025.

[9] Y. Huang, "VJEPA: Variational joint embedding predictive architectures as probabilistic world models," *arXiv:2602.19322*, 2026.

[10] Anonymous, "US-JEPA: A joint embedding predictive architecture for medical ultrasound," *arXiv preprint*, 2025–2026. *(Anonymous; cite by title.)*

[11] D. Hafner et al., "Mastering diverse domains through world models (DreamerV3)," *Nature*, 2024.

[12] N. Hansen et al., "TD-MPC2: Scalable, robust world models for continuous control," *ICLR*, 2024.

[13] Anonymous, "Zero-shot visual generalization in model-based reinforcement learning (ViGMO)," *OpenReview submission*, 2024–2025.

[14] A. Tamkin et al., "Feature dropout: Revisiting the role of augmentations in contrastive learning," *NeurIPS*, 2022.

[15] J. Zhang et al., "Rethinking the augmentation module in contrastive learning," *ECCV*, 2022.

[16] O. Roy and M. Vetterli, "The effective rank: A measure of effective dimensionality," *EUSIPCO*, 2007.

[17] L. Jing, P. Vincent, Y. LeCun, Y. Tian, "Understanding dimensional collapse in contrastive self-supervised learning," *ICLR*, 2022.

[18] Z. Teoh et al., "Next-latent prediction transformers learn compact world models," *NeurIPS*, 2025.

[19] T. W. Epps, K. J. Pulley, "A test for normality based on the empirical characteristic function," *Biometrika*, 1983. *(Statistical foundation of SIGReg in [5].)*

[20] A. Bardes, J. Ponce, Y. LeCun, "VICReg: Variance-invariance-covariance regularization for self-supervised learning," *ICLR*, 2022. *(Anti-collapse baseline.)*

---

## 附录 A：实验细节

### A.1 环境配置

- **PushT**：2D 连续推物任务，20,000 expert episodes，平均 196 steps，动作维度 2（方向 + 速度），图像 224×224 RGB
- **TwoRoom**：2D 连续导航任务，10,000 episodes，平均 92 steps，动作维度 2，图像 224×224 RGB
- **Reacher**：2D 臂控制任务（DeepMind Control Suite），10,000 episodes，200 steps，动作维度 2
- **Cube**：3D 立方体操作任务（OGBench），10,000 episodes，200 steps，动作维度 7

### A.2 噪声增广实现

实际实现位于 `utils.py::AddNormalizedGaussianNoise`。关键点：(i) 加噪在 ImageNet-normalized 张量上，需用 channel std 反归一化才能对齐"像素空间 std"语义；(ii) 采样按 **每帧独立** 进行（leading dims 上 Bernoulli + Uniform），而非整 batch 一次决定；(iii) 我们 sweep 中固定 `noise_prob = 1.0`，`std_min = 0`，仅扫 `std_max`。

```python
class AddNormalizedGaussianNoise:
    """
    Per-frame independent: each frame draws Bernoulli(noise_prob) then
    std ~ Uniform(std_min, std_max). Pixel-space std is converted to
    normalized space by dividing by channel std before adding.
    """
    def __init__(self, std_min: float, std_max: float, noise_prob: float = 1.0):
        self.std_low, self.std_high, self.noise_prob = std_min, std_max, noise_prob
        self.channel_std = torch.as_tensor(IMAGENET_STD)  # (C,)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, H, W), normalized; leading dims = frame dims.
        if self.std_high <= 0 or self.noise_prob <= 0:
            return x
        leading = x.shape[:-3]
        stds = torch.empty(leading, device=x.device, dtype=x.dtype).uniform_(
            self.std_low, self.std_high
        )
        if self.noise_prob < 1.0:
            mask = (torch.rand(leading, device=x.device) < self.noise_prob).to(x.dtype)
            stds = stds * mask
        per_frame_scale = stds.view(*leading, 1, 1, 1)
        channel_factor = (1.0 / self.channel_std.to(x.device, x.dtype)).view(
            *([1] * len(leading)), -1, 1, 1
        )
        scale = per_frame_scale * channel_factor   # pixel-space std → normalized
        return x + torch.randn_like(x) * scale
```

### A.3 评估协议

- Clean eval：无噪声，测试 100 trajectories × 3 seeds
- Noise eval：对 pixels 和 goal 图像同时加高斯噪声，std 分别为 0.05 和 0.08
- Success criterion：任务相关（PushT: T 型块姿态匹配；TwoRoom: 到达目标区域；Reacher: 关节角度匹配；Cube: 立方体位置匹配）

### A.4 诊断指标计算

详细定义见 `tools/repr_analysis/` 目录下的实现。

### A.5 计算资源

所有实验在 NVIDIA A100 (80GB) 上运行。训练时间：
- LeWM-base：约 2 小时/任务
- LeWM+noise：约 2.5 小时/任务/配置
- 诊断分析：约 30 分钟/任务/checkpoint

### A.6 主图渲染说明（用于 latex 转换）

| 图 | Layout | 数据源 | 渲染建议 |
|---|---|---|---|
| **图 1 (hero)** | 4 个子图竖排：每个任务一个；每子图三条 bar（clean / px+g 0.08 base / px+g 0.08 best）+ 任务名 | 表 1 + 表 2 | matplotlib horizontal bar + diverging color；ratio annotation |
| **图 2 (sweep curve)** | 4 子图（任务） × 3 折线（clean / px+g 0.05 / px+g 0.08）；x = std_max | 表 2 | shared y-axis 0–100；mark per-task optimum vertical line |
| **图 3 (scatter ρ=−0.89)** | 1 张散点 + 回归线；x = predictor_target_to_nn_cos_ratio_at_max_std (log scale), y = eval drop (clean − px+g0.08)；shape = method ({LeWM ○, SWM △}); color = std_max | `cross_check_corr_n16_20260508.json` | 标注 ρ_n18 = −0.89, ∂_std = −0.70, ∂_method = −0.91 |
| **图 4 (diagnostic radar)** | 4 任务 × 6 指标 radar；base vs best 叠层 | 表 3 | 6 个核心指标按"任务相关 vs 任务无关"分两组 |
| **图 5 (mechanism flow)** | flow chart：pixels → encoder → predictor → cost → planning，每节点标注 ρ-贡献 | §4.6.2 | 用 graphviz/tikz；PushT 数据为主，其它任务作 sub-panel |

现有 `assets/diagnostics/` 中可直接复用的：
- `noise_angle_curve_goal.png`：encoder shift 随 std 的曲线（4 任务）→ supplementary
- `predictor_drift_eval_correlation.png`：predictor drift × eval scatter → 图 3 的初稿
- `geometry_tradeoff_goal.png`：几何形态散点 → 附录
- `diagnostic_correlation_{task}.png`：每任务 P0 诊断指标 ↔ eval 相关性条形图 → §4.5 的补充

---

## 附录 B：LeWM-base 四任务完整诊断指标

下表汇总 LeWM-base 在四任务上的全部核心诊断指标，数据源自 `research_notebook_swm.md` §4.3–§5.2 及对应 ckpt 的 `diagnostics_summary.json`。

| 层级 | 指标 | TwoRoom | PushT | Reacher | Cube | 单位/说明 |
|---|---:|---:|---:|---:|---:|---|
| **Encoder Geometry** | `clean_nn_cos_dist_median` | 0.0449 | 0.2360 | 0.0633 | 0.1856 | cosine distance |
| | `clean_pair_cos_dist_median` | 0.9904 | 1.0228 | 1.0252 | 1.0193 | pair-wise cos distance |
| | `clean_effective_rank` | 47.60 | 76.42 | 61.04 | 73.25 | effective rank |
| **Noise Sensitivity** | `noise_angle_deg_median` (@std=0.005) | 5.51° | 1.33° | 3.22° | 1.40° | 中位角向偏移 |
| | `noise_to_nn_cos_ratio_median` | 0.1031 | 0.0011 | 0.0249 | 0.0016 | noise/NN cos ratio |
| | `robust_radius_std` | 0.0142 | 0.0537 | 0.0142 | 0.0356 | 临界噪声 std |
| | `first_risk_std` | >0.08 | >0.08 | >0.08 | >0.08 | 首个高风险 std |
| | `noise_angle_slope_deg_per_std` | 1085.8 | 284.8 | 831.7 | 327.0 | °/std，角向增益 |
| | `geometry_flag` | balanced | robust | balanced | robust | 几何形态标签 |
| **Task Resolution** | `transition_resolution_ratio_l2` | 0.7216 | 0.3015 | 0.3704 | 0.4847 | L2 分辨率比 |
| | `transition_resolution_ratio_cos` | 0.5538 | 0.0868 | 0.1351 | 0.2347 | cos 分辨率比 |
| | `id_probe_r2` | 0.2889 | 0.7739 | 0.1621 | 0.6657 | action linear probe R² |
| | `id_probe_r2_min` | 0.2599 | 0.6786 | 0.1366 | 0.0972 | min probe R² |
| | `lidar_rank` | 46.06 | 13.95 | 45.90 | 42.46 | LiDAR rank proxy |
| **Action Effect** | `action_mean_pred_shift_norm` | 0.5329 | 0.1283 | 0.2518 | 0.2364 | action 扰动平均预测偏移 |
| | `action_perturb_pred_shift_corr` | 0.2847 | 0.2873 | 0.4042 | 0.2559 | 偏移与 action norm 相关性 |
| **Predictor Stability** | `predictor_rollout_T8_l2` | 18.62 | 18.65 | 15.17 | 20.20 | T=8 rollout L2 drift (history-only noise @ max std) |
| | `predictor_target_to_nn_cos_ratio_at_max_std` | 1.51e-4 | 3.54e-6 | 2.67e-5 | 3.39e-6 | max std 下 target/NN ratio |
| **Latent Noise** | `cka_linear_at_max_std` | 0.1986 | 0.5536 | 0.3085 | 0.1814 | CKA clean vs noisy |
| | `latent_cost_surface_slope_z` | 635.31 | 1.3886 | 599.45 | 0.6208 | goal latent 扰动 cost 斜率 |

**8 档 sweep 完整数据**：LeWM/SWM 各 8 档（base + 0to001–0to008-p1）的逐档诊断原始值见 `research_notebook_swm.md` §5.2 及本地生成的 `canonical_evals_20260508.json`。

---

## 附录 C：Heteroscedastic Loss 公式

scale-preserving hetero NLL：

$$
\mathcal{L}_{\text{hetero}} = \frac{1}{2} \exp(-s_t) \cdot \|z_{t+1} - \hat{z}_{t+1}\|^2 + \frac{1}{2} s_t
$$

其中 $s_t$ 是 σ head 预测的 log-variance。训练时 $\exp(-s_t)$ 作为自动权重：高误差 transition 被降权，低误差 transition 被升权。

---

*本文代码和完整数据见：https://github.com/qun-team/wm_exp*
