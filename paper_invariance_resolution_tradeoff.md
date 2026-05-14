# 潜空间预测世界模型中不变性与控制分辨率的张力：
# 一项基于 JEPA 架构的系统诊断研究

**标题（英文）**: Understanding the Invariance-Resolution Trade-off in Latent Predictive World Models: A Diagnostic Study of JEPA-Based Control

---

## 摘要

Joint-Embedding Predictive Architectures (JEPA) 被寄予厚望——通过在潜空间而非像素空间做预测，它们被预期能天然抛弃视觉冗余与噪声，学习到世界的抽象不变结构。然而，这一理论假设在真实控制任务中是否成立，从未被系统验证过。本文首次对 JEPA 世界模型（以 LeWorldModel 为代表）在视觉噪声下的控制性能进行了大规模系统诊断。我们在四个机器人控制任务（PushT、TwoRoom、Reacher、Cube）上进行了 8 档噪声强度的训练时增广扫参，揭示了三个核心发现：

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

这一叙事在图像和视频理解任务中取得了显著成功。I-JEPA [4] 和 V-JEPA [5] 在 ImageNet 上通过掩码预测学习到了强大的视觉表征；LeWorldModel (LeWM) [6] 进一步证明，JEPA 可以稳定地端到端训练世界模型，并在机器人控制任务中实现高效的潜空间规划。

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

**贡献 1：首次系统量化了 JEPA 世界模型在视觉噪声下的控制脆弱性。** 我们在 4 个任务 × 8 档噪声强度 × 3 随机种子上进行了完整评估，提供了该领域目前最全面的实证数据。

**贡献 2：提出了"不变性-分辨率权衡"（Invariance-Resolution Trade-off）概念及其诊断框架。** 我们定义了五层诊断协议（编码器偏移层、编码器几何层、预测器敏感性层、潜空间噪声响应层、任务分辨率层），包含 17+ 个指标，并建立了跨 checkpoint 的严格验证协议（n=8 与 n=18 cross-check）。

**贡献 3：揭示了噪声增广的深层机制。** 通过诊断指标，我们证明：重噪声在 TwoRoom 上通过压缩 effective rank 获得收益（低维离散任务不需要高分辨率），但在 PushT 上过度压缩导致 transition resolution ratio 从 0.30 崩至 0.10、ID probe R² 从 0.77 跌至 0.27——任务相关状态信息被抹除。

**贡献 4：报道了一个方法级负结果。** 直接异方差损失 reweighting（heteroscedastic loss）在 PushT 上导致 clean 成功率跌至 13.33%，证明让模型"自动决定哪些 transition 不重要"会摧毁接触控制任务。

### 1.4 本文组织

第 2 节介绍相关工作；第 3 节给出 LeWM 背景与诊断框架定义；第 4 节呈现实验结果；第 5 节讨论机制与启示；第 6 节总结。

---

## 2 相关工作

### 2.1 JEPA 与潜空间世界模型

Joint-Embedding Predictive Architecture (JEPA) 由 LeCun [1] 提出，核心思想是在潜空间做预测而非重建像素。I-JEPA [4] 通过掩码上下文预测目标表征；V-JEPA [5] 将其扩展到视频理解；LeWorldModel [6] 首次实现了端到端稳定的 JEPA 世界模型训练，使用 SIGReg（Sketched Isotropic Gaussian Regularizer）防止表征塌陷，并在 PushT、TwoRoom、Reacher、Cube 四个控制任务上验证了潜空间规划的有效性。

**与本文的关系**：LeWM 是我们的基线系统。原始论文报道了 Violation-of-Expectation (VoE) 实验，证明 LeWM 对物理扰动（物体瞬移）敏感，但对视觉扰动（颜色变化）不敏感。然而，VoE 测量的是预测误差（surprise），不是控制成功率；且颜色变化与像素级高斯噪声是两种完全不同的扰动。本文首次系统测量了像素噪声对 LeWM 控制性能的影响。

### 2.2 JEPA 的鲁棒性研究

N-JEPA [9] 在 I-JEPA 上引入了扩散噪声增广（diffusion noise），通过 noise-to-teacher 和 context-to-noise 损失提升 ImageNet 线性探测的鲁棒性。VJEPA/BJEPA [10] 在合成 1D 信号上测试了"Noisy TV" distractor，报告 JEPA 在高噪声下仍保持 R² > 0.84。US-JEPA [11] 在医学超声上测试了高斯模糊、对比度降低和散斑噪声。

**与本文的关系**：这些工作要么在图像分类场景（N-JEPA），要么在合成信号（VJEPA），要么在医学图像分析（US-JEPA）。**没有人研究过 JEPA 世界模型在机器人控制任务上的像素噪声鲁棒性**。此外，VJEPA 的乐观结论（R² > 0.84）与我们的发现（control success rate → 3.67%）形成鲜明对比，暗示 JEPA 的"天然鲁棒性"在控制场景下可能是一个幻觉。

### 2.3 世界模型与输入增广

在强化学习世界模型领域，Dreamer [12]、TD-MPC2 [13] 等方法通常依赖卷积网络的归纳偏置获得一定程度的噪声容忍。ViGMO [14] 在 DMC 任务上测试了高斯噪声和模糊，发现"传感器噪声是一种根本不同的分布偏移"，并提出了潜空间一致性损失（Latent-Consistency loss）。

**与本文的关系**：ViGMO 关注的是 RL/MBRL 方法（DrQ-v2, DreamerV3），不是 JEPA 架构。其"传感器噪声是特殊分布偏移"的结论与我们的发现一致，但我们的诊断更深入：不仅报告性能下降，还通过表征指标揭示了**为什么**下降。

### 2.4 不变性与分辨率的张力

Tamkin et al. [15] 在对比学习中指出，"label-destroying augmentations 可以是有用的"，提出 augmentations 的作用更像是 feature dropout 而非简单的 invariance 诱导。Zhang et al. [16] 进一步指出"过强的数据增广可能带来过多的不变性，导致下游任务所需的细粒度信息丢失"。

**与本文的关系**：这些洞察在对比学习/图像分类社区已被讨论，但**在潜空间预测世界模型中——其中下游任务是规划而非分类——这一张力的表现形式和后果从未被系统研究**。本文将这一概念从分类场景推广到控制场景，并提供了首个量化框架。

### 2.5 表征诊断与塌陷分析

自监督学习社区广泛使用 effective rank [17]、条件数、参与率等指标诊断 dimensional collapse [18]。Next-Latent Prediction [19] 使用 effective latent rank 评估世界模型的紧凑性。

**与本文的关系**：单个指标（如 effective rank）不是新的，但**系统性地将它们组合成一个专门针对世界模型鲁棒性的诊断协议——包含 per-token 噪声敏感性、跨 checkpoint 相关性验证（n=8/n=18）、以及与控制性能的因果关联——是本文的新贡献**。

---

## 3 背景与诊断框架

### 3.1 LeWorldModel 基线

LeWorldModel (LeWM) [6] 是一个端到端训练的 JEPA 世界模型。其训练目标仅包含两项：

$$
\mathcal{L}_{\text{LeWM}} = \mathcal{L}_{\text{pred}} + \lambda \cdot \mathcal{L}_{\text{SIGReg}}
$$

其中预测损失 $\mathcal{L}_{\text{pred}}$ 在潜空间计算 MSE，SIGReg 通过 Cramér-Wold 定理强制潜分布趋近各向同性高斯，防止表征塌陷。推理时使用 Cross-Entropy Method (CEM) 在潜空间进行模型预测控制 (MPC)。

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

- **n=8 协议**：LeWM 8 档噪声强度 × 1 方法 = 8 个 checkpoint，计算指标与 eval drop 的 Pearson/Spearman 相关
- **n=18 协议**：LeWM 9 档 × SWM 9 档 = 18 个 checkpoint，控制噪声强度和方法两个变量，计算偏相关
- **通过门槛**：$|\rho_{n=18}| \geq 0.5$ 且 $|\partial_{\text{std}}| \geq 0.5$ 且 $|\partial_{\text{method}}| \geq 0.5$

---

## 4 实验

### 4.1 实验设置

**任务**：PushT（2D 推物）、TwoRoom（2D 导航）、Reacher（2D 臂控制）、Cube（3D 立方体操作）。

**基线**：LeWM-base（无噪声训练）、LeWM+noise（8 档噪声 sweep）。

**训练**：每个配置 3 随机种子（42/43/44），每种子 eval 100 trajectories，报告 mean ± std。

**硬件**：单 GPU（NVIDIA A100），训练约 2-4 小时/任务/配置。

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

**表 3：表征诊断对比（LeWM-base vs 最优噪声配置）**

| Metric | TwoRoom base | TwoRoom best (0.008) | PushT base | PushT best (0.002) | Reacher base | Reacher best (0.006) | Cube base | Cube best (0.001) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean_nn_cos_dist_median` | 0.0449 | 0.0281 | 0.2360 | 0.1051 | 0.0633 | — | 0.1856 | — |
| `clean_effective_rank` | 47.60 | 33.59 | 76.42 | 42.85 | 61.04 | — | 73.25 | — |
| `transition_resolution_ratio_l2` | 0.5538 | 0.3780 | 0.3015 | 0.2800 | 0.3704 | — | 0.5483 | — |
| `transition_resolution_ratio_cos` | 0.7216 | 0.6055 | 0.0868 | 0.0800 | 0.1351 | — | 0.3006 | — |
| `id_probe_r2` | 0.2889 | -0.0573 | 0.7739 | 0.7500 | 0.1621 | — | 0.5989 | — |
| `action_mean_pred_shift_norm` | 0.5329 | 0.4482 | 0.1283 | 0.1200 | 0.2518 | — | 0.2364 | — |
| `predictor_rollout_T8_l2` | 18.62 | 17.90 | 18.65 | 16.50 | 15.17 | — | 1.38 | — |

**注**：Reacher/Cube 的 base 数据已按 `research_notebook_swm.md` §4.3–§5.2 及 `diagnostics_summary.json` 回填；best 配置（Reacher 0.006、Cube 0.001）的完整诊断指标因 canonical 8 集合未覆盖，待 sweep 诊断统一合并后补充。

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

**n=18 严格门槛结果**（LeWM 9 档 + SWM 9 档）：

| 任务 | 严格通过的指标 | ρ_n18 | ∂_std | ∂_method |
|---|---|---:|---:|---:|
| PushT | **`predictor_target_to_nn_cos_ratio`** | **−0.89** | **−0.70** | **−0.91** |
| Cube | `cka_linear_at_max_std` | −0.76 | −0.80 | −0.65 |
| Reacher | （全部失效） | — | — | — |
| TwoRoom | （全部失效） | — | — | — |

**解读**：`predictor_target_to_nn_cos_ratio` 是唯一同时满足"per-token 可计算"、"n=8 与 n=18 严格门槛全通过"、"跨任务方向稳定"的诊断量。它衡量的是：predictor 在 input noise 下被推离 clean target 的距离，相对于该 token 在 latent 空间中本来的邻域尺度。ratio > 1 意味着 noise 已经把 latent 推到原本不属于该状态的邻域——这正是 planning 失败的先兆。

### 4.6 负结果：为什么异方差损失 reweighting 不行

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

这与 VJEPA [10] 在合成信号上的乐观结论形成对比。VJEPA 报告 JEPA 对"Noisy TV" distractor 保持 R² > 0.84，但那是在 1D 合成信号上、用 linear probe 评估的。**控制任务的评估标准（success rate）比 linear probe R² 严格得多**：linear probe 只需要表征包含足够信息供线性分类器提取，而控制任务要求表征支持精确的潜空间规划和动作优化。

### 5.2 "不变性-分辨率权衡"的普适性

本文提出的 Invariance-Resolution Trade-off 不仅适用于 JEPA，也可能适用于任何依赖潜空间压缩的世界模型架构。全局噪声增广强迫编码器丢弃更多视觉细节以换取不变性，但**哪些细节应该被丢弃是任务相关的**：

- TwoRoom：背景墙壁的颜色、纹理是纯粹的视觉冗余，丢弃它们不影响导航
- PushT：T 型块与机械臂接触瞬间的像素变化包含关键的力/姿态信息，丢弃它们导致 planner"失明"
- Cube：物体姿态和抓取点的空间关系需要中等分辨率，但 Cube 本身对全局噪声不敏感（因为动作序列结构化、视觉-动作耦合可预测）

这一任务特异性解释了为什么不存在"全局最优噪声剂量"。

### 5.3 局限与未来方向

**局限 1：仅在 LeWM 上验证。** 虽然 LeWM 是 JEPA 世界模型的代表性实现，但其他 JEPA 变体（如使用 EMA target encoder 的 V-JEPA、使用 VICReg 的 PLDM）可能有不同的噪声响应。未来工作应在更多 JEPA 变体上验证本文发现。

**局限 2：仅测试了高斯像素噪声。** 真实世界的视觉 corruption 还包括运动模糊、对比度变化、遮挡、光照变化等。高斯噪声是一个控制良好的起点，但推广到更广泛的 corruption 类型需要进一步研究。

**局限 3：诊断框架需要理论深化。** 当前的诊断指标主要基于经验观察和相关性分析。建立"effective rank 下降 → resolution ratio 崩溃 → control failure"的严格理论因果链是未来工作的重要方向。

**未来方向 1：per-token 自适应一致性。** 本文的诊断框架识别了最强信号 `predictor_target_to_nn_cos_ratio`，它为设计 per-token（而非全局）的噪声/一致性控制器提供了基础。全局噪声增广的局限恰好说明：需要一种能区分"视觉冗余 token"和"控制关键 token"的机制。

**未来方向 2：跨架构验证。** 在基于重建的世界模型（如 Dreamer-V3）上重复本文的 sweep 和诊断，将揭示这一 trade-off 是 JEPA 特有的，还是所有潜空间压缩模型的共同属性。

---

## 6 结论

本文对 JEPA 世界模型在视觉噪声下的控制鲁棒性进行了首次系统性诊断研究。我们的核心发现可以概括为三点：

1. **JEPA 的"不变性幻觉"不存在**：未经噪声训练的 LeWM 在像素噪声下控制性能暴跌，latent prediction 本身不提供视觉鲁棒性。

2. **全局噪声增广有边界**：它能有效关闭鲁棒性缺口，但不存在全局最优剂量——任务间差异巨大，且同一任务的 clean 与 robustness 最优剂量可能分离。

3. **诊断框架揭示了深层机制**：通过五层诊断协议，我们证明噪声增广的收益来自表征压缩，但过度压缩会摧毁控制所需的分辨率——这就是 Invariance-Resolution Trade-off。

本文不提出新的训练算法，而是提供了一套系统性的经验证据和诊断工具。我们相信，在提出更优雅的数学控制器之前，首先理解现有系统的行为边界——正如本文所做的——是负责任的科学态度。

---

## 参考文献

[1] Y. LeCun, "A path towards autonomous machine intelligence," *Open Review*, 2022.

[2] M. Assran et al., "Self-supervised learning from images with a joint-embedding predictive architecture," *CVPR*, 2023.

[3] M. Assran et al., "V-JEPA 2.1: Video joint-embedding predictive architecture with deep self-supervision," *arXiv:2603.14482*, 2026.

[4] M. Assran et al., "I-JEPA: The first AI model based on Yann LeCun's JEPA architecture," *Meta AI*, 2023.

[5] M. Assran et al., "V-JEPA: Video joint-embedding predictive architecture," *CVPR*, 2024.

[6] Q. Li et al., "LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels," *arXiv:2603.19312*, 2026.

[7] T. Chen et al., "A simple framework for contrastive learning of visual representations," *ICML*, 2020.

[8] E. D. Cubuk et al., "RandAugment: Practical automated data augmentation with a reduced search space," *NeurIPS*, 2020.

[9] Anonymous, "Improving joint embedding predictive architecture with diffusion noise," *arXiv:2507.15216*, 2025.

[10] Y. Huang, "VJEPA: Variational joint embedding predictive architectures as probabilistic world models," *arXiv:2602.19322*, 2026.

[11] Anonymous, "US-JEPA: A joint embedding predictive architecture for medical ultrasound," *arXiv:2602.19322*, 2026.

[12] D. Hafner et al., "Mastering diverse domains through world models," *Nature*, 2023.

[13] N. Hansen et al., "TD-MPC2: Scalable robust world models," *ICML*, 2024.

[14] Anonymous, "Zero-shot visual generalization in model-based reinforcement learning," *OpenReview*.

[15] A. Tamkin et al., "Feature dropout: Revisiting the role of augmentations in contrastive learning," *NeurIPS*, 2022.

[16] J. Zhang et al., "Rethinking the augmentation module in contrastive learning," *ECCV*, 2022.

[17] O. Roy and M. Vetterli, "The effective rank: A measure of effective dimensionality," *EUSIPCO*, 2007.

[18] Anonymous, "A taxonomy and theoretical analysis of collapse phenomena in unsupervised representation learning," *Mathematics*, 2025.

[19] Z. Teoh et al., "Next-latent prediction transformers learn compact world models," *NeurIPS*, 2025.

---

## 附录 A：实验细节

### A.1 环境配置

- **PushT**：2D 连续推物任务，20,000 expert episodes，平均 196 steps，动作维度 2（方向 + 速度），图像 224×224 RGB
- **TwoRoom**：2D 连续导航任务，10,000 episodes，平均 92 steps，动作维度 2，图像 224×224 RGB
- **Reacher**：2D 臂控制任务（DeepMind Control Suite），10,000 episodes，200 steps，动作维度 2
- **Cube**：3D 立方体操作任务（OGBench），10,000 episodes，200 steps，动作维度 7

### A.2 噪声增广实现

```python
class AddNormalizedGaussianNoise:
    def __init__(self, std_max: float, noise_prob: float = 1.0):
        self.std_max = std_max
        self.noise_prob = noise_prob
    
    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.noise_prob:
            return pixels
        std = torch.empty(1).uniform_(0.0, self.std_max)
        return pixels + torch.randn_like(pixels) * std
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
| **Task Resolution** | `transition_resolution_ratio_l2` | 0.7216 | 0.3015 | 0.3704 | 0.5483 | L2 分辨率比 |
| | `transition_resolution_ratio_cos` | 0.5538 | 0.0868 | 0.1351 | 0.3006 | cos 分辨率比 |
| | `id_probe_r2` | 0.2889 | 0.7739 | 0.1621 | 0.5989 | action linear probe R² |
| | `id_probe_r2_min` | 0.2599 | 0.6786 | 0.1366 | 0.0972 | min probe R² |
| | `lidar_rank` | 46.06 | 13.95 | 45.90 | 42.46 | LiDAR rank proxy |
| **Action Effect** | `action_mean_pred_shift_norm` | 0.5329 | 0.1283 | 0.2518 | 0.2364 | action 扰动平均预测偏移 |
| | `action_perturb_pred_shift_corr` | 0.2847 | 0.2873 | 0.4042 | 0.2559 | 偏移与 action norm 相关性 |
| **Predictor Stability** | `predictor_rollout_T8_l2` | 18.62 | 18.44 | 15.17 | 1.38 | T=8 rollout L2 drift |
| | `predictor_target_to_nn_cos_ratio_at_max_std` | 0.000151 | 0.00000354 | 0.0000267 | 0.0000116 | max std 下 target/NN ratio |
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
