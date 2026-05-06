# Adaptive Latent Resolution via Heteroscedastic JEPA

> **Status**: 设计阶段，未实现。等 plan_v3 §6 P0 真实数据出来后再启动 Pilot。
> **关系**: 不是 plan_v3 的替换，而是 plan_v3 §6 P4 "Adaptive Resolution Method" 的具体化方案，也是 plan_v2 V1/V2 ladder（vMF / ball-cap）的更普适版本。
> **状态字段**: 当前文件供查阅和后续讨论；落到主 plan 之前需要 Pilot-1 验证 σ_x 是否真的在不同任务上分布不同。

---

## 0. TL;DR

把 LeWM/SWM 从单一 latent 输出 `enc(x) → z` 改成 **heteroscedastic 双头** `enc(x) → (μ_x, σ_x)`，predictor 同样输出 `(μ̂, σ̂)`。Loss 从 information bottleneck 一行推导出来，包含两部分：

- **heteroscedastic NLL**：每个样本按自己的 σ 加权预测损失；
- **IB 压缩项**：`-β·E[log σ²]`，给 σ 一个理论上界。

KKT 平衡点天然给出 **per-sample 的分辨率分配**——结构密集 / 任务关键的状态自动获得小 σ（高分辨率），噪声大 / 信息冗余的状态自动获得大 σ（高鲁棒性）。**任务自适应不是外加约束，而是 rate-distortion 的副产品。**

只暴露一个 hyperparameter `β`（IB 强度），跨任务**无需手调**。

---

## 1. 为什么需要这个方案

### 1.1 plan_v3 暴露的死结

我们在表征分析工具上花了大量功夫（17+ 诊断指标，4 个 task × 多 ckpt 的相关性分析），但 plan_v3 §5.2 主线"task-aware latent geometry"在落地时遇到三层困难：

1. **τ 选择问题**：在 (robust_radius, resolution_ratio) 平面上给每个任务设目标点 → "frame_motion_var → τ_res" 这种映射本身就是手设计的，跟手调 λ 没区别。
2. **指标依赖任务**：不同任务可能对同一诊断指标呈现 **相反** 趋势（TwoRoom vs PushT 的 §4.2 已经是经验证据），universal 阈值不存在。
3. **外部 controller 不优雅**：周期性测 + PI 调权本质是自动化 grid search，理论上 weak，论文叙事 weak。

### 1.2 已尝试的方向以及为什么不够

| 方向 | 问题 |
|---|---|
| outer-loop PI controller | 频率低、信号噪声大、震荡、本质是工程脚本而非方法 |
| 固定 τ + Lagrangian 乘子 λ | τ 仍需任务级人为设定 |
| cheap-proxy bilevel eval | 需要在训练中跑 rollout，工程复杂；只是把代价从 hyperparameter search 移到 outer loop |
| 多任务 head 让数据自决 | 假设 head 集合足够 task-fitness；如果不全则优化错误 |

共同症结：**所有方案都把 trade-off 控制器放在 loss 之外**，模型自己没有"分辨率"这个内禀概念。

### 1.3 真正的范式转换

**让分辨率本身成为模型的输出维度。** 模型对每个观察 x 输出一个 σ_x，σ_x 就是该状态在 latent 空间被分配的"邻域半径"。Loss 推动 σ_x 在不同样本上不同——这是模型架构决定的内禀机制，不是外部脚本调权。

球面 / 欧氏 / SIGReg / uniformity 等都变成本框架的特例（详见 §4.4）。

---

## 2. 架构设计

### 2.1 双头 encoder

LeWM 现状：
```
enc_backbone(x) → h ∈ R^h_dim
projection_head(h) → z ∈ R^d        # d = 192 (LeWM) 或 64 (SWM)
```

修改为：
```
enc_backbone(x) → h
mean_head(h)    → μ_x ∈ R^d
logvar_head(h)  → log σ_x² ∈ R^d (per-dim) 或 R^1 (scalar)
```

**两条 head 并列共享 backbone**。Pilot-1 用 **scalar σ**（更稳、更易诊断）；Pilot-2 升级到 **per-dim σ**（各向异性，更表达）。

参数量增加：`d_hidden × d`（约 +0.1M params for d=192），可忽略。

### 2.2 双头 predictor

```
pred_backbone(μ_t, a_t) → h_pred
mean_head_pred(h_pred)    → μ̂_{t+1} ∈ R^d
logvar_head_pred(h_pred)  → log σ̂_{t+1}² ∈ R^d
```

**Var-JEPA 没做这一步**（它只在 encoder 侧加 σ）。在 predictor 侧加 σ̂ 是本工作的关键扩展，因为：
- multi-step rollout 时 σ̂ 沿 horizon **累积**，给 CEM 自然的不确定性 horizon-decay；
- σ̂ 区分 "encoder 不确定"（aleatoric）和 "predictor 不确定"（epistemic）两种来源。

### 2.3 Target encoder：保留 LeWM 单 encoder 哲学，**不引入 EMA**

LeWM 论文的核心方法论卖点是：**SIGReg 让 anti-collapse 成为对称单 encoder 的训练问题，不需要 BYOL-style 的 EMA + stop-grad asymmetry**。如果我们退回 EMA target，等于把 LeWM 的方法论价值废掉。

本框架沿用 LeWM 做法：

```
μ_{t+1}^target = enc(x_{t+1}).μ      # 同一个 encoder，不是 EMA 副本
```

是否对 target 做 stop-grad 与 LeWM 保持一致（LeWM 在 predictor target 侧做 stop-grad，但没有 EMA decoupling）。Anti-collapse 完全交给 §3.2 的 **extended SIGReg**——SIGReg 的存在本身就是 EMA 的替代，我们要做的只是把它从 deterministic z 推广到 stochastic (μ, σ)。

---

## 3. Loss 推导（从 Information Bottleneck）

### 3.1 Heteroscedastic NLL

把 predictor 看成参数化条件 Gaussian：
$$
p(z_{t+1} | μ_t, a_t) = \mathcal{N}(\hat\mu_{t+1},\ \mathrm{diag}(\hat\sigma_{t+1}^2))
$$

最大化对数似然 → 等价最小化负对数似然：
$$
\mathcal{L}_{\text{pred}} = \frac{1}{2}\,\mathbb{E}_t\!\left[\frac{\|\hat\mu_{t+1} - \mu_{t+1}^{\text{target}}\|^2}{\hat\sigma_{t+1}^2} + \log \hat\sigma_{t+1}^2\right] + \text{const}
$$

`μ^target` 来自同一个 encoder（无 EMA，见 §2.3）。

**直观解读**：
- 第一项：预测越准的样本（小 σ̂）权重越大；
- 第二项：log σ̂² 复杂度惩罚阻止 trivial solution σ̂ → ∞；
- 自然形成 **样本权重的 self-paced learning**：模型对不确定的样本主动调高 σ̂，从而降低对该样本的预测损失贡献。

文献：Kendall & Gal NeurIPS 2017 "What Uncertainties Do We Need in Bayesian Deep Learning"。

### 3.2 Extended SIGReg：anti-collapse 同时给 σ 软下界（替代 EMA 的核心改动）

LeWM 的 SIGReg 是 anti-collapse 的核心机制——把 marginal `p(z)` 通过 moment matching 推向 `N(0, I)`。我们要把它**从 deterministic z 推广到 stochastic (μ, σ)**。

**Reparametrization**：
$$
\tilde z_x = \mu_x + \sigma_x \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

每个 batch 跑一次 reparametrization，`z̃_x` 直接喂给 LeWM 现有的 `SIGReg(z)` 实现——**SIGReg 代码本身不改一行**。

**aggregate marginal 的二阶矩闭式**（Gaussian mixture 标准结果）：
$$
\mathbb{E}_z[z] = \mathbb{E}_x[\mu_x],\qquad
\mathrm{Cov}_z[z] = \mathrm{Cov}_x[\mu_x] + \mathbb{E}_x[\sigma_x^2]\cdot I
$$

SIGReg 推这两个矩到 `N(0, I)` 的 `(0, I)` ⇒ 给出**统一的 anti-collapse 约束**：
$$
\boxed{\;\;
\mathrm{Cov}_x[\mu_x] + \mathbb{E}_x[\sigma_x^2]\cdot I \;\approx\; I
\;\;}
$$

**这一项同时做两件事**：
- 强迫 μ_x 在样本间分散（防 μ collapse）——同 LeWM SIGReg
- 给 σ_x² 一个**软全局下界**（防 σ → 0 退化），同时通过资源 trade-off 形成软上界

**实现**：`L_SIGReg = SIGReg(μ + σ⊙ε)`，权重 `λ_SIG` 沿用 LeWM 默认值。

### 3.2.1 设计权衡：高阶矩松弛

LeWM 原 SIGReg 用 Stein identity 做完整的 Gaussianization，包括 skewness、kurtosis 等高阶矩。一个 Gaussian mixture **几乎不可能**在高阶矩上严格匹配 Gaussian（除非所有 component 的 σ 相等）——也就是说严格版 SIGReg 会和 heteroscedasticity 在高阶矩上**冲突**，把 σ_x 推向均匀。

**为了让自适应分辨率活下来，本框架明确把 SIGReg 放宽到一/二阶矩匹配**（aggregate mean = 0, aggregate cov = I），高阶矩允许偏离。

代价 / 收益：
- 失去：marginal 几何在三阶以上的"整洁度"
- 获得：per-sample σ 自由度，整个自适应机制的存在前提

实现上，让 `SIGReg` 的高阶 Stein 项权重为 0（或显著减小），保留一/二阶矩 + entropy proxy。这是论文里需要老实写出来的 **deliberate weakening**，作为 Pilot-1 必检项（看 σ_x 方差是否显著 > 0）。

### 3.3 Information Bottleneck 项：σ 的 per-sample 上界

aggregate SIGReg 给 σ 全局下界，但还需要一个**逐点上界**阻止"少数样本 σ 爆炸 + 多数样本 σ 极小"的退化分布。Information Bottleneck 给出原理性版本：

$$
\max\ I(z_{t+1}; z_t, a_t) - \beta\, I(z_t; x_t)
$$

第一项 = 预测可达性（已在 L_pred 里隐含）；第二项 = encoder 对 input 的压缩率。

对 Gaussian latent `z_t ~ N(μ_x, σ_x²)`：
$$
I(z_t; x_t) \approx -\frac{1}{2}\,\mathbb{E}_x[\log \sigma_x^2] + \text{const}
$$
（DeepVIB Alemi 2017 标准近似。）

代入 IB 目标：
$$
\mathcal{L}_{\text{IB}} = -\frac{\beta}{2}\,\mathbb{E}_x[\log \sigma_x^2]
$$

### 3.4 总目标：双侧 σ 闭包

$$
\boxed{\quad
\mathcal{L} = \underbrace{\frac{1}{2}\mathbb{E}_t\!\left[\frac{\|\hat\mu - \mu^{\text{target}}\|^2}{\hat\sigma^2} + \log \hat\sigma^2\right]}_{\text{heteroscedastic NLL}} \;+\; \underbrace{\lambda_{\text{SIG}}\cdot \mathrm{SIGReg}(\mu + \sigma\odot\epsilon)}_{\text{aggregate σ 下界 + μ anti-collapse}} \;-\; \underbrace{\frac{\beta}{2}\mathbb{E}_x[\log \sigma_x^2]}_{\text{per-sample σ 上界}}
\quad}
$$

**双侧 σ 闭包**：
- **下界（aggregate）**：SIGReg 锁住总信息预算 `Cov[μ] + E[σ²] ≈ I`
- **上界（per-sample）**：IB 项给每个 σ_x 一个 log-惩罚
- **中间**：NLL 决定每个样本的具体 σ_x 投标——预测越难的样本 σ 越小

**资源拍卖解释**：总 σ 预算固定（SIGReg），per-sample σ 投标由 NLL pressure 决定，IB 项给个体 σ 上限阻止"赢者通吃"。**这是 rate-distortion 的字面实现**。

**Hyperparameter**：
- `λ_SIG`：沿用 LeWM 默认（不算新参数）
- `β`：唯一新增 hyperparameter

**β 的作用范围**：
- β → 0：IB 项 vanish，σ 仅受 SIGReg 全局约束 + NLL 自由竞争 → σ 可能趋向集体压低
- β → ∞：IB 项主导，所有 σ 推向相同大值 → 信息平均分配
- 中间 β：σ_x 在不同样本上自动分布不同 → **task-adaptive resource allocation**

### 3.5 与 ELBO / Constrained Optimization 的关系

总目标恰好对应一个 **constrained ELBO**：
$$
\max\ \mathbb{E}[\log p(z_{t+1}|μ_t, a_t)]\quad \text{s.t.}\quad \mathrm{KL}(q(z)\,\|\,\mathcal{N}(0,I)) \leq \tau
$$

SIGReg 是 KL 约束的 moment-matching surrogate（避开 KL 的 Monte Carlo 估计），IB 项是 Lagrangian 形式的逐点版本。两套数学等价，但 SIGReg 实现稳定（LeWM 已验证），不依赖 KL 的 Monte Carlo 估计。

**论文叙事**：
> "We extend LeWM's SIGReg from deterministic to stochastic latents. The same Gaussian moment-matching that prevents collapse in LeWM now also bounds per-sample σ from below in aggregate. Combined with an information-bottleneck term that bounds σ from above per sample, the latent geometry is bracketed without EMA, stop-grad asymmetry, or task-specific tuning."

---

## 4. 关键性质

### 4.1 (Robust, Resolution) trade-off 的内禀涌现

定义 task-conditional：
- **Local resolution** at x: `1/σ_x²` —— σ 越小，邻域越紧，分辨率越高；
- **Local robust radius** at x: `σ_x` —— σ 越大，对输入扰动的容忍越大。

对每个样本 x，`σ_x` 是模型自己输出的，**它就是该样本的 robust ↔ resolution 平衡点**。
- 任务关键状态（PushT 接触瞬间）：预测损失梯度大 → 推动 σ 变小 → 高分辨率
- 噪声 / 视觉冗余状态（TwoRoom 长走廊）：预测损失梯度小 → IB 项主导 → 大 σ → 高鲁棒

**任务差异从数据梯度自然涌现，不需要任何 task-specific 配置。**

### 4.2 单一 β 跨任务

β 不需要 per-task 调，因为 σ_x **per-sample 自己分配**：
- TwoRoom 整体偏向高 σ 区域分布
- PushT 整体偏向低 σ 区域分布
- 同一 β 下，两者自动到达不同 KKT 平衡

这正是之前所有方案试图通过外部 controller 做到却做不到的事。

### 4.3 多步 rollout 下的不确定性传播

predictor 输出 σ̂_{t+1}。multi-step rollout 时：
$$
\sigma_{t+k}^2 \approx \sigma_t^2 + \sum_{i=1}^{k} \hat\sigma_{t+i}^2
$$
（独立性假设下；耦合情形需要 propagate 协方差矩阵，per-dim σ 时即 element-wise）

**结果**：σ 沿 horizon 自然增大，匹配 "long-horizon prediction is harder" 的物理直觉，**完全无需手设计 horizon decay schedule**。

### 4.4 现有方法作为本框架的特例（**口径已修正**）

之前简单说"σ ≡ const + β = 0 → LeWM"不严谨。正确口径是：**LeWM 是本框架在 σ-homoscedastic restriction 下的特例**。

**Homoscedastic restriction**：把 σ_x 限制为**单一可学习标量** `σ_x ≡ σ_const`（不是 0、不是 per-sample，而是全数据集共享一个标量）。在这个约束下：

- aggregate SIGReg `Cov[μ] + σ_const²·I ≈ I` ⇒ 退化为 `Cov[μ] ≈ (1−σ_const²)·I`，即 LeWM 的 **SIGReg-on-μ**（差一个常数尺度因子）
- heteroscedastic NLL `‖μ̂−μ‖²/σ² + log σ²` ⇒ σ 是常数时退化为 **MSE + 常数项** = LeWM 的 prediction loss
- IB 项 `−β·log σ²` ⇒ σ 是 single scalar 时变成训练中的常数，对梯度无贡献，drop out
- reparametrization `μ + σ_const·ε` ⇒ LeWM 没显式做但等价于一个固定幅度的 input noise（LeWM 的 `image_noise` 机制本来就有；理论上吸收到现有 noise schedule 即可）

**完整对照表**：

| 现有方法 | 在本框架下的特例条件 | 备注 |
|---|---|---|
| LeWM + SIGReg | σ-homoscedastic restriction（σ_x ≡ σ_const） | β 项 drop out；NLL 退化为 MSE |
| SWM (V0 spherical) | σ-homoscedastic restriction + μ_x 投影到单位球 | 球面 vMF 的 σ 退化 |
| VICReg | σ-homoscedastic + 协方差 decorrelation 等价于 SIGReg 二阶矩匹配的另一实现 | 可视为本框架的另一 surrogate |
| vMF (plan_v2 V1) | μ 在球面 + 1D 角度 σ_x（per-sample 但限制为 1D） | 几何特化版的 heteroscedastic |
| Ball-cap (plan_v2 V2) | σ_x 学 hard cutoff（quantile clip on σ_x） | OOD detection 延伸 |

### 4.5 强于"特例包含"的 smoothness 主张

更进一步，可以陈述 **smoothness / regime collapse**：

> 当训练数据的 per-sample 预测难度接近常数时（数据 difficulty 同质），本框架最优解的 σ_x 收敛到 homoscedastic，等价于 LeWM。当数据 difficulty 异质（如世界模型中的接触瞬间 vs 自由运动），本框架的 σ 异质化解严格优于 LeWM。

这是比"特例"更强的论文叙事：**不是简单"我们包含 LeWM"，而是"我们在 LeWM 失效的场景下精确填补了 gap"**。Pilot-1 的关键验证：在 PushT 这种 difficulty 异质性强的任务上，σ_x 方差应显著 > 0；在 TwoRoom 这种相对同质的任务上，σ_x 方差可能较小但依然存在。

---

## 5. 与表征诊断工具的同构

这是本框架最优雅的副产品：**所有 17 个诊断指标都是 (μ, σ) 的函数**，可以重新解释为 IB 平面上的不同切片。

| 现有诊断 | (μ, σ) 框架下的含义 |
|---|---|
| `clean_effective_rank` | μ + σ 联合协方差谱 |
| `clean_nn_cos_dist` | μ NN 距离，应与 σ_x 同向（密集 μ → 小 σ） |
| `clean_pair_cos_dist` | μ 全局扩散，受 IB 项 β 控制 |
| `transition_resolution_ratio` | Mahalanobis 时间距离 `‖μ_{t+1}−μ_t‖² / σ²` —— 自然归一化 |
| `noise_to_nn_cos_ratio` | encoder shift 相对 *local σ_x* 的比 —— per-sample 自适应 |
| `robust_radius_std` | σ_x 分布的某个百分位数（例如 p50 或 p90） |
| `id_probe_r2` | μ 子空间在 action 上的 mutual information |
| `predictor_target_shift` | predictor 输出 μ̂ 在 noisy history 下的偏移 |
| `predictor_rollout_drift(T)` | σ̂ 沿 horizon 的累积量 |
| `cost_surface_slope_z` | latent 噪声下 NLL 对 σ 的灵敏度 |

**含义**：原本 17 个 ad-hoc composite 指标 → **本框架下的 2–3 个 principal axes**：
1. **μ-axis**：μ 的分布、几何、协方差谱（取代 effective_rank、clean_pair_dist、id_probe）
2. **σ-axis**：σ 的分布、per-sample 异质性（取代 robust_radius、resolution_ratio、noise_to_nn）
3. **(μ, σ) coupling axis**：两者的耦合关系（取代 transition_resolution、target_shift）

**这正是表征分析工具的真正价值**——不是堆 17 个指标，而是用框架把它们压成 framework-driven principal components。可以在 plan_v3 §7.2 加一条新的 novelty："Diagnostic toolkit as IB landscape probes"。

---

## 6. Uncertainty-Aware Planning（CEM 的免费副产品）

predictor 输出 σ̂ 后，CEM cost 可以自然加权：

```
cost(trajectory) = Σ_t  ‖μ̂_t − goal‖² / σ̂_t² + log σ̂_t²
```

或更简单：用 σ̂_t 给每个时间步的 cost 一个置信度衰减权重：
```
cost(trajectory) = Σ_t  w_t · ‖μ̂_t − goal‖²,    w_t = 1 / (1 + σ̂_t)
```

**论文 contribution**：planning 在长 horizon 自动降权高不确定 trajectory，无需手设计。

---

## 7. 论文 Novelty 主张（4 条互不重复）

> 替代 plan_v3 §7.2 的现有 3 条，方法论分量明显更强。

1. **Heteroscedastic JEPA for planning latents.**  
   per-sample (μ, σ) head 在 encoder 和 predictor 双侧。Var-JEPA (Gögl & Yau) 是 tabular + 单步 + 仅 encoder σ；本工作首次扩到 vision + multi-step rollout + planning + predictor σ。

2. **Two-sided σ bracketing via extended SIGReg + IB.**  
   - **下界（aggregate）**：把 LeWM SIGReg 从 deterministic z 推广到 stochastic `q(z|x) = N(μ_x, σ_x²I)`，aggregate marginal 的二阶矩约束 `Cov[μ] + E[σ²]·I ≈ I` **同时**给 μ anti-collapse 和 σ 全局软下界。
   - **上界（per-sample）**：IB 项 `−β/2 · E[log σ²]` 给每个 σ_x 一个逐点 log-惩罚，阻止"赢者通吃"型分布退化。
   - **对偶关系**：aggregate 下界 + per-sample 上界 = rate-distortion 字面闭包；σ_x 在两端约束之间被 NLL pressure 推向 task-adaptive 的资源分配。
   - **保留 LeWM 单 encoder 哲学**：无 EMA、无 stop-grad asymmetry。LeWM/SWM 严格对应"σ-homoscedastic restriction"特例；当数据 difficulty 同质时本框架收敛到 LeWM，异质时严格优于（§4.5 smoothness 主张）。
   - 一个 β 跨任务**无需手调**，σ_x 在不同任务上 per-sample 自适应分配。

3. **Uncertainty-aware CEM planning.**  
   利用 predictor σ̂ 加权 cost，长 horizon 自动 horizon-decay。比 fixed-discount 或 receding-horizon truncation 更原理化。

4. **Diagnostic toolkit as rate-distortion landscape probes.**  
   重新解释表征分析的 17 个指标为 (μ, σ) 框架下的不同切片；用 PCA-on-diagnostics 找 framework 的 2–3 个本征轴；validate 实证上确实压缩成低维。

---

## 8. 潜在风险与对策

| 风险 | 对策 |
|---|---|
| **SIGReg 高阶矩残留压力把 σ 拉向 homoscedastic** | **本框架的 critical 风险**。LeWM 原 SIGReg 用 Stein identity 做完整 Gaussianization；高阶矩对 Gaussian mixture 几乎不可满足，会推 σ_x → const，自适应失效。对策：把 SIGReg 高阶 Stein 项权重设 0 或显著减小，仅保留一/二阶矩 + entropy proxy（§3.2.1）。Pilot-1 必检：σ_x 跨样本方差是否显著 > 0（KS test 或 std/mean 比） |
| **Target collapse**：σ → ∞ + μ → 0 退化解 | aggregate SIGReg 锁住 `Cov[μ] + E[σ²] ≈ I`，μ 不能塌缩到 0 + σ 不能集体爆炸；IB 项给逐点 σ 上界。三层约束应足够，不需 EMA |
| **σ 无变化**：σ_x 退化成全局常数 | 与第一行连带——若高阶 SIGReg 太强会触发；其它原因：β 太小、数据 difficulty 同质（这种情况下 σ 退化为常数本身 = 退化为 LeWM，不算失败）。Pilot-1 在 difficulty 异质的 PushT 上必须看到 σ_x 显著异质 |
| **NLL 训练不稳定**：log σ 容易 NaN | 用 log σ² 参数化；clamp log σ² ∈ [-10, 10]；warmup（前 1k step 关闭 IB 项 + reparametrization 噪声幅度从 0 渐进） |
| **Reparametrization 噪声破坏预测信号** | 反向传播经过 `μ + σε` 时 ε 应 detach 但 σ 应保留梯度（标准 reparametrization trick）；早期训练 σ 过大会让 SIGReg 主导而 NLL 无法收敛 → warmup 必需 |
| **Predictor σ̂ 退化成 encoder σ 简单复制** | 加 σ̂ 与 input σ 解耦的 ablation；强制 σ̂ 至少包含 "predictor 自身不确定" 信号；calibration loss 监督 |
| **Multi-step σ propagation 欠 calibrated** | 训练时显式监督 multi-step σ̂ 是否匹配实际预测误差（calibration loss）：`(σ̂_T − ‖drift_T‖)²` |
| **不带 EMA 的对称训练训练动力学不稳** | LeWM 已经验证 SIGReg 单 encoder 能稳；本框架 SIGReg 仍在（只是改成 stochastic 输入）。如果新增的 reparametrization + IB 项导致不稳，回退顺序：先关 IB（β=0），再缩小 reparametrization 幅度。EMA 是**最后的兜底**，论文叙事上承认这种回退会损失方法论简洁性 |

---

## 9. Pilot 实验计划

> **触发条件**: plan_v3 P0.6 holdout 跑通 + std_max 加密 sweep 数据出来后启动。
> 在此之前不动手，避免 framework 设计与真实数据脱节。

### 9.1 Pilot-1: 最小可行版

**目标**: 验证 σ_x 在不同任务呈现明显不同分布。

| 项 | 设置 |
|---|---|
| 起点 | LeWM 公开 ckpt（不冷启动训练） |
| 改动 | 加 scalar σ encoder head + IB 项 |
| 任务 | TwoRoom + PushT |
| β | 1e-3, 1e-2, 1e-1 三档扫 |
| Seed | 1 |
| Eval | num_eval=100 单 seed |

**Critical signal**:
- σ_x 直方图在 TwoRoom 与 PushT 上**统计上不同**（KS test p < 0.05）。
- σ_x 与现有诊断 `clean_nn_cos_dist` 的 Spearman 相关 > 0.5。

**失败判据**: σ_x 退化成几乎常数 → 优先排查 SIGReg 高阶矩压力（§3.2.1）；其次 β 太弱；最后才考虑数据 difficulty 是否真的同质（同质则退化为 LeWM 不算失败）。先排查再决定是否调整框架。

### 9.2 Pilot-2: Predictor σ̂ 验证

**目标**: 验证 σ̂ 沿 horizon 单调上升 + 与实际 multi-step error 校准。

| 项 | 设置 |
|---|---|
| 起点 | Pilot-1 通过的 ckpt |
| 改动 | 加 predictor σ̂ head |
| 任务 | TwoRoom + PushT |
| 监测 | σ̂_{t+1...t+8} 趋势；σ̂ vs 实际 multi-step drift 的 calibration ratio |

**Critical signal**: σ̂ 沿 horizon 增大（monotonic trend p < 0.05）；calibration ratio ∈ [0.5, 2]（一阶近似）。

### 9.3 Validation: 4-task 全套

**触发条件**: Pilot-1 + Pilot-2 通过。

| 项 | 设置 |
|---|---|
| 任务 | TwoRoom + PushT + Reacher + Cube |
| Seed | 3 |
| Eval | num_eval=300（每 seed 100） |
| 对照 | LeWM-base / SWM-best / Heteroscedastic-JEPA(本工作) |
| Ablation | β = 0 / β = best / β = 10·best ；scalar σ vs per-dim σ |

### 9.4 Diagnostic 重写

把 17 个指标 dump 成 (μ, σ) 函数，做 PCA 找 principal axes。验证是否能压成 2–3 维。这部分**与 §9.3 并行**，不阻塞 paper writing。

---

## 10. 与 plan_v3 / plan_v2 的关系

### 10.1 与 plan_v3 §6 P4 的关系

替代 plan_v3 §6 P4 当前的 outer-controller 草案。**条件**: Pilot-1 通过后才正式合并；在此之前 plan_v3 §6 P4 保留现状作为 fallback。

### 10.2 与 plan_v2 V1/V2 ladder 的关系

- **V1 (vMF)**: 本框架的球面 + per-sample κ 特例；如果欧氏 heteroscedastic 不稳，可作为几何特化版回退。
- **V2 (ball-cap)**: 本框架的 σ_x 加 hard quantile clip；可作为 OOD detection 的延伸。

### 10.3 与现有诊断工具的关系

诊断工具**不变**——它们继续作为 framework 的 instrumentation。但解释口径升级：从 "ad-hoc composite metrics" → "rate-distortion landscape probes"。这给 plan_v3 §7.2 加一条 novelty。

---

## 11. 论文 Story Outline (草稿)

```
§1 Introduction
   - World model planning 的 latent geometry 需要在 robustness 和 resolution 之间 trade-off
   - 现有方法（LeWM/SWM/SIGReg/VICReg）选了一个固定点，跨任务表现不一致
   - 我们提出 heteroscedastic JEPA + IB framework，让 trade-off 由 rate-distortion 自动平衡

§2 Background
   - JEPA / LeWM
   - Heteroscedastic regression (Kendall & Gal)
   - Information Bottleneck (Tishby; Alemi DeepVIB)

§3 Method: Heteroscedastic JEPA with IB
   - 双头架构（encoder + predictor）
   - Loss derivation（§3.1, §3.2, §3.3）
   - 单一 β，跨任务

§4 Diagnostic Toolkit as IB Landscape Probes
   - 现有 17 指标的 framework reinterpretation
   - PCA 找 principal axes

§5 Uncertainty-Aware Planning
   - σ̂ 在 CEM cost 中的角色
   - Multi-step horizon decay

§6 Experiments
   - 4 task × {LeWM, SWM, Hetero-JEPA} × 3 seed
   - σ_x distribution per task
   - Ablation: β scan, scalar vs per-dim σ
   - Diagnostic PCA validation

§7 Related Work
   - Var-JEPA: tabular, single-step, encoder-only σ → 我们 vision + multi-step + predictor σ
   - vMF, BNN heteroscedastic, DeepVIB, RankMe
   - LeWM/SWM/SIGReg 作为 ablation case

§8 Discussion
   - τ-free adaptive 的实现路径
   - 何时 framework 退化（β=0 → LeWM）
   - 跨任务无需手调的实证

§9 Future Work
   - V1: per-dim σ 的协方差 propagation
   - V2: σ-aware 的 OOD detection
   - V3: σ 作为 active learning 的 acquisition function
```

---

## 12. 开放问题

1. **β 调整策略**: 单值 β 跨 4 task 是否真的够用？如果某任务需要不同 β，是否退化为本方案试图避免的"task-specific tuning"？
   - 应对：β 在量级 [1e-4, 1e-1] 之间扫，看是否有 single β 在 4 任务上都接近最优；如果不行，退到 per-task β 但仍主张比 per-task τ 优雅（β 是物理量级参数）。

2. **SIGReg 高阶矩松弛幅度**: 把 Stein 高阶项权重设为 0 是粗暴做法；是否有更优雅的方式只保留必要的 Gaussianization 而不挤压 σ 异质性？候选：用 mixture-aware 的 moment matching（针对 Gaussian mixture 的二阶 + skewness/kurtosis 闭式而不是用整体 Stein 算子）。
   - 应对：Pilot-1 先用 "高阶项权重=0" 的暴力版；如果 σ 异质性 OK 但 marginal 几何明显歪，再升级到 mixture-aware。

3. **IB 项的高阶近似**: `I(z; x) ≈ -E[log σ²] / 2` 在 marginal 不接近标准高斯时不准。是否需要更精确的 mutual information 估计？
   - 应对：在 fixed prior 下偏差是常数，对优化无影响；如果 marginal 偏离严重，加 KL(q(z) ‖ N(0,I)) 项作为高阶修正。

4. **Multi-step σ 的 calibration**: σ̂ 沿 horizon 增大是否与真实预测误差 quantitatively 匹配？欠 calibrated 会让 uncertainty-aware planning 误导。
   - 应对：加 calibration loss `‖σ̂_T − actual_drift_T‖²` 在 Pilot-2 中显式监督。

5. **诊断 PCA 是否真的压到 2–3 维**: 如果还是 5+ 维，framework reinterpretation 论点弱化。
   - 应对：是 empirical question，需要 §9.4 实证。

---

## 13. References

- **JEPA / LeWM**: LeCun 2022 "A Path Towards Autonomous Machine Intelligence"; LeWM 2024
- **Variational JEPA**: Gögl & Yau 2026, "Var-JEPA: A Variational Formulation of JEPA" (arXiv:2603.20111)
- **Heteroscedastic regression**: Kendall & Gal, NeurIPS 2017
- **Information Bottleneck**: Tishby & Zaslavsky 2015; Alemi et al. ICLR 2017 (Deep VIB)
- **EMA target / BYOL / DINO**: Grill et al. NeurIPS 2020; Caron et al. ICCV 2021
- **Anti-collapse 工具线**: VICReg (Bardes 2022), RankMe (Garrido 2023), SIGReg (LeWM)
- **Aleatoric vs Epistemic**: Gal 2016 thesis; Kendall & Gal 2017
- **Spherical / vMF**: Davidson et al. 2018 (Hyperspherical VAE); Mardia & Jupp 1999

---

## 14. 维护说明

- 本文件供查阅与设计迭代；**不**作为 plan_v3 的替换，**不**作为 implementation tracker。
- 每次新讨论后，在 §8 风险表 / §9 实验计划 / §12 开放问题 中追加新条目。
- Pilot-1 启动前，重读本文件 §3.3（loss form）+ §9.1（验证 signal）—— 这两处是判定 framework 是否成立的核心。
- Pilot-1 通过后，把 §3 §4 §5 §7 的内容合并进 plan_v3 §6 P4，本文件归入 archived/ 目录。
