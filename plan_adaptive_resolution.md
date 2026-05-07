# Adaptive Latent Resolution via Heteroscedastic JEPA (Minimal Version)

> **Status**: 设计阶段，未实现。等 plan_v3 §6 P0 真实数据出来后再启动 Pilot。
> **关系**: 不是 plan_v3 的替换，而是 plan_v3 §6 P4 "Adaptive Resolution Method" 的具体化方案。
> **设计原则**: **不增加超参数数量**——和 LeWM 同一量级（仅 1 个 λ_SIGReg）。
> **重要历史记录**: 本文件早期版本曾包含 IB term / aggregate covariance Frobenius / Fisher manifold planning 等多层架构，hyperparameter 数量涨到 4–5 个。经过严格审视后**全部回退**——它们都需要新超参却没有可论证的额外收益。详见 §10 设计回退记录。

---

## 0. TL;DR

把 LeWM/SWM 从单一 latent 输出 `enc(x) → z` 改成**双头** `enc(x) → (μ_x, σ_x)`，predictor 同样输出 `(μ̂, σ̂)`。Loss **唯一改动**：MSE 替换为 heteroscedastic NLL。

$$
\mathcal{L} = \underbrace{\frac{1}{2}\,\mathbb{E}\!\left[\frac{\|\hat\mu - \mu_{\text{target}}\|^2}{\hat\sigma^2} + \log \hat\sigma^2\right]}_{\text{heteroscedastic NLL（替换 MSE）}} + \underbrace{\lambda_{\text{SIGReg}}\cdot \mathrm{SIGReg}(\mu)}_{\text{LeWM 原汁原味}}
$$

**关键性质**：σ_x 通过 NLL 自校准到 per-sample 预测难度，**无需 IB 上界、无需 aggregate 下界**。SIGReg 一字不动作用在 deterministic μ 上（避免 mixture 高阶矩冲突）。

**超参数对比**：
- LeWM: λ_SIGReg
- 本框架: λ_SIGReg（**真新增 = 0**，仅 +1 个 σ head）

LeWM 是严格特例：σ_x ≡ const → MSE + SIGReg = LeWM。

---

## 1. 动机

plan_v3 §5.2 的主线"task-aware latent geometry"在落地时遇到的死结：**所有"自适应"方案都把 trade-off 控制器放在 loss 之外**，模型自己没有"分辨率"这个内禀概念。

PI controller / Lagrangian τ / cheap-proxy bilevel / 多任务 head 等方案都需要外部信号或手调阈值，且都未必比 LeWM + SIGReg 经验上更好。

**真正的范式转换**：让分辨率本身成为模型的输出维度。模型对每个观察 x 输出一个 σ_x——σ_x 就是该状态在 latent 空间被分配的"邻域半径"。Loss 推动 σ_x 在不同样本上不同，**这是模型架构决定的内禀机制**，不是外部脚本调权。

但前提是：**这个改动不能引入新的可调超参数**，否则只是把"调外部 controller"换成"调 IB / σ-floor / σ-cap / calibration"。本文件最终采用的最简版本严格满足这个约束。

---

## 2. 架构设计

### 2.1 双头 encoder

LeWM 现状：
```
enc_backbone(x) → h ∈ R^h_dim
projection_head(h) → z ∈ R^d
```

修改：
```
enc_backbone(x) → h
mean_head(h)    → μ_x ∈ R^d
logvar_head(h)  → log σ_x² ∈ R^d (per-dim) 或 R^1 (scalar)
```

Pilot-1 用 **scalar σ**（更稳、更易诊断）；Pilot-2 才考虑升级 per-dim。

参数量增加：约 +0.1M params for d=192，可忽略。

### 2.2 双头 predictor

```
pred_backbone(μ_t, a_t) → h_pred
mean_head_pred(h_pred)    → μ̂_{t+1} ∈ R^d
logvar_head_pred(h_pred)  → log σ̂_{t+1}² ∈ R^d
```

predictor σ̂ 不参与训练 loss 之外的事——**仅作为 inference 时的可选 uncertainty 信号**（OOD 拒答 / planning budget 调度）。是否真的有用要 Pilot-2 验证。

### 2.3 Target encoder：保留 LeWM 单 encoder 哲学

target latent `μ_{t+1}^target = enc(x_{t+1}).μ`——同一个 encoder，无 EMA、无 stop-grad asymmetry（沿用 LeWM 做法，是否对 target stop-grad 跟 LeWM 保持一致即可）。

Anti-collapse 完全交给 LeWM 现成的 SIGReg(μ)，**不引入额外机制**。

---

## 3. Loss

### 3.1 Heteroscedastic NLL（唯一改动）

把 predictor 看成参数化条件 Gaussian `p(z_{t+1} | μ_t, a_t) = N(μ̂_{t+1}, σ̂_{t+1}²)`。NLL：
$$
\mathcal{L}_{\text{NLL}} = \frac{1}{2}\,\mathbb{E}_t\!\left[\frac{\|\hat\mu_{t+1} - \mu_{t+1}^{\text{target}}\|^2}{\hat\sigma_{t+1}^2} + \log \hat\sigma_{t+1}^2\right] + \text{const}
$$

直观：第一项是误差的 σ-加权（hard sample 自降权），第二项 log σ² 是 σ 的复杂度惩罚（防 σ 任意大）。

### 3.2 σ_x 自校准（无需 IB / aggregate）

NLL 对 σ 求导，最优条件给出闭式：
$$
\sigma_x^{2*} = \mathbb{E}[\|e_x\|^2 \mid x]\quad\text{即该样本的预测误差平方}
$$

→ **σ_x 自动收敛到 per-sample 预测难度**：
- 难预测的样本（接触瞬间、高频动态）→ σ 大 → NLL 权重 1/σ² 小 → 训练梯度弱 → encoder 不必把这些样本 μ-精确分开
- 易预测的样本（自由运动、静止）→ σ 小 → 权重大 → encoder 必须 μ-精确分开

这个"两端互锁"完全由 NLL 闭式给出——**不需要 IB 上界（σ → ∞ 自然被 log σ² 项拉回），不需要 aggregate 下界（σ → 0 时分子项爆炸）**。

### 3.3 SIGReg 仍然作用在 μ 上（一字不动）

由于 SIGReg 输入是 deterministic μ（不是 stochastic z），**没有 Gaussian mixture 高阶矩冲突**。LeWM 的 SIGReg 实现不修改、权重不调整，整套 anti-collapse 机制原样保留。

### 3.4 总目标（最终）

$$
\boxed{\quad
\mathcal{L} = \frac{1}{2}\,\mathbb{E}\!\left[\frac{\|\hat\mu - \mu_{\text{target}}\|^2}{\hat\sigma^2} + \log \hat\sigma^2\right] + \lambda_{\text{SIGReg}}\cdot \mathrm{SIGReg}(\mu)
\quad}
$$

**唯一新增超参数：0**。λ_SIGReg 沿用 LeWM 默认值。

工程实现额外可能需要 `log σ²` 的 clamp 范围（数值稳定性，不算超参数）和 warmup 期间冻结 σ-head（先按 const σ 训几个 epoch 再放开）。

---

## 4. 关键性质

### 4.1 训练时的自适应分辨率

σ_x 收敛到预测难度后：
- **高 σ_x 区域**：μ 几何被允许"粗"（NLL 不强迫 μ 精确分开）
- **低 σ_x 区域**：μ 几何被强迫"细"（NLL 推 μ 精确分开）

→ **训练完之后，μ 空间天然就有不同区域不同分辨率**。

### 4.2 推理时的自适应分辨率

CEM **在 μ 上做标准欧氏 planning**（不改 cost），自动受益于上面的 μ 几何——粗区域的搜索步效率低（移动 μ 但状态没变多少）、细区域的搜索步效率高。这就是"adaptive resolution at inference"——**不依赖把 σ 推进 metric**。

σ̂ 在 inference 仍然有可选用法：
- OOD 拒答（σ̂ 超阈值时 planner 弃权）
- planning budget 调度（高 σ̂ 时多采样）
- 长 horizon 截断（σ̂ 累积超阈值时停止 rollout）

但这些都是**附加 nice-to-have**，不是核心 contribution。是否值得做要 Pilot-2 实证。

### 4.3 LeWM 是严格特例

把 σ_x 限制为单一可学习标量 `σ_x ≡ σ_const`：
- heteroscedastic NLL 退化为 MSE + 常数项
- log σ² 项变成一个全局常数对 σ_const 求最优
- SIGReg(μ) 不变

**结果**：σ-homoscedastic restriction 下 = LeWM。

更进一步：**当训练数据 difficulty 同质时**，本框架最优解收敛到 σ_x ≈ const ≈ LeWM；**异质时**严格优于（异质性是收益的来源）。这给出比"特例"更强的论文主张：不是"我们包含 LeWM"，而是"我们在 LeWM 失效（异质数据）的场景填补了 gap"。

### 4.4 现有方法对照

| 现有方法 | 在本框架下 |
|---|---|
| LeWM + SIGReg | σ_x ≡ const 特例 |
| SWM (V0 spherical) | σ_x ≡ const + μ 投影到单位球 |
| VICReg | σ_x ≡ const + 协方差 decorrelation 风格的 SIGReg 替代 |

现有方法都是 σ-homoscedastic 退化。

---

## 5. 与诊断工具的关系（弱化版）

之前版本主张"17 个诊断指标 = (μ, σ) 框架的 2–3 个本征轴"。**这个主张过于激进**——它假设所有诊断都能被 (μ, σ) 解释，且压缩比可观。这是 empirical question，需要 Pilot-1 数据验证。

本最简版的诚实主张：
- σ_x 输出本身**就是**新增的 per-sample 诊断量
- 现有诊断（`clean_nn_dist`, `effective_rank`, `transition_resolution_ratio` 等）和 σ_x 的相关性是**值得测的事后分析**，但不作为 a priori 的论文主张
- 如果实证发现 σ_x 和某些诊断高相关 → 加分项；如果不相关 → σ_x 提供独立的新信息，也是加分项

→ **诊断工具的价值在 plan_v3 P0 的相关性分析里独立证明**（论文 §4），与本框架的成败解耦。

---

## 6. 论文 Novelty 主张（缩到 1 条）

> **Heteroscedastic JEPA on planning latents**: 在 LeWM 上加 per-sample σ head（encoder + predictor），用 heteroscedastic NLL 替代 MSE，SIGReg 完全保留作用在 μ 上。σ_x 通过 NLL 闭式自校准到 per-sample 预测难度，**不引入新超参数**。LeWM 严格对应 σ-homoscedastic restriction 特例；当数据 difficulty 异质（接触瞬间 vs 自由运动）时，本框架的 μ 几何分辨率自适应分配，CEM planning 自动受益，无需修改 planner。

这一条的优势：
- LeWM 是严格特例（论文叙事）
- 加了 1 个 head + 0 个超参数（实现成本极低）
- σ 自校准有数学闭式（不是经验调出来的）
- 故事简洁：**heteroscedastic 是 LeWM 的最小自然延伸**
- 即使实证不超过 LeWM，也只是一条 ablation 失效，**不伤主表（plan_v3 的诊断工具线）**

不再主张 IB / Fisher manifold / "诊断 = 2–3 个本征轴" 等强主张——这些都需要更多超参或 empirical 假设，得不偿失。

---

## 7. 风险与对策

| 风险 | 评估 | 对策 |
|---|---|---|
| **σ_x 退化成全局常数** | 数据 difficulty 同质时这本身就 = LeWM，**不算失败**。Pilot-1 在 PushT 这种 difficulty 异质强的任务上必须看到 σ_x 跨样本方差显著 > 0 | KS test 或 σ_x std/mean 比；阈值 0.1（待校准） |
| **NLL 训练不稳定**：1/σ² 在 σ̂ 小时梯度爆炸 | heteroscedastic regression 的标准问题 | log σ² 参数化；clamp log σ² ∈ [-10, 10]；warmup 期 σ 冻结为 const |
| **σ_x 早期训练贴在 clamp 边界** | 容易发生 | 起步 1–2 epoch 用 σ ≡ const_init，再放开；初始化 log σ² ≈ 0 (σ ≈ 1) |
| **predictor σ̂ 退化成 encoder σ 简单复制** | 可能但不严重——退化版本仍然给 inference 一个 uncertainty 信号 | 加可选 calibration loss（Pilot-2 候选） |
| **Multi-step σ propagation 公式不准** | 本最简版**不主张**手写 σ 累积公式；让 predictor σ̂ 自学 multi-step uncertainty | 用 multi-step rollout NLL 做训练监督 |
| **不超过 LeWM 的经验风险** | 真实存在——heteroscedastic 在视觉 + planning 上未必有效 | 只伤 1 条 novelty，主表（plan_v3 诊断线）不受影响；fallback 是把这条降为 future work |

---

## 8. Pilot 实验计划

> **触发条件**: plan_v3 §6 P0.6 holdout 跑通 + std_max 加密 sweep 数据出来后启动。

### 8.1 Pilot-1: 最小可行版

**目标**: 验证 σ_x 异质且 μ 几何因此自适应。

| 项 | 设置 |
|---|---|
| 起点 | LeWM 公开 ckpt fine-tune 或 from scratch |
| 改动 | 加 scalar σ head（encoder + predictor），MSE → heteroscedastic NLL |
| SIGReg | 不动 |
| 任务 | TwoRoom + PushT（difficulty 同质 vs 异质对照）|
| Seed | 1 |
| Eval | num_eval=100 单 seed |

**Critical signals**:
1. **σ_x 跨样本方差**: PushT 显著大于 TwoRoom（KS test p < 0.05）；TwoRoom 上 σ_x 接近 const 也是 OK 的（→ LeWM）
2. **σ_x 与预测难度的相关性**: σ_x vs 实际预测误差的 Spearman ρ > 0.5
3. **Eval 不显著差于 LeWM**: 至少不退步；如果 PushT 超过 LeWM 即为成功

**失败判据**: σ_x 在 PushT 上仍然集体常数 → 排查 SIGReg 是否压制了 σ-head 信号 / clamp 是否过紧 / warmup 是否过短。

### 8.2 Pilot-2: Predictor σ̂ + 可选 inference 用法

**触发条件**: Pilot-1 通过。

- 验证 σ̂ 是否 calibration（vs 实际 multi-step error）
- 试验 σ̂-based OOD 拒答 / budget 调度对 success rate 的影响
- 不一定全部正面——有些 inference 用法可能加复杂度但不加性能，老实记录

### 8.3 Validation: 4-task 全套

**触发条件**: Pilot-1 + Pilot-2 通过且经验上至少持平 LeWM。

| 项 | 设置 |
|---|---|
| 任务 | 4 task |
| Seed | 3 |
| Eval | num_eval=300（每 seed 100） |
| 对照 | LeWM-base / 本框架 |
| Ablation | scalar σ vs per-dim σ；SIGReg on μ vs SIGReg on (μ + σε) reparametrization 等价性 |

---

## 9. 与 plan_v3 / plan_v2 的关系

### 9.1 与 plan_v3 §6 P4 的关系

替代 plan_v3 §6 P4 现有 outer-controller 草案的**条件**: Pilot-1 通过后才考虑合并；在此之前 plan_v3 §6 P4 保留为 fallback（PI controller 至少不需要 σ-head 训练成本）。

### 9.2 与 plan_v2 V1/V2 的关系

- V1 (vMF): 球面 + 1D 角度 σ 的特化版，本框架的 spherical projection 限制
- V2 (ball-cap): σ_x quantile clip 的 OOD 延伸

V1/V2 都是更复杂版本，**本最简版本不预设走那个方向**，看 Pilot 结果再决定。

### 9.3 与诊断工具的关系

诊断工具**完全不变**，独立作为 plan_v3 §4 的论文 contribution。本框架的 σ_x 是新增的 per-sample 诊断量，**事后分析**它和现有诊断的相关性是 nice-to-have，不预设为 a priori 主张。

---

## 10. 设计回退记录（Honest Engineering Notes）

本文件早期版本曾包含以下加层，**全部已经被回退**：

| 加层 | 移除原因 |
|---|---|
| EMA target encoder | 违反 LeWM 单 encoder 哲学；SIGReg 已经替代了 EMA 的 anti-collapse 功能 |
| 把 SIGReg 推广到 stochastic (μ, σ) via reparametrization | Gaussian mixture 高阶矩与 heteroscedasticity 冲突，需要"deliberate weakening"——把 SIGReg 砍到只剩二阶矩。这就是放弃了 SIGReg 大半价值 |
| Aggregate covariance Frobenius regularizer | 替代上一项，但额外引入 λ_agg；和 LeWM 比超参数 +1 |
| Information Bottleneck term `−β/2·E[log σ²]` | σ_x 通过 NLL 已经自校准（σ² = e²），IB 上界 redundant；β 是新超参数 |
| Fisher manifold planning（CEM 用 Mahalanobis cost） | (a) 不是真正 Fisher 距离（仅一阶近似）；(b) σ-drift hallucination 风险（CEM 会优化到高 σ 状态）；(c) 修改 planner 违反 SWM 设计承诺；(d) σ_goal 没明确来源 |
| σ propagation closed form `σ_{t+k}² ≈ σ_t² + Σσ̂²` | 假设 predictor 误差独立，autoregressive 下严重不成立 |
| "诊断 = (μ, σ) 框架 2–3 个本征轴" 强主张 | empirical question；提前预设是给论文挖坑 |
| 多 head GradNorm / PCGrad / Lagrangian | 引入新 hyperparameter + 额外训练复杂度，得不偿失 |

**核心教训**：
1. **每加一项都要数 hyperparameter**——如果新增 hyperparameter > 0 而经验收益不明，回退。
2. **数学优雅 ≠ 经验有效**：Fisher / IB 等理论框架在论文里好讲，但 Pilot 没跑过的情况下都是 speculative。
3. **LeWM 已验证有效**：任何替代方案的默认假设是"不超过 LeWM"，需要 empirical evidence 才能逆转。
4. **简单主张更稳**：1 条 novelty + 充分实证 > 4 条互相依赖的理论叠塔。

---

## 11. References

- **JEPA / LeWM**: LeCun 2022; LeWM 2024
- **Heteroscedastic regression**: Kendall & Gal NeurIPS 2017 "What Uncertainties Do We Need in Bayesian Deep Learning"
- **Variational JEPA (rejected as direct borrow)**: Gögl & Yau 2026 (arXiv:2603.20111)——tabular only，本工作扩到 vision + multi-step
- **Anti-collapse 工具线**: SIGReg (LeWM), VICReg (Bardes 2022), RankMe (Garrido 2023)

---

## 12. 维护说明

- 本文件供查阅与设计迭代；**不**作为 plan_v3 的替换。
- 每次新讨论后追加新条目到 §7 风险表 或 §10 回退记录。
- Pilot-1 启动前必读：§3.4（loss form）+ §8.1（critical signals）。
- Pilot-1 通过后，把 §3 §4 §6 合并进 plan_v3 §6 P4；本文件归档。
- **下一次想加新机制前**: 先回看 §10，问自己"它会增加几个超参数？经验收益的证据是什么？"。如果两个问题答不清楚，不加。
