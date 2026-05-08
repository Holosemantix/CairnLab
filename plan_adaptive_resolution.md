# Adaptive Latent Resolution via Sigma-Conditioned JEPA

> **Status**: 设计阶段，未实现。plan_v3 2026-05-08 审计后，本方案升为下一步主线 Pilot，但本文件已从“直接 NLL 替换 MSE”收紧为分阶段验证。
> **关系**: 不是 plan_v3 的替换，而是 plan_v3 §6 P4 "Adaptive Resolution Method" 的具体化方案。
> **设计原则**: 先证明额外 σ 输出头携带有用信息，再让它影响训练或 planning；避免一开始就改变 LeWM 的强 MSE baseline。
> **重要历史记录**: 本文件早期版本曾包含 IB term / aggregate covariance Frobenius / Fisher manifold planning 等多层架构，hyperparameter 数量涨到 4–5 个。经过严格审视后**全部回退**——它们都需要新超参却没有可论证的额外收益。详见 §10 设计回退记录。

---

## 0. TL;DR

不要默认假设 heteroscedastic NLL 会优于 MSE。LeWM 的 MSE + SIGReg 已经很强，直接替换成 NLL 会改变 pred loss 与 SIGReg 的相对尺度，而且 NLL 会 downweight 高误差样本；在 PushT 这类任务里，高误差样本可能正是接触/精细控制的关键状态。

当前路线（**2026-05-08 更新**：原 Pilot-1A "probe-only σ head" 已折叠为 Pilot-1B 训练时的实时监控量；不再单独训练一轮 detached probe）：

1. **Pilot-1B：Scale-preserving heteroscedastic loss（Stage B，**首选执行项**）**：直接用 scale-preserving 形式替代 MSE，初始化 `logvar_hat ≡ 0` 让 loss 数值与 LeWM MSE 等价；训练时实时监控 σ 的语义稳定性 + μ-梯度 reweight 比值（详 §3.2 / §7 / §8.1 监控量）。如果 σ 全程退化为常数（PushT 特别关注），等价于"Probe-only 失败"——early-stop 即可，无需另起一轮训练。
2. **Pilot-2：Use σ in training/planning（Stage C）**：只有当 σ 在 Pilot-1B 中表现出非平凡异质性、PushT clean eval 不退步、weight reweight 比值未失衡时，才考虑让 σ 影响 planning budget、uncertainty gating、或和 noise consistency / guardrail 结合。

> **为什么跳过原 Pilot-1A？** Probe-only σ head（detached 学 `log(err_token)`、不反传到 μ）只能验证"σ head 容量足够"——这是**廉价 smoke test，不是研究步骤**：(a) logvar head 是个标量小 MLP，容量风险极低；(b) probe 阶段不改变 μ 几何，无法验证 hetero loss 真正的核心风险（PushT 高 error 关键 transition 被 downweight）；(c) 单独训练一轮再切到 Pilot-1B 浪费算力。把"σ↔err 相关性"作为 Pilot-1B 的实时监控指标已足够。

核心批判点：**额外输出头本身不会自动变成动态分辨率。** 如果 σ 只作为日志或 detached probe，它是诊断量，不改变 μ 几何；如果 σ 进入 NLL，它改变训练梯度，但可能只是学会“忽略难样本”；如果 σ 进入 planner，它才成为决策逻辑的一部分，但会引入新的策略风险。因此必须逐级验证。

LeWM 是第一性 baseline。任何 σ 方案都必须先证明至少不破坏 LeWM+noise 的 clean / robustness tradeoff。

---

## 1. 动机

plan_v3 §5.2 的主线"task-aware latent geometry"在落地时遇到的死结：**所有"自适应"方案都把 trade-off 控制器放在 loss 之外**，模型自己没有"分辨率"这个内禀概念。

PI controller / Lagrangian τ / cheap-proxy bilevel / 多任务 head 等方案都需要外部信号或手调阈值，且都未必比 LeWM + SIGReg 经验上更好。

**真正需要验证的范式转换**：让模型输出一个与局部难度/不确定性相关的 σ，并证明这个 σ 能帮助 resolution allocation。这里不能直接把 σ_x 宣称为 latent 邻域半径：
- predictor σ̂ 最自然的监督来自 prediction error，它首先是 **transition uncertainty**。
- encoder σ_x 如果没有额外使用逻辑，只是一个未监督 head，容易不可辨识。
- planning resolution 需要的是“哪些状态差异应该保留”，不等价于“哪些 transition 难预测”。

因此第一步不应直接改主 loss，而应先问：额外输出头是否能稳定学到有意义的异质性？如果不能，后续 NLL / planner 使用都没有基础。如果能，再逐步让 σ 影响训练或 inference。

---

## 2. 架构设计

### 2.1 Pilot-1：只给 predictor 加 σ head

LeWM 现状：
```
enc_backbone(x) → h ∈ R^h_dim
projection_head(h) → z ∈ R^d
predictor(z_t, a_t) → pred_hidden
pred_proj(pred_hidden) → z_hat ∈ R^d
```

Pilot-1 修改：
```
pred_hidden → μ_hat ∈ R^d
pred_hidden → logvar_hat ∈ R^1  # scalar per token
```

encoder 仍输出单一 `μ=z`，不加 encoder σ。原因：
- predictor σ̂ 有天然目标：当前 transition 的 prediction error。
- encoder σ_x 没有直接监督，若同时加 encoder/predictor σ，会有不可辨识问题。
- 最小改动可保持 rollout / CEM cost 全部不变，先隔离 σ head 是否有信息。

### 2.2 Pilot-2：可选 encoder σ head

只有当 Pilot-1 证明 σ̂ 与 prediction difficulty 稳定相关，才考虑 encoder 输出：

```
enc_backbone(x) → h
mean_head(h)    → μ_x ∈ R^d
logvar_head(h)  → log σ_x² ∈ R^1
```

encoder σ_x 的用途必须明确，否则不加：
- 作为 goal / observation uncertainty 的 inference signal。
- 作为 state-wise noise strength / consistency weight 的 controller。
- 与 predictor σ̂ 做 calibration 对齐。

### 2.3 Target encoder：保留 LeWM 单 encoder 哲学

target latent `μ_{t+1}^target = enc(x_{t+1})`——同一个 encoder，无 EMA、无 stop-grad asymmetry（沿用 LeWM 做法，是否对 target stop-grad 跟 LeWM 保持一致即可）。

Anti-collapse 完全交给 LeWM 现成的 SIGReg(μ)，**不引入额外机制**。

---

## 3. Loss 设计：先 probe，再 intervention

### 3.1 Stage A：detached σ calibration probe（首选第一步）

主训练目标完全保持 LeWM：

```text
pred_loss = mean((mu_hat - mu_target)^2)
loss = pred_loss + lambda_SIGReg * SIGReg(mu)
```

新增 σ head 只做 detached calibration：

```text
err_token = mean((mu_hat.detach() - mu_target.detach())^2, dim=-1)
s_hat = pred_logvar_hat.squeeze(-1)
sigma_probe_loss = smooth_l1(s_hat, log(err_token + eps))
```

关键点：
- `sigma_probe_loss` 只更新 σ head，不反向影响 encoder / predictor mean path。
- 这一步**不会**改变 latent resolution；它只是检验额外输出头是否能学到 transition difficulty。
- 如果 σ probe 都学不出稳定结构，NLL 版没有继续做的基础。

### 3.2 Stage B：scale-preserving heteroscedastic loss（候选第二步）

普通 Gaussian NLL：

```text
0.5 * (err * exp(-s) + s)
```

不适合直接替换 MSE，因为初始 `s=0` 时变成 `0.5 * err`，等于把 pred loss 缩小一半，SIGReg 相对变强；后续 `s` 还能让 loss 尺度漂移。

候选替代是尺度保持版本：

```text
err = mean((mu_hat - mu_target)^2, dim=-1)
s = pred_logvar_hat.squeeze(-1)
tau = stopgrad(EMA(mean(err)))  # 或当前 batch mean(err).detach()

hetero_loss = mean(exp(-s) * err + tau * s)
loss = hetero_loss + lambda_SIGReg * SIGReg(mu)
```

性质：
- 初始 `s=0` 时 `hetero_loss = mean(err)`，与原 MSE 同尺度。
- `mu` path 初始梯度接近 LeWM，SIGReg 权重可先不改。
- 最优条件是 `exp(s) ≈ err / tau`，σ 学相对 difficulty，而不是任意改变全局 loss scale。

但这一步仍有核心风险：它会 downweight 高误差 transition。若 PushT 的 hard transition 正好是控制关键区域，eval 可能下降。因此 Stage B 必须以 Stage A 的 σ probe 和 transition/action resolution 诊断为前置条件。

### 3.3 Stage C：σ 进入使用逻辑（不是默认）

如果 σ 只被训练为 calibration probe，它不是 dynamic resolution 方法。要真正影响系统，必须至少进入以下一类逻辑：

| 使用方式 | 作用 | 风险 |
|---|---|---|
| training weight | 通过 hetero loss 改变 μ 的梯度分配 | 可能忽略 hard-but-important states |
| noise/controller | 高 σ 区域降低/提高 noise consistency 强度 | 需要额外规则，容易变成 controller |
| planner budget | 高 σ rollout 增加 CEM samples / 缩短 horizon | 不改变表示，只改 inference compute |
| uncertainty gate | 高 σ 时拒绝或降权候选 plan | 可能过度保守 |

因此 Stage C 只能在 Stage A/B 证明 σ 有语义后再做。否则额外 head 只是日志，不是方法。

### 3.4 SIGReg 仍然作用在 μ 上

无论 Stage A/B/C，SIGReg 都只作用在 deterministic μ 上。不要把 SIGReg 推广到 `(μ, σ)` 或 reparameterized sample；那会引入 Gaussian mixture 高阶矩问题，并破坏 LeWM 已验证的 anti-collapse 机制。

---

## 4. 关键性质

### 4.1 额外 σ head 不自动等于动态分辨率

必须区分三种层级：

| 层级 | σ 做了什么 | 是否改变 resolution |
|---|---|---|
| Probe | 预测 detached error | 否，只是诊断 |
| Loss weighting | 改变不同 transition 的 μ 梯度 | 可能，但可能忽略关键 hard states |
| Planner/controller | 影响 CEM budget / gating / consistency strength | 是系统级 adaptive，但需要额外逻辑 |

所以论文中不能把“加一个 σ head”直接等同于 dynamic resolution。真正要证明的是：σ 与任务相关 difficulty 对齐，并且它进入训练或 inference 后改善了 LeWM+noise 的手调 tradeoff。

### 4.2 NLL 的好处和风险

NLL/hetero loss 的潜在好处：
- 让模型不要为了不可预测或视觉噪声细节浪费 μ 分辨率。
- 为 planning 提供 uncertainty signal。
- 可能减少按任务选择 `std_max` 的需求。

核心风险：
- 高误差不等于低价值。PushT 的接触瞬间可能 high error 但 high value。
- downweight hard samples 可能降低 clean control，而不是提升 robustness。
- loss scale 改变会干扰 SIGReg 权重，必须用 scale-preserving 版本或重新调权。

### 4.3 LeWM 是严格特例

在 Stage B 中，如果 `s ≡ 0` 或 `sigma` 被固定：
- `hetero_loss = mean(err)`
- SIGReg(μ) 不变

**结果**：退化回 LeWM 的 MSE + SIGReg。这个特例关系只有在 scale-preserving 形式下最干净；普通 NLL 会额外改变常数和尺度。

更进一步的“异质时优于 LeWM”目前只是 hypothesis，不应在 Pilot 前写成结论。

### 4.4 现有方法对照

| 现有方法 | 在本框架下 |
|---|---|
| LeWM + SIGReg | 无 σ 使用逻辑；等价于 Stage A 中忽略 σ head |
| SWM (V0 spherical) | 固定单位球几何 prior；无动态 σ |
| VICReg | 固定 covariance / variance prior；无动态 σ |

现有方法都没有把 per-transition uncertainty 明确输出并用于训练或 planning。

---

## 5. 与诊断工具的关系（弱化版）

之前版本主张"17 个诊断指标 = (μ, σ) 框架的 2–3 个本征轴"。**这个主张过于激进**——它假设所有诊断都能被 (μ, σ) 解释，且压缩比可观。这是 empirical question，需要 Pilot-1 数据验证。

本最简版的诚实主张：
- predictor σ̂ 输出本身**就是**新增的 per-transition 诊断量
- 现有诊断（`clean_nn_dist`, `effective_rank`, `transition_resolution_ratio` 等）和 σ̂ 的相关性是**值得测的事后分析**，但不作为 a priori 的论文主张
- 如果实证发现 σ̂ 和某些诊断高相关 → 加分项；如果不相关 → σ̂ 提供独立的新信息，也是加分项

→ **诊断工具的价值主要是设计约束和机制解释**，不再要求先证明它们能独立预测 eval。它们与本框架的成败解耦：即使 P0.6 盲分桶不强，σ-head 仍可能作为更直接的 adaptive resolution 方法成立。

---

## 6. 论文 Novelty 主张（待 Pilot 验证）

> **Sigma-conditioned JEPA for adaptive latent resolution**: 在 LeWM predictor 上加 scalar σ head，先作为 detached prediction-difficulty probe；若 σ 与 transition difficulty / task resolution 对齐，再用 scale-preserving heteroscedastic loss 或 planner/controller logic 让 σ 影响训练或 inference。LeWM 是 σ 不参与使用逻辑时的严格 baseline。

这一条成立的前提：
- Stage A 证明 σ head 学到非平凡、任务相关的 prediction difficulty。
- Stage B/C 证明使用 σ 后能接近或超过 LeWM+noise oracle，而不是只超过 LeWM-base。
- σ 的收益不是来自重新调 SIGReg / loss scale。

不再主张：
- “NLL 一定比 MSE 好”。
- “σ head 自然就是 latent resolution”。
- “不改 planner 就一定能在 inference 自动受益”。
- IB / Fisher manifold / “诊断 = (μ, σ) 本征轴”等强理论叙事。

---

## 7. 风险与对策

| 风险 | 评估 | 对策 |
|---|---|---|
| **σ 退化成全局常数** | Probe 阶段若 PushT 也近似常数，说明额外 head 没有学到有用异质性 | 先不进入 NLL；检查 err target、head capacity、是否需要更长训练 |
| **NLL 改变 MSE/SIGReg 权重比** | 普通 NLL 初始就是 0.5× MSE，且尺度会随 σ 漂移 | 只用 scale-preserving hetero loss；初始 loss/gradient 对齐 MSE |
| **hard-but-important states 被 downweight** | PushT 接触/精细控制可能高误差但高价值 | 必须监控 transition/action resolution；必要时 fallback 到 guarded consistency |
| **σ 只是 uncertainty，不是 resolution** | calibration 成功不等于 planning 提升 | Stage C 必须明确 σ 的使用逻辑；否则只作为诊断输出 |
| **encoder σ 不可辨识** | encoder σ 无天然监督，和 predictor σ 同时学会互相逃逸 | Pilot-1 不加 encoder σ；只在 predictor σ 成立后再加 |
| **Multi-step σ propagation 公式不准** | 本最简版**不主张**手写 σ 累积公式；让 predictor σ̂ 自学 multi-step uncertainty | 用 multi-step rollout NLL 做训练监督 |
| **不超过 LeWM+noise oracle** | 很可能；LeWM+noise 已经很强 | 目标先设为减少手调且接近 oracle；若明显低于 oracle，降级为 analysis/future work |

---

## 8. Pilot 实验计划

> **触发条件**: 已满足。LeWM+noise 已经强于 LeWM-base，但不同任务的最优 noise 强度不同；SWM 没有成为主方法，只保留为 geometry intervention。因此下一步应直接启动 Pilot-1，而不是等待 P0.6 holdout。

### 8.1 Pilot-1B: scale-preserving hetero loss（**首选起点**，已合并 Pilot-1A）

**触发条件**: 已满足。LeWM+noise 已强于 LeWM-base，下一步直接做 hetero loss 验证；σ 语义检验作为本阶段实时监控量（不再单独跑 Pilot-1A）。

设置：
- 配置开关：`loss.hetero.enabled=true`（见 `config/train/lewm.yaml::loss.hetero` block）。
- 用 §3.2 的 scale-preserving hetero loss 替代 MSE（已实现于 `train.py::compute_hetero_pred_loss`）；`logvar_hat` 初始化 weight=bias=0，loss 起点严格 = LeWM MSE。
- 监控量已通过 `self.log_dict` 自动进入 swanlab/wandb（见 `train.py::lejepa_forward` 末尾的 metrics_dict 过滤器，已包含 `hetero_*` 与 `pred_loss_mse_equiv`）；不需另写 callback。
- 不加 encoder σ。
- 不改 planner。
- 任务仍先 TwoRoom + PushT，单 seed × num_eval=100 起步。

**实时监控量**（已自动 log，参见 `train.py::compute_hetero_pred_loss::monitors_dict`）：

| 监控量 | 警戒值 | 含义 / 行动 |
|---|---|---|
| `hetero_s_mean` | \|·\| > 1.0 | tau 与 err 失配，loss scale 漂移；检查 EMA / clamp 是否生效 |
| `hetero_s_std` | > 2.0 | σ 学到极端异质，可能在 clamp 边界失效 |
| `hetero_s_abs_max` | 持续 = `s_max` | s 顶在 clamp 上限——`s_max` 设得太小或 σ head 发散 |
| `hetero_weight_q10_q90_ratio` | < 0.3 | hard transition μ-梯度被 downweight 过度（**PushT 风险信号**），应转 guarded consistency |
| `hetero_s_logerr_corr` | < 0.3 | σ 与预测难度对齐弱（即原 Pilot-1A "失败判据"）；通常意味着 σ 仍是常数 |
| `hetero_tau` | 与 LeWM-base `pred_loss` 数量级偏 > 5× | err 与 EMA 失同步 |
| PushT clean eval | 跌 > 3pt vs LeWM-base | 直接结果，触发 fallback |

成功标准：
- clean eval 不低于 LeWM-base，最好接近 LeWM+noise best。
- PushT `transition_resolution_ratio_cos` 不显著低于 LeWM+noise。
- σ 分布仍与 prediction error calibrated（`hetero_s_logerr_corr` ≥ 0.5），没有贴 clamp。

失败解释：
- 若 PushT 掉分但 σ calibration 好，说明 downweight hard transition 伤害控制；转 guarded consistency。
- 若 σ 全程退化为常数（`hetero_s_std` ≈ 0），等价于"原 Pilot-1A 失败"——early-stop，回 probe-only 改训练（仅 σ head 单独训）或直接转 guarded consistency。
- 若 σ 贴边（`hetero_s_abs_max` 长期 = `s_max`），说明 NLL 数值路径不稳；放宽 clamp 或重新设计 scale handling。

### 8.2 Pilot-2: σ 使用逻辑

**触发条件**: Pilot-1B 不退步，且 σ 语义稳定。

可选项：
- σ-based CEM budget：高 uncertainty rollout 分配更多 samples。
- σ-based horizon gating：高 uncertainty 长 rollout 降权或截断。
- σ-conditioned noise consistency：高/低 σ 区域使用不同 consistency 强度，但必须避免新超参膨胀。

这一步才真正检验“额外输出头是否被系统用起来”。如果只停在 Pilot-1A，它是诊断；如果停在 Pilot-1B，它是 loss weighting；进入 Pilot-2 后才是完整 adaptive system。

### 8.3 Validation: 4-task 全套

**触发条件**: Pilot-1B 或 Pilot-2 通过且经验上至少接近 LeWM+noise oracle。

| 项 | 设置 |
|---|---|
| 任务 | 4 task |
| Seed | 3 |
| Eval | num_eval=300（每 seed 100） |
| 对照 | LeWM-base / LeWM+noise shared std / LeWM+noise per-task oracle / 本框架 |
| Ablation | probe-only vs hetero loss vs σ planning use；scalar σ vs per-dim σ（仅最后考虑） |

---

## 9. 与 plan_v3 / plan_v2 的关系

### 9.1 与 plan_v3 §6 P4 的关系

本文件现在是 plan_v3 §6 P4 的首选思考路线，但执行上必须分阶段。guarded noise consistency / PI controller 保留为 fallback：只有当 Pilot-1A 显示 σ 没有语义、Pilot-1B 显示 NLL 伤害关键 transition，或 eval 明显退步时再回退。

### 9.2 与 plan_v2 V1/V2 的关系

- V1 (vMF): 球面 + 1D 角度 σ 的特化版，本框架的 spherical projection 限制
- V2 (ball-cap): σ_x quantile clip 的 OOD 延伸

V1/V2 都是更复杂版本，**本最简版本不预设走那个方向**，看 Pilot 结果再决定。

### 9.3 与诊断工具的关系

诊断工具**完全不变**，但定位从“独立预测工具”降为“设计约束 + 机制解释”。本框架的 σ̂ 是新增的 per-token / per-transition 诊断量；**事后分析**它和现有诊断的相关性是 nice-to-have，不预设为 a priori 主张。

---

## 10. 设计回退记录（Honest Engineering Notes）

本文件早期版本曾包含以下加层，**全部已经被回退**：

| 加层 | 移除原因 |
|---|---|
| EMA target encoder | 违反 LeWM 单 encoder 哲学；SIGReg 已经替代了 EMA 的 anti-collapse 功能 |
| 把 SIGReg 推广到 stochastic (μ, σ) via reparametrization | Gaussian mixture 高阶矩与 heteroscedasticity 冲突，需要"deliberate weakening"——把 SIGReg 砍到只剩二阶矩。这就是放弃了 SIGReg 大半价值 |
| Aggregate covariance Frobenius regularizer | 替代上一项，但额外引入 λ_agg；和 LeWM 比超参数 +1 |
| Information Bottleneck term `−β/2·E[log σ²]` | 即便 σ 可以通过 NLL calibration，IB 上界仍会引入 β 新超参数；先不加 |
| Fisher manifold planning（CEM 用 Mahalanobis cost） | (a) 不是真正 Fisher 距离（仅一阶近似）；(b) σ-drift hallucination 风险（CEM 会优化到高 σ 状态）；(c) 修改 planner 违反 SWM 设计承诺；(d) σ_goal 没明确来源 |
| σ propagation closed form `σ_{t+k}² ≈ σ_t² + Σσ̂²` | 假设 predictor 误差独立，autoregressive 下严重不成立 |
| "诊断 = (μ, σ) 框架 2–3 个本征轴" 强主张 | empirical question；提前预设是给论文挖坑 |
| 多 head GradNorm / PCGrad / Lagrangian | 引入新 hyperparameter + 额外训练复杂度，得不偿失 |

**核心教训**：
1. **每加一项都要数 hyperparameter**——如果新增 hyperparameter > 0 而经验收益不明，回退。
2. **数学优雅 ≠ 经验有效**：Fisher / IB 等理论框架在论文里好讲，但 Pilot 没跑过的情况下都是 speculative。
3. **LeWM+noise 已验证有效**：任何替代方案的默认假设是"不超过 LeWM+noise oracle"，需要 empirical evidence 才能逆转。
4. **简单主张更稳**：1 条 novelty + 充分实证 > 4 条互相依赖的理论叠塔。

---

## 11. References

- **JEPA / LeWM**: LeCun 2022 ("A Path Towards Autonomous Machine Intelligence"); **Maes et al. 2026, "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"** (arXiv:2603.19312, Mar 2026; Lucas Maes / Quentin Le Lidec / Damien Scieur / Yann LeCun / Randall Balestriero)
- **Heteroscedastic regression**: Kendall & Gal NeurIPS 2017 "What Uncertainties Do We Need in Bayesian Deep Learning"
- **Variational JEPA (rejected as direct borrow)**: Gögl & Yau 2026 (arXiv:2603.20111, Mar 2026) — tabular only，本工作扩到 vision + multi-step
- **Anti-collapse 工具线**: SIGReg (Maes et al. 2026), VICReg (Bardes 2022), RankMe (Garrido 2023)

---

## 12. 维护说明

- 本文件供查阅与设计迭代；**不**作为 plan_v3 的替换。
- 每次新讨论后追加新条目到 §7 风险表 或 §10 回退记录。
- Pilot-1A 是下一步主线；启动前必读：§3.1（probe loss）+ §8.1（critical signals）。
- Pilot-1A/1B 通过后，把 §3 §4 §6 合并进 plan_v3 §6 P4；本文件归档。
- **下一次想加新机制前**: 先回看 §10，问自己"它会增加几个超参数？经验收益的证据是什么？"。如果两个问题答不清楚，不加。
