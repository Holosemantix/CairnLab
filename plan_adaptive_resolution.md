# Adaptive Latent Resolution via Sigma-Conditioned JEPA

> **Status**: Pilot-1B 已完成首轮 TwoRoom + PushT 验证（2026-05-09）。结果支持“σ head 能学到 prediction difficulty”，但否定了“直接用 hetero loss 替换 MSE”作为 PushT 上的主方法：PushT clean eval 从 LeWM-base 87.33/86.00 降到 13.33，诊断显示 transition/action resolution 被严重压缩。下一步主线改为 **probe-only σ + resolution guardrail + σ inference/controller use**，而不是继续加大 hetero loss。
> **关系**: 不是 plan_v3 的替换，而是 plan_v3 §6 P4 "Adaptive Resolution Method" 的具体化方案。
> **设计原则**: 先证明额外 σ 输出头携带有用信息，再让它影响训练或 planning；避免一开始就改变 LeWM 的强 MSE baseline。
> **重要历史记录**: 本文件早期版本曾包含 IB term / aggregate covariance Frobenius / Fisher manifold planning 等多层架构，hyperparameter 数量涨到 4–5 个。经过严格审视后**全部回退**——它们都需要新超参却没有可论证的额外收益。详见 §10 设计回退记录。

---

## 0. TL;DR

不要默认假设 heteroscedastic NLL 会优于 MSE。LeWM 的 MSE + SIGReg 已经很强，直接替换成 NLL 会改变 pred loss 与 SIGReg 的相对尺度，而且 NLL 会 downweight 高误差样本；在 PushT 这类任务里，高误差样本可能正是接触/精细控制的关键状态。

当前路线（**2026-05-09 更新**：Pilot-1B 已跑完，直接 hetero loss 在 PushT 上失败，因此恢复 probe-only / guarded 路线）：

1. **Pilot-1B 结论：Scale-preserving heteroscedastic loss 语义成功、控制失败。** `hetero_s_logerr_corr` 在 TwoRoom/PushT 后期分别约 0.89/0.95，说明 σ head 学到了 prediction difficulty；但 PushT `hetero_weight_q10_q90_ratio` 掉到约 0.008，hard transition 被强 downweight，clean eval 崩到 13.33。
2. **下一步首选：Probe-only σ / MSE-preserving guarded σ。** μ path 保持 LeWM MSE + SIGReg，σ head detached 学 `log(error)`；若再让 σ 影响训练，必须只作为小权重辅助项并加入 transition/action resolution guardrail。
3. **Pilot-2：Use σ in inference/controller。** 优先让 σ 进入 planning budget、uncertainty gating、或 noise consistency controller，而不是再直接替换 pred loss。真正的 adaptive resolution 应该利用 σ 分配计算或一致性强度，同时保住 PushT 的连续控制分辨率。

> **为什么恢复 probe-only？** 2026-05-09 Pilot-1B 已经证明核心风险真实存在：σ calibration 很好，但 PushT 失败。此时 probe-only 不再是“容量 smoke test”，而是把 σ 语义从 μ 几何更新中解耦，避免 hard-but-important transition 被训练权重抹掉。

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

> **历史触发条件**: 已满足。LeWM+noise 已经强于 LeWM-base，但不同任务的最优 noise 强度不同；SWM 没有成为主方法，只保留为 geometry intervention。因此 2026-05-09 已执行 Pilot-1B；结果见 §8.2，下一步不再继续扩大直接 hetero loss。

### 8.1 Pilot-1B: scale-preserving hetero loss（已完成的历史设置）

**触发条件**: 已满足并已执行。LeWM+noise 已强于 LeWM-base，因此本轮直接做 hetero loss 验证；σ 语义检验作为本阶段实时监控量（未单独跑 Pilot-1A）。

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

### 8.2 Pilot-1B 结果（2026-05-09）

运行：

| Task | Run | SwanLab id | Local output |
|---|---|---|---|
| TwoRoom | `tworoom_lewm_hetero_default` | `gps6asjv22tmflag9af5m` | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_lewm_hetero_default` |
| PushT | `pusht_lewm_hetero_default` | `tge50bhmtws06xc7n4wtq` | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt/pusht_lewm_hetero_default` |

设置：
- LeWM baseline architecture，`loss.hetero.enabled=true`。
- `image_noise.std_max=0.0`，不使用 noise training。
- `s_min=-4.0, s_max=4.0`，`logvar_hat` final layer zero-init。
- eval 为 epoch 10，`num_eval=300`，seeds 42/43/44 聚合。

#### 8.2.1 训练曲线

| Metric | TwoRoom hetero | PushT hetero | 解释 |
|---|---:|---:|---|
| `fit/hetero_s_logerr_corr` tail100 | 0.894 | 0.950 | σ 与 prediction error 强正相关，σ head 语义成立 |
| `validate/hetero_s_logerr_corr_epoch` last | 0.912 | 0.957 | validation 上同样成立，不是 train-only artifact |
| `fit/hetero_s_std` tail100 | 1.232 | 1.836 | PushT 的 σ 异质性明显更强 |
| `fit/hetero_s_abs_max` last | 3.236 | 4.000 | PushT 已贴到 clamp 上限 |
| `fit/hetero_weight_q10` last | 0.495 | 0.369 | 高 σ / hard token 被 downweight |
| `fit/hetero_weight_q90` last | 11.026 | 47.802 | low-error token 被大幅 upweight |
| `fit/hetero_weight_q10_q90_ratio` last | 0.045 | 0.008 | PushT 梯度权重极端失衡 |
| `fit/pred_loss_mse_equiv` tail100 | 0.0438 | 0.0394 | true MSE-equivalent loss 仍下降，但不保证任务 resolution 保留 |
| `validate/pred_loss_mse_equiv_epoch` last | 0.0274 | 0.0332 | validation MSE 也下降；失败不是简单 underfit |

关键判定：
- **σ calibration 成功。** 两个任务 `hetero_s_logerr_corr` 后期都很高，说明 σ head 不是常数，也不是噪声。
- **PushT reweight 过强。** `q10/q90_ratio` 低到 0.008，远低于 §8.1 的 0.3 警戒线；这正是 hard-but-important transition downweight 风险。
- **hetero loss 可以为负。** `pred_loss` 后期略为负是公式 `exp(-s) * err + tau * s` 的结果，不代表 prediction quality “负误差”；真实对照应看 `pred_loss_mse_equiv`。

#### 8.2.2 Eval 结果

| Task / model | Clean | goal 0.05 | pixels 0.05 | pixels+goal 0.05 | goal 0.08 | pixels+goal 0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom LeWM-base | 93.00 | 71.00 | 70.33 | 62.33 | 55.67 | 44.33 |
| TwoRoom LeWM+noise best (`0to008-p1`) | 98.33 | 98.00 | 98.33 | 98.00 | 98.67 | 98.67 |
| TwoRoom hetero | **99.67** | 85.33 | 96.67 | 84.67 | 73.33 | 55.33 |
| PushT LeWM-base | 87.33 / 86.00 | 38.00 | 17.33 | 15.00 | 15.00 | 3.67 |
| PushT LeWM+noise best (`0to002-p1`) | **90.00** | 85.00 | 87.67 | 86.00 | 83.00 | 70.67 |
| PushT hetero | **13.33** | 7.67 | 7.67 | 7.67 | 9.67 | 6.00 |

结论：
- TwoRoom clean 提升到 99.67，符合低维离散任务受益于 stronger invariance / clustering 的预期。
- TwoRoom hetero 不能替代 noise training：goal/pixels+goal 高噪声仍明显低于 LeWM+noise best。
- PushT clean 只有 13.33，是方法级失败，不是 robustness tradeoff。

#### 8.2.3 Diagnostics

| Metric | TwoRoom LeWM-base | TwoRoom hetero | PushT LeWM-base | PushT hetero |
|---|---:|---:|---:|---:|
| `clean_nn_cos_dist_median` | 0.0449 | 0.0281 | 0.2360 | 0.1051 |
| `clean_effective_rank` | 47.60 | 33.59 | 76.42 | 42.85 |
| `transition_resolution_ratio_cos` | 0.5538 | 0.3780 | 0.0868 | 0.0101 |
| `transition_resolution_ratio_l2` | 0.7216 | 0.6055 | 0.3015 | 0.1023 |
| `id_probe_r2` | 0.2889 | -0.0573 | 0.7739 | 0.2678 |
| `action_mean_pred_shift_norm` | 0.5329 | 0.4482 | 0.1283 | 0.0841 |
| `action_interpolation_endpoint_shift` | 1.0474 | 0.8907 | 0.3361 | 0.1702 |
| `predictor_rollout_T8_l2` | 18.62 | 17.90 | 18.65 | 14.01 |

机制解释：
- Hetero loss 在两个任务上都压缩表征：NN distance 降低，effective rank 降低，action-induced shift 降低。
- TwoRoom 低维、离散、视觉冗余，压缩表征是可接受甚至有利的。
- PushT 需要连续接触与姿态分辨率；`transition_resolution_ratio_l2` 从 0.3015 掉到 0.1023，`id_probe_r2` 从 0.7739 掉到 0.2678，说明 task-relevant state information 被抹掉。
- PushT 的 `predictor_rollout_T8_l2` 下降不是好消息：它意味着 latent 更容易预测，但不是更适合控制。预测稳定性是通过牺牲 resolution 得到的。

#### 8.2.4 结论

Pilot-1B 的结果是“语义成功、系统失败”：

1. **σ head 值得保留。** 它稳定学到了 per-transition prediction difficulty。
2. **直接 hetero training 不适合 PushT。** 它会把 high-error hard transitions 当成低权重样本，而这些 transition 很可能正是 PushT 的接触和精细控制关键区域。
3. **adaptive resolution 不能只靠 loss reweight。** 真正需要的是：μ 表征保留控制分辨率，σ 作为额外信号去调节 planning / consistency / compute，而不是让 σ 直接决定哪些 transition 不训练。

### 8.3 下一步主线：MSE-preserving σ + PushT resolution guardrail

目标不是“让 hetero loss 变温和一点”这么简单，而是让系统具备 **自适应分辨率**：低价值或不可预测扰动可以被 σ 标记和处理，PushT 的 task-critical continuous state/action resolution 必须保留。

#### 8.3.1 第一优先级：Probe-only σ head（恢复 Stage A，但目的改变）

训练：

```text
pred_loss = MSE(mu_hat, mu_target)
loss = pred_loss + lambda_SIGReg * SIGReg(mu)

err_token = mean((mu_hat.detach() - mu_target.detach())^2, dim=-1)
s_hat = pred_logvar_hat.squeeze(-1)
sigma_probe_loss = smooth_l1(s_hat, log(err_token + eps))

loss_total = loss + beta_probe * sigma_probe_loss
```

约束：
- `sigma_probe_loss` 只更新 σ head；不反传到 encoder / predictor mean path。
- μ path 必须退化为严格 LeWM baseline，避免 PushT resolution 再次被重加权破坏。
- 先跑 TwoRoom + PushT；如果 PushT clean 回到 LeWM-base 附近，同时 σ 仍有 `s_logerr_corr >= 0.5`，说明 σ 可以作为独立 adaptive signal 使用。

这一步回答：**能不能在不改变 μ 几何的前提下得到有语义的 σ？**

#### 8.3.2 第二优先级：σ-conditioned planner / controller，而不是训练重加权

优先实现不改变 μ 训练目标的使用逻辑：

| 方案 | 作用 | PushT 风险 | 推荐度 |
|---|---|---|---|
| σ-based CEM budget | 高 σ rollout 分配更多 candidates / restarts | 不改 μ，风险低 | 高 |
| σ uncertainty gate | 对高 σ plan 加轻量 penalty 或截断长 horizon | 可能过保守 | 中 |
| σ-conditioned noise consistency | 高/低 σ 区域使用不同 consistency 强度 | 需要 guardrail，风险中 | 中 |
| hetero loss 替换 MSE | 改 μ 梯度分配 | 已在 PushT 失败 | 暂停 |

最小可跑版本：

```text
train: LeWM MSE + SIGReg + detached sigma probe
eval/planning: rollout 时累计 mean(sigma_hat)，作为 uncertainty score
planner: 在候选 plan cost 上加 very small alpha * uncertainty，或高 uncertainty 时增加 CEM samples
```

先不要让 σ 直接改变 latent cost 的主项；PushT 对 cost surface 很敏感。

#### 8.3.3 第三优先级：如果必须让 σ 进 training，只能做 guarded auxiliary

候选 loss：

```text
loss = MSE + lambda_SIGReg * SIGReg
     + beta_probe * sigma_probe_loss
     + alpha * stopgrad_clip(hetero_loss - MSE)
```

或更简单：

```text
loss = (1 - alpha) * MSE + alpha * hetero_loss + lambda_SIGReg * SIGReg
```

约束：
- `alpha` 从 0.01 / 0.05 起步，不允许直接 `alpha=1`。
- `hetero_weight_q10_q90_ratio < 0.1` 立即 early-stop。
- `hetero_s_abs_max` 贴 `s_max=4` 立即降 `alpha` 或禁用 hetero branch。
- PushT 必须同时监控 `id_probe_r2`、`transition_resolution_ratio_l2`、`action_mean_pred_shift_norm`。

Guardrail 建议阈值（相对 PushT LeWM-base）：

| Metric | PushT LeWM-base | Stop / reject if |
|---|---:|---:|
| `id_probe_r2` | 0.774 | < 0.65 |
| `transition_resolution_ratio_l2` | 0.301 | < 0.24 |
| `action_mean_pred_shift_norm` | 0.128 | < 0.10 |
| clean eval | 87.33 / 86.00 | < 84 |

这些 guardrail 比单看 `pred_loss_mse_equiv` 更重要，因为本轮已经证明 MSE 可以下降但 planning 失败。

#### 8.3.4 最推荐的下一轮实验

只跑两个任务，先不扩到 4-task：

| Experiment | TwoRoom | PushT | 目的 |
|---|---:|---:|---|
| `lewm_sigma_probe_default` | yes | yes | 验证 σ 独立语义，不改 μ |
| `lewm_sigma_probe_planner_uncertainty_alpha001` | optional | yes | 测 σ 进入 planner 是否能改善 PushT noisy robustness |
| `lewm_hetero_alpha001_guarded` | optional | yes | 只作为训练重加权的极小权重对照 |

判定标准：
1. PushT clean 必须接近 LeWM-base（≥84）才继续。
2. σ calibration 必须保持：`validate/hetero_s_logerr_corr_epoch >= 0.5`。
3. PushT resolution guardrail 不得破。
4. 若 probe-only 成立但 planner use 无收益，方法降级为 uncertainty diagnostic；若 planner use 有收益，再谈 adaptive resolution 主方法。

### 8.4 Pilot-2: σ 使用逻辑

**触发条件**: Probe-only σ 保持 PushT clean / resolution，且 σ 语义稳定。当前 Pilot-1B 已显示直接 hetero loss 不能作为进入 Pilot-2 的前置成功条件。

可选项：
- σ-based CEM budget：高 uncertainty rollout 分配更多 samples。
- σ-based horizon gating：高 uncertainty 长 rollout 降权或截断。
- σ-conditioned noise consistency：高/低 σ 区域使用不同 consistency 强度，但必须避免新超参膨胀。

这一步才真正检验“额外输出头是否被系统用起来”。如果只停在 probe-only，它是诊断；如果停在 Pilot-1B，它是 loss weighting；进入 Pilot-2 后才是完整 adaptive system。

### 8.5 Validation: 4-task 全套

**触发条件**: Probe-only + σ 使用逻辑通过，且经验上至少接近 LeWM+noise oracle。

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

本文件现在是 plan_v3 §6 P4 的首选思考路线，但执行上必须分阶段。2026-05-09 Pilot-1B 已触发关键 fallback 条件：直接 hetero loss 伤害 PushT critical transition resolution。后续应优先做 probe-only σ 与 σ inference/controller use；guarded noise consistency / PI controller 仍可作为备选实现。

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
- 下一步主线是 §8.3：probe-only σ + PushT resolution guardrail + σ inference/controller use。
- 后续若 probe-only / Pilot-2 通过，把 §3 §4 §6 §8.3 合并进 plan_v3 §6 P4；本文件归档。
- **下一次想加新机制前**: 先回看 §10，问自己"它会增加几个超参数？经验收益的证据是什么？"。如果两个问题答不清楚，不加。
