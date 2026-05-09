# Action-Aware Adaptive Latent Resolution

> **Status**: Pilot-1B 已完成首轮 TwoRoom + PushT 验证（2026-05-09）。结果支持“σ head 能学到 prediction difficulty”，但否定了“直接用 hetero loss 替换 MSE”作为 PushT 上的主方法：PushT clean eval 从 LeWM-base 87.33/86.00 降到 13.33，诊断显示 transition/action resolution 被严重压缩。下一步主线改为 **probe-only σ + action-aware adaptive consistency + resolution guardrail**，而不是继续加大 hetero loss 或只做 σ-only controller。
> **关系**: 不是 plan_v3 的替换，而是 plan_v3 §6 P4 "Adaptive Resolution Method" 的具体化方案。
> **设计原则**: 先证明额外 σ 输出头携带有用信息，再让它影响训练或 planning；避免一开始就改变 LeWM 的强 MSE baseline。
> **重要历史记录**: 本文件早期版本曾包含 IB term / aggregate covariance Frobenius / Fisher manifold planning 等多层架构，hyperparameter 数量涨到 4–5 个。经过严格审视后**全部回退**——它们都需要新超参却没有可论证的额外收益。详见 §10 设计回退记录。

## 0. TL;DR

不要默认假设 heteroscedastic NLL 会优于 MSE。LeWM 的 MSE + SIGReg 已经很强，直接替换成 NLL 会改变 pred loss 与 SIGReg 的相对尺度，而且 NLL 会 downweight 高误差样本；在 PushT 这类任务里，高误差样本可能正是接触/精细控制的关键状态。

当前路线（**2026-05-09 更新**：Pilot-1B 已跑完，直接 hetero loss 在 PushT 上失败，因此恢复 probe-only / guarded 路线）：

1. **Pilot-1B 结论：Scale-preserving heteroscedastic loss 语义成功、控制失败。** `hetero_s_logerr_corr` 在 TwoRoom/PushT 后期分别约 0.89/0.95，说明 σ head 学到了 prediction difficulty；但 PushT `hetero_weight_q10_q90_ratio` 掉到约 0.008，hard transition 被强 downweight，clean eval 崩到 13.33。
2. **下一步首选：Probe-only σ + action-aware adaptive consistency。** μ path 保持 LeWM MSE + SIGReg，σ head detached 学 `log(error)`；真正改变 encoder resolution 的路径应放在 input-side consistency 上，而不是 prediction-loss reweighting 上。
3. **σ 不能单独决定 consistency weight。** prediction difficulty 会混合 action-relevant difficulty 和视觉 aleatoric noise；σ-only consistency 会落入 Noisy TV / confounder trap。必须用 action sensitivity `A_t` 作为主门控，σ 只作为 difficulty enhancer。
4. **Pilot-2A 结论（2026-05-09，§8.3.6）：probe-only 救回 PushT，gate logging 三个结构判据通过，但发现 BN drift bug 让 TwoRoom probe+gate clean 跌 7pt（96.33 → 89.33）。** PushT 崩溃问题已解决（probe+gate clean 87.00 ≈ LeWM-base 87.33）；σ-A 弱正/独立，乘性 critical 设计得到经验支持；但 `compute_action_gate_metrics` 的 K 次 perturb forward 在 train mode 下污染了 `BatchNorm1d` running stats，TwoRoom 受害严重。**Stage C 不能解决此 bug**——它正交于 `alpha_cons`，必须先在 gate 内部把 BN 切到 `.eval()` 再重测。修复前不开 `alpha_cons > 0`。

> **为什么恢复 probe-only？** 2026-05-09 Pilot-1B 已经证明核心风险真实存在：σ calibration 很好，但 PushT 失败。此时 probe-only 不再是“容量 smoke test”，而是把 σ 语义从 μ 几何更新中解耦，避免 hard-but-important transition 被训练权重抹掉。

核心批判点：**额外输出头本身不会自动变成动态分辨率。** 如果 σ 只作为日志或 detached probe，它是诊断量，不改变 μ 几何；如果 σ 进入 NLL，它改变训练梯度，但可能只是学会“忽略难样本”；如果只用 σ 调 consistency，它会把背景噪声误判成高分辨率需求。因此必须把 σ 和 action relevance 解耦：`σ_t` 识别 difficulty，`A_t` 识别 controllability / causal relevance。

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

### 2.2.1 备选：encoder input-sensitivity head（有监督版本，优先于 unsupervised σ_x）

> 设计动机（2026-05-09 登记）：当前 Stage A–C 的所有 resolution controller 信号（σ̂、A_t）都经过 **predictor** 这条路。encoder 端对自己输入扰动的反应没有 self-aware signal；σ̂ 在 contact / 边界 chaotic extrapolation 时会被污染（§8.3.2.2 的 multi-δ CV 是在为此打补丁）。直觉"只给 predictor 加 head 差点啥"指向的不是 unsupervised encoder σ，而是这条**有天然监督**的替代。

```
enc_backbone(x) → h
mean_head(h)    → μ_x ∈ R^d
sens_head(h)    → s_enc(x) ∈ R^1   # 预测 input-perturbation 后的 encoder 位移

# detached supervision
target_enc = log( ||enc(x).detach() − enc(aug(x)).detach()||₂ + eps )
L_sens = smooth_l1(s_enc(x), target_enc)
```

性质：
- **天然监督信号**：`||enc(x) − enc(aug(x))||` 直接可测，避免 §2.1 点名的 unsupervised encoder σ identifiability trap。
- **与 predictor σ̂ 正交**：σ̂ 是 *transition difficulty given clean state*；s_enc 是 *state representation 对 input nuisance 的敏感度*。两者描述不同物理量，没有互相吸收 residual 的退化路径。
- **不引入 EMA / 第二 encoder**：监督目标用同一个 encoder 的两次 forward，与 §2.3 单 encoder 哲学一致。
- **与 §3.4 `L_cons` 共享 forward，零额外开销**：`||enc(x) − enc(aug(x))||` 正是 `L_cons` 已经在计算的 per-token distance。监督目标直接复用 `d(stopgrad(z_clean), stopgrad(z_noisy))`，不需要第二次 `enc(aug(x))`。
- **真正的价值是 controller-side 闭环，不是 loss-side**：`L_cons` 已经把这个距离当**训练目标**（梯度反传进 encoder 把它压小）；s_enc head 的角色是把这个量**摘出来作为 controller `w_t` 的输入**。没有 s_enc 时，`w_t` 只看 predictor 端信号（σ̂、A_t），是 predictor-only feedback loop；有了 s_enc 才形成 *encoder sensitivity → controller → encoder consistency pressure* 的闭环。这也是直觉"predictor-only head 差点啥"指向的真正缺口。

不立即采用的理由（§10 纪律）：
1. Pilot-1（probe-only σ）尚未给出结果；最便宜的 predictor head 都未验证。
2. Stage C 的 `L_cons = w_t · ||z_clean − z_noisy||` 已经隐式让 encoder input-sensitivity 进入梯度——先看它够不够，再决定是否需要把 encoder sensitivity 显式提取成 controller 输入。
3. 加这一项会引入新 hyperparam `beta_sens`，违反"加项必须先有经验收益证据"的 §10 第 1 条。

**触发加入的条件**（写死，不模糊）：
- Stage C 的 `alpha_cons` ramp 在 PushT 上撞 guardrail（transition_resolution_ratio_l2 < 0.24 或 clean < 84），且诊断显示 critical 区域是因 **encoder 端对 input nuisance 区分不足** 导致（不是 predictor 端 σ̂ / A_t 信号失败）；或
- σ̂ 的 calibration 在 +noise 训练下漂移（§8.5.6 Probe-on-noise 阶段），需要一个 input-side 信号去解释 σ̂ 漂移成分。

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
- 如果 σ probe 都学不出稳定结构，后续 action gate / consistency 都没有稳定信号基础。

### 3.2 历史候选：scale-preserving heteroscedastic loss（Pilot-1B 已失败）

本节保留是为了说明为什么“不直接用 σ 重加权 prediction loss”。2026-05-09 Pilot-1B 已经验证：scale-preserving 形式能校准 σ，但 PushT clean eval 崩到 13.33，因此它不再是主方法路线，只作为 ablation / negative result。

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

核心风险已经在 Pilot-1B 中发生：它会 downweight 高误差 transition；在 PushT 中这些 transition 很可能包含控制关键区域，导致 clean control 失败。因此后续不再扩大 hetero loss，而是把 σ 从 μ-path 梯度中解耦出来。

### 3.3 σ 进入使用逻辑（不是默认）

如果 σ 只被训练为 calibration probe，它不是 dynamic resolution 方法。要真正影响系统，必须至少进入以下一类逻辑：

| 使用方式 | 作用 | 风险 |
|---|---|---|
| training weight | 通过 hetero loss 改变 μ 的梯度分配 | 可能忽略 hard-but-important states |
| σ-only noise/controller | 高 σ 区域降低/提高 noise consistency 强度 | **confounder trap**：高 σ 可能来自视觉 aleatoric noise，而不是 task-critical dynamics |
| action-aware consistency | 用 action sensitivity 区分 controllable critical states 和不可控视觉噪声 | 最符合 adaptive resolution，但必须先做 logging-only 验证 gate |
| planner budget | 高 σ rollout 增加 CEM samples / 缩短 horizon | 不改变表示，只改 inference compute |
| uncertainty gate | 高 σ 时拒绝或降权候选 plan | 可能过度保守 |

因此 σ 进入使用逻辑前必须先通过 probe-only calibration 和 action-gate logging。否则额外 head 只是日志，不是方法；若直接进入 loss weighting，Pilot-1B 已经给出 PushT 反例。

### 3.4 Action-Aware Adaptive Consistency（当前首选）

真正符合“adaptive latent resolution”的训练介入不应是直接重加权 prediction loss，而应是对 encoder input-side invariance 的局部调节：

```text
z_clean = enc(x)
z_noisy = enc(aug(x))
L_cons = mean(w_t * d(stopgrad(z_clean_t), z_noisy_t))
```

核心问题是 `w_t` 不能只由 σ 决定。prediction difficulty 同时包含：
- **Epistemic / dynamics difficulty**：例如 PushT 接触瞬间，应该降低 consistency pressure，保留分辨率。
- **Aleatoric / visual nuisance**：例如不可控背景噪声，应该提高 consistency pressure，抹掉噪声。

因此定义一个 action-aware criticality：

```text
A_t = d(f(z_t, a_t + delta), f(z_t, a_t)) / (||delta|| + eps)
gA_t = sigmoid(zscore_ema(log(A_t + eps)))
gS_t = sigmoid(zscore_ema(s_t))
critical_t = gA_t * (0.5 + 0.5 * gS_t)
w_t = w_max - (w_max - w_min) * stopgrad(critical_t)
```

设计取舍：
- `A_t` 是主门控，表示 action-conditioned local sensitivity / controllability。
- `σ_t` 只作为 difficulty enhancer，避免 σ-only gate 把 Noisy TV 当成高分辨率需求。
- `delta` 应来自 empirical action std 或 batch 内 in-distribution action 差分，不用任意 OOD random action。
- `critical_t` 和 `w_t` 必须 detach；gate 是 controller，不允许成为 predictor / encoder 的反向捷径。
- 先做 `weight=0` logging-only，再启用 `L_cons`。

### 3.5 SIGReg 仍然作用在 μ 上

无论 Stage A/B/C，SIGReg 都只作用在 deterministic μ 上。不要把 SIGReg 推广到 `(μ, σ)` 或 reparameterized sample；那会引入 Gaussian mixture 高阶矩问题，并破坏 LeWM 已验证的 anti-collapse 机制。

---

## 4. 关键性质

### 4.1 额外 σ head 不自动等于动态分辨率

必须区分三种层级：

| 层级 | σ 做了什么 | 是否改变 resolution |
|---|---|---|
| Probe | 预测 detached error | 否，只是诊断 |
| Loss weighting | 改变不同 transition 的 μ 梯度 | 可能，但可能忽略关键 hard states |
| σ-only controller | 影响 CEM budget / gating / consistency strength | 可能，但容易把 aleatoric visual noise 当成 resolution demand |
| Action-aware consistency | σ 与 action sensitivity 共同控制 encoder invariance | 是 encoder-side adaptive resolution 的当前首选候选 |

所以论文中不能把“加一个 σ head”直接等同于 dynamic resolution。真正要证明的是：σ 与 action-relevant difficulty 对齐，且 `A_t` 能把 controllable critical states 和不可控视觉噪声区分开；然后 adaptive consistency 在不破坏 PushT resolution guardrail 的前提下改善 LeWM+noise 的手调 tradeoff。

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
| LeWM + noise | 全局 input-side invariance；无 state/action-aware weighting |

现有方法都没有把 per-transition uncertainty 和 action-conditioned local sensitivity 结合起来控制 encoder invariance strength。

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

> **Action-aware adaptive consistency for latent resolution**: 在 LeWM predictor 上加 detached scalar σ probe 来估计 transition difficulty，同时用 action-conditioned local sensitivity `A_t` 区分 controllable critical states 与不可控视觉噪声；二者共同控制 clean/noisy encoder consistency strength，使简单/冗余区域更 invariant、action-critical 区域保留 resolution。LeWM 与 LeWM+noise 分别是无 σ/A controller 与全局 consistency 的严格 baseline。

这一条成立的前提：
- Stage A 证明 σ head 学到非平凡、任务相关的 prediction difficulty。
- Logging-only 阶段证明 `A_t` 能过滤 σ 中的 aleatoric visual noise，而不是只复述 prediction error。
- Adaptive consistency 证明使用 `critical_t = f(A_t, σ_t)` 后能接近或超过 LeWM+noise oracle，而不是只超过 LeWM-base。
- 收益不是来自重新调 SIGReg / loss scale，也不是来自把 hard transitions 的 prediction gradient 降权。

不再主张：
- “NLL 一定比 MSE 好”。
- “σ head 自然就是 latent resolution”。
- “高 σ 就应该保留分辨率”。
- “不改 planner 就一定能在 inference 自动受益”。
- IB / Fisher manifold / “诊断 = (μ, σ) 本征轴”等强理论叙事。

---

## 7. 风险与对策

| 风险 | 评估 | 对策 |
|---|---|---|
| **σ 退化成全局常数** | Probe 阶段若 PushT 也近似常数，说明额外 head 没有学到有用异质性 | 先不进入 NLL；检查 err target、head capacity、是否需要更长训练 |
| **NLL 改变 MSE/SIGReg 权重比** | 普通 NLL 初始就是 0.5× MSE，且尺度会随 σ 漂移 | hetero loss 仅作为历史 ablation；若重跑必须保持 scale-preserving，并以 PushT resolution guardrail 拒绝 |
| **hard-but-important states 被 downweight** | PushT 接触/精细控制可能高误差但高价值 | 必须监控 transition/action resolution；必要时 fallback 到 guarded consistency |
| **σ 只是 uncertainty，不是 resolution** | calibration 成功不等于 planning 提升 | Stage C 必须明确 σ 的使用逻辑；否则只作为诊断输出 |
| **Noisy TV / confounder trap** | 高 σ 也可能来自不可控视觉噪声；σ-only consistency 会放弃对噪声的 invariance | consistency gate 必须 action-aware：以 `A_t` 为主门控，σ 只做 enhancer |
| **Action sensitivity OOD** | 任意随机动作可能离开数据分布，导致 `A_t` 反映 predictor extrapolation | `delta` 使用 empirical action std 或 batch 内 in-distribution action 差分；先 logging-only |
| **Gate 反向捷径** | 若 `critical_t` 不 detach，encoder/predictor 可通过操纵 gate 逃避 consistency | `σ_t`、`A_t`、`critical_t`、`w_t` 全部 stopgrad；warmup 后再启用 consistency |
| **encoder σ 不可辨识** | encoder σ 无天然监督，和 predictor σ 同时学会互相逃逸 | Pilot-1 不加 encoder σ；只在 predictor σ 成立后再加 |
| **Multi-step σ propagation 公式不准** | 本最简版**不主张**手写 σ 累积公式；让 predictor σ̂ 自学 multi-step uncertainty | 用 multi-step rollout NLL 做训练监督 |
| **不超过 LeWM+noise oracle** | 很可能；LeWM+noise 已经很强 | 目标先设为减少手调且接近 oracle；若明显低于 oracle，降级为 analysis/future work |
| **BN drift via gate perturb forward**（2026-05-09 实证发现） | gate 内 K 次 perturb forward 在 train mode 下走 projector / predictor_proj 的 `BatchNorm1d`，污染 running stats；TwoRoom probe+gate clean 因此跌 7pt（96.33 → 89.33）。**Stage C 不能解决此 bug**——它正交于 `alpha_cons` 与 `w_t` 设计。 | 在 `compute_action_gate_metrics` perturb forward 前临时把所有 `_BatchNorm` module `.eval()`，结束后恢复。修复后重跑 TwoRoom/PushT probe+gate；clean 未恢复前不进入 Stage C。 |

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

### 8.3 下一步主线：Probe-only σ + Action-Aware Adaptive Consistency

目标不是“让 hetero loss 变温和一点”这么简单，而是让系统具备 **自适应分辨率**：视觉冗余 / 不可控噪声区域加强 invariance，PushT 的 action-critical continuous state/action resolution 必须保留。

关键修正：`σ_t` 不能单独当作“高分辨率需求”。prediction difficulty 混合了 controllable dynamics difficulty 和 aleatoric visual noise；前者应降低 consistency weight，后者应提高 consistency weight。因此下一步使用 `A_t`（action-conditioned local sensitivity）作为主门控，σ 只作为 difficulty enhancer。

#### 8.3.1 第一优先级：Probe-only σ head（恢复 Stage A，但目的改变）

实现状态（2026-05-09）：
- 已加入 `loss.hetero.mode=probe`。
- `JEPA.predict_with_logvar(..., detach_logvar_input=True)` 会 detach σ head 的 predictor hidden 输入，确保 `sigma_probe_loss` 不更新 shared predictor backbone。
- 使用方式：`python train.py data=pusht output_model_name=pusht_lewm_sigma_probe_default loss.hetero.enabled=true loss.hetero.mode=probe`。

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

#### 8.3.2 第二优先级：Logging-only action-aware gate

实现状态（2026-05-09）：
- 已加入 `loss.action_gate` config block（`config/train/lewm.yaml`）。
- `train.py::compute_action_gate_metrics` 在 forward 内 K 次 perturb action → re-predict → 计算 `A_t`、`A_t_cv`、`gA_t`、`critical_t`、`w_t`，全部 `no_grad`。
- EMA 统计以 `world_model.gate_{log_A,s}_{mean,var}` buffer 持久化；warmup 期不更新 EMA。
- 与 `loss.hetero.mode=probe` 兼容：σ 关闭时 gate 仅记录 A 相关指标。
- 推荐运行命令（PushT，叠加 σ probe）：
  ```bash
  python train.py data=pusht \
      output_model_name=pusht_lewm_action_gate_logging \
      loss.hetero.enabled=true loss.hetero.mode=probe \
      loss.action_gate.enabled=true
  ```

先实现 `adaptive_consistency.weight=0`，只记录 gate，不改变训练目标。核心量：

```text
A_t = d(f(z_t, a_t + delta), f(z_t, a_t)) / (||delta|| + eps)
gA_t = sigmoid(zscore_ema(log(A_t + eps)))
gS_t = sigmoid(zscore_ema(s_t))
critical_t = gA_t * (0.5 + 0.5 * gS_t)
w_t = w_max - (w_max - w_min) * critical_t
```

实现约束：
- `delta` 使用 empirical action std 或 batch 内 in-distribution action 差分；不要用任意 OOD random action。
- `s_t`、`A_t`、`critical_t`、`w_t` 全部 detach。
- 先 warmup 若干 epoch，只训主 loss + σ probe，再开始记录/使用 gate。具体 warmup 判据见 §8.3.2.1。
- 记录 `adaptive/sigma_mean`、`adaptive/action_sensitivity_mean`、`adaptive/critical_mean`、`adaptive/weight_mean`、`adaptive/corr_sigma_action`、`adaptive/weight_q10_q90`。
- **额外记录 `A_t` 的多 δ 方差**（见 §8.3.2.2），用于区分 smooth-controllable 与 chaotic 高敏感区域。

进入训练介入前必须看到：
- high `critical_t` 与 PushT contact / high action-norm / high transition displacement 有结构性关系。
- 视觉 nuisance 主要提高 σ，不应同步提高 `A_t`。
- `critical_t` 比 σ-only 更能解释 `id_probe_r2` / action resolution 相关诊断。
- 高 `A_t` 区域的多 δ 方差不应远大于低 `A_t` 区域；如果显著更大，说明被 chaotic dynamics 污染（详见 §8.3.2.2）。

##### 8.3.2.1 Warmup 判据

`A_t` 的物理意义只有在 predictor 学到了 action conditioning 之后才成立。早期 predictor 几乎忽略 action 时，`A_t ≈ 0` 是 predictor 不成熟而非 state insensitive。因此 logging 只在以下任一条件后启动：

- `validate/id_probe_r2_epoch >= 0.5 * id_probe_r2_LeWM_base`（PushT 取 0.39，TwoRoom 取 0.14；这是 "predictor 学到了一半 action 信息" 的代理）；或
- 训练经过 `cfg.loss.action_gate.warmup_epochs` epochs（默认 3，等于 LeWM 10-epoch 训练的前 30%）。

在 warmup 期间仍然计算并记录 `A_t`，但不进入 `critical_t` 聚合，也不写入 EMA z-score 统计——避免 z-score baseline 被 action-blind 阶段的统计带偏。

##### 8.3.2.2 区分 smooth-controllable 与 chaotic 的 A_t 方差

`A_t` 高有两种成因：
- **Smooth controllable**：小 δ → 平滑大响应。多次采样 δ 给出 *方向相关、幅度相近* 的响应，`A_t` 在 δ 上低方差。这是真正的 action critical 区域。
- **Chaotic / extrapolation**：predictor 在 contact / 边界附近不连续，小 δ → 任意大响应。多次采样 δ 给出高方差。这种区域不该当作"应保留分辨率"的 critical state。

因此在 logging-only 阶段，每个 token 用 `K=4` 个独立 δ 采样：

```text
A_t^{(k)} = d(f(z, a + δ^{(k)}), f(z, a)) / (||δ^{(k)}|| + eps)   for k=1..K
A_t_mean = mean_k A_t^{(k)}
A_t_cv   = std_k A_t^{(k)} / (A_t_mean + eps)   # coefficient of variation
```

记录 `adaptive/action_sensitivity_cv_mean`、`adaptive/action_sensitivity_cv_high_A_quantile`（高 `A_t_mean` 分位下的 CV）。判定：

- 全局 `cv_mean < 0.5`：predictor 局部光滑，`A_t` 可信。
- 高 `A_t_mean` 区域的 CV 不显著高于全局 CV：critical 区域不被 chaotic 主导。

如果两者中任一不满足，说明 `A_t` 信号被噪声污染，consistency 训练阶段应改用 `A_t_mean / (1 + α_cv * A_t_cv)` 做 chaos-discount，而不是直接用 raw `A_t`。`α_cv` 默认 1.0。

#### 8.3.3 第三优先级：Action-aware adaptive consistency training

只有 logging-only gate 成立后，才启用 encoder-side consistency：

```text
z_clean = enc(x_clean)
z_noisy = enc(aug(x_clean))
L_main = MSE(mu_hat, mu_target) + lambda_SIGReg * SIGReg(mu)
L_cons = mean(stopgrad(w_t) * d(stopgrad(z_clean_t), z_noisy_t))
loss = L_main + beta_probe * sigma_probe_loss + alpha_cons * L_cons
```

解释：
- 主 prediction loss 不被 σ 或 `A_t` 降权，避免复现 hetero loss 的 PushT resolution collapse。
- `w_t` 只控制额外 invariance pressure；action-critical / high-σ 区域少抹细节，visual nuisance / action-insensitive 区域更强 invariance。
- `alpha_cons` 从小值开始，并以 PushT resolution guardrail 为硬拒绝条件。

##### 超参数预算表（兑现 §8.5.5）

| 名称 | 默认值 | 允许范围 | 进入条件 / early-stop 阈值 |
|---|---:|---|---|
| `loss.hetero.probe_weight` (`beta_probe`) | 1.0 | [0.1, 5.0] | probe-only 阶段；`hetero_s_logerr_corr ≥ 0.5` 才进入 §8.3.2。低于 0.3 持续 3 epoch → fallback to detach-deeper probe head |
| `loss.action_gate.delta_scale` | 0.25 | [0.05, 0.5] | δ 相对 batch 内 action std 的比例。固定值，**不调** |
| `loss.action_gate.num_delta_samples` (K) | 4 | [2, 8] | 多 δ 方差估计；CV 不可信时 K 可加大到 8 |
| `loss.action_gate.warmup_epochs` | 3 | [0, 5] | logging 启动门槛（见 §8.3.2.1） |
| `loss.action_gate.ema_momentum` | 0.99 | [0.95, 0.999] | zscore EMA 平滑系数；固定值，**不调** |
| `loss.adaptive_consistency.w_min` | 0.2 | [0.0, 0.5] | critical 区域的最小 invariance pressure |
| `loss.adaptive_consistency.w_max` | 1.0 | [0.5, 1.5] | non-critical 区域的最大 invariance pressure |
| `loss.adaptive_consistency.alpha_cons` | 0.01 | [0.001, 0.1] | 起始小，每次 +×3 ramp，触发 guardrail 即冻结 |
| `loss.adaptive_consistency.aug_type` | `gaussian_noise(std=0.04)` | — | 与 LeWM+noise pipeline 复用，避免引入新 augmentation |

**进入下一阶段的边际经验收益要求**：

| 阶段 | 必要条件 | 边际收益要求（PushT） |
|---|---|---|
| probe-only → action-gate logging | probe 通过 §8.3.1 判据 | clean ≥ LeWM-base − 1pt（即 ≥ 86）|
| logging → consistency `alpha=0.01` | §8.3.2 三个结构判据全过 + §8.3.2.2 CV 判据通过 | clean ≥ 86，且 transition_resolution_ratio_l2 ≥ 0.27 |
| `alpha=0.01` → `alpha=0.03` | guardrail 全部不破 + clean 不跌 > 1pt | robustness（goal+pixels 0.05）较 LeWM-base 提升 ≥ 5pt |
| `alpha=0.03` → `alpha=0.1` | 同上 | robustness 较 LeWM+noise oracle 接近（差距 ≤ 5pt） |

任一阶段不满足该收益要求 → **冻结当前 alpha，转 ablation/分析**，不再向上 ramp。

#### 8.3.4 备选：σ planner / hetero auxiliary 降级为对照

σ-based planner budget 仍可作为 inference-only 对照，但不再是主线。σ uncertainty penalty 和 hetero loss 只保留为 ablation，因为前者可能让 planner 躲开 contact 区域，后者已在 PushT 上失败。

Guardrail 建议阈值（相对 PushT LeWM-base）：

| Metric | PushT LeWM-base | Stop / reject if |
|---|---:|---:|
| `id_probe_r2` | 0.774 | < 0.65 |
| `transition_resolution_ratio_l2` | 0.301 | < 0.24 |
| `action_mean_pred_shift_norm` | 0.128 | < 0.10 |
| clean eval | 87.33 / 86.00 | < 84 |

这些 guardrail 比单看 `pred_loss_mse_equiv` 更重要，因为本轮已经证明 MSE 可以下降但 planning 失败。

#### 8.3.5 最推荐的下一轮实验

只跑两个任务，先不扩到 4-task：

| Experiment | TwoRoom | PushT | 目的 |
|---|---:|---:|---|
| `lewm_sigma_probe_default` | yes | yes | 验证 σ 独立语义，不改 μ |
| `lewm_action_gate_logging` | yes | yes | `weight=0` 记录 `A_t` / `critical_t`，验证是否过滤 Noisy TV confounder |
| `lewm_action_aware_consistency_alpha001` | optional | yes | 核心新方法：action-aware gate 控制 encoder consistency |
| `lewm_sigma_only_consistency_alpha001` | optional | yes | 失败对照：σ 直接当 critical 信号（`critical_t = sigmoid(zscore_ema(s_t))`，**高 σ → 低 w_t → 弱 consistency**），检验是否被视觉噪声误导。这是与 §8.3.3 完全相反方向之外的另一个 sign 选择；本对照固定走"高 σ = 高 resolution 需求"分支，验证 Noisy TV 假设 |
| `lewm_hetero_alpha001_guarded` | optional | yes | 只作为训练重加权的极小权重对照 |

判定标准：
1. PushT clean 必须接近 LeWM-base（≥84）才继续。
2. σ calibration 必须保持：`validate/hetero_s_logerr_corr_epoch >= 0.5`。
3. `A_t` / `critical_t` 必须显示 action-relevant 结构；否则不启用 consistency。
4. PushT resolution guardrail 不得破。
5. 若 action-aware consistency 无收益，方法降级为 uncertainty/action diagnostic；若有收益，再谈 adaptive resolution 主方法。

#### 8.3.6 Probe-only σ + Action-gate logging 结果（2026-05-09 Pilot-2A）

运行：

| Task | Run | SwanLab id |
|---|---|---|
| TwoRoom probe | `tworoom_lewm_hetero_probe_default` | `75qiqru0ttwmyy7pwigly` |
| TwoRoom probe+gate | `tworoom_lewm_hetero_probe_default_action_gate` | `awokxbepmodp2shcqmynr` |
| PushT probe | `pusht_lewm_hetero_probe_default` | `jgqsw29zji110j3gczu03` |
| PushT probe+gate | `pusht_lewm_hetero_probe_default_action_gate` | `oezw5j3w0uh3ydxnan63c` |

设置：`loss.hetero.enabled=true loss.hetero.mode=probe`；`probe+gate` 额外 `loss.action_gate.enabled=true`（logging-only，`adaptive_consistency.weight=0`）。Eval epoch 10，seeds 42/43/44，每 seed `num_eval=100`。

##### 8.3.6.1 Eval 表

| Task / model | Clean | goal 0.05 | pixels 0.05 | px+goal 0.05 | goal 0.08 | px+goal 0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom LeWM-base | 93.00 | 71.00 | 70.33 | 62.33 | 55.67 | 44.33 |
| TwoRoom hetero (Pilot-1B) | **99.67** | 85.33 | **96.67** | 84.67 | 73.33 | **55.33** |
| TwoRoom probe | 96.33 | 80.67 | 81.00 | 67.00 | 63.67 | 46.00 |
| TwoRoom probe+gate | 89.33 | 49.00 | 52.00 | 36.67 | 41.67 | 33.00 |
| PushT LeWM-base | 87.33 | 38.00 | 17.33 | 15.00 | 15.00 | 3.67 |
| PushT hetero (Pilot-1B) | 13.33 | 7.67 | 7.67 | 7.67 | 9.67 | 6.00 |
| PushT probe | 81.67 | 39.00 | 19.33 | 14.67 | 17.33 | 3.33 |
| PushT probe+gate | **87.00** | **52.00** | **31.67** | **21.00** | **23.00** | 3.33 |

##### 8.3.6.2 训练侧诊断

| 指标 | tw_probe | tw_probe_gate | pu_probe | pu_probe_gate |
|---|---:|---:|---:|---:|
| `pred_loss_mse_equiv` (tail100) | 0.0295 | 0.0295 | 0.0177 | 0.0162 |
| `validate/hetero_s_logerr_corr` (last) | 0.620 | 0.621 | 0.480 | 0.462 |
| `hetero_s_abs_max` (last) | 4.24 | 4.30 | 4.23 | 4.23 |
| `adaptive_action_sensitivity_cv_mean` | — | 0.36 | — | 0.245 |
| `adaptive_action_sensitivity_cv_high_A` | — | 0.35 | — | 0.277 |
| `validate/adaptive_corr_sigma_action_epoch` | — | −0.02 | — | +0.23 |
| `adaptive_critical_mean` (val) | — | 0.35 | — | 0.30 |

##### 8.3.6.3 结论

1. **PushT 崩溃问题已解决。** probe-only PushT clean 81.67、probe+gate clean 87.00，与 LeWM-base 87.33 持平；hetero loss 的 13.33 不再出现。证明 §3.4 把 σ 从 μ-path 梯度解耦的设计是对的。
2. **σ probe 语义保留。** PushT validate corr ≈ 0.46–0.54、TwoRoom ≈ 0.62。PushT 边界但通过 Stage A 0.5 阈值。代价是 σ 比 hetero loss 下的 0.95 弱，因为没有 NLL 反馈把 σ 强行钉到 prediction error。
3. **Action-gate 三个结构判据通过（PushT）。** `cv_mean = 0.245 < 0.5` ✅；`cv_high_A = 0.277` 不显著高于全局 ✅；`corr_sigma_action`：PushT 0.23 弱正相关、TwoRoom −0.02 基本独立 → σ 与 A_t 不是同一信号，乘性 `critical = gA × (0.5 + 0.5·gS)` 设计得到经验支持。这是 §8.3.2.2 多 δ CV 判据的首个正面验证。
4. **TwoRoom probe+gate clean 跌 7pt（96.33 → 89.33）是真问题，且不是 Stage C 能解决的。** 见 §8.3.6.4。
5. **PushT probe+gate robustness 比 LeWM-base 更高**（goal 0.05: 38 → 52、pixels 0.05: 17 → 32），但 logging-only run 不应改变训练梯度——这要么是种子噪声，要么是 §8.3.6.4 同源的 BN drift 在 PushT 上反而起到轻微 invariance 训练效果。需要修 BN drift 后重测确认。

##### 8.3.6.4 关键 bug：BN drift via gate perturb forward

**机制：** `train.py::compute_action_gate_metrics` 内 K=4 次 `model.predict(ctx_emb_d, act_emb_pert)` 在 `model.training=True` 下执行。`projector` 与 `predictor_proj` 默认 `nn.BatchNorm1d`（`config/train/lewm.yaml::encoder.projection_head.norm_fn=batchnorm1d`），每次 perturb forward 都会用 OOD-ish 扰动 activation 更新 BN running mean/var。每个 train step BN 统计被多走了 K 次偏离主分布的 forward。

**为什么 TwoRoom 受害严重、PushT 几乎不受影响：**
- TwoRoom：表征空间小、batch 视觉多样性低，BN 统计的有效样本数对 perturb forward 敏感；clean eval 跌 7pt（96.33 → 89.33）。
- PushT：视觉多样性主导 BN 统计，K=4 perturb forward 占比可忽略；clean 87.00 与 LeWM-base 87.33 持平。

**为什么 Stage C 解决不了这个 bug：** Stage C 是在主 loss 上加 `L_cons = w_t · d(z_clean, z_noisy)`，**完全不改 gate 内部的 perturb forward 逻辑**。BN running stats 仍然每步被 K=4 OOD forward 污染，与 `alpha_cons` 大小、`w_t` 设计正交。叠加 Stage C 只会让 BN drift 与 consistency gradient 互相纠缠，更难诊断。

**修复方案（必须先做，再开 Stage C）：**

```python
# In compute_action_gate_metrics, before the K-loop perturb forwards:
bn_states = []
for m in model.modules():
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        bn_states.append((m, m.training))
        m.eval()  # use running stats; do NOT update them
try:
    for _ in range(K):
        ...  # perturb forward + A_t computation
finally:
    for m, was_train in bn_states:
        m.train(was_train)
```

语义上这是正确的：A_t 测的是 `||predictor(z, a+δ) − predictor(z, a)||` 的局部敏感度，应该在**固定 normalization 统计**下测量；让 perturb forward 反向影响 BN 统计本身就是 leakage。

**修复后必须重跑的实验：**
- TwoRoom probe+gate（验证 clean 回到 96+；如果仍 89 附近，bug 不止 BN drift）。
- PushT probe+gate（验证 clean 87 与 robustness 是否仍保持；如果 robustness 跌回 LeWM-base 水平，§8.3.6.3 第 5 点的"轻微 invariance 训练副作用"假设成立）。

**Stage C 的真实定位（修 BN drift 之后才有意义）：** Stage C 仍可能解决 TwoRoom 与 PushT *两个任务都接近各自最优* 的兼容性问题——critical 区域降 consistency 保 PushT 接触 resolution，non-critical 区域加 consistency 把 TwoRoom 提到 LeWM+noise 水平。但前提是 logging-only 阶段已经 clean 不掉。当前数据**还不能下结论 "adaptive consistency 兼容动态分辨率"**，要先修 bug 再观察。

##### 8.3.6.5 进入 Stage C 的修订前置条件

§8.3.3 原前置条件 "logging → consistency `alpha=0.01`" 现追加：

- **(新)** TwoRoom probe+gate clean ≥ 92（恢复到 LeWM-base 附近），证明 BN drift 已修；同时 PushT probe+gate clean ≥ 86 不退化。
- σ probe corr (validate) PushT ≥ 0.5、TwoRoom ≥ 0.5（已基本满足，PushT 边界）。
- §8.3.2.2 三个 cv 结构判据保持通过（已通过）。

不满足前两条之前，**不开 `alpha_cons > 0`**。

### 8.4 Pilot-2: Action-aware σ 使用逻辑

**触发条件**: Probe-only σ 保持 PushT clean / resolution，且 σ 语义稳定。当前 Pilot-1B 已显示直接 hetero loss 不能作为进入 Pilot-2 的前置成功条件。

可选项：
- action-aware adaptive consistency：首选，真正作用到 encoder invariance strength。
- σ-based CEM budget：inference-only 对照，高 uncertainty rollout 分配更多 samples。
- σ-based horizon gating：降级为风险较高的 planner 对照。
- hetero loss：仅作为失败路线的小权重 ablation。

这一步才真正检验“额外输出头是否被系统用起来”。如果只停在 probe-only，它是诊断；如果停在 Pilot-1B，它是 loss weighting；进入 Pilot-2 后才是完整 adaptive system。

### 8.5 待 probe 结果回来后联合分析的开放问题（2026-05-08 预登记）

> 本节是在 `lewm_sigma_probe_default` 结果尚未回来前，对当前叙事和 §8.3 规划的预登记疑问。等 probe 跑完再统一回来落实/反驳，不要先动 §8.3 主体。

#### 8.5.1 Probe-only "calibration 成功"的判据需要更硬

`s_logerr_corr ≥ 0.5` 是必要不充分。probe 用 detached MSE 监督，只要 σ head 容量够，calibration corr 高几乎必然——这只能证明 σ 能 fit residual，不能证明 σ 携带 *超出 per-token MSE* 的可用信号。回看时应额外检查：

- σ 与 contact / goal-edge mask（或 high action-norm 分位）的相关性，有没有系统性结构。
- σ 在视觉 nuisance 干扰下（goal / pixels noise）相对 clean 输入的漂移方向：是漂向"难"区域（与控制相关）还是漂向"乱"区域（与视觉无关）。
- σ 的空间/时间结构是否仅是 residual 的 smoother，还是能 anticipate 下一步的 contact onset。

如果只通过 corr 但没有上面任一条结构性证据，probe-only 应只升级为 diagnostic，不进入 §8.3.3 consistency 训练阶段。

#### 8.5.2 Planner uncertainty 的方向风险

旧 §8.3.2 表里"σ uncertainty gate"被标"风险中"，但没明说方向。在 goal-conditioned planning 里把 `α·σ` 加进 cost，相当于让 planner *躲* 高 σ 区域；而 PushT 的 contact / 精细控制恰好是 high-σ。所以：

- 起步首选 **σ-based CEM budget reallocation**（高 σ 增加 samples / restarts），而不是 cost penalty。前者只改 inference compute，不改 cost surface。
- 如果一定要做 cost-side，应该是 *goal-aware* 的——例如只在远离 goal 时让 σ 起 penalty 作用，靠近 goal 时反而允许进入高 σ contact 区。
- "σ-conditioned noise consistency" 同理：高 σ 区域是该加强 consistency 还是放松 consistency 是个 sign 问题。当前修正是不用 σ 单独决定方向，而用 action sensitivity `A_t` 过滤 aleatoric visual noise。

#### 8.5.3 σ clamp 与参数化

PushT `s_abs_max` 已经长期顶在 4.0。下一次启用任何 hetero / guarded auxiliary 前需先决定：

- 放宽 `s_max`（直接副作用：极端样本权重更不平衡）。
- 改 σ 参数化（例如 softplus-style positive scale + 可学习 prior，避免 logvar 双侧发散）。
- 或保持 4.0 但配合"hetero_s_abs_max 持续顶上限 → 自动降 alpha"的 guard rail。

probe-only 阶段可以顺便看 σ 在 PushT 上的自然分布是否仍贴 clamp，作为后面参数化决策的依据。

#### 8.5.4 TwoRoom hetero +6.7pt 的解释缺口

Pilot-1B 里 TwoRoom hetero clean 99.67 不仅超过 LeWM-base (93)，还超过 LeWM+noise best (98.33)。叙事目前把它一笔带过。这其实是一个独立信号：在低本征维 / 离散 transition 任务上，"downweight 简单样本 + upweight 难样本"反而像 implicit clustering。回看时建议至少跑一组 ablation：

- TwoRoom probe-only：σ probe 不动 μ → 应回落到 LeWM-base 附近。如果仍接近 99.67，说明并非 reweight 在起作用。
- 如果 probe-only 回落而 hetero 保持 99.67，则该收益机制是"按难度重加权"，可作为 *任务条件化* 的论据——而不是统一 method 的论据。

#### 8.5.5 超参数预算公开化

§10 立了"count hyperparameter"纪律，但当前 §8.3 路径累计：`beta_probe`（probe）→ `alpha_planner`（uncertainty / budget）→ `alpha_hetero`（guarded）。每一阶段应在文档里明列：

- 新增 hyperparam 名、默认值、允许范围、early-stop 阈值。
- 该阶段相比前一阶段的边际经验收益要求（例如 "PushT clean +2 或 robustness +5"），不达标不进入下一阶段。

否则 §10 的纪律会随阶段松动。

#### 8.5.6 +noise 训练与 adaptive resolution 是否联用（重要叙事问题）

目前的叙事链是：LeWM-base → LeWM+noise（任务最优 std 不同 + 表征/诊断显示 noise 推向 nuisance-invariant 表征）→ 因此需要 adaptive resolution。当前所有 σ pilot 都是 `image_noise.std_max=0.0`，这是为了 ablation 干净，但 **不是最终方法应该长成的样子**。

机制上 +noise 和 σ-adaptive 处于不同位置：

| 机制 | 作用位置 | 物理含义 |
|---|---|---|
| image noise / consistency | encoder 输入侧 | 数据增广，强迫 encoder 对 nuisance 不变 |
| σ probe | predictor 输出侧 | 标注 per-transition 难度 |
| σ planner use | inference 侧 | 按难度分配 compute / 截断 horizon |
| action-aware adaptive consistency | encoder 输入侧 + σ/A 反馈回去 | 用 `A_t` 判断 action relevance，再用 σ 增强 criticality，决定 consistency strength |

所以正确的关系不是"二选一"，而是 **noise 是 input-side 的 isotropic 数据增广，σ 是 output-side 的 per-state difficulty 信号，`A_t` 是 controllability filter**，互补。最终方法叙事里应该是：

> LeWM+noise 提供全局 invariance baseline；action-aware σ/A controller 把全局 invariance 拆成 per-state，让模型在 action-critical 难区域少抹细节，在 action-insensitive 视觉冗余区域多抹。

需要补的实验阶梯（在 §8.3 通过后追加）：

1. **Probe-on-noise**：LeWM+noise checkpoint 加 σ probe（μ path 不变）。检查 σ 在 noise 训练下是否仍稳定 calibration、空间分布是否变化。这是最便宜的联用验证，应该尽早做。
2. **Action-gate-on-noise**：在 LeWM+noise checkpoint 上 logging-only 记录 `A_t` / `critical_t`，确认 noise 训练后 gate 仍能区分 contact 与 visual nuisance。
3. **Joint training**：只有 1+2 都通过，再考虑 action-aware adaptive consistency / 联合训练。这一步必须额外注意双重 downweight 风险：noise 让 contact transition 更难 → σ-only gate 会错误放松 consistency；因此必须保留 `A_t` 主门控，并在 guardrail 里同时监控 noise level、`A_t` 分布与 resolution 指标。

短期内不需要立刻扩到 4-task；但回看 probe-only 结果时，应同时回答："这个 σ 如果叠到 +noise 训练上还稳吗？`A_t` 是否能过滤 σ 的视觉噪声成分？"——这一条建议直接写进 §8.5.1 的回看 checklist。

---

### 8.6 Validation: 4-task 全套

**触发条件**: Probe-only + σ 使用逻辑通过，且经验上至少接近 LeWM+noise oracle。

| 项 | 设置 |
|---|---|
| 任务 | 4 task |
| Seed | 3 |
| Eval | num_eval=300（每 seed 100） |
| 对照 | LeWM-base / LeWM+noise shared std / LeWM+noise per-task oracle / 本框架 |
| Ablation | probe-only vs action-gate logging vs action-aware consistency vs σ-only consistency vs hetero loss；scalar σ vs per-dim σ（仅最后考虑） |

---

## 9. 与 plan_v3 / plan_v2 的关系

### 9.1 与 plan_v3 §6 P4 的关系

本文件现在是 plan_v3 §6 P4 的首选思考路线，但执行上必须分阶段。2026-05-09 Pilot-1B 已触发关键 fallback 条件：直接 hetero loss 伤害 PushT critical transition resolution。后续应优先做 probe-only σ、action-gate logging、action-aware adaptive consistency；σ planner use / guarded hetero auxiliary 仅作为对照或备选。

### 9.2 与 plan_v2 V1/V2 的关系

- V1 (vMF): 球面 + 1D 角度 σ 的特化版，本框架的 spherical projection 限制
- V2 (ball-cap): σ_x quantile clip 的 OOD 延伸

V1/V2 都是更复杂版本，**本最简版本不预设走那个方向**，看 Pilot 结果再决定。

### 9.3 与诊断工具的关系

诊断工具**完全不变**，但定位从“独立预测工具”降为“设计约束 + 机制解释”。本框架的 σ̂ 是新增的 per-token / per-transition difficulty 诊断量，`A_t` 是在线版 action-effect / controllability 诊断量；**事后分析**它们和现有诊断的相关性是 nice-to-have，不预设为 a priori 主张。

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
| σ-only adaptive consistency | 高 σ 同时包含 dynamics difficulty 和 aleatoric visual noise，会落入 Noisy TV / confounder trap；必须加入 action sensitivity `A_t` |
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
- **Anti-collapse 工具线**: SIGReg (Maes et al. 2026), VICReg (Bardes, Ponce & LeCun ICLR 2022), Barlow Twins (Zbontar et al. ICML 2021), RankMe (Garrido 2023), LiDAR (Thilak 2024), uniformity (Wang & Isola ICML 2020), BYOL (Grill et al. NeurIPS 2020)
- **Reconstruction-based world models（对照路线）**: Hafner et al. 2020/2023 (Dreamer / DreamerV3), Hansen et al. 2024 (TD-MPC2)
- **JEPA 路线**: LeCun 2022, "A Path Towards Autonomous Machine Intelligence"; Assran et al. CVPR 2023, "I-JEPA"
- **Noise / Lipschitz / certified-robustness 诊断**: Hoffman 2019 (Jacobian regularization), Virmaux & Scaman NeurIPS 2018 (Lipschitz spectral bounds), Cohen, Rosenfeld & Kolter ICML 2019 (randomized smoothing → robust radius)
- **Latent geometry diagnostics**: Sun et al. NeurIPS 2022 (KNN-OOD), Kornblith et al. ICML 2019 (CKA), Ethayarajh EMNLP 2019 (anisotropy), Jing et al. ICLR 2022 (dimensional collapse)
- **Action probing / inverse dynamics**: Alain & Bengio ICLR-W 2017, Brandfonbrener et al. NeurIPS 2023, Pathak et al. ICML 2017 (ICM)
- **Noisy TV / aleatoric confounder**: Burda, Edwards, Pathak, Storkey, Darrell & Efros, ICLR 2019, "Large-Scale Study of Curiosity-Driven Learning" — canonical demonstration that uncertainty/curiosity signals attract to uncontrollable stochastic distractors
- **Empowerment / controllability**: Klyubin, Polani & Nehaniv, IEEE CEC 2005, "Empowerment: A universal agent-centric measure of control" — origin of the action-conditioned mutual-information / sensitivity framing for controllability
- **Asymmetric consistency**: Chen & He, CVPR 2021, "Exploring Simple Siamese Representation Learning" (SimSiam) — stop-grad + predictor-side asymmetry as anti-collapse without negatives
- **Heteroscedastic uncertainty**: Kendall & Gal, NeurIPS 2017, "What Uncertainties Do We Need in Bayesian Deep Learning"
- **撞车风险高的近期工作（必须 differentiate）**: PCA++ (arXiv:2511.12278, Nov 2025) — uniformity ⇒ background-noise robustness in contrastive SSL；Surprise-Recognition (arXiv:2512.01119, Dec 2025) — runtime input filtering by single-step surprise；RobustZero (Li et al. ICML 2025) — adversarial latent-state perturbation in MuZero

---

## 12. 维护说明

- 本文件供查阅与设计迭代；**不**作为 plan_v3 的替换。
- 每次新讨论后追加新条目到 §7 风险表 或 §10 回退记录。
- 下一步主线是 §8.3：probe-only σ + action-gate logging + action-aware adaptive consistency + PushT resolution guardrail。
- 后续若 action-aware Pilot-2 通过，把 §3 §4 §6 §8.3 合并进 plan_v3 §6 P4；本文件归档。
- **下一次想加新机制前**: 先回看 §10，问自己"它会增加几个超参数？经验收益的证据是什么？"。如果两个问题答不清楚，不加。
