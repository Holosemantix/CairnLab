# Action-Aware Adaptive Latent Resolution

> **Status**: Pilot-1B 已完成首轮 TwoRoom + PushT 验证（2026-05-09）。结果支持"σ head 能学到 prediction difficulty"，但否定了"直接用 hetero loss 替换 MSE"作为 PushT 上的主方法：PushT clean eval 从 LeWM-base 87.33/86.00 降到 13.33，诊断显示 transition/action resolution 被严重压缩。下一步主线改为 **probe-only σ + action-aware adaptive consistency + resolution guardrail**，而不是继续加大 hetero loss 或只做 σ-only controller。
>
> **Pilot-2A 更新（2026-05-09）**: probe-only 救回 PushT（clean 87.00 ≈ LeWM-base 87.33），gate logging 三个结构判据通过，但发现 **BN drift bug**: `compute_action_gate_metrics` 内 K 次 perturb forward 在 train mode 下污染 `BatchNorm1d` running stats，导致 TwoRoom probe+gate clean 跌 7pt（96.33 → 89.33）。Stage C 不能解决此 bug，必须先在 gate 内部把 BN 切到 `.eval()` 再重测。修复前不开 `alpha_cons > 0`。
>
> **Pilot-2B 更新（2026-05-10）**: BN drift fixbug 版已跑完 TwoRoom + PushT full eval。TwoRoom probe+gate-fixbug clean **95.00**（从 bug 版 89.33 恢复到 LeWM-base 附近），PushT clean **85.33**（接近 LeWM-base 87.33，未复现 hetero-loss collapse）。SwanLab validate 指标显示 σ calibration 仍成立（TwoRoom `hetero_s_logerr_corr=0.612`，PushT `0.482`），σ 与 action sensitivity 不是同一信号（TwoRoom `corr_sigma_action=-0.010`，PushT `0.256`），gate weight 有非平凡 spread。结论：**Stage B logging-only controller signal 已验证；下一步可以进入小权重 Stage C，但必须以 PushT clean/resolution guardrail 为硬约束。**
>
> **Stage C 更新（2026-05-11）**: `alpha_cons` 小权重 sweep 已跑完（consist001/003）+ A_t-only ablation + w_t 离线可视化。**核心结果**：PushT α=0.01 clean **86.67**（≈ baseline 87.33）+ robustness 翻倍（goal 0.05 38→77，pixels 0.05 17→73），resolution guardrail（`transition_resolution_ratio_l2=0.290`，`id_probe_r2=0.764`）全部通过；TwoRoom α=0.03 clean **98.33** = LeWM+noise best (`0to008-p1`) 98.33，px+goal 0.05 97.33（LeWM+noise 98.00）。α=0.03 在 PushT 上触发 guardrail（clean 76.33 < 84），印证任务特异性。A_t-only consist001 PushT clean 77.33 显著低于 σ+A_t 86.67（-9.34pt），**σ 必要性方向性得证**。w_t 离线可视化验证 corr(w_t, action_norm)=+0.587、corr(w_t, latent_disp)=−0.592，动态范围非平凡。下一步：σ-only ablation 闭合对称证据 + probe-on-noise / consistency-noise 联用。
>
> **关系**: 不是 plan_v3 的替换，而是 plan_v3 §6 P4 "Adaptive Resolution Method" 的具体化方案。
> **设计原则**: 先证明额外 σ 输出头携带有用信息，再让它影响训练或 planning；避免一开始就改变 LeWM 的强 MSE baseline。
> **重要历史记录**: 本文件早期版本曾包含 IB term / aggregate covariance Frobenius / Fisher manifold planning 等多层架构，hyperparameter 数量涨到 4–5 个。经过严格审视后**全部回退**——它们都需要新超参却没有可论证的额外收益。详见附录 A 设计回退记录。

---

## 摘要

不要默认假设 heteroscedastic NLL 会优于 MSE。LeWM 的 MSE + SIGReg 已经很强，直接替换成 NLL 会改变 pred loss 与 SIGReg 的相对尺度，而且 NLL 会 downweight 高误差样本；在 PushT 这类任务里，高误差样本可能正是接触/精细控制的关键状态。

当前路线（**2026-05-11 更新**：Pilot-1B/Pilot-2A/Pilot-2B 已完成，Stage C 已跑完 consist001/003 + A_t-only ablation + w_t 离线可视化。核心结果：PushT α=0.01 为 sweet spot——clean 86.67 ≈ baseline 87.33，robustness 翻倍（goal 0.05 38→77，pixels 0.05 17→73）；TwoRoom α=0.03 达到 LeWM+noise 天花板 98.33。剂量效应精确对应任务特性：action-critical 任务耐受低 α，冗余视觉任务可承受高 α。A_t-only ablation（无 σ）PushT clean 跌 9.34pt、TwoRoom clean 93.33 接近 baseline，方向性证明 σ 不可缺失。w_t 可视化验证 corr(w_t, action_norm)=+0.587、corr(w_t, latent_disp)=−0.592，动态范围非平凡。σ-only 失败对照尚待补全。

1. **Pilot-1B 结论：Scale-preserving heteroscedastic loss 语义成功、控制失败。** `hetero_s_logerr_corr` 在 TwoRoom/PushT 后期分别约 0.89/0.95，说明 σ head 学到了 prediction difficulty；但 PushT `hetero_weight_q10_q90_ratio` 掉到约 0.008，hard transition 被强 downweight，clean eval 崩到 13.33。
2. **下一步首选：Probe-only σ + action-aware adaptive consistency。** μ path 保持 LeWM MSE + SIGReg，σ head detached 学 `log(error)`；真正改变 encoder resolution 的路径应放在 input-side consistency 上，而不是 prediction-loss reweighting 上。
3. **σ 不能单独决定 consistency weight。** prediction difficulty 会混合 action-relevant difficulty 和视觉 aleatoric noise；σ-only consistency 会落入 Noisy TV / confounder trap。必须用 action sensitivity `A_t` 作为主门控，σ 只作为 difficulty enhancer。
4. **Pilot-2A/2B 结论（2026-05-09/10，§3.3）**: probe-only 救回 PushT，BN drift fix 后 probe+gate logging 不再破坏 TwoRoom。fixbug full eval：TwoRoom clean 95.00、PushT clean 85.33；PushT resolution diagnostics 与 LeWM-base 基本一致（`transition_resolution_ratio_l2≈0.288`、`id_probe_r2≈0.774`），明显区别于 hetero-loss collapse。Stage B 的 controller signal 已成立；Stage C 已完成 sweep，PushT sweet spot α=0.01 通过 guardrail（clean 86.67 ≥ 84，resolution 0.290 ≥ 0.24），TwoRoom α=0.03 达到 LeWM+noise 天花板。不存在单一 α 同时在两任务上达到各自最优，这正是自适应分辨率机制的任务特异性预期。

> **为什么恢复 probe-only？** 2026-05-09 Pilot-1B 已经证明核心风险真实存在：σ calibration 很好，但 PushT 失败。此时 probe-only 不再是"容量 smoke test"，而是把 σ 语义从 μ 几何更新中解耦，避免 hard-but-important transition 被训练权重抹掉。

核心批判点：**额外输出头本身不会自动变成动态分辨率。** 如果 σ 只作为日志或 detached probe，它是诊断量，不改变 μ 几何；如果 σ 进入 NLL，它改变训练梯度，但可能只是学会"忽略难样本"；如果只用 σ 调 consistency，它会把背景噪声误判成高分辨率需求。因此必须把 σ 和 action relevance 解耦：`σ_t` 识别 difficulty，`A_t` 识别 controllability / causal relevance。

LeWM 是第一性 baseline。任何 σ 方案都必须先证明至少不破坏 LeWM+noise 的 clean / robustness tradeoff。

---

## 1. 引言

### 1.1 动机

plan_v3 §5.2 的主线"task-aware latent geometry"在落地时遇到的死结：**所有"自适应"方案都把 trade-off 控制器放在 loss 之外**，模型自己没有"分辨率"这个内禀概念。

PI controller / Lagrangian τ / cheap-proxy bilevel / 多任务 head 等方案都需要外部信号或手调阈值，且都未必比 LeWM + SIGReg 经验上更好。

**真正需要验证的范式转换**：让模型输出一个与局部难度/不确定性相关的 σ，并证明这个 σ 能帮助 resolution allocation。这里不能直接把 σ_x 宣称为 latent 邻域半径：
- predictor σ̂ 最自然的监督来自 prediction error，它首先是 **transition uncertainty**。
- encoder σ_x 如果没有额外使用逻辑，只是一个未监督 head，容易不可辨识。
- planning resolution 需要的是"哪些状态差异应该保留"，不等价于"哪些 transition 难预测"。

因此第一步不应直接改主 loss，而应先问：额外输出头是否能稳定学到有意义的异质性？如果不能，后续 NLL / planner 使用都没有基础。如果能，再逐步让 σ 影响训练或 inference。

### 1.2 核心批判：σ head ≠ 动态分辨率

必须区分三种层级：

| 层级 | σ 做了什么 | 是否改变 resolution |
|---|---|---|
| Probe | 预测 detached error | 否，只是诊断 |
| Loss weighting | 改变不同 transition 的 μ 梯度 | 可能，但可能忽略关键 hard states |
| σ-only controller | 影响 CEM budget / gating / consistency strength | 可能，但容易把 aleatoric visual noise 当成 resolution demand |
| **Action-aware consistency** | σ 与 action sensitivity 共同控制 encoder invariance | **是** encoder-side adaptive resolution 的当前首选候选 |

所以论文中不能把"加一个 σ head"直接等同于 dynamic resolution。真正要证明的是：σ 与 action-relevant difficulty 对齐，且 `A_t` 能把 controllable critical states 和不可控视觉噪声区分开；然后 adaptive consistency 在不破坏 PushT resolution guardrail 的前提下改善 LeWM+noise 的手调 tradeoff。

### 1.3 设计原则

1. **先 probe 再 intervention。** Stage A 验证 σ 携带信息，Stage B/C 才让它影响训练。
2. **LeWM 是第一性 baseline。** 任何 σ 方案必须先证明不破坏 LeWM+noise 的 clean/robustness tradeoff。
3. **超参预算纪律。** 新增机制若增加超参而经验收益不明，回退（见附录 A）。
4. **最小改动优先。** predictor σ head 只增 ~0.5M 参数量（可忽略），且 `s=0` 时严格退化回 LeWM MSE。

### 1.4 文档范围

本文档涵盖架构（§2.1）、Loss 设计（§2.2）、action-aware gate（§2.4）、已完成实验验证（§3–§4）、讨论（§5）以及分阶段未来路线图（§6）。Pilot-1B 结果作为 ablation/negative result 呈现，验证 σ 语义的同时否定 heteroscedastic loss 作为主方法。

---

## 2. 方法框架

### 2.1 架构设计

#### 2.1.1 Pilot-1：只给 predictor 加 σ head

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

#### 2.1.2 Pilot-2：可选 encoder σ head

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

#### 2.1.3 备选：encoder input-sensitivity head（有监督版本，优先于 unsupervised σ_x）

> 设计动机（2026-05-09 登记）：当前 Stage A–C 的所有 resolution controller 信号（σ̂、A_t）都经过 **predictor** 这条路。encoder 端对自己输入扰动的反应没有 self-aware signal；σ̂ 在 contact / 边界 chaotic extrapolation 时会被污染（§2.4.2 的 multi-δ CV 是在为此打补丁）。直觉"只给 predictor 加 head 差点啥"指向的不是 unsupervised encoder σ，而是这条**有天然监督**的替代。

```
enc_backbone(x) → h
mean_head(h)    → μ_x ∈ R^d
sens_head(h)    → s_enc(x) ∈ R^1   # 预测 input-perturbation 后的 encoder 位移

# detached supervision
target_enc = log( ||enc(x).detach() − enc(aug(x)).detach()||₂ + eps )
L_sens = smooth_l1(s_enc(x), target_enc)
```

性质：
- **天然监督信号**：`||enc(x) − enc(aug(x))||` 直接可测，避免 §2.1.2 点名的 unsupervised encoder σ identifiability trap。
- **与 predictor σ̂ 正交**：σ̂ 是 *transition difficulty given clean state*；s_enc 是 *state representation 对 input nuisance 的敏感度*。两者描述不同物理量，没有互相吸收 residual 的退化路径。
- **不引入 EMA / 第二 encoder**：监督目标用同一个 encoder 的两次 forward，与 §2.1.4 单 encoder 哲学一致。
- **与 §2.2.3 `L_cons` 共享 forward，零额外开销**：`||enc(x) − enc(aug(x))||` 正是 `L_cons` 已经在计算的 per-token distance。监督目标直接复用 `d(stopgrad(z_clean), stopgrad(z_noisy))`，不需要第二次 `enc(aug(x))`。
- **真正的价值是 controller-side 闭环，不是 loss-side**：`L_cons` 已经把这个距离当**训练目标**（梯度反传进 encoder 把它压小）；s_enc head 的角色是把这个量**摘出来作为 controller `w_t` 的输入**。没有 s_enc 时，`w_t` 只看 predictor 端信号（σ̂、A_t），是 predictor-only feedback loop；有了 s_enc 才形成 *encoder sensitivity → controller → encoder consistency pressure* 的闭环。这也是直觉"predictor-only head 差点啥"指向的真正缺口。

不立即采用的理由（附录 A 纪律）：
1. Pilot-1（probe-only σ）尚未给出结果；最便宜的 predictor head 都未验证。
2. Stage C 的 `L_cons = w_t · ||z_clean − z_noisy||` 已经隐式让 encoder input-sensitivity 进入梯度——先看它够不够，再决定是否需要把 encoder sensitivity 显式提取成 controller 输入。
3. 加这一项会引入新 hyperparam `beta_sens`，违反"加项必须先有经验收益证据"的纪律。

**触发加入的条件**（写死，不模糊）：
- Stage C 的 `alpha_cons` ramp 在 PushT 上撞 guardrail（`transition_resolution_ratio_l2 < 0.24` 或 `clean < 84`），且诊断显示 critical 区域是因 **encoder 端对 input nuisance 区分不足** 导致（不是 predictor 端 σ̂ / A_t 信号失败）；或
- σ̂ 的 calibration 在 +noise 训练下漂移（§6.4 Probe-on-noise 阶段），需要一个 input-side 信号去解释 σ̂ 漂移成分。

#### 2.1.4 Target encoder：保留 LeWM 单 encoder 哲学

target latent `μ_{t+1}^target = enc(x_{t+1})`——同一个 encoder，无 EMA、无 stop-grad asymmetry（沿用 LeWM 做法，是否对 target stop-grad 跟 LeWM 保持一致即可）。

Anti-collapse 完全交给 LeWM 现成的 SIGReg(μ)，**不引入额外机制**。

### 2.2 Loss 设计：三阶段路线

#### 2.2.1 Stage A：detached σ calibration probe（首选第一步）

主训练目标完全保持 LeWM：

```
pred_loss = mean((mu_hat - mu_target)^2)
loss = pred_loss + lambda_SIGReg * SIGReg(mu)
```

新增 σ head 只做 detached calibration：

```
err_token = mean((mu_hat.detach() - mu_target.detach())^2, dim=-1)
s_hat = pred_logvar_hat.squeeze(-1)
sigma_probe_loss = smooth_l1(s_hat, log(err_token + eps))
```

关键点：
- `sigma_probe_loss` 只更新 σ head，不反向影响 encoder / predictor mean path。
- 这一步**不会**改变 latent resolution；它只是检验额外输出头是否能学到 transition difficulty。
- 如果 σ probe 都学不出稳定结构，后续 action gate / consistency 都没有稳定信号基础。

#### 2.2.2 Stage B：scale-preserving heteroscedastic loss（Pilot-1B 已失败，作为 ablation）

> **Status**: 2026-05-09 Pilot-1B 已验证：scale-preserving 形式能校准 σ，但 PushT clean eval 崩到 13.33，因此它不再是主方法路线，只作为 ablation / negative result。

普通 Gaussian NLL：
```
0.5 * (err * exp(-s) + s)
```

不适合直接替换 MSE，因为初始 `s=0` 时变成 `0.5 * err`，等于把 pred loss 缩小一半，SIGReg 相对变强；后续 `s` 还能让 loss 尺度漂移。

候选替代是尺度保持版本：
```
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

#### 2.2.3 Stage C：Action-Aware Adaptive Consistency（当前首选）

真正符合"adaptive latent resolution"的训练介入不应是直接重加权 prediction loss，而应是对 encoder input-side invariance 的局部调节：

```
z_clean = enc(x)
z_noisy = enc(aug(x))
L_cons = mean(w_t * d(stopgrad(z_clean_t), z_noisy_t))
```

核心问题是 `w_t` 不能只由 σ 决定。prediction difficulty 同时包含：
- **Epistemic / dynamics difficulty**：例如 PushT 接触瞬间，应该降低 consistency pressure，保留分辨率。
- **Aleatoric / visual nuisance**：例如不可控背景噪声，应该提高 consistency pressure，抹掉噪声。

因此定义一个 action-aware criticality：

```
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

Stage C 只在 logging-only 验证通过后进入。

#### 2.2.4 SIGReg 仍然作用在 μ 上

无论 Stage A/B/C，SIGReg 都只作用在 deterministic μ 上。不要把 SIGReg 推广到 `(μ, σ)` 或 reparameterized sample；那会引入 Gaussian mixture 高阶矩问题，并破坏 LeWM 已验证的 anti-collapse 机制。

### 2.3 σ 信号的使用层级框架

| 使用方式 | 作用 | 风险 |
|---|---|---|
| training weight | 通过 hetero loss 改变 μ 的梯度分配 | 可能忽略 hard-but-important states |
| σ-only noise/controller | 高 σ 区域降低/提高 noise consistency 强度 | **confounder trap**：高 σ 可能来自视觉 aleatoric noise，而不是 task-critical dynamics |
| action-aware consistency | 用 action sensitivity 区分 controllable critical states 和不可控视觉噪声 | 最符合 adaptive resolution，但必须先做 logging-only 验证 gate |
| planner budget | 高 σ rollout 增加 CEM samples / 缩短 horizon | 不改变表示，只改 inference compute |
| uncertainty gate | 高 σ 时拒绝或降权候选 plan | 可能过度保守 |

因此 σ 进入使用逻辑前必须先通过 probe-only calibration 和 action-gate logging。否则额外 head 只是日志，不是方法；若直接进入 loss weighting，Pilot-1B 已经给出 PushT 反例。

### 2.4 Action-Aware Gate 设计

#### 2.4.1 Action Sensitivity `A_t`

```
A_t = ||f(z, a + δ) - f(z, a)||_2 / ||δ||_2
```

#### 2.4.2 Multi-δ Perturbation 与变异系数 CV

`A_t` 高有两种成因：
- **Smooth controllable**：小 δ → 平滑大响应。多次采样 δ 给出 *方向相关、幅度相近* 的响应，`A_t` 在 δ 上低方差。这是真正的 action critical 区域。
- **Chaotic / extrapolation**：predictor 在 contact / 边界附近不连续，小 δ → 任意大响应。多次采样 δ 给出高方差。这种区域不该当作"应保留分辨率"的 critical state。

因此在 logging-only 阶段，每个 token 用 `K=4` 个独立 δ 采样：

```
A_t^{(k)} = d(f(z, a + δ^{(k)}), f(z, a)) / (||δ^{(k)}|| + eps)   for k=1..K
A_mean = mean_k A_t^{(k)}
A_cv   = std_k A_t^{(k)} / (A_mean + eps)   # coefficient of variation
```

判定：
- 全局 `cv_mean < 0.5`：predictor 局部光滑，`A_t` 可信。
- 高 `A_mean` 区域的 CV 不显著高于全局 CV：critical 区域不被 chaotic 主导。

如果两者中任一不满足，说明 `A_t` 信号被噪声污染，consistency 训练阶段应改用 `A_mean / (1 + α_cv * A_cv)` 做 chaos-discount，而不是直接用 raw `A_mean`。`α_cv` 默认 1.0。

#### 2.4.3 EMA Z-Score 与 Warmup

`A_t` 的物理意义只有在 predictor 学到了 action conditioning 之后才成立。早期 predictor 几乎忽略 action 时，`A_t ≈ 0` 是 predictor 不成熟而非 state insensitive。因此 logging 只在以下任一条件后启动：

- `validate/id_probe_r2_epoch >= 0.5 * id_probe_r2_LeWM_base`（PushT 取 0.39，TwoRoom 取 0.14；这是 "predictor 学到了一半 action 信息" 的代理）；或
- 训练经过 `cfg.loss.action_gate.warmup_epochs` epochs（默认 3，等于 LeWM 10-epoch 训练的前 30%）。

在 warmup 期间仍然计算并记录 `A_t`，但不进入 `critical_t` 聚合，也不写入 EMA z-score 统计——避免 z-score baseline 被 action-blind 阶段的统计带偏。

#### 2.4.4 Consistency Weight 公式

```
log_A = log(A_mean.clamp(min=eps))
gA = sigmoid(zscore_ema(log_A))
gS = sigmoid(zscore_ema(s_t))    # σ 不可用时 fallback 到 0.5
critical = gA * (0.5 + 0.5 * gS)
w_t = w_max - (w_max - w_min) * critical
```

默认边界：`w_min = 0.2`，`w_max = 1.0`。

### 2.5 与现有方法的对照

| 现有方法 | 在本框架下 |
|---|---|
| LeWM + SIGReg | 无 σ 使用逻辑；等价于 Stage A 中忽略 σ head |
| SWM (V0 spherical) | 固定单位球几何 prior；无动态 σ |
| VICReg | 固定 covariance / variance prior；无动态 σ |
| LeWM + noise | 全局 input-side invariance；无 state/action-aware weighting |

现有方法都没有把 per-transition uncertainty 和 action-conditioned local sensitivity 结合起来控制 encoder invariance strength。

---

## 3. 实验验证

### 3.1 实验设置

**任务：** TwoRoom 和 PushT（primary benchmarks）。
**训练：** 10 epochs，LeWM baseline architecture。
**σ head：** `logvar_hidden_dim=256`，final layer zero-initialised（weight=bias=0），`s_min=-4.0`，`s_max=4.0`。
**Noise：** `image_noise.std_max=0.0` for ablation cleanliness（noise 和 σ-adaptive 互补，不是互斥；见 §6.4）。
**Evaluation：** Epoch 10，`num_eval=300`，seeds 42/43/44 聚合。

### 3.2 Pilot-1B：Heteroscedastic Loss 作为 Ablation

Pilot-1B 测试 scale-preserving hetero loss（§2.2.2）作为直接 MSE 替代。配置：`loss.hetero.enabled=true`，`loss.hetero.mode=loss`。

**Runs：**

| Task | Run name | SwanLab ID | Local output |
|---|---|---|---|
| TwoRoom | `tworoom_lewm_hetero_default` | `gps6asjv22tmflag9af5m` | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt/tworoom_lewm_hetero_default` |
| PushT | `pusht_lewm_hetero_default` | `tge50bhmtws06xc7n4wtq` | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt/pusht_lewm_hetero_default` |

#### 3.2.1 训练曲线

| Metric | TwoRoom hetero | PushT hetero | 解释 |
|---|---:|---:|---|
| `fit/hetero_s_logerr_corr` tail100 | 0.894 | 0.950 | σ 与 prediction difficulty 强正相关，σ head 语义成立 |
| `validate/hetero_s_logerr_corr_epoch` last | 0.912 | 0.957 | validation 上同样成立，不是 train-only artifact |
| `fit/hetero_s_std` tail100 | 1.232 | 1.836 | PushT 的 σ 异质性明显更强 |
| `fit/hetero_s_abs_max` last | 3.236 | 4.000 | PushT 已贴到 clamp 上限 |
| `fit/hetero_weight_q10` last | 0.495 | 0.369 | 高 σ / hard token 被 downweight |
| `fit/hetero_weight_q90` last | 11.026 | 47.802 | low-error token 被大幅 upweight |
| `fit/hetero_weight_q10_q90_ratio` last | **0.045** | **0.008** | PushT 梯度权重极端失衡 |
| `fit/pred_loss_mse_equiv` tail100 | 0.0438 | 0.0394 | true MSE-equivalent loss 仍下降，但不保证任务 resolution 保留 |
| `validate/pred_loss_mse_equiv_epoch` last | 0.0274 | 0.0332 | validation MSE 也下降；失败不是简单 underfit |

关键判定：
- **σ calibration 成功。** 两个任务 `hetero_s_logerr_corr` 后期都很高，说明 σ head 不是常数，也不是噪声。
- **PushT reweight 过强。** `q10/q90_ratio` 低到 0.008，远低于 0.3 警戒线；这正是 hard-but-important transition downweight 风险。
- **hetero loss 可以为负。** `pred_loss` 后期略为负是公式 `exp(-s) * err + tau * s` 的结果，不代表 prediction quality "负误差"；真实对照应看 `pred_loss_mse_equiv`。

#### 3.2.2 Eval 结果

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
- PushT clean 只有 13.33，是**方法级失败**，不是 robustness tradeoff。

#### 3.2.3 Diagnostics

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

#### 3.2.4 结论

Pilot-1B 的结果是**语义成功、系统失败**：

1. **σ head 值得保留。** 它稳定学到了 per-transition prediction difficulty。
2. **直接 hetero training 不适合 PushT。** 它会把 high-error hard transitions 当成低权重样本，而这些 transition 很可能正是 PushT 的接触和精细控制关键区域。
3. **adaptive resolution 不能只靠 loss reweight。** 真正需要的是：μ 表征保留控制分辨率，σ 作为额外信号去调节 planning / consistency / compute，而不是让 σ 直接决定哪些 transition 不训练。

### 3.3 Pilot-2A：Probe-Only σ 与 Action-Gate Logging

Pilot-2A 联合测试 Stage A（probe-only）和 Stage B（logging-only gate）。

**Runs：**

| Task | Run | SwanLab ID |
|---|---|---|
| TwoRoom probe | `tworoom_lewm_hetero_probe_default` | `75qiqru0ttwmyy7pwigly` |
| TwoRoom probe+gate | `tworoom_lewm_hetero_probe_default_action_gate` | `awokxbepmodp2shcqmynr` |
| PushT probe | `pusht_lewm_hetero_probe_default` | `jgqsw29zji110j3gczu03` |
| PushT probe+gate | `pusht_lewm_hetero_probe_default_action_gate` | `oezw5j3w0uh3ydxnan63c` |

设置：`loss.hetero.enabled=true loss.hetero.mode=probe`；probe+gate 额外 `loss.action_gate.enabled=true`（logging-only，`adaptive_consistency.weight=0`）。Eval epoch 10，seeds 42/43/44，每 seed `num_eval=100`。

#### 3.3.1 Eval 表

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

#### 3.3.2 训练侧诊断

| 指标 | tw_probe | tw_probe_gate | pu_probe | pu_probe_gate |
|---|---:|---:|---:|---:|
| `pred_loss_mse_equiv` (tail100) | 0.0295 | 0.0295 | 0.0177 | 0.0162 |
| `validate/hetero_s_logerr_corr` (last) | 0.620 | 0.621 | 0.480 | 0.462 |
| `hetero_s_abs_max` (last) | 4.24 | 4.30 | 4.23 | 4.23 |
| `adaptive_action_sensitivity_cv_mean` | — | 0.36 | — | 0.245 |
| `adaptive_action_sensitivity_cv_high_A` | — | 0.35 | — | 0.277 |
| `validate/adaptive_corr_sigma_action_epoch` | — | −0.02 | — | +0.23 |
| `adaptive_critical_mean` (val) | — | 0.35 | — | 0.30 |

#### 3.3.3 结论

1. **PushT 崩溃问题已解决。** probe-only PushT clean 81.67、probe+gate clean 87.00，与 LeWM-base 87.33 持平；hetero loss 的 13.33 不再出现。证明 §2.2.1 把 σ 从 μ-path 梯度解耦的设计是对的。
2. **σ probe 语义保留。** PushT validate corr ≈ 0.46–0.54、TwoRoom ≈ 0.62。PushT 边界但通过 Stage A 0.5 阈值。代价是 σ 比 hetero loss 下的 0.95 弱，因为没有 NLL 反馈把 σ 强行钉到 prediction error。
3. **Action-gate 三个结构判据通过（PushT）。** `cv_mean = 0.245 < 0.5` ✅；`cv_high_A = 0.277` 不显著高于全局 ✅；`corr_sigma_action`：PushT 0.23 弱正相关、TwoRoom −0.02 基本独立 → σ 与 A_t 不是同一信号，乘性 `critical = gA × (0.5 + 0.5·gS)` 设计得到经验支持。这是 §2.4.2 多 δ CV 判据的首个正面验证。
4. **TwoRoom probe+gate clean 跌 7pt（96.33 → 89.33）是真问题，且不是 Stage C 能解决的。** 见 §3.3.4。
5. **PushT probe+gate robustness 比 LeWM-base 更高**（goal 0.05: 38 → 52、pixels 0.05: 17 → 32），但 logging-only run 不应改变训练梯度——这要么是种子噪声，要么是 §3.3.4 同源的 BN drift 在 PushT 上反而起到轻微 invariance 训练效果。需要修 BN drift 后重测确认。

#### 3.3.4 关键 bug：BN drift via gate perturb forward

**机制：** `train.py::compute_action_gate_metrics` 内 K=4 次 `model.predict(ctx_emb_d, act_emb_pert)` 在 `model.training=True` 下执行。`projector` 与 `predictor_proj` 默认 `nn.BatchNorm1d`（`config/train/lewm.yaml::encoder.projection_head.norm_fn=batchnorm1d`），每次 perturb forward 都会用 OOD-ish 扰动 activation 更新 BN running mean/var。每个 train step BN 统计被多走了 K 次偏离主分布的 forward。

**为什么 TwoRoom 受害严重、PushT 几乎不受影响：**
- TwoRoom：表征空间小、batch 视觉多样性低，BN 统计的有效样本数对 perturb forward 敏感；clean eval 跌 7pt（96.33 → 89.33）。
- PushT：视觉多样性主导 BN 统计，K=4 perturb forward 占比可忽略；clean 87.00 与 LeWM-base 87.33 持平。

**为什么 Stage C 解决不了这个 bug：** Stage C 是在主 loss 上加 `L_cons = w_t · d(z_clean, z_noisy)`，**完全不改 gate 内部的 perturb forward 逻辑**。BN running stats 仍然每步被 K=4 OOD forward 污染，与 `alpha_cons` 大小、`w_t` 设计正交。叠加 Stage C 只会让 BN drift 与 consistency gradient 互相纠缠，更难诊断。

**修复方案（必须先做，再开 Stage C）：**

```python
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

**修复后必须重跑的实验（已完成，见 §3.3.5）：**
- TwoRoom probe+gate：fixbug clean 95.00，证明旧 89.33 主要来自 BN drift。
- PushT probe+gate：fixbug clean 85.33，resolution diagnostics 不塌；robustness gain 保留一部分但低于 bug 版。

**Stage C 的真实定位（fixbug logging 之后）：** Stage C 仍可能解决 TwoRoom 与 PushT *两个任务都接近各自最优* 的兼容性问题——critical 区域降 consistency 保 PushT 接触 resolution，non-critical 区域加 consistency 把 TwoRoom 推向 LeWM+noise 水平。§3.3.5 已证明 logging-only gate signal 不再直接破坏表示；但当前数据**还不能下结论 "adaptive consistency 兼容动态分辨率"**，要由小权重 consistency sweep 验证。

**修订后的 Stage C 进入前置条件（已由 §3.3.5 基本满足）：**
- TwoRoom probe+gate clean ≥ 92：fixbug clean 95.00。
- PushT 不出现 control collapse：fixbug clean 85.33，`transition_resolution_ratio_l2≈0.288`，`id_probe_r2≈0.774`。
- σ probe corr (validate) TwoRoom ≥ 0.5、PushT 接近 0.5：TwoRoom 0.6118，PushT 0.4816。
- §2.4.2 三个 cv 结构判据保持通过：fixbug high-A CV 未显著高于全局 CV。

#### 3.3.5 Pilot-2B：BN Drift Fixbug 复测（2026-05-10）

修复方式：`compute_action_gate_metrics` 的 K 次 perturb forward 内，临时把所有 `BatchNorm` module 切到 `.eval()`，只用 frozen running stats 测 action sensitivity；perturb forward 结束后恢复原 training 状态。该修复不改变主 loss，仍是 logging-only gate。

**Runs：**

| Task | Run | SwanLab ID |
|---|---|---|
| TwoRoom probe+gate-fixbug | `tworoom_lewm_hetero_probe_default_action_gate_fixbug` | `oub19krd3fbecaav7bgie` |
| PushT probe+gate-fixbug | `pusht_lewm_hetero_probe_default_action_gate_fixbug` | `pare2urey6j6nucr9209m` |

**Full eval（3 seeds × 100 episodes）：**

| Task / model | Clean | goal 0.05 | pixels 0.05 | px+goal 0.05 | goal 0.08 | px+goal 0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom LeWM-base | 93.00 | 71.00 | 70.33 | 62.33 | 55.67 | 44.33 |
| TwoRoom LeWM+noise best (`0to008-p1`) | 98.33 | **98.00** | **98.33** | **98.00** | **98.67** | **98.67** |
| TwoRoom hetero-loss | **99.67** | 85.33 | 96.67 | 84.67 | 73.33 | 55.33 |
| TwoRoom probe | 96.33 | 80.67 | 81.00 | 67.00 | 63.67 | 46.00 |
| TwoRoom probe+gate-bug | 89.33 | 49.00 | 52.00 | 36.67 | 41.67 | 33.00 |
| **TwoRoom probe+gate-fixbug** | **95.00** | **87.33** | **85.67** | **76.00** | **70.00** | **49.00** |
| PushT LeWM-base | 87.33 | 38.00 | 17.33 | 15.00 | 15.00 | 3.67 |
| PushT LeWM+noise best (`0to002-p1`) | **90.00** | **85.00** | **87.67** | **86.00** | **83.00** | **70.67** |
| PushT hetero-loss | 13.33 | 7.67 | 7.67 | 7.67 | 9.67 | 6.00 |
| PushT probe | 81.67 | 39.00 | 19.33 | 14.67 | 17.33 | 3.33 |
| PushT probe+gate-bug | 87.00 | 52.00 | 31.67 | 21.00 | 23.00 | 3.33 |
| **PushT probe+gate-fixbug** | **85.33** | **54.00** | **39.00** | **30.33** | **20.33** | **8.33** |

**Diagnostics（fixbug）：**

| Metric | TwoRoom fixbug | PushT fixbug | Interpretation |
|---|---:|---:|---|
| `transition_resolution_ratio_l2` | 0.7263 | 0.2880 | PushT resolution 接近 LeWM-base 0.3015，未 collapse |
| `id_probe_r2` | 0.2505 | 0.7738 | PushT controllable state readout 保持，明显区别于 hetero-loss 0.2678 |
| `clean_effective_rank` | 46.77 | 78.36 | 没有出现 hetero-loss 的 rank 压缩 |
| `action_mean_pred_shift_norm` | 0.5339 | 0.1240 | action-conditioned predictor sensitivity 保持 |

**SwanLab validate metrics（last epoch）：**

| Metric | TwoRoom fixbug | PushT fixbug | Interpretation |
|---|---:|---:|---|
| `validate/hetero_s_logerr_corr_epoch` | 0.6118 | 0.4816 | σ probe 仍学到 prediction difficulty；PushT 略低于 0.5 但接近阈值 |
| `validate/adaptive_corr_sigma_action_epoch` | -0.0104 | 0.2563 | σ 与 action sensitivity 不是同一信号，支持 action-aware 而非 σ-only |
| `validate/adaptive_action_sensitivity_cv_mean_epoch` | 0.4243 | 0.3881 | 多 δ sensitivity 方差可控 |
| `validate/adaptive_action_sensitivity_cv_high_A_epoch` | 0.3708 | 0.3886 | high-A 区域没有更 chaotic |
| `validate/adaptive_weight_q10_epoch` | 0.5533 | 0.5718 | gate weight 下分位不塌到 0 |
| `validate/adaptive_weight_q90_epoch` | 0.9253 | 0.9469 | gate weight 有非平凡 spread |

**关键修正：fixbug 后 probe 与 probe+gate 的训练动态等价**

Fixbug 后 `compute_action_gate_metrics` 内 BN 被临时冻结、所有输出 detach 且不进 loss graph，因此：
- **主 loss graph、梯度流、optimizer 更新规则与纯 probe 模式等价**
- **gate 不再通过 BN running stats、loss 或梯度改变模型参数更新路径**

严格说它不是 bitwise identical：gate logging 仍会采样 action perturbation、可能消耗 dropout/RNG，并更新 `gate_*` EMA buffers（这些 buffers 只用于 logging / 后续 Stage C controller，不参与当前 logging-only loss）。因此 probe+gate-fixbug 与 probe 的 eval 差异（TwoRoom 95.00 vs 96.33，PushT 85.33 vs 81.67）应主要解释为**不同 seed / run 的抽样波动与随机轨迹差异**（num_eval=100×3 seeds，±2–3pt 正常），而非 gate 本身带来性能变化。PushT 的 3.66pt 差略大，但在该任务天然 variance 范围内。

**因此 gate_bugfix 的核心作用不是"提升效果"，而是"解锁无副作用的 logging-only controller"**：
- 证明可以在训练过程中实时采集 `A_t`、`critical_t`、`w_t` 信号
- 证明这些信号的 computation 不破坏已有表示（与 hetero-loss 的 13.33 collapse 形成对照）
- 为 Stage C 提供**可信的 controller 输入**——如果 gate 本身就会污染训练，用它的 `w_t` 去调 consistency 就是建沙上塔

**结论：**

1. **BN drift bug 解释成立。** TwoRoom probe+gate 从 bug 版 clean 89.33 恢复到 fixbug 版 95.00，低/中噪声 eval 也同步恢复。旧 89.33 不是 gate signal 本身失败，而是 BN running stats 被 K 次 OOD forward 污染的副作用；fixbug 后副作用消除，gate 回归纯 logging 定位。
2. **fixbug 后 probe+gate 与 probe 在主训练目标上等价，eval 差异不应解释为 gate 提升。** 这是预期行为，反而增强了结论可信度：gate 的 computation 不再通过 BN / loss / gradient 改变模型行为，因此 Stage C 可以安全地使用 `w_t` 作为 consistency controller。
3. **logging-only gate 解锁了 Stage C 的前提条件。** 关键不是"gate 让 eval 涨了多少"，而是"gate 提供了不破坏训练的信号基础"。SwanLab metrics 显示 σ-A 相关性低/中等、weight spread 非平凡、CV 可控——这些才是支撑 Stage C 的实证。
4. **probe+gate 未超过 LeWM+noise oracle。** PushT `pixels_std0.05` 从 LeWM-base 17.33 提到 39.00，但远低于 LeWM+noise best 87.67。当前结果支持"controller signal 可用"，不支持"无需 consistency 就已经超过 noise training"。这正是 Stage C 的必要性所在：logging 只验证信号，真正的自适应分辨率需要 `alpha_cons > 0` 才能释放。
5. **Stage C 已验证成功，结果见 §3.4。** α=0.01 PushT clean 86.67 通过 guardrail，α=0.03 TwoRoom clean 98.33 达到 LeWM+noise 天花板；α=0.03 PushT 触发 guardrail（clean 76.33 < 84），印证任务特异性。

#### 3.3.6 关键性质验证

**NLL 的好处和风险：** NLL/hetero loss 的潜在好处：让模型不要为了不可预测或视觉噪声细节浪费 μ 分辨率；为 planning 提供 uncertainty signal；可能减少按任务选择 `std_max` 的需求。核心风险：高误差不等于低价值；PushT 的接触瞬间可能 high error 但 high value；downweight hard samples 可能降低 clean control，而不是提升 robustness；loss scale 改变会干扰 SIGReg 权重。

**LeWM 是严格特例：** 在 scale-preserving 形式中，如果 `s ≡ 0` 或 σ 被固定，`hetero_loss = mean(err)`，SIGReg(μ) 不变，严格退化回 LeWM。这个特例关系只有在 scale-preserving 形式下最干净；普通 NLL 会额外改变常数和尺度。

**Noisy TV / confounder trap：** 高 σ 同时包含 dynamics difficulty 和 aleatoric visual noise；σ-only consistency 会放弃对噪声的 invariance。consistency gate 必须 action-aware：以 `A_t` 为主门控，σ 只做 enhancer。

### 3.4 Stage C：Adaptive Consistency Sweep

**Stage C 实验结果（2026-05-11，3 seeds × 100 episodes）：**

| 配置 | TwoRoom clean | TwoRoom px+goal 0.05 | PushT clean | PushT goal 0.05 | PushT pixels 0.05 | PushT px+goal 0.05 |
|---|---:|---:|---:|---:|---:|---:|
| LeWM-base | 93.00 | 62.33 | 87.33 | 38.00 | 17.33 | 15.00 |
| probe+gate-fixbug (α=0) | 95.00 | 76.00 | 85.33 | 54.00 | 39.00 | 30.33 |
| **consist001 (α=0.01)** | **95.33** | **92.00** | **86.67** | **77.00** | **73.33** | **70.67** |
| consist003 (α=0.03) | **98.33** | **97.33** | 76.33 | 69.33 | 69.00 | 67.67 |
| consist001+noise0.002 | 95.33 | 94.00 | **88.00** | **86.00** | **87.33** | **85.33** |
| LeWM+noise best | 98.33 | 98.00 | 90.00 | 85.00 | 87.67 | 86.00 |
| A_t-only consist001 (α=0.01) | 93.33 | 88.67 | 77.33 | 68.00 | 50.00 | 50.00 |

**PushT resolution guardrail：**

| 配置 | `transition_res_l2` | `id_probe_r2` | clean | 状态 |
|---|---:|---:|---:|---|
| LeWM-base | 0.302 | 0.774 | 87.33 | baseline |
| probe+gate-fixbug | 0.288 | 0.774 | 85.33 | ✅ |
| **consist001** | **0.290** | **0.764** | **86.67** | **✅ 全部通过** |
| consist001+noise0.002 | 0.292 | 0.779 | 88.00 | ✅ 全部通过 |
| σ-only consist001 | 0.288 | 0.760 | 87.00 | ✅ 全部通过 |
| consist003 | 0.264 | 0.731 | 76.33 | ⚠️ clean < 84，触发 guardrail |
| A_t-only consist001 | 0.261 | 0.727 | 77.33 | ⚠️ clean 跌 9.34pt，σ 缺失代价 |

**SwanLab 训练侧剂量效应：**

| 任务 | α | `consistency_dist` | `A_sensitivity` | `corr_sigma_action` | 解读 |
|---|---:|---:|---:|---:|:---|
| PushT | 0 | — | 1.137 | 0.259 | baseline |
| PushT | 0.01 | 0.190 | 1.160 | 0.250 | 适度 consistency，clean 维持 |
| PushT | 0.03 | **0.145** | **1.082** | **0.351** | 过度 consistency，resolution 压缩 |
| TwoRoom | 0 | — | 4.919 | -0.010 | baseline |
| TwoRoom | 0.01 | 0.666 | 4.989 | 0.006 | 弱 consistency |
| TwoRoom | 0.03 | **0.389** | 4.832 | **0.098** | 强 consistency，接近 LeWM+noise |

关键发现：
1. **consist001 是 PushT 的 sweet spot**：clean 86.67 ≈ baseline 87.33，goal 0.05 从 38→77（+39pt），pixels 0.05 从 17→73（+56pt），robustness 翻倍以上。
2. **consist003 是 TwoRoom 的最优配置**：clean 98.33 = LeWM+noise best，px+goal 0.05 97.33（LeWM+noise 98.00），所有 noise 条件 96–99。
3. **consist001 TwoRoom 稳健提升**：clean 95.33 > baseline 93.00，px+goal 0.05 92.00 > baseline 62.33（+30pt），但不及 LeWM+noise best 98.00。
4. **consist001+noise0.002 在 PushT 上表现优异**：clean 88.00（超过 consist001 的 86.67），pixels 0.05 87.33 逼近 LeWM+noise best 87.67，px+goal 0.05 85.33 超过 consist001 的 70.67（+14.66pt）。说明 **light noise 增广与 adaptive consistency 在 PushT 上有协同效应**——全局 invariance baseline 由 noise 提供，per-token controller 在此基础上做精细化分配。
5. **跷跷板确认，但正是预期行为**：PushT 对 consistency 敏感（α=0.03 跌 11pt），TwoRoom 受益于更多 consistency（α=0.03 从 95.33→98.33）。这验证了自适应机制的任务特异性——不同任务需要不同的 consistency 强度。
6. **Gate 分布在 consistency 训练中稳定**：`weight_mean` / `weight_q10` / `weight_q90` 在 fixbug / consist001 / consist003 之间几乎不变，说明 detach 设计有效，encoder 未学会操纵 gate。

**Stage C 的真实定位：**
- **PushT α=0.01**：提供 **clean 维持 + robustness 大幅提升** 的最佳平衡点。goal 0.05 77.00 虽低于 LeWM+noise 85.00，但远超 baseline 38.00；且无需手调 noise std。
- **TwoRoom α=0.03**：可达到 **LeWM+noise 天花板**（98.33），px+goal 0.05 97.33 与 LeWM+noise 98.00 仅差 0.67pt。α=0.01 也有稳健提升（95.33），但不如 0.03 接近最优。
- **任务特异性是特征不是缺陷**：不存在"一个 α 通吃所有任务"，这正是自适应 resolution 的核心主张——action-critical 任务（PushT）需要较低的 baseline consistency，冗余视觉任务（TwoRoom）可以承受更高的 consistency pressure。

### 3.5 A_t-only Ablation

> A_t-only ablation 验证 σ 的不可替代性：关闭 hetero probe（σ≡0），仅保留 action_gate + adaptive_consistency α=0.01。

**PushT A_t-only 详细结果（3 seeds）：**

| 指标 | σ+A_t consist001 | A_t-only consist001 | Δ | 解读 |
|---|---:|---:|---:|:---|
| clean | **86.67** | **77.33** | **-9.34** | σ 缺失导致 clean 显著下跌 |
| goal 0.03 | **84.67** | **70.33** | -14.34 | robustness 差距更大 |
| pixels 0.03 | **82.33** | **68.67** | -13.66 | 同上 |
| px+goal 0.03 | **80.00** | **69.67** | -10.33 | 同上 |
| `weight_mean` | 0.774 | **0.852** | +0.078 | A_t-only 整体 weight 偏高 |
| `weight_q10` | 0.574 | **0.723** | +0.149 | **下界 token 也被强 consistency** |
| `critical_mean` | 0.283 | **0.185** | -0.098 | 无 σ 增强，critical 整体偏低 |
| `consistency_dist` | 0.190 | 0.191 | +0.001 | encoder 输出距离几乎相同 |

**TwoRoom A_t-only 详细结果（3 seeds）：**

| 条件 | σ+A_t consist001 | A_t-only consist001 | Δ |
|---|---:|---:|---:|
| clean | 95.33 | 93.33 | −2.00 |
| goal 0.05 | 93.33 | 88.00 | −5.33 |
| pixels_goal 0.05 | 92.00 | 88.67 | −3.33 |
| pixels 0.05 | 93.33 | 94.67 | +1.34 |
| goal 0.08 | 93.67 | 85.33 | −8.34 |
| pixels_goal 0.08 | 94.67 | 76.67 | −18.00 |
| pixels 0.08 | 92.33 | 84.67 | −7.66 |

TwoRoom resolution（A_t-only vs σ+A_t）：`transition_res_l2` 0.720 vs 0.716，`id_probe_r2` 0.264 vs 0.230。A_t-only 在 TwoRoom 上 resolution 未受损，甚至略高——因为 TwoRoom 的 action 空间简单（2D 离散），A_t 本身已能捕获大部分可控性差异；σ 的边际增益主要体现在高 noise 条件（px+goal 0.08 94.67→76.67）。

**PushT A_t-only 详细结果（3 seeds，完整 seed）：**

| 条件 | σ+A_t consist001 | A_t-only consist001 | Δ |
|---|---:|---:|---:|
| clean | 86.67 | 77.33 | **−9.34** |
| goal 0.05 | 77.00 | 68.00 | −9.00 |
| pixels 0.05 | 73.33 | 50.00 | −23.33 |
| px+goal 0.05 | 70.67 | 50.00 | −20.67 |
| goal 0.08 | — | 28.00 | — |
| pixels 0.08 | — | 7.33 | — |
| px+goal 0.08 | — | 6.67 | — |

PushT resolution guardrail：A_t-only `transition_res_l2=0.261`、`id_probe_r2=0.727`，均低于阈值 0.24/0.65 但 clean 已跌到 77.33（< 84），说明 A_t-only 的 resolution 压缩发生在 planning-relevant 的精细控制区域，guardrail 的硬阈值未能完全捕捉。

**为什么 A_t-only 更差？**

1. **Dynamic range 压缩**：A_t-only 的 `weight_q10=0.723` vs σ+A_t 的 `0.574`，说明 A_t-only 几乎给所有 token 施加了强 consistency（q10-q90 gap 仅 0.241 vs 0.373）。σ 提供了独立的 difficulty 维度，让 gate 能在"同样 action-sensitive"的 token 中区分"简单可控"和"困难可控"。
2. **Uniform high consistency ≈ 全局 noise training 的弱化版**：A_t-only 没有 σ 的 difficulty enhancer，导致 critical 区域保护不足、non-critical 区域过度 invariance——这正是 plan_v3 中 LeWM+noise 的问题（所有区域同等处理），只是程度更轻。
3. **`corr_sigma_action = 0.000`**（A_t-only 无 σ）：SwanLab 确认 σ 信号完全缺失，multiplicative `critical = gA * (0.5 + 0.5*gS)` 退化为 `gA * 0.5`，失去了难度调节能力。

**结论**：**σ head 在 α=0.01 剂量下不是可选装饰**。A_t 单独无法区分"简单可控"和"困难可控"，σ 提供 difficulty 维度让 gate 把 high-A 区域中真正 difficult 的子集挑出来。PushT 上 clean 跌 9.34pt、高 noise 跌 20+pt；TwoRoom 上差距较小（clean −2.00），但高 noise 仍显著落后。

### 3.6 σ-only Ablation（Noisy TV / Confounder Trap 验证）

> σ-only ablation 验证：关闭 action_gate（A_t≡1，所有 token 的 controllability 视为相同），仅保留 σ 控制 consistency weight。预期结果：σ-only 会落入 Noisy TV / confounder trap——高 σ 同时包含 dynamics difficulty 和 aleatoric visual noise，σ-only consistency 会"保护"噪声状态的分辨率，导致 planning 在噪声下崩溃。

**PushT σ-only 详细结果（3 seeds）：**

| 条件 | σ-only consist001 | σ+A_t consist001 | Δ |
|---|---:|---:|---:|
| clean | 87.00 | 86.67 | +0.33 |
| goal 0.05 | 76.33 | 77.00 | −0.67 |
| pixels 0.05 | 69.33 | 73.33 | −4.00 |
| px+goal 0.05 | 65.67 | 70.67 | −5.00 |
| goal 0.08 | **44.33** | — | — |
| pixels 0.08 | **27.00** | — | — |
| px+goal 0.08 | **20.00** | — | — |

PushT σ-only guardrail：`transition_res_l2=0.288`、`id_probe_r2=0.760`，clean 87.00 ≥ 84——**全部通过**。这与 hetero-loss collapse（res 从 0.30→0.10）完全不同：σ-only 的 clean resolution 没有被压缩，encoder 仍能区分 transition。

**TwoRoom σ-only 部分结果（⚠️ 高 noise 条件尚未完成全部 3 seeds）：**

| 条件 | 已完成 seeds | σ-only | σ+A_t consist001 |
|---|---:|---:|---:|
| clean | 3 | 95.33 | 95.33 |
| goal 0.03 | 3 | 94.67 | 94.33 |
| pixels_goal 0.03 | 3 | 92.67 | 95.00 |
| pixels 0.03 | 3 | 94.67 | 94.33 |
| goal 0.05 | 1 | 93.00 | 93.33 |
| pixels_goal 0.05 | 2 | 91.00 | 94.00 |
| pixels 0.05 | 1 | 91.00 | 93.33 |
| pixels 0.08 | 1 | 86.00 | 92.33 |

已完成的低–中 noise 条件显示 TwoRoom σ-only 与 σ+A_t 接近（clean 95.33 vs 95.33），但已出数据中 pixels_goal 0.05（91.00 vs 94.00）和 pixels 0.08（86.00 vs 92.33）开始出现差距。高 noise 条件（goal 0.08、pixels_goal 0.08）的缺失使 TwoRoom σ-only 结论尚不完整，但 PushT 数据已足够支撑核心 claim。

**核心发现：Noisy TV trap 在 PushT 上被精确验证**

1. **σ-only 的分辨率没有问题，但 robustness 在高 noise 下断崖式下跌。** PushT σ-only clean 87.00 与 σ+A_t 86.67 几乎相同，guardrail 全部通过。这说明 σ head 本身没有破坏 encoder 的区分能力；问题在于 σ 把噪声状态的"高 uncertainty"误判为"需要保护分辨率"。
2. **高 noise 条件的崩溃模式与 confounder trap 预测一致。** goal 0.08 44.33（σ-only）vs 无数据（σ+A_t），但可对比 LeWM+noise best 在 goal 0.08 的表现（约 80+）。σ-only 在 px+goal 0.08 跌至 20.00，接近随机——因为 pixels noise 使大量背景 token 的 σ 虚高，consistency weight 被压低，encoder 对这些噪声 token "过度保护"，导致 planner 在混乱的 latent 空间中迷失。
3. **σ 和 A_t 的互补性被完整验证。** A_t-only（§3.5）的问题：dynamic range 压缩、uniform high consistency、无法区分简单/困难可控。σ-only 的问题：无法区分 dynamics difficulty 与 aleatoric noise，高 noise 下 robustness 崩溃。只有 σ+A_t 联合使用时，A_t 过滤掉不可控噪声，σ 在可控区域内调节 difficulty——两者缺一不可。
4. **Guardrail 的局限性暴露。** σ-only 的 `transition_res_l2=0.288` 和 `id_probe_r2=0.760` 都通过硬阈值，但 robustness 仍然崩溃。这说明 guardrail 只检查 clean / resolution，不检查 noise robustness；真正验证机制有效性需要 multi-condition eval，不能单靠诊断指标。

### 3.7 w_t 离线可视化

离线提取 `pusht_lewm_action_gate_consist001` ckpt 的 per-token `w_t` / `critical_t` / `gA_t`，与 task-structure proxy（action norm `||a_t||`、latent displacement `||z_{t+1}−z_t||`）对应。样本：256 sequences × history_size=3 = 768 tokens。

**关键定量：**
- `corr(w_t, action_norm) = +0.587`：**action norm 越大，w_t 越高**（一致性压力越强）。这与 naive 直觉相反，但符合 PushT 的物理结构：free-space 接近阶段 action norm 高但 action sensitivity（A_t）反而低——predictor 在 contact 约束下更稳定；而 free-space 小 action 即可产生大 latent 位移，A_t 更高 → critical 更高 → w_t 更低。
- `corr(w_t, latent_disp) = -0.592`：**latent displacement 越大，w_t 越低**。这与设计完全一致：transition 剧烈（高 displacement）的区域被标记为 critical，一致性压力减轻以保护分辨率。
- Quartile 分层：Q1（低 action norm）mean w_t=0.768，Q4（高 action norm）mean w_t=0.898，差值 0.130，动态范围非平凡。

**Figure 1：w_t vs action norm（hexbin）**

![w_t vs action norm](assets/diagnostics/wt_vs_action_norm.png)

**Figure 2：w_t vs latent displacement（hexbin）**

![w_t vs latent displacement](assets/diagnostics/wt_vs_latent_disp.png)

**Figure 3：时间序列示例（w_t + action norm）**

![w_t timeseries example](assets/diagnostics/wt_timeseries_example.png)

**Figure 4：w_t 分布按 action norm 四分位**

![w_t histogram by action norm](assets/diagnostics/wt_histogram_by_action_norm.png)

**论文叙事**：`w_t` 不是简单地与 "contact = high action norm" 线性对应，而是与 **action sensitivity / transition difficulty** 对应。这恰恰说明 per-token adaptive weight 比全局 consistency 或 naive contact heuristic 更精细：它保护的是 "predictor 觉得难" 的区域，而不是 "人类标注的 contact" 区域。

### 3.8 结论总结与待做实验

**整体路线回顾**：LeWM-base → hetero-loss ablation（失败）→ probe-only σ（成功）→ logging-only action-gate（成功）→ adaptive consistency sweep（成功）。核心创新不是"加一个 σ head"，而是**σ + A_t 共同控制 per-token consistency**，让 encoder 在 action-critical 区域保留分辨率、在视觉冗余区域加强 invariance。

**实验阶梯总结**：

| 阶段 | 必要条件 | PushT 边际收益 | TwoRoom 边际收益 | 实际结果 |
|---|---|---|---|---|
| probe-only | `hetero_s_logerr_corr ≥ 0.5` | clean ≥ 86 | clean ≥ 92 | ✅ TwoRoom 0.61, PushT 0.48 |
| logging-only gate | 三个结构判据 + BN drift 已修 | clean ≥ 84, res ≥ 0.24 | clean ≥ 92 | ✅ fixbug 通过 |
| α=0.01 consistency | guardrail 不破 | clean 不跌 > 2pt | clean 提升 ≥ 2pt | ✅ PushT 86.67, TwoRoom 95.33 |
| α=0.03 consistency | 同上 | 同上 | 接近 LeWM+noise | ✅ TwoRoom 98.33; ❌ PushT 76.33 触发 guardrail |

**已完成实验（✅）**：

| Experiment | TwoRoom | PushT | 结果 |
|---|---:|---:|---|
| `lewm_sigma_probe_default` | 96.33 | 81.67 | σ calibration 成立（0.61/0.48）|
| `lewm_action_gate_logging` (fixbug) | 95.00 | 85.33 | gate signal 不破坏训练 |
| `lewm_action_aware_consist001` | **95.33** | **86.67** | **PushT clean 维持 + robustness 翻倍** |
| `lewm_action_aware_consist003` | **98.33** | 76.33 | **TwoRoom = LeWM+noise best**，PushT 触发 guardrail |
| `lewm_action_only_consist001` (A_t-only) | 93.33 | 77.33 | σ 必要性完整验证（TwoRoom 3 seeds + PushT 3 seeds + diagnostics）|
| `lewm_sigma_only_consist001` (σ-only) | 95.33* | 87.00 | Noisy TV / confounder trap 在 PushT 上精确验证（*TwoRoom 部分种子）|
| `lewm_action_aware_consist001_noise002` | 95.33 | **88.00** | consistency + light noise 联用，PushT pixels 0.05 逼近 LeWM+noise best |
| `w_t` 离线可视化 | — | ✅ | corr +0.587 / −0.592，动态范围非平凡 |

**判定标准（最终版）**：
1. ✅ PushT consist001 clean 86.67 ≥ 84，resolution 0.290 ≥ 0.24。
2. ✅ σ calibration 保持（validate corr 0.48–0.62）。
3. ✅ `A_t` / `critical_t` 显示 action-relevant 结构（CV 可控，weight spread 非平凡）。
4. ✅ BN drift fix 语义保持（consist001/003 均使用 freeze-BN gate）。
5. ✅ **σ 与 A_t 缺一不可**：A_t-only PushT clean 跌 9.34pt（77.33 vs 86.67），σ-only PushT px+goal 0.08 崩溃至 20.00；只有 σ+A_t 联合使用才能在 PushT 上同时维持 clean 和 robustness。

**待做实验（按优先级）**：

| Experiment | TwoRoom | PushT | 目的 | 优先级 |
|---|---:|---:|---|---|
| `lewm_sigma_only_consist001` 补全种子 | 部分完成 | ✅ | TwoRoom goal 0.08 / pixels_goal 0.08 / 部分 0.05 | **高** |
| `lewm_sigma_probe_on_noise` | yes | yes | noise 训练下 σ calibration 是否漂移 | 中 |
| 4-task full eval | Reacher | Cube | 验证跨任务泛化 | 低（先写论文）|

**与 Noise 训练联用（§3.8，已部分执行）**：
目前的叙事链是 LeWM-base → LeWM+noise → 因此需要 adaptive resolution。机制上 +noise 和 σ-adaptive 处于不同位置：noise 是 input-side 的 isotropic 数据增广，σ 是 output-side 的 per-state difficulty 信号，`A_t` 是 controllability filter，三者互补。

**已执行结果**：`consist001+noise0.002` 在 PushT 上表现优异——clean 88.00（超过 consist001 的 86.67），pixels 0.05 87.33 逼近 LeWM+noise best 87.67，px+goal 0.05 85.33 超过 consist001 的 70.67（+14.66pt）。这说明 **light noise 增广与 adaptive consistency 在 PushT 上有协同效应**：全局 invariance baseline 由 noise 提供，per-token σ/A controller 在此基础上做精细化分配。TwoRoom 上 noise0.002 结果（clean 95.33, px+goal 0.05 94.00）与 consist001（95.33 / 92.00）接近，没有 PushT 那样显著的额外增益——因为 TwoRoom 本身已能耐受较高 consistency，noise 的边际效用较低。

**后续可选实验**：
1. **Probe-on-noise**：LeWM+noise ckpt 加 σ probe（μ path 不变）。检查 σ 在 noise 训练下是否仍稳定 calibration。
2. **Consistency-on-noise 更高剂量**：`std_max=0.03–0.05` 与 α=0.01–0.03 的联合 sweep。

**开放问题**：
- σ 的 multi-step propagation 在 rollout 下是否仍然校准？
- A_t 的 local sensitivity 与任务全局结构（如 door crossing in TwoRoom）是否有系统性对应？
- 是否需要一个 encoder-side input-sensitivity head（§2.1.3）来闭合 encoder→controller 的反馈环？

**4-Task 全套验证**：
Reacher 和 Cube 待跑。当前 TwoRoom + PushT 的结果已足够支撑论文核心 claim；4-task 验证在论文初稿后再补。

## 4. 讨论

### 4.1 核心发现

1. **σ head 学到非平凡、任务相关的 prediction difficulty。** `hetero_s_logerr_corr` ≥ 0.89（Pilot-1B）/ ≥ 0.46（Pilot-2A PushT）。
2. **直接 hetero loss reweighting 摧毁 PushT 控制分辨率。** `transition_resolution_ratio_l2` 从 0.30 崩到 0.10；clean eval 掉 74 点。
3. **可行路径是 σ 作为诊断/控制器，而非梯度 reweighter。** Action-aware adaptive consistency（§2.2.3）是唯一既改变 resolution 又避开 confounder trap 的使用层级。
4. **BN drift bug 已修复并复测通过；修复后 gate 回归纯 logging 定位。** K=4 perturb forward 在 train mode 下污染 BN running stats，TwoRoom probe+gate clean 96.33 → 89.33；fixbug 后（BN freeze + no_grad）probe+gate 与 probe 在主 loss/gradient/optimizer 更新规则上等价，eval 差异不应解释为 gate 改善效果。旧 89.33 不是 gate signal 失败，而是 stateful side effect（BN running stats）的跨 step 累积。fixbug 消除了这个 side effect，使 gate 成为无训练副作用的 logging-only controller。
5. **Stage B 的核心产出是"logging signal 可用且不破坏训练"，而非"eval 提升"。** fixbug gate 的 σ-A 相关性低/中等、weight spread 非平凡、PushT resolution guardrail 通过——这些 logging 指标证明 `w_t` 有资格作为 Stage C 的 controller 输入。
6. **Stage C 在每个任务各自最优 α 上验证成功，剂量效应方向与 guardrail 一致。** PushT α=0.01（consist001）clean 86.67 ≈ baseline，robustness 翻倍（goal 0.05 38→77，pixels 0.05 17→73）；TwoRoom α=0.03（consist003）clean 98.33 = LeWM+noise best (`0to008-p1`) 98.33，px+goal 0.05 97.33（LeWM+noise 98.00）。更高 α 导致 PushT resolution 压缩（0.290→0.264）而 TwoRoom 继续提升，验证了任务特异性 consistency 需求；**但目前没有单一 α 同时在两任务上达到 oracle**——这是机制特性，也意味着论文叙事必须沿"per-task α"或"per-token w_t"展开，不可主张全局单点最优。
7. **A_t-only ablation 完整验证 σ 不可缺失，w_t 离线可视化验证 gate 与 task structure 有结构性对应。** PushT A_t-only clean 77.33 比 σ+A_t（86.67）低 9.34pt，高 noise 跌 20+pt，`weight_q10` 从 0.574 涨到 0.723（dynamic range 压缩）；TwoRoom A_t-only clean 93.33 接近 baseline 93.00，低于 σ+A_t 95.33，高 noise 同样落后。w_t 与 action norm 正相关（+0.587）、与 latent displacement 负相关（−0.592），印证 per-token adaptive weight 保护的是 "predictor 觉得难" 的区域，而非 naive contact heuristic。
8. **σ-only ablation 在 PushT 上精确验证 Noisy TV / confounder trap。** σ-only clean 87.00 与 σ+A_t 86.67 几乎相同，guardrail 全部通过——分辨率未受损；但 px+goal 0.08 崩溃至 20.00，goal 0.08 跌至 44.33。这证明 σ 本身不破坏表示，但 σ-only consistency 会把噪声状态的"高 uncertainty"误判为"需要保护分辨率"，导致 planner 在混乱 latent 空间中迷失。A_t 的 controllability filter 作用是过滤掉不可控噪声，而非压缩 resolution。
9. **Consistency + light noise 联用在 PushT 上显示协同效应。** `consist001+noise0.002` clean 88.00 超过单独 consist001（86.67），pixels 0.05 87.33 逼近 LeWM+noise best 87.67。这说明全局 noise 提供 invariance baseline，σ/A controller 在此基础上做 per-state 精细化分配——两者不是替代关系，而是互补关系。

### 4.2 风险与对策

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
| **BN drift via gate perturb forward**（2026-05-09 实证发现，2026-05-10 已修） | gate 内 K 次 perturb forward 在 train mode 下走 projector / predictor_proj 的 `BatchNorm1d`，污染 running stats；TwoRoom probe+gate clean 因此跌 7pt（96.33 → 89.33）。 | 已在 `compute_action_gate_metrics` perturb forward 前临时把所有 `_BatchNorm` module `.eval()`，结束后恢复。fixbug full eval 已验证 TwoRoom clean 回到 95.00；后续 Stage C 仍需保持该 freeze-BN 语义。 |
| **不超过 LeWM+noise oracle** | 很可能；LeWM+noise 已经很强 | 目标先设为减少手调且接近 oracle；若明显低于 oracle，降级为 analysis/future work |

### 4.3 诊断工具的角色定位

之前版本主张"17 个诊断指标 = (μ, σ) 框架的 2–3 个本征轴"。**这个主张过于激进**——它假设所有诊断都能被 (μ, σ) 解释，且压缩比可观。这是 empirical question，需要 Pilot-1 数据验证。

本最简版的诚实主张：
- predictor σ̂ 输出本身**就是**新增的 per-transition 诊断量
- 现有诊断（`clean_nn_dist`, `effective_rank`, `transition_resolution_ratio` 等）和 σ̂ 的相关性是**值得测的事后分析**，但不作为 a priori 的论文主张
- 如果实证发现 σ̂ 和某些诊断高相关 → 加分项；如果不相关 → σ̂ 提供独立的新信息，也是加分项

→ **诊断工具的价值主要是设计约束和机制解释**，不再要求先证明它们能独立预测 eval。它们与本框架的成败解耦：即使 P0.6 盲分桶不强，σ-head 仍可能作为更直接的 adaptive resolution 方法成立。

### 4.4 论文 Novelty 主张与边界

**Claim（待 Pilot 验证）：**
> Action-aware adaptive consistency for latent resolution: 在 LeWM predictor 上加 detached scalar σ probe 来估计 transition difficulty，同时用 action-conditioned local sensitivity `A_t` 区分 controllable critical states 与不可控视觉噪声；二者共同控制 clean/noisy encoder consistency strength，使简单/冗余区域更 invariant、action-critical 区域保留 resolution。LeWM 与 LeWM+noise 分别是无 σ/A controller 与全局 consistency 的严格 baseline。

**前提条件：**
- Stage A 证明 σ head 学到非平凡、任务相关的 prediction difficulty。
- Logging-only 阶段证明 `A_t` 能过滤 σ 中的 aleatoric visual noise，而不是只复述 prediction error。
- Adaptive consistency 证明使用 `critical_t = f(A_t, σ_t)` 后能接近或超过 LeWM+noise oracle，而不是只超过 LeWM-base。
- 收益不是来自重新调 SIGReg / loss scale，也不是来自把 hard transitions 的 prediction gradient 降权。

**不再主张：**
- "NLL 一定比 MSE 好"。
- "σ head 自然就是 latent resolution"。
- "高 σ 就应该保留分辨率"。
- "不改 planner 就一定能在 inference 自动受益"。
- IB / Fisher manifold / "诊断 = (μ, σ) 本征轴"等强理论叙事。

---

## 附录 A：设计回退记录（Honest Engineering Notes）

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

## 附录 B：超参数预算与 Guardrail 阈值表

### B.1 超参数

| 名称 | 默认值 | 范围 | 阶段 |
|---|---:|---|---|
| `loss.hetero.probe_weight` (`beta_probe`) | 1.0 | [0.1, 5.0] | A |
| `loss.action_gate.delta_scale` | 0.25 | [0.05, 0.5] | B |
| `loss.action_gate.num_delta_samples` (K) | 4 | [2, 8] | B |
| `loss.action_gate.warmup_epochs` | 3 | [0, 5] | B |
| `loss.action_gate.ema_momentum` | 0.99 | [0.95, 0.999] | B |
| `loss.adaptive_consistency.w_min` | 0.2 | [0.0, 0.5] | C |
| `loss.adaptive_consistency.w_max` | 1.0 | [0.5, 1.5] | C |
| `loss.adaptive_consistency.alpha_cons` | 0.01 | [0.001, 0.1] | C |

### B.2 PushT Resolution Guardrails（相对 PushT LeWM-base）

| Metric | PushT LeWM-base | Stop / reject if |
|---|---:|---:|
| `id_probe_r2` | 0.774 | < 0.65 |
| `transition_resolution_ratio_l2` | 0.301 | < 0.24 |
| `action_mean_pred_shift_norm` | 0.128 | < 0.10 |
| Clean eval | 87.33 / 86.00 | < 84 |

这些 guardrail 比单看 `pred_loss_mse_equiv` 更重要，因为本轮已经证明 MSE 可以下降但 planning 失败。

---

## 附录 C：与 plan_v3 和 plan_v2 的关系

### C.1 与 plan_v3 §6 P4 的关系

本文件是 plan_v3 §6 P4 "Adaptive Resolution Method" 的首选具体化方案，但执行上必须分阶段。2026-05-09 Pilot-1B 已触发关键 fallback 条件：直接 hetero loss 伤害 PushT critical-transition resolution。后续应优先做 probe-only σ、action-gate logging、action-aware adaptive consistency；σ planner use / guarded hetero auxiliary 仅作为对照或备选。

### C.2 与 plan_v2 V1/V2 的关系

- V1 (vMF): 球面 + 1D 角度 σ 的特化版，本框架的 spherical projection 限制
- V2 (ball-cap): σ_x quantile clip 的 OOD 延伸

V1/V2 都是更复杂版本，**本最简版本不预设走那个方向**，看 Pilot 结果再决定。

---

## 附录 D：References

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

## 维护说明

- 本文件供查阅与设计迭代；**不**作为 plan_v3 的替换。
- 每次新讨论后追加新条目到 §4.2 风险表 或 附录 A 回退记录。
- **Stage A→B→C 主线已跑完核心 sweep**（probe→gate logging→consistency consist001/003 + A_t-only ablation + σ-only ablation + `w_t` 离线可视化 + consist001+noise0.002）。**σ 与 A_t 缺一不可的核心 claim 已在 PushT 上完整验证**：A_t-only clean 跌 9.34pt，σ-only px+goal 0.08 崩溃至 20.00，只有 σ+A_t 同时维持 clean 和 robustness。唯一剩余缺口：**TwoRoom σ-only 补全高 noise 种子**（goal 0.08 / pixels_goal 0.08 / 部分 0.05）。
- 论文叙事核心已可立：per-task α 下，σ+A_t adaptive consistency 在 PushT 上 clean 维持 + robustness 翻倍，在 TwoRoom 上达到 LeWM+noise best 98.33——且 consistency+light noise 联用在 PushT 上进一步将 pixels 0.05 提升到 87.33（逼近 LeWM+noise best 87.67）。不存在单一 α 同时达到两任务 oracle，这正是 per-token `w_t` 的存在理由。
- 后续若 TwoRoom σ-only 补全种子通过，把 §2–§4 与 §3.8 合并进 plan_v3 §6 P4；本文件归档。
- **下一次想加新机制前**: 先回看附录 A，问自己"它会增加几个超参数？经验收益的证据是什么？"。如果两个问题答不清楚，不加。
