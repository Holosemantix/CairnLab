# Planner-side Robustification 实验计划：Dynamic Risk-Aware CEM

> 目标：在**不重训世界模型、不改 encoder / predictor 权重**的前提下，只修改 CEM 推理阶段，验证视觉 OOD collapse 是否主要来自 point-estimate CEM 对 noisy latent 的脆弱性。  
> 核心原则：先做最小可行的 robust reranking；只有出现正结果后，再做动态半径、belief filter、manifold projection 等复杂版本。

---

## 0. 背景与实验问题

当前 Paper1 已经证明：LeWM-base 在 pixels+goal Gaussian noise 下会严重掉点；LeWM+noise 能修复，但需要 task-specific `std_max` 调参。Planner-side robustification 要回答一个更直接的问题：

> 标准 CEM 是否因为只在单个 noisy latent state / noisy goal latent 上优化，导致 action sequence 对 latent encode shift 过拟合？

如果训练不变、只把 CEM 评价准则从单点 cost 改成 latent belief 下的 risk cost，就能显著恢复 OOD 控制，则说明 failure 不只是 encoder training 问题，也是 planning-time risk aggregation 问题。

---

## 1. 非目标

本轮不要做以下事情：

1. 不重训 LeWM / PLDM。
2. 不新增 σ head、AAAC、consistency loss。
3. 不使用 clean test image、clean goal image 或 simulator hidden state 做校正。
4. 不做复杂 learned denoiser。
5. 不把方法一开始写成新 robust MPC framework；先做 inference-time intervention。

---

## 2. 总体方案

新增一个 planner-side 模块：`RobustCEM` 或 `RiskAwareCEMWrapper`。

标准 CEM 当前近似做：

```text
a* = argmin_a J(a; z_obs, z_goal)
```

改成：

```text
a* = argmin_a Risk_s J(a; z_obs + delta_obs_s, z_goal + delta_goal_s)
```

其中 `delta_obs_s / delta_goal_s` 来自 observed corrupted image 周围的 encoder ensemble，或来自 latent perturbation 分布。

第一版只做 **top-K robust reranking**：

1. 普通 CEM 正常运行，得到最后一轮 top-K action sequences。
2. 对 top-K candidates 做 robust risk evaluation。
3. 选择 robust risk 最小的 action sequence。
4. 执行第一步 action。

这样改动最小，也方便和原始 CEM 对照。

---

## 3. 代码改动范围

Codex 先在仓库里定位以下入口：

```bash
grep -R "class.*CEM\|def.*cem\|CrossEntropy\|plan" -n .
grep -R "encode\|encoder" -n . | head
grep -R "goal" -n . | head
grep -R "eval_results\|num_eval\|seed" -n . | head
```

优先只改 inference / eval 路径。默认配置必须完全等价于原始 CEM。

建议新增或修改：

```text
configs/planner/robust_cem.yaml          # 新增 planner 配置
planners/robust_cem.py                   # 如果仓库已有 planner 目录
utils/latent_belief.py                   # latent belief / TTA / perturbation 工具
tools/eval_robust_cem.py                 # 批量评估脚本，也可复用现有 eval 脚本加 flag
tests/test_robust_cem.py                 # shape、no-leak、default-off 测试
```

如果仓库没有这些目录，不要强行重构；在现有 CEM 文件旁边加 wrapper。

---

## 4. 新增配置项

建议所有字段挂在 `planner.robust` 下：

```yaml
planner:
  robust:
    enabled: false

    # integration mode
    mode: rerank_topk        # rerank_topk | inner_loop
    topk: 64                # final CEM top-K candidates to rerank

    # which latents to robustify
    robust_current: true
    robust_goal: true

    # latent belief construction
    belief_mode: input_tta_empirical   # none | latent_iso | latent_diag | input_tta_empirical
    tta_num: 8
    tta_noise_std: 0.005
    tta_include_identity: true
    latent_samples: 8
    latent_radius: 1.0

    # risk aggregation
    risk: cvar              # mean | mean_std | quantile | cvar | max
    cvar_q: 0.8
    quantile_q: 0.8
    lambda_std: 1.0

    # dynamic scale, off by default in Stage 1
    dynamic_scale: false
    clean_nn_radius_path: null
    u_low: 1.0
    u_high: 3.0
    max_radius_mult: 2.0

    # compute controls
    compute_matched_tag: false
    log_debug: true
```

Default `enabled=false` 必须完全复现原 eval。

---

## 5. Latent belief 构造

### 5.1 输入

对当前 observation image `x_obs` 和 goal image `g_obs` 分别构造 latent belief。

注意：

- TTA 必须作用在**当前观测到的图像**上，即 `x_obs` / `g_obs`。
- 禁止从 clean image 重新采样噪声。
- 禁止访问 simulator hidden state。
- encoder forward 必须在 `model.eval()` 和 `torch.no_grad()` 下运行。
- 如果模型里有 BatchNorm，TTA forward 不得更新 running stats。

### 5.2 `input_tta_empirical`

给 observed image 做小扰动 ensemble：

```text
x_i = T_i(x_obs)
z_i = encoder(x_i)
z_center = mean_i z_i      # 第一版用 mean；geometric median 可后置
Delta_i = z_i - z_center
```

TTA 第一版只做 Gaussian jitter，避免引入太多图像增强变量：

```text
T_i(x) = clamp(x + Normal(0, tta_noise_std), valid_range)
```

`tta_include_identity=true` 时，第一个 sample 是原始 observed image。

### 5.3 `latent_iso`

不做 input TTA，直接在 latent 上采样 isotropic perturbation：

```text
z_center = encoder(x_obs)
Delta_s ~ Normal(0, latent_radius * scale)
```

这是最便宜的 sanity baseline，但不一定真实反映 encoder shift。

### 5.4 `latent_diag`

用 input TTA 估计 diagonal covariance，再采样：

```text
sigma_diag = std_i(z_i)
Delta_s ~ Normal(0, sigma_diag * latent_radius)
```

### 5.5 输出结构

实现一个通用返回对象：

```python
@dataclass
class LatentBelief:
    center: torch.Tensor          # same shape as model latent
    deltas: torch.Tensor          # [S, ...latent_shape]
    uncertainty: torch.Tensor     # scalar or per-batch scalar
    meta: dict
```

---

## 6. Risk aggregation

对每个 candidate action sequence，得到多组 latent perturbation 下的 cost：

```text
cost_samples: [num_candidates, num_samples]
```

实现以下 aggregator：

```python
def aggregate_risk(cost_samples, risk, lambda_std=1.0, q=0.8):
    if risk == "mean":
        return cost_samples.mean(dim=-1)
    if risk == "mean_std":
        return cost_samples.mean(dim=-1) + lambda_std * cost_samples.std(dim=-1)
    if risk == "quantile":
        return torch.quantile(cost_samples, q, dim=-1)
    if risk == "cvar":
        # average of worst tail costs
        threshold = torch.quantile(cost_samples, q, dim=-1, keepdim=True)
        mask = cost_samples >= threshold
        return (cost_samples * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    if risk == "max":
        return cost_samples.max(dim=-1).values
```

主实验默认：

```yaml
risk: cvar
cvar_q: 0.8
latent_samples: 8
```

`max` 只作为 ablation，不作为主方法。

---

## 7. CEM 集成方式

### 7.1 Stage 1：`rerank_topk`

推荐第一版只做 rerank。

伪代码：

```python
# 1. 普通 CEM，保持原逻辑
cem_result = original_cem(..., return_candidates=True)
candidates = cem_result.topk_action_sequences  # [K, H, action_dim]

# 2. 构造 latent belief
belief_current = build_latent_belief(x_obs, encoder, cfg) if robust_current else point_belief(z_obs)
belief_goal = build_latent_belief(g_obs, encoder, cfg) if robust_goal else point_belief(z_goal)

# 3. robust rerank
cost_samples = []
for s in range(S):
    z0_s = belief_current.center + belief_current.deltas[s]
    zg_s = belief_goal.center + belief_goal.deltas[s]
    costs_s = rollout_cost(candidates, z0_s, zg_s, predictor, cost_fn)
    cost_samples.append(costs_s)

cost_samples = stack(cost_samples, dim=-1)  # [K, S]
risk_cost = aggregate_risk(cost_samples, cfg.risk, cfg.lambda_std, cfg.cvar_q)
best_idx = risk_cost.argmin()
return candidates[best_idx, 0]
```

要求：

- `original_cem` 不被破坏。
- 如果原 CEM 不返回 candidates，就在最后一轮保留 elite / sampled actions。
- rerank 只作用于最后 action selection，不改变 CEM proposal distribution。

### 7.2 Stage 2：`inner_loop`

只有 Stage 1 有正结果后再做。把每轮 CEM 的 cost 改为 robust risk cost，让 proposal distribution 本身对 uncertainty 稳健。

---

## 8. 动态适应：先留接口，Stage 2 再启用

固定 perturbation radius 可能变成另一个全局旋钮。Stage 2 增加 online uncertainty score：

```text
u = median(||Delta_i||) / clean_nn_radius
```

其中 `clean_nn_radius` 来自训练/验证集 clean latent 的最近邻尺度，优先读取已有 diagnostics JSON；若没有，Codex 实现一个离线 calibration 脚本。

动态强度：

```python
alpha = clip((u - u_low) / (u_high - u_low), 0, 1)
radius_mult = 1 + alpha * (max_radius_mult - 1)
lambda_std_t = alpha * lambda_std
```

Stage 1 默认 `dynamic_scale=false`。如果 fixed robust CEM 有正结果，再启用动态版本。

---

## 9. 可选校正：latent belief filter，暂不进入 Stage 1

如果 robust rerank 只能部分修复，再实现 dynamics-prior filter：

```text
z_prior = predictor(z_hat_prev, a_prev)
z_obs_center = encoder_tta_center(x_obs)
K = P / (P + R)
z_hat = (1 - K) * z_prior + K * z_obs_center
```

其中：

- `R` 来自 TTA latent variance。
- `P` 第一版可设常数，后续可由 predictor residual 估计。
- clean / low-noise 下应退化为 observation-dominant。
- 这部分不要和 Stage 1 混在一起。

---

## 10. 实验矩阵

### Phase 0：工程验证

只跑少量 batch / 1-2 条 episode：

| Test | 要求 |
|---|---|
| default-off | `robust.enabled=false` 与原始 CEM 输出一致 |
| shape test | candidate、latent、cost_samples shape 正确 |
| no clean leakage | TTA 只接收 observed image |
| no BN update | eval/TTA 不更新 BN running stats |
| deterministic seeds | 相同 seed 下可复现 |
| overhead logging | 记录每 step 额外 rollout 次数和 wall time |

### Phase 1：PushT 快速判定

优先任务：PushT。  
模型：LeWM-base。  
评估协议：seeds 42/43/44，每 seed `num_eval=100`。  
条件：clean、px+goal 0.05、px+goal 0.08。

候选配置：

| ID | robust_current | robust_goal | belief | risk | samples | 说明 |
|---|---:|---:|---|---|---:|---|
| base_cem | false | false | none | mean | 1 | 原始 CEM |
| compute_matched | false | false | none | mean | matched | 普通 CEM 增加等量 compute |
| rcem_curr | true | false | input_tta_empirical | cvar | 8 | 只 robust current |
| rcem_goal | false | true | input_tta_empirical | cvar | 8 | 只 robust goal |
| rcem_both | true | true | input_tta_empirical | cvar | 8 | current+goal 主配置 |
| rcem_meanstd | true | true | input_tta_empirical | mean_std | 8 | risk ablation |
| rcem_latentiso | true | true | latent_iso | cvar | 8 | latent perturbation baseline |

Phase 1 不要扫太多超参。默认：

```yaml
topk: 64
tta_num: 8
tta_noise_std: 0.005
latent_samples: 8
risk: cvar
cvar_q: 0.8
```

### Phase 2：四任务验证

若 Phase 1 有中等及以上结果，扩展到：

| Task | ckpt | conds |
|---|---|---|
| PushT | LeWM-base | clean, px+goal 0.08 |
| TwoRoom | LeWM-base | clean, px+goal 0.08 |
| Reacher | LeWM-base | clean, px+goal 0.08 |
| Cube | LeWM-base | clean, px+goal 0.08 |

只保留 Phase 1 最好的 1-2 个 robust configs。

### Phase 3：和 noise-trained ckpt / PLDM 对照

如果 Phase 2 仍有明显收益，再跑：

1. LeWM+noise point-best ckpt + robust CEM，测试是否能推进 oracle frontier。
2. PLDM-base + robust CEM，如果 PLDM baseline 已经接入。

---

## 11. 必须记录的 metrics

每个 eval run 输出：

```text
success_rate
return / score if available
num_eval
seed
condition
planner_config
wall_time_per_episode
cem_particles / cem_iterations / topk / latent_samples
```

robust planner 额外输出：

```text
belief_current_uncertainty_mean
belief_goal_uncertainty_mean
risk_cost_mean
risk_cost_std
selected_candidate_point_cost
selected_candidate_risk_cost
risk_minus_point_cost
per_step_uncertainty_trace
per_step_action_norm_trace
fallback_count
```

聚合文件建议：

```text
assets/paper1_data/robust_cem_evals_YYYYMMDD.json
assets/paper1_data/robust_cem_phase1_pusht.csv
assets/paper1_data/robust_cem_debug_traces/*.jsonl
```

---

## 12. 必做 ablation

1. **current-only vs goal-only vs both**：判断 px+goal noise 的主要瓶颈在哪端。
2. **cvar vs mean vs mean_std vs max**：证明 risk aggregation 不是简单平均。
3. **input_tta_empirical vs latent_iso**：区分真实 encoder shift ensemble 和任意 latent smoothing。
4. **fixed radius vs dynamic radius**：只有 fixed 有效后再做 dynamic。
5. **compute-matched CEM**：robust CEM 额外 rollout 必须和普通 CEM 增加 compute 比较。
6. **clean degradation guardrail**：clean success 不能明显下降。

---

## 13. 成功 / 失败判据

### Phase 1 PushT 判据

以 LeWM-base PushT px+goal 0.08 为核心。

| 结果 | 判定 | 后续 |
|---|---|---|
| px+goal 0.08 ≥ 70，clean drop ≤ 5pt，且优于 compute-matched ≥ 5pt | 强结果 | 进入 Paper1 主文候选 |
| px+goal 0.08 40–70，clean drop ≤ 5pt | 中等结果 | 作为 causal intervention / appendix，继续 Phase 2 |
| px+goal 0.08 20–40，或 clean drop 5–10pt | 弱结果 | 只保留机制分析，不作为方法主线 |
| px+goal 0.08 < 20，或 clean drop > 10pt | 负结果 | 停止扩展；说明 encoder shift 不能靠 planner smoothing 修复 |

### Phase 2 四任务判据

强结果需要满足：

1. 四任务 clean drop 平均 ≤ 3pt，单任务不超过 5pt。
2. 至少 PushT / Reacher 中一个 contact/continuous task 有 ≥ 20pt OOD 提升。
3. TwoRoom / Cube 不明显退化。
4. 同一组 planner 超参能跨任务使用，或 dynamic_scale 能减少手调。

---

## 14. Paper 定位预案

根据结果决定论文放置：

1. **强结果**：并入 Paper1 顶会版，作为 `training-free robust planning intervention`。Paper1 从纯诊断升级为“诊断 + 最小推理侧干预”。
2. **中等结果**：Paper1 主文一小节或 appendix，定位为 causal test：OOD collapse 部分来自 CEM point-estimate brittleness。
3. **弱 / 负结果**：Paper1 appendix 负结果，强化“必须训练侧改 representation robustness”的结论。
4. **方法变复杂后才有效**：不要塞 Paper1，转 Paper2 robust latent MPC 路线。

---

## 15. Codex 实现 checklist

请按以下顺序提交改动：

1. 找到原始 CEM / eval 入口，确认 default-off 可复现。
2. 加 `planner.robust.enabled=false` 配置，不改变任何默认行为。
3. 实现 `aggregate_risk`，加 unit tests。
4. 实现 `LatentBelief` 和 `build_latent_belief`。
5. 实现 `rerank_topk` wrapper。
6. 加 eval logging 到 JSON / CSV。
7. 跑 Phase 0 工程测试。
8. 跑 Phase 1 PushT 小矩阵。
9. 生成聚合表：base CEM、compute-matched、rcem_curr、rcem_goal、rcem_both。
10. 只有 Phase 1 通过中等阈值后，再做 Phase 2。

---

## 16. 最小命令模板

实际命令按仓库脚本名调整。Codex 需要把下面模板映射到现有 eval 脚本：

```bash
# default-off sanity
python tools/eval_robust_cem.py \
  --task pusht \
  --ckpt /path/to/pusht_lewm_base_epoch_10_object.ckpt \
  --cond clean \
  --seed 42 \
  --num_eval 10 \
  planner.robust.enabled=false

# Phase 1 main robust config
python tools/eval_robust_cem.py \
  --task pusht \
  --ckpt /path/to/pusht_lewm_base_epoch_10_object.ckpt \
  --cond px_goal_0.08 \
  --seeds 42 43 44 \
  --num_eval 100 \
  planner.robust.enabled=true \
  planner.robust.mode=rerank_topk \
  planner.robust.topk=64 \
  planner.robust.robust_current=true \
  planner.robust.robust_goal=true \
  planner.robust.belief_mode=input_tta_empirical \
  planner.robust.tta_num=8 \
  planner.robust.tta_noise_std=0.005 \
  planner.robust.latent_samples=8 \
  planner.robust.risk=cvar \
  planner.robust.cvar_q=0.8
```

---

## 17. 实验完成后的汇总表模板

```markdown
| Task | Ckpt | Cond | Planner | Clean | px+g 0.05 | px+g 0.08 | Clean drop | Compute matched? |
|---|---|---|---:|---:|---:|---:|---:|---|
| PushT | LeWM-base | all | Standard CEM |  |  |  |  | no |
| PushT | LeWM-base | all | Standard CEM, matched compute |  |  |  |  | yes |
| PushT | LeWM-base | all | Robust CEM current-only |  |  |  |  | yes |
| PushT | LeWM-base | all | Robust CEM goal-only |  |  |  |  | yes |
| PushT | LeWM-base | all | Robust CEM current+goal |  |  |  |  | yes |
```

---

## 18. 最重要的防错提醒

1. 不要用 clean image 构造 TTA。
2. 不要在 training loop 里改 loss。
3. 不要让 TTA forward 更新 BN running stats。
4. 不要只报 robust CEM，不报 compute-matched CEM。
5. 不要只看 px+goal 0.08，必须同时看 clean drop。
6. 不要把 input TTA mean 宣称为 clean latent correction；它只是 observed corrupted image 周围的 local uncertainty estimate。
7. 如果 top-K rerank 无效，不要直接堆复杂模块；先检查 current-only / goal-only / cost shape / candidate diversity。
