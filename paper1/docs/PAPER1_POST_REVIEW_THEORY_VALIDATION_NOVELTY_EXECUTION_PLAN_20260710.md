# Paper 1 顶会复审后：理论、外推验证、判别性与创新性整改执行计划

> **目标分支**：`qun-team/wm_exp@ag/dev`  
> **计划基线**：`bfd2c95b6c7fdfdddbd4b28bb67211d26a21ba1c`（`Polish Paper1 visuals and release checks`）  
> **日期**：2026-07-10  
> **执行对象**：Codex  
> **主原则**：不伪造结果；不手工改 canonical JSON；所有新数值必须由脚本生成、带 schema/provenance/hash，并通过一致性检查后才进入正文。

---

## 0. 本轮整改的最终目标

本轮不是继续堆同一 Gaussian sweep 的图，也不是优先增加 LeWM 训练种子，而是解决六个会直接影响顶会判断的问题：

1. 把现有 blur/resize、PLDM 和 target-view 资产组织成**真正冻结门限的多轴外推验证**；
2. 加入 `std_max`、encoder-only、one-step 等简单 baseline，回答 ACPC 是否只是读取训练噪声或 encoder smoothing；
3. 统一理论量、实验量和概率空间，修正当前 ATR/Jacobian/κ 的不一致，并提供非空的 fixed-pool 定量结果；
4. 将 `flip | cert-pass = 0` 从“独立证据”改回它应有的角色：定理/实现 sanity check；
5. 把 SMPR 从单一 median-bin proxy 提升为有敏感性分析、负对照和任务语义锚点的 guard；
6. 正面处理 2026 年并发工作带来的 novelty 压力，更新参考文献、相关工作、命名和最近方法 baseline。

完成 P0 后，论文可合理主张：

> 在 LeWM Gaussian development split 上冻结的 paired-rollout radius/guard protocol，能够在 held-out LeWM seeds、PLDM 模型族、blur/resize 扰动族和至少一个不同训练机制的失败干预上接受前瞻式检验；理论与实验使用同一个 horizon-level rollout radius；fixed-pool 结果报告可审计的 certificate coverage，而不是把描述性 q90 当成概率证书。

仍然不应主张：

- universal robustness predictor；
- 任意 corruption family 的 transfer theorem；
- adaptive CEM、重复 replanning 或闭环轨迹的形式保证；
- SMPR 已证明 oracle-level semantic sufficiency；
- “action-conditioned consistency” 这一一般概念本身是本文独占的新颖性。

---

# 1. 对六个问题的明确回答

## 1.1 问题 1：blur/resize 与 PLDM 是否已经足以回答 off-axis validation？

### 结论

**两者已经是很好的外推证据，但按当前分析方式还不能完整回答疑问。**它们分别覆盖两个不同的轴：

| 现有资产 | 覆盖的外推轴 | 已经能说明什么 | 目前不能说明什么 |
|---|---|---|---|
| LeWM blur/resize，3 training seeds | **扰动族轴** | Gaussian-trained endpoint 在未训练 blur/resize 下，行为和 stressor-specific ATR/SMPR 有正、弱、负三类一致变化；不是只看 matched Gaussian | 当前主要是 base/end 两点和相关性；并未把 Gaussian 上选定的**同一个 frozen gate**无调参地应用到全部 blur/resize rows |
| PLDM 4 tasks × 9 Gaussian sweep，1 training seed | **模型族轴** | 相同 Gaussian intervention 在另一 JEPA latent-dynamics model 上也出现任务相关 cliff/recovery/diagnostic movement | 只有一个 PLDM training seed；且仍使用相同 Gaussian augmentation repair，不是“不同修复机制” |
| target-view ablation | **训练机制/失败机制轴** | `perturbed history -> original future` 是与 full-sequence perturbed-target 不同的训练设计，并在 noisy closed loop 上明显失败 | 当前还没有用最新 ATR/SMPR frozen gate 对这组 checkpoints 做同协议诊断 |

因此，最低充分包不是“blur/resize + PLDM”两个数据集直接宣称已证明，而是：

1. **LeWM Gaussian seed 3072**：只用于选 protocol 和 gate；
2. **LeWM Gaussian seeds 3073/3074**：training-seed confirmation；
3. **PLDM Gaussian full sweep**：模型族 transfer，绝不重调门限；
4. **LeWM blur/resize full 24-row slice**：扰动族 transfer，绝不重调门限；
5. **target-view ablation**：训练机制 falsification，绝不重调门限。

这五部分足以支持一个收敛的顶会说法：

> 该 diagnostic 的 movement 不只存在于 LeWM matched-Gaussian held-out seeds；它还接受了模型族、扰动族和失败训练机制三个轴的 frozen-protocol 检验。

但仍不能支持“不同成功修复算法上普遍有效”。要升级到这一层，P1 再加入 DINO-WM、Delta-JEPA 或其他正向非 Gaussian repair。

### 为什么当前 blur/resize 表还不等于 frozen-threshold validation

当前主文表已经很有价值：TwoRoom/Reacher blur 同时有行为增益与 ATR/SMPR 改善，PushT resize 近似不动，Cube resize 行为略负且 diagnostic 也反向。问题不在结果，而在 protocol：

- 当前 Gaussian held-out gate 脚本会在每个 leave-one split 上重新搜索阈值；
- blur/resize 只有 endpoint pair，没有 recovery onset；
- 现有 unseen summary 中的 composite rule 主要来自旧 ACPC/PCC/CRA/MAF，不是当前 paper-facing ATR/SMPR gate。

整改后应保留现有表作为 scope evidence，同时新增严格的 frozen external validation artifact。

---

## 1.2 问题 2：既然 `std_max` 在不同任务上的敏感度明显不同，还需要单独比较吗？

### 结论

**需要，但把它定位为“元数据 confound control”，不是可部署 diagnostic baseline。**

任务曲线不同只能说明“一个 universal `std_max` 阈值不合理”，不能证明 ATR/SMPR 相对 `std_max` 或 encoder q90 有增量。Reviewer 仍可以提出两个反解释：

1. ATR/SMPR 只是 `std_max` 的单调函数；
2. ATR 的效果完全由 encoder clean/noisy q90 决定，action-conditioned rollout 没有额外价值。

必须用一个小而严格的 baseline audit 消除这两个疑问。这里最重要的不是让 ACPC 在 LeWM Gaussian 内赢过 `std_max`，而是看它是否能在**模型/扰动/训练机制外推**时继续工作：

- `std_max` 在 target-view 高噪声 checkpoints 上仍然很大，但行为可能仍然失败；这正是区分“训练元数据”与“checkpoint 实际 predictive behavior”的理想 falsification；
- encoder q90 若在 blur/resize、PLDM、target-view 上与 ATR 一样好，则必须收缩“rollout is empirically necessary”的说法；
- 若 ATR/SMPR 在外推轴上明显优于 encoder-only，则中心主张才获得真正的增量证据。

### baseline 最小集合

| ID | baseline | 是否 training-free | 回答的问题 |
|---|---|---:|---|
| B0 | `std_max` | 否，训练元数据 | diagnostic 是否只是在反推 augmentation strength？ |
| B1 | clean closed-loop score | 否，行为元数据 | 是否仅靠 clean quality 就能预测 corruption behavior？ |
| B2 | encoder clean/noisy q90，按 clean NN scale 归一化 | 是 | action-conditioned rollout 是否比 encoder invariance 增量更强？ |
| B3 | one-step paired prediction q90 | 是 | multi-step horizon 是否必要？ |
| B4 | H-step paired rollout q90，但 action shuffled/zeroed | 是 | 改善是否真的依赖正确 action intervention？ |
| B5 | ATR only | 是 | radius 单独有多强？ |
| B6 | SMPR only | 是 | guard 单独有多强？ |
| B7 | frozen ATR+SMPR | 是 | 论文主 diagnostic |
| B8 | fixed-pool cost-drift / top-1 audit | 是，但 planner-coupled | planner-side 上限参照，不作为轻量 baseline |
| B9 | ATM-style action-transfer probe（P1） | 是 | 与最直接并发 diagnostic 比较 |

### 决策规则

- 若 B7 只在 LeWM Gaussian 内有效，而在 PLDM/blur/resize/target-view 上不优于 B0/B2，则删去“action-conditioned rollout is empirically necessary”，改为“a control-facing formulation and one useful diagnostic family”。
- 若 B7 在至少两个 external axes 上明显优于 B0/B2，保留强中心主张。
- 若 B3 与 B5 持平，弱化 multi-step 必要性；若 B5 在长 horizon 或 contact-heavy tasks 上更稳，保留 H-step 论证。
- 若 B4 与 B5 没差异，说明 action conditioning 未被实验证明有增量，必须增加 action-distinct guard 或收缩命名。

---

# 2. 统一的冻结协议与数据分区

## 2.1 新建唯一 protocol lockbox

创建：

```text
paper1/config/frozen_diagnostic_protocol_v1.json
```

只能使用以下 calibration data：

```text
model_family = LeWM
training_seed = 3072
training_stressor = Gaussian input noise
rho_grid = 0.00 ... 0.08
tasks = TwoRoom, PushT, Reacher, Cube
```

文件至少记录：

```json
{
  "schema_version": "paper1-frozen-diagnostic-protocol-1.0",
  "calibration_source": "LeWM seed3072 Gaussian full sweep",
  "radius_metric": "horizon_weighted_l2_v2",
  "rollout_horizon": 8,
  "horizon_weights": "uniform",
  "atr_quantile": 0.90,
  "normalization": "clean_transition_scale",
  "smpr_pair_rule": "task_grounded_near_boundary_v2",
  "smpr_margin_delta_normalized": 0.10,
  "smpr_local_quantile": 0.35,
  "tau_atr": "generated from calibration only",
  "tau_smpr": "generated from calibration only",
  "behavior_label_rule": {
    "gaussian_sweep": "80% recovery-to-task-grid-max plus <=5pp clean drop",
    "external_endpoint": ">=80% stress-score retention relative to clean score"
  },
  "frozen_at_utc": "...",
  "source_hashes": {}
}
```

阈值必须由 builder 写入，不能人工抄数。冻结后：

- seeds 3073/3074 的 label 不得参与阈值选择；
- PLDM 的任何结果不得参与阈值选择；
- blur/resize 的任何结果不得参与阈值选择；
- target-view 的任何结果不得参与阈值选择；
- external split 上只允许读 protocol，不允许调用 threshold search。

当前 `paper1/scripts/heldout_diagnostic_validation.py` 保留为 retrospective cross-validation，不再把它描述为“一个全局 frozen gate”。另建严格外推脚本。

## 2.2 验证矩阵

| split | 模型 | 训练机制 | evaluation stressor | 用途 | 主要指标 |
|---|---|---|---|---|---|
| CAL | LeWM seed3072 | Gaussian full-sequence | Gaussian | 只选 metric/gate | calibration objective |
| E1 | LeWM seeds3073/3074 | Gaussian full-sequence | Gaussian | training-seed confirmation | onset MAE、AUPRC、precision/recall |
| E2 | PLDM seed canonical | Gaussian full-sequence | Gaussian | model-family transfer | per-task onset error、AUPRC、signed rank |
| E3 | LeWM 3 seeds | Gaussian full-sequence | strongest blur/resize | stressor-family transfer | retention classification、continuous delta correlation、pairwise ordering |
| E4 | LeWM target-view runs | perturbed history → original future | Gaussian | mechanism falsification | false-pass rate、behavior/diagnostic mismatch |
| E5（P1） | DINO-WM/Delta-JEPA | different model/objective | Gaussian/other | positive mechanism transfer | frozen-gate metrics |

## 2.3 新建脚本与输出

创建：

```text
paper1/scripts/freeze_diagnostic_protocol.py
paper1/scripts/frozen_external_validation.py
paper1/scripts/diagnostic_baseline_comparison.py
paper1/scripts/plot_frozen_external_validation.py
```

输出：

```text
paper1/results/frozen_diagnostic_protocol_calibration.json
paper1/results/frozen_external_validation_rows.csv
paper1/results/frozen_external_validation_summary.csv
paper1/results/diagnostic_baseline_comparison.csv
paper1/tables/table_frozen_external_validation.tex
paper1/tables/table_diagnostic_baselines.tex
assets/paper1_figs/fig_frozen_external_validation.png
```

所有输出写入：

- exact source artifact path；
- source SHA-256；
- protocol JSON SHA-256；
- 是否 calibration/test；
- 是否出现缺失 checkpoint；
- 运行参数；
- 不允许自动忽略 error row。

---

# 3. P0-A：off-axis validation 的具体执行

## 3.1 LeWM held-out seeds：确认而非重新调参

### 输入

```text
assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json
assets/paper1_data/acpc_phase0_lewm_three_seed.json
assets/paper1_data/semantic_task_grounded_margin_lewm_full_sweep_20260708.json
paper1/results/prospective_diagnostic/diagnostics_all_ckpts.csv
```

### 操作

1. 用 seed3072 CAL rows 选择一次 `tau_atr/tau_smpr`；
2. 将同一 gate 应用于 seeds3073/3074；
3. 报告每个 task-seed block 的：
   - behavioral onset；
   - predicted onset；
   - onset error；
   - precision/recall；
   - false early / false late；
4. 结果不得与现有 leave-one-seed-out 重新调参结果混写。

### 成功标准

- 不要求每个 block 完美；
- 主要标准是平均 onset error 不超过两格，并公开全部错误方向；
- 若严格 frozen gate 明显差于当前 cross-validation，正文必须使用严格结果，现有表降为 sensitivity appendix。

## 3.2 PLDM full sweep：模型族 transfer

### 现有输入

```text
assets/paper1_data/canonical_evals_pldm_20260522.json
assets/paper1_data/canonical_diagnostics_pldm_20260522.json
assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json
assets/paper1_data/acpc_basin_diagnostics_pldm.json
assets/paper1_data/acpc_phase0_clean_goal_seed9101.json
```

PLDM 已有完整 `4 tasks × 9 configs`，因此 P0 不新增 PLDM 训练。需要做的是把它升级到与当前 paper-facing protocol 相同：

1. 扩展 `tools/paper1_semantic_margin.py`：
   - 新增 `--method {LeWM,PLDM}`；
   - 新增 `--evals`/`--manifest`；
   - 不再硬编码 `lewm_seed*.json`；
2. 用 canonical v2 horizon radius 重算 PLDM ATR；
3. 用 frozen LeWM gate 原样预测 PLDM recovery rows；
4. 报告每任务 onset error 和整体 AUPRC；
5. 单独标注：PLDM 只有一个 training seed，结论是**model-family transfer direction**，不是 PLDM training-run stability。

### 失败判据

- 若 frozen gate 在 PLDM 上系统性提前/滞后超过两格，不能写 cross-model diagnostic validity；
- 若 ATR 有方向但 SMPR 不稳定，改写为 radius transfer、guard method-dependent；
- 不得用 PLDM label 重新选阈值后再称 external validation。

## 3.3 blur/resize：扰动族 transfer

### 输入

```text
assets/paper1_data/unseen_origin_vs_std008_strongest_s3072.json
assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json
assets/paper1_data/unseen_origin_vs_std008_strongest_s3074.json
assets/paper1_data/unseen_phase0_acpc_fullstress.json
assets/paper1_data/unseen_atr_smpr_summary_20260707.json
assets/paper1_data/prospective_validation_summary.json
```

### 不能沿用 Gaussian onset MAE 的原因

blur/resize 当前是 base vs `std_max=0.08` endpoint，不是 9 点训练 sweep。因此不应人为制造 onset。采用两个冻结指标：

1. **绝对 retention label**：

```text
stress_score / clean_score >= 0.80
```

2. **连续 pair response**：

```text
Δbehavior = stress_score(noise-trained) - stress_score(base)
ΔATR      = ATR(base) - ATR(noise-trained)       # 正数为改善
ΔSMPR     = SMPR(noise-trained) - SMPR(base)     # 正数为改善
```

报告：

- frozen gate pass 与 retention label 的 balanced accuracy/AUPRC；
- `Δdiagnostic` 与 `Δbehavior` 的 Spearman/Pearson，仅作为小样本描述；
- task × stress family × seed 的 signed agreement；
- 所有 24 rows，不只挑 TwoRoom/Reacher 正例与 PushT/Cube 边界例。

### 关键限制

- 不把 blur/resize 结果写成 ACPC universal transfer；
- 不重新搜索 stressor-specific thresholds；
- stressor-specific ATR/SMPR 可以重算，但 gate 必须来自 Gaussian CAL；
- 若 scale shift 导致所有 ATR 超阈值，优先检查 normalization 是否真正 method/stressor invariant，而不是调阈值。

## 3.4 target-view ablation：修复机制 falsification

### 现有行为资产

```text
assets/paper1_data/target_view_closed_loop_summary.json
```

其中 `perturbed_history_to_original_future` 与 full-sequence perturbed-target 使用不同训练目标；在 PushT/Reacher 等任务上，前者 noisy closed-loop 明显失败。这组数据最适合回答：

> 高 `std_max` 是否必然意味着 diagnostic 会判为 recovered？ACPC 是否能识别一个“用了噪声训练但修复机制失败”的 checkpoint family？

### 新建

```text
tools/paper1_target_view_diagnostic_manifest.py
paper1/scripts/target_view_frozen_gate_validation.py
```

### 操作

1. 从已有 32+32 target-view/full-sequence runs 构建 canonical-shaped manifest；
2. 对相同 task/std checkpoint 计算 canonical v2 ATR 和 SMPR；
3. frozen gate 不做任何重调；
4. 与 B0 `std_max`、B2 encoder q90、B3 one-step、B7 ATR+SMPR 比较；
5. 报告：
   - false-pass rate；
   - behavior collapse but gate pass 的 counterexample rows；
   - gate fail but behavior pass 的 false negatives；
   - matched std pairwise ordering。

### 解释规则

- **理想但不得预设**：`std_max` 在高噪声 target-view runs 上误判，而 ATR+SMPR 能识别失败；
- 若 ATR+SMPR 也误判，这不是删数据的理由，而是论文最重要的边界结果；
- 若只有 fixed-pool cost drift 能识别，说明 ATR/SMPR 还不足以承担 planner-facing claim；
- target-view 是失败机制验证，不是成功替代 repair 的证明。

## 3.5 P1 正向不同机制

P0 完成后再决定是否新增：

- DINO-WM no-prop：PushT/TwoRoom，base + `{0.02,0.05,0.08}`，共 8 runs；
- Delta-JEPA：若作者代码/checkpoints 可用，优先作为 action-sensitive objective；
- ACID：属于 decision-time planner intervention，适合作为 planner-side external mechanism，不适合作为 representation diagnostic 的唯一外部模型。

不在代码/权重未确认时把它们写成已完成实验。

---

# 4. P0-B：简单 baseline 与增量价值审计

## 4.1 新建统一 row schema

`diagnostic_baseline_comparison.py` 对每个 task/model/seed/checkpoint/stressor 输出：

```text
std_max
clean_score
stress_score
behavior_label
encoder_q90_norm
one_step_q90_norm
horizon_atr_q90_norm
action_shuffled_horizon_q90_norm
smpr
frozen_gate_pass
fixed_pool_flip
fixed_pool_cert_score
split_name
```

所有表示诊断必须：

- 同一批 anchors；
- 同一 corruption draws；
- 同一 embedding/cost space；
- 同一 clean transition/NN normalization；
- 同一 horizon；
- 同一 checkpoint；
- 不允许某个 baseline 使用更多数据。

## 4.2 比较协议

### 阈值型比较

每个 baseline 的阈值只在 CAL 上选一次，然后应用 E1–E4。报告：

- AUPRC；
- balanced accuracy；
- precision/recall；
- onset MAE（仅 sweep split）；
- external false-pass rate；
- pairwise ordering accuracy。

### 增量型比较

使用 block bootstrap（task/seed/model 为 block）比较：

```text
B2 encoder-only
B3 one-step
B5 ATR-only
B7 ATR+SMPR
```

不要在 108 rows 上做普通 iid p-value。可以报告：

- metric difference 的 block-bootstrap 95% interval；
- 每个 external split 的 raw value；
- 不把小样本 CI 写成“显著”。

### action intervention test

对相同 clean/noisy histories：

1. 正确 recorded action；
2. batch 内 action permutation；
3. zero action（任务允许时）；
4. 时间顺序 shuffle。

若正确 action 下的 ATR 比 shuffled action 更能解释 behavior/flip，支持 action-conditioned 增量。若没有差异，收缩论文表述。

## 4.3 论文中的结果决策

| 结果 | 论文处理 |
|---|---|
| ATR+SMPR 在 E2/E3/E4 优于 `std_max` 和 encoder q90 | 强化 “checkpoint behavior, not augmentation metadata” |
| ATR 与 encoder q90 持平 | 将 encoder geometry 提升为共同主角，删除“rollout is necessary”的经验强说法 |
| one-step 与 H-step 持平 | 将 H-step 降为理论自然扩展，不声称实证必要 |
| action shuffled 与正确 action 持平 | 删除 action-conditioned 独立增量主张，或增加 action-distinct guard |
| target-view 上所有 lightweight diagnostics 都失败 | 明确 ACPC 只 localize matched repair，不是外部 selector；保留 fixed-pool/closed-loop authority |

---

# 5. P0-C：理论与实验的四项一致性整改

## 5.1 子问题 1：q90 与 `Kα` 空洞——改成非空的 sample-level certificate coverage

### 5.1.1 保留但降级当前 union-bound theorem

当前

```text
P(flip) <= K alpha + P(margin <= 2 L_J epsilon)
```

可以留在 appendix 作为 tail motivation，但正文不得把 ATR q90 代入其中。明确：

- ATR q90 是描述性 checkpoint diagnostic；
- theorem 的单候选 `alpha` 不是 `0.1` 的直接替代；
- `K=65` 时朴素代入为空洞；
- q90 解释“为什么看 tail”，不产生数值概率证书。

### 5.1.2 引入更锐利的 deterministic candidate-wise certificate

设 clean winner 为 `j*`，定义：

```math
\Delta_j = C_h(a^j)-C_h(a^{j^*}),
\qquad
d_j = |C_{\tilde h}(a^j)-C_h(a^j)|.
```

若对所有 `j != j*`：

```math
\Delta_j > d_j+d_{j^*},
```

则 top-1 不变。定义 sample slack：

```math
S_{\rm sharp}(h,\tilde h,\mathcal A)
=
\min_{j\ne j^*}\left[\Delta_j-d_j-d_{j^*}\right].
```

`S_sharp > 0` 即 cert-pass。它严格包含当前较粗条件：

```text
max_j d_j < top1_top2_margin / 2
```

修改 `tools/paper1_sample_level_certificate.py`，同时输出：

```text
coarse_cert_pass
sharp_cert_pass
sharp_cert_slack
flip
flip_when_sharp_cert_fail
coverage_by_K
```

### 5.1.3 报告可解释、非空的风险上界

因为 deterministic theorem 给出：

```math
\{\text{flip}\}\subseteq\{S_{\rm sharp}\le0\},
```

所以在明确的 sampled fixed-pool distribution 下：

```math
P(\text{flip})\le 1-P(S_{\rm sharp}>0).
```

实验报告：

- `p_cert`；
- 一侧 95% lower confidence bound `p_cert,L`；
- 因而 `flip-risk upper = 1 - p_cert,L`；
- observed flip rate 仅作为 sharpness 对照。

统计口径：

- checkpoint 内 anchors 的 one-sided binomial interval；
- task/seed/checkpoint 层面另做 block bootstrap；
- 不把所有 anchors 混成完全 iid 的一个超大 `n`；
- 主文给 coverage 与 risk upper，appendix 给分层结果。

### 5.1.4 ACPC radius 到 cost drift 的可选 P1 校准

直接 cost certificate 是 planner-coupled。若要让 ACPC radius 真正进入数值证书，新增 calibration：

```math
L_{\rm emp} = Q_{0.99}^{\rm CAL}
\left(\frac{|C_h(a)-C_{\tilde h}(a)|}{R_H(h,\tilde h,a)+\epsilon}\right).
```

在 E1–E4 冻结验证 exceedance rate，构造：

```math
\widehat d_j=L_{\rm emp}R_H(h,\tilde h,a^j).
```

只有外部 exceedance 得到覆盖支持时，才报告 ACPC-only proxy certificate。否则将 direct cost certificate 与 ATR 机制分析明确分开。

### 输出

```text
paper1/results/fixed_pool_candidatewise_certificate.csv
paper1/results/fixed_pool_certificate_coverage_by_block.csv
paper1/tables/table_fixed_pool_certificate_coverage.tex
assets/paper1_figs/fig_fixed_pool_certificate_calibration.png
```

---

## 5.2 子问题 2：统一 probability space

### 5.2.1 在理论中显式定义随机变量

```math
H\sim\mu_{\rm task},
\qquad
\Xi\sim P_\tau(\cdot\mid H),
\qquad
A_{\rm rec}\sim\nu(\cdot\mid H),
\qquad
\mathcal A=(A^1,\ldots,A^K)\sim q(\cdot\mid H)^K.
```

另定义 task-different pair：

```math
(H,H')\sim\pi_{\rm diff}.
```

区分三个风险：

```math
\alpha_R(\epsilon)
= P_{H,\Xi,A_{\rm rec}}[R_H>\epsilon],
```

```math
\beta_{\rm plan}
= P_{H,\Xi,\mathcal A}[\text{fixed-pool top1 flip}],
```

```math
\beta_{\rm guard}
= P_{H,H',\Xi}[M_{\rm diff}\le R_H+\delta].
```

### 5.2.2 修改 selective theorem

当前将 planning risk 与 guard risk 直接相加，容易让读者误以为实证数据来自同一 joint sample。整改二选一：

**推荐方案**：

- theorem 只保留 fixed-pool planning statement；
- guard 另列 proposition/definition；
- empirical selective region 定义为两个 independently audited criteria 的 conjunction；
- 不报告一个伪装成校准总概率的 `beta_plan + beta_guard`。

只有未来确实构造同一 history/noise/candidate/pair joint protocol 时，才恢复 union bound。

### 5.2.3 新增 theory-to-data map

正文或 appendix 表必须逐项写：

| quantity | random source | empirical unit | current estimator | claim |
|---|---|---|---|---|
| horizon radius | history × noise draw × recorded action | anchor | q90 | diagnostic tail |
| cert coverage | history × noise draw × fixed pool | anchor/pool | pass rate + one-sided CI | sampled fixed-pool risk upper |
| guard failure | task-different pair × noise | pair | SMPR | proxy/oracle guard |
| closed-loop score | training seed × eval seed × trajectory | checkpoint | success mean/std | behavioral authority |

---

## 5.3 子问题 3：统一 ACPC metric 与 Jacobian 命题

### 5.3.1 当前实现审计结论

必须在整改记录中明确：当前 `compute_acpc_prediction_metrics` 调用 `_shift_stats(pred_clean, pred_noisy)`；后者把 `(B,H,D)` reshape 为 `(B×H,D)` 后做 q90。因此当前 `acpc_h_l2_p90` 更接近**step/token-level rollout disagreement tail**，不是 theorem 中每条 rollout 的 `d_H` horizon distance。

这不是简单改文字能解决的，需要 canonical v2 metric。

### 5.3.2 定义 canonical horizon-level radius

创建公共模块：

```text
tools/paper1_acpc_metrics.py
```

定义：

```math
\bar G_{\mathbf a}(z)
=
\left[
\sqrt{\alpha_1}\Pi(\hat z_{t+1});\ldots;
\sqrt{\alpha_H}\Pi(\hat z_{t+H})
\right],
```

```math
R_H^{(2)}(i)
=
\frac{
\|\bar G_{\mathbf a_i}(E(h_i))-
  \bar G_{\mathbf a_i}(E(\tilde h_i))\|_2
}{s_i+\epsilon}.
```

ATR 为 anchors 上的：

```math
\mathrm{ATR}_q=Q_q\{R_H^{(2)}(i)\}_{i=1}^n.
```

要求：

- default `H=8`；
- uniform `alpha_k=1/H`，写入 artifact；
- normalization `s_i` 明确定义，不能一处用 per-token transition median、一处用全局 median；
- 每个 anchor 一条 radius；
- 多 noise draw 时先形成 anchor-level conditional tail，再聚合；
- 旧 per-step q90 保留为 `stepwise_rollout_q90`，只进 appendix/compatibility，不再叫 theorem-aligned ATR。

### 5.3.3 统一 JVP map

`tools/paper1_jvp_hutchinson_sensitivity_audit.py` 中 composed output 必须使用与 `bar G` 完全相同的：

- rollout steps；
- horizon weights；
- projection；
- embedding space；
- normalization前的 vectorization。

然后 proposition 才能写：

```math
E\|\bar G(E(o+\xi))-\bar G(E(o))\|_2^2
=
\sigma^2\|J_{\bar G}J_E\|_F^2+O(\sigma^3).
```

### 5.3.4 线性化实证校准

对 small sigmas `{0.0025,0.005,0.01,0.02}`：

- empirical `E[R^2]/sigma^2`；
- JVP trace estimate；
- ratio/relative error；
- remainder growth；
- base/onset/endpoint；
- 每任务每 seed。

不要只比较 endpoint/base 同方向；增加“JVP 是否能预测 measured local slope”的 scatter 和 calibration error。

### 5.3.5 horizon/quantile sensitivity

至少重算：

```text
H ∈ {1, 2, 4, 8}
q ∈ {0.80, 0.90, 0.95}
```

若 H=1 与 H=8 无差异，论文不再声称 long-horizon 是必要成分；若 PushT/Cube 随 H 增强，则作为 task-dependent result。

---

## 5.4 子问题 4：修正 κ 定义与表格

### 当前不一致

正文公式：

```math
\kappa_{\rm sub}
=
\frac{\|J_GJ_E\|_F^2}
{\|J_G\|_F^2\|J_E\|_F^2+\epsilon},
```

按 Frobenius 次乘性应不超过 1。

当前代码实际计算：

```math
\kappa_{\rm rel}
=
\frac{d_z\|J_GJ_E\|_F^2}
{\|J_G\|_F^2\|J_E\|_F^2},
```

因为 rollout trace 先除以 latent dimension，所以可以超过 1。表格 caption 使用的是第二个定义。

### 推荐整改

1. 主文不再把 κ 作为主要机制量；主文只报告：
   - encoder trace；
   - rollout trace；
   - composed trace；
2. appendix 同时输出：

```text
kappa_submultiplicative   # 理论上 <= 1
kappa_relative_isotropic # 乘 d_z，可 > 1
```

3. 将第二个重命名为：

```text
relative isotropic alignment gain
```

不得叫 angle/cosine/certificate；
4. 正文公式、代码、CSV 字段、table caption 四处完全一致；
5. 新增 synthetic linear-map unit test：
   - exact matrix norms 对照 estimator；
   - `kappa_sub <= 1 + tolerance`；
   - `kappa_rel == d_z * kappa_sub`；
6. Hutchinson noise 可能让比值略越界时，报告 estimator CI，不静默 clip。

### 修改文件

```text
paper1/main.tex
tools/paper1_jvp_hutchinson_sensitivity_audit.py
paper1/tables/table_jvp_hutchinson_sensitivity_audit.tex
paper1/scripts/plot_gaussian_sensitivity_mechanism.py
tests/test_paper1_jvp_alignment_metrics.py
```

---

# 6. P0-D：`flip | cert-pass = 0` 的整改

## 6.1 论文措辞

从主文删除/替换以下推理：

```text
flip conditioned on cert-pass is zero,
therefore the sufficient event is empirically conservative rather than overfit.
```

改成：

> By construction, cert-pass is a deterministic sufficient condition for preserving the clean winner on the shared candidate pool. The observed zero conditional flip rate is therefore an implementation check, not independent empirical evidence. The informative quantities are certificate coverage, observed flip risk outside the certified subset, and how both change across checkpoints.

中文执行含义：

- `flip | cert = 0` 只在 appendix/checker 中作为 theorem implementation sanity；
- 主图不画全零 conditional rate；
- 主文强调 coverage 与 observed flip；
- `flip | cert-fail` 才反映 certificate sharpness。

## 6.2 新主结果

主图建议三项，最多两 panel：

### Panel A

```text
coarse certificate coverage
sharp candidate-wise certificate coverage
```

### Panel B

```text
observed top-1 flip rate
flip rate among certificate-fail anchors
```

appendix 报告：

- `flip | cert-pass = 0` sanity；
- certificate slack distribution；
- one-sided risk upper；
- q10/q95 negative gap 作为 quantile aggregation 过保守的负结果。

## 6.3 `K` 敏感性

从相同 65-candidate pool 构造 nested pools：

```text
K ∈ {8, 16, 32, 65}
```

要求：

- 相同 random seed；
- expert candidate 始终保留；
- random candidate 顺序固定；
- 报告 coverage、flip、sharpness 随 K 的变化；
- 不用不同 K 各自重新采样后直接比较。

这会实证连接 theorem 中 pool size，而不是只在公式里出现 K。

## 6.4 hierarchical uncertainty

当前 Wilson interval 以 pooled anchors 为单位，会低估 task/checkpoint/seed cluster。新增：

- checkpoint 内 binomial interval；
- seed/task block bootstrap；
- main text 用 block-level interval；
- pooled Wilson 仅 appendix 作为 measurement precision。

## 6.5 adaptive CEM（P1）

基于现有 `cem_trace_audit`，新做 common-random-number audit：

- clean/noisy 从同一初始 Gaussian candidate samples 出发；
- 每轮记录 elite overlap、distribution mean/cov drift、first-action difference；
- 最重要指标是 final first-action agreement，不是内部 candidate index 是否一致；
- 报告 iteration 0 fixed-pool 与 later adaptive iterations 的差异；
- 仍不称 closed-loop guarantee。

---

# 7. P0-E：SMPR anti-collapse / discriminability 整改

## 7.1 先修正文中的机制解释

现有数据表明 noise-trained endpoint 的 task-different rollout distance大体保持，而 same-state noisy radius 大幅下降。SMPR pass rate 上升主要是“tube contraction while separation is retained”，不是不同状态 margin 被训练得越来越大。

因此正文从：

```text
SMPR guard improves
```

优先改为：

```text
task-grounded separations remain outside the contracted nuisance tube
```

这更符合 selective consistency，也避免把 pass-rate 的比值效应误写成 semantic margin growth。

## 7.2 参数敏感性网格

扩展 `tools/paper1_semantic_margin.py`，支持：

```text
noise_draws ∈ {1, 5}
radius_q ∈ {0.80, 0.90, 0.95}
local_state_quantile ∈ {0.10, 0.25, 0.35, 0.50}
margin_delta_norm ∈ {0.00, 0.05, 0.10, 0.25}
label_binning ∈ {median, quartile, fixed_physical}
```

`margin_delta_norm` 乘 clean transition scale，避免绝对 latent scale 不可比。

每个设置必须分开报告：

```text
same-state radius distribution
semantic-different distance distribution
raw margin distribution
pass rate
pair count
skipped anchor count
```

不能只给 SMPR 一个数。

## 7.3 负对照与正对照

创建：

```text
paper1/scripts/smpr_controls.py
```

至少包含：

| control | 目的 | 预期用途，不预设数值 |
|---|---|---|
| constant/collapsed rollout | 检查 low ATR 是否会被 SMPR 拒绝 | strict `diff > radius + delta` 应使 guard 失败 |
| identical clean/noisy positive control | 检查 radius=0 时真实 different pairs 是否通过 | 校验实现与 label coverage |
| state-label permutation | 检查 task labels 是否比随机标签更贴近 behavior | 比较 behavior association，而非强求 pass rate 必然更低 |
| action permutation | 检查 guard 是否保留 action relevance | 与正确 shared action 对照 |
| same-label nearest neighbor | 检查 label-crossing pair 是否确实更 task-distinct | 应报告 effect size |
| global far-neighbor | 揭示“随便找很远状态就高 SMPR”的 triviality | near-boundary 应比 far-neighbor 更严格 |

### 必须通过的最小 correctness gate

- constant collapse 在所有任务的 positive-margin SMPR 上不能被判为 robust；
- pair count/skip rate 不得因参数变化悄然降到很小；
- random labels 若与真实 labels 一样好，不能保留“task-grounded”强说法；
- near-boundary 与 far-neighbor 结果必须同时展示，避免选择容易通过的 pairs。

## 7.4 更强任务语义标签

### TwoRoom

新增：

- room identity；
- doorway side；
- 是否必须过门才能到 goal；
- shortest-path/topological cost difference；
- next useful action region。

优先从 simulator state/map geometry 程序生成，不做人手标注。

### PushT

新增：

- pusher–T contact/non-contact；
- object pose relative to goal；
- keyframe/contact mode；
- clean planner candidate-cost vector；
- clean top-1 action/first-action region。

最低可行 oracle：用 simulator state 计算 object-goal pose cost 和 pusher-object distance，构造近状态但 cost/optimal candidate 不同的 pairs。

### Reacher

新增：

- end-effector–target distance；
- target quadrant；
- joint configuration aliasing；
- one-step oracle reward/cost；
- clean top-1 candidate difference。

### Cube

新增：

- gripper–cube contact/grasp mode；
- cube–goal position/orientation error；
- grasped/ungasped topology；
- clean planner cost-to-go；
- top-1 candidate difference。

## 7.5 planner/value-grounded guard

新增一个比 state bins 更直接的 guard，暂不另造大术语：

选择满足以下任一条件的 near-state pairs：

```math
|V_{\rm oracle}(s_i)-V_{\rm oracle}(s_j)|>\tau_V,
```

或

```math
\arg\min_a C(s_i,a)\ne\arg\min_a C(s_j,a).
```

然后检查其 clean projected rollout distance 是否超过 same-state noise radius + positive margin。

这会直接回答“不同状态是否真的导致不同控制决策”，是最强 guard 版本。

## 7.6 输出与 claim 决策

输出：

```text
assets/paper1_data/smpr_sensitivity_v2_20260710.json
assets/paper1_data/smpr_controls_v2_20260710.json
assets/paper1_data/smpr_oracle_guard_v2_20260710.json
paper1/tables/table_smpr_sensitivity.tex
paper1/tables/table_smpr_controls.tex
assets/paper1_figs/fig_smpr_radius_margin_decomposition.png
```

决策：

- oracle/value guard 成功：主文可说 task/planner-grounded discriminability preservation；
- 只有 programmatic bins 成功：继续写 proxy-level anti-collapse guard；
- constant collapse 被拒绝但 random labels 同样好：只能说 generic non-collapse，不说 task-grounded；
- behavior 与 SMPR 在 external axes 不共变：SMPR 降到 appendix，主 diagnostic 改为 radius + fixed-pool planner guard。

---

# 8. P0-F：2026 并发工作与参考文献整改

## 8.1 必须新增的五篇工作

截至 2026-07-10：

| 工作 | 日期 | 与本文的直接重叠 | 本文应强调的差异 | 需要的实验/写作回应 |
|---|---|---|---|---|
| MWM: Mobile World Models for Action-Conditioned Consistent Prediction | 2026-03-08 | 明确使用 Action-Conditioned Consistency，关注 multi-step rollout 与 planning | MWM 是生成式导航 world model 的 self-forcing/post-training；本文是 fixed checkpoint 上 clean/corrupted same-state paired diagnostic + guard | 不再把一般“action-conditioned consistency”命名本身当 novelty；related work 正面对比 |
| ATM: Action-Consistency Transfer Matrix for Diagnosing and Improving Latent World Models | 2026-06-08 | post-hoc、轻量、action-consistency diagnostic，可筛 checkpoints/world models | ATM 比较 real encoded vs model-predicted transition 中的 action semantics；本文比较同状态视觉扰动在 shared action 下的 rollout radius，并有 nuisance/guard/fixed-pool链条 | 最强直接 novelty risk；P1 做 ATM-style baseline 或至少 action-transfer probe |
| Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding | 2026-06-30 | action sensitivity、anti-collapse、latent transition geometry | Delta-JEPA 是训练目标/模型；本文是 no-retraining robustness diagnostic | 作为正向不同机制 external model 的优先候选；相关工作说明互补 |
| ACID: Action Consistency via Inverse Dynamics for Planning with World Models | 2026-07-02 | inverse-dynamics action consistency、planner candidate validity | ACID 是 decision-time planner intervention；本文诊断 observation corruption 对 fixed checkpoint 的影响 | planner-side related work；inverse-dynamics residual 可作 baseline |
| Imagined Rollouts are Kinematic, Not Dynamic | 2026-07-07 | 提出 world-model rollout diagnostic 与 perturbation protocol，并展示 diagnostic/behavior 可脱钩 | iKCE 诊断 kinematic-vs-dynamic failure；本文诊断 nuisance perturbation 下 paired predictive radius/guard | 用作重要 negative precedent：diagnostic 必须做 responsiveness/falsification，而不能只展示 endpoint correlation |

官方页面：

```text
https://arxiv.org/abs/2603.07799
https://arxiv.org/abs/2606.09028
https://arxiv.org/abs/2606.31232
https://arxiv.org/abs/2607.02403
https://arxiv.org/abs/2607.05966
```

## 8.2 新 novelty 定位

不要再把 novelty 主要写成：

```text
we introduce action-conditioned predictive consistency
```

改为三个更难碰撞的具体对象：

1. **paired same-state visual intervention**：clean/corrupted histories 描述同一底层状态，使用相同动作序列；
2. **selective radius–guard diagnostic**：nuisance pair 应收缩，task/planner-distinct pair 应保持分离；
3. **frozen checkpoint audit through the encoder–rollout–candidate chain**：外推验证、candidate-wise certificate coverage 和局部 composed sensitivity。

建议标题候选：

```text
Selective Paired-Rollout Diagnostics for Gaussian Visual Robustness in Latent World Models
```

或保留当前标题，但摘要/贡献中首次出现 ACPC 时立即加：

```text
Here ACPC denotes paired clean/corrupted same-state rollout consistency,
not the broader real-versus-imagined rollout consistency used in recent training methods.
```

## 8.3 related-work 建议段落

### 在 latent prediction / rollout consistency 段加入 MWM 与 Delta-JEPA

建议含义：

> Recent methods directly train action-sensitive or rollout-consistent world models. MWM reduces real-versus-imagined autoregressive drift through action-conditioned consistency post-training, whereas Delta-JEPA supervises latent displacement with action decoding to prevent action-insensitive collapse. Our object is different: given a fixed checkpoint, we compare clean and visually corrupted views of the same state under a shared action intervention, and require task/planner-distinct cases to remain separated.

### 在 diagnostic / planning 段加入 ATM、ACID、iKCE

建议含义：

> ATM diagnoses whether real encoded and predicted transitions preserve transferable action information; ACID uses inverse-dynamics cycle consistency as a decision-time planning cost. These are neighboring action-semantics tools. ACPC instead conditions on a paired visual intervention and measures nuisance-radius contraction together with a discriminability guard. The iKCE study further shows that a plausible rollout diagnostic can remain flat while policy reward collapses under a physical perturbation sweep, motivating our frozen off-axis and falsification tests rather than relying on endpoint correlation alone.

不要逐字照抄，Codex 应按全文风格压缩。

## 8.4 建议 BibTeX

在提交前仍需由 `reference_audit.md` 按官方 arXiv 页面复核；以下字段可直接作为起点：

```bibtex
@article{yan2026mwm,
  title         = {{MWM}: Mobile World Models for Action-Conditioned Consistent Prediction},
  author        = {Yan, Han and Xiang, Zishang and Zhang, Zeyu and Tang, Hao},
  journal       = {arXiv preprint arXiv:2603.07799},
  year          = {2026},
  eprint        = {2603.07799},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2603.07799},
  url           = {https://arxiv.org/abs/2603.07799}
}

@article{chen2026atm,
  title         = {{ATM}: Action-Consistency Transfer Matrix for Diagnosing and Improving Latent World Models},
  author        = {Chen, Jiaheng},
  journal       = {arXiv preprint arXiv:2606.09028},
  year          = {2026},
  eprint        = {2606.09028},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2606.09028},
  url           = {https://arxiv.org/abs/2606.09028}
}

@article{zhang2026deltajepa,
  title         = {{Delta-JEPA}: Learning Action-Sensitive World Models via Latent Difference Decoding},
  author        = {Zhang, Zhenghao and Wang, Yuanxiang and Guan, Zhenyu and Yang, Yujia and Shi, Bingkang and Zong, Tianyu and Yi, Hongzhu and Chao, Guoqing and Chen, Xingchen and Yang, Tiankun and Bao, Chenxi and Yu, Tao and Zhou, Jingjing and Xu, Jungang},
  journal       = {arXiv preprint arXiv:2606.31232},
  year          = {2026},
  eprint        = {2606.31232},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2606.31232},
  url           = {https://arxiv.org/abs/2606.31232}
}

@article{seo2026acid,
  title         = {{ACID}: Action Consistency via Inverse Dynamics for Planning with World Models},
  author        = {Seo, Gawon and Kim, Dongwon and Kwak, Suha},
  journal       = {arXiv preprint arXiv:2607.02403},
  year          = {2026},
  eprint        = {2607.02403},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2607.02403},
  url           = {https://arxiv.org/abs/2607.02403}
}

@article{schaefer2026kinematic,
  title         = {Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure},
  author        = {Sch{\"a}fer, Finn Rasmus and Moller, Korbinian and Gao, Yuan and Oefinger, Christian and Schmidt, Sebastian and Betz, Johannes},
  journal       = {arXiv preprint arXiv:2607.05966},
  year          = {2026},
  eprint        = {2607.05966},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2607.05966},
  url           = {https://arxiv.org/abs/2607.05966}
}
```

## 8.5 参考文献执行项

修改：

```text
paper1/references.bib
paper1/reference_audit.md
paper1/main.tex
tools/check_paper1_consistency.py
```

要求：

- `reference_audit.md` 记录 2026-07-10 live recheck；
- 不手工写死 citation count，或由脚本计算；
- checker 要求五个 citation key 都被引用；
- checker 禁止旧式“ACPC is the first action-conditioned consistency”类措辞；
- bibliography build 和 blind/arXiv bundle 均通过。

## 8.6 ATM baseline 的可执行路线

ATM 是最强直接相近工作，优先级高于再加一个普通 encoder metric。若官方代码未公开：

1. 只按论文定义实现最小 probe，不声称官方复现；
2. 使用同一 real encoded transition / predicted transition；
3. 训练轻量 action probe；
4. 训练/验证 split 按 trajectory，不能把相邻 transition 泄漏到两侧；
5. 记录 real→real、pred→pred、real→pred、pred→real transfer matrix；
6. 用其 screening score 与 B2/B3/B7 同表比较；
7. 明确标注 `ATM-style reimplementation`。

如果无法无歧义复现，则 related work 保留，实验 baseline 改为 inverse-dynamics action transfer，不伪称 ATM。

---

# 9. 其他补强项

## 9.1 统计层级

主行为统计继续以 training seed 为主要单位。诊断/anchor 统计采用：

```text
trajectory/anchor within checkpoint
checkpoint within training seed
training seed within task
model/stressor as external block
```

推荐：

- checkpoint-level summary 先算；
- block bootstrap resample task/seed/checkpoint；
- anchor-level Wilson/CP interval只表示 measurement precision；
- 不把几千 anchors 当成几千独立模型证据。

## 9.2 会议模板与篇幅

当前 article/arXiv 技术报告可以保留，但新增：

```text
paper1/docs/main_conference.tex
```

使用目标会议官方模板后重新检查：

- main text 页数；
- theorem 与 proof 移 appendix；
- 主文最多保留：行为 sweep、external validation、radius/guard、certificate coverage、local sensitivity；
- baseline exact table 与敏感性细节放 appendix；
- current 28-page PDF 不能作为主会篇幅完成的依据。

## 9.3 术语收敛

主文只保留：

```text
ACPC
ATR
SMPR
fixed-pool certificate coverage
composed local sensitivity
```

避免重新引入大量 acronym。新 oracle guard 可先用描述性名字，不急着命名。

---

# 10. 执行顺序与优先级

## P0-0：冻结旧结果与 correctness audit

- [ ] 给当前 main/result artifacts 写 snapshot hash；
- [ ] 新结果全部使用 `_v2`/新 schema，不覆盖旧 canonical；
- [ ] 在 docs 记录 current ATR 是 stepwise tail；
- [ ] 确认 target-view checkpoint manifest 可解析；
- [ ] 确认 PLDM 36 checkpoints 可加载。

## P0-1：canonical metric 与 κ 修复

- [ ] 新建 `tools/paper1_acpc_metrics.py`；
- [ ] horizon-level radius 与 stepwise metric 分名；
- [ ] JVP 使用同一个 weighted stacked map；
- [ ] 修正 κ 两个定义；
- [ ] 单元测试通过；
- [ ] 小规模 2-task smoke run。

## P0-2：rebuild LeWM/PLDM/blur-resize/target-view diagnostics

- [ ] seed3072 CAL；
- [ ] seeds3073/3074 E1；
- [ ] PLDM E2；
- [ ] blur/resize E3；
- [ ] target-view E4；
- [ ] 每个 artifact 无 missing/error rows 或显式列出。

## P0-3：freeze gate 与 baseline audit

- [ ] 生成 protocol JSON；
- [ ] 外部脚本只读 protocol；
- [ ] B0–B8 同表；
- [ ] action shuffle；
- [ ] block-bootstrap differences；
- [ ] 根据结果决定 claim 强度。

## P0-4：fixed-pool certificate 重写

- [ ] candidate-wise sharp certificate；
- [ ] coverage 与 one-sided risk upper；
- [ ] `flip|cert=0` 降为 sanity；
- [ ] `flip|fail` 与 K sensitivity；
- [ ] hierarchical interval；
- [ ] 主图/正文更新。

## P0-5：SMPR sensitivity 与 controls

- [ ] positive margins；
- [ ] neighborhood/binning sensitivity；
- [ ] radius/diff 分解；
- [ ] collapse/label/action/far-neighbor controls；
- [ ] 至少 TwoRoom + PushT oracle MVE；
- [ ] 决定是否升级或收缩 guard claim。

## P0-6：novelty 与 references

- [ ] 加五篇 BibTeX；
- [ ] 更新 related work；
- [ ] 加 direct comparison matrix；
- [ ] ATM-style baseline 可行性检查；
- [ ] 修改 title/abstract/contributions 的 novelty wording；
- [ ] reference audit/checker 通过。

## P0-7：全文与 release gate

- [ ] `python -m tools.check_paper1_consistency`；
- [ ] `pytest -q`；
- [ ] `bash paper1/scripts/run_all_paper1_diagnostics.sh`；
- [ ] `cd paper1 && bash build.sh --clean`；
- [ ] `bash paper1/check_arxiv_ready.sh`；
- [ ] `bash paper1/docs/check_blind_ready.sh`；
- [ ] isolated source bundles 编译无 warning；
- [ ] 更新 `DATA_MANIFEST.md` 和 SHA-256。

## P1

- [ ] adaptive CEM common-random-number audit；
- [ ] full oracle/value-grounded guard 四任务；
- [ ] ATM-style action-transfer baseline；
- [ ] empirical ACPC-to-cost Lipschitz calibration；
- [ ] Gaussian quadratic-form concentration / spectral estimates；
- [ ] DINO-WM 8-run positive external family；
- [ ] Delta-JEPA/ACID 外部代码和 checkpoint 可用性核查。

## P2

- [ ] 更多 model families/tasks；
- [ ] 更多成功非 Gaussian repair；
- [ ] adaptive/repeated replanning theory；
- [ ] 将 diagnostic 变成训练 objective 的后续方法论文。

---

# 11. 资源与降级方案

## 11.1 P0 原则上不新增模型训练

P0 主要复用：

- 108 LeWM Gaussian rows；
- 36 PLDM rows；
- 24 unseen stress rows；
- target-view runs；
- fixed-pool/JVP checkpoint artifacts。

checkpoint-level diagnostic 仍需要 GPU。Codex 先做小规模 benchmark：

```text
2 tasks × 2 checkpoints × 16 anchors × 2 noise draws
```

记录每 row wall time、peak GPU memory、I/O time，再外推 full run。不要在未 benchmark 前写死总耗时。

### 降级版 MVE

若 full recomputation 过慢：

1. TwoRoom + PushT；
2. LeWM seed3072 CAL；
3. LeWM seed3073 test；
4. PLDM PushT/TwoRoom；
5. target-view PushT；
6. blur TwoRoom + resize PushT；
7. 32 anchors、3 noise draws；
8. protocol 不变，之后只扩样本，不重调。

MVE 必须先验证：

- metric/JVP 数学量一致；
- frozen external script 不会偷偷重调；
- target-view 可以形成 falsification；
- collapse control 被 guard 拒绝。

## 11.2 P1 新训练预算

DINO-WM no-prop：

```text
2 tasks × 4 configs = 8 runs
```

按仓库既有估计每 run 约 2–3 GPU 小时，粗略 16–24 GPU 小时；实际以首个 run benchmark 为准。未完成前不写入论文结果。

---

# 12. Codex 文件级改动清单

## 新建

```text
paper1/config/frozen_diagnostic_protocol_v1.json
paper1/scripts/freeze_diagnostic_protocol.py
paper1/scripts/frozen_external_validation.py
paper1/scripts/diagnostic_baseline_comparison.py
paper1/scripts/plot_frozen_external_validation.py
paper1/scripts/fixed_pool_certificate_calibration.py
paper1/scripts/smpr_controls.py
paper1/scripts/smpr_sensitivity.py
tools/paper1_acpc_metrics.py
tools/paper1_target_view_diagnostic_manifest.py
tests/test_paper1_acpc_metrics.py
tests/test_paper1_jvp_alignment_metrics.py
tests/test_paper1_frozen_gate_no_recalibration.py
tests/test_paper1_candidatewise_certificate.py
```

## 修改

```text
paper1/main.tex
paper1/references.bib
paper1/reference_audit.md
paper1/scripts/run_all_paper1_diagnostics.sh
paper1/scripts/build_diagnostic_manifest.py
paper1/scripts/build_full_sweep_diagnostics.py
paper1/scripts/heldout_diagnostic_validation.py
tools/paper1_phase0_acpc.py
tools/paper1_semantic_margin.py
tools/paper1_sample_level_certificate.py
tools/paper1_gaussian_sensitivity_audit.py
tools/paper1_jvp_hutchinson_sensitivity_audit.py
tools/check_paper1_consistency.py
tools/README_paper1.md
DATA_MANIFEST.md
```

## 不允许

- 覆盖旧 canonical result；
- 手工把预期数字写入 table；
- external split 上选择阈值；
- 因 external result 不好而更换 q/H/normalization；
- 只报告 selected stress positive rows；
- 把 PLDM one seed 写成 multi-seed model-family robustness；
- 把 target-view failure 写成成功 repair；
- 把 conditional zero flip 写成独立 statistical evidence；
- 把 κ_rel 写成 bounded angle；
- 将 ATM-style 重实现冒充官方 ATM 代码。

---

# 13. 最终验收标准

## 13.1 Coverage audit

- [ ] Gaussian matched axis；
- [ ] held-out training seed；
- [ ] different model family；
- [ ] different perturbation family；
- [ ] different training/failure mechanism；
- [ ] simple metadata/encoder/one-step/action-shuffle baselines；
- [ ] negative diagnostic control；
- [ ] recent/concurrent work。

## 13.2 Theory audit

- [ ] ATR 是 per-rollout horizon metric，不是误命名 per-token q90；
- [ ] JVP map 与 ATR map 完全一致；
- [ ] κ 公式/代码/表格一致；
- [ ] probability spaces 分开；
- [ ] fixed-pool 数值结果非空；
- [ ] q90 不再冒充 theorem alpha；
- [ ] `flip|cert=0` 只作 sanity。

## 13.3 Execution audit

- [ ] 每项有输入、脚本、输出、failure criterion；
- [ ] external gate 有 hash lock；
- [ ] 所有新 artifacts 有 schema/provenance/hash；
- [ ] missing/error rows 不被静默丢弃；
- [ ] 运行成本由 smoke benchmark 外推；
- [ ] 文本 claim 自动从 audited artifact 生成或由 checker 锁定。

## 13.4 Claim decision gate

最终正文强度由结果决定：

### Stronger diagnostic claim

仅当：

- frozen ATR+SMPR 在 PLDM 与至少一个 stressor/mechanism external split 上优于 encoder-only/std metadata；
- candidate-wise coverage 与 observed flip 有合理 sharpness；
- SMPR controls 与至少一个 oracle/value guard 通过；
- no unreported counterexample。

### Bounded diagnostic claim

若 external transfer 部分成立但 baseline/guard 增量有限：

- 论文定位为 controlled Gaussian robustness analysis；
- ACPC 是组织理论和 mechanism localization 的框架；
- closed-loop evaluation 仍是唯一 authority；
- 不把 ATR/SMPR 写成 selector。

### Negative but publishable finding

若 target-view/PLDM/blur-resize 揭示 diagnostic 与 behavior 脱钩：

- 不隐藏；
- 将贡献转为“何时 paired predictive diagnostics 有效、何时失败”；
- 使用 iKCE 并发工作作为诊断 responsiveness 的相关背景；
- 这比在同一 Gaussian sweep 上继续堆高相关更有顶会价值。

---

# 14. 推荐的最短执行路径

在不新增训练的情况下，按以下顺序最划算：

1. 修 canonical horizon radius 与 κ；
2. 生成一个 seed3072 frozen protocol；
3. 现有 LeWM held-out seeds 重跑 strict frozen evaluation；
4. 现有 PLDM full sweep 重算当前 ATR/SMPR；
5. 现有 blur/resize 24 rows 做 frozen external test；
6. 现有 target-view 做 mechanism falsification；
7. 同时跑 `std_max`、encoder q90、one-step、action shuffle baseline；
8. fixed-pool 改为 candidate-wise certificate coverage；
9. SMPR 做 positive-margin/controls，先 TwoRoom+PushT oracle MVE；
10. 更新五篇并发文献与 related work；
11. 根据真实结果决定是否投入 DINO-WM/Delta-JEPA P1。

这条路径直接回答 reviewer 最可能提出的疑问，而且大部分利用已有 checkpoint 和 artifact，优先级高于继续增加 LeWM seed 或同轴图表。
