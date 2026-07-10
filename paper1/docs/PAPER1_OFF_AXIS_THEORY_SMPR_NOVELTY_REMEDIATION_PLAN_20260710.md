# Paper 1：Off-axis Validation、理论定量化、Fixed-pool Audit、SMPR 与新颖性整改执行计划

> **目标仓库**：`qun-team/wm_exp`  
> **目标分支**：`ag/dev`  
> **修订日期**：2026-07-10  
> **本次修订主题**：重新界定 `std_max` 在 Gaussian 内部审计与跨模型/跨扰动外部验证中的角色  
> **执行者**：Codex  
> **关联总计划**：`paper1/docs/PAPER1_POST_REVIEW_THEORY_VALIDATION_NOVELTY_EXECUTION_PLAN_20260710.md`

---

## 0. 本文件的权威性与适用范围

本文件是对关联总计划中以下内容的**权威修订**：

1. 问题 2：是否必须把 `std_max` 作为正式 baseline；
2. simple-baseline 工作包；
3. PLDM 与 blur/resize external validation 的成功标准；
4. Phase 1 停止条件；
5. 最终 claim 决策表。

关联总计划中关于以下部分的整改方案保持不变：

- probability space 与 metric 对齐；
- q90 与 fixed-pool theorem 的定量桥梁；
- `kappa` 公式/实现一致性；
- `flip | cert-pass = 0` 降级为 invariant test；
- fixed-pool risk–coverage；
- SMPR positive margin、collapse/action/label controls；
- MWM、ATM、Delta-JEPA、ACID、Future Compatible、Kinematic Not Dynamic 的相关工作与参考文献更新。

若本文件与关联总计划在 `std_max`、`rho` baseline 或 external validation 验收条件上冲突，以本文件为准。

---

# 1. 问题 2 的复议结论

## 1.1 最终结论

**不需要把 `std_max` 作为跨模型、跨扰动类别或跨修复机制的正式竞争 baseline，也不应要求 ATR/SMPR 在 external validation 中“击败 `std_max`”。**

`std_max` 只保留为 matched-Gaussian 子集上的一个低成本、附录级、privileged confound sanity check。

这个结论相对上一版发生了实质变化：

- 删除“joint diagnostic 必须优于 `rho/std_max` 才能保留中心主张”的要求；
- 删除“`rho-only` 匹配 onset MAE 就触发整篇论文降级”的停止条件；
- 不再把 `rho` 放进 PLDM、blur/resize、off-repair 的 external leaderboard；
- external validation 的主要竞争对象改为可从 checkpoint/trajectory 共同计算的指标。

## 1.2 为什么 `std_max` 不是通用 diagnostic baseline

`std_max` 是某一种 Gaussian input augmentation 配方的训练元数据，而不是 checkpoint 自身的可测风险量。它有四个结构性限制。

### 限制 A：跨修复机制不可定义

以下方法没有统一的 Gaussian `std_max`：

- target-view training；
- heteroscedastic objective；
- AITS / Delta-JEPA 风格 action-sensitive objective；
- test-time observation intervention；
- planner-side robust CEM；
- inverse-dynamics planning repair。

若一个 baseline 在这些 repair 上没有定义，就不能作为跨 repair 的统一诊断指标。

### 限制 B：跨模型数值未必同尺度

即使 LeWM 与 PLDM 使用同名 Gaussian augmentation，数值相同也不保证功能扰动相同。模型输入归一化、encoder sensitivity、latent scale、优化动态和 predictor gain 都可能不同。

因此，`std_max=0.04` 在 LeWM 与 PLDM 中不是天然可比的 checkpoint-risk 数值。

### 限制 C：跨扰动类别没有共同语义

Gaussian `std_max` 与以下 severity 没有统一的一维映射：

- blur kernel；
- resize factor；
- occlusion；
- background distractor；
- compression；
- geometry shift。

用 Gaussian training strength 作为 blur/resize external diagnostic，会把训练配方标签误当成可迁移风险量。

### 限制 D：不是 training-free checkpoint diagnostic

论文的定位是 no-retraining / post-hoc checkpoint diagnostic。`std_max` 依赖训练日志或命名，不能保证在外部 checkpoint 上可获得，也不能反映实际 learned predictive behavior。

---

# 2. 更强的回应：fixed-`rho` off-axis discrimination

用户提出的核心思路是正确的：最有力的反驳不是让 ACPC 在同一 Gaussian sweep 上击败 `std_max`，而是验证从 Gaussian 理论与实验中得到的指标，能否在 `std_max` 不可区分或没有定义的外部轴上继续有效。

## 2.1 Blur/resize 是问题 2 的直接反证场景

在只保留 noise-trained endpoint 的 external slice 中：

- 4 tasks；
- 2 stressor families：blur / resize；
- 3 training seeds；
- 所有 row 都是相同的 Gaussian training endpoint：`rho=0.08`。

此时训练 `std_max` 在 12 个 endpoint rows 中是常数，没有任何区分能力；但行为迁移包含：

- 明确正向；
- 弱向或中性；
- 负向。

因此应把下面的检验作为问题 2 的主实验：

> 在固定 `rho=0.08` 的情况下，stressor-specific ATR/SMPR 与 frozen joint gate 是否能区分 transfer-positive、neutral 和 negative rows？

若能区分，说明 diagnostic 读取的是 checkpoint 在当前 perturbation 下的 predictive behavior，而不是训练噪声强度。

## 2.2 必须使用全部数据，而不是四个代表行

正式统计必须使用：

- 全部 12 个 noise-trained fixed-`rho` endpoint rows；
- 全部 24 个 base→endpoint paired rows。

主文四个代表行可以保留为 compact boundary table，但不能替代完整 external audit。

## 2.3 建议输出

新增：

```text
paper1/results/external_validation/cross_stressor_fixed_rho_rows.csv
paper1/results/external_validation/cross_stressor_fixed_rho_summary.json
paper1/results/external_validation/cross_stressor_all24_pairs.csv
paper1/figures/fig_cross_stressor_fixed_rho.png
```

每个 fixed-`rho` row 至少包含：

```text
task
training_seed
stressor_family
stressor_severity
rho
clean_score
stressed_score
retention
behavior_class
encoder_q90
h1_q90
action_shuffled_h8_q90
atr_h8_q90
smpr
joint_gate_pass
```

每个 paired row 至少包含：

```text
delta_behavior
delta_encoder_q90
delta_h1_q90
delta_atr
delta_smpr
delta_joint_score
```

## 2.4 预注册行为标签

不要根据 diagnostics 重新定义 external target。建议在读取 diagnostic result 前冻结：

- `positive transfer`：stressed score 提升至少 5 pp，且 clean score 不下降超过 5 pp；
- `neutral`：变化在 ±5 pp；
- `negative`：stressed score 下降超过 5 pp，或 clean degradation 超限。

同时报告连续 retention，不只报告分类结果。

---

# 3. PLDM 的角色

## 3.1 PLDM 可以作为跨模型补充验证

PLDM 4 tasks × 9 Gaussian sweep、一个 training run，可以用于：

- model-family transfer；
- frozen LeWM gate 的外部 onset/error audit；
- encoder/H1/H8/joint diagnostic 的跨模型比较。

但必须明确：

- 只有一个 PLDM training run；
- 三个 evaluation seeds 不是三个 training seeds；
- 不能声称 PLDM training-run variability 已验证；
- PLDM 仍使用 Gaussian augmentation，因此它不是跨 repair positive validation。

## 3.2 `rho` 在 PLDM 中只作协议内 nuisance

PLDM 也有 Gaussian grid，所以可在 appendix 记录 `rho-only` 的内部参考；但：

- 不进入 external leaderboard；
- 不作为 frozen gate 成功与否的门槛；
- 不要求 ATR/SMPR 必须在 onset MAE 上击败它；
- 主结论来自 LeWM 上冻结的 checkpoint-level diagnostic 无调参转移到 PLDM。

## 3.3 PLDM 成功标准修订

原先“必须优于 metadata-global `rho`”的标准删除。

修订后的成功标准：

1. frozen gate 不在 PLDM 结果出来后重新调参；
2. 至少 3/4 tasks 的 onset error 在两格以内，或 row-level classification/calibration 保持可解释；
3. joint diagnostic 相比 encoder q90、H=1 或 action-shuffled 等 checkpoint-level baseline 有增量；
4. 所有失败 task 与 false-early/false-late 均保留。

---

# 4. `std_max` 仍保留什么？

## 4.1 只保留 Gaussian-only confound audit

保留它只有一个目的：判断 LeWM matched-Gaussian sweep 中的 recovery-onset 结果，有多少只是离散 training grid 的显然结构。

新增：

```text
paper1/scripts/gaussian_rho_confound_audit.py
paper1/results/diagnostic_baselines/gaussian_rho_confound_rows.csv
paper1/results/diagnostic_baselines/gaussian_rho_confound_summary.json
```

仅在 LeWM/PLDM Gaussian rows 上比较：

| Audit model | 目的 |
|---|---|
| `rho_only` | training strength 能解释多少 Gaussian recovery label |
| `rho + encoder_q90` | encoder 指标是否有条件增量 |
| `rho + h1_q90` | one-step 是否有条件增量 |
| `rho + atr_h8` | rollout radius 是否有条件增量 |
| `rho + atr_h8 + smpr` | joint diagnostic 是否有条件增量 |

## 4.2 评价方式

优先报告：

- held-out row-level AUPRC；
- balanced accuracy；
- calibration-only logistic/ordinal deviance；
- held-out log loss；
- 同一 `rho` 下跨 task/seed/model 的 residual ordering。

onset MAE 只作为次要结果，因为离散九点 grid 容易被一个粗糙 threshold 匹配。

## 4.3 结果解释

- 若 `rho-only` 匹配 Gaussian onset：不构成中心失败；将 onset 结果降级为 internal consistency check。
- 若 `rho + ATR + SMPR` 仍有增量：可写“contains checkpoint-level information beyond augmentation strength”。
- 若没有内部增量，但 fixed-`rho` cross-stressor 与 PLDM external 成立：仍可保留跨轴 diagnostic claim。
- 若内部无增量且 external 也失败：退回 matched-Gaussian mechanism-localization。

## 4.4 不再做的内容

删除或取消以下要求：

- `rho_task_conditioned` privileged upper bound；
- `rho` 进入 cross-stressor leaderboard；
- `rho` 进入 cross-repair leaderboard；
- joint gate 必须击败 `rho` 才能通过 P0；
- `rho-only` 匹配 onset 就停止全部后续实验。

---

# 5. 真正不能省的简单 baseline

`std_max` 可以降级，但下列 baseline 直接检验论文中心主张，不能省。

| Baseline | 核心问题 | 可迁移性 |
|---|---|---|
| `encoder_q90` | raw encoder invariance 是否已经足够 | 高 |
| `h1_predictive_q90` | one-step prediction 是否已经足够 | 高 |
| `action_shuffled_h8_q90` | H-step 改善是否真的依赖正确 action intervention | 高 |
| `action_zeroed_h8_q90` | 一般平滑度是否冒充 action-conditioned consistency | 高 |
| `atr_h8_q90` | multi-step action-conditioned radius | 高 |
| `smpr` | guard 单独贡献 | 取决于 task labels |
| `atr_h8 + smpr` | 主 joint diagnostic | 高 |
| `ATM-style action-transfer` | 与最接近并发 diagnostic 比较 | 中/高 |

这些 baseline 应使用：

- 相同 calibration/held-out split；
- 相同 normalization leakage rule；
- 相同 threshold/model-complexity budget；
- 相同 external rows；
- 完整 confusion/calibration 输出。

---

# 6. 修订后的证据层级

## 6.1 内部证据：LeWM Gaussian

定位：

- metric/protocol development；
- training-seed confirmation；
- theory-to-measurement alignment；
- internal consistency。

不再把 onset MAE 单独写成 strongest validity evidence。

## 6.2 跨模型证据：PLDM Gaussian

定位：

- frozen model-family transfer；
- 不重新调 threshold；
- 一个 training run 的边界明确披露。

## 6.3 跨扰动证据：LeWM blur/resize

定位：

- fixed-`rho=0.08` external discrimination；
- 完整 12 endpoint rows + 24 paired rows；
- mixed positive/neutral/negative transfer 是关键，而不是需要所有结果都正向。

## 6.4 跨修复证据：off-repair controls

定位：

- target-view / heteroscedastic / robust-CEM 等作为 falsification controls；
- 不使用 `rho`；
- 若没有非 Gaussian augmentation 的正向 repair，不写 cross-repair generalization。

---

# 7. 修订后的 P0-B 工作包

## 7.1 新增/修改文件

```text
paper1/scripts/diagnostic_baseline_benchmark.py
paper1/scripts/gaussian_rho_confound_audit.py
paper1/scripts/build_cross_stressor_external_validation.py
paper1/results/diagnostic_baselines/heldout_baseline_rows.csv
paper1/results/diagnostic_baselines/heldout_baseline_summary.csv
paper1/results/diagnostic_baselines/gaussian_rho_confound_rows.csv
paper1/results/diagnostic_baselines/gaussian_rho_confound_summary.json
paper1/results/external_validation/cross_stressor_fixed_rho_rows.csv
paper1/results/external_validation/cross_stressor_fixed_rho_summary.json
paper1/tables/table_diagnostic_baselines.tex
```

## 7.2 Phase 1 执行顺序

1. encoder q90；
2. H=1 predictive q90；
3. action-shuffled / action-zeroed H=8；
4. ATR H=8；
5. SMPR；
6. ATR+SMPR；
7. fixed-`rho=0.08` cross-stressor discrimination；
8. 最后单独运行 appendix-only `rho` confound audit。

## 7.3 修订后的停止条件

触发 claim 收缩的条件：

- encoder q90 与 joint diagnostic 相当或更好；
- H=1 与 H=8 相当，且 contact/planner-sensitive task 上也无增量；
- action-shuffled 与正确 action rollout 相当；
- fixed-`rho` cross-stressor mixed transfer 无法被 diagnostic 区分；
- PLDM frozen gate 系统性失败。

**以下情况不单独触发停止：**

- `rho-only` 在 LeWM Gaussian onset MAE 上表现相当；
- `rho` 在 PLDM Gaussian grid 上有较强单调性。

这些结果只要求降低 in-domain onset 证据权重。

---

# 8. 修订后的 claim 决策

## 情形 A：强结果

满足：

- PLDM frozen validation 成立；
- fixed-`rho` blur/resize discrimination 成立；
- joint diagnostic 优于 encoder/H1/action-shuffled；
- theory calibration 与 SMPR controls 通过。

可写：

> The frozen paired-rollout radius/guard diagnostic transfers across a second model family and distinguishes mixed non-Gaussian stressor outcomes at a fixed Gaussian training strength.

若 Gaussian confound audit 还有条件增量，可补：

> The diagnostic also contains checkpoint-level information beyond augmentation strength within the matched Gaussian sweep.

## 情形 B：中等结果

满足：

- `rho-only` 匹配 in-domain onset；
- 但 PLDM 或 fixed-`rho` cross-stressor 仍有外部增量；
- 部分 checkpoint-level baseline 相当。

应写：

> The Gaussian sweep supplies internal mechanism evidence, while the main diagnostic validity comes from frozen cross-model and fixed-training-strength cross-stressor tests.

删除：

- reliable onset predictor；
- checkpoint selector；
- beats training-noise metadata。

## 情形 C：负结果

出现：

- fixed-`rho` cross-stressor discrimination 失败；
- PLDM frozen gate 失败；
- encoder/H1/action-shuffled 不弱于 ACPC；
- collapse control 未被 SMPR 拒绝。

定位改为：

> A controlled audit of when predictive-consistency diagnostics align with, and fail to explain, visual robustness.

---

# 9. 论文正文建议措辞

## 9.1 不建议写

> ATR/SMPR outperforms training noise strength as a universal robustness diagnostic.

原因：两者不是同类别量，且 `std_max` 在 external repair/stressor 上没有统一定义。

## 9.2 建议写

> Training-noise magnitude is a protocol variable rather than a checkpoint diagnostic and is not comparable across repair or perturbation families. We therefore use it only as a privileged confound reference inside the matched Gaussian sweeps. External validity is tested with frozen checkpoint-level diagnostics on PLDM and on non-Gaussian stressors, including a fixed-training-strength slice in which all checkpoints share the same Gaussian training endpoint.

## 9.3 对内部 onset 结果的建议写法

> Within the matched Gaussian grid, held-out onset alignment is an internal consistency check. Because a coarse threshold on the training grid can approximate some task onsets, the main evidence for diagnostic validity comes from frozen cross-model and fixed-training-strength cross-stressor evaluations.

---

# 10. Codex 文件级整改清单

## 必改

- [ ] `paper1/main.tex`
  - [ ] 将 `std_max/rho` 描述为 protocol metadata；
  - [ ] onset validation 降级为 internal consistency；
  - [ ] 加 fixed-`rho` cross-stressor 结果；
  - [ ] external baseline 只保留 checkpoint-level metrics；
  - [ ] `rho` confound audit 放 appendix。
- [ ] `paper1/scripts/heldout_diagnostic_validation.py`
  - [ ] 保留内部 retrospective validation；
  - [ ] 不再把结果描述为 universal external gate。
- [ ] `paper1/scripts/diagnostic_baseline_benchmark.py`
  - [ ] encoder/H1/action-shuffled/H8/SMPR/joint；
  - [ ] external protocol 不接受 `rho` 特征。
- [ ] `paper1/scripts/gaussian_rho_confound_audit.py`
  - [ ] 仅加载 Gaussian rows；
  - [ ] 输出 privileged-reference 标记。
- [ ] `paper1/scripts/build_cross_stressor_external_validation.py`
  - [ ] 12 fixed-`rho` endpoint rows；
  - [ ] 24 paired rows；
  - [ ] 不重新调 gate。
- [ ] `tools/check_paper1_consistency.py`
  - [ ] checker 阻止 external results 使用 `rho` gate；
  - [ ] checker 验证固定 endpoint slice 的 `rho` 全相同；
  - [ ] checker 验证完整 24 rows 未被选择性过滤。
- [ ] `DATA_MANIFEST.md`
  - [ ] 新增 fixed-`rho` 与 confound artifacts；
  - [ ] 记录 source hashes/provenance。

## 硬验收

- [ ] external leaderboard 不含 `rho/std_max`；
- [ ] Gaussian appendix 有 `rho-only` confound reference；
- [ ] `rho-only` 不作为 P0 pass/fail 条件；
- [ ] 12 个 `rho=0.08` external rows 全部进入统计；
- [ ] 24 个 base→endpoint pairs 全部进入统计；
- [ ] encoder/H1/action-shuffled/H8/joint 使用同 split；
- [ ] external gate 无 threshold re-search；
- [ ] 负向与 discordant rows 不删除。

---

# 11. 与其他整改工作包的接口

## 理论工作包

不变。仍需：

- probability space 分层；
- stacked-L2 metric 对齐；
- non-vacuous radius→cost bridge；
- `kappa_iso/kappa_sub` 修正。

## Fixed-pool 工作包

不变。仍需：

- `flip | cert-pass = 0` 只作 invariant；
- certificate coverage；
- continuous score；
- risk–coverage；
- block bootstrap；
- PLDM transfer（可行时）。

## SMPR 工作包

不变。仍需：

- positive normalized margins；
- neighborhood/granularity sensitivity；
- same-state radius 与 different-state margin 分开；
- progressive collapse；
- action shuffle；
- label shuffle；
- PushT/Cube stronger semantic guard。

## Novelty 工作包

不变。仍需：

- 增加 MWM、ATM、Delta-JEPA、ACID、Future Compatible、Kinematic Not Dynamic 的 BibTeX；
- 更新 `reference_audit.md`；
- ATM/ATM-style baseline；
- novelty 不押在 action-conditioned consistency 名称本身。

---

# 12. 最终执行优先级

1. **先做 fixed-`rho=0.08` cross-stressor discrimination**：这是对问题 2 最直接、最有说服力的回答。
2. **再做 encoder/H1/action-shuffled/H8/joint checkpoint-level baseline**：检验 action-conditioned multi-step 的真实增量。
3. **做 PLDM frozen validation**：补跨模型族。
4. **最后做 Gaussian-only `rho` confound audit**：用于决定 onset 结果在正文还是 appendix 的证据权重，不决定论文整体成败。
5. 继续执行理论、fixed-pool、SMPR 和 novelty 的既定 P0 计划。

---

## 最终判断

本次复议后，结论调整为：

> `std_max` 不应是跨模型、跨扰动、跨修复的正式 baseline。真正的主检验应是在 `std_max` 固定或没有定义时，checkpoint-level ATR/SMPR 是否仍能解释行为差异。Gaussian 内部的 `rho` 比较只保留为混杂审计，用来限制 onset-prediction 语言，而不是作为 ACPC 中心主张的生死门槛。
