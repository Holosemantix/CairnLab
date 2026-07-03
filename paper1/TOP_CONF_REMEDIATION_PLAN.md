# Paper1 顶会投稿整改计划

> 目标：把当前 `paper1` 从“matched Gaussian-noise post-hoc diagnostic study”提升为更接近顶会主会标准的诊断/世界模型论文。
>
> 本文档根据对 `paper1/main.tex` 更新版的客观审稿意见整理，供 Codex 后续执行。当前评价倾向为 **Weak Reject / Borderline Reject**，主要原因不是问题不重要，而是证据链仍偏事后诊断、主实验统计不足、prospective validation 尚未建立。

---

## 0. 当前论文核心定位

论文当前最有价值的贡献是：

1. 指出 **latent prediction / JEPA 本身不等于视觉扰动下的 closed-loop control robustness**。
2. 提出 **Action-Conditioned Predictive Consistency, ACPC** 作为诊断视角：同一底层状态的 clean/noisy 观测，在相同动作序列下经过 encoder + predictor 后，应产生一致的 rollout/readout。
3. 强调 ACPC 必须是 **selective consistency**：不能通过 collapse 表征来获得低 ACPC，必须保留 action-/transition-/cost-relevant discriminability。
4. 用 LeWM 主实验和 PLDM 复现说明 Gaussian observation noise 会导致 closed-loop cliff，而 full-sequence input-side noise training 可恢复部分任务表现。
5. 用 ACPC-basin、PCC/CRA/MAF、rank/transition-resolution/ID-probe 等指标做 post-hoc localization。

当前最大风险是：

- 贡献容易被审稿人认为是 **post-hoc diagnostic framing + known noise augmentation effect**。
- 主 LeWM grid 使用的是 **3 个 evaluation seeds**，不是 independent training seeds。
- ACPC 目前没有证明能在 held-out checkpoints / held-out perturbations 上做 **prospective prediction / selection**。
- Discriminability 主要靠 proxy metrics，而不是 task-semantic margins。
- 理论结果是合理的 formal link，但新意偏弱。

---

## 1. P0：必须优先完成的顶会级补强

### P0.1 建立 prospective validation protocol

**问题：** 当前 ACPC/PCC/CRA/MAF 主要用于事后解释已知 recovered checkpoints。审稿人会质疑：这些指标是否真的能预测鲁棒性，还是只是在好 checkpoint 上看起来合理？

**整改目标：** 把 ACPC 从 post-hoc localization 提升为 prospective diagnostic。

**建议执行：**

1. 固定一个 development set：
   - 若资源有限：每个任务选择若干 training noise levels / checkpoints 作为 dev。
   - 若资源允许：使用部分 training seeds +部分 checkpoint grid 作为 dev。
2. 在 dev set 上冻结以下内容：
   - ACPC-H 计算方式。
   - PCC / CRA / MAF 计算方式。
   - candidate-action set 采样规则。
   - 归一化方式。
   - checkpoint ranking 或 threshold rule。
3. 在 held-out set 上测试：
   - held-out training seeds。
   - held-out checkpoints。
   - held-out perturbation families，例如 blur / resize / brightness / contrast。
4. 报告 ACPC-style diagnostics 对以下量的预测能力：
   - noisy success。
   - clean-to-noisy drop。
   - drop improvement。
   - ranking agreement with closed-loop noisy evaluation。

**建议新增表格：**

```latex
Table: Prospective diagnostic validation.
Columns:
Task | Held-out split | Metric | Spearman rho vs noisy success | Pearson r | Top-k selection hit | Notes
```

**验收标准：**

- 主文中必须出现一个明确的 prospective validation 表，而不只是 appendix。
- 文中明确说明：哪些规则在 dev set 冻结，哪些结果是在 held-out set 上得到。
- 如果结果不强，也要诚实报告，并把 claim 收紧为“localization diagnostic”，不要说 predictor。

**建议改动位置：**

- `Section 4 Experiments` 增加一个 subsection：`Prospective diagnostic validation`。
- `Discussion and limitations` 中减少“当前没有 prospective validity”的负面表述，改成基于新结果的边界说明。

---

### P0.2 把 independent training seeds 提升为主实验统计

**问题：** 当前 canonical LeWM grid 使用 3 个 evaluation seeds，但这些不是 independent training seeds。顶会 reviewer 会认为结论可能依赖单次训练偶然性。

**整改目标：** 主结果至少覆盖多个 independent training seeds。

**建议执行：**

1. 对关键设置补跑 independent training seeds：
   - no-noise baseline。
   - representative robust noise-trained checkpoint，例如 `std_max = 0.08` 或任务最优/预设高噪声点。
   - 如资源允许，补完整 9-level noise sweep 的 3 个 training seeds。
2. 报告：
   - training-seed mean ± std。
   - evaluation-seed mean ± std 可以作为附加分解。
3. 避免只报告 2 个 seeds 的 appendix lockbox。至少主文里应有 seed-level evidence。

**最低可接受版本：**

- 每个任务至少 3 个 independent training seeds。
- 每个 seed 下固定 evaluation seeds。
- 主表报告 training-seed uncertainty。

**建议新增表格：**

```latex
Table: Independent-training-seed robustness summary.
Columns:
Task | Model row | Clean success | Obs-noise 0.08 success | Drop | ACPC-H | PCC | CRA | MAF
Rows:
base vs fixed std_max=0.08 or prospectively selected robust row
```

**验收标准：**

- 主文不再仅依赖 canonical 36-checkpoint single-training-seed grid。
- 论文明确区分 training variance 与 evaluation variance。
- 如果资源不足，必须在 abstract / limitation 中显著收紧 claim。

**建议改动位置：**

- `Study protocol`：重新描述 seed protocol。
- `Experiments`：把 independent-seed table 放到主文。
- `Discussion`：更新 limitation。

---

### P0.3 加入 task-semantic discriminability guard

**问题：** 当前 discriminability guard 主要是 effective rank、transition-resolution、ID probe R²。这些是合理 proxy，但不能直接证明 action-relevant state distinctions 被保留。

**整改目标：** 让 selective ACPC 的“selective”部分有更语义化的证据。

**建议执行：**

对每个任务设计一个轻量 semantic margin：

1. **PushT**
   - 使用 T-block pose / contact state / object-agent relative pose。
   - 构造 action-relevant pair：相似图像背景但 block pose/contact state 不同。
   - 测量 clean/noisy same-state ACPC contraction 是否小于 different-state semantic distance。
2. **TwoRoom**
   - 使用 room ID、doorway crossing state、target-region relation。
   - 检查 doorway/topology distinct states 是否保持可分。
3. **Reacher**
   - 使用 joint angles、end-effector-to-target distance、target quadrant。
   - 检查不同 joint/target geometry 下 rollout readout 是否保持 margin。
4. **Cube**
   - 使用 cube position/orientation、gripper-object relation 或 action-relevant object pose。

**建议新增指标：**

```text
Semantic Discriminability Ratio = median different-state semantic rollout distance / median same-state noisy rollout distance
```

或：

```text
Selective Margin Pass Rate = P[different-state distance > same-state noisy radius + margin]
```

**建议新增表格：**

```latex
Table: Task-semantic selective ACPC guard.
Columns:
Task | Semantic factor | Same-state noisy radius | Different-state semantic margin | Pass rate | Base -> Robust
```

**验收标准：**

- 主文至少给一个 compact table。
- Appendix 给每个任务的具体 pair construction。
- 不能只说 rank/probe 没 collapse；必须说明 action-relevant distinctions 是否保留。

**建议改动位置：**

- `Action-Conditioned Predictive Consistency`：把 semantic discriminability guard 和定义对应起来。
- `Experiments / Collapse guard and negative checks`：替换或增强当前 proxy guard table。
- Appendix 增加 pair construction 和实现细节。

---

## 2. P1：强烈建议完成的增强项

### P1.1 统一 representative row 选择规则，减少 cherry-picking 风险

**问题：** 当前不同表中的 representative robust rows 不完全一致，例如 ACPC-basin compact rows、Phase-0 table 和 sweep-best rows有时使用不同 `std_max`。虽然文中解释了原因，但 reviewer 仍可能认为选择带有事后性。

**整改目标：** 建立统一、冻结、可复现的 row selection protocol。

**建议执行：**

选择以下一种策略并全文统一：

1. **Fixed high-noise endpoint：** 所有任务统一使用 `std_max = 0.08`。
2. **Prospective dev-selected row：** 在 dev set 上用固定 rule 选择，再在 held-out eval。
3. **Plateau-aware summary：** 不选单点，报告 plateau mean / best-within-CI / area under noise-sweep curve。

**推荐方案：**

- 主文使用 fixed `std_max = 0.08` 做跨任务对照。
- Appendix 补充 full sweep 和 plateau analysis。
- 如果某任务 `0.08` 不是最佳点，明确说这是 endpoint diagnostic，不是 point-best claim。

**验收标准：**

- 每张主文表都能解释为什么用这个 checkpoint。
- 不出现“同一任务不同表为了更好看而换 row”的观感。

---

### P1.2 证明 ACPC 相比 simpler metrics 有增量价值

**问题：** 当前 reviewer 可能问：为什么不用 encoder distance、rollout drift、clean score、training noise level 这些简单指标？ACPC 是否真的提供额外信息？

**整改目标：** 给出 ACPC-style metrics 的 incremental explanatory value。

**建议执行：**

1. 对 full grid 计算相关性：
   - encoder radius `R_E` vs noisy success。
   - rollout radius `R_F` vs noisy success。
   - ACPC-H vs noisy success。
   - PCC / CRA / MAF vs noisy success。
   - rollout drift vs noisy success。
2. 做 partial correlation 或 small regression：
   - noisy success ~ clean success + encoder metric。
   - noisy success ~ clean success + encoder metric + ACPC。
   - 比较 R² / rank accuracy。
3. 报告 failure/boundary cases：
   - 哪些任务或 perturbations 下 ACPC 不预测。
   - 这会增强可信度。

**建议新增图/表：**

```latex
Figure: Metric-vs-noisy-success scatter across full grid.
Table: Incremental predictive value of ACPC-style readouts.
```

**验收标准：**

- 主文能回答：ACPC 比 encoder invariance 更有 control-facing value。
- 不是只靠概念论证，而有 full-grid 定量支持。

---

### P1.3 加强 baseline/context comparison

**问题：** 当前 related work 覆盖较广，但缺少 competing baseline。若定位为 diagnostic paper，可以不做完整 SOTA，但需要更清楚地说明为什么这些 baseline 不在 scope，或者加入轻量比较。

**建议执行：**

可选路线：

1. **轻量 baseline 实验：**
   - 标准 pixel augmentation baseline。
   - encoder consistency regularization baseline。
   - latent consistency regularization baseline。
   - direct denoising / target-view branch 已有，可保留为 negative baseline。
2. **若不加实验：**
   - Related Work 中更明确区分：ACPC 是 checkpoint diagnostic，不是 robust visual control method。
   - Discussion 中说明 method comparison 是后续方向。

**更优方案：**

至少加入一个简单 baseline：

```text
Input-side Gaussian noise training vs encoder-consistency regularization vs target-view denoising
```

并报告 closed-loop success + ACPC + semantic guard。

**验收标准：**

- reviewer 不会误以为论文在和 DrQ/TD-MPC/Dreamer 做鲁棒方法竞争。
- 如果声称 diagnostic insight，就要说明 insight 对 baseline failure/success 的解释力。

---

### P1.4 压缩主文指标，降低读者负担

**问题：** 当前指标很多：ACPC-H、PCC、CRA、MAF、ADM、SPRR、R_E、R_F、rollout drift、transition-resolution、ID probe 等。主文容易显得像 metric dump。

**整改目标：** 主文只保留最核心的 3–4 个诊断，其他放 appendix。

**建议主文保留：**

1. Closed-loop noisy success / drop。
2. ACPC-H 或 `R_F`。
3. PCC/CRA 二选一或 compact paired readout。
4. Semantic discriminability guard。

**建议放 appendix：**

- ADM / SPRR。
- full diagnostic five-layer table。
- CKA / effective rank detailed analysis。
- latent-noise mechanism probes。

**验收标准：**

- Introduction 的 contribution 与 main experiments 一一对应。
- 每个主文 metric 都回答一个必要问题。

---

## 3. P2：写作和定位修改

### P2.1 调整 title / abstract，避免过度泛化

**当前风险：** 标题和摘要容易让 reviewer 期待一个完整 robustness method 或 robust-control guarantee，但正文实际是 diagnostic-localization study。

**建议标题方向：**

```text
Action-Conditioned Predictive Consistency for Diagnosing Gaussian-Noise Robustness in JEPA World Models
```

或：

```text
Diagnosing Visual Robustness in JEPA World Models via Action-Conditioned Predictive Consistency
```

**摘要建议强化：**

- 第一段明确：diagnostic study, not new objective。
- 中段突出：prospective validation / seed-level evidence，如果 P0 完成。
- 末尾保留 scope boundary，但不要过多“我们没做什么”，避免削弱贡献。

---

### P2.2 重写 contributions，使其更像顶会主贡献

当前 C1/C2/C3 比较诚实，但有些像 release-package 描述。建议改成：

1. **Problem/formalization:** 定义 selective ACPC，将 visual robustness 从 encoder invariance 推到 action-conditioned predictive dynamics。
2. **Diagnostic validation:** 在 LeWM/PLDM、多任务、多 seed 下验证 ACPC-style readouts 对 matched Gaussian stressor 的 localization/prediction value。
3. **Discriminability and failure analysis:** 用 semantic/proxy guards 和 negative ablations 说明 consistency 必须 selective，避免 collapse。

如果 P0 prospective validation 完成，可以把第二点写得更强。

---

### P2.3 重新平衡 limitations 的语气

当前版本的 limitation 很诚实，但“不是 method / 不是 predictor / 不是 guarantee”出现太多，会让 reviewer 觉得贡献被作者自己削弱。

**建议：**

- 保留边界，但集中到 `Limitations` 一节。
- 在 Introduction / Abstract 中正向表达：本文建立了什么、验证了什么。
- 不要在每个结果后都反复强调“不是 predictor”，改成一次性说明：`We evaluate diagnostics as localization tools unless otherwise stated; Section X tests prospective validity.`

---

## 4. 建议 Codex 执行顺序

### Step 1：先做不需要新实验的文本结构整改

1. 统一 contribution 表述。
2. 简化主文 metric set。
3. 统一 representative row selection policy。
4. 把 post-hoc / prospective 的边界集中写清楚。
5. 更新 title/abstract。

### Step 2：整理现有 artifact，先做 full-grid metric analysis

1. 从现有 JSON 中抽取 full-grid rows。
2. 计算 ACPC/R_E/R_F/PCC/CRA/MAF 与 noisy success 的相关性。
3. 生成 scatter/table。
4. 判断是否已有足够证据支持 P1.2。

### Step 3：补 semantic discriminability guard

1. 先实现 PushT 和 TwoRoom 的 semantic pair construction。
2. 如时间允许扩展到 Reacher/Cube。
3. 主文先放 compact 4-task table，appendix 放细节。

### Step 4：补 independent training seeds / prospective validation

1. 最低版本：每任务 3 training seeds 的 base vs fixed `std_max=0.08`。
2. 更强版本：dev/held-out split + frozen diagnostic rule。
3. 更新主表和 claims。

---

## 5. 推荐新增文件/脚本

可由 Codex 按实际项目结构调整名称。

```text
paper1/TOP_CONF_REMEDIATION_PLAN.md             # 本整改计划
paper1/prospective_validation_protocol.md       # 冻结诊断协议说明，可选
scripts/paper1_metric_correlation.py            # full-grid metric correlation
scripts/paper1_semantic_discriminability.py      # semantic guard computation
tools/paper1_prospective_validation.py          # dev/held-out validation, if needed
assets/paper1_data/prospective_validation.json  # 新 artifact
assets/paper1_figs/metric_correlation_*.pdf/png # 新图
```

---

## 6. 最小可接受顶会改版 checklist

投稿前至少完成：

- [ ] 主文加入 independent training seed evidence。
- [ ] 主文加入 prospective 或 quasi-prospective diagnostic validation。
- [ ] 主文加入 semantic discriminability guard，不能只靠 effective rank / ID probe。
- [ ] 统一 representative checkpoint selection policy。
- [ ] 给出 ACPC 相对 simpler metrics 的增量证据。
- [ ] 明确标题/摘要定位为 diagnostic，不夸大为 robust-control method。
- [ ] 保留 negative checks，但避免主文过度堆叠指标。
- [ ] Appendix 保留 full artifact map、full sweep、PLDM replication、target-view ablation、heteroscedastic failure。

---

## 7. 建议最终 claim 边界

如果 P0 补强完成，建议主张：

> We introduce selective action-conditioned predictive consistency as a control-facing diagnostic for visual robustness in JEPA world models. Across LeWM and PLDM, Gaussian-noise robustness failures and recoveries are better localized by paired action-conditioned rollout and candidate-cost readouts than by encoder invariance alone. Prospective held-out validation and semantic discriminability guards show when these diagnostics predict or fail to predict closed-loop robustness under bounded perturbation settings.

如果 P0 补强无法完成，建议保守主张：

> We provide a controlled post-hoc diagnostic study showing that Gaussian-noise recovery in JEPA world-model control coincides with contraction of same-state action-conditioned rollout readouts while preserving proxy action-relevant discriminability. The diagnostics localize failures under matched Gaussian stress tests but are not yet validated as general checkpoint selectors.

---

## 8. 一句话执行重点

优先把论文从“我们观察到 noise training 恢复，并用 ACPC 解释”改成：

> **我们定义了 selective ACPC，并证明/验证它在 held-out 设置中比 encoder invariance 更能诊断或预测 visual world-model control robustness，同时用 semantic guard 排除了 collapse。**
