# Paper1 指标压缩整改计划

> 目标：把 `paper1` 当前较多的诊断读数压缩成理论上闭合、主文可读、可被 Codex 执行的一组核心指标。
>
> 最终判断：**应该压缩，而且应该压缩到两个理论成分；不建议压成单一无保护标量。** 主文核心诊断轴应为：
>
> 1. `ATR`：ACPC Tail Risk，即 same-state clean/noisy action-conditioned rollout disagreement 的 tail risk；
> 2. `SMPR`：Selective Margin Pass Rate，即 task-grounded different-state / near-boundary separation 是否大于 same-state noisy radius。
>
> 旧的 `PCC/CRA/MAF`、CEM trace、rank/transition/ID-probe 等不再作为主文核心指标；它们应退出 paper-facing 主叙事，迁入 `paper1/docs/LEGACY_DIAGNOSTIC_AUDITS.md` 和 `DATA_MANIFEST.md` provenance。**不要物理删除 artifacts；可以从主文删除旧指标表，只保留一句 shadow-audit 指针。**

---

## 0. 极具批判性的结论

你的思路是对的：如果文章想有连贯性，指标必须和理论最后剩下的成分一一对应。当前理论真正需要的不是一串平行指标，而是两个必要条件：

```text
same-state predictive-stability tail 小
+
action-relevant / task-grounded selective margin 大
```

也就是：

```text
low ACPC-tail risk + high selective-margin pass rate
```

但有一个关键限制：**不能把这两个条件再压成一个主文单分数**。单分数会把 stability 和 selectivity 混掉，正好掩盖 ACPC 理论里最重要的 collapse 反例。

因此，主文应该大胆删除 metric zoo，但不能删除证据资产。推荐策略是：

```text
Paper main text: only ATR + SMPR as core diagnostics.
Paper appendix: only minimal proof/calibration + perhaps one short pointer to legacy audits.
Legacy MD / manifest: PCC/CRA/MAF, CEM trace, rank/ID probe, older ADM/SPRR, full selector audits.
Artifacts: all retained, hash-checked, no physical deletion.
```

---

## 1. 压缩是否有风险？

有风险，但风险来自错误压缩，而不是来自压缩本身。

### 1.1 错误压缩：一个万能分数

不建议主文只报告一个分数，例如：

```text
S = ACPC + lambda * PCC - gamma * CRA + beta * MAF
```

原因：

1. **collapse 会被掩盖。** 低 ACPC 可以由 collapsed encoder/predictor 产生。如果单分数主要奖励低 ACPC，就可能把坏模型误判成好模型。
2. **planner margin 会被掩盖。** ACPC 小不等于 top action 稳定。clean top-1/top-2 margin 很小时，微小 cost drift 也能 flip。
3. **task semantics 会被掩盖。** PushT/Cube 的 contact/object-pose distinctions 是局部关键因素；全局平均分容易掩盖局部 failure。
4. **plateau 内排序会被过拟合。** 当前 evidence 支持 plateau-entry enrichment，不支持 point-best ranking。单分数会诱导 reviewer 以为你在做 checkpoint ranker。
5. **会和现有 selector audit 冲突。** 当前 aggregate ACPC/PCC/CRA/MAF 并不明显优于 MAF-only 或 high-std top-half reference；强推单分数容易被质疑。

### 1.2 正确压缩：两个必要条件

压成两个轴是合理的，因为这两个轴正好对应理论上不可替代的两个成分：

1. `ATR`：控制 same-state clean/noisy predictive disagreement 的 tail；
2. `SMPR`：检查 action-relevant / task-grounded different-state pairs 没被 collapse。

这两个成分不应再互相抵消。主文表格应该让 reviewer 直接看到：

```text
ATR base -> robust endpoint: 明显下降
SMPR base -> robust endpoint: 明显上升或保持高位
```

如果某任务，尤其 Cube，出现弱项，要如实作为 boundary case，而不是用单分数遮掉。

---

## 2. 理论到指标的一一对应

### 2.1 理论核心 1：ACPC-tail term

当前 sampled-pool stability theorem 的核心事件是：

```text
D(a) = d_H( rollout_readout_clean(a), rollout_readout_noisy(a) )
```

如果：

```text
Pr_{a ~ q}[D(a) > epsilon] <= delta
```

则一次 sampled candidate pool 的 flip risk 有项：

```text
K * delta
```

因此，经验指标不应只是 median `R_F`。median 只能说明典型样本变好了，不能对应 theorem 中的 tail event。

**压缩指标 A：ATR — ACPC Tail Risk**

```text
D_same_i = normalized ACPC-H/trans for paired clean/noisy same-state item i
ATR_q90 = Q_90({D_same_i})
ATR_CVaR90 = mean(D_same_i | D_same_i >= ATR_q90)
```

推荐主文报告：

```text
ATR_q90
```

appendix / artifact 可同时保留：

```text
ATR_CVaR90
```

理论对应：

| 理论对象 | 指标 | 解释 |
|---|---|---|
| `Pr[D > epsilon] <= delta` | `ATR_q90` / `ATR_CVaR90` | 估计 ACPC-tail risk |
| fixed-candidate cost drift bound | `ATR` upstream | rollout readout 不稳定会传到 cost drift |
| local Gaussian sensitivity | `ATR` under Gaussian noise | 测的是 action-conditioned `J_G J_E` sensitivity，不是 encoder invariance |

### 2.2 理论核心 2：clean-margin / top-1 instability term

sampled-pool theorem 另一项是：

```text
Pr[Delta_A <= 2 L_J epsilon]
```

这说明：即使 ACPC-tail 小，如果 clean candidate margin 很小，top-1 仍可能不稳定。

这部分在最终压缩中不建议变成第三个核心指标，因为它更接近 planner audit，而不是 representation/predictive diagnostic 的本体。处理方式：

- `MAF` 保留为 shadow audit；
- `PCC/CRA` 保留为 cost/ranking companion audit；
- `CEM trace` 保留为 fixed-pool 到 optimizer 的 gap audit；
- 主文只用一句说明：planner-facing audits are consistent and archived, but core diagnostic is ATR + SMPR。

如果顶会版需要一个最小 planner audit，优先保留 `MAF`，而不是 PCC/CRA 全表。因为 `MAF` 最贴近 clean-margin flip failure。

### 2.3 理论核心 3：selectivity / no-collapse condition

ACPC alone permits collapse。这个不是辅助细节，而是理论定义的一半。

因此必须有第二个核心指标：

```text
D_same_i = same-state clean/noisy rollout radius
D_diff_i = closest task-label-crossing proxy-different clean rollout distance
M_i = 1[D_diff_i > D_same_i + m]
SMPR = mean_i M_i
SMR = 1 - SMPR
```

推荐主文报告：

```text
SMPR_m0
```

appendix / artifact 可做：

```text
m > 0 sensitivity
```

理论对应：

| 理论对象 | 指标 | 解释 |
|---|---|---|
| discriminability countercondition | `SMPR` | action-/transition-/cost-relevant distinctions must remain separated |
| collapse counterexample | `SMPR` | low ATR alone is insufficient |
| selective ACPC pseudo-metric | `SMPR` companion | same-state contraction only meaningful if different-state margin survives |

### 2.4 最终 theory-to-metric mapping

主文必须出现类似下表，确保文章连贯：

```latex
\begin{table}[H]
\centering
\caption{Theory-to-metric mapping for compressed selective ACPC.}
\label{tab:theory-metric-map}
\small
\begin{tabularx}{\textwidth}{l l X}
\toprule
Theory object & Paper-facing readout & Role \\
\midrule
$\Pr[D>\epsilon]$ / ACPC tail & ATR & same-state action-conditioned predictive-stability tail \\
$d_{\mathrm{diff}}>d_{\mathrm{same}}+m$ & SMPR & selective anti-collapse margin \\
$\Delta_{\mathcal A}\le 2L_J\epsilon$ / realized top-1 flips & MAF legacy audit & planner-facing clean-margin failure audit \\
Cost/ranking drift under shared candidates & PCC/CRA legacy audit & sanity check for fixed-candidate planner link \\
Adaptive optimizer sensitivity & CEM trace legacy audit & gap check beyond fixed candidate pools \\
\bottomrule
\end{tabularx}
\end{table}
```

---

## 3. 是否要从 paper 删除旧指标？

### 3.1 批判性判断

可以从 paper 主文删除旧指标表，而且建议删除大部分旧指标表。否则压缩只是口头说法。

但是不建议完全从项目中删除旧指标，因为：

1. 旧指标是 reviewer 质疑时的 defense material；
2. 旧指标能解释 ATR/SMPR 到 planner/cost/ranking 的过渡；
3. CEM trace 虽然预算有限，但能说明 fixed-pool diagnostics 不是完全脱离 optimizer；
4. rank/ID probe 对 heteroscedastic-loss failure 的解释仍有 provenance 价值；
5. artifact 已经存在并有 manifest/hash，物理删除会破坏可复现记录。

### 3.2 最终处理策略

采用三层结构：

```text
Layer 1 — Main paper:
  ATR + SMPR + closed-loop behavior only.

Layer 2 — Minimal paper appendix, optional:
  one compact paragraph/table pointing to MAF/PCC/CRA/CEM trace as shadow audits.
  No long metric zoo.

Layer 3 — Legacy/shadow document:
  paper1/docs/LEGACY_DIAGNOSTIC_AUDITS.md records all旧指标、为什么降级、何时仍有用、对应 artifacts。
```

### 3.3 旧指标迁移表

| 当前读数 | 主文处理 | Legacy/shadow 处理 | 保留原因 |
|---|---|---|---|
| `R_E` | 删除或仅 figure caption 一句 | legacy | encoder-entry signal，不是最终 robustness diagnostic |
| `R_F` median | 被 ATR tail 替代 | legacy / figure provenance | median 不对应 tail theorem |
| ACPC-H/trans mean/median | 被 ATR tail 替代 | legacy | fallback/provenance |
| PCC | 从主文核心表删除 | legacy planner audit | cost drift companion |
| CRA | 从主文核心表删除 | legacy planner audit | ranking companion |
| MAF | 可保留一句或移 legacy | legacy，必要时最小 planner audit | 最贴近 clean-margin flip term |
| CEM trace | 主文删除或一句 bounded audit | legacy | optimizer-gap audit，预算有限 |
| effective rank | 主文删除 | legacy | proxy anti-collapse check，已被 SMPR 替代 |
| transition L2 | 主文删除 | legacy | proxy guard / hetero failure provenance |
| ID probe | 主文删除 | legacy | proxy guard / hetero failure provenance |
| 8-step rollout drift | 主文删除 | legacy | rollout companion，不是核心 |
| ADM/SPRR | 主文删除 | legacy/provenance | exploratory |

结论：**从 paper 删除旧指标表是可以的；但不要删除 artifacts，也不要删除 manifest provenance。**

---

## 4. Codex 执行计划

### Task A：生成压缩指标 artifact

基于现有 artifacts 生成：

```text
assets/paper1_data/compressed_metrics_summary_20260706.json
assets/paper1_data/compressed_metrics_summary_20260706.md
```

输入优先级：

```text
acpc_phase0_lewm_three_seed.json
semantic_task_grounded_margin_lewm_three_seed.json
margin_flip_curve_lewm_three_seed.json
training_seed_gaussian_lockbox.json
```

最低字段：

```json
{
  "metadata": {
    "metric_version": "compressed_v1",
    "same_state_metric": "ATR_q90",
    "selective_metric": "SMPR_m0",
    "training_seeds": [3072, 3073, 3074],
    "tasks": ["TwoRoom", "PushT", "Reacher", "Cube"],
    "notes": "ATR/SMPR are paper-facing; PCC/CRA/MAF/CEM/rank/ID are legacy audits."
  },
  "rows": [
    {
      "task": "PushT",
      "std": 0.0,
      "clean_success": 0.0,
      "obs008_success": 0.0,
      "ATR_q90": 0.0,
      "ATR_cvar90": 0.0,
      "SMPR": 0.0,
      "SMR": 0.0,
      "MAF_q75_legacy": 0.0,
      "notes": "base or std0.08 endpoint"
    }
  ]
}
```

必须注意：不要把 `MAF_q75_legacy` 写成主指标，只作为 legacy audit 字段。

### Task B：新增主文 compressed table

新增主文表：

```latex
\begin{table}[H]
\centering
\caption{Compressed selective-ACPC diagnostics over training seeds 3072/3073/3074. ATR is the 90th percentile of normalized same-state clean/noisy ACPC-H/trans.; lower is better. SMPR is the task-grounded near-boundary selective-margin pass rate; higher is better.}
\label{tab:compressed-selective-acpc}
\small
\begin{tabular}{lrrrr}
\toprule
Task & ATR base & ATR std0.08 & SMPR base & SMPR std0.08 \\
\midrule
TwoRoom & ... & ... & ... & ... \\
PushT & ... & ... & ... & ... \\
Reacher & ... & ... & ... & ... \\
Cube & ... & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

如果 ATR q90 不能立刻从 artifact 中计算，不要伪造。临时 fallback 只允许在 draft 中标注：

```text
Current table uses ACPC-H/trans. mean/median fallback; final submission must replace this with ATR_q90 or CVaR.
```

### Task C：删除/移动主文旧指标表

从 main text 中删除或降级以下表格/长段落：

```text
compact LeWM shared-candidate readouts: ACPC-H/PCC/CRA/MAF
margin-conditioned action-flip full table
compact representation discriminability table: eff rank / trans L2 / ID probe
CEM trace long table, if currently in main text
```

替换为短段：

```text
Planner-facing and representation-proxy audits are archived in LEGACY_DIAGNOSTIC_AUDITS.md and DATA_MANIFEST.md. They are used as consistency checks, not as independent core diagnostics. The compressed paper-facing diagnostics are ATR and SMPR.
```

### Task D：新增或更新 legacy/shadow doc

新增：

```text
paper1/docs/LEGACY_DIAGNOSTIC_AUDITS.md
```

该文档应包含：

1. PCC / CRA / MAF 的定义、旧角色、新角色；
2. CEM trace 的目的和限制；
3. rank / transition / ID probe 的目的和限制；
4. 为什么这些不再 paper-facing；
5. 对应 artifact 路径；
6. 不得在主文中声称它们是核心指标。

### Task E：更新 theory-to-metric paragraph

在 sampled-pool theorem 后写清楚：

```text
ATR estimates the ACPC-tail term Kδ through high-quantile/CVaR paired rollout disagreement. SMPR is not part of the flip bound; it is the discriminability condition required to make low ATR meaningful rather than collapse. MAF/PCC/CRA are planner-facing audits of the finite candidate-pool link and are archived as legacy checks.
```

这句话很关键：**SMPR 不属于 flip bound，但属于 selective ACPC 定义必需项。** 这样理论上不会混乱。

### Task F：更新 abstract / contributions

建议 abstract 加一句：

```text
We compress the diagnostic to two theory-aligned readouts: a tail risk of same-state action-conditioned rollout disagreement and a task-grounded selective-margin pass rate. Planner-side cost, ranking, and optimizer-trace readouts are retained as shadow audits rather than separate objectives.
```

Contributions 改成：

```text
C1 — Selective ACPC principle.
Robustness diagnostics require low same-state action-conditioned predictive-disagreement tails and preserved task-grounded near-boundary separations.

C2 — Theory-matched compression.
Fixed-candidate and sampled-pool analyses map ACPC tails to candidate-cost/top-1 instability terms; a local Gaussian linearization identifies the measured tail as action-conditioned encoder–predictor sensitivity. The selective-margin readout rules out collapse.

C3 — Evidence chain.
Across three LeWM training seeds and PLDM boundary checks, ATR and SMPR localize matched-Gaussian robustness plateaus, while legacy planner/proxy audits provide consistency checks and negative ablations explain failure modes.
```

---

## 5. 防幻觉和一致性检查

Codex 执行时必须通过以下 gates。

### Gate 1：不要伪造 ATR

如果现有 artifact 没有 sample-level ACPC-H/trans，不能凭表格均值反推 q90。必须从 sample-level diagnostic rows 计算，或者明确 fallback。

### Gate 2：不要把 legacy audit 写成核心指标

`PCC/CRA/MAF/CEM/rank/ID` 可以保留在 legacy MD 和 manifest，但主文不能再把它们和 ATR/SMPR 并列为核心。

### Gate 3：不要删 artifact

不删除 JSON、schema、manifest entries。只改 paper-facing hierarchy。

### Gate 4：不要掩盖 selector audit 事实

必须保留事实：aggregate diagnostic screen 不显著优于 MAF-only 或 high-std top-half reference。正确说法是：compressed metrics are theory-facing diagnostics, not a dominated selector claim.

### Gate 5：SMPR 不可删除

任何压缩版本都不能只留 ATR。必须保留：

```text
low ATR is meaningful only with high SMPR / low SMR.
```

### Gate 6：Cube 不可美化

Cube 是 boundary case。压缩表如果 Cube 的 ATR/SMPR 或 behavior 不如其他任务，必须如实写，不用单分数盖掉。

### Gate 7：claim 边界

可以写：

```text
ATR and SMPR summarize selective ACPC for matched Gaussian plateau localization.
```

不要写：

```text
ATR/SMPR universally predict robustness.
ATR/SMPR prove CEM stability.
ATR/SMPR outperform all selectors.
```

---

## 6. 推荐最终主文结构

```text
1 Introduction
  - latent prediction is not robustness
  - selective ACPC = low ATR + high SMPR

2 ACPC theory and compressed diagnostics
  - ACPC definition
  - sampled-pool bound
  - local Gaussian sensitivity
  - theory-to-metric table

3 Protocol
  - three training seeds
  - Gaussian endpoint
  - artifact/legacy policy

4 Experiments
  4.1 Closed-loop Gaussian cliff and recovery
  4.2 Compressed selective-ACPC diagnostics: ATR + SMPR
  4.3 Bounded validation / plateau localization
  4.4 Shadow audits and failure checks summarized briefly, details in legacy doc / appendix

5 Discussion
  - diagnostic not method
  - legacy audits support consistency but not core metrics
  - future method objective: CVaR(D_same) + margin loss
```

---

## 7. 后续方法版本的自然目标

压缩后，新方法可自然写成：

```text
L_selective_ACPC = CVaR_alpha(D_same) + lambda * max(0, m + D_same - D_diff)
```

可选 cost/planner term：

```text
+ mu * PCC_tail
```

但这应该是后续 method-paper 路线，不应放进 arXiv v1 的主 claim。

---

## 8. Codex 最小执行清单

- [ ] 生成 `compressed_metrics_summary_20260706.json/.md`。
- [ ] 从 sample-level rows 计算 `ATR_q90` 和 `ATR_CVaR90`；不能计算就标注 fallback，不得伪造。
- [ ] 从 task-grounded margin artifact 读取/汇总 `SMPR` 和 `SMR`。
- [ ] 主文新增 `tab:compressed-selective-acpc`。
- [ ] 主文新增 `tab:theory-metric-map`。
- [ ] 主文删除或移动 ACPC-H/PCC/CRA/MAF 大表。
- [ ] 主文删除或移动 rank/transition/ID probe 大表。
- [ ] 主文删除或移动 CEM trace 大表，仅留 bounded audit 一句。
- [ ] 新增 `paper1/docs/LEGACY_DIAGNOSTIC_AUDITS.md`。
- [ ] 更新 `DATA_MANIFEST.md`，加入 compressed metrics artifact，并保留 legacy audit artifact 路径。
- [ ] 运行 `tools/check_paper1_consistency.py` 或现有一致性检查脚本。

---

## 9. 最终推荐口径

压缩后的论文口径：

> Selective ACPC is summarized by two theory-aligned diagnostics: low tail risk of same-state clean/noisy action-conditioned rollout disagreement, and high task-grounded selective-margin pass rate. The first captures predictive stability under nuisance perturbations; the second prevents collapse by checking action-relevant separation. Planner-side PCC/CRA/MAF, CEM traces, and representation probes are retained as legacy/shadow audits, not as independent core metrics.

一句话版：

```text
最终核心不是一堆指标，而是两件事：同态噪声 rollout tail 要小，异态任务边界 margin 要大。
```
