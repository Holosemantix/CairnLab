# Paper1 指标压缩整改计划

> 目标：把 `paper1` 当前较多的诊断读数压缩成理论上闭合、主文可读、可被 Codex 执行的一组核心指标。
>
> 结论先行：**可以压缩，但不要压成单一无保护标量。主文应压缩为两个核心诊断轴：`ACPC-tail / predictive stability risk` + `selective margin / anti-collapse risk`。** 其他读数保留为 appendix / shadow audit，不再作为主贡献指标。

---

## 0. 当前判断

当前论文已经有一条比较完整的诊断链：

1. closed-loop Gaussian robustness cliff / recovery；
2. ACPC basin / ACPC-H；
3. PCC / CRA / MAF fixed-pool planner readouts；
4. margin-conditioned action-flip audit；
5. task-grounded near-boundary proxy margin；
6. reduced-budget CEM trace audit；
7. representation proxy guards：effective rank、transition L2、ID probe；
8. PLDM、blur/resize、target-view、heteroscedastic-loss 等边界检查。

这条链现在足够完整，但主文指标仍偏多。顶会版本应避免给 reviewer 一种 “metric zoo / diagnostics dump” 的观感。

**推荐主张从：**

> 我们有 ACPC-H、PCC、CRA、MAF、R_E、R_F、rank、ID probe、semantic margin、CEM trace 等一组诊断。

**收敛为：**

> selective ACPC 由两个必要成分构成：
>
> 1. same-state corrupted/clean action-conditioned rollout disagreement 的 tail risk 要低；
> 2. task-grounded different-state / near-boundary separations 相对于 same-state noisy radius 要保持 margin。

即：

```text
low ACPC tail  +  high selective margin
```

这是最能和当前理论部分匹配的压缩方式。

---

## 1. 是否有压缩风险？

有风险，但可以控制。

### 1.1 压缩成一个单标量的风险

不建议直接把所有指标合成一个主文单标量，例如：

```text
S = ACPC + lambda * PCC - gamma * CRA + beta * MAF
```

风险包括：

1. **collapse 被掩盖。** 低 ACPC 可以由 collapsed encoder/predictor 产生。如果单标量主要奖励低 ACPC，就会把坏模型误判为好模型。
2. **planner margin 被掩盖。** ACPC 小不等于 top action 稳定；clean top-1/top-2 margin 小时，微小 cost drift 仍可能 flip。
3. **task semantics 被掩盖。** PushT/Cube 这类 contact/object-pose-heavy 任务需要保留局部 state distinction；一个全局平均分可能掩盖局部 failure。
4. **plateau 内排序过拟合。** 当前数据支持 plateau-entry enrichment，不支持 point-best ranking。单标量容易诱导读者以为可以排序所有 checkpoint。
5. **和已有结果冲突。** 当前 plateau-membership audit 中 aggregate ACPC/PCC/CRA/MAF 并不明显优于 MAF-only 或 high-std top-half reference；如果强推单标量，容易被 reviewer 抓住。

### 1.2 压缩成两个理论成分的风险

压成两个轴的风险小很多，因为这两个轴分别对应当前理论里的必要条件：

- same-state predictive stability；
- action-relevant discriminability / margin。

但仍需要保留 shadow audit：PCC、CRA、MAF、CEM trace、rank/ID probe 不应删除，只应降级为 appendix checks。这样即使主文压缩，也不会丢掉 planner/cost/ranking 的证据链。

---

## 2. 理论上应匹配的最终成分

当前理论已经暗示最终指标应该是两个成分，而不是很多平行指标。

### 2.1 成分 A：same-state ACPC tail

理论里的 sampled-pool stability bound 使用的是 tail event：

```text
Pr[D(a) > epsilon] <= delta
```

而不是 median。对于 sampled candidate pool：

```text
Pr[top-1 flip] <= K * delta + Pr[clean margin <= 2 L_J epsilon]
```

因此主指标不应只用 median `R_F`。更适合的是高分位或 CVaR：

```text
ACPC-tail risk = Q_90 or CVaR_90 of normalized ACPC-H/trans.
```

建议命名：

```text
ATR = ACPC Tail Risk
```

其中：

```text
D_same = ACPC-H / natural transition scale
ATR_q90 = quantile_0.90(D_same)
ATR_cvar90 = mean(D_same | D_same >= quantile_0.90(D_same))
```

主文可以报告 `ATR_q90`，appendix 报 `ATR_cvar90` sanity check。

### 2.2 成分 B：selective margin / anti-collapse risk

ACPC alone permits collapse，因此第二个指标必须检查 different-state / action-relevant pairs 是否仍可分。

当前 task-grounded near-boundary proxy margin 已经是正确方向。建议把它形式化为：

```text
D_diff = task-grounded label-crossing proxy-different clean rollout distance
D_same = same-state clean/noisy rollout radius
```

定义 pass event：

```text
M = 1[D_diff > D_same + margin]
```

建议主指标：

```text
SMPR = Selective Margin Pass Rate = mean(M)
```

也可以给 risk 形式：

```text
SMR = Selective Margin Risk = 1 - SMPR
```

为了和 “risk 越低越好” 统一，主文可以写：

```text
Selective risk = 1 - pass-rate
```

但表格更直观时可报告 pass-rate。

### 2.3 成分 C：planner-facing audit，不作为核心轴

PCC、CRA、MAF、CEM trace 都是重要的，但它们应解释理论链条，而不是和 ATR/SMPR 平级。

推荐定位：

- PCC：ACPC 经 cost readout 后的 cost drift audit；
- CRA：candidate ranking alignment audit；
- MAF：high-margin top-1 flip audit，和 sampled-pool theorem 最贴近；
- reduced-budget CEM trace：fixed-pool 到实际 optimizer 的 gap audit。

这些读数应保留在 appendix 或 compact planner-audit table，不作为主贡献指标。

---

## 3. 最终主文应保留的指标

### 3.1 主文核心指标：两个

#### Metric 1：ATR — ACPC Tail Risk

**定义：**

```text
D_same_i = normalized ACPC-H/trans for paired clean/noisy same-state item i
ATR_q90 = Q_90({D_same_i})
```

可选：

```text
ATR_CVaR90 = mean(D_same_i | D_same_i >= ATR_q90)
```

**越低越好。**

**理论对应：**

- sampled-pool bound 中的 `delta = Pr[D > epsilon]`；
- local Gaussian sensitivity 中的 `||J_G J_E||_F`；
- fixed-candidate cost drift bound 的 upstream stability term。

**取代/吸收：**

- `R_F` median；
- ACPC-H/trans mean/median；
- 8-step rollout drift 的主文地位。

`R_F` 可以作为可视化 companion；不要让它成为最终核心指标。

#### Metric 2：SMPR — Selective Margin Pass Rate

**定义：**

```text
D_same_i = same-state clean/noisy rollout radius
D_diff_i = closest task-label-crossing proxy-different clean rollout distance
SMPR = mean_i 1[D_diff_i > D_same_i + m]
```

当前可用 `m=0`，后续可在 appendix 做 `m>0` sensitivity。

**越高越好。**

或 risk 形式：

```text
SMR = 1 - SMPR
```

**理论对应：**

- discriminability countercondition；
- ACPC alone permits collapse 的反例防护；
- selective ACPC pseudo-metric 的必要补充。

**取代/吸收：**

- effective rank；
- transition L2；
- ID probe；
- ADM/SPRR；
- semantic local/global sanity pass。

rank/transition/ID probe 仍保留在 appendix，作为 failure-case provenance 和 heteroscedastic-loss negative check 的辅助说明。

---

## 4. 主文指标结构建议

### 4.1 行为结果仍然必须保留

Closed-loop success/drop 不是诊断指标，而是行为 endpoint。主文必须保留：

```text
clean success
obs-noise 0.08 success
drop / gain
```

推荐主文结果结构：

1. `Table 1`: three-training-seed closed-loop Gaussian cliff/recovery；
2. `Table 2`: compressed diagnostic table with ATR + SMPR；
3. `Table 3`: planner audit compact table, only MAF or MAF + CEM trace summary；
4. appendix: PCC/CRA/full ACPC/PCC/CRA/MAF, full selector audit, PLDM, blur/resize, rank/ID probes。

### 4.2 主文不要再并列堆很多指标

当前主文中 `ACPC-H/trans.`, `PCC`, `CRA`, `MAF` 全部并列表格会让读者觉得有四个核心指标。整改后建议：

- 主文用 ATR + SMPR 做核心；
- PCC/CRA/MAF 表改为 “planner audit”，可移到 appendix 或缩成一段文字；
- MAF 可以在主文留一行，因为它直接对应 clean-margin flip failure；
- CRA/PCC 主要放 appendix。

---

## 5. Codex 执行任务

### Task A：生成压缩指标 artifact

新增或更新脚本，基于现有 `acpc_phase0_lewm_three_seed.json`、`semantic_task_grounded_margin_lewm_three_seed.json`、`margin_flip_curve_lewm_three_seed.json` 生成统一摘要。

建议输出：

```text
assets/paper1_data/compressed_metrics_summary_20260706.json
assets/paper1_data/compressed_metrics_summary_20260706.md
```

最低字段：

```json
{
  "metadata": {
    "metric_version": "compressed_v1",
    "same_state_metric": "ATR_q90",
    "selective_metric": "SMPR_m0",
    "training_seeds": [3072, 3073, 3074],
    "tasks": ["TwoRoom", "PushT", "Reacher", "Cube"]
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
      "MAF_q75": 0.0,
      "notes": "base or std0.08 endpoint"
    }
  ]
}
```

注意：字段名可按现有 JSON schema 调整，但必须明确 `ATR_q90` 和 `SMPR`。

### Task B：新增主文 compressed diagnostic table

建议表格：

```latex
\begin{table}[H]
\centering
\caption{Compressed selective-ACPC diagnostics over training seeds 3072/3073/3074. ATR is the 90th percentile of normalized same-state clean/noisy ACPC-H/trans.; lower is better. SMPR is the task-grounded near-boundary selective margin pass-rate; higher is better.}
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

如果 ATR 不能立刻从 artifact 直接获得 q90，则临时使用 `ACPC-H/trans.` 作为 fallback，但文档中必须标注：

```text
Current table uses mean/median ACPC-H/trans. as a fallback; final submission should use ATR_q90 or CVaR.
```

不要在 final paper 中长期保留 fallback。

### Task C：重写 Operational consistency diagnostics subsection

把当前描述改成：

```text
The main paper uses two compressed diagnostics. ATR measures the tail of same-state clean/noisy action-conditioned rollout disagreement. SMPR measures whether task-grounded near-boundary different-state rollouts stay outside the same-state noisy radius. Together they instantiate selective ACPC: predictive stability without collapse.
```

再说明：

```text
PCC, CRA, MAF, and CEM traces are planner-facing audits of the same stability chain and remain in appendix/main compact audit; they are not independent core objectives.
```

### Task D：重写 theory-to-metric paragraph

在 sampled-pool theorem 后加入明确映射：

```text
ATR estimates the ACPC-tail term K delta. MAF estimates the realized high-margin flip event under a finite sampled pool. SMPR checks the discriminability condition that is absent from the flip bound but required to rule out collapse.
```

建议用一个小表：

```latex
\begin{table}[H]
\centering
\caption{Theory-to-metric mapping for compressed selective ACPC.}
\label{tab:theory-metric-map}
\begin{tabularx}{\textwidth}{l l X}
\toprule
Theory object & Main metric & Role \\
\midrule
$\Pr[D>\epsilon]$ & ATR & same-state predictive-stability tail \\
$d_{diff} > d_{same}+m$ & SMPR & selective anti-collapse margin \\
$\Delta \le 2L_J\epsilon$ / top-1 flips & MAF audit & planner-facing clean-margin failure audit \\
\bottomrule
\end{tabularx}
\end{table}
```

### Task E：降级旧指标

主文中旧指标处理建议：

| 当前读数 | 新位置 | 原因 |
|---|---|---|
| `R_E` | appendix / figure caption | encoder-entry signal，不是最终 robustness diagnostic |
| `R_F` median | appendix / visual reference | 被 ATR tail 替代 |
| ACPC-H/trans mean/median | appendix / fallback | 被 ATR tail 替代 |
| PCC | appendix planner audit | cost drift companion |
| CRA | appendix planner audit | ranking companion |
| MAF | compact main audit or appendix | 和 margin flip 理论贴近，可主文保留一小表 |
| effective rank | appendix | proxy anti-collapse check |
| transition L2 | appendix | proxy anti-collapse check |
| ID probe | appendix | proxy anti-collapse check |
| 8-step rollout drift | appendix | rollout stability companion |
| CEM trace | appendix or one sentence main | planner-gap audit，预算有限 |
| ADM/SPRR | appendix/provenance | exploratory |

不要删除 artifacts；只改 paper-facing hierarchy。

---

## 6. 验收门槛：防止压缩后丢关键因素

Codex 完成后必须通过以下检查。

### Gate 1：理论闭合检查

主文中必须能明确回答：

```text
为什么核心指标不是 encoder distance？
为什么不是 median ACPC？
为什么需要 discriminability guard？
为什么 planner readouts 变成 audit 而不是核心指标？
```

必须出现以下概念映射：

```text
ATR -> ACPC-tail term
SMPR -> selective/discriminability condition
MAF/PCC/CRA -> planner-facing audits
```

### Gate 2：数值不倒退检查

压缩指标摘要必须验证：

1. `ATR_q90` 在 base -> std0.08 上显著下降；
2. `SMPR` 在 base -> std0.08 上显著上升或保持高位；
3. 结果覆盖四个任务和三个 training seeds；
4. Cube 作为 boundary case 不能被过度美化；
5. PushT / Reacher recovery 不应只靠 selected seed。

### Gate 3：与现有 frozen validation 不冲突

新增 compressed metrics 后，必须重新检查或至少引用现有 plateau audit：

- aggregate ACPC/PCC/CRA/MAF 不明显优于 high-std reference；
- MAF-only 目前是 competitive single-readout reference；
- 因此不要声称 compressed metric dominates all alternatives。

正确表述：

```text
The compressed metrics express the theory-facing diagnostic target; planner-facing and high-std references remain audits/baselines, not defeated competitors.
```

### Gate 4：selectivity 不可删除

任何版本都不能只保留 ACPC/ATR，而删掉 SMPR 或 discriminability guard。

必须保留结论：

```text
low ACPC is meaningful only with selective margin evidence.
```

### Gate 5：arXiv/顶会 claim 边界

压缩后主张应是：

```text
Selective ACPC is summarized by two diagnostics: low same-state action-conditioned ACPC tail and high task-grounded selective margin. These localize matched Gaussian robustness plateaus and provide theory-aligned targets for future methods.
```

不要写：

```text
The compressed metric universally predicts robustness.
The compressed metric proves CEM stability.
The compressed metric outperforms all baselines.
```

---

## 7. 推荐改写后的主文结构

### Abstract 中建议改一句

当前 abstract 指标名较多，可改为：

```text
We compress the diagnostic to two theory-aligned readouts: a tail risk of same-state action-conditioned rollout disagreement and a task-grounded selective-margin pass rate. Planner-side cost, ranking, and CEM traces audit the same chain rather than define separate objectives.
```

### Contributions 建议改写

```text
C1 — Selective ACPC principle.
We define robustness diagnostics through two necessary components: low same-state action-conditioned predictive-disagreement tails and preserved task-grounded near-boundary separations.

C2 — Theory-matched diagnostic compression.
Fixed-candidate and sampled-pool analyses show that the ACPC tail controls candidate-cost drift and exposes top-1 instability through ACPC-tail and clean-margin terms; local Gaussian linearization identifies the measured tail as action-conditioned encoder–predictor sensitivity.

C3 — Evidence chain.
Across three LeWM training seeds and PLDM boundary checks, the compressed diagnostics localize matched-Gaussian robustness plateaus; planner readouts and CEM traces audit the same direction, while negative ablations show why the selective margin is necessary.
```

### Experiments 建议结构

```text
4.1 Closed-loop Gaussian cliff and recovery
4.2 Compressed selective-ACPC diagnostics: ATR + SMPR
4.3 Planner-facing audits: MAF / PCC / CRA / CEM trace
4.4 Bounded unseen stressor and failure checks
```

---

## 8. 是否需要一个最终单分数？

不建议主文只用一个分数。

如果需要给 Codex 做 checkpoint screening，可在 appendix 或 artifact 中提供一个非主张单分数：

```text
SACPC-risk = zscore(ATR_q90) + lambda * zscore(1 - SMPR)
```

但必须满足：

1. 主文仍分别报告 ATR 和 SMPR；
2. SACPC-risk 只用于 engineering screening；
3. 不声称 SACPC-risk 是 universal robustness predictor；
4. 必须做 held-out seed audit，不能只在 seed 3072 上调 lambda。

推荐默认：

```text
lambda = 1
```

不要调参到最好看。

---

## 9. 和后续新方法的衔接

指标压缩后，新方法可以自然写成 selective ACPC objective：

```text
L_method = CVaR_alpha(D_same) + lambda * max(0, m + D_same - D_diff)
```

可选 planner/cost term：

```text
+ mu * PCC_tail
```

但 Paper1 arXiv v1 不应声称这个方法已经解决问题。当前 Robust CEM 路线已有 pilot/iteration 记录，但 manifest 里显示它还不够强，不适合作为 main method claim。后续如果做方法版，优先考虑训练时 paired predictive-dynamics consistency，而不是只做 planner reranking。

---

## 10. Codex 最小执行清单

- [ ] 新增 `compressed_metrics_summary_20260706.json/.md`。
- [ ] 计算或抽取 `ATR_q90`、`ATR_CVaR90`、`SMPR`、`SMR`。
- [ ] 主文新增 `tab:compressed-selective-acpc`。
- [ ] 主文新增或更新 `tab:theory-metric-map`。
- [ ] 将 `PCC/CRA/MAF` 从“核心指标表”改成 “planner-facing audit”。
- [ ] 将 `effective rank / transition L2 / ID probe` 降级为 appendix proxy guard。
- [ ] 保留 task-grounded near-boundary margin 为主文 selectivity 证据。
- [ ] 确认 compressed metrics 与三训练种子 Gaussian endpoint、frozen validation、unseen-stressor appendix 不冲突。
- [ ] 运行 `tools/check_paper1_consistency.py` 或现有一致性检查脚本。
- [ ] 更新 `DATA_MANIFEST.md`，加入 compressed metrics artifact 和 SHA。

---

## 11. 最终推荐口径

压缩后的论文口径建议是：

> Selective ACPC has two theory-aligned diagnostics: low tail risk of same-state clean/noisy action-conditioned rollout disagreement, and high task-grounded selective-margin pass rate. The first captures predictive stability under nuisance perturbations; the second prevents collapse by checking action-relevant separation. Planner-side PCC/CRA/MAF and reduced-budget CEM traces audit the same stability chain but are not independent core metrics.

一句话版：

```text
最终核心不是一堆指标，而是两件事：同态噪声 rollout tail 要小，异态任务边界 margin 要大。
```
