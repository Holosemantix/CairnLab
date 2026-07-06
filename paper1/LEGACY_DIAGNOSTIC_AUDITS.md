# Paper1 Legacy / Shadow Diagnostic Audits

> 本文档用于承接从 `paper1` 主文和 PDF 附录退出的旧诊断读数、负 ablation 和项目排查记录。
>
> 主文核心只应保留：
>
> ```text
> ATR: ACPC Tail Risk, lower is better
> SMPR: Selective Margin Pass Rate, higher is better
> ```
>
> 本文档中的内容只作为 repository-level / team-internal provenance、debug、reviewer defense 和后续方法设计材料，不再作为 Paper1 PDF 的正文或附录内容。**Paper1 PDF 中不应出现“legacy audits are retained in the release package”这类项目管理句子。**

---

## 1. 为什么要建 legacy 文档

指标压缩后，主文不能继续堆叠 `ACPC-H/PCC/CRA/MAF/R_E/R_F/rank/ID/CEM trace`，否则 reviewer 会认为核心贡献不清楚。

负 ablation 表也不应继续留在 Paper1 PDF。它们解释了方法探索失败路径，但压缩版 Paper1 不是方法论文；这些表会把读者注意力从 `ATR + SMPR` 拖回训练目标排查和项目报告。

这些材料不物理删除，原因是：

1. 它们解释了从 ACPC 到 planner/cost/ranking 的中间链条；
2. 它们保留了 heteroscedastic-loss failure、target-view ablation、PLDM replication 等 provenance；
3. 许多 artifacts 已经有 hash 和 manifest，删除会破坏可复现记录；
4. 后续方法迭代仍可能需要这些读数做 debug；
5. 如果 reviewer 以后要求补充材料，可以从这些文档中恢复，但不应默认进入 Paper1 PDF。

因此采用策略：

```text
Paper main text: closed-loop behavior + ATR + SMPR.
Paper appendix: no legacy-audit tables; only proof/calibration and essential protocol details.
This legacy doc: old metrics, negative ablations, roles, artifact paths.
DATA_MANIFEST.md: artifact inventory and hashes.
```

---

## 2. Core vs legacy hierarchy

| Level | 内容 | Paper-facing role |
|---|---|---|
| Core behavior endpoint | clean success, obs-noise 0.08 success, drop/gain | 必须主文保留，说明 closed-loop behavior |
| Core diagnostic 1 | ATR / ACPC-tail risk | 主文核心，理论对应 `Pr[D>epsilon]` |
| Core diagnostic 2 | SMPR / selective margin pass-rate | 主文核心，理论对应 discriminability / no-collapse |
| Legacy planner audits | PCC, CRA, MAF, CEM trace | 不进 Paper1 PDF；仅 repository/internal provenance |
| Legacy representation probes | effective rank, transition L2, ID probe, CKA, rollout drift | 不进 Paper1 PDF；仅 failure provenance / debug |
| Legacy negative ablations | target-view ablation, heteroscedastic-loss ablation | 不进 Paper1 PDF；仅方法探索 provenance |
| Legacy exploratory summaries | ADM, SPRR, top-k overlap, latent-noise probes | 不进 Paper1 PDF；optional debug |

---

## 3. Legacy planner audits

### 3.1 PCC — Predictive Cost Consistency

**Old role:** candidate-cost drift readout.

**Definition sketch:**

```text
PCC = | J(F^H(E(o), a), g) - J(F^H(E(o_tilde), a), g) |
```

**Why downgraded out of Paper1:**

- PCC is downstream of ACPC and cost readout `J`.
- It is useful for auditing fixed-candidate cost stability, but it is not the primitive diagnostic.
- It should not compete with ATR as a separate paper-facing metric.
- Including it in the PDF would invite planner-stability review questions that the compressed paper no longer claims to answer.

**New role:**

```text
repository-level planner audit: cost-drift companion to ATR
```

**Relevant artifact families:**

```text
assets/paper1_data/acpc_phase0_clean_goal_seed9101.json
assets/paper1_data/acpc_phase0_lewm_three_seed.json
assets/paper1_data/unseen_phase0_acpc_subset.json
assets/paper1_data/unseen_phase0_acpc_fullstress.json
```

### 3.2 CRA — Candidate Ranking Agreement

**Old role:** ranking stability readout between clean/noisy candidate cost vectors.

**Why downgraded out of Paper1:**

- CRA is useful but indirect.
- It can be high even when top-1 action changes under small margins.
- It does not directly prevent collapse.
- It depends on candidate-pool construction.

**New role:**

```text
repository-level planner audit: ranking companion to ATR/PCC
```

**Relevant artifact families:**

```text
assets/paper1_data/acpc_phase0_clean_goal_seed9101.json
assets/paper1_data/acpc_phase0_lewm_three_seed.json
assets/paper1_data/unseen_phase0_acpc_subset.json
assets/paper1_data/unseen_phase0_acpc_fullstress.json
```

### 3.3 MAF — Margin-Conditioned Action Flip

**Old role:** high-margin top-1 flip readout.

**Why still useful internally:**

- MAF is closest to the clean-margin failure term in sampled-pool theory.
- Existing selector audit shows MAF-only is a competitive single-readout reference.

**Why downgraded out of Paper1:**

- MAF depends on finite candidate pool construction and thresholding.
- It is planner-facing, not a general representation/predictive-dynamics diagnostic.
- Reporting it would imply empirical planner-action stability validation, which the compressed paper should not claim.

**New role:**

```text
repository-level finite-pool margin-failure audit
```

**Relevant artifact families:**

```text
assets/paper1_data/margin_flip_curve_lewm_three_seed.json
assets/paper1_data/selector_plateau_audit_20260704.json
assets/paper1_data/acpc_phase0_lewm_three_seed.json
```

### 3.4 Reduced-budget CEM trace

**Old role:** bridge from fixed-pool diagnostics to actual CEM optimizer traces.

**Why downgraded out of Paper1:**

- Budget is reduced and offline.
- It is not a replacement for closed-loop evaluation.
- It does not prove adaptive CEM stability.
- Some task-specific results, e.g. TwoRoom seeded top-1 flip, remain mixed.
- Including it in the PDF creates more attack surface than evidence value for the compressed ATR/SMPR claim.

**New role:**

```text
repository-level optimizer-gap audit
```

**Relevant artifact families:**

```text
assets/paper1_data/cem_trace_audit_20260704.json
assets/paper1_data/cem_trace_audit_20260704.md
```

---

## 4. Legacy representation / anti-collapse proxies

### 4.1 Effective rank

**Old role:** representation collapse sanity check.

**Why downgraded out of Paper1:**

- It is a generic representation statistic.
- It does not directly measure action-relevant discriminability.
- SMPR is closer to selective ACPC because it compares same-state noisy radius against task-grounded label-crossing pairs.

**New role:**

```text
repository-level proxy guard / provenance
```

### 4.2 Transition L2 / transition-resolution ratio

**Old role:** checks whether transition-scale distinctions collapse.

**Why downgraded out of Paper1:**

- Useful for failure analysis, especially heteroscedastic-loss failure.
- Still proxy-level and not task-semantic enough for the core selective claim.
- Replaced paper-facing by SMPR.

**New role:**

```text
repository-level proxy guard, especially for failure cases
```

### 4.3 ID probe R²

**Old role:** inverse-dynamics information sanity check.

**Why downgraded out of Paper1:**

- Useful but linear-probe dependent.
- Does not directly check near-boundary task distinctions.
- Replaced paper-facing by SMPR.

**New role:**

```text
repository-level proxy guard / heteroscedastic failure provenance
```

### 4.4 8-step rollout drift

**Old role:** rollout-space stability companion.

**Why downgraded out of Paper1:**

- It is not selective.
- It does not by itself rule out collapse.
- ATR tail is the theory-matched replacement.

**New role:**

```text
repository-level rollout companion
```

**Relevant artifact families for this section:**

```text
assets/paper1_data/canonical_diagnostics_20260517.json
assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json
assets/paper1_data/canonical_diagnostics_pldm_20260522.json
assets/paper1_data/canonical_evals_20260517.json
```

---

## 5. Legacy negative ablations

### 5.1 Target-view ablation

**Old role:** show that perturbed-history `->` original-future target-view denoising does not reproduce the main Gaussian-noise recovery.

**Why it is redundant for compressed Paper1:**

- The compressed paper is a diagnostic paper, not a method-comparison paper.
- ATR + SMPR do not require this ablation to be interpretable.
- Keeping the table invites method-claim questions: why not compare more objectives, why only this ablation, why not tune it?
- It distracts from the clean ATR/SMPR story.

**New role:**

```text
repository-level method-exploration provenance
```

**Relevant artifact family:**

```text
assets/paper1_data/target_view_closed_loop_summary.json
```

### 5.2 Heteroscedastic-loss negative result

**Old role:** show that prediction difficulty / loss reweighting can collapse PushT-relevant distinctions and harm clean control.

**Why it is redundant for compressed Paper1:**

- SMPR already provides the paper-facing anti-collapse check.
- The heteroscedastic result is a method-development failure case, not part of the compressed diagnostic proof.
- Including the table reintroduces rank/transition/ID-probe proxy metrics that the compressed paper is trying to remove.
- It creates the impression of a broader objective-design paper without enough competing objective baselines.

**New role:**

```text
repository-level failure-case provenance for future method design
```

**Relevant artifact families:**

```text
assets/paper1_data/target_view_closed_loop_summary.json
assets/paper1_data/canonical_diagnostics_20260517.json
```

---

## 6. Legacy exploratory metrics

### ADM / SPRR

**Old role:** exploratory action-relevant discriminability and selective robustness summaries.

**Why downgraded out of Paper1:**

- Useful for debug but not necessary for the final core story.
- SMPR is simpler, task-grounded, and easier to explain.

**New role:**

```text
repository-level exploratory / provenance only
```

### R_E / R_F median

**Old role:** ACPC-basin visualization and same-state perturbation contraction.

**Why downgraded out of Paper1:**

- `R_E` measures encoder-entry, not control-facing predictive stability.
- `R_F` median is useful but does not match the sampled-pool tail theorem.
- ATR tail replaces `R_F` median as the main paper-facing diagnostic.

**New role:**

```text
repository-level visualization companion / historical basin table provenance
```

Paper-facing exception:

```text
A single qualitative encoder/predictor feature-neighborhood illustration may reuse cached neighborhood points only as visual intuition. It must not report R_E/R_F or revive the ACPC-basin table as a diagnostic claim.
```

**Relevant artifact families:**

```text
assets/paper1_data/acpc_basin_diagnostics.json
assets/paper1_data/acpc_basin_diagnostics_pldm.json
```

---

## 7. What remains paper-facing

After compression, the paper-facing diagnostic section should focus only on:

```text
ATR: normalized same-state ACPC-H/trans tail risk
SMPR: task-grounded near-boundary selective margin pass-rate
```

Do not write in the Paper1 PDF:

```text
Planner-facing cost/ranking/flip and reduced-budget optimizer-trace audits are retained as shadow checks in the release package.
```

Do not write:

```text
PCC, CRA, MAF, rank, ID probe, and CEM trace are all equally important diagnostics.
```

Do not include negative ablation tables unless Paper1 is later rewritten as a method-comparison paper.

---

## 8. Safety / no-hallucination policy for Codex

1. Do not report ATR q90 unless it is computed from sample-level or per-row diagnostic values.
2. Do not infer ATR q90 from a mean/median table.
3. Do not delete artifacts from `assets/paper1_data`.
4. Do not remove manifest entries; update their role as legacy if needed.
5. Do not claim legacy audits prove CEM stability.
6. Do not claim compressed diagnostics outperform MAF-only or high-std references unless a new audit actually shows it.
7. Do not remove SMPR or selective-margin evidence from the paper.
8. Do not hide Cube or other boundary cases behind a combined scalar.
9. Do not put release-package / legacy-audit sentences into the Paper1 PDF.
10. Do not put target-view or heteroscedastic-loss negative ablation tables into the Paper1 PDF.

---

## 9. Checklist for moving old content out of main paper

- [ ] Remove ACPC-H/PCC/CRA/MAF table from main text and appendix.
- [ ] Remove rank/transition/ID probe table from main text and appendix.
- [ ] Remove CEM trace table/prose from main text and appendix.
- [ ] Remove target-view ablation table from PDF appendix.
- [ ] Remove heteroscedastic-loss negative-result table from PDF appendix.
- [ ] Keep ATR + SMPR table in main text.
- [ ] Keep theory-to-metric mapping only for ATR and SMPR.
- [ ] Keep closed-loop behavior table in main text.
- [ ] Update `DATA_MANIFEST.md` as repository provenance, not paper content.
- [ ] Run consistency checks.

---

## 10. Bottom line

The final compressed story should be:

```text
Behavior endpoint: closed-loop Gaussian success/drop.
Core diagnostic A: ATR, low same-state ACPC tail.
Core diagnostic B: SMPR, high selective margin.
Repository/internal provenance: PCC/CRA/MAF/CEM/rank/ID/negative ablations.
```

This keeps the paper theoretically coherent without losing useful internal evidence.
