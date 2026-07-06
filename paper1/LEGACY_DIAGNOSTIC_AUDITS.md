# Paper1 Legacy / Shadow Diagnostic Audits

> 本文档用于承接从 `paper1` 主文退出的旧诊断读数。目标不是删除证据，而是把 paper-facing 叙事压缩到两个核心指标：`ATR` 和 `SMPR`。
>
> 主文核心：
>
> ```text
> ATR: ACPC Tail Risk, lower is better
> SMPR: Selective Margin Pass Rate, higher is better
> ```
>
> 本文档中的旧读数只作为 legacy / shadow audits，用于 provenance、debug、reviewer defense 和后续方法设计，不再作为主文核心指标。

---

## 1. 为什么要建 legacy 文档

指标压缩后，主文不能继续堆叠 `ACPC-H/PCC/CRA/MAF/R_E/R_F/rank/ID/CEM trace`，否则 reviewer 会认为核心贡献不清楚。

但也不能把这些读数物理删除，原因是：

1. 它们解释了从 ACPC 到 planner/cost/ranking 的中间链条；
2. 它们能在 reviewer 质疑时证明压缩不是 cherry-picking；
3. 它们保留了 heteroscedastic-loss failure、target-view ablation、PLDM replication 等 provenance；
4. 许多 artifacts 已经有 hash 和 manifest，删除会破坏可复现性；
5. 后续方法迭代仍可能需要这些读数做 debug。

因此采用策略：

```text
Paper main text: ATR + SMPR.
Paper appendix: minimal reference to shadow audits if needed.
This legacy doc: old metrics, definitions, role, artifact paths.
DATA_MANIFEST.md: artifact inventory and hashes.
```

---

## 2. Core vs legacy hierarchy

| Level | 内容 | Paper-facing role |
|---|---|---|
| Core behavior endpoint | clean success, obs-noise 0.08 success, drop/gain | 必须主文保留，说明 closed-loop behavior |
| Core diagnostic 1 | ATR / ACPC-tail risk | 主文核心，理论对应 `Pr[D>epsilon]` |
| Core diagnostic 2 | SMPR / selective margin pass-rate | 主文核心，理论对应 discriminability / no-collapse |
| Legacy planner audits | PCC, CRA, MAF, CEM trace | 不再作为核心指标；仅用于支持 planner-facing consistency |
| Legacy representation probes | effective rank, transition L2, ID probe, CKA, rollout drift | 不再作为核心指标；仅用于 failure provenance / sanity checks |
| Legacy exploratory summaries | ADM, SPRR, top-k overlap, latent-noise probes | provenance / optional debug |

---

## 3. Legacy planner audits

### 3.1 PCC — Predictive Cost Consistency

**Old role:** candidate-cost drift readout.

**Definition sketch:**

```text
PCC = | J(F^H(E(o), a), g) - J(F^H(E(o_tilde), a), g) |
```

**Why downgraded:**

- PCC is downstream of ACPC and cost readout `J`.
- It is useful for auditing fixed-candidate cost stability, but it is not the most primitive diagnostic.
- It should not compete with ATR as a separate core metric.

**New role:**

```text
legacy planner audit: cost-drift companion to ATR
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

**Why downgraded:**

- CRA is useful but indirect.
- It can be high even when top-1 action still changes under small margins.
- It does not directly prevent collapse.

**New role:**

```text
legacy planner audit: ranking companion to ATR/PCC
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

**Why still important:**

- MAF is closest to the clean-margin failure term in sampled-pool theory.
- Existing selector audit shows MAF-only is a competitive single-readout reference.

**Why downgraded anyway:**

- MAF depends on finite candidate pool construction and thresholding.
- It is planner-facing, not a general representation/predictive-dynamics diagnostic.
- It should audit the theory, not replace ATR/SMPR.

**New role:**

```text
legacy or minimal appendix audit: realized finite-pool margin failure
```

**Relevant artifact families:**

```text
assets/paper1_data/margin_flip_curve_lewm_three_seed.json
assets/paper1_data/selector_plateau_audit_20260704.json
assets/paper1_data/acpc_phase0_lewm_three_seed.json
```

### 3.4 Reduced-budget CEM trace

**Old role:** bridge from fixed-pool diagnostics to actual CEM optimizer traces.

**Why downgraded:**

- Budget is reduced and offline.
- It is not a replacement for closed-loop evaluation.
- It does not prove adaptive CEM stability.
- Some task-specific results, e.g. TwoRoom seeded top-1 flip, remain mixed.

**New role:**

```text
legacy optimizer-gap audit
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

**Why downgraded:**

- It is a generic representation statistic.
- It does not directly measure action-relevant discriminability.
- SMPR is closer to selective ACPC because it compares same-state noisy radius against task-grounded label-crossing pairs.

**New role:**

```text
legacy proxy guard / provenance
```

### 4.2 Transition L2 / transition-resolution ratio

**Old role:** checks whether transition-scale distinctions collapse.

**Why downgraded:**

- Useful for failure analysis, especially heteroscedastic-loss failure.
- Still proxy-level and not task-semantic enough for the core selective claim.

**New role:**

```text
legacy proxy guard, especially for failure cases
```

### 4.3 ID probe R²

**Old role:** inverse-dynamics information sanity check.

**Why downgraded:**

- Useful but linear-probe dependent.
- Does not directly check near-boundary task distinctions.

**New role:**

```text
legacy proxy guard / heteroscedastic failure provenance
```

### 4.4 8-step rollout drift

**Old role:** rollout-space stability companion.

**Why downgraded:**

- It is not selective.
- It does not by itself rule out collapse.
- ATR tail is the theory-matched replacement.

**New role:**

```text
legacy rollout companion
```

**Relevant artifact families for this section:**

```text
assets/paper1_data/canonical_diagnostics_20260517.json
assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json
assets/paper1_data/canonical_diagnostics_pldm_20260522.json
assets/paper1_data/canonical_evals_20260517.json
```

---

## 5. Legacy exploratory metrics

### ADM / SPRR

**Old role:** exploratory action-relevant discriminability and selective robustness summaries.

**Why downgraded:**

- Useful for debug but not necessary for the final core story.
- SMPR is simpler, task-grounded, and easier to explain.

**New role:**

```text
legacy exploratory / provenance only
```

### R_E / R_F median

**Old role:** ACPC-basin visualization and same-state perturbation contraction.

**Why downgraded:**

- `R_E` measures encoder-entry, not control-facing predictive stability.
- `R_F` median is useful but does not match the sampled-pool tail theorem.
- ATR tail replaces `R_F` median as the main paper-facing diagnostic.

**New role:**

```text
legacy visualization companion / historical basin table provenance
```

**Relevant artifact families:**

```text
assets/paper1_data/acpc_basin_diagnostics.json
assets/paper1_data/acpc_basin_diagnostics_pldm.json
```

---

## 6. What remains paper-facing

After compression, the paper-facing diagnostic section should focus on:

```text
ATR: normalized same-state ACPC-H/trans tail risk
SMPR: task-grounded near-boundary selective margin pass-rate
```

Minimal paper-facing language for legacy audits:

```text
Planner-facing cost/ranking/flip and reduced-budget optimizer-trace audits are retained as shadow checks in the release package. They support the same stability direction but are not independent core metrics.
```

Do not write:

```text
PCC, CRA, MAF, rank, ID probe, and CEM trace are all equally important diagnostics.
```

---

## 7. Safety / no-hallucination policy for Codex

1. Do not report ATR q90 unless it is computed from sample-level or per-row diagnostic values.
2. Do not infer ATR q90 from a mean/median table.
3. Do not delete artifacts from `assets/paper1_data`.
4. Do not remove manifest entries; update their role as legacy if needed.
5. Do not claim legacy audits prove CEM stability.
6. Do not claim compressed diagnostics outperform MAF-only or high-std references unless a new audit actually shows it.
7. Do not remove SMPR or selective-margin evidence from the paper.
8. Do not hide Cube or other boundary cases behind a combined scalar.

---

## 8. Recommended paper-facing sentence

Use this sentence when moving old tables out of the paper:

> The paper-facing selective-ACPC summary uses two theory-aligned readouts: ATR for same-state predictive-stability tails and SMPR for task-grounded anti-collapse margins. Planner-side PCC/CRA/MAF, reduced-budget CEM traces, and representation-proxy probes are retained in the release package as legacy shadow audits rather than independent core metrics.

---

## 9. Checklist for moving old content out of main paper

- [ ] Remove or shrink ACPC-H/PCC/CRA/MAF table from main text.
- [ ] Remove or shrink rank/transition/ID probe table from main text.
- [ ] Remove CEM trace table from main text unless one compact audit sentence is needed.
- [ ] Keep ATR + SMPR table in main text.
- [ ] Keep theory-to-metric mapping table in main text.
- [ ] Keep closed-loop behavior table in main text.
- [ ] Update `DATA_MANIFEST.md` to point to this legacy doc and compressed metric artifact.
- [ ] Run consistency checks.

---

## 10. Bottom line

The final compressed story should be:

```text
Behavior endpoint: closed-loop Gaussian success/drop.
Core diagnostic A: ATR, low same-state ACPC tail.
Core diagnostic B: SMPR, high selective margin.
Legacy audits: PCC/CRA/MAF/CEM/rank/ID, retained for provenance and reviewer defense.
```

This keeps the paper theoretically coherent without throwing away useful evidence.
