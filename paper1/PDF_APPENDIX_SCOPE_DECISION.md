# Paper1 PDF / Appendix Scope Decision for Legacy Diagnostics

> Decision date: 2026-07-06
>
> This note supersedes any earlier ambiguous wording that suggested putting `PCC/CRA/MAF`, CEM traces, representation-proxy probes, release-package notes, manifest references, or legacy-audit pointers into the Paper1 PDF by default.

---

## 1. Final decision: zero legacy-audit mentions in the Paper1 PDF

**Do not include `PCC/CRA/MAF`, CEM trace tables, rank/transition/ID-probe tables, legacy-audit pointers, release-package provenance notes, or `DATA_MANIFEST.md` references in the Paper1 PDF by default, including the appendix.**

They should be treated as team/internal or repository-level release artifacts, not as paper-facing evidence.

The Paper1 PDF should focus on exactly this structure:

```text
Behavior endpoint: closed-loop Gaussian success/drop
Core diagnostic A: ATR, same-state ACPC-tail risk
Core diagnostic B: SMPR, selective-margin pass rate
```

No sentence like the following should appear in the PDF:

```text
Additional planner-facing and representation-proxy audits are retained in the release package as legacy checks.
```

That sentence is project-management / release-note content, not scientific paper content. If someone needs those audits, they can find them through repository documentation, not through the Paper1 narrative.

---

## 2. Why this cut is necessary

### 2.1 The paper becomes theoretically coherent

Paper1's theory-facing claim is:

```text
low same-state ACPC tail + high selective margin
```

`ATR` maps to the ACPC-tail term in the sampled-pool analysis. `SMPR` maps to the discriminability/no-collapse condition. That is enough for the paper's main diagnostic claim.

Adding `PCC/CRA/MAF`, CEM trace, rank, transition L2, ID probe, artifact paths, or release-manifest sentences back into the PDF recreates the metric-zoo / project-report impression and weakens the compression.

### 2.2 PCC/CRA/MAF are useful internally but unnecessary for Paper1

`PCC/CRA/MAF` explain the chain from ACPC to cost/ranking/action flips, but they are not required if the PDF claim is kept to:

```text
ATR and SMPR localize matched-Gaussian robustness plateaus as a no-retraining diagnostic.
```

They become necessary only if the paper claims:

```text
the diagnostic empirically validates planner/cost/ranking stability
```

or

```text
the diagnostic predicts CEM behavior / action selection
```

Paper1 should not make those stronger claims in the compressed version.

### 2.3 Putting old metrics in appendix creates avoidable attack surface

The old audits contain mixed or non-dominating evidence:

- aggregate ACPC/PCC/CRA/MAF does not clearly beat MAF-only or high-std top-half references;
- CEM trace is reduced-budget and has task-specific mixed behavior;
- rank/transition/ID probes are proxy-level and weaker than SMPR;
- PCC/CRA/MAF depend on candidate-pool construction;
- manifest / artifact sentences make the paper read like a release report instead of a research article.

Including them in the PDF invites reviewers to evaluate them as if they were central claims. Keeping them out of Paper1 avoids that failure mode.

---

## 3. Full-text deletion / rewrite audit for current `paper1/main.tex`

This section records what Codex should remove or rewrite when compressing the actual paper. It is intentionally strict.

### 3.1 Preamble / macros

**Delete or stop using paper-facing macros tied only to legacy diagnostics:**

```text
fragmetric
fragkey
fragility
transres
transreskey
idprobe
stdmaxkey, if used only for artifact/key mapping
codebrk/dataset, if only used for manifest paths or implementation keys
```

Keep only notation required by the paper-facing math and reproducible method description.

### 3.2 Abstract

Current abstract still mentions old or non-paper-facing evidence types:

```text
fixed-candidate cost drift
margin-failure
planner-trace audits
optimizer sensitivity
blur/resize diagnostics track ...
Diagnostic scores fixed on one development seed identify held-out candidate regions
```

Rewrite to ATR + SMPR only. Suggested abstract logic:

```text
ACPC is compressed to two readouts: ATR, a tail risk of same-state clean/noisy action-conditioned rollout disagreement, and SMPR, a task-grounded selective-margin pass rate. Across three LeWM training seeds, noise training yields matched-Gaussian robustness plateaus; ATR decreases and SMPR increases at recovered endpoints. The claim is a no-retraining diagnostic for matched Gaussian robustness, not planner-stability, universal transfer, or a new objective.
```

Do not mention planner traces, release package, legacy audits, PCC/CRA/MAF, or manifest artifacts in the abstract.

### 3.3 Code/data/reproducibility notes near title

The following is acceptable only as metadata, not as scientific narrative:

```text
Code and data release: <URL>
```

But remove any sentence that says:

```text
submission package includes aggregate artifacts and scripts
release package contains legacy checks
artifact links are withheld
```

For arXiv, a single code URL is fine. For double-blind, use the venue template outside the scientific argument; do not discuss release-package contents in the paper body.

### 3.4 Introduction contributions

Current contribution text still says things like:

```text
candidate costs
planner-side checks
larger grids / artifact paths remain in appendix and release manifest
```

Rewrite contributions to:

```text
C1: selective ACPC principle = low same-state ACPC tail + preserved task-grounded separations.
C2: theory-matched compression = ATR maps to ACPC-tail / Gaussian sensitivity; SMPR maps to discriminability / no-collapse.
C3: evidence = three-training-seed matched-Gaussian behavior plus ATR/SMPR diagnostic movement.
```

Do not mention release manifest, artifact paths, planner-side checks, or legacy metrics.

### 3.5 Operational diagnostics subsection

Current text talks about:

```text
R_E / R_F median
PCC / CRA / MAF
DATA_MANIFEST.md
three readout families
CEM trace audit
ADM/SPRR
full representation diagnostics
```

Replace the entire subsection with:

```text
The paper-facing diagnostic uses two readouts. ATR is the q90 or CVaR tail of normalized same-state clean/noisy ACPC-H/trans. SMPR is the pass rate of task-grounded label-crossing pairs whose clean different-state rollout distance exceeds the same-state noisy radius. ATR measures predictive-stability tail risk; SMPR rules out collapse. Encoder geometry and planner/cost readouts are not paper-facing diagnostics in this compressed version.
```

No `DATA_MANIFEST.md`, release artifacts, or old metric names should appear here.

### 3.6 Theory section

Keep the ACPC definition, discriminability condition, sampled-pool intuition, finite-sample tail calibration, and Gaussian sensitivity.

Rewrite theory-to-metric mapping so it maps only:

```text
Pr[D > epsilon] -> ATR
D_diff > D_same + m -> SMPR
```

The clean candidate-margin term may remain as a theorem term, but do not map it to MAF in the paper text. Do not add empirical planner-audit language.

Acceptable wording:

```text
The clean-margin term explains why ATR is not a closed-loop guarantee. We do not use a separate planner-facing empirical metric as a paper-facing diagnostic.
```

Delete / avoid:

```text
PCC estimates...
CRA checks...
MAF estimates...
shared-candidate diagnostics are reported as a family
```

### 3.7 Study protocol

Keep:

```text
models
tasks
noise intervention
training seeds
evaluation seeds
closed-loop endpoint
ATR/SMPR diagnostic construction
```

Delete:

```text
artifact paths
release manifest
Phase-0 artifact names
selector audit provenance
legacy diagnostic validation names
```

Protocol should describe what is scientifically evaluated, not where files live.

### 3.8 Experiments main text

Keep or rewrite:

1. closed-loop Gaussian cliff/recovery table;
2. compressed ATR + SMPR table;
3. concise plateau wording if based on ATR/SMPR;
4. SMPR construction details if not already in protocol.

Delete or move out of PDF entirely:

```text
Three-seed frozen diagnostic validation based on aggregate ACPC/PCC/CRA/MAF ranks
selector-plateau audit table
ACPC-basin R_E/R_F median table
Downstream shared-candidate readouts table
PCC/CRA/MAF compact table
margin-conditioned action-flip table
CEM trace table or CEM trace prose
representation discriminability table with effective rank / transition L2 / ID probe
unseen-stressor diagnostic-delta table with ACPC/PCC/CRA columns
```

If unseen-stressor content is retained, keep only behavior scores and scope boundary. Do not report diagnostic deltas with old metrics.

### 3.9 Discussion and conclusion

Rewrite discussion to avoid old metric language.

Delete / avoid phrases like:

```text
ACPC rollout/cost readouts
margin-conditioned flips
reduced-budget CEMSolver trace
optimizer sensitivity
planner-side evidence
release package
legacy checks
```

Suggested compressed discussion:

```text
Closed-loop evaluation establishes the behavior. ATR localizes whether same-state noisy views produce small action-conditioned predictive tails. SMPR checks whether task-grounded near-boundary distinctions survive that contraction. The results do not prove universal perturbation transfer, closed-loop CEM stability, or a training objective. Future methods can turn ATR/SMPR into a selective predictive-dynamics loss.
```

### 3.10 Appendix

Keep only paper-necessary appendix material:

```text
proofs / calibration for ATR and SMPR
minimal experimental details
noise implementation details if needed
semantic label construction for SMPR
negative target-view / heteroscedastic ablation only if directly used to motivate the future-method boundary
```

Delete from PDF appendix:

```text
full artifact maps
DATA_MANIFEST references
file paths / JSON names / hashes
main-figure rendering commands
atlas rendering commands
full LeWM sweep grids, if they duplicate main behavior table
full ACPC-basin grids
five-layer diagnostic framework
full diagnostic profile
PLDM full diagnostic tables, if not part of core ATR/SMPR claim
blur/resize diagnostic audit tables with old metrics
Phase-0 paired ACPC diagnostics
selector audit
CEM trace audit
legacy target/readout tables not needed for compressed claim
```

These can live in repository docs, not in the paper PDF.

### 3.11 Acknowledgements / release notes

Do not include release-package inventory language in the PDF. If arXiv needs code disclosure, use a single code URL near the title or in an unnumbered note. Avoid:

```text
JSON hashes are listed in the data manifest
public repository contains aggregate evaluation files, rendering scripts...
```

This belongs in `DATA_MANIFEST.md`, not the paper.

---

## 4. What to keep outside the PDF

Keep these materials outside the Paper1 PDF:

```text
paper1/LEGACY_DIAGNOSTIC_AUDITS.md
DATA_MANIFEST.md
assets/paper1_data/acpc_phase0_clean_goal_seed9101.json
assets/paper1_data/acpc_phase0_lewm_three_seed.json
assets/paper1_data/margin_flip_curve_lewm_three_seed.json
assets/paper1_data/cem_trace_audit_20260704.json
assets/paper1_data/selector_plateau_audit_20260704.json
canonical diagnostics / PLDM diagnostics / representation-probe artifacts
```

For arXiv v1, these can remain in the public repository as provenance. For a double-blind venue, include only if the anonymous supplemental policy allows it and only if needed for reviewer defense.

---

## 5. How this affects the theory section

The theory section should map only the two paper-facing metrics directly:

| Theory component | Paper-facing metric | Status |
|---|---|---|
| ACPC-tail event `Pr[D > epsilon]` | ATR | core |
| discriminability / no-collapse condition `D_diff > D_same + m` | SMPR | core |
| clean candidate-margin flip term | theorem-only caveat | not paper-facing metric |
| cost/ranking drift | theorem-only background or removed | not paper-facing metric |
| adaptive CEM sensitivity | out of scope | not paper-facing metric |

The sampled-pool theorem can stay only if it is used to motivate why tail risk matters and why ATR is not a closed-loop guarantee. It should not force empirical PCC/CRA/MAF tables into the paper.

---

## 6. Codex action items

- [ ] Keep only ATR + SMPR as Paper1 diagnostic tables.
- [ ] Remove `PCC/CRA/MAF` tables from the Paper1 PDF, including appendix.
- [ ] Remove CEM trace tables and CEM trace prose from the Paper1 PDF, including appendix.
- [ ] Remove rank/transition/ID-probe tables from the Paper1 PDF, including appendix.
- [ ] Remove release-package / legacy-audit / `DATA_MANIFEST.md` references from the Paper1 PDF.
- [ ] Remove file paths, JSON artifact names, hashes, and rendering commands from the PDF appendix unless required for a method definition.
- [ ] Keep a single code URL only if desired for arXiv metadata.
- [ ] Keep `paper1/LEGACY_DIAGNOSTIC_AUDITS.md` as the destination for old metric definitions and roles.
- [ ] Keep artifact references in `DATA_MANIFEST.md`; do not delete JSON files.
- [ ] Ensure the abstract/contributions do not promise planner/cost/ranking validation as a main result.
- [ ] Ensure the conclusion says future methods may use planner/cost terms, but Paper1's compressed diagnostic is ATR + SMPR.

---

## 7. Bottom line

For the compressed Paper1, `PCC/CRA/MAF`, CEM trace, rank/transition/ID-probe tables, artifact paths, release notes, and legacy-audit references should be removed from the paper and kept only as repository-level or internal provenance.

The clean story is:

```text
Closed-loop behavior establishes the phenomenon.
ATR measures same-state predictive-stability tail.
SMPR checks selective anti-collapse margin.
Everything else is non-paper-facing audit/provenance.
```
