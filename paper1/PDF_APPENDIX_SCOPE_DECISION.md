# Paper1 PDF / Appendix Scope Decision for Legacy Diagnostics

> Decision date: 2026-07-06
>
> This note supersedes any earlier ambiguous wording that suggested putting `PCC/CRA/MAF`, CEM traces, or representation-proxy probes into the Paper1 appendix by default.

---

## 1. Final decision

**Do not include `PCC/CRA/MAF`, CEM trace tables, or rank/transition/ID-probe tables in the Paper1 PDF by default, including the appendix.**

They should be treated as team/internal or release-package legacy audits, not as paper-facing evidence.

The Paper1 PDF should focus on:

```text
Behavior endpoint: closed-loop Gaussian success/drop
Core diagnostic A: ATR, same-state ACPC-tail risk
Core diagnostic B: SMPR, selective-margin pass rate
```

A single sentence may point to archived shadow audits if needed, but the PDF should not carry the old metric tables unless a reviewer or venue explicitly asks for supplemental evidence.

---

## 2. Why this is the right cut

### 2.1 The paper becomes more coherent

Paper1's theory-facing claim is now:

```text
low same-state ACPC tail + high selective margin
```

`ATR` maps to the ACPC-tail term in the sampled-pool analysis. `SMPR` maps to the discriminability/no-collapse condition. That is enough for the paper's main diagnostic claim.

Adding `PCC/CRA/MAF`, CEM trace, rank, transition L2, and ID probe tables back into the appendix would re-create the old metric-zoo impression and weaken the compression.

### 2.2 PCC/CRA/MAF are useful but not necessary for the paper-facing claim

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

### 2.3 Putting old metrics in the appendix creates avoidable attack surface

The old audits contain mixed or non-dominating evidence:

- aggregate ACPC/PCC/CRA/MAF does not clearly beat MAF-only or high-std top-half references;
- CEM trace is reduced-budget and has task-specific mixed behavior;
- rank/transition/ID probes are proxy-level and weaker than SMPR;
- PCC/CRA/MAF depend on candidate-pool construction.

Including them in the PDF invites reviewers to evaluate them as if they were central claims. Keeping them out of Paper1 avoids that failure mode.

---

## 3. What to keep outside the PDF

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

For arXiv v1, these can remain in the public release package as provenance. For a double-blind venue, include only if the anonymous supplemental policy allows it and only if needed for reviewer defense.

---

## 4. Minimal PDF wording

If Paper1 needs to acknowledge old audits at all, use one sentence only:

> Additional planner-facing and representation-proxy audits are retained in the release package as legacy checks; they are not used as paper-facing core diagnostics.

Do not include tables for:

```text
PCC / CRA / MAF
CEM trace
rank / transition L2 / ID probe
ADM / SPRR
R_E / R_F median grids
```

unless the paper's claim is expanded again.

---

## 5. How this affects the theory section

The theory section should map only the two paper-facing metrics directly:

| Theory component | Paper-facing metric | Status |
|---|---|---|
| ACPC-tail event `Pr[D > epsilon]` | ATR | core |
| discriminability / no-collapse condition | SMPR | core |
| clean candidate-margin flip term | theorem-only or legacy MAF audit | not paper-facing |
| cost/ranking drift | theorem-only or legacy PCC/CRA audit | not paper-facing |
| adaptive CEM sensitivity | out of scope / legacy CEM trace | not paper-facing |

The sampled-pool theorem can stay because it motivates why tails matter and why clean margins matter. But the PDF should not need to show PCC/CRA/MAF tables unless it claims empirical planner-stability validation.

---

## 6. Codex action items

- [ ] Keep only ATR + SMPR as Paper1 diagnostic tables.
- [ ] Remove `PCC/CRA/MAF` tables from the Paper1 PDF, including appendix, unless explicitly requested later.
- [ ] Remove CEM trace tables from the Paper1 PDF, including appendix.
- [ ] Remove rank/transition/ID-probe tables from the Paper1 PDF, including appendix, except possibly one short failure-case sentence if needed.
- [ ] Keep `paper1/LEGACY_DIAGNOSTIC_AUDITS.md` as the destination for old metric definitions and roles.
- [ ] Keep artifact references in `DATA_MANIFEST.md`; do not delete JSON files.
- [ ] Ensure the abstract/contributions do not promise planner/cost/ranking validation as a main result.
- [ ] Ensure the conclusion says future methods may use planner/cost terms, but Paper1's compressed diagnostic is ATR + SMPR.

---

## 7. Bottom line

Yes: for the compressed Paper1, `PCC/CRA/MAF`, CEM trace, rank/transition/ID-probe tables can be removed from the paper and kept only as legacy/private/release-package evidence.

The clean story is:

```text
Closed-loop behavior establishes the phenomenon.
ATR measures same-state predictive-stability tail.
SMPR checks selective anti-collapse margin.
Everything else is non-paper-facing audit/provenance.
```
