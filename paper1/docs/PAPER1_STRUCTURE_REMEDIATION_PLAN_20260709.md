# Paper1 structure remediation plan for Codex

Target branch: `ag/dev`  
Target file family: `paper1/main.tex`, `paper1/tables/*`, `paper1/scripts/*`, `paper1/docs/*`, `assets/paper1_figs/*`  
Purpose: make Paper1 read like a formal top-conference diagnostic paper rather than an internal experiment report.

This plan is a writing-and-evidence restructuring pass. It should not invent results, edit numerical artifacts by hand, or silently change claims. Every promoted figure/table must trace to an existing artifact or a reproducible script.

---

## 0. Core diagnosis

Current Paper1 has enough scientific material for a serious diagnostic paper, but the article structure hides the strongest evidence.

Main issues:

1. **Section and subsection names still sound internal.** Several headings read like audit logs or remediation notes instead of paper results. This makes the paper feel assembled from internal reports even when the evidence is strong.
2. **Theory-supporting evidence is too fragmented.** The radius--margin theorem, Gaussian sensitivity proposition, SMPR guard, fixed-pool audit, held-out validation, and JVP/Hutchinson evidence are scattered across main text and appendix. A reader can miss that the theory is empirically supported.
3. **Some important appendix tables should be figures.** Tables are useful for reproducibility, but several current tables encode trends or before/after contrasts that would be clearer as figures.
4. **The main text lacks an explicit theory-to-evidence narrative.** The theory defines radius, margin, planner instability, discriminability failure, and local Gaussian sensitivity. The experiments should be organized around exactly these objects.
5. **The appendix mixes proof, protocol, audit, and raw data tables.** This is useful internally but harder for reviewers to navigate.

Target reader reaction after remediation:

> I can see the paper's logic in one pass: Gaussian noise breaks no-noise JEPA checkpoints; Gaussian training creates recovery bands; ACPC explains the recovery through a radius term, a margin/guard term, and local composed sensitivity; the evidence is full-sweep, held-out, fixed-pool, and bounded-scope rather than endpoint-only.

---

## 1. Target paper story

The paper should read as a diagnostic study with a single evidence chain:

```text
Problem:
  Encoder invariance is not enough for control robustness.

Diagnostic target:
  Same-state clean/noisy views should have small action-conditioned predictive radius,
  while task/action-relevant different states remain separated.

Theory:
  If the same-state predictive radius is small relative to candidate margins,
  fixed-pool planning decisions are stable; if discriminability margins collapse,
  low radius is meaningless. Local Gaussian linearization links ATR to the
  composed encoder--rollout Jacobian.

Evidence:
  1. Closed-loop Gaussian noise reveals fragility and recovery bands.
  2. ATR/SMPR track recovered-vs-fragile rows across the full sweep.
  3. Held-out seed/task validation checks diagnostic-region transfer.
  4. Fixed-pool event-rate audit tests the planner margin link.
  5. Finite-difference and JVP/Hutchinson audits test the local sensitivity link.
  6. Non-Gaussian stressors delimit scope.

Boundary:
  This is a fixed-checkpoint matched-Gaussian diagnostic, not a robust-control
  method, not a universal JEPA theorem, and not a calibrated closed-loop certificate.
```

This story should be visible in the section order, figure order, and captions.

---

## 2. Proposed main-text structure

Replace the current structure with the following conference-style outline.

### 2.1 Title / abstract / contribution box

Keep the current title only if the abstract makes the scope explicit. Safer title options:

1. `Action-Conditioned Predictive Consistency as a Diagnostic for Gaussian Robustness in JEPA World Models`
2. `A Diagnostic Study of Gaussian Visual Robustness in JEPA Latent World Models`
3. `Diagnosing Gaussian Visual Robustness in JEPA World Models through Action-Conditioned Predictive Consistency`

The abstract should follow this structure:

1. Problem: latent prediction does not define closed-loop visual robustness.
2. Diagnostic: ACPC = same-state predictive stability + task/action discriminability guard.
3. Study: fixed LeWM checkpoints, four tasks, three training seeds, matched Gaussian stressor.
4. Key result: base fragility and broad matched-Gaussian recovery bands.
5. Mechanism evidence: low ATR/high SMPR, held-out validation, fixed-pool audit, local sensitivity audit.
6. Boundary: fixed-checkpoint matched-Gaussian diagnostic, not a new robust-control method.

Do not make the abstract a list of every audit. It should sell the evidence chain, not the artifact inventory.

### 2.2 Section 1: Introduction

Suggested length: 1.25--1.5 pages.

Recommended flow:

1. Latent prediction helps avoid pixel reconstruction, but control uses predicted futures.
2. Same-state noisy views need action-conditioned agreement, not necessarily raw encoder closeness.
3. Agreement alone can collapse task distinctions; therefore a guard is required.
4. This paper studies this as a matched-Gaussian diagnostic problem for fixed JEPA world-model checkpoints.
5. Contributions, exactly three:
   - ACPC diagnostic framing with ATR and SMPR.
   - Controlled three-training-seed Gaussian robustness study.
   - Theory-aligned empirical audits: held-out diagnostic validation, fixed-pool event rates, and local Gaussian sensitivity.

Avoid listing too many caveats in the Introduction. Put scope boundaries in one short sentence and defer details to Discussion.

### 2.3 Section 2: Related Work

Suggested length: 0.75--1 page.

Use three compact subsections:

1. `Latent predictive world models`  
   JEPA, V-JEPA, LeWM, PLDM, LeJEPA.
2. `Visual robustness in control`  
   DrQ/DrQ-v2, SODA, DreamerV3, TD-MPC2, robust visual MPC, ReOI.
3. `Control-relevant representation diagnostics`  
   bisimulation, value-aware models, group-action/action-conditioned world-model metrics.

Do not over-explain every baseline. The goal is to position ACPC as a diagnostic lens, not to claim method superiority.

### 2.4 Section 3: ACPC diagnostic theory

Rename current `Action-Conditioned Predictive Consistency` section to:

```latex
\section{Action-Conditioned Predictive Consistency as a Diagnostic}
```

Recommended subsections:

1. `Same-state predictive radius and selective margin`  
   Combine the current setup, ACPC-H, and discriminability guard. Keep notation minimal.
2. `Fixed-pool radius--margin link to planning`  
   Present the cost-drift proposition, top-1 stability proposition, and theorem. Keep proof in appendix.
3. `Local Gaussian sensitivity`  
   Present only the proposition/corollary intuition and the composed sensitivity object. Put detailed proof in appendix.
4. `Operational metrics`  
   Define ATR and SMPR after the theory objects, not before. This makes the metrics look derived from theory rather than invented.

Important wording:

- Use `diagnostic bound`, `fixed-pool sufficient condition`, or `radius--margin audit`.
- Avoid implying a calibrated closed-loop certificate.
- Say q90 ATR is a fixed reporting summary and not the theorem's calibrated tail probability.

### 2.5 Section 4: Experimental setup

Rename current `Study protocol` to:

```latex
\section{Experimental Setup}
```

Keep this short. It should define:

- tasks: TwoRoom, PushT, Reacher, Cube;
- model family: LeWM fixed checkpoints;
- training sweep: `stdmax in {0.00,...,0.08}`;
- three training seeds: 3072/3073/3074;
- evaluation seeds: 42/43/44, 100 trajectories each;
- primary stressor: observation-only Gaussian `sigma=0.08`, clean goal;
- diagnostics: ATR, SMPR, fixed-pool candidate audit, sensitivity audits;
- rule: diagnostics are computed without retraining.

Move raw artifact and manifest details to appendix.

### 2.6 Section 5: Results

Use this section as the main narrative. Suggested subsections:

#### 5.1 Gaussian observation noise breaks no-noise checkpoints

Current subsection:

```latex
\subsection{JEPA control fragility under Gaussian observation noise}
```

Suggested rename:

```latex
\subsection{Gaussian observation noise breaks no-noise checkpoints}
```

Keep Table `noise-cliff` or convert to a compact figure. This result establishes the behavioral phenomenon.

Recommended figure/table choice:

- If space is tight, use a small bar chart of clean vs `sigma=0.08` for four tasks and move the full multi-sigma table to appendix.
- If keeping the table, highlight endpoint drops in the paragraph.

#### 5.2 Full-sequence Gaussian augmentation yields task-dependent recovery bands

Current subsection:

```latex
\subsection{Noise augmentation reveals matched-Gaussian recovery bands}
```

Suggested rename:

```latex
\subsection{Full-sequence Gaussian augmentation yields task-dependent recovery bands}
```

Keep `fig2_sweep` as main Figure 1 or Figure 2. It is central.

Caption should explicitly say:

- three training seeds;
- evaluation seeds averaged within seed;
- error bars across training seeds;
- recovery is broad and task-dependent;
- this is not a checkpoint ranking claim.

#### 5.3 Selective ACPC diagnostics track recovered checkpoints

This should be a major main-text subsection, not an endpoint-only paragraph.

Merge the following current material here:

- endpoint ATR/SMPR table;
- qualitative ACPC neighborhood t-SNE;
- full-sweep diagnostic separation;
- held-out seed/task validation;
- threshold sensitivity only as a sentence pointing to appendix.

Suggested structure:

1. Start with one sentence: `The theory predicts that recovered rows should have smaller same-state predictive radius and preserved task-grounded separations.`
2. Show endpoint ATR/SMPR as a figure or compact table.
3. Show full-sweep diagnostic curves as a main figure.
4. Report held-out validation in text or a tiny inset/table.
5. Explain SMPR proxy labels and limitations once.

Recommended main figures:

- `fig_acpc_endpoint_atr_smpr.png`: two-panel paired plot, ATR base→std0.08 and SMPR base→std0.08.
- `fig_full_sweep_diagnostics.png`: keep or redesign to show score, normalized ATR, SMPR failure, top-1 disagreement.

The current endpoint ATR/SMPR numeric table can move to appendix if replaced by a figure.

#### 5.4 Planner-side radius--margin audit supports the fixed-pool link

Current subsection:

```latex
\subsection{Fixed-pool radius--margin calibration}
```

Suggested rename:

```latex
\subsection{Planner-side radius--margin audit}
```

This is where the fixed-pool theorem gets its empirical support. It should not be buried after held-out validation without a clear bridge.

Main-text content:

1. Restate the theory object: top-1 stability requires cost drift below half the clean candidate margin.
2. Report summary-level proxy gap as mechanism orientation, not certificate.
3. Promote the sample-level event-rate audit:
   - cert-pass fragile→recovered;
   - top-1 flip fragile→recovered;
   - flip conditioned on cert-pass.
4. State the negative calibration result: strict q10/q95 gaps remain negative.

Recommended figure:

- `fig_fixed_pool_event_rates.png`: forest/dumbbell plot with Wilson intervals for cert-pass and top-1 flip, fragile vs recovered, by task plus ALL.
- Optional small inset: flip | cert-pass is zero, with CI.

Move detailed event-rate table to appendix.

#### 5.5 Local Gaussian sensitivity explains ATR contraction

Current subsection:

```latex
\subsection{Local Gaussian sensitivity analysis}
```

Suggested rename:

```latex
\subsection{Local Gaussian sensitivity explains ATR contraction}
```

This is currently one of the strongest theory-supporting pieces but it reads like an audit note. Make it a mechanism result.

Main-text content:

1. Link to theory: local Gaussian ACPC radius scales with composed encoder--rollout sensitivity.
2. Show finite-difference endpoint/base ratios.
3. Show JVP/Hutchinson composed-trace endpoint/base ratios.
4. State decomposition: composed trace falls sharply; encoder-side reduction dominates; rollout-side trace is task-dependent.
5. Keep caveat: local mechanism evidence, not global robustness proof.

Recommended figure:

- `fig_gaussian_sensitivity_mechanism.png`: log-scale bar or dot plot with two rows/panels:
  - finite-difference slope endpoint/base;
  - JVP/Hutchinson composed trace endpoint/base.
- Optional heatmap panel for encoder / rollout / composed / alignment ratios.

Move raw numeric tables to appendix.

#### 5.6 Boundary check outside matched Gaussian stressor

Current subsection:

```latex
\subsection{Evaluation under bounded non-Gaussian stressors}
```

Suggested rename:

```latex
\subsection{Boundary check outside the matched Gaussian stressor}
```

This should be short. It exists to prevent overclaiming broad corruption transfer.

Main-text content:

- TwoRoom/Reacher blur improve.
- PushT/Cube resize do not clearly improve.
- Stressor-specific ATR/SMPR co-move with behavior.
- This supports bounded association, not general transfer.

A small table is acceptable here because there are only four rows and mixed outcomes are important.

### 2.7 Section 6: Discussion and limitations

Suggested subsections or paragraphs:

1. `What the evidence supports`
2. `Where the diagnostic stops`
3. `What would turn this into a method paper`

Avoid repeating every caveat from earlier sections. Use a compact limitations box if the target venue allows it.

Key limitations to state once:

- matched Gaussian stressor;
- fixed checkpoints;
- no robust-control method claim;
- fixed-pool planning audit, not adaptive CEM/replanning guarantee;
- local Gaussian sensitivity, not global certificate;
- SMPR uses proxy labels;
- stronger contact/topology/action-value labels remain future work;
- cross-model evidence is limited unless PLDM/DINO-WM is promoted.

### 2.8 Conclusion

Short. Do not re-list all experiments. End with the diagnostic implication:

> Robust visual world-model control should be evaluated through stable action-conditioned predictive dynamics under nuisance variation, while preserving distinctions needed for planning.

---

## 3. Appendix restructuring

The appendix should be navigable. Reorganize into the following order.

### Appendix A: Proofs

Include:

- cost-drift proposition proof;
- top-1 stability proof;
- radius--margin theorem proof;
- local Gaussian sensitivity derivation.

Do not mix proofs with experimental details.

### Appendix B: Experimental protocol and artifacts

Include:

- tasks;
- seeds;
- checkpoint grid;
- evaluation corruption definitions;
- diagnostic sampling protocol;
- artifact provenance and checker command;
- release manifest pointer.

### Appendix C: Additional Gaussian sweep tables

Include:

- full Gaussian sweep table;
- full multi-sigma base cliff table if removed from main;
- per-training-seed values if needed.

### Appendix D: Diagnostic validation details

Include:

- held-out gate parameters;
- threshold sensitivity;
- full-sweep diagnostic-region rows;
- exact endpoint ATR/SMPR table if figure replaces it.

### Appendix E: Fixed-pool planner audit details

Include:

- sample-level event-rate table;
- endpoint fixed-pool certificate audit;
- strict q10/q95 negative calibration;
- top-1 agreement table or line plots.

### Appendix F: Local sensitivity details

Include:

- finite-difference raw table;
- JVP/Hutchinson decomposition table;
- estimator details;
- why finite difference and JVP/Hutchinson need not match numerically.

### Appendix G: Boundary stressors

Include:

- blur/resize details;
- stressor-specific ATR/SMPR details;
- any additional unseen-stressor rows.

### Appendix H: Reproducibility and source release

Include:

- build command;
- consistency checker;
- arXiv/blind source bundle notes;
- known artifact audit notes.

---

## 4. Heading rewrite map

Use this map when editing `main.tex`.

| Current wording | Suggested wording | Reason |
|---|---|---|
| `Study protocol` | `Experimental Setup` | Standard paper heading. |
| `JEPA control fragility under Gaussian observation noise` | `Gaussian observation noise breaks no-noise checkpoints` | Direct result statement. |
| `Noise augmentation reveals matched-Gaussian recovery bands` | `Full-sequence Gaussian augmentation yields task-dependent recovery bands` | More precise and formal. |
| `ATR/SMPR selective-ACPC diagnostics` | `Selective ACPC diagnostics track recovered checkpoints` | Reads as a result, not a metric inventory. |
| `Full-sweep diagnostic separation and held-out validation` | Fold into `Selective ACPC diagnostics track recovered checkpoints` | This is central evidence, not a detached audit. |
| `Fixed-pool radius--margin calibration` | `Planner-side radius--margin audit` | Avoids implying calibrated certificate. |
| `Local Gaussian sensitivity analysis` | `Local Gaussian sensitivity explains ATR contraction` | Makes theory--evidence link explicit. |
| `Evaluation under bounded non-Gaussian stressors` | `Boundary check outside the matched Gaussian stressor` | Makes scope role clear. |
| `Supplementary diagnostic analyses` | Split into targeted appendices | Current heading is too broad/internal. |
| `Finite-sample empirical fixed-pool risk audit` | `Fixed-pool event-rate audit` | Shorter, paper-facing. |

Forbidden or risky wording in section titles:

- remediation;
- retained-summary;
- audit anchors;
- paper-facing;
- internal;
- selector;
- release package;
- certificate calibration, unless immediately qualified;
- robustness oracle.

---

## 5. What to promote from appendix to main text

Promote the following evidence into the main text narrative, either as figures or compact paragraphs.

| Evidence | Current role | New role |
|---|---|---|
| Held-out diagnostic-region validation | Main text but visually minor | Make it part of the main ACPC diagnostic evidence chain. |
| Full-sweep diagnostic separation | Main text but reads like supplemental audit | Keep as one of the central results. |
| Sample-level fixed-pool event rates | Appendix/table-heavy | Promote as planner-side radius--margin evidence. |
| Wilson intervals for cert-pass/top-1 flip | Appendix table | Convert to figure or compact main-text result. |
| Finite-difference Gaussian sensitivity | Main text paragraph + appendix table | Promote as mechanism evidence with figure. |
| JVP/Hutchinson composed trace | Main text paragraph + appendix table | Promote as mechanism evidence with figure. |
| Theory-to-evidence map | Appendix table | Move a compact version to main text or convert into a schematic figure. |
| Threshold sensitivity | Appendix | Keep appendix; mention one sentence in main. |
| Strict q10/q95 negative calibration | Appendix/main paragraph | Keep in planner audit; this protects against overclaiming. |

---

## 6. Table-to-figure conversion recommendations

Do not convert every table. Convert only tables where the visual trend is the point.

### 6.1 Convert: endpoint ATR/SMPR table

Current table: `tab:atr-smpr-selective-acpc`.

Problem: before/after contraction is easier to see visually than in four numeric rows.

Suggested figure:

`assets/paper1_figs/fig_endpoint_atr_smpr.png`

Design:

- two panels;
- panel A: ATR base vs std0.08 by task, log y-scale if needed;
- panel B: SMPR base vs std0.08 by task;
- use paired lines/dumbbells per task;
- include error bars across training seeds;
- lower-is-better / higher-is-better labels.

Keep exact numeric table in appendix.

### 6.2 Keep or redesign: full-sweep diagnostics figure

Current `fig_full_sweep_diagnostics.png` is important. If it is crowded, split it:

1. `fig_full_sweep_behavior_acpc.png`: score + normalized ATR + SMPR failure.
2. `fig_full_sweep_planner_guard.png`: top-1 flip + proxy gap.

Avoid putting too many curves on one axis.

### 6.3 Convert: held-out validation table, optional

Current table has only two rows. It can stay as a table.

Optional figure:

- small point plot with mean absolute onset error and max error;
- annotate precision/recall in caption.

Recommendation: keep table unless the main results page needs visual balance.

### 6.4 Convert: sample-level event-rate CI table

Current table is important but dense.

Suggested figure:

`assets/paper1_figs/fig_fixed_pool_event_rates.png`

Design:

- forest plot or paired dot plot with Wilson intervals;
- rows: TwoRoom, PushT, Reacher, Cube, ALL;
- columns/panels:
  - cert-pass fragile vs recovered;
  - top-1 flip fragile vs recovered;
  - optional flip | cert-pass inset.

Main caption should say:

- fixed 65-candidate pool;
- intervals quantify event-rate estimation uncertainty;
- not calibrated theorem probabilities.

Keep exact CI table in appendix.

### 6.5 Convert: Gaussian sensitivity tables

Current finite-difference and JVP/Hutchinson tables are mechanism evidence. They should be visual.

Suggested figure:

`assets/paper1_figs/fig_gaussian_sensitivity_mechanism.png`

Design:

- panel A: finite-difference slope endpoint/base ratio by task;
- panel B: JVP/Hutchinson composed trace endpoint/base ratio by task;
- y-axis log scale;
- horizontal line at 1.0;
- optional text labels for ratios.

Optional second figure or appendix heatmap:

`assets/paper1_figs/fig_jvp_trace_decomposition_heatmap.png`

Rows: task. Columns: encoder, rollout, composed, alignment. Values: endpoint/base ratios.

Keep raw tables in appendix.

### 6.6 Keep: non-Gaussian stressor table

The table has four rows and mixed outcomes. A table is fine.

Optional paired score plot is possible, but not necessary. The mixed result is clearer with exact values.

### 6.7 Convert or simplify: theory-to-evidence map

Current table is useful but may feel appendix-like.

Two options:

1. Main-text schematic figure: `Theory object -> diagnostic -> evidence -> limitation`.
2. Compact main table with only five rows, exact details in appendix.

Recommendation: create a small schematic figure for the main text and keep the detailed map as appendix table.

### 6.8 Convert: threshold sensitivity table, optional appendix heatmap

Current threshold table is repetitive. Convert to appendix heatmap only if easy.

Design:

- x-axis: clean tolerance;
- y-axis: recovery fraction;
- cell: number of tasks passing ATR direction / SMPR direction.

Recommendation: optional. This is not central enough to block the restructure.

---

## 7. Main-text figure order

Recommended final main figure/table sequence:

1. **Figure 1: ACPC concept schematic**  
   Clean/noisy same-state branches under shared action sequence; radius term + margin/guard term.
2. **Figure 2: Gaussian sweep recovery bands**  
   Current `fig2_sweep`.
3. **Figure 3: Endpoint ATR/SMPR selective diagnostics**  
   New paired plot or compact table.
4. **Figure 4: Full-sweep diagnostic separation and held-out validation**  
   Current or redesigned `fig_full_sweep_diagnostics`, plus held-out result in text/inset.
5. **Figure 5: Planner-side fixed-pool event-rate audit**  
   New Wilson interval event-rate plot.
6. **Figure 6: Local Gaussian sensitivity mechanism**  
   New finite-difference + JVP/Hutchinson ratio figure.
7. **Table 1: Noise cliff or non-Gaussian boundary check**  
   Depending on space, keep one or two compact tables.

If page budget is tight, combine Figures 5 and 6 into a two-column mechanism figure, but do not bury both in appendix.

---

## 8. Claim-boundary rules during restructure

Codex must preserve these boundaries.

### Allowed claims

- No-noise JEPA world-model checkpoints can be fragile under observation-only Gaussian noise in this protocol.
- Full-sequence Gaussian input augmentation yields broad task-dependent matched-Gaussian recovery bands.
- Recovered rows occupy low-ATR/high-SMPR regions across the full sweep.
- Held-out seed/task validation supports recovered-vs-fragile diagnostic separation under matched Gaussian stressor.
- Fixed-pool event-rate audits support the radius--margin mechanism for shared candidate pools.
- Finite-difference and JVP/Hutchinson audits support reduced local composed encoder--rollout sensitivity at endpoints.
- Non-Gaussian stressor rows delimit scope and show mixed transfer.

### Disallowed claims

- ACPC is a closed-loop robustness guarantee.
- ATR alone predicts robustness.
- SMPR alone proves semantic discriminability.
- q90 ATR gives the theorem's calibrated tail probability.
- The radius--margin theorem covers adaptive CEM resampling or repeated replanning.
- Gaussian training provides general corruption robustness.
- The paper beats DrQ, DreamerV3, TD-MPC2, ReOI, or robust MPC methods.
- The method is a new training algorithm.

### Preferred language

Use:

- `supports`;
- `is consistent with`;
- `diagnostic evidence`;
- `fixed-pool audit`;
- `matched-Gaussian setting`;
- `local mechanism evidence`;
- `proxy-level discriminability guard`.

Avoid:

- `proves robustness`;
- `guarantees`;
- `oracle`;
- `universal`;
- `method-invariant`;
- `calibrated certificate`, unless negated.

---

## 9. Execution plan for Codex

### Phase 0: preflight

Run from repo root:

```bash
git checkout ag/dev
git pull --ff-only
python -m tools.check_paper1_consistency | tee /tmp/paper1_check_before.log
cd paper1 && bash build.sh --clean | tee /tmp/paper1_build_before.log
cd ..
```

Also list current paper structure:

```bash
rg -n "^\\section|^\\subsection|^\\subsubsection|^\\paragraph" paper1/main.tex > /tmp/paper1_headings_before.txt
```

### Phase 1: outline rewrite without changing evidence

Edit `paper1/main.tex` only.

Tasks:

1. Rename sections/subsections using the heading rewrite map.
2. Move full-sweep diagnostic and held-out validation closer to ATR/SMPR endpoint results.
3. Move fixed-pool event-rate audit into the planner-side radius--margin result subsection.
4. Move local Gaussian sensitivity paragraph into a mechanism result subsection.
5. Consolidate limitations.
6. Do not add new numerical claims.

Acceptance:

```bash
rg -n "remediation|retained-summary|audit anchors|paper-facing|selector|robustness oracle|calibrated certificate" paper1/main.tex
```

Any hits must be intentionally justified or removed.

### Phase 2: create or redesign figures

Create scripts under `paper1/scripts/` if existing scripts do not already generate the needed plots.

Recommended new scripts:

```text
paper1/scripts/plot_endpoint_atr_smpr.py
paper1/scripts/plot_fixed_pool_event_rates.py
paper1/scripts/plot_gaussian_sensitivity_mechanism.py
```

Recommended outputs:

```text
assets/paper1_figs/fig_endpoint_atr_smpr.png
assets/paper1_figs/fig_fixed_pool_event_rates.png
assets/paper1_figs/fig_gaussian_sensitivity_mechanism.png
```

Inputs should be existing CSV/JSON/table artifacts, not hand-entered numbers. If a script must parse `.tex` tables temporarily, document that as a transitional implementation and prefer CSV/JSON sources.

Acceptance:

- every figure has a reproducible script;
- every script has input paths at top or CLI args;
- no manual edits to JSON result rows;
- figures are referenced from `main.tex` only after they exist.

### Phase 3: table relocation

For each converted table:

1. Keep exact numeric table in appendix.
2. Replace main-text table with figure and a short result paragraph.
3. Ensure labels are unique and captions match the claim boundary.

Tables likely to move to appendix:

- endpoint ATR/SMPR numeric table;
- sample-level event-rate CI table if figure promoted;
- finite-difference sensitivity table if figure promoted;
- JVP/Hutchinson sensitivity table if figure promoted.

Tables likely to remain in main:

- noise cliff table, unless converted to figure;
- non-Gaussian boundary table, because mixed exact values matter.

### Phase 4: appendix reorganization

Reorder appendix according to Section 3 of this plan.

Acceptance:

```bash
rg -n "^\\section|^\\subsection" paper1/main.tex
```

The appendix order should be:

1. Proofs;
2. Experimental protocol;
3. Additional Gaussian tables;
4. Diagnostic validation details;
5. Fixed-pool planner audit details;
6. Local sensitivity details;
7. Boundary stressors;
8. Reproducibility/source release.

### Phase 5: release and source bundle checks

Run:

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
rg -n "Citation .* undefined|Reference .* undefined|There were undefined references|Undefined control sequence|Fatal error|No file main.bbl" main.log
```

If source bundle scripts or README are touched, ensure all figures referenced in `main.tex` are included in arXiv/blind source packaging.

Known risk to check:

- If `main.tex` references `fig_full_sweep_diagnostics.png`, `fig_radius_margin_overlap.png`, or new figures, the source bundle copy list must include them.

---

## 10. Suggested concrete edits in `main.tex`

### 10.1 ACPC theory section reordering

Current flow likely starts with definitions, then radius--margin, then operational diagnostics. Keep that, but make the section titles more reader-facing.

Suggested skeleton:

```latex
\section{Action-Conditioned Predictive Consistency as a Diagnostic}
\label{sec:acpc}

\subsection{Same-state predictive radius and selective margin}
...

\subsection{Fixed-pool radius--margin link to planning}
...

\subsection{Local Gaussian sensitivity}
...

\subsection{Operational diagnostics: ATR and SMPR}
...
```

### 10.2 Results section skeleton

Suggested skeleton:

```latex
\section{Experiments}
\label{sec:exp}

\subsection{Gaussian observation noise breaks no-noise checkpoints}
...

\subsection{Full-sequence Gaussian augmentation yields task-dependent recovery bands}
...

\subsection{Selective ACPC diagnostics track recovered checkpoints}
...

\subsection{Planner-side radius--margin audit}
...

\subsection{Local Gaussian sensitivity explains ATR contraction}
...

\subsection{Boundary check outside the matched Gaussian stressor}
...
```

### 10.3 Bridge paragraphs to add

Before ATR/SMPR result:

```text
The theory separates robustness into a radius term and a guard term. The next experiments follow this split: ATR measures high-tail same-state predictive radius, while SMPR and fixed-pool top-1 flip test whether task and planner distinctions survive that contraction.
```

Before fixed-pool audit:

```text
The radius--margin theorem is a fixed-candidate statement. We therefore test the corresponding finite-sample event directly on a shared 65-candidate pool rather than treating q90 ATR as a calibrated theorem probability.
```

Before Gaussian sensitivity figure:

```text
The local Gaussian proposition predicts that ATR contraction should coincide with lower sensitivity of the composed encoder--rollout map along input perturbation directions. We test this with two complementary local estimators.
```

Before boundary stressor:

```text
The main diagnostic region is matched to Gaussian training and Gaussian observation noise. We therefore use blur/resize only as a boundary check, not as a transfer claim.
```

---

## 11. Suggested figure captions

### `fig_endpoint_atr_smpr.png`

```latex
\caption{Endpoint selective-ACPC diagnostics. ATR is the q90 normalized same-state clean/noisy rollout disagreement; lower is better. SMPR is the task-grounded near-boundary margin pass rate; higher is better. Points report mean across LeWM training seeds 3072/3073/3074 and error bars report population standard deviation across training seeds. Noise-trained endpoints use full-sequence Gaussian augmentation with $\sigma_{\max}=0.08$.}
```

### `fig_fixed_pool_event_rates.png`

```latex
\caption{Fixed-pool radius--margin event-rate audit. Cert-pass denotes sampled histories where the maximum paired cost drift over the shared 65-candidate pool is below one half of the clean top-1/top-2 margin. Top-1 flip is the observed clean/noisy fixed-pool top-candidate disagreement. Intervals are Wilson 95\% intervals over sampled anchors and quantify event-rate uncertainty, not calibrated theorem probabilities.}
```

### `fig_gaussian_sensitivity_mechanism.png`

```latex
\caption{Local Gaussian sensitivity mechanism. Finite-difference slopes estimate small-noise rollout-radius response along sampled Gaussian perturbations; exact-JVP/Hutchinson traces estimate local Frobenius averages of the composed encoder--rollout map. Values are endpoint/base ratios within each task. Ratios below one support reduced local composed sensitivity at the noise-trained endpoint; these are local mechanism diagnostics rather than global robustness guarantees.}
```

---

## 12. Final acceptance checklist

### Structure

- [ ] Main section titles sound like formal paper results, not internal audit logs.
- [ ] Theory section introduces radius, margin, fixed-pool planning link, and Gaussian sensitivity before metrics.
- [ ] Results section follows the theory objects.
- [ ] Held-out validation is visible in main text.
- [ ] Fixed-pool event-rate audit is visible in main text.
- [ ] Gaussian sensitivity evidence is visible in main text.
- [ ] Appendix is organized by purpose, not by historical artifact order.

### Figures and tables

- [ ] Endpoint ATR/SMPR before/after trend is shown clearly.
- [ ] Fixed-pool event rates are shown as a figure or highly readable table.
- [ ] Gaussian sensitivity mechanism is shown as a figure or highly readable table.
- [ ] Dense exact numeric tables remain in appendix.
- [ ] No figure is created from hand-edited numbers.
- [ ] Every referenced figure is included in source bundle scripts.

### Claims

- [ ] q90 ATR is not described as calibrated theorem probability.
- [ ] Fixed-pool audit is not described as adaptive CEM/replanning guarantee.
- [ ] SMPR is described as proxy-level task-grounded guard.
- [ ] Non-Gaussian results are described as bounded scope checks.
- [ ] The paper does not claim method superiority over robust visual-control baselines.

### Build and release

- [ ] `python -m tools.check_paper1_consistency` passes.
- [ ] `cd paper1 && bash build.sh --clean` passes.
- [ ] No undefined references/citations in `main.log`.
- [ ] arXiv/blind source packaging includes all figures referenced by `main.tex`.
- [ ] Author/public-code metadata are correct for the intended release mode.

---

## 13. If time is limited

Do the following minimum restructure first:

1. Rename headings using the map in Section 4.
2. Move held-out validation, fixed-pool event rates, and Gaussian sensitivity into the main Results narrative.
3. Add one theory-to-evidence bridge paragraph before each promoted evidence block.
4. Convert only two tables to figures:
   - endpoint ATR/SMPR;
   - Gaussian sensitivity mechanism.
5. Keep fixed-pool event-rate CI as a main table if plotting takes too long.
6. Rebuild and run release checks.

This minimum pass should already make the paper read much less like an internal report and much more like a coherent diagnostic paper.
