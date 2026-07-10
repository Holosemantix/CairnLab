# Paper1 visual / text / table remediation plan for Codex

Target branch: `ag/dev`  
Primary files: `paper1/main.tex`, `paper1/scripts/*.py`, `paper1/tables/*.tex`, `paper1/docs/*.md`, `assets/paper1_figs/*`, `tools/check_paper1_consistency.py`  
Purpose: review the post-structure-remediation Paper1 and make a second-pass plan focused on top-conference figure clarity, figure/table/text alignment, source packaging, and final paper polish.

This plan assumes the current `ag/dev` already contains the first structure remediation: renamed sections, the ACPC concept figure, endpoint ATR/SMPR figure, full-sweep diagnostics figure, fixed-pool event-rate figure, and Gaussian sensitivity mechanism figure.

---

## 0. Executive assessment

The paper is now much closer to a formal diagnostic paper. The main results are visible in the main text, and the theory-to-evidence chain is much easier to see than before.

However, the current update introduced a new class of issues:

1. **The paper is now figure-rich but not yet figure-polished.** Several plots are technically correct but visually dense or not intuitive at first glance.
2. **Some evidence is promoted to main text, but the visual hierarchy is still uneven.** A reader should immediately know which figure establishes behavior, which figure supports the radius term, which supports the margin/planner term, and which supports local sensitivity.
3. **Some new headings are still awkward.** The biggest example is a `\paragraph{Full-sweep and held-out evidence}` inside a main result subsection. This is better than before, but it still feels like a report heading rather than a paper subsection.
4. **The theory section still has duplicated-sounding subheadings.** `Same-state predictive consistency and selective margin` and `Same-state predictive radius and selective margin` are too similar. They should be separated conceptually.
5. **Source packaging is now stale.** `main.tex` references new figures, but arXiv/blind source packaging instructions/scripts still copy only a subset of old figures.
6. **The consistency checker may now preserve wording that should be improved.** If a phrase is required only because it was useful during remediation, the checker should be updated together with the prose.

The next pass should not add new experiments. It should make the existing evidence readable, reviewer-facing, and visually professional.

---

## 1. Top-conference figure/table/text rubric

Use this rubric before editing any figure.

### 1.1 A top-conference figure must answer one question

For every main-text figure, write the question it answers before plotting:

- `fig2_sweep`: Does Gaussian observation noise break checkpoints, and does Gaussian training recover them?
- `fig_endpoint_atr_smpr`: Do recovered endpoints move in the radius and guard directions?
- `fig_full_sweep_diagnostics`: Do diagnostics separate recovered and fragile rows across the whole sweep?
- `fig_fixed_pool_event_rates`: Does the fixed-pool sufficient event align with lower candidate instability?
- `fig_gaussian_sensitivity_main`: Does ATR contraction coincide with lower local composed encoder--rollout sensitivity?

If a figure answers more than one question, split it or move part to appendix.

### 1.2 Main-text figure design rules

1. **One figure, one message.** Do not combine behavior, margin, sensitivity, and guard evidence into one crowded panel.
2. **Prefer small multiples over dual axes.** Dual axes are acceptable only when unavoidable and must be visually subtle.
3. **Use direct visual encodings.** If lower is better, add an arrow or subtitle. Do not make readers infer direction from captions.
4. **Use consistent colors across figures.** Suggested semantic mapping:
   - base / fragile: neutral gray or muted red;
   - recovered / noise-trained: blue or green;
   - behavior score: dark neutral;
   - ATR/radius failure: orange;
   - SMPR failure / guard failure: purple;
   - top-1 flip / planner instability: red.
5. **Use colorblind-safe palettes and grayscale-safe markers.** Combine color with marker shape or line style.
6. **Use readable text after scaling.** Assume the figure is printed in a one-column or full-width conference layout. Tick labels should remain readable at final PDF size.
7. **Avoid tiny numeric labels.** Use annotations only for key ratios or takeaways, not every data point.
8. **Remove nonessential chart ink.** Light grids, no heavy boxes, no unnecessary legends if direct labels work.
9. **Make uncertainty visible but not dominant.** Use error bars, Wilson intervals, or shaded bands only where they answer the claim.
10. **Captions should state the conclusion first, then define protocol.** Do not bury the claim after a long artifact description.

### 1.3 Main-text table rules

Use tables for exact values and mixed qualitative outcomes. Use figures for trends, before/after movement, and multi-row comparisons.

Keep in main text:

- small mixed-result tables, such as non-Gaussian boundary stressors;
- compact protocol tables if they replace long prose;
- tiny held-out validation table if no figure is needed.

Move to appendix:

- exact numeric values behind main figures;
- large sweep tables;
- dense audit/event-rate tables after a figure summarizes them;
- implementation/provenance tables.

Every table should use:

- `booktabs` only;
- no vertical lines;
- aligned decimals where possible;
- short captions with the takeaway first;
- footnotes for seed and protocol details if the caption becomes long.

### 1.4 Text around figures

Each main result block should have this pattern:

```text
Question / theory prediction.
Figure.
Answer in one paragraph: what moved, how large, what it means, what it does not mean.
```

Avoid repeating all numbers from the figure in text. Text should interpret, not transcribe.

### 1.5 Float and layout rules

The current paper uses many `[H]` floats. This makes the PDF feel rigid and can create awkward page breaks. For a conference-style paper:

- Use `[t]` or `[tbp]` for most figures.
- Keep `[H]` only when a figure must appear exactly at that point for readability.
- Avoid more than two consecutive full-width figures without substantial text.
- Put qualitative or supplemental visuals in appendix unless they carry a central claim.

---

## 2. Current paper issues to fix

### 2.1 Theory headings are still partially redundant

Current pattern:

```latex
\subsection{Same-state predictive consistency and selective margin}
...
\subsection{Same-state predictive radius and selective margin}
...
```

Problem: the two headings sound almost identical. A reviewer skimming the paper will not see the conceptual progression.

Recommended replacement:

```latex
\subsection{Paired rollout consistency and discriminability}
...
\subsection{Predictive tubes and planner margins}
...
```

Rationale:

- The first subsection defines ACPC and discriminability target.
- The second subsection introduces radius/tube/margin quantities used by the theorem.

If changing required phrases breaks `tools/check_paper1_consistency.py`, update the checker to require the new headings rather than preserving the weaker old wording.

### 2.2 `Full-sweep and held-out evidence` should not be a paragraph heading

Current pattern:

```latex
\paragraph{Full-sweep and held-out evidence.}\label{sec:exp-full-sweep-diagnostics}
```

Problem: this is semantically important enough to be a subsection/subsubsection. A paragraph heading makes it look like a local note.

Recommended fix:

Option A, if keeping it inside the ACPC diagnostic subsection:

```latex
\subsubsection{Full-sweep and held-out validation}
```

Option B, if page layout is cleaner:

```latex
\subsection{Full-sweep ACPC validation}
```

Recommendation: use **Option A**. It preserves the result chain while improving navigation.

### 2.3 The theorem still sounds too strong in a few places

Current terms such as `ACPC radius--margin certificate` are acceptable in a theorem title only if immediately qualified. In prose and figure captions, prefer:

- `fixed-pool radius--margin bound`;
- `fixed-pool sufficient condition`;
- `radius--margin diagnostic audit`;
- `event-rate audit`.

Recommended change:

- The theorem title can become `Fixed-pool radius--margin bound`.
- The phrase `diagnostic certificate` in prose should become `diagnostic bound` or `fixed-pool diagnostic condition`.
- The checker should stop requiring `ACPC radius--margin diagnostic certificate` if that phrase forces overclaiming.

### 2.4 Too many main figures may dilute the story

The current main text should keep only evidence-bearing figures:

1. Gaussian sweep;
2. endpoint ATR/SMPR;
3. full-sweep diagnostics;
4. fixed-pool event rates;
5. Gaussian sensitivity main plot;
6. non-Gaussian boundary table.

The concept schematic is retired from the main paper because it makes the paper look like a method/architecture paper rather than a diagnostic evidence paper. The qualitative PushT t-SNE remains appendix-only unless redesigned as a small qualitative inset.

Recommendation:

- Let the Gaussian sweep be the first main figure; it establishes the behavior before diagnostics.
- Keep endpoint/full-sweep ACPC, fixed-pool event-rate, and Gaussian sensitivity in main.
- Keep the non-Gaussian table because mixed results are clearer as exact rows.

### 2.5 Source packaging is stale

`main.tex` now references new figures, but the arXiv source bundle instructions and blind bundle script still copy only old figure subsets.

Required fix:

- Update `paper1/docs/README.md` source tarball commands.
- Update `paper1/docs/check_blind_ready.sh`.
- Prefer a script that parses `\includegraphics{...}` from `main.tex` and copies exactly the referenced figures, rather than a manually maintained list.

Minimum figure copy list should include all figures referenced in current `main.tex`, including at least:

```text
fig2_sweep.png
fig_endpoint_atr_smpr.png
fig_full_sweep_diagnostics.png
fig_fixed_pool_event_rates.png
fig_gaussian_sensitivity_main.png
fig_acpc_basin_tsne.png
fig_full_sweep_planner_guard.png
fig_radius_margin_overlap.png
fig_jvp_trace_decomposition_heatmap.png
```

If t-SNE or radius-margin overlap move to appendix/source-only status, still include them when the TeX source references them.

---

## 3. Figure-by-figure review and redesign plan

### 3.1 Concept schematic status

Decision: retire `fig1_concept.png` from the main paper. The first main figure should establish the behavioral phenomenon with the Gaussian sweep, not introduce a schematic that makes the paper look like a method/architecture paper. The ACPC radius, guard, and fixed-pool planning link should be carried by the definitions and theorem text, then supported by endpoint/full-sweep diagnostics.

If a concept schematic is reintroduced later, it must be a compact supplement or a redesigned inset, not the first main figure.

Retired schematic design target, only if needed later:

- Full-width or compact two-column schematic.
- Left half: same-state clean/noisy histories under shared actions, producing a predictive radius `R_sigma` / ATR.
- Right half: task/action-distinct pairs, producing margin `M_diff` / SMPR.
- Bottom strip: fixed-pool planning link, `cost drift < margin/2 -> top-1 stable`.
- Use three named objects only: `Radius`, `Guard`, `Planner margin`.
- Avoid long text inside boxes.
- Use vector source if possible (`.svg` or `.pdf`) and export PNG/PDF for LaTeX.

Implementation task:

- Create or update `paper1/scripts/draw_acpc_concept.py` or store an editable SVG under `assets/paper1_figs/source/`.
- Use consistent color semantics with the rest of the paper.

Acceptance:

- A reader can explain ACPC after viewing the figure for 10 seconds.
- Text is readable at final PDF size.
- Caption states: low radius is meaningful only with guard preservation.

### 3.2 `fig2_sweep.png`: Gaussian sweep recovery bands

Current role: main behavioral result.

Keep in main. It is central.

Potential improvements:

- Ensure each task panel has the same y-axis range if scores are comparable success rates. If task-specific y limits are used, state that clearly.
- Add light recovery-band shading or onset markers only if they do not clutter.
- Direct-label clean vs observation-noise curves near the right edge and reduce legend dependence.
- Keep error bars visible but not visually dominant.
- If four task panels are dense, use a 2x2 layout with identical x-axis ticks and aligned y labels.

Do not add more metrics to this figure. It should remain behavior-only.

### 3.3 `fig_endpoint_atr_smpr.png`: endpoint selective diagnostics

Current script: `paper1/scripts/plot_endpoint_atr_smpr.py`.

Current design issues:

- It uses two horizontal panels and log-scale ATR. This is correct but not instantly intuitive.
- It does not visually state the theorem link: ATR is the radius term, SMPR is the guard term.
- Error bars may be small and visually hard to interpret.
- The paired line direction is not annotated; for ATR, left is better, while for SMPR, right is better. This mixed direction can confuse readers.

Recommended redesign:

Option A, preferred main-text figure:

- Two panels titled:
  - `(a) Radius tail contracts: ATR q90`
  - `(b) Guard improves: SMPR`
- Use paired dumbbells by task.
- Add arrows in each panel:
  - ATR: `lower better` arrow pointing left.
  - SMPR: `higher better` arrow pointing right.
- Directly label base and noise-trained endpoints only once.
- Keep ATR log-scale but label ticks as interpretable values: `0.05, 0.1, 0.3, 1, 3`.
- Add small ratio annotations only if they do not clutter, e.g. `14x lower` for PushT or all-task median.

Option B, if absolute values remain too hard:

- Main panel: endpoint/base ratio for ATR and SMPR gain, normalized by task.
- Appendix table: exact absolute values.

Recommendation: implement **Option A** first.

Script changes:

- Increase figure width only if necessary, not more than full text width.
- Use shared y labels and direct annotation.
- Remove legend frame if direct labels are added.
- Add `ax.annotate` arrows for better directionality.

Acceptance:

- Without reading the caption, a reader sees ATR moves strongly downward and SMPR moves upward on all tasks.

### 3.4 `fig_acpc_basin_tsne.png`: qualitative ACPC neighborhood t-SNE

Current role: qualitative intuition.

Issue: t-SNE figures are often visually attractive but weak as evidence. In this paper, it risks distracting from the quantitative ATR/SMPR/full-sweep evidence.

Recommendation:

- Move to appendix by default.
- In main text, replace it with one sentence: qualitative PushT neighborhoods are shown in Appendix X and are not used for quantitative claims.

If keeping in main:

- Use a small cropped two-panel version: base rollout features vs noise-trained rollout features only.
- Remove encoder panels unless they directly support a claim.
- Add a clear visual annotation: `same-state noisy views tighten after rollout`.
- Make the caption very short and clearly qualitative.

Acceptance:

- The main result section should not depend on t-SNE to make the quantitative claim.

### 3.5 `fig_full_sweep_diagnostics.png`: full-sweep diagnostic separation

Current role: show diagnostic movement across 108 rows.

Likely issue: too many curves and bands in one figure. Current caption includes behavior score, normalized ATR, SMPR failure, fixed-pool top-1 disagreement, green recovery bands, and blue proxy-positive bands. This is a lot.

Recommended redesign:

Split into two figures or two stacked rows.

Preferred main figure:

`fig_full_sweep_behavior_acpc.png`

- 4 task panels.
- x-axis: training `stdmax`.
- left/primary y-axis: observation-only `sigma=0.08` success.
- secondary or lower row: normalized ATR and SMPR failure only.
- Recovery band shown as light neutral shading.
- Do not include fixed-pool top-1 disagreement in this figure; move planner curve to the fixed-pool section.

Planner companion figure, main or appendix:

`fig_full_sweep_planner_guard.png`

- fixed-pool top-1 flip or agreement;
- proxy gap positive/negative;
- recovery band as reference.

If page budget is tight, keep only the behavior+ACPC figure in main and move planner guard to appendix, because the event-rate figure already supports planner-side claims.

Script changes:

- Update `paper1/scripts/plot_full_sweep_diagnostics.py` to support `--mode behavior_acpc` and `--mode planner_guard`, or create two scripts.
- Reduce dual-axis complexity.
- Use direct labels and consistent colors.

Acceptance:

- A reader can distinguish behavior recovery from diagnostic failure curves without consulting the caption.
- No panel contains more than three plotted semantic objects.

### 3.6 `fig_fixed_pool_event_rates.png`: fixed-pool event-rate audit

Current script: `paper1/scripts/plot_fixed_pool_event_rates.py`.

Current design issues:

- Three side-by-side panels are dense.
- `top-1 flip | cert-pass` is mostly zero and wastes a full panel.
- Red/green can be colorblind-unfriendly and semantically overloaded.
- The figure does not clearly separate `higher is better` cert-pass from `lower is better` top-1 flip.

Recommended redesign:

Main figure:

- Two panels only:
  - `(a) Cert-pass rate (higher is better)`
  - `(b) Top-1 flip rate (lower is better)`
- Rows: TwoRoom, PushT, Reacher, Cube, ALL.
- Show fragile vs recovered as paired dots with Wilson intervals.
- Use neutral gray for fragile and blue for recovered, or muted red/blue with marker-shape redundancy.
- Add a small textbox below or inside panel B:
  - `flip | cert-pass = 0 in sampled anchors; Wilson intervals in Appendix`.
- Keep exact conditional rates in appendix table.

Script changes:

- Remove the third main axis from the default output.
- Add a CLI option `--include-conditional-panel` for appendix version if desired.
- Use `figsize` around 7.0 x 3.8 for two panels.
- Add directional arrows to titles or x-axis labels.

Acceptance:

- The figure should communicate: recovered rows certify more often and flip less often.
- Conditional zero should not dominate the layout.

### 3.7 `fig_gaussian_sensitivity_mechanism.png`: local sensitivity mechanism

Current script: `paper1/scripts/plot_gaussian_sensitivity_mechanism.py`.

Current design issues:

- It combines two log-scale bar panels plus a four-column heatmap. This is too much for one main-text figure.
- The heatmap uses `coolwarm` around log10 ratio; this is scientifically okay but visually loaded.
- The alignment coefficient has a different interpretation than encoder/rollout/composed ratios, so putting them in one heatmap can confuse readers.
- Small numeric labels inside the heatmap may be hard to read.

Recommended redesign:

Main figure:

`fig_gaussian_sensitivity_main.png`

- Two panels only:
  - finite-difference slope endpoint/base ratio;
  - JVP/Hutchinson composed trace endpoint/base ratio.
- Use dot/lollipop or horizontal bars on log x-axis.
- Add vertical line at 1.0.
- Add title: `lower = less local composed sensitivity`.
- Keep values as unobtrusive labels only if legible.

Appendix figure:

`fig_jvp_trace_decomposition_heatmap.png`

- Encoder, rollout, composed, alignment ratios.
- Explain that alignment is a diagnostic coefficient, not the same type of scale as traces.
- Consider separating alignment into its own small panel or line plot.

Script changes:

- Split current script into two outputs:
  - `--out-main assets/paper1_figs/fig_gaussian_sensitivity_main.png`
  - `--out-decomp assets/paper1_figs/fig_jvp_trace_decomposition_heatmap.png`
- Keep the current combined figure only as an optional transitional artifact.

Main-text changes:

- Replace `fig_gaussian_sensitivity_mechanism.png` with `fig_gaussian_sensitivity_main.png`.
- Move decomposition paragraph and heatmap to Appendix `Local Sensitivity Details`.

Acceptance:

- Main text figure should make one point: both local estimators drop below one at the endpoint.
- Decomposition should be available but not visually overloaded in main text.

### 3.8 `fig_radius_margin_overlap.png`: radius--margin overlap proxy

Current role: appendix/supporting mechanism.

Recommendation:

- Keep in appendix only.
- If referenced in main, use one sentence and point to appendix.
- Do not duplicate its message with the event-rate main figure.

If it remains as appendix figure:

- Make sure caption clearly says proxy gap is not a calibrated planner-margin certificate.
- Consider using a simpler positive/negative band plot instead of multiple curves if it is currently dense.

### 3.9 Non-Gaussian boundary table

Current table should stay in main. The result is mixed and exact values matter.

Minor improvements:

- Add arrow notation consistently: no-noise `->` noise-trained.
- Bold only if there is a clear directional improvement, but avoid over-boldening mixed outcomes.
- Keep caption short: `Mixed blur/resize outcomes delimit matched-Gaussian scope`.

---

## 4. Main-text layout recommendation after visual cleanup

Target main figure/table order:

1. **Figure 1: Gaussian sweep recovery bands**
2. **Figure 2: Endpoint ATR/SMPR selective diagnostics**
3. **Figure 3: Full-sweep behavior + ACPC validation**
4. **Small held-out validation table or text block**
5. **Figure 4: Fixed-pool event-rate audit**
6. **Figure 5: Local Gaussian sensitivity main plot**
7. **Table 2: Boundary stressor evaluation**
9. **Table 2: Boundary stressor evaluation**

If page budget is tight:

- Move t-SNE to appendix.
- Move Gaussian sensitivity decomposition heatmap to appendix.
- Move exact endpoint ATR/SMPR table to appendix.
- Move full-sweep planner guard plot to appendix.

---

## 5. Text polish recommendations

### 5.1 Introduction

Current introduction is substantially improved. Remaining issue: contribution 2 still uses `diagnostic certificate`, which may read too strong.

Recommended replacement idea:

```text
Second, it derives a fixed-pool radius--margin diagnostic bound and a local Gaussian sensitivity interpretation: high-probability same-state predictive radius, planner candidate margins, and task-grounded discriminability margins identify the failure modes tested by the empirical audits. ATR and SMPR are empirical diagnostics aligned with the radius and guard sides, not calibrated flip-probability bounds.
```

### 5.2 ACPC theory section

Reduce repeated caveats by moving the general limitation sentence to the end of the section. Keep precise caveats near theorem statements.

Replace:

- repeated `not a closed-loop guarantee` statements in multiple paragraphs;

with:

- one theorem-local caveat;
- one section-ending caveat.

### 5.3 Results section

For each subsection, make the first sentence a claim, not a procedure.

Examples:

- Current procedural style: `For the main diagnostic, ATR is...`
- Better: `Recovered checkpoints move in both parts of the selective ACPC diagnostic: the same-state radius contracts and the task-grounded guard improves.`

- Current procedural style: `To evaluate the local Gaussian sensitivity interpretation...`
- Better: `The ATR contraction is accompanied by a large drop in local composed encoder--rollout sensitivity.`

Then define measurement details after the claim.

### 5.4 Captions

Every caption should follow:

```text
Takeaway sentence. Metric/protocol details. Scope caveat if needed.
```

Do not start every caption with raw protocol unless the figure is a protocol figure.

Example fixed-pool caption revision:

```latex
\caption{Recovered rows certify more fixed-pool histories and flip fewer top candidates. Cert-pass denotes histories where maximum paired cost drift over the shared 65-candidate pool is below one half of the clean top-1/top-2 margin; top-1 flip is clean/noisy top-candidate disagreement. Intervals are Wilson 95\% intervals over sampled anchors and are not calibrated theorem probabilities.}
```

---

## 6. Checker and release cleanup

### 6.1 Update consistency checker after prose changes

`tools/check_paper1_consistency.py` currently requires many exact snippets. This is useful for release safety but can freeze awkward wording.

If headings or phrases are improved, update:

- `REQUIRED_MAIN_TEXT_SNIPPETS`;
- forbidden snippets if needed;
- required figure list if figure names change.

Do not keep a weaker phrase only because the checker requires it.

### 6.2 Update figure source packaging

Required updates:

- `paper1/docs/README.md`
- `paper1/docs/check_blind_ready.sh`
- any arXiv/source-bundle helper scripts if present.

Preferred implementation:

Create a helper script:

```text
paper1/scripts/collect_tex_figures.py
```

Function:

- parse `main.tex` and `docs/main_blind.tex` after resolving `\input{main.tex}`;
- collect every `\includegraphics{...}` target;
- search `paper1/figures`, `assets/paper1_figs`, and configured `\graphicspath` paths;
- copy exactly referenced figures into source bundle;
- fail if any referenced figure is missing.

If this is too much, at least manually update copy lists.

### 6.3 Build checks

After visual changes, run:

```bash
python -m paper1.scripts.run_all_paper1_diagnostics
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
rg -n "Citation .* undefined|Reference .* undefined|There were undefined references|Undefined control sequence|Fatal error|No file main.bbl" main.log
```

If checkpoint-dependent audits are not available, do not rerun them. Regenerate only the training-free plots from existing CSV/JSON artifacts.

---

## 7. Concrete Codex execution phases

### Phase 0: current-state audit

Run:

```bash
git checkout ag/dev
git pull --ff-only
rg -n "^\\section|^\\subsection|^\\subsubsection|^\\paragraph|includegraphics" paper1/main.tex > /tmp/paper1_structure_figs_before.txt
python -m tools.check_paper1_consistency | tee /tmp/paper1_check_before.log
cd paper1 && bash build.sh --clean | tee /tmp/paper1_build_before.log
cd ..
```

Record:

- list of main-text figures;
- list of appendix figures;
- current PDF page count;
- any overfull/undefined warnings.

### Phase 1: prose/headings cleanup

Tasks:

1. Rename duplicate theory subsections:
   - `Same-state predictive consistency and selective margin` -> `Paired rollout consistency and discriminability`.
   - `Same-state predictive radius and selective margin` -> `Predictive tubes and planner margins`.
2. Promote `Full-sweep and held-out evidence` from paragraph to subsubsection or subsection.
3. Replace over-strong `certificate` prose with `fixed-pool bound`, `sufficient condition`, or `diagnostic audit` where appropriate.
4. Make result subsection first sentences claim-first.
5. Update checker snippets accordingly.

### Phase 2: figure redesign scripts

Tasks:

1. Update `plot_endpoint_atr_smpr.py`:
   - add direction arrows;
   - improve axis labels;
   - reduce legend dependence;
   - keep exact values in appendix.
2. Update `plot_fixed_pool_event_rates.py`:
   - make default output two panels;
   - move conditional metric to annotation or appendix-only optional panel.
3. Split `plot_gaussian_sensitivity_mechanism.py`:
   - main output: two-panel sensitivity ratios;
   - appendix output: decomposition heatmap.
4. Split or simplify `plot_full_sweep_diagnostics.py`:
   - main output: behavior + ATR/SMPR failure;
   - optional planner-guard output: top-1 flip/proxy gap.
5. Ensure all scripts read CSV/JSON artifacts and do not hand-enter numbers.

### Phase 3: main-text figure relocation

Tasks:

1. Move `fig_acpc_basin_tsne` to appendix unless a simplified two-panel version is created.
2. Replace main `fig_gaussian_sensitivity_mechanism` with `fig_gaussian_sensitivity_main`.
3. Move JVP decomposition heatmap to `Local Sensitivity Details` appendix.
4. If full-sweep plot is split, update main references and captions.
5. Ensure no section has three full-width figures in a row.

### Phase 4: source packaging and checker

Tasks:

1. Update README source tarball copy list or implement `collect_tex_figures.py`.
2. Update blind source bundle script.
3. Update `REQUIRED_ARTIFACTS` and `REQUIRED_MAIN_TEXT_SNIPPETS` in checker.
4. Run all build and grep checks.

### Phase 5: final visual inspection

Open the compiled PDF and inspect every main figure at actual page size.

Checklist:

- Are all labels readable without zoom?
- Can the key claim be understood from the figure alone?
- Are colors consistent across figures?
- Is uncertainty visible but not dominant?
- Are legends unnecessary or placed unobtrusively?
- Does the caption start with the takeaway?
- Are exact values available in appendix if the main figure is visual summary?
- Are qualitative figures clearly marked qualitative?

---

## 8. Priority if time is limited

Do these first:

1. Fix source packaging for all referenced figures.
2. Promote `Full-sweep and held-out evidence` to a real subsubsection.
3. Move t-SNE to appendix.
4. Redesign `fig_fixed_pool_event_rates` as two panels.
5. Split `fig_gaussian_sensitivity_mechanism` into main two-panel ratio figure + appendix heatmap.
6. Add direction arrows / clearer labels to `fig_endpoint_atr_smpr`.
7. Update checker snippets and rebuild.

This minimum pass will remove the most visible top-conference polish problems.

---

## 9. Final acceptance checklist

### Paper structure

- [x] No internal-report-like heading remains in main text.
- [x] Theory headings are not redundant.
- [x] Full-sweep and held-out validation are visible in navigation.
- [x] Main figures follow the theory objects: radius, guard, planner margin, local sensitivity.
- [x] Qualitative t-SNE is either appendix-only or clearly secondary.

### Figures

- [x] Endpoint ATR/SMPR figure has clear directionality.
- [x] Full-sweep figure has no more than three semantic objects per panel.
- [x] Fixed-pool event-rate figure uses two main panels and does not waste space on all-zero conditional rates.
- [x] Gaussian sensitivity main figure is not overloaded with decomposition heatmap.
- [x] Colors are consistent and colorblind-safe.
- [x] Text is readable at final PDF size.
- [x] Captions start with takeaways.

### Tables

- [x] Exact values behind converted figures remain in appendix.
- [x] Non-Gaussian boundary table remains compact and interpretable.
- [x] Dense audit tables are not duplicated in main text.

### Claims

- [x] No calibrated closed-loop certificate claim.
- [x] No adaptive CEM/replanning guarantee claim.
- [x] SMPR remains proxy-level unless stronger labels are added.
- [x] Non-Gaussian rows remain scope-boundary evidence.

### Release

- [x] All figures referenced by `main.tex` are copied into arXiv and blind source bundles.
- [x] `python -m tools.check_paper1_consistency` passes.
- [x] `cd paper1 && bash build.sh --clean` passes.
- [x] No undefined citations/references or fatal LaTeX diagnostics.

---

## 10. Completion record (2026-07-10)

Phases 0--5 are complete. The final pass also replaced the undersized four-across Gaussian sweep with a native-width 2x2 figure and added subsection float barriers so the local-sensitivity figure and non-Gaussian boundary table cannot drift into the following result section.

Validation evidence:

- `bash paper1/scripts/run_all_paper1_diagnostics.sh` completed for the checked-in, training-free artifact path.
- `python -m tools.check_paper1_consistency` passed, including the main/appendix figure-set, paper-facing terminology, and isolated-bundle gates.
- `pytest -q` passed all 33 tests; the collector-specific regression shard passed all 3 tests.
- `cd paper1 && bash build.sh --clean` produced a 28-page PDF with no undefined citation/reference, fatal, overfull, or underfull diagnostics.
- The arXiv source bundle compiled in isolation to 28 pages; the blind bundle compiled in isolation to 29 pages because of anonymous front-matter layout. Each contains exactly the 9 figures referenced by its TeX entry point.
- Final PDF inspection covered every main figure plus the t-SNE, planner-guard, radius--margin, and JVP decomposition appendix figures at actual page size.

The remaining author placeholder in `paper1/arxiv_metadata.tex` is an intentional release-time human input and must be replaced before a public arXiv upload.
