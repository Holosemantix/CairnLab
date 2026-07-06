# Paper1 post-update review and remediation plan

Date: 2026-07-06  
Branch: `ag/dev`  
Scope: updated `paper1/main.tex` after the arXiv-v1 convergence pass.  
Companion decision note: `paper1/THREE_SEED_FIGURE1_TABLE2_DECISION_20260706.md`.

This memo is a review/planning document for Codex execution. Do not include it in arXiv source packages.

---

## 0. Updated review verdict

### 0.1 What improved

The newest version is materially better than the previous one.

The strongest improvements are:

- The Introduction now reads as a narrative instead of starting abruptly with subsection 1.1.
- Contributions are integrated into prose rather than formatted as project-outline C1/C2/C3 blocks.
- The old theory-to-metric table has been removed.
- The main diagnostic set is now cleanly compressed to ATR and SMPR.
- The new ACPC t-SNE figure direction is closer to the intended old ACPC-basin visualization.
- Goal-corrupted evaluation is no longer mixed into the main observation-only robustness endpoint.

### 0.2 Current arXiv v1 readiness

**Nearly ready, but I would do one more small cleanup pass before posting.**

The paper now has a defensible arXiv v1 identity as:

> a controlled diagnostic study of matched-Gaussian visual robustness in JEPA latent world-model checkpoints.

The remaining issues are not fatal, but they are exactly the kind of small phrasing/table/theory issues that make reviewers feel the paper is still an internal project report. Clean those up before arXiv.

### 0.3 Current top-conference review estimate

As a top-conference main-track submission today, my review would improve from the earlier borderline WR to:

```text
Overall: 6/10, weak reject / borderline.
Confidence: medium.
```

The paper is now much more coherent as a diagnostic paper. It still lacks the weight of a top-conference method paper: no new training method, no strong method baselines, and no causal demonstration that optimizing ATR/SMPR improves closed-loop robustness. If v2 adds a mature method with proper baselines and the diagnostic-to-behavior causal chain, the ceiling becomes much higher.

---

## 1. Highest-priority fixes

These should be the next Codex task.

1. Remove the premature 90th-percentile sentence from the theory definition paragraph.
2. Rename `Action-relevant discriminability (countercondition)` to a cleaner label such as `Action-relevant discriminability` or `Discriminability guard`.
3. Add a short statement that discriminability can also be posed over state-action pairs, which covers same-state/different-action futures when those actions induce different external future readouts.
4. Remove the project-management phrase `The compressed diagnostic keeps only the two empirical quantities required by the selective-ACPC logic` from the paper.
5. Update the main sweep figure to three-training-seed mean/std and delete the current main three-seed endpoint table. Put full all-seed Gaussian sweep/evaluation numbers in appendix or artifact tables.
6. Replace Appendix C with a broader `Additional Gaussian Evaluation Tables` appendix, or move the current observation+goal-only table to artifact documentation.
7. Add `\clearpage` before `\appendix` so appendices do not run directly after the reference list on the same page.
8. Slim the theory section: move finite-sample calibration, replanning union bound, and pseudo-metric material to appendix or remove if not needed.

---

## 2. Point-by-point response to the user's new comments

## 2.1 The sentence: “In experiments, ATR uses the 90th percentile ...”

### Current issue

This sentence appears immediately after the formal ACPC-H definition. That is the wrong location. The reader is still trying to understand the mathematical object, and suddenly a specific empirical quantile appears. This makes the theory feel ad hoc.

### Decision

**Remove this sentence from the theory definition paragraph.**

Do not remove the fact that ATR is q90. Move it to the operational diagnostics subsection or experimental details.

Recommended replacement in the theory definition paragraph:

```tex
Thus \Cref{eq:acpc-h} defines the clean/perturbed projected-rollout distance under the shared action sequence.
```

Then in `Operational consistency diagnostics`, write:

```tex
ATR reports a fixed upper-tail summary of this same-state projected-rollout disagreement; we use the 90th percentile in all experiments. The sampled-pool argument below motivates a tail readout, but it does not privilege 0.90 as a theoretical constant.
```

Rationale:

- The theorem motivates a tail probability / upper-tail diagnostic.
- The 90th percentile is an empirical reporting choice, not a theorem-derived number.
- If space permits, appendix can state that q90 was chosen as a stable tail summary under finite diagnostic sample size; do not over-explain in main text.

---

## 2.2 ACPC consistency, discriminability, and missing same-state/different-action condition

### Current issue

The current definition separates:

1. Same-state clean/noisy views under the same action sequence should yield close predicted futures.
2. Different states under an action sequence should remain separated if their external future readouts differ.

This is correct for the paper's visual-perturbation diagnostic, but it leaves one conceptual gap: what about the same clean state under two different actions? If the actions produce different futures, the predictor should not collapse them.

### Is this missing condition necessary?

**It is conceptually important, but it does not need to become a new paper-facing metric in v1.**

The main paper's diagnostic question is visual robustness under matched action interventions, so the same-state/different-action condition is not directly used by ATR. However, it belongs to the broader anti-collapse / action-faithfulness side of the definition. A predictor that ignores actions could satisfy same-state clean/noisy consistency while being useless for planning.

### Recommended fix

Generalize the discriminability explanation from state pairs to **state-action pairs**.

Minimal text patch:

```tex
Although we state the main consistency condition for clean and perturbed views under a shared action sequence, the discriminability side can be posed over state-action pairs. If two pairs $(s_i, \mathbf a_i)$ and $(s_j, \mathbf a_j)$ have external task-relevant futures that differ by more than a margin, their projected rollouts should remain separated. This includes the important special case of the same clean state evaluated under two dynamically distinct actions.
```

Then keep SMPR as the implemented state-neighborhood proxy:

```tex
The reported SMPR instantiates this guard with task-grounded near-boundary state pairs rather than a separate action-faithfulness metric; action-distinct audits are a natural stronger extension but are not required for the matched-visual-perturbation claim.
```

### Rename recommendation

Change:

```tex
\paragraph{Action-relevant discriminability (countercondition).}
```

to one of:

```tex
\paragraph{Action-relevant discriminability.}
```

or:

```tex
\paragraph{Discriminability guard.}
```

I prefer `Action-relevant discriminability.` It is cleaner and avoids the awkward parenthetical `countercondition`.

---

## 2.3 The phrase: “The compressed diagnostic keeps only ... required ...”

### Current issue

This sentence sounds like internal metric-compression bookkeeping. It is also too strong: the theory does not prove that exactly two empirical quantities are “required.” It only motivates a tail readout and a discriminability guard.

### Decision

**Remove or rewrite. Do not keep this sentence in the paper.**

Recommended replacement:

```tex
We report two readouts matched to this logic. ATR summarizes the upper tail of same-state action-conditioned projected-rollout disagreement. SMPR checks the no-collapse side by asking whether task-grounded near-boundary pairs remain separated beyond the same-state noisy radius. Low ATR is therefore interpreted only together with high SMPR.
```

Avoid:

- `compressed diagnostic` in main prose;
- `required by the selective-ACPC logic`;
- `paper-facing`;
- `reported metric set` unless in appendix or tool docs.

---

## 2.4 Main Table 2 versus Figure 1

### Updated decision

**Yes: update Figure 1 to show the three-training-seed Gaussian sweep, delete the current main Table 2, and put the full all-seed sweep/evaluation data in the appendix.**

This is now the cleanest solution. It removes the remaining leaderboard smell from the main text while preserving the evidence needed for the “three training seeds” claim.

### Why this is better than keeping a small Table 2

The earlier compromise was to keep a minimal two-column endpoint table. After reconsidering the revised manuscript, a stronger move is to let Figure 1 carry the three-seed evidence directly:

- The current Figure 1 is already the natural place to show the sweep/plateau story.
- If Figure 1 becomes three-seed mean ± std, it subsumes the current endpoint table.
- Deleting Table 2 avoids `gain`, `gap`, `best obs`, and point-best checkpoint language entirely.
- The appendix can carry the exact numeric sweep rows for reproducibility.

### Required Figure 1 change

Regenerate the sweep figure so that each plotted point is:

```text
mean over LeWM training seeds 3072/3073/3074
```

and the error bar is:

```text
population std across those three training seeds
```

For each training seed, the plotted value should first average evaluation seeds 42/43/44 with 100 trajectories per evaluation seed.

The figure should show at least:

```text
unperturbed evaluation success
observation-only Gaussian evaluation success at σ=0.08 with a clean goal image
```

across the full training-noise sweep:

```text
stdmax ∈ {0.0, 0.01, 0.02, ..., 0.08}
```

If space permits, a supplementary version can include additional evaluation severities such as `obs σ=0.03` and `obs σ=0.05`, but the main figure should not become visually overloaded.

### Caption wording

Recommended caption:

```tex
Three-training-seed LeWM Gaussian sweep. Each point averages evaluation seeds 42/43/44 within each training seed and then reports the mean across training seeds 3072/3073/3074; error bars show population std across training seeds. Blue circles show unperturbed evaluation and red squares show observation-only Gaussian evaluation at σ=0.08 with a clean goal image. The curves show broad task-dependent recovery plateaus rather than point-best checkpoint rankings.
```

### Main-text replacement for current Table 2 prose

After Figure 1, replace the current endpoint-table paragraph with:

```tex
Because \Cref{fig:sweep} already aggregates the full sweep across three training seeds, we do not report a separate point-best or endpoint leaderboard in the main text. The fixed \stdmax{}=0.08 endpoint substantially improves the observation-noise endpoint on TwoRoom, PushT, and Reacher, with a weaker positive Cube signal. Exact all-seed sweep values are reported in Appendix~\ref{sec:appendix-gaussian-evals} for reproducibility.
```

Avoid the terms:

```text
best obs
gain
gap
regret
point-best row
```

in main results prose, except when explicitly explaining that point-best ranking is not the claim.

### Appendix replacement

Replace current Appendix C with:

```tex
\section{Additional Gaussian Evaluation Tables}\label{sec:appendix-gaussian-evals}
```

This appendix should include exact all-seed sweep/evaluation data. Preferred structure:

```tex
\paragraph{Three-seed observation-only sweep.}
A compact table or landscape table containing task, training stdmax, clean success, obs σ=0.08 success, and optionally obs σ=0.03 / obs σ=0.05 if available.

\paragraph{Auxiliary observation+goal stress.}
A small table for obs+goal σ=0.08, clearly labeled auxiliary and not part of the main endpoint.
```

If the all-seed sweep table is too wide, use one table per task or a landscape table. If it is too large for the PDF, include a compact appendix summary and cite the generated artifact paths.

### Required new artifacts

Generate artifact files so the figure and appendix are not hand-entered:

```text
assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json
assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.md
```

The artifact should include:

```text
task
stdmax
per-training-seed values for each eval condition
mean/std across training seeds
source manifests
```

Do not invent or interpolate missing severities. If `obs σ=0.03` or `obs σ=0.05` are not available for all three training seeds, either omit those columns or mark them unavailable in the appendix artifact.

---

## 2.5 Appendix should start on a new page

### Current issue

The current source has:

```tex
\bibliographystyle{plain}
\bibliography{references}

\appendix
\section{Proofs and Calibration for ACPC Diagnostics}
```

Without a page break, the appendix can start immediately after references on the same page, which looks unpolished.

### Decision

**Add `\clearpage` before `\appendix`.**

Recommended patch:

```tex
\bibliographystyle{plain}
\bibliography{references}

\clearpage
\appendix
```

Keeping references before appendix is acceptable for an arXiv-style technical report. If a target conference requires appendix as supplement, handle that in the venue-specific source later. For the current arXiv v1, the priority is simply to make the appendix visually separate.

---

## 2.6 Theory completeness and trimming

### Current verdict

The theory is **complete enough** for a diagnostic paper, but it is now slightly too long and risks overpromising. The issue is not missing theory; the issue is too many calibration-style statements in the main text.

### Keep in main text

Keep these elements:

1. ACPC definition under shared action sequence.
2. Discriminability guard, revised to allow state-action pairs.
3. Cost-drift proposition.
4. Top-1 margin stability proposition or corollary.
5. Sampled-pool theorem, if kept short.
6. Local Gaussian sensitivity proposition.
7. Collapse counterexample or a short paragraph equivalent.

### Move to appendix or remove from main

Move or remove:

1. `Finite-sample tail calibration` proposition. It is useful but not central.
2. `Replanning union bound`. It is loose and invites closed-loop guarantee questions.
3. `Selective ACPC pseudo-metric`. It is mathematically fine but interrupts the empirical diagnostic story.
4. Duplicate proof text. If a proposition is one-line Lipschitz, either keep the proof inline or put it in appendix, not both.
5. Any repeated language saying the paper is not a guarantee. Say it once clearly after the sampled-pool theorem and once in limitations.

### Recommended main-theory structure

```text
3.1 Setup and ACPC definition
3.2 Discriminability guard, including state-action-pair note
3.3 Operational diagnostics: ATR and SMPR
3.4 Why ACPC connects to planning: cost drift + top-1 margin
3.5 Sampled-pool/tail motivation and Gaussian sensitivity
3.6 Collapse caveat / why SMPR is needed
```

Target length: roughly 1.5--2 pages in main text.

---

## 3. Additional issues I noticed

### 3.1 Abstract wording

Current abstract says:

```text
These results support ACPC as a bounded diagnostic...
```

This is mostly fine, but a more cautious diagnostic-paper wording is:

```text
These results support ATR and SMPR as bounded ACPC diagnostics for matched-Gaussian robustness plateaus in fixed JEPA world-model checkpoints.
```

Reason: ACPC is the conceptual frame; ATR/SMPR are the empirical diagnostics.

### 3.2 Figure caption is okay, but the plot should avoid tiny bottom text

The current t-SNE figure caption is in the right direction. The plot-generation script also hides legacy visible stats in paper-facing mode and keeps high-D cluster stats in the sidecar JSON only. That is good.

Remaining risk: if the bottom annotation in the PNG is too small, it may become illegible in the PDF. Codex should visually inspect `fig_acpc_basin_tsne.png` in the compiled PDF. If the ATR/SMPR annotation is too small, either enlarge it or remove the in-figure text and keep ATR/SMPR only in the caption/table.

### 3.3 “Compressed” appears too often

The term `compressed` is useful in internal docs but not ideal in the manuscript. Search and replace most main-text occurrences with:

- `reported diagnostic`,
- `two diagnostic readouts`,
- `ATR/SMPR diagnostics`,
- or simply remove the qualifier.

### 3.4 Table numbering after changes

If Table 2 is deleted, rerun a grep for hard-coded table numbers and stale labels:

```bash
rg -n "Table~|Table |tab:training-seed-gaussian-lockbox|best obs|std0.08 gap|gain|regret" paper1/main.tex paper1/*.md tools
```

Use labels, not table numbers, in prose.

---

## 4. Concrete Codex task list

Ask Codex to implement these in order.

### Patch A: theory wording and structure

1. Remove the q90 sentence from the ACPC definition paragraph.
2. Move q90 explanation to `Operational consistency diagnostics` and explicitly say q90 is a fixed reporting choice, not a theoretical constant.
3. Rename `Action-relevant discriminability (countercondition)` to `Action-relevant discriminability`.
4. Add a short state-action-pair note to cover same-state/different-action futures.
5. Replace the `compressed diagnostic keeps only... required...` paragraph with the softer two-readout wording.
6. Move finite-sample calibration, replanning union bound, and pseudo-metric material to appendix, or delete if the paper becomes cleaner without them.

### Patch B: main Figure 1, tables, and appendix

1. Regenerate the main sweep figure as a three-training-seed mean/std plot over LeWM seeds 3072/3073/3074.
2. Error bars must be population std across training seeds, not evaluation seeds.
3. Each training-seed value should first average evaluation seeds 42/43/44.
4. Replace the current `fig2_sweep.png` or add a new canonical figure such as:

```text
assets/paper1_figs/fig1_three_seed_sweep.png
```

5. Update `main.tex` to use the new figure and caption.
6. Delete the current main Table 2 (`tab:training-seed-gaussian-lockbox`) and remove connected `best obs`, `gain`, `gap`, `regret`, or point-best prose from the main text.
7. Generate exact all-seed sweep artifacts:

```text
assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json
assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.md
```

8. Replace the standalone observation+goal appendix with:

```tex
\section{Additional Gaussian Evaluation Tables}\label{sec:appendix-gaussian-evals}
```

9. Put full or compact all-seed sweep/evaluation data in that appendix. Put obs+goal σ=0.08 as a small auxiliary subsection, not the whole appendix.
10. Add `\clearpage` before `\appendix`.

### Patch C: figure and build hygiene

1. Confirm `fig_acpc_basin_tsne.png` is readable in the built PDF.
2. If bottom annotation is too small, enlarge or remove it.
3. Ensure the paper-facing figure does not visibly show `R_E`, `R_F`, `r/NN`, or `disjoint`.
4. Keep sidecar high-D stats as audit metadata only.
5. Run consistency/build checks.

---

## 5. Ready-to-paste Codex prompt

```text
You are editing qun-team/wm_exp on branch ag/dev. Use paper1/POST_UPDATE_REVIEW_AND_REMEDIATION_PLAN_20260706.md as the source of truth for this cleanup pass.

Do not change the paper's scientific claim: it is an arXiv v1 diagnostic / empirical analysis paper, not a method paper.

Tasks:
1. In paper1/main.tex, remove the sentence “In experiments, ATR uses the 90th percentile...” from the ACPC definition paragraph. Move the q90 explanation to Operational consistency diagnostics and state that q90 is a fixed empirical reporting choice, not a theoretical constant.
2. Rename “Action-relevant discriminability (countercondition)” to “Action-relevant discriminability” or “Discriminability guard”. Add a short note that the guard can be posed over state-action pairs, covering same-state/different-action futures when the external future readout differs. Do not add a new paper-facing metric unless existing artifacts support it.
3. Replace the sentence “The compressed diagnostic keeps only the two empirical quantities required by the selective-ACPC logic” with softer paper prose: “We report two readouts matched to this logic...” Avoid “compressed”, “required”, “paper-facing”, and project-management wording in the manuscript.
4. Update the main sweep figure so it aggregates LeWM training seeds 3072/3073/3074. Each point should average eval seeds 42/43/44 within each training seed, then plot mean ± population std across training seeds. Show at least clean evaluation and observation-only σ=0.08 evaluation over stdmax ∈ {0.0,0.01,...,0.08}. Update the caption accordingly.
5. Delete the current main Table 2 / `tab:training-seed-gaussian-lockbox`. Remove main-text `best obs`, `gain`, `std0.08 gap`, `regret`, and point-best leaderboard language. Exact sweep values should move to appendix or generated artifacts.
6. Generate exact all-seed sweep/evaluation artifacts at `assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json` and `.md`. Do not invent or interpolate missing eval severities; omit unavailable columns or mark them unavailable in the artifact.
7. Replace Appendix C “Auxiliary Observation+Goal Gaussian Stress” with a broader “Additional Gaussian Evaluation Tables” appendix. Include full or compact all-seed sweep/evaluation data there, and keep observation+goal σ=0.08 only as a small auxiliary subsection.
8. Add \clearpage before \appendix so the appendix starts on a new page after references.
9. Slim the main theory section if needed: move finite-sample calibration, replanning union bound, and pseudo-metric material to appendix or delete from main. Keep the core ACPC definition, discriminability guard, cost-drift/top-1 link, sampled-pool tail motivation, Gaussian sensitivity, and collapse caveat.
10. Build and inspect the PDF. Confirm fig_acpc_basin_tsne.png is readable and does not visibly show legacy metrics R_E/R_F/rNN/disjoint. If the bottom ATR/SMPR annotation is too small, enlarge it or remove it and rely on the caption/table.

Acceptance checks:
- main.tex builds.
- Main sweep figure is three-training-seed mean/std, not seed-3072-only.
- Main Table 2 is gone.
- No main-text `best obs`, `std0.08 gap`, `gain`, `regret`, or table leaderboard language remains.
- No awkward `compressed diagnostic keeps only... required...` wording remains.
- q90 is introduced only as an empirical reporting choice.
- Appendix starts on a new page and contains/points to exact all-seed sweep data.
- The paper remains honest: diagnostic only, no closed-loop guarantee, no new method claim.
```

---

## 6. Final recommendation

Proceed with this cleanup, then post arXiv v1.

Updating Figure 1 to the three-seed sweep and deleting the current endpoint table is the best presentation route. It makes the main story visual and avoids point-best leaderboard framing, while the appendix/artifacts keep exact reproducibility data. After this pass, further large manuscript restructuring should stop; the next scientific step should be the v2 method work with closed-loop evidence and baselines.
