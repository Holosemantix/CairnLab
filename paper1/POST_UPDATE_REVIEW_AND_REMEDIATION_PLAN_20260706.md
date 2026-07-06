# Paper1 post-update review and remediation plan

Date: 2026-07-06  
Branch: `ag/dev`  
Scope: updated `paper1/main.tex` after the arXiv-v1 convergence pass.

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
5. Simplify or move the current main Table 2. At minimum remove `best obs`, `gain`, and `gap` columns. Do not use `best obs` language in the main paper.
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

## 2.4 Current main Table 2: should it stay?

### Current issue

The current Table 2 has columns:

```text
base obs σ=0.08 | std0.08 obs σ=0.08 | gain | best obs σ=0.08 | std0.08 gap
```

The `best obs` wording is a serious presentation problem. It reintroduces the point-best checkpoint story that the paper is trying to avoid. `gain` and `gap` also make the table feel like an internal leaderboard.

### Is the table fully redundant with Figure 1?

**Not fully.**

Figure 1 shows the seed-3072 sweep. The table summarizes the three-training-seed endpoint. That is the main support for the “across three training seeds” statement. So I do **not** recommend deleting all three-seed evidence from the main paper.

### Recommended compromise

Keep a very small main table, but remove the leaderboard columns.

Preferred main table:

```text
Task | no-noise ckpt, obs σ=0.08 | stdmax=0.08 ckpt, obs σ=0.08
```

Optional if space allows:

```text
Task | no-noise ckpt | stdmax=0.08 ckpt | Δ
```

But since the user explicitly dislikes `gain`, the safest main-table version is the two-score-column table. Put gains in prose only for PushT/Reacher if needed.

Recommended main prose:

```tex
Across three training seeds, the fixed stdmax=0.08 endpoint substantially improves the observation-noise endpoint on TwoRoom, PushT, and Reacher, with a weaker positive Cube signal. Because the full sweep forms broad task-dependent plateaus, we do not use point-best rows as a main result.
```

### Where should full sweep numbers go?

Move full sweep or point-best bookkeeping to appendix or artifact docs.

Recommended appendix replacement:

```tex
\section{Additional Gaussian Evaluation Tables}\label{sec:appendix-gaussian-evals}
```

This appendix can include:

- a full seed-3072 sweep table if compact;
- a three-seed endpoint table if removed from main;
- the observation+goal stress row as a small auxiliary subsection;
- a note that point-best rows are reproducibility bookkeeping, not a checkpoint-selection claim.

### What to do with current Appendix C

The current `Auxiliary Observation+Goal Gaussian Stress` appendix is too low-value by itself because it only reports std 0.08 rows. I agree with replacing it.

Recommended structure:

```tex
\section{Additional Gaussian Evaluation Tables}\label{sec:appendix-gaussian-evals}
\paragraph{Observation-only sweep.} ...
\paragraph{Auxiliary observation+goal stress.} ...
```

If the full sweep table is too large for the PDF, keep a compact appendix table and point to artifact paths.

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

If Table 2 is simplified or moved, rerun a grep for hard-coded table numbers and stale labels:

```bash
rg -n "Table~|Table |tab:training-seed-gaussian-lockbox|best obs|std0.08 gap|gain" paper1/main.tex paper1/*.md tools
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

### Patch B: tables and appendix

1. Simplify current main Table 2 to two score columns:

```text
Task | no-noise ckpt obs σ=0.08 | stdmax=0.08 ckpt obs σ=0.08
```

2. Remove `best obs`, `gain`, and `gap` from main table and connected prose.
3. Do not mention point-best rows in main except as a short plateau caveat.
4. Replace `Auxiliary Observation+Goal Gaussian Stress` appendix with `Additional Gaussian Evaluation Tables`.
5. Put observation+goal stress under that appendix as a small auxiliary subsection, not a standalone appendix section.
6. Add `\clearpage` before `\appendix`.

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
4. Simplify the current main Table 2. Remove `gain`, `best obs`, and `std0.08 gap`. Prefer a two-score-column table: no-noise checkpoint obs σ=0.08 and stdmax=0.08 checkpoint obs σ=0.08. Move point-best / full-sweep bookkeeping to appendix or artifact text.
5. Replace Appendix C “Auxiliary Observation+Goal Gaussian Stress” with a broader “Additional Gaussian Evaluation Tables” appendix. Observation+goal σ=0.08 can remain as a small auxiliary subsection, but not as the whole appendix.
6. Add \clearpage before \appendix so the appendix starts on a new page after references.
7. Slim the main theory section if needed: move finite-sample calibration, replanning union bound, and pseudo-metric material to appendix or delete from main. Keep the core ACPC definition, discriminability guard, cost-drift/top-1 link, sampled-pool tail motivation, Gaussian sensitivity, and collapse caveat.
8. Build and inspect the PDF. Confirm fig_acpc_basin_tsne.png is readable and does not visibly show legacy metrics R_E/R_F/rNN/disjoint. If the bottom ATR/SMPR annotation is too small, enlarge it or remove it and rely on the caption/table.

Acceptance checks:
- main.tex builds.
- No main-text `best obs`, `std0.08 gap`, or table leaderboard language remains.
- No awkward `compressed diagnostic keeps only... required...` wording remains.
- q90 is introduced only as an empirical reporting choice.
- Appendix starts on a new page.
- The paper remains honest: diagnostic only, no closed-loop guarantee, no new method claim.
```

---

## 6. Final recommendation

Proceed with this cleanup, then post arXiv v1.

The paper is now close enough that more large restructuring would have diminishing returns. The important remaining task is to remove small phrases and tables that trigger reviewer distrust. The next scientific step should not be more manuscript polishing; it should be the v2 method work with closed-loop evidence and baselines.
