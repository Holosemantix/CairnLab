# Paper1 arXiv v1 / top-conference review and Codex execution plan

Date: 2026-07-06  
Branch: `ag/dev`  
Scope: `paper1/main.tex`, `paper1/docs/README.md`, `tools/README_paper1.md`, compressed ATR/SMPR artifacts, legacy diagnostic policy docs.

This document is written as a pre-arXiv and pre-Codex review memo. It should not be included in any arXiv source tarball.

---

## 0. Current high-level verdict

### 0.1 Is this suitable as arXiv v1?

**Yes, but only after one manuscript-convergence pass.**

The current paper has enough substance to post as an arXiv v1 **if it is framed exactly as a controlled diagnostic / empirical analysis paper**:

> A no-retraining diagnostic study of matched-Gaussian visual robustness in JEPA latent world-model control, centered on action-conditioned predictive consistency and a discriminability guard.

This is a defensible arXiv v1 because the paper already has:

- a clear phenomenon: no-noise LeWM checkpoints fail under observation-only Gaussian noise;
- a controlled intervention: full-sequence input-side Gaussian noise training over a fixed sweep;
- replication over three LeWM training seeds;
- four heterogeneous control tasks;
- two compressed theory-aligned diagnostics, ATR and SMPR;
- explicit scope boundaries: not a new training algorithm, not a universal robustness theorem, not a closed-loop stability guarantee.

However, the manuscript should not be posted in its current exact form if the goal is to look polished. The current version still has too many project-report signals: a table for theory-to-metric mapping, redundant sweep/endpoint tables, a feature-neighborhood figure that does not match the intended ACPC-basin visualization, and an introduction that starts with subsection headings before giving the reader a narrative runway.

### 0.2 Does the current paper have top-conference weight?

**As of this version: likely Weak Reject / borderline reject at a top ML venue if submitted as a main-track paper.**

This is not because the idea is weak. The central diagnostic idea is good. The likely reviewer concern is that the paper is currently between categories:

- If read as a **method paper**, it lacks a new algorithm and lacks comparison to robust-control or augmentation baselines.
- If read as a **theory paper**, the theory is calibration-level and assumption-based rather than a deep theorem.
- If read as a **diagnostic / empirical analysis paper**, the framing is promising, but the current presentation needs to be tighter and less internally motivated.

A more accurate top-conference review score right now would be around:

```text
Overall: 5/10 or 6/10 depending on reviewer taste.
Confidence: medium.
Recommendation: Weak reject unless the venue strongly values diagnostic empirical studies.
```

The best strategic path is therefore:

1. Post arXiv v1 after the convergence edits below.
2. Keep v1 honest and scoped.
3. Use v1 to establish the ACPC diagnostic lens and artifact trail.
4. Develop the new method separately.
5. Update v2 only when the method has mature, causal, closed-loop evidence.

This is a suitable “先站住坑位” strategy as long as v1 does not overclaim novelty as a method.

---

## 1. Objective reviewer-style assessment

### 1.1 Summary in reviewer language

The paper argues that visual robustness for JEPA latent world-model control should not be defined as encoder-level invariance. Instead, same-state clean/corrupted observations should induce consistent action-conditioned predicted futures, while action-relevant state distinctions remain separable. The paper introduces a compressed diagnostic pair: ACPC Tail Risk (ATR) for same-state rollout-disagreement tails and Selective Margin Pass Rate (SMPR) for an anti-collapse discriminability guard. Experiments on LeWM checkpoints across four control tasks show severe Gaussian observation-noise fragility for no-noise checkpoints, broad robustness plateaus under matched Gaussian input noise training, and large ATR/SMPR movement at recovered endpoints. The paper explicitly limits the claim to matched Gaussian stress and fixed-checkpoint diagnostics.

### 1.2 Strengths

1. **Good problem reframing.** The paper correctly attacks a real weakness in naive JEPA robustness claims: latent prediction does not by itself define control robustness.
2. **Selective criterion is conceptually strong.** Pairing predictive consistency with discriminability is much better than only measuring encoder closeness.
3. **The main claim is now scoped.** The abstract and discussion already avoid claiming a new training method or universal robustness guarantee.
4. **Three-training-seed Gaussian replication is valuable.** This makes the empirical claim much stronger than a single-seed sweep.
5. **ATR/SMPR compression is the right direction.** It reduces earlier metric sprawl and aligns the paper with a clean diagnostic story.

### 1.3 Main weaknesses reviewers will notice

1. **Presentation still looks over-instrumented.** Multiple tables and figures restate similar sweep or endpoint facts. This makes the paper feel like an internal experiment report.
2. **The Introduction is structurally abrupt.** It starts immediately with subsection 1.1 and reaches “Contributions” as another subsection. That makes the paper feel outline-like rather than narrative.
3. **The theory section is slightly overbuilt for the claim.** The propositions are useful, but the theory-to-metric table and repeated proof/calibration material make it feel more formal than the evidence supports.
4. **Figure 3 does not yet match the intended mechanism visualization.** The current feature-neighborhood figure is a PCA-style qualitative illustration plus side readouts. The intended figure is closer to the old ACPC basin / t-SNE cluster figure showing encoder and predictor features under repeated perturbations.
5. **Unseen-stressor material must remain bounded.** The blur/resize checks are useful scope checks but should not compete with the matched Gaussian claim.
6. **No new method yet.** For a top main track, the paper will need either a stronger empirical-analysis identity or a mature v2 method.

### 1.4 Bottom-line recommendation

For arXiv v1:

```text
Proceed after cleanup. The paper has enough substance as a diagnostic technical report.
```

For top conference submission now:

```text
Do not submit as-is. Clean up the manuscript first, and preferably wait for the method extension unless targeting a venue receptive to empirical diagnostic papers.
```

---

## 2. Decisions on the six requested issues

## 2.1 Table 1: theory-to-metric mapping

### Current issue

`Table 1` maps theorem quantities to ATR/SMPR. This is clear, but it is not ideal for a top-conference-style manuscript because it reads like an internal summary table. It also makes the theory section look heavier than it is.

### Decision

**Remove Table 1 from the main text. Replace it with 2--4 sentences in `sec:acpc-diag`.**

Recommended replacement paragraph:

```tex
The compressed diagnostic keeps only the two empirical quantities required by the selective-ACPC logic. ATR estimates the high-tail term \(\Pr[D > \epsilon]\) that appears in the sampled-pool stability calibration, using the 90th percentile of normalized same-state action-conditioned rollout disagreement. SMPR estimates the discriminability countercondition by checking whether task-grounded near-boundary pairs remain separated beyond the same-state noisy radius. Thus low ATR without high SMPR is not interpreted as robustness, because collapse can also make clean and corrupted views agree.
```

### Numbering consequences

Removing the table will renumber all later tables. That is fine if all references use labels via `\Cref{...}`. Codex must still run:

```bash
rg -n "Table~|Table |tab:theory-metric-map|theory-metric" paper1/main.tex paper1/*.md tools
```

and remove or update any hard-coded table-number wording.

### Theory completeness

The theory section is **more than complete enough** for a diagnostic paper. It should not be expanded. If anything, it should be slimmed.

Keep:

- ACPC definition and discriminability countercondition;
- cost-drift proposition;
- top-1 fixed-candidate stability proposition/corollary;
- sampled-pool stability theorem;
- local Gaussian sensitivity proposition;
- collapse counterexample.

Consider trimming or moving to appendix:

- finite-sample calibration if space is tight;
- duplicated proof text in the appendix if it repeats main-text proof too closely;
- pseudo-metric material if it interrupts the empirical story.

Do **not** add more theory unless it directly repairs a reviewer-facing logical gap.

---

## 2.2 Table 2: noise cliff table

### Current issue

The current table includes:

- `drop` column;
- `obs+goal 0.08` column;
- only `obs 0.05` and `obs 0.08` among noisy observation-only settings.

This is not ideal because `drop` is redundant, `obs+goal` distracts from the primary clean-goal observation-noise endpoint, and the user wants a smoother evaluation-noise progression.

### Decision

**Revise the main table to show only observation-only evaluation noise at std 0, 0.03, 0.05, 0.08. Remove `drop`. Remove `obs+goal 0.08` from the main table.**

New intended columns:

```text
Task | eval σ=0 | eval σ=0.03 | eval σ=0.05 | eval σ=0.08
```

Caption should say:

```text
LeWM-base under observation-only Gaussian evaluation noise. Values are mean ± population std across training seeds 3072/3073/3074; each training-seed value averages evaluation seeds 42/43/44 with 100 trajectories per evaluation seed. The goal image is kept clean.
```

### Where should obs+goal eval go?

**Put obs+goal in the appendix, not in the legacy document.**

Reason:

- It is a real auxiliary stress condition, not a deprecated diagnostic.
- It helps justify why the main endpoint keeps the goal clean.
- But it should not compete with the main matched Gaussian observation-only endpoint.

Appendix placement:

```tex
\section{Auxiliary Observation+Goal Gaussian Stress}\label{sec:appendix-obs-goal}
```

Add a small table only if the data are cleanly available. Otherwise, add a short paragraph pointing to the artifact and keep it out of the PDF.

### Required cleanup

Codex must update all connected prose:

- Abstract: keep observation-only as primary; do not mention obs+goal unless necessary.
- Introduction: if using `loses X points`, compute it from the new table or keep it as text only.
- Study protocol: say obs+goal is auxiliary and appendix-only.
- Experiments subsection: remove the paragraph that interprets `obs+goal` as a main-table column.
- README arXiv tarball list: no change unless new appendix figure/table assets are added.

---

## 2.3 Table 3: sweep summary

### Current issue

The sweep summary table duplicates what the curve already communicates and invites point-best checkpoint interpretation, which the paper explicitly wants to avoid.

### Decision

**Remove Table 3 from the main paper. Keep the sweep curve.**

The curve is clearer for the intended claim: broad task-dependent plateaus rather than a universal point optimum. The exact point-best rows can be moved to appendix or kept in artifact docs.

Recommended main-text replacement:

```tex
Figure~\ref{fig:sweep} shows the full seed-3072 Gaussian sweep. The main qualitative pattern is a broad, task-dependent recovery plateau rather than a stable point-best training-noise level. Exact point-best rows are artifact-level bookkeeping rather than a ranking claim.
```

If the final paper needs exact rows, place a compact appendix table titled:

```text
Seed-3072 sweep endpoints and point-best rows, for reproducibility only.
```

Do not keep it in the main text.

---

## 2.4 Table 4: three-training-seed Gaussian replication

### Current issue

This table looks superficially similar to the sweep summary table, so it feels redundant. But it plays a different evidential role: it is the main three-training-seed replication evidence.

### Decision

**Do not delete Table 4. Keep it, but simplify it.**

This is the table that supports the “across three training seeds” claim. Removing it would weaken the paper more than it would improve readability.

Recommended compact columns:

```text
Task | base obs σ=0.08 | std0.08 obs σ=0.08 | gain | best obs σ=0.08 | std0.08 regret
```

Move `best std range` into a prose sentence or appendix. The current `best std range` column is useful internally but makes the table look like a checkpoint-selection report.

Recommended caption:

```text
Three-training-seed Gaussian endpoint for LeWM. Values are mean ± population std across training seeds 3072/3073/3074; each seed point averages evaluation seeds 42/43/44 with 100 trajectories per evaluation seed. The endpoint is observation-only Gaussian evaluation at σ=0.08 with a clean goal image.
```

Recommended prose:

```tex
The fixed std0.08 endpoint is within a few percentage points of each task's best obs-σ=0.08 row, supporting plateau wording rather than a universal optimum at std0.08.
```

---

## 2.5 Figures 2 and 3

### Current issue

The paper currently has:

- an ATR/SMPR diagnostic-plane figure;
- a qualitative PushT feature-neighborhood figure.

The diagnostic-plane figure mostly restates the compressed ATR/SMPR table. The feature-neighborhood figure is more conceptually useful, but the current rendering does not match the intended “old ACPC basin figure” style.

### Decision

**Keep only one of the two. Keep a redesigned ACPC-basin / t-SNE feature-cluster figure and remove the ATR/SMPR plane from the main paper.**

Reason:

- The ATR/SMPR table already carries the quantitative claim.
- A correct t-SNE feature-cluster figure adds visual intuition that the table cannot.
- The t-SNE figure must be explicitly qualitative and must not introduce extra paper-facing metrics.

### Intended Figure 3 replacement

Use the existing ACPC selective-contraction cluster machinery rather than the current PCA-style feature-neighborhood renderer.

Use this script path as the base:

```text
tools/paper1_selective_contraction.py
```

Not this as the primary renderer:

```text
tools/paper1_feature_neighborhood_figure.py
```

The current `paper1_feature_neighborhood_figure.py` can be deleted, archived, or retained only as a legacy qualitative renderer. It should not be the source of the paper-facing figure if the intent is the old ACPC basin / t-SNE style.

### Reuse existing points and plotting path

Use the cached feature arrays if present:

```text
/tmp/paper1_selective_contraction_cache
```

The relevant runner already supports:

- encoder and predictor features;
- repeated perturbation views;
- t-SNE projection;
- fixed-seed random anchors;
- sidecar point-count JSON;
- paper-facing output to `assets/paper1_figs`.

Baseline command shape:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mplconfig \
python -m tools.paper1_selective_contraction \
  --plot-clusters --plot-tasks PushT \
  --n-sequences 128 --cluster-anchor-count 16 \
  --view-stds 0.0 0.01 0.04 0.08 \
  --cluster-perturb-repeats 6 \
  --feature-cache-dir /tmp/paper1_selective_contraction_cache \
  --cluster-out-dir assets/paper1_figs \
  --cluster-envelope ellipse --cluster-envelope-coverage 0.90
```

### Required modification to the plotting script

The visible figure must not show old paper-facing metric names such as `R_E`, `R_F`, `r/NN`, `r<NN`, or `disjoint` as if they are current claims.

Modify the cluster rendering path so that:

1. The panel layout remains 2 × 2:
   - baseline encoder features;
   - baseline 8-step rollout predicted features;
   - std0.08 encoder features;
   - std0.08 8-step rollout predicted features.
2. The t-SNE points and selected anchors are reused from the existing cached feature arrays.
3. The only visible quantitative annotations are the two current theory-matched metrics:
   - ATR: base → std0.08;
   - SMPR: base → std0.08.
4. ATR/SMPR are read from:

```text
assets/paper1_data/compressed_metrics_summary_20260706.json
```

5. ATR/SMPR are not computed in t-SNE space.
6. Any old high-D cluster-isolation stats can remain in the sidecar JSON for audit, but not as visible paper-facing metrics.
7. The caption says:

```text
The t-SNE projection is qualitative. ATR and SMPR are the quantitative paper-facing diagnostics and are computed in the original rollout-readout diagnostic space.
```

### File and LaTeX target

Preferred new canonical output:

```text
assets/paper1_figs/fig_acpc_basin_tsne.png
```

Then update `paper1/main.tex`:

```tex
\includegraphics[width=\linewidth]{fig_acpc_basin_tsne.png}
\caption{Qualitative PushT ACPC-basin visualization...}
\label{fig:acpc-basin-tsne}
```

Remove the main-text reference to:

```text
fig_atr_smpr_plane.png
```

and remove it from the arXiv source copy list unless the figure remains referenced.

---

## 2.6 Introduction and section-opening structure

### Current issue

The Introduction begins immediately with `\subsection{Latent prediction is not a robustness definition}`. This is too abrupt. It looks like an outline rather than a paper narrative.

The `Contributions` subsection also feels abrupt because it uses `C1/C2/C3` paragraph blocks before the paper has fully built narrative momentum.

Several other sections also start directly with a subsection or paragraph label, especially Related Work and Study Protocol.

### Decision

**Rewrite the Introduction as a continuous narrative without subsections.**

Recommended structure:

1. Paragraph 1: JEPA latent prediction helps avoid pixel reconstruction but does not define robustness for control.
2. Paragraph 2: Control robustness should be checked after action-conditioned prediction, not only at the encoder.
3. Paragraph 3: Define the selective tension: same-state perturbations should agree; action-relevant distinct states should remain separated.
4. Paragraph 4: Controlled Gaussian intervention and empirical setup.
5. Paragraph 5: What the evidence shows: cliffs, recovery plateaus, ATR/SMPR movement.
6. Final paragraph or compact bullet list: contributions.

Use natural contribution prose instead of `C1 ---`, `C2 ---`, `C3 ---` labels, unless the venue style strongly prefers labeled contributions.

Recommended contribution wording:

```tex
This paper makes three contributions. First, it formulates visual robustness for JEPA world-model control as selective action-conditioned predictive consistency rather than encoder invariance. Second, it compresses the diagnostic into two theory-aligned readouts: ATR for same-state predictive tail risk and SMPR for task-grounded anti-collapse margins. Third, it provides a fixed-checkpoint Gaussian robustness study across four tasks and three LeWM training seeds, with bounded unseen-stressor checks that delimit rather than broaden the claim.
```

### Other section openings

Add a 2--3 sentence lead-in before the first subsection of:

- `Related Work`;
- `Study protocol`.

`ACPC` already has a useful opening paragraph. `Experiments` already has a useful opening paragraph. `Discussion` and `Conclusion` are acceptable.

---

## 3. Codex execution plan

## 3.1 Patch A: manuscript convergence in `paper1/main.tex`

Tasks:

1. Remove Table 1 (`tab:theory-metric-map`) and replace with paragraph text.
2. Rewrite Introduction as narrative prose without subsections.
3. Convert contributions from `C1/C2/C3` paragraphs into one compact contribution paragraph or short list.
4. Add opening lead paragraphs to Related Work and Study Protocol.
5. Revise the noise-cliff table:
   - remove `drop`;
   - remove `obs+goal 0.08`;
   - add observation-only eval std columns 0, 0.03, 0.05, 0.08;
   - update caption and prose.
6. Remove the sweep summary table from the main text.
7. Keep and compact the three-training-seed Gaussian endpoint table.
8. Remove the ATR/SMPR plane figure from main text.
9. Replace the current qualitative feature-neighborhood figure with the ACPC-basin t-SNE cluster figure.
10. Move obs+goal stress details to appendix or artifact note.
11. Check all labels and references after renumbering.

## 3.2 Patch B: data/artifact generation for revised Table 2

Create or adapt a script that aggregates LeWM-base observation-only eval noise at:

```text
σ = 0, 0.03, 0.05, 0.08
```

across training seeds:

```text
3072, 3073, 3074
```

Output:

```text
assets/paper1_data/base_noise_cliff_multistd_20260706.json
assets/paper1_data/base_noise_cliff_multistd_20260706.md
```

The MD should contain the exact table rows used by the manuscript.

If the manifests do not contain `obs σ=0.03`, Codex must not invent values. Instead:

- report available stds only; or
- add a TODO/error explaining the missing field;
- do not silently interpolate.

## 3.3 Patch C: ACPC-basin t-SNE figure

Modify `tools/paper1_selective_contraction.py` or add a thin wrapper script:

```text
tools/paper1_acpc_basin_tsne_figure.py
```

Preferred: modify the existing cluster path to accept:

```bash
--metric-summary assets/paper1_data/compressed_metrics_summary_20260706.json
--cluster-paper-facing
```

In paper-facing mode:

- hide legacy visible stats;
- use panel titles without `R_E` / `R_F`;
- annotate ATR and SMPR only;
- keep old stats in sidecar JSON only if needed;
- write `assets/paper1_figs/fig_acpc_basin_tsne.png`;
- write `assets/paper1_figs/fig_acpc_basin_tsne_point_counts.json`.

Do not commit `/tmp/paper1_selective_contraction_cache`.

## 3.4 Patch D: README / arXiv source-list cleanup

Update `paper1/docs/README.md`:

- remove `fig_atr_smpr_plane.png` from the arXiv source copy list if it is no longer referenced;
- replace `fig_feature_neighborhood_atr_smpr.png` with `fig_acpc_basin_tsne.png`;
- update the figure regeneration commands;
- state that the t-SNE figure uses cached feature arrays but the cache itself is not committed.

Update `tools/README_paper1.md`:

- mark `paper1_feature_neighborhood_figure.py` as legacy if it remains;
- document the new paper-facing ACPC-basin t-SNE command;
- explicitly say ATR/SMPR are the only visible paper-facing metrics on the qualitative figure.

## 3.5 Patch E: consistency and build checks

Run:

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
rg -n "Overfull|undefined references|Citation .* undefined|Reference .* undefined|Fatal error|Undefined control sequence" paper1/main.log || true
```

Then run grep checks:

```bash
rg -n "tab:theory-metric-map|theory-to-metric|fig:atr-smpr-plane|fig_atr_smpr_plane" paper1 tools assets || true
rg -n "obs\+goal 0\.08|drop" paper1/main.tex || true
rg -n "R_E|R_F|r/NN|disjoint" paper1/main.tex tools/paper1_selective_contraction.py tools/README_paper1.md || true
```

Expected outcome:

- no references to deleted table/figure labels in main text;
- no `drop` column in main noise-cliff table;
- no main-text `obs+goal 0.08` table column;
- no visible legacy metric names in the paper-facing figure caption or panel labels;
- clean PDF build.

---

## 4. Ready-to-paste Codex prompt

```text
You are editing qun-team/wm_exp on branch ag/dev. Please implement the manuscript convergence plan in paper1/docs/ARXIV_V1_REVIEW_AND_CODEX_PLAN_20260706.md.

Main goals:
1. Keep Paper1 as an arXiv v1 diagnostic / empirical analysis paper, not a method paper.
2. Remove Table 1 theory-to-metric mapping and replace it with prose.
3. Rewrite Introduction into a continuous narrative without subsections; convert C1/C2/C3 into compact contribution prose.
4. Revise the current noise-cliff table to observation-only eval std columns 0, 0.03, 0.05, 0.08; remove drop and obs+goal columns; move obs+goal to appendix or artifact note.
5. Remove the sweep summary table from the main text and rely on the sweep curve.
6. Keep but compact the three-training-seed Gaussian endpoint table.
7. Keep only one of the current Figures 2/3: remove the ATR/SMPR plane and replace the current feature-neighborhood figure with a proper ACPC-basin t-SNE cluster figure based on tools/paper1_selective_contraction.py.
8. The replacement figure should reuse existing cached feature arrays if available, show encoder and post-predictor features for baseline/std0.08, and visibly annotate only ATR and SMPR from assets/paper1_data/compressed_metrics_summary_20260706.json. Do not show R_E/R_F/rNN/disjoint as paper-facing metrics.
9. Update paper1/docs/README.md and tools/README_paper1.md so the build/arXiv figure list matches the new figure set.
10. Run consistency, LaTeX build, and grep checks. Do not invent missing numeric values; if σ=0.03 is absent from artifacts, stop and report the missing source rather than interpolating.

Acceptance criteria:
- main.tex builds.
- No deleted labels remain referenced.
- Main tables/figures tell one clean story: base fragility, sweep plateau curve, three-seed endpoint, ATR/SMPR diagnostics, bounded unseen scope.
- arXiv source list contains only actually referenced figures.
- The manuscript still states clearly that this is a diagnostic study, not a closed-loop robustness guarantee or new method.
```

---

## 5. Final strategic recommendation

For arXiv v1, the right move is:

```text
Post after this cleanup pass.
```

For v2/top-conference, the right move is:

```text
Use v1 to establish the diagnostic lens. Then add a new method only after it has mature closed-loop evidence, proper baselines, and a causal link to ATR/SMPR improvement.
```

The current version has enough material to claim a diagnostic contribution. It does not yet have enough to claim a new robust-control method.
