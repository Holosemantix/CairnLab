# Paper1 noise-response revision plan

> Target branch: `ag/dev`
> Target paper source: `paper1/main.tex`
> Intended executor: Codex / implementation agent
> Scope: plan only. Do not modify paper claims without regenerating the corresponding artifacts.
> Last updated: 2026-07-09

## 0. Executive decision

The proposed direction should be merged into paper1, but as a **diagnostic/profiling upgrade**, not as a new robust-training method claim.

Current paper1 already has the core ACPC frame, ATR/SMPR diagnostics, radius--margin certificate, local Gaussian sensitivity, full-sweep diagnostic dynamics, fixed-pool proxy audit, and matched-Gaussian limitations. The next useful upgrade is to make the multi-level noise scan an explicit contribution:

> Given a trained JEPA world-model checkpoint, a multi-level perturbation scan estimates its capability decay rate, effective robustness radius, and task-specific safe/effective training-noise budget. Different tasks have different noise budgets because training noise simultaneously reduces nuisance sensitivity and can erode task-relevant predictive margins.

This is stronger than the current wording "broad task-dependent recovery bands" because it turns the sweep into an actionable diagnostic protocol:

1. **Fixed-checkpoint noise-response profiling**: scan evaluation noise levels and measure behavior + ACPC diagnostics.
2. **Capability decay and effective radius**: summarize how fast score/diagnostics degrade as noise grows.
3. **Task-specific training-noise budget**: use train-noise x eval-noise response surfaces to identify safe/effective bands rather than a universal best std.
4. **Theory link**: explain the budget as a radius--margin trade-off: noise training lowers same-state predictive radius, but excessive noise can shrink discriminability/planner margins.

The revised paper should still be described as a controlled diagnostic study. Avoid claiming that ATR/SMPR automatically select the globally optimal checkpoint or that the certificate proves closed-loop robustness under adaptive CEM.

---

## 1. Current paper1 status to preserve

Read `paper1/main.tex` before editing. Preserve the following current structure and claims.

### 1.1 Main thesis already present

Current title and abstract frame ACPC as a no-retraining diagnostic for Gaussian-noise robustness in JEPA world models. The diagnostic combines ATR and SMPR, reports three LeWM training seeds across four tasks, and says recovered rows occupy low-ATR/high-SMPR regions with held-out onset errors within two grid steps.

Do not revert this framing. The noise-response upgrade should refine it, not replace it.

### 1.2 Existing theory already present

The paper already defines:

- `ACPC_H`: clean/noisy projected rollout distance under shared action sequence.
- Task-grounded discriminability guard.
- Same-state predictive radius `R_sigma` and quantile `r_{1-alpha}`.
- Planner margin `Delta_A`.
- Different-state margin `M_diff`.
- ATR as an empirical upper-tail radius estimate.
- SMPR as a margin-preservation / anti-collapse estimate.
- ACPC radius--margin certificate.
- Matched-perturbation diagnostic region.
- Local Gaussian sensitivity and local Gaussian radius quantile.
- Composed Jacobian sensitivity `||J_G J_E||_F^2` and local diagnostic `sigma*`.

Do not duplicate these statements. Add the new noise-response definitions as a layer above them.

### 1.3 Existing experimental evidence already present

The paper already reports:

- LeWM-base fragility under observation-only Gaussian noise.
- Three-training-seed Gaussian train-noise sweep.
- Broad task-dependent recovery bands.
- ATR/SMPR endpoint diagnostics.
- Full-sweep diagnostic dynamics.
- Held-out diagnostic validation.
- Radius--margin proxy overlap.
- Fixed-pool top-1 agreement audit.
- Bounded blur/resize severe-stressor checks.
- Appendix tables for full Gaussian sweep and radius--margin interpretation.

The new work should reorganize/extend this evidence into a more explicit **noise-response profile** rather than adding unrelated experiments.

---

## 2. Revised contribution statement

Update the contribution list in the Introduction to four contributions if space allows.

### Proposed C1: ACPC reframing

Visual robustness for JEPA world-model control should be evaluated after action-conditioned rollout, not only at the encoder output. Same-state visual perturbations should yield consistent predicted futures under the same action sequence, while action-, transition-, cost-, or goal-distinct cases must remain separable.

### Proposed C2: Radius--margin diagnostic theory

ATR and SMPR estimate two sides of a radius--margin diagnostic certificate. ATR measures the high-tail same-state perturbation tube after action-conditioned rollout. SMPR guards against collapse by testing task-grounded margin preservation. Local Gaussian sensitivity relates ATR to the composed encoder--predictor Jacobian.

### Proposed C3: Noise-response profiling and robustness radius

Given a fixed checkpoint, scanning several evaluation-noise levels gives a noise-response profile. From this profile the paper estimates:

- capability decay rate,
- score-based effective robustness radius,
- ACPC diagnostic radius,
- failure mode decomposition using ATR, SMPR, and fixed-pool cost-drift/margin proxies.

This is the new conceptual upgrade.

### Proposed C4: Task-specific training-noise budgets

Across four tasks and three LeWM training seeds, train-noise response surfaces show that different tasks require different safe/effective noise budgets. A single universal `std_max` should not be claimed. The useful claim is:

> Noise augmentation helps when it reduces same-state predictive radius faster than it erodes clean performance and task-relevant margins; the resulting budget is task-specific.

---

## 3. Theory additions

Add a compact subsection either at the end of `sec:acpc-sampled-gaussian` or immediately after the current `Matched-perturbation diagnostic region` definition.

Recommended title:

```latex
\subsection{Noise-response profiles and task-specific noise budgets}
```

### 3.1 Definition: fixed-checkpoint noise-response profile

For a trained checkpoint `theta`, define the response curves over evaluation noise `sigma`:

```latex
S_\theta(\sigma) = \text{closed-loop score under eval noise }\sigma,
```

```latex
A_\theta(\sigma) = \mathrm{ATR}_\theta(\sigma),
\qquad
M_\theta(\sigma) = \mathrm{SMPR}_\theta(\sigma).
```

The **noise-response profile** is:

```latex
\mathcal P_\theta
= \{(\sigma, S_\theta(\sigma), A_\theta(\sigma), M_\theta(\sigma)):\sigma\in\Sigma_{\rm eval}\}.
```

In the main paper, use the available eval grid first:

```latex
\Sigma_{\rm eval}=\{0,0.03,0.05,0.08\}.
```

If additional evals are later run, extend to a denser grid:

```latex
\{0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.10\}.
```

Be explicit that the current four-point grid gives a **coarse grid radius**, not a smooth certified radius.

### 3.2 Definition: capability decay rate

Define a score-normalized response:

```latex
P_\theta(\sigma)=\frac{S_\theta(\sigma)}{S_\theta(0)+\varepsilon}.
```

Finite-difference decay over `[sigma_a, sigma_b]`:

```latex
\widehat\kappa^S_\theta(\sigma_a,\sigma_b)
= -\frac{\log(S_\theta(\sigma_b)+\varepsilon)-\log(S_\theta(\sigma_a)+\varepsilon)}
{\sigma_b-\sigma_a}.
```

Use this as a descriptive fragility slope. Do not call it an asymptotic derivative unless a dense grid is available.

Recommended reported variants:

- `kappa_0_08`: score decay from eval sigma 0 to 0.08.
- `kappa_0_03`: early fragility from 0 to 0.03.
- Optional: diagnostic decay slopes for ATR and `1-SMPR`.

### 3.3 Definition: score-based effective robustness radius

For a retention threshold `tau`:

```latex
\rho^S_\tau(\theta)
=\max\{\sigma\in\Sigma_{\rm eval}: S_\theta(\sigma)\ge \tau S_\theta(0)\}.
```

Use `tau=0.8` as the main value. Appendix sensitivity should include `tau in {0.7,0.8,0.9}`.

With the current eval grid, report it as:

```latex
\widehat\rho^S_{\tau,\rm grid}(\theta)
```

and explain that it is lower-resolution than a continuous radius.

### 3.4 Definition: ACPC diagnostic radius

Use the existing radius--margin quantities:

```latex
\widehat\rho^{\rm ACPC}_{\gamma}(\theta)
=\max\{\sigma\in\Sigma_{\rm eval}:
\widehat Q_{0.90}(R_\sigma)\le \widehat r_{\rm margin},
\;\widehat{\rm SMPR}_\theta(\sigma)\ge \gamma\}.
```

If true planner-margin traces are unavailable, define a proxy variant instead:

```latex
\widehat\rho^{\rm proxy}_{\gamma}(\theta)
=\max\{\sigma\in\Sigma_{\rm eval}:
\widehat\Gamma_{\theta,\sigma}>0,
\;\widehat{\rm SMPR}_\theta(\sigma)\ge \gamma\},
```

where the current main text already defines:

```latex
\widehat\Gamma_{\rho,0.08}
=\widehat Q_{0.50}(\Delta_{\mathcal A})
-2\widehat Q_{0.90}(|C_h-C_{\tilde h}|).
```

Important wording:

> If only retained summary statistics are available, `rho_proxy` is a mechanism proxy, not a calibrated planner-margin certificate.

### 3.5 Corollary: task-specific noise budget

Add a short corollary after the local Gaussian quantile result.

Suggested statement:

```latex
\begin{corollary}[Local task-specific noise budget]
Under the local Gaussian linearization, suppose
Q_{1-\alpha}(R_\sigma)\approx \sigma c_\theta
for a checkpoint \theta and let m_\theta denote the relevant planner or task-discriminability margin. A sufficient local diagnostic condition for fixed-pool stability is
\sigma c_\theta < m_\theta/(2L_J).
Thus the local diagnostic radius scales as
\rho^{\rm diag}(\theta)\approx m_\theta/(2L_J c_\theta).
For a training-noise family \theta_\rho, input-side noise augmentation is useful only when it reduces c_{\theta_\rho} faster than it erodes m_{\theta_\rho} and clean performance. Hence the safe/effective training-noise budget is task-dependent.
\end{corollary}
```

Then immediately add the caveat:

> This corollary is a diagnostic orientation for the matched perturbation family. It does not prove an optimal augmentation level, and it does not cover adaptive CEM, repeated replanning, or environment-feedback stability.

### 3.6 Training-noise budget selection rule

Define the descriptive budget set for a task `T`:

```latex
\mathcal B_T(\tau,\delta,\gamma)
=\{\rho:\widehat\rho^S_\tau(\theta_\rho)\text{ is maximal or near-maximal},
S_{\theta_\rho}(0)\ge S_{\theta_0}(0)-\delta,
\widehat{\rm SMPR}_{\theta_\rho}\ge\gamma\}.
```

Do not force a single optimum. Report a band:

```latex
\rho^{\rm safe}_{T,\rm start} \;\text{to}\; \rho^{\rm safe}_{T,\rm end}.
```

Recommended parameters for main text:

- `tau = 0.8` score retention.
- `delta = 5` clean-score points.
- `gamma = 0.9` SMPR or task-specific high-SMPR criterion.

Appendix sensitivity:

- `tau in {0.7,0.8,0.9}`.
- `delta in {3,5,10}`.
- `gamma in {0.8,0.9,0.95}` if SMPR summaries support this.

---

## 4. Main-text restructuring plan

### 4.1 Abstract

Current abstract already says no-noise checkpoints are fragile, noise training gives broad recovery bands, and recovered rows occupy low-ATR/high-SMPR regions.

Add one sentence after the recovery-band sentence:

> We further treat the Gaussian sweep as a noise-response profiling protocol: multi-level perturbation scans estimate capability decay rates, coarse effective robustness radii, and task-specific safe/effective training-noise budgets.

Keep the final sentence bounded:

> These results support ATR/SMPR and noise-response profiles as bounded diagnostics for matched-Gaussian recovery bands in fixed JEPA world-model checkpoints.

### 4.2 Introduction

Add a paragraph after the current observation that evidence shows broad matched-Gaussian recovery bands.

Proposed content:

> The same sweep also exposes a practical question: if a fixed checkpoint degrades under visual perturbation, how fast does it degrade, and what training-noise budget is sufficient without damaging clean task-relevant margins? We therefore read the train-noise/eval-noise grid as a dose-response surface rather than as a leaderboard. For each checkpoint, evaluation-noise scans define a capability decay rate and a coarse effective robustness radius. Across the training-noise family, these radii reveal task-specific safe/effective noise budgets.

Then update contribution list as in Section 2 above.

### 4.3 Related Work

Do not expand the related work heavily unless a new external literature audit is performed. The current related work already covers JEPA, robustness augmentation, bisimulation/value-aware representations, group-action metrics, and ReOI.

Recommended minimal edit:

- Add 2-3 sentences at the end of the robustness subsection saying this paper does not propose another augmentation method; it uses controlled perturbation response curves to diagnose fixed JEPA world-model checkpoints and to expose task-specific noise budgets.
- Use only already-present citations unless Codex performs a fresh literature audit.
- Do not introduce new references from memory.

### 4.4 Theory section

Add the definitions and corollary in Section 3 of this plan. Keep the subsection concise in main text, and move derivation details to appendix.

Target text economy:

- Main text: one definition block, one corollary, one explanatory paragraph.
- Appendix: full computation details and threshold variants.

### 4.5 Study protocol

Add a sentence that the Gaussian grid is used in two ways:

1. behavioral endpoint evaluation at `sigma_eval=0.08`, and
2. multi-level response profiling over `sigma_eval in {0,0.03,0.05,0.08}`.

If the data source already has more eval sigmas, use the denser grid and update all formulas/figures accordingly.

### 4.6 Experiments section

Add a new subsection immediately after `Noise augmentation supplies recovered checkpoints for diagnosis` and before `ATR/SMPR selective-ACPC diagnostics`.

Recommended title:

```latex
\subsection{Noise-response profiles reveal task-specific noise budgets}
```

Core claims to write:

1. Base checkpoints have task-dependent decay rates under eval noise.
2. Noise-trained checkpoints expand the score-based radius, but not identically across tasks.
3. The useful region is a band, not a point optimum.
4. Different tasks show different minimum training-noise levels needed for recovery.
5. Over-noising should be treated as a possible margin/clean-performance risk, even if the current grid does not always show severe over-noising up to 0.08.

Do not claim severe over-noising unless the score/SMPR tables actually show it. Use cautious wording:

> The present sweep identifies safe/effective bands within the tested range; stronger train noise would be needed to map the full over-noising side of the dose-response curve.

---

## 5. Figures and tables to add

### Main Figure A: train-noise x eval-noise score surface

File name:

```text
assets/paper1_figs/fig_noise_response_surface.png
```

Suggested LaTeX label:

```latex
\label{fig:noise-response-surface}
```

Content:

- 4 panels, one per task.
- x-axis: eval noise `sigma_eval`.
- y-axis: train noise `std_max`.
- color: closed-loop success score.
- overlay or annotation: recovery-band start and proxy-positive start if available.

Use the existing full sweep table source. Current paper already has eval columns `0, 0.03, 0.05, 0.08`; this is enough for a coarse surface. If denser eval data exists, use it.

Caption should say:

> Score surfaces show that matched-Gaussian robustness is a task-specific dose-response phenomenon: the train-noise level needed to recover high eval-noise performance differs across tasks, and the grid forms broad bands rather than universal point optima.

### Main Figure B: robustness radius and decay summary

File name:

```text
assets/paper1_figs/fig_noise_response_radius_decay.png
```

Suggested LaTeX label:

```latex
\label{fig:noise-response-radius-decay}
```

Content options:

- Bar/line plot per task showing:
  - base checkpoint `rho^S_0.8`,
  - best/selected noise-trained checkpoint `rho^S_0.8`,
  - base decay slope `kappa_0_08`,
  - selected checkpoint decay slope.

If one figure gets too dense, split:

- Main text: score radius only.
- Appendix: decay slopes.

Caption should emphasize coarse grid radius.

### Main Table A: task-specific noise budget summary

File name if generated as LaTeX input:

```text
paper1/tables/table_noise_budget_summary.tex
```

Columns:

```text
Task
base score radius rho^S_0.8
base decay kappa_0_08
behavioral recovery band
proxy/diagnostic-positive band
selected safe/effective training-noise band
interpretation
```

Populate from generated artifacts. Do not hand-invent values. Use `mean ± population std across training seeds` if reporting numeric scores; for bands, report grid intervals.

Possible interpretations by current evidence direction:

- TwoRoom: early recovery; likely large safe band.
- PushT: sharp base fragility; needs moderate train noise; strong recovery after onset.
- Reacher: strong recovery but cost-margin proxy conservative at low noise levels.
- Cube: boundary case; diagnostic improves but behavioral gain weaker / plateau less decisive.

Use exact wording only after checking generated numbers.

### Main Figure C: optional diagnostic response overlay

If space allows, add or adapt existing `fig_full_sweep_diagnostics` rather than adding a redundant figure.

Desired content:

- x-axis: training `std_max`.
- left y-axis: eval score at `sigma_eval=0.08`.
- right y-axis or normalized overlay: ATR q90, `1-SMPR`, fixed-pool top-1 flip, proxy gap.
- mark behavioral recovery band and diagnostic/proxy-positive band.

If the current `fig_full_sweep_diagnostics` already does this, just update caption/text to explicitly call it a noise-budget diagnostic plot.

### Appendix Figure Set

Add per-task supplemental figures if main text is too crowded:

```text
assets/paper1_figs/appendix_noise_response_surface_tworoom.png
assets/paper1_figs/appendix_noise_response_surface_pusht.png
assets/paper1_figs/appendix_noise_response_surface_reacher.png
assets/paper1_figs/appendix_noise_response_surface_cube.png
```

Each appendix figure should show:

- score vs eval noise for each train-noise level,
- optional ATR/SMPR if recomputed at multiple eval-noise levels,
- clean retention line.

---

## 6. Experiment / artifact plan

### 6.1 Minimum experiment: use existing data

Minimum version requires no new closed-loop runs if the existing artifacts contain the full Gaussian eval columns in the appendix table.

Required computation:

1. Load three-training-seed Gaussian sweep scores.
2. For each task and training noise `rho`, compute:
   - clean score `S_theta(0)`,
   - high-noise endpoint `S_theta(0.08)`,
   - `rho^S_0.8` over eval grid,
   - decay slope `kappa_0_08`,
   - recovery-band label using existing paper definition,
   - clean-retention label using 5-point tolerance.
3. Join available diagnostic summaries:
   - ATR q90,
   - SMPR,
   - fixed-pool top1 agreement,
   - cost-margin proxy gap.
4. Generate score surface, radius/decay summary, and noise-budget summary table.

This minimum version supports the new claim as a **coarse grid diagnostic**.

### 6.2 Stronger experiment: denser eval-noise scan

If resources allow, add eval noise levels:

```text
0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10
```

Use the same evaluation seeds 42/43/44 and 100 trajectories/seed. This would let the paper report smoother decay slopes and more credible radii.

Recommended priority if compute is limited:

1. base checkpoint and selected recovered endpoint per task,
2. behavioral onset checkpoint per task,
3. all 9 train-noise checkpoints only if budget allows.

### 6.3 Strongest version: diagnostic-only prospective selection

This would most strongly support the "training guidance" claim.

Protocol:

1. Use only diagnostic summaries from seed 3072 to select a recommended safe/effective train-noise band for each task.
2. Do not use seed 3073/3074 closed-loop scores during selection.
3. Validate whether the selected band overlaps high-score recovery bands on seeds 3073/3074.

Expected claim:

> Diagnostic-only noise-budget selection lands inside or near the replicated behavioral recovery band.

Caveat:

> This is a band-level validation, not exact checkpoint ranking.

If current artifacts already include held-out seed/task audits, reuse them and only reframe as noise-budget validation.

---

## 7. Scripts / code changes for Codex

Before adding new scripts, inspect existing tools:

```bash
grep -R "three_seed_gaussian\|fig_full_sweep\|radius_margin\|ATR\|SMPR\|fixed_pool" -n tools paper1 assets | head -200
```

Likely files to inspect first:

```text
tools/paper1_three_seed_gaussian_sweep.py
tools/paper1_figs.py
tools/paper1_selective_contraction.py
paper1/tables/*.tex
assets/paper1_data/*.json
assets/paper1_data/*.csv
```

### 7.1 New script option

Add:

```text
tools/paper1_noise_response_profile.py
```

Responsibilities:

- load existing Gaussian sweep score summaries,
- compute `rho^S_tau`, decay slopes, clean retention, recovery band,
- join diagnostics if available,
- emit machine-readable summary:

```text
assets/paper1_data/noise_response_profile_summary.json
assets/paper1_data/noise_budget_summary.csv
```

- emit LaTeX table:

```text
paper1/tables/table_noise_budget_summary.tex
```

- emit figures:

```text
assets/paper1_figs/fig_noise_response_surface.png
assets/paper1_figs/fig_noise_response_radius_decay.png
```

### 7.2 Or extend existing figure script

If `tools/paper1_figs.py` already owns all paper figures, prefer adding functions there:

```python
make_noise_response_surface(...)
make_noise_response_radius_decay(...)
write_noise_budget_summary_table(...)
```

Keep the data computation in a separate helper module if the figure script is already large.

### 7.3 Reproducibility metadata

Every generated artifact should include:

- input file paths,
- git commit if available,
- task list,
- train seeds,
- eval seeds,
- eval trajectory count,
- thresholds used: `tau`, `delta`, `gamma`,
- whether radius is grid/coarse or interpolated.

If writing JSON, include:

```json
{
  "tau_score_retention": 0.8,
  "clean_tolerance_points": 5,
  "eval_noise_grid": [0.0, 0.03, 0.05, 0.08],
  "train_noise_grid": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
  "training_seeds": [3072, 3073, 3074],
  "evaluation_seeds": [42, 43, 44],
  "trajectories_per_eval_seed": 100,
  "radius_type": "coarse_grid"
}
```

---

## 8. Appendix additions

### Appendix A: Noise-response profile computation

Add after current experimental details.

Content:

- exact eval noise grid,
- score normalization,
- decay slope definition,
- `rho^S_tau` definition,
- interpolation policy if used,
- threshold values,
- why grid radius is coarse.

### Appendix B: Threshold sensitivity

Add a table:

```text
paper1/tables/table_noise_budget_threshold_sensitivity.tex
```

Rows:

```text
Task | tau | clean tolerance | behavioral band | selected/proxy band | start error | comment
```

Use:

- tau = 0.7 / 0.8 / 0.9,
- clean tolerance = 3 / 5 / 10 points.

This prevents the reviewer complaint that the budget is arbitrary.

### Appendix C: Full per-task response curves

Add per-task figures or one multi-page figure with:

- score vs eval noise for all train-noise levels,
- optional clean retention line,
- optional selected band shading.

### Appendix D: Diagnostic failure cases and limits

Move/extend the current Reacher and Cube caveats here:

- Reacher: fixed-pool cost-margin proxy is conservative at low noise levels even though top-1 agreement improves.
- Cube: ATR/SMPR improve strongly but closed-loop Gaussian recovery is weaker; closed-loop behavior remains authority.
- Over-noising: current train grid may not be heavy enough to show the full harmful side for every task; do not claim the optimum is found unless larger std levels are evaluated.

### Appendix E: Algorithm box

Add pseudocode:

```text
Algorithm: ACPC noise-response profiling
Input: checkpoint theta, eval noise grid Sigma_eval, diagnostic dataset D, thresholds tau/delta/gamma
1. For each sigma in Sigma_eval, run closed-loop eval or load retained score S_theta(sigma).
2. For each sigma, compute paired clean/noisy ACPC radius samples R_sigma.
3. Compute ATR = Q_0.90(R_sigma), SMPR, and fixed-pool cost-drift/margin proxy when available.
4. Compute score retention P_theta(sigma), decay slopes, rho^S_tau, and rho^ACPC/proxy.
5. For train-noise family theta_rho, report safe/effective band satisfying clean retention and diagnostic/score radius criteria.
Output: profile curves, radius/decay table, safe/effective training-noise band.
```

---

## 9. Main text wording guardrails

### Allowed strong wording

Use these confidently:

- "Noise-response profiling turns the Gaussian sweep into a diagnostic of capability decay and effective robustness radius."
- "The safe/effective training-noise budget is task-specific."
- "Noise training improves robustness when it contracts same-state predictive radius without eroding task-relevant margins."
- "ATR and SMPR provide a radius--margin decomposition of the response profile."
- "The present evidence supports matched-Gaussian diagnostic guidance, not general corruption transfer."

### Avoid overclaiming

Do not write:

- "ATR/SMPR select the optimal checkpoint."
- "We prove closed-loop robustness."
- "The training noise optimum is universal."
- "Adding more noise always helps until a threshold."
- "The radius is continuous/certified" if using only four eval-noise levels.
- "The fixed-pool proxy is a planner-margin certificate" unless raw lower-tail margin and upper-tail drift data are recomputed.
- "Blur/resize transfer is solved."
- "This is a new training algorithm" unless ACPC regularization experiments are added.

### Best paper-level phrasing

Use:

> We estimate a coarse effective radius on the retained evaluation grid.

Use:

> The diagnostic identifies a safe/effective training-noise band.

Avoid:

> We find the optimal noise level.

Use:

> Task-specific budgets arise because training noise changes both nuisance sensitivity and task margins.

Avoid:

> Task difficulty determines the optimal noise level.

---

## 10. Reviewer-facing argument

### Likely reviewer question 1

"Is this just a hyperparameter sweep?"

Answer in paper:

No. A sweep would report best `std_max`. The revised paper extracts a perturbation-response profile: decay slope, effective radius, diagnostic radius, and radius--margin failure components. The contribution is the diagnostic decomposition and task-specific budget interpretation, not leaderboard tuning.

### Likely reviewer question 2

"Why not just use closed-loop score to choose noise?"

Answer:

Closed-loop score is the behavioral authority but expensive and not mechanistic. ATR/SMPR localize whether recovery comes from same-state predictive contraction while retaining task-grounded separability. The diagnostic does not replace closed-loop evaluation; it explains and helps narrow safe/effective noise ranges.

### Likely reviewer question 3

"Does the radius guarantee robustness?"

Answer:

Only under the fixed-candidate, matched-perturbation diagnostic assumptions. The paper explicitly does not prove adaptive CEM or repeated-replanning closed-loop guarantees.

### Likely reviewer question 4

"Why do tasks need different noise budgets?"

Answer:

Because the local radius--margin ratio differs by task. Some tasks have large planner/discriminability margins or contract nuisance directions easily; others have contact-sensitive or pose-sensitive boundaries where visual smoothing can harm task-relevant distinctions.

### Likely reviewer question 5

"Where is over-noising shown?"

Answer:

If the current grid up to 0.08 does not show clear over-noising, state it honestly: the current grid identifies the safe/effective band within the tested range, while heavier train-noise levels are needed to fully map the harmful side. Do not invent an over-noising result.

---

## 11. Concrete edit checklist for Codex

### Step 1: Inspect current source

```bash
git checkout ag/dev
git pull
sed -n '1,220p' paper1/main.tex
sed -n '220,520p' paper1/main.tex
sed -n '520,820p' paper1/main.tex
ls paper1/tables
ls assets/paper1_figs
ls assets/paper1_data
```

### Step 2: Locate data generation paths

```bash
grep -R "fig2_sweep\|appendix-gaussian-sweep\|3072\|3073\|3074\|stdmax\|sigma=0.08" -n tools paper1 assets | head -300
```

### Step 3: Implement response-profile computation

Add or extend script to compute:

- `score_retention = S(sigma_eval)/S(0)`
- `rho_score_tau_grid`
- `kappa_score_0_08`
- `kappa_score_0_03`
- `clean_retention_ok`
- `behavioral_recovery_band`
- `noise_budget_band`
- optional join: ATR, SMPR, top1 agree, proxy gap

### Step 4: Generate artifacts

Expected outputs:

```text
assets/paper1_data/noise_response_profile_summary.json
assets/paper1_data/noise_budget_summary.csv
paper1/tables/table_noise_budget_summary.tex
paper1/tables/table_noise_budget_threshold_sensitivity.tex
assets/paper1_figs/fig_noise_response_surface.png
assets/paper1_figs/fig_noise_response_radius_decay.png
```

### Step 5: Modify `paper1/main.tex`

Edit in this order:

1. Abstract: add noise-response profiling sentence.
2. Introduction: add paragraph and update contribution list.
3. Theory: add noise-response profile definitions and local task-specific budget corollary.
4. Study protocol: mention multi-level response profiling.
5. Experiments: add subsection after Gaussian sweep.
6. Figures/tables: include new surface figure and budget summary table.
7. Discussion: add one sentence that the estimated radius is grid/coarse and matched-stressor only.
8. Appendix: add computation details, threshold sensitivity, full response curves if generated.

### Step 6: Build and lint

```bash
cd paper1
bash build.sh --clean
bash build.sh
```

Check:

```bash
grep -R "??" -n main.tex tables || true
grep -R "TODO\|TBD\|PLACEHOLDER" -n main.tex tables docs || true
```

If using generated tables, verify every `\input{tables/...}` file exists.

### Step 7: Commit

```bash
git status
git add paper1/main.tex paper1/tables assets/paper1_figs assets/paper1_data tools paper1/docs/NOISE_RESPONSE_REVISION_PLAN.md
git commit -m "paper1: add noise-response profiling plan and artifacts"
git push origin ag/dev
```

---

## 12. Desired final paper narrative after revision

The paper should read as follows:

1. Latent prediction is not a robustness definition for control.
2. Control robustness should be evaluated after action-conditioned rollout.
3. ACPC requires same-state predictive consistency plus different-state/action-relevant separability.
4. Gaussian noise is a controlled matched perturbation probe.
5. No-noise LeWM checkpoints show task-dependent visual fragility.
6. Noise training recovers broad matched-Gaussian bands.
7. Multi-level noise-response profiling estimates capability decay and coarse robustness radius.
8. Different tasks have different safe/effective train-noise budgets.
9. ATR/SMPR and fixed-pool proxies explain these budgets through a radius--margin mechanism.
10. The scope remains diagnostic: fixed checkpoints, matched Gaussian stressor, fixed-pool approximation, no closed-loop theorem, no method-superiority claim.

---

## 13. Minimal acceptance criteria

The revision is acceptable if it adds:

- one theory subsection with response profile, radius, decay, and budget definitions;
- one main experiment subsection on task-specific noise-response budgets;
- one score surface figure or equivalent table;
- one budget summary table;
- one appendix subsection explaining computations and threshold sensitivity;
- clear caveats that the radius is coarse/grid-based and matched-Gaussian only.

The revision is strong if it additionally adds:

- denser eval-noise scan;
- diagnostic-only held-out seed budget selection;
- ATR/SMPR response curves across eval noise, not only endpoint;
- sample-level q10/q95 fixed-pool certificate recomputation.

The revision should not proceed if:

- the figures cannot be regenerated from committed artifacts;
- clean/eval seed protocol is inconsistent with the existing paper;
- the text claims over-noising harm without data;
- the text claims planner-margin certification from retained q50/q90 summaries alone.

---

## 14. One-paragraph version for the paper

If space is tight, insert this compressed version into the experiments transition:

> Beyond endpoint recovery, the Gaussian grid gives a perturbation-response profile for each fixed checkpoint. For evaluation noise level `sigma`, we track closed-loop score, ATR, SMPR, and available fixed-pool cost-margin proxies. This profile yields a coarse score-retention radius and a finite-difference capability-decay rate on the retained evaluation grid. Reading the train-noise family through these profiles shows that matched-Gaussian robustness is governed by task-specific noise budgets: noise training is beneficial when it contracts same-state action-conditioned predictive radius without eroding clean performance or task-grounded margins. We therefore report safe/effective bands rather than a universal optimal `std_max`.

---

## 15. Final recommendation

Do this revision. It increases paper weight because it adds a practical and theoretical layer between diagnostics and training design:

```text
multi-level perturbation scan
  -> capability decay rate
  -> effective robustness radius
  -> task-specific training-noise budget
  -> radius--margin interpretation
```

This keeps paper1 within its safest identity: a controlled diagnostic paper for fixed JEPA world-model checkpoints under matched Gaussian perturbations, while making the diagnostics visibly useful for deciding how much noise augmentation is safe and effective.
