# Paper1 fixed-checkpoint noise-response revision plan

> Target branch: `ag/dev`  
> Target paper source: `paper1/main.tex`  
> Intended executor: Codex / implementation agent  
> Scope: planning document only. Do not modify paper claims without regenerating the corresponding artifacts.  
> Last updated: 2026-07-09  
> Status: **V1/V2 split. V1 prioritizes fixed-checkpoint capability decay and robustness radius. Training-noise budget selection is moved to V2.**

---

## 0. Executive decision

The previous version of this plan pushed two ideas together:

1. fixed-checkpoint noise-response profiling, and
2. task-specific training-noise budget selection.

After reconsidering the experimental dependency, they should be separated.

**V1 / current paper priority:** add a bounded, low-risk diagnostic upgrade:

> Given a trained checkpoint, scan several evaluation-noise levels and estimate its capability decay rate, coarse score-retention radius, and ACPC/radius--margin diagnostic radius.

This is directly aligned with current paper1. It uses the existing ACPC/ATR/SMPR theory and can be supported by the retained evaluation-noise grid. It strengthens the diagnostic contribution without requiring new training runs.

**V2 / later paper or revision priority:** estimate task-specific safe/effective **training-noise budgets**. This requires a前置实验: for each task, train or evaluate enough higher-noise checkpoints to observe where training noise starts hurting clean performance, task-relevant margin, or closed-loop behavior. Without that harmful-side boundary, claiming a principled training-noise budget is premature.

The revised V1 claim should therefore be:

> Noise-response profiling estimates how fast a fixed JEPA world-model checkpoint loses capability as evaluation noise increases, and gives a coarse matched-Gaussian robustness radius. The result can motivate future noise-augmentation choices, but does not yet select an optimal training-noise strength.

Do **not** make the current paper depend on proving when training noise becomes harmful.

---

## 1. Why training-noise budget selection should move to V2

### 1.1 Missing前置实验

To say "how much training noise should be added", the paper needs to know three task-dependent boundaries:

1. **benefit onset**: the smallest train-noise level that recovers evaluation-noise performance;
2. **clean/margin damage onset**: the train-noise level where clean score, SMPR, planner margin, or task-relevant discriminability starts to degrade;
3. **over-noising frontier**: the region where more noise no longer improves robustness and starts damaging task-relevant behavior.

Current paper1 already supports benefit/recovery bands under the tested Gaussian sweep, but it does not necessarily map the harmful side. If the sweep only goes to `std_max=0.08`, and all or most tasks remain acceptable there, then the data identifies a safe tested range, not a full budget optimum.

### 1.2 Reviewer risk if kept as a main claim

If the paper claims task-specific training-noise budget selection now, likely reviewer questions are:

- Where is the evidence that heavier noise hurts each task?
- Did the grid extend far enough to observe over-noising?
- Is the selected noise level chosen from closed-loop score, diagnostics, or both?
- Does ATR/SMPR predict the training-noise budget prospectively, or only explain a completed sweep?
- Is this still a diagnostic paper, or now a hyperparameter-selection method paper?

Those questions would force a heavier V2 experiment. For V1, avoid this burden.

### 1.3 Better V1 positioning

V1 should say:

- We can estimate **current checkpoint robustness radius**.
- We can estimate **capability decay rate** under evaluation noise.
- We can decompose failure using ATR/SMPR and fixed-pool margin proxies.
- We can show task heterogeneity in noise response.
- We can state that this suggests a future route for training-noise budget selection.

V1 should **not** say:

- We find the optimal training-noise level.
- We know exactly how much noise to add during training.
- We prove that one `std_max` is safe or optimal for a task.
- We have mapped the over-noising side unless new heavier-noise data is generated.

---

## 2. Revised contribution structure

Keep paper1 as a diagnostic paper. If the introduction has room, use three contributions plus one future implication, not four equal contributions.

### C1 — ACPC reframing

Visual robustness for JEPA world-model control should be evaluated after action-conditioned rollout, not only at the encoder output. Same-state visual perturbations should yield consistent predicted futures under the same action sequence, while action-, transition-, cost-, or goal-distinct cases remain separable.

### C2 — Radius--margin diagnostic theory

ATR and SMPR estimate two sides of a radius--margin diagnostic certificate. ATR measures the high-tail same-state perturbation tube after action-conditioned rollout. SMPR guards against collapse by testing task-grounded margin preservation. Local Gaussian sensitivity relates ATR to the composed encoder--predictor Jacobian.

### C3 — Fixed-checkpoint noise-response profiling

Given a fixed trained checkpoint, evaluation-noise scans define:

- capability decay rate,
- coarse score-retention radius,
- ACPC diagnostic radius or proxy radius,
- failure-mode decomposition through ATR, SMPR, fixed-pool top-1 agreement, and cost-drift/margin proxies.

This is the main new upgrade for V1.

### Future implication — task-specific training-noise budgets

Different tasks likely require different training-noise ranges because noise training changes both nuisance sensitivity and task-relevant margins. However, selecting a principled training-noise budget requires an explicit over-noising-frontier experiment. Keep this as V2 / future work unless the additional data is produced.

---

## 3. Theory changes for V1

Add a compact subsection after the existing matched-perturbation diagnostic region or after the local Gaussian sensitivity discussion.

Recommended title:

```latex
\subsection{Fixed-checkpoint noise-response profiles}
```

### 3.1 Definition: fixed-checkpoint profile

For a trained checkpoint `theta`, define response curves over evaluation noise `sigma`:

```latex
S_\theta(\sigma)=\text{closed-loop score under evaluation noise }\sigma,
```

```latex
A_\theta(\sigma)=\mathrm{ATR}_\theta(\sigma),
\qquad
M_\theta(\sigma)=\mathrm{SMPR}_\theta(\sigma).
```

The profile is:

```latex
\mathcal P_\theta
=\{(\sigma,S_\theta(\sigma),A_\theta(\sigma),M_\theta(\sigma)):
\sigma\in\Sigma_{\rm eval}\}.
```

For V1, the retained grid can be:

```latex
\Sigma_{\rm eval}=\{0,0.03,0.05,0.08\}.
```

If additional evaluations are later available, use a denser grid. Until then, always write **coarse grid radius**, not continuous radius.

### 3.2 Definition: capability decay rate

Score-normalized response:

```latex
P_\theta(\sigma)=\frac{S_\theta(\sigma)}{S_\theta(0)+\varepsilon}.
```

Finite-difference decay:

```latex
\widehat\kappa^S_\theta(\sigma_a,\sigma_b)
= -\frac{\log(S_\theta(\sigma_b)+\varepsilon)-\log(S_\theta(\sigma_a)+\varepsilon)}
{\sigma_b-\sigma_a}.
```

Use this as a descriptive fragility slope. Report:

- `kappa_0_08`: full retained-grid decay from 0 to 0.08;
- `kappa_0_03`: early decay from 0 to 0.03;
- optional diagnostic slopes for ATR and `1-SMPR` if the diagnostics are recomputed across eval noise levels.

### 3.3 Definition: score-retention radius

For retention threshold `tau`:

```latex
\widehat\rho^S_{\tau,\rm grid}(\theta)
=\max\{\sigma\in\Sigma_{\rm eval}:S_\theta(\sigma)\ge \tau S_\theta(0)\}.
```

Use `tau=0.8` in the main text. Appendix sensitivity can include `tau in {0.7,0.8,0.9}`.

This is the cleanest V1 robustness-radius object because it only requires the current checkpoint and evaluation-noise scan.

### 3.4 Definition: ACPC diagnostic radius

If ATR/SMPR are available at multiple evaluation-noise levels, define:

```latex
\widehat\rho^{\rm ACPC}_{\gamma,\rm grid}(\theta)
=\max\{\sigma\in\Sigma_{\rm eval}:
\widehat Q_{0.90}(R_\sigma)\le \widehat r_{\rm margin},
\;\widehat{\rm SMPR}_\theta(\sigma)\ge\gamma\}.
```

If true planner-margin traces are unavailable, use the retained proxy only:

```latex
\widehat\rho^{\rm proxy}_{\gamma,\rm grid}(\theta)
=\max\{\sigma\in\Sigma_{\rm eval}:
\widehat\Gamma_{\theta,\sigma}>0,
\;\widehat{\rm SMPR}_\theta(\sigma)\ge\gamma\}.
```

Important caveat:

> If only retained summary statistics are available, `rho_proxy` is a mechanism proxy, not a calibrated planner-margin certificate.

### 3.5 Local radius interpretation

Keep a local diagnostic-radius corollary, but remove the training-noise-budget conclusion from the main theorem/corollary.

Suggested statement:

```latex
\begin{corollary}[Local fixed-checkpoint diagnostic radius]
Under the local Gaussian linearization, suppose
Q_{1-\alpha}(R_\sigma)\approx \sigma c_\theta
for a fixed checkpoint \theta and let m_\theta denote the relevant planner or task-discriminability margin. A sufficient local diagnostic condition for fixed-pool stability is
\sigma c_\theta < m_\theta/(2L_J).
Thus the local diagnostic radius scales as
\rho^{\rm diag}(\theta)\approx m_\theta/(2L_J c_\theta).
\end{corollary}
```

Then add:

> This explains why different checkpoints and tasks can have different evaluation-noise radii. Turning this into a training-noise budget requires an additional learning-family experiment that maps how training noise changes both `c_theta` and `m_theta`.

This keeps the theory useful without forcing the V2 experiment into V1.

---

## 4. V1 experimental plan: fixed-checkpoint decay and radius

### 4.1 Minimum V1 using existing data

No new training runs are required if the existing artifacts already contain score columns for evaluation noise:

```text
sigma_eval in {0, 0.03, 0.05, 0.08}
```

For each task and selected checkpoint row, compute:

- clean score `S_theta(0)`;
- noisy scores `S_theta(0.03)`, `S_theta(0.05)`, `S_theta(0.08)`;
- score retention `P_theta(sigma)`;
- `rho^S_0.8_grid`;
- `kappa_0_03` and `kappa_0_08`;
- ATR/SMPR if available for the same stressor;
- fixed-pool top-1 agreement / proxy gap if available.

Recommended checkpoint rows for main text:

1. no-noise baseline checkpoint per task;
2. matched recovered endpoint per task, e.g. current `std_max=0.08` endpoint if already used;
3. optional behavioral-onset checkpoint per task if already defined in current artifacts.

### 4.2 Optional stronger V1: denser evaluation-noise scan

If compute is available, add evaluation sigmas:

```text
0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10
```

Priority order:

1. base checkpoint and selected recovered endpoint per task;
2. behavioral-onset checkpoint per task;
3. all nine train-noise checkpoints only if cheap.

This strengthens the decay curve and radius estimate but is not required for the basic V1 revision.

### 4.3 Main claim supported by V1

Allowed V1 claim:

> The evaluation-noise response profile gives a coarse, fixed-checkpoint estimate of capability decay and matched-Gaussian robustness radius. ATR/SMPR explain whether the score radius is accompanied by predictive-tube contraction and margin preservation.

Do not claim:

> The profile tells us exactly what training noise to use.

---

## 5. V2 roadmap: training-noise damage frontier and budget selection

Move the training-noise budget work here. Do not make it a current-paper acceptance requirement.

### 5.1 Required V2前置实验

For each task, train or evaluate a wider noise-augmentation grid, likely extending beyond current `std_max=0.08`:

```text
std_max in {0.00, 0.01, ..., 0.08, 0.10, 0.12, 0.16, 0.20}
```

The exact high end should be chosen based on where clean score or task margin visibly degrades. Do not precommit to these exact values if unstable.

For each train-noise checkpoint, evaluate:

- clean score;
- matched Gaussian endpoint score;
- dense evaluation-noise response curve if possible;
- ATR/SMPR;
- fixed-pool top-1 agreement;
- cost-drift/margin proxy;
- if possible, stricter sample-level q10/q95 certificate fields.

### 5.2 V2 quantities

Define harmful onset:

```latex
\rho^{\rm harm}_{T}
=\min\{\rho:S_{\theta_\rho}(0)<S_{\theta_0}(0)-\delta
\;\text{or}\;\mathrm{SMPR}_{\theta_\rho}<\gamma
\;\text{or margin proxy degrades}\}.
```

Define benefit onset:

```latex
\rho^{\rm benefit}_{T}
=\min\{\rho:S_{\theta_\rho}(\sigma_{\rm stress})
\ge \tau_{\rm rec}\cdot S^{\rm best}_{T}(\sigma_{\rm stress})\}.
```

Define safe/effective budget band:

```latex
\mathcal B_T
=\{\rho:\rho\ge \rho^{\rm benefit}_{T},
\rho<\rho^{\rm harm}_{T},
\widehat\rho^S_{0.8,\rm grid}(\theta_\rho)\text{ is high}\}.
```

This should be V2 because it requires observing both the benefit side and the harmful side.

### 5.3 V2 claim if experiments succeed

Only after the harmful frontier is measured, the paper can claim:

> The diagnostic profile can narrow task-specific safe/effective training-noise budgets by balancing nuisance-radius contraction against clean-performance and margin erosion.

Until then, keep it as future work.

---

## 6. Main-text restructuring for V1

### 6.1 Abstract

Replace any training-budget sentence with a fixed-checkpoint radius sentence.

Recommended sentence:

> We further treat the Gaussian evaluation sweep as a fixed-checkpoint noise-response profile, estimating capability decay rates and coarse score-retention radii under the matched stressor.

Final abstract scope sentence:

> These results support ATR/SMPR and noise-response profiles as bounded diagnostics for matched-Gaussian recovery bands in fixed JEPA world-model checkpoints.

### 6.2 Introduction

Add a paragraph after the broad matched-Gaussian recovery-band paragraph:

> The same evaluation grid also asks a more local diagnostic question: for a fixed checkpoint, how quickly does capability decay as observation noise increases? We therefore read the evaluation-noise sweep as a noise-response profile rather than only as an endpoint stress test. The profile yields a finite-difference decay rate and a coarse score-retention radius on the retained noise grid. These quantities characterize the checkpoint's matched-Gaussian tolerance and provide a concrete target for future training interventions, without claiming that the current paper selects an optimal training-noise strength.

Update contribution C3 accordingly.

### 6.3 Theory section

Add the definitions in Section 3 of this plan. Do not include a main-text training-noise-budget selection rule.

### 6.4 Study protocol

Add:

> The Gaussian evaluation grid is used in two ways: the endpoint at `sigma=0.08` establishes the main behavior, while the multi-level grid `{0,0.03,0.05,0.08}` defines a coarse fixed-checkpoint noise-response profile.

### 6.5 Experiments section

Add a subsection after the base fragility table or after the Gaussian sweep figure.

Recommended title:

```latex
\subsection{Fixed-checkpoint noise-response profiles}
```

Core claims:

1. Base checkpoints have task-dependent decay rates.
2. Matched noise-trained endpoints have larger coarse score-retention radii.
3. ATR/SMPR and fixed-pool proxies explain the radius increase through predictive-tube contraction and margin preservation.
4. This is a fixed-checkpoint diagnostic, not a training-noise optimizer.

### 6.6 Discussion / limitations

Add:

> Estimating the training-noise budget itself requires a separate over-noising-frontier experiment: the current analysis measures evaluation-noise tolerance of fixed checkpoints, not the train-noise level at which clean performance or task margins begin to degrade.

---

## 7. Figures and tables for V1

### Main Figure A: fixed-checkpoint noise-response curves

File:

```text
assets/paper1_figs/fig_ckpt_noise_response_profiles.png
```

Content:

- one panel per task;
- x-axis: evaluation noise `sigma_eval`;
- y-axis: closed-loop score or score retention;
- curves: base checkpoint and selected recovered endpoint;
- optional: behavioral-onset checkpoint if already available.

Caption emphasis:

> The curves estimate fixed-checkpoint capability decay under the matched Gaussian stressor. The retained grid gives a coarse radius, not a continuous robustness certificate.

### Main Table A: fixed-checkpoint radius/decay summary

File:

```text
paper1/tables/table_ckpt_noise_response_radius.tex
```

Columns:

```text
Task
checkpoint
S(0)
S(0.08)
rho^S_0.8_grid
kappa_0_03
kappa_0_08
ATR
SMPR
interpretation
```

Use generated values only. Do not hand-enter unverified numbers.

### Optional Main Figure B: diagnostic radius/margin overlay

If current `fig_full_sweep_diagnostics` already overlays score, ATR, SMPR failure, and top-1 disagreement, update its caption/text to say it supports the noise-response/radius interpretation. Do not add a redundant figure unless useful.

### Appendix Table: threshold sensitivity

File:

```text
paper1/tables/table_ckpt_radius_threshold_sensitivity.tex
```

Rows:

```text
Task | checkpoint | tau | rho^S_tau_grid | comment
```

Use `tau in {0.7,0.8,0.9}`.

### Appendix V2 placeholder table only if needed

Do not add a training-noise budget table in V1 unless the harmful frontier is measured. A placeholder roadmap table can go in docs, not in the paper.

---

## 8. Script / artifact plan for Codex

### 8.1 Inspect existing artifacts

```bash
git checkout ag/dev
git pull

grep -R "three_seed_gaussian\|fig2_sweep\|appendix-gaussian-sweep\|ATR\|SMPR\|fixed_pool\|radius_margin" -n tools paper1 assets | head -300
ls assets/paper1_data
ls assets/paper1_figs
ls paper1/tables
```

### 8.2 Add V1 script

Recommended file:

```text
tools/paper1_ckpt_noise_response.py
```

Responsibilities:

- load existing Gaussian sweep score summaries;
- select rows: base, endpoint, optional onset;
- compute score retention;
- compute `rho^S_tau_grid` for `tau={0.7,0.8,0.9}`;
- compute `kappa_0_03` and `kappa_0_08`;
- join ATR/SMPR/fixed-pool summaries if available;
- emit machine-readable summary:

```text
assets/paper1_data/ckpt_noise_response_summary.json
assets/paper1_data/ckpt_noise_response_radius.csv
```

- emit paper table:

```text
paper1/tables/table_ckpt_noise_response_radius.tex
paper1/tables/table_ckpt_radius_threshold_sensitivity.tex
```

- emit figure:

```text
assets/paper1_figs/fig_ckpt_noise_response_profiles.png
```

### 8.3 Metadata requirements

Each JSON artifact should record:

```json
{
  "analysis_type": "fixed_checkpoint_noise_response",
  "eval_noise_grid": [0.0, 0.03, 0.05, 0.08],
  "radius_type": "coarse_grid",
  "tau_values": [0.7, 0.8, 0.9],
  "training_seeds": [3072, 3073, 3074],
  "evaluation_seeds": [42, 43, 44],
  "trajectories_per_eval_seed": 100,
  "claims_excluded": ["optimal_training_noise", "closed_loop_certificate", "general_corruption_transfer"]
}
```

### 8.4 Build commands

After editing `main.tex`:

```bash
cd paper1
bash build.sh --clean
bash build.sh
```

Check:

```bash
grep -R "TODO\|TBD\|PLACEHOLDER\|optimal training-noise\|training-noise budget" -n main.tex tables docs || true
```

If `training-noise budget` remains in the main paper, it should appear only as a future-work limitation, not as a completed claim.

---

## 9. V2 script / artifact plan, not V1

If/when V2 starts, add:

```text
tools/paper1_train_noise_budget_v2.py
```

Expected V2 outputs:

```text
assets/paper1_data/train_noise_budget_v2_summary.json
assets/paper1_data/train_noise_damage_frontier.csv
assets/paper1_figs/fig_train_noise_budget_v2_surface.png
paper1/tables/table_train_noise_budget_v2.tex
```

V2 must include heavier train-noise checkpoints or otherwise demonstrate the clean/margin damage onset. Without that, the table should not be included in the paper.

---

## 10. Wording guardrails

### Allowed V1 wording

Use:

- "fixed-checkpoint noise-response profile"
- "capability decay rate"
- "coarse score-retention radius"
- "matched-Gaussian evaluation-noise tolerance"
- "diagnostic radius / proxy radius"
- "suggests a future training-noise budget route"
- "does not select an optimal training-noise strength"

### Avoid in V1

Avoid:

- "we choose the optimal training noise"
- "we determine the task-specific training-noise budget"
- "training noise should be X for this task"
- "over-noising starts at X" unless measured
- "continuous robustness radius" if using four grid points
- "closed-loop guarantee"
- "adaptive CEM certificate"

### Best one-sentence positioning

> The V1 revision estimates how robust each fixed checkpoint is to evaluation noise; V2 can use the same profiling idea to choose training-noise budgets once the over-noising frontier is measured.

---

## 11. Acceptance criteria

### Minimum V1 acceptance

The revision is acceptable if it adds:

- one theory subsection defining fixed-checkpoint noise-response profile, decay rate, and coarse score-retention radius;
- one main experiment subsection on fixed-checkpoint noise-response profiles;
- one figure showing score/retention vs evaluation noise for base and recovered checkpoints;
- one table summarizing `rho^S_0.8_grid` and decay slopes;
- one appendix threshold-sensitivity table;
- explicit limitation that training-noise budget selection is V2 because harmful-side experiments are missing.

### Stronger V1

The revision is stronger if it additionally adds:

- denser evaluation-noise scan;
- ATR/SMPR curves across multiple evaluation-noise levels;
- diagnostic radius/proxy radius aligned with score radius;
- fixed-pool top-1 agreement response curves.

### V2 acceptance, later

A V2 training-noise-budget claim requires:

- task-wise heavier train-noise grid;
- evidence of clean-score or margin degradation onset;
- safe/effective band definition with threshold sensitivity;
- diagnostic-only or held-out-seed validation if claiming predictive guidance.

---

## 12. Concrete edit checklist for Codex

1. Keep the current ACPC/radius--margin theory intact.
2. Add fixed-checkpoint noise-response definitions.
3. Remove or demote main-paper training-noise budget language.
4. Generate `fig_ckpt_noise_response_profiles.png`.
5. Generate `table_ckpt_noise_response_radius.tex`.
6. Add appendix threshold sensitivity.
7. Add limitation paragraph: training-noise damage frontier is V2.
8. Build paper.
9. Check that no unsupported training-budget claim remains.

Suggested commit message after implementation:

```bash
git commit -m "paper1: add fixed-checkpoint noise-response radius analysis"
```

---

## 13. Final recommendation

Do the fixed-checkpoint noise-response/radius upgrade now. It is well aligned with paper1's diagnostic identity and can be supported by the current artifacts.

Delay the training-noise budget claim to V2. It is a good direction, but it needs the extra前置实验: each task must be pushed far enough in training-noise strength to reveal when clean performance or task-relevant margins begin to degrade. Without that frontier, the current paper should only say that the fixed-checkpoint profile can **motivate** future training-noise selection, not that it already solves it.
