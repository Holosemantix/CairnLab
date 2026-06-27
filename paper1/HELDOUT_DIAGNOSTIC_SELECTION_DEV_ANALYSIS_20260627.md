# Held-out Diagnostic Selection Development Analysis (2026-06-27)

Status: standalone development note, not yet integrated into `main.tex`.

Purpose: decide whether the ACPC/CRA/MAF diagnostics are worth strengthening into a prospective checkpoint-selection experiment for the main-conference version. This document uses existing seed-3072 artifacts only as a development/smoke-test set. It must not be described as held-out evidence in the paper.

## Executive Decision

Proceed with the held-out diagnostic-selection direction, but keep the claim boundary strict.

Existing seed-3072 data are strong enough to justify continuing: a fixed rank-based diagnostic selector with a 5 percentage-point clean-performance guard selects LeWM checkpoints inside the high-performance plateau on all four tasks for the primary observation-only `pixels_std0.08` target. Reported regret to the top mean is small (<=1.0 pp), but this should be treated as a descriptive check, not as evidence that there is a single best noise level. The same fixed rule transfers reasonably to PLDM, selecting plateau checkpoints on all four tasks with <=2.33 pp regret to the top mean.

The main-conference contribution should be framed as a checkpoint-level perturbation-resilience readout, not as hyperparameter optimization. The core value claim is: under a simple controlled stressor, here additive Gaussian observation noise, the representation geometry and action-conditioned predictive-dynamics space expose measurable signatures of whether a trained world-model checkpoint has entered an anti-perturbation plateau. If a fixed diagnostic protocol can identify that plateau before looking at the target corrupted closed-loop evaluation, then ACPC/CRA/MAF become a prospective diagnostic instrument for world-model robustness rather than only a post-hoc explanation. The same measurement recipe can guide analyses of other perturbation families later, but the paper should not claim cross-perturbation validity until those stressors are tested.

This is the stronger and more defensible contribution than "finding the best std." The best-mean checkpoint is noisy and often not unique; the scientific claim is that a controlled perturbation reveals stable predictive-space structure associated with robustness. The lockbox experiment should therefore ask whether the diagnostic protocol identifies robust-plateau membership, not whether it exactly predicts the top noisy evaluation mean.

However, this is not yet main-conference-grade evidence because the rule was developed after seeing the seed-3072 sweep. The right use is:

- seed 3072: development / protocol calibration only;
- new independent training seeds: lockbox validation;
- no further selector tuning after this protocol is frozen.

## Critical Assessment of the Intended Claim

The intended main-conference claim is directionally correct, but only in a bounded form.

Supported by the first step:

- The current data show a real perturbation-resilience phenomenon under the controlled Gaussian observation-noise stressor. In LeWM, no-noise checkpoints lose large amounts of `pixels_std0.08` success on PushT, Reacher, TwoRoom, and Cube, while noise-trained checkpoints recover into broad high-performance regions.
- The diagnostic quantities are not just arbitrary correlations with a single top-mean checkpoint. They track a coarser and more defensible target: whether the checkpoint is in the robust plateau rather than in the no-noise collapse region.
- The diagnostic selector uses representation/predictive-dynamics readouts before seeing the target corrupted closed-loop metric. This is the right shape for a top-conference contribution because it converts ACPC/CRA/MAF from retrospective explanation into a testable checkpoint-triage protocol.
- The PLDM sanity check suggests the rule is not purely a LeWM-only artifact, although PLDM clean-only selection is already strong and should not be oversold.

Not yet supported:

- The current evidence does not prove a causal mechanism. It shows predictive-space signatures associated with robustness; causal language needs either an intervention objective or a stronger ablation.
- The current evidence does not prove cross-perturbation generality. Gaussian noise can validate the diagnostic protocol, but blur, occlusion, lighting, camera shift, and distractors need their own paired-corruption runs.
- The current seed-3072 analysis is not true held-out evidence. It is a development check. Main-paper strength requires the same frozen selector to work on independent training seeds.

Verdict: the first step does meet the expected direction and is worth continuing, provided the paper frames the contribution as **controlled-stressor diagnostic triage into a robust plateau**, not as a universal robustness oracle or exact hyperparameter optimizer. The two new training seeds are therefore not just extra statistics; they are the lockbox test that determines whether this claim can move into the main paper.

## Available Artifacts Used

All analysis below is computed from existing JSON artifacts. No checkpoint files are required.

| Artifact | Role in this note |
|---|---|
| `assets/paper1_data/canonical_evals_20260517.json` | LeWM closed-loop success for 4 tasks x 9 noise-training configs, eval seeds 42/43/44. |
| `assets/paper1_data/acpc_basin_diagnostics.json` | LeWM observation-history Gaussian ACPC basin diagnostics for 4 tasks x 9 configs. |
| `assets/paper1_data/acpc_phase0_diagnostics.json` | LeWM and PLDM Phase-0 ACPC/PCC/CRA/MAF diagnostics. Existing artifact uses observation+goal corruption unless rerun with `--clean-goal`. |
| `assets/paper1_data/acpc_basin_diagnostics_pldm.json` | PLDM observation-history Gaussian ACPC basin diagnostics. |
| `assets/paper1_data/canonical_evals_pldm_20260522.json` | PLDM closed-loop success for 4 tasks x 9 configs. |

Important mismatch: the existing ACPC-basin artifact matches the main observation-only target, while the existing Phase-0 artifact computes PCC/CRA/MAF under observation+goal corruption. Therefore Phase-0 results here are useful as a retrospective sanity check, but the lockbox run should rerun Phase-0 with `--clean-goal` before selection.

## Equal-coverage Reanalysis: LeWM vs PLDM

The current LeWM and PLDM artifacts have matched experimental coverage:

| Method | Closed-loop eval grid | ACPC-basin rows | Phase-0 rows |
|---|---:|---:|---:|
| LeWM | 4 tasks x 9 `std_max` = 36 groups | 36 ok rows | 36 ok rows |
| PLDM | 4 tasks x 9 `std_max` = 36 groups | 36 ok rows | 36 ok rows |

Using the same frozen 5 pp clean guard and hybrid rank selector, both methods select checkpoints inside the target high-performance plateau on all four tasks.

Primary observation-only target, `pixels_std0.08_success`:

| Method | Hybrid plateau hits | Hybrid mean regret to top mean | Hybrid max regret | Clean-only plateau hits | Fixed-0.08 plateau hits | No-noise plateau hits |
|---|---:|---:|---:|---:|---:|---:|
| LeWM | 4/4 | 0.50 pp | 1.00 pp | 2/4 | 3/4 | 0/4 |
| PLDM | 4/4 | 1.25 pp | 2.33 pp | 4/4 | 4/4 | 1/4 |

Secondary observation+goal target, `pixels_goal_std0.08_success`:

| Method | Hybrid plateau hits | Hybrid mean regret to top mean | Hybrid max regret | Clean-only plateau hits | Fixed-0.08 plateau hits | No-noise plateau hits |
|---|---:|---:|---:|---:|---:|---:|
| LeWM | 4/4 | 0.42 pp | 1.67 pp | 2/4 | 3/4 | 0/4 |
| PLDM | 4/4 | 1.17 pp | 2.67 pp | 4/4 | 3/4 | 1/4 |

Interpretation:

- The plateau-triage conclusion is consistent across LeWM and PLDM: the same diagnostic selector lands in the high-performance plateau for every task under both target metrics.
- The comparative strength differs. For LeWM, the diagnostic selector is clearly more informative than clean-only selection and fixed `std_max=0.08` on the seed-3072 development grid. For PLDM, clean-only selection is already strong, so PLDM supports cross-method consistency but not diagnostic superiority over clean-only.
- The no-noise-collapse phenomenon is much stronger in LeWM. PLDM has large no-noise drops on PushT and TwoRoom, but Reacher and Cube are already close to plateau under no-noise for the primary target. This is another reason to keep PLDM as a bounded replication/sanity check rather than making it the main mechanism story.

## Candidate Checkpoint Set

For each task and training seed, define the candidate checkpoint set:

```text
Theta_task,seed = {std_max = 0.00, 0.01, ..., 0.08}
```

The target metric is hidden from the selector:

```text
primary target:   pixels_std0.08_success
secondary target: pixels_goal_std0.08_success
```

The target metric is not used to define a unique "best" checkpoint. Because closed-loop evaluation variance is non-negligible and several adjacent noise-training levels often form a performance plateau, the primary success criterion is plateau membership:

```text
plateau_tolerance = max(3 pp, evaluation-seed std of the top-mean checkpoint)
plateau = {theta: success(theta) >= top_mean_success - plateau_tolerance}
plateau_hit = 1[theta_hat in plateau]
```

Regret to the top mean remains useful as a compact descriptive number, but it should not be framed as the main scientific target.

For the current development analysis there is only one independent training seed, the released seed-3072 grid.

## Selector Protocol to Freeze for Lockbox Seeds

### Clean-performance guard

A checkpoint is eligible only if its clean success is within 5 percentage points of the best clean checkpoint for the same task and training seed:

```text
G(theta) = 1[clean_success(theta) >= max_clean_success - 5 pp]
```

Rationale: a robustness selector that sacrifices clean performance is not a useful model selector for the paper's setting. A looser 10 pp guard admits an over-contracted Cube checkpoint (`std_max=0.05`) with poor closed-loop success; that is a real development-set failure mode and the main reason to use a 5 pp guard. This guard must now be frozen before evaluating new training seeds.

### Rank-based score

Within the eligible set only, compute average ranks. Lower score is better. No learned weights are used.

Basin-only score:

```text
S_basin(theta) =
  rank_low(pred_view_pair_l2_norm_by_transition)
+ rank_low(pred_to_clean_l2_norm_by_transition_p90)
+ rank_low(basin_contraction_to_clean_norm_median)
```

Phase-0 cost score:

```text
S_phase0(theta) =
  rank_low(acpc_h_norm_by_transition)
+ rank_low(pcc_abs_p90)
+ rank_low(maf_flip_rate)
+ rank_high(cra_spearman_median)
+ rank_high(elite_overlap_mean)
```

Hybrid selector for the lockbox experiment:

```text
S_hybrid(theta) =
  rank_low(pred_view_pair_l2_norm_by_transition)
+ rank_low(pred_to_clean_l2_norm_by_transition_p90)
+ rank_low(pcc_abs_p90)
+ rank_low(maf_flip_rate)
+ rank_high(cra_spearman_median)
+ rank_high(elite_overlap_mean)
```

Selection rule:

```text
theta_hat = argmin_{theta in Theta_task,seed, G(theta)=1} S_hybrid(theta)
```

Tie-breaker: choose the smaller `std_max` if scores are equal. This tie-breaker is conservative and prevents silently preferring the highest training noise level.

## Seed-3072 Development Results: LeWM

These results are retrospective. They justify continuing the direction but should not be written as held-out evidence.

### Primary target: `pixels_std0.08_success`

Hybrid selector with 5 pp clean guard:

| Task | Selected std | Selected success | Top-mean std | Top mean | Plateau stds | Plateau hit | Regret to top mean |
|---|---:|---:|---:|---:|---|---:|---:|
| Cube | 0.07 | 67.3 | 0.03 | 68.3 | 0.03, 0.06, 0.07 | yes | 1.0 |
| PushT | 0.08 | 89.0 | 0.08 | 89.0 | 0.04, 0.06, 0.08 | yes | 0.0 |
| Reacher | 0.02 | 83.7 | 0.07 | 84.7 | 0.02, 0.04, 0.06, 0.07, 0.08 | yes | 1.0 |
| TwoRoom | 0.08 | 97.7 | 0.08 | 97.7 | 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 | yes | 0.0 |

Aggregate: 4/4 plateau hits; mean regret to the top mean 0.5 pp, max regret 1.0 pp, 4/4 tasks within 3 pp of the top mean.

Baselines on the same primary target:

| Selector | Mean regret to top mean | Median regret | Max regret | Within 3 pp of top mean |
|---|---:|---:|---:|---:|
| Hybrid diagnostic selector | 0.5 | 0.5 | 1.0 | 4/4 |
| Clean-only selector | 7.42 | 5.17 | 19.33 | 1/4 |
| Fixed `std_max=0.08` | 2.25 | 1.33 | 6.33 | 3/4 |
| No-noise baseline | 51.08 | 49.17 | 84.67 | 0/4 |

Interpretation: the selector avoids the no-noise collapse and lands in the robust plateau on every task. The most important qualitative win over fixed-0.08 is Cube, where fixed 0.08 falls outside the top plateau under the current tolerance, while the diagnostic selector picks 0.07, which is plateau-compatible. This should be framed as plateau triage, not as identifying a uniquely optimal training-noise level.

### Secondary target: `pixels_goal_std0.08_success`

Hybrid selector with 5 pp clean guard:

| Task | Selected std | Selected success | Top-mean std | Top mean | Plateau stds | Plateau hit | Regret to top mean |
|---|---:|---:|---:|---:|---|---:|---:|
| Cube | 0.07 | 68.0 | 0.07 | 68.0 | 0.03, 0.04, 0.06, 0.07 | yes | 0.0 |
| PushT | 0.08 | 85.3 | 0.06 | 87.0 | 0.06, 0.08 | yes | 1.7 |
| Reacher | 0.02 | 85.7 | 0.02 | 85.7 | 0.02, 0.06, 0.08 | yes | 0.0 |
| TwoRoom | 0.08 | 98.7 | 0.08 | 98.7 | 0.05, 0.06, 0.07, 0.08 | yes | 0.0 |

Aggregate: 4/4 plateau hits; mean regret to the top mean 0.42 pp, max regret 1.67 pp, 4/4 tasks within 3 pp of the top mean.

Baselines on the same secondary target:

| Selector | Mean regret to top mean | Median regret | Max regret | Within 3 pp of top mean |
|---|---:|---:|---:|---:|
| Hybrid diagnostic selector | 0.42 | 0.0 | 1.67 | 4/4 |
| Clean-only selector | 5.42 | 2.5 | 16.67 | 2/4 |
| Fixed `std_max=0.08` | 3.0 | 2.17 | 7.67 | 3/4 |
| No-noise baseline | 55.83 | 59.67 | 82.33 | 0/4 |

## Cross-method Sanity Check: PLDM

PLDM is not a true lockbox because the same selector was inspected after the LeWM development pass, but it is useful as a cross-method sanity check. The same 5 pp clean guard and rank rule is used.

### Primary target: `pixels_std0.08_success`

| Task | Selected std | Selected success | Top-mean std | Top mean | Plateau stds | Plateau hit | Regret to top mean |
|---|---:|---:|---:|---:|---|---:|---:|
| Cube | 0.08 | 53.7 | 0.04 | 54.7 | 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 | yes | 1.0 |
| PushT | 0.05 | 71.7 | 0.03 | 73.0 | 0.03, 0.04, 0.05, 0.06, 0.08 | yes | 1.3 |
| Reacher | 0.08 | 79.3 | 0.03 | 81.7 | 0.0, 0.01, 0.02, 0.03, 0.04, 0.07, 0.08 | yes | 2.3 |
| TwoRoom | 0.08 | 97.7 | 0.06 | 98.0 | 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 | yes | 0.3 |

Aggregate: 4/4 plateau hits; mean regret to the top mean 1.25 pp, max regret 2.33 pp, 4/4 tasks within 3 pp of the top mean.

Baselines on the same PLDM primary target:

| Selector | Mean regret to top mean | Median regret | Max regret | Within 3 pp of top mean |
|---|---:|---:|---:|---:|
| Hybrid diagnostic selector | 1.25 | 1.17 | 2.33 | 4/4 |
| Clean-only selector | 0.5 | 0.33 | 1.33 | 4/4 |
| Fixed `std_max=0.08` | 2.17 | 1.67 | 5.0 | 3/4 |
| No-noise baseline | 23.33 | 18.67 | 54.67 | 1/4 |

Interpretation: on PLDM, clean-only is already very strong because clean and robust success are aligned more tightly in this artifact. The diagnostic selector still lands in the same high-performance plateau, but it does not beat clean-only by regret-to-top-mean. This should be framed as cross-method sanity, not superiority evidence.

### Secondary target: `pixels_goal_std0.08_success`

| Task | Selected std | Selected success | Top-mean std | Top mean | Plateau stds | Plateau hit | Regret to top mean |
|---|---:|---:|---:|---:|---|---:|---:|
| Cube | 0.08 | 56.3 | 0.08 | 56.3 | 0.03, 0.06, 0.08 | yes | 0.0 |
| PushT | 0.05 | 69.3 | 0.03 | 72.0 | 0.03, 0.05, 0.06 | yes | 2.7 |
| Reacher | 0.08 | 80.7 | 0.04 | 81.3 | 0.0, 0.02, 0.03, 0.04, 0.08 | yes | 0.7 |
| TwoRoom | 0.08 | 97.0 | 0.05 | 98.3 | 0.04, 0.05, 0.06, 0.07, 0.08 | yes | 1.3 |

Aggregate: 4/4 plateau hits; mean regret to the top mean 1.17 pp, max regret 2.67 pp, 4/4 tasks within 3 pp of the top mean.

## Failure Mode Found During Development

A 10 pp clean guard is too loose. With that guard, both basin and Phase-0 selectors choose Cube `std_max=0.05`, which has very low ACPC/PCC values but poor closed-loop success:

| Target | Selected std under 10 pp guard | Selected success | Top mean | Regret to top mean |
|---|---:|---:|---:|---:|
| Cube `pixels_std0.08` | 0.05 | 60.0 | 68.3 | 8.3 |
| Cube `pixels_goal_std0.08` | 0.05 | 59.7 | 68.0 | 8.3 |

This is exactly the collapse/over-contraction concern in the ACPC theory. Low same-state predictive movement is not sufficient; the checkpoint must also preserve clean task competence. The 5 pp clean guard is therefore not cosmetic. It is the minimal current guard available from existing artifacts.

## Recommended Lockbox Experiment

Use seed 3072 as the development set and independent training seeds as lockbox:

```text
development/calibration: training seed 3072
lockbox validation:      training seeds 3073 and 3074, optionally 3075/3076
```

For each lockbox `task x training_seed`:

1. Generate candidate checkpoints for `std_max=0.00...0.08`.
2. Compute ACPC-basin diagnostics with observation-history Gaussian noise.
3. Rerun Phase-0 diagnostics with clean goal:

```bash
python -m tools.paper1_phase0_acpc \
  --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
  --noise-std 0.08 \
  --clean-goal \
  --seed 9101 \
  --out assets/paper1_data/heldout_selection_phase0_seed9101.json
```

4. Apply the frozen 5 pp clean guard and hybrid rank selector without looking at `pixels_std0.08_success`.
5. Write selected checkpoints to a machine-readable file, for example:

```text
assets/paper1_data/heldout_diagnostic_selected_checkpoints_2026xxxx.json
```

6. Only then aggregate closed-loop eval and compute regret.

Primary reporting unit:

```text
one independent sample = task x training_seed
```

Do not treat the three evaluation seeds 42/43/44 as independent training samples.

Recommended summary metrics:

```text
top_mean = max_theta pixels_std0.08_success(theta)
plateau_tolerance = max(3 pp, evaluation-seed std of the top-mean checkpoint)
plateau_hit = 1[selected_success >= top_mean - plateau_tolerance]
regret_to_top_mean = top_mean - selected_success
```

`plateau_hit` should be the primary binary readout. `regret_to_top_mean` is a secondary descriptive readout, because the top-mean checkpoint may not be distinguishable from neighboring plateau checkpoints under only three evaluation seeds.

Compare against:

```text
clean-only selector
fixed std_max=0.08 selector
no-noise baseline
random eligible checkpoint expectation
```

## Theory / Text Impact

The existing fixed-candidate ACPC derivation does not need a deeper theorem for this experiment. It already supports the diagnostic link:

- ACPC bounds candidate-cost drift on a fixed candidate set.
- Margin controls top-1 candidate stability.
- ACPC alone admits collapse, so a discriminability/competence guard is necessary.

What should be added later, if lockbox results are favorable, is a short empirical-protocol definition, not a stronger mathematical guarantee. That protocol should explicitly target high-performance plateau selection rather than unique best-checkpoint identification.

Draft text to add later:

```text
Diagnostic checkpoint selection protocol. For each task and independent training seed, let Theta be the set of checkpoints trained with std_max in {0.00,...,0.08}. Before observing the held-out corrupted closed-loop success, we compute paired predictive diagnostics on fixed dataset samples and form a rank-based score from ACPC, PCC, CRA, and MAF readouts. We first apply a clean-performance guard that excludes checkpoints more than 5 percentage points below the best clean checkpoint for that task and seed. Among the remaining checkpoints we select the lowest diagnostic score. The selected checkpoint is then evaluated against held-out corrupted closed-loop success. Because several neighboring noise-training levels often form a performance plateau under evaluation-seed variance, the primary readout is whether the selected checkpoint lies in the held-out high-performance plateau; regret to the top mean is reported only as a descriptive secondary statistic.
```

Claim boundary if results are favorable:

```text
Under a controlled Gaussian observation-noise stressor, paired representation and action-conditioned predictive-dynamics diagnostics identify whether a trained checkpoint has entered the high-performance anti-perturbation plateau. This supports fixed-protocol checkpoint triage under the tested stress protocol.
```

Broader interpretation if future perturbation families are added:

```text
Gaussian noise is the controlled stressor used to validate the diagnostic protocol. The contribution is the paired predictive-dynamics measurement and checkpoint-triage procedure; the same procedure can be reused for blur, occlusion, lighting, or camera-shift stressors by replacing the paired corruption operator and preserving the fixed selection protocol.
```

Do not claim:

```text
ACPC is a robustness oracle.
ACPC guarantees CEM or closed-loop stability.
The selector generalizes beyond the tested perturbation family.
```

## Current Recommendation

Continue this direction. The development evidence is strong enough that the new independent seeds are worth using for lockbox diagnostic selection. The highest-value next action is to freeze this selector before looking at the new seed results.

If the lockbox repeats the seed-3072 pattern, this becomes a meaningful main-conference strengthening: the paper can move from post-hoc localization to fixed-protocol diagnostic triage into the robust plateau. If the lockbox fails, the result is still useful but should be reported as a limitation: ACPC localizes recovery but is not reliable as a standalone selector.
