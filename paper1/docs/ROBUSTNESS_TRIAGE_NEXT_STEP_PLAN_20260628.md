# Paper1 Robustness-Triage Next-Step Plan (2026-06-28)

Status: development plan. Do not merge into `main.tex` until the lockbox
training seeds validate the frozen protocol.

## 1. Core Direction

The next step should use the regularities exposed by noise-trained checkpoints
as a **robustness-triage protocol**, not as a new training objective.

Working claim:

> Under a controlled Gaussian observation-noise stressor, robust checkpoints
> occupy a measurable range in representation geometry and action-conditioned
> predictive-dynamics space. A frozen diagnostic protocol can test whether a
> checkpoint has entered the high-performance anti-perturbation plateau before
> looking at the target corrupted closed-loop evaluation.

This is stronger than saying "we found the best training-noise std." Closed-loop
evaluation variance makes a unique optimum unstable; the useful scientific
target is plateau membership.

## 2. What Is Supported Now

Existing seed-3072 analysis supports continuing this route:

- robust LeWM checkpoints show smaller clean/noisy predictive drift under the
  same action sequence;
- ACPC/PCC/MAF/CRA-style diagnostics select plateau checkpoints on all four
  LeWM tasks in the development grid;
- the same frozen rule also lands in PLDM plateau checkpoints, although PLDM
  clean-only selection is already strong;
- the no-noise collapse is much stronger in LeWM than PLDM, so PLDM should be
  used as a bounded replication/sanity check, not as the main mechanism story.

The current evidence is still development evidence because the selector was
designed after seeing seed 3072. The new independent training seeds are the real
lockbox test.

## 3. What Should Not Be Claimed

Do not claim any of the following at this stage:

- the diagnostic is a universal robustness oracle;
- the protocol identifies a unique best noise level;
- Gaussian-noise diagnostics automatically transfer to blur, occlusion,
  lighting, camera shift, or distractors;
- representation closeness alone is sufficient;
- naive consistency losses are expected to work.

Paper2 records already rule out several direct regularization routes under the
tested settings: GLC, one-step SNAP-ACPC, paired no-aux, in-forward noisy-only,
heteroscedastic loss reweighting, and fixed temporal hinge. These negative
results should make the paper more careful, not less ambitious: the diagnostic
regularities are useful, but they should not be naively converted into a loss
without a repaired data path and action/discriminability guards.

## 4. Frozen Lockbox Protocol

Use seed 3072 only for development. Freeze the selector now.

Lockbox seeds:

- primary: training seeds 3073 and 3074;
- optional extension: 3075/3076 if compute allows.

Candidate set per task and seed:

```text
Theta_task,seed = {std_max = 0.00, 0.01, ..., 0.08}
```

Hidden target metrics:

```text
primary:   pixels_std0.08_success
secondary: pixels_goal_std0.08_success
```

Plateau definition:

```text
plateau_tolerance = max(3 pp, evaluation-seed std of the top-mean checkpoint)
plateau = {theta: success(theta) >= top_mean_success - plateau_tolerance}
plateau_hit = 1[theta_hat in plateau]
```

Clean-performance guard:

```text
eligible(theta) = clean_success(theta) >= max_clean_success - 5 pp
```

Hybrid diagnostic score, lower is better:

```text
S_hybrid(theta) =
  rank_low(pred_view_pair_l2_norm_by_transition)
+ rank_low(pred_to_clean_l2_norm_by_transition_p90)
+ rank_low(pcc_abs_p90)
+ rank_low(maf_flip_rate)
+ rank_high(cra_spearman_median)
+ rank_high(elite_overlap_mean)
```

Selection:

```text
theta_hat = argmin S_hybrid(theta), among eligible checkpoints
tie-breaker = smaller std_max
```

Theory-to-diagnostic mapping:

The current theory section in `main.tex` is sufficient for this lockbox plan if
the claim remains diagnostic. It gives sufficient conditions for
fixed-candidate planning stability:

1. Bounded same-action rollout disagreement bounds candidate-cost drift under a
   local Lipschitz cost readout.
2. Bounded candidate-cost drift plus a clean top-1/top-2 margin preserves the
   selected candidate within a shared candidate set.
3. Low predictive disagreement is not meaningful without a discriminability
   guard, because collapsed representations can make ACPC zero while destroying
   action-relevant distinctions.

The hybrid selector is a finite-sample proxy for those sufficient conditions,
not a theorem that a checkpoint must lie in the closed-loop robustness plateau:

| Selector component | Theory role | Interpretation |
|---|---|---|
| `pred_view_pair_l2_norm_by_transition` | ACPC rollout disagreement proxy | Smaller values approximate the `epsilon` term in the fixed-candidate stability corollary. |
| `pred_to_clean_l2_norm_by_transition_p90` | Tail-risk rollout disagreement proxy | Penalizes checkpoints whose typical median is small but high-disagreement pairs remain large. |
| `pcc_abs_p90` | Candidate-cost drift proxy | Empirical downstream counterpart of the Lipschitz cost-drift bound. |
| `cra_spearman_median` | Candidate-ranking stability proxy | Tests whether the clean/noisy branches keep similar candidate orderings. |
| `elite_overlap_mean` | Top-set stability proxy | Weaker than top-1 equality but more stable under noisy finite candidate samples. |
| `maf_flip_rate` | Margin-conditioned action-stability proxy | Directly mirrors the top-1 stability proposition under observed clean margins. |
| clean guard, effective rank, transition resolution, ID probe | Discriminability guard | Prevents selecting over-contracted checkpoints that are stable only because they collapsed action-relevant distinctions. |

Therefore the lockbox success criterion should be written as:

> The theory motivates a frozen diagnostic score for the sufficient conditions
> of fixed-candidate stability. The closed-loop robust plateau remains an
> empirical target because CEM resampling, repeated replanning, unknown local
> Lipschitz constants, and environment feedback are outside the theorem.

Do not add another theorem unless the empirical protocol changes. More formal
claims would require estimating candidate margins/Lipschitz constants on the
actual planner distribution, or proving stability under CEM resampling and
closed-loop feedback. That is not needed for the present diagnostic contribution
and would likely overpromise.

Required diagnostic rerun before final lockbox analysis:

```bash
python -m tools.paper1_phase0_acpc \
  --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
  --noise-std 0.08 \
  --clean-goal \
  --seed 9101 \
  --out assets/paper1_data/heldout_selection_phase0_seed9101.json
```

The archived Phase-0 artifact `assets/paper1_data/acpc_phase0_diagnostics.json`
mixes observation+goal corruption. The seed-3072 clean-goal rerun is complete
for LeWM and PLDM and merged at
`assets/paper1_data/acpc_phase0_clean_goal_seed9101.json`. Any lockbox rerun on
new training seeds should keep the same observation-only corruption with a clean
goal to match the primary target.

## 5. Analysis Outputs To Produce

For each new training seed, produce:

1. Per-task selected checkpoint table:
   task, selected `std_max`, selected clean, selected corrupted success, top
   mean, plateau set, plateau hit, regret to top mean.

2. Aggregate selector comparison:
   hybrid diagnostic selector vs clean-only vs fixed `std_max=0.08` vs no-noise.

3. Robustness-triage summary:
   plateau-hit rate, mean regret, max regret, and within-3pp count.

4. Diagnostic guard table:
   effective rank, transition-resolution, ID probe, and any collapse warning.

5. Failure audit:
   cases where the selector misses the plateau, cases where clean guard removes
   a low-drift but behaviorally bad checkpoint, and cases where PLDM differs
   from LeWM.

## 6. Paper Integration Decision Gate

Integrate into `main.tex` only if the frozen selector satisfies all three gates:

- **Plateau gate:** at least 3/4 tasks hit the primary plateau per lockbox seed,
  and the aggregate across seeds is clearly above clean-only/no-noise baselines.
- **Guard gate:** selected checkpoints do not show obvious rank, transition, or
  inverse-dynamics collapse.
- **Stability gate:** selector behavior is stable across 3073/3074, or any
  failures have a clear, bounded explanation that can be written honestly.

If the gates pass, add a compact main-text paragraph and an appendix table:

> A fixed diagnostic selector, developed on seed 3072 and frozen before two
> independent training seeds, identifies robust-plateau checkpoints under the
> Gaussian observation-noise stressor. This supports ACPC/CRA/MAF as a
> prospective robustness-triage instrument, not merely a post-hoc explanation.

If the gates do not pass, keep the result in appendix or development notes and
frame ACPC as localization/interpretation only.

## 7. Relation To Training Objectives

The diagnostic regularities can guide future methods, but the immediate Paper1
route should remain diagnostic.

Safe wording:

> The observed ranges suggest that useful training objectives should target
> action-conditioned predictive stability while preserving action-relevant
> discriminability. Existing negative controls show that naive encoder
> invariance, one-step predictive matching, and error-based loss reweighting are
> insufficient.

Do not write:

> We can directly add the diagnostic metric as a regularizer.

Any future training-objective experiment should first repair the dataset-level
paired-view transform, verify no-aux equivalence to ordinary noise training, and
only then test a selective/action-conditioned objective.

## 8. Longer-Term Extensions

After the Gaussian lockbox test:

1. Add one or two unseen perturbation families: brightness/contrast, occlusion,
   camera shift, or background distractors.
2. Reuse the same paired-corruption diagnostic protocol, but do not reuse
   Gaussian thresholds without recalibration.
3. Add semantic discriminability guards where available:
   PushT keypoint/contact/pose, TwoRoom topology/doorway state, Reacher
   joint-target relation, Cube pose/goal relation.
4. If method work resumes, compare selective/action-conditioned objectives
   against ordinary noise training under the same training seed and same
   evaluation protocol.

## 9. Recommended Route And Current Status

This section is the cross-machine execution ledger. Update it whenever a run
finishes on any machine.

| ID | Item | Current status | Why it matters |
|---|---|---|---|
| R1 | Independent LeWM training seeds 3073/3074 | Complete; 4 tasks x 9 std grid available for both seeds | Main-conference reviewers will not accept evaluation seeds as a substitute for training-run variability. |
| R2 | Frozen seed-3072 robustness-triage selector | Development analysis done; selector frozen in this document | Converts ACPC/CRA/MAF from post-hoc explanation into a testable plateau-triage protocol. |
| R3 | Gaussian lockbox application to 3073/3074 | Complete; see `paper1/docs/LOCKBOX_RESULTS_20260703.md` | Confirms the no-noise cliff, noise-training recovery, and broad plateau reading under independent training seeds. |
| R4 | Seed-3072 unseen-perturbation pilot | Four-task strongest-only formal pass complete; TwoRoom/Reacher positive, PushT weak/mixed, Cube neutral | Tests whether the protocol is Gaussian-specific or reusable with a different paired-corruption operator. |
| R5 | Unseen-perturbation lockbox on new seeds | Complete for strongest-only blur/resize; bounded positive transfer on TwoRoom/Reacher, weak/mixed PushT, neutral Cube | Checks whether the Gaussian-trained endpoint transfers to non-Gaussian visual stressors without upgrading the claim to universal robustness. |
| R6 | Representative unseen Phase-0 ACPC subset | Complete; `assets/paper1_data/unseen_phase0_acpc_subset.json`, `missing=0` | Tests whether selected unseen score movements also have paired ACPC/PCC/CRA/MAF movement; supports TwoRoom/Reacher, bounds Cube, leaves PushT mixed. |
| R7 | Training-objective follow-up | Paper2 / future work | Existing negative controls make naive consistency losses unsafe for Paper1. |

Recommended execution order status:

1. R1 is complete for seeds 3073/3074.
2. R4 remains development evidence and its seed-3072 readout is frozen.
3. R3 is complete and supports the Gaussian plateau-recovery claim.
4. R5 is complete for strongest-only blur/resize and remains bounded
   cross-stressor evidence.
5. R6 is complete and supports an appendix/boundary reading: TwoRoom/Reacher
   align, Cube bounds the claim, and PushT remains seed-sensitive.

## 10. Seed-3072 Unseen-Perturbation Pilot

This is a development pilot, not confirmatory evidence. Its purpose is to decide
whether Paper1 should add a bounded cross-stressor diagnostic extension.

Start with perturbations already supported by the code:

- `gaussian_blur`: existing eval and diagnostics support;
- `resize`: existing eval and diagnostics support.

Then add new families only if the existing pilot is interpretable:

- brightness/contrast;
- occlusion/cutout.

Use the same checkpoint grid:

```text
std_max = 0.00, 0.01, ..., 0.08
tasks = PushT, TwoRoom, Reacher, Cube
training seed = 3072
eval seeds = 42, 43, 44
```

Minimum pilot:

- all four tasks if compute is available;
- otherwise PushT + TwoRoom first, because they expose contact-heavy and
  low-dimensional/discrete regimes.

Readouts:

1. Closed-loop success under the new perturbation family.
2. Same-family paired diagnostics using the same rank formula where possible.
3. Selector comparison:
   Gaussian-frozen selector, same-family diagnostic selector, clean-only,
   fixed `std_max=0.08`, and no-noise.

Interpretation levels:

- Strong: Gaussian-frozen selector also hits the unseen-perturbation plateau.
- Moderate: Gaussian-frozen selector does not transfer, but the same diagnostic
  protocol with the new perturbation operator hits the plateau.
- Negative but useful: neither transfers; report stressor specificity.

Do not claim cross-perturbation generality from this pilot alone.

Fast smoke status (2026-06-29): a reduced `std=0.0` vs `std=0.08` check on
PushT and TwoRoom used strongest-severity blur/resize only and `30 x 3` eval
episodes. PushT showed little or no cross-stressor gain, while TwoRoom showed a
large positive signal for `std=0.08` under both blur and resize. This justified
the formal strongest-only pass below, but did not by itself justify a claim of
general cross-perturbation robustness.

Formal strongest-only status (2026-06-29): the four-task `std=0.0` vs
`std=0.08` pass used `num_eval=300` total episodes (`100` per seed for eval seeds
42/43/44), no-op plus `gaussian_blur` kernel 15 and `resize` factor 0.25. The
review artifacts are
`assets/paper1_data/unseen_origin_vs_std008_strongest_tworoom.json` and
`assets/paper1_data/unseen_origin_vs_std008_strongest_reacher.json`; together
they cover all 16 task/std/family eval summaries with `missing=0`.

| Task | Strongest stress | `std=0.0` | `std=0.08` | Stress delta | Origin delta | Drop improvement |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom | blur k=15 | 41.00 | 96.67 | +55.67 | +3.33 | +52.33 |
| TwoRoom | resize 0.25 | 44.33 | 96.67 | +52.33 | +3.33 | +49.00 |
| Reacher | blur k=15 | 19.00 | 69.67 | +50.67 | +16.00 | +34.67 |
| Reacher | resize 0.25 | 44.67 | 74.33 | +29.67 | +16.00 | +13.67 |
| Cube | blur k=15 | 53.67 | 51.00 | -2.67 | -4.67 | +2.00 |
| Cube | resize 0.25 | 54.33 | 55.33 | +1.00 | -4.67 | +5.67 |
| PushT | blur k=15 | 61.33 | 72.00 | +10.67 | +6.33 | +4.33 |
| PushT | resize 0.25 | 71.67 | 75.67 | +4.00 | +6.00 | -2.00 |

Interpretation: TwoRoom is the cleanest positive transfer signal: both unseen
stressors move from roughly 40--45% to 96.67%, and origin performance also
improves slightly. Reacher is also positive, but the origin checkpoint improves
by +16.00 pp, so the more conservative robustness readout is the residual drop
improvement: +34.67 pp under blur and +13.67 pp under resize. PushT is weak/mixed
and Cube is effectively neutral under this protocol. The correct conclusion is
task-dependent transfer on seed 3072, not a universal robustness recipe. Do not
promote this to the main paper claim unless independent training seeds reproduce
the TwoRoom/Reacher pattern.

## 10.1 Seed-3073/3074 Lockbox Result

Final status (2026-07-03): the independent training-seed lockbox is complete.
The detailed result note is `paper1/docs/LOCKBOX_RESULTS_20260703.md`.

Gaussian lockbox summary, averaged across seeds 3073 and 3074:

| Task | baseline obs 0.08 | std 0.08 obs 0.08 | std 0.08 gain | best obs 0.08 | std 0.08 regret to best |
|---|---:|---:|---:|---:|---:|
| TwoRoom | 70.33 | 96.83 | +26.50 | 97.67 | 0.83 |
| PushT | 8.67 | 84.17 | +75.50 | 87.00 | 2.83 |
| Reacher | 18.17 | 81.33 | +63.17 | 83.33 | 2.00 |
| Cube | 41.17 | 62.83 | +21.67 | 65.50 | 2.67 |

The same runs preserve the main diagnostic reading: baseline-to-std0.08
predictor rollout T8 drift drops by roughly 17.7x--51.2x, and clean/noisy CKA
rises to about 0.984--0.997. This is strong evidence for mechanism
localization and plateau triage, not a universal robustness-oracle claim.

Strongest-only unseen perturbation summary, averaged across seeds and both
families where relevant:

| Task | base stress | std 0.08 stress | stress delta | positive rows |
|---|---:|---:|---:|---:|
| TwoRoom | 51.58 | 88.58 | +37.00 | 4/4 |
| PushT | 59.08 | 63.08 | +4.00 | 3/4 |
| Reacher | 31.42 | 74.00 | +42.58 | 4/4 |
| Cube | 57.08 | 56.00 | -1.08 | 1/4 |

Interpretation: TwoRoom and Reacher replicate the positive transfer pattern;
PushT remains weak/mixed; Cube is neutral to slightly negative. This confirms
that blur/resize transfer is task-dependent. The representative unseen
Phase-0 ACPC subset below is now complete, so the result can be written as
bounded appendix evidence rather than a pending diagnostic question.

Representative unseen Phase-0 ACPC subset (2026-07-03):

| Task | stress | stress delta | drop improvement | delta ACPC-H/trans. | delta PCC | delta CRA | reading |
|---|---|---:|---:|---:|---:|---:|---|
| TwoRoom | blur k=15 | +36.83 | +35.17 | -0.590 | -42.7 | +0.567 | aligned |
| Reacher | blur k=15 | +48.50 | +28.00 | -1.770 | -47.0 | +0.568 | aligned |
| PushT | resize 0.25 | +2.33 | -4.67 | -0.249 | -6.4 | +0.009 | mixed |
| Cube | resize 0.25 | -1.83 | +1.33 | +0.088 | -0.1 | -0.042 | boundary |

Artifact: `assets/paper1_data/unseen_phase0_acpc_subset.json`. The subset keeps
goal images clean and applies blur/resize only to observation history. It
strengthens the TwoRoom/Reacher appendix story, but Cube and PushT prevent any
universal cross-perturbation predictor claim.

## 11. Eval-Logic Recommendation

Do not rewrite `run_trainer.sh` into a paper-specific grid runner. Keep it as
the single-checkpoint primitive and add a thin Paper1 orchestration layer around
it.

Current relevant behavior:

- `run_trainer.sh` already supports training skip via `skip_train=1`.
- Eval sweep supports `eval_corruption_type=gaussian_noise|gaussian_blur|resize`.
- Diagnostics default to the eval corruption family through
  `diagnostic_corruption_type=${eval_corruption_type}` unless overridden.
- Multiple eval seeds are already handled by `eval_seeds` and `eval_base_seed`.
- Non-Gaussian diagnostics are written to suffixed directories by
  `run_full_diagnostics.py`, avoiding overwrite of canonical Gaussian outputs.

Best next code change:

1. Add a Paper1-specific batch wrapper, for example:

```text
tools/paper1_unseen_eval_grid.py
```

or a shell wrapper if the cluster environment is simpler:

```text
run_paper1_unseen_eval_grid.sh
```

2. The wrapper should enumerate:

```text
task x std_max checkpoint x corruption_family x severity x apply_to x eval_seed
```

and call `run_trainer.sh` with an eval-only default:

```bash
skip_train=1
post_train_eval_mode=full
skip_diagnostics=1
eval_corruption_type=<family>
diagnostic_corruption_type=<family>
eval_corruption_apply_to=1
eval_seeds=3
eval_base_seed=42
ckpt_override=$DATA_ROOT/<task-root>/ckpt/<subdir>/<subdir>_epoch_10_object.ckpt
```

Then rerun only representative checkpoints with `skip_diagnostics=0` (the
wrapper flag is `--diagnostics`) after the closed-loop pilot shows a signal.
This keeps the first pass cheap enough to answer whether the unseen stressor is
worth expanding, without conflating eval coverage with diagnostic coverage.

3. Keep one corruption family per `run_trainer.sh` invocation. This preserves
simple filenames and summaries. The wrapper can run blur and resize as separate
passes.

4. Add a separate artifact builder after the runs, for example:

```text
tools/build_paper1_unseen_eval_artifact.py
```

It should parse each checkpoint's `eval_results/eval_summary.csv` and
diagnostics JSONs into a single canonical pilot artifact:

```text
assets/paper1_data/unseen_perturbation_pilot_seed3072.json
```

5. Only after this wrapper works should `eval.py` / `utils.py` be extended with
brightness/contrast or occlusion/cutout. Those are new corruption transforms,
not reasons to refactor the existing eval runner.

Do not put Paper1 plateau-selection logic inside `run_trainer.sh`. Selection and
paper-specific aggregation should live in `tools/`, because `run_trainer.sh` is
already serving training, eval, and diagnostics across several projects.

## 12. Immediate Checklist

- Seeds 3073/3074 Gaussian eval grid is complete.
- Strongest-only unseen perturbation artifacts for seeds 3073/3074 are complete.
- Lockbox result note has been written: `paper1/docs/LOCKBOX_RESULTS_20260703.md`.
- Before touching `main.tex`, run a claim audit deciding whether the lockbox
  belongs in the main text, appendix, or rebuttal package.
- Representative unseen diagnostics subset is complete:
  `assets/paper1_data/unseen_phase0_acpc_subset.json` (`missing=0`).
- Do not change `run_trainer.sh`; continue using the Paper1 wrappers and artifact
  builders for paper-specific orchestration.
