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

Required diagnostic rerun before final lockbox analysis:

```bash
python -m tools.paper1_phase0_acpc \
  --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
  --noise-std 0.08 \
  --clean-goal \
  --seed 9101 \
  --out assets/paper1_data/heldout_selection_phase0_seed9101.json
```

The existing Phase-0 artifact mixes observation+goal corruption; the lockbox
selector should use observation-only corruption with a clean goal to match the
primary target.

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
| R1 | Independent LeWM training seeds 3073/3074 | In progress | Main-conference reviewers will not accept evaluation seeds as a substitute for training-run variability. |
| R2 | Frozen seed-3072 robustness-triage selector | Development analysis done; selector frozen in this document | Converts ACPC/CRA/MAF from post-hoc explanation into a testable plateau-triage protocol. |
| R3 | Lockbox application to 3073/3074 | Pending new seed artifacts | This is the confirmatory test. Do not tune the selector after looking at these results. |
| R4 | Seed-3072 unseen-perturbation pilot | Pending | Tests whether the protocol is Gaussian-specific or reusable with a different paired-corruption operator. |
| R5 | Unseen-perturbation lockbox on new seeds | Conditional | Only needed if R4 shows interpretable signal and R3 is strong enough to justify expansion. |
| R6 | Training-objective follow-up | Paper2 / future work | Existing negative controls make naive consistency losses unsafe for Paper1. |

Recommended execution order:

1. Finish R1.
2. Run R4 on seed 3072 while R1 is finishing.
3. Freeze the unseen-perturbation pilot protocol if R4 shows a usable signal.
4. Apply R2 exactly as written to 3073/3074 once the artifacts exist.
5. Decide whether R5 is worth the compute after seeing R3 and the R4 pilot.

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

- Wait for LeWM seeds 3073/3074 to finish.
- Build canonical eval JSONs for the new seeds.
- Rerun Phase-0 diagnostics with `--clean-goal`.
- Apply the frozen selector without further tuning.
- Write a lockbox result note before touching `main.tex`.
- Decide whether the result qualifies for main-text integration.
- In parallel, run the seed-3072 unseen-perturbation pilot through a wrapper
  around `run_trainer.sh`, not by changing the training/eval entrypoint.
- Use `tools/paper1_unseen_eval_grid.py --dry-run` first, then run the same
  wrapper without `--dry-run`; aggregate with
  `tools/build_paper1_unseen_eval_artifact.py`.
