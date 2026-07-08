# Paper 1 top-conference weak-reject response plan (2026-06-26)

## Verdict being addressed

The review reads the manuscript as a careful diagnostic paper but gives a weak-reject / borderline-reject judgment for a main-conference submission. The main concerns are that ACPC is currently a modest formalization, the diagnostic metrics are mostly post-hoc localization rather than held-out predictors, the empirical evidence lacks independent training seeds and broader perturbation families, and the comparisons do not include stronger non-JEPA visual-control baselines.

This plan separates what can be fixed in the manuscript now from experiments that require new training or new evaluation.

## Non-negotiable boundaries

- Do not claim that ACPC is a new training objective in the current paper.
- Do not claim that ACPC metrics independently predict robustness across held-out checkpoints, seeds, or perturbation families.
- Do not claim broad visual robustness from the Gaussian-noise sweep.
- Do not treat the fixed-candidate theorem as a proof about CEM resampling, repeated replanning, or closed-loop trajectories.
- Do not replace missing independent training seeds with evaluation-seed standard deviations.
- Do not describe proxy rank, transition-resolution, or ID-probe diagnostics as oracle task-margin guarantees.

## Can be fixed now without new training

### 1. Reposition ACPC novelty as a diagnostic organization principle

Action: tighten the introduction, contributions, and conclusion so that ACPC is presented as a way to organize an encoder--predictor--planner diagnostic, not as a deep theoretical contribution or a solved robustness method.

Acceptance: the paper should say that the formal fixed-candidate result is a sufficient-condition link that motivates readouts, while the empirical contribution is the release package and bounded diagnostic evidence.

### 2. Make the diagnostic-predictiveness limitation explicit

Action: promote the partial-correlation null/weak residual result from a caveat into an explicit finding: after conditioning on training-noise level, the scalar fragility metric and rollout drift do not function as robust model-selection or prediction oracles.

Acceptance: readers should not infer that ACPC/fragility can select a robust checkpoint without closed-loop evaluation.

### 3. Clarify post-hoc ACPC-basin status

Action: state in the ACPC-basin section and discussion that the representative rows are selected after the closed-loop sweep, so the basin analysis localizes an already observed recovery rather than predicting it prospectively.

Acceptance: no sentence should imply held-out or pre-registered checkpoint selection.

### 4. Bound the perturbation claim

Action: explicitly state that Gaussian-noise training and Gaussian-noise evaluation are matched, while blur is eval-only and shows perturbation-specific ordering. The current paper should not claim cross-perturbation robustness.

Acceptance: the blur appendix and discussion should frame blur as evidence that the failure is not only a Gaussian artifact, but also evidence that task ordering and recovery do not transfer automatically.

### 5. Bound the baseline comparison

Action: make clear that PLDM is a second JEPA-family or latent-predictive family replication check, not a comprehensive strong-baseline comparison against DrQ-v2, Dreamer/TD-MPC2, frozen-pretrained visual backbones, or robust visual MPC interventions.

Acceptance: the paper can cite these families as context and future comparison targets, but must not imply it has beaten or exhausted them.

### 6. Strengthen the discriminability-guard limitation

Action: state that effective rank, transition-resolution, and ID probe are sanity checks against coarse collapse, not task-semantic guarantees. Add examples of stronger guards that would require new artifacts: PushT keypoints/contact/pose, TwoRoom topology/doorway state, Reacher joint-target relation, and Cube pose/goal relation.

Acceptance: the current proxy guard remains useful but explicitly bounded.

### 7. Clarify theory-to-CEM gap

Action: add a direct sentence connecting the fixed-candidate theorem to the existing Phase-0 shared-candidate readouts and explaining why it does not prove actual CEM action stability.

Acceptance: the paper should avoid a theory/implementation mismatch by naming CEM resampling and repeated replanning as outside the theorem.

### 8. Handle double-blind venue risk outside the arXiv build

Action: add a submission-note paragraph in the paper README explaining that the current source is arXiv/non-anonymous and that a double-blind venue variant must remove author names, acknowledgements, public GitHub URL, and any self-identifying artifact pointers.

Acceptance: arXiv readiness remains non-anonymous; double-blind risk is documented for future venue packaging.

## Requires future experiments, not manuscript-only edits

- Independent training seeds for the LeWM and PLDM grids.
- Held-out diagnostic validation: choose checkpoints by ACPC/CRA/MAF without seeing the target closed-loop noise endpoint.
- Held-out perturbation families: brightness/contrast, occlusion, camera shift, background distractors, crop/resize, sensor noise variants, and goal perturbations.
- A trained paired predictive-dynamics consistency objective with a discriminability guard.
- Strong non-JEPA or non-matching baselines such as DrQ-v2, Dreamer/TD-MPC2, frozen/pretrained visual backbones, or robust visual MPC intervention methods.
- Task-semantic discriminability metrics: keypoint/contact/pose for PushT, doorway/topology for TwoRoom, joint-target relation for Reacher, and cube pose/goal relation for Cube.

## Concrete manuscript edits

1. Abstract: make the "diagnostic framework and release package" sentence more explicit about boundedness and remove any reading that recovered checkpoints imply a generally predictive metric.
2. Contributions: revise C2/C3 so the theorem is a support layer and PLDM is a bounded replication check.
3. Experiments preamble: add that the study does not compare against strong visual-control baselines and should not be read as benchmark superiority.
4. ACPC basin: add post-hoc selection and non-predictive status directly before the compact table.
5. Cross-checkpoint correlations: label the result as a negative diagnostic-predictiveness check.
6. Discussion scope: add a concise "main-conference evidence still missing" sentence without sounding defensive.
7. Negative checks and next steps: name the strongest next experiments, but only as future work.
8. README: add double-blind packaging note.

## Final validation checklist

- Targeted `rg` checks on `main.tex` and `README.md` find no overclaiming phrases such as "predicts robustness", "guarantees robustness", "general visual robustness", or "strong baseline superiority".
- `python3 -m tools.check_paper1_consistency` passes.
- `bash build.sh --clean` passes.
- `ALLOW_AUTHOR_PLACEHOLDER=1 bash paper1/check_arxiv_ready.sh` passes for the current arXiv-style draft.
- The final diff touches only planning/manuscript/readme/PDF files, unless a checker reveals a true artifact inconsistency.
