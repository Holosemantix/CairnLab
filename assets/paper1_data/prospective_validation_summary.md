# Prospective Validation Remediation Summary

This artifact separates completed validation evidence from the protocol pieces that are now frozen but still require state-margin or full held-out-grid runs.

## Three-seed unseen score aggregate

Scores include training seeds 3072/3073/3074. Diagnostics below remain the matched 3073/3074 slice because matching Phase-0 diagnostic rows are released for those lockbox seeds.

| Task | selected stress | baseline stress | std0.08 stress | stress delta | drop improvement |
|---|---|---:|---:|---:|---:|
| TwoRoom | gaussian_blur | 47.67 +/- 5.44 | 90.78 +/- 5.38 | 43.11 +/- 8.90 | 40.89 +/- 8.23 |
| PushT | resize | 63.44 +/- 14.05 | 66.33 +/- 8.38 | 2.89 +/- 17.98 | -3.78 +/- 15.56 |
| Reacher | gaussian_blur | 22.00 +/- 3.78 | 71.22 +/- 1.10 | 49.22 +/- 3.29 | 30.22 +/- 4.16 |
| Cube | resize | 57.00 +/- 1.96 | 56.11 +/- 0.57 | -0.89 +/- 1.40 | 2.78 +/- 3.40 |

## Matched held-out unseen diagnostic validation slice

Split: training seeds 3073/3074; unseen perturbations gaussian_blur and resize; fixed comparison std_max 0.0 -> 0.08.

| Metric | rho vs stress delta | r vs stress delta | rho vs drop improvement | r vs drop improvement | n |
|---|---:|---:|---:|---:|---:|
| ACPC-H/trans. delta | 0.90 | 0.88 | 0.71 | 0.68 | 8 |
| PCC delta | 0.88 | 0.93 | 0.81 | 0.88 | 8 |
| CRA delta | 0.86 | 0.90 | 0.90 | 0.88 | 8 |
| MAF delta | 0.95 | 0.92 | 0.71 | 0.77 | 8 |
| Composite signed-rank rule | 0.92 | 0.93 | 0.81 | 0.83 | 8 |

Top-4 agreement: composite signed-rank top-k hits 4/4 for stress-success delta and 4/4 for drop improvement.

## Semantic discriminability protocol ledger

| Task | semantic factor | available source | release status |
|---|---|---|---|
| PushT | T-block pose/contact relative to pusher | dataset state column is configured for PushT analysis | protocol frozen; state-margin run still required for a result table |
| TwoRoom | room/doorway/topology and target-region relation | derive from trajectory position/proprio and map topology | protocol frozen; task-topology extraction still required for a result table |
| Reacher | joint/target geometry and end-effector-to-target relation | qpos/goal_qpos are used by eval set-state callables | protocol frozen; state-margin run still required for a result table |
| Cube | cube pose and gripper-object/goal relation | qpos plus goal block position/quaternion are used by eval callables | protocol frozen; state-margin run still required for a result table |

## Remaining validation work

- Run the fixed ACPC/PCC/CRA/MAF rule on full held-out training-seed checkpoint grids, not only the current matched diagnostic slice.
- Run the task-semantic state-margin probes defined in this artifact and report pass rates before claiming semantic discriminability results.
- Use the three-training-seed Gaussian table as the primary behavior statistic; keep evaluation-seed variance as a secondary decomposition.
