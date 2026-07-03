# Prospective Validation Remediation Summary

This artifact separates main completed validation evidence from appendix scope checks and reproducibility details.

## Three-seed unseen score aggregate

Scores include training seeds 3072/3073/3074 and are treated as a bounded unseen-stressor scope check.

| Task | selected stress | baseline stress | std0.08 stress | stress delta | drop improvement |
|---|---|---:|---:|---:|---:|
| TwoRoom | gaussian_blur | 47.67 +/- 5.44 | 90.78 +/- 5.38 | 43.11 +/- 8.90 | 40.89 +/- 8.23 |
| PushT | resize | 63.44 +/- 14.05 | 66.33 +/- 8.38 | 2.89 +/- 17.98 | -3.78 +/- 15.56 |
| Reacher | gaussian_blur | 22.00 +/- 3.78 | 71.22 +/- 1.10 | 49.22 +/- 3.29 | 30.22 +/- 4.16 |
| Cube | resize | 57.00 +/- 1.96 | 56.11 +/- 0.57 | -0.89 +/- 1.40 | 2.78 +/- 3.40 |

## Three-seed fixed-rule Gaussian diagnostic validation

Exact best hits: 2/12; within-5pp hits: 10/12; mean regret to best: 2.25 +/- 2.51 pp.

## Semantic margin pass-rate

| Task | std | pass-rate | ratio | margin |
|---|---:|---:|---:|---:|
| TwoRoom | 0.0 | 0.44 +/- 0.06 | 0.98 +/- 0.04 | -0.71 +/- 0.77 |
| TwoRoom | 0.08 | 1.00 +/- 0.00 | 29.21 +/- 1.46 | 15.77 +/- 0.27 |
| PushT | 0.0 | 0.27 +/- 0.14 | 0.87 +/- 0.07 | -2.48 +/- 1.66 |
| PushT | 0.08 | 1.00 +/- 0.00 | 23.66 +/- 2.28 | 15.06 +/- 0.13 |
| Reacher | 0.0 | 0.58 +/- 0.03 | 1.12 +/- 0.03 | 0.83 +/- 0.37 |
| Reacher | 0.08 | 1.00 +/- 0.00 | 61.26 +/- 3.37 | 16.50 +/- 0.23 |
| Cube | 0.0 | 0.25 +/- 0.06 | 0.85 +/- 0.01 | -3.32 +/- 0.19 |
| Cube | 0.08 | 1.00 +/- 0.00 | 34.44 +/- 4.02 | 15.99 +/- 0.28 |

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

## Semantic state proxies

| Task | semantic factor | available source | status |
|---|---|---|---|
| PushT | T-block pose/contact relative to pusher | dataset state column is configured for PushT analysis | reported in semantic_margin_passrate_lewm_three_seed.json; broader pair construction remains future work |
| TwoRoom | room/doorway/topology and target-region relation | derive from trajectory position/proprio and map topology | reported with pos_agent proxy in semantic_margin_passrate_lewm_three_seed.json; topology-specific extraction remains future work |
| Reacher | joint/target geometry and end-effector-to-target relation | qpos/goal_qpos are used by eval set-state callables | reported in semantic_margin_passrate_lewm_three_seed.json; broader pair construction remains future work |
| Cube | cube pose and gripper-object/goal relation | qpos plus goal block position/quaternion are used by eval callables | reported in semantic_margin_passrate_lewm_three_seed.json; broader pair construction remains future work |

## Remaining validation work

- Extend the fixed diagnostic rule to additional perturbation families and method families after this three-seed Gaussian validation.
- Broaden semantic-pair construction beyond one state proxy per task if the claim is expanded beyond matched Gaussian diagnostics.
- Keep training-seed uncertainty as the primary behavior statistic and evaluation-seed variance as the secondary decomposition.
