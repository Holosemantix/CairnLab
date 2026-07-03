# Paper1 Lockbox Results — Seeds 3073/3074

Status: development review note, not merged into `main.tex`.

This note records the independent-training-seed lockbox pass after the original
seed-3072 development analysis. The purpose is to decide whether the Paper1
claim should stay as a controlled Gaussian diagnostic paper, move toward a
diagnostic-triage claim, or expand to unseen perturbations.

## Scope

- Training seeds: `3073`, `3074`.
- Tasks: TwoRoom, PushT, Reacher, Cube.
- Gaussian sweep: LeWM baseline plus `std_max = 0.01 ... 0.08`, evaluated with
  seeds `42/43/44`, `100` episodes per eval seed.
- Unseen strongest-only stress: `std_max in {0.0, 0.08}`,
  `gaussian_blur` kernel `15`, and `resize` factor `0.25`.
- Unseen artifacts:
  - `assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json`
  - `assets/paper1_data/unseen_origin_vs_std008_strongest_s3074.json`

## Gaussian Lockbox

The Gaussian lockbox strongly reproduces the canonical Paper1 pattern:
no-noise checkpoints suffer large observation-noise cliffs, while noise-trained
checkpoints recover into broad task-dependent plateaus.

| Task | seed | baseline obs 0.08 | best obs 0.08 | best std | std 0.08 obs 0.08 | std 0.08 gain over baseline |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom | 3073 | 71.67 | 97.33 | 0.06 | 97.33 | +25.67 |
| TwoRoom | 3074 | 69.00 | 98.00 | 0.05 | 96.33 | +27.33 |
| PushT | 3073 | 2.00 | 86.67 | 0.04 | 81.00 | +79.00 |
| PushT | 3074 | 15.33 | 87.33 | 0.08 | 87.33 | +72.00 |
| Reacher | 3073 | 17.67 | 83.33 | 0.07 | 82.33 | +64.67 |
| Reacher | 3074 | 18.67 | 83.33 | 0.05 | 80.33 | +61.67 |
| Cube | 3073 | 39.00 | 65.67 | 0.05 | 64.67 | +25.67 |
| Cube | 3074 | 43.33 | 65.33 | 0.03 | 61.00 | +17.67 |

Average across the two lockbox training seeds:

| Task | baseline obs 0.08 | std 0.08 obs 0.08 | gain | best obs 0.08 | std 0.08 regret to best |
|---|---:|---:|---:|---:|---:|
| TwoRoom | 70.33 | 96.83 | +26.50 | 97.67 | 0.83 |
| PushT | 8.67 | 84.17 | +75.50 | 87.00 | 2.83 |
| Reacher | 18.17 | 81.33 | +63.17 | 83.33 | 2.00 |
| Cube | 41.17 | 62.83 | +21.67 | 65.50 | 2.67 |

Reading:

- The lockbox supports the main Gaussian diagnostic claim.
- It also supports the plateau wording: the best observed std is task- and
  seed-dependent (`0.03`--`0.08`), so the result should not be written as a
  universal `std=0.08` optimum.
- `std=0.08` itself is still a strong endpoint reference: it is within roughly
  three points of the best mean on average, but PushT seed 3073 and Cube seed
  3074 are reminders to use plateau language rather than point-best language.

## Gaussian Diagnostics

The existing per-checkpoint diagnostics move in the same direction as closed-loop
recovery. The clearest common movement is a large reduction in multi-step
predictor rollout drift, with high clean/noisy CKA at `std=0.08`.

| Task | seed | T8 rollout drift baseline | T8 rollout drift std 0.08 | fold reduction | CKA baseline | CKA std 0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom | 3073 | 18.891 | 0.853 | 22.2x | 0.167 | 0.984 |
| TwoRoom | 3074 | 18.597 | 0.605 | 30.8x | 0.202 | 0.989 |
| PushT | 3073 | 19.914 | 1.111 | 17.9x | 0.434 | 0.996 |
| PushT | 3074 | 18.034 | 1.021 | 17.7x | 0.564 | 0.997 |
| Reacher | 3073 | 16.236 | 0.371 | 43.7x | 0.357 | 0.997 |
| Reacher | 3074 | 16.944 | 0.331 | 51.2x | 0.422 | 0.997 |
| Cube | 3073 | 20.044 | 0.622 | 32.2x | 0.278 | 0.993 |
| Cube | 3074 | 19.602 | 0.652 | 30.1x | 0.355 | 0.993 |

Reading:

- This strengthens the mechanism-localization story.
- It does not by itself prove that the diagnostic predicts robustness. The
  correct claim remains diagnostic triage / mechanism localization, not a
  universal robustness oracle.
- PushT does not show a harmful rank-compression narrative in the lockbox
  diagnostics, matching the corrected Paper1 interpretation.

## Unseen Strongest-Only Stress

The unseen perturbation pass reproduces the seed-3072 boundary:
TwoRoom and Reacher transfer strongly, PushT is weak/mixed, and Cube is neutral
to slightly negative.

| Task | family | baseline stress | std 0.08 stress | stress delta | drop improvement |
|---|---|---:|---:|---:|---:|
| TwoRoom | gaussian_blur | 51.00 | 87.83 | +36.83 | +35.17 |
| TwoRoom | resize | 52.17 | 89.33 | +37.17 | +35.50 |
| PushT | gaussian_blur | 58.83 | 64.50 | +5.67 | -1.50 |
| PushT | resize | 59.33 | 61.67 | +2.33 | -4.67 |
| Reacher | gaussian_blur | 23.50 | 72.00 | +48.50 | +28.00 |
| Reacher | resize | 39.33 | 76.00 | +36.67 | +16.17 |
| Cube | gaussian_blur | 55.83 | 55.50 | -0.33 | +2.83 |
| Cube | resize | 58.33 | 56.50 | -1.83 | +1.33 |

Average across both unseen families:

| Task | baseline stress | std 0.08 stress | stress delta | positive rows |
|---|---:|---:|---:|---:|
| TwoRoom | 51.58 | 88.58 | +37.00 | 4/4 |
| PushT | 59.08 | 63.08 | +4.00 | 3/4 |
| Reacher | 31.42 | 74.00 | +42.58 | 4/4 |
| Cube | 57.08 | 56.00 | -1.08 | 1/4 |

Reading:

- The unseen pass is useful, but it should not become a broad cross-perturbation
  robustness claim.
- It supports a bounded statement: Gaussian noise training can transfer to some
  unseen perturbation families on some tasks, especially TwoRoom and Reacher,
  but transfer is task-dependent and can be weak or absent.
- The follow-up unseen Phase-0 ACPC subset is now complete for representative
  positive and boundary cases. It supports TwoRoom/Reacher as aligned
  score-plus-diagnostic transfer, keeps Cube as a negative boundary, and leaves
  PushT mixed because the two seeds disagree under resize.

## Unseen Phase-0 ACPC Subset

Artifact: `assets/paper1_data/unseen_phase0_acpc_subset.json` (`missing=0`).
The runner is `run_paper1_unseen_phase0_acpc_subset.sh`; it keeps goal images
clean and applies the unseen stress to observation history, matching the
strongest-only unseen eval protocol.

| Task | stress | stress delta | drop improvement | delta ACPC-H/trans. | delta PCC | delta CRA | reading |
|---|---|---:|---:|---:|---:|---:|---|
| TwoRoom | blur k=15 | +36.83 | +35.17 | -0.590 | -42.7 | +0.567 | aligned |
| Reacher | blur k=15 | +48.50 | +28.00 | -1.770 | -47.0 | +0.568 | aligned |
| PushT | resize 0.25 | +2.33 | -4.67 | -0.249 | -6.4 | +0.009 | mixed |
| Cube | resize 0.25 | -1.83 | +1.33 | +0.088 | -0.1 | -0.042 | boundary |

Reading: the unseen diagnostics strengthen the bounded appendix story, not the
main Gaussian claim. TwoRoom/Reacher show score and diagnostic alignment; Cube
is the negative boundary; PushT remains seed-sensitive and should not be used as
a broad transfer example.

## Recommendation

1. Keep Paper1 as a reframing + diagnostic paper centered on Gaussian ACPC.
2. Add the lockbox result as a strengthening note / appendix candidate only
   after a claim audit, emphasizing independent training-seed replication and
   plateau recovery.
3. Keep unseen perturbation as bounded appendix/development evidence. The
   completed diagnostics subset supports TwoRoom/Reacher and bounds PushT/Cube,
   but it does not justify a universal cross-perturbation claim.
4. Do not shift Paper1 into a method paper. The next method direction should
   remain Paper2, where selective/action-conditioned objectives can be compared
   against ordinary noise training under matched training seeds.
