# Paper 1 Data Manifest

This manifest documents the released evaluation aggregate for Paper 1:

- Canonical aggregate: `assets/paper1_data/canonical_evals_20260517.json`
- Schema: `assets/paper1_data/canonical_evals_20260517.schema.json`
- Canonical diagnostics: `assets/paper1_data/canonical_diagnostics_20260517.json` (2026-06-10 revision: the TwoRoom/PushT representative diagnostics used by `tab:diag-base-vs-best` were re-extracted from the per-checkpoint diagnostics after an audit found they duplicated the `*_lewm_hetero_default` values; see the JSON `metadata.table3_revision_20260610` note)
- Diagnostics schema: `assets/paper1_data/canonical_diagnostics_20260517.schema.json`
- External baseline sanity check: `assets/paper1_data/canonical_external_baselines_20260520.json`
- External baseline schema: `assets/paper1_data/canonical_external_baselines_20260520.schema.json`
- **PLDM cross-method replication aggregate**: `assets/paper1_data/canonical_evals_pldm_20260522.json` (36 ckpts: 4 tasks × 9 configs)
- **PLDM cross-method replication diagnostics**: `assets/paper1_data/canonical_diagnostics_pldm_20260522.json`
- **PLDM full diagnostics**: `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json` (full diagnostics-summary rows for the same 36 PLDM ckpts)
- **PLDM full diagnostics schema**: `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.schema.json`
- **Cross-method correlations**: `assets/paper1_data/cross_method_corr_pldm_20260522.json` (within-LeWM / within-PLDM / joint partial Spearman; consumed by the PLDM appendix)
- **PLDM full ACPC basin replication**: `assets/paper1_data/acpc_basin_diagnostics_pldm.json` (36 rows: 4 tasks × 9 PLDM configs)
- **Partial-correlation bootstrap CIs**: `assets/paper1_data/partial_corr_bootstrap_20260523.json` (95% percentile bootstrap intervals for the LeWM, PLDM, and joint partial-correlation tables)
- **Phase-0 paired ACPC diagnostics**: `assets/paper1_data/acpc_phase0_clean_goal_seed9101.json` (72 rows: LeWM + PLDM, 4 tasks × 9 configs; clean-goal observation-noise run consumed by the Phase-0 appendix)
- **Gaussian-noise ACPC basin diagnostics**: `assets/paper1_data/acpc_basin_diagnostics.json` (LeWM 36 ckpts, epoch-10 model objects, paired clean/noised views at Gaussian std 0.01..0.08; source for the ACPC basin table)
- **No-noise-baseline blur sanity check**: `assets/paper1_data/canonical_blur_baselines_20260523.json` (LeWM + PLDM baselines trained without input-noise augmentation, 4 tasks, blur eval only)
- **No-noise-baseline blur schema**: `assets/paper1_data/canonical_blur_baselines_20260523.schema.json`
- **Seed-3072 strongest-only unseen-perturbation pilot**: `assets/paper1_data/unseen_origin_vs_std008_strongest_tworoom.json` and `assets/paper1_data/unseen_origin_vs_std008_strongest_reacher.json` (review artifacts, not paper-facing release evidence; together cover 4 tasks × 2 std keys × 2 stress families with no-op plus blur k=15 / resize factor 0.25)
- **Seed-3073/3074 strongest-only unseen-perturbation lockbox artifacts**: `assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json` and `assets/paper1_data/unseen_origin_vs_std008_strongest_s3074.json` (review artifacts for independent training-seed follow-up; each covers 4 tasks × 2 std keys × 2 stress families, `missing=0`)
- **Unseen Phase-0 ACPC subset artifact**: `assets/paper1_data/unseen_phase0_acpc_subset.json` (review artifact: 8 selected seed/task/stress case rows joining 16 clean-goal Phase-0 diagnostic rows with strongest-only unseen eval scores; `missing=0`)
- **Three-training-seed Gaussian lockbox summary**: `assets/paper1_data/training_seed_gaussian_lockbox.json` and `.md` (canonical seed 3072 plus completed lockbox seeds 3073/3074; reports training-seed mean/std for the main observation-noise 0.08 endpoint)
- **Validation remediation summary**: `assets/paper1_data/prospective_validation_summary.json` and `.md` (summarizes the held-out seed/perturbation validation slice, full-grid frozen-rule audit summary, and semantic state-margin protocol ledger)
- Scope: 36 LeWM canonical checkpoints = 4 tasks × 9 configs (`base` + `std_max` 0.01..0.08); 36 PLDM checkpoints on the same grid; completed LeWM Gaussian lockbox sweeps for independent training seeds 3073/3074 are summarized separately.
- Evaluation protocol: **3 evaluation seeds** (`42`, `43`, `44`) × **100 trajectories per seed** for each checkpoint/seed point.
- Seed clarification: canonical LeWM/PLDM grid tables use evaluation-seed variance; the main LeWM Gaussian endpoint now additionally reports independent training-seed mean/std across seeds 3072/3073/3074 via `training_seed_gaussian_lockbox.json`.

## Release Provenance Notes

- Paper-facing main evidence: `canonical_evals_20260517.json`, `training_seed_gaussian_lockbox.json`, `prospective_validation_summary.json`, `canonical_diagnostics_20260517.json`, `acpc_basin_diagnostics.json`, `canonical_evals_pldm_20260522.json`, `canonical_diagnostics_pldm_20260522.json`, `canonical_full_diagnostics_pldm_20260523.json`, `acpc_basin_diagnostics_pldm.json`, `partial_corr_bootstrap_20260523.json`, and `acpc_phase0_clean_goal_seed9101.json`.
- Scope-boundary / sanity artifacts: `canonical_blur_baselines_20260523.json` is eval-only blur stress; `acpc_phase0_diagnostics.json` is the archived observation+goal Phase-0 sanity run; `target_view_closed_loop_summary.json` is a negative target-view ablation; `canonical_external_baselines_20260520.json` is retained for backward-compatible sanity checks. The strongest-only unseen-perturbation artifacts and `unseen_phase0_acpc_subset.json` support the held-out seed/perturbation validation slice summarized in `prospective_validation_summary.json`. The full task-semantic state-margin probes are specified but not yet reported as result artifacts.
- Contamination fix: the 2026-06-10 audit found that the TwoRoom and PushT representative diagnostic rows in `canonical_diagnostics_20260517.json` duplicated heteroscedastic-loss diagnostics. The affected representative fields were re-extracted from the intended per-checkpoint `diagnostics_summary.json` files. The release checker now guards these values so the PushT noise-sweep row cannot regress to the heteroscedastic `rank 76.4 -> 42.9` narrative.
- Manual revision status: no released JSON artifact is hand-edited for paper prose. The only documented corrective revision is the representative-diagnostics re-extraction above, recorded in JSON metadata and checked by `tools/check_paper1_consistency.py`.

| Artifact | Role | SHA-256 |
|---|---|---|
| `canonical_evals_20260517.json` | LeWM evaluation aggregate | `394c21142311e628232e510d0087b17828ab78d973727cff6b049cb50ed98e1a` |
| `canonical_diagnostics_20260517.json` | LeWM diagnostics and representative rows | `8012fd3bc5fb445bd5d00ea78d3c5df30f57c10f69b5175f8479b48370517443` |
| `acpc_basin_diagnostics.json` | LeWM Gaussian ACPC basin | `e0e468d2d7a94666e6bbcf8dafd24f32fe2dade03d66deb5619a86145a8dc521` |
| `canonical_evals_pldm_20260522.json` | PLDM evaluation aggregate | `e9bf3a49b91f3d17151db2cd94c696cd56f53d8c2be9670351f059ebb671df7a` |
| `canonical_diagnostics_pldm_20260522.json` | PLDM predictor diagnostics | `efb15cc8baafd1f80e5d5b67ffce59666ef909f7a4106dd50cc6a7c9fcf4c536` |
| `canonical_full_diagnostics_pldm_20260523.json` | PLDM five-layer diagnostics | `6a5b2ae47b09b4bd6fd6fba87846e7d9484e6beea2cfe24f75380c238c73fc7d` |
| `acpc_basin_diagnostics_pldm.json` | PLDM Gaussian ACPC basin | `dd6aeaa3e793ce09294b049b31ff2f7791c7b83fe0a6a375e6adba806f23e6e2` |
| `partial_corr_bootstrap_20260523.json` | Bootstrap CI aggregate | `e6cbba0893defd152b540150dcf86ee091fbe5cf4061131871228b6a59d51465` |
| `acpc_phase0_clean_goal_seed9101.json` | Clean-goal Phase-0 ACPC/PCC/CRA/MAF diagnostics | `0207daddc972bfd66829d5a521101284daed3cc60933046fa13766da73cde021` |
| `heldout_selection_phase0_seed9101.json` | LeWM component for clean-goal Phase-0 diagnostics | `4d8daf17a0d43b57c996bd3fc0ab4b1cba830844c9bbb8dae0b002cee2fb409c` |
| `heldout_selection_phase0_pldm_seed9101.json` | PLDM component for clean-goal Phase-0 diagnostics | `e46db5f49011ca199f213096a813db1ca69c2c6023ffde123d11f60dce751f91` |
| `acpc_phase0_diagnostics.json` | Archived observation+goal Phase-0 sanity run | `9654759b576216b7249a3bf5e2ee7b778318cf4de22babf3395a0757b3e644fd` |
| `target_view_closed_loop_summary.json` | Negative target-view ablation | `04f75ad72543fb98d51304a6dec12ceb1b2dc099e915e24a49862ed3451744d0` |
| `canonical_blur_baselines_20260523.json` | Eval-only blur sanity check | `8e4c18d9f354a585770e6eb389e4ceb1b449eea7a5e7758af40a324878e0700b` |
| `unseen_origin_vs_std008_strongest_tworoom.json` | Seed-3072 strongest-only unseen pilot, TwoRoom | `e64640cd902e65c40215dd54670120efa46e78c3215c88d8c5fcb5dd818b02c4` |
| `unseen_origin_vs_std008_strongest_reacher.json` | Seed-3072 strongest-only unseen pilot, Reacher/Cube/PushT | `8a9ae1a9a058770c647535e8ea4bb5b9f3cc1eb5b977499e0b26aba2f94bf34a` |
| `unseen_origin_vs_std008_strongest_s3073.json` | Seed-3073 strongest-only unseen lockbox | `2d69d99e742ce5cffe6c37eec004a5ac255289106d9e847b34fabfd44cf92ee7` |
| `unseen_origin_vs_std008_strongest_s3074.json` | Seed-3074 strongest-only unseen lockbox | `a1015cdbb8062f60fc7835d87ed237c62b8003427964266aa63d2637a771ae5b` |
| `unseen_phase0_acpc_subset.json` | Unseen Phase-0 ACPC subset review artifact | `0fa4138533867ee6bd3eb768b0d0d84205de10b36e8c1bc25bdd6db94421f8b3` |
| `unseen_phase0_acpc_subset.schema.json` | Unseen Phase-0 ACPC subset schema | `8f688e77a27fbf69bb750bf26f90e8745b0b2e369ad6e62128b4a3040095b85f` |
| `training_seed_gaussian_lockbox.json` | Three-training-seed Gaussian lockbox summary | `526ff2fad2ba86c3c865341ae4bc9db9fad52f4825bf12818c3ab54a2af4aabb` |
| `prospective_validation_summary.json` | Validation remediation summary and semantic protocol ledger | `4c203c6c34baa42aacedf94ff46ea15c1987994dd6e4122daf9e31a61d79b732` |

## Data Semantics

- Unit: every success-rate field is stored in **percentage points** on `[0, 100]`
- Aggregation: each metric stores `values = [seed42, seed43, seed44]`, `mean`, and `std`
- `std` convention: the JSON stores the **population standard deviation** across the 3 evaluation seeds (`ddof = 0`)
- Conditions released per checkpoint:
  - `clean`
  - `goal_std0.03`, `goal_std0.05`, `goal_std0.08`
  - `pixels_std0.03`, `pixels_std0.05`, `pixels_std0.08`
  - `pixels_goal_std0.03`, `pixels_goal_std0.05`, `pixels_goal_std0.08`
- Paper 1 primary corrupted endpoint: `pixels_std0.08` (observation pixels corrupted, goal image kept clean). `pixels_goal_std0.08` is retained as a stronger full-visual-stream stress condition.
- Canonical lookup key for portability: use `subdir`
  - The JSON also stores an absolute local `path`, but downstream tools should not rely on that exact absolute prefix

## Per-Seed File Pattern

For every checkpoint subdirectory below, the raw per-seed metrics are expected at:

`<ckpt>/eval_results/<cond>_seed42_metrics.txt`

`<ckpt>/eval_results/<cond>_seed43_metrics.txt`

`<ckpt>/eval_results/<cond>_seed44_metrics.txt`

where `<cond>` is one of:

- `clean`
- `goal_std0.03`, `goal_std0.05`, `goal_std0.08`
- `pixels_std0.03`, `pixels_std0.05`, `pixels_std0.08`
- `pixels_goal_std0.03`, `pixels_goal_std0.05`, `pixels_goal_std0.08`

## Task Roots

Set `DATA_ROOT` to the machine-local prefix that contains the released
`lewm-*` task directories. Keep that prefix outside the manifest and commands;
the released task roots below stay relative to `DATA_ROOT` so the same
artifacts can be moved between machines.

| Task | Local ckpt root |
|---|---|
| TwoRoom | `$DATA_ROOT/lewm-tworooms/ckpt` |
| PushT | `$DATA_ROOT/lewm-pusht/ckpt` |
| Reacher | `$DATA_ROOT/lewm-reacher/ckpt` |
| Cube | `$DATA_ROOT/lewm-cube/ckpt` |

## Released Checkpoints

| Task | `std_max` | Canonical `subdir` | Raw per-seed file pattern |
|---|---:|---|---|
| TwoRoom | 0.0 | `tworoom_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.01 | `tworoom_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.02 | `tworoom_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.03 | `tworoom_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.04 | `tworoom_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.05 | `tworoom_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.06 | `tworoom_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.07 | `tworoom_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.08 | `tworoom_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.0 | `pusht_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.01 | `pusht_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.02 | `pusht_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.03 | `pusht_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.04 | `pusht_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.05 | `pusht_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.06 | `pusht_lewm_noise_0to006_p1_20260507` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.07 | `pusht_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.08 | `pusht_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.0 | `reacher_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.01 | `reacher_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.02 | `reacher_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.03 | `reacher_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.04 | `reacher_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.05 | `reacher_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.06 | `reacher_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.07 | `reacher_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.08 | `reacher_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.0 | `cube_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.01 | `cube_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.02 | `cube_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.03 | `cube_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.04 | `cube_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.05 | `cube_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.06 | `cube_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.07 | `cube_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.08 | `cube_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |

## Consumer Notes

- `tools/paper1_figs.py` treats `assets/paper1_data/canonical_evals_20260517.json` as the source of truth for the script-generated main evaluation figures and tables.
- `tools/paper1_figs.py` treats `assets/paper1_data/canonical_diagnostics_20260517.json` as the source of truth for predictor metrics, the representative diagnostic table (`tab:diag-base-vs-best`), and the PushT fragility scatter (`fig:scatter`).
- The canonical diagnostics release stores:
  - 4 tasks × 9 ckpts of `predictor_target_to_nn_cos_ratio_at_max_std`
  - 4 tasks × 9 ckpts of `predictor_rollout_T8_l2_at_max_std`
  - finalized representative diagnostic values used by `tab:diag-base-vs-best`
  - published LeWM cross-checkpoint correlation and partial-correlation values
- The historical external baseline release stores one PushT PLDM run trained without input-noise augmentation (`pusht_pldm_baseline`) for backward compatibility. The primary PLDM release is now the full 36-checkpoint aggregate in `canonical_evals_pldm_20260522.json`.
- The PLDM aggregate uses the same condition names, success-rate units, 3 evaluation seeds, and population-std convention as the LeWM aggregate. It is used only for the PLDM cross-method appendix, not for the main LeWM-only sweep and correlation tables.
- `canonical_full_diagnostics_pldm_20260523.json` stores the full `diagnostics_summary.json` row for every PLDM checkpoint and a compact base-vs-representative table used by the PLDM appendix. It is interpreted as a mechanism-boundary check: PLDM replicates the task-level fragility/recovery signature but does not reuse LeWM's exact compression-chain profile.
- `acpc_basin_diagnostics_pldm.json` stores the full PLDM 4 tasks × 9 configs Gaussian ACPC basin replication. It uses the same same-state clean/noised view protocol as `acpc_basin_diagnostics.json`; the PLDM appendix reports a compact baseline-vs-pixels-0.08-point-best summary from this full artifact.
- `partial_corr_bootstrap_20260523.json` stores the 95% percentile bootstrap CIs for the LeWM, PLDM, and joint partial-correlation claims quoted in the main text and PLDM appendix. It is generated by `tools/build_partial_corr_bootstrap.py` from the canonical eval/diagnostic JSONs.
- `acpc_phase0_clean_goal_seed9101.json` stores the clean-goal Phase-0 paired ACPC diagnostics (ACPC-1/H, PCC, CRA, MAF, ADM action-distance proxy, SPRR) for the LeWM and PLDM full sweep. It uses Gaussian observation-history noise at std 0.08 while keeping the goal image clean, matching the paper's primary `pixels_std0.08` endpoint. It is an exploratory diagnostic artifact for the Phase-0 appendix, not a method-result file or a robustness-predictor benchmark. The two component files `heldout_selection_phase0_seed9101.json` and `heldout_selection_phase0_pldm_seed9101.json` record the separate LeWM and PLDM runs merged into this release artifact. The older `acpc_phase0_diagnostics.json` is retained as an archived observation+goal sanity run.
- `acpc_basin_diagnostics.json` stores the paired Gaussian-noise ACPC basin diagnostic for all 36 LeWM canonical checkpoints. For each checkpoint it uses clean plus noised views at std 0.01..0.08, all rolled out under the same recorded action sequence. The main summary fields are `encoder_view_pair_l2_norm_by_nn`, `pred_view_pair_l2_norm_by_transition`, and `basin_contraction_pair_norm`. This artifact is intentionally Gaussian-noise-only to match the training sweep family; blur/resize are not mixed into the ACPC basin evidence.
- `unseen_phase0_acpc_subset.json` stores the independent-seed unseen follow-up subset: TwoRoom/Reacher blur as positive-transfer cases and PushT/Cube resize as boundary cases, each comparing `std_key` 0.0 vs 0.08 for training seeds 3073/3074 with clean-goal Phase-0 paired diagnostics. It joins diagnostics with strongest-only unseen eval scores and is a review artifact for appendix/boundary analysis, not a main Gaussian evidence source.
- Some released JSON rows retain the absolute checkpoint paths from the machine that produced the artifact. Treat those fields as historical provenance only. Portable reruns should resolve checkpoints from `DATA_ROOT` plus the relative task roots above.
- `canonical_blur_baselines_20260523.json` stores blur evals of LeWM/PLDM baselines trained without input-noise augmentation for kernel sizes 3/7/11/15 on `pixels`, `goal`, and `pixels_goal`. This is an eval-only cross-corruption sanity check for the blur appendix; it is not a blur-training sweep and is not mixed into the Gaussian-noise canonical tables.
- `tools/check_paper1_consistency.py` verifies:
  - the JSON exists
  - the released structure is 4 tasks × 9 configs
  - each config contains `clean`, `pixels_std0.05`, `pixels_std0.08`, `pixels_goal_std0.05`, and `pixels_goal_std0.08` with `mean`/`std`
  - the stored `mean`/`std` agree with the 3 released seed values
  - the historical external PLDM sanity-check aggregate is trained without input-noise augmentation and recomputes its reported means/stds/drop
  - the full PLDM aggregate is 4 tasks × 9 configs, contains required eval metrics, and recomputes means/stds from the three released seed values
  - the PLDM full-diagnostics aggregate is 4 tasks × 9 configs and contains the required five-layer diagnostic fields
  - the bootstrap CI aggregate exists, has the expected n=9 / n=18 scopes, and reproduces the headline PushT CI values quoted in the paper
  - the clean-goal Phase-0 ACPC aggregate is 2 methods × 4 tasks × 9 configs, all rows are `ok`, `corrupt_goal=false`, and the paired diagnostic fields used by the Phase-0 appendix are finite
  - the ACPC basin artifact covers LeWM 4 tasks × 9 configs, contains only Gaussian-noise std 0.01..0.08 variants, and stores finite encoder/prediction basin radii
  - the PLDM ACPC basin replication covers the full 4 tasks × 9 configs grid, all `ok`
  - the blur sanity-check aggregate covers 2 methods × 4 tasks × 12 blur conditions and recomputes means/stds/worst-blur drops from the three seed values
