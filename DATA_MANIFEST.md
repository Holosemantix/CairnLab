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
- **Three-training-seed strongest-only unseen score artifacts**: `assets/paper1_data/unseen_origin_vs_std008_strongest_s3072.json`, `assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json`, and `assets/paper1_data/unseen_origin_vs_std008_strongest_s3074.json` (audited score artifacts for the unseen score aggregate over training seeds 3072/3073/3074; the `s3072` wrapper is derived from the earlier seed-3072 task-split artifacts `unseen_origin_vs_std008_strongest_tworoom.json` and `unseen_origin_vs_std008_strongest_reacher.json`; together cover 4 tasks × 3 training seeds × 2 std keys × 2 stress families with no-op plus blur k=15 / resize factor 0.25)
- **Unseen Phase-0 ACPC subset artifacts**: `assets/paper1_data/unseen_phase0_acpc_subset.json` (12 selected seed/task/stress case rows for training seeds 3072/3073/3074 joining 24 clean-goal Phase-0 diagnostic rows with strongest-only unseen eval scores) and `assets/paper1_data/unseen_phase0_acpc_fullstress.json` (24 task-family-seed rows over all four tasks, blur/resize strongest endpoints, and training seeds 3072/3073/3074; reduces selected-slice risk while remaining a bounded two-family scope audit)
- **Three-training-seed Gaussian replication summary**: `assets/paper1_data/training_seed_gaussian_lockbox.json` and `.md` (legacy filename; canonical seed 3072 plus independent seeds 3073/3074; reports training-seed mean/std for the main observation-noise 0.08 endpoint)
- **Three-seed LeWM eval manifests**: `assets/paper1_data/training_seed_eval_manifests/lewm_seed{3072,3073,3074}_evals.json` (canonical-shaped manifests with local checkpoint paths and seed-specific eval metrics for the three-training-seed full-grid diagnostic run)
- **Three-seed LeWM Phase-0 full-grid diagnostics**: `assets/paper1_data/acpc_phase0_lewm_three_seed.json` plus per-seed component files (108 ok rows: 3 training seeds × 4 tasks × 9 Gaussian checkpoint rows; ACPC/PCC/CRA/MAF computed with clean goal and observation-history Gaussian noise)
- **Three-seed fixed-rule diagnostic validation**: `assets/paper1_data/three_seed_diagnostic_validation.json` and `.md` (ACPC/PCC/CRA/MAF ranking rule fixed on development seed 3072 and evaluated on held-out training seeds 3073/3074; reports 7/8 held-out within-5pp selection hits, 2.21 ± 1.83 pp held-out regret, and 10/12 within-5pp over all seeds)
- **Selector-baseline audit**: `assets/paper1_data/selector_baseline_audit_20260704.json` and `.md` (derived from the three-seed Phase-0 diagnostics and eval manifests; compares the aggregate ACPC/PCC/CRA/MAF triage rule with fixed `std=0.08`, single-metric selectors, exact random nonzero-std expected regret, and closed-loop oracle; intended as plateau-context audit rather than a dominance claim)
- **Task-state proxy margin pass-rate**: `assets/paper1_data/semantic_margin_passrate_lewm_three_seed.json` and `.md` (24 ok rows: training seeds 3072/3073/3074 × 4 tasks × {0.0, 0.08}; reports task-state proxy pass rates plus same-state noisy radius and state-proxy-different rollout distance)
- **Validation remediation summary**: `assets/paper1_data/prospective_validation_summary.json` and `.md` (summarizes three-training-seed Gaussian behavior, development/held-out fixed-rule diagnostic validation, task-state proxy margin pass-rate, and bounded unseen-stressor scope checks)
- Scope: 36 LeWM canonical checkpoints = 4 tasks × 9 configs (`base` + `std_max` 0.01..0.08); 36 PLDM checkpoints on the same grid; completed LeWM Gaussian replication sweeps for independent training seeds 3073/3074 are summarized separately; strongest-only unseen score artifacts cover training seeds 3072/3073/3074.
- Evaluation protocol: canonical Gaussian grids use **3 evaluation seeds** (`42`, `43`, `44`) × **100 trajectories per seed** for each checkpoint/seed point; strongest-only unseen score artifacts store `eval_base_seed=42`, `eval_seeds=3`, and `num_eval=300`.
- Seed clarification: canonical LeWM/PLDM grid tables use evaluation-seed variance; the main LeWM Gaussian endpoint, the full-grid LeWM Phase-0 diagnostic validation, and the task-state proxy margin pass-rate now all report independent training seeds 3072/3073/3074, with evaluation-seed variance kept as secondary measurement scale where applicable.

## Release Provenance Notes

- Paper-facing main evidence: `canonical_evals_20260517.json`, `training_seed_gaussian_lockbox.json`, `acpc_phase0_lewm_three_seed.json`, `three_seed_diagnostic_validation.json`, `selector_baseline_audit_20260704.json`, `semantic_margin_passrate_lewm_three_seed.json`, `prospective_validation_summary.json`, `unseen_phase0_acpc_fullstress.json`, `canonical_diagnostics_20260517.json`, `acpc_basin_diagnostics.json`, `canonical_evals_pldm_20260522.json`, `canonical_diagnostics_pldm_20260522.json`, `canonical_full_diagnostics_pldm_20260523.json`, `acpc_basin_diagnostics_pldm.json`, `partial_corr_bootstrap_20260523.json`, and `acpc_phase0_clean_goal_seed9101.json`.
- Scope-boundary / sanity artifacts: `canonical_blur_baselines_20260523.json` is eval-only blur stress; `acpc_phase0_diagnostics.json` is the archived observation+goal Phase-0 sanity run; `target_view_closed_loop_summary.json` is a negative target-view ablation; `canonical_external_baselines_20260520.json` is retained for backward-compatible sanity checks. The strongest-only unseen score artifacts support the three-training-seed unseen score aggregate in `prospective_validation_summary.json`; `unseen_phase0_acpc_subset.json` supports the matched selected three-training-seed appendix diagnostic slice, and `unseen_phase0_acpc_fullstress.json` supports the full blur/resize 24-row appendix scope audit. Task-state proxy margin pass-rates, same-state noisy radii, and state-proxy-different rollout distances are now reported in `semantic_margin_passrate_lewm_three_seed.json`; finer oracle contact/topology/goal-relation labels remain future work if claims expand beyond matched Gaussian diagnostics.
- Contamination fix: the 2026-06-10 audit found that the TwoRoom and PushT representative diagnostic rows in `canonical_diagnostics_20260517.json` duplicated heteroscedastic-loss diagnostics. The affected representative fields were re-extracted from the intended per-checkpoint `diagnostics_summary.json` files. The release checker now guards these values so the PushT noise-sweep row cannot regress to the heteroscedastic `rank 76.4 -> 42.9` narrative.
- Manual revision status: no released numerical JSON result row is hand-edited for paper prose. The representative-diagnostics re-extraction above is recorded in JSON metadata and checked by `tools/check_paper1_consistency.py`; the strongest-only unseen score artifacts received a status-only metadata update after the three-seed score aggregate audit, with numeric rows unchanged.

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
| `unseen_origin_vs_std008_strongest_tworoom.json` | Seed-3072 strongest-only unseen score, TwoRoom | `af0a4329155b2bd4acd59964318ebab2935f8a48f145e64279d5130fc709aa68` |
| `unseen_origin_vs_std008_strongest_reacher.json` | Seed-3072 strongest-only unseen score, Reacher/Cube/PushT | `90debe607cd302c31707890375c500ca2e62a4feeb05bc61f580f2dc5097566c` |
| `unseen_origin_vs_std008_strongest_s3072.json` | Seed-3072 strongest-only unseen score wrapper, all tasks | `5b12fd4d19484199178b13a3eae3c12867d6dd4a218b431d03ea69666eaaf0dd` |
| `unseen_origin_vs_std008_strongest_s3072_manifest.json` | Seed-3072 strongest-only unseen score wrapper manifest | `0f1cb6ed765e96f479dc5b60fe4c81d2ab05a9220fb1f09392cc389bb8aa1017` |
| `unseen_origin_vs_std008_strongest_s3072.schema.json` | Seed-3072 strongest-only unseen score wrapper schema | `525d59956dc1e8180e795c23f5667c552e05bb4155808f7fb317d7571f0d13da` |
| `unseen_origin_vs_std008_strongest_s3073.json` | Seed-3073 strongest-only unseen score lockbox | `c933785bc39b4ac556fbe69a3b00e3451402accfd3b64f907b8a29f306b8636a` |
| `unseen_origin_vs_std008_strongest_s3074.json` | Seed-3074 strongest-only unseen score lockbox | `243ef17d77ed2e72adfd6fbdf16459ed09dd33558869c31421c562ae520db637` |
| `unseen_phase0_acpc_subset.json` | Unseen Phase-0 ACPC subset review artifact | `7f8ff8a8f85c170b7eed0abdf16109e0f2d4f0f94f66dd498c5ab76a1c314ead` |
| `unseen_phase0_acpc_subset.schema.json` | Unseen Phase-0 ACPC subset schema | `8f688e77a27fbf69bb750bf26f90e8745b0b2e369ad6e62128b4a3040095b85f` |
| `unseen_phase0_acpc_fullstress.json` | Full blur/resize unseen Phase-0 ACPC scope audit | `77636682649c1f7a7099eca6fe7d2ba86244550a4c54571551d88298cd1ca51c` |
| `unseen_phase0_acpc_fullstress.schema.json` | Full blur/resize unseen Phase-0 ACPC scope-audit schema | `8f688e77a27fbf69bb750bf26f90e8745b0b2e369ad6e62128b4a3040095b85f` |
| `training_seed_gaussian_lockbox.json` | Three-training-seed Gaussian replication summary (legacy filename) | `526ff2fad2ba86c3c865341ae4bc9db9fad52f4825bf12818c3ab54a2af4aabb` |
| `training_seed_eval_manifests/lewm_seed3072_evals.json` | Seed-3072 LeWM eval manifest for three-seed Phase-0 | `0f34532e78bf228eaa3f470b5728fb28fc88d34f47b40e728c7aadab6b4397c4` |
| `training_seed_eval_manifests/lewm_seed3073_evals.json` | Seed-3073 LeWM eval manifest for three-seed Phase-0 | `03cfcf889825cf2022a3b0e3d54a67e1f70e412b8d704226609cbe89360b0b5b` |
| `training_seed_eval_manifests/lewm_seed3074_evals.json` | Seed-3074 LeWM eval manifest for three-seed Phase-0 | `7a7dbd8c37992cf12f5b77bf1f7f841e9314c7e9eb31ddf67ce8afa4b5699316` |
| `acpc_phase0_lewm_seed3072.json` | Seed-3072 LeWM Phase-0 full-grid diagnostics | `1163c0f033fb8ce5b6afc9307d52db4bec4ad9f6f16a62f048defc4865e578ba` |
| `acpc_phase0_lewm_seed3073.json` | Seed-3073 LeWM Phase-0 full-grid diagnostics | `77d78b4a3193779b9b7d8d6a2e4db0f4d280ad3b408abeb0edd1f686e78a1394` |
| `acpc_phase0_lewm_seed3074.json` | Seed-3074 LeWM Phase-0 full-grid diagnostics | `ef68d50e294df3b8cce795f47f0adba33803fb964f043ff1ed95fb4e4de50cb6` |
| `acpc_phase0_lewm_three_seed.json` | Three-seed LeWM Phase-0 full-grid diagnostics | `8742c9dfbbb45f884fc4aa93d1adecfe03ca9d7b9f037e959b5ddda988d4e989` |
| `three_seed_diagnostic_validation.json` | Development/held-out three-seed fixed-rule diagnostic validation | `6275b0ea68e367aa302dbe1e19f1ab96f2a0485aee28d656fce9d8e028025ef0` |
| `selector_baseline_audit_20260704.json` | Selector-baseline audit for fixed-rule triage | `95d7ffc04949b6c93f567ab6f1097280e963865fbe341159fdd38630b01e6219` |
| `selector_baseline_audit_20260704.md` | Human-readable selector-baseline audit summary | `1584f65093c1328a970c25d79efb57cea7e5f7f4752d9a50cbf12431e0393ea5` |
| `semantic_margin_passrate_lewm_three_seed.json` | Three-seed task-state proxy margin pass-rate | `b4cea993839043f7a5759fbed9645dc290fc4373b3203aa5ca96442b81a1c3fa` |
| `prospective_validation_summary.json` | Validation remediation summary, fixed-rule diagnostic split, task-state proxy margin pass-rate, and unseen scope check | `6534a97cf6130350f47a36404c75f1e9cca2de12dd2f5da5682cb854bad8a6d5` |

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
- `acpc_phase0_clean_goal_seed9101.json` stores the clean-goal Phase-0 paired ACPC diagnostics (ACPC-1/H, PCC, CRA, MAF, ADM action-distance proxy, SPRR) for the LeWM and PLDM full sweep. It uses Gaussian observation-history noise at std 0.08 while keeping the goal image clean, matching the paper's primary `pixels_std0.08` endpoint. It is an appendix reference file for the paired-diagnostic definitions and is not used as the paper-facing held-out selector benchmark. The two component files `heldout_selection_phase0_seed9101.json` and `heldout_selection_phase0_pldm_seed9101.json` record the separate LeWM and PLDM runs merged into this release artifact. The older `acpc_phase0_diagnostics.json` is retained as an archived observation+goal sanity run.
- `acpc_basin_diagnostics.json` stores the paired Gaussian-noise ACPC basin diagnostic for all 36 LeWM canonical checkpoints. For each checkpoint it uses clean plus noised views at std 0.01..0.08, all rolled out under the same recorded action sequence. The main summary fields are `encoder_view_pair_l2_norm_by_nn`, `pred_view_pair_l2_norm_by_transition`, and `basin_contraction_pair_norm`. This artifact is intentionally Gaussian-noise-only to match the training sweep family; blur/resize are not mixed into the ACPC basin evidence.
- `unseen_phase0_acpc_subset.json` stores the matched unseen diagnostic subset: TwoRoom/Reacher blur as positive-transfer cases and PushT/Cube resize as boundary cases, each comparing `std_key` 0.0 vs 0.08 for training seeds 3072/3073/3074 with clean-goal Phase-0 paired diagnostics. It joins diagnostics with strongest-only unseen eval scores and is a review artifact for matched diagnostic analysis.
- Some released JSON rows retain the absolute checkpoint paths from the machine that produced the artifact. Treat those fields as historical provenance only. Portable reruns should resolve checkpoints from `DATA_ROOT` plus the relative task roots above.
- `canonical_blur_baselines_20260523.json` stores blur evals of LeWM/PLDM baselines trained without input-noise augmentation for kernel sizes 3/7/11/15 on `pixels`, `goal`, and `pixels_goal`. This is an eval-only cross-corruption sanity check for the blur appendix; it is not a blur-training sweep and is not mixed into the Gaussian-noise canonical tables.
- `selector_baseline_audit_20260704.json` stores the no-retraining selector-baseline audit for the three-training-seed Gaussian grid. It is derived from `acpc_phase0_lewm_three_seed.json` and `three_seed_diagnostic_validation.json`; random nonzero-std regret is the exact average over the eight nonzero Gaussian checkpoints, and oracle best is reported only as a lower bound.
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
