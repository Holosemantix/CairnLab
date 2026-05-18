# Paper 1 Data Manifest

This manifest documents the released evaluation aggregate for Paper 1:

- Canonical aggregate: `assets/paper1_data/canonical_evals_20260517.json`
- Schema: `assets/paper1_data/canonical_evals_20260517.schema.json`
- Canonical diagnostics: `assets/paper1_data/canonical_diagnostics_20260517.json`
- Diagnostics schema: `assets/paper1_data/canonical_diagnostics_20260517.schema.json`
- Scope: 36 LeWM checkpoints = 4 tasks × 9 configs (`base` + `std_max` 0.001..0.008)
- Evaluation protocol: **3 evaluation seeds** (`42`, `43`, `44`) × **100 trajectories per seed**
- Important clarification: these are **evaluation seeds**, not 3 independently trained models per configuration

## Data Semantics

- Unit: every success-rate field is stored in **percentage points** on `[0, 100]`
- Aggregation: each metric stores `values = [seed42, seed43, seed44]`, `mean`, and `std`
- `std` convention: the JSON stores the **population standard deviation** across the 3 evaluation seeds (`ddof = 0`)
- Conditions released per checkpoint:
  - `clean`
  - `goal_std0.03`, `goal_std0.05`, `goal_std0.08`
  - `pixels_std0.03`, `pixels_std0.05`, `pixels_std0.08`
  - `pixels_goal_std0.03`, `pixels_goal_std0.05`, `pixels_goal_std0.08`
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

| Task | Local ckpt root |
|---|---|
| TwoRoom | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-tworooms/ckpt` |
| PushT | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt` |
| Reacher | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-reacher/ckpt` |
| Cube | `/home/ag/dataset/ag_data/data/world_model/quentinll/lewm-cube/ckpt` |

## Released Checkpoints

| Task | `std_max` | Canonical `subdir` | Raw per-seed file pattern |
|---|---:|---|---|
| TwoRoom | 0.0 | `tworoom_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.001 | `tworoom_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.002 | `tworoom_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.003 | `tworoom_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.004 | `tworoom_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.005 | `tworoom_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.006 | `tworoom_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.007 | `tworoom_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.008 | `tworoom_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.0 | `pusht_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.001 | `pusht_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.002 | `pusht_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.003 | `pusht_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.004 | `pusht_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.005 | `pusht_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.006 | `pusht_lewm_noise_0to006_p1_20260507` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.007 | `pusht_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.008 | `pusht_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.0 | `reacher_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.001 | `reacher_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.002 | `reacher_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.003 | `reacher_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.004 | `reacher_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.005 | `reacher_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.006 | `reacher_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.007 | `reacher_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.008 | `reacher_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.0 | `cube_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.001 | `cube_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.002 | `cube_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.003 | `cube_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.004 | `cube_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.005 | `cube_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.006 | `cube_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.007 | `cube_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.008 | `cube_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |

## Consumer Notes

- `tools/paper1_figs.py` treats `assets/paper1_data/canonical_evals_20260517.json` as the sole source of truth for Figure 1, Figure 2, Figure 3 eval values, and Figure 6.
- `tools/paper1_figs.py` treats `assets/paper1_data/canonical_diagnostics_20260517.json` as the source of truth for Figure 3 predictor metrics and Table 3 / Figure 4 representative diagnostic values.
- The canonical diagnostics release stores:
  - 4 tasks × 9 ckpts of `predictor_target_to_nn_cos_ratio_at_max_std`
  - 4 tasks × 9 ckpts of `predictor_rollout_T8_l2_at_max_std`
  - finalized Table 3 representative diagnostic values
  - published Table 4 / Table 4b / Table 5 correlation numbers
- `tools/check_paper1_consistency.py` verifies:
  - the JSON exists
  - the released structure is 4 tasks × 9 configs
  - each config contains `clean`, `pixels_goal_std0.05`, and `pixels_goal_std0.08` with `mean`/`std`
  - the stored `mean`/`std` agree with the 3 released seed values
