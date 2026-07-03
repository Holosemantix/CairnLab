# Paper 1 工具说明

本目录下和 Paper 1 直接相关的脚本分三类：release gate、图表/统计复现、从本地原始实验目录重新聚合 canonical artifact。除非特别说明，命令都从仓库根目录执行。

## 最常用命令

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

`check_paper1_consistency.py` 是提交前的主检查入口。它验证必需 artifact 是否存在、旧口径字符串是否消失、LeWM/PLDM/blur/ACPC-basin canonical JSON 结构是否完整、正文引用的关键相关系数和 bootstrap CI 是否能从 artifact 复算到相同数值。Paper 1 当前主 corrupted endpoint 是 `pixels_std0.08`：只扰动 observation pixels，goal 保持 clean；`pixels_goal_std0.08` 只作为更强 stress condition。

`paper1/build.sh --clean` 用 `latexmk` 重建 PDF。构建后建议检查 log：

```bash
rg -n "Overfull|undefined references|Citation .* undefined|Reference .* undefined|Fatal error|Undefined control sequence" paper1/main.log || true
```

## 图和统计

| 脚本 | 作用 | 输入 | 输出 / 用途 |
|---|---|---|---|
| `tools/paper1_figs.py` | 渲染主文中由脚本生成的图 | `assets/paper1_data/canonical_evals_20260517.json`, `assets/paper1_data/canonical_diagnostics_20260517.json` | 默认只输出 `assets/paper1_figs/fig2_sweep.png`；已下线的 `fig3_pareto.png`, `fig4_radar.png`, `fig5_scatter.png`, `fig6_mechanism.png` 仍可用 `--only` 生成但当前不进正文 |
| `tools/build_partial_corr_bootstrap.py` | 为 partial Spearman 相关计算 95% percentile bootstrap CI | LeWM/PLDM canonical eval + diagnostics artifact | `assets/paper1_data/partial_corr_bootstrap_20260523.json`，用于主文 partial-correlation tables 和 PLDM appendix |
| `tools/pldm_correlation_analysis.py` | 复算 LeWM/PLDM within-method 与 joint partial correlation | LeWM/PLDM canonical eval + diagnostics artifact | `assets/paper1_data/cross_method_corr_pldm_20260522.json`，用于 PLDM appendix 和 consistency checker |
| `tools/paper1_acpc_basin.py` | Paper-facing Gaussian-noise ACPC basin runner：dense std 0.01--0.08 same-state views，统计 encoder radius / prediction radius / contraction | LeWM/PLDM canonical eval manifest + 本地 epoch-10 model object checkpoints | `assets/paper1_data/acpc_basin_diagnostics.json`；PLDM appendix 的 full-sweep replication 用 `assets/paper1_data/acpc_basin_diagnostics_pldm.json` |
| `tools/paper1_phase0_acpc.py` | 低频 paired ACPC 诊断 runner：ACPC-1/H、PCC、CRA、MAF、ADM proxy、SPRR | LeWM/PLDM canonical eval manifest + 本地 loadable model checkpoints | `assets/paper1_data/acpc_phase0_clean_goal_seed9101.json`；three-seed LeWM run 输出 `assets/paper1_data/acpc_phase0_lewm_three_seed.json`，旧 `acpc_phase0_diagnostics.json` 仅作 observation+goal archived sanity |
| `tools/paper1_training_seed_eval_manifests.py` | 生成 LeWM seed 3072/3073/3074 的 canonical-shaped eval manifests | canonical seed-3072 JSON + seed3073/3074 checkpoint `eval_summary.csv` | `assets/paper1_data/training_seed_eval_manifests/lewm_seed*_evals.json` |
| `tools/paper1_three_seed_diagnostic_validation.py` | 固定 ACPC/PCC/CRA/MAF rank rule 并汇总 three-seed full-grid validation | `assets/paper1_data/acpc_phase0_lewm_three_seed.json` | `assets/paper1_data/three_seed_diagnostic_validation.json` / `.md` |
| `tools/paper1_semantic_margin.py` | 任务语义 margin pass-rate：same-state noisy radius vs semantic-different rollout distance | seed-specific eval manifests + 本地 ckpt/data | `assets/paper1_data/semantic_margin_passrate_lewm_three_seed.json` / `.md` |
| `tools/paper1_selective_contraction.py` | Phase-1 前的 selective-contraction branch probe；可选渲染同 state clean/noised rollout cluster 图 | ACPC basin + Phase-0 diagnostics；plot 模式还需要本地 checkpoint/data | `assets/paper1_data/selective_contraction_fullseq_branch.*`；cluster 图默认输出到 `assets/phase1_figs/selective_contraction_clusters/`，paper-facing 输出可用 `--cluster-out-dir assets/paper1_figs` 生成 `assets/paper1_figs/pusht_fullseq_selective_contraction_clusters.png`；用 repeated perturbation samples，默认用 fixed-seed random anchors 选点并绘制低权重的 90% 2-D covariance ellipse，只作 qualitative visualization |
| `tools/paper1_unseen_eval_grid.py` / `run_paper1_unseen_origin_vs_std008_seeded.sh` | unseen-perturbation pilot / lockbox launcher；只包装 `run_trainer.sh`，默认 eval-only，按需加 `--diagnostics` | canonical eval JSON or seed-remapped temporary canonical + `$DATA_ROOT/lewm-*/ckpt/*epoch_10_object.ckpt` | seed-specific `unseen_origin_vs_std008_strongest_s<seed>*.json` review artifacts |
| `tools/build_paper1_unseen_eval_artifact.py` | 汇总 unseen-perturbation pilot 的 `eval_summary.csv` 和可选 diagnostics summary | `tools/paper1_unseen_eval_grid.py` 写出的 manifest + 本地 eval 输出 | `assets/paper1_data/unseen_perturbation_pilot_seed3072.json`；进入正文前必须人工审查 |

常用重生成命令：

```bash
python -m tools.paper1_figs --out-dir assets/paper1_figs

python -m tools.pldm_correlation_analysis \
  --evals-lewm assets/paper1_data/canonical_evals_20260517.json \
  --evals-pldm assets/paper1_data/canonical_evals_pldm_20260522.json \
  --diag-lewm assets/paper1_data/canonical_diagnostics_20260517.json \
  --diag-pldm assets/paper1_data/canonical_diagnostics_pldm_20260522.json \
  --out assets/paper1_data/cross_method_corr_pldm_20260522.json

python -m tools.build_partial_corr_bootstrap \
  --out assets/paper1_data/partial_corr_bootstrap_20260523.json \
  --n-bootstrap 1000 --seed 42

python -m tools.paper1_acpc_basin \
  --dry-run \
  --out /tmp/acpc_basin_dry.json

python -m tools.paper1_acpc_basin \
  --out assets/paper1_data/acpc_basin_diagnostics.json

python -m tools.paper1_acpc_basin \
  --methods PLDM \
  --out assets/paper1_data/acpc_basin_diagnostics_pldm.json

python -m tools.paper1_phase0_acpc \
  --dry-run --methods LeWM PLDM --tasks PushT \
  --out /tmp/acpc_phase0_dry.json

python -m tools.paper1_phase0_acpc \
  --methods LeWM PLDM \
  --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
  --noise-std 0.08 --clean-goal --seed 9101 \
  --n-sequences 100 --random-action-trials 64 \
  --out assets/paper1_data/acpc_phase0_clean_goal_seed9101.json

python -m tools.paper1_training_seed_eval_manifests

STABLEWM_HOME=/tmp/paper1_stablewm_home CUDA_VISIBLE_DEVICES=0 \
python -m tools.paper1_phase0_acpc \
  --methods LeWM --evals-lewm assets/paper1_data/training_seed_eval_manifests/lewm_seed3072_evals.json \
  --clean-goal --seed 9101 --out assets/paper1_data/acpc_phase0_lewm_seed3072.json
# repeat for seeds 3073/3074, then merge component files into acpc_phase0_lewm_three_seed.json

python -m tools.paper1_three_seed_diagnostic_validation

STABLEWM_HOME=/tmp/paper1_stablewm_home CUDA_VISIBLE_DEVICES=3 \
python -m tools.paper1_semantic_margin --seeds 3072 --out /tmp/semantic_margin_seed3072.json
# repeat/merge for seeds 3073/3074 to regenerate semantic_margin_passrate_lewm_three_seed.json

OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mplconfig \
python -m tools.paper1_selective_contraction \
  --plot-clusters --plot-tasks PushT \
  --n-sequences 128 --cluster-anchor-count 16 \
  --view-stds 0.0 0.01 0.04 0.08 \
  --cluster-perturb-repeats 6 \
  --feature-cache-dir /tmp/paper1_selective_contraction_cache \
  --cluster-out-dir assets/paper1_figs \
  --cluster-envelope ellipse --cluster-envelope-coverage 0.90

DATA_ROOT=/path/to/world_model/quentinll \
python -m tools.paper1_unseen_eval_grid --dry-run

DATA_ROOT=/path/to/world_model/quentinll \
python -m tools.paper1_unseen_eval_grid --only-missing

DATA_ROOT=/path/to/world_model/quentinll \
bash run_paper1_unseen_origin_vs_std008_eval.sh

python -m tools.build_paper1_unseen_eval_artifact \
  --manifest assets/paper1_data/unseen_perturbation_pilot_seed3072_manifest.json \
  --out assets/paper1_data/unseen_perturbation_pilot_seed3072.json \
  --allow-missing
```

95% CI 的口径：脚本对 checkpoint rows 做 with-replacement bootstrap。within-LeWM 和 within-PLDM 是每个 task 的 9 个 checkpoint rows；joint 分析是 LeWM+PLDM 共 18 个 rows，并在 partial correlation 中同时 conditioning on `std_max` 和 `method`。CI 是 bootstrap 分布的 2.5/97.5 percentile，不是额外 evaluation seed 的置信区间。

ACPC basin runner 默认只接受 Gaussian-noise corruption specs，并使用 dense eval grid `0.01 ... 0.08`，以匹配 Paper 1 的 Gaussian-noise training sweep；它默认只扰动 observation history、保持 goal clean。不要把 blur/resize 混入这个 artifact。默认命令只跑 LeWM dense 4 tasks × 9 configs；PLDM appendix replication 使用 `--methods PLDM` 跑同样的 4 tasks × 9 configs full sweep。`--base-vs-best` 仅保留作快速本地审计，不作为 paper-facing artifact。`--dry-run` 只解析 manifest 和 epoch-10 checkpoint 路径，不加载模型。实际计算需要当前 Python 环境能 import `torch`、`stable_pretraining`、`stable_worldmodel`，且 `$DATA_ROOT/<task-root>/ckpt/<subdir>/` 下存在唯一 `*epoch_10_object.ckpt`。`DATA_ROOT` 也可以通过 `PAPER1_DATA_ROOT`、`STABLEWM_HOME` 或 `--model-root` 传入。

Phase 0 ACPC runner 的 `--dry-run` 只解析 manifest 和 checkpoint 路径，不需要 `torch`；shell wrapper `run_phase0_acpc.sh --dry-run` 输出到 `/tmp/acpc_phase0_dry_run.json`，避免覆盖 canonical artifact。实际计算需要当前 Python 环境能 import `torch`、`stable_pretraining`、`stable_worldmodel`，且 canonical eval 里的 `path` 或 `--model-root` 下存在可 `torch.load` 的 model object checkpoint。当前 ADM 是 action-distance latent proxy，不是 oracle state/keypoint ADM。

Selective-contraction cluster plots are paper-facing qualitative illustrations, not standalone proof. The default paper-facing path selects colored anchors by a fixed-seed random subset and writes the selected indices to the sidecar JSON; neither t-SNE coordinates nor high-D statistics are used for anchor selection. The small panel summaries (`median r/NN`, `r < NN`, `disjoint balls`) are computed in the original high-D feature space and must not be replaced by bottom-of-figure screenshot tables or in-axis legend boxes. The low-opacity 90% covariance ellipses are 2-D t-SNE visual summaries only, not high-D basin boundaries. Use `--cluster-envelope none` for point-only audits, `--cluster-envelope circle` only to reproduce the legacy max-distance circle view, and `--cluster-envelope hull` only as a sample hull.

For label/style-only redraws of these figures, pass `--feature-cache-dir /tmp/paper1_selective_contraction_cache`. The first run writes cached encoder/8-step-rollout feature arrays keyed by task, method, checkpoints, view stds, seed, and rendering sample parameters; later runs with the same parameters reuse the cache and do not reload checkpoints. Use `--refresh-feature-cache` only when the checkpoints, sampled windows, or feature-extraction parameters intentionally change. Do not commit the cache directory.

The unseen-perturbation pilot is deliberately staged. Run
`paper1_unseen_eval_grid.py` without `--diagnostics` first to test closed-loop
transfer under `gaussian_blur` and `resize`. If the pilot shows a meaningful
plateau/ranking signal, rerun a smaller representative subset with
`--diagnostics` before making any manuscript claim. The generated artifact is a
review artifact, not an automatic paper-facing source.

Seed-3072 fast smoke result (2026-06-29): `PushT` and `TwoRoom`, `std_max` in
`{0.0, 0.08}`, strongest-severity only (`gaussian_blur` kernel 15 and `resize`
factor 0.25), `30 x 3` eval episodes. The output artifact was
`assets/paper1_data/unseen_origin_vs_std008_fast.json`. PushT did not show a
clear cross-stressor gain (`std=0.08` vs `std=0.0`: blur +2.22 pp, resize 0.00
pp), while TwoRoom showed a large positive smoke signal (blur +56.67 pp, resize
+55.56 pp) with no clean-performance tradeoff. Treat this as a development smoke
check only. The formal pass uses `num_eval=300` total episodes (`100` per seed
for 3 eval seeds) and the dedicated
`run_paper1_unseen_origin_vs_std008_eval.sh` launcher. The launcher
intentionally runs only no-op plus strongest stress (`gaussian_blur=1,15`,
`resize=1.0,0.25`) because the next question is whether `std=0.08` helps, not
to estimate a full severity curve. It writes to a separate
`paper1_unseen_origin_vs_std008_s3072_strongest_*` output prefix so fast-run rows
do not mix with formal eval rows.

Four-task formal strongest-only pilot result (2026-06-29): `num_eval=300` total
episodes (`100` per seed for 3 eval seeds), `std_max` in `{0.0, 0.08}`, no-op
plus strongest stress only. The review artifacts are
`assets/paper1_data/unseen_origin_vs_std008_strongest_tworoom.json` with
`assets/paper1_data/unseen_origin_vs_std008_strongest_tworoom_manifest.json`
for TwoRoom, and `assets/paper1_data/unseen_origin_vs_std008_strongest_reacher.json`
with `assets/paper1_data/unseen_origin_vs_std008_strongest_reacher_manifest.json`
for Reacher/Cube/PushT. Both artifacts report `missing=0`; the primary stress
groups are `pixels_blur_ks15` and `pixels_rs_factor0.25`.

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

Interpretation: TwoRoom is a strong positive seed-3072 pilot signal. Reacher is
also positive, but part of the gain is a better origin checkpoint, so the
drop-improvement columns are the cleaner robustness readout. PushT and Cube have score movements small enough to read as no clear effect
under this strongest-only check. This supports a task-dependent transfer reading,
not a universal cross-perturbation robustness claim; keep it out of paper-facing
claims until independent training seeds are evaluated.

Seed-3073/3074 strongest-only lockbox result (2026-07-03): both independent
training seeds completed the same four-task pass through
`run_paper1_unseen_origin_vs_std008_seeded.sh`, producing
`assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json` and
`assets/paper1_data/unseen_origin_vs_std008_strongest_s3074.json` with
`missing=0`. Averaged across the two seeds, TwoRoom remains strongly positive
under blur and resize (stress delta about +37 pp), Reacher remains strongly
positive (+42.58 pp averaged across the two families), while PushT (+4.00 pp
averaged, with one negative resize row) and Cube (-1.08 pp averaged) should be
read as no clear effect at this evaluation variance. Treat these artifacts as
bounded cross-stressor development evidence, not universal robustness.

Representative unseen Phase-0 ACPC subset (2026-07-03):
`run_paper1_unseen_phase0_acpc_subset.sh` reruns clean-goal Phase-0 paired
ACPC/PCC/CRA/MAF on selected positive and boundary cases, then
`tools/build_paper1_unseen_phase0_acpc_subset.py` joins those rows with the
completed unseen eval artifacts. The artifact is
`assets/paper1_data/unseen_phase0_acpc_subset.json` (`missing=0`, 8 case rows /
16 diagnostic rows). TwoRoom blur and Reacher blur show score gains and
diagnostic improvements in the same direction; Cube resize and PushT resize
remain no-clear-effect boundary cases rather than negative or mixed-transfer
claims.

Optional PLDM sanity plots can use the same runner without changing paper-facing claims:

```bash
python -m tools.paper1_selective_contraction \
  --method PLDM \
  --acpc-basin assets/paper1_data/acpc_basin_diagnostics_pldm.json \
  --plot-clusters --plot-tasks PushT \
  --feature-cache-dir /tmp/paper1_selective_contraction_cache \
  --cluster-out-dir assets/phase1_figs/selective_contraction_clusters \
  --cluster-envelope ellipse --cluster-envelope-coverage 0.90
```

The paper-facing PLDM appendix figure (`assets/paper1_figs/pusht_pldm_noise_selective_contraction_clusters.png`) uses the full-quality parameters and the paper-facing output dir:

```bash
STABLEWM_HOME=<dataset-root> python -m tools.paper1_selective_contraction \
  --method PLDM \
  --acpc-basin assets/paper1_data/acpc_basin_diagnostics_pldm.json \
  --plot-clusters --plot-tasks PushT \
  --n-sequences 128 --cluster-anchor-count 16 \
  --view-stds 0.0 0.01 0.04 0.08 \
  --cluster-perturb-repeats 6 \
  --feature-cache-dir /tmp/paper1_selective_contraction_cache \
  --cluster-out-dir assets/paper1_figs \
  --cluster-envelope ellipse --cluster-envelope-coverage 0.90
```

The projection-free local-atlas companion figure (`assets/paper1_figs/pusht_fullseq_selective_contraction_atlas.png`) is rendered with:

```bash
STABLEWM_HOME=<dataset-root> python -m tools.paper1_selective_contraction \
  --plot-atlas --plot-tasks PushT \
  --n-sequences 128 --atlas-anchor-count 16 \
  --view-stds 0.0 0.01 0.04 0.08 \
  --feature-cache-dir /tmp/paper1_selective_contraction_cache \
  --atlas-out-dir assets/paper1_figs
```

Both load checkpoints and the HDF5 datasets; `STABLEWM_HOME` must point at the dataset root that contains `pusht_expert_train.h5` (the resolver checks `$STABLEWM_HOME/<name>.h5` and `$STABLEWM_HOME/datasets/<name>.h5`).

For quick path checks, reduce the render size, for example add `--n-sequences 48 --cluster-anchor-count 10 --cluster-perturb-repeats 3 --cluster-perplexity 12 --cluster-tsne-max-iter 350`; this is a smoke test, not a paper-facing figure. For non-LeWM methods, the default summary output is method-specific (for example `selective_contraction_pldm_noise_branch.*`) so that sanity runs do not overwrite the LeWM paper-facing branch summary. Do not compare LeWM and PLDM t-SNE coordinates directly; if PLDM visualization is used, treat it as a qualitative method-family sanity check and keep the high-D ACPC basin table as the evidence.

## Canonical artifact builders

这些脚本需要本机存在原始实验目录。用 `DATA_ROOT` 传入前缀，例如：

```text
$DATA_ROOT
```

| 脚本 | 作用 | 输出 |
|---|---|---|
| `tools/build_canonical_evals_pldm.py` | 从 PLDM 4 tasks x 9 checkpoints 的 `eval_results` 聚合 unperturbed（artifact key: `clean`）/ goal / observation-noise / observation+goal eval，3 evaluation seeds x 100 trajectories，population std | `assets/paper1_data/canonical_evals_pldm_20260522.json` |
| `tools/build_canonical_diagnostics_pldm.py` | 聚合 PLDM full-coverage predictor metrics：fragility ratio 和 8-step rollout drift | `assets/paper1_data/canonical_diagnostics_pldm_20260522.json` |
| `tools/build_canonical_full_diagnostics_pldm.py` | 聚合 PLDM five-layer diagnostics summary rows，并生成 schema | `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json` |
| `tools/build_canonical_blur_baselines.py` | 聚合 LeWM/PLDM no-noise baseline 的 blur eval-only 结果 | `assets/paper1_data/canonical_blur_baselines_20260523.json` |

示例：

```bash
python -m tools.build_canonical_evals_pldm \
  --root "$DATA_ROOT" \
  --out assets/paper1_data/canonical_evals_pldm_20260522.json

python -m tools.build_canonical_diagnostics_pldm \
  --root "$DATA_ROOT" \
  --out assets/paper1_data/canonical_diagnostics_pldm_20260522.json

python -m tools.build_canonical_full_diagnostics_pldm \
  --root "$DATA_ROOT" \
  --out assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json \
  --schema-out assets/paper1_data/canonical_full_diagnostics_pldm_20260523.schema.json

python -m tools.build_canonical_blur_baselines \
  --root "$DATA_ROOT" \
  --out assets/paper1_data/canonical_blur_baselines_20260523.json \
  --schema-out assets/paper1_data/canonical_blur_baselines_20260523.schema.json
```

## 建议执行顺序

1. 原始实验数据没有变化时，不要重跑 canonical builders，只运行 checker 和 LaTeX build。
2. PLDM 或 blur 原始结果变化时，先重建对应 canonical JSON，再重跑 `pldm_correlation_analysis.py` 和 `build_partial_corr_bootstrap.py`。
3. LeWM canonical eval/diagnostics 变化时，重跑 `paper1_figs.py`，再运行 consistency checker。
4. 提交前固定执行 `python -m tools.check_paper1_consistency` 和 `cd paper1 && bash build.sh --clean`。

## 低频工具

`tools/remap_canonical_std_keys.py` 是历史 artifact key remap 工具。正常 Paper 1 release 不需要执行，除非旧 JSON 的 `std_max` key 需要一次性迁移。
