# Paper 1 工具说明

本目录下和 Paper 1 直接相关的脚本分三类：release gate、图表/统计复现、从本地原始实验目录重新聚合 canonical artifact。除非特别说明，命令都从仓库根目录 `/home/ag/projects/wm_exp` 执行。

## 最常用命令

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

`check_paper1_consistency.py` 是提交前的主检查入口。它验证必需 artifact 是否存在、旧口径字符串是否消失、LeWM/PLDM/blur canonical JSON 结构是否完整、正文引用的关键相关系数和 bootstrap CI 是否能从 artifact 复算到相同数值。

`paper1/build.sh --clean` 用 `latexmk` 重建 PDF。构建后建议检查 log：

```bash
rg -n "Overfull|undefined references|Citation .* undefined|Reference .* undefined|Fatal error|Undefined control sequence" paper1/main.log || true
```

## 图和统计

| 脚本 | 作用 | 输入 | 输出 / 用途 |
|---|---|---|---|
| `tools/paper1_figs.py` | 渲染主文图 | `assets/paper1_data/canonical_evals_20260517.json`, `assets/paper1_data/canonical_diagnostics_20260517.json` | `assets/paper1_figs/fig1_concept.png`, `fig2_sweep.png`, `fig4_radar.png`, `fig5_scatter.png`, `fig6_mechanism.png`；`fig3_pareto.png` 可用 `--only 3` 生成但当前不进正文 |
| `tools/build_partial_corr_bootstrap.py` | 为 partial Spearman 相关计算 95% percentile bootstrap CI | LeWM/PLDM canonical eval + diagnostics artifact | `assets/paper1_data/partial_corr_bootstrap_20260523.json`，用于主文 Table 7 和 Appendix F |
| `tools/pldm_correlation_analysis.py` | 复算 LeWM/PLDM within-method 与 joint partial correlation | LeWM/PLDM canonical eval + diagnostics artifact | `assets/paper1_data/cross_method_corr_pldm_20260522.json`，用于 Appendix F 和 consistency checker |
| `tools/paper1_phase0_acpc.py` | Phase 0 paired ACPC 诊断 runner：ACPC-1/H、PCC、CRA、MAF、ADM proxy、SPRR | LeWM/PLDM canonical eval manifest + 本地 loadable model checkpoints | `assets/paper1_data/acpc_phase0_diagnostics.json`；dry-run 可先检查哪些 checkpoint 能解析 |

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

python -m tools.paper1_phase0_acpc \
  --dry-run --methods LeWM PLDM --tasks PushT \
  --out /tmp/acpc_phase0_dry.json

python -m tools.paper1_phase0_acpc \
  --methods LeWM --tasks PushT --std-keys 0.0 0.03 0.06 \
  --n-sequences 100 --random-action-trials 64 \
  --out assets/paper1_data/acpc_phase0_diagnostics.json
```

95% CI 的口径：脚本对 checkpoint rows 做 with-replacement bootstrap。within-LeWM 和 within-PLDM 是每个 task 的 9 个 checkpoint rows；joint 分析是 LeWM+PLDM 共 18 个 rows，并在 partial correlation 中同时 conditioning on `std_max` 和 `method`。CI 是 bootstrap 分布的 2.5/97.5 percentile，不是额外 evaluation seed 的置信区间。

Phase 0 ACPC runner 的 `--dry-run` 只解析 manifest 和 checkpoint 路径，不需要 `torch`。实际计算需要当前 Python 环境能 import `torch`、`stable_pretraining`、`stable_worldmodel`，且 canonical eval 里的 `path` 或 `--model-root` 下存在可 `torch.load` 的 model object checkpoint。当前 ADM 是 action-distance latent proxy，不是 oracle state/keypoint ADM。

## Canonical artifact builders

这些脚本需要本机存在原始实验目录，默认 root 是：

```text
/home/ag/dataset/ag_data/data/world_model/quentinll
```

| 脚本 | 作用 | 输出 |
|---|---|---|
| `tools/build_canonical_evals_pldm.py` | 从 PLDM 4 tasks x 9 checkpoints 的 `eval_results` 聚合 unperturbed（artifact key: `clean`）/ goal / pixels / pixels+goal eval，3 evaluation seeds x 100 trajectories，population std | `assets/paper1_data/canonical_evals_pldm_20260522.json` |
| `tools/build_canonical_diagnostics_pldm.py` | 聚合 PLDM full-coverage predictor metrics：fragility ratio 和 T8 drift | `assets/paper1_data/canonical_diagnostics_pldm_20260522.json` |
| `tools/build_canonical_full_diagnostics_pldm.py` | 聚合 PLDM five-layer diagnostics summary rows，并生成 schema | `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json` |
| `tools/build_canonical_blur_baselines.py` | 聚合 LeWM/PLDM no-noise baseline 的 blur eval-only 结果 | `assets/paper1_data/canonical_blur_baselines_20260523.json` |

示例：

```bash
python -m tools.build_canonical_evals_pldm \
  --root /home/ag/dataset/ag_data/data/world_model/quentinll \
  --out assets/paper1_data/canonical_evals_pldm_20260522.json

python -m tools.build_canonical_diagnostics_pldm \
  --root /home/ag/dataset/ag_data/data/world_model/quentinll \
  --out assets/paper1_data/canonical_diagnostics_pldm_20260522.json

python -m tools.build_canonical_full_diagnostics_pldm \
  --root /home/ag/dataset/ag_data/data/world_model/quentinll \
  --out assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json \
  --schema-out assets/paper1_data/canonical_full_diagnostics_pldm_20260523.schema.json

python -m tools.build_canonical_blur_baselines \
  --root /home/ag/dataset/ag_data/data/world_model/quentinll \
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
