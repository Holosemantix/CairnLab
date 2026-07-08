# Prospective ATR/SMPR Diagnostic Validation

This directory is an internal Paper1 validation artifact. It tests whether thresholds calibrated on training seed 3072 predict robust intervals on held-out training seeds 3073/3074 before reading their closed-loop scores.

Full-sweep SMPR computation command:

```bash
python -m tools.paper1_semantic_margin \
  --seeds 3072 3073 3074 \
  --tasks TwoRoom PushT Reacher Cube \
  --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
  --n-sequences 100 \
  --device cuda \
  --pair-rule task_grounded_near_boundary \
  --out assets/paper1_data/semantic_task_grounded_margin_lewm_full_sweep_20260708.json
```

Validation command after the SMPR artifact exists:

```bash
python -m tools.paper1_prospective_atr_smpr_validation \
  --smpr assets/paper1_data/semantic_task_grounded_margin_lewm_full_sweep_20260708.json \
  --rho 0.80 \
  --out-dir paper1/results/prospective_diagnostic
```

Leakage rule: `predictions_heldout.csv` is written without score, return, success, eval_score, or true-label columns. Held-out closed-loop scores are joined only in `validation_rows_with_scores.csv` and the validation summary files.

`figures/` contains review plots for the default `per_task` + `atr_smpr` held-out rule; these are internal result-inspection figures, not paper figures by default.
