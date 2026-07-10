# Paper1 Diagnostic Remediation Scripts

These scripts rebuild the training-free diagnostic artifacts added for the Paper1 experiment remediation. They do not retrain models or rerun closed-loop evaluation.

Run summary-level diagnostics from repository root:

```bash
bash paper1/scripts/run_all_paper1_diagnostics.sh
```

Rebuild checkpoint-level audits only when the LeWM checkpoints and datasets are available:

```bash
RUN_CHECKPOINT_AUDITS=1 bash paper1/scripts/run_all_paper1_diagnostics.sh
```

The checkpoint-level path recomputes the sample-level fixed-pool audit, finite-difference Gaussian sensitivity audit, and exact-autograd JVP/Hutchinson trace audit; it is slower than the summary-level path.

Individual steps:

```bash
python -m paper1.scripts.build_diagnostic_manifest
python -m paper1.scripts.build_full_sweep_diagnostics
python -m paper1.scripts.plot_full_sweep_diagnostics
python -m paper1.scripts.plot_endpoint_atr_smpr
python -m paper1.scripts.fixed_pool_tail_audit
python -m paper1.scripts.heldout_diagnostic_validation
python -m paper1.scripts.threshold_quantile_sensitivity
python -m tools.paper1_sample_level_certificate --out-json paper1/results/sample_level_certificate_full_sweep_audit.json --out-csv paper1/results/sample_level_certificate_full_sweep_audit.csv --sample-csv paper1/results/sample_level_certificate_full_sweep_samples.csv
python -m paper1.scripts.full_sweep_sample_level_certificate_summary
python -m paper1.scripts.sample_level_event_rate_ci
python -m paper1.scripts.plot_fixed_pool_event_rates
python -m tools.paper1_gaussian_sensitivity_audit --num-noise-draws 5
python -m tools.paper1_jvp_hutchinson_sensitivity_audit --n-sequences 100 --hutchinson-probes 8
python -m paper1.scripts.joint_guard_side_validation
python -m paper1.scripts.plot_gaussian_sensitivity_mechanism
```

Plot output notes:

- `plot_full_sweep_diagnostics` writes a main figure with separate behavior and direct ATR/SMPR axes per task, the diagnostic-region scatter, and a compact four-across appendix planner-guard figure; recovery shading is rendered as continuous majority-recovered ranges.
- `plot_endpoint_atr_smpr` writes the two-panel endpoint dumbbell figure with base-to-noise-trained movement arrows.
- `plot_gaussian_sensitivity_mechanism` writes a two-panel endpoint/base lollipop figure for the main text and the trace-decomposition heatmap plus separate alignment panel for the appendix.
- `plot_fixed_pool_event_rates` writes the two-panel paired event-rate figure; conditional flip-given-cert-pass rates remain in the appendix table.

Important scope constraints:

- Full-sweep diagnostics join existing Gaussian evaluation, ATR, SMPR, and retained fixed-pool summaries.
- Full-sweep sample-level fixed-pool event rates are recomputed from checkpoints; strict q10/q95 gaps remain negative and are not treated as calibrated probability bounds.
- Wilson intervals quantify sample event-rate estimation uncertainty; they are not calibrated theorem probability bounds. The event-rate figure is regenerated from `paper1/results/sample_level_event_rate_wilson_ci.csv`.
- Held-out validation freezes diagnostic gates on calibration rows before evaluating held-out rows.
- Gaussian sensitivity is audited with finite differences using 100 sampled sequences and 5 noise draws per small sigma, plus an exact-autograd JVP/Hutchinson trace decomposition using 100 sampled sequences and 8 Rademacher probes per checkpoint; both are local checkpoint audits, not closed-loop guarantees.
- SMPR and fixed-pool top-1 flip are guard-side checks interpreted only jointly with ATR, not standalone robustness metrics.
- ATR q80/q95 and positive-margin SMPR variants are not inferred from retained ATR/SMPR summaries.
