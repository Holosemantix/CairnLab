# Paper1 Diagnostic Remediation Scripts

These scripts rebuild the training-free diagnostic artifacts added for the Paper1 experiment remediation. They do not retrain models or rerun closed-loop evaluation.

Run from repository root:

```bash
bash paper1/scripts/run_all_paper1_diagnostics.sh
```

Individual steps:

```bash
python -m paper1.scripts.build_diagnostic_manifest
python -m paper1.scripts.build_full_sweep_diagnostics
python -m paper1.scripts.plot_full_sweep_diagnostics
python -m paper1.scripts.fixed_pool_tail_audit
python -m paper1.scripts.heldout_diagnostic_validation
python -m paper1.scripts.threshold_quantile_sensitivity
```

Important scope constraints:

- Full-sweep diagnostics join existing Gaussian evaluation, ATR, SMPR, and retained fixed-pool summaries.
- q80/q95 ATR, q10 clean margins, q95/q99 drift tails, and pool-level cert-pass rates are not inferred from summaries.
- Held-out validation freezes diagnostic gates on calibration rows before evaluating held-out rows.
- Gaussian finite-difference/JVP sensitivity and stronger action-distinct guard audits are not run here; they require additional checkpoint-level computation.
