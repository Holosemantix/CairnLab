#!/usr/bin/env bash
# Rebuild Paper1 training-free diagnostic remediation artifacts.
set -euo pipefail

python -m paper1.scripts.build_diagnostic_manifest
python -m paper1.scripts.build_full_sweep_diagnostics
python -m paper1.scripts.plot_full_sweep_diagnostics
python -m paper1.scripts.fixed_pool_tail_audit
python -m paper1.scripts.heldout_diagnostic_validation
python -m paper1.scripts.threshold_quantile_sensitivity

if [[ "${RUN_CHECKPOINT_AUDITS:-0}" == "1" ]]; then
  python -m tools.paper1_sample_level_certificate \
    --tasks TwoRoom PushT Reacher Cube \
    --seeds 3072 3073 3074 \
    --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
    --n-sequences 100 \
    --include-samples \
    --out-json paper1/results/sample_level_certificate_full_sweep_audit.json \
    --out-csv paper1/results/sample_level_certificate_full_sweep_audit.csv \
    --sample-csv paper1/results/sample_level_certificate_full_sweep_samples.csv
  python -m paper1.scripts.full_sweep_sample_level_certificate_summary
  python -m paper1.scripts.sample_level_event_rate_ci
  python -m tools.paper1_gaussian_sensitivity_audit \
    --tasks TwoRoom PushT Reacher Cube \
    --seeds 3072 3073 3074 \
    --n-sequences 100 \
    --num-noise-draws 1 \
    --small-sigmas 0.005 0.01 0.02
  python -m tools.paper1_jvp_hutchinson_sensitivity_audit \
    --tasks TwoRoom PushT Reacher Cube \
    --seeds 3072 3073 3074 \
    --n-sequences 16 \
    --hutchinson-probes 8
  python -m paper1.scripts.joint_guard_side_validation
fi
