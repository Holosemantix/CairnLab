#!/usr/bin/env python3
"""Create Paper1 diagnostic remediation manifest."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .utils_paper1_io import ROOT, RHO_GRID, SEEDS, TASKS

DEFAULT_OUT = ROOT / "paper1" / "results" / "diagnostic_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = {
        "tasks": TASKS,
        "training_seeds": SEEDS,
        "eval_seeds": [42, 43, 44],
        "rho_grid": [float(x) for x in RHO_GRID],
        "eval_noise_sigmas": [0.00, 0.03, 0.05, 0.08],
        "checkpoint_epoch": 10,
        "closed_loop_eval_source": "assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json",
        "diagnostic_source": "paper1/results/prospective_diagnostic/diagnostics_all_ckpts.csv",
        "smpr_label_source": "assets/paper1_data/semantic_task_grounded_margin_lewm_full_sweep_20260708.json",
        "fixed_pool_summary_source": "assets/paper1_data/acpc_phase0_lewm_three_seed.json",
        "raw_fixed_pool_source": "full-sweep recomputation implemented: paper1/results/sample_level_certificate_full_sweep_audit.json; retained summaries are still used for the q50/q90 proxy overlay",
        "jacobian_audit_source": "finite-difference checkpoint audit: paper1/results/gaussian_sensitivity_audit.json; exact-autograd JVP/Hutchinson decomposition: paper1/results/jvp_hutchinson_sensitivity_audit.json",
        "stronger_smpr_guard_source": "joint guard-side validation: paper1/results/joint_guard_side_validation.csv; SMPR and fixed-pool top1 flip are interpreted only with ATR",
        "generated_outputs": [
            "paper1/results/full_sweep_diagnostics.csv",
            "paper1/results/full_sweep_diagnostics_summary.csv",
            "paper1/results/heldout_diagnostic_validation.csv",
            "paper1/results/heldout_gate_params.json",
            "paper1/results/fixed_pool_tail_audit.csv",
            "paper1/results/fixed_pool_tail_audit_summary.csv",
            "paper1/results/threshold_quantile_sensitivity.csv",
            "assets/paper1_figs/fig_full_sweep_diagnostics.png",
            "assets/paper1_figs/fig_full_sweep_diagnostic_region.png",
            "assets/paper1_figs/fig_endpoint_atr_smpr.png",
            "assets/paper1_figs/fig_heldout_diagnostic_validation.png",
            "assets/paper1_figs/fig_fixed_pool_tail_audit.png",
            "assets/paper1_figs/fig_top1_agreement_full_sweep.png",
            "assets/paper1_figs/fig_threshold_sensitivity.png",
            "paper1/results/sample_level_certificate_full_sweep_audit.csv",
            "paper1/results/sample_level_certificate_full_sweep_audit.json",
            "paper1/results/sample_level_certificate_full_sweep_samples.csv",
            "paper1/results/sample_level_certificate_full_sweep_summary.csv",
            "paper1/results/sample_level_certificate_recovery_alignment.csv",
            "paper1/results/sample_level_event_rate_wilson_ci.csv",
            "paper1/tables/table_sample_level_certificate_full_sweep.tex",
            "paper1/tables/table_sample_level_event_rate_ci.tex",
            "assets/paper1_figs/fig_fixed_pool_event_rates.png",
            "paper1/results/sample_level_certificate_endpoint_audit.csv",
            "paper1/results/sample_level_certificate_endpoint_audit.json",
            "paper1/results/sample_level_certificate_endpoint_samples.csv",
            "paper1/results/sample_level_certificate_endpoint_summary.csv",
            "paper1/tables/table_sample_level_certificate_endpoint.tex",
            "paper1/results/gaussian_sensitivity_audit.csv",
            "paper1/results/gaussian_sensitivity_audit.json",
            "paper1/results/gaussian_sensitivity_summary.csv",
            "paper1/tables/table_gaussian_sensitivity_audit.tex",
            "paper1/results/jvp_hutchinson_sensitivity_audit.csv",
            "paper1/results/jvp_hutchinson_sensitivity_audit.json",
            "paper1/results/jvp_hutchinson_sensitivity_summary.csv",
            "paper1/tables/table_jvp_hutchinson_sensitivity_audit.tex",
            "assets/paper1_figs/fig_gaussian_sensitivity_mechanism.png",
            "paper1/tables/table_endpoint_atr_smpr.tex",
            "paper1/tables/table_theory_evidence_map.tex",
            "paper1/results/joint_guard_side_validation.csv",
            "paper1/tables/table_joint_guard_side_validation.tex",
        ],
        "sample_level_certificate_audit": {
            "scope": "full Gaussian sweep for four tasks and training seeds 3072/3073/3074",
            "n_sequences_per_seed_checkpoint": 100,
            "candidate_count": 65,
            "result_csv": "paper1/results/sample_level_certificate_full_sweep_summary.csv",
            "interpretation": "cert-pass and top1-flip separation strengthens fixed-pool mechanism audit; strict q10/q95 gaps remain negative, so this is not a calibrated certificate",
        },
        "gaussian_sensitivity_audit": {
            "scope": "base, recovery-onset, and endpoint checkpoints for four tasks and training seeds 3072/3073/3074",
            "small_sigmas": [0.005, 0.01, 0.02],
            "n_sequences_per_checkpoint": 100,
            "num_noise_draws_per_small_sigma": 5,
            "result_csv": "paper1/results/gaussian_sensitivity_summary.csv",
            "interpretation": "finite-difference local sensitivity proxy; not a global robustness guarantee",
        },
        "jvp_hutchinson_sensitivity_audit": {
            "scope": "base, recovery-onset, and endpoint checkpoints for four tasks and training seeds 3072/3073/3074",
            "n_sequences_per_checkpoint": 100,
            "hutchinson_probes_per_checkpoint": 8,
            "result_csv": "paper1/results/jvp_hutchinson_sensitivity_summary.csv",
            "interpretation": "exact-autograd JVP/Hutchinson local Frobenius-trace decomposition; not a full Jacobian matrix or closed-loop guarantee",
        },
        "joint_guard_side_validation": {
            "result_csv": "paper1/results/joint_guard_side_validation.csv",
            "interpretation": "ATR is the radius term; SMPR and fixed-pool top1 flip are guard-side checks, not standalone robustness metrics",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "No retraining or closed-loop re-evaluation is performed by this remediation.",
            "Full-sweep sample-level fixed-pool event rates are recomputed from checkpoints; strict q10/q95 gaps remain negative and are not calibrated probability bounds.",
            "Wilson intervals quantify sample event-rate estimation uncertainty, not theorem-calibrated probabilities.",
            "Held-out gates are calibrated on calibration rows only; held-out labels are used only for evaluation.",
            "SMPR and fixed-pool top1 flip are guard-side checks interpreted jointly with ATR, not standalone robustness metrics.",
            "Exact-autograd JVP/Hutchinson traces decompose local encoder, rollout, and composed sensitivity but do not materialize a full Jacobian or prove closed-loop robustness.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
