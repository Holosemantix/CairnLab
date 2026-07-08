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
        "raw_fixed_pool_source": "missing: retained artifacts contain q50/q90 summaries and fixed-pool flip rates, not sample-level candidate-cost traces",
        "jacobian_audit_source": "omitted: finite-difference/JVP checkpoint audit not run in this training-free remediation",
        "stronger_smpr_guard_source": "omitted: stronger simulator labels/action-distinct guard require additional state/action trace computation",
        "generated_outputs": [
            "paper1/results/full_sweep_diagnostics.csv",
            "paper1/results/full_sweep_diagnostics_summary.csv",
            "paper1/results/heldout_diagnostic_validation.csv",
            "paper1/results/heldout_gate_params.json",
            "paper1/results/fixed_pool_tail_audit.csv",
            "paper1/results/fixed_pool_tail_audit_summary.csv",
            "paper1/results/threshold_quantile_sensitivity.csv",
            "paper1/figures/fig_full_sweep_diagnostics.png",
            "paper1/figures/fig_full_sweep_diagnostic_region.png",
            "paper1/figures/fig_heldout_diagnostic_validation.png",
            "paper1/figures/fig_fixed_pool_tail_audit.png",
            "paper1/figures/fig_top1_agreement_full_sweep.png",
            "paper1/figures/fig_threshold_sensitivity.png",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "No retraining or closed-loop re-evaluation is performed by this remediation.",
            "q80/q95 ATR, q10 margin, q95/q99 drift, and pool cert-pass rates are not inferred from retained summaries.",
            "Held-out gates are calibrated on calibration rows only; held-out labels are used only for evaluation.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
