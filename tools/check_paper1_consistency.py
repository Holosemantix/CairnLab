#!/usr/bin/env python3
"""Release consistency checks for Paper 1.

Usage:
    python -m tools.check_paper1_consistency
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RELEASE_FILES = [
    # PDF-facing text gate. Legacy diagnostics remain in repository artifacts,
    # but Paper1 must not re-import them into the main claim.
    ROOT / "paper1" / "main.tex",
]

REQUIRED_ARTIFACTS = [
    ROOT / "assets" / "paper1_data" / "canonical_evals_20260517.json",
    ROOT / "assets" / "paper1_data" / "canonical_evals_20260517.schema.json",
    ROOT / "assets" / "paper1_data" / "canonical_diagnostics_20260517.json",
    ROOT / "assets" / "paper1_data" / "canonical_diagnostics_20260517.schema.json",
    ROOT / "assets" / "paper1_data" / "canonical_external_baselines_20260520.json",
    ROOT / "assets" / "paper1_data" / "canonical_external_baselines_20260520.schema.json",
    # PLDM cross-method replication (added 2026-05-22)
    ROOT / "assets" / "paper1_data" / "canonical_evals_pldm_20260522.json",
    ROOT / "assets" / "paper1_data" / "canonical_diagnostics_pldm_20260522.json",
    ROOT / "assets" / "paper1_data" / "cross_method_corr_pldm_20260522.json",
    ROOT / "assets" / "paper1_data" / "canonical_full_diagnostics_pldm_20260523.json",
    ROOT / "assets" / "paper1_data" / "canonical_full_diagnostics_pldm_20260523.schema.json",
    ROOT / "assets" / "paper1_data" / "canonical_blur_baselines_20260523.json",
    ROOT / "assets" / "paper1_data" / "canonical_blur_baselines_20260523.schema.json",
    ROOT / "assets" / "paper1_data" / "acpc_basin_diagnostics.json",
    ROOT / "assets" / "paper1_data" / "acpc_basin_diagnostics_pldm.json",
    ROOT / "assets" / "paper1_data" / "partial_corr_bootstrap_20260523.json",
    ROOT / "assets" / "paper1_data" / "acpc_phase0_clean_goal_seed9101.json",
    ROOT / "assets" / "paper1_data" / "target_view_closed_loop_summary.json",
    ROOT / "assets" / "paper1_data" / "no_retrain_diagnostic_audit.json",
    ROOT / "assets" / "paper1_data" / "unseen_origin_vs_std008_strongest_tworoom.json",
    ROOT / "assets" / "paper1_data" / "unseen_origin_vs_std008_strongest_reacher.json",
    ROOT / "assets" / "paper1_data" / "unseen_origin_vs_std008_strongest_s3072.json",
    ROOT / "assets" / "paper1_data" / "unseen_origin_vs_std008_strongest_s3072.schema.json",
    ROOT / "assets" / "paper1_data" / "unseen_origin_vs_std008_strongest_s3072_manifest.json",
    ROOT / "assets" / "paper1_data" / "unseen_origin_vs_std008_strongest_s3073.json",
    ROOT / "assets" / "paper1_data" / "unseen_origin_vs_std008_strongest_s3074.json",
    ROOT / "assets" / "paper1_data" / "training_seed_gaussian_lockbox.json",
    ROOT / "assets" / "paper1_data" / "training_seed_gaussian_lockbox.md",
    ROOT / "assets" / "paper1_data" / "prospective_validation_summary.json",
    ROOT / "assets" / "paper1_data" / "prospective_validation_summary.md",
    ROOT / "assets" / "paper1_data" / "unseen_phase0_acpc_fullstress.json",
    ROOT / "assets" / "paper1_data" / "unseen_phase0_acpc_fullstress.schema.json",
    ROOT / "assets" / "paper1_data" / "training_seed_eval_manifests" / "lewm_seed3072_evals.json",
    ROOT / "assets" / "paper1_data" / "training_seed_eval_manifests" / "lewm_seed3073_evals.json",
    ROOT / "assets" / "paper1_data" / "training_seed_eval_manifests" / "lewm_seed3074_evals.json",
    ROOT / "assets" / "paper1_data" / "acpc_phase0_lewm_seed3072.json",
    ROOT / "assets" / "paper1_data" / "acpc_phase0_lewm_seed3073.json",
    ROOT / "assets" / "paper1_data" / "acpc_phase0_lewm_seed3074.json",
    ROOT / "assets" / "paper1_data" / "acpc_phase0_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "three_seed_diagnostic_validation.json",
    ROOT / "assets" / "paper1_data" / "three_seed_diagnostic_validation.md",
    ROOT / "assets" / "paper1_data" / "selector_baseline_audit_20260704.json",
    ROOT / "assets" / "paper1_data" / "selector_baseline_audit_20260704.md",
    ROOT / "assets" / "paper1_data" / "selector_plateau_audit_20260704.json",
    ROOT / "assets" / "paper1_data" / "selector_plateau_audit_20260704.md",
    ROOT / "assets" / "paper1_data" / "residual_diagnostic_audit_20260704.json",
    ROOT / "assets" / "paper1_data" / "residual_diagnostic_audit_20260704.md",
    ROOT / "assets" / "paper1_data" / "selector_incremental_audit_20260704.json",
    ROOT / "assets" / "paper1_data" / "selector_incremental_audit_20260704.md",
    ROOT / "assets" / "paper1_data" / "margin_flip_curve_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_margin_passrate_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_margin_passrate_lewm_three_seed.md",
    ROOT / "assets" / "paper1_data" / "semantic_local_margin_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_task_grounded_margin_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_task_grounded_margin_lewm_full_sweep_20260708.json",
    ROOT / "assets" / "paper1_data" / "semantic_task_grounded_margin_lewm_three_seed.md",
    ROOT / "assets" / "paper1_data" / "semantic_task_grounded_margin_unseen_blur_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_task_grounded_margin_unseen_resize_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "unseen_atr_smpr_summary_20260707.json",
    ROOT / "assets" / "paper1_data" / "unseen_atr_smpr_summary_20260707.md",
    ROOT / "assets" / "paper1_data" / "cem_trace_audit_20260704.json",
    ROOT / "assets" / "paper1_data" / "cem_trace_audit_20260704.md",
    ROOT / "assets" / "paper1_data" / "compressed_metrics_summary_20260706.json",
    ROOT / "assets" / "paper1_data" / "compressed_metrics_summary_20260706.md",
    ROOT / "assets" / "paper1_data" / "base_noise_cliff_multistd_20260706.json",
    ROOT / "assets" / "paper1_data" / "base_noise_cliff_multistd_20260706.md",
    ROOT / "assets" / "paper1_data" / "three_seed_gaussian_sweep_summary_20260706.json",
    ROOT / "assets" / "paper1_data" / "three_seed_gaussian_sweep_summary_20260706.md",
    ROOT / "assets" / "paper1_figs" / "fig1_concept.png",
    ROOT / "assets" / "paper1_figs" / "fig_acpc_basin_tsne.png",
    ROOT / "assets" / "paper1_figs" / "fig_full_sweep_diagnostics.png",
    ROOT / "assets" / "paper1_figs" / "fig_full_sweep_diagnostic_region.png",
    ROOT / "assets" / "paper1_figs" / "fig_endpoint_atr_smpr.png",
    ROOT / "assets" / "paper1_figs" / "fig_heldout_diagnostic_validation.png",
    ROOT / "assets" / "paper1_figs" / "fig_fixed_pool_tail_audit.png",
    ROOT / "assets" / "paper1_figs" / "fig_top1_agreement_full_sweep.png",
    ROOT / "assets" / "paper1_figs" / "fig_threshold_sensitivity.png",
    ROOT / "assets" / "paper1_figs" / "fig_radius_margin_interval_overlay.png",
    ROOT / "assets" / "paper1_figs" / "fig_radius_margin_overlap.png",
    ROOT / "assets" / "paper1_figs" / "fig_fixed_pool_event_rates.png",
    ROOT / "assets" / "paper1_figs" / "fig_gaussian_sensitivity_mechanism.png",
    ROOT / "paper1" / "results" / "radius_margin_certificate_summary.csv",
    ROOT / "paper1" / "results" / "radius_margin_gate_ablation.csv",
    ROOT / "paper1" / "results" / "radius_margin_boundary_alignment.csv",
    ROOT / "paper1" / "results" / "fixed_pool_top1_agreement.csv",
    ROOT / "paper1" / "results" / "prospective_diagnostic" / "diagnostics_all_ckpts.csv",
    ROOT / "paper1" / "results" / "diagnostic_region" / "diagnostic_region_rows.csv",
    ROOT / "paper1" / "results" / "diagnostic_region" / "diagnostic_region_summary.csv",
    ROOT / "paper1" / "results" / "diagnostic_region" / "direction_consistency_by_block.csv",
    ROOT / "paper1" / "results" / "diagnostic_region" / "direction_consistency_summary.csv",
    ROOT / "paper1" / "results" / "diagnostic_region" / "robust_fragile_separation.csv",
    ROOT / "paper1" / "results" / "diagnostic_region" / "README.md",
    ROOT / "paper1" / "results" / "diagnostic_manifest.json",
    ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv",
    ROOT / "paper1" / "results" / "full_sweep_diagnostics_summary.csv",
    ROOT / "paper1" / "results" / "heldout_diagnostic_validation.csv",
    ROOT / "paper1" / "results" / "heldout_gate_params.json",
    ROOT / "paper1" / "results" / "fixed_pool_tail_audit.csv",
    ROOT / "paper1" / "results" / "fixed_pool_tail_audit_summary.csv",
    ROOT / "paper1" / "results" / "threshold_quantile_sensitivity.csv",
    ROOT / "paper1" / "results" / "MISSING_DATA_fixed_pool_tail_audit.md",
    ROOT / "paper1" / "results" / "sample_level_certificate_endpoint_audit.json",
    ROOT / "paper1" / "results" / "sample_level_certificate_endpoint_audit.csv",
    ROOT / "paper1" / "results" / "sample_level_certificate_endpoint_samples.csv",
    ROOT / "paper1" / "results" / "sample_level_certificate_endpoint_summary.csv",
    ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_audit.json",
    ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_audit.csv",
    ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_samples.csv",
    ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_summary.csv",
    ROOT / "paper1" / "results" / "sample_level_certificate_recovery_alignment.csv",
    ROOT / "paper1" / "results" / "sample_level_event_rate_wilson_ci.csv",
    ROOT / "paper1" / "results" / "gaussian_sensitivity_audit.json",
    ROOT / "paper1" / "results" / "gaussian_sensitivity_audit.csv",
    ROOT / "paper1" / "results" / "gaussian_sensitivity_summary.csv",
    ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_audit.json",
    ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_audit.csv",
    ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_summary.csv",
    ROOT / "paper1" / "results" / "joint_guard_side_validation.csv",
    ROOT / "paper1" / "tables" / "table_heldout_diagnostic_validation.tex",
    ROOT / "paper1" / "tables" / "table_endpoint_atr_smpr.tex",
    ROOT / "paper1" / "tables" / "table_fixed_pool_tail_audit.tex",
    ROOT / "paper1" / "tables" / "table_sample_level_certificate_full_sweep.tex",
    ROOT / "paper1" / "tables" / "table_sample_level_event_rate_ci.tex",
    ROOT / "paper1" / "tables" / "table_sample_level_certificate_endpoint.tex",
    ROOT / "paper1" / "tables" / "table_joint_guard_side_validation.tex",
    ROOT / "paper1" / "tables" / "table_gaussian_sensitivity_audit.tex",
    ROOT / "paper1" / "tables" / "table_jvp_hutchinson_sensitivity_audit.tex",
    ROOT / "paper1" / "tables" / "table_theory_evidence_map.tex",
    ROOT / "paper1" / "tables" / "table_threshold_quantile_sensitivity.tex",
    ROOT / "paper1" / "scripts" / "build_diagnostic_manifest.py",
    ROOT / "paper1" / "scripts" / "build_full_sweep_diagnostics.py",
    ROOT / "paper1" / "scripts" / "fixed_pool_tail_audit.py",
    ROOT / "paper1" / "scripts" / "heldout_diagnostic_validation.py",
    ROOT / "paper1" / "scripts" / "plot_full_sweep_diagnostics.py",
    ROOT / "paper1" / "scripts" / "plot_endpoint_atr_smpr.py",
    ROOT / "paper1" / "scripts" / "plot_fixed_pool_event_rates.py",
    ROOT / "paper1" / "scripts" / "plot_gaussian_sensitivity_mechanism.py",
    ROOT / "paper1" / "scripts" / "threshold_quantile_sensitivity.py",
    ROOT / "paper1" / "scripts" / "utils_paper1_io.py",
    ROOT / "paper1" / "scripts" / "sample_level_certificate_summary.py",
    ROOT / "paper1" / "scripts" / "full_sweep_sample_level_certificate_summary.py",
    ROOT / "paper1" / "scripts" / "sample_level_event_rate_ci.py",
    ROOT / "paper1" / "scripts" / "joint_guard_side_validation.py",
    ROOT / "tools" / "paper1_sample_level_certificate.py",
    ROOT / "tools" / "paper1_gaussian_sensitivity_audit.py",
    ROOT / "tools" / "paper1_jvp_hutchinson_sensitivity_audit.py",
    ROOT / "paper1" / "scripts" / "run_all_paper1_diagnostics.sh",
    ROOT / "paper1" / "scripts" / "README.md",
    ROOT / "paper1" / "docs" / "codex_paper1_experiment_remediation_plan.md",
    ROOT / "paper1" / "docs" / "EXPERIMENT_REMEDIATION_STATUS_20260708.md",
    ROOT / "tools" / "paper1_radius_margin_certificate.py",
    ROOT / "tools" / "paper1_boundary_mechanism_audit.py",
    ROOT / "tools" / "paper1_diagnostic_region_validation.py",
    ROOT / "DATA_MANIFEST.md",
]


REQUIRED_MAIN_TEXT_SNIPPETS = [
    "ACPC Tail Risk (ATR)",
    "Selective Margin Pass Rate (SMPR)",
    "The reported diagnostic uses two metrics matched to this radius--margin logic",
    "ATR and SMPR are empirical diagnostics aligned with the radius and margin sides, not calibrated flip-probability bounds",
    "fixed empirical reporting choice, not a theoretical constant",
    "The same guard can be posed over state--action pairs",
    "fig:acpc-concept",
    "Encoder geometry remains an indispensable first-stage risk signal",
    "raw encoder distance alone is not a complete robustness criterion",
    "It need not reduce the rollout-side Jacobian uniformly",
    "lower composed encoder--rollout response to actual noise-induced perturbations",
    "Low ATR without high SMPR is not interpreted as robustness",
    "Three-training-seed LeWM Gaussian sweep",
    "Because \Cref{fig:sweep} already aggregates the full sweep across three training seeds",
    "fig:endpoint-atr-smpr",
    "Qualitative PushT ACPC neighborhood t-SNE visualization",
    "fig_fixed_pool_event_rates.png",
    "fig_gaussian_sensitivity_mechanism.png",
    "stressor-specific ATR/SMPR diagnostics",
    "raw no-noise $\\to$ noise-trained diagnostic values under the row stressor",
    "We treat this as bounded behavior outside the matched Gaussian setting",
    "These rows therefore support only a bounded severe-stressor association rather than a general perturbation-transfer claim",
    "programmatic task-state proxy labels",
    "The joint fixed-pool top-1 flip guard mitigates this proxy-label limitation",
    "does not replace stronger oracle-level contact, topology, action-value, or cost-to-go semantics",
    "closest 35\% state-distance neighborhood",
    "Hand-labeled or simulator-derived contact, topology, action-value, or cost-to-go labels remain future validation",
    "These proofs support the diagnostic use of ATR and SMPR",
    "Additional Gaussian Evaluation Tables",
    "These tables report the full observation-only Gaussian evaluation columns available",
    "Future methods can turn ATR/SMPR into objectives",
    "ACPC radius--margin diagnostic certificate",
    "Same-state predictive radius and selective margin",
    "ACPC radius--margin certificate",
    "Matched-perturbation diagnostic region",
    "Local Gaussian ACPC radius quantile",
    "encoder-side repair, rollout-side contraction, or alignment repair",
    "Planner-side radius--margin audit",
    "mechanism proxy, not a planner-margin certificate",
    "finite-sample empirical fixed-pool risk audit",
    "normalized ATR tail and SMPR failure",
    "Those q10/q95 gaps remain negative",
    "aggregate grid-point F1 or interval IoU is not used as the primary validation criterion",
    "Full-sweep and held-out evidence",
    "full-sweep analysis over all $4\\times3\\times9=108$",
    "mean absolute recovery-onset error $0.007$",
    "leave-one-task-out validation",
    "Threshold sensitivity is reported",
    "not as universal checkpoint rankers",
    "Boundary-aware interpretation of the fixed-pool radius--margin proxy",
    "Fixed-pool top-1 agreement analysis derived from the measured fixed-pool flip rate",
    "Diagnostic Validation Details",
    "Local Sensitivity Details",
    "Boundary Stressors",
    "Reproducibility",
    "The separate sample-level recomputation evaluates the sufficient event directly over the fixed 65-candidate pool for all 108 rows",
    "Those q10/q95 gaps remain negative",
    "median cert-pass rises from $0.06$",
    "Local Gaussian sensitivity explains ATR contraction",
    "endpoint/base reduction in this local slope",
    "using 100 sampled sequences and 5 noise draws per small $\sigma$ and checkpoint",
    "exact-autograd JVP/Hutchinson Frobenius traces",
    "using 100 sampled sequences and 8 Rademacher probes per checkpoint",
    "The two estimators are not expected to match numerically",
    "not as a cross-task scale or a shared threshold",
    "Complementary finite-difference and exact-JVP/Hutchinson analyses",
    "maps each theoretical object to its empirical evidence and limitation",
    "the decomposition attributes the composed-trace reduction mainly to encoder-side sensitivity reduction",
    "rollout-side trace is task-dependent rather than uniformly smaller",
    "tighter post-rollout feature clouds measure the composed response",
    "not claim a standalone predictor-Jacobian repair",
    "observed top-1 flips conditioned on cert-pass are zero",
    "rather than calibrated probabilities",
    "SMPR and fixed-pool top-1 flip are guard-side criteria and are not standalone robustness metrics",
    "unavailable ATR/SMPR tail variants are reported as unavailable rather than inferred",
    "q10/q95 rule is too conservative for the current fixed-pool cost scale",
    "$\\beta_{\\mathrm{plan}}$ and $\\beta_{\\mathrm{disc}}$ name empirical failure components rather than calibrated probabilities",
    "The radius--margin certificate is fixed-pool and matched-perturbation only",
    "adaptive CEM resampling, repeated replanning, or environment-feedback trajectory guarantees",
    "The Gaussian sensitivity analyses are local: finite-difference slopes and exact-JVP/Hutchinson trace estimates do not provide a global robustness or closed-loop guarantee",
    "SMPR is only a guard-side component of the joint diagnostic and uses programmatic proxy labels",
    "fixed-pool top-1 flip guard evaluates planning consistency but does not replace stronger semantic labels",
    "radius--margin diagnostic theory for fixed-checkpoint Gaussian robustness",
    "Radius--margin parameter interpretation",
    "candidate count is $K=65$",
    "A q90 summary descriptively leaves 10\\%",
    "not as a calibrated probability guarantee",
    "65\\times0.1=6.5",
    "tab:appendix-radius-margin-params",
    "Across the full Gaussian training sweep, recovered rows occupy low-ATR/high-SMPR regions",
    "full-sweep fixed-pool event-rate recomputation links the radius--margin mechanism to candidate stability",
    "The separate sample-level recomputation in \Cref{tab:sample-level-certificate-full-sweep} provides q10/q95 gaps and event rates",
    "q10/q95 gaps remain negative, so the analysis supports the mechanism rather than a calibrated planner-margin certificate",
    "observed top-1 flips conditioned on cert-pass are zero",
    "paired event-rate calibration",
    "\\hat p_{\\mathrm{flip}\\mid\\mathrm{cert}}",
    "Fixed-pool event-rate audit",
    "maps each theoretical object to its empirical evidence and limitation",
    "The sufficient-event interpretation is sample-level",
    "event rates remain informative even though the distribution-level q10/q95 gap is negative",
]




FORBIDDEN_SNIPPETS = [
    "H8 predictor",
    "H=8 predictor",
    "8-step predictor basin",
    "ACPC rollout $R_F$",
    "R_E",
    "R_F",
    "PCC",
    "CRA",
    "MAF",
    "CEM trace",
    "CEMSolver",
    "ID probe",
    "effective rank",
    "transition-resolution",
    "Phase-0",
    "selector",
    "legacy",
    "DATA_MANIFEST",
    "manifest",
    "release package",
    "JSON",
    "hash",
    "rendering",
    "scripts",
    "selective_contraction",
    "fig:selective",
    "tab:acpc-basin",
    "tab:acpc-downstream",
    "tab:diag-base",
    "tab:margin-flip",
    "tab:selector",
    "tab:cem",
    "appendix-phase0",
    "appendix-selector",
    "appendix-cem",
    "appendix-diagnostic-framework",
    "appendix-pldm",
    "appendix-A-atlas",
    "public repository",
    "data manifest",
    "coarse collapse",
    "The data reject",
    "bought by",
    "small CEM trace audit",
    "point-optimal std prediction",
    "heteroscedastic",
    "Heteroscedastic",
    "target-view",
    "Target-view",
    "Target-View",
    "clean-target denoising",
    "negative ablation",
]


EXPECTED_TASKS = {"TwoRoom", "PushT", "Reacher", "Cube"}
EXPECTED_CONFIGS = {
    "0.0",
    "0.01",
    "0.02",
    "0.03",
    "0.04",
    "0.05",
    "0.06",
    "0.07",
    "0.08",
}
REQUIRED_METRICS = {
    "clean",
    "pixels_std0.05",
    "pixels_std0.08",
    "pixels_goal_std0.05",
    "pixels_goal_std0.08",
}
THREE_SEED_SWEEP_METRICS = {
    "clean",
    "obs_sigma_0.03",
    "obs_sigma_0.05",
    "obs_sigma_0.08",
    "obs_goal_sigma_0.08",
}
REQUIRED_DIAG_TASKS = EXPECTED_TASKS
EXPECTED_METHODS = {"LeWM", "PLDM"}
EXPECTED_BLUR_CONDITIONS = {
    f"{scope}_blur_ks{kernel}"
    for scope in ("pixels", "goal", "pixels_goal")
    for kernel in (3, 7, 11, 15)
}
EXPECTED_PLDM_FULL_DIAG_METRICS = {
    "clean_effective_rank",
    "clean_nn_cos_dist_median",
    "transition_resolution_ratio_l2",
    "transition_resolution_ratio_cos",
    "id_probe_r2",
    "action_mean_pred_shift_norm",
    "predictor_target_to_nn_cos_ratio_at_max_std",
    "predictor_rollout_T8_l2",
}
EXPECTED_ACPC_PHASE0_METRICS = {
    "encoder_shift_to_nn_l2",
    "acpc_1_norm_by_transition",
    "acpc_h_norm_by_transition",
    "pcc_abs_median",
    "pcc_abs_p90",
    "cra_spearman_mean",
    "elite_overlap_mean",
    "maf_flip_rate",
    "adm_l2_median",
    "sprr",
}
EXPECTED_BOOTSTRAP_SCOPES = {"within_lewm", "within_pldm", "joint"}
EXPECTED_BOOTSTRAP_METRICS = {"frag", "drift"}
EXPECTED_ACPC_BASIN_CORRUPTIONS = {round(i / 100, 2) for i in range(1, 9)}
REQUIRED_ACPC_BASIN_FIELDS = {
    "pixels_std0.08_success",
    "pixels_goal_std0.08_success",
    "corruption_drop",
    "pixels_goal_corruption_drop",
    "encoder_view_pair_l2_norm_by_nn",
    "pred_view_pair_l2_norm_by_transition",
    "basin_contraction_pair_norm",
    "encoder_to_clean_l2_norm_by_nn_median",
    "pred_to_clean_l2_norm_by_transition_median",
    "basin_contraction_to_clean_norm_median",
}
TOL = 1e-9


def fail(msg: str) -> None:
    raise AssertionError(msg)


def check_artifacts() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_ARTIFACTS if not path.exists()]
    if missing:
        fail(f"Missing release artifacts: {', '.join(missing)}")


def check_forbidden_text() -> None:
    hits: list[str] = []
    for path in RELEASE_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                hits.append(f"{path.relative_to(ROOT)} contains forbidden snippet: {snippet!r}")
    main_tex = (ROOT / "paper1" / "main.tex").read_text(encoding="utf-8")
    paper_facing_files = [ROOT / "paper1" / "main.tex"] + sorted((ROOT / "paper1" / "tables").glob("table_*.tex"))
    top_conference_forbidden = [
        "Remediation audit tables",
        "Bounded unseen-stressor check",
        "Bounded unseen-stressor score check",
        "Gaussian sensitivity audits",
        "Finite-difference Gaussian sensitivity audit",
        "Fixed-pool top-1 agreement audit",
        "Retained-summary fixed-pool top-1 audit",
        "Full-sweep sample-level fixed-pool event-rate audit",
        "The audit uses exact autograd JVPs",
        "held-out seed/task audits",
        "joint ATR-plus-guard audits",
        "unseen-stressor score checks",
        "training-free full-sweep audit",
        "retained full-sweep ATR/SMPR artifact",
        "retained-summary overlay",
        "recorded fixed-pool summaries",
        "recorded fixed-pool flip rate",
        "sampled fixed-pool audit anchors",
    ]
    for path in paper_facing_files:
        if not path.exists():
            continue
        paper_text = path.read_text(encoding="utf-8")
        for snippet in top_conference_forbidden:
            if snippet in paper_text:
                hits.append(f"{path.relative_to(ROOT)} contains paper-facing internal-review wording: {snippet!r}")
    main_forbidden = [
        "residual association with reduced drop",
        "selector-baseline audit",
        "fixed high-noise and MAF-only baselines",
        "then select the lowest aggregate rank",
        "Three-seed fixed-rule diagnostic validation",
        "Task & exact best",
        "regret to best",
        "top-2 overlap",
        "The claim is",
        "readout",
        "paper-facing claim",
        "baseline stress",
        "std0.08 stress",
        "stress $\\Delta$",
        "& reading",
        "appendix-unseen-transfer",
        "unseen-score-three-seed-appendix",
        "drop impr.",
        "In experiments, ATR uses the 90th percentile",
        "neither necessary",
        "nor sufficient",
        "Action-relevant discriminability (countercondition)",
        "The compressed diagnostic keeps",
        "Compressed selective-ACPC",
        "tab:training-seed-gaussian-lockbox",
        "best obs",
        "std0.08 gap",
        "point-best",
        "point-optimal",
        "full seed-3072 Gaussian sweep",
        "population standard deviation across the three evaluation seeds",
        "Auxiliary Observation+Goal Gaussian Stress",
        "appendix-unseen-transfer",
        "appendix-obs-goal",
        "Replanning union bound",
        "Selective ACPC pseudo-metric",
        "obs+goal $\\sigma=0.08$ eval",
        "auxiliary observation+goal Gaussian evaluation",
        "Auxiliary observation+goal Gaussian runs",
        "Goal-corrupted Gaussian evaluation",
    ]
    for snippet in main_forbidden:
        if snippet in main_tex:
            hits.append(f"paper1/main.tex contains retired main-text snippet: {snippet!r}")
    for snippet in REQUIRED_MAIN_TEXT_SNIPPETS:
        if snippet not in main_tex:
            hits.append(f"paper1/main.tex missing required scope-boundary snippet: {snippet!r}")
    if hits:
        fail("\n".join(hits))


def check_appendix_internal_heading_gate() -> None:
    main_tex = (ROOT / "paper1" / "main.tex").read_text(encoding="utf-8")
    marker = "\\appendix"
    if marker not in main_tex:
        fail("paper1/main.tex missing appendix marker")
    appendix = main_tex.split(marker, 1)[1]
    forbidden = ["\\paragraph{Reading.}", "\\paragraph{Reading:}"]
    hits = [snippet for snippet in forbidden if snippet in appendix]
    if hits:
        fail("Appendix contains internal Reading heading(s): " + ", ".join(hits))


def approx_equal(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=TOL)


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    dx = [v - mean_x for v in x]
    dy = [v - mean_y for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom <= 1e-12:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def residualize_against_z(values: list[float], z: list[float]) -> list[float]:
    mean_v = statistics.fmean(values)
    mean_z = statistics.fmean(z)
    dz = [v - mean_z for v in z]
    var_z = sum(v * v for v in dz)
    if var_z <= 1e-12:
        return [0.0] * len(values)
    cov = sum((v - mean_v) * zz for v, zz in zip(values, dz))
    slope = cov / var_z
    intercept = mean_v - slope * mean_z
    return [v - (intercept + slope * zz) for v, zz in zip(values, z)]


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> float | None:
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)
    ex = residualize_against_z(rx, rz)
    ey = residualize_against_z(ry, rz)
    if max(ex) - min(ex) <= 1e-12 or max(ey) - min(ey) <= 1e-12:
        return None
    return pearson(ex, ey)


def round2(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)


def check_metric_summary(task: str, std_key: str, metric_name: str, summary: dict) -> None:
    for key in ("n", "mean", "std", "values"):
        if key not in summary:
            fail(f"{task}/{std_key}/{metric_name} missing key {key!r}")

    values = summary["values"]
    if summary["n"] != 3:
        fail(f"{task}/{std_key}/{metric_name} expected n=3, got {summary['n']}")
    if not isinstance(values, list) or len(values) != 3:
        fail(f"{task}/{std_key}/{metric_name} expected 3 seed values, got {values!r}")

    if not all(isinstance(v, (int, float)) for v in values):
        fail(f"{task}/{std_key}/{metric_name} has non-numeric seed values: {values!r}")

    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if not approx_equal(summary["mean"], mean):
        fail(
            f"{task}/{std_key}/{metric_name} mean mismatch: "
            f"stored={summary['mean']} recomputed={mean}"
        )
    if not approx_equal(summary["std"], std):
        fail(
            f"{task}/{std_key}/{metric_name} std mismatch: "
            f"stored={summary['std']} recomputed={std}"
        )
    if not (0.0 <= summary["mean"] <= 100.0):
        fail(f"{task}/{std_key}/{metric_name} mean out of success-rate range: {summary['mean']}")
    if not (0.0 <= summary["std"] <= 100.0):
        fail(f"{task}/{std_key}/{metric_name} std out of success-rate range: {summary['std']}")


def check_canonical_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "canonical_evals_20260517.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    if set(data) != EXPECTED_TASKS:
        fail(f"Canonical tasks mismatch: expected {sorted(EXPECTED_TASKS)}, got {sorted(data)}")

    total_configs = 0
    seen_subdirs: set[str] = set()
    for task, configs in data.items():
        if set(configs) != EXPECTED_CONFIGS:
            fail(
                f"{task} config mismatch: expected {sorted(EXPECTED_CONFIGS)}, "
                f"got {sorted(configs)}"
            )
        total_configs += len(configs)
        for std_key, entry in configs.items():
            for key in ("path", "subdir", "metrics"):
                if key not in entry:
                    fail(f"{task}/{std_key} missing key {key!r}")
            subdir = entry["subdir"]
            if not isinstance(subdir, str) or not subdir:
                fail(f"{task}/{std_key} has invalid subdir: {subdir!r}")
            if subdir in seen_subdirs:
                fail(f"Duplicate canonical subdir: {subdir}")
            seen_subdirs.add(subdir)

            metrics = entry["metrics"]
            missing_metrics = REQUIRED_METRICS - set(metrics)
            if missing_metrics:
                fail(f"{task}/{std_key} missing required metrics: {sorted(missing_metrics)}")
            for metric_name in REQUIRED_METRICS:
                check_metric_summary(task, std_key, metric_name, metrics[metric_name])

    if total_configs != 36:
        fail(f"Expected 36 canonical configs, got {total_configs}")


def check_pldm_canonical_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "canonical_evals_pldm_20260522.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    if set(data) != EXPECTED_TASKS:
        fail(f"PLDM tasks mismatch: expected {sorted(EXPECTED_TASKS)}, got {sorted(data)}")

    total_configs = 0
    for task, configs in data.items():
        if set(configs) != EXPECTED_CONFIGS:
            fail(
                f"PLDM {task} config mismatch: expected {sorted(EXPECTED_CONFIGS)}, "
                f"got {sorted(configs)}"
            )
        total_configs += len(configs)
        for std_key, entry in configs.items():
            for key in ("path", "subdir", "metrics"):
                if key not in entry:
                    fail(f"PLDM {task}/{std_key} missing key {key!r}")
            metrics = entry["metrics"]
            missing_metrics = REQUIRED_METRICS - set(metrics)
            if missing_metrics:
                fail(f"PLDM {task}/{std_key} missing required metrics: {sorted(missing_metrics)}")
            for metric_name in REQUIRED_METRICS:
                check_metric_summary(f"PLDM/{task}", std_key, metric_name, metrics[metric_name])

    if total_configs != 36:
        fail(f"Expected 36 PLDM canonical configs, got {total_configs}")


def check_canonical_diagnostics_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "canonical_diagnostics_20260517.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    predictor = data.get("predictor_metrics_by_task")
    if not isinstance(predictor, dict) or set(predictor) != REQUIRED_DIAG_TASKS:
        fail(
            "canonical diagnostics predictor tasks mismatch: "
            f"expected {sorted(REQUIRED_DIAG_TASKS)}, got {sorted(predictor or {})}"
        )

    for task, configs in predictor.items():
        if set(configs) != EXPECTED_CONFIGS:
            fail(
                f"canonical diagnostics {task} config mismatch: "
                f"expected {sorted(EXPECTED_CONFIGS)}, got {sorted(configs)}"
            )
        for std_key, entry in configs.items():
            for key in (
                "subdir",
                "diagnostic_max_std",
                "predictor_target_to_nn_cos_ratio_at_max_std",
                "predictor_rollout_T8_l2_at_max_std",
            ):
                if key not in entry:
                    fail(f"canonical diagnostics {task}/{std_key} missing key {key!r}")

    rep = data.get("table3_representative_diagnostics", {})
    if set(rep.get("representative_std_by_task", {})) != REQUIRED_DIAG_TASKS:
        fail("canonical diagnostics representative std map is incomplete")
    for task, std in rep.get("representative_std_by_task", {}).items():
        if abs(float(std) - 0.08) > 1e-12:
            fail(
                "canonical diagnostics Table 3 must use the fixed high-noise "
                f"std=0.08 checkpoint for every task; got {task}={std}"
            )
    values = rep.get("values", {})
    if set(values) != REQUIRED_DIAG_TASKS:
        fail("canonical diagnostics representative value map is incomplete")
    metric_order = rep.get("metric_order", [])
    expected_metric_order = [
        "clean_effective_rank",
        "clean_nn_cos_dist_median",
        "transition_resolution_ratio_l2",
        "transition_resolution_ratio_cos",
        "id_probe_r2",
        "action_mean_pred_shift_norm",
    ]
    if metric_order != expected_metric_order:
        fail(
            "canonical diagnostics metric order mismatch: "
            f"expected {expected_metric_order}, got {metric_order}"
        )
    for task, task_values in values.items():
        for which in ("base", "representative"):
            if which not in task_values:
                fail(f"canonical diagnostics {task} missing {which!r} values")
            for metric in expected_metric_order:
                if metric not in task_values[which]:
                    fail(f"canonical diagnostics {task}/{which} missing metric {metric!r}")

    # Regression guard for the 2026-06-25 Table 3 audit: the compact main-text
    # diagnostic rows must stay pinned to the fixed std=0.08 per-checkpoint
    # diagnostics. Earlier drafts mixed per-task representative std values,
    # which read like an implicit selector.
    expected_representative = {
        "TwoRoom": {
            "clean_effective_rank": 37.69,
            "clean_nn_cos_dist_median": 0.0321,
            "transition_resolution_ratio_l2": 0.6621,
            "transition_resolution_ratio_cos": 0.461,
            "id_probe_r2": 0.1419,
            "action_mean_pred_shift_norm": 0.4843,
        },
        "PushT": {
            "clean_effective_rank": 78.08,
            "clean_nn_cos_dist_median": 0.2191,
            "transition_resolution_ratio_l2": 0.2789,
            "transition_resolution_ratio_cos": 0.0759,
            "id_probe_r2": 0.7647,
            "action_mean_pred_shift_norm": 0.1208,
        },
        "Reacher": {
            "clean_effective_rank": 66.2,
            "clean_nn_cos_dist_median": 0.0664,
            "transition_resolution_ratio_l2": 0.3831,
            "transition_resolution_ratio_cos": 0.144,
            "id_probe_r2": 0.1767,
            "action_mean_pred_shift_norm": 0.2619,
        },
        "Cube": {
            "clean_effective_rank": 74.97,
            "clean_nn_cos_dist_median": 0.1587,
            "transition_resolution_ratio_l2": 0.5085,
            "transition_resolution_ratio_cos": 0.2557,
            "id_probe_r2": 0.6342,
            "action_mean_pred_shift_norm": 0.2573,
        },
    }
    for task, expected in expected_representative.items():
        got = values[task]["representative"]
        for metric, want in expected.items():
            if abs(float(got[metric]) - want) > 1e-9:
                fail(
                    f"canonical diagnostics table3 {task}/representative/{metric}: "
                    f"got {got[metric]}, want {want} (fixed-0.08 Table 3 guard)"
                )


def check_pldm_diagnostics_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "canonical_diagnostics_pldm_20260522.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    predictor = data.get("predictor_metrics_by_task")
    if not isinstance(predictor, dict) or set(predictor) != REQUIRED_DIAG_TASKS:
        fail(
            "PLDM diagnostics predictor tasks mismatch: "
            f"expected {sorted(REQUIRED_DIAG_TASKS)}, got {sorted(predictor or {})}"
        )

    for task, configs in predictor.items():
        if set(configs) != EXPECTED_CONFIGS:
            fail(
                f"PLDM diagnostics {task} config mismatch: "
                f"expected {sorted(EXPECTED_CONFIGS)}, got {sorted(configs)}"
            )
        for std_key, entry in configs.items():
            for key in (
                "subdir",
                "diagnostic_max_std",
                "predictor_target_to_nn_cos_ratio_at_max_std",
                "predictor_rollout_T8_l2_at_max_std",
            ):
                if key not in entry:
                    fail(f"PLDM diagnostics {task}/{std_key} missing key {key!r}")
            for key in (
                "predictor_target_to_nn_cos_ratio_at_max_std",
                "predictor_rollout_T8_l2_at_max_std",
            ):
                if not math.isfinite(float(entry[key])):
                    fail(f"PLDM diagnostics {task}/{std_key}/{key} is not finite")


def check_pldm_full_diagnostics_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "canonical_full_diagnostics_pldm_20260523.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    by_task = data.get("diagnostics_by_task")
    if not isinstance(by_task, dict) or set(by_task) != EXPECTED_TASKS:
        fail(
            "PLDM full diagnostics task mismatch: "
            f"expected {sorted(EXPECTED_TASKS)}, got {sorted(by_task or {})}"
        )

    for task, configs in by_task.items():
        if set(configs) != EXPECTED_CONFIGS:
            fail(
                f"PLDM full diagnostics {task} config mismatch: "
                f"expected {sorted(EXPECTED_CONFIGS)}, got {sorted(configs)}"
            )
        for std_key, entry in configs.items():
            for key in ("path", "subdir", "diagnostics_summary"):
                if key not in entry:
                    fail(f"PLDM full diagnostics {task}/{std_key} missing key {key!r}")
            summary = entry["diagnostics_summary"]
            missing = EXPECTED_PLDM_FULL_DIAG_METRICS - set(summary)
            if missing:
                fail(f"PLDM full diagnostics {task}/{std_key} missing metrics: {sorted(missing)}")
            for metric in EXPECTED_PLDM_FULL_DIAG_METRICS:
                value = summary[metric]
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    fail(f"PLDM full diagnostics {task}/{std_key}/{metric} is not finite")

    rep_std = data.get("representative_std_by_task", {})
    if set(rep_std) != EXPECTED_TASKS:
        fail("PLDM full diagnostics representative std map is incomplete")
    reps = data.get("representative_diagnostics", {}).get("values", {})
    if set(reps) != EXPECTED_TASKS:
        fail("PLDM full diagnostics representative values are incomplete")
    for task, entry in reps.items():
        if rep_std[task] != entry.get("representative_std"):
            fail(f"PLDM full diagnostics representative std mismatch for {task}")
        for side in ("base", "representative"):
            values = entry.get(side)
            if not isinstance(values, dict):
                fail(f"PLDM full diagnostics representative {task}/{side} missing")
            missing = EXPECTED_PLDM_FULL_DIAG_METRICS - set(values)
            if missing:
                fail(
                    f"PLDM full diagnostics representative {task}/{side} missing metrics: "
                    f"{sorted(missing)}"
                )


def check_acpc_phase0_diagnostics_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "acpc_phase0_clean_goal_seed9101.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-acpc-phase0-0.1":
        fail(f"unexpected ACPC Phase-0 schema: {meta.get('schema_version')!r}")
    if set(meta.get("methods", [])) != EXPECTED_METHODS:
        fail(f"ACPC Phase-0 methods mismatch: {meta.get('methods')}")
    if set(meta.get("tasks", [])) != EXPECTED_TASKS:
        fail(f"ACPC Phase-0 tasks mismatch: {meta.get('tasks')}")
    if set(meta.get("std_keys", [])) != EXPECTED_CONFIGS:
        fail(f"ACPC Phase-0 std keys mismatch: {meta.get('std_keys')}")
    if meta.get("dry_run") is not False:
        fail("ACPC Phase-0 artifact must be from a real run, not dry-run")
    if meta.get("corrupt_goal") is not False:
        fail("ACPC Phase-0 artifact must use clean-goal observation-noise diagnostics")

    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_METHODS) * len(EXPECTED_TASKS) * len(EXPECTED_CONFIGS):
        fail(f"ACPC Phase-0 row count mismatch: {len(rows) if isinstance(rows, list) else type(rows)}")

    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row.get("method"), row.get("task"), row.get("std_key"))
        if key in seen:
            fail(f"duplicate ACPC Phase-0 row: {key}")
        seen.add(key)
        method, task, std_key = key
        if method not in EXPECTED_METHODS or task not in EXPECTED_TASKS or std_key not in EXPECTED_CONFIGS:
            fail(f"unexpected ACPC Phase-0 row key: {key}")
        if row.get("status") != "ok":
            fail(f"ACPC Phase-0 row {key} is not ok: {row.get('status')}")
        if int(row.get("candidate_count", -1)) != 65:
            fail(f"ACPC Phase-0 row {key} unexpected candidate_count: {row.get('candidate_count')}")
        if int(row.get("rollout_horizon_actual", -1)) != 8:
            fail(f"ACPC Phase-0 row {key} unexpected rollout horizon: {row.get('rollout_horizon_actual')}")
        if int(row.get("n_sequences", -1)) != 100:
            fail(f"ACPC Phase-0 row {key} unexpected n_sequences: {row.get('n_sequences')}")
        if abs(float(row.get("noise_std", float("nan"))) - 0.08) > TOL:
            fail(f"ACPC Phase-0 row {key} unexpected noise_std: {row.get('noise_std')}")
        if row.get("corrupt_goal") is not False:
            fail(f"ACPC Phase-0 row {key} must keep the goal clean")
        missing = EXPECTED_ACPC_PHASE0_METRICS - set(row)
        if missing:
            fail(f"ACPC Phase-0 row {key} missing metrics: {sorted(missing)}")
        for metric in EXPECTED_ACPC_PHASE0_METRICS:
            value = row[metric]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                fail(f"ACPC Phase-0 row {key}/{metric} is not finite")

    expected_seen = {
        (method, task, std_key)
        for method in EXPECTED_METHODS
        for task in EXPECTED_TASKS
        for std_key in EXPECTED_CONFIGS
    }
    if seen != expected_seen:
        fail("ACPC Phase-0 row coverage mismatch")


def check_blur_baselines_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "canonical_blur_baselines_20260523.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    baselines = data.get("baselines")
    if not isinstance(baselines, dict) or set(baselines) != EXPECTED_METHODS:
        fail(
            "blur baseline methods mismatch: "
            f"expected {sorted(EXPECTED_METHODS)}, got {sorted(baselines or {})}"
        )

    for method, by_task in baselines.items():
        if set(by_task) != EXPECTED_TASKS:
            fail(f"blur baseline {method} tasks mismatch: {sorted(by_task)}")
        for task, entry in by_task.items():
            for key in ("path", "subdir", "clean", "blur", "worst_pixels_goal_blur"):
                if key not in entry:
                    fail(f"blur baseline {method}/{task} missing key {key!r}")
            check_metric_summary(f"blur/{method}/{task}", "clean", "clean", entry["clean"])
            blur = entry["blur"]
            if set(blur) != EXPECTED_BLUR_CONDITIONS:
                fail(
                    f"blur baseline {method}/{task} condition mismatch: "
                    f"expected {sorted(EXPECTED_BLUR_CONDITIONS)}, got {sorted(blur)}"
                )
            for condition, summary in blur.items():
                check_metric_summary(f"blur/{method}/{task}", condition, condition, summary)
            worst = entry["worst_pixels_goal_blur"]
            condition = worst.get("condition")
            if condition not in blur or not condition.startswith("pixels_goal_blur_ks"):
                fail(f"blur baseline {method}/{task} has invalid worst condition {condition!r}")
            expected_worst = min(
                (blur[f"pixels_goal_blur_ks{k}"]["mean"], f"pixels_goal_blur_ks{k}")
                for k in (3, 7, 11, 15)
            )[1]
            if condition != expected_worst:
                fail(
                    f"blur baseline {method}/{task} worst mismatch: "
                    f"got {condition}, want {expected_worst}"
                )
            drop = entry["clean"]["mean"] - blur[condition]["mean"]
            if not approx_equal(drop, entry["clean_to_worst_pixels_goal_blur_drop"]):
                fail(f"blur baseline {method}/{task} drop mismatch: {drop}")


def check_acpc_basin_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "acpc_basin_diagnostics.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-acpc-basin-0.1":
        fail(f"unexpected ACPC basin schema: {meta.get('schema_version')!r}")
    if meta.get("method") != "LeWM":
        fail(f"ACPC basin method should be LeWM, got {meta.get('method')!r}")
    if meta.get("corrupt_goal") is not False:
        fail("ACPC basin metadata should mark corrupt_goal=false")

    corruptions = meta.get("corruptions")
    if not isinstance(corruptions, list) or len(corruptions) != 8:
        fail("ACPC basin metadata must list exactly 8 Gaussian-noise corruptions")
    got_magnitudes = set()
    for spec in corruptions:
        if spec.get("type") != "gaussian_noise":
            fail(f"ACPC basin contains non-noise corruption: {spec}")
        got_magnitudes.add(round(float(spec.get("magnitude")), 2))
    if got_magnitudes != EXPECTED_ACPC_BASIN_CORRUPTIONS:
        fail(
            "ACPC basin corruption grid mismatch: "
            f"got {sorted(got_magnitudes)}, want {sorted(EXPECTED_ACPC_BASIN_CORRUPTIONS)}"
        )

    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_TASKS) * len(EXPECTED_CONFIGS):
        fail("ACPC basin rows must cover 4 tasks x 9 configs")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("status") != "ok":
            fail(f"ACPC basin row is not ok: {row.get('task')}/{row.get('std_key')}")
        task = row.get("task")
        std_key = row.get("std_key")
        if task not in EXPECTED_TASKS or std_key not in EXPECTED_CONFIGS:
            fail(f"unexpected ACPC basin row key: {task}/{std_key}")
        key = (task, std_key)
        if key in seen:
            fail(f"duplicate ACPC basin row: {task}/{std_key}")
        seen.add(key)
        if row.get("method") != "LeWM":
            fail(f"ACPC basin row method should be LeWM: {task}/{std_key}")
        if row.get("corrupt_goal") is not False:
            fail(f"ACPC basin {task}/{std_key} should keep the goal clean by default")
        model_file = str(row.get("model_file", ""))
        if not model_file.endswith("epoch_10_object.ckpt"):
            fail(f"ACPC basin row does not use epoch_10 object ckpt: {model_file}")
        variants = row.get("variant_rows")
        if not isinstance(variants, list) or len(variants) != 8:
            fail(f"ACPC basin {task}/{std_key} must contain 8 variant rows")
        variant_magnitudes = set()
        for variant in variants:
            if variant.get("corruption_type") != "gaussian_noise":
                fail(f"ACPC basin {task}/{std_key} has non-noise variant: {variant}")
            variant_magnitudes.add(round(float(variant.get("magnitude")), 2))
        if variant_magnitudes != EXPECTED_ACPC_BASIN_CORRUPTIONS:
            fail(f"ACPC basin {task}/{std_key} variant grid mismatch")
        missing = REQUIRED_ACPC_BASIN_FIELDS - set(row)
        if missing:
            fail(f"ACPC basin {task}/{std_key} missing fields: {sorted(missing)}")
        for field in REQUIRED_ACPC_BASIN_FIELDS:
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                fail(f"ACPC basin {task}/{std_key}/{field} is not finite")
    if seen != {(task, std) for task in EXPECTED_TASKS for std in EXPECTED_CONFIGS}:
        fail("ACPC basin task/config coverage mismatch")


def check_acpc_basin_artifact_pointer() -> None:
    main_tex = ROOT / "paper1" / "main.tex"
    if not main_tex.exists():
        return
    tex = main_tex.read_text(encoding="utf-8")
    required = [
        "Full LeWM ACPC-basin grid",
        r"assets/paper1\_data/acpc\_basin\_diagnostics.json",
        r"\Cref{tab:acpc-basin}",
    ]
    missing = [snippet for snippet in required if snippet not in tex]
    if missing:
        fail("main.tex is missing ACPC-basin artifact pointer snippets: " + ", ".join(missing))


def check_pldm_acpc_basin_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "acpc_basin_diagnostics_pldm.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-acpc-basin-0.1":
        fail(f"unexpected PLDM ACPC basin schema: {meta.get('schema_version')!r}")
    if meta.get("method") != "PLDM" or meta.get("methods") != ["PLDM"]:
        fail(f"PLDM ACPC basin method mismatch: {meta.get('method')!r}/{meta.get('methods')!r}")
    if meta.get("base_vs_best") is not False:
        fail("PLDM ACPC basin must be the full sweep, not base-vs-best")
    if meta.get("robust_metric") != "pixels_std0.08":
        fail(f"PLDM ACPC basin robust metric mismatch: {meta.get('robust_metric')!r}")
    if meta.get("corrupt_goal") is not False:
        fail("PLDM ACPC basin metadata should mark corrupt_goal=false")
    if meta.get("dry_run") is not False:
        fail("PLDM ACPC basin artifact must be from a real run, not dry-run")

    corruptions = meta.get("corruptions")
    if not isinstance(corruptions, list) or len(corruptions) != 8:
        fail("PLDM ACPC basin metadata must list exactly 8 Gaussian-noise corruptions")
    got_magnitudes = set()
    for spec in corruptions:
        if spec.get("type") != "gaussian_noise":
            fail(f"PLDM ACPC basin contains non-noise corruption: {spec}")
        got_magnitudes.add(round(float(spec.get("magnitude")), 2))
    if got_magnitudes != EXPECTED_ACPC_BASIN_CORRUPTIONS:
        fail("PLDM ACPC basin corruption grid mismatch")

    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_TASKS) * len(EXPECTED_CONFIGS):
        fail(f"PLDM ACPC basin row count mismatch: {len(rows) if isinstance(rows, list) else type(rows)}")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        task = row.get("task")
        std_key = row.get("std_key")
        key = (task, std_key)
        if task not in EXPECTED_TASKS or std_key not in EXPECTED_CONFIGS:
            fail(f"unexpected PLDM ACPC basin row key: {key}")
        if key in seen:
            fail(f"duplicate PLDM ACPC basin row: {key}")
        seen.add(key)
        if row.get("status") != "ok":
            fail(f"PLDM ACPC basin row {key} is not ok: {row.get('status')}")
        if row.get("method") != "PLDM":
            fail(f"PLDM ACPC basin row method mismatch: {key}")
        if row.get("corrupt_goal") is not False:
            fail(f"PLDM ACPC basin row should keep the goal clean: {key}")
        model_file = str(row.get("model_file", ""))
        if not model_file.endswith("epoch_10_object.ckpt"):
            fail(f"PLDM ACPC basin row does not use epoch_10 object ckpt: {model_file}")
        variants = row.get("variant_rows")
        if not isinstance(variants, list) or len(variants) != 8:
            fail(f"PLDM ACPC basin {key} must contain 8 variant rows")
        variant_magnitudes = set()
        for variant in variants:
            if variant.get("corruption_type") != "gaussian_noise":
                fail(f"PLDM ACPC basin {key} has non-noise variant: {variant}")
            variant_magnitudes.add(round(float(variant.get("magnitude")), 2))
        if variant_magnitudes != EXPECTED_ACPC_BASIN_CORRUPTIONS:
            fail(f"PLDM ACPC basin {key} variant grid mismatch")
        missing = REQUIRED_ACPC_BASIN_FIELDS - set(row)
        if missing:
            fail(f"PLDM ACPC basin {key} missing fields: {sorted(missing)}")
        for field in REQUIRED_ACPC_BASIN_FIELDS:
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                fail(f"PLDM ACPC basin {key}/{field} is not finite")
    if seen != {(task, std) for task in EXPECTED_TASKS for std in EXPECTED_CONFIGS}:
        fail("PLDM ACPC basin task/config coverage mismatch")


def check_external_baselines_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "canonical_external_baselines_20260520.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    entry = data.get("baselines", {}).get("PushT", {}).get("PLDM_clean_trained")
    if not isinstance(entry, dict):
        fail("external baseline JSON missing PushT/PLDM_clean_trained")
    if entry.get("subdir") != "pusht_pldm_baseline":
        fail(f"unexpected PLDM subdir: {entry.get('subdir')!r}")
    if (
        entry.get("citation")
        != "sobal2022jointembeddingpredictivearchitectures;sobal2025stresstesting;maes2026stableworldmodel"
    ):
        fail(f"unexpected PLDM citation key: {entry.get('citation')!r}")

    training = entry.get("training", {})
    if training.get("image_noise_std_max") != 0.0 or training.get("image_noise_noise_prob") != 0.0:
        fail("PLDM external baseline is expected to be clean-trained")

    required_eval = {"clean", "pixels_std0.08", "pixels_goal_std0.05", "pixels_goal_std0.08"}
    evaluation = entry.get("evaluation", {})
    missing = required_eval - set(evaluation)
    if missing:
        fail(f"PLDM external baseline missing eval conditions: {sorted(missing)}")
    for metric_name, summary in evaluation.items():
        check_metric_summary("PushT/PLDM_clean_trained", "external", metric_name, summary)

    clean = evaluation["clean"]["mean"]
    px08 = evaluation["pixels_std0.08"]["mean"]
    if round(clean - px08, 2) != 57.00:
        fail(f"unexpected PLDM clean-to-pixels0.08 drop: {clean - px08}")


def check_pldm_correlations_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "cross_method_corr_pldm_20260522.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    if set(data) != EXPECTED_TASKS:
        fail(f"PLDM correlation tasks mismatch: expected {sorted(EXPECTED_TASKS)}, got {sorted(data)}")

    expected_push = {
        ("within_pldm", "partial_metric_drop_on_std"): -0.05,
        ("joint", "partial_metric_drop_on_std_method"): 0.22,
    }
    for task, block in data.items():
        rows = block.get("rows", {})
        if len(rows.get("pldm", [])) != 9 or len(rows.get("lewm", [])) != 9:
            fail(f"PLDM correlation {task} expected 9 LeWM rows and 9 PLDM rows")
        within = block.get("within_pldm", {}).get("frag", {})
        joint = block.get("joint", {}).get("frag", {})
        if within.get("n") != 9:
            fail(f"PLDM correlation {task} within-PLDM n mismatch: {within.get('n')}")
        if joint.get("n") != 18:
            fail(f"PLDM correlation {task} joint n mismatch: {joint.get('n')}")
        for key in (
            "partial_metric_clean_on_std",
            "partial_metric_px08_on_std",
            "partial_metric_drop_on_std",
        ):
            if key not in within or not math.isfinite(float(within[key])):
                fail(f"PLDM correlation {task}/within_pldm/frag missing finite {key}")
        if (
            "partial_metric_drop_on_std_method" not in joint
            or not math.isfinite(float(joint["partial_metric_drop_on_std_method"]))
        ):
            fail(f"PLDM correlation {task}/joint/frag missing finite partial drop")

    for (section, key), want in expected_push.items():
        got = round2(data["PushT"][section]["frag"][key])
        if got != want:
            fail(f"PLDM PushT correlation mismatch for {section}/{key}: got {got}, want {want}")


def _check_bootstrap_cell(
    data: dict,
    task: str,
    scope: str,
    metric: str,
    key: str,
    point: float,
    ci: tuple[float, float],
) -> None:
    cell = data["by_task"][task][scope][metric][key]
    got_point = round2(cell.get("point"))
    if got_point != point:
        fail(
            f"bootstrap point mismatch for {task}/{scope}/{metric}/{key}: "
            f"got {got_point}, want {point}"
        )
    got_ci = cell.get("ci")
    if not isinstance(got_ci, list) or len(got_ci) != 2:
        fail(f"bootstrap CI missing for {task}/{scope}/{metric}/{key}")
    if round2(got_ci[0]) != ci[0] or round2(got_ci[1]) != ci[1]:
        fail(
            f"bootstrap CI mismatch for {task}/{scope}/{metric}/{key}: "
            f"got {[round2(got_ci[0]), round2(got_ci[1])]}, want {list(ci)}"
        )


def check_partial_corr_bootstrap_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "partial_corr_bootstrap_20260523.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    meta = data.get("metadata", {})
    if meta.get("n_bootstrap") != 1000 or meta.get("seed") != 42:
        fail(f"unexpected bootstrap metadata: {meta}")
    if meta.get("ci_low_pct") != 2.5 or meta.get("ci_high_pct") != 97.5:
        fail(f"unexpected bootstrap CI percentiles: {meta}")

    by_task = data.get("by_task")
    if not isinstance(by_task, dict) or set(by_task) != EXPECTED_TASKS:
        fail(
            "bootstrap tasks mismatch: "
            f"expected {sorted(EXPECTED_TASKS)}, got {sorted(by_task or {})}"
        )
    for task, block in by_task.items():
        if set(block) != EXPECTED_BOOTSTRAP_SCOPES:
            fail(f"bootstrap {task} scopes mismatch: {sorted(block)}")
        for scope, scope_block in block.items():
            expected_n = 18 if scope == "joint" else 9
            if scope_block.get("n") != expected_n:
                fail(f"bootstrap {task}/{scope} n mismatch: {scope_block.get('n')}")
            if not EXPECTED_BOOTSTRAP_METRICS.issubset(scope_block):
                fail(f"bootstrap {task}/{scope} missing metrics")
            for metric in EXPECTED_BOOTSTRAP_METRICS:
                cells = scope_block[metric]
                if not isinstance(cells, dict):
                    fail(f"bootstrap {task}/{scope}/{metric} is not a dict")
                for cell_name, cell in cells.items():
                    if "point" not in cell or "n_valid" not in cell or "ci" not in cell:
                        fail(f"bootstrap {task}/{scope}/{metric}/{cell_name} malformed")
                    if cell["point"] is not None and not math.isfinite(float(cell["point"])):
                        fail(f"bootstrap {task}/{scope}/{metric}/{cell_name} point not finite")
                    if not isinstance(cell["n_valid"], int) or cell["n_valid"] < 0:
                        fail(f"bootstrap {task}/{scope}/{metric}/{cell_name} invalid n_valid")

    # Values quoted in main.tex contributions / Table 7 / Appendix F. These are rounded
    # checks, not a substitute for rerunning the bootstrap.
    _check_bootstrap_cell(
        data, "PushT", "within_lewm", "frag", "partial_metric_clean_on_std",
        -0.59, (-0.97, -0.10),
    )
    _check_bootstrap_cell(
        data, "PushT", "within_lewm", "frag", "partial_metric_px08_on_std",
        -0.53, (-0.84, 0.00),
    )
    _check_bootstrap_cell(
        data, "PushT", "within_lewm", "frag", "partial_metric_drop_on_std",
        0.19, (-0.00, 0.70),
    )
    _check_bootstrap_cell(
        data, "PushT", "within_pldm", "frag", "partial_metric_drop_on_std",
        -0.05, (-0.92, 0.61),
    )
    _check_bootstrap_cell(
        data, "PushT", "joint", "frag", "partial_metric_drop_on_std_method",
        0.22, (-0.59, 0.61),
    )
    _check_bootstrap_cell(
        data, "Reacher", "within_lewm", "drift", "partial_metric_drop_on_std",
        0.37, (-0.35, 0.99),
    )


def check_three_seed_gaussian_sweep_summary_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "three_seed_gaussian_sweep_summary_20260706.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-three-seed-gaussian-sweep-summary-20260706-v1":
        fail(f"three-seed Gaussian sweep schema changed: {meta.get('schema_version')!r}")
    if meta.get("tasks") != ["TwoRoom", "PushT", "Reacher", "Cube"]:
        fail(f"three-seed Gaussian sweep task order changed: {meta.get('tasks')}")
    if meta.get("training_seeds") != [3072, 3073, 3074]:
        fail(f"three-seed Gaussian sweep seeds changed: {meta.get('training_seeds')}")
    if set(meta.get("sweep_stdmax", [])) != EXPECTED_CONFIGS:
        fail(f"three-seed Gaussian sweep std grid changed: {meta.get('sweep_stdmax')}")
    if set(meta.get("metric_keys", {})) != THREE_SEED_SWEEP_METRICS:
        fail(f"three-seed Gaussian sweep metric keys changed: {meta.get('metric_keys')}")

    summary_rows = data.get("summary_rows", [])
    per_seed_rows = data.get("per_seed_rows", [])
    if len(summary_rows) != len(EXPECTED_TASKS) * len(EXPECTED_CONFIGS):
        fail(f"three-seed Gaussian sweep expected 36 summary rows, got {len(summary_rows)}")
    if len(per_seed_rows) != len(EXPECTED_TASKS) * len(EXPECTED_CONFIGS) * 3:
        fail(f"three-seed Gaussian sweep expected 108 per-seed rows, got {len(per_seed_rows)}")

    per_seed = {}
    for row in per_seed_rows:
        key = (row.get("task"), str(row.get("stdmax")), int(row.get("training_seed")))
        if key in per_seed:
            fail(f"duplicate three-seed Gaussian per-seed row: {key}")
        if key[0] not in EXPECTED_TASKS or key[1] not in EXPECTED_CONFIGS or key[2] not in (3072, 3073, 3074):
            fail(f"unexpected three-seed Gaussian per-seed key: {key}")
        metrics = row.get("metrics", {})
        if set(metrics) != THREE_SEED_SWEEP_METRICS:
            fail(f"three-seed Gaussian per-seed metrics changed for {key}: {sorted(metrics)}")
        for metric, cell in metrics.items():
            values = cell.get("eval_seed_values", [])
            if not isinstance(values, list) or len(values) != 3:
                fail(f"three-seed Gaussian {key}/{metric} must contain three eval-seed values")
            if not all(isinstance(v, (int, float)) for v in values):
                fail(f"three-seed Gaussian {key}/{metric} has non-numeric eval-seed values")
            mean = statistics.fmean(values)
            if not math.isclose(float(cell.get("mean_over_eval_seeds")), mean, rel_tol=0.0, abs_tol=1e-6):
                fail(f"three-seed Gaussian {key}/{metric} eval-seed mean mismatch")
        per_seed[key] = row

    summary = {}
    for row in summary_rows:
        key = (row.get("task"), str(row.get("stdmax")))
        if key in summary:
            fail(f"duplicate three-seed Gaussian summary row: {key}")
        if key[0] not in EXPECTED_TASKS or key[1] not in EXPECTED_CONFIGS:
            fail(f"unexpected three-seed Gaussian summary key: {key}")
        if row.get("training_seeds") != [3072, 3073, 3074] or row.get("n_training_seeds") != 3:
            fail(f"three-seed Gaussian summary row must use seeds 3072/3073/3074: {key}")
        metrics = row.get("metrics", {})
        if set(metrics) != THREE_SEED_SWEEP_METRICS:
            fail(f"three-seed Gaussian summary metrics changed for {key}: {sorted(metrics)}")
        for metric, cell in metrics.items():
            values = [
                float(per_seed[(key[0], key[1], seed)]["metrics"][metric]["mean_over_eval_seeds"])
                for seed in (3072, 3073, 3074)
            ]
            if [round(float(v), 6) for v in cell.get("per_training_seed_means", [])] != [round(v, 6) for v in values]:
                fail(f"three-seed Gaussian {key}/{metric} per-training-seed means mismatch")
            if not math.isclose(float(cell.get("mean")), statistics.fmean(values), rel_tol=0.0, abs_tol=1e-6):
                fail(f"three-seed Gaussian {key}/{metric} mean mismatch")
            if not math.isclose(float(cell.get("pstdev")), statistics.pstdev(values), rel_tol=0.0, abs_tol=1e-6):
                fail(f"three-seed Gaussian {key}/{metric} pstdev mismatch")
        summary[key] = row

    expected_obs08 = {
        "TwoRoom": 97.11,
        "PushT": 85.78,
        "Reacher": 81.56,
        "Cube": 62.56,
    }
    for task, want in expected_obs08.items():
        got = round2(float(summary[(task, "0.08")]["metrics"]["obs_sigma_0.08"]["mean"]))
        if got != want:
            fail(f"three-seed Gaussian std=0.08 obs endpoint changed for {task}: {got} != {want}")


def check_training_seed_gaussian_lockbox_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "training_seed_gaussian_lockbox.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("task_summary_rows", [])
    by_task = {row.get("task"): row for row in rows}
    expected = {
        "TwoRoom": (97.10888888888888, 28.333333333333332),
        "PushT": (85.77666666666666, 78.55555555555556),
        "Reacher": (81.55333333333333, 63.33555555555555),
        "Cube": (62.55666666666667, 19.44666666666667),
    }
    if set(by_task) != set(expected):
        fail(f"training-seed lockbox tasks mismatch: got {sorted(by_task)}")
    for task, (want_std08, want_gain) in expected.items():
        row = by_task[task]
        if row.get("training_seeds") != [3072, 3073, 3074]:
            fail(f"{task} training-seed lockbox must use seeds 3072/3073/3074")
        got_std08 = float(row["std_0p08_obs_0p08_mean"])
        got_gain = float(row["std_0p08_gain_over_baseline_mean"])
        if not approx_equal(got_std08, want_std08) or not approx_equal(got_gain, want_gain):
            fail(
                f"{task} training-seed lockbox mismatch: "
                f"got std08={got_std08}, gain={got_gain}; "
                f"want std08={want_std08}, gain={want_gain}"
            )


def check_prospective_validation_summary_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "prospective_validation_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    required_sources = {
        "assets/paper1_data/unseen_origin_vs_std008_strongest_s3072.json",
        "assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json",
        "assets/paper1_data/unseen_origin_vs_std008_strongest_s3074.json",
        "assets/paper1_data/unseen_phase0_acpc_subset.json",
        "assets/paper1_data/unseen_phase0_acpc_fullstress.json",
    }
    sources = set(data.get("metadata", {}).get("source_artifacts", []))
    if not required_sources.issubset(sources):
        fail("prospective validation summary must cite all three-seed unseen score artifacts")
    for source in sorted(s for s in required_sources if "origin_vs" in s):
        source_data = json.loads((ROOT / source).read_text(encoding="utf-8"))
        status = source_data.get("metadata", {}).get("status", "")
        if "audited score artifact" not in status:
            fail(f"{source} must be marked as an audited unseen score artifact")

    score_summary = data.get("three_seed_unseen_score_summary", {})
    selected_policy = {
        "TwoRoom": "gaussian_blur",
        "PushT": "resize",
        "Reacher": "gaussian_blur",
        "Cube": "resize",
    }
    if score_summary.get("selected_stress_policy") != selected_policy:
        fail("three-seed unseen score summary selected-stress policy changed")

    coverage = score_summary.get("coverage", {})
    expected_coverage = {
        f"{task}:{family}": [3072, 3073, 3074]
        for task in ("TwoRoom", "PushT", "Reacher", "Cube")
        for family in ("gaussian_blur", "resize")
    }
    if coverage != expected_coverage:
        fail(f"three-seed unseen score coverage mismatch: {coverage}")

    selected = {
        (row.get("task"), row.get("family")): row
        for row in score_summary.get("selected_stress_rows", [])
    }
    expected_selected = {
        ("TwoRoom", "gaussian_blur"): (47.67, 90.78, 43.11, 40.89),
        ("PushT", "resize"): (63.44, 66.33, 2.89, -3.78),
        ("Reacher", "gaussian_blur"): (22.00, 71.22, 49.22, 30.22),
        ("Cube", "resize"): (57.00, 56.11, -0.89, 2.78),
    }
    if set(selected) != set(expected_selected):
        fail(f"three-seed unseen selected rows mismatch: {sorted(selected)}")
    for row_key, expected_values in expected_selected.items():
        row = selected[row_key]
        if row.get("training_seeds") != [3072, 3073, 3074] or row.get("n_training_seeds") != 3:
            fail(f"{row_key} unseen score row must use training seeds 3072/3073/3074")
        got_values = (
            round2(float(row["baseline_stress_success_mean"])),
            round2(float(row["std008_stress_success_mean"])),
            round2(float(row["stress_success_delta_mean"])),
            round2(float(row["drop_improvement_mean"])),
        )
        if got_values != expected_values:
            fail(f"{row_key} three-seed unseen score mismatch: got {got_values}, want {expected_values}")

    heldout = data.get("heldout_unseen_validation", {})
    if heldout.get("n_rows") != 12:
        fail("prospective validation summary must contain the 12-row three-seed unseen diagnostic slice")
    rows = {row.get("metric"): row for row in heldout.get("metric_rows", [])}
    composite = rows.get("Composite signed-rank rule")
    if composite is None:
        fail("prospective validation summary missing composite signed-rank row")
    checks = {
        "spearman_vs_stress_success_delta": 0.94,
        "pearson_vs_stress_success_delta": 0.96,
        "spearman_vs_drop_improvement": 0.83,
        "pearson_vs_drop_improvement": 0.86,
    }
    for key, want in checks.items():
        got = round2(float(composite[key]))
        if got != want:
            fail(f"prospective validation composite {key} mismatch: got {got}, want {want}")
    topk = heldout.get("topk_summary", {})
    if topk.get("stress_success_delta_topk_hit_count") != 4 or topk.get("drop_improvement_topk_hit_count") != 2:
        fail("prospective validation top-4 agreement must remain 4/4 for stress delta and 2/4 for drop improvement on the three-seed unseen slice")
    fullstress = data.get("fullstress_unseen_validation", {})
    if fullstress.get("n_rows") != 24:
        fail("prospective validation summary must contain the 24-row full blur/resize unseen diagnostic slice")
    full_rows = {row.get("metric"): row for row in fullstress.get("metric_rows", [])}
    full_composite = full_rows.get("Composite signed-rank rule")
    if full_composite is None:
        fail("prospective validation summary missing fullstress composite signed-rank row")
    full_checks = {
        "spearman_vs_stress_success_delta": 0.94,
        "pearson_vs_stress_success_delta": 0.94,
        "spearman_vs_drop_improvement": 0.82,
        "pearson_vs_drop_improvement": 0.84,
    }
    for key, want in full_checks.items():
        got = round2(float(full_composite[key]))
        if got != want:
            fail(f"fullstress validation composite {key} mismatch: got {got}, want {want}")
    full_topk = fullstress.get("topk_summary", {})
    if full_topk.get("stress_success_delta_topk_hit_count") != 4 or full_topk.get("drop_improvement_topk_hit_count") != 2:
        fail("fullstress validation top-4 agreement must remain 4/4 for stress delta and 2/4 for drop improvement")
    diag = data.get("three_seed_full_grid_diagnostic_validation", {})
    if diag.get("n_task_seed_blocks") != 12 or diag.get("within_5pp_hits") != 10:
        fail("prospective validation summary must include completed three-seed full-grid diagnostic validation")
    split_rows = {row.get("split"): row for row in data.get("three_seed_diagnostic_split_summaries", [])}
    heldout = split_rows.get("heldout_training_seeds_3073_3074")
    if heldout is None:
        fail("prospective validation summary missing held-out training-seed diagnostic split")
    if (heldout.get("n_task_seed_blocks"), heldout.get("n_checkpoint_candidates"), heldout.get("within_5pp_hits")) != (8, 64, 7):
        fail(f"held-out training-seed diagnostic split changed: {heldout}")
    if round2(float(heldout.get("mean_selected_regret_to_best_pp"))) != 2.21:
        fail("held-out diagnostic split mean regret changed")
    semantic_rows = data.get("semantic_margin_passrate", [])
    if len(semantic_rows) != 8:
        fail("prospective validation summary must include completed task-state proxy margin pass-rate rows")
    semantic_cov = data.get("semantic_margin_coverage", {})
    expected_semantic_cov = {
        f"{task}:{std}": [3072, 3073, 3074]
        for task in ("TwoRoom", "PushT", "Reacher", "Cube")
        for std in ("0.0", "0.08")
    }
    if semantic_cov != expected_semantic_cov:
        fail(f"prospective validation task-state proxy margin coverage mismatch: {semantic_cov}")





def check_prospective_atr_smpr_validation() -> None:
    smpr_path = ROOT / "assets" / "paper1_data" / "semantic_task_grounded_margin_lewm_full_sweep_20260708.json"
    smpr = json.loads(smpr_path.read_text(encoding="utf-8"))
    rows = [row for row in smpr.get("rows", []) if row.get("status") == "ok"]
    if len(rows) != 108:
        fail(f"full-sweep SMPR must contain 108 ok rows, got {len(rows)}")
    coverage = {(row["task"], int(row["training_seed"]), str(row["std_key"])) for row in rows}
    expected = {(task, seed, std) for task in EXPECTED_TASKS for seed in (3072, 3073, 3074) for std in EXPECTED_CONFIGS}
    if coverage != expected:
        fail("full-sweep SMPR coverage mismatch")
    if len(smpr.get("summary_rows", [])) != 36:
        fail("full-sweep SMPR summary must cover 4 tasks x 9 std keys")

    main_text = (ROOT / "paper1" / "main.tex").read_text(encoding="utf-8")
    forbidden_main = (
        "per-task ATR+SMPR precision",
        "mean interval IoU",
        "F1 0.96",
        "separate ATR/SMPR threshold audit",
    )
    for snippet in forbidden_main:
        if snippet in main_text:
            fail(f"paper-facing prospective validation must not contain old threshold-classifier wording: {snippet}")

    out_dir = ROOT / "paper1" / "results" / "diagnostic_region"
    region = list(csv.DictReader((out_dir / "diagnostic_region_summary.csv").open(encoding="utf-8")))
    direction = list(csv.DictReader((out_dir / "direction_consistency_summary.csv").open(encoding="utf-8")))
    separation = list(csv.DictReader((out_dir / "robust_fragile_separation.csv").open(encoding="utf-8")))

    def row_for(table: list[dict[str, str]], **criteria: str) -> dict[str, str]:
        row = next((r for r in table if all(r.get(k) == v for k, v in criteria.items())), None)
        if row is None:
            fail(f"diagnostic-region table missing row: {criteria}")
        return row

    expected_counts = {
        ("heldout", "fragile"): 8,
        ("heldout", "transition"): 13,
        ("heldout", "robust"): 51,
        ("all", "fragile"): 13,
        ("all", "transition"): 19,
        ("all", "robust"): 76,
    }
    for (split, regime), want in expected_counts.items():
        got = int(row_for(region, split=split, regime=regime)["n"])
        if got != want:
            fail(f"diagnostic-region {split}/{regime} count mismatch: got {got}, want {want}")

    heldout_fragile = row_for(region, split="heldout", regime="fragile")
    heldout_robust = row_for(region, split="heldout", regime="robust")
    checks = {
        "heldout fragile median normalized ATR": (float(heldout_fragile["atr_rel_q50"]), 1.0),
        "heldout fragile median SMPR": (float(heldout_fragile["smpr_q50"]), 0.4349999874830246),
        "heldout robust median normalized ATR": (float(heldout_robust["atr_rel_q50"]), 0.08666369301340819),
        "heldout robust median SMPR": (float(heldout_robust["smpr_q50"]), 0.9999999403953552),
    }
    for label, (got, want) in checks.items():
        if abs(got - want) > 1e-9:
            fail(f"{label} mismatch: got {got}, want {want}")

    heldout_direction = row_for(direction, split="heldout")
    all_direction = row_for(direction, split="all")
    if (
        int(heldout_direction["eligible_blocks"]),
        int(heldout_direction["atr_direction_ok"]),
        int(heldout_direction["smpr_direction_ok"]),
        int(heldout_direction["joint_direction_ok"]),
    ) != (8, 8, 8, 8):
        fail(f"held-out diagnostic direction consistency mismatch: {heldout_direction}")
    if (
        int(all_direction["eligible_blocks"]),
        int(all_direction["atr_direction_ok"]),
        int(all_direction["smpr_direction_ok"]),
        int(all_direction["joint_direction_ok"]),
    ) != (12, 12, 12, 12):
        fail(f"all-seed diagnostic direction consistency mismatch: {all_direction}")

    heldout_sep = row_for(separation, split="heldout")
    if int(heldout_sep["robust_atr_q75_below_fragile_q25"]) != 1:
        fail("held-out robust ATR IQR must stay below fragile ATR IQR")
    if int(heldout_sep["robust_smpr_q25_above_fragile_q75"]) != 1:
        fail("held-out robust SMPR IQR must stay above fragile SMPR IQR")
    if float(heldout_sep["atr_rel_median_gap"]) <= 0.9 or float(heldout_sep["smpr_median_gap"]) <= 0.5:
        fail(f"held-out recovered-vs-fragile separation is weaker than expected: {heldout_sep}")

def check_selector_baseline_audit_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "selector_baseline_audit_20260704.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-selector-baseline-audit-20260704-v1":
        fail(f"selector-baseline audit schema changed: {meta.get('schema_version')!r}")
    if meta.get("score") != "pixels_std0.08_success":
        fail("selector-baseline audit must target pixels_std0.08_success")
    rows = data.get("selection_rows", [])
    if len(rows) != 96:
        fail(f"selector-baseline audit expected 96 selection rows, got {len(rows)}")
    selectors = {
        "aggregate_rank_acpc_pcc_cra_maf",
        "fixed_std_0.08",
        "best_acpc_only",
        "best_pcc_only",
        "best_cra_only",
        "best_maf_only",
        "random_nonzero_std",
        "oracle_best",
    }
    keys = {(row.get("task"), int(row.get("training_seed")), row.get("selector")) for row in rows}
    expected_keys = {
        (task, seed, selector)
        for task in EXPECTED_TASKS
        for seed in (3072, 3073, 3074)
        for selector in selectors
    }
    if keys != expected_keys:
        fail("selector-baseline audit row coverage mismatch")

    splits = {entry.get("split"): entry for entry in data.get("split_summaries", [])}
    expected_splits = {
        "all_three_training_seeds",
        "development_seed_3072",
        "heldout_training_seeds_3073_3074",
    }
    if set(splits) != expected_splits:
        fail(f"selector-baseline audit splits changed: {sorted(splits)}")

    expected = {
        "all_three_training_seeds": {
            "aggregate_rank_acpc_pcc_cra_maf": (12, 10, 3, 2.25),
            "fixed_std_0.08": (12, 10, 4, 2.14),
            "best_maf_only": (12, 10, 5, 1.89),
            "random_nonzero_std": (12, None, None, 7.02),
            "oracle_best": (12, 12, 12, 0.00),
        },
        "heldout_training_seeds_3073_3074": {
            "aggregate_rank_acpc_pcc_cra_maf": (8, 7, 1, 2.21),
            "fixed_std_0.08": (8, 7, 2, 2.08),
            "best_maf_only": (8, 7, 2, 1.62),
            "random_nonzero_std": (8, None, None, 7.23),
            "oracle_best": (8, 8, 8, 0.00),
        },
    }
    for split, split_expected in expected.items():
        row_map = {row.get("selector"): row for row in splits[split].get("rows", [])}
        for selector, (want_n, want_within, want_exact, want_regret) in split_expected.items():
            row = row_map.get(selector)
            if row is None:
                fail(f"selector-baseline audit missing {split}/{selector}")
            if int(row.get("n_task_seed_blocks")) != want_n:
                fail(f"selector-baseline audit {split}/{selector} n changed")
            if want_within is None:
                if row.get("within_5pp_hits") is not None:
                    fail(f"selector-baseline audit {split}/{selector} within should be None")
            elif int(row.get("within_5pp_hits")) != want_within:
                fail(f"selector-baseline audit {split}/{selector} within changed")
            if want_exact is None:
                if row.get("exact_best_hits") is not None:
                    fail(f"selector-baseline audit {split}/{selector} exact should be None")
            elif int(row.get("exact_best_hits")) != want_exact:
                fail(f"selector-baseline audit {split}/{selector} exact changed")
            got_regret = round2(float(row.get("mean_regret_to_best_pp")))
            if got_regret != want_regret:
                fail(
                    f"selector-baseline audit {split}/{selector} regret changed: "
                    f"got {got_regret}, want {want_regret}"
                )


def check_selector_plateau_audit_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "selector_plateau_audit_20260704.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-plateau-membership-audit-20260704-v2":
        fail(f"selector plateau audit schema changed: {meta.get('schema_version')!r}")
    if round2(float(meta.get("tolerance_pp"))) != 5.00:
        fail("selector plateau audit tolerance must remain 5pp")
    if int(meta.get("screen_size_per_block")) != 4:
        fail("selector plateau audit must screen top half of nonzero candidates")
    interp = meta.get("interpretation", "")
    if "Plateau-membership screen" not in interp or "candidate label" not in interp:
        fail("selector plateau audit must state plateau-membership/candidate-label framing")
    if "point-optimal selector target" not in interp:
        fail("selector plateau audit must reject point-optimal selector framing")

    rows = data.get("membership_summaries", [])
    if len(rows) != 7:
        fail(f"selector plateau audit expected 7 membership summaries, got {len(rows)}")
    row_map = {row.get("rule"): row for row in rows}
    expected = {
        "Aggregate ACPC/PCC/CRA/MAF": (12, 48, 68, 42, 6, 26, 22, 0.875, 0.618),
        "ACPC only": (12, 48, 68, 42, 6, 26, 22, 0.875, 0.618),
        "PCC only": (12, 48, 68, 42, 6, 26, 22, 0.875, 0.618),
        "CRA only": (12, 48, 68, 42, 6, 26, 22, 0.875, 0.618),
        "MAF only": (12, 48, 68, 44, 4, 24, 24, 0.917, 0.647),
        "High-std top-half reference": (12, 48, 68, 42, 6, 26, 22, 0.875, 0.618),
    }
    random_name = "Random top-half reference (exact expectation)"
    if set(row_map) != set(expected) | {random_name}:
        fail(f"selector plateau audit membership rows changed: {sorted(row_map)}")
    for rule, want in expected.items():
        row = row_map[rule]
        got = (
            int(row.get("plateau_presence_hits")),
            int(row.get("screened_rows")),
            int(row.get("true_plateau_rows")),
            int(row.get("true_positive_rows")),
            int(row.get("false_positive_rows")),
            int(row.get("false_negative_rows")),
            int(row.get("true_negative_rows")),
            round(float(row.get("screen_precision")), 3),
            round(float(row.get("plateau_recall")), 3),
        )
        if got != want:
            fail(f"selector plateau audit {rule} changed: got {got}, want {want}")
    random_row = row_map[random_name]
    got_random = (
        round2(float(random_row.get("plateau_presence_hits_expected"))),
        int(random_row.get("screened_rows")),
        int(random_row.get("true_plateau_rows")),
        round2(float(random_row.get("true_positive_rows_expected"))),
        round2(float(random_row.get("false_positive_rows_expected"))),
        round2(float(random_row.get("false_negative_rows_expected"))),
        round2(float(random_row.get("true_negative_rows_expected"))),
        round(float(random_row.get("screen_precision_expected")), 3),
        round(float(random_row.get("plateau_recall_expected")), 3),
    )
    want_random = (11.96, 48, 68, 34.00, 14.00, 34.00, 14.00, 0.708, 0.500)
    if got_random != want_random:
        fail(f"selector plateau audit random reference changed: got {got_random}, want {want_random}")

def check_residual_diagnostic_audit_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "residual_diagnostic_audit_20260704.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-residual-diagnostic-audit-0.1":
        fail(f"residual diagnostic audit schema changed: {meta.get('schema_version')!r}")
    controls = meta.get("controls", "")
    if "std_max" not in controls or "task fixed effects" not in controls or "training-seed" not in controls:
        fail("residual diagnostic audit controls must include std_max, task, and training seed")
    rows = data.get("metric_rows", [])
    if len(rows) != 8:
        fail(f"residual diagnostic audit expected 8 metric rows, got {len(rows)}")
    expected = {
        ("ACPC-H/trans.", "obs0.08 success"): (0.41, 0.07, -0.22, 0.30),
        ("ACPC-H/trans.", "reduced drop"): (0.62, 0.19, 0.06, 0.36),
        ("PCC", "obs0.08 success"): (0.38, 0.09, -0.16, 0.33),
        ("PCC", "reduced drop"): (0.60, 0.20, 0.06, 0.37),
        ("CRA", "obs0.08 success"): (0.15, 0.23, -0.07, 0.47),
        ("CRA", "reduced drop"): (0.54, 0.29, 0.14, 0.45),
        ("MAF", "obs0.08 success"): (-0.02, 0.30, 0.07, 0.48),
        ("MAF", "reduced drop"): (0.45, 0.23, 0.11, 0.34),
    }
    got_keys = {(row.get("metric"), row.get("outcome")) for row in rows}
    if got_keys != set(expected):
        fail(f"residual diagnostic audit row keys changed: {sorted(got_keys)}")
    for row in rows:
        key = (row.get("metric"), row.get("outcome"))
        if int(row.get("n_rows")) != 96:
            fail(f"residual diagnostic audit {key} n_rows changed")
        if int(row.get("n_task_seed_blocks")) != 12:
            fail(f"residual diagnostic audit {key} block count changed")
        if int(row.get("n_bootstrap_valid")) != 2000:
            fail(f"residual diagnostic audit {key} bootstrap count changed")
        want_ord, want_partial, want_lo, want_hi = expected[key]
        got_ord = round2(float(row.get("ordinary_spearman_signed")))
        got_partial = round2(float(row.get("partial_spearman_signed_controlling_std_task_seed")))
        got_lo, got_hi = [round2(float(x)) for x in row.get("block_bootstrap_ci95", [])]
        got = (got_ord, got_partial, got_lo, got_hi)
        want = (want_ord, want_partial, want_lo, want_hi)
        if got != want:
            fail(f"residual diagnostic audit {key} changed: got {got}, want {want}")


def check_selector_incremental_audit_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "selector_incremental_audit_20260704.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-selector-incremental-audit-0.1":
        fail(f"selector incremental audit schema changed: {meta.get('schema_version')!r}")
    controls = set(meta.get("controls", []))
    required_controls = {"std_max", "std_max^2", "task fixed effects", "training-seed fixed effects"}
    if not required_controls.issubset(controls):
        fail(f"selector incremental audit controls changed: {controls}")
    rows = data.get("compact_rows", [])
    if len(rows) != 6:
        fail(f"selector incremental audit expected 6 compact rows, got {len(rows)}")
    row_map = {row.get("metric"): row for row in rows}
    expected = {
        "Aggregate ACPC/PCC/CRA/MAF": (0.16, 0.03, 0.01, 0.07, 0.11),
        "ACPC-H/trans.": (0.12, 0.01, 0.01, 0.16, -0.08),
        "PCC": (0.13, 0.02, 0.01, 0.12, -0.04),
        "CRA": (0.23, 0.05, 0.03, 0.005, 0.10),
        "MAF": (0.15, 0.02, 0.01, 0.12, 0.15),
        "Elite overlap": (0.13, 0.02, 0.01, 0.18, -0.01),
    }
    if set(row_map) != set(expected):
        fail(f"selector incremental audit metrics changed: {sorted(row_map)}")
    for metric, want in expected.items():
        row = row_map[metric]
        got = (
            round2(float(row["reduced_drop_partial_r"])),
            round2(float(row["reduced_drop_partial_r2"])),
            round2(float(row["reduced_drop_incremental_r2"])),
            round(float(row["reduced_drop_block_permutation_p"]), 3) if metric == "CRA" else round2(float(row["reduced_drop_block_permutation_p"])),
            round2(float(row["obs008_success_partial_r"])),
        )
        want_tuple = (want[0], want[1], want[2], want[3], want[4])
        if got != want_tuple:
            fail(f"selector incremental audit {metric} changed: got {got}, want {want_tuple}")
    metric_rows = data.get("metric_rows", [])
    if len(metric_rows) != 12:
        fail(f"selector incremental audit expected 12 full metric rows, got {len(metric_rows)}")
    if any(int(row.get("n", 0)) != 96 for row in metric_rows):
        fail("selector incremental audit full rows must use 96 nonzero checkpoint rows")


def check_semantic_task_grounded_margin_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "semantic_task_grounded_margin_lewm_three_seed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("pair_rule") != "task_grounded_near_boundary":
        fail("task-grounded semantic margin artifact must use task_grounded_near_boundary")
    if round2(float(meta.get("local_quantile"))) != 0.35:
        fail("task-grounded semantic margin local quantile changed")
    rows = data.get("rows", [])
    if len(rows) != 24 or any(row.get("status") != "ok" for row in rows):
        fail("task-grounded semantic margin artifact must contain 24 ok rows")
    coverage = data.get("coverage", {})
    expected_coverage = {
        f"{task}:{std}": [3072, 3073, 3074]
        for task in ("TwoRoom", "PushT", "Reacher", "Cube")
        for std in ("0.0", "0.08")
    }
    if coverage != expected_coverage:
        fail(f"task-grounded semantic margin coverage mismatch: {coverage}")
    expected_pair_counts = {"TwoRoom": 61, "PushT": 98, "Reacher": 100, "Cube": 100}
    for row in rows:
        task = row.get("task")
        if int(row.get("semantic_pair_count")) != expected_pair_counts[task]:
            fail(f"task-grounded semantic pair count changed for {task}: {row.get('semantic_pair_count')}")
        if "task-grounded" not in row.get("semantic_factor", ""):
            fail(f"task-grounded semantic factor missing for {task}")
    summary = {(row["task"], row["std_key"]): row for row in data.get("summary_rows", [])}
    expected_pass = {
        ("TwoRoom", "0.0"): 0.34,
        ("TwoRoom", "0.08"): 0.99,
        ("PushT", "0.0"): 0.44,
        ("PushT", "0.08"): 1.00,
        ("Reacher", "0.0"): 0.73,
        ("Reacher", "0.08"): 1.00,
        ("Cube", "0.0"): 0.45,
        ("Cube", "0.08"): 1.00,
    }
    if set(summary) != set(expected_pass):
        fail(f"task-grounded semantic summary rows mismatch: {sorted(summary)}")
    for key, want in expected_pass.items():
        got = round2(float(summary[key]["semantic_margin_pass_rate_mean"]))
        if got != want:
            fail(f"task-grounded semantic pass-rate mismatch for {key}: got {got}, want {want}")
    if round2(float(summary[("TwoRoom", "0.08")]["semantic_margin_median_mean"])) != 15.27:
        fail("task-grounded TwoRoom high-noise margin changed")
    if round2(float(summary[("PushT", "0.0")]["semantic_margin_median_mean"])) != -0.58:
        fail("task-grounded PushT base margin changed")


def check_cem_trace_audit_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "cem_trace_audit_20260704.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-cem-trace-audit-0.1":
        fail(f"CEM trace audit schema changed: {meta.get('schema_version')!r}")
    expected_meta = {"n_sequences": 4, "plan_horizon": 5, "action_block": 5, "cem_num_samples": 64, "cem_n_steps": 8, "cem_topk": 8}
    for key, want in expected_meta.items():
        if int(meta.get(key, -1)) != want:
            fail(f"CEM trace audit metadata {key} changed: {meta.get(key)} != {want}")
    rows = data.get("rows", [])
    if len(rows) != 24 or any(row.get("status") != "ok" for row in rows):
        fail("CEM trace audit must contain 24 ok rows")
    got = {(row.get("task"), int(row.get("training_seed")), row.get("std_key")) for row in rows}
    expected = {
        (task, seed, std)
        for task in ("TwoRoom", "PushT", "Reacher", "Cube")
        for seed in (3072, 3073, 3074)
        for std in ("0.0", "0.08")
    }
    if got != expected:
        fail("CEM trace audit row coverage changed")
    summary = {(row["task"], row["std_key"]): row for row in data.get("summary_rows", [])}
    expected_plan = {
        ("TwoRoom", "0.0"): 1.24,
        ("TwoRoom", "0.08"): 0.61,
        ("PushT", "0.0"): 1.96,
        ("PushT", "0.08"): 0.64,
        ("Reacher", "0.0"): 1.15,
        ("Reacher", "0.08"): 0.38,
        ("Cube", "0.0"): 1.30,
        ("Cube", "0.08"): 0.51,
    }
    if set(summary) != set(expected_plan):
        fail(f"CEM trace summary rows mismatch: {sorted(summary)}")
    for key, want in expected_plan.items():
        got_plan = round2(float(summary[key]["final_plan_l2_per_dim_mean_mean"]))
        if got_plan != want:
            fail(f"CEM trace final plan L2/dim changed for {key}: got {got_plan}, want {want}")
    for task in ("TwoRoom", "PushT", "Reacher", "Cube"):
        base = float(summary[(task, "0.0")]["final_plan_l2_per_dim_mean_mean"])
        robust = float(summary[(task, "0.08")]["final_plan_l2_per_dim_mean_mean"])
        if not robust < base:
            fail(f"CEM trace high-noise plan shift no longer below base for {task}: {robust} >= {base}")
    if round2(float(summary[("TwoRoom", "0.08")]["final_seeded_top1_flip_rate_mean"])) != 0.92:
        fail("CEM trace TwoRoom boundary flip rate changed")
    if round2(float(summary[("Reacher", "0.08")]["final_seeded_top1_flip_rate_mean"])) != 0.25:
        fail("CEM trace Reacher high-noise flip rate changed")


def check_three_seed_diagnostic_validation_json() -> None:
    phase0_path = ROOT / "assets" / "paper1_data" / "acpc_phase0_lewm_three_seed.json"
    phase0 = json.loads(phase0_path.read_text(encoding="utf-8"))
    rows = phase0.get("rows", [])
    if len(rows) != 108 or any(row.get("status") != "ok" for row in rows):
        fail("three-seed Phase-0 LeWM artifact must contain 108 ok rows")
    expected = {
        (task, seed, std)
        for task in EXPECTED_TASKS
        for seed in (3072, 3073, 3074)
        for std in EXPECTED_CONFIGS
    }
    got = {(row.get("task"), int(row.get("training_seed")), str(row.get("std_key"))) for row in rows}
    if got != expected:
        fail(f"three-seed Phase-0 coverage mismatch: missing={sorted(expected - got)[:5]}")
    required_fields = {
        "pixels_std0.08_success",
        "pixels_goal_std0.08_success",
        "corruption_drop",
        "acpc_h_norm_by_transition",
        "pcc_abs_median",
        "cra_spearman_mean",
        "maf_flip_rate",
    }
    for row in rows:
        missing = required_fields - set(row)
        if missing:
            fail(f"three-seed Phase-0 row missing fields {missing}: {row.get('task')} {row.get('training_seed')} {row.get('std_key')}")

    validation_path = ROOT / "assets" / "paper1_data" / "three_seed_diagnostic_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    summary = validation.get("summary", {})
    expected_summary = {
        "n_task_seed_blocks": 12,
        "exact_best_hits": 2,
        "within_5pp_hits": 10,
        "checkpoint_candidates_per_block": 8,
    }
    for key, want in expected_summary.items():
        if summary.get(key) != want:
            fail(f"three-seed diagnostic validation {key} mismatch: {summary.get(key)} != {want}")
    if round2(float(summary.get("mean_selected_regret_to_best_pp"))) != 2.25:
        fail("three-seed diagnostic validation mean regret changed")
    splits = {row.get("split"): row for row in validation.get("split_summaries", [])}
    heldout = splits.get("heldout_training_seeds_3073_3074")
    if heldout is None:
        fail("three-seed diagnostic validation missing held-out split summary")
    expected_heldout = {
        "n_task_seed_blocks": 8,
        "n_checkpoint_candidates": 64,
        "exact_best_hits": 0,
        "within_5pp_hits": 7,
    }
    for key, want in expected_heldout.items():
        if heldout.get(key) != want:
            fail(f"three-seed held-out split {key} mismatch: {heldout.get(key)} != {want}")
    if round2(float(heldout.get("mean_selected_regret_to_best_pp"))) != 2.21:
        fail("three-seed held-out split mean regret changed")
    ci = heldout.get("bootstrap_ci95_mean_selected_regret_to_best_pp")
    if [round2(float(v)) for v in ci] != [1.04, 3.54]:
        fail(f"three-seed held-out split CI changed: {ci}")
    selection = validation.get("selection_rows", [])
    if len(selection) != 12:
        fail("three-seed diagnostic validation must contain 12 selection rows")
    if sorted({int(row["training_seed"]) for row in selection}) != [3072, 3073, 3074]:
        fail("three-seed diagnostic validation must cover seeds 3072/3073/3074")


def check_semantic_margin_passrate_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "semantic_margin_passrate_lewm_three_seed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if len(rows) != 24 or any(row.get("status") != "ok" for row in rows):
        fail("task-state proxy margin pass-rate artifact must contain 24 ok rows")
    coverage = data.get("coverage", {})
    expected_coverage = {
        f"{task}:{std}": [3072, 3073, 3074]
        for task in ("TwoRoom", "PushT", "Reacher", "Cube")
        for std in ("0.0", "0.08")
    }
    if coverage != expected_coverage:
        fail(f"task-state proxy margin coverage mismatch: {coverage}")
    summary = {(row["task"], row["std_key"]): row for row in data.get("summary_rows", [])}
    expected_pass = {
        ("TwoRoom", "0.0"): 0.44,
        ("TwoRoom", "0.08"): 1.00,
        ("PushT", "0.0"): 0.27,
        ("PushT", "0.08"): 1.00,
        ("Reacher", "0.0"): 0.58,
        ("Reacher", "0.08"): 1.00,
        ("Cube", "0.0"): 0.25,
        ("Cube", "0.08"): 1.00,
    }
    if set(summary) != set(expected_pass):
        fail(f"task-state proxy margin summary rows mismatch: {sorted(summary)}")
    for key, want in expected_pass.items():
        got = round2(float(summary[key]["semantic_margin_pass_rate_mean"]))
        if got != want:
            fail(f"task-state proxy margin pass-rate mismatch for {key}: got {got}, want {want}")



def check_semantic_local_margin_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "semantic_local_margin_lewm_three_seed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("pair_rule") != "local_task_feature_contrast":
        fail("local task-feature margin artifact must use local_task_feature_contrast")
    if round2(float(meta.get("local_quantile"))) != 0.35:
        fail("local task-feature margin artifact local quantile changed")
    rows = data.get("rows", [])
    if len(rows) != 24 or any(row.get("status") != "ok" for row in rows):
        fail("local task-feature margin artifact must contain 24 ok rows")
    coverage = data.get("coverage", {})
    expected_coverage = {
        f"{task}:{std}": [3072, 3073, 3074]
        for task in ("TwoRoom", "PushT", "Reacher", "Cube")
        for std in ("0.0", "0.08")
    }
    if coverage != expected_coverage:
        fail(f"local task-feature margin coverage mismatch: {coverage}")
    summary = {(row["task"], row["std_key"]): row for row in data.get("summary_rows", [])}
    expected_pass = {
        ("TwoRoom", "0.0"): 0.59,
        ("TwoRoom", "0.08"): 1.00,
        ("PushT", "0.0"): 0.54,
        ("PushT", "0.08"): 1.00,
        ("Reacher", "0.0"): 0.81,
        ("Reacher", "0.08"): 1.00,
        ("Cube", "0.0"): 0.53,
        ("Cube", "0.08"): 1.00,
    }
    if set(summary) != set(expected_pass):
        fail(f"local task-feature margin summary rows mismatch: {sorted(summary)}")
    for key, want in expected_pass.items():
        got = round2(float(summary[key]["semantic_margin_pass_rate_mean"]))
        if got != want:
            fail(f"local task-feature margin pass-rate mismatch for {key}: got {got}, want {want}")
    for key in (("PushT", "0.08"), ("Cube", "0.08")):
        margin = round2(float(summary[key]["semantic_margin_median_mean"]))
        if margin < 18.0:
            fail(f"local task-feature high-noise margin unexpectedly low for {key}: {margin}")


def check_margin_flip_curve_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "margin_flip_curve_lewm_three_seed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-margin-flip-curve-0.1":
        fail(f"margin flip schema mismatch: {meta.get('schema_version')}")
    if meta.get("seeds") != [3072, 3073, 3074]:
        fail(f"margin flip seeds mismatch: {meta.get('seeds')}")
    if meta.get("tasks") != ["TwoRoom", "PushT", "Reacher", "Cube"]:
        fail(f"margin flip tasks mismatch: {meta.get('tasks')}")
    if meta.get("std_keys") != ["0.0", "0.08"]:
        fail(f"margin flip std keys mismatch: {meta.get('std_keys')}")
    if [round(float(q), 2) for q in meta.get("threshold_quantiles", [])] != [0.0, 0.5, 0.75, 0.9]:
        fail(f"margin flip threshold quantiles mismatch: {meta.get('threshold_quantiles')}")
    rows = data.get("rows", [])
    samples = data.get("sample_rows", [])
    if len(rows) != 96:
        fail(f"margin flip row count mismatch: {len(rows)}")
    if len(samples) != 2400:
        fail(f"margin flip sample row count mismatch: {len(samples)}")
    if any(row.get("status") != "ok" for row in rows):
        fail("margin flip rows contain non-ok status")
    coverage = {
        (row["task"], row["std_key"], round(float(row["threshold_quantile"]), 2), int(row["training_seed"]))
        for row in rows
    }
    expected = {
        (task, std, q, seed)
        for task in ("TwoRoom", "PushT", "Reacher", "Cube")
        for std in ("0.0", "0.08")
        for q in (0.0, 0.5, 0.75, 0.9)
        for seed in (3072, 3073, 3074)
    }
    if coverage != expected:
        fail("margin flip coverage mismatch")
    summary = {
        (row["task"], row["std_key"], round(float(row["threshold_quantile"]), 2)): row
        for row in data.get("summary_rows", [])
    }
    if set(summary) != {
        (task, std, q)
        for task in ("TwoRoom", "PushT", "Reacher", "Cube")
        for std in ("0.0", "0.08")
        for q in (0.0, 0.5, 0.75, 0.9)
    }:
        fail("margin flip summary coverage mismatch")
    expected_q75 = {
        ("TwoRoom", "0.0"): 0.67,
        ("TwoRoom", "0.08"): 0.00,
        ("PushT", "0.0"): 0.92,
        ("PushT", "0.08"): 0.00,
        ("Reacher", "0.0"): 0.79,
        ("Reacher", "0.08"): 0.00,
        ("Cube", "0.0"): 0.84,
        ("Cube", "0.08"): 0.00,
    }
    for key, want in expected_q75.items():
        got = round2(float(summary[(key[0], key[1], 0.75)]["flip_rate_mean"]))
        if got != want:
            fail(f"margin flip q75 mean mismatch for {key}: got {got}, want {want}")
    for task in ("TwoRoom", "PushT", "Reacher", "Cube"):
        base_all = float(summary[(task, "0.0", 0.0)]["flip_rate_mean"])
        robust_all = float(summary[(task, "0.08", 0.0)]["flip_rate_mean"])
        if base_all < 0.75:
            fail(f"margin flip base all-sample flip unexpectedly low for {task}: {base_all}")
        if robust_all > 0.04:
            fail(f"margin flip robust all-sample flip unexpectedly high for {task}: {robust_all}")



def check_unseen_atr_smpr_summary_json() -> None:
    for rel, expected_coverage in {
        "semantic_task_grounded_margin_unseen_blur_lewm_three_seed.json": {
            "TwoRoom:0.0": [3072, 3073, 3074],
            "TwoRoom:0.08": [3072, 3073, 3074],
            "Reacher:0.0": [3072, 3073, 3074],
            "Reacher:0.08": [3072, 3073, 3074],
        },
        "semantic_task_grounded_margin_unseen_resize_lewm_three_seed.json": {
            "PushT:0.0": [3072, 3073, 3074],
            "PushT:0.08": [3072, 3073, 3074],
            "Cube:0.0": [3072, 3073, 3074],
            "Cube:0.08": [3072, 3073, 3074],
        },
    }.items():
        data = json.loads((ROOT / "assets" / "paper1_data" / rel).read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        if meta.get("pair_rule") != "task_grounded_near_boundary":
            fail(f"{rel} must use task_grounded_near_boundary")
        if round2(float(meta.get("local_quantile"))) != 0.35:
            fail(f"{rel} local quantile changed")
        if data.get("coverage") != expected_coverage:
            fail(f"{rel} coverage mismatch: {data.get('coverage')}")
        if len(data.get("rows", [])) != 12 or any(row.get("status") != "ok" for row in data.get("rows", [])):
            fail(f"{rel} must contain 12 ok rows")

    path = ROOT / "assets" / "paper1_data" / "unseen_atr_smpr_summary_20260707.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-unseen-atr-smpr-summary-20260707-v1":
        fail(f"unseen ATR/SMPR schema changed: {meta.get('schema_version')!r}")
    if "same unseen stressor" not in meta.get("smpr_definition", ""):
        fail("unseen SMPR definition must remain stressor-specific")
    rows = {row.get("task"): row for row in data.get("summary_rows", [])}
    expected = {
        "TwoRoom": (47.67, 90.78, 1.61, 1.24, 0.16, 0.77),
        "Reacher": (22.00, 71.22, 2.81, 0.54, 0.60, 0.98),
        "PushT": (63.44, 66.33, 1.77, 1.53, 0.93, 0.96),
        "Cube": (57.00, 56.11, 1.35, 1.59, 0.98, 0.95),
    }
    if set(rows) != set(expected):
        fail(f"unseen ATR/SMPR summary task coverage changed: {sorted(rows)}")
    for task, want in expected.items():
        row = rows[task]
        got = (
            round2(float(row["baseline_stress_success"]["mean"])),
            round2(float(row["std008_stress_success"]["mean"])),
            round2(float(row["ATR_q90_0.0"]["mean"])),
            round2(float(row["ATR_q90_0.08"]["mean"])),
            round2(float(row["SMPR_0.0"]["mean"])),
            round2(float(row["SMPR_0.08"]["mean"])),
        )
        if got != want:
            fail(f"unseen ATR/SMPR summary {task} changed: got {got}, want {want}")
    corr = data.get("correlations", {})
    if int(corr.get("seed_rows_n", 0)) != 12:
        fail("unseen ATR/SMPR correlations must use 12 seed rows")
    if round2(float(corr.get("spearman_stress_delta_vs_ATR_drop"))) != 0.84:
        fail("unseen ATR-drop Spearman association changed")
    if round2(float(corr.get("spearman_stress_delta_vs_SMPR_gain"))) != 0.87:
        fail("unseen SMPR-gain Spearman association changed")

def check_compressed_metrics_summary_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "compressed_metrics_summary_20260706.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != "paper1-compressed-selective-acpc-20260706-v1":
        fail(f"compressed metrics schema changed: {meta.get('schema_version')!r}")
    if meta.get("same_state_metric") != "ATR_q90" or meta.get("selective_metric") != "SMPR_m0":
        fail("compressed metrics must remain ATR_q90 + SMPR_m0")
    rows = data.get("summary_rows", [])
    if len(rows) != 4:
        fail(f"compressed metrics expected 4 summary rows, got {len(rows)}")
    expected = {
        "TwoRoom": (1.509, 0.111, 0.34, 0.99, 68.78, 97.11),
        "PushT": (3.580, 0.247, 0.44, 1.00, 7.22, 85.78),
        "Reacher": (2.628, 0.082, 0.73, 1.00, 18.22, 81.56),
        "Cube": (2.320, 0.100, 0.45, 1.00, 43.11, 62.56),
    }
    row_map = {row.get("task"): row for row in rows}
    if set(row_map) != set(expected):
        fail(f"compressed metrics task coverage changed: {sorted(row_map)}")
    for task, want in expected.items():
        row = row_map[task]
        got = (
            round(float(row["ATR_q90_0.0"]["mean"]), 3),
            round(float(row["ATR_q90_0.08"]["mean"]), 3),
            round(float(row["SMPR_0.0"]["mean"]), 2),
            round(float(row["SMPR_0.08"]["mean"]), 2),
            round(float(row["obs008_success_0.0"]["mean"]), 2),
            round(float(row["obs008_success_0.08"]["mean"]), 2),
        )
        if got != want:
            fail(f"compressed metrics {task} changed: got {got}, want {want}")

def check_target_view_closed_loop_summary_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "target_view_closed_loop_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    rows = data.get("closed_loop_pixels_std0.08_across_eight_checkpoints")
    expected = {
        "tworoom": (94.708333125, 61.75, 32.958333125),
        "pusht": (72.833333125, 6.749999875, 66.08333325),
        "reacher": (76.166666625, 19.624999875, 56.54166675),
        "cube": (59.83333325, 39.625, 20.20833325),
    }
    if not isinstance(rows, dict) or set(rows) != set(expected):
        fail(
            "target-view summary tasks mismatch: "
            f"expected {sorted(expected)}, got {sorted(rows or {})}"
        )

    for task, want in expected.items():
        row = rows[task]
        got = (
            float(row["full_sequence_mean"]),
            float(row["origin_target_mean"]),
            float(row["full_sequence_advantage"]),
        )
        if any(not approx_equal(g, w) for g, w in zip(got, want)):
            fail(f"target-view closed-loop summary mismatch for {task}: got {got}, want {want}")

    probe = data.get("representative_pusht_0to008", {})
    canonical = probe.get("canonical_seeds_42_43_44", {})
    if canonical.get("full_sequence_pixels_std0.08_raw") != [88.0, 82.0, 89.0]:
        fail("target-view PushT full-sequence canonical raw values changed")
    if canonical.get("origin_target_pixels_std0.08_raw") != [12.0, 4.0, 10.0]:
        fail("target-view PushT origin-target canonical raw values changed")


def check_published_correlations() -> None:
    evals = json.loads((ROOT / "assets" / "paper1_data" / "canonical_evals_20260517.json").read_text(encoding="utf-8"))
    diag = json.loads((ROOT / "assets" / "paper1_data" / "canonical_diagnostics_20260517.json").read_text(encoding="utf-8"))

    predictor = diag["predictor_metrics_by_task"]
    published = diag["published_correlations"]

    metrics = (
        "predictor_target_to_nn_cos_ratio_at_max_std",
        "predictor_rollout_T8_l2_at_max_std",
    )

    for task in sorted(EXPECTED_TASKS):
        std_keys = sorted(evals[task], key=float)
        z = [float(std_key) for std_key in std_keys]
        clean = [float(evals[task][std_key]["metrics"]["clean"]["mean"]) for std_key in std_keys]
        px08 = [
            float(evals[task][std_key]["metrics"]["pixels_std0.08"]["mean"])
            for std_key in std_keys
        ]
        drop = [c - p for c, p in zip(clean, px08)]

        for metric in metrics:
            xs = [float(predictor[task][std_key][metric]) for std_key in std_keys]
            got_pearson = round2(pearson(xs, drop))
            got_spearman = round2(spearman(xs, drop))
            want = published["table4_ood_drop"][task][metric]
            if got_pearson != round2(want["pearson"]) or got_spearman != round2(want["spearman"]):
                fail(
                    f"published Table 4 mismatch for {task}/{metric}: "
                    f"got pearson={got_pearson}, spearman={got_spearman}; "
                    f"want pearson={want['pearson']}, spearman={want['spearman']}"
                )

            got_partial = round2(partial_spearman(xs, drop, z))
            want_partial = published["table4b_partial_spearman_ood_drop_given_std_max"][task][metric]
            if got_partial != round2(want_partial):
                fail(
                    f"published Table 4b mismatch for {task}/{metric}: "
                    f"got partial={got_partial}; want partial={want_partial}"
                )

    push_keys = sorted(evals["PushT"], key=float)
    z = [float(std_key) for std_key in push_keys]
    fragility = [
        float(predictor["PushT"][std_key]["predictor_target_to_nn_cos_ratio_at_max_std"])
        for std_key in push_keys
    ]
    clean = [float(evals["PushT"][std_key]["metrics"]["clean"]["mean"]) for std_key in push_keys]
    px08 = [
        float(evals["PushT"][std_key]["metrics"]["pixels_std0.08"]["mean"])
        for std_key in push_keys
    ]
    drop = [c - p for c, p in zip(clean, px08)]
    table5 = published["table5_pusht_fragility_metric"]["spearman"]
    recomputed = {
        "rho_std_max_metric": round2(spearman(z, fragility)),
        "rho_std_max_clean": round2(spearman(z, clean)),
        "rho_std_max_pixels_std0.08": round2(spearman(z, px08)),
        "rho_std_max_ood_drop": round2(spearman(z, drop)),
        "rho_metric_clean_unconditional": round2(spearman(fragility, clean)),
        "rho_metric_clean_partial_given_std_max": round2(partial_spearman(fragility, clean, z)),
        "rho_metric_pixels_std0.08_unconditional": round2(spearman(fragility, px08)),
        "rho_metric_pixels_std0.08_partial_given_std_max": round2(
            partial_spearman(fragility, px08, z)
        ),
        "rho_metric_ood_drop_unconditional": round2(spearman(fragility, drop)),
        "rho_metric_ood_drop_partial_given_std_max": round2(partial_spearman(fragility, drop, z)),
    }
    for key, got in recomputed.items():
        want = round2(table5[key])
        if got != want:
            fail(f"published Table 5 mismatch for {key}: got {got}, want {want}")


def check_radius_margin_certificate_outputs() -> None:
    summary_path = ROOT / "paper1" / "results" / "radius_margin_certificate_summary.csv"
    gate_path = ROOT / "paper1" / "results" / "radius_margin_gate_ablation.csv"
    boundary_path = ROOT / "paper1" / "results" / "radius_margin_boundary_alignment.csv"
    top1_path = ROOT / "paper1" / "results" / "fixed_pool_top1_agreement.csv"
    summary_rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
    gate_rows = list(csv.DictReader(gate_path.open(newline="", encoding="utf-8")))
    boundary_rows = list(csv.DictReader(boundary_path.open(newline="", encoding="utf-8")))
    top1_rows = list(csv.DictReader(top1_path.open(newline="", encoding="utf-8")))

    if len(summary_rows) != 36:
        fail(f"radius-margin summary expected 36 rows, got {len(summary_rows)}")
    if {row["task"] for row in summary_rows} != EXPECTED_TASKS:
        fail("radius-margin summary task set mismatch")
    expected_stdmax_csv = {f"{float(std):.2f}" for std in EXPECTED_CONFIGS}
    if {row["train_stdmax"] for row in summary_rows} != expected_stdmax_csv:
        fail("radius-margin summary stdmax grid mismatch")

    required_columns = {
        "task",
        "train_stdmax",
        "eval_sigma",
        "n_training_seeds",
        "score_clean_mean",
        "score_obs_sigma_0p08_mean",
        "behavioral_plateau_label",
        "atr_q90_mean",
        "clean_margin_q50_mean",
        "cost_drift_q90_mean",
        "certificate_gap_q50_q90_mean",
        "certificate_pass_proxy",
        "candidate_count_mean",
        "notes",
    }
    missing = required_columns - set(summary_rows[0])
    if missing:
        fail(f"radius-margin summary missing columns: {sorted(missing)}")

    for row in summary_rows:
        if row["n_training_seeds"] != "3":
            fail(f"radius-margin summary expected three training seeds: {row}")
        if row["eval_sigma"] != "0.08":
            fail(f"radius-margin summary expected eval_sigma 0.08: {row}")
        if float(row["candidate_count_mean"]) != 65.0:
            fail(f"radius-margin summary expected candidate_count_mean 65.0: {row}")
        for key in (
            "score_clean_mean",
            "score_obs_sigma_0p08_mean",
            "atr_q90_mean",
            "clean_margin_q50_mean",
            "cost_drift_q90_mean",
            "certificate_gap_q50_q90_mean",
        ):
            value = float(row[key])
            if not math.isfinite(value):
                fail(f"radius-margin summary has non-finite {key}: {row}")
        if "Phase-0" in row["notes"] or "artifact" in row["notes"]:
            fail(f"radius-margin summary note uses internal wording: {row['notes']}")

    expected_proxy_ranges = {
        "TwoRoom": ("0.02-0.08", "0.01-0.08", "none", "0.01"),
        "PushT": ("0.02-0.08", "0.03-0.08", "0.02", "none"),
        "Reacher": ("0.04-0.08", "0.02-0.08", "none", "0.02;0.03"),
        "Cube": ("0.03-0.08", "0.03-0.08", "none", "none"),
    }
    proxy_rows = {
        row["task"]: row
        for row in gate_rows
        if row["criterion"] == "fixed-pool cost-margin proxy gap > 0"
    }
    if set(proxy_rows) != EXPECTED_TASKS:
        fail("radius-margin gate missing fixed-pool proxy rows")
    for task, (pred, plateau, fp, fn) in expected_proxy_ranges.items():
        row = proxy_rows[task]
        got = (
            row["predicted_robust_stdmax_range"],
            row["behavioral_plateau_range"],
            row["false_positive_stdmax"],
            row["false_negative_stdmax"],
        )
        if got != (pred, plateau, fp, fn):
            fail(f"radius-margin gate mismatch for {task}: got {got}")
        if "Phase-0" in row["notes"] or "artifact" in row["notes"]:
            fail(f"radius-margin gate note uses internal wording: {row['notes']}")

    joint_rows = [row for row in gate_rows if row["criterion"] == "ATR+SMPR joint gate"]
    if len(joint_rows) != 4:
        fail(f"radius-margin gate expected four ATR+SMPR rows, got {len(joint_rows)}")
    for row in joint_rows:
        if row["predicted_robust_stdmax_range"] != "not computed":
            fail("radius-margin table must not mix the separate ATR/SMPR diagnostic-region audit into the cost-proxy gate row")
        if "reported separately" not in row["notes"] or "cost-proxy table" not in row["notes"]:
            fail(f"radius-margin joint-gate note missing scope explanation: {row['notes']}")
        if "artifact" in row["notes"]:
            fail(f"radius-margin joint-gate note uses internal wording: {row['notes']}")

    expected_boundary = {
        "TwoRoom": ("0.01-0.08", "0.02-0.08", "+0.01", "+0.00", "yes"),
        "PushT": ("0.03-0.08", "0.02-0.08", "-0.01", "+0.00", "yes"),
        "Reacher": ("0.02-0.08", "0.04-0.08", "+0.02", "+0.00", "partial"),
        "Cube": ("0.03-0.08", "0.03-0.08", "+0.00", "+0.00", "yes"),
    }
    if len(boundary_rows) != 4:
        fail(f"boundary alignment expected four rows, got {len(boundary_rows)}")
    for row in boundary_rows:
        task = row["task"]
        got = (
            row["recovery_band"],
            row["diagnostic_proxy_interval"],
            row["start_boundary_error_stdmax"],
            row["end_boundary_error_stdmax"],
            row["within_one_grid_tolerance"],
        )
        if got != expected_boundary.get(task):
            fail(f"boundary alignment mismatch for {task}: got {got}")

    if len(top1_rows) != 12:
        fail(f"fixed-pool top1 audit expected 12 rows, got {len(top1_rows)}")
    top1_map = {(row["task"], row["row_role"]): row for row in top1_rows}
    expected_top1 = {
        ("TwoRoom", "base"): ("0.00", "0.207", "no"),
        ("TwoRoom", "std0.08_endpoint"): ("0.08", "0.960", "yes"),
        ("PushT", "recovery_onset"): ("0.03", "0.937", "yes"),
        ("Reacher", "recovery_onset"): ("0.02", "0.810", "yes"),
        ("Cube", "std0.08_endpoint"): ("0.08", "0.997", "yes"),
    }
    for key, expected in expected_top1.items():
        row = top1_map.get(key)
        if row is None:
            fail(f"fixed-pool top1 audit missing row {key}")
        got = (row["stdmax"], row["empirical_top1_agree"], row["recovery_band_member"])
        if got != expected:
            fail(f"fixed-pool top1 audit mismatch for {key}: got {got}, want {expected}")

    sample_rows = list(csv.DictReader((ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_audit.csv").open(newline="", encoding="utf-8")))
    if len(sample_rows) != 108:
        fail(f"full-sweep sample-level certificate audit expected 108 rows, got {len(sample_rows)}")
    if any(row.get("status") != "ok" for row in sample_rows):
        fail("full-sweep sample-level certificate audit must have all rows ok")
    sample_alignment = list(csv.DictReader((ROOT / "paper1" / "results" / "sample_level_certificate_recovery_alignment.csv").open(newline="", encoding="utf-8")))
    align = {(row["task"], row["split"]): row for row in sample_alignment}
    all_fragile = align[("ALL", "fragile")]
    all_recovered = align[("ALL", "recovered")]
    got_alignment = (
        round(float(all_fragile["cert_pass_rate_median"]), 2),
        round(float(all_recovered["cert_pass_rate_median"]), 2),
        round(float(all_fragile["top1_flip_rate_median"]), 2),
        round(float(all_recovered["top1_flip_rate_median"]), 2),
    )
    if got_alignment != (0.06, 0.61, 0.54, 0.04):
        fail(f"full-sweep sample-level alignment changed: got {got_alignment}")

    ci_rows = list(csv.DictReader((ROOT / "paper1" / "results" / "sample_level_event_rate_wilson_ci.csv").open(newline="", encoding="utf-8")))
    if len(ci_rows) != 30:
        fail(f"sample-level Wilson CI expected 30 rows, got {len(ci_rows)}")
    ci = {(row["task"], row["split"], row["metric"]): row for row in ci_rows}
    got_ci = (
        round(float(ci[("ALL", "fragile", "cert-pass")]["rate"]), 2),
        round(float(ci[("ALL", "recovered", "cert-pass")]["rate"]), 2),
        round(float(ci[("ALL", "fragile", "top-1 flip")]["rate"]), 2),
        round(float(ci[("ALL", "recovered", "top-1 flip")]["rate"]), 2),
        round(float(ci[("ALL", "fragile", "top-1 flip | cert-pass")]["rate"]), 2),
        round(float(ci[("ALL", "recovered", "top-1 flip | cert-pass")]["rate"]), 2),
        int(ci[("ALL", "fragile", "top-1 flip | cert-pass")]["n"]),
        int(ci[("ALL", "recovered", "top-1 flip | cert-pass")]["n"]),
    )
    if got_ci != (0.20, 0.61, 0.53, 0.06, 0.00, 0.00, 678, 4479):
        fail(f"sample-level Wilson CI rates changed: got {got_ci}")

    sensitivity_rows = list(csv.DictReader((ROOT / "paper1" / "results" / "gaussian_sensitivity_summary.csv").open(newline="", encoding="utf-8")))
    if len(sensitivity_rows) != 12:
        fail(f"Gaussian sensitivity summary expected 12 rows, got {len(sensitivity_rows)}")
    sens = {(row["task"], row["checkpoint_type"]): row for row in sensitivity_rows}
    expected_sens = {"TwoRoom": 0.003, "PushT": 0.008, "Reacher": 0.002, "Cube": 0.008}
    for task, expected in expected_sens.items():
        got = round(float(sens[(task, "endpoint")]["sensitivity_slope_vs_base"]), 3)
        if got != expected:
            fail(f"Gaussian sensitivity endpoint/base changed for {task}: got {got}, want {expected}")

    jvp_rows = list(csv.DictReader((ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_audit.csv").open(newline="", encoding="utf-8")))
    if len(jvp_rows) != 36:
        fail(f"JVP/Hutchinson audit expected 36 rows, got {len(jvp_rows)}")
    if any(row.get("status") != "ok" for row in jvp_rows):
        fail("JVP/Hutchinson audit must have all rows ok")
    if sorted({row.get("n_sequences") for row in jvp_rows}) != ["100"] or sorted({row.get("hutchinson_probes") for row in jvp_rows}) != ["8"]:
        fail("JVP/Hutchinson audit must use n_sequences=100 and hutchinson_probes=8")
    jvp_summary = list(csv.DictReader((ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_summary.csv").open(newline="", encoding="utf-8")))
    if len(jvp_summary) != 12:
        fail(f"JVP/Hutchinson summary expected 12 rows, got {len(jvp_summary)}")
    jvp = {(row["task"], row["checkpoint_type"]): row for row in jvp_summary}
    expected_jvp_composed = {"TwoRoom": 0.003, "PushT": 0.010, "Reacher": 0.006, "Cube": 0.026}
    expected_jvp_encoder = {"TwoRoom": 0.003, "PushT": 0.017, "Reacher": 0.003, "Cube": 0.023}
    for task, expected in expected_jvp_composed.items():
        got = round(float(jvp[(task, "endpoint")]["composed_trace_per_pixel_dim_vs_base"]), 3)
        if got != expected:
            fail(f"JVP/Hutchinson composed endpoint/base changed for {task}: got {got}, want {expected}")
    for task, expected in expected_jvp_encoder.items():
        got = round(float(jvp[(task, "endpoint")]["encoder_trace_per_pixel_dim_vs_base"]), 3)
        if got != expected:
            fail(f"JVP/Hutchinson encoder endpoint/base changed for {task}: got {got}, want {expected}")

    guard_rows = list(csv.DictReader((ROOT / "paper1" / "results" / "joint_guard_side_validation.csv").open(newline="", encoding="utf-8")))
    if len(guard_rows) != 10:
        fail(f"joint guard-side validation expected 10 rows, got {len(guard_rows)}")
    guard = {(row["task"], row["split"]): row for row in guard_rows}
    all_guard_fragile = guard[("ALL", "fragile")]
    all_guard_recovered = guard[("ALL", "recovered")]
    got_guard = (
        round(float(all_guard_fragile["atr_normalized_q90_median"]), 2),
        round(float(all_guard_recovered["atr_normalized_q90_median"]), 2),
        round(float(all_guard_fragile["smpr_delta0_median"]), 2),
        round(float(all_guard_recovered["smpr_delta0_median"]), 2),
        round(float(all_guard_fragile["fixed_pool_top1_flip_median"]), 2),
        round(float(all_guard_recovered["fixed_pool_top1_flip_median"]), 2),
    )
    if got_guard != (0.84, 0.09, 0.86, 1.00, 0.54, 0.04):
        fail(f"joint guard-side validation changed: got {got_guard}")

    for rel in (
        "assets/paper1_figs/fig_radius_margin_interval_overlay.png",
        "assets/paper1_figs/fig_radius_margin_overlap.png",
        "assets/paper1_figs/fig_endpoint_atr_smpr.png",
        "assets/paper1_figs/fig_fixed_pool_event_rates.png",
        "assets/paper1_figs/fig_gaussian_sensitivity_mechanism.png",
    ):
        path = ROOT / rel
        if path.stat().st_size < 10_000:
            fail(f"paper1 figure looks too small: {rel}")


def main() -> int:
    checks = [
        ("artifacts", check_artifacts),
        ("forbidden text", check_forbidden_text),
        ("appendix internal heading gate", check_appendix_internal_heading_gate),
        ("canonical json", check_canonical_json),
        ("pldm canonical json", check_pldm_canonical_json),
        ("canonical diagnostics json", check_canonical_diagnostics_json),
        ("pldm diagnostics json", check_pldm_diagnostics_json),
        ("pldm full diagnostics json", check_pldm_full_diagnostics_json),
        ("acpc phase0 diagnostics json", check_acpc_phase0_diagnostics_json),
        ("blur baselines json", check_blur_baselines_json),
        ("acpc basin json", check_acpc_basin_json),
        ("pldm acpc basin json", check_pldm_acpc_basin_json),
        ("compressed metrics summary json", check_compressed_metrics_summary_json),
        ("radius-margin certificate outputs", check_radius_margin_certificate_outputs),
        ("unseen ATR/SMPR summary json", check_unseen_atr_smpr_summary_json),
        ("target-view closed-loop json", check_target_view_closed_loop_summary_json),
        ("training-seed Gaussian lockbox json", check_training_seed_gaussian_lockbox_json),
        ("three-seed Gaussian sweep summary json", check_three_seed_gaussian_sweep_summary_json),
        ("three-seed diagnostic validation json", check_three_seed_diagnostic_validation_json),
        ("selector-baseline audit json", check_selector_baseline_audit_json),
        ("selector plateau audit json", check_selector_plateau_audit_json),
        ("residual diagnostic audit json", check_residual_diagnostic_audit_json),
        ("selector incremental audit json", check_selector_incremental_audit_json),
        ("margin-conditioned flip json", check_margin_flip_curve_json),
        ("task-state proxy margin pass-rate json", check_semantic_margin_passrate_json),
        ("local task-feature margin json", check_semantic_local_margin_json),
        ("task-grounded semantic margin json", check_semantic_task_grounded_margin_json),
        ("CEM trace audit json", check_cem_trace_audit_json),
        ("prospective validation summary json", check_prospective_validation_summary_json),
        ("prospective ATR/SMPR validation", check_prospective_atr_smpr_validation),
        ("external baselines json", check_external_baselines_json),
        ("pldm correlations json", check_pldm_correlations_json),
        ("partial-corr bootstrap json", check_partial_corr_bootstrap_json),
        ("published correlations", check_published_correlations),
    ]
    for name, fn in checks:
        try:
            fn()
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            return 1
        print(f"[OK] {name}")
    print("[OK] paper1 release consistency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
