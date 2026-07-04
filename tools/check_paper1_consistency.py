#!/usr/bin/env python3
"""Release consistency checks for Paper 1.

Usage:
    python -m tools.check_paper1_consistency
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RELEASE_FILES = [
    ROOT / "paper1" / "main.tex",
    ROOT / "tools" / "paper1_figs.py",
    ROOT / "tools" / "paper1_margin_flip_curve.py",
    ROOT / "tools" / "README_paper1.md",
    ROOT / "paper1" / "references.bib",
    ROOT / "DATA_MANIFEST.md",
    ROOT / "assets" / "paper1_data" / "selective_contraction_fullseq_branch.md",
    ROOT / "assets" / "paper1_data" / "canonical_diagnostics_20260517.json",
    ROOT / "assets" / "paper1_data" / "canonical_external_baselines_20260520.json",
    ROOT / "assets" / "paper1_data" / "canonical_blur_baselines_20260523.json",
    ROOT / "assets" / "paper1_data" / "acpc_basin_diagnostics.json",
    ROOT / "assets" / "paper1_data" / "acpc_basin_diagnostics_pldm.json",
    ROOT / "assets" / "paper1_data" / "canonical_full_diagnostics_pldm_20260523.json",
    ROOT / "assets" / "paper1_data" / "partial_corr_bootstrap_20260523.json",
    ROOT / "assets" / "paper1_data" / "acpc_phase0_clean_goal_seed9101.json",
    ROOT / "assets" / "paper1_data" / "target_view_closed_loop_summary.json",
    ROOT / "assets" / "paper1_data" / "training_seed_gaussian_lockbox.json",
    ROOT / "assets" / "paper1_data" / "prospective_validation_summary.json",
    ROOT / "assets" / "paper1_data" / "unseen_phase0_acpc_fullstress.json",
    ROOT / "assets" / "paper1_data" / "acpc_phase0_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "three_seed_diagnostic_validation.json",
    ROOT / "assets" / "paper1_data" / "selector_baseline_audit_20260704.json",
    ROOT / "assets" / "paper1_data" / "selector_baseline_audit_20260704.md",
    ROOT / "assets" / "paper1_data" / "margin_flip_curve_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_margin_passrate_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_local_margin_lewm_three_seed.json",
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
    ROOT / "assets" / "paper1_data" / "margin_flip_curve_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_margin_passrate_lewm_three_seed.json",
    ROOT / "assets" / "paper1_data" / "semantic_margin_passrate_lewm_three_seed.md",
    ROOT / "assets" / "paper1_data" / "semantic_local_margin_lewm_three_seed.json",
    ROOT / "DATA_MANIFEST.md",
]


REQUIRED_MAIN_TEXT_SNIPPETS = [
    "local task-feature proxy margin pass-rate",
    "Cube seed 3072",
    "regret $8.33$ pp",
    "We do not claim superiority over DrQ-style augmentation, TD-MPC2, DreamerV3, or robust MPC methods",
    "not an oracle hand-labeled semantic proof",
    "closest 35\% state-distance neighborhood",
    "Sampled-pool ACPC stability",
    "ACPC-tail and clean-margin terms",
    "action-conditioned encoder--predictor sensitivity",
    "not a convergence theorem for adaptive multi-round CEM",
    "Hoeffding gives",
    "Proofs and calibration for ACPC diagnostics",
    "Finite-sample tail calibration for paired diagnostics",
    "diagnostic calibration rather than as closed-loop control guarantees",
    "selector-baseline audit",
    "comparable to fixed $\\stdmax{}=0.08$",
    "Bounded unseen-stressor scope check",
    "not a universal transfer claim",
    "universal cross-perturbation robustness claim",
    "ACPC rollout readout $R_F$",
    "margin-conditioned action-flip audit",
    "top clean-margin quartile",
]

FORBIDDEN_SNIPPETS = [
    "either 1 seed × 300",
    "single-seed",
    "mixed convention",
    "canonical_evals_20260508",
    "summary.txt",
    "clean_300",
    "ρ ≈ −0.8",
    "ρ ≈ −0.3",
    "noise-best",
    "within-protocol",
    "Within-protocol",
    "LeWM + SWM",
    "PushT n=18 scatter",
    "best (σ*=",
    "same-state encoder/predictor basins shrink",
    "same-state basin shrinkage is not monotone",
    "smaller same-state perturbation basin",
    "high-D basin support",
    "H8 predictor",
    "H=8 predictor",
    "8-step predictor basin",
    "prediction radius",
    "selective-consistency certificate",
    "low-cliff external baseline",
    "The external baseline is therefore",
    "strong simple endpoint baseline",
    "comparable strong baseline",
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
    for snippet in REQUIRED_MAIN_TEXT_SNIPPETS:
        if snippet not in main_tex:
            hits.append(f"paper1/main.tex missing required scope-boundary snippet: {snippet!r}")
    if hits:
        fail("\n".join(hits))


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


def check_acpc_basin_full_grid_table() -> None:
    main_tex = ROOT / "paper1" / "main.tex"
    if not main_tex.exists():
        return
    tex = main_tex.read_text(encoding="utf-8")
    marker = r"\label{tab:acpc-basin-full-grid}"
    start = tex.find(marker)
    if start < 0:
        fail("main.tex is missing tab:acpc-basin-full-grid")
    end = tex.find(r"\end{tabular}", start)
    if end < 0:
        fail("tab:acpc-basin-full-grid does not close its tabular")
    table = tex[start:end]

    row_re = re.compile(
        r"^(TwoRoom|PushT|Reacher|Cube)\s*&\s*"
        r"([0-9.]+)\s*&\s*"
        r"([-0-9.]+)\s*&\s*"
        r"([-0-9.]+)\s*&\s*"
        r"([-0-9.]+)\s*&\s*"
        r"([-0-9.]+)\s*&\s*"
        r"([-0-9.]+)\s*&\s*"
        r"([-0-9.]+)\s*&\s*"
        r"([^\\\\]*)\\\\"
    )
    parsed: dict[tuple[str, str], dict[str, object]] = {}
    for raw_line in table.splitlines():
        line = raw_line.strip()
        match = row_re.match(line)
        if not match:
            continue
        task, std_display, unpert, obs, drop, radius_e, radius_f, ratio, note = match.groups()
        std_key = "0.0" if std_display == "0" else std_display
        key = (task, std_key)
        if key in parsed:
            fail(f"duplicate LaTeX full-grid row: {task}/{std_key}")
        parsed[key] = {
            "unpert": float(unpert),
            "obs": float(obs),
            "drop": float(drop),
            "radius_e": float(radius_e),
            "radius_f": float(radius_f),
            "ratio": float(ratio),
            "note": note.strip(),
        }

    expected_keys = {(task, std) for task in EXPECTED_TASKS for std in EXPECTED_CONFIGS}
    if set(parsed) != expected_keys:
        fail(
            "LaTeX ACPC full-grid coverage mismatch: "
            f"missing={sorted(expected_keys - set(parsed))}, extra={sorted(set(parsed) - expected_keys)}"
        )

    basin = json.loads((ROOT / "assets" / "paper1_data" / "acpc_basin_diagnostics.json").read_text(encoding="utf-8"))
    evals = json.loads((ROOT / "assets" / "paper1_data" / "canonical_evals_20260517.json").read_text(encoding="utf-8"))
    rows = {(row["task"], row["std_key"]): row for row in basin["rows"]}
    obs_best = {
        task: max(EXPECTED_CONFIGS, key=lambda std: evals[task][std]["metrics"]["pixels_std0.08"]["mean"])
        for task in EXPECTED_TASKS
    }

    def check_display(label: str, got: float, want: float, digits: int) -> None:
        rounded = round(float(want), digits)
        if got != rounded:
            fail(f"{label} mismatch: table has {got}, artifact rounds to {rounded}")

    for key, shown in parsed.items():
        task, std_key = key
        row = rows[key]
        eval_cell = evals[task][std_key]["metrics"]
        check_display(f"{task}/{std_key}/unpert", shown["unpert"], eval_cell["clean"]["mean"], 2)
        check_display(f"{task}/{std_key}/obs0.08", shown["obs"], eval_cell["pixels_std0.08"]["mean"], 2)
        check_display(f"{task}/{std_key}/drop", shown["drop"], row["corruption_drop"], 2)
        check_display(f"{task}/{std_key}/R_E", shown["radius_e"], row["encoder_view_pair_l2_norm_by_nn"], 3)
        check_display(f"{task}/{std_key}/R_F", shown["radius_f"], row["pred_view_pair_l2_norm_by_transition"], 3)
        check_display(f"{task}/{std_key}/R_F/R_E", shown["ratio"], row["basin_contraction_pair_norm"], 3)

        expected_note = "base" if std_key == "0.0" else ("obs0.08 ref" if std_key == obs_best[task] else "")
        if shown["note"] != expected_note:
            fail(f"{task}/{std_key}/note mismatch: table has {shown['note']!r}, want {expected_note!r}")


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


def main() -> int:
    checks = [
        ("artifacts", check_artifacts),
        ("forbidden text", check_forbidden_text),
        ("canonical json", check_canonical_json),
        ("pldm canonical json", check_pldm_canonical_json),
        ("canonical diagnostics json", check_canonical_diagnostics_json),
        ("pldm diagnostics json", check_pldm_diagnostics_json),
        ("pldm full diagnostics json", check_pldm_full_diagnostics_json),
        ("acpc phase0 diagnostics json", check_acpc_phase0_diagnostics_json),
        ("blur baselines json", check_blur_baselines_json),
        ("acpc basin json", check_acpc_basin_json),
        ("acpc basin full-grid table", check_acpc_basin_full_grid_table),
        ("pldm acpc basin json", check_pldm_acpc_basin_json),
        ("target-view closed-loop json", check_target_view_closed_loop_summary_json),
        ("training-seed Gaussian lockbox json", check_training_seed_gaussian_lockbox_json),
        ("three-seed diagnostic validation json", check_three_seed_diagnostic_validation_json),
        ("selector-baseline audit json", check_selector_baseline_audit_json),
        ("margin-conditioned flip json", check_margin_flip_curve_json),
        ("task-state proxy margin pass-rate json", check_semantic_margin_passrate_json),
        ("local task-feature margin json", check_semantic_local_margin_json),
        ("prospective validation summary json", check_prospective_validation_summary_json),
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
