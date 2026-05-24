#!/usr/bin/env python3
"""Release consistency checks for Paper 1.

Usage:
    python -m tools.check_paper1_consistency
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RELEASE_FILES = [
    ROOT / "paper1" / "main.tex",
    ROOT / "tools" / "paper1_figs.py",
    ROOT / "paper1" / "references.bib",
    ROOT / "DATA_MANIFEST.md",
    ROOT / "assets" / "paper1_data" / "canonical_diagnostics_20260517.json",
    ROOT / "assets" / "paper1_data" / "canonical_external_baselines_20260520.json",
    ROOT / "assets" / "paper1_data" / "canonical_blur_baselines_20260523.json",
    ROOT / "assets" / "paper1_data" / "canonical_full_diagnostics_pldm_20260523.json",
    ROOT / "assets" / "paper1_data" / "partial_corr_bootstrap_20260523.json",
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
    ROOT / "assets" / "paper1_data" / "partial_corr_bootstrap_20260523.json",
    ROOT / "DATA_MANIFEST.md",
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
REQUIRED_METRICS = {"clean", "pixels_goal_std0.05", "pixels_goal_std0.08"}
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
EXPECTED_BOOTSTRAP_SCOPES = {"within_lewm", "within_pldm", "joint"}
EXPECTED_BOOTSTRAP_METRICS = {"frag", "drift"}
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
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                hits.append(f"{path.relative_to(ROOT)} contains forbidden snippet: {snippet!r}")
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

    required_eval = {"clean", "pixels_goal_std0.05", "pixels_goal_std0.08"}
    evaluation = entry.get("evaluation", {})
    missing = required_eval - set(evaluation)
    if missing:
        fail(f"PLDM external baseline missing eval conditions: {sorted(missing)}")
    for metric_name, summary in evaluation.items():
        check_metric_summary("PushT/PLDM_clean_trained", "external", metric_name, summary)

    clean = evaluation["clean"]["mean"]
    px08 = evaluation["pixels_goal_std0.08"]["mean"]
    if round(clean - px08, 2) != 65.33:
        fail(f"unexpected PLDM clean-to-px+goal0.08 drop: {clean - px08}")


def check_pldm_correlations_json() -> None:
    path = ROOT / "assets" / "paper1_data" / "cross_method_corr_pldm_20260522.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    if set(data) != EXPECTED_TASKS:
        fail(f"PLDM correlation tasks mismatch: expected {sorted(EXPECTED_TASKS)}, got {sorted(data)}")

    expected_push = {
        ("within_pldm", "partial_metric_drop_on_std"): -0.14,
        ("joint", "partial_metric_drop_on_std_method"): 0.11,
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

    # Values quoted in main.tex C4 / Table 7 / Appendix F. These are rounded
    # checks, not a substitute for rerunning the bootstrap.
    _check_bootstrap_cell(
        data, "PushT", "within_lewm", "frag", "partial_metric_clean_on_std",
        -0.59, (-0.97, -0.10),
    )
    _check_bootstrap_cell(
        data, "PushT", "within_lewm", "frag", "partial_metric_px08_on_std",
        -0.41, (-0.77, 0.00),
    )
    _check_bootstrap_cell(
        data, "PushT", "within_lewm", "frag", "partial_metric_drop_on_std",
        0.06, (-0.00, 0.25),
    )
    _check_bootstrap_cell(
        data, "PushT", "within_pldm", "frag", "partial_metric_drop_on_std",
        -0.14, (-1.00, 0.87),
    )
    _check_bootstrap_cell(
        data, "PushT", "joint", "frag", "partial_metric_drop_on_std_method",
        0.11, (-0.54, 0.71),
    )
    _check_bootstrap_cell(
        data, "Reacher", "within_lewm", "drift", "partial_metric_drop_on_std",
        0.79, (0.00, 1.00),
    )


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
            float(evals[task][std_key]["metrics"]["pixels_goal_std0.08"]["mean"])
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
        float(evals["PushT"][std_key]["metrics"]["pixels_goal_std0.08"]["mean"])
        for std_key in push_keys
    ]
    drop = [c - p for c, p in zip(clean, px08)]
    table5 = published["table5_pusht_fragility_metric"]["spearman"]
    recomputed = {
        "rho_std_max_metric": round2(spearman(z, fragility)),
        "rho_std_max_clean": round2(spearman(z, clean)),
        "rho_std_max_pixels_goal_std0.08": round2(spearman(z, px08)),
        "rho_std_max_ood_drop": round2(spearman(z, drop)),
        "rho_metric_clean_unconditional": round2(spearman(fragility, clean)),
        "rho_metric_clean_partial_given_std_max": round2(partial_spearman(fragility, clean, z)),
        "rho_metric_pixels_goal_std0.08_unconditional": round2(spearman(fragility, px08)),
        "rho_metric_pixels_goal_std0.08_partial_given_std_max": round2(
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
        ("blur baselines json", check_blur_baselines_json),
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
