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
    ROOT / "paper_invariance_resolution_tradeoff.md",
    ROOT / "paper_invariance_resolution_tradeoff_zh.md",
    ROOT / "paper1" / "main.tex",
    ROOT / "tools" / "paper1_figs.py",
    ROOT / "paper1" / "references.bib",
    ROOT / "DATA_MANIFEST.md",
    ROOT / "canonical_diagnostics_20260517.json",
]

REQUIRED_ARTIFACTS = [
    ROOT / "canonical_evals_20260517.json",
    ROOT / "canonical_evals_20260517.schema.json",
    ROOT / "canonical_diagnostics_20260517.json",
    ROOT / "canonical_diagnostics_20260517.schema.json",
    ROOT / "DATA_MANIFEST.md",
]

FORBIDDEN_SNIPPETS = [
    "either 1 seed × 300",
    "single-seed",
    "mixed convention",
    "n=18",
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
    "0.001",
    "0.002",
    "0.003",
    "0.004",
    "0.005",
    "0.006",
    "0.007",
    "0.008",
}
REQUIRED_METRICS = {"clean", "pixels_goal_std0.05", "pixels_goal_std0.08"}
REQUIRED_DIAG_TASKS = EXPECTED_TASKS
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
    path = ROOT / "canonical_evals_20260517.json"
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


def check_canonical_diagnostics_json() -> None:
    path = ROOT / "canonical_diagnostics_20260517.json"
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


def main() -> int:
    checks = [
        ("artifacts", check_artifacts),
        ("forbidden text", check_forbidden_text),
        ("canonical json", check_canonical_json),
        ("canonical diagnostics json", check_canonical_diagnostics_json),
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
