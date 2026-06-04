#!/usr/bin/env python3
"""Aggregate Paper 1 perturbed-input -> original-target LeWM sweep.

This artifact is the paper-facing target-view ablation branch.  It reads the
completed ``lewm_baseline_unperturbed_target_noise_*`` checkpoints, verifies
that all expected eval and diagnostic files exist, and emits a compact JSON/MD
summary for Paper 1 and the release consistency checker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_OUT_JSON = DEFAULT_DATA_DIR / "unperturbed_target_mainline_20260604.json"
DEFAULT_OUT_MD = DEFAULT_DATA_DIR / "unperturbed_target_mainline_20260604.md"
DEFAULT_CANONICAL_EVALS = DEFAULT_DATA_DIR / "canonical_evals_20260517.json"

EXPECTED_STD_KEYS = tuple(f"0.0{i}" for i in range(1, 9))
EXPECTED_EVAL_GROUPS = ("origin", "pixels_std0.03", "pixels_std0.05", "pixels_std0.08")
REQUIRED_DIAGNOSTICS = {
    "noise_robust_radius_std",
    "noise_angle_slope_deg_per_std",
    "clean_nn_cos_dist_median",
    "clean_effective_rank",
    "cka_linear_at_max_std",
    "predictor_rollout_T8_l2",
    "predictor_target_to_nn_cos_ratio_at_max_std",
    "transition_resolution_ratio_cos",
    "transition_resolution_ratio_l2",
    "id_probe_r2",
    "latent_robust_radius_z",
    "latent_predictor_rollout_T8_l2_history",
    "action_mean_pred_shift_norm",
    "action_perturb_pred_shift_corr",
}


@dataclass(frozen=True)
class TaskSpec:
    task: str
    ckpt_root_rel: str
    subdir_prefix: str


TASKS = (
    TaskSpec("TwoRoom", "lewm-tworooms/ckpt", "tworoom"),
    TaskSpec("PushT", "lewm-pusht/ckpt", "pusht"),
    TaskSpec("Reacher", "lewm-reacher/ckpt", "reacher"),
    TaskSpec("Cube", "lewm-cube/ckpt", "cube"),
)


def _stablewm_root() -> Path:
    env = os.environ.get("STABLEWM_HOME")
    if env:
        return Path(env).expanduser()
    return Path("/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll")


def _std_to_fragment(std_key: str) -> str:
    value = int(round(float(std_key) * 100))
    return f"0to{value:03d}"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, digits: int = 2) -> str:
    if not _finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _metric_summary(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals),
        "values": vals,
    }


def _parse_values(raw: str) -> list[float]:
    return [float(v) for v in str(raw).split(";") if v != ""]


def _read_eval_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing eval summary: {path}")

    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["group"]
            values = _parse_values(row["values"])
            summary = _metric_summary(values)
            stored_mean = float(row["mean"])
            stored_std = float(row["std"])
            if not math.isclose(summary["mean"], stored_mean, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"{path}: mean mismatch for {group}: {summary['mean']} != {stored_mean}")
            if not math.isclose(summary["std"], stored_std, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"{path}: std mismatch for {group}: {summary['std']} != {stored_std}")
            out[group] = summary
    return out


def _read_diagnostics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing diagnostics summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"{path}: expected one diagnostics row, got {len(payload)}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: diagnostics payload is not an object")
    return payload


def _checkpoint_files(run_dir: Path, subdir: str) -> list[Path]:
    return sorted(run_dir.glob(f"{subdir}_epoch_*_object.ckpt"))


def _epoch10_file(run_dir: Path, subdir: str) -> Path:
    path = run_dir / f"{subdir}_epoch_10_object.ckpt"
    if not path.exists():
        raise FileNotFoundError(f"missing epoch-10 checkpoint: {path}")
    return path


def _read_canonical_baseline(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for spec in TASKS:
        base = payload[spec.task]["0.0"]["metrics"]
        out[spec.task] = {
            "origin_success": base["clean"],
            "pixels_std0.08_success": base["pixels_std0.08"],
        }
    return out


def _read_full_sequence_best(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for spec in TASKS:
        rows = []
        for std_key, entry in payload[spec.task].items():
            if std_key == "0.0":
                continue
            metrics = entry["metrics"]
            rows.append(
                {
                    "std_key": std_key,
                    "subdir": entry["subdir"],
                    "origin_success": metrics["clean"],
                    "pixels_std0.08_success": metrics["pixels_std0.08"],
                }
            )
        out[spec.task] = max(
            rows,
            key=lambda r: (
                float(r["pixels_std0.08_success"]["mean"]),
                float(r["origin_success"]["mean"]),
            ),
        )
    return out


def build_payload(
    *,
    stablewm_root: Path,
    canonical_evals_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for spec in TASKS:
        task_root = stablewm_root / spec.ckpt_root_rel
        for std_key in EXPECTED_STD_KEYS:
            fragment = _std_to_fragment(std_key)
            subdir = f"{spec.subdir_prefix}_lewm_baseline_unperturbed_target_noise_{fragment}_p1"
            run_dir = task_root / subdir
            if not run_dir.exists():
                missing.append(f"{spec.task}/{std_key}: missing run dir {run_dir}")
                continue

            try:
                ckpts = _checkpoint_files(run_dir, subdir)
                model_file = _epoch10_file(run_dir, subdir)
                evals = _read_eval_summary(run_dir / "eval_results" / "eval_summary.csv")
                diagnostics = _read_diagnostics(
                    run_dir / "eval_results" / "diagnostics" / "diagnostics_summary.json"
                )
            except (FileNotFoundError, ValueError) as exc:
                missing.append(f"{spec.task}/{std_key}: {exc}")
                continue

            group_missing = sorted(set(EXPECTED_EVAL_GROUPS) - set(evals))
            diag_missing = sorted(REQUIRED_DIAGNOSTICS - set(diagnostics))
            if group_missing:
                missing.append(f"{spec.task}/{std_key}: missing eval groups {group_missing}")
            if diag_missing:
                missing.append(f"{spec.task}/{std_key}: missing diagnostics {diag_missing}")
            if len(ckpts) != 10:
                missing.append(f"{spec.task}/{std_key}: expected 10 epoch checkpoints, got {len(ckpts)}")

            row: dict[str, Any] = {
                "task": spec.task,
                "std_key": std_key,
                "target_view_branch": "perturbed_input_original_target",
                "subdir": subdir,
                "path": str(run_dir),
                "model_file": str(model_file),
                "checkpoint_count": len(ckpts),
                "eval": {group: evals[group] for group in EXPECTED_EVAL_GROUPS if group in evals},
                "diagnostics_summary": {
                    key: diagnostics[key]
                    for key in sorted(REQUIRED_DIAGNOSTICS)
                    if key in diagnostics
                },
            }
            row["origin_success"] = row["eval"].get("origin", {}).get("mean")
            row["pixels_std0.08_success"] = row["eval"].get("pixels_std0.08", {}).get("mean")
            row["pixels_std0.08_drop"] = (
                row["origin_success"] - row["pixels_std0.08_success"]
                if _finite(row["origin_success"]) and _finite(row["pixels_std0.08_success"])
                else float("nan")
            )
            rows.append(row)

    if missing:
        return {
            "metadata": {
                "schema_version": "paper1-unperturbed-target-mainline-0.1",
                "status": "incomplete",
                "stablewm_root": str(stablewm_root),
            },
            "missing": missing,
            "rows": rows,
        }

    baseline = _read_canonical_baseline(canonical_evals_path)
    fullseq_best = _read_full_sequence_best(canonical_evals_path)
    by_task: dict[str, list[dict[str, Any]]] = {spec.task: [] for spec in TASKS}
    for row in rows:
        by_task[row["task"]].append(row)

    best_by_task: dict[str, dict[str, Any]] = {}
    comparison: dict[str, dict[str, Any]] = {}
    for spec in TASKS:
        best = max(
            by_task[spec.task],
            key=lambda r: (
                float(r["pixels_std0.08_success"]),
                float(r["origin_success"]),
            ),
        )
        best_summary = {
            "std_key": best["std_key"],
            "subdir": best["subdir"],
            "origin_success": best["eval"]["origin"],
            "pixels_std0.08_success": best["eval"]["pixels_std0.08"],
            "pixels_std0.08_drop": best["pixels_std0.08_drop"],
            "diagnostics_summary": best["diagnostics_summary"],
        }
        best_by_task[spec.task] = best_summary

        fullseq = fullseq_best[spec.task]
        comparison[spec.task] = {
            "no_perturb_training": baseline[spec.task],
            "full_sequence_perturbed_target_best": fullseq,
            "perturbed_input_original_target_best": best_summary,
            "delta_vs_full_sequence_px08": (
                best["eval"]["pixels_std0.08"]["mean"]
                - fullseq["pixels_std0.08_success"]["mean"]
            ),
            "delta_vs_full_sequence_origin": (
                best["eval"]["origin"]["mean"] - fullseq["origin_success"]["mean"]
            ),
            "delta_vs_no_perturb_px08": (
                best["eval"]["pixels_std0.08"]["mean"]
                - baseline[spec.task]["pixels_std0.08_success"]["mean"]
            ),
        }

    return {
        "metadata": {
            "schema_version": "paper1-unperturbed-target-mainline-0.1",
            "status": "ok",
            "source_pattern": "lewm_baseline_unperturbed_target_noise_0to00{1..8}_p1",
            "stablewm_root": str(stablewm_root),
            "canonical_full_sequence_source": str(canonical_evals_path),
            "tasks": [spec.task for spec in TASKS],
            "std_keys": list(EXPECTED_STD_KEYS),
            "eval_groups": list(EXPECTED_EVAL_GROUPS),
            "diagnostics_fields": sorted(REQUIRED_DIAGNOSTICS),
            "interpretation": (
                "Target-view ablation branch: train with perturbed input/history while "
                "keeping the prediction target at the original unperturbed view. "
                "Evaluation is observation-pixel perturbation with an unperturbed goal."
            ),
        },
        "rows": rows,
        "best_by_task": best_by_task,
        "comparison_to_full_sequence": comparison,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Perturbed-Input -> Original-Target Mainline Sweep",
        "",
        "Scope: LeWM target-view ablation, 4 tasks x 8 nonzero train-time perturbation levels.",
        "",
    ]
    meta = payload.get("metadata", {})
    if meta.get("status") != "ok":
        lines.extend(["Status: incomplete.", ""])
        for item in payload.get("missing", []):
            lines.append(f"- {item}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "| Task | branch | best std | origin eval | pixels 0.08 | delta vs full-seq px0.08 | delta vs base px0.08 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task, comp in payload["comparison_to_full_sequence"].items():
        branch = comp["perturbed_input_original_target_best"]
        lines.append(
            "| {task} | perturbed input -> original target | {std} | {origin} | {px08} | {d_full} | {d_base} |".format(
                task=task,
                std=branch["std_key"],
                origin=_fmt(branch["origin_success"]["mean"]),
                px08=_fmt(branch["pixels_std0.08_success"]["mean"]),
                d_full=_fmt(comp["delta_vs_full_sequence_px08"]),
                d_base=_fmt(comp["delta_vs_no_perturb_px08"]),
            )
        )
    lines.extend(
        [
            "",
            "Comparison rows use the best pixels 0.08 checkpoint within each branch. "
            "The full-sequence branch is read from the canonical LeWM sweep JSON.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stablewm-root", type=Path, default=_stablewm_root())
    parser.add_argument("--canonical-evals", type=Path, default=DEFAULT_CANONICAL_EVALS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(stablewm_root=args.stablewm_root, canonical_evals_path=args.canonical_evals)
    _write_json(args.out_json, payload)
    write_markdown(args.out_md, payload)

    status = payload.get("metadata", {}).get("status")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    if status != "ok":
        print("status=incomplete")
        for item in payload.get("missing", []):
            print(f"  - {item}")
        return 1

    print("status=ok")
    for task, comp in payload["comparison_to_full_sequence"].items():
        branch = comp["perturbed_input_original_target_best"]
        print(
            f"{task}: std={branch['std_key']} origin={branch['origin_success']['mean']:.2f} "
            f"px0.08={branch['pixels_std0.08_success']['mean']:.2f} "
            f"delta_vs_fullseq={comp['delta_vs_full_sequence_px08']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
