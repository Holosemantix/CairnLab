"""Validation/protocol remediation summaries for Paper 1.

This script builds a compact artifact for the top-conference remediation pass.
It uses only released JSON/Markdown summaries and does not load models,
checkpoints, or datasets.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "assets" / "paper1_data"
TRAINING_SEED_LOCKBOX = DATA_DIR / "training_seed_gaussian_lockbox.json"
UNSEEN_PHASE0 = DATA_DIR / "unseen_phase0_acpc_subset.json"
NO_RETRAIN_AUDIT = DATA_DIR / "no_retrain_diagnostic_audit.json"
DEFAULT_OUT_JSON = DATA_DIR / "prospective_validation_summary.json"
DEFAULT_OUT_MD = DATA_DIR / "prospective_validation_summary.md"

SIGNED_METRICS = [
    ("ACPC-H/trans. delta", "delta_acpc_h_norm_by_transition", -1.0, "lower is better"),
    ("PCC delta", "delta_pcc_abs_median", -1.0, "lower is better"),
    ("CRA delta", "delta_cra_spearman_mean", 1.0, "higher is better"),
    ("MAF delta", "delta_maf_flip_rate", -1.0, "lower is better"),
]
SEMANTIC_GUARD_PROTOCOL = [
    {
        "task": "PushT",
        "semantic_factor": "T-block pose/contact relative to pusher",
        "available_source": "dataset state column is configured for PushT analysis",
        "probe_rule": "different-state pair must cross a pose/contact threshold while same-state clean/noisy views share state",
        "release_status": "protocol frozen; state-margin run still required for a result table",
    },
    {
        "task": "TwoRoom",
        "semantic_factor": "room/doorway/topology and target-region relation",
        "available_source": "derive from trajectory position/proprio and map topology",
        "probe_rule": "different-state pair must differ in room or doorway side under comparable visual nuisance",
        "release_status": "protocol frozen; task-topology extraction still required for a result table",
    },
    {
        "task": "Reacher",
        "semantic_factor": "joint/target geometry and end-effector-to-target relation",
        "available_source": "qpos/goal_qpos are used by eval set-state callables",
        "probe_rule": "different-state pair must differ in target quadrant or end-effector-target distance bin",
        "release_status": "protocol frozen; state-margin run still required for a result table",
    },
    {
        "task": "Cube",
        "semantic_factor": "cube pose and gripper-object/goal relation",
        "available_source": "qpos plus goal block position/quaternion are used by eval callables",
        "probe_rule": "different-state pair must differ in object pose/goal relation beyond tolerance",
        "release_status": "protocol frozen; state-margin run still required for a result table",
    },
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _rankdata(values: Sequence[float]) -> list[float]:
    pairs = sorted((float(v), i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    j = 0
    while j < len(pairs):
        k = j + 1
        while k < len(pairs) and pairs[k][0] == pairs[j][0]:
            k += 1
        rank = 0.5 * (j + k - 1)
        for _, idx in pairs[j:k]:
            ranks[idx] = rank
        j = k
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_rankdata(xs), _rankdata(ys))


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _heldout_metric_rows(unseen: dict) -> tuple[list[dict], dict]:
    rows = unseen["rows"]
    stress_delta = [float(row["eval"]["stress_success_delta"]) for row in rows]
    drop_improvement = [float(row["eval"]["drop_improvement"]) for row in rows]
    out = []
    signed_values = []
    for label, key, sign, direction in SIGNED_METRICS:
        values = [sign * float(row["diagnostics"][key]) for row in rows]
        signed_values.append(values)
        out.append(
            {
                "metric": label,
                "direction": direction,
                "n": len(rows),
                "spearman_vs_stress_success_delta": _spearman(values, stress_delta),
                "pearson_vs_stress_success_delta": _pearson(values, stress_delta),
                "spearman_vs_drop_improvement": _spearman(values, drop_improvement),
                "pearson_vs_drop_improvement": _pearson(values, drop_improvement),
            }
        )

    composite_scores = [0.0] * len(rows)
    for values in signed_values:
        ranks = _rankdata(values)
        # Higher signed values are better; negative rank makes lower composite score better.
        for i, rank in enumerate(ranks):
            composite_scores[i] -= rank
    out.append(
        {
            "metric": "Composite signed-rank rule",
            "direction": "aggregate rank over ACPC/PCC/CRA/MAF directions",
            "n": len(rows),
            "spearman_vs_stress_success_delta": _spearman([-s for s in composite_scores], stress_delta),
            "pearson_vs_stress_success_delta": _pearson([-s for s in composite_scores], stress_delta),
            "spearman_vs_drop_improvement": _spearman([-s for s in composite_scores], drop_improvement),
            "pearson_vs_drop_improvement": _pearson([-s for s in composite_scores], drop_improvement),
        }
    )
    top_diag = set(sorted(range(len(rows)), key=lambda i: composite_scores[i])[:4])
    top_stress = set(sorted(range(len(rows)), key=lambda i: stress_delta[i], reverse=True)[:4])
    top_drop = set(sorted(range(len(rows)), key=lambda i: drop_improvement[i], reverse=True)[:4])
    topk = {
        "k": 4,
        "topk_by_composite": [rows[i]["case"] + f":seed{rows[i]['seed']}" for i in sorted(top_diag)],
        "stress_success_delta_topk_hit_count": len(top_diag & top_stress),
        "drop_improvement_topk_hit_count": len(top_diag & top_drop),
        "stress_success_delta_topk_total": len(top_stress),
        "drop_improvement_topk_total": len(top_drop),
    }
    return out, topk


def build_payload() -> dict:
    training = _load(TRAINING_SEED_LOCKBOX)
    unseen = _load(UNSEEN_PHASE0)
    no_retrain = _load(NO_RETRAIN_AUDIT)
    heldout_rows, topk = _heldout_metric_rows(unseen)
    return {
        "metadata": {
            "schema_version": "paper1-validation-remediation-0.1",
            "source_artifacts": [
                str(TRAINING_SEED_LOCKBOX.relative_to(ROOT)),
                str(UNSEEN_PHASE0.relative_to(ROOT)),
                str(NO_RETRAIN_AUDIT.relative_to(ROOT)),
            ],
            "scope": "No model loading or retraining. Summarizes completed three-training-seed Gaussian behavior, a held-out training-seed/unseen-perturbation validation slice, and the semantic-guard protocol ledger.",
        },
        "three_training_seed_gaussian_summary": training["task_summary_rows"],
        "heldout_unseen_validation": {
            "split": "training seeds 3073/3074; unseen perturbation families gaussian_blur and resize; fixed std_max comparison 0.0 vs 0.08",
            "n_rows": len(unseen["rows"]),
            "metric_rows": heldout_rows,
            "topk_summary": topk,
            "summary_by_task": unseen["summary_by_task"],
        },
        "existing_full_grid_frozen_rule_audit": no_retrain["summary"],
        "semantic_discriminability_protocol": SEMANTIC_GUARD_PROTOCOL,
        "remaining_validation_work": [
            "Run the fixed ACPC/PCC/CRA/MAF rule on full held-out training-seed checkpoint grids, not only the current unseen slice.",
            "Run the task-semantic state-margin probes defined in this artifact and report pass rates before claiming semantic discriminability results.",
            "Use the three-training-seed Gaussian table as the primary behavior statistic; keep evaluation-seed variance as a secondary decomposition.",
        ],
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Prospective Validation Remediation Summary",
        "",
        "This artifact separates completed validation evidence from the protocol pieces that are now frozen but still require state-margin or full held-out-grid runs.",
        "",
        "## Held-out unseen validation slice",
        "",
        "Split: training seeds 3073/3074; unseen perturbations gaussian_blur and resize; fixed comparison std_max 0.0 -> 0.08.",
        "",
        "| Metric | rho vs stress delta | r vs stress delta | rho vs drop improvement | r vs drop improvement | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["heldout_unseen_validation"]["metric_rows"]:
        lines.append(
            "| {metric} | {rho_s} | {r_s} | {rho_d} | {r_d} | {n} |".format(
                metric=row["metric"],
                rho_s=_fmt(row["spearman_vs_stress_success_delta"]),
                r_s=_fmt(row["pearson_vs_stress_success_delta"]),
                rho_d=_fmt(row["spearman_vs_drop_improvement"]),
                r_d=_fmt(row["pearson_vs_drop_improvement"]),
                n=row["n"],
            )
        )
    topk = payload["heldout_unseen_validation"]["topk_summary"]
    lines.extend(
        [
            "",
            f"Top-{topk['k']} agreement: composite signed-rank top-k hits {topk['stress_success_delta_topk_hit_count']}/{topk['stress_success_delta_topk_total']} for stress-success delta and {topk['drop_improvement_topk_hit_count']}/{topk['drop_improvement_topk_total']} for drop improvement.",
            "",
            "## Semantic discriminability protocol ledger",
            "",
            "| Task | semantic factor | available source | release status |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["semantic_discriminability_protocol"]:
        lines.append(
            "| {task} | {factor} | {source} | {status} |".format(
                task=row["task"],
                factor=row["semantic_factor"],
                source=row["available_source"],
                status=row["release_status"],
            )
        )
    lines.extend(
        [
            "",
            "## Remaining validation work",
            "",
        ]
    )
    for item in payload["remaining_validation_work"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)
    payload = build_payload()
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.out_md, payload)
    print(f"wrote {args.out_json.relative_to(ROOT)}")
    print(f"wrote {args.out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
