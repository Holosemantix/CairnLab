"""Plateau-aware selector audit for Paper 1.

The audit treats std_max as a candidate label inside each task/training-seed
block, not as a continuous covariate. It evaluates whether a fixed diagnostic
rule lands inside the closed-loop robustness plateau and whether diagnostic
rankings agree on checkpoint pairs whose closed-loop scores differ by more than
the plateau tolerance.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_IN = DATA_DIR / "acpc_phase0_lewm_three_seed.json"
DEFAULT_OUT_JSON = DATA_DIR / "selector_plateau_audit_20260704.json"
DEFAULT_OUT_MD = DATA_DIR / "selector_plateau_audit_20260704.md"

TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
NONZERO_STD_KEYS = ("0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
RULE = (
    ("ACPC-H/trans.", "acpc_h_norm_by_transition", -1.0),
    ("PCC", "pcc_abs_median", -1.0),
    ("CRA", "cra_spearman_mean", 1.0),
    ("MAF", "maf_flip_rate", -1.0),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values)


def _std(values: Sequence[float]) -> float:
    return statistics.pstdev([float(v) for v in values])


def _rankdata(values: Sequence[float], *, higher_better: bool) -> list[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: item[1], reverse=higher_better)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _aggregate_scores(block: Sequence[dict]) -> list[float]:
    scores = [0.0] * len(block)
    for _, key, sign in RULE:
        signed = [sign * float(row[key]) for row in block]
        ranks = _rankdata(signed, higher_better=True)
        for i, rank in enumerate(ranks):
            scores[i] -= rank
    return scores


def _metric_scores(key: str, sign: float) -> Callable[[Sequence[dict]], list[float]]:
    return lambda block: [sign * float(row[key]) for row in block]


def _fixed_std_selector(std_key: str) -> Callable[[Sequence[dict]], int]:
    return lambda block: next(i for i, row in enumerate(block) if str(row["std_key"]) == std_key)


def _best_index(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=lambda i: scores[i])


def _prepare_rows(data: dict) -> list[dict]:
    rows = [row for row in data["rows"] if row.get("status") == "ok" and str(row["std_key"]) != "0.0"]
    expected = {(task, seed, std) for task in TASKS for seed in SEEDS for std in NONZERO_STD_KEYS}
    got = {(row["task"], int(row["training_seed"]), str(row["std_key"])) for row in rows}
    if got != expected:
        raise ValueError(f"coverage mismatch missing={sorted(expected - got)[:5]} extra={sorted(got - expected)[:5]}")
    return sorted(rows, key=lambda r: (r["task"], int(r["training_seed"]), float(r["std_key"])))


def _blocks(rows: Sequence[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for task in TASKS:
        for seed in SEEDS:
            block = [row for row in rows if row["task"] == task and int(row["training_seed"]) == seed]
            out.append(sorted(block, key=lambda r: float(r["std_key"])))
    return out


def _selection_summary(
    rows: Sequence[dict],
    *,
    name: str,
    select_index: Callable[[Sequence[dict]], int],
    tolerance_pp: float,
) -> dict:
    block_rows = []
    for block in _blocks(rows):
        values = [float(row["pixels_std0.08_success"]) for row in block]
        best = max(values)
        chosen = select_index(block)
        regret = best - values[chosen]
        block_rows.append(
            {
                "task": block[0]["task"],
                "training_seed": int(block[0]["training_seed"]),
                "selected_std": str(block[chosen]["std_key"]),
                "selected_success": values[chosen],
                "best_success": best,
                "best_std": str(block[values.index(best)]["std_key"]),
                "point_regret_pp": regret,
                "plateau_hit": regret <= tolerance_pp + 1e-9,
                "regret_to_plateau_pp": max(0.0, regret - tolerance_pp),
            }
        )
    point_regrets = [row["point_regret_pp"] for row in block_rows]
    plateau_regrets = [row["regret_to_plateau_pp"] for row in block_rows]
    return {
        "selector": name,
        "n_blocks": len(block_rows),
        "plateau_hit_count": sum(1 for row in block_rows if row["plateau_hit"]),
        "bad_pick_count": sum(1 for row in block_rows if not row["plateau_hit"]),
        "point_regret_mean_pp": _mean(point_regrets),
        "point_regret_std_pp": _std(point_regrets),
        "regret_to_plateau_mean_pp": _mean(plateau_regrets),
        "regret_to_plateau_std_pp": _std(plateau_regrets),
        "block_rows": block_rows,
    }


def _ranking_summary(
    rows: Sequence[dict],
    *,
    name: str,
    score_fn: Callable[[Sequence[dict]], list[float]],
    tolerance_pp: float,
) -> dict:
    correct = 0.0
    total = 0
    by_block = []
    for block in _blocks(rows):
        scores = score_fn(block)
        values = [float(row["pixels_std0.08_success"]) for row in block]
        block_correct = 0.0
        block_total = 0
        for i, j in itertools.combinations(range(len(block)), 2):
            delta = values[i] - values[j]
            if abs(delta) <= tolerance_pp:
                continue
            better = i if delta > 0 else j
            worse = j if delta > 0 else i
            if abs(scores[better] - scores[worse]) <= 1e-12:
                block_correct += 0.5
            elif scores[better] > scores[worse]:
                block_correct += 1.0
            block_total += 1
        correct += block_correct
        total += block_total
        by_block.append(
            {
                "task": block[0]["task"],
                "training_seed": int(block[0]["training_seed"]),
                "decisive_pair_count": block_total,
                "decisive_pair_correct": block_correct,
                "decisive_pair_accuracy": block_correct / block_total if block_total else None,
            }
        )
    return {
        "ranker": name,
        "decisive_pair_correct": correct,
        "decisive_pair_count": total,
        "decisive_pair_accuracy": correct / total if total else None,
        "block_rows": by_block,
    }


def _random_nonzero_summary(rows: Sequence[dict], tolerance_pp: float) -> dict:
    hits = []
    point_regrets = []
    plateau_regrets = []
    for block in _blocks(rows):
        values = [float(row["pixels_std0.08_success"]) for row in block]
        best = max(values)
        for value in values:
            regret = best - value
            hits.append(regret <= tolerance_pp + 1e-9)
            point_regrets.append(regret)
            plateau_regrets.append(max(0.0, regret - tolerance_pp))
    return {
        "selector": "Random nonzero std (exact expectation)",
        "n_blocks": 12,
        "plateau_hit_count_expected": 12.0 * sum(hits) / len(hits),
        "bad_pick_count_expected": 12.0 * (1.0 - sum(hits) / len(hits)),
        "point_regret_mean_pp": _mean(point_regrets),
        "point_regret_std_pp": _std(point_regrets),
        "regret_to_plateau_mean_pp": _mean(plateau_regrets),
        "regret_to_plateau_std_pp": _std(plateau_regrets),
    }


def build_audit(data: dict, tolerance_pp: float) -> dict:
    rows = _prepare_rows(data)
    score_fns = {
        "Aggregate ACPC/PCC/CRA/MAF": _aggregate_scores,
        "ACPC only": _metric_scores("acpc_h_norm_by_transition", -1.0),
        "PCC only": _metric_scores("pcc_abs_median", -1.0),
        "CRA only": _metric_scores("cra_spearman_mean", 1.0),
        "MAF only": _metric_scores("maf_flip_rate", -1.0),
        "Monotone high-std baseline": _metric_scores("std_key", 1.0),
    }
    selection_specs = {
        "Aggregate ACPC/PCC/CRA/MAF": lambda block: _best_index(_aggregate_scores(block)),
        "Fixed std=0.08": _fixed_std_selector("0.08"),
        "MAF only": lambda block: _best_index(_metric_scores("maf_flip_rate", -1.0)(block)),
    }
    return {
        "metadata": {
            "schema_version": "paper1-selector-plateau-audit-20260704-v1",
            "source_artifact": str(DEFAULT_IN.relative_to(ROOT)),
            "row_unit": "task-training-seed block, nonzero Gaussian std candidates",
            "tolerance_pp": tolerance_pp,
            "interpretation": "Plateau-localization and decisive-pair audit; std_max is treated as a candidate label, not a continuous covariate.",
        },
        "selection_summaries": [
            _selection_summary(rows, name=name, select_index=fn, tolerance_pp=tolerance_pp)
            for name, fn in selection_specs.items()
        ]
        + [_random_nonzero_summary(rows, tolerance_pp)],
        "ranking_summaries": [
            _ranking_summary(rows, name=name, score_fn=fn, tolerance_pp=tolerance_pp)
            for name, fn in score_fns.items()
        ],
    }


def write_markdown(audit: dict, path: Path) -> None:
    meta = audit["metadata"]
    lines = [
        "# Selector plateau audit",
        "",
        f"- Tolerance: {meta['tolerance_pp']:.1f} pp",
        "- Unit: task-training-seed block over the eight nonzero Gaussian candidates.",
        "- Reading: plateau localization and bad-checkpoint triage, not point-optimal std selection.",
        "",
        "## Selection summaries",
        "",
        "| Selector | plateau hit | bad picks | point regret mean±std | regret-to-plateau mean±std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in audit["selection_summaries"]:
        if "plateau_hit_count" in row:
            hit = f"{row['plateau_hit_count']}/{row['n_blocks']}"
            bad = f"{row['bad_pick_count']}/{row['n_blocks']}"
        else:
            hit = f"{row['plateau_hit_count_expected']:.1f}/{row['n_blocks']}"
            bad = f"{row['bad_pick_count_expected']:.1f}/{row['n_blocks']}"
        lines.append(
            f"| {row['selector']} | {hit} | {bad} | "
            f"{row['point_regret_mean_pp']:.2f}±{row['point_regret_std_pp']:.2f} | "
            f"{row['regret_to_plateau_mean_pp']:.2f}±{row['regret_to_plateau_std_pp']:.2f} |"
        )
    lines += [
        "",
        "## Decisive-pair ranking summaries",
        "",
        "| Ranker | decisive pairs | accuracy |",
        "|---|---:|---:|",
    ]
    for row in audit["ranking_summaries"]:
        lines.append(
            f"| {row['ranker']} | {row['decisive_pair_count']} | "
            f"{100.0 * row['decisive_pair_accuracy']:.1f}% |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--tolerance-pp", type=float, default=5.0)
    args = parser.parse_args()

    audit = build_audit(_load(args.input), args.tolerance_pp)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_markdown(audit, args.out_md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
