"""
cross_check_correlations.py — Cross-check P0.5 "main indicator" claims by
running three additional analyses on the canonical 8 ckpts/task that the
univariate Spearman in §6 P0.4/P0.5 does *not* do:

1. Within-method Spearman (n=4 LeWM, n=4 SWM): separates "high-vs-low eval
   model"-axis correlations from the LeWM↔SWM cluster-axis. Many aggregate
   ρ collapse or sign-flip when restricted to a single architecture.

2. Partial Spearman conditioning on `std_max` (training noise intensity):
   removes the obvious confound that noise training simultaneously shifts
   eval and many diagnostics (T8 drift, noise_angle_slope, etc.). Metrics
   whose aggregate ρ disappears under partial corr are mostly carrying
   noise-training-intensity information, not structural latent geometry.

3. Top-2 vs bottom-2 group mean contrast: relative diff of metric means
   between the two highest-eval and two lowest-eval ckpts in each task.
   Validates whether the rank correlation reflects a sizeable group-mean
   gap (vs. monotonic but tiny separation).

Usage:
    STABLEWM_HOME=<dataset_root> python -m tools.repr_analysis.cross_check_correlations
        [--out cross_check_corr.json]

Reads each ckpt's `eval_results/diagnostics/diagnostics_summary.json` and
the canonical eval JSON shipped at repo root. Pure stdlib; no scipy needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

CANONICAL_EVALS = "canonical_evals_20260506.json"

CANON = [
    ("LeWM-base",        "{tk}_lewm_20260430",                                                                "LeWM", 0.000),
    ("LeWM-0to001-p1",   "{tk}_lewm_noise_0to001_p1",                                                          "LeWM", 0.001),
    ("LeWM-0to002-p1",   "{tk}_lewm_noise_0to002_p1",                                                          "LeWM", 0.002),
    ("LeWM-0to005-p1",   "{tk}_lewm_noise_0to005_p1",                                                          "LeWM", 0.005),
    ("SWM-base",         "{tk}_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64",                             "SWM",  0.000),
    ("SWM-0to001-p1",    "{tk}_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64",             "SWM",  0.001),
    ("SWM-0to002-p1",    "{tk}_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64",             "SWM",  0.002),
    ("SWM-0to005-p1",    "{tk}_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64",             "SWM",  0.005),
]

TASKS = [
    ("TwoRoom", "tworooms", "tworoom"),
    ("PushT",   "pusht",    "pusht"),
    ("Reacher", "reacher",  "reacher"),
    ("Cube",    "cube",     "cube"),
]

METRICS = [
    "predictor_target_to_nn_cos_ratio_at_max_std",
    "predictor_rollout_T8_l2",
    "clean_effective_rank",
    "lidar_rank",
    "noise_angle_slope_deg_per_std",
    "cka_linear_at_max_std",
    "latent_rollout_angle_slope_per_std_z",
    "latent_cost_surface_slope_z",
    "latent_predictor_rollout_T8_l2_history",
    "id_probe_r2",
    "transition_resolution_ratio_cos",
    "clean_nn_cos_dist_median",
]


def avg_ranks(xs: Sequence[float]) -> list[float]:
    n = len(xs)
    idx = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (sx * sy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(avg_ranks(xs), avg_ranks(ys))


def partial_spearman(xs: Sequence[float], ys: Sequence[float], zs: Sequence[float]) -> float:
    """Pearson of rank residuals after regressing both on rank(z). Approximates
    partial Spearman for small n."""
    rx, ry, rz = avg_ranks(xs), avg_ranks(ys), avg_ranks(zs)
    n = len(rz)
    mz = sum(rz) / n
    sz2 = sum((z - mz) ** 2 for z in rz)
    if sz2 == 0:
        return pearson(rx, ry)
    bx = sum((rx[i] - sum(rx) / n) * (rz[i] - mz) for i in range(n)) / sz2
    ax = sum(rx) / n - bx * mz
    rx_ = [rx[i] - (ax + bx * rz[i]) for i in range(n)]
    by = sum((ry[i] - sum(ry) / n) * (rz[i] - mz) for i in range(n)) / sz2
    ay = sum(ry) / n - by * mz
    ry_ = [ry[i] - (ay + by * rz[i]) for i in range(n)]
    return pearson(rx_, ry_)


def load_diag(stablewm_home: Path, sub: str, ckpt_subdir: str) -> dict | None:
    p = stablewm_home / f"lewm-{sub}" / "ckpt" / ckpt_subdir / "eval_results" / "diagnostics" / "diagnostics_summary.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    if isinstance(d, list):
        d = d[0]
    return d


def build_task_rows(stablewm_home: Path, evals: Mapping[str, Mapping[str, float]], task: str, sub: str) -> list[dict]:
    pre = "tworoom" if sub == "tworooms" else sub
    rows = []
    for label, tpl, method, std in CANON:
        d = load_diag(stablewm_home, sub, tpl.format(tk=pre))
        if d is None:
            continue
        if label not in evals[task]:
            continue
        row = {"label": label, "method": method, "std": std, "eval": evals[task][label]}
        for m in METRICS:
            v = d.get(m)
            row[m] = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
        rows.append(row)
    return rows


def cross_check(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    evals = [r["eval"] for r in rows]
    stds = [r["std"] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: evals[i])
    bot2, top2 = order[:2], order[-2:]
    out = []
    for m in METRICS:
        vals = [r[m] for r in rows]
        if any(v is None for v in vals):
            out.append({"metric": m, "skipped": True})
            continue
        rho_all = spearman(vals, evals)
        l_idx = [i for i, r in enumerate(rows) if r["method"] == "LeWM"]
        s_idx = [i for i, r in enumerate(rows) if r["method"] == "SWM"]
        rho_l = spearman([vals[i] for i in l_idx], [evals[i] for i in l_idx]) if len(l_idx) >= 3 else float("nan")
        rho_s = spearman([vals[i] for i in s_idx], [evals[i] for i in s_idx]) if len(s_idx) >= 3 else float("nan")
        rho_partial = partial_spearman(vals, evals, stds)
        top_mean = sum(vals[i] for i in top2) / 2
        bot_mean = sum(vals[i] for i in bot2) / 2
        denom = max(abs(top_mean), abs(bot_mean), 1e-9)
        rel = (top_mean - bot_mean) / denom
        out.append({
            "metric": m,
            "rho_all_n8": round(rho_all, 4),
            "rho_within_LeWM_n4": round(rho_l, 4),
            "rho_within_SWM_n4": round(rho_s, 4),
            "rho_partial_given_std": round(rho_partial, 4),
            "top2_minus_bot2_relative": round(rel, 4),
            "top2_mean": round(top_mean, 6),
            "bot2_mean": round(bot_mean, 6),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stablewm-home", default=os.environ.get("STABLEWM_HOME"))
    ap.add_argument("--evals", default=str(Path(__file__).resolve().parent.parent.parent / CANONICAL_EVALS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.stablewm_home:
        raise SystemExit("STABLEWM_HOME or --stablewm-home required (dir containing lewm-<task>/)")
    home = Path(args.stablewm_home)

    raw = json.loads(Path(args.evals).read_text())
    evals_clean = {tk: {label: v["evals"]["clean"] for label, v in raw[tk].items()} for tk in raw}

    result = {}
    print(f"{'Task':<8} {'Metric':<46} {'rho_n8':>7} {'LeWM_n4':>9} {'SWM_n4':>8} {'partial':>9} {'top2-bot2':>11}")
    print("-" * 102)
    for task, sub, _ in TASKS:
        rows = build_task_rows(home, evals_clean, task, sub)
        result[task] = {"n_models": len(rows), "rows": cross_check(rows)}
        for r in result[task]["rows"]:
            if r.get("skipped"):
                continue
            sign = "↑" if r["top2_mean"] > r["bot2_mean"] else "↓"
            contrast = f"{sign}{abs(r['top2_minus_bot2_relative']) * 100:5.1f}%"
            print(f"{task:<8} {r['metric']:<46} "
                  f"{r['rho_all_n8']:>+7.2f} {r['rho_within_LeWM_n4']:>+9.2f} "
                  f"{r['rho_within_SWM_n4']:>+8.2f} {r['rho_partial_given_std']:>+9.2f} "
                  f"{contrast:>11}")
        print()
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
