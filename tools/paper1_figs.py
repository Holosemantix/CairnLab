"""Render the 5 main paper figures for the Invariance-Resolution paper.

Run:
    python -m tools.paper1_figs --out-dir assets/paper1_figs

Generates:
    fig1_hero.png      — 4-task OOD cliff + per-task best recovery
    fig2_sweep.png     — 4 panels of clean / px+g 0.05 / px+g 0.08 vs std_max
    fig3_scatter.png   — PushT n=18 scatter: predictor_target_to_nn_cos_ratio
                         (max-std) vs eval-drop, LeWM (○) + SWM (△)
    fig4_radar.png     — 4-task diagnostic radar (base vs best on 6 metrics)
    fig5_mechanism.png — 3-layer attribution flow chart (PushT-centric)

Data sources (no new computation needed — pulls from existing JSONs and
the hard-coded numbers reported in §3-§4 of the paper):

- Eval tables (§4.2, §4.3): hard-coded from the paper (mirrors of
  research_notebook §4.2 and plan §3.2.2; cross-checked at paper-write
  time).
- Per-ckpt diagnostics for the scatter: globbed from
  /home/ag/dataset/ag_data/data/world_model/quentinll/lewm-pusht/ckpt/
  *_lewm_noise_0to00[1-8]_p1 / *_swm_..._noise_0to00[1-8]_p1_dim64 +
  the two base ckpts.
- Diagnostic Table 3 / radar data: hard-coded from the paper §4.4 + §A.6.

All paths follow the canonical_evals_20260508.json scheme. Adjust the
data-root constant below if your local checkout differs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Hard-coded data (mirrored from the paper). Update here if numbers change.
# ============================================================================

# §4.2 Table 1 — LeWM-base OOD cliff (4 tasks)
EVAL_BASE = {
    "TwoRoom": {"clean": 93.00, "px08": 44.33},
    "PushT":   {"clean": 87.33, "px08":  3.67},
    "Reacher": {"clean": 57.67, "px08": 14.67},
    "Cube":    {"clean": 72.33, "px08": 52.33},
}

# §4.3 Table 2 — LeWM+noise per-task best (clean / px+g 0.08)
EVAL_BEST = {
    "TwoRoom": {"std": 0.008, "clean": 98.33, "px08": 98.67},
    "PushT":   {"std": 0.006, "clean": 89.33, "px08": 87.00},
    "Reacher": {"std": 0.006, "clean": 86.00, "px08": 84.67},
    "Cube":    {"std": 0.003, "clean": 65.00, "px08": 67.33},
}

# §4.3 Table 2 — full sweep (clean / px+g 0.08) for the line plot
SWEEP_STDS = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
SWEEP = {
    "TwoRoom": {
        "clean": [93.00, 92.00, 94.33, 96.33, 96.33, 94.00, 96.67, 96.00, 98.33],
        "px08":  [44.33, 84.67, 91.00, 94.67, 95.00, 94.00, 96.67, 96.33, 98.67],
    },
    "PushT": {
        "clean": [87.33, 89.67, 90.00, 89.67, 89.33, 82.00, 89.33, 85.67, 88.33],
        "px08":  [ 3.67, 46.33, 70.67, 83.00, 81.33, 78.00, 87.00, 82.33, 85.33],
    },
    "Reacher": {
        "clean": [57.67, 55.67, 80.33, 78.67, 84.00, 73.33, 86.00, 83.67, 84.00],
        "px08":  [14.67, 45.33, 80.67, 73.67, 80.00, 71.33, 84.67, 81.33, 83.00],
    },
    "Cube": {
        "clean": [72.33, 73.00, 64.67, 65.00, 69.00, 61.33, 66.67, 67.67, 62.33],
        "px08":  [52.33, 53.33, 63.00, 67.33, 67.00, 60.67, 65.00, 68.00, 60.33],
    },
}

# §4.4 Table 3 — 6 diagnostic metrics × {base, best} × 4 tasks
# Metric order chosen so "compression" metrics group on one side of the radar.
DIAG_METRICS = [
    "clean_effective_rank",
    "clean_nn_cos_dist_median",
    "transition_resolution_ratio_l2",
    "transition_resolution_ratio_cos",
    "id_probe_r2",
    "action_mean_pred_shift_norm",
]
DIAG_DATA = {
    "TwoRoom": {
        "base": [47.60, 0.0449, 0.7216, 0.5538, 0.2889,  0.5329],
        "best": [33.59, 0.0281, 0.6055, 0.3780,-0.0573,  0.4482],
    },
    "PushT": {
        "base": [76.42, 0.2360, 0.3015, 0.0868, 0.7739,  0.1283],
        "best": [42.85, 0.1051, 0.2800, 0.0800, 0.7500,  0.1200],
    },
    "Reacher": {
        "base": [61.04, 0.0633, 0.3704, 0.1351, 0.1621,  0.2518],
        "best": [65.92, 0.0676, 0.3791, 0.1399, 0.1729,  0.2585],
    },
    "Cube": {
        "base": [73.25, 0.1856, 0.4847, 0.2347, 0.6657,  0.2364],
        "best": [71.83, 0.1879, 0.4629, 0.2168, 0.6720,  0.2320],
    },
}

# §4.6 Mechanism attribution — three layers × per-task contribution
# Numbers are |ρ| of the strongest per-layer metric vs eval drop (n=8 / canonical).
# pixel-encoder = noise_angle_slope or clean_nn (encoder geometry under noise)
# predictor     = predictor_target_to_nn_cos_ratio_at_max_std (single step)
#               + predictor_rollout_T8_l2 (multi-step)
# cost          = latent_cost_surface_slope_z
MECH = {
    "TwoRoom": {"encoder": 0.93, "predictor": 0.79, "cost": 0.61},
    "PushT":   {"encoder": 0.31, "predictor": 0.93, "cost": 0.93},
    "Reacher": {"encoder": 0.74, "predictor": 0.83, "cost": 0.14},
    "Cube":    {"encoder": 0.96, "predictor": 0.76, "cost": 0.37},
}


# ============================================================================
# Helpers
# ============================================================================

def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================================================================
# Figure 1 — Hero: OOD cliff vs noise-training recovery (4-task grouped bar)
# ============================================================================

def fig1_hero(out_path: Path):
    tasks = list(EVAL_BASE.keys())
    base_clean = [EVAL_BASE[t]["clean"] for t in tasks]
    base_px08  = [EVAL_BASE[t]["px08"]  for t in tasks]
    best_px08  = [EVAL_BEST[t]["px08"]  for t in tasks]
    best_stds  = [EVAL_BEST[t]["std"]   for t in tasks]

    x = np.arange(len(tasks))
    w = 0.26

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(x - w, base_clean, w, label="LeWM-base, clean",
           color="#4477AA", edgecolor="black", linewidth=0.5)
    ax.bar(x,     base_px08,  w, label="LeWM-base, px+g 0.08",
           color="#EE6677", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, best_px08,  w, label="LeWM+noise (best), px+g 0.08",
           color="#228833", edgecolor="black", linewidth=0.5)

    for i, t in enumerate(tasks):
        drop = base_clean[i] - base_px08[i]
        recover = best_px08[i] - base_px08[i]
        # drop label above the px+g 0.08 base bar
        ax.text(x[i], base_px08[i] + 2,
                f"−{drop:.0f}", ha="center", va="bottom",
                color="#EE6677", fontsize=9, fontweight="bold")
        # recover label above the best bar
        ax.text(x[i] + w, best_px08[i] + 2,
                f"+{recover:.0f}", ha="center", va="bottom",
                color="#228833", fontsize=9, fontweight="bold")
        # σ* annotation under the x-tick label, in green
        ax.text(x[i] + w, -7,
                f"σ*={best_stds[i]:.3f}", ha="center", va="top",
                color="#228833", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(-14, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("Fig. 1. Visual OOD cliff in LeWM and recovery by noise training "
                 "(success rate, 4 tasks)", loc="left", pad=10)
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Figure 2 — Sweep curves: clean and OOD vs std_max, per task
# ============================================================================

def fig2_sweep(out_path: Path):
    tasks = list(SWEEP.keys())
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, t in zip(axes, tasks):
        ax.plot(SWEEP_STDS, SWEEP[t]["clean"], "o-", color="#4477AA",
                label="clean", linewidth=1.6, markersize=4)
        ax.plot(SWEEP_STDS, SWEEP[t]["px08"], "s-", color="#EE6677",
                label="px+g 0.08", linewidth=1.6, markersize=4)
        best_std = EVAL_BEST[t]["std"]
        ax.axvline(best_std, color="#228833", linestyle="--", alpha=0.7, linewidth=1.0)
        ax.set_title(f"{t}   (σ*={best_std:.3f})", fontsize=10.5)
        ax.set_xlabel("std_max")
        ax.set_xticks([0, 0.002, 0.004, 0.006, 0.008])
        ax.set_xticklabels(["0", ".002", ".004", ".006", ".008"])
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("Fig. 2. Noise-training sweep: clean vs OOD per task; "
                 "no single std_max is jointly optimal across tasks",
                 x=0.01, y=1.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Figure 3 — Scatter: predictor_target_to_nn_cos_ratio_at_max_std vs eval drop
# Globs n=18 PushT ckpts (LeWM 9 + SWM 9), pulls per-ckpt metric and eval drop.
# ============================================================================

def _glob_n9_pusht_lewm(data_root: Path) -> List[Tuple[str, str, float, str]]:
    """Return [(label, method, std_max, ckpt_dir), ...] for the 9 LeWM PushT
    sweep ckpts (base + std 0.001..0.008). Prefers `_20260507` retrained
    canonical paths when they exist (matches `canonical_evals_20260508.json`)."""
    out = []
    pusht_root = data_root / "lewm-pusht" / "ckpt"
    # base
    base_dir = pusht_root / "pusht_lewm_20260430"
    if base_dir.exists():
        out.append(("LeWM-base", "LeWM", 0.0, str(base_dir)))
    # noise sweep 0to001..0to008
    for i in range(1, 9):
        sx = f"0to00{i}_p1"
        retrained = pusht_root / f"pusht_lewm_noise_{sx}_20260507"
        default = pusht_root / f"pusht_lewm_noise_{sx}"
        chosen = retrained if retrained.exists() else default
        if not chosen.exists():
            continue
        out.append((f"LeWM-noise-0to00{i}", "LeWM", float(f"0.00{i}"), str(chosen)))
    return out


# Backward-compat alias for any code that imports the older name.
_glob_n18_pusht = _glob_n9_pusht_lewm


def _read_predictor_target_ratio(ckpt_dir: Path) -> float:
    """Find the max-std, history-only row in predictor_sensitivity.json and
    return target_to_nn_cos_ratio. NaN if missing."""
    fp = Path(ckpt_dir) / "eval_results" / "diagnostics" / "predictor_sensitivity.json"
    if not fp.exists():
        return float("nan")
    with open(fp) as f:
        rows = json.load(f)
    if not rows:
        return float("nan")
    rows = [r for r in rows if r.get("history_noise_only") in (True, "true", "True", None)]
    if not rows:
        return float("nan")
    max_std = max(r["std"] for r in rows)
    for r in rows:
        if r["std"] == max_std:
            return float(r.get("target_to_nn_cos_ratio", float("nan")))
    return float("nan")


def _read_eval_metrics(ckpt_dir: Path) -> Dict[str, float]:
    """Parse summary.txt and return {'clean': float, 'px08': float}."""
    import re
    summ = Path(ckpt_dir) / "eval_results" / "summary.txt"
    if not summ.exists():
        return {"clean": float("nan"), "px08": float("nan")}
    text = summ.read_text(errors="ignore")
    sections: Dict[str, float] = {}
    block_re = re.compile(
        r"==\s*(?P<name>[\w.+]+)\s*==\s*\n\s*\{'success_rate':\s*(?P<sr>[-+0-9.eE]+)"
    )
    for m in block_re.finditer(text):
        try:
            sections[m.group("name")] = float(m.group("sr"))
        except ValueError:
            continue

    def _resolve(prefer_names, seed_prefix):
        for n in prefer_names:
            if n in sections:
                return sections[n]
        seed_vals = [v for k, v in sections.items() if k.startswith(seed_prefix + "_seed")]
        if seed_vals:
            return sum(seed_vals) / len(seed_vals)
        return float("nan")

    return {
        "clean": _resolve(["clean_300", "clean"], "clean"),
        "px08":  _resolve(["pixels_goal_std0.08"], "pixels_goal_std0.08"),
    }


def _read_eval_drop(ckpt_dir: Path) -> float:
    """clean − px+g 0.08 from summary.txt.

    Two summary formats coexist in the repo:
      (A) single-seed × 300: blocks `== clean ==`, `== clean_300 ==`,
          `== pixels_goal_std0.08 ==`, ...
      (B) 3-seed × 100:      blocks `== clean_seed42/43/44 ==`,
          `== pixels_goal_std0.08_seed42/43/44 ==`, ...
    Each block is followed by a python-repr dict starting with
    `'success_rate': <float>`. We parse all blocks and:
      - prefer `clean_300` then `clean`; otherwise mean of `clean_seed*`.
      - prefer `pixels_goal_std0.08`; otherwise mean of `pixels_goal_std0.08_seed*`.
    """
    import re
    summ = Path(ckpt_dir) / "eval_results" / "summary.txt"
    if not summ.exists():
        return float("nan")
    text = summ.read_text(errors="ignore")
    sections: Dict[str, float] = {}
    block_re = re.compile(
        r"==\s*(?P<name>[\w.+]+)\s*==\s*\n\s*\{'success_rate':\s*(?P<sr>[-+0-9.eE]+)"
    )
    for m in block_re.finditer(text):
        try:
            sections[m.group("name")] = float(m.group("sr"))
        except ValueError:
            continue

    def _resolve(prefer_names, seed_prefix):
        for n in prefer_names:
            if n in sections:
                return sections[n]
        seed_vals = [v for k, v in sections.items() if k.startswith(seed_prefix + "_seed")]
        if seed_vals:
            return sum(seed_vals) / len(seed_vals)
        return float("nan")

    clean = _resolve(["clean_300", "clean"], "clean")
    px08 = _resolve(["pixels_goal_std0.08"], "pixels_goal_std0.08")
    if math.isnan(clean) or math.isnan(px08):
        return float("nan")
    return clean - px08


def _spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    """Spearman ρ via rank-Pearson (no scipy required)."""
    if len(xs) < 3:
        return float("nan")
    def _rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks
    rx, ry = _rank(xs), _rank(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = max(float((rx ** 2).sum() ** 0.5 * (ry ** 2).sum() ** 0.5), 1e-9)
    return float((rx * ry).sum() / denom)


def fig3_scatter(out_path: Path, data_root: Path):
    ckpts = _glob_n9_pusht_lewm(data_root)
    if not ckpts:
        print(f"  WARN: no PushT LeWM ckpts found under {data_root}; skipping fig3")
        return
    rows = []
    for label, method, std, ck in ckpts:
        ratio = _read_predictor_target_ratio(Path(ck))
        ev = _read_eval_metrics(Path(ck))
        if math.isnan(ratio) or math.isnan(ev["clean"]) or math.isnan(ev["px08"]):
            continue
        rows.append((label, std, ratio, ev["clean"], ev["px08"]))
    if len(rows) < 5:
        print(f"  WARN: only {len(rows)} valid PushT LeWM ckpts; skipping fig3")
        return

    # Two side-by-side panels: vs clean (strong) | vs OOD drop (weak)
    fig, (ax_clean, ax_drop) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    xs = np.array([r[2] for r in rows])
    cleans = np.array([r[3] for r in rows])
    px08s = np.array([r[4] for r in rows])
    drops = cleans - px08s
    stds_ax = np.array([r[1] for r in rows])

    def _panel(ax_, ys, ylabel, title, anchor_y_top):
        sc = ax_.scatter(xs, ys, marker="o", s=90,
                         c=stds_ax, cmap="Blues", edgecolor="black", linewidth=0.5,
                         vmin=0, vmax=0.008)
        # log-linear fit
        logxs = np.log10(xs)
        p = np.polyfit(logxs, ys, 1)
        xf = np.linspace(logxs.min() - 0.1, logxs.max() + 0.1, 50)
        ax_.plot(10 ** xf, p[0] * xf + p[1], "k--", linewidth=1.0, alpha=0.6)
        rho = _spearman(xs, ys)
        ax_.text(0.04, anchor_y_top,
                 f"Spearman ρ (LeWM, n={len(rows)}): {rho:+.2f}\n"
                 f"linear-fit slope (on log x): {p[0]:+.1f}",
                 transform=ax_.transAxes,
                 va="top" if anchor_y_top > 0.5 else "bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           edgecolor="gray", alpha=0.92), fontsize=8.5,
                 family="DejaVu Sans Mono")
        ax_.set_xscale("log")
        ax_.set_xlabel(r"$\mathtt{predictor\_target\_to\_nn\_cos\_ratio}$")
        ax_.set_ylabel(ylabel)
        ax_.set_title(title, fontsize=10, loc="left", pad=8)
        ax_.grid(alpha=0.25, linewidth=0.4)
        return rho

    rho_clean = _panel(ax_clean, cleans, "PushT clean success rate (%)",
                       "(a)  vs clean success — strong cross-ckpt signal",
                       anchor_y_top=0.18)
    rho_drop = _panel(ax_drop, drops, "PushT eval drop  (clean − px+g 0.08, pts)",
                      "(b)  vs OOD drop — weak / dominated by training protocol",
                      anchor_y_top=0.96)

    # colourbar for std_max
    cbar = fig.colorbar(ax_clean.collections[0], ax=[ax_clean, ax_drop],
                        fraction=0.025, pad=0.04)
    cbar.set_label("std_max during training", fontsize=8.5)

    fig.suptitle("Fig. 3. PushT LeWM noise sweep (n=9): fragility metric is a ckpt-quality "
                 "predictor (a), not an OOD-specific predictor (b)",
                 x=0.01, y=1.02, ha="left", fontsize=11)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path} (n={len(rows)} LeWM ckpts;  "
          f"ρ_clean={rho_clean:+.2f}, ρ_drop={rho_drop:+.2f})")


# ============================================================================
# Figure 4 — Radar: 4 tasks × 6 diagnostic metrics (base vs best)
# Metrics are normalized per-task to [0,1] using the wider of {base, best}
# extents so radial axes are visually comparable.
# ============================================================================

def fig4_radar(out_path: Path):
    metrics = DIAG_METRICS
    short_names = [
        "eff. rank",
        "NN cos dist",
        "trans. res. L2",
        "trans. res. cos",
        "id-probe R²",
        "action shift",
    ]
    tasks = list(DIAG_DATA.keys())

    # Normalize each metric across all 8 (4 tasks × 2 ckpts) values to [0,1].
    metric_min = [float("inf")] * len(metrics)
    metric_max = [-float("inf")] * len(metrics)
    for t in tasks:
        for which in ("base", "best"):
            for i, v in enumerate(DIAG_DATA[t][which]):
                metric_min[i] = min(metric_min[i], v)
                metric_max[i] = max(metric_max[i], v)
    def _norm(vals):
        out = []
        for i, v in enumerate(vals):
            lo, hi = metric_min[i], metric_max[i]
            out.append((v - lo) / max(hi - lo, 1e-9))
        return out

    angles = np.linspace(0, 2 * math.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    # 2x2 grid is more readable than 1x4 for radar charts with 6 long labels.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    for ax, t in zip(axes, tasks):
        base = _norm(DIAG_DATA[t]["base"])
        best = _norm(DIAG_DATA[t]["best"])
        base += base[:1]
        best += best[:1]
        ax.plot(angles, base, "o-", linewidth=1.6, color="#4477AA",
                label="base", markersize=4.5)
        ax.fill(angles, base, alpha=0.18, color="#4477AA")
        ax.plot(angles, best, "s-", linewidth=1.6, color="#228833",
                label=f"best (σ*={EVAL_BEST[t]['std']:.3f})", markersize=4.5)
        ax.fill(angles, best, alpha=0.18, color="#228833")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_names, fontsize=9)
        ax.tick_params(axis="x", pad=14)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", ".5", "1"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(t, fontsize=12, pad=28)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16),
                  frameon=False, fontsize=8.5, ncol=2)

    fig.suptitle("Fig. 4. Per-task diagnostic profile: base vs noise-best on 6 metrics  "
                 "(min-max normalized across 4 tasks)",
                 y=0.995, fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.88, bottom=0.05,
                        wspace=0.6, hspace=0.65)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Figure 5 — Mechanism attribution flow + per-task |ρ| stacked bar
# ============================================================================

def fig5_mechanism(out_path: Path):
    # Left panel: schematic flow with arrows; right panel: |ρ| per layer per task.
    fig = plt.figure(figsize=(11.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.25)

    # --- Left: schematic ---
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.axis("off")

    # Bigger gap between boxes so arrows + layer labels don't crowd
    box_hw, box_hh = 0.075, 0.085
    box_centers = [0.10, 0.37, 0.64, 0.92]  # widened from 0.36/0.62/0.88
    box_labels = ["pixels\n(noise +)", "encoder f", "predictor g", "cost / CEM"]
    for x, lab in zip(box_centers, box_labels):
        ax_left.add_patch(plt.Rectangle((x - box_hw, 0.45), 2 * box_hw, 2 * box_hh,
                                        edgecolor="black", facecolor="#EEEEEE", linewidth=1.0))
        ax_left.text(x, 0.535, lab, ha="center", va="center", fontsize=9)
    for i in range(len(box_centers) - 1):
        x0 = box_centers[i] + box_hw + 0.012
        x1 = box_centers[i + 1] - box_hw - 0.012
        ax_left.annotate("", xy=(x1, 0.535), xytext=(x0, 0.535),
                         arrowprops=dict(arrowstyle="->", lw=1.4))

    # Layer labels in the gaps BELOW the boxes, with a thin bracket-style cue
    layer_labels = ["encoder\nshift", "predictor\ndrift", "cost\nsurface"]
    for i, name in enumerate(layer_labels):
        x_mid = (box_centers[i] + box_centers[i + 1]) / 2
        ax_left.text(x_mid, 0.32, name, ha="center", va="center", fontsize=8.5,
                     style="italic", color="#555555")

    ax_left.text(0.50, 0.85,
                 "Eval-only cost swap (cos→mse, raw):\n"
                 "TwoRoom std=0.03  36.0 → 42.0  (+6 only)",
                 ha="center", va="center", fontsize=8, color="#995500",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF6E5",
                           edgecolor="#995500", linewidth=0.6))
    ax_left.text(0.50, 0.12,
                 "→ encoder shift transduced by predictor dominates;\n   cost surface is not the bottleneck",
                 ha="center", va="center", fontsize=8.5)
    ax_left.set_xlim(-0.02, 1.05)
    ax_left.set_ylim(0, 1)
    ax_left.set_title("Three-layer attribution path", loc="left", fontsize=10, pad=8)

    # --- Right: per-task |ρ| of strongest per-layer signal ---
    ax_right = fig.add_subplot(gs[0, 1])
    tasks = list(MECH.keys())
    x = np.arange(len(tasks))
    w = 0.26
    enc = [MECH[t]["encoder"]   for t in tasks]
    pred = [MECH[t]["predictor"] for t in tasks]
    cost = [MECH[t]["cost"]      for t in tasks]
    ax_right.bar(x - w, enc,  w, label="encoder shift",
                 color="#4477AA", edgecolor="black", linewidth=0.4)
    ax_right.bar(x,     pred, w, label="predictor drift",
                 color="#EE6677", edgecolor="black", linewidth=0.4)
    ax_right.bar(x + w, cost, w, label="cost surface",
                 color="#CCBB44", edgecolor="black", linewidth=0.4)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(tasks)
    ax_right.set_ylim(0, 1.05)
    ax_right.set_ylabel(r"$|\rho|$ of strongest per-layer signal vs eval drop  (n=8)")
    ax_right.set_title("Strongest per-layer signal magnitude",
                       loc="left", fontsize=10)
    ax_right.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax_right.legend(loc="lower left", frameon=False, fontsize=8.5)

    fig.suptitle("Fig. 5. Mechanism attribution: which pipeline stage explains noise-induced control failure",
                 x=0.01, y=1.04, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Figure 6 — Pareto frontier: (clean, px+g 0.08) per (task, std_max)
# Each task gets a connected curve; markers coloured by std_max.
# ============================================================================

def fig6_pareto(out_path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    colors = {"TwoRoom": "#4477AA", "PushT": "#EE6677",
              "Reacher": "#228833", "Cube": "#CCBB44"}
    markers = {"TwoRoom": "o", "PushT": "s", "Reacher": "^", "Cube": "D"}

    for task in ["TwoRoom", "PushT", "Reacher", "Cube"]:
        xs = SWEEP[task]["clean"]
        ys = SWEEP[task]["px08"]
        ax.plot(xs, ys, "-", color=colors[task], linewidth=1.0, alpha=0.6)
        # mark base (std=0) separately
        ax.scatter(xs[0], ys[0], s=90, marker=markers[task],
                   facecolor="white", edgecolor=colors[task],
                   linewidth=1.5, zorder=4,
                   label=f"{task} (base, σ=0)")
        # mark sweep points
        for x, y, s in zip(xs[1:], ys[1:], SWEEP_STDS[1:]):
            ax.scatter(x, y, s=55, marker=markers[task],
                       color=colors[task], alpha=0.55 + 0.05 * SWEEP_STDS[1:].index(s),
                       edgecolor="black", linewidth=0.3, zorder=3)
        # mark per-task best (the configuration that's reported)
        best_std = EVAL_BEST[task]["std"]
        if best_std in SWEEP_STDS:
            i = SWEEP_STDS.index(best_std)
            ax.scatter(xs[i], ys[i], s=180, marker=markers[task],
                       facecolor="none", edgecolor=colors[task],
                       linewidth=2.0, zorder=5)

    # Diagonal y=x — ckpts on or above this diagonal are "robust ≥ clean"
    lo, hi = 0, 100
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, alpha=0.5)
    ax.text(95, 97, "y = x", color="gray", fontsize=8, alpha=0.7,
            ha="right", va="bottom")

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Clean success rate (%)")
    ax.set_ylabel("OOD success rate (%)  —  px+g 0.08")
    ax.set_title("Fig. 6. Per-task Pareto trajectory of (clean, OOD) under noise sweep\n"
                 "(open marker = base; solid markers = std_max sweep; ringed marker = per-task best)",
                 loc="left", fontsize=10, pad=10)
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="assets/paper1_figs",
                    help="output directory (relative to repo root)")
    ap.add_argument("--data-root",
                    default="/home/ag/dataset/ag_data/data/world_model/quentinll",
                    help="root of ckpt directories (lewm-pusht/, lewm-cube/, ...)")
    ap.add_argument("--only", nargs="+", choices=["1", "2", "3", "4", "5", "6"],
                    help="render only these figures (default: all)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()

    _setup_style()
    print(f"Output dir: {out_dir}")
    print(f"Data root:  {data_root}")
    selected = set(args.only or ["1", "2", "3", "4", "5", "6"])

    if "1" in selected:
        fig1_hero(out_dir / "fig1_hero.png")
    if "2" in selected:
        fig2_sweep(out_dir / "fig2_sweep.png")
    if "3" in selected:
        fig3_scatter(out_dir / "fig3_scatter.png", data_root)
    if "4" in selected:
        fig4_radar(out_dir / "fig4_radar.png")
    if "5" in selected:
        fig5_mechanism(out_dir / "fig5_mechanism.png")
    if "6" in selected:
        fig6_pareto(out_dir / "fig6_pareto.png")

    print("done.")


if __name__ == "__main__":
    main()
