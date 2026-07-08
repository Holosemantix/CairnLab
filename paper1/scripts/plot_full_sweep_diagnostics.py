#!/usr/bin/env python3
"""Plot Paper1 full-sweep diagnostic dynamics."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .utils_paper1_io import ROOT, TASKS, fnum, read_csv, safe_mean

DEFAULT_DIAGNOSTICS = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_FIG = ROOT / "paper1" / "figures" / "fig_full_sweep_diagnostics.png"
DEFAULT_REGION_FIG = ROOT / "paper1" / "figures" / "fig_full_sweep_diagnostic_region.png"


def _by_task(rows):
    out = defaultdict(list)
    for row in rows:
        out[row["task"]].append(row)
    return out


def _task_rhos(task_rows):
    return sorted({fnum(r["rho"]) for r in task_rows})


def _mean_by_rho(task_rows, key):
    out = []
    for rho in _task_rhos(task_rows):
        vals = [r[key] for r in task_rows if abs(fnum(r["rho"]) - rho) < 1e-12]
        out.append(safe_mean(vals))
    return out


def _rate_by_rho(task_rows, key):
    out = []
    for rho in _task_rhos(task_rows):
        vals = []
        for r in task_rows:
            if abs(fnum(r["rho"]) - rho) < 1e-12:
                vals.append(1.0 if str(r.get(key, "")).lower() == "true" else 0.0)
        out.append(safe_mean(vals))
    return out


def plot_dynamics(rows, out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), sharex=True)
    axes = axes.ravel()
    by_task = _by_task(rows)
    legend_handles = None
    legend_labels = None
    for ax, task in zip(axes, TASKS):
        trs = by_task[task]
        x = _task_rhos(trs)
        score = _mean_by_rho(trs, "obs_sigma_008_score")
        atr = _mean_by_rho(trs, "atr_normalized_q90")
        smpr_fail = [1.0 - v if math.isfinite(v) else math.nan for v in _mean_by_rho(trs, "smpr_delta0")]
        top1_fail = [1.0 - v if math.isfinite(v) else math.nan for v in _mean_by_rho(trs, "top1_agree")]
        recovery = _rate_by_rho(trs, "recovery_label")
        proxy = []
        for rho in x:
            vals = [1.0 if fnum(r["proxy_gap_q50q90"]) > 0 else 0.0 for r in trs if abs(fnum(r["rho"]) - rho) < 1e-12]
            proxy.append(safe_mean(vals))
        for rho, rec, prox in zip(x, recovery, proxy):
            if rec >= 0.5:
                ax.axvspan(rho - 0.0038, rho + 0.0038, color="#d9ead3", alpha=0.65, lw=0)
            if prox >= 0.5:
                ax.axvspan(rho - 0.0021, rho + 0.0021, color="#c9daf8", alpha=0.45, lw=0)
        ax.plot(x, score, color="#1f4e79", marker="o", lw=1.8, label="obs score")
        ax.set_title(task, fontsize=10)
        ax.set_ylabel("score")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(x, atr, color="#a61c00", marker="s", lw=1.3, label="ATR rel")
        ax2.plot(x, smpr_fail, color="#674ea7", marker="^", lw=1.3, ls="--", label="1-SMPR")
        ax2.plot(x, top1_fail, color="#38761d", marker="d", lw=1.0, ls=":", label="1-Top1Agree")
        ax2.set_ylim(0, 1.10)
        if ax in (axes[1], axes[3]):
            ax2.set_ylabel("diagnostic failure")
        if legend_handles is None:
            h1, _ = ax.get_legend_handles_labels()
            h2, _ = ax2.get_legend_handles_labels()
            legend_handles = [
                Patch(facecolor="#d9ead3", edgecolor="none", alpha=0.65, label="recovery band"),
                Patch(facecolor="#c9daf8", edgecolor="none", alpha=0.45, label="proxy gap > 0"),
                *h1,
                *h2,
            ]
            legend_labels = [h.get_label() for h in legend_handles]
    for ax in axes:
        ax.set_xlabel(r"training noise $\sigma_{\max}^{\mathrm{train}}$")
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center right",
            bbox_to_anchor=(0.99, 0.5),
            fontsize=8,
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0, 0.83, 1])
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


def plot_region(rows, out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0), sharex=True, sharey=True)
    axes = axes.ravel()
    by_task = _by_task(rows)
    for ax, task in zip(axes, TASKS):
        for row in by_task[task]:
            recovered = str(row.get("recovery_label", "")).lower() == "true"
            color = "#38761d" if recovered else "#a61c00"
            marker = "o" if int(float(row["training_seed"])) == 3072 else "s" if int(float(row["training_seed"])) == 3073 else "^"
            ax.scatter(fnum(row["atr_normalized_q90"]), fnum(row["smpr_delta0"]), s=26, alpha=0.78, c=color, marker=marker)
        ax.set_title(task, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.02, 1.08)
        ax.set_ylim(-0.02, 1.04)
    for ax in axes[2:]:
        ax.set_xlabel("normalized ATR q90")
    for ax in axes[::2]:
        ax.set_ylabel("SMPR")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--region-out", type=Path, default=DEFAULT_REGION_FIG)
    args = parser.parse_args()
    rows = read_csv(args.diagnostics)
    plot_dynamics(rows, args.out)
    plot_region(rows, args.region_out)
    print(f"wrote {args.out}")
    print(f"wrote {args.region_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
