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
DEFAULT_FIG = ROOT / "assets" / "paper1_figs" / "fig_full_sweep_diagnostics.png"
DEFAULT_REGION_FIG = ROOT / "assets" / "paper1_figs" / "fig_full_sweep_diagnostic_region.png"
DEFAULT_PLANNER_FIG = ROOT / "assets" / "paper1_figs" / "fig_full_sweep_planner_guard.png"


def _by_task(rows: list[dict[str, str]]):
    out = defaultdict(list)
    for row in rows:
        out[row["task"]].append(row)
    return out


def _task_rhos(task_rows: list[dict[str, str]]) -> list[float]:
    return sorted({fnum(r["rho"]) for r in task_rows})


def _mean_by_rho(task_rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for rho in _task_rhos(task_rows):
        vals = [r[key] for r in task_rows if abs(fnum(r["rho"]) - rho) < 1e-12]
        out.append(safe_mean(vals))
    return out


def _rate_by_rho(task_rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for rho in _task_rhos(task_rows):
        vals = []
        for r in task_rows:
            if abs(fnum(r["rho"]) - rho) < 1e-12:
                vals.append(1.0 if str(r.get(key, "")).lower() == "true" else 0.0)
        out.append(safe_mean(vals))
    return out


def _proxy_positive_by_rho(task_rows: list[dict[str, str]]) -> list[float]:
    out = []
    for rho in _task_rhos(task_rows):
        vals = [1.0 if fnum(r["proxy_gap_q50q90"]) > 0 else 0.0 for r in task_rows if abs(fnum(r["rho"]) - rho) < 1e-12]
        out.append(safe_mean(vals))
    return out


def _shade_recovery(ax: plt.Axes, x: list[float], recovery: list[float]) -> None:
    for rho, rec in zip(x, recovery):
        if rec >= 0.5:
            ax.axvspan(rho - 0.0038, rho + 0.0038, color="#d9ead3", alpha=0.62, lw=0)


def plot_dynamics(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.35), sharex=True)
    axes = axes.ravel()
    by_task = _by_task(rows)
    legend_handles = None

    for ax, task in zip(axes, TASKS):
        trs = by_task[task]
        x = _task_rhos(trs)
        score = _mean_by_rho(trs, "obs_sigma_008_score")
        atr = _mean_by_rho(trs, "atr_normalized_q90")
        smpr_fail = [1.0 - v if math.isfinite(v) else math.nan for v in _mean_by_rho(trs, "smpr_delta0")]
        recovery = _rate_by_rho(trs, "recovery_label")

        _shade_recovery(ax, x, recovery)
        score_line, = ax.plot(x, score, color="#222222", marker="o", lw=1.8, ms=4.4, label="score")
        ax.set_title(task, fontsize=10)
        ax.set_ylabel("obs-noise score")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.22)

        ax2 = ax.twinx()
        atr_line, = ax2.plot(x, atr, color="#d95f02", marker="s", lw=1.35, ms=4.0, label="ATR rel")
        smpr_line, = ax2.plot(x, smpr_fail, color="#7570b3", marker="^", lw=1.35, ms=4.0, ls="--", label="1-SMPR")
        ax2.set_ylim(0, 1.10)
        if ax in (axes[1], axes[3]):
            ax2.set_ylabel("diagnostic failure")

        if legend_handles is None:
            legend_handles = [
                Patch(facecolor="#d9ead3", edgecolor="none", alpha=0.62, label="recovered rows"),
                score_line,
                atr_line,
                smpr_line,
            ]

    for ax in axes:
        ax.set_xlabel(r"training noise $\sigma_{\max}^{\mathrm{train}}$")
    if legend_handles is not None:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_fig, dpi=230)
    plt.close(fig)


def plot_planner_guard(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 6.2), sharex=True)
    axes = axes.ravel()
    by_task = _by_task(rows)
    legend_handles = None

    for ax, task in zip(axes, TASKS):
        trs = by_task[task]
        x = _task_rhos(trs)
        top1_fail = [1.0 - v if math.isfinite(v) else math.nan for v in _mean_by_rho(trs, "top1_agree")]
        proxy_pos = _proxy_positive_by_rho(trs)
        recovery = _rate_by_rho(trs, "recovery_label")
        _shade_recovery(ax, x, recovery)
        flip_line, = ax.plot(x, top1_fail, color="#b2182b", marker="d", lw=1.55, ms=4.0, label="top-1 flip")
        proxy_line, = ax.plot(x, proxy_pos, color="#2166ac", marker="s", lw=1.25, ms=3.8, ls="--", label="proxy gap > 0")
        ax.set_title(task, fontsize=10)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, alpha=0.22)
        ax.set_ylabel("event rate")
        if legend_handles is None:
            legend_handles = [Patch(facecolor="#d9ead3", edgecolor="none", alpha=0.62, label="recovered rows"), flip_line, proxy_line]

    for ax in axes:
        ax.set_xlabel(r"training noise $\sigma_{\max}^{\mathrm{train}}$")
    if legend_handles is not None:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_fig, dpi=230)
    plt.close(fig)


def plot_region(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0), sharex=True, sharey=True)
    axes = axes.ravel()
    by_task = _by_task(rows)
    for ax, task in zip(axes, TASKS):
        for row in by_task[task]:
            recovered = str(row.get("recovery_label", "")).lower() == "true"
            color = "#1f77b4" if recovered else "#8c8c8c"
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
    parser.add_argument("--planner-out", type=Path, default=DEFAULT_PLANNER_FIG)
    args = parser.parse_args()
    rows = read_csv(args.diagnostics)
    plot_dynamics(rows, args.out)
    plot_region(rows, args.region_out)
    plot_planner_guard(rows, args.planner_out)
    print(f"wrote {args.out}")
    print(f"wrote {args.region_out}")
    print(f"wrote {args.planner_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
