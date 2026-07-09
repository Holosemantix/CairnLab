#!/usr/bin/env python3
"""Plot endpoint ATR/SMPR selective-ACPC diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils_paper1_io import ROOT, TASKS, fnum, read_csv, safe_mean, safe_pstdev

DEFAULT_DIAGNOSTICS = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_FIG = ROOT / "assets" / "paper1_figs" / "fig_endpoint_atr_smpr.png"


def _endpoint_stats(rows: list[dict[str, str]], task: str, rho: str, key: str) -> tuple[float, float]:
    vals = [fnum(r[key]) for r in rows if r["task"] == task and f"{fnum(r['rho']):.2f}" == rho]
    return safe_mean(vals), safe_pstdev(vals)


def _direction_arrow(ax: plt.Axes, direction: str, text: str) -> None:
    if direction == "left":
        xy, xytext = (0.18, 0.91), (0.50, 0.91)
    else:
        xy, xytext = (0.82, 0.91), (0.50, 0.91)
    ax.annotate(
        "",
        xy=xy,
        xytext=xytext,
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
    )
    ax.text(0.50, 0.925, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


def plot(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    y = list(range(len(TASKS)))
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.75), sharey=True)
    specs = [
        ("atr_q90", "(a) Radius tail contracts: ATR q90", "lower is better", "log", "left"),
        ("smpr_delta0", "(b) Guard improves: SMPR", "higher is better", "linear", "right"),
    ]
    colors = {"0.00": "#6c757d", "0.08": "#1f77b4"}
    labels = {"0.00": "base", "0.08": "noise-trained"}

    for ax, (key, title, direction_text, scale, direction) in zip(axes, specs):
        for i, task in enumerate(TASKS):
            base_mean, base_std = _endpoint_stats(rows, task, "0.00", key)
            end_mean, end_std = _endpoint_stats(rows, task, "0.08", key)
            ax.plot([base_mean, end_mean], [i, i], color="#b7b7b7", lw=1.35, zorder=1)
            ax.errorbar(
                base_mean,
                i,
                xerr=base_std,
                fmt="o",
                color=colors["0.00"],
                ecolor=colors["0.00"],
                elinewidth=1.0,
                capsize=2.0,
                ms=5.0,
                zorder=2,
                label=labels["0.00"] if i == 0 else None,
            )
            ax.errorbar(
                end_mean,
                i,
                xerr=end_std,
                fmt="s",
                color=colors["0.08"],
                ecolor=colors["0.08"],
                elinewidth=1.0,
                capsize=2.0,
                ms=5.0,
                zorder=3,
                label=labels["0.08"] if i == 0 else None,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(direction_text)
        ax.grid(True, axis="x", alpha=0.24)
        _direction_arrow(ax, direction, direction_text)
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xlim(0.05, 5.0)
            ax.set_xticks([0.05, 0.1, 0.3, 1.0, 3.0])
            ax.set_xticklabels(["0.05", "0.1", "0.3", "1", "3"])
        else:
            ax.set_xlim(-0.03, 1.04)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(TASKS)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    plot(read_csv(args.diagnostics), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
