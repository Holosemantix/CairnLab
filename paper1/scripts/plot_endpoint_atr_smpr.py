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


def plot(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    y = list(range(len(TASKS)))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)
    specs = [
        ("atr_q90", "ATR q90", "lower is better", "log"),
        ("smpr_delta0", "SMPR", "higher is better", "linear"),
    ]
    colors = {"0.00": "#6c757d", "0.08": "#1f77b4"}
    labels = {"0.00": "LeWM-base", "0.08": r"LeWM+noise $\sigma_{\max}=0.08$"}

    for ax, (key, xlabel, subtitle, scale) in zip(axes, specs):
        for i, task in enumerate(TASKS):
            base_mean, base_std = _endpoint_stats(rows, task, "0.00", key)
            end_mean, end_std = _endpoint_stats(rows, task, "0.08", key)
            ax.plot([base_mean, end_mean], [i, i], color="#b7b7b7", lw=1.5, zorder=1)
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
                fmt="o",
                color=colors["0.08"],
                ecolor=colors["0.08"],
                elinewidth=1.0,
                capsize=2.0,
                ms=5.0,
                zorder=3,
                label=labels["0.08"] if i == 0 else None,
            )
        ax.set_xlabel(f"{xlabel} ({subtitle})")
        ax.grid(True, axis="x", alpha=0.25)
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xlim(0.05, 5.0)
        else:
            ax.set_xlim(-0.03, 1.04)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(TASKS)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
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
