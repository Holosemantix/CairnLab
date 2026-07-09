#!/usr/bin/env python3
"""Plot fixed-pool event-rate Wilson intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils_paper1_io import ROOT, fnum, read_csv

DEFAULT_CI = ROOT / "paper1" / "results" / "sample_level_event_rate_wilson_ci.csv"
DEFAULT_FIG = ROOT / "assets" / "paper1_figs" / "fig_fixed_pool_event_rates.png"
TASK_ORDER = ["TwoRoom", "PushT", "Reacher", "Cube", "ALL"]
METRICS = ["cert-pass", "top-1 flip", "top-1 flip | cert-pass"]


def _lookup(rows: list[dict[str, str]], task: str, split: str, metric: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["split"] == split and row["metric"] == metric:
            return row
    raise KeyError((task, split, metric))


def plot(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.25), sharey=True)
    colors = {"fragile": "#a61c00", "recovered": "#38761d"}
    offsets = {"fragile": 0.13, "recovered": -0.13}
    y_base = list(range(len(TASK_ORDER)))

    for ax, metric in zip(axes, METRICS):
        for task_idx, task in enumerate(TASK_ORDER):
            for split in ("fragile", "recovered"):
                row = _lookup(rows, task, split, metric)
                rate = fnum(row["rate"])
                low = fnum(row["wilson_low"])
                high = fnum(row["wilson_high"])
                y = task_idx + offsets[split]
                ax.errorbar(
                    rate,
                    y,
                    xerr=[[max(rate - low, 0.0)], [max(high - rate, 0.0)]],
                    fmt="o",
                    color=colors[split],
                    ecolor=colors[split],
                    elinewidth=1.2,
                    capsize=2.0,
                    ms=4.8,
                    label=split if task_idx == 0 else None,
                )
        ax.set_title(metric, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlabel("event rate")

    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(TASK_ORDER)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", type=Path, default=DEFAULT_CI)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    plot(read_csv(args.ci), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
