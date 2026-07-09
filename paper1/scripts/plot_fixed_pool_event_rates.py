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
MAIN_METRICS = [
    ("cert-pass", "(a) Cert-pass rate", "higher is better", "right"),
    ("top-1 flip", "(b) Top-1 flip rate", "lower is better", "left"),
]


def _lookup(rows: list[dict[str, str]], task: str, split: str, metric: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["split"] == split and row["metric"] == metric:
            return row
    raise KeyError((task, split, metric))


def _direction_arrow(ax: plt.Axes, direction: str, text: str) -> None:
    if direction == "left":
        xy, xytext = (0.18, 0.92), (0.50, 0.92)
    else:
        xy, xytext = (0.82, 0.92), (0.50, 0.92)
    ax.annotate(
        "",
        xy=xy,
        xytext=xytext,
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
    )
    ax.text(0.50, 0.935, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


def plot(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.85), sharey=True)
    colors = {"fragile": "#8c8c8c", "recovered": "#1f77b4"}
    markers = {"fragile": "o", "recovered": "s"}
    offsets = {"fragile": 0.13, "recovered": -0.13}
    y_base = list(range(len(TASK_ORDER)))

    for ax, (metric, title, direction_text, direction) in zip(axes, MAIN_METRICS):
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
                    fmt=markers[split],
                    color=colors[split],
                    ecolor=colors[split],
                    elinewidth=1.15,
                    capsize=2.0,
                    ms=4.8,
                    label=split if task_idx == 0 else None,
                )
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, axis="x", alpha=0.24)
        ax.set_xlabel("event rate")
        _direction_arrow(ax, direction, direction_text)

    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(TASK_ORDER)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].text(
        0.98,
        0.06,
        "flip | cert-pass = 0\nin sampled cert-pass anchors",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.25", alpha=0.9),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
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
