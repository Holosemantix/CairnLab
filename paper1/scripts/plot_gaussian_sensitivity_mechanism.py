#!/usr/bin/env python3
"""Plot local Gaussian sensitivity mechanism diagnostics."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .utils_paper1_io import ROOT, TASKS, fnum, read_csv

DEFAULT_FD = ROOT / "paper1" / "results" / "gaussian_sensitivity_summary.csv"
DEFAULT_JVP = ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_summary.csv"
DEFAULT_MAIN_FIG = ROOT / "assets" / "paper1_figs" / "fig_gaussian_sensitivity_main.png"
DEFAULT_DECOMP_FIG = ROOT / "assets" / "paper1_figs" / "fig_jvp_trace_decomposition_heatmap.png"


def _row(rows: list[dict[str, str]], task: str, checkpoint_type: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["checkpoint_type"] == checkpoint_type:
            return row
    raise KeyError((task, checkpoint_type))


def _log10_ratio(value: float) -> float:
    return math.log10(max(value, 1e-6))


def plot_main(fd_rows: list[dict[str, str]], jvp_rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.75), sharey=True)
    y = list(range(len(TASKS)))
    panels = [
        (
            axes[0],
            [fnum(_row(fd_rows, task, "endpoint")["sensitivity_slope_vs_base"]) for task in TASKS],
            "(a) Finite-difference slope",
        ),
        (
            axes[1],
            [fnum(_row(jvp_rows, task, "endpoint")["composed_trace_per_pixel_dim_vs_base"]) for task in TASKS],
            "(b) JVP/Hutchinson composed trace",
        ),
    ]
    for ax, vals, title in panels:
        ax.barh(y, vals, color="#1f77b4", alpha=0.82)
        ax.axvline(1.0, color="#555555", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("endpoint/base ratio (lower is better)")
        ax.grid(True, axis="x", alpha=0.24)
        for yi, val in zip(y, vals):
            ax.text(max(val * 1.18, 1.35e-3), yi, f"{val:.3f}", va="center", fontsize=7.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(TASKS)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


def plot_decomposition(jvp_rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_heat, ax_align) = plt.subplots(1, 2, figsize=(8.8, 3.65), gridspec_kw={"width_ratios": [1.45, 0.95]})
    y = list(range(len(TASKS)))
    trace_cols = [
        ("encoder", "encoder_trace_per_pixel_dim_vs_base"),
        ("rollout", "rollout_trace_per_latent_dim_vs_base"),
        ("composed", "composed_trace_per_pixel_dim_vs_base"),
    ]
    matrix = []
    labels = []
    align_vals = []
    for task in TASKS:
        row = _row(jvp_rows, task, "endpoint")
        matrix.append([_log10_ratio(fnum(row[key])) for _label, key in trace_cols])
        labels.append([fnum(row[key]) for _label, key in trace_cols])
        align_vals.append(fnum(row["alignment_coefficient_vs_base"]))

    im = ax_heat.imshow(matrix, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5), aspect="auto")
    ax_heat.set_xticks(range(len(trace_cols)))
    ax_heat.set_xticklabels([label for label, _key in trace_cols], rotation=25, ha="right")
    ax_heat.set_yticks(y)
    ax_heat.set_yticklabels(TASKS)
    ax_heat.set_title("(a) Trace ratios", fontsize=10)
    for i, row in enumerate(labels):
        for j, val in enumerate(row):
            ax_heat.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=7.5)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\log_{10}$ endpoint/base", fontsize=8)

    ax_align.barh(y, align_vals, color="#4c78a8", alpha=0.82)
    ax_align.axvline(1.0, color="#555555", lw=1.0, ls="--")
    ax_align.set_yticks(y)
    ax_align.set_yticklabels([])
    ax_align.invert_yaxis()
    ax_align.set_title("(b) Alignment coefficient", fontsize=10)
    ax_align.set_xlabel("endpoint/base ratio")
    ax_align.grid(True, axis="x", alpha=0.24)
    for yi, val in zip(y, align_vals):
        ax_align.text(val + 0.03, yi, f"{val:.2f}", va="center", fontsize=7.5)
    ax_align.set_xlim(0.0, max(1.35, max(align_vals) * 1.18))

    fig.tight_layout()
    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite-diff", type=Path, default=DEFAULT_FD)
    parser.add_argument("--jvp", type=Path, default=DEFAULT_JVP)
    parser.add_argument("--out-main", type=Path, default=DEFAULT_MAIN_FIG)
    parser.add_argument("--out-decomp", type=Path, default=DEFAULT_DECOMP_FIG)
    args = parser.parse_args()
    fd_rows = read_csv(args.finite_diff)
    jvp_rows = read_csv(args.jvp)
    plot_main(fd_rows, jvp_rows, args.out_main)
    plot_decomposition(jvp_rows, args.out_decomp)
    print(f"wrote {args.out_main}")
    print(f"wrote {args.out_decomp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
