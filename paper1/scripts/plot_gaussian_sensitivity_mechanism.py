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
DEFAULT_FIG = ROOT / "assets" / "paper1_figs" / "fig_gaussian_sensitivity_mechanism.png"


def _row(rows: list[dict[str, str]], task: str, checkpoint_type: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["checkpoint_type"] == checkpoint_type:
            return row
    raise KeyError((task, checkpoint_type))


def _log10_ratio(value: float) -> float:
    return math.log10(max(value, 1e-6))


def plot(fd_rows: list[dict[str, str]], jvp_rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.8, 4.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.35)
    ax_fd = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[0, 1], sharey=ax_fd)
    ax_heat = fig.add_subplot(gs[0, 2])

    y = list(range(len(TASKS)))
    fd_vals = [fnum(_row(fd_rows, task, "endpoint")["sensitivity_slope_vs_base"]) for task in TASKS]
    comp_vals = [fnum(_row(jvp_rows, task, "endpoint")["composed_trace_per_pixel_dim_vs_base"]) for task in TASKS]

    for ax, vals, title in [
        (ax_fd, fd_vals, "finite-difference slope"),
        (ax_comp, comp_vals, "JVP/Hutchinson composed trace"),
    ]:
        ax.barh(y, vals, color="#1f77b4", alpha=0.82)
        ax.axvline(1.0, color="#555555", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("endpoint/base ratio")
        ax.grid(True, axis="x", alpha=0.25)
        for yi, val in zip(y, vals):
            ax.text(max(val * 1.15, 1.4e-3), yi, f"{val:.3f}", va="center", fontsize=7)

    ax_fd.set_yticks(y)
    ax_fd.set_yticklabels(TASKS)
    ax_fd.invert_yaxis()
    ax_comp.tick_params(axis="y", labelleft=False)

    heat_cols = [
        ("encoder", "encoder_trace_per_pixel_dim_vs_base"),
        ("rollout", "rollout_trace_per_latent_dim_vs_base"),
        ("composed", "composed_trace_per_pixel_dim_vs_base"),
        ("alignment", "alignment_coefficient_vs_base"),
    ]
    matrix = []
    labels = []
    for task in TASKS:
        row = _row(jvp_rows, task, "endpoint")
        matrix.append([_log10_ratio(fnum(row[key])) for _label, key in heat_cols])
        labels.append([fnum(row[key]) for _label, key in heat_cols])
    im = ax_heat.imshow(matrix, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5), aspect="auto")
    ax_heat.set_xticks(range(len(heat_cols)))
    ax_heat.set_xticklabels([label for label, _key in heat_cols], rotation=30, ha="right")
    ax_heat.set_yticks(y)
    ax_heat.set_yticklabels(TASKS)
    ax_heat.set_title("JVP/Hutchinson decomposition", fontsize=10)
    for i, row in enumerate(labels):
        for j, val in enumerate(row):
            ax_heat.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\log_{10}$ endpoint/base", fontsize=8)

    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite-diff", type=Path, default=DEFAULT_FD)
    parser.add_argument("--jvp", type=Path, default=DEFAULT_JVP)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    plot(read_csv(args.finite_diff), read_csv(args.jvp), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
