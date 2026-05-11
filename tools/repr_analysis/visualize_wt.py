#!/usr/bin/env python3
"""
Offline w_t visualization for PushT.

Loads a checkpoint with action_gate, extracts per-token w_t / critical_t / gA,
and correlates them with task-structure proxies:
- action norm ||a_t||
- latent displacement ||z_{t+1} - z_t||

Outputs:
- scatter_wt_vs_action_norm.png
- scatter_wt_vs_latent_disp.png
- timeseries_example.png
- histogram_wt_by_action_norm_quartile.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stable_pretraining as spt
import stable_worldmodel as swm

from tools.repr_analysis.analyze_repr import (
    encode_sequences,
    infer_history_size,
    load_dataset_samples,
    load_model,
)
from utils import get_column_normalizer, get_img_preprocessor


def compute_action_gate_metrics_offline(
    model,
    ctx_emb: torch.Tensor,
    ctx_action_raw: torch.Tensor,
    pred_emb_clean: torch.Tensor,
    *,
    delta_scale: float = 0.25,
    num_delta_samples: int = 4,
    delta_norm_floor: float = 1e-6,
    log_a_floor: float = 1e-8,
    w_min: float = 0.2,
    w_max: float = 1.0,
    ema_momentum: float = 0.99,
):
    """
    Reproduce train.py::compute_action_gate_metrics in eval mode.
    Assumes model already has gate_* EMA buffers from training.
    For offline viz we do NOT update EMA buffers (in_warmup=True logic).
    """
    with torch.no_grad():
        ctx_emb_d = ctx_emb.detach()
        pred_clean_d = pred_emb_clean.detach()
        B, T_ctx = ctx_emb_d.shape[:2]

        action_std = ctx_action_raw.float().std(dim=(0, 1), unbiased=False).clamp(min=1e-6)

        # Freeze BN during perturbation forwards (same as training)
        bn_states = []
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training:
                bn_states.append(m)
                m.eval()
        try:
            A_samples = []
            for _ in range(num_delta_samples):
                delta = torch.randn_like(ctx_action_raw) * (delta_scale * action_std)
                act_pert = ctx_action_raw + delta
                act_emb_pert = model.action_encoder(act_pert)
                pred_pert = model.predict(ctx_emb_d, act_emb_pert)
                diff = (pred_pert - pred_clean_d).pow(2).sum(dim=-1).clamp(min=0).sqrt()
                delta_norm = delta.pow(2).sum(dim=-1).clamp(min=0).sqrt().clamp(min=delta_norm_floor)
                A_samples.append(diff / delta_norm)
        finally:
            for m in bn_states:
                m.train()

        A_stack = torch.stack(A_samples, dim=0)  # (K, B, T_ctx)
        A_mean = A_stack.mean(dim=0)              # (B, T_ctx)
        log_A = torch.log(A_mean.clamp(min=log_a_floor))

        def _zscore(x: torch.Tensor, name: str) -> torch.Tensor:
            inited = getattr(model, f"gate_{name}_inited").item() > 0.5
            if inited:
                m = getattr(model, f"gate_{name}_mean")
                v = getattr(model, f"gate_{name}_var")
            else:
                m = x.mean()
                v = x.var(unbiased=False)
            return (x - m) / v.clamp(min=1e-6).sqrt()

        gA = torch.sigmoid(_zscore(log_A, "log_A"))

        # For A_t-only (hetero disabled) s_t is None
        critical = gA * 0.5
        w_t = w_max - (w_max - w_min) * critical

    return {
        "w_t": w_t,              # (B, T_ctx)
        "critical": critical,    # (B, T_ctx)
        "gA": gA,                # (B, T_ctx)
        "A_mean": A_mean,        # (B, T_ctx)
        "log_A": log_A,          # (B, T_ctx)
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline w_t visualization for PushT")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model object checkpoint")
    p.add_argument("--dataset", type=str, default="pusht_expert_train", help="Dataset name")
    p.add_argument("--n-sequences", type=int, default=256, help="Number of sequences to sample")
    p.add_argument("--future-steps", type=int, default=8, help="Future steps for loading data")
    p.add_argument("--frameskip", type=int, default=5, help="Frameskip")
    p.add_argument("--img-size", type=int, default=224, help="Image size")
    p.add_argument("--seed", type=int, default=3072, help="Random seed")
    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument("--save-dir", type=str, default="assets/diagnostics", help="Output directory for figures")
    return p


def main():
    args = build_parser().parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    model = load_model(args.ckpt, device)
    model.eval().requires_grad_(False)
    history_size = infer_history_size(model)

    print(f"[visualize_wt] model loaded, history_size={history_size}")

    batch = load_dataset_samples(
        dataset_name=args.dataset,
        state_key=None,
        n_sequences=args.n_sequences,
        history_size=history_size,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=args.seed,
        device=device,
    )
    outputs = encode_sequences(model, batch)

    emb = outputs["emb"]          # (B, T, D)
    action = outputs["action"]    # (B, T, A)
    B, T, D = emb.shape
    T_ctx = history_size  # gate is computed on the context window only

    # Context windows for gate computation
    ctx_emb = emb[:, :T_ctx, :]          # (B, T_ctx, D)
    ctx_action = action[:, :T_ctx, :]    # (B, T_ctx, A)
    # predictor output for the context window (model.predict predicts next step)
    pred_emb = model.predict(ctx_emb, outputs["act_emb"][:, :T_ctx, :])  # (B, T_ctx, D)

    gate_out = compute_action_gate_metrics_offline(
        model, ctx_emb, ctx_action, pred_emb,
        delta_scale=0.25,
        num_delta_samples=4,
        delta_norm_floor=1e-6,
        log_a_floor=1e-8,
        w_min=0.2,
        w_max=1.0,
    )

    w_t = gate_out["w_t"]                # (B, T_ctx)
    critical = gate_out["critical"]      # (B, T_ctx)
    gA = gate_out["gA"]                  # (B, T_ctx)
    A_mean = gate_out["A_mean"]          # (B, T_ctx)

    # Task-structure proxies
    action_norm = action[:, :T_ctx, :].pow(2).sum(dim=-1).sqrt()  # (B, T_ctx)
    latent_disp = (emb[:, 1:T_ctx+1, :] - emb[:, :T_ctx, :]).pow(2).sum(dim=-1).sqrt()  # (B, T_ctx)

    # Flatten for plotting
    w_flat = w_t.cpu().numpy().reshape(-1)
    crit_flat = critical.cpu().numpy().reshape(-1)
    gA_flat = gA.cpu().numpy().reshape(-1)
    A_flat = A_mean.cpu().numpy().reshape(-1)
    an_flat = action_norm.cpu().numpy().reshape(-1)
    ld_flat = latent_disp.cpu().numpy().reshape(-1)

    # Mask out any NaNs
    mask = np.isfinite(w_flat) & np.isfinite(an_flat) & np.isfinite(ld_flat)
    w_flat = w_flat[mask]
    crit_flat = crit_flat[mask]
    gA_flat = gA_flat[mask]
    A_flat = A_flat[mask]
    an_flat = an_flat[mask]
    ld_flat = ld_flat[mask]

    print(f"[visualize_wt] tokens after masking: {len(w_flat)}")
    print(f"  w_t mean={w_flat.mean():.3f} std={w_flat.std():.3f}")
    print(f"  action_norm mean={an_flat.mean():.3f} std={an_flat.std():.3f}")
    print(f"  corr(w_t, action_norm) = {np.corrcoef(w_flat, an_flat)[0,1]:.3f}")
    print(f"  corr(w_t, latent_disp) = {np.corrcoef(w_flat, ld_flat)[0,1]:.3f}")

    # ------------------------------------------------------------------
    # Figure 1: w_t vs action norm scatter
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    hexbin = ax.hexbin(an_flat, w_flat, gridsize=50, cmap="YlOrRd", mincnt=1, alpha=0.85)
    ax.set_xlabel(r"Action norm $||a_t||$", fontsize=11)
    ax.set_ylabel(r"Adaptive weight $w_t$", fontsize=11)
    ax.set_title("PushT: $w_t$ vs action norm (A_t-only consist001)", fontsize=12)
    ax.set_ylim(0.15, 1.05)
    cbar = fig.colorbar(hexbin, ax=ax)
    cbar.set_label("Count", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_dir / "wt_vs_action_norm.png", dpi=200)
    plt.close(fig)
    print(f"[visualize_wt] saved {save_dir / 'wt_vs_action_norm.png'}")

    # ------------------------------------------------------------------
    # Figure 2: w_t vs latent displacement scatter
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    hexbin = ax.hexbin(ld_flat, w_flat, gridsize=50, cmap="YlOrRd", mincnt=1, alpha=0.85)
    ax.set_xlabel(r"Latent displacement $||z_{t+1} - z_t||$", fontsize=11)
    ax.set_ylabel(r"Adaptive weight $w_t$", fontsize=11)
    ax.set_title("PushT: $w_t$ vs latent displacement", fontsize=12)
    ax.set_ylim(0.15, 1.05)
    cbar = fig.colorbar(hexbin, ax=ax)
    cbar.set_label("Count", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_dir / "wt_vs_latent_disp.png", dpi=200)
    plt.close(fig)
    print(f"[visualize_wt] saved {save_dir / 'wt_vs_latent_disp.png'}")

    # ------------------------------------------------------------------
    # Figure 3: timeseries example (first 3 sequences)
    # ------------------------------------------------------------------
    n_show = min(3, B)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        ax = axes[i]
        t = np.arange(T_ctx)
        ax.plot(t, w_t[i].cpu().numpy(), label=r"$w_t$", color="C0", linewidth=1.5)
        ax_twin = ax.twinx()
        ax_twin.plot(t, action_norm[i].cpu().numpy(), label=r"$||a_t||$", color="C1", linewidth=1.5, alpha=0.7)
        ax.set_ylabel(r"$w_t$", color="C0")
        ax_twin.set_ylabel(r"$||a_t||$", color="C1")
        ax.set_ylim(0.15, 1.05)
        ax.set_title(f"Sequence {i}")
        if i == n_show - 1:
            ax.set_xlabel("Time step $t$")
        # combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_dir / "wt_timeseries_example.png", dpi=200)
    plt.close(fig)
    print(f"[visualize_wt] saved {save_dir / 'wt_timeseries_example.png'}")

    # ------------------------------------------------------------------
    # Figure 4: histogram of w_t by action-norm quartile
    # ------------------------------------------------------------------
    q25, q50, q75 = np.percentile(an_flat, [25, 50, 75])
    masks = {
        "Q1 (low)": an_flat <= q25,
        "Q2": (an_flat > q25) & (an_flat <= q50),
        "Q3": (an_flat > q50) & (an_flat <= q75),
        "Q4 (high)": an_flat > q75,
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ca02c", "#98df8a", "#ff7f0e", "#d62728"]
    for (label, m), color in zip(masks.items(), colors):
        ax.hist(w_flat[m], bins=30, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel(r"Adaptive weight $w_t$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("PushT: $w_t$ distribution by action-norm quartile", fontsize=12)
    ax.legend(title="Action norm quartile", fontsize=9)
    ax.set_xlim(0.15, 1.05)
    plt.tight_layout()
    fig.savefig(save_dir / "wt_histogram_by_action_norm.png", dpi=200)
    plt.close(fig)
    print(f"[visualize_wt] saved {save_dir / 'wt_histogram_by_action_norm.png'}")

    # Print summary statistics by quartile
    print("\n[visualize_wt] w_t mean by action-norm quartile:")
    for label, m in masks.items():
        print(f"  {label}: mean={w_flat[m].mean():.3f}, std={w_flat[m].std():.3f}, n={m.sum()}")

    print("\n[visualize_wt] Done.")


if __name__ == "__main__":
    main()
