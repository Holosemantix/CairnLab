"""
latent_visualization.py — Project the latent embedding `emb` (and `emb_raw`
for SWM) of one or more checkpoints onto 2-D via PCA and save scatter +
trajectory PNGs. Companion to the rest of `run_full_diagnostics`: gives a
qualitative picture of "what the latent looks like" alongside the
quantitative noise / predictor / resolution / action_effect tables.

What it produces (per `--save-dir`):

    latent_pca_2d.png         scatter colored by frame index (T axis); one
                              panel per checkpoint; shared coordinates within
                              the figure (all ckpts PCA'd together to the same
                              2-D basis).
    latent_pca_2d_per_ckpt.png each ckpt fitted independently — shows the
                              shape of *its own* latent rather than the cross-
                              ckpt embedding.
    latent_trajectory.png     for the first 8 sequences, plot PCA-2D path
                              (T points connected) per ckpt; helps see whether
                              consecutive frames trace smooth curves vs noisy
                              clouds.
    latent_pca_data.npz       saved coordinates + sequence/frame indices so
                              users can re-render with a different style.

Standalone usage (one ckpt):

    python -m tools.repr_analysis.latent_visualization \\
        --model SWM-base=/path/to/...epoch_10_object.ckpt \\
        --dataset tworoom --frameskip 5 --save-dir <out>

The same entry point is also called from `run_full_diagnostics` when
`--skip-visualization` is not set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

from tools.repr_analysis.analyze_repr import (
    encode_sequences,
    get_embedding_space,
    get_model_spaces,
    infer_history_size,
    load_dataset_samples,
    load_model,
    resolve_space_name,
)


@torch.no_grad()
def _embed_ckpt(ckpt: str, batch: dict, device: str, embedding_space: str | None):
    model = load_model(ckpt, device)
    spaces = get_model_spaces(model)
    space = resolve_space_name(embedding_space or spaces["inference_cost_space"])
    cloned = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    outputs = encode_sequences(model, cloned)
    z = get_embedding_space(outputs, space).detach().cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return z, space


def _pca_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (proj_2d, components) for a (N, D) matrix x. Float64 to keep the
    eigendecomposition well-conditioned on small N."""
    xd = x.double()
    mu = xd.mean(dim=0, keepdim=True)
    xc = xd - mu
    # SVD-based PCA (top-2 components)
    u, s, vh = torch.linalg.svd(xc, full_matrices=False)
    comps = vh[:2]  # (2, D)
    proj = xc @ comps.T  # (N, 2)
    return proj.float(), comps.float()


def run_latent_visualization(
    *,
    models: Mapping[str, str],
    dataset: str,
    state_key: str | None = None,
    n_sequences: int = 64,
    history_size: int | None = None,
    future_steps: int = 8,
    frameskip: int = 1,
    img_size: int = 224,
    embedding_space: str | None = None,
    seed: int = 3072,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str | Path | None = None,
    log=print,
) -> dict:
    if not models:
        raise ValueError("models must contain at least one label -> ckpt path")

    first_ckpt = next(iter(models.values()))
    first_model = load_model(first_ckpt, device)
    H = history_size or infer_history_size(first_model)
    del first_model

    batch = load_dataset_samples(
        dataset_name=dataset,
        state_key=state_key,
        n_sequences=n_sequences,
        history_size=H,
        future_steps=future_steps,
        frameskip=frameskip,
        img_size=img_size,
        seed=seed,
        device=device,
    )

    embeddings: dict[str, torch.Tensor] = {}
    space_used: dict[str, str] = {}
    for label, ckpt in models.items():
        if log is not None:
            log(f"[viz] embedding {label}")
        z, sp = _embed_ckpt(ckpt, batch, device, embedding_space)
        embeddings[label] = z
        space_used[label] = sp

    # ---- Shared-basis PCA: stack all ckpts together, fit one 2-D space ----
    flat_per_ckpt = {k: v.reshape(-1, v.size(-1)) for k, v in embeddings.items()}
    # Only valid if all ckpts share latent dim — otherwise fall back to per-ckpt.
    dims = {v.size(-1) for v in flat_per_ckpt.values()}
    shared_proj: dict[str, torch.Tensor] | None = None
    if len(dims) == 1:
        all_x = torch.cat(list(flat_per_ckpt.values()), dim=0)
        proj, _ = _pca_2d(all_x)
        cursor = 0
        shared_proj = {}
        for label, v in flat_per_ckpt.items():
            n = v.size(0)
            shared_proj[label] = proj[cursor:cursor + n].reshape(*embeddings[label].shape[:-1], 2)
            cursor += n

    # Per-ckpt independent PCA for shape comparison
    indep_proj: dict[str, torch.Tensor] = {}
    for label, v in flat_per_ckpt.items():
        proj, _ = _pca_2d(v)
        indep_proj[label] = proj.reshape(*embeddings[label].shape[:-1], 2)

    # ---- Save artifacts ----
    save_dir_path = Path(save_dir) if save_dir is not None else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        # Pack coordinates
        npz_payload = {
            "labels": list(embeddings.keys()),
            "spaces": [space_used[k] for k in embeddings],
        }
        if shared_proj is not None:
            for k, v in shared_proj.items():
                npz_payload[f"shared_{k}"] = v.numpy()
        for k, v in indep_proj.items():
            npz_payload[f"indep_{k}"] = v.numpy()
        try:
            import numpy as np

            np.savez_compressed(save_dir_path / "latent_pca_data.npz", **npz_payload)
        except ImportError:
            (save_dir_path / "latent_pca_data.json").write_text(json.dumps({
                k: v if isinstance(v, list) else v.tolist() for k, v in npz_payload.items()
            }))

        if HAS_MPL:
            _render_panel(shared_proj, save_dir_path / "latent_pca_2d.png",
                          title="PCA-2D (shared basis across ckpts)") if shared_proj else None
            _render_panel(indep_proj, save_dir_path / "latent_pca_2d_per_ckpt.png",
                          title="PCA-2D (per-ckpt fit)")
            _render_trajectory(indep_proj, save_dir_path / "latent_trajectory.png",
                               n_seq=min(8, n_sequences))
        else:
            if log is not None:
                log("[viz] matplotlib not available; only .npz saved")

    return {
        "embeddings": embeddings,
        "shared_proj": shared_proj,
        "indep_proj": indep_proj,
        "spaces": space_used,
        "save_dir": save_dir_path,
    }


def _render_panel(proj_dict: Mapping[str, torch.Tensor], out_path: Path, title: str):
    """Scatter PCA-2D, one subpanel per ckpt, color = frame index along T."""
    n = len(proj_dict)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for ax, (label, p) in zip(axes.flat, proj_dict.items()):
        # p is (B, T, 2)
        b, t, _ = p.shape
        x = p[..., 0].reshape(-1).numpy()
        y = p[..., 1].reshape(-1).numpy()
        frame_idx = torch.arange(t).unsqueeze(0).expand(b, -1).reshape(-1).numpy()
        sc = ax.scatter(x, y, c=frame_idx, cmap="viridis", s=4, alpha=0.65)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="frame")
    for ax in axes.flat[len(proj_dict):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_trajectory(proj_dict: Mapping[str, torch.Tensor], out_path: Path, n_seq: int):
    n = len(proj_dict)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for ax, (label, p) in zip(axes.flat, proj_dict.items()):
        b, t, _ = p.shape
        for s in range(min(n_seq, b)):
            xy = p[s]  # (T, 2)
            ax.plot(xy[:, 0].numpy(), xy[:, 1].numpy(), "-", linewidth=0.7, alpha=0.7)
            ax.scatter(xy[:, 0].numpy(), xy[:, 1].numpy(),
                       c=torch.arange(t).numpy(), cmap="viridis", s=8, alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(proj_dict):]:
        ax.set_visible(False)
    fig.suptitle("Latent trajectories (first 8 seqs, color=frame)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_model_specs(specs: Sequence[str]) -> dict:
    out = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--model spec must be label=path, got: {s}")
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render latent PCA-2D for one or more ckpts.")
    p.add_argument("--model", action="append", required=True,
                   help="label=ckpt; repeat for multi-ckpt overlay/comparison.")
    p.add_argument("--dataset", default="tworoom")
    p.add_argument("--state-key", default=None)
    p.add_argument("--n-sequences", type=int, default=64)
    p.add_argument("--future-steps", type=int, default=8)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embedding-space", default=None, choices=[None, "raw", "normalized"])
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", required=True)
    return p


def main():
    args = build_parser().parse_args()
    run_latent_visualization(
        models=_parse_model_specs(args.model),
        dataset=args.dataset,
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        embedding_space=args.embedding_space,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
