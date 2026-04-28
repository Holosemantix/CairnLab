"""
noise_sensitivity.py - Noise robustness diagnostics for LeWM / SWM embeddings.

Notebook use:

    from tools.repr_analysis.noise_sensitivity import run_noise_sensitivity, format_noise_table

    rows = run_noise_sensitivity(
        models={"swm": "/path/to/swm/model_object.ckpt", "lewm": "/path/to/lewm/model_object.ckpt"},
        dataset="tworoom",
        stds=[0.0, 0.01, 0.03, 0.05],
    )
    format_noise_table(rows)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from tools.repr_analysis.analyze_repr import (
    encode_sequences,
    get_embedding_space,
    get_model_spaces,
    infer_history_size,
    load_dataset_samples,
    load_model,
    resolve_space_name,
    to_serializable,
)
from utils import AddNormalizedGaussianNoise


def _clone_batch(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def _add_eval_noise(x: torch.Tensor, std: float, seed: int) -> torch.Tensor:
    if std <= 0:
        return x.clone()
    with torch.random.fork_rng(devices=[x.device] if x.device.type == "cuda" else []):
        torch.manual_seed(seed)
        return AddNormalizedGaussianNoise(std)(x)


def _select_frames(z: torch.Tensor, frame_scope: str) -> torch.Tensor:
    if frame_scope == "goal":
        return z[:, -1]
    if frame_scope == "all":
        return z.reshape(-1, z.size(-1))
    raise ValueError(f"Unsupported frame_scope: {frame_scope}")


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float().cpu(), q))


def _pairwise_reference(z: torch.Tensor) -> Dict[str, float]:
    if z.size(0) < 2:
        return {
            "clean_nn_cos_dist_median": float("nan"),
            "clean_pair_cos_dist_median": float("nan"),
            "clean_nn_l2_median": float("nan"),
            "clean_pair_l2_median": float("nan"),
        }

    z_norm = F.normalize(z, dim=-1, eps=1e-8)
    cos_dist = 1.0 - z_norm @ z_norm.T
    l2_dist = torch.cdist(z, z, p=2)
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    cos_offdiag = cos_dist[~eye].clamp_min(0.0)
    l2_offdiag = l2_dist[~eye]
    cos_nn = cos_dist.masked_fill(eye, float("inf")).min(dim=1).values.clamp_min(0.0)
    l2_nn = l2_dist.masked_fill(eye, float("inf")).min(dim=1).values

    return {
        "clean_nn_cos_dist_median": _safe_quantile(cos_nn, 0.5),
        "clean_pair_cos_dist_median": _safe_quantile(cos_offdiag, 0.5),
        "clean_nn_l2_median": _safe_quantile(l2_nn, 0.5),
        "clean_pair_l2_median": _safe_quantile(l2_offdiag, 0.5),
    }


def _shift_metrics(clean: torch.Tensor, noisy: torch.Tensor) -> Dict[str, float]:
    clean_norm = F.normalize(clean, dim=-1, eps=1e-8)
    noisy_norm = F.normalize(noisy, dim=-1, eps=1e-8)
    cos = (clean_norm * noisy_norm).sum(dim=-1).clamp(-1.0, 1.0)
    cos_dist = (1.0 - cos).clamp_min(0.0)
    angle_deg = torch.rad2deg(torch.acos(cos))
    l2_shift = torch.linalg.vector_norm(noisy - clean, dim=-1)

    return {
        "noise_cos_sim_mean": float(cos.mean()),
        "noise_cos_dist_median": _safe_quantile(cos_dist, 0.5),
        "noise_cos_dist_p90": _safe_quantile(cos_dist, 0.9),
        "noise_angle_deg_median": _safe_quantile(angle_deg, 0.5),
        "noise_angle_deg_p90": _safe_quantile(angle_deg, 0.9),
        "noise_l2_median": _safe_quantile(l2_shift, 0.5),
        "noise_l2_p90": _safe_quantile(l2_shift, 0.9),
        "clean_norm_mean": float(torch.linalg.vector_norm(clean, dim=-1).mean()),
        "noisy_norm_mean": float(torch.linalg.vector_norm(noisy, dim=-1).mean()),
    }


def _risk_label(ratio_median: float, ratio_p90: float) -> str:
    if ratio_median >= 1.0 or ratio_p90 >= 2.0:
        return "high"
    if ratio_median >= 0.5 or ratio_p90 >= 1.0:
        return "medium"
    return "low"


def analyze_model_noise(
    *,
    label: str,
    ckpt: str,
    batch: Mapping[str, torch.Tensor],
    stds: Sequence[float],
    embedding_space: str | None = None,
    seed: int = 3072,
    device: str = "cuda",
) -> list[Dict[str, Any]]:
    model = load_model(ckpt, device)
    spaces = get_model_spaces(model)
    space = resolve_space_name(embedding_space or spaces["inference_cost_space"])

    clean_outputs = encode_sequences(model, _clone_batch(batch))
    clean_z = get_embedding_space(clean_outputs, space).detach()

    rows: list[Dict[str, Any]] = []
    for std_idx, std in enumerate(stds):
        noisy_batch = _clone_batch(batch)
        noisy_batch["pixels"] = _add_eval_noise(
            noisy_batch["pixels"], float(std), seed + 1009 * std_idx
        )
        noisy_outputs = encode_sequences(model, noisy_batch)
        noisy_z = get_embedding_space(noisy_outputs, space).detach()

        for frame_scope in ("goal", "all"):
            clean_frame = _select_frames(clean_z, frame_scope)
            noisy_frame = _select_frames(noisy_z, frame_scope)
            shift = _shift_metrics(clean_frame, noisy_frame)
            ref = _pairwise_reference(clean_frame)

            cos_ratio = (
                shift["noise_cos_dist_median"] / ref["clean_nn_cos_dist_median"]
                if ref["clean_nn_cos_dist_median"] > 0
                else float("nan")
            )
            l2_ratio = (
                shift["noise_l2_median"] / ref["clean_nn_l2_median"]
                if ref["clean_nn_l2_median"] > 0
                else float("nan")
            )
            cos_ratio_p90 = (
                shift["noise_cos_dist_p90"] / ref["clean_nn_cos_dist_median"]
                if ref["clean_nn_cos_dist_median"] > 0
                else float("nan")
            )

            rows.append(
                {
                    "model": label,
                    "ckpt": ckpt,
                    "std": float(std),
                    "frame_scope": frame_scope,
                    "embedding_space": space,
                    "n_points": int(clean_frame.size(0)),
                    **shift,
                    **ref,
                    "noise_to_nn_cos_ratio_median": float(cos_ratio),
                    "noise_to_nn_cos_ratio_p90": float(cos_ratio_p90),
                    "noise_to_nn_l2_ratio_median": float(l2_ratio),
                    "risk": _risk_label(float(cos_ratio), float(cos_ratio_p90)),
                }
            )

    return rows


def run_noise_sensitivity(
    *,
    models: Mapping[str, str],
    dataset: str = "tworoom",
    stds: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05),
    state_key: str | None = None,
    n_sequences: int = 256,
    future_steps: int = 8,
    frameskip: int = 1,
    img_size: int = 224,
    embedding_space: str | None = None,
    seed: int = 3072,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[Dict[str, Any]]:
    if not models:
        raise ValueError("models must contain at least one label -> checkpoint path.")

    first_ckpt = next(iter(models.values()))
    first_model = load_model(first_ckpt, device)
    history_size = infer_history_size(first_model)
    del first_model

    batch = load_dataset_samples(
        dataset_name=dataset,
        state_key=state_key,
        n_sequences=n_sequences,
        history_size=history_size,
        future_steps=future_steps,
        frameskip=frameskip,
        img_size=img_size,
        seed=seed,
        device=device,
    )

    rows: list[Dict[str, Any]] = []
    for label, ckpt in models.items():
        rows.extend(
            analyze_model_noise(
                label=label,
                ckpt=ckpt,
                batch=batch,
                stds=stds,
                embedding_space=embedding_space,
                seed=seed,
                device=device,
            )
        )
    return rows


def format_noise_table(rows: Sequence[Mapping[str, Any]], frame_scope: str = "goal"):
    """Return a compact pandas table for notebook display."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("format_noise_table requires pandas.") from exc

    columns = [
        "model",
        "std",
        "frame_scope",
        "embedding_space",
        "noise_cos_sim_mean",
        "noise_angle_deg_median",
        "noise_angle_deg_p90",
        "clean_nn_cos_dist_median",
        "noise_to_nn_cos_ratio_median",
        "noise_to_nn_cos_ratio_p90",
        "noise_l2_median",
        "clean_nn_l2_median",
        "noise_to_nn_l2_ratio_median",
        "risk",
    ]
    df = pd.DataFrame(rows)
    df = df[df["frame_scope"] == frame_scope].copy()
    df = df[columns].sort_values(["model", "std"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


def _parse_model_specs(specs: Sequence[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be label=/path/to/ckpt, got: {spec}")
        label, ckpt = spec.split("=", 1)
        models[label.strip()] = ckpt.strip()
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose pixel-noise sensitivity in latent space.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as label=/path/to/model_object.ckpt. Repeat for comparisons.",
    )
    parser.add_argument("--dataset", default="tworoom")
    parser.add_argument("--stds", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--n-sequences", type=int, default=256)
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embedding-space", default=None, choices=[None, "raw", "normalized"])
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    rows = run_noise_sensitivity(
        models=_parse_model_specs(args.model),
        dataset=args.dataset,
        stds=args.stds,
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        embedding_space=args.embedding_space,
        seed=args.seed,
        device=args.device,
    )

    print(format_noise_table(rows, frame_scope="goal").to_string(index=False))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "noise_sensitivity.json").open("w") as f:
            json.dump(to_serializable(rows), f, indent=2)
        with (save_dir / "noise_sensitivity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[noise_sensitivity] saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
