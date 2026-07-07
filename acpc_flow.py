import math

import torch
from torch import nn


class ResidualTransportHead(nn.Module):
    """Small residual latent transport head for ACPC-Flow."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 32,
        scale_init: float = 0.0,
        norm: str = "layernorm",
    ):
        super().__init__()
        norm = norm.lower()
        if norm in {"layernorm", "ln"}:
            norm_layer = nn.LayerNorm(dim)
        elif norm in {"none", "identity"}:
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"Unsupported ACPC-Flow transport norm: {norm}")
        self.net = nn.Sequential(
            norm_layer,
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(dtype=z.dtype, device=z.device)
        return z + alpha * self.net(z)


def token_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"token_mse shape mismatch: {a.shape} vs {b.shape}")
    return (a - b).pow(2).mean(dim=-1)


def sample_latent_noise(
    z: torch.Tensor,
    *,
    std_min: float,
    std_max: float,
    mode: str = "token_std",
    relative: bool = True,
    sample_per_token: bool = True,
) -> torch.Tensor:
    if std_min < 0.0 or std_max < 0.0:
        raise ValueError("ACPC-Flow noise std_min/std_max must be non-negative")
    if std_min > std_max:
        raise ValueError("ACPC-Flow noise std_min must be <= std_max")
    if std_max == 0.0:
        return torch.zeros_like(z)

    mode = mode.lower()
    if mode == "token_std":
        base_scale = z.detach().std(dim=-1, unbiased=False, keepdim=True)
    elif mode == "rms":
        base_scale = z.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
    elif mode == "fixed":
        base_scale = torch.ones_like(z[..., :1])
    else:
        raise ValueError(f"Unsupported ACPC-Flow noise mode: {mode}")
    if not relative:
        base_scale = torch.ones_like(base_scale)
    base_scale = base_scale.clamp_min(1e-8)

    std_shape = z.shape[:-1] + (1,) if sample_per_token else (z.shape[0],) + (1,) * (z.ndim - 1)
    if std_min == std_max:
        sampled_std = z.new_full(std_shape, float(std_max))
    else:
        sampled_std = torch.empty(std_shape, device=z.device, dtype=z.dtype)
        sampled_std.uniform_(float(std_min), float(std_max))
    return torch.randn_like(z) * sampled_std * base_scale


def cvar_loss(values: torch.Tensor, q: float = 0.90) -> torch.Tensor:
    if not 0.0 <= q < 1.0:
        raise ValueError(f"CVaR q must be in [0, 1), got {q}")
    flat = values.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("CVaR requires at least one value")
    tail_count = max(1, math.ceil((1.0 - float(q)) * flat.numel()))
    return flat.topk(tail_count, largest=True).values.mean()


def diagnostic_distance(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    *,
    normalize: torch.Tensor | None = None,
    tail_mode: str = "mean",
    q: float = 0.90,
) -> torch.Tensor:
    values = token_mse(pred_a, pred_b)
    if normalize is not None:
        values = values / normalize.detach().clamp_min(1e-6)
    tail_mode = tail_mode.lower()
    if tail_mode == "mean":
        return values.mean()
    if tail_mode == "cvar":
        return cvar_loss(values, q=q)
    raise ValueError(f"Unsupported diagnostic tail_mode: {tail_mode}")


def acpc_flow_loss_terms(
    *,
    mode: str,
    clean_ctx: torch.Tensor,
    clean_ctx_trans: torch.Tensor,
    transported_ctx: torch.Tensor,
    transported_pred: torch.Tensor | None = None,
    clean_pred: torch.Tensor | None = None,
    transition_scale: torch.Tensor | None = None,
    identity_weight: float = 0.1,
    diagnostic_tail_mode: str = "mean",
    diagnostic_q: float = 0.90,
    hybrid_latent_weight: float = 0.1,
    hybrid_acpc_weight: float = 1.0,
    detach_clean_pred: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mode = mode.lower()
    identity_raw = token_mse(clean_ctx_trans, clean_ctx.detach()).mean()
    latent_raw = token_mse(transported_ctx, clean_ctx.detach()).mean()
    terms: dict[str, torch.Tensor] = {
        "identity_raw": identity_raw,
        "latent_raw": latent_raw,
    }

    if mode == "latent_z":
        variant_raw = latent_raw
    else:
        if transported_pred is None or clean_pred is None:
            raise ValueError(f"ACPC-Flow mode={mode} requires predictor outputs")
        clean_pred_target = clean_pred.detach() if detach_clean_pred else clean_pred
        pred_raw = token_mse(transported_pred, clean_pred_target).mean()
        diag_raw = diagnostic_distance(
            transported_pred,
            clean_pred_target,
            normalize=transition_scale,
            tail_mode=diagnostic_tail_mode,
            q=diagnostic_q,
        )
        terms["pred_raw"] = pred_raw
        terms["diagnostic_raw"] = diag_raw
        if mode == "predictor":
            variant_raw = pred_raw
        elif mode == "diagnostic":
            variant_raw = diag_raw
        elif mode == "hybrid":
            variant_raw = hybrid_latent_weight * latent_raw + hybrid_acpc_weight * diag_raw
        else:
            raise ValueError(f"Unsupported ACPC-Flow mode: {mode}")

    raw = variant_raw + float(identity_weight) * identity_raw
    terms["variant_raw"] = variant_raw
    terms["raw"] = raw
    return raw, terms
