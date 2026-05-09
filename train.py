from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch import nn

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import (
    ARPredictor,
    Embedder,
    MLP,
    SIGReg,
    inverse_dynamics_loss,
    temporal_straightness,
    transition_distance_prediction_loss,
)
from utils import (
    get_column_normalizer,
    get_img_noise_transform,
    get_img_preprocessor,
    ModelObjectCallBack,
    TransformDataset,
)


def get_pred_loss_tensor(tensor: torch.Tensor, *, space: str) -> torch.Tensor:
    space = space.lower()
    if space == "raw":
        return tensor
    if space in {"normalized", "l2_norm", "sphere"}:
        return F.normalize(tensor, dim=-1, eps=1e-8)
    raise ValueError(f"Unsupported loss.pred.space: {space}")


def resolve_norm_fn(norm_name: str):
    """Resolve a config string to an nn norm class (or None for identity).
    Mirrors train_swm.resolve_norm_fn so both training paths accept the same
    encoder.projection_head.norm_fn vocabulary."""
    norm_name = norm_name.lower()
    if norm_name in {"none", "identity"}:
        return None
    if norm_name in {"ln", "layernorm"}:
        return nn.LayerNorm
    if norm_name in {"bn", "batchnorm", "batchnorm1d"}:
        return nn.BatchNorm1d
    raise ValueError(f"Unsupported encoder.projection_head.norm_fn: {norm_name}")


def compute_hetero_pred_loss(
    pred_loss_emb: torch.Tensor,
    tgt_loss_emb: torch.Tensor,
    logvar_hat: torch.Tensor,
    *,
    s_min: float = -4.0,
    s_max: float = 4.0,
    tau_floor: float = 1e-6,
):
    """Scale-preserving heteroscedastic prediction loss (sigma-conditioned JEPA).

    err_token  = mean((mu_hat - mu_target)^2, dim=-1)              # (B, T)
    s          = clamp(logvar_hat.squeeze(-1), s_min, s_max)        # (B, T)
    tau        = stop_grad(mean(err_token))                         # scalar (per-batch)
    hetero     = mean( exp(-s) * err_token + tau * s )

    At s ≡ 0 this equals plain MSE — so SIGReg's relative weight need not be
    retuned. After training, exp(-s) downweights high-error samples (the known
    hard-transition risk; monitor `hetero/weight_q10_q90_ratio` for this).

    Returns (hetero_loss, monitors_dict). The monitors dict carries logging
    fields only — none of them are part of the optimization graph.
    """
    err = (pred_loss_emb - tgt_loss_emb).pow(2).mean(dim=-1)        # (B, T)
    s = logvar_hat.squeeze(-1).clamp(min=s_min, max=s_max)          # (B, T)
    tau = err.detach().mean().clamp(min=tau_floor)
    weight = torch.exp(-s)                                          # exp(-s); large = upweight
    hetero_loss = (weight * err + tau * s).mean()

    with torch.no_grad():
        s_flat = s.reshape(-1)
        w_flat = weight.reshape(-1)
        e_flat = err.reshape(-1)
        # Spearman-ish: Pearson on rank — cheaper to just use Pearson on values
        # since exp/log are monotone, sign tells us the calibration direction.
        s_centered = s_flat - s_flat.mean()
        loge_centered = torch.log(e_flat.clamp(min=tau_floor)) - torch.log(
            e_flat.clamp(min=tau_floor)
        ).mean()
        denom = s_centered.norm() * loge_centered.norm()
        s_loge_corr = (s_centered * loge_centered).sum() / denom.clamp(min=1e-8)
        # Per-sample weight quantiles → measure of hard-transition downweighting.
        q10 = torch.quantile(w_flat, 0.1)
        q90 = torch.quantile(w_flat, 0.9)
        monitors = {
            "hetero_s_mean": s_flat.mean(),
            "hetero_s_std": s_flat.std(unbiased=False),
            "hetero_s_abs_max": s_flat.abs().max(),
            "hetero_weight_mean": w_flat.mean(),
            "hetero_weight_q10": q10,
            "hetero_weight_q90": q90,
            "hetero_weight_q10_q90_ratio": q10 / q90.clamp(min=1e-8),
            "hetero_tau": tau.detach(),
            "hetero_err_mean": e_flat.mean(),
            "hetero_s_logerr_corr": s_loge_corr,
        }
    return hetero_loss, monitors


def compute_sigma_probe_loss(
    pred_loss_emb: torch.Tensor,
    tgt_loss_emb: torch.Tensor,
    logvar_hat: torch.Tensor,
    *,
    s_min: float = -4.0,
    s_max: float = 4.0,
    tau_floor: float = 1e-6,
):
    """Detached sigma calibration loss.

    The prediction error target is detached, and callers should also detach the
    logvar head input so this loss only trains pred_logvar_proj. The mean path
    remains the exact LeWM MSE + SIGReg objective.
    """
    err = (pred_loss_emb.detach() - tgt_loss_emb.detach()).pow(2).mean(dim=-1)
    target_logerr = torch.log(err.clamp(min=tau_floor)).clamp(min=s_min, max=s_max)
    s = logvar_hat.squeeze(-1)
    sigma_probe_loss = F.smooth_l1_loss(s, target_logerr)

    with torch.no_grad():
        s_flat = s.reshape(-1)
        e_flat = err.reshape(-1)
        target_flat = target_logerr.reshape(-1)
        weight = torch.exp(-s.clamp(min=s_min, max=s_max))
        w_flat = weight.reshape(-1)
        s_centered = s_flat - s_flat.mean()
        loge_centered = target_flat - target_flat.mean()
        denom = s_centered.norm() * loge_centered.norm()
        s_loge_corr = (s_centered * loge_centered).sum() / denom.clamp(min=1e-8)
        q10 = torch.quantile(w_flat, 0.1)
        q90 = torch.quantile(w_flat, 0.9)
        monitors = {
            "hetero_s_mean": s_flat.mean(),
            "hetero_s_std": s_flat.std(unbiased=False),
            "hetero_s_abs_max": s_flat.abs().max(),
            "hetero_weight_mean": w_flat.mean(),
            "hetero_weight_q10": q10,
            "hetero_weight_q90": q90,
            "hetero_weight_q10_q90_ratio": q10 / q90.clamp(min=1e-8),
            "hetero_tau": e_flat.mean(),
            "hetero_err_mean": e_flat.mean(),
            "hetero_s_logerr_corr": s_loge_corr,
            "sigma_probe_target_logerr_mean": target_flat.mean(),
            "sigma_probe_target_logerr_std": target_flat.std(unbiased=False),
        }
    return sigma_probe_loss, monitors


def compute_temporal_hinge(output, *, model, cfg):
    """Upper hinge loss on consecutive latent pairs (LeWM variant).

    Mirrors the SWM compute_temporal_hinge exactly, except distance is
    computed with L2 (Euclidean) instead of cosine because LeWM does not
    L2-normalise its embeddings.
    """
    emb = output["emb"]
    if emb.size(1) <= 1:
        output["temporal_hinge_active_ratio"] = emb.new_tensor(0.0)
        return emb.new_tensor(0.0)

    hinge_cfg = cfg.loss.temporal_hinge
    dynamic_cfg = hinge_cfg.get("dynamic", {})
    z_t = emb[:, :-1]
    z_tp1 = emb[:, 1:]

    if not dynamic_cfg.get("enabled", False):
        dist = torch.linalg.vector_norm(z_tp1 - z_t, dim=-1)
        margin = hinge_cfg.margin
        hinge = torch.clamp_min(dist - margin, 0.0)
        output["temporal_hinge_active_ratio"] = (hinge > 0).float().mean()
        if hinge_cfg.squared:
            hinge = hinge.square()
        return hinge.mean()

    if not hasattr(model, "dynamic_margin_head"):
        raise AttributeError(
            "loss.temporal_hinge.dynamic.enabled=True requires "
            "model.dynamic_margin_head to be initialized"
        )

    act_emb = output["act_emb"][:, :-1]
    margin_input = torch.cat([z_t.detach(), act_emb.detach()], dim=-1)
    raw_score = model.dynamic_margin_head(margin_input).squeeze(-1)
    score = torch.sigmoid(raw_score)
    score = score / score.detach().mean().clamp_min(1e-6)

    margin = dynamic_cfg.get("base_margin", hinge_cfg.margin) * score
    margin = margin.clamp(
        min=dynamic_cfg.get("min_margin", 0.05),
        max=dynamic_cfg.get("max_margin", 1.0),
    )

    dist = torch.linalg.vector_norm(z_tp1 - z_t, dim=-1)
    hinge = torch.clamp_min(dist - margin, 0.0)
    output["temporal_hinge_active_ratio"] = (hinge > 0).float().mean()
    output["temporal_margin_mean"] = margin.mean()
    output["temporal_margin_std"] = margin.std(unbiased=False)
    margin_flat = margin.detach().float().flatten()
    output["temporal_margin_p10"] = torch.quantile(margin_flat, 0.10)
    output["temporal_margin_p50"] = torch.quantile(margin_flat, 0.50)
    output["temporal_margin_p90"] = torch.quantile(margin_flat, 0.90)

    if hinge_cfg.squared:
        hinge = hinge.square()
    return hinge.mean()


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    sigreg_lambd = cfg.loss.sigreg.weight
    hinge_cfg = cfg.loss.temporal_hinge

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]
    pred_cfg = cfg.loss.get("pred", {})
    pred_space = pred_cfg.get("space", "raw")

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    # Optional SimSiam-style stop-grad on target. Together with the existing
    # predictor head this turns the JEPA into an asymmetric architecture and
    # is the standard recipe for BN-free non-contrastive training (see Chen
    # & He, "Exploring Simple Siamese Representation Learning", CVPR 2021).
    # Without this asymmetry, dropping BN tends to collapse because pred and
    # target share gradient through the same encoder.
    if cfg.loss.get("target_stop_grad", False):
        tgt_emb = tgt_emb.detach()
    hetero_cfg = cfg.loss.get("hetero", {})
    hetero_enabled = bool(hetero_cfg.get("enabled", False))
    hetero_mode = hetero_cfg.get("mode", "loss").lower()
    if hetero_enabled and hetero_mode not in {"loss", "probe"}:
        raise ValueError(f"Unsupported loss.hetero.mode: {hetero_mode}")
    if hetero_enabled:
        pred_emb, logvar_hat = self.model.predict_with_logvar(
            ctx_emb,
            ctx_act,
            detach_logvar_input=(hetero_mode == "probe"),
        )
        if logvar_hat is None:
            raise RuntimeError(
                "loss.hetero.enabled=True requires model.pred_logvar_proj to be built"
            )
    else:
        pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred
        logvar_hat = None
    pred_loss_emb = get_pred_loss_tensor(pred_emb, space=pred_space)
    tgt_loss_emb = get_pred_loss_tensor(tgt_emb, space=pred_space)

    # LeWM loss, optional hetero replacement, or detached sigma probe.
    if hetero_enabled and hetero_mode == "loss":
        hetero_loss, hetero_monitors = compute_hetero_pred_loss(
            pred_loss_emb,
            tgt_loss_emb,
            logvar_hat,
            s_min=hetero_cfg.get("s_min", -4.0),
            s_max=hetero_cfg.get("s_max", 4.0),
            tau_floor=hetero_cfg.get("tau_floor", 1e-6),
        )
        output["pred_loss"] = hetero_loss
        # Also report the underlying MSE for direct comparability with the
        # LeWM baseline (loss curves stay readable when toggling hetero).
        output["pred_loss_mse_equiv"] = (pred_loss_emb - tgt_loss_emb).pow(2).mean().detach()
        for k, v in hetero_monitors.items():
            output[k] = v
    else:
        output["pred_loss"] = (pred_loss_emb - tgt_loss_emb).pow(2).mean()
        if hetero_enabled and hetero_mode == "probe":
            sigma_probe_loss, probe_monitors = compute_sigma_probe_loss(
                pred_loss_emb,
                tgt_loss_emb,
                logvar_hat,
                s_min=hetero_cfg.get("s_min", -4.0),
                s_max=hetero_cfg.get("s_max", 4.0),
                tau_floor=hetero_cfg.get("tau_floor", 1e-6),
            )
            output["sigma_probe_loss"] = sigma_probe_loss
            output["pred_loss_mse_equiv"] = output["pred_loss"].detach()
            for k, v in probe_monitors.items():
                output[k] = v
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["temporal_hinge_loss"] = compute_temporal_hinge(
        output, model=self.model, cfg=cfg
    )
    inverse_cfg = cfg.loss.get("inverse_dynamics", {})
    inverse_weight = inverse_cfg.get("weight", 0.0)
    if inverse_weight > 0.0:
        if not hasattr(self.model, "inverse_dynamics_head"):
            raise AttributeError(
                "loss.inverse_dynamics requires model.inverse_dynamics_head"
            )
        output["inverse_dynamics_loss"] = inverse_dynamics_loss(
            emb[:, :-1],
            emb[:, 1:],
            output["action"][:, :-1],
            self.model.inverse_dynamics_head,
            detach_input=inverse_cfg.get("detach_input", False),
        )

    dist_cfg = cfg.loss.get("transition_distance", {})
    dist_weight = dist_cfg.get("weight", 0.0)
    if dist_weight > 0.0:
        if not hasattr(self.model, "transition_distance_head"):
            raise AttributeError(
                "loss.transition_distance requires model.transition_distance_head"
            )
        (
            output["transition_distance_loss"],
            pred_dist,
            target_dist,
        ) = transition_distance_prediction_loss(
            emb[:, :-1],
            emb[:, 1:],
            self.model.transition_distance_head,
            metric=dist_cfg.get("metric", "l2"),
            detach_input=dist_cfg.get("detach_input", True),
        )
        output["transition_distance_pred_mean"] = pred_dist.mean()
        output["transition_distance_target_mean"] = target_dist.mean()
        output["transition_distance_target_std"] = target_dist.std(unbiased=False)

    output["loss"] = (
        output["pred_loss"]
        + sigreg_lambd * output["sigreg_loss"]
        + hinge_cfg.weight * output["temporal_hinge_loss"]
    )
    if "inverse_dynamics_loss" in output:
        output["loss"] = output["loss"] + inverse_weight * output["inverse_dynamics_loss"]
    if "transition_distance_loss" in output:
        output["loss"] = (
            output["loss"]
            + dist_weight * output["transition_distance_loss"]
        )
    if "sigma_probe_loss" in output:
        output["loss"] = (
            output["loss"]
            + hetero_cfg.get("probe_weight", 1.0) * output["sigma_probe_loss"]
        )
    output["temporal_straightness"] = temporal_straightness(emb)

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if (
            torch.is_tensor(v)
            and (
                k == "temporal_straightness"
                or k == "temporal_hinge_active_ratio"
                or k.startswith("temporal_margin_")
                or (
                    k.startswith("transition_distance_")
                    and not k.endswith("_loss")
                )
                or k.startswith("hetero_")
                or k.startswith("sigma_probe_")
                or k == "pred_loss_mse_equiv"
            )
        )
    }
    self.log_dict({**losses_dict, **metrics_dict}, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    img_noise = get_img_noise_transform(cfg.get("image_noise"))
    if img_noise is not None:
        train_set = TransformDataset(train_set, img_noise)
        if cfg.image_noise.get("apply_to_val", False):
            val_set = TransformDataset(val_set, img_noise)

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    head_cfg = cfg.get("encoder", {}).get("projection_head", {})
    proj_norm_name = head_cfg.get("norm_fn", "batchnorm1d")
    proj_hidden_dim = head_cfg.get("hidden_dim", 2048)
    proj_norm_fn = resolve_norm_fn(proj_norm_name)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden_dim,
        norm_fn=proj_norm_fn,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden_dim,
        norm_fn=proj_norm_fn,
    )

    hetero_cfg = cfg.loss.get("hetero", {})
    if hetero_cfg.get("enabled", False):
        # scalar log-variance head sharing the predictor backbone hidden state.
        # Adds ~0.5M params for hidden_dim=192 + hidden=2048; negligible.
        # Initialised to output 0 so the loss starts at MSE-equivalent.
        pred_logvar_head = MLP(
            input_dim=hidden_dim,
            output_dim=1,
            hidden_dim=hetero_cfg.get("logvar_hidden_dim", 256),
            norm_fn=proj_norm_fn,
        )
        # Zero the final linear so logvar starts at 0 (i.e. weight = exp(-0) = 1
        # everywhere; loss reduces to plain MSE on the first step).
        with torch.no_grad():
            final = pred_logvar_head.net[-1]
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()
    else:
        pred_logvar_head = None

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        pred_logvar_proj=pred_logvar_head,
    )
    if cfg.loss.temporal_hinge.get("dynamic", {}).get("enabled", False):
        world_model.dynamic_margin_head = nn.Linear(2 * embed_dim, 1)
        nn.init.zeros_(world_model.dynamic_margin_head.weight)
        nn.init.zeros_(world_model.dynamic_margin_head.bias)
    inverse_cfg = cfg.loss.get("inverse_dynamics", {})
    if inverse_cfg.get("weight", 0.0) > 0.0:
        world_model.inverse_dynamics_head = MLP(
            input_dim=2 * embed_dim,
            hidden_dim=inverse_cfg.get("hidden_dim", embed_dim),
            output_dim=effective_act_dim,
            norm_fn=None,
        )
    dist_cfg = cfg.loss.get("transition_distance", {})
    if dist_cfg.get("weight", 0.0) > 0.0:
        world_model.transition_distance_head = MLP(
            input_dim=2 * embed_dim,
            hidden_dim=dist_cfg.get("hidden_dim", embed_dim),
            output_dim=1,
            norm_fn=None,
        )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    backend = cfg.get("logger_backend", "swanlab")
    if backend == "swanlab" and cfg.swanlab.enabled:
        if SwanLabLogger is None:
            raise ImportError("swanlab is not installed. Run: pip install swanlab")
        logger = SwanLabLogger(**cfg.swanlab.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    elif backend == "wandb" and cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
