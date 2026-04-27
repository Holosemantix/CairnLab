import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
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
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


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

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["temporal_hinge_loss"] = compute_temporal_hinge(
        output, model=self.model, cfg=cfg
    )
    inverse_cfg = cfg.loss.get("inverse_dynamics", {})
    inverse_enabled = (
        inverse_cfg.get("enabled", False)
        or inverse_cfg.get("weight", 0.0) > 0.0
    )
    if inverse_enabled:
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
    dist_enabled = (
        dist_cfg.get("enabled", False)
        or dist_cfg.get("weight", 0.0) > 0.0
    )
    if dist_enabled:
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
        output["loss"] = (
            output["loss"]
            + inverse_cfg.get("weight", 0.0) * output["inverse_dynamics_loss"]
        )
    if "transition_distance_loss" in output:
        output["loss"] = (
            output["loss"]
            + dist_cfg.get("weight", 0.0) * output["transition_distance_loss"]
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

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )
    if cfg.loss.temporal_hinge.get("dynamic", {}).get("enabled", False):
        world_model.dynamic_margin_head = nn.Linear(2 * embed_dim, 1)
        nn.init.zeros_(world_model.dynamic_margin_head.weight)
        nn.init.zeros_(world_model.dynamic_margin_head.bias)
    inverse_cfg = cfg.loss.get("inverse_dynamics", {})
    if inverse_cfg.get("enabled", False) or inverse_cfg.get("weight", 0.0) > 0.0:
        world_model.inverse_dynamics_head = MLP(
            input_dim=2 * embed_dim,
            hidden_dim=inverse_cfg.get("hidden_dim", embed_dim),
            output_dim=effective_act_dim,
            norm_fn=None,
        )
    dist_cfg = cfg.loss.get("transition_distance", {})
    if dist_cfg.get("enabled", False) or dist_cfg.get("weight", 0.0) > 0.0:
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
