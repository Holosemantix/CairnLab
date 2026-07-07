from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from acpc_flow import (
    ResidualTransportHead,
    acpc_flow_loss_terms,
    cvar_loss,
    diagnostic_distance,
    sample_latent_noise,
)
from jepa import JEPA
from tools.acpc_flow.coverage_audit import (
    _add_predictor_levels,
    _amplification_metrics,
    _candidate_rank_bundle,
    _candidate_rank_metrics,
)


class DummyEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(5, dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        cls = self.proj(pixels)
        return SimpleNamespace(last_hidden_state=cls.unsqueeze(1))


class DummyPredictor(nn.Module):
    def forward(self, x, c):
        return x + 0.1 * c[..., : x.size(-1)]


def build_dummy_jepa(dim=4):
    return JEPA(
        encoder=DummyEncoder(dim),
        predictor=DummyPredictor(),
        action_encoder=nn.Linear(2, dim),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    )


def test_residual_transport_identity_at_zero_scale():
    z = torch.randn(2, 3, 4)
    head = ResidualTransportHead(dim=4, hidden_dim=8, scale_init=0.0)

    out = head(z)

    assert torch.allclose(out, z)


def test_latent_noise_shape_and_cvar_tail_mean():
    z = torch.randn(2, 3, 4)

    noise = sample_latent_noise(
        z,
        std_min=0.01,
        std_max=0.02,
        mode="token_std",
        relative=True,
    )

    assert noise.shape == z.shape
    assert torch.isfinite(noise).all()
    assert torch.allclose(cvar_loss(torch.tensor([1.0, 2.0, 3.0, 4.0]), q=0.5), torch.tensor(3.5))


def test_diagnostic_distance_modes_are_finite():
    a = torch.zeros(2, 3, 4)
    b = torch.ones(2, 3, 4)

    mean_loss = diagnostic_distance(a, b, tail_mode="mean")
    cvar = diagnostic_distance(a, b, normalize=torch.tensor(2.0), tail_mode="cvar", q=0.9)

    assert mean_loss.ndim == 0
    assert cvar.ndim == 0
    assert torch.isfinite(mean_loss)
    assert torch.isfinite(cvar)


def test_acpc_flow_loss_terms_support_all_modes():
    clean = torch.randn(2, 3, 4)
    clean_trans = clean.clone()
    transported = clean + 0.1
    pred_a = torch.randn(2, 3, 4)
    pred_b = pred_a + 0.2

    for mode in ("latent_z", "predictor", "diagnostic", "hybrid"):
        kwargs = {}
        if mode != "latent_z":
            kwargs = {
                "transported_pred": pred_a,
                "clean_pred": pred_b,
                "transition_scale": torch.tensor(1.0),
            }
        raw, terms = acpc_flow_loss_terms(
            mode=mode,
            clean_ctx=clean,
            clean_ctx_trans=clean_trans,
            transported_ctx=transported,
            diagnostic_tail_mode="cvar",
            **kwargs,
        )

        assert raw.ndim == 0
        assert torch.isfinite(raw)
        assert torch.isfinite(terms["identity_raw"])
        assert torch.isfinite(terms["latent_raw"])


def test_jepa_encode_emits_emb_trans_identity_when_disabled():
    model = build_dummy_jepa()
    batch = {
        "pixels": torch.randn(2, 3, 5),
        "action": torch.randn(2, 3, 2),
    }

    out = model.encode(batch)

    assert out["encoder_feat"].shape == (2, 3, 4)
    assert out["emb"].shape == (2, 3, 4)
    assert out["emb_trans"] is out["emb"]
    assert torch.allclose(out["emb_trans"], out["emb"])
    assert out["act_emb"].shape == (2, 3, 4)


def test_jepa_encode_uses_transport_head_and_backpropagates():
    model = build_dummy_jepa()
    model.acpc_flow_enabled = True
    model.acpc_flow_head = ResidualTransportHead(dim=4, hidden_dim=8, scale_init=1.0)
    with torch.no_grad():
        model.acpc_flow_head.net[-1].weight.zero_()
        model.acpc_flow_head.net[-1].bias.fill_(0.5)
    batch = {
        "pixels": torch.randn(2, 3, 5),
        "action": torch.randn(2, 3, 2),
    }

    out = model.encode(batch)
    loss = out["emb_trans"].sum()
    loss.backward()

    assert not torch.allclose(out["emb_trans"], out["emb"])
    assert model.acpc_flow_head.alpha.grad is not None


def test_coverage_audit_v2_predictor_levels_and_amplification():
    model = build_dummy_jepa()
    batch = {
        "pixels": torch.randn(3, 4, 5),
        "action": torch.randn(3, 4, 2),
    }

    clean = _add_predictor_levels(model, model.encode(dict(batch)), history_size=3)
    corrupt_batch = {"pixels": batch["pixels"] + 0.05, "action": batch["action"]}
    corrupt = _add_predictor_levels(model, model.encode(corrupt_batch), history_size=3)
    amp = _amplification_metrics(
        {k: clean[k][:, :3] for k in ("encoder_feat", "emb", "predictor_hidden", "pred_emb")},
        {k: corrupt[k][:, :3] for k in ("encoder_feat", "emb", "predictor_hidden", "pred_emb")},
    )

    assert clean["predictor_hidden"].shape == (3, 3, 4)
    assert clean["pred_emb"].shape == (3, 3, 4)
    assert "amp_P_q90" in amp
    assert "amp_B_q90" in amp
    assert "amp_R_q90" in amp


def test_coverage_audit_candidate_rank_metrics_identity_costs():
    clean = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    metrics = _candidate_rank_metrics(clean, clean.clone(), topk=2)

    assert abs(metrics["candidate_rank_spearman"] - 1.0) < 1e-6
    assert metrics["candidate_top1_flip_rate"] == 0.0
    assert metrics["candidate_topk_overlap_rate"] == 1.0


def test_coverage_audit_candidate_rank_bundle_smoke():
    model = build_dummy_jepa()
    batch = {
        "pixels": torch.randn(3, 5, 5),
        "action": torch.randn(3, 5, 2),
    }
    clean = _add_predictor_levels(model, model.encode(dict(batch)), history_size=2)
    corrupt_batch = {"pixels": batch["pixels"] + 0.01, "action": batch["action"]}
    corrupt = _add_predictor_levels(model, model.encode(corrupt_batch), history_size=2)

    bundle = _candidate_rank_bundle(
        model,
        batch,
        clean,
        corrupt,
        history_size=2,
        future_steps=3,
        random_action_trials=2,
        topk=2,
        std_grid=[0.01],
        noise_mode="token_std",
        seed=11,
    )

    assert bundle["computed"] is True
    assert "candidate_rank_spearman" in bundle["pixel"]
    assert "0.01" in bundle["synthetic_encoder_by_alpha"]
    assert bundle["synthetic_predictor_hidden"]["computed"] is False
