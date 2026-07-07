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


class DummyEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(5, dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        cls = self.proj(pixels)
        return SimpleNamespace(last_hidden_state=cls.unsqueeze(1))


def build_dummy_jepa(dim=4):
    return JEPA(
        encoder=DummyEncoder(dim),
        predictor=nn.Identity(),
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
