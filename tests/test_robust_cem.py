from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from robust_cem import RobustCEMSolver, aggregate_view_costs
from stable_worldmodel.solver import CEMSolver


class QuadraticCostModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def get_cost(self, info_dict, action_candidates):
        return action_candidates.square().sum(dim=(-1, -2)) * self.weight


class MutatingCostModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def get_cost(self, info_dict, action_candidates):
        info_dict["goal_emb"] = torch.ones(action_candidates.shape[:2])
        if "pixels" in info_dict:
            info_dict["pixels"].add_(1.0)
        return action_candidates.square().sum(dim=(-1, -2)) * self.weight


class SequentialCostModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = []

    def get_cost(self, info_dict, action_candidates):
        sample_count = action_candidates.shape[1]
        self.calls.append(sample_count)
        values = torch.arange(sample_count, device=action_candidates.device, dtype=action_candidates.dtype)
        return values.unsqueeze(0).expand(action_candidates.shape[0], -1) * self.weight


class CountingCostModel(QuadraticCostModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_cost(self, info_dict, action_candidates):
        self.calls += 1
        return super().get_cost(info_dict, action_candidates)


def _configured_solver(model, **kwargs):
    solver_kwargs = {
        "batch_size": 2,
        "num_samples": 6,
        "n_steps": 2,
        "topk": 2,
        "device": "cpu",
        "seed": 7,
    }
    solver_kwargs.update(kwargs)
    solver = RobustCEMSolver(model=model, **solver_kwargs)
    action_space = Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)
    solver.configure(
        action_space=action_space,
        n_envs=2,
        config=SimpleNamespace(horizon=3, action_block=1),
    )
    return solver


def test_standard_mode_shape_and_protocol():
    solver = _configured_solver(QuadraticCostModel(), enabled=False, num_views=1)
    info = {"pixels": torch.zeros(2, 1, 3, 4, 4)}

    out = solver.solve(info)

    assert out["actions"].shape == (2, 3, 2)
    assert len(out["costs"]) == 2
    assert out["robust"]["enabled"] is False


def test_robust_cost_penalizes_instability():
    view_costs = torch.tensor([[[0.0, 10.0, 10.0], [1.0, 1.0, 1.0]]])

    mean_std, _ = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="mean_std",
        beta=1.0,
    )
    worst, _ = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="worst",
    )

    assert int(mean_std.argmin(dim=1).item()) == 1
    assert int(worst.argmin(dim=1).item()) == 1


def test_no_mutation_leakage_to_original_info_dict():
    solver = _configured_solver(
        MutatingCostModel(),
        enabled=True,
        num_views=2,
        include_identity=True,
        view_type="gaussian_noise",
        view_std=0.0,
        score_mode="mean",
    )
    pixels = torch.zeros(2, 1, 3, 4, 4)
    info = {"pixels": pixels.clone()}

    solver.solve(info)

    assert "goal_emb" not in info
    assert torch.equal(info["pixels"], pixels)


def test_num_views_one_reduces_to_base_scoring():
    model = CountingCostModel()
    solver = _configured_solver(
        model,
        enabled=True,
        num_views=1,
        include_identity=True,
        score_mode="mean",
        n_steps=1,
    )
    info = {"pixels": torch.zeros(2, 1, 3, 4, 4)}

    out = solver.solve(info)

    assert out["robust"]["enabled"] is False
    assert model.calls == 1



def test_disabled_solver_matches_upstream_cem_for_same_seed():
    model = QuadraticCostModel()
    info = {"pixels": torch.zeros(2, 1, 3, 4, 4)}
    action_space = Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)
    config = SimpleNamespace(horizon=3, action_block=1)

    ours = RobustCEMSolver(
        model=model,
        batch_size=2,
        num_samples=8,
        n_steps=3,
        topk=2,
        device="cpu",
        seed=123,
        enabled=False,
    )
    upstream = CEMSolver(
        model=model,
        batch_size=2,
        num_samples=8,
        n_steps=3,
        topk=2,
        device="cpu",
        seed=123,
    )
    ours.configure(action_space=action_space, n_envs=2, config=config)
    upstream.configure(action_space=action_space, n_envs=2, config=config)

    ours_out = ours.solve(info)
    upstream_out = upstream.solve(info)

    assert torch.allclose(ours_out["actions"], upstream_out["actions"])
    assert np.allclose(ours_out["costs"], upstream_out["costs"])


def test_robust_view_infos_keep_each_view_at_sample_zero():
    solver = _configured_solver(
        QuadraticCostModel(),
        enabled=True,
        num_views=3,
        include_identity=True,
        view_type="gaussian_noise",
        view_std=0.0,
        score_mode="mean",
        n_steps=1,
    )
    expanded_info = {"pixels": torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 2)}

    view_infos = solver._build_robust_view_infos(expanded_info)

    assert len(view_infos) == 3
    for view_info in view_infos:
        assert view_info["pixels"].shape == (1, 3, 1, 2)
        assert torch.equal(view_info["pixels"][:, 0], expanded_info["pixels"][:, 0])


def test_robust_scoring_supports_view_chunking():
    model = SequentialCostModel()
    solver = _configured_solver(
        model,
        enabled=True,
        num_samples=5,
        n_steps=1,
        topk=2,
        num_views=3,
        score_mode="mean",
        max_view_batch=4,
    )

    score, stats = solver._score_candidates(
        {"pixels": torch.zeros(1, 5, 1, 2, 2)},
        torch.zeros(1, 5, 3, 2),
    )

    assert model.calls == [4, 1, 4, 1, 4, 1]
    assert score.shape == (1, 5)
    assert stats["effective_model_call_multiplier"] == 3.0


def test_elite_rescore_branch_runs_and_logs_stats():
    solver = _configured_solver(
        QuadraticCostModel(),
        enabled=True,
        num_views=2,
        view_std=0.0,
        score_mode="mean",
        robust_rescore="elite",
        elite_rescore_multiplier=2,
        n_steps=1,
    )
    info = {"pixels": torch.zeros(2, 1, 3, 4, 4)}

    out = solver.solve(info)

    assert out["actions"].shape == (2, 3, 2)
    assert out["robust"]["enabled"] is True
    assert out["robust"]["robust_rescore"] == "elite"
    assert out["robust"]["batches"][0]["elite_pool_size"] == 4



def test_gaussian_view_noise_uses_solver_seed():
    kwargs = dict(
        enabled=True,
        num_views=2,
        view_type="gaussian_noise",
        view_std=0.04,
        score_mode="mean",
        seed=99,
    )
    s1 = _configured_solver(QuadraticCostModel(), **kwargs)
    s2 = _configured_solver(QuadraticCostModel(), **kwargs)
    x = torch.zeros(1, 1, 1, 3, 4, 4)

    assert torch.allclose(s1._apply_view_transform(x), s2._apply_view_transform(x))


def test_base_std_preserves_base_order_when_candidate_is_stable():
    view_costs = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 10.0, 10.0]]])

    score, components = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="base_std",
        beta=1.0,
    )

    assert int(score.argmin(dim=1).item()) == 0
    assert components["margin_gate"] is None


def test_margin_std_penalizes_only_clean_margin_candidates():
    view_costs = torch.tensor(
        [
            [
                [0.0, 10.0, 10.0],
                [0.2, 0.2, 0.2],
                [5.0, 100.0, 100.0],
            ]
        ]
    )

    score, components = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="margin_std",
        beta=1.0,
        margin_topk=2,
    )

    assert int(score.argmin(dim=1).item()) == 1
    assert torch.allclose(components["margin_gate"], torch.tensor([[1.0, 0.0, 0.0]]))


def test_final_rescore_branch_runs_and_selects_action():
    solver = _configured_solver(
        QuadraticCostModel(),
        enabled=True,
        num_views=2,
        view_std=0.0,
        score_mode="base_std",
        robust_rescore="final",
        elite_rescore_multiplier=2,
        final_accept_rank=1,
        n_steps=2,
    )
    info = {"pixels": torch.zeros(2, 1, 3, 4, 4)}

    out = solver.solve(info)

    assert out["actions"].shape == (2, 3, 2)
    assert out["robust"]["enabled"] is True
    assert out["robust"]["robust_rescore"] == "final"
    assert out["robust"]["batches"][0]["final_pool_size"] == 4


def test_rank_mean_prefers_consistently_high_rank_candidate():
    view_costs = torch.tensor([[[0.0, 100.0, 100.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0]]])

    score, components = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="rank_mean",
    )

    assert int(score.argmin(dim=1).item()) == 1
    assert components["mean_rank"].shape == (1, 3)


def test_rank_mean_std_penalizes_rank_instability():
    view_costs = torch.tensor([[[0.0, 3.0, 3.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]])

    score, _ = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="rank_mean_std",
        beta=1.0,
    )

    assert int(score.argmin(dim=1).item()) == 1


def test_base_rank_std_keeps_stable_identity_best():
    view_costs = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 10.0, 10.0], [1.0, 1.0, 1.0]]])

    score, components = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="base_rank_std",
        beta=0.5,
    )

    assert int(score.argmin(dim=1).item()) == 0
    assert components["base_rank"].shape == (1, 3)


def test_final_switch_gate_blocks_weak_robust_advantage():
    solver = _configured_solver(
        QuadraticCostModel(),
        robust_rescore="final",
        final_switch_min_score_gain=1.0,
        final_switch_min_base_rank_std=0.5,
    )
    raw = torch.tensor([1])
    robust_score = torch.tensor([[2.0, 1.5, 3.0]])
    components = {"std_rank": torch.tensor([[1.0, 0.0, 0.0]])}
    view_costs = torch.tensor([[[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]]])

    selected, stats = solver._final_switch_selection(
        raw, robust_score, components=components, view_costs=view_costs
    )

    assert int(selected.item()) == 0
    assert stats["final_switch_allowed_rate"] == 0.0


def test_final_switch_gate_allows_strong_unstable_base_best():
    solver = _configured_solver(
        QuadraticCostModel(),
        robust_rescore="final",
        final_switch_min_score_gain=1.0,
        final_switch_min_base_rank_std=0.5,
    )
    raw = torch.tensor([1])
    robust_score = torch.tensor([[3.0, 1.0, 4.0]])
    components = {"std_rank": torch.tensor([[1.0, 0.0, 0.0]])}
    view_costs = torch.tensor([[[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]]])

    selected, stats = solver._final_switch_selection(
        raw, robust_score, components=components, view_costs=view_costs
    )

    assert int(selected.item()) == 1
    assert stats["final_switch_allowed_rate"] == 1.0
    assert stats["final_switch_score_gain_mean"] == 2.0


def test_final_switch_gate_blocks_stable_base_best():
    solver = _configured_solver(
        QuadraticCostModel(),
        robust_rescore="final",
        final_switch_min_score_gain=1.0,
        final_switch_min_base_rank_std=0.5,
    )
    raw = torch.tensor([1])
    robust_score = torch.tensor([[3.0, 1.0, 4.0]])
    components = {"std_rank": torch.tensor([[0.0, 0.0, 0.0]])}
    view_costs = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])

    selected, stats = solver._final_switch_selection(
        raw, robust_score, components=components, view_costs=view_costs
    )

    assert int(selected.item()) == 0
    assert stats["final_switch_allowed_rate"] == 0.0


def test_rank_vote_prefers_candidate_with_more_top1_votes():
    view_costs = torch.tensor([[[0.0, 3.0, 3.0], [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]]])

    score, components = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="rank_vote",
    )

    assert int(score.argmin(dim=1).item()) == 1
    assert torch.allclose(components["top1_vote"], torch.tensor([[1.0, 2.0, 0.0]]))


def test_rank_vote_keeps_base_order_on_vote_ties():
    view_costs = torch.tensor([[[0.0, 3.0], [1.0, 0.0], [2.0, 1.0]]])

    score, components = aggregate_view_costs(
        view_costs,
        include_identity=True,
        score_mode="rank_vote",
    )

    assert int(score.argmin(dim=1).item()) == 0
    assert torch.allclose(components["top1_vote"], torch.tensor([[1.0, 1.0, 0.0]]))


def test_final_rescore_elite_mean_branch_runs():
    solver = _configured_solver(
        QuadraticCostModel(),
        enabled=True,
        num_views=2,
        view_std=0.0,
        score_mode="rank_vote",
        robust_rescore="final",
        final_output_mode="elite_mean",
        elite_rescore_multiplier=2,
        n_steps=2,
    )
    info = {"pixels": torch.zeros(2, 1, 3, 4, 4)}

    out = solver.solve(info)

    assert out["actions"].shape == (2, 3, 2)
    assert out["robust"]["enabled"] is True
    assert out["robust"]["robust_rescore"] == "final"
    assert out["robust"]["final_output_mode"] == "elite_mean"
