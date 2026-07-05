"""Planner-side robust CEM solver for evaluation-time interventions.

The standard path intentionally follows ``stable_worldmodel.solver.CEMSolver``.
When robust scoring is enabled, candidate costs are evaluated under a small
observation-view ensemble before elite selection, so the robust score drives
the CEM distribution updates rather than only a final rerank.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging


_SCORE_MODES = {"base", "mean", "mean_std", "base_std", "margin_std", "base_rank_std", "rank_mean", "rank_mean_std", "rank_vote", "rank_worst", "worst", "max", "quantile"}
_VIEW_TYPES = {"gaussian_noise", "gaussian_blur", "resize"}
_RESCORE_MODES = {"all", "elite", "final"}
_FINAL_OUTPUT_MODES = {"selected", "elite_mean"}


def aggregate_view_costs(
    view_costs: torch.Tensor,
    *,
    include_identity: bool = True,
    score_mode: str = "mean_std",
    beta: float = 0.5,
    quantile: float = 0.8,
    margin_topk: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Aggregate ``(B, N, K)`` view costs into one score per candidate."""
    if view_costs.ndim != 3:
        raise ValueError(f"Expected view_costs shape (B, N, K), got {tuple(view_costs.shape)}")

    mode = score_mode.lower()
    if mode == "max":
        mode = "worst"
    if mode not in _SCORE_MODES:
        raise ValueError(f"Unsupported robust CEM score_mode: {score_mode}")

    base_cost = view_costs[..., 0] if include_identity else view_costs.mean(dim=-1)
    mean_cost = view_costs.mean(dim=-1)
    std_cost = view_costs.std(dim=-1, unbiased=False)
    ranks = view_costs.argsort(dim=1).argsort(dim=1).to(view_costs.dtype)
    base_rank = ranks[..., 0] if include_identity else ranks.mean(dim=-1)
    mean_rank = ranks.mean(dim=-1)
    std_rank = ranks.std(dim=-1, unbiased=False)
    worst_rank = ranks.max(dim=-1).values
    top1_vote = (ranks == 0).to(view_costs.dtype).sum(dim=-1)

    margin_gate = None
    if mode == "margin_std":
        pool_k = view_costs.shape[1] if margin_topk is None else int(margin_topk)
        pool_k = max(1, min(pool_k, view_costs.shape[1]))
        best_cost = base_cost.min(dim=1, keepdim=True).values
        kth_cost = torch.topk(base_cost, k=pool_k, dim=1, largest=False).values[:, -1:]
        margin_scale = (kth_cost - best_cost).clamp_min(torch.finfo(base_cost.dtype).eps)
        clean_margin = base_cost - best_cost
        margin_gate = (1.0 - clean_margin / margin_scale).clamp(min=0.0, max=1.0)

    if mode == "base":
        score = base_cost
    elif mode == "mean":
        score = mean_cost
    elif mode == "mean_std":
        score = mean_cost + float(beta) * std_cost
    elif mode == "base_std":
        score = base_cost + float(beta) * std_cost
    elif mode == "margin_std":
        score = base_cost + float(beta) * std_cost * margin_gate
    elif mode == "base_rank_std":
        score = base_rank + float(beta) * std_rank
    elif mode == "rank_mean":
        score = mean_rank
    elif mode == "rank_mean_std":
        score = mean_rank + float(beta) * std_rank
    elif mode == "rank_vote":
        # Preserve base-pool order on ties: the final argmin returns the first
        # candidate among equal vote counts, and final pools are base-cost sorted.
        score = -top1_vote
    elif mode == "rank_worst":
        score = worst_rank
    elif mode == "worst":
        score = view_costs.max(dim=-1).values
    elif mode == "quantile":
        score = torch.quantile(view_costs, q=float(quantile), dim=-1)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unsupported robust CEM score_mode: {score_mode}")

    return score, {
        "base_cost": base_cost,
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "range_cost": view_costs.max(dim=-1).values - view_costs.min(dim=-1).values,
        "base_rank": base_rank,
        "mean_rank": mean_rank,
        "std_rank": std_rank,
        "worst_rank": worst_rank,
        "top1_vote": top1_vote,
        "margin_gate": margin_gate,
    }


def _json_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


class RobustCEMSolver:
    """Cross-Entropy Method solver with optional robust candidate scoring."""

    def __init__(
        self,
        model: Any | None = None,
        cost: Any | None = None,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cuda",
        seed: int = 1234,
        callbacks: list[Any] | None = None,
        *,
        enabled: bool = True,
        num_views: int = 4,
        include_identity: bool = True,
        view_type: str = "gaussian_noise",
        view_std: float = 0.04,
        view_kernel_size: int = 7,
        view_resize_factor: float = 0.75,
        perturb_pixels: bool = True,
        perturb_goal: bool = False,
        score_mode: str = "mean_std",
        beta: float = 0.5,
        quantile: float = 0.8,
        margin_topk: int | None = None,
        final_accept_rank: int | None = None,
        final_switch_min_score_gain: float | None = None,
        final_switch_min_base_rank_std: float | None = None,
        final_switch_min_top1_flip: float | None = None,
        final_output_mode: str = "selected",
        robust_rescore: str = "all",
        elite_rescore_multiplier: int = 2,
        max_view_batch: int | None = None,
        record_history: bool = True,
    ) -> None:
        self.model = model if model is not None else cost
        if self.model is None:
            raise ValueError("RobustCEMSolver requires either model= or cost=")

        self.batch_size = int(batch_size)
        self.var_scale = float(var_scale)
        self.num_samples = int(num_samples)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.device = torch.device(device)
        self._dtype = self._infer_model_dtype()
        self.torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.callbacks = list(callbacks) if callbacks else []

        self.enabled = bool(enabled)
        self.num_views = int(num_views)
        self.include_identity = bool(include_identity)
        self.view_type = str(view_type)
        self.view_std = float(view_std)
        self.view_kernel_size = int(view_kernel_size)
        self.view_resize_factor = float(view_resize_factor)
        self.perturb_pixels = bool(perturb_pixels)
        self.perturb_goal = bool(perturb_goal)
        self.score_mode = str(score_mode).lower()
        self.beta = float(beta)
        self.quantile = float(quantile)
        self.margin_topk = None if margin_topk is None else int(margin_topk)
        self.final_accept_rank = None if final_accept_rank is None else int(final_accept_rank)
        self.final_switch_min_score_gain = (
            None if final_switch_min_score_gain is None else float(final_switch_min_score_gain)
        )
        self.final_switch_min_base_rank_std = (
            None if final_switch_min_base_rank_std is None else float(final_switch_min_base_rank_std)
        )
        self.final_switch_min_top1_flip = (
            None if final_switch_min_top1_flip is None else float(final_switch_min_top1_flip)
        )
        self.final_output_mode = str(final_output_mode).lower()
        self.robust_rescore = str(robust_rescore).lower()
        self.elite_rescore_multiplier = int(elite_rescore_multiplier)
        self.max_view_batch = None if max_view_batch is None else int(max_view_batch)
        self.record_history = bool(record_history)

        if self.score_mode == "max":
            self.score_mode = "worst"
        if self.score_mode not in _SCORE_MODES:
            raise ValueError(f"Unsupported robust CEM score_mode: {score_mode}")
        if self.view_type not in _VIEW_TYPES:
            raise ValueError(f"Unsupported robust CEM view_type: {view_type}")
        if self.robust_rescore not in _RESCORE_MODES:
            raise ValueError(f"Unsupported robust CEM robust_rescore: {robust_rescore}")
        if self.final_output_mode not in _FINAL_OUTPUT_MODES:
            raise ValueError(f"Unsupported robust CEM final_output_mode: {final_output_mode}")
        if self.num_samples <= 0 or self.n_steps <= 0 or self.topk <= 0:
            raise ValueError("num_samples, n_steps, and topk must be positive")
        if self.topk > self.num_samples:
            raise ValueError("topk must be <= num_samples")
        if self.num_views <= 0:
            raise ValueError("num_views must be positive")
        if self.margin_topk is not None and self.margin_topk <= 0:
            raise ValueError("margin_topk must be positive when set")
        if self.final_accept_rank is not None and self.final_accept_rank < 0:
            raise ValueError("final_accept_rank must be non-negative when set")
        for name, value in (
            ("final_switch_min_score_gain", self.final_switch_min_score_gain),
            ("final_switch_min_base_rank_std", self.final_switch_min_base_rank_std),
            ("final_switch_min_top1_flip", self.final_switch_min_top1_flip),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when set")
        if self.elite_rescore_multiplier <= 0:
            raise ValueError("elite_rescore_multiplier must be positive")

        # Small tensors are cloned to guard against in-place model mutation.
        # Large expanded image tensors are left as views to avoid materializing
        # B*N*history*C*H*W copies during CEM scoring.
        self._max_info_clone_numel = 1_000_000

        self.last_robust_stats: dict[str, Any] = {}
        self.robust_history: list[dict[str, Any]] = []

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(
                f"Action space is discrete, got {type(action_space)}. "
                "RobustCEMSolver may not work as expected."
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _infer_model_dtype(self) -> torch.dtype:
        if hasattr(self.model, "parameters"):
            try:
                return next(self.model.parameters()).dtype
            except StopIteration:
                pass
        return torch.float32

    def _robust_active(self) -> bool:
        return bool(self.enabled and self.num_views > 1 and self.score_mode != "base")

    def _tensor_to_solver(self, value: torch.Tensor) -> torch.Tensor:
        target_dtype = self.dtype if value.is_floating_point() else None
        return value.to(device=self.device, dtype=target_dtype)

    def prepare_init_action(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        *,
        n_envs: int,
    ) -> torch.Tensor | None:
        if init_action is None:
            actions = None
        else:
            actions = init_action.to(device=self.device, dtype=self.dtype)
            assert actions.shape[0] == n_envs, (
                f"init_action batch size {actions.shape[0]} != n_envs {n_envs}"
            )
            assert actions.shape[2] == self.action_dim, (
                f"init_action action_dim {actions.shape[2]} != action_dim {self.action_dim}"
            )

        n_prev = actions.shape[1] if actions is not None else 0
        remaining = self.horizon - n_prev
        if remaining <= 0:
            return actions[:, : self.horizon]

        tail = self._actionable_warm_start_tail(info_dict, remaining, actions, n_envs=n_envs)
        if tail is None:
            tail = torch.zeros(
                [n_envs, remaining, self.action_dim],
                device=self.device,
                dtype=self.dtype,
            )
        if actions is not None:
            return torch.cat([actions.to(tail.device), tail], dim=1)
        return tail

    def _actionable_warm_start_tail(
        self,
        info_dict: dict,
        remaining: int,
        prefix_actions: torch.Tensor | None,
        *,
        n_envs: int,
    ) -> torch.Tensor | None:
        if not hasattr(self.model, "get_action"):
            return None

        prepared_info = {}
        for k, v in info_dict.items():
            prepared_info[k] = self._tensor_to_solver(v) if torch.is_tensor(v) else v

        try:
            tail = self.model.get_action(
                prepared_info,
                horizon=remaining,
                prefix_actions=prefix_actions,
            )
        except TypeError as exc:  # pragma: no cover - legacy Actionable fallback
            logging.warning(
                "Actionable warm-start did not accept horizon/prefix_actions; "
                f"falling back to repeated one-step get_action: {exc}"
            )
            return self._legacy_actionable_tail(prepared_info, remaining, n_envs=n_envs)
        except Exception as exc:  # pragma: no cover - compatibility fallback
            logging.warning(f"Actionable warm-start failed; falling back to zeros: {exc}")
            return None

        if isinstance(tail, np.ndarray):
            tail = torch.from_numpy(tail)
        if not torch.is_tensor(tail):
            tail = torch.as_tensor(tail)
        tail = tail.to(device=self.device, dtype=self.dtype)
        if tail.ndim == 2:
            tail = tail.unsqueeze(1)
        if tail.shape != (n_envs, remaining, self.action_dim):
            logging.warning(
                "Actionable warm-start produced shape "
                f"{tuple(tail.shape)}, expected {(n_envs, remaining, self.action_dim)}; "
                "falling back to zeros."
            )
            return None
        return tail

    def _legacy_actionable_tail(
        self,
        prepared_info: dict,
        remaining: int,
        *,
        n_envs: int,
    ) -> torch.Tensor | None:
        tail = []
        for _ in range(remaining):
            try:
                action = self.model.get_action(prepared_info)
            except Exception as exc:  # pragma: no cover - compatibility fallback
                logging.warning(f"Legacy Actionable warm-start failed; falling back to zeros: {exc}")
                return None
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            if not torch.is_tensor(action):
                action = torch.as_tensor(action)
            action = action.to(device=self.device, dtype=self.dtype).reshape(n_envs, -1)
            if action.shape[-1] == self._action_dim:
                action = action.repeat(1, self._config.action_block)
            if action.shape[-1] != self.action_dim:
                logging.warning(
                    "Legacy Actionable warm-start produced action dim "
                    f"{action.shape[-1]}, expected {self.action_dim}; falling back to zeros."
                )
                return None
            tail.append(action.unsqueeze(1))
        return torch.cat(tail, dim=1) if tail else None

    def init_action_distrib(
        self,
        n_envs: int,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var = self.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim],
            device=self.device,
            dtype=self.dtype,
        )
        mean = (
            torch.zeros([n_envs, 0, self.action_dim], device=self.device, dtype=self.dtype)
            if actions is None
            else actions.to(device=self.device, dtype=self.dtype)
        )

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            new_mean = torch.zeros(
                [n_envs, remaining, self.action_dim],
                device=self.device,
                dtype=self.dtype,
            )
            mean = torch.cat([mean, new_mean], dim=1)
        return mean, var

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": []}

        total_envs = len(next(iter(info_dict.values())))
        init_action = self.prepare_init_action(info_dict, init_action, n_envs=total_envs)
        mean, var = self.init_action_distrib(total_envs, init_action)
        final_stats: list[dict[str, Any]] = []
        iteration_records: list[dict[str, Any]] = []

        for cb in self.callbacks:
            cb.reset()

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            expanded_infos = self._expand_info(info_dict, start_idx, end_idx, self.num_samples)
            final_batch_cost = None

            for cb in self.callbacks:
                cb.start_batch()

            for step in range(self.n_steps):
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                selected_final_candidate = None
                if self._robust_active() and self.robust_rescore == "elite":
                    score, topk_inds, topk_candidates, step_stats = self._elite_rescore_step(
                        expanded_infos, candidates
                    )
                    topk_vals = torch.gather(score, 1, topk_inds)
                elif self._robust_active() and self.robust_rescore == "final":
                    if step == self.n_steps - 1:
                        (
                            score,
                            topk_inds,
                            topk_candidates,
                            step_stats,
                            selected_final_candidate,
                        ) = self._final_rescore_step(expanded_infos, candidates)
                        topk_vals = torch.gather(score, 1, topk_inds)
                    else:
                        score, step_stats = self._score_candidates(
                            expanded_infos, candidates, force_base=True
                        )
                        self._assert_cost_shape(score, current_bs, self.num_samples)
                        topk_vals, topk_inds = torch.topk(score, k=self.topk, dim=1, largest=False)
                        batch_indices = (
                            torch.arange(current_bs, device=self.device)
                            .unsqueeze(1)
                            .expand(-1, self.topk)
                        )
                        topk_candidates = candidates[batch_indices, topk_inds]
                else:
                    score, step_stats = self._score_candidates(expanded_infos, candidates)
                    self._assert_cost_shape(score, current_bs, self.num_samples)
                    topk_vals, topk_inds = torch.topk(score, k=self.topk, dim=1, largest=False)
                    batch_indices = (
                        torch.arange(current_bs, device=self.device)
                        .unsqueeze(1)
                        .expand(-1, self.topk)
                    )
                    topk_candidates = candidates[batch_indices, topk_inds]

                prev_mean = batch_mean
                prev_var = batch_var
                batch_mean = (
                    selected_final_candidate
                    if selected_final_candidate is not None
                    else topk_candidates.mean(dim=1)
                )
                batch_var = topk_candidates.std(dim=1)

                if step_stats is not None:
                    record = {
                        **step_stats,
                        "batch_start": int(start_idx),
                        "batch_end": int(end_idx),
                        "step": int(step),
                    }
                    if self.record_history:
                        iteration_records.append(record)
                    if step == self.n_steps - 1:
                        final_stats.append(record)

                for cb in self.callbacks:
                    cb(
                        step=step,
                        candidates=candidates,
                        costs=score,
                        topk_vals=topk_vals,
                        topk_inds=topk_inds,
                        topk_candidates=topk_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=prev_mean,
                        prev_var=prev_var,
                    )

                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        if self.callbacks:
            outputs["callbacks"] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs["callbacks"][cb.output_key] = cb.history

        solve_time = time.time() - start_time
        if self.record_history and iteration_records:
            self.robust_history.extend(iteration_records)
        self.last_robust_stats = self._summarize_solve(final_stats, solve_time)
        outputs["robust"] = self.last_robust_stats

        print(f"CEM solve time: {solve_time:.4f} seconds")
        return outputs

    def _elite_rescore_step(
        self,
        expanded_infos: dict,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        current_bs = candidates.shape[0]
        base_score, _ = self._score_candidates(expanded_infos, candidates, force_base=True)
        self._assert_cost_shape(base_score, current_bs, self.num_samples)

        pool_k = min(
            self.num_samples,
            max(self.topk, self.topk * self.elite_rescore_multiplier),
        )
        _, pool_inds = torch.topk(base_score, k=pool_k, dim=1, largest=False)
        batch_indices = (
            torch.arange(current_bs, device=self.device)
            .unsqueeze(1)
            .expand(-1, pool_k)
        )
        pool_candidates = candidates[batch_indices, pool_inds]
        pool_info = self._gather_info_samples(expanded_infos, pool_inds)
        robust_score, step_stats = self._score_candidates(pool_info, pool_candidates)
        self._assert_cost_shape(robust_score, current_bs, pool_k)

        robust_vals, robust_pool_inds = torch.topk(
            robust_score, k=self.topk, dim=1, largest=False
        )
        topk_candidates = pool_candidates[
            torch.arange(current_bs, device=self.device).unsqueeze(1),
            robust_pool_inds,
        ]
        topk_inds = torch.gather(pool_inds, 1, robust_pool_inds)

        selection_score = base_score.clone()
        selection_score.scatter_(1, pool_inds, robust_score)
        selected_scores = torch.gather(selection_score, 1, topk_inds)
        if not torch.allclose(selected_scores, robust_vals):
            selection_score.scatter_(1, topk_inds, robust_vals)

        if step_stats is not None:
            step_stats = {
                **step_stats,
                "robust_rescore": self.robust_rescore,
                "elite_pool_size": int(pool_k),
            }
        return selection_score, topk_inds, topk_candidates, step_stats

    def _final_rescore_step(
        self,
        expanded_infos: dict,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any] | None, torch.Tensor | None]:
        current_bs = candidates.shape[0]
        base_score, _ = self._score_candidates(expanded_infos, candidates, force_base=True)
        self._assert_cost_shape(base_score, current_bs, self.num_samples)

        pool_k = min(
            self.num_samples,
            max(self.topk, self.topk * self.elite_rescore_multiplier),
        )
        base_pool_vals, pool_inds = torch.topk(base_score, k=pool_k, dim=1, largest=False)
        batch_indices = (
            torch.arange(current_bs, device=self.device)
            .unsqueeze(1)
            .expand(-1, pool_k)
        )
        pool_candidates = candidates[batch_indices, pool_inds]
        pool_info = self._gather_info_samples(expanded_infos, pool_inds)
        robust_score, step_stats, components, view_costs = self._score_candidates_with_components(
            pool_info, pool_candidates
        )
        self._assert_cost_shape(robust_score, current_bs, pool_k)

        raw_selected_pool_inds = robust_score.argmin(dim=1)
        selected_pool_inds, gate_stats = self._final_switch_selection(
            raw_selected_pool_inds,
            robust_score,
            components=components,
            view_costs=view_costs,
        )
        selected_candidates = pool_candidates[
            torch.arange(current_bs, device=self.device), selected_pool_inds
        ]
        selected_global_inds = torch.gather(pool_inds, 1, selected_pool_inds.unsqueeze(1))

        selection_score = base_score.clone()
        selection_score.scatter_(1, pool_inds, robust_score)
        if self.final_output_mode == "elite_mean":
            _, robust_topk_pool_inds = torch.topk(
                robust_score, k=self.topk, dim=1, largest=False
            )
            topk_candidates = pool_candidates[
                torch.arange(current_bs, device=self.device).unsqueeze(1),
                robust_topk_pool_inds,
            ]
            topk_inds = torch.gather(pool_inds, 1, robust_topk_pool_inds)
            selected_output_candidate = None
        else:
            topk_candidates = pool_candidates[:, : self.topk]
            topk_inds = pool_inds[:, : self.topk]
            topk_inds = torch.cat([selected_global_inds, topk_inds[:, 1:]], dim=1)
            topk_candidates = torch.cat(
                [selected_candidates.unsqueeze(1), topk_candidates[:, 1:]], dim=1
            )
            selected_output_candidate = selected_candidates

        if step_stats is not None:
            base_best_global = pool_inds[:, 0]
            selected_base_rank = (pool_inds == selected_global_inds).float().argmax(dim=1)
            step_stats = {
                **step_stats,
                "robust_rescore": self.robust_rescore,
                "final_pool_size": int(pool_k),
                "final_accept_rank": None if self.final_accept_rank is None else int(self.final_accept_rank),
                "final_output_mode": self.final_output_mode,
                "final_selected_base_rank_mean": _json_float(selected_base_rank.float().mean()),
                "final_changed_from_base_best_rate": _json_float(
                    (selected_global_inds.squeeze(1) != base_best_global).float().mean()
                ),
                "final_mean_base_pool_cost": _json_float(base_pool_vals.mean()),
                **gate_stats,
            }
        return selection_score, topk_inds, topk_candidates, step_stats, selected_output_candidate

    def _score_candidates(
        self,
        expanded_infos: dict,
        candidates: torch.Tensor,
        *,
        force_base: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        score, stats, _, _ = self._score_candidates_with_components(
            expanded_infos, candidates, force_base=force_base
        )
        return score, stats

    def _score_candidates_with_components(
        self,
        expanded_infos: dict,
        candidates: torch.Tensor,
        *,
        force_base: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any] | None, dict[str, torch.Tensor] | None, torch.Tensor | None]:
        if force_base or not self._robust_active():
            costs = self._cost_in_chunks(expanded_infos, candidates)
            return costs, None, None, None

        view_costs = []
        for view_info in self._build_robust_view_infos(expanded_infos):
            view_costs.append(
                self._cost_in_chunks(
                    view_info,
                    candidates,
                    max_sample_batch=self.max_view_batch,
                )
            )
        view_costs = torch.stack(view_costs, dim=-1)
        score, components = aggregate_view_costs(
            view_costs,
            include_identity=self.include_identity,
            score_mode=self.score_mode,
            beta=self.beta,
            quantile=self.quantile,
            margin_topk=self.margin_topk or self.topk,
        )
        return score, self._view_stats(view_costs, score, components), components, view_costs

    def _final_switch_selection(
        self,
        raw_selected_pool_inds: torch.Tensor,
        robust_score: torch.Tensor,
        *,
        components: dict[str, torch.Tensor] | None,
        view_costs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        selected_scores = robust_score.gather(1, raw_selected_pool_inds.unsqueeze(1)).squeeze(1)
        base_scores = robust_score[:, 0]
        score_gain = base_scores - selected_scores
        switch_allowed = raw_selected_pool_inds != 0

        if self.final_accept_rank is not None:
            accept_rank = min(int(self.final_accept_rank), robust_score.shape[1] - 1)
            switch_allowed = switch_allowed & (raw_selected_pool_inds <= accept_rank)
        if self.final_switch_min_score_gain is not None:
            switch_allowed = switch_allowed & (score_gain >= self.final_switch_min_score_gain)

        base_rank_std = None
        if components is not None and components.get("std_rank") is not None:
            base_rank_std = components["std_rank"][:, 0]
            if self.final_switch_min_base_rank_std is not None:
                switch_allowed = switch_allowed & (base_rank_std >= self.final_switch_min_base_rank_std)

        top1_flip = None
        if view_costs is not None and view_costs.shape[-1] > 1:
            anchor_best = view_costs[..., 0].argmin(dim=1)
            all_view_best = view_costs.argmin(dim=1)
            top1_flip = (all_view_best[:, 1:] != anchor_best.unsqueeze(1)).float().mean(dim=1)
            if self.final_switch_min_top1_flip is not None:
                switch_allowed = switch_allowed & (top1_flip >= self.final_switch_min_top1_flip)

        selected_pool_inds = torch.where(
            switch_allowed, raw_selected_pool_inds, torch.zeros_like(raw_selected_pool_inds)
        )
        return selected_pool_inds, {
            "final_switch_min_score_gain": self.final_switch_min_score_gain,
            "final_switch_min_base_rank_std": self.final_switch_min_base_rank_std,
            "final_switch_min_top1_flip": self.final_switch_min_top1_flip,
            "final_raw_selected_base_rank_mean": _json_float(raw_selected_pool_inds.float().mean()),
            "final_switch_allowed_rate": _json_float(switch_allowed.float().mean()),
            "final_switch_score_gain_mean": _json_float(score_gain.mean()),
            "final_base_rank_std_mean": _json_float(base_rank_std.mean()) if base_rank_std is not None else None,
            "final_top1_flip_mean": _json_float(top1_flip.mean()) if top1_flip is not None else None,
        }

    def _assert_cost_shape(self, costs: torch.Tensor, batch_size: int, num_samples: int) -> None:
        assert isinstance(costs, torch.Tensor), f"Expected cost tensor, got {type(costs)}"
        assert costs.ndim == 2 and costs.shape == (batch_size, num_samples), (
            f"Expected cost shape ({batch_size}, {num_samples}), got {tuple(costs.shape)}"
        )

    def _expand_info(self, info_dict: dict, start_idx: int, end_idx: int, num_samples: int) -> dict:
        current_bs = end_idx - start_idx
        expanded_infos = {}
        for k, v in info_dict.items():
            v_batch = v[start_idx:end_idx]
            if torch.is_tensor(v):
                v_batch = self._tensor_to_solver(v_batch).unsqueeze(1)
                v_batch = v_batch.expand(current_bs, num_samples, *v_batch.shape[2:])
            elif isinstance(v, np.ndarray):
                v_batch = np.repeat(v_batch[:, None, ...], num_samples, axis=1)
            expanded_infos[k] = v_batch
        return expanded_infos

    def _fresh_info_dict(self, info: dict) -> dict:
        out = {}
        for k, v in info.items():
            if torch.is_tensor(v):
                out[k] = v.clone() if v.numel() <= self._max_info_clone_numel else v
            elif isinstance(v, np.ndarray):
                out[k] = v.copy() if v.size <= self._max_info_clone_numel else v
            else:
                out[k] = v
        return out

    def _cost_in_chunks(
        self,
        info: dict,
        candidates: torch.Tensor,
        *,
        max_sample_batch: int | None = None,
    ) -> torch.Tensor:
        total_samples = candidates.shape[1]
        if max_sample_batch is None or max_sample_batch <= 0 or total_samples <= max_sample_batch:
            costs = self.model.get_cost(self._fresh_info_dict(info), candidates)
            self._assert_cost_shape(costs, candidates.shape[0], total_samples)
            return costs

        chunks = []
        for start in range(0, total_samples, max_sample_batch):
            end = min(start + max_sample_batch, total_samples)
            chunk_info = self._slice_info_samples(info, start, end)
            chunk_candidates = candidates[:, start:end]
            costs = self.model.get_cost(self._fresh_info_dict(chunk_info), chunk_candidates)
            self._assert_cost_shape(costs, candidates.shape[0], end - start)
            chunks.append(costs)
        return torch.cat(chunks, dim=1)

    def _slice_info_samples(self, info: dict, start: int, end: int) -> dict:
        out = {}
        for k, v in info.items():
            if torch.is_tensor(v):
                out[k] = v[:, start:end]
            elif isinstance(v, np.ndarray):
                out[k] = v[:, start:end]
            else:
                out[k] = v
        return out

    def _gather_info_samples(self, info: dict, indices: torch.Tensor) -> dict:
        out = {}
        batch_size, sample_count = indices.shape
        for k, v in info.items():
            if torch.is_tensor(v):
                gather_index = indices.reshape(batch_size, sample_count, *([1] * (v.ndim - 2)))
                gather_index = gather_index.expand(batch_size, sample_count, *v.shape[2:])
                out[k] = torch.gather(v, 1, gather_index)
            elif isinstance(v, np.ndarray):
                np_indices = indices.detach().cpu().numpy()
                gather_index = np_indices.reshape(batch_size, sample_count, *([1] * (v.ndim - 2)))
                gather_index = np.broadcast_to(gather_index, (batch_size, sample_count, *v.shape[2:]))
                out[k] = np.take_along_axis(v, gather_index, axis=1)
            else:
                out[k] = v
        return out

    def _build_robust_view_infos(self, expanded_infos: dict) -> list[dict]:
        # LeWM's get_cost()/rollout() reads info[k][:, 0] and broadcasts that
        # observation to all candidates. Robust views therefore must be separate
        # cost calls, not extra entries on the sample axis.
        view_infos = [dict() for _ in range(self.num_views)]
        for key, value in expanded_infos.items():
            if torch.is_tensor(value):
                views = self._make_tensor_views(key, value)
                for view_idx, view_value in enumerate(views):
                    view_infos[view_idx][key] = view_value
            elif isinstance(value, np.ndarray):
                for view_idx in range(self.num_views):
                    view_infos[view_idx][key] = value.copy() if value.size <= self._max_info_clone_numel else value
            else:
                for view_idx in range(self.num_views):
                    view_infos[view_idx][key] = value
        return view_infos

    def _make_tensor_views(self, key: str, value: torch.Tensor) -> list[torch.Tensor]:
        if not self._should_perturb_key(key):
            return [value for _ in range(self.num_views)]

        base = value[:, :1].clone()
        views = []
        for view_idx in range(self.num_views):
            if self.include_identity and view_idx == 0:
                view = base
            else:
                view = self._apply_view_transform(base.clone())
            views.append(view.expand(value.shape[0], value.shape[1], *value.shape[2:]))
        return views

    def _should_perturb_key(self, key: str) -> bool:
        if key == "pixels":
            return self.perturb_pixels
        if key == "goal":
            return self.perturb_goal
        return False

    def _apply_view_transform(self, value: torch.Tensor) -> torch.Tensor:
        if self.view_type == "gaussian_noise":
            if self.view_std <= 0:
                return value
            return self._add_normalized_gaussian_noise(value)
        if self.view_type == "gaussian_blur":
            if self.view_kernel_size <= 1:
                return value
            from utils import AddGaussianBlur

            return AddGaussianBlur(self.view_kernel_size, self.view_kernel_size)(value)
        if self.view_type == "resize":
            if self.view_resize_factor >= 1.0:
                return value
            from utils import AddResize

            return AddResize(self.view_resize_factor, self.view_resize_factor)(value)
        raise ValueError(f"Unsupported robust CEM view_type: {self.view_type}")

    def _add_normalized_gaussian_noise(self, value: torch.Tensor) -> torch.Tensor:
        from stable_pretraining import data as dt

        noise = torch.randn(
            value.shape,
            generator=self.torch_gen,
            device=value.device,
            dtype=value.dtype,
        )
        if value.ndim < 3:
            return value + noise * self.view_std

        stats = dt.dataset_stats.ImageNet
        channel_std = stats["std"] if isinstance(stats, dict) else stats.std
        channel_std = torch.as_tensor(channel_std, device=value.device, dtype=value.dtype)
        if value.shape[-3] != channel_std.numel():
            return value + noise * self.view_std

        leading_dims = [1] * (value.ndim - 3)
        scale = (self.view_std / channel_std).view(*leading_dims, -1, 1, 1)
        return value + noise * scale

    def _view_stats(
        self,
        view_costs: torch.Tensor,
        score: torch.Tensor,
        components: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        anchor_best = view_costs[..., 0].argmin(dim=1)
        all_view_best = view_costs.argmin(dim=1)
        if self.num_views > 1:
            compare = all_view_best[:, 1:]
            top1_flip_rate = (compare != anchor_best.unsqueeze(1)).float().mean()
        else:
            top1_flip_rate = None

        robust_best = score.argmin(dim=1)
        changed_rate = (robust_best != anchor_best).float().mean()

        return {
            "enabled": True,
            "num_views": int(self.num_views),
            "include_identity": bool(self.include_identity),
            "view_type": self.view_type,
            "score_mode": self.score_mode,
            "beta": float(self.beta),
            "quantile": float(self.quantile),
            "margin_topk": None if self.margin_topk is None else int(self.margin_topk),
            "final_accept_rank": None if self.final_accept_rank is None else int(self.final_accept_rank),
            "final_switch_min_score_gain": self.final_switch_min_score_gain,
            "final_switch_min_base_rank_std": self.final_switch_min_base_rank_std,
            "final_switch_min_top1_flip": self.final_switch_min_top1_flip,
            "final_output_mode": self.final_output_mode,
            "robust_rescore": self.robust_rescore,
            "mean_base_cost": _json_float(components["base_cost"].mean()),
            "mean_robust_score": _json_float(score.mean()),
            "mean_view_std": _json_float(components["std_cost"].mean()),
            "mean_view_range": _json_float(components["range_cost"].mean()),
            "mean_base_rank": _json_float(components["base_rank"].mean()),
            "mean_rank_std": _json_float(components["std_rank"].mean()),
            "mean_top1_vote": _json_float(components["top1_vote"].mean()),
            "mean_margin_gate": _json_float(components["margin_gate"].mean()) if components.get("margin_gate") is not None else None,
            "top1_flip_rate": _json_float(top1_flip_rate),
            "robust_changed_top1_rate": _json_float(changed_rate),
            "effective_model_call_multiplier": self._effective_model_call_multiplier(),
        }

    def _effective_model_call_multiplier(self) -> float:
        if not self._robust_active():
            return 1.0
        if self.robust_rescore == "all":
            return float(self.num_views)
        pool_k = min(self.num_samples, max(self.topk, self.topk * self.elite_rescore_multiplier))
        if self.robust_rescore == "final":
            return 1.0 + float(self.num_views * pool_k) / float(self.num_samples * self.n_steps)
        return 1.0 + float(self.num_views * pool_k) / float(self.num_samples)

    def _summarize_solve(self, final_stats: list[dict[str, Any]], solve_time: float) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "enabled": bool(self._robust_active()),
            "configured_enabled": bool(self.enabled),
            "num_views": int(self.num_views),
            "include_identity": bool(self.include_identity),
            "view_type": self.view_type,
            "score_mode": self.score_mode,
            "beta": float(self.beta),
            "quantile": float(self.quantile),
            "margin_topk": None if self.margin_topk is None else int(self.margin_topk),
            "final_accept_rank": None if self.final_accept_rank is None else int(self.final_accept_rank),
            "final_switch_min_score_gain": self.final_switch_min_score_gain,
            "final_switch_min_base_rank_std": self.final_switch_min_base_rank_std,
            "final_switch_min_top1_flip": self.final_switch_min_top1_flip,
            "final_output_mode": self.final_output_mode,
            "robust_rescore": self.robust_rescore,
            "perturb_pixels": bool(self.perturb_pixels),
            "perturb_goal": bool(self.perturb_goal),
            "solve_time_sec": float(solve_time),
            "effective_model_call_multiplier": self._effective_model_call_multiplier(),
        }
        if final_stats:
            numeric_keys = [
                "mean_base_cost",
                "mean_robust_score",
                "mean_view_std",
                "mean_view_range",
                "mean_base_rank",
                "mean_rank_std",
                "mean_top1_vote",
                "mean_margin_gate",
                "top1_flip_rate",
                "robust_changed_top1_rate",
                "final_selected_base_rank_mean",
                "final_changed_from_base_best_rate",
                "final_raw_selected_base_rank_mean",
                "final_switch_allowed_rate",
                "final_switch_score_gain_mean",
                "final_base_rank_std_mean",
                "final_top1_flip_mean",
            ]
            for key in numeric_keys:
                values = [row[key] for row in final_stats if row.get(key) is not None]
                summary[key] = float(np.mean(values)) if values else None
            summary["batches"] = final_stats
        else:
            summary.update(
                {
                    "mean_base_cost": None,
                    "mean_robust_score": None,
                    "mean_view_std": None,
                    "mean_view_range": None,
                    "mean_base_rank": None,
                    "mean_rank_std": None,
                    "mean_top1_vote": None,
                    "mean_margin_gate": None,
                    "top1_flip_rate": None,
                    "robust_changed_top1_rate": None,
                    "batches": [],
                }
            )
        return summary


class RiskAwareCEMSolver(RobustCEMSolver):
    """Backward-compatible adapter for the previous local robust CEM config."""

    def __init__(self, *args: Any, robust: Any | None = None, **kwargs: Any) -> None:
        if robust is not None:
            kwargs.setdefault("enabled", bool(_cfg_get(robust, "enabled", False)))
            kwargs.setdefault("num_views", int(_cfg_get(robust, "tta_num", 8)))
            kwargs.setdefault(
                "include_identity",
                bool(_cfg_get(robust, "tta_include_identity", True)),
            )
            kwargs.setdefault("view_type", "gaussian_noise")
            kwargs.setdefault("view_std", float(_cfg_get(robust, "tta_noise_std", 0.005)))
            kwargs.setdefault("perturb_pixels", bool(_cfg_get(robust, "robust_current", True)))
            kwargs.setdefault("perturb_goal", bool(_cfg_get(robust, "robust_goal", True)))
            kwargs.setdefault("robust_rescore", "elite")
            risk = str(_cfg_get(robust, "risk", "mean_std")).lower()
            risk_map = {"cvar": "quantile", "max": "worst"}
            kwargs.setdefault("score_mode", risk_map.get(risk, risk))
            kwargs.setdefault("beta", float(_cfg_get(robust, "lambda_std", 1.0)))
            kwargs.setdefault("quantile", float(_cfg_get(robust, "quantile_q", 0.8)))
            kwargs.setdefault("elite_rescore_multiplier", 1)
            kwargs.setdefault("record_history", bool(_cfg_get(robust, "log_debug", True)))
        super().__init__(*args, **kwargs)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
