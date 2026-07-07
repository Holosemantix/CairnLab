from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from acpc_flow import sample_latent_noise
from tools.repr_analysis.analyze_repr import (
    effective_rank,
    infer_history_size,
    load_dataset_samples,
    load_model,
    to_serializable,
)
from tools.repr_analysis.noise_sensitivity import _add_eval_corruption


DATASET_ROOTS = {
    "tworoom": "lewm-tworooms",
    "reacher": "lewm-reacher",
    "pusht": "lewm-pusht",
}
DATASET_NAMES = {
    "tworoom": "tworoom",
    "reacher": "reacher",
    "pusht": "pusht_expert_train",
}
STATE_KEYS = {
    "tworoom": "proprio",
    "reacher": "proprio",
    "pusht": None,
}


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.size(-1))


def _q(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float().cpu(), q))


def _quantiles(prefix: str, x: torch.Tensor) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(x.float().mean().cpu()),
        f"{prefix}_median": _q(x, 0.50),
        f"{prefix}_q75": _q(x, 0.75),
        f"{prefix}_q90": _q(x, 0.90),
        f"{prefix}_q95": _q(x, 0.95),
        f"{prefix}_q99": _q(x, 0.99),
    }


def _clean_knn_distance(clean: torch.Tensor, k: int) -> torch.Tensor:
    dist = torch.cdist(clean.float(), clean.float())
    eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
    dist = dist.masked_fill(eye, float("inf"))
    k_eff = min(k, max(1, dist.size(0) - 1))
    return dist.topk(k_eff, largest=False).values.mean(dim=1).clamp_min(1e-6)


def _paired_rank_and_crossing(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    state: torch.Tensor | None,
    *,
    k: int,
) -> dict[str, float]:
    dist = torch.cdist(corrupt.float(), clean.float())
    paired = dist.diag()
    paired_rank = (dist < paired[:, None]).sum(dim=1).float() + 1.0
    result = {
        "paired_clean_rank_median": _q(paired_rank, 0.50),
        "paired_clean_rank_q90": _q(paired_rank, 0.90),
    }
    if state is None:
        result.update(
            {
                "wrong_label_nn_rate": float("nan"),
                "closer_to_wrong_than_pair_rate": float("nan"),
                "same_label_topk_rate": float("nan"),
                "state_proxy_missing": 1.0,
            }
        )
        return result

    state = state.float()
    state_dist = torch.cdist(state, state)
    state_knn = _clean_knn_distance(state, k=k)
    wrong = state_dist > state_knn[:, None]
    nearest = dist.argmin(dim=1)
    wrong_nn = wrong[torch.arange(dist.size(0), device=dist.device), nearest]
    closer_wrong = (dist.masked_fill(~wrong, float("inf")) < paired[:, None]).any(dim=1)
    topk = dist.topk(min(k, dist.size(1)), largest=False).indices
    same_topk = (~wrong.gather(1, topk)).float().mean(dim=1)
    result.update(
        {
            "wrong_label_nn_rate": float(wrong_nn.float().mean().cpu()),
            "closer_to_wrong_than_pair_rate": float(closer_wrong.float().mean().cpu()),
            "same_label_topk_rate": float(same_topk.mean().cpu()),
            "state_proxy_missing": 0.0,
        }
    )
    return result


def _anisotropy(delta: torch.Tensor) -> dict[str, float]:
    x = delta.float() - delta.float().mean(dim=0, keepdim=True)
    if x.size(0) < 2:
        return {
            "effective_rank": float("nan"),
            "lambda_max_over_trace": float("nan"),
            "top1_eigen_ratio": float("nan"),
            "top5_eigen_ratio": float("nan"),
        }
    cov = x.T @ x / float(x.size(0) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0).sort(descending=True).values
    trace = eig.sum().clamp_min(1e-12)
    p = eig / trace
    entropy_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
    return {
        "effective_rank": float(entropy_rank.cpu()),
        "lambda_max_over_trace": float((eig[0] / trace).cpu()),
        "top1_eigen_ratio": float((eig[0] / trace).cpu()),
        "top5_eigen_ratio": float((eig[:5].sum() / trace).cpu()),
    }


def _task_alignment(delta: torch.Tensor, clean: torch.Tensor, state: torch.Tensor | None, k: int) -> dict[str, float]:
    if state is None:
        return {"task_alignment_mean": float("nan"), "task_alignment_q90": float("nan"), "task_alignment_q95": float("nan")}
    state_dist = torch.cdist(state.float(), state.float())
    state_knn = _clean_knn_distance(state.float(), k=k)
    wrong = state_dist > state_knn[:, None]
    clean_diff = clean[None, :, :] - clean[:, None, :]
    delta_n = F.normalize(delta.float(), dim=-1, eps=1e-8)
    diff_n = F.normalize(clean_diff.float(), dim=-1, eps=1e-8)
    cos = (delta_n[:, None, :] * diff_n).sum(dim=-1).masked_fill(~wrong, -1.0)
    max_align = cos.max(dim=1).values
    return {
        "task_alignment_mean": float(max_align.mean().cpu()),
        "task_alignment_q90": _q(max_align, 0.90),
        "task_alignment_q95": _q(max_align, 0.95),
    }


def _synthetic_coverage(
    clean: torch.Tensor,
    delta_norm: torch.Tensor,
    std_grid: list[float],
    mode: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    coverage: dict[str, dict[str, float]] = {}
    radii: dict[str, dict[str, float]] = {}
    clean_shape = clean.unsqueeze(1)
    for std in std_grid:
        noise = sample_latent_noise(
            clean_shape,
            std_min=std,
            std_max=std,
            mode=mode,
            relative=True,
            sample_per_token=True,
        ).squeeze(1)
        radius = torch.linalg.vector_norm(noise, dim=-1)
        key = f"{std:.4g}"
        radii[key] = {"q90": _q(radius, 0.90), "q95": _q(radius, 0.95), "q99": _q(radius, 0.99)}
        coverage[key] = {
            "q90": float((delta_norm <= radii[key]["q90"]).float().mean().cpu()),
            "q95": float((delta_norm <= radii[key]["q95"]).float().mean().cpu()),
            "q99": float((delta_norm <= radii[key]["q99"]).float().mean().cpu()),
        }
    return coverage, radii


def _predict_gap(model, clean_ctx: torch.Tensor, corrupt_ctx: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    clean_pred = model.predict(clean_ctx, act)
    corrupt_pred = model.predict(corrupt_ctx, act)
    return (clean_pred - corrupt_pred).pow(2).mean(dim=-1).reshape(-1)


def _synthetic_acpc_gaps(model, clean_seq: torch.Tensor, act: torch.Tensor, std_grid: list[float], mode: str) -> dict[str, dict[str, float]]:
    gaps: dict[str, dict[str, float]] = {}
    for std in std_grid:
        noise = sample_latent_noise(
            clean_seq,
            std_min=std,
            std_max=std,
            mode=mode,
            relative=True,
            sample_per_token=True,
        )
        gap = _predict_gap(model, clean_seq, clean_seq + noise, act)
        gaps[f"{std:.4g}"] = {"mean": float(gap.mean().cpu()), "q90": _q(gap, 0.90), "q95": _q(gap, 0.95)}
    return gaps


def _decision(metrics: Mapping[str, Any]) -> str:
    ratio_q90 = metrics.get("ratio_to_knn_q90", float("inf"))
    wrong_rate = metrics.get("wrong_label_nn_rate", float("nan"))
    closer_wrong = metrics.get("closer_to_wrong_than_pair_rate", float("nan"))
    cov_004 = metrics.get("coverage_q95_by_alpha", {}).get("0.04", 0.0)
    cov_008 = metrics.get("coverage_q95_by_alpha", {}).get("0.08", 0.0)
    pixel_gap = metrics.get("acpc_gap_q90", float("nan"))
    synth = metrics.get("synthetic_acpc_gap_q90_by_alpha", {})
    max_synth_gap = max([v for v in synth.values() if not math.isnan(v)], default=float("nan"))
    if (not math.isnan(wrong_rate) and wrong_rate >= 0.30) or (not math.isnan(closer_wrong) and closer_wrong >= 0.40):
        return "no_go"
    if not math.isnan(pixel_gap) and not math.isnan(max_synth_gap) and max_synth_gap > 0 and pixel_gap > 2.0 * max_synth_gap:
        return "no_go"
    if ratio_q90 >= 0.8 or (not math.isnan(wrong_rate) and wrong_rate >= 0.15) or cov_008 < 0.50:
        return "low"
    if ratio_q90 < 0.3 and (math.isnan(wrong_rate) or wrong_rate < 0.05) and cov_004 > 0.80:
        return "high"
    return "medium"


def _encode(model, pixels: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
    info = model.encode({"pixels": pixels, "action": action})
    return {k: v.detach() for k, v in info.items() if torch.is_tensor(v)}


def _audit_space(
    *,
    model,
    space: str,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    clean_emb_seq: torch.Tensor,
    corrupt_emb_seq: torch.Tensor,
    act_emb: torch.Tensor,
    state: torch.Tensor | None,
    std_grid: list[float],
    noise_mode: str,
    knn_k: int,
) -> dict[str, Any]:
    clean_flat = _flatten(clean)
    corrupt_flat = _flatten(corrupt)
    delta = corrupt_flat - clean_flat
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    knn = _clean_knn_distance(clean_flat, k=knn_k)
    ratio = delta_norm / knn
    state_flat = _flatten(state) if state is not None else None
    coverage, radii = _synthetic_coverage(clean_flat, delta_norm, std_grid, noise_mode)

    metrics: dict[str, Any] = {
        **_quantiles("delta_norm", delta_norm),
        **_quantiles("ratio_to_knn", ratio),
        **_paired_rank_and_crossing(clean_flat, corrupt_flat, state_flat, k=knn_k),
        **_anisotropy(delta),
        **_task_alignment(delta, clean_flat, state_flat, k=knn_k),
        "coverage_q95_by_alpha": {k: v["q95"] for k, v in coverage.items()},
        "coverage_by_alpha": coverage,
        "synthetic_radius_by_alpha": radii,
        "candidate_rank_metrics_computed": False,
    }
    if space == "emb":
        gap = _predict_gap(model, clean_emb_seq, corrupt_emb_seq, act_emb)
        synth_gaps = _synthetic_acpc_gaps(model, clean_emb_seq, act_emb, std_grid, noise_mode)
        metrics.update(
            {
                "acpc_gap_mean": float(gap.mean().cpu()),
                "acpc_gap_q90": _q(gap, 0.90),
                "acpc_gap_q95": _q(gap, 0.95),
                "synthetic_acpc_gap_q90_by_alpha": {k: v["q90"] for k, v in synth_gaps.items()},
                "synthetic_acpc_gap_by_alpha": synth_gaps,
            }
        )
    else:
        metrics.update(
            {
                "acpc_gap_mean": float("nan"),
                "acpc_gap_q90": float("nan"),
                "acpc_gap_q95": float("nan"),
                "synthetic_acpc_gap_q90_by_alpha": {},
                "acpc_gap_not_computed_reason": "predictor consumes post-projector emb, not encoder_feat",
            }
        )
    metrics["coverage_decision"] = _decision(metrics)
    return metrics


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.stablewm_home:
        os.environ["STABLEWM_HOME"] = args.stablewm_home
    task = args.task
    dataset_name = args.dataset_name or DATASET_NAMES[task]
    state_key = args.state_key if args.state_key != "auto" else STATE_KEYS.get(task)
    model = load_model(args.checkpoint, args.device)
    history_size = infer_history_size(model)
    batch = load_dataset_samples(
        dataset_name=dataset_name,
        state_key=state_key,
        n_sequences=args.num_samples,
        history_size=history_size,
        future_steps=max(1, args.future_steps),
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=args.seed,
        device=args.device,
    )
    clean_info = _encode(model, batch["pixels"], batch["action"])
    state = batch.get("state")
    act_emb = clean_info["act_emb"][:, :history_size]
    std_grid = [float(x) for x in args.synthetic_std_grid.split(",") if x]

    results: dict[str, dict[str, Any]] = {"emb": {}, "encoder_feat": {}}
    for c_idx, spec in enumerate(args.corruption):
        ctype, mag_s = spec.split(":", 1)
        mag = float(mag_s)
        corrupt_pixels = _add_eval_corruption(
            batch["pixels"], mag, args.seed + 7919 * (c_idx + 1), corruption_type=ctype
        )
        corrupt_info = _encode(model, corrupt_pixels, batch["action"])
        label = _corruption_label(ctype, mag)
        for space in ("emb", "encoder_feat"):
            if space not in clean_info or space not in corrupt_info:
                continue
            results[space][label] = _audit_space(
                model=model,
                space=space,
                clean=clean_info[space],
                corrupt=corrupt_info[space],
                clean_emb_seq=clean_info["emb"][:, :history_size],
                corrupt_emb_seq=corrupt_info["emb"][:, :history_size],
                act_emb=act_emb,
                state=state,
                std_grid=std_grid,
                noise_mode=args.synthetic_noise_mode,
                knn_k=args.knn_k,
            )

    decisions = [m["coverage_decision"] for by_space in results.values() for m in by_space.values()]
    emb_decisions = [m["coverage_decision"] for m in results.get("emb", {}).values()]
    recommendation = _recommendation(emb_decisions)
    return {
        "schema_version": "acpc-flow-coverage-core-v1",
        "task": task,
        "checkpoint": args.checkpoint,
        "dataset_name": dataset_name,
        "num_samples": args.num_samples,
        "history_size": history_size,
        "feature_spaces": [s for s in ("encoder_feat", "emb") if results.get(s)],
        "synthetic_noise_grid": std_grid,
        "candidate_rank_metrics_computed": False,
        "results": results,
        "recommendation": recommendation,
        "overall_decisions": decisions,
    }


def _recommendation(emb_decisions: list[str]) -> dict[str, Any]:
    no_go = emb_decisions.count("no_go")
    low = emb_decisions.count("low")
    worth = bool(emb_decisions) and no_go == 0 and low <= max(1, len(emb_decisions) // 2)
    if no_go:
        reason = "At least one emb stressor is no_go; do not make broad generalization claims."
    elif not worth:
        reason = "Most emb stressors are low coverage; run only targeted ablations if at all."
    else:
        reason = "Core coverage is not ruled out; proceed with small offline/latent ablations before training scale."
    return {
        "train_clean_only_latent_noise": worth,
        "preferred_feature_space": "emb",
        "suggested_noise_std_max": 0.04,
        "no_go_stressors": [],
        "worth_doing": worth,
        "reason": reason,
    }


def _corruption_label(ctype: str, magnitude: float) -> str:
    if ctype == "gaussian_noise":
        return f"gaussian_std{magnitude:g}"
    if ctype == "gaussian_blur":
        return f"blur_ks{int(magnitude)}"
    if ctype == "resize":
        return f"resize_factor{magnitude:g}"
    return f"{ctype}_{magnitude:g}"


def _write_outputs(report: Mapping[str, Any], output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    with json_path.open("w") as f:
        json.dump(to_serializable(report), f, indent=2, sort_keys=True)
    rows = []
    for space, by_corr in report["results"].items():
        for corr, metrics in by_corr.items():
            flat = {"feature_space": space, "corruption": corr}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    continue
                flat[k] = v
            rows.append(flat)
    fieldnames = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ACPC-Flow core coverage audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(DATASET_NAMES), default="tworoom")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--stablewm-home", default=None)
    parser.add_argument("--state-key", default="auto")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--future-steps", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic-std-grid", default="0.01,0.02,0.04,0.08,0.12")
    parser.add_argument("--synthetic-noise-mode", default="token_std")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument(
        "--corruption",
        action="append",
        default=None,
        help="Corruption spec type:magnitude; repeatable.",
    )
    parser.add_argument("--output-dir", default="assets/paper1_data")
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()
    if args.corruption is None:
        args.corruption = [
            "gaussian_noise:0.03",
            "gaussian_noise:0.05",
            "gaussian_noise:0.08",
            "gaussian_blur:7",
            "resize:0.5",
        ]

    report = run_audit(args)
    date = datetime.utcnow().strftime("%Y%m%d")
    prefix = args.prefix or f"acpc_flow_coverage_{args.task}_{Path(args.checkpoint).parent.name}_{date}"
    json_path, csv_path = _write_outputs(report, Path(args.output_dir), prefix)
    print(f"[coverage_audit] wrote {json_path}")
    print(f"[coverage_audit] wrote {csv_path}")
    for space, by_corr in report["results"].items():
        print(f"[{space}]")
        for corr, metrics in by_corr.items():
            decision = metrics["coverage_decision"]
            ratio_q90 = metrics["ratio_to_knn_q90"]
            coverage_004 = metrics["coverage_q95_by_alpha"].get("0.04", float("nan"))
            wrong_nn = metrics["wrong_label_nn_rate"]
            print(
                f"  {corr}: decision={decision} "
                f"ratio_q90={ratio_q90:.3g} "
                f"coverage@0.04/q95={coverage_004:.3g} "
                f"wrong_nn={wrong_nn:.3g}"
            )
    print("recommendation: " + report["recommendation"]["reason"])


if __name__ == "__main__":
    main()
