"""Run Paper 1 task-semantic margin pass-rate probes.

This is a lightweight selective-ACPC guard: for each checkpoint, compare the
same-state clean/noisy rollout radius against hard semantic-different clean
rollout distances on a task-specific state proxy. It is intended to complement
ACPC/PCC/CRA/MAF with one compact semantic discriminability readout per task.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_MANIFEST_DIR = DATA_DIR / "training_seed_eval_manifests"
DEFAULT_OUT = DATA_DIR / "semantic_margin_passrate_lewm_three_seed.json"
SEEDS = (3072, 3073, 3074)
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEMANTIC_STATE_KEYS = {
    "TwoRoom": "pos_agent",
    "PushT": "state",
    "Reacher": "observation",
    "Cube": "observation",
}
SEMANTIC_FACTORS = {
    "TwoRoom": "agent room/doorway/topology proxy from pos_agent",
    "PushT": "T-block and pusher pose proxy from state",
    "Reacher": "joint/target geometry proxy from observation",
    "Cube": "cube pose and gripper-object proxy from observation",
}
STD_KEYS = ("0.0", "0.08")


def _jsonable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.detach().float().cpu(), q))


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values)


def _pstdev(values: Sequence[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / len(values))


def _success(entry: Mapping[str, Any], metric: str) -> float:
    return float(entry.get("metrics", {}).get(metric, {}).get("mean", float("nan")))


def _semantic_metrics(
    *,
    model,
    task: str,
    batch: Mapping[str, torch.Tensor],
    noisy_batch: Mapping[str, torch.Tensor],
    history_size: int,
    rollout_horizon: int,
    embedding_space: str,
    semantic_quantile: float,
    margin_delta: float,
) -> dict[str, Any]:
    clean_outputs = phase0.encode_sequences(model, phase0._clone_batch(batch))
    noisy_outputs = phase0.encode_sequences(model, phase0._clone_batch(noisy_batch))
    clean_emb = phase0.get_embedding_space(clean_outputs, embedding_space).detach()
    noisy_emb = phase0.get_embedding_space(noisy_outputs, embedding_space).detach()
    act_emb = clean_outputs["act_emb"].detach()
    max_steps = min(rollout_horizon, max(0, act_emb.size(1) - history_size + 1))
    clean_chain = phase0._autoregressive_rollout(model, clean_emb[:, :history_size], act_emb, history_size, max_steps)
    noisy_chain = phase0._autoregressive_rollout(model, noisy_emb[:, :history_size], act_emb, history_size, max_steps)
    final_idx = history_size + max_steps - 1 if max_steps else history_size - 1
    clean_final = clean_chain[:, final_idx].float()
    noisy_final = noisy_chain[:, final_idx].float()
    same_radius = torch.linalg.vector_norm(clean_final - noisy_final, dim=-1)

    state = batch["state"].float()
    state_idx = min(state.size(1) - 1, final_idx)
    state_final = state[:, state_idx].reshape(state.size(0), -1)
    state_dist = torch.cdist(state_final, state_final, p=2)
    latent_dist = torch.cdist(clean_final, clean_final, p=2)
    n = state_dist.size(0)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=state_dist.device)
    candidate_state = state_dist[offdiag]
    if candidate_state.numel() == 0:
        return {
            "semantic_state_key": SEMANTIC_STATE_KEYS[task],
            "semantic_factor": SEMANTIC_FACTORS[task],
            "semantic_pair_rule": "hard semantic-different pair above state-distance quantile",
            "semantic_pair_count": 0,
            "semantic_distance_threshold": float("nan"),
            "same_state_noisy_radius_median": _safe_quantile(same_radius, 0.5),
            "semantic_diff_l2_median": float("nan"),
            "semantic_margin_median": float("nan"),
            "semantic_margin_pass_rate": float("nan"),
            "semantic_discriminability_ratio": float("nan"),
        }
    threshold = torch.quantile(candidate_state, float(semantic_quantile))
    valid = offdiag & (state_dist >= threshold)
    hard_diffs = []
    aligned_same = []
    for i in range(n):
        row = valid[i]
        if bool(row.any()):
            hard_diffs.append(torch.min(latent_dist[i][row]))
            aligned_same.append(same_radius[i])
    if not hard_diffs:
        hard = torch.empty(0, device=clean_final.device)
        same = torch.empty(0, device=clean_final.device)
    else:
        hard = torch.stack(hard_diffs)
        same = torch.stack(aligned_same)
    margins = hard - same
    passes = margins > float(margin_delta)
    same_med = _safe_quantile(same, 0.5)
    diff_med = _safe_quantile(hard, 0.5)
    return {
        "semantic_state_key": SEMANTIC_STATE_KEYS[task],
        "semantic_factor": SEMANTIC_FACTORS[task],
        "semantic_pair_rule": "hard semantic-different pair above state-distance quantile",
        "semantic_pair_count": int(hard.numel()),
        "semantic_distance_threshold": float(threshold.detach().cpu()),
        "same_state_noisy_radius_median": same_med,
        "semantic_diff_l2_median": diff_med,
        "semantic_margin_median": _safe_quantile(margins, 0.5),
        "semantic_margin_pass_rate": float(passes.float().mean().detach().cpu()) if passes.numel() else float("nan"),
        "semantic_discriminability_ratio": diff_med / same_med if same_med > 0 else float("nan"),
    }


def _run_row(task: str, std_key: str, entry: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model_file, tried = phase0.resolve_model_file(str(entry.get("path", "")), str(entry.get("subdir", "")), [])
    base = {
        "training_seed": int(args.training_seed),
        "task": task,
        "std_key": std_key,
        "subdir": entry.get("subdir"),
        "run_path": entry.get("path"),
        "model_file": str(model_file) if model_file else None,
        "clean_success": _success(entry, "clean"),
        "pixels_std0.08_success": _success(entry, "pixels_std0.08"),
        "corruption_drop": _success(entry, "clean") - _success(entry, "pixels_std0.08"),
    }
    if model_file is None:
        return {**base, "status": "skipped_missing_model", "model_search_dirs": tried}
    try:
        phase0._ensure_runtime_deps()
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            model = phase0.load_model(str(model_file), device)
            history_size = phase0.infer_history_size(model)
            future_steps = max(args.future_steps, args.rollout_horizon + 1)
            state_key = SEMANTIC_STATE_KEYS[task]
            batch = phase0.load_dataset_samples(
                dataset_name=phase0.TASK_DATASETS[task],
                state_key=state_key,
                n_sequences=args.n_sequences,
                history_size=history_size,
                future_steps=future_steps,
                frameskip=args.frameskip,
                img_size=args.img_size,
                seed=args.seed,
                device=device,
            )
            noisy_batch = phase0.make_paired_noisy_batch(
                batch,
                history_size=history_size,
                noise_std=args.noise_std,
                seed=args.seed + 1009,
                corruption_type=args.corruption_type,
                corrupt_goal=False,
            )
            spaces = phase0.get_model_spaces(model)
            embedding_space = args.embedding_space or spaces["inference_cost_space"]
            metrics = _semantic_metrics(
                model=model,
                task=task,
                batch=batch,
                noisy_batch=noisy_batch,
                history_size=history_size,
                rollout_horizon=args.rollout_horizon,
                embedding_space=embedding_space,
                semantic_quantile=args.semantic_quantile,
                margin_delta=args.margin_delta,
            )
            return {
                **base,
                "status": "ok",
                "history_size": int(history_size),
                "n_sequences": int(args.n_sequences),
                "rollout_horizon": int(args.rollout_horizon),
                "embedding_space": embedding_space,
                "noise_std": float(args.noise_std),
                **metrics,
            }
    except Exception as exc:  # noqa: BLE001 - row-level artifact should record failures.
        return {**base, "status": "error", "error": repr(exc)}


def _summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task in TASKS:
        for std_key in STD_KEYS:
            task_rows = [r for r in rows if r["task"] == task and r["std_key"] == std_key and r["status"] == "ok"]
            if not task_rows:
                continue
            summary = {"task": task, "std_key": std_key, "training_seeds": sorted(int(r["training_seed"]) for r in task_rows), "n_training_seeds": len(task_rows)}
            for key in ("semantic_margin_pass_rate", "semantic_discriminability_ratio", "same_state_noisy_radius_median", "semantic_diff_l2_median", "semantic_margin_median"):
                vals = [float(r[key]) for r in task_rows]
                summary[f"{key}_mean"] = _mean(vals)
                summary[f"{key}_pstdev"] = _pstdev(vals)
            out.append(summary)
    return out


def _iter_rows(manifest: Mapping[str, Any], tasks: Sequence[str], std_keys: Sequence[str]) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    for task in tasks:
        for std_key in std_keys:
            yield task, std_key, manifest[task][std_key]


def run_for_seed(seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    args.training_seed = seed
    manifest = _load(args.manifest_dir / f"lewm_seed{seed}_evals.json")
    rows = []
    for task, std_key, entry in _iter_rows(manifest, args.tasks, args.std_keys):
        rows.append(_run_row(task, std_key, entry, args))
        if args.limit is not None and len(rows) >= args.limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--std-keys", nargs="+", default=list(STD_KEYS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--semantic-quantile", type=float, default=0.5)
    parser.add_argument("--margin-delta", type=float, default=0.0)
    parser.add_argument("--state-key-seed", dest="seed", type=int, default=9101)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embedding-space", choices=["raw", "normalized"], default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows.extend(run_for_seed(seed, args))
    coverage = {
        f"{task}:{std_key}": sorted(int(r["training_seed"]) for r in rows if r["task"] == task and r["std_key"] == std_key and r["status"] == "ok")
        for task in args.tasks
        for std_key in args.std_keys
    }
    payload = {
        "metadata": {
            "schema_version": "paper1-semantic-margin-passrate-0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "fixed endpoint semantic guard over LeWM training seeds 3072/3073/3074; no retraining",
            "semantic_state_keys": SEMANTIC_STATE_KEYS,
            "semantic_factors": SEMANTIC_FACTORS,
            "std_keys": list(args.std_keys),
            "training_seeds": list(args.seeds),
            "semantic_quantile": float(args.semantic_quantile),
            "margin_delta": float(args.margin_delta),
        },
        "rows": rows,
        "summary_rows": _summaries(rows),
        "coverage": coverage,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    try:
        display_out = args.out.relative_to(ROOT)
    except ValueError:
        display_out = args.out
    print(f"wrote {display_out}")
    print("status counts:", counts)


if __name__ == "__main__":
    main()
