#!/usr/bin/env python3
"""Sample-level fixed-pool certificate audit for Paper 1.

This script recomputes fixed-pool candidate costs from checkpoints. Unlike the
retained phase-0 summaries, it keeps the sample-level maximum paired cost drift
needed for the theorem's sufficient fixed-pool event:

    max_j |C_h(a_j)-C_tilde_h(a_j)| < clean_top1_top2_margin / 2.

The output is an audit artifact. It does not evaluate adaptive CEM, repeated
replanning, or closed-loop behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0
from tools.paper1_margin_flip_curve import MANIFEST_DIR, SEEDS, TASKS, _success

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / 'paper1' / 'results' / 'sample_level_certificate_audit.json'
DEFAULT_CSV = ROOT / 'paper1' / 'results' / 'sample_level_certificate_audit.csv'
DEFAULT_SAMPLE_CSV = ROOT / 'paper1' / 'results' / 'sample_level_certificate_samples.csv'
STD_KEYS = ('0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08')
EPS_QUANTILES = (0.90, 0.95, 0.99, 0.995, 0.999)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _q(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float('nan')
    return float(torch.quantile(x.detach().float().cpu(), float(q)).item())


def _mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float('nan')
    return float(x.detach().float().mean().cpu().item())


def _costs_for_branch(model, batch: Mapping[str, torch.Tensor], candidates: torch.Tensor, *, history_size: int) -> torch.Tensor:
    return model.get_cost(phase0._cost_info(batch, history_size), candidates)


def _resolve(entry: Mapping[str, Any], model_roots: Sequence[Path]) -> tuple[Path | None, list[str]]:
    return phase0.resolve_model_file(str(entry.get('path', '')), str(entry.get('subdir', '')), model_roots)


def run_checkpoint(*, seed: int, task: str, std_key: str, entry: Mapping[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    phase0._ensure_runtime_deps()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_roots = [Path(p).expanduser() for p in args.model_root]
    model_file, tried = _resolve(entry, model_roots)
    base = {
        'training_seed': int(seed),
        'task': task,
        'std_key': std_key,
        'subdir': entry.get('subdir'),
        'run_path': entry.get('path'),
        'model_file': str(model_file) if model_file else None,
        'model_search_dirs': tried,
        'clean_success': _success(entry, 'clean'),
        'pixels_std0.08_success': _success(entry, 'pixels_std0.08'),
        'noise_std': float(args.noise_std),
    }
    if model_file is None:
        return {**base, 'status': 'skipped_missing_model'}, []

    with torch.no_grad():
        model = phase0.load_model(str(model_file), device)
        history_size = phase0.infer_history_size(model)
        future_steps = max(args.future_steps, args.rollout_horizon + 1)
        batch = phase0.load_dataset_samples(
            dataset_name=phase0.TASK_DATASETS[task],
            state_key=args.state_key,
            n_sequences=args.n_sequences,
            history_size=history_size,
            future_steps=future_steps,
            frameskip=args.frameskip,
            img_size=args.img_size,
            seed=seed,
            device=device,
        )
        noisy_batch = phase0.make_paired_noisy_batch(
            batch,
            history_size=history_size,
            noise_std=args.noise_std,
            seed=seed + 1009,
            corruption_type=args.corruption_type,
            corrupt_goal=False,
        )
        candidates = phase0.build_action_candidates(
            batch['action'],
            history_size=history_size,
            future_steps=future_steps,
            random_action_trials=args.random_action_trials,
            seed=seed + 2027,
        )
        clean_costs = _costs_for_branch(model, batch, candidates, history_size=history_size)
        noisy_costs = _costs_for_branch(model, noisy_batch, candidates, history_size=history_size)

    clean = clean_costs.detach().float().cpu()
    noisy = noisy_costs.detach().float().cpu()
    abs_diff = (clean - noisy).abs()
    clean_sorted = torch.sort(clean, dim=1).values
    margins = clean_sorted[:, 1] - clean_sorted[:, 0]
    max_drift = abs_diff.max(dim=1).values
    mean_drift = abs_diff.mean(dim=1)
    flat_drift = abs_diff.reshape(-1)
    clean_best = torch.argmin(clean, dim=1)
    noisy_best = torch.argmin(noisy, dim=1)
    flips = clean_best != noisy_best
    cert_pass = max_drift < (margins / 2.0)

    eps_rows: dict[str, dict[str, float]] = {}
    for q in args.eps_quantiles:
        eps = _q(flat_drift, q)
        alpha_hat = float((flat_drift > eps).float().mean().item()) if math.isfinite(eps) else float('nan')
        margin_fail = float((margins <= 2.0 * eps).float().mean().item()) if math.isfinite(eps) else float('nan')
        eps_rows[f'q{int(round(q * 1000)):03d}'] = {
            'epsilon': eps,
            'alpha_hat': alpha_hat,
            'k_alpha': float(clean.size(1)) * alpha_hat if math.isfinite(alpha_hat) else float('nan'),
            'margin_fail_rate': margin_fail,
            'union_bound_proxy': min(1.0, float(clean.size(1)) * alpha_hat + margin_fail) if math.isfinite(alpha_hat) and math.isfinite(margin_fail) else float('nan'),
        }

    row = {
        **base,
        'status': 'ok',
        'n_sequences': int(args.n_sequences),
        'candidate_count': int(clean.size(1)),
        'future_steps': int(future_steps),
        'cost_drift_abs_q50': _q(flat_drift, 0.50),
        'cost_drift_abs_q90': _q(flat_drift, 0.90),
        'cost_drift_abs_q95': _q(flat_drift, 0.95),
        'cost_drift_abs_q99': _q(flat_drift, 0.99),
        'sample_mean_drift_q90': _q(mean_drift, 0.90),
        'sample_max_drift_q50': _q(max_drift, 0.50),
        'sample_max_drift_q90': _q(max_drift, 0.90),
        'sample_max_drift_q95': _q(max_drift, 0.95),
        'sample_max_drift_q99': _q(max_drift, 0.99),
        'clean_margin_q10': _q(margins, 0.10),
        'clean_margin_q25': _q(margins, 0.25),
        'clean_margin_q50': _q(margins, 0.50),
        'clean_margin_q90': _q(margins, 0.90),
        'certificate_gap_q10_q95': _q(margins, 0.10) - 2.0 * _q(max_drift, 0.95),
        'certificate_gap_q50_q95': _q(margins, 0.50) - 2.0 * _q(max_drift, 0.95),
        'sample_cert_pass_rate': _mean(cert_pass.float()),
        'sample_top1_flip_rate': _mean(flips.float()),
        'flip_when_cert_pass_rate': _mean(flips[cert_pass].float()) if bool(cert_pass.any()) else float('nan'),
        'flip_when_cert_fail_rate': _mean(flips[~cert_pass].float()) if bool((~cert_pass).any()) else float('nan'),
        'epsilon_tail_rows': eps_rows,
        'notes': 'sample-level fixed-pool sufficient-event audit; not adaptive CEM or closed-loop guarantee',
    }

    sample_rows: list[dict[str, Any]] = []
    if args.include_samples:
        for i in range(clean.size(0)):
            sample_rows.append({
                'training_seed': int(seed),
                'task': task,
                'std_key': std_key,
                'sample_index': int(i),
                'candidate_count': int(clean.size(1)),
                'clean_margin': float(margins[i].item()),
                'sample_max_drift': float(max_drift[i].item()),
                'sample_mean_drift': float(mean_drift[i].item()),
                'cert_pass': bool(cert_pass[i].item()),
                'top1_flip': bool(flips[i].item()),
                'clean_best': int(clean_best[i].item()),
                'noisy_best': int(noisy_best[i].item()),
            })
    return row, sample_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields = [k for k in rows[0].keys() if k not in {'model_search_dirs', 'epsilon_tail_rows'}]
    for key in sorted({k for r in rows for k in r.keys() if k not in fields and k not in {'model_search_dirs', 'epsilon_tail_rows'}}):
        fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS))
    p.add_argument('--std-keys', nargs='+', default=list(STD_KEYS))
    p.add_argument('--eval-manifest-dir', type=Path, default=MANIFEST_DIR)
    p.add_argument('--model-root', action='append', default=[])
    p.add_argument('--out-json', type=Path, default=DEFAULT_JSON)
    p.add_argument('--out-csv', type=Path, default=DEFAULT_CSV)
    p.add_argument('--sample-csv', type=Path, default=DEFAULT_SAMPLE_CSV)
    p.add_argument('--include-samples', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--n-sequences', type=int, default=100)
    p.add_argument('--future-steps', type=int, default=9)
    p.add_argument('--rollout-horizon', type=int, default=8)
    p.add_argument('--random-action-trials', type=int, default=64)
    p.add_argument('--noise-std', type=float, default=0.08)
    p.add_argument('--corruption-type', default='gaussian_noise')
    p.add_argument('--state-key', default=None)
    p.add_argument('--frameskip', type=int, default=5)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--eps-quantiles', type=float, nargs='+', default=list(EPS_QUANTILES))
    p.add_argument('--device', default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    specs: list[tuple[int, str, str, Mapping[str, Any]]] = []
    for seed in args.seeds:
        manifest = _load(args.eval_manifest_dir / f'lewm_seed{seed}_evals.json')
        for task in args.tasks:
            for std_key in args.std_keys:
                entry = manifest.get(task, {}).get(std_key)
                if entry is None:
                    rows.append({'status': 'skipped_missing_manifest', 'training_seed': seed, 'task': task, 'std_key': std_key})
                    continue
                specs.append((seed, task, std_key, entry))
    if args.limit is not None:
        specs = specs[: args.limit]
    for idx, (seed, task, std_key, entry) in enumerate(specs, start=1):
        print(f'[{idx}/{len(specs)}] {task} seed{seed} std{std_key}', flush=True)
        try:
            row, sample_rows = run_checkpoint(seed=seed, task=task, std_key=std_key, entry=entry, args=args)
        except Exception as exc:  # noqa: BLE001 - audit should record per-row failures.
            row, sample_rows = {'status': 'error', 'training_seed': seed, 'task': task, 'std_key': std_key, 'error': repr(exc)}, []
        rows.append(row)
        samples.extend(sample_rows)
    payload = {
        'metadata': {
            'schema_version': 'paper1-sample-level-certificate-0.1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'seeds': list(args.seeds),
            'tasks': list(args.tasks),
            'std_keys': list(args.std_keys),
            'n_sequences': int(args.n_sequences),
            'candidate_count': int(args.random_action_trials) + 1,
            'noise_std': float(args.noise_std),
            'note': 'Fixed-pool sample-level sufficient-event audit; closed-loop evaluation is not run.',
        },
        'rows': rows,
    }
    if args.include_samples:
        payload['sample_rows'] = samples
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(payload), indent=2))
    _write_csv(args.out_csv, rows)
    if args.include_samples:
        _write_csv(args.sample_csv, samples)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get('status'))] = counts.get(str(row.get('status')), 0) + 1
    print(f'wrote {args.out_json}')
    print(f'wrote {args.out_csv}')
    if args.include_samples:
        print(f'wrote {args.sample_csv}')
    print('status counts:', counts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
