"""Selective-contraction branch diagnostics for Paper 1.

This script is intentionally separate from the paper-facing figure generator.
It answers a narrower mechanism question on existing full-sequence
perturbed-target checkpoints:

    do same-state perturbation basins shrink, and what happens to simple
    state/action discriminability proxies?

The default path only reads released JSON artifacts and writes a compact branch
table.  The optional ``--plot-3d`` path loads checkpoints and dataset windows to
render real encoder/predictor feature clouds with PCA-to-3D, so it is eval-only
but not artifact-only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
DEFAULT_DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_FIG_DIR = ROOT / "assets" / "paper1_figs" / "selective_contraction_3d"
DEFAULT_OUT_JSON = DEFAULT_DATA_DIR / "selective_contraction_fullseq_branch.json"
DEFAULT_OUT_MD = DEFAULT_DATA_DIR / "selective_contraction_fullseq_branch.md"
TASK_DATASETS = {
    "TwoRoom": "tworoom",
    "PushT": "pusht_expert_train",
    "Reacher": "reacher",
    "Cube": "ogbench/cube_single_expert",
}


@dataclass(frozen=True)
class CkptSpec:
    label: str
    task: str
    std_key: str
    subdir: str
    model_file: Path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _ratio(num: Any, den: Any) -> float:
    if not _finite(num) or not _finite(den) or abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num) / float(den)


def _drop_frac(base: Any, best: Any) -> float:
    if not _finite(base) or abs(float(base)) <= 1e-12 or not _finite(best):
        return float("nan")
    return 1.0 - float(best) / float(base)


def _fmt(x: Any, digits: int = 3) -> str:
    if not _finite(x):
        return "n/a"
    return f"{float(x):.{digits}g}"


def _fmt_arrow(a: Any, b: Any, digits: int = 3) -> str:
    return f"{_fmt(a, digits)} -> {_fmt(b, digits)}"


def _rows_by_task_std(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (str(row["task"]), str(row["std_key"]))
        out[key] = row
    return out


def _phase_rows_by_task_std(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("method") != "LeWM":
            continue
        out[(str(row["task"]), str(row["std_key"]))] = row
    return out


def _best_row(rows: Sequence[Mapping[str, Any]], robust_metric: str) -> Mapping[str, Any]:
    return max(
        rows,
        key=lambda r: (
            float(r.get(robust_metric, float("nan"))),
            float(r.get("clean_success", float("nan"))),
        ),
    )


def build_summary(
    *,
    acpc_basin_path: Path,
    acpc_phase0_path: Path,
    robust_metric: str,
) -> dict[str, Any]:
    basin_payload = _load_json(acpc_basin_path)
    phase_payload = _load_json(acpc_phase0_path)
    basin_rows = [r for r in basin_payload["rows"] if r.get("status") == "ok"]
    phase_by_key = _phase_rows_by_task_std(phase_payload["rows"])

    out_rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [r for r in basin_rows if r["task"] == task]
        base = next(r for r in task_rows if r["std_key"] == "0.0")
        best = _best_row(task_rows, robust_metric)
        phase_base = phase_by_key.get((task, "0.0"), {})
        phase_best = phase_by_key.get((task, str(best["std_key"])), {})

        re_base = base["encoder_view_pair_l2_norm_by_nn"]
        re_best = best["encoder_view_pair_l2_norm_by_nn"]
        rf_base = base["pred_view_pair_l2_norm_by_transition"]
        rf_best = best["pred_view_pair_l2_norm_by_transition"]
        clean_nn_base = base["clean_nn_l2_median"]
        clean_nn_best = best["clean_nn_l2_median"]
        trans_base = base["clean_transition_l2_median"]
        trans_best = best["clean_transition_l2_median"]

        adm_base = phase_base.get("adm_l2_median", float("nan"))
        adm_best = phase_best.get("adm_l2_median", float("nan"))
        sprr_base = phase_base.get("sprr", float("nan"))
        sprr_best = phase_best.get("sprr", float("nan"))

        out_rows.append(
            {
                "task": task,
                "target_view_branch": "full_sequence_perturbed_target",
                "best_std_key": str(best["std_key"]),
                "best_subdir": best.get("subdir"),
                "clean_success_base": base["clean_success"],
                f"{robust_metric}_base": base[robust_metric],
                "clean_success_best": best["clean_success"],
                f"{robust_metric}_best": best[robust_metric],
                "encoder_radius_RE_base": re_base,
                "encoder_radius_RE_best": re_best,
                "encoder_radius_RE_drop_frac": _drop_frac(re_base, re_best),
                "prediction_radius_RF_base": rf_base,
                "prediction_radius_RF_best": rf_best,
                "prediction_radius_RF_drop_frac": _drop_frac(rf_base, rf_best),
                "prediction_selective_ratio_base": _ratio(1.0, rf_base),
                "prediction_selective_ratio_best": _ratio(1.0, rf_best),
                "clean_nn_l2_base": clean_nn_base,
                "clean_nn_l2_best": clean_nn_best,
                "clean_nn_l2_ratio_best_over_base": _ratio(clean_nn_best, clean_nn_base),
                "clean_transition_l2_base": trans_base,
                "clean_transition_l2_best": trans_best,
                "clean_transition_l2_ratio_best_over_base": _ratio(trans_best, trans_base),
                "phase0_aux_pxgoal_adm_l2_base": adm_base,
                "phase0_aux_pxgoal_adm_l2_best": adm_best,
                "phase0_aux_pxgoal_adm_ratio_best_over_base": _ratio(adm_best, adm_base),
                "phase0_aux_pxgoal_sprr_base": sprr_base,
                "phase0_aux_pxgoal_sprr_best": sprr_best,
                "readable_conclusion": _readable_conclusion(
                    re_base=re_base,
                    re_best=re_best,
                    rf_base=rf_base,
                    rf_best=rf_best,
                    trans_base=trans_base,
                    trans_best=trans_best,
                    adm_base=adm_base,
                    adm_best=adm_best,
                ),
            }
        )

    return {
        "metadata": {
            "schema_version": "paper1-selective-contraction-branch-0.1",
            "source_acpc_basin": str(acpc_basin_path),
            "source_acpc_phase0": str(acpc_phase0_path),
            "robust_metric": robust_metric,
            "branch": "existing full-sequence perturbed-target LeWM sweep",
            "interpretation": (
                "RE/RF are same-state perturbation basin radii from the primary "
                "observation-only ACPC basin diagnostic. ADM/SPRR are auxiliary "
                "pixels+goal Phase-0 proxies and should be read only as branch "
                "sanity checks, not paper-facing proof."
            ),
        },
        "rows": out_rows,
    }


def _readable_conclusion(
    *,
    re_base: float,
    re_best: float,
    rf_base: float,
    rf_best: float,
    trans_base: float,
    trans_best: float,
    adm_base: float,
    adm_best: float,
) -> str:
    same_state = (
        "same-state encoder/predictor basins shrink"
        if re_best < re_base and rf_best < rf_base
        else "same-state basin shrinkage is not monotone"
    )
    trans = _ratio(trans_best, trans_base)
    adm = _ratio(adm_best, adm_base)
    if _finite(adm):
        if adm >= 0.95:
            disc = "auxiliary ADM is preserved"
        elif adm >= 0.8:
            disc = "auxiliary ADM mildly decreases"
        else:
            disc = "auxiliary ADM decreases"
    elif _finite(trans):
        if trans >= 0.95:
            disc = "transition scale is preserved"
        elif trans >= 0.8:
            disc = "transition scale mildly decreases"
        else:
            disc = "transition scale decreases"
    else:
        disc = "discriminability proxy unavailable"
    return f"{same_state}; {disc}."


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Selective-Contraction Branch Table",
        "",
        "Scope: existing LeWM full-sequence perturbed-target sweep. This is a branch diagnostic, not a new main claim.",
        "",
        "| Task | best std | px0.08 success | encoder radius R_E | prediction radius R_F | clean NN L2 | transition L2 | aux ADM | aux SPRR | read |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    metric = payload["metadata"]["robust_metric"]
    for row in payload["rows"]:
        lines.append(
            "| {task} | {std} | {succ} | {re} | {rf} | {nn} | {tr} | {adm} | {sprr} | {read} |".format(
                task=row["task"],
                std=row["best_std_key"],
                succ=_fmt_arrow(row[f"{metric}_base"], row[f"{metric}_best"], 3),
                re=_fmt_arrow(row["encoder_radius_RE_base"], row["encoder_radius_RE_best"], 3),
                rf=_fmt_arrow(row["prediction_radius_RF_base"], row["prediction_radius_RF_best"], 3),
                nn=_fmt_arrow(row["clean_nn_l2_base"], row["clean_nn_l2_best"], 3),
                tr=_fmt_arrow(row["clean_transition_l2_base"], row["clean_transition_l2_best"], 3),
                adm=_fmt_arrow(
                    row["phase0_aux_pxgoal_adm_l2_base"],
                    row["phase0_aux_pxgoal_adm_l2_best"],
                    3,
                ),
                sprr=_fmt_arrow(
                    row["phase0_aux_pxgoal_sprr_base"],
                    row["phase0_aux_pxgoal_sprr_best"],
                    3,
                ),
                read=row["readable_conclusion"],
            )
        )
    lines.extend(
        [
            "",
            "Reading: lower R_E/R_F means a smaller same-state perturbation basin. "
            "Higher SPRR means the auxiliary action-distance margin is larger relative "
            "to paired rollout disagreement. ADM/SPRR come from the exploratory pixels+goal "
            "Phase-0 diagnostic, so they are supportive visualization/branch evidence only.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_plot_deps():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    return plt


def _ensure_runtime_deps():
    from tools import paper1_phase0_acpc as phase0

    phase0._ensure_runtime_deps()
    return phase0


def _clone_batch(phase0, batch: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v.clone() if phase0.torch.is_tensor(v) else copy.deepcopy(v) for k, v in batch.items()}


def _checkpoint_specs(
    *,
    task: str,
    summary: Mapping[str, Any],
    acpc_basin_path: Path,
) -> list[CkptSpec]:
    rows = _load_json(acpc_basin_path)["rows"]
    task_rows = [r for r in rows if r.get("status") == "ok" and r["task"] == task]
    base = next(r for r in task_rows if r["std_key"] == "0.0")
    branch_row = next(r for r in summary["rows"] if r["task"] == task)
    best = next(r for r in task_rows if str(r["std_key"]) == str(branch_row["best_std_key"]))
    specs = []
    for label, row in (("base", base), ("fullseq_robust", best)):
        model_file = Path(str(row["model_file"]))
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model file for {task}/{label}: {model_file}")
        specs.append(
            CkptSpec(
                label=label,
                task=task,
                std_key=str(row["std_key"]),
                subdir=str(row["subdir"]),
                model_file=model_file,
            )
        )
    return specs


def _pca_fit_transform(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    flat = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].T
    out = []
    for a in arrays:
        z = (a.reshape(-1, a.shape[-1]) - mean) @ components
        out.append(z.reshape(a.shape[:-1] + (3,)))
    return out


def _nearest_original_indices(features: np.ndarray, anchors: Sequence[int]) -> dict[int, int]:
    """Nearest other original-state index for each anchor in original feature space."""
    clean = features[0].reshape(features.shape[1], features.shape[2])
    dists = np.linalg.norm(clean[:, None, :] - clean[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return {int(i): int(np.argmin(dists[int(i)])) for i in anchors}


def _axis_limits(arrays: Sequence[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    flat = np.concatenate([a.reshape(-1, 3) for a in arrays], axis=0)
    limits = []
    for dim in range(3):
        lo = float(np.nanmin(flat[:, dim]))
        hi = float(np.nanmax(flat[:, dim]))
        pad = 0.08 * max(hi - lo, 1e-6)
        limits.append((lo - pad, hi + pad))
    return limits[0], limits[1], limits[2]


def _extract_view_features(
    *,
    phase0,
    model,
    batch: Mapping[str, Any],
    history_size: int,
    rollout_horizon: int,
    view_stds: Sequence[float],
    seed: int,
    embedding_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    encoder_views = []
    predictor_views = []
    clean_outputs_for_actions = phase0.encode_sequences(model, _clone_batch(phase0, batch))
    act_emb = clean_outputs_for_actions["act_emb"].detach()
    for idx, std in enumerate(view_stds):
        if float(std) == 0.0:
            view_batch = _clone_batch(phase0, batch)
        else:
            view_batch = phase0.make_paired_noisy_batch(
                batch,
                history_size=history_size,
                noise_std=float(std),
                seed=seed + 131 * (idx + 1),
                corruption_type="gaussian_noise",
                corrupt_goal=False,
            )
        outputs = phase0.encode_sequences(model, _clone_batch(phase0, view_batch))
        emb = phase0.get_embedding_space(outputs, embedding_space).detach()
        encoder_views.append(emb[:, history_size - 1].detach().float().cpu().numpy())
        chain = phase0._autoregressive_rollout(
            model,
            emb[:, :history_size],
            act_emb,
            history_size,
            rollout_horizon,
        )
        final = chain[:, history_size + rollout_horizon - 1]
        predictor_views.append(final.detach().float().cpu().numpy())
    return np.stack(encoder_views, axis=0), np.stack(predictor_views, axis=0)


def render_3d_task(
    *,
    task: str,
    summary: Mapping[str, Any],
    acpc_basin_path: Path,
    out_dir: Path,
    n_sequences: int,
    view_stds: Sequence[float],
    rollout_horizon: int,
    seed: int,
    device: str | None,
    img_size: int,
    frameskip: int,
    anchor_count: int,
) -> Path:
    plt = _ensure_plot_deps()
    phase0 = _ensure_runtime_deps()
    device_value = device or "cpu"
    specs = _checkpoint_specs(task=task, summary=summary, acpc_basin_path=acpc_basin_path)

    encoded: dict[str, dict[str, np.ndarray]] = {}
    for spec in specs:
        with phase0.torch.no_grad():
            model = phase0.load_model(str(spec.model_file), device_value)
            history_size = phase0.infer_history_size(model)
            future_steps = max(rollout_horizon + 1, 9)
            batch = phase0.load_dataset_samples(
                dataset_name=TASK_DATASETS[task],
                state_key=None,
                n_sequences=n_sequences,
                history_size=history_size,
                future_steps=future_steps,
                frameskip=frameskip,
                img_size=img_size,
                seed=seed,
                device=device_value,
            )
            spaces = phase0.get_model_spaces(model)
            embedding_space = spaces["inference_cost_space"]
            enc, pred = _extract_view_features(
                phase0=phase0,
                model=model,
                batch=batch,
                history_size=history_size,
                rollout_horizon=rollout_horizon,
                view_stds=view_stds,
                seed=seed,
                embedding_space=embedding_space,
            )
            encoded[spec.label] = {"encoder": enc, "predictor": pred}

    enc_pca = _pca_fit_transform([encoded["base"]["encoder"], encoded["fullseq_robust"]["encoder"]])
    pred_pca = _pca_fit_transform([encoded["base"]["predictor"], encoded["fullseq_robust"]["predictor"]])
    encoded["base"]["encoder_3d"], encoded["fullseq_robust"]["encoder_3d"] = enc_pca
    encoded["base"]["predictor_3d"], encoded["fullseq_robust"]["predictor_3d"] = pred_pca
    nearest_indices = {
        "base": {
            "encoder_3d": _nearest_original_indices(encoded["base"]["encoder"], []),
            "predictor_3d": _nearest_original_indices(encoded["base"]["predictor"], []),
        },
        "fullseq_robust": {
            "encoder_3d": _nearest_original_indices(encoded["fullseq_robust"]["encoder"], []),
            "predictor_3d": _nearest_original_indices(encoded["fullseq_robust"]["predictor"], []),
        },
    }
    axis_limits = {
        "encoder_3d": _axis_limits([encoded["base"]["encoder_3d"], encoded["fullseq_robust"]["encoder_3d"]]),
        "predictor_3d": _axis_limits([encoded["base"]["predictor_3d"], encoded["fullseq_robust"]["predictor_3d"]]),
    }

    rng = np.random.default_rng(seed)
    anchor_count = min(anchor_count, n_sequences)
    anchors = np.linspace(0, n_sequences - 1, anchor_count, dtype=int)
    if anchor_count > 0:
        anchors = np.unique(anchors)
    for label in ("base", "fullseq_robust"):
        nearest_indices[label]["encoder_3d"] = _nearest_original_indices(
            encoded[label]["encoder"], anchors
        )
        nearest_indices[label]["predictor_3d"] = _nearest_original_indices(
            encoded[label]["predictor"], anchors
        )
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(anchors))))
    label_by_spec = {
        "base": f"base std=0.0",
        "fullseq_robust": f"full-seq robust std={specs[1].std_key}",
    }

    fig = plt.figure(figsize=(12, 9))
    panels = [
        ("base", "encoder_3d", "Encoder"),
        ("base", "predictor_3d", "Predictor H8"),
        ("fullseq_robust", "encoder_3d", "Encoder"),
        ("fullseq_robust", "predictor_3d", "Predictor H8"),
    ]
    for i, (label, feature, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        arr = encoded[label][feature]
        clean = arr[0]
        ax.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=8, c="#999999", alpha=0.22, depthshade=False)
        for ci, state_idx in enumerate(anchors):
            color = colors[ci % len(colors)]
            pts = arr[:, state_idx, :]
            nn_idx = nearest_indices[label][feature][int(state_idx)]
            nn_pt = clean[nn_idx]
            origin_pt = pts[0]
            ax.plot(
                [origin_pt[0], nn_pt[0]],
                [origin_pt[1], nn_pt[1]],
                [origin_pt[2], nn_pt[2]],
                color="#222222",
                alpha=0.35,
                linewidth=0.9,
                linestyle="--",
            )
            ax.scatter(
                nn_pt[0:1],
                nn_pt[1:2],
                nn_pt[2:3],
                s=34,
                color="#222222",
                marker="x",
                alpha=0.72,
                depthshade=False,
            )
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.7, linewidth=1.0)
            ax.scatter(pts[0:1, 0], pts[0:1, 1], pts[0:1, 2], s=42, color=[color], marker="o", depthshade=False)
            ax.scatter(pts[1:, 0], pts[1:, 1], pts[1:, 2], s=24, color=[color], marker="^", alpha=0.78, depthshade=False)
        ax.set_title(f"{label_by_spec[label]}: {title}", pad=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        xlim, ylim, zlim = axis_limits[feature]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=22, azim=38)
    fig.suptitle(
        f"{task}: original-state points and same-state perturbation clusters "
        f"(view stds={','.join(f'{s:g}' for s in view_stds)})",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Gray dots: original-view states. Colored circles: selected originals. "
        "Colored triangles/lines: same-state perturbed views. Black x/dashed line: nearest other original state.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{task.lower()}_fullseq_selective_contraction_3d.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Paper 1 selective-contraction branch diagnostics.")
    p.add_argument("--acpc-basin", type=Path, default=DEFAULT_DATA_DIR / "acpc_basin_diagnostics.json")
    p.add_argument("--acpc-phase0", type=Path, default=DEFAULT_DATA_DIR / "acpc_phase0_diagnostics.json")
    p.add_argument("--robust-metric", default="pixels_std0.08_success")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--plot-3d", action="store_true")
    p.add_argument("--plot-tasks", nargs="+", choices=TASKS, default=["PushT"])
    p.add_argument("--plot-out-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--n-sequences", type=int, default=48)
    p.add_argument("--view-stds", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.08])
    p.add_argument("--rollout-horizon", type=int, default=8)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default=None, help="Default: cpu, to avoid interfering with active training.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--anchor-count", type=int, default=8)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary = build_summary(
        acpc_basin_path=args.acpc_basin,
        acpc_phase0_path=args.acpc_phase0,
        robust_metric=args.robust_metric,
    )
    _write_json(args.out_json, summary)
    write_markdown(args.out_md, summary)
    print(f"[selective-contraction] wrote {args.out_json}")
    print(f"[selective-contraction] wrote {args.out_md}")

    if args.plot_3d:
        for task in args.plot_tasks:
            out = render_3d_task(
                task=task,
                summary=summary,
                acpc_basin_path=args.acpc_basin,
                out_dir=args.plot_out_dir,
                n_sequences=args.n_sequences,
                view_stds=args.view_stds,
                rollout_horizon=args.rollout_horizon,
                seed=args.seed,
                device=args.device,
                img_size=args.img_size,
                frameskip=args.frameskip,
                anchor_count=args.anchor_count,
            )
            print(f"[selective-contraction] wrote {out}")


if __name__ == "__main__":
    main()
