#!/usr/bin/env python3
"""Patch Paper 1 wording around sweep plateaus and highest observed grid points.

Run from the repository root:
    python paper1/apply_plateau_uncertainty_patch.py

The patch avoids treating single-run/single-grid maximum means as statistically
unique optima. It also patches the Figure 2 renderer label/title so regenerated
fig2_sweep.png no longer displays "Robust-eval optimum" or sigma-star language.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper1" / "main.tex"
FIGS = ROOT / "tools" / "paper1_figs.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_all(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        raise RuntimeError(f"{label}: expected at least one match, found 0")
    return text.replace(old, new)


def patch_main_tex() -> None:
    text = TEX.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "Full-sequence input-side noise augmentation recovers much of this loss, but the best recovery point and plateau structure vary by task.",
        "Full-sequence input-side noise augmentation recovers much of this loss, but the high-performing range and plateau structure vary by task.",
        "abstract plateau wording",
    )

    text = replace_once(
        text,
        "\\caption{LeWM noise-sweep summary. Values are success rate mean $\\pm$ population std across evaluation seeds; parenthesised values are the train-time \\stdmax{} of the point-best checkpoint.}",
        "\\caption{LeWM noise-sweep summary. Values are success rate mean $\\pm$ population std across evaluation seeds. Parenthesised \\stdmax{} values mark the highest observed mean in this grid and should be read as plateau reference points, not statistically unique optima.}",
        "LeWM sweep caption",
    )

    text = replace_once(
        text,
        "Task & unpert. base & obs 0.08 base & unpert. best & obs 0.08 best & Main reading \\\",
        "Task & unpert. base & obs 0.08 base & unpert. high & obs 0.08 high & Main reading \\\",
        "LeWM sweep headers",
    )

    text = replace_once(
        text,
        "green dashed: each task's corrupted-eval point-best \\stdmax{}.",
        "green dashed: the highest observed corrupted-eval mean in the grid, shown as a reference marker rather than a unique optimum.",
        "Figure 2 caption marker wording",
    )

    text = replace_once(
        text,
        "PushT's unperturbed and noisy-eval point-bests differ ($0.03$ vs $0.08$), Reacher has a broad plateau, and Cube responds weakly. Several point-bests sit within plateau regions rather than isolated peaks; the robust pattern is that one scalar perturbation strength expresses the selective-consistency tradeoff only coarsely.",
        "PushT's highest observed unperturbed and noisy-eval means occur at different grid points ($0.03$ vs $0.08$), but neighbouring high-noise settings are close under the reported evaluation variability. Reacher has a broad plateau, and Cube responds weakly. The robust pattern is therefore not a unique \\stdmax{} optimum; it is that one scalar perturbation strength expresses the selective-consistency tradeoff only coarsely.",
        "LeWM sweep reading",
    )

    text = replace_once(
        text,
        "This comparison is diagnostic, not predictive: the robust endpoint is selected by direct closed-loop evaluation, and the basin radius is used afterward to localise what changed.",
        "This comparison is diagnostic, not predictive: the reported high-corruption checkpoint is chosen from direct closed-loop evaluation, and the basin radius is used afterward to localise what changed.",
        "ACPC basin selection wording",
    )

    text = replace_once(
        text,
        "\\caption{LeWM Gaussian-noise ACPC-basin proxy: no-noise baseline vs.\\ each task's observation-noise 0.08 point-best checkpoint. $R_E$ is normalised encoder-view spread and $R_F$ is normalised action-conditioned rollout-view spread under the same action sequence.}",
        "\\caption{LeWM Gaussian-noise ACPC-basin proxy: no-noise baseline vs.\\ each task's highest-observed observation-noise 0.08 grid point. $R_E$ is normalised encoder-view spread and $R_F$ is normalised action-conditioned rollout-view spread under the same action sequence; the selected grid point is a representative high-corruption checkpoint, not a statistically unique optimum.}",
        "LeWM ACPC basin caption",
    )

    text = replace_once(
        text,
        "The direct-evaluation-selected endpoints have much smaller same-action prediction radii under same-family visual perturbations:",
        "The reported high-corruption checkpoints have much smaller same-action prediction radii under same-family visual perturbations:",
        "LeWM ACPC basin reading start",
    )

    text = replace_once(
        text,
        "The compact base-vs-selected endpoint view localises what changed at already selected checkpoints, while the full grid in Appendix~\\ref{sec:appendix-acpc-basin-grid} shows that the basin radius is not a standalone success predictor.",
        "The compact base-vs-representative view localises what changed at high-corruption grid points, while the full grid in Appendix~\\ref{sec:appendix-acpc-basin-grid} shows that the basin radius is not a standalone success predictor.",
        "Discussion ACPC compact view wording",
    )

    text = replace_once(
        text,
        "\\caption{Heteroscedastic-loss evaluation (canonical 3-seed $\\times$ 100 protocol). The LeWM+noise rows are each task's observation+goal 0.08 point-best checkpoint (TwoRoom \\stdmax{} $= 0.08$, PushT \\stdmax{} $= 0.06$), matching the strongest stress column of this ablation; the observation-only point-bests in \\Cref{tab:sweep-px08} differ.}",
        "\\caption{Heteroscedastic-loss evaluation (canonical 3-seed $\\times$ 100 protocol). The LeWM+noise rows use each task's highest-observed observation+goal 0.08 grid point (TwoRoom \\stdmax{} $= 0.08$, PushT \\stdmax{} $= 0.06$), matching the strongest stress column of this ablation; the observation-only high-observed grid points in \\Cref{tab:sweep-px08} differ.}",
        "hetero caption point-best wording",
    )

    text = replace_once(
        text,
        "TwoRoom LeWM+noise point-best",
        "TwoRoom LeWM+noise high-observed",
        "hetero TwoRoom row label",
    )
    text = replace_once(
        text,
        "PushT LeWM+noise point-best",
        "PushT LeWM+noise high-observed",
        "hetero PushT row label",
    )

    text = replace_once(
        text,
        "\\caption{PLDM+noise sweep, unperturbed success rate (\\%). $\\dagger$ marks the per-task unperturbed point-best.}",
        "\\caption{PLDM+noise sweep, unperturbed success rate (\\%). $\\dagger$ marks the highest observed mean in this grid, not a statistically unique optimum.}",
        "PLDM unperturbed caption",
    )

    text = replace_once(
        text,
        "\\caption{PLDM+noise sweep, observation-noise 0.08 success rate with unperturbed goal (\\%). $\\ddagger$ marks the per-task observation-noise point-best.}",
        "\\caption{PLDM+noise sweep, observation-noise 0.08 success rate with unperturbed goal (\\%). $\\ddagger$ marks the highest observed mean in this grid, not a statistically unique optimum.}",
        "PLDM obs caption",
    )

    text = replace_once(
        text,
        "The unperturbed and observation-noise 0.08 point-bests are co-located at \\stdmax{}~$= 0.03$ on PLDM.",
        "The highest observed unperturbed and observation-noise 0.08 means are co-located at \\stdmax{}~$= 0.03$ on PLDM, but this should be read as a grid-level marker rather than a unique optimum.",
        "PLDM PushT point-best wording",
    )

    text = replace_once(
        text,
        "The no-noise PLDM drop is only $2.33$\\,pt, and the observation-noise point-best sits at \\stdmax{}~$= 0.03$ ($81.67\\%$).",
        "The no-noise PLDM drop is only $2.33$\\,pt, and the highest observed observation-noise mean occurs at \\stdmax{}~$= 0.03$ ($81.67\\%$).",
        "PLDM Reacher point-best wording",
    )

    text = replace_once(
        text,
        "PLDM's PushT unperturbed point-best ($76.67\\%$ at \\stdmax{}~$= 0.03$) is lower than LeWM's ($89.67\\%$ at \\stdmax{}~$= 0.03$),",
        "PLDM's highest observed PushT unperturbed mean ($76.67\\%$ at \\stdmax{}~$= 0.03$) is lower than LeWM's corresponding high-observed mean ($89.67\\%$ at \\stdmax{}~$= 0.03$),",
        "PLDM method-specific point-best wording",
    )

    text = replace_once(
        text,
        "while the exact unperturbed/robust point-best location is method-dependent. In particular, the LeWM PushT point-best split should be read as a LeWM-family signature, not a cross-method law.",
        "while the exact high-observed grid location is method-dependent. In particular, the LeWM PushT split between unperturbed and noisy high-observed means should be read as a LeWM-family signature, not a cross-method law.",
        "PLDM method-specific location wording",
    )

    text = replace_once(
        text,
        "\\paragraph{PLDM full-sweep ACPC basin replication.} We also run the Gaussian-noise basin protocol on all 36 PLDM checkpoints. \\Cref{tab:pldm-acpc-basin} reports the compact baseline-vs-observation-noise-0.08 point-best summary; the full 4 tasks $\\times$ 9 configurations grid is listed in Appendix~\\ref{sec:appendix-artifact-map}.",
        "\\paragraph{PLDM full-sweep ACPC basin replication.} We also run the Gaussian-noise basin protocol on all 36 PLDM checkpoints. \\Cref{tab:pldm-acpc-basin} reports the compact baseline-vs-high-observed observation-noise summary; the full 4 tasks $\\times$ 9 configurations grid is listed in Appendix~\\ref{sec:appendix-artifact-map}.",
        "PLDM ACPC paragraph point-best wording",
    )

    text = replace_once(
        text,
        "\\caption{PLDM full-sweep Gaussian-noise ACPC basin replication, summarised by the no-noise baseline and each task's PLDM observation-noise 0.08 point-best checkpoint. The protocol matches \\Cref{tab:acpc-basin}: same-state Gaussian-noise views of the observation history (std 0.01--0.08), unperturbed goal, and the same recorded action sequence.}",
        "\\caption{PLDM full-sweep Gaussian-noise ACPC basin replication, summarised by the no-noise baseline and each task's highest-observed PLDM observation-noise 0.08 grid point. The protocol matches \\Cref{tab:acpc-basin}: same-state Gaussian-noise views of the observation history (std 0.01--0.08), unperturbed goal, and the same recorded action sequence.}",
        "PLDM ACPC caption",
    )

    text = replace_once(
        text,
        "The PLDM basin replication is directionally consistent with the LeWM ACPC-basin reading: the observation-noise 0.08 point-best checkpoint reduces $R_F$ on all four tasks, sharply on TwoRoom, PushT, and Cube.",
        "The PLDM basin replication is directionally consistent with the LeWM ACPC-basin reading: the reported high-observed observation-noise 0.08 checkpoint reduces $R_F$ on all four tasks, sharply on TwoRoom, PushT, and Cube.",
        "PLDM ACPC reading",
    )

    text = replace_once(
        text,
        "\\caption{PLDM full-diagnostic check, no-noise baseline $\\to$ representative noise-trained checkpoint. Representatives are each task's PLDM observation-noise 0.08 point-best checkpoint (TwoRoom \\stdmax{}~$= 0.06$, PushT \\stdmax{}~$= 0.03$, Reacher \\stdmax{}~$= 0.03$, Cube \\stdmax{}~$= 0.04$).}",
        "\\caption{PLDM full-diagnostic check, no-noise baseline $\\to$ representative noise-trained checkpoint. Representatives are the highest-observed PLDM observation-noise 0.08 grid points (TwoRoom \\stdmax{}~$= 0.06$, PushT \\stdmax{}~$= 0.03$, Reacher \\stdmax{}~$= 0.03$, Cube \\stdmax{}~$= 0.04$), not statistically unique optima.}",
        "PLDM full diag caption",
    )

    text = replace_once(
        text,
        "\\caption{Phase-0 paired ACPC diagnostics, no-noise baseline $\\to$ observation+goal 0.08 point-best checkpoint. Lower is better for ACPC-$H$/transition, PCC, and MAF; higher is better for CRA and top-8 overlap. Values are exploratory diagnostics, not predictor-selection rules.}",
        "\\caption{Phase-0 paired ACPC diagnostics, no-noise baseline $\\to$ representative high observation+goal 0.08 checkpoint. Lower is better for ACPC-$H$/transition, PCC, and MAF; higher is better for CRA and top-8 overlap. Values are exploratory diagnostics, not predictor-selection rules.}",
        "Phase-0 caption",
    )

    text = replace_once(
        text,
        "and the observation+goal point-best checkpoints reduce these quantities sharply.",
        "and the representative high observation+goal checkpoints reduce these quantities sharply.",
        "Phase-0 reading",
    )

    TEX.write_text(text, encoding="utf-8")


def patch_fig_script() -> None:
    text = FIGS.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'label=r"Robust-eval optimum $\\sigma^\\ast$"',
        'label=r"Highest observed robust eval (not unique)"',
        "fig2 legend label",
    )
    text = replace_once(
        text,
        'ax.set_title(rf"{t}   ($\\sigma^\\ast={best_std:.2f}$)", fontsize=11)',
        'ax.set_title(rf"{t}   (ref. $\\sigma={best_std:.2f}$)", fontsize=11)',
        "fig2 panel title",
    )
    FIGS.write_text(text, encoding="utf-8")


def main() -> None:
    patch_main_tex()
    patch_fig_script()
    print("Updated plateau/uncertainty language in paper1/main.tex and tools/paper1_figs.py")


if __name__ == "__main__":
    main()
