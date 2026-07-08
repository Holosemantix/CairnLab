#!/usr/bin/env python3
"""Apply the Paper 1 theory-strengthening editorial patch.

Run from the repository root:
    python paper1/docs/apply_theory_strengthening_patch.py

The script only edits paper1/main.tex and uses exact-string replacements. It is
idempotent for an already-patched manuscript and fails loudly on other drift.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper1" / "main.tex"


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    new_count = text.count(new)
    if new_count == 1:
        print(f"{label}: already applied")
        return text, False
    if new_count > 1:
        raise RuntimeError(f"{label}: patched text appears {new_count} times")

    old_count = text.count(old)
    if old_count != 1:
        raise RuntimeError(f"{label}: expected exactly one original match, found {old_count}")
    print(f"{label}: applied")
    return text.replace(old, new, 1), True


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    changed = False

    text, did_change = replace_once(
        text,
        """\\newtheorem{proposition}{Proposition}""",
        """\\newtheorem{proposition}{Proposition}
\\newtheorem{corollary}{Corollary}""",
        "corollary theorem environment",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """A natural remedy is input-side noise augmentation. We sweep \\stdmax{} $\\in \\{0.01,\\dots,0.08\\}$ across four tasks and find recovery, but the useful range varies: TwoRoom benefits from heavy noise, PushT's unperturbed and noisy-eval point-best checkpoints differ, Reacher has a broad plateau, and Cube responds weakly. A single Gaussian-noise strength therefore recovers performance unevenly across tasks. We use this sweep to motivate future predictive-consistency objectives, not to identify a universal training-noise level. Appendix~\\ref{sec:appendix-target-view} records a negative target-view ablation: perturbed-history $\\to$ original-future denoising is not a sufficient closed-loop fix, so the retained empirical branch is full-sequence perturbed-target training.""",
        """A natural remedy is input-side noise augmentation. We sweep \\stdmax{} $\\in \\{0.01,\\dots,0.08\\}$ across four tasks and find recovery, but the useful range varies: TwoRoom reaches its highest robust values at high noise, PushT's unperturbed and noisy-eval point-best checkpoints differ, Reacher has a broad plateau, and Cube responds weakly. A single Gaussian-noise strength therefore recovers performance unevenly across tasks. We use this sweep to motivate future predictive-consistency objectives, not to identify a universal training-noise level. Appendix~\\ref{sec:appendix-target-view} records a negative target-view ablation: perturbed-history $\\to$ original-future denoising is not a sufficient closed-loop fix, so the retained empirical branch is full-sequence perturbed-target training.""",
        "intro TwoRoom high-noise wording",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """\\paragraph{C1 --- Problem reframing.} We formulate a diagnostic lens for visual robustness in world-model control: action-conditioned predictive consistency plus an action-relevant discriminability countercondition, rather than encoder-level invariance.""",
        """\\paragraph{C1 --- Diagnostic principle and planner link.} We formulate visual robustness for world-model control as action-conditioned predictive consistency with an action-relevant discriminability countercondition. On a fixed candidate set, bounded ACPC implies bounded candidate-cost drift and preserves the selected candidate under a clean action-margin condition.""",
        "C1 contribution",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """\\subsection{A planner-stability view}\\label{sec:acpc-planner-stability}""",
        """\\subsection{ACPC as a fixed-candidate stability condition}\\label{sec:acpc-planner-stability}""",
        "theory subsection title",
    )
    changed = changed or did_change

    top1_proof = """\\begin{proof}
For any $j\\neq 1$,
\\[
C_{\\tilde h}(\\mathbf a^j,g)-C_{\\tilde h}(\\mathbf a^{(1)},g)
\\ge
C_h(\\mathbf a^j,g)-C_h(\\mathbf a^{(1)},g)-2\\eta
\\ge \\Delta-2\\eta >0 .
\\]
Thus every non-top candidate remains more costly than the clean top-1 candidate on the corrupted branch.
\\end{proof}"""

    top1_proof_with_corollary = """\\begin{proof}
For any candidate $\\mathbf a^j \\neq \\mathbf a^{(1)}$,
\\[
C_{\\tilde h}(\\mathbf a^j,g)-C_{\\tilde h}(\\mathbf a^{(1)},g)
\\ge
C_h(\\mathbf a^j,g)-C_h(\\mathbf a^{(1)},g)-2\\eta
\\ge \\Delta-2\\eta >0 .
\\]
Thus every non-top candidate remains more costly than the clean top-1 candidate on the corrupted branch.
\\end{proof}

\\begin{corollary}[ACPC-margin condition for candidate stability]\\label{cor:acpc-margin}
Let $\\mathcal A=\\{\\mathbf a^1,\\ldots,\\mathbf a^K\\}$ be a shared candidate set and assume the cost readout $J$ is $L_J$-Lipschitz as in Proposition~\\ref{prop:cost-drift}. If every candidate in $\\mathcal A$ has rollout-readout discrepancy at most $\\epsilon$ between the clean and corrupted branches, and the clean branch has top-1/top-2 margin $\\Delta > 2L_J\\epsilon$, then the two branches select the same top-1 candidate from $\\mathcal A$.
\\end{corollary}

\\begin{proof}
Proposition~\\ref{prop:cost-drift} gives $|C_h(\\mathbf a^j,g)-C_{\\tilde h}(\\mathbf a^j,g)|\\le L_J\\epsilon$ for every candidate. Applying Proposition~\\ref{prop:top1-stability} with $\\eta=L_J\\epsilon$ gives the result.
\\end{proof}"""

    text, did_change = replace_once(
        text,
        top1_proof,
        top1_proof_with_corollary,
        "top-1 proof insertion point",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """\\noindent\\textbf{Notes.} (i) All representative diagnostics are taken from the corresponding checkpoint's per-tool JSON outputs under the diagnostics directory (max-std $= 0.1$, history-only noise); (ii) the near-collapse of multi-step rollout drift at the heavier-noise representatives (TwoRoom $18.62 \\to 0.66$ at \\stdmax{} $= 0.08$; Reacher $15.17 \\to 0.44$ at \\stdmax{} $= 0.06$) is not a recording error: sufficient training noise collapses the multi-step predictor drift while leaving the single-step task-resolution metrics comparatively flat --- precisely the dissociation that motivates the analysis in Section~\\ref{sec:diag-cross}. The low-noise representatives reduce drift far less (PushT $18.65 \\to 6.00$ at \\stdmax{} $= 0.02$; Cube $20.20 \\to 19.25$ at \\stdmax{} $= 0.01$).""",
        """\\noindent\\textbf{Notes.} (i) All representative diagnostics are taken from the corresponding checkpoint's per-tool JSON outputs under the diagnostics directory (max-std $= 0.1$, history-only noise); (ii) the near-collapse of multi-step rollout drift at the heavier-noise representatives (TwoRoom $18.62 \\to 0.66$ at \\stdmax{} $= 0.08$; Reacher $15.17 \\to 0.44$ at \\stdmax{} $= 0.06$) is not a recording error: these checkpoints sharply reduce multi-step predictor drift while leaving the single-step task-resolution metrics comparatively flat --- precisely the dissociation that motivates the analysis in Section~\\ref{sec:diag-cross}. The low-noise representatives reduce drift far less (PushT $18.65 \\to 6.00$ at \\stdmax{} $= 0.02$; Cube $20.20 \\to 19.25$ at \\stdmax{} $= 0.01$).""",
        "diagnostic-note causal wording",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """These propositions explain the roles of the downstream diagnostics. ACPC controls candidate-cost drift; if the drift is small relative to the clean action margin, the selected candidate is stable. Candidate-ranking agreement and margin-conditioned action flips are therefore readouts of the same chain. The chain can fail in two simple ways. Encoder closeness is neither necessary nor sufficient: an invertible nuisance-subspace change can move $E_\\theta(h)$ and $E_\\theta(\\tilde h)$ far apart while leaving $\\Pi(F_\\theta^{1:H}(\\cdot,\\mathbf a))$ and the cost unchanged, whereas a small encoder displacement can still flip a low-margin cost ranking if the predictor or cost readout has high local sensitivity. ACPC without discriminability also admits collapse: a constant encoder and predictor give zero same-state ACPC for every perturbation pair, but merge states requiring different actions, costs, or transitions. Low ACPC is meaningful only with a guard on action-relevant separation.""",
        """The corollary is the formal reason that ACPC is a planning diagnostic rather than only a representation-distance measurement. It links paired predictive agreement to candidate-cost stability and then to action selection under a margin condition. Candidate-ranking agreement and margin-conditioned action flips are downstream readouts of this chain. The result is deliberately limited to a fixed candidate set; CEM resampling, repeated replanning, and environment feedback can still change the closed-loop trajectory.

Encoder closeness is not the right substitute for this condition. It is not necessary: a nuisance-subspace change can move $E_\\theta(h)$ and $E_\\theta(\\tilde h)$ far apart while leaving $\\Pi(F_\\theta^{1:H}(\\cdot,\\mathbf a))$ and the cost unchanged. It is not sufficient either: a small encoder displacement can flip a low-margin cost ranking when the predictor or cost readout has high local sensitivity.

\\begin{proposition}[ACPC alone permits collapse]\\label{prop:acpc-collapse}
There exist encoder--predictor pairs for which $\\mathrm{ACPC}_H=0$ for every same-state perturbation pair and every action sequence, while action-distinct states are mapped to identical rollout readouts.
\\end{proposition}

\\begin{proof}
Let $E_\\theta(h)=z_0$ for every history and let $F_\\theta^k(z_0,\\mathbf a)=u_0$ for every horizon $k$ and action sequence $\\mathbf a$. Then clean and corrupted branches always have identical rollout readouts, so same-state ACPC is zero. However, any two states that require different actions, costs, or transitions also have identical readouts, violating the discriminability condition in \\Cref{eq:acpc-disc}.
\\end{proof}

Thus low ACPC is meaningful only with a guard on action-relevant separation. This is the role of the transition-resolution and inverse-dynamics probes in the main diagnostics, and it is the failure exposed by the heteroscedastic-loss ablation in Appendix~\\ref{sec:appendix-D}.""",
        "post-proposition explanation and collapse proposition",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """\\item \\textbf{TwoRoom.} Low-dimensional, discrete, visually redundant. Compressing the representation (effective rank $47.6 \\to 37.7$) is acceptable and even beneficial: a more compact latent space is easier to plan in.""",
        """\\item \\textbf{TwoRoom.} Low-dimensional, discrete, visually redundant. The observed compression (effective rank $47.6 \\to 37.7$) is compatible with high success in this task; the data do not show that compactness itself causes the improvement.""",
        "TwoRoom mechanism wording",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """\\paragraph{Scope of this arXiv version.}
This version is a controlled diagnostic study.""",
        """\\paragraph{Scope.}
This paper is a controlled diagnostic study.""",
        "Scope paragraph wording",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """The data challenge a too-strong reading of latent prediction: predicting future latents does not by itself guarantee robustness to visual noise. The corrective is also not encoder invariance. The useful target is agreement of action-conditioned predictions in task-relevant coordinates while retaining distinctions that affect actions, costs, or transitions. This explains the task variation: TwoRoom can tolerate substantial nuisance contraction, PushT contact control needs fine pose resolution, Reacher sits in a lower-dimensional continuous regime, and Cube is less sensitive to global Gaussian noise in this sweep.""",
        """The data challenge a too-strong reading of latent prediction: predicting future latents does not by itself guarantee robustness to visual noise. The corrective is also not encoder invariance. The useful target is agreement of action-conditioned predictions in task-relevant coordinates while retaining distinctions that affect actions, costs, or transitions. This reading is compatible with the task variation: TwoRoom can tolerate substantial nuisance contraction, PushT contact control needs fine pose resolution, Reacher sits in a lower-dimensional continuous regime, and Cube is less sensitive to global Gaussian noise in this sweep.""",
        "discussion interpretation wording",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """TwoRoom hetero reaches $99.67\\%$ on unperturbed evaluation --- consistent with the prior that low-dimensional discrete tasks benefit from stronger nuisance contraction / clustering --- but lags noise training on high-noise robustness. \\textbf{PushT hetero unperturbed success is $13.33\\%$ --- a method-level failure}, not evidence for a useful robustness compromise.""",
        """TwoRoom hetero reaches $99.67\\%$ on unperturbed evaluation --- compatible with the prior that low-dimensional discrete tasks can tolerate stronger nuisance contraction / clustering --- but lags noise training on high-noise robustness. \\textbf{PushT hetero unperturbed success is $13.33\\%$ --- a method-level failure}, not evidence for a useful robustness compromise.""",
        "hetero TwoRoom prior wording",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """Hetero-loss compresses the representation in both tasks. For TwoRoom (low-dimensional, discrete, redundant), compression is acceptable. For PushT, the L2 \\transres{} collapses from $0.3015$ to $0.1023$ and \\idprobe{} from $0.7739$ to $0.2678$ --- task-relevant state information is erased. The contrast with input-side noise training is sharp: the PushT noise-sweep representative in \\Cref{tab:diag-base-vs-best} leaves rank and probe nearly unchanged, so this collapse is specific to error-based downweighting. The drop in multi-step rollout drift is not good news either: the latent has become more \\emph{predictable} without being more \\emph{controllable}.""",
        """Hetero-loss compresses the representation in both tasks. In TwoRoom (low-dimensional, discrete, redundant), the observed compression coexists with high unperturbed success, but the robustness shortfall shows that compression alone is not the target. For PushT, the L2 \\transres{} collapses from $0.3015$ to $0.1023$ and \\idprobe{} from $0.7739$ to $0.2678$ --- task-relevant state information is erased. The contrast with input-side noise training is sharp: the PushT noise-sweep representative in \\Cref{tab:diag-base-vs-best} leaves rank and probe nearly unchanged, so this collapse is specific to error-based downweighting. The drop in multi-step rollout drift is not good news either: the latent has become more \\emph{predictable} without being more \\emph{controllable}.""",
        "hetero compression wording",
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        """  \\item \\textbf{TwoRoom (visually redundant) benefits from heavy noise on both methods.} PLDM observation-noise 0.08 success rises from $67.00\\%$ at the no-noise baseline to $98.00\\%$ at \\stdmax{}~$= 0.06$, then remains on a high-noise plateau. This mirrors the LeWM reading that TwoRoom tolerates strong same-state nuisance contraction.""",
        """  \\item \\textbf{TwoRoom (visually redundant) shows high-noise recovery on both methods.} PLDM observation-noise 0.08 success rises from $67.00\\%$ at the no-noise baseline to $98.00\\%$ at \\stdmax{}~$= 0.06$, then remains on a high-noise plateau. This mirrors the LeWM reading that TwoRoom tolerates strong same-state nuisance contraction.""",
        "PLDM TwoRoom high-noise wording",
    )
    changed = changed or did_change

    if changed:
        TEX.write_text(text, encoding="utf-8")
        print(f"Updated {TEX.relative_to(ROOT)}")
    else:
        print(f"No changes needed for {TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
