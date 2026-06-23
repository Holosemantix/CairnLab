#!/usr/bin/env python3
"""Apply the Paper 1 theory-strengthening editorial patch.

Run from the repository root:
    python paper1/apply_theory_strengthening_patch.py

The script only edits paper1/main.tex and uses exact-string replacements so that
it fails loudly if the manuscript has drifted.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper1" / "main.tex"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TEX.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """\\paragraph{C1 --- Problem reframing.} We formulate a diagnostic lens for visual robustness in world-model control: action-conditioned predictive consistency plus an action-relevant discriminability countercondition, rather than encoder-level invariance.""",
        """\\paragraph{C1 --- Diagnostic principle and planner link.} We formulate visual robustness for world-model control as action-conditioned predictive consistency with an action-relevant discriminability countercondition. On a fixed candidate set, bounded ACPC implies bounded candidate-cost drift and preserves the selected candidate under a clean action-margin condition.""",
        "C1 contribution",
    )

    text = replace_once(
        text,
        """\\subsection{A planner-stability view}\\label{sec:acpc-planner-stability}""",
        """\\subsection{ACPC as a fixed-candidate stability condition}\\label{sec:acpc-planner-stability}""",
        "theory subsection title",
    )

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

    top1_proof_with_corollary = top1_proof + """

\\begin{corollary}[ACPC-margin condition for candidate stability]\\label{cor:acpc-margin}
Let $\\mathcal A$ be a shared candidate set and assume the cost readout $J$ is $L_J$-Lipschitz as in Proposition~\\ref{prop:cost-drift}. If every candidate in $\\mathcal A$ has rollout-readout discrepancy at most $\\epsilon$ between the clean and corrupted branches, and the clean branch has top-1/top-2 margin $\\Delta > 2L_J\\epsilon$, then the two branches select the same top-1 candidate from $\\mathcal A$.
\\end{corollary}

\\begin{proof}
Proposition~\\ref{prop:cost-drift} gives $|C_h(\\mathbf a^j,g)-C_{\\tilde h}(\\mathbf a^j,g)|\\le L_J\\epsilon$ for every candidate. Applying Proposition~\\ref{prop:top1-stability} with $\\eta=L_J\\epsilon$ gives the result.
\\end{proof}"""

    text = replace_once(
        text,
        top1_proof,
        top1_proof_with_corollary,
        "top-1 proof insertion point",
    )

    text = replace_once(
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

    text = replace_once(
        text,
        """\\item \\textbf{TwoRoom.} Low-dimensional, discrete, visually redundant. Compressing the representation (effective rank $47.6 \\to 37.7$) is acceptable and even beneficial: a more compact latent space is easier to plan in.""",
        """\\item \\textbf{TwoRoom.} Low-dimensional, discrete, visually redundant. The observed compression (effective rank $47.6 \\to 37.7$) is compatible with high success in this task; the data do not show that compactness itself causes the improvement.""",
        "TwoRoom mechanism wording",
    )

    text = replace_once(
        text,
        """\\paragraph{Scope of this arXiv version.}
This version is a controlled diagnostic study.""",
        """\\paragraph{Scope.}
This paper is a controlled diagnostic study.""",
        "Scope paragraph wording",
    )

    TEX.write_text(text, encoding="utf-8")
    print(f"Updated {TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
