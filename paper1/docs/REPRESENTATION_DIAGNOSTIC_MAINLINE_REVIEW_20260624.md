# Paper 1 representation-diagnostic mainline review — 2026-06-24

This review answers the central positioning question: the paper should not read as “visual perturbations hurt; noise training helps.” The core contribution should be **representation analysis and diagnostic findings**, with Gaussian perturbations and noise training used as the controlled stressor. The theory section should support this diagnostic logic, not dominate it as a standalone theory contribution.

## High-level verdict

The current paper is close, but not fully aligned with the strongest story. It still has places where the narrative reads like a noise-augmentation sweep: Figure 2 marker language, compact rows called point-bests/endpoints, and an abstract that mentions release artifacts more than the multi-stage diagnostic account. The theory is useful and mathematically safe under its stated assumptions, but it should be presented as a bridge inside a representation-diagnostic pipeline.

The strongest positioning is:

> This paper is a representation-diagnostic study of JEPA latent world-model control. Gaussian visual perturbations are the stressor. Noise training is the intervention. The contribution is to trace how perturbations move through encoder geometry, action-conditioned rollout geometry, candidate costs/rankings, and discriminability. The theory supplies a fixed-candidate explanation of why rollout consistency matters for planning.

Do not frame the contribution as: “we found the best noise level,” “we propose a new robust method,” or “ACPC alone predicts robustness.”

---

## Reviewer 1 — Theory formalist

### Assessment

The theory section is correct in scope: fixed-candidate-set sufficient conditions, not full CEM or closed-loop guarantees. The cost-drift proposition, top-1 margin proposition, ACPC-margin corollary, and collapse proposition are valid under their assumptions.

### Remaining issue

The current theory begins at rollout discrepancy. That is safe, but it leaves the encoder role informal. Since the paper’s true contribution is representation diagnosis, add one small proposition that connects encoder geometry to rollout disagreement under local predictor sensitivity.

### Required addition

Insert before `Proposition~\ref{prop:cost-drift}`:

```tex
\begin{proposition}[Encoder shift as one route to rollout disagreement]\label{prop:encoder-route}
Fix an action sequence $\mathbf a$ and write
$G_{\mathbf a}(z)=\Pi(F_\theta^{1:H}(z,\mathbf a))$ for the rollout-readout map. If $G_{\mathbf a}$ is locally $L_G$-Lipschitz between $E_\theta(h)$ and $E_\theta(\tilde h)$ under metrics $d_Z$ and $d_H$, then
\[
 d_H\!\left(G_{\mathbf a}(E_\theta(h)),G_{\mathbf a}(E_\theta(\tilde h))\right)
 \le L_G\, d_Z\!\left(E_\theta(h),E_\theta(\tilde h)\right).
\]
\end{proposition}

\begin{proof}
This is the Lipschitz condition applied to the two encoded histories.
\end{proof}
```

Then add:

```tex
This bound makes encoder geometry a first-stage risk signal. A small encoder shift is sufficient for small rollout disagreement only when the action-conditioned predictor is locally insensitive in that direction. It is not necessary, because a large encoder shift can lie in a nuisance direction contracted by $G_{\mathbf a}$. It is not sufficient without the local-sensitivity condition, because a small encoder shift can be amplified by the predictor or cost readout. This is why the empirical analysis reports both encoder radii $R_E$ and rollout radii $R_F$.
```

### Why this is safe

This is a direct local Lipschitz bound. It introduces no hallucinated guarantee and does not claim the learned networks are globally Lipschitz. It connects encoder diagnostics to the existing ACPC theory.

---

## Reviewer 2 — Representation/diagnostics reviewer

### Assessment

The paper’s real contribution is the diagnostic decomposition, not the noise sweep. Make this explicit throughout the abstract, contribution list, and results section.

### Required abstract repositioning

The current abstract ending is too weak because it says “diagnostic framing, release artifacts, and negative ablations.” Replace the ending with a multi-stage account:

```tex
We use the corruption sweep as a stressor for representation analysis rather than as a hyperparameter search. Paired ACPC-basin probes localise the recovery in rollout space (PushT $R_F$: $1.543\to0.088$), while representation diagnostics check whether encoder contraction preserves action-relevant information. The result is a multi-stage diagnostic account: encoder geometry remains informative, but robustness for control is decided by whether perturbation-induced latent changes survive action-conditioned rollout into candidate costs while action-relevant distinctions remain separable.
```

### Required contribution rewrite

Replace C1 with:

```tex
\paragraph{C1 --- Multi-stage diagnostic principle.}
We formulate visual robustness for JEPA world-model control as a multi-stage diagnostic rather than encoder invariance alone. Encoder perturbation geometry measures whether visual noise enters and reshapes the latent state; action-conditioned predictive consistency measures whether that shift changes predicted futures and candidate costs; the discriminability countercondition prevents contraction from erasing action-relevant distinctions.
```

Then keep the planner link either in C1 as a second sentence or in C2:

```tex
Under a Lipschitz cost readout, bounded rollout disagreement bounds candidate-cost drift on a fixed candidate set, and a sufficiently large clean action margin preserves the selected candidate within that set.
```

### Required results framing

At the beginning of `Experiments`, insert or replace with:

```tex
The experiments use Gaussian corruption as a controlled stressor for a representation-diagnostic question. We first show that the stressor changes closed-loop behaviour. We then ask where the recovery appears in the model: encoder geometry ($R_E$, rank, transition resolution, ID probe), action-conditioned rollout geometry ($R_F$, rollout drift), and downstream candidate/cost readouts. The goal is not to tune a unique noise level, but to trace how visual perturbations move through the encoder--predictor--planner chain.
```

---

## Reviewer 3 — Statistical reviewer

### Assessment

The strongest remaining statistical flaw is single-point language. Any marker or term suggesting a robust-eval optimum will invite objections. Figure 2 must not mark a single sigma, even as “highest observed.”

### Required Figure 2 changes

In `tools/paper1_figs.py`, remove from `fig2_sweep`:

- `best_std = ...`
- `ax.axvline(...)`
- marker legend label
- title text containing `sigma*`, `ref. sigma`, or any single selected sigma

Use:

```python
ax.set_title(t, fontsize=11)
```

Legend should have only two entries and `ncol=2`.

### Required Figure 2 caption

```tex
\caption{Noise-training sweep across four tasks. Blue: unperturbed evaluation; red: observation-only Gaussian noise $\sigma=0.08$ with an unperturbed goal. Error bars show the population standard deviation across the three evaluation seeds. The curves show task-dependent high-performing regions rather than a statistically unique training-noise optimum.}
```

### Terms to remove from main text and Figure 2 generator

Run:

```bash
rg -n "robust-eval optimum|highest observed robust|sigma\\*|sigma\\^\\*|green dashed|unique optimum|optimal sigma|point-best|best checkpoint|selected endpoint|noise-best" paper1/main.tex tools/paper1_figs.py tools/README_paper1.md
```

Main text and Figure 2 generator should have no hits. Appendix table captions may keep “grid reference row” or “compact row,” but not “point-best” unless explicitly defined as a descriptive grid maximum.

---

## Reviewer 4 — RL/world-model reviewer

### Assessment

The story becomes meaningful when framed as a pipeline diagnosis:

```text
pixels -> encoder geometry -> action-conditioned rollout geometry -> candidate costs/rankings -> closed-loop success
```

Do not say or imply that encoder invariance is irrelevant. It is a first-stage diagnostic. The paper’s point is that encoder invariance alone cannot define robustness for control.

### Required wording near ACPC diagnostics

After the paragraph defining $R_E$ and $R_F$, add:

```tex
Thus $R_E$ and $R_F$ answer different questions. $R_E$ measures whether the visual perturbation changes the encoded neighbourhood; $R_F$ measures whether that change survives action-conditioned rollout in coordinates used for planning. Robust checkpoints in this Gaussian-noise study often reduce both, so the claim is not predictor-only contraction. The claim is that encoder contraction must be interpreted through its downstream predictive effect and checked against discriminability.
```

This directly addresses the concern that the paper ignores encoder representations.

---

## Reviewer 5 — Writing and contribution reviewer

### Assessment

The paper risks underselling itself by repeating what it is not: not a benchmark, not an oracle, not a method, not training-seed study. Keep these limitations, but concentrate them in the Scope paragraph and write the main story positively.

### Desired positive story

Use this as the manuscript’s guiding sentence:

```text
This paper is not an augmentation sweep. It is a multi-stage diagnostic study of visual robustness in JEPA world-model control: encoder geometry shows how visual perturbations enter the latent state, action-conditioned rollout consistency shows whether those perturbations affect predicted futures and candidate costs, and discriminability checks whether robustness is bought by erasing action-relevant distinctions.
```

### Prose edits

- Replace “The corrective is also not encoder invariance” with “Encoder invariance is useful but incomplete.”
- Replace “local mechanism read” with “local diagnostic readout.”
- Avoid “post-hoc” more than once in the main text.
- Avoid “best,” “selected endpoint,” and “optimum.”

---

## Final Codex execution checklist

1. Remove Figure 2 marker from `tools/paper1_figs.py`.
2. Regenerate `assets/paper1_figs/fig2_sweep.png`.
3. Reposition abstract around multi-stage representation diagnosis.
4. Rewrite C1 contribution as multi-stage diagnostic principle.
5. Add encoder-to-rollout Lipschitz proposition.
6. Add `R_E`/`R_F` interpretation paragraph.
7. Replace single-point sweep/ACPC/PLDM wording with plateau/range/reference-row wording.
8. Run:

```bash
python -m tools.paper1_figs --out-dir assets/paper1_figs
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

9. Run final grep:

```bash
rg -n "robust-eval optimum|highest observed robust|sigma\\*|sigma\\^\\*|green dashed|unique optimum|optimal sigma|point-best|best checkpoint|selected endpoint|noise-best" paper1/main.tex tools/paper1_figs.py tools/README_paper1.md
```

10. Replace `\arxivauthors` with the real author list and generate `main.bbl` before arXiv upload.

---

## Bottom line

The core logic is valid only if the paper is framed as representation analysis and diagnostic discovery. Visual perturbations and noise training are the controlled stressor/intervention. Theory is a compact support layer explaining why rollout-space diagnostics matter for planning. Encoder analysis remains central as the first diagnostic stage; ACPC is the downstream planning-relevant continuation, not a replacement for encoder analysis.
