# Paper 1 five-reviewer final audit — 2026-06-24

This audit reviews the current `ag/dev` manuscript after the arXiv/submission readiness passes and the ACPC theory-strengthening pass. It is written from five strict reviewer perspectives and includes the remaining submission blockers.

## Overall verdict

The manuscript is now a coherent diagnostic paper. The central contribution is no longer just experimental: the ACPC section now provides a fixed-candidate-set theoretical bridge from paired predictive agreement to candidate-cost drift and then to action selection under a margin condition. The collapse proposition also makes the discriminability guard necessary rather than cosmetic.

The paper is not yet final-upload-ready because the author field still contains an arXiv placeholder and `main.bbl` must be generated locally. These are submission metadata/build blockers, not conceptual blockers.

---

## Reviewer 1 — Theory formalist

**Score: weak accept as diagnostic/arXiv paper; reject as theory paper.**

The theory section is now materially stronger. It contains:

- a fixed paper-facing ACPC instance: identity readout on inference/cost latent rollouts, L2 over rollout tokens, horizon `H=8`, reported as `R_F`;
- Proposition: rollout-readout discrepancy bounds candidate-cost drift under an `L_J`-Lipschitz cost readout;
- Proposition: bounded per-candidate cost perturbation plus clean top-1/top-2 margin greater than `2 eta` preserves the selected candidate in a shared candidate set;
- Corollary: bounded rollout discrepancy plus `Delta > 2 L_J epsilon` gives fixed-candidate top-1 stability;
- Proposition: ACPC alone permits collapse, so a discriminability guard is necessary.

I do not see a mathematical hallucination in these claims. The proofs are direct Lipschitz and triangle-inequality arguments. The important boundary is correctly stated: these results do not prove full CEM stability, repeated-replanning stability, or closed-loop success.

**Required tiny wording polish.** C1 currently says bounded ACPC implies bounded candidate-cost drift. Strictly, this requires the Lipschitz cost-readout assumption. Safer C1 wording:

```tex
\paragraph{C1 --- Diagnostic principle and planner link.} We formulate visual robustness for world-model control as action-conditioned predictive consistency with an action-relevant discriminability countercondition. Under a Lipschitz cost readout, bounded ACPC bounds candidate-cost drift on a fixed candidate set; with a sufficiently large clean action margin, it preserves the selected candidate within that set.
```

This keeps the contribution strong while matching the formal assumptions.

**Optional prose polish.** The sentence “The corollary is the formal reason...” is a bit too strong stylistically. Prefer:

```tex
The corollary gives a formal link between ACPC and planning without turning the diagnostic into a closed-loop guarantee.
```

---

## Reviewer 2 — Empirical/statistical reviewer

**Score: borderline/weak reject for main conference unless training seeds are added; acceptable for arXiv diagnostic study.**

The empirical scope is now stated honestly. The canonical LeWM grid uses one epoch-10 checkpoint per `std_max` cell, evaluated with three evaluation seeds and 100 trajectories per seed. This is not independent training-run variability, and the paper now says so.

The full ACPC-basin grid in the appendix is important. It reduces the earlier concern that only base-vs-selected endpoints were shown. The manuscript still should not imply ACPC is a selection rule. The current text correctly says the basin diagnostic is post-hoc and localises changes at checkpoints selected by closed-loop evaluation.

Remaining concern: all point-best language must remain point-best, not optimum. The current main sweep table uses “point-bests,” which is good.

Submission blocker: ensure the full-grid ACPC table has been checked against `assets/paper1_data/acpc_basin_diagnostics.json` after any artifact changes.

---

## Reviewer 3 — World-model / RL reviewer

**Score: weak accept as a diagnostic paper; reject as a method paper.**

The paper is now better positioned. It does not claim a new optimizer or training recipe. Instead, it claims that visual robustness for latent world-model control should be diagnosed after action-conditioned prediction and guarded by action-relevant discriminability.

This is a meaningful contribution because the theory section connects the diagnostic to candidate costs and action rankings. The negative ablations are also useful: target-view denoising is not enough, and error-based reweighting can erase PushT contact transitions. These negative checks make the diagnostic principle more credible.

Do not strengthen the method claim. The natural next objective is still future work: paired predictive-dynamics consistency after the action-conditioned predictor, gated by action sensitivity and protected by discriminability.

---

## Reviewer 4 — Writing / AI-feel reviewer

**Score: accept after small polish.**

The paper now reads less like a defensive AI-generated report. Related Work is shorter, the main story is clearer, and repeated caveats have been consolidated into a compact scope paragraph.

Remaining style issues:

1. Replace “The corollary is the formal reason...” with a softer, less promotional sentence.
2. Avoid repeated “diagnostic, not predictive” wording outside the ACPC basin paragraph, cross-checkpoint paragraph, and Discussion scope paragraph.
3. The appendix remains long, but acceptable for arXiv. For a conference submission, compress Related Work and move some diagnostic details to appendix.

---

## Reviewer 5 — Release/artifact reviewer

**Score: not yet final-upload-ready.**

The release hygiene is much better: `build.sh` checks BibTeX and undefined references, and `check_arxiv_ready.sh` creates a minimal source bundle.

Hard blockers:

1. `main.tex` still has `\arxivauthors` placeholder. arXiv must not be uploaded with this placeholder.
2. `main.bbl` is not tracked in the repository. It must be generated by `cd paper1 && bash build.sh --clean` and included in the arXiv source bundle.
3. The acknowledgement points to `https://github.com/Holosemantix/le-wm/tree/ag/dev`. Confirm this is intended as a public code/data URL. A branch URL is acceptable for a draft, but a tag or release archive is preferable for a stable arXiv version.
4. Run the final gate locally:

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
bash paper1/check_arxiv_ready.sh
```

`check_arxiv_ready.sh` should fail until the author placeholder is replaced.

---

## Final required edits before submission

### Must do

1. Replace `\arxivauthors` with the real author list.
2. Generate and include `main.bbl` in the arXiv source bundle.
3. Verify the public code/data URL; preferably use a stable tag/release rather than a mutable branch if possible.
4. Run the consistency/build/arXiv readiness commands locally.

### Recommended small wording patch

Apply these exact replacements in `paper1/main.tex`:

```tex
% Replace C1 paragraph with:
\paragraph{C1 --- Diagnostic principle and planner link.} We formulate visual robustness for world-model control as action-conditioned predictive consistency with an action-relevant discriminability countercondition. Under a Lipschitz cost readout, bounded ACPC bounds candidate-cost drift on a fixed candidate set; with a sufficiently large clean action margin, it preserves the selected candidate within that set.
```

```tex
% Replace the sentence beginning "The corollary is the formal reason..." paragraph with:
The corollary gives a formal link between ACPC and planning without turning the diagnostic into a closed-loop guarantee. It connects paired predictive agreement to candidate-cost stability and then to action selection under a margin condition. Candidate-ranking agreement and margin-conditioned action flips are downstream readouts of this chain. The result is limited to a fixed candidate set; CEM resampling, repeated replanning, and environment feedback can still change the closed-loop trajectory.
```

These changes are not conceptual blockers, but they make the theory claim maximally precise and reduce promotional tone.

## Bottom line

The theory is now sufficiently complete for the paper's intended contribution. It is not a full theory paper, but it is strong enough to make the diagnostic framing central rather than ornamental. The remaining blockers are metadata/build/release issues and two small wording polishes.
