# Paper 1 final submission audit — 2026-06-23

This audit reflects the manuscript state after the ACPC theory-strengthening pass was applied to `paper1/main.tex`.

## Verdict

The paper is now a coherent arXiv-ready **diagnostic empirical study** once the remaining human-only metadata blockers are filled. The theory section is no longer just motivational prose: it gives a fixed-candidate-set bridge from ACPC to candidate-cost drift and action-ranking stability, and it formalises why ACPC without a discriminability guard permits collapse.

The paper should still not be described as a new training algorithm, a robustness benchmark, a full CEM theorem, or an independent-training-seed study.

---

## 1. Current theory status

### What is now in the paper

The ACPC section now contains:

1. A fixed paper instance of ACPC: identity readout on inference/cost latent rollouts, L2 over rollout tokens, horizon `H=8`, reported as `R_F`.
2. Proposition: ACPC-style rollout discrepancy bounds candidate-cost drift under an `L_J`-Lipschitz cost readout.
3. Proposition: for a shared candidate set, if every candidate's cost perturbation is at most `eta` and the clean top-1/top-2 cost margin is larger than `2 eta`, the corrupted branch selects the same top-1 candidate.
4. Corollary: if every candidate has rollout-readout discrepancy at most `epsilon`, then the margin condition `Delta > 2 L_J epsilon` gives fixed-candidate top-1 stability.
5. Proposition: ACPC alone permits collapse; a constant encoder/predictor can drive same-state ACPC to zero while merging action-distinct states.

### Theory correctness assessment

The statements are mathematically sound under their stated assumptions. The key limitation is also stated correctly: these are fixed-candidate-set sufficient conditions. They do not prove stability of the CEM sampling distribution, repeated replanning, environment feedback, or closed-loop trajectory success.

Do not strengthen the theory language beyond this scope.

Safe phrases:

- `fixed-candidate-set stability condition`
- `candidate-cost drift bound`
- `candidate-ranking stability under a margin condition`
- `diagnostic principle with a discriminability guard`

Unsafe phrases:

- `ACPC guarantees robustness`
- `ACPC proves CEM stability`
- `ACPC is sufficient for closed-loop success`
- `method-invariant theorem`

---

## 2. Remaining blockers and final checks

### 2.1 Real author list

`main.tex` still contains:

```tex
\newcommand{\arxivauthors}{Author names to be supplied for arXiv v1}
\author{\arxivauthors}
```

This must be replaced with the real arXiv author list before submission. Do not upload arXiv with placeholder or anonymous authors.

### 2.2 Final public URL

The acknowledgement now points to the paper-facing branch:

```tex
https://github.com/Holosemantix/le-wm/tree/ag/dev
```

Public access was checked on 2026-06-23 with `git ls-remote`; the `ag/dev`
branch is visible. The local repository contains the paper-facing source,
`DATA_MANIFEST.md`, JSON artifacts, and manifest hashes. If the final public
location differs, update the acknowledgement before submission.

### 2.3 Build and arXiv package

Before submission, run:

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
```

After the real author list is filled, run:

```bash
bash paper1/check_arxiv_ready.sh
```

The readiness script should fail while the author placeholder remains; that is intentional.

---

## 3. Writing and claim status

The major claim risks have been fixed:

- The title is appropriately narrow: Gaussian visual robustness in JEPA latent world models.
- The contribution now includes the planner link, not only a diagnostic lens.
- The ACPC basin comparison is described as post-hoc diagnostic/localisation, not model selection.
- The full LeWM ACPC-basin grid is in the appendix, reducing cherry-pick concerns.
- The Discussion uses a compact scope paragraph rather than repeating defensive caveats throughout the paper.

Remaining optional polish:

1. If page length matters, compress the Related Work subsections and move more detail to appendix.
2. If the prose still feels defensive, keep only three limitation reminders in the main text: ACPC basin paragraph, cross-checkpoint paragraph, and Discussion scope paragraph.
3. Keep `point-best` rather than `optimum` unless the statement is mathematically about an actual optimum.

---

## 4. Final readiness status

Ready after human metadata + final build checks:

- Theory bridge: ready.
- Claim boundary: ready.
- Full-grid transparency: ready.
- Negative ablation framing: ready.
- arXiv packaging script: ready.
- Public branch URL: ready for `ag/dev`, subject to final human location choice.

Not ready until completed:

- Real author list replaces `\arxivauthors`.
- `main.bbl` is generated and included in the arXiv source bundle.
- `tools.check_paper1_consistency`, `paper1/build.sh --clean`, and `paper1/check_arxiv_ready.sh` pass in the final local environment.
