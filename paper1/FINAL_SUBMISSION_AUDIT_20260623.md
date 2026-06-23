# Paper 1 final submission audit — 2026-06-23

This is the post-rewrite audit after the submission-readiness and arXiv-v1 readiness passes. It is written as an execution note for the next Codex/editorial pass.

## Verdict

The paper is now much closer to a submit-ready **diagnostic empirical study**. The title, abstract, contribution framing, ACPC instantiation, full-grid ACPC appendix, and release-gate notes have moved in the right direction.

The remaining issues are not about missing a large theorem. The formal section is enough for an arXiv diagnostic paper if it is kept as a fixed-candidate-set planner-stability view. The remaining blockers are mostly:

1. arXiv author placeholder and final public code/data URL.
2. A few wording issues that still sound like rebuttal or over-claim mechanism.
3. A small theorem-proof precision pass.
4. Main-text length and appendix density.
5. Final build/tarball verification after real author names are filled.

Do not invent author names, new experiments, or stronger theory.

---

## 1. Hard blockers before arXiv submission

### 1.1 Author placeholder remains

Current `main.tex` still has an arXiv author placeholder:

```tex
\newcommand{\arxivauthors}{Author names to be supplied for arXiv v1}
\author{\arxivauthors}
```

This is not acceptable for final arXiv upload. Replace with the real author list. Do not use anonymous authors or generated placeholder names.

### 1.2 Final public code/data URL must be verified

Current acknowledgement says:

```tex
The complete code and data are available at \url{https://github.com/qun-team/wm_exp};
```

Before submission, verify that this is the actual public repository containing the paper-facing code, artifacts, manifest, and hashes. If the final public repo is `Holosemantix/le-wm` or another URL, update the acknowledgement and any metadata accordingly.

### 1.3 `main.bbl` must exist in the arXiv source package

`main.bbl` is not necessarily tracked in the repo. The README tarball command assumes `bash build.sh --clean` creates it before packaging. Verify this locally and ensure the final tarball contains `main.bbl` whose basename matches `main.tex`.

---

## 2. Theory/formal section audit

### 2.1 Overall assessment

The formal section is now useful and mostly correct. It adds the missing bridge:

```text
ACPC -> bounded candidate-cost drift -> stable candidate ranking when clean margin is large enough.
```

This is sufficient for a diagnostic paper. It should not be promoted to a theorem about full CEM, repeated replanning, or closed-loop trajectory stability.

### 2.2 Required precision edits

#### Candidate top-1 stability should state cost direction and uniqueness

Current proposition assumes, but does not explicitly say, that lower cost is preferred and that the clean top-1/top-2 are ordered by increasing cost.

Edit the statement to include:

```tex
Assume lower cost is preferred, and let $\mathbf a^{(1)}$ and $\mathbf a^{(2)}$ denote the lowest- and second-lowest-cost candidates on the clean branch, with a strict margin.
```

This avoids ambiguity about whether `top-1` means max score or min cost.

#### Proof should write the key inequality explicitly

Replace the current proof sentence with a more explicit derivation:

```tex
For any $j\neq 1$,
\[
C_{\tilde h}(\mathbf a^j,g)-C_{\tilde h}(\mathbf a^{(1)},g)
\ge
C_h(\mathbf a^j,g)-C_h(\mathbf a^{(1)},g)-2\eta
\ge \Delta-2\eta >0 .
\]
Thus every non-top candidate remains more costly than the clean top-1 candidate on the corrupted branch.
```

This is mathematically cleaner and removes any possible confusion about ties.

#### Link Proposition 1 to the paper-facing ACPC metric

The proposition uses a generic sequence metric `d_H`, while the paper-facing ACPC basin uses L2 over rollout tokens. Add one sentence after Proposition 1:

```tex
In the reported basin diagnostic, this condition is instantiated with the L2 rollout-token distance underlying $R_F$; other downstream readouts are treated only as exploratory diagnostics.
```

Do not claim that the weighted sum definition of ACPC automatically equals every downstream `d_H` unless the paper explicitly defines it that way.

### 2.3 Theory language to avoid

Do not write:

- `ACPC guarantees robustness`.
- `ACPC proves CEM stability`.
- `ACPC is sufficient for closed-loop success`.
- `The propositions explain all empirical failures`.

Use:

- `fixed-candidate-set sufficient condition`.
- `planner-stability view`.
- `candidate-ranking stability under a margin condition`.
- `motivation for the diagnostic readouts`.

---

## 3. Main-text writing and claim audit

### 3.1 The title and abstract are now aligned with the evidence

The current title is appropriately narrow:

```text
A Diagnostic Study of Gaussian Visual Robustness in JEPA Latent World Models
```

The abstract is substantially better than the old version. It is diagnostic, not method-claiming. Keep the last sentence limiting training algorithm and independent training-run variability.

### 3.2 Replace remaining “arXiv version” wording

Current discussion has:

```tex
\paragraph{Scope of this arXiv version.}
This version is a controlled diagnostic study.
```

For a paper manuscript, prefer:

```tex
\paragraph{Scope.}
This paper is a controlled diagnostic study.
```

This reads less like an internal release note.

### 3.3 Soften unsupported mechanism speculation

Current text says:

```tex
Compressing the representation ... is acceptable and even beneficial: a more compact latent space is easier to plan in.
```

This is stronger than the data prove. Replace with:

```tex
The observed compression appears compatible with high success in this task; the data do not by themselves show that compactness is the cause of the improvement.
```

Similar caution should be applied to any claim that a task is easier because the latent is more compact.

### 3.4 Replace “optima” with “point-bests” where possible

The sweep table still uses phrases like:

```text
unperturbed and noisy optima dissociate
```

Prefer:

```text
unperturbed and noisy point-bests differ
```

Reason: each cell is one training run, so `optimum` reads too strong.

### 3.5 Avoid repeated “not a predictor / not an oracle” phrasing

The current paper is much improved, but the main text still repeats the diagnostic limitation in several places. Keep the limitation in:

- ACPC basin paragraph.
- Discussion scope paragraph.
- Cross-checkpoint paragraph.

Do not repeat it in every result transition.

---

## 4. Empirical/table audit

### 4.1 Full-grid ACPC appendix is a good addition

The full LeWM ACPC-basin grid is now included. This addresses the earlier post-hoc selection concern. The wording correctly says the basin radius is not used to select checkpoints and is not a reliable standalone predictor.

### 4.2 Verify full-grid numbers against artifact before submission

The table appears manually embedded. Before final upload, run the consistency checker or a small validation script to confirm every row matches:

```text
assets/paper1_data/acpc_basin_diagnostics.json
```

At minimum verify:

- all 4 tasks × 9 std levels are present;
- `obs 0.08` values match `canonical_evals_20260517.json`;
- `R_E`, `R_F`, and `R_F/R_E` match the basin artifact after rounding;
- `obs-best` labels match the table point-bests.

### 4.3 Partial correlation remains correctly bounded

The text correctly says `n=9`, partial correlation is a small-sample diagnostic check, and CIs are wide. Do not strengthen this into selection or prediction.

### 4.4 PLDM remains appendix-level replication/boundary

The PLDM section is appropriate as an appendix. Avoid moving too much PLDM detail into the main text unless the main text becomes too LeWM-only.

---

## 5. Length and structure audit

### 5.1 Current structure is acceptable for arXiv, still long for a conference main paper

Main text is now much shorter than before, and long tables moved to appendix. For arXiv v1 this is acceptable.

For a conference submission, further compress:

- Related Work: merge some subsections into 3 paragraphs.
- Experiments setup: remove repeated checkpoint/evaluation text between `Study protocol` and `Experiments`.
- Diagnostic analysis: shrink mechanistic bullets to a compact paragraph or move all bullets to appendix.

### 5.2 Duplicate setup issue

There is still some repetition between:

- `Study protocol` section;
- `Experiments -> Setup` subsection.

Keep one concise setup block in main text. The other should become a short pointer or be removed.

Suggested compression:

```tex
\section{Study protocol}
```

can contain tasks, models, checkpoints, and evaluation. Then `Experiments -> Setup` can be removed or reduced to one sentence:

```tex
All results use the protocol in Section~X; we first report closed-loop corruption behaviour, then the paired ACPC-basin diagnostic, and finally diagnostic limitations.
```

---

## 6. “AI feel” audit

### 6.1 Improved but still slightly over-defensive

The manuscript now has less of the old “not X, not Y, scope boundary” pattern. Remaining phrases to grep and reduce if too frequent:

```bash
rg -n "not a|does not|do not claim|scope|boundary|predictor|oracle|mechanism localisation|controlled diagnostic|post-hoc" paper1/main.tex
```

Do not delete every occurrence. The goal is to avoid a defensive rhythm in the main text.

### 6.2 Specific phrases to consider replacing

- `mechanism localisation` -> `localise what changed` or `local diagnostic`.
- `boundary check` -> `replication check` or `scope check`, depending context.
- `coarse pressure` -> `single global augmentation strength`.
- `paper-facing` / `method-facing` should stay out of final prose unless needed in appendix tooling notes.

---

## 7. Release gate checklist

Run after final author and URL edits:

```bash
python -m tools.check_paper1_consistency
cd paper1 && bash build.sh --clean
rg -n "undefined references|Citation .* undefined|Reference .* undefined|Fatal error|Undefined control sequence|No file main.bbl" main.log || true
```

Then prepare the arXiv source tarball from `paper1/README.md` and inspect:

```bash
tar -tzf /tmp/paper1_arxiv_v1_src.tar.gz | sort
```

The tarball must not contain:

- `PLAN.md`;
- `CODEX_SUBMISSION_READINESS.md`;
- `ARXIV_V1_READINESS_PLAN.md`;
- this audit file;
- raw JSON artifacts unless intentionally uploaded as ancillary files;
- build logs or temporary files;
- unused figures.

---

## 8. Final readiness status

### Ready after minor edits

- Diagnostic framing.
- ACPC formal bridge.
- Full-grid ACPC transparency.
- Main claim boundary.
- Build script no longer suppresses BibTeX errors.
- README arXiv packaging guidance.

### Not yet final-submission-ready until fixed

- Real author list must replace `\arxivauthors`.
- Public code/data URL must be verified.
- `main.bbl` must be generated and included in source package.
- The small formal-section precision edits above should be applied.
- Final build and tarball audit must be run after those edits.

