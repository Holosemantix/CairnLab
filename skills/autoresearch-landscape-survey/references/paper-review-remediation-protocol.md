# Paper Review and Remediation Protocol

This protocol is for top-conference manuscript review, writing improvement,
claim calibration, citation audit, and iterative remediation. It is a CairnLab
workflow asset: it can produce reviewer evidence, issue lists, and remediation
plans, but it does not authorize scientific claim lifecycle transitions.

Core boundary:

- No artifact, no claim.
- LLM reviewers propose; deterministic verifiers, provenance records, and
  humans decide under CairnLab governance.
- Manuscript reviews, scores, and rebuttal plans are non-authoritative review evidence until imported through explicit evidence objects, verifier certificates, human gates, and append-only transition events.

## Required Inputs

Before judging or editing a manuscript, collect the current artifact set:

- paper source, compiled PDF, figures, tables, bibliography, and appendix;
- data manifests, generated artifacts, scripts, logs, hashes, and checkpoints;
- train/eval seed definitions and exact aggregation conventions;
- reviewer comments, prior remediation plans, and unresolved findings;
- conference format constraints and anonymity requirements.

If a referenced artifact is missing, the claim supported by that artifact is not
ready. Do not fill gaps from memory.

## Independent Review Rounds

Every round is an independent review of the current manuscript.

- Reset the score at the start of each round; do not reward the paper merely
  because it improved relative to a previous draft.
- Judge the current manuscript as if it were newly submitted today.
- Keep a round-local finding ledger with severity, evidence, concrete fix, and
  whether the fix needs retraining, new evaluation, new diagnostics, or only
  writing/reanalysis.
- Separate acceptance probability from effort invested. A long remediation
  history is not evidence of contribution quality.
- Re-score only after verifying the changed manuscript, not after accepting the
  intent of a fix.

Use at least these decisions:

- `strong_accept_baseline`: the current paper would be clearly competitive for
  the target track without relying on author promises.
- `weak_accept_or_borderline`: evidence is real but remaining weaknesses could
  decide the review.
- `weak_reject`: useful paper, but contribution, evidence, or clarity is still
  below the target bar.
- `reject`: central claims unsupported, misleading, unreproducible, or out of
  scope for the target venue.

## Strong-Accept Continuation Rule

When the user asks for top-conference remediation, do not stop merely because
the paper reaches `weak_accept_or_borderline`. Continue independent remediation
rounds until one of these conditions holds:

- the current artifact set reaches `strong_accept_baseline` under a fresh,
  objective review of the target track;
- all remaining decision-changing weaknesses require retraining, new
  large-scale experiments, missing author/private metadata, or user
  authorization that is outside the current task scope;
- the user explicitly pauses or narrows the objective.

Each continuation round must still begin from a fresh review score. Never lift
the score because previous rounds were extensive, difficult, or improved the
draft. If the paper cannot honestly reach `strong_accept_baseline` without
retraining or new large-scale evaluation, say so and keep the remaining blockers
separate from feasible fixed-checkpoint or writing-only work.

## Round Execution Order

Run every remediation round in this order:

1. Review the current artifact set and assign the recommendation, score, and confidence before proposing fixes.
2. List findings by decision impact, then classify each fix as `no-retraining`, `new evaluation/diagnostic with fixed checkpoints`, `retraining-required`, or `writing-only`.
3. Execute the feasible `no-retraining`, `new evaluation/diagnostic with fixed checkpoints`, and `writing-only` items that are in scope for the current task.
4. Leave `retraining-required` items as explicit blockers or future work unless the user authorizes retraining.
5. Rebuild, rerun consistency checks, and re-review the changed manuscript as a fresh submission before updating the score.

## Writing And Structure Audit

Review every section, paragraph, sentence, figure, table, formula, and caption.

Check that:

- the title, abstract, introduction, and contributions state a clear thesis
  with enough weight for the venue;
- the abstract contains the right problem, method or diagnostic contribution,
  evidence scale, key limitation, and no over-claimed scope;
- each paragraph has one job, a topic sentence, and a reason to exist;
- each sentence is specific, non-AI-generated in tone, and free of vague
  filler, ritual caveats, or repetitive reviewer-defense language;
- section titles match the content and expose the main line instead of artifact
  bookkeeping;
- main text contains must-read evidence, while reproduction ledgers, artifact
  maps, extra diagnostics, and audit trails move to appendices;
- terminology is stable and reader-facing names are not implementation keys;
- the paper avoids diagnostic name soup and acronym overload unless each term
  answers a necessary question;
- the length is justified by evidence density, not by defensive patching.

Remove prose that sounds like a checklist of reviewer anxieties. A strong paper
is honest about limits while still making a crisp contribution.

## Claim Calibration

Claims must be strong enough to matter and narrow enough to be true.

Require:

- a visible central contribution, not just a collection of caveats;
- no fake novelty from stacking conditions around a natural idea;
- no over-claim such as universal robustness, general transfer, closed-loop
  guarantee, or method superiority unless directly supported;
- no under-claim that makes the paper read as "not a method, not a predictor,
  not a guarantee" without a positive contribution;
- explicit distinction between method, diagnostic, benchmark, empirical
  analysis, theory, and release artifact;
- clear authority of closed-loop scores when diagnostics are only triage or
  localization signals;
- post-hoc, representative, and prospective evidence labeled separately.

When scores are noisy or based on finite sampling, do not emphasize exact best
checkpoints or tiny ordering differences. Prefer plateau, within-tolerance,
mean/std, confidence interval, regret-to-best, and failure-case reporting.

## Diagnostic-Paper Acceptance Gates

When a manuscript is primarily a diagnostic, analysis, or reframing paper, apply
extra pressure before calling it top-conference ready. A careful diagnostic paper
can still be a weak reject if it lacks decision-changing evidence beyond a
natural framing.

Check these gates explicitly:

- **Contribution type**: decide whether the current paper is a method,
  diagnostic, benchmark, theory, or empirical-analysis submission. Do not let a
  diagnostic paper borrow method-paper language unless it introduces and tests a
  training objective, algorithm, or intervention.
- **Technical depth**: definitions such as same-state predictive consistency
  plus different-state separability are natural. Treat Lipschitz drift, margin
  stability, union-bound, and local Taylor arguments as supporting analysis
  unless they prove a non-obvious guarantee actually used by the experiments.
- **Matched-stressor discount**: if the main behavior is "train with Gaussian
  noise, evaluate on Gaussian noise", discount novelty unless the paper shows a
  causal diagnostic link, stronger external comparison, or broader held-out
  evidence. State that matched augmentation recovery is expected and explain
  what the diagnostics add beyond that sanity check.
- **Selector baseline demotion**: if an aggregate diagnostic selector is only
  comparable to fixed-endpoint, single-metric, or random/plateau baselines, do
  not sell it as a superior selector. Reframe it as plateau localization,
  triage, or failure-case explanation, and put the baseline audit near the
  claim it qualifies.
- **Discriminability strength**: proxy guards such as effective rank, ID probes,
  transition resolution, or state-distance margin pass-rate do not establish
  oracle task semantics. For selective-consistency claims, prefer near-boundary
  or contact/topology/goal-relation pairs; otherwise label the result as a
  state-proxy guard and keep oracle semantic preservation as a blocker.
- **External baseline need**: for a main-conference robustness or method claim,
  require at least one meaningful external baseline, objective ablation, or
  competing method family unless the venue and thesis are explicitly diagnostic.
  Examples include standard pixel augmentation, encoder-level consistency,
  reconstruction-based world models, Dreamer/TD-MPC-style baselines, robust MPC,
  or a prototype paired-predictive consistency objective.
- **Metric budget**: the main paper should usually carry three evidence types:
  closed-loop behavior, one rollout/cost diagnostic family, and one selective
  margin guard. Move extra acronym families, ledgers, probes, and audit trails
  to appendix unless each answers a necessary main-text question.

If these gates fail, keep the score at `weak_reject` or
`weak_accept_or_borderline` even if the paper is honest, reproducible, and much
improved. Record which remaining gates are fixable without retraining and which
are decision-changing blockers requiring new training, external baselines, or
new labeled/constructed data.

## Theory Audit

The theory section must deepen the paper rather than decorate it.

Check that:

- definitions are necessary, minimal, and connected to the empirical protocol;
- assumptions are stated in measurable terms and mapped to reported quantities;
- theorem conclusions match what the experiments can test;
- finite-sample or calibration statements say which sampling distribution they
  cover;
- stability, margin, Lipschitz, or concentration results are not sold as
  adaptive closed-loop guarantees unless the proof covers adaptive replanning
  and environment feedback;
- theory explains why the chosen diagnostics should predict, localize, or bound
  the relevant failure mode;
- limitations are integrated into the theory-to-experiment bridge instead of
  appended as apologies.

If new theoretical framing, inequalities, or proof techniques are introduced,
audit whether new references are needed before finalizing the manuscript.

## Experiment And Artifact Audit

For every empirical claim:

- trace the number to a source artifact, script, command, seed set, and hash;
- verify train seeds, evaluation seeds, sample counts, aggregation convention,
  and units;
- distinguish independent training-seed variance from evaluation-seed variance;
- report uncertainty when sampling randomness can change interpretation;
- identify matched-stressor, held-out seed, held-out checkpoint, and unseen
  perturbation evidence separately;
- avoid cherry-picking stressors or endpoints; if rows are selected for
  localization, say so;
- compare against obvious non-retraining baselines such as fixed endpoint,
  single-metric selector, random candidate, and oracle lower bound when making
  selector claims;
- keep fixed-checkpoint remediation honest: with fixed checkpoints, allowed
  work includes new diagnostics, new evaluation, reaggregation, consistency
  checks, figures, and writing; it does not include implying retrained models;
- mark any claim that still needs retraining or new data as future work.

Numerical tables must round consistently and agree with artifacts. Captions
must state what seeds and sample counts are averaged and whether the table is a
main result, scope check, sanity check, or appendix audit.

## Figures, Tables, Formulas, And Layout

Inspect the compiled PDF, not just source.

Require:

- no stale labels, old terminology, or implementation strings in figure text;
- axes, legends, captions, and table headers that are interpretable without
  reading scripts;
- formulas whose symbols are introduced before use and reused consistently;
- theorem/proposition names that match the actual claim strength;
- tables that fit, align units, avoid overprecision, and expose the conclusion;
- figures that support the argument rather than decorate it;
- appendix tables that do not distract from the main narrative.

If a table or figure exists only because a reviewer might ask, either connect it
to a necessary question or move it to reproducibility material.

## Reference And Citation Audit

Reference checks require current source verification.

- Use web lookup for bibliography entries, author order, venue, year, URL/DOI,
  arXiv version, and project names when they could have changed or are not
  already verified by a local source artifact.
- Prefer primary sources: official papers, official documentation, project
  repositories, released artifacts, or publisher pages.
- Verify that the cited paper actually supports the sentence that cites it.
- Do not cite related work as a placeholder for unsupported novelty.
- If a new theoretical or empirical analysis relies on a known result, add the
  right reference and confirm bibliographic details.
- For double-blind submissions, remove public self-identifying URLs, author
  acknowledgments, non-anonymous artifact pointers, and wording that reveals
  identity.

Record the citation audit date and unresolved bibliography risks in the review
ledger.

## Review Output Format

Each round should produce:

- recommendation and score with confidence;
- strengths that matter for acceptance;
- weaknesses ordered by decision impact;
- concrete remediation items grouped as no-retraining, new evaluation/diagnostic
  with fixed checkpoints, retraining-required, and writing-only;
- artifact or source references for every numeric concern;
- current over-claim and under-claim risks;
- updated acceptance estimate after verification.

The final recommendation must stay objective. Do not make the score drift upward
because the paper has been through many rounds.

## Mature Patterns To Reuse

Borrow workflow ideas from strong AutoResearch and paper-audit systems while
preserving CairnLab boundaries:

- reviewer finding -> concrete issue -> patch/eval/diagnostic -> re-review;
- claim -> evidence artifact -> verifier/check -> human approval path;
- structured artifact ledger and hash recording;
- adversarial review plus material-dissent handling;
- separate producer, reviewer, verifier, and human-gate roles;
- decision trace packages for high-impact or release-stage claims.

Do not borrow the weak pattern of letting an LLM reviewer, consensus panel,
paper verifier prose, dashboard score, or polished manuscript decide scientific
release.
