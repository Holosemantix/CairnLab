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
- When a fresh external, simulated, or user-supplied review is materially lower
  than a prior internal score, treat the review as new evidence about the
  current manuscript. Do not defend the prior score. First identify which gate
  was overestimated (for example method novelty, external baselines,
  discriminability strength, selector value, or matched-stressor scope), then
  reset the recommendation before planning fixes.

## Score-Drift Calibration Gate

Before accepting any score increase, explicitly check why an internal
multi-round review might diverge from a fresh reviewer. Common root causes are:

- improvement bias: scoring the delta from the previous draft instead of the
  absolute current submission;
- track mismatch: judging a diagnostic or analysis paper by a friendlier track
  while reporting the score as if it were for the main method/general track;
- evidence substitution: treating honest limitations, artifact volume,
  appendix breadth, or polished wording as if they were new causal evidence,
  external baselines, stronger semantics, or a trained method;
- local overfitting: optimizing against the previous review ledger rather than
  asking whether a new reviewer would still see the central weakness;
- future-work optimism: giving credit for objectives, baselines, labels, or
  evaluations that are only proposed;
- plateau/selector optimism: treating a diagnostic rule that matches simple
  fixed-endpoint or single-metric baselines as a superior selector;
- matched-stressor optimism: treating train-Gaussian/evaluate-Gaussian recovery
  as surprising robustness evidence without a stronger causal or external
  comparison.

Every top-conference review round must report scores separately for at least:

- main-conference method/general track;
- diagnostic, empirical-analysis, or benchmark track;
- workshop/resource positioning, when relevant.

If an independent or user-supplied review is lower by at least one
recommendation category or 1.0 score point, create a score-disagreement ledger
before editing:

- prior internal recommendation and score;
- fresh external/user recommendation and score;
- target track assumed by each score;
- gates the internal review overestimated;
- whether any verified new artifact directly closes each criticized weakness;
- corrected score ceiling before further work.

The corrected main-track score must not exceed the lower fresh-review score by
more than 0.5 points unless a verified new artifact, new evaluation, new
baseline, new label/protocol, or accepted proof directly closes the cited
decision-changing weakness. Writing-only claim calibration can improve clarity
and trust, but it cannot by itself convert a weak-reject method-track paper into
a strong-accept baseline.

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
2. List findings by decision impact, then classify each fix as `no-retraining`, `new evaluation/diagnostic with fixed checkpoints`, `retraining-required`, or `writing-only`; in the same pass, list deletion candidates for content that should be removed, merged, demoted, or moved to appendix.
3. Execute the feasible `no-retraining`, `new evaluation/diagnostic with fixed checkpoints`, `writing-only`, and subtractive cleanup items that are in scope for the current task.
4. Leave `retraining-required` items as explicit blockers or future work unless the user authorizes retraining.
5. Rebuild, rerun consistency checks, and re-review the changed manuscript as a fresh submission before updating the score.

## Subtractive Remediation Gate

Every round must ask what should be removed, not only what can be added.
Before adding new diagnostics, tables, caveats, appendix ledgers, or reviewer
patches, produce a subtractive ledger with deletion candidates.

For each manuscript section, figure, table, formula, appendix block, diagnostic
family, artifact ledger, and repeated caveat, decide whether it should be:

- kept in main text because it answers a necessary reader question;
- merged with a nearby result or limitation;
- demoted from a claim to a scope check, sanity check, or reproduction note;
- moved to appendix or artifact documentation;
- deleted because it creates main-text burden, acronym load, defensive patching,
  fake novelty, or no decision-changing evidence.

Apply these checks before expanding the paper:

- If a paragraph only reassures reviewers without advancing the thesis, remove
  or merge it.
- If a diagnostic name, acronym, table, or ledger does not answer a necessary
  question, delete it from the main narrative or move it to reproducibility
  material.
- If several caveats say the same thing, keep the strongest precise one and
  remove the rest.
- If a result is weak, mixed, or redundant with a stronger table, demote it
  rather than adding more explanation around it.
- If the paper's contribution appears to depend on stacking conditions,
  terminology, or artifact volume, simplify the claim instead of adding more
  qualifiers.

The review output must include a `remove/merge/demote` group alongside the
`no-retraining`, `new evaluation/diagnostic with fixed checkpoints`,
`retraining-required`, and `writing-only` groups. A remediation round is incomplete if it only lists additions.

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

## Weak-Reject Diagnostic-Paper Calibration Gate

When a fresh review still says Weak Reject or Borderline Reject after several
rounds of remediation, treat that as evidence that the paper may have been
locally optimized around prior comments rather than lifted across the target
venue bar. Before editing, classify the paper under this gate.

For diagnostic or empirical-analysis manuscripts, require the review ledger to
answer these questions explicitly:

- **Method-paper ceiling**: does the paper introduce and evaluate a new training
  objective, algorithm, or intervention? If not, do not score it as a strong
  method paper. A no-retraining diagnostic can be valuable, but it cannot close
  an algorithmic-novelty weakness by writing alone.
- **Matched-stressor ceiling**: if the main behavior is matched train/eval
  augmentation, such as Gaussian training evaluated on Gaussian noise, treat the
  behavioral recovery as an expected axis unless the paper adds causal
  diagnostic evidence, held-out perturbation transfer, or an external baseline
  comparison that changes the review decision.
- **Selector-increment ceiling**: if an aggregate diagnostic selector is
  comparable to a fixed endpoint, a single metric, or a broad plateau baseline,
  count it as triage or plateau localization. Do not call it a reliable
  prospective selector, and do not let selector wording raise the score.
- **Theory-link ceiling**: a new calibration table, margin-conditioned flip
  audit, or finite-sample bound strengthens theory--experiment alignment only
  if it measures the theorem's actual terms. It does not become an adaptive CEM,
  closed-loop, or environment-feedback guarantee.
- **Semantic-proxy ceiling**: state-distance, effective-rank, transition, or ID
  probe guards are not oracle task semantics. If contact/topology/goal-relation
  or near-boundary labels are absent, keep that weakness open or mark it as a
  fixed-checkpoint diagnostic opportunity only when such labels can be
  constructed without retraining.
- **External-baseline ceiling**: PLDM-style method-family replication, negative
  ablations, or artifact completeness reduce narrowness concerns but do not
  replace strong baselines such as standard augmentation objectives,
  reconstruction-based world models, Dreamer/TD-MPC-style systems, robust MPC,
  or an ACPC-derived training objective when the target track is a robustness
  method paper.
- **Appendix-burden ceiling**: extra ledgers, audits, and diagnostic families
  do not raise the score unless they simplify the decision path. If the main
  text now has more than three or four must-read evidence layers, verify that
  each answers a necessary question and move the rest to reproducibility
  material.
- **Fixed-checkpoint ceiling**: with fixed checkpoints and no retraining,
  feasible work is limited to new diagnostics, eval-only tests, reaggregation,
  consistency checks, figures, and writing. If the remaining decision-changing
  blockers are a trained objective, strong external baselines, broad
  perturbation training, or oracle/hand-labeled semantic data, record them as
  blockers instead of inflating the score.

Classification is mandatory before edits:

- `no-retraining`: artifact checks, reaggregation, consistency fixes, figure or
  table relabeling, manuscript restructuring, and claim calibration that use
  existing data/checkpoints.
- `new evaluation/diagnostic with fixed checkpoints`: eval-only baselines,
  unseen-stressor scoring, constructed near-boundary semantic audits,
  margin-conditioned flip curves, or additional uncertainty estimates that use
  existing checkpoints and logged data.
- `retraining-required`: new objectives, robustness training, strong external
  baselines that require training, broad perturbation-family training, or
  method claims requiring new learned models.
- `writing-only`: abstract, title, contribution framing, limitation balance,
  appendix slimming, reference/citation cleanup, and non-AI-sounding prose.

Scoring rule: if the paper lacks a new objective/trained method, lacks strong
external baselines, remains centered on matched train/eval stressors, and uses
proxy rather than oracle semantic discriminability, the main-conference
method/general score should normally remain at `weak_reject` to
`weak_accept_or_borderline` even after no-retraining cleanup. A diagnostic or
empirical-analysis track score may be higher, but it must be reported
separately and justified by current evidence, not by remediation effort.

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
  with fixed checkpoints, retraining-required, writing-only, and
  remove/merge/demote;
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
