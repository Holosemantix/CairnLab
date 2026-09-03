# Paper Writing Quality Module

This module defines the pure manuscript-writing capability expected by
CairnLab-facing paper workflows. It is not a claim-transition authority. It
turns research artifacts into a readable, venue-shaped manuscript draft and
produces reviewable writing-quality evidence.

Core boundary:

- No artifact, no claim.
- LLM writers propose manuscript text.
- Writing reviews, style scores, and polished PDFs are non-authoritative evidence.
- CairnLab claim lifecycle transitions still require explicit evidence items,
  verifier certificates, provenance records, governance gates, human gates, and
  append-only transition events.

## Surveyed Patterns To Reuse

Mature writing workflows are strongest when they separate writing, review,
format validation, and evidence audit.

- The local `scientific-writing` skill contributes the useful
  outline-to-prose pattern, full-paragraph final manuscripts, IMRAD defaults,
  citation hygiene, and reporting-guideline awareness. Do not inherit its
  blanket "always generate many AI figures" rule; figures must be evidence- or
  explanation-bearing, not decorative.
- The local `venue-templates` skill contributes venue-specific format and
  style constraints: page limits, citation style, anonymization, audience
  level, contribution bullets, ablations, reproducibility expectations, and
  reviewer expectations.
- ARIS paper-writing contributes reusable workflow mechanics: paper plan,
  figure plan, section writing, compilation, review/improvement loops,
  negotiated acceptance contract before writing, proof audit, paper-claim audit,
  citation audit, and rendered-PDF inspection.
- AutoResearchClaw contributes stage contracts, stage/context skill matching,
  HITL checkpoints, peer review, verification reports, and failure/retry
  discipline.
- data-to-paper contributes backward traceability from manuscript numbers to
  code, results, and data, plus human oversight for accuracy.
- Agent Laboratory contributes a simple phase split: literature review,
  experimentation, and report writing, with human feedback at each phase.

Reusable conclusion: a good paper-writing module needs an acceptance contract,
section-level writing gates, artifact traceability, independent review, and
compiled-PDF checks. It is not enough to ask a model to "write better".

## Open-Source Reuse Decision

The closest direct open-source base is PaperOrchestra. It is Apache-2.0 and
specifically targets automated AI research paper writing from unconstrained raw
materials to LaTeX manuscripts. It has a practical CLI, target-template input,
outline agent, literature review agent, section writing agent, plotting
workflow, content refinement loop, PDF compilation, VLM layout review, and
autoraters. If CairnLab later wants an executable paper-writing engine rather
than only a workflow asset, PaperOrchestra is the strongest candidate to wrap as
an external artifact producer.

Do not vendor PaperOrchestra directly into CairnLab authority code. Direct reuse
would need an adapter layer because it brings model/API assumptions
(Gemini/OpenAI/Semantic Scholar), a full generation pipeline, plotting and PDF
dependencies, and writer-side prompts that are not always aligned with
CairnLab's claim-calibration rules. In particular, any prompt or refinement
logic that hides limitations, treats reviewer score as success, or writes a
"submission-ready" claim must be replaced or wrapped by CairnLab evidence,
verifier, governance, and human-gate rules.

ARIS is the best direct base for a Markdown skill workflow. It is MIT licensed
and its paper-writing skill already contains a plan -> figure -> write ->
compile -> improvement loop, style-reference isolation, an acceptance contract,
proof/claim/citation audits, and a verifier-as-truth submission gate. Reuse it
as a structural template for skill authoring, but do not copy its whole path,
model, or tool assumptions into CairnLab. The reusable part is the gate shape:
contract before writing, fresh auditors independent from the writer, mandatory
artifact emission, and refusal to call a paper green when audits fail.

data-to-paper is the strongest reference for data-chained manuscript
traceability. It should not be the base for this pure-writing module because it
is a raw-data-to-paper research platform, not a narrow writing skill. Reuse its
principle that every numeric manuscript value should trace to code/results/data
and remain human-verifiable.

Agent Laboratory and AI Scientist provide useful section tips, LaTeX edit/repair
loops, citation helpers, and compile checks. They are not suitable as the base
for this module: their writing components are embedded in broader autonomous
research systems, their prompts are less strict about concision and claim
calibration, and AI Scientist's license is not a standard permissive open-source
license for unrestricted reuse.

Current implementation decision:

- use ARIS as the closest reusable skill-level structure;
- use PaperOrchestra as the future executable-engine candidate, behind an
  external adapter if needed;
- use data-to-paper for traceability requirements;
- use Agent Laboratory and AI Scientist only for section-level and LaTeX repair
  patterns;
- keep this module as CairnLab's quality contract so any external writer remains
  a non-authoritative artifact producer.

## Required Inputs

Before writing or rewriting a paper, collect:

- target venue, track, page limit, anonymity requirement, citation style, and
  expected paper type;
- current source files, compiled PDF if present, bibliography, figures, tables,
  appendix, and template/style files;
- claim/evidence inventory with source artifacts for every headline result,
  number, theorem, method claim, comparison, limitation, and released asset;
- figure/table inventory with source data, scripts, seeds, aggregation
  conventions, and captions;
- notation, acronym, and terminology inventory;
- prior reviewer comments, unresolved findings, and any accepted human
  waivers.

If any artifact needed for a sentence is missing, write around the gap honestly
or mark the sentence blocked. Do not fill the gap from memory.

## Phase 0: Writing Contract

Before drafting, create a writing contract that is checkable after compilation.
It should be short enough to use and strict enough to block bad manuscripts.

The contract must name:

- target venue and track;
- paper type: method, diagnostic, benchmark, empirical analysis, theory,
  dataset/resource, systems, or survey;
- central thesis in one sentence;
- claim frame sentence in the form "This paper is X, not Y";
- allowed claim strength and disallowed over-claims;
- positive claim and boundary claim that must both appear in the abstract;
- required evidence-bearing figures and tables;
- abstract obligations: problem, contribution, evidence scale, key result,
  key limitation, and target word count;
- for revisions, the last accepted abstract and the evidence change that could
  justify preserving, compressing, replacing, or expanding it;
- section order and the one job of each section;
- notation/acronym budget;
- page budget for main text, appendix, and references;
- formatting gate: compile cleanly and inspect the rendered PDF;
- acceptance checks that can be answered from the final PDF plus artifacts.

Bad contract assertion:

```text
The writing is clear and strong.
```

Good contract assertion:

```text
Every acronym used in the abstract is defined before first use, and the abstract
contains no more than three nonstandard acronyms.
```

## Phase 1: Evidence-First Paper Plan

Plan the paper around claims and evidence, not around sections alone.

Before writing section prose, run a claim-compression pass. The manuscript is
not a project report, metric zoo, ablation dump, release note, or artifact
index. It should expose the smallest evidence loop that supports the
paper-facing claim.

### Claim Compression And Metric Scope Gate

Require:

- name the paper-facing claim before choosing tables, figures, or appendix
  material;
- admit a table, figure, theorem, appendix subsection, or negative ablation
  only when it is necessary for that claim or materially changes the reader's
  interpretation of that claim;
- map every core metric to a specific theory object, assumption, or final
  necessary condition;
- map every theory object used in the final claim to either an empirical
  readout, a stated assumption, or a weakened claim;
- keep behavioral endpoints separate from diagnostics: task score, reward,
  accuracy, or downstream success establishes the phenomenon, while
  diagnostics explain or localize it;
- do not collapse multiple necessary theoretical conditions into one scalar
  unless the paper also reports the components and explains the trade-off;
- classify each non-core display as core evidence, scoped audit, reproducibility
  support, internal provenance, debug material, or reviewer-defense material;
- move internal provenance, pilot failures, debug traces, artifact hashes,
  rendering commands, repository paths, and release-package bookkeeping out
  of the paper and into repository documentation or manifests;
- treat appendix as paper, not storage. Appendix material must extend evidence,
  prove/calibrate a claim, document essential protocol, or support
  reproducibility;
- include a negative ablation only when it rules out a key alternative
  explanation for the main claim and does not create a new method-comparison
  obligation;
- run an attack-surface check on every table: if the table looks like method
  comparison, selector validation, stability proof, or anti-collapse proof,
  the evidence must actually satisfy that reading or the table must be
  removed/demoted;
- preserve boundary cases with per-task, per-domain, per-seed, or per-scope
  visibility instead of hiding them inside averages or composite scores;
- allow theory caveats to remain caveats. Do not add a paper-facing metric for
  every intermediate theorem term unless the final paper-facing claim needs
  it;
- match verbs to evidence strength: use language such as localizes, audits,
  is consistent with, or calibrates for post-hoc diagnostics; reserve
  predicts, improves, proves, guarantees, solves, and generalizes for evidence
  that supports those exact claims.

Hard numeric gates:

- do not report a quantile, tail risk, confidence interval, or CVaR unless the
  required sample-level or distributional data exists;
- do not infer fine-grained diagnostics from means, medians, or aggregate
  summaries;
- do not call a result held-out, prospective, pre-registered, or lockbox if the
  selection rule used outcome labels or was fixed after seeing the result;
- every number in the paper must trace to a source artifact, script, or
  documented manual calculation.

The plan must include:

- central thesis;
- reader promise: what the reader should know after the paper;
- claim/evidence matrix;
- section map with each section's single job and argument order;
- theory-to-evidence map when theory terms support empirical claims;
- figure/table map with the question each display answers and whether it belongs
  in main text, appendix, or repository documentation;
- appendix/provenance split for extra audits, ledgers, diagnostic families,
  failed experiments, artifact inventories, and reproducibility material;
- citation scaffold with primary sources for novelty, methods, baselines, and
  theory;
- known blockers that require new experiments, retraining, source lookup, or
  human metadata.

Reject plans that are a list of topics without an argument.

### Structure-First Manuscript Plan Gate

Run a structure pass before creating new figures, converting tables, or adding
more caveats. Display work should follow the argument order, not compensate for
a confused structure.

Require:

- title, abstract, contribution paragraph, section order, and appendix split
  are fixed before figure/table conversion begins;
- each section has one reader-facing job and advances the claim chain rather
  than mirroring the remediation history;
- for theory-heavy, diagnostic, benchmark, or empirical-analysis papers, the
  main narrative introduces the theoretical or conceptual object before the
  operational metric, then reports the behavioral endpoint, diagnostic
  validation, mechanism evidence, boundary checks, and limitations in that
  order unless the venue requires a different structure;
- when a theory contains multiple necessary quantities, include a
  theory-to-evidence map that names the quantity, empirical audit or measured
  object, observed evidence, and limitation;
- main text displays answer must-read questions only. Promote theory-supporting
  evidence that is essential to the claim into the main narrative, and move
  dense exact values, threshold grids, sensitivity sweeps, artifact inventories,
  and raw audit ledgers to appendix or repository documentation;
- convert tables to figures only when the display is about a trend, region,
  uncertainty interval, event rate, mechanism contrast, or before/after
  comparison. Keep exact lookup values and compact supplementary numerics as
  tables;
- order main figures and tables by the evidence chain: concept or failure mode,
  primary behavior, core diagnostics, held-out or transfer validation, mechanism
  evidence, boundary checks, then supplementary exact values;
- if the claim requires two diagnostics or conditions jointly, the main display
  or adjacent prose must show the joint reading. Do not promote a single
  component as if it carried the full claim;
- captions should state scope, aggregation, sample or seed count when material,
  uncertainty meaning, and the non-claim boundary. They should not explain
  which internal review issue a display closes.

## Phase 2: Section Drafting Standards

Draft section by section from the plan. Final manuscript text must be flowing
prose except where the venue or method section explicitly expects lists.

Every section must satisfy:

- the section title matches the section's actual job;
- the table-of-contents structure reads like a scientific argument, not a remediation log, issue tracker, or engineering record;
- headings name the scientific object, mechanism, result, or limitation rather than the review process that produced the text;
- the first paragraph tells the reader why the section exists;
- every paragraph has one role and one topic sentence;
- every sentence either advances the argument, reports evidence, defines a
  necessary concept, or states a necessary limitation;
- transitions explain why the next paragraph follows;
- terminology is stable and reader-facing, not implementation-key-facing;
- no paragraph exists only to soothe a reviewer anxiety;
- no result sentence reports a number without a source artifact;
- no limitation paragraph repeats the same caveat in different words.

Use these section-specific gates.

## Generic Manuscript Gates

These gates are venue-independent and domain-independent. They apply to any
paper or technical manuscript regardless of field, task, metric, or system under
study. Each gate has a stable name so plans, reviews, and ledgers can cite it.

Policy and thresholds are separated on purpose: the requirements below are
policy and do not change per project, while every number lives in
`Configurable Gate Thresholds` at the end of this section and may be overridden
in the writing contract.

### G1. Progressive Disclosure Gate

A reader should be able to stop at any layer and still hold a correct, smaller
version of the paper.

Require:

- after the title and abstract, a newcomer can state the problem, the
  contribution, and the boundary of the contribution;
- after the introduction and the main figures, a reader can state the argument
  and the evidence chain without reading the methods;
- exact protocols, parameter grids, derivations, and per-condition values come
  later, in later sections or the appendix;
- technical completeness never requires every internal log, run, or intermediate
  result to appear in the main narrative. Completeness is reached by pointing to
  reproducible artifacts, not by transcribing them;
- each layer stays consistent with the layers above it. A later section may
  refine or bound an earlier claim, but must not silently contradict it;
- if a claim can only be understood after reading an appendix, either lift the
  minimal statement into the main text or narrow the claim.

### G2. Concrete-Before-Abstract Gate

Notation compresses something the reader already understands. Introduce the
thing, then the symbol.

Require:

- every formal object is motivated by one plain-language example, minimal
  instance, or concrete failure case before or immediately after its definition;
- every nonstandard term, symbol, and acronym is defined exactly once, at first
  use, where a reader will look for it;
- one concept has one stable name across text, equations, figures, tables,
  captions, and appendix. Synonyms used for prose variety are a defect, not
  style;
- a term that exists only to name an internal component is either promoted to a
  reader-facing scientific object or removed;
- when notation density is unavoidable, add a notation table and keep the same
  symbol from denoting unrelated objects;
- abstraction runs concrete, then general, then formal, not the reverse.

### G3. One-Job Gate

Every unit of the manuscript answers one question.

Require:

- one job per section, per paragraph, and per display;
- the topic sentence comes first and names that job;
- paragraph order is claim or reason, then evidence, then boundary or
  transition;
- a paragraph carrying two claims is split; a display answering two questions is
  split or re-scoped;
- project chronology is removed. The paper reports what is true, not the order
  in which it was discovered or repaired;
- reviewer-response prose is removed. If a concern is scientifically real, state
  it as a limitation or boundary check, not as a reply;
- section titles name the scientific job, so the table of contents alone reads
  as the argument.

### G4. Claim-Evidence Identity Gate

Absence of a result and a negative result are different scientific statements.
So are two numbers produced under different conditions.

Require:

- distinguish and label these states wherever they matter:
  - `not-run`: the experiment was never executed;
  - `unavailable`: it exists or was run, but cannot be reported here;
  - `inconclusive`: it was run and does not separate the hypotheses;
  - `failed`: it was run and contradicts the expectation;
- never let "we do not report X" stand where the honest statement is "X was not
  run", "X is unavailable", or "X did not work";
- whenever it could change interpretation, a reported quantity identifies its
  split or subset, the distinction between training seeds and evaluation
  episodes or trials, the budget and checkpoint-selection rule, the comparator
  identity, the aggregation function, and the uncertainty semantics;
- comparisons are matched: compared conditions differ only in the factor under
  study, or the mismatch is stated in the same sentence as the number;
- never convert absolute endpoint performance into a method effect without a
  matched comparator. "A reaches v" is not "A improves by v" and is not "A
  improves over B";
- a difference measured against a comparator trained, tuned, selected, or
  evaluated differently is labeled a scope check, not an effect estimate;
- a selection rule that used outcome values is never described as held-out,
  prospective, or pre-registered.

### G5. Conceptual And Statistical Precision Gate

Require:

- keep property-of-the-setup claims separate from behavior-of-the-model claims.
  Identifiability, realizability, sufficiency, and well-posedness are properties
  of assumptions, data, and the estimator class. They are not established by a
  model scoring well, and not refuted by a model scoring poorly;
- keep four evidence roles distinct and labeled: diagnostics that localize or
  explain, endpoints that establish the phenomenon, mechanism evidence that
  identifies why, and condition claims that assert necessity or sufficiency;
- a necessary condition is not a sufficient condition. Do not report a satisfied
  necessary condition as if the conclusion followed;
- correlation, ablation, intervention, and proof license different verbs. Match
  the verb to the evidence: observe, is consistent with, localizes, predicts,
  causes, guarantees;
- statistical vocabulary is used in its technical sense. Significant, unbiased,
  robust, converged, calibrated, and stationary require the corresponding test
  or definition, or must be replaced with plain description;
- an assumption the paper cannot check is stated inside the claim, not buried in
  a limitations paragraph.

### G6. Self-Contained Display Gate

A figure or table must survive being read alone, out of order, by a reader who
skipped the body text.

Every caption states:

- the question the display answers;
- what is compared against what;
- protocol and scope: split, condition, budget, and selection rule as far as
  they matter;
- units and axis meaning;
- sample, seed, or episode count and the aggregation, when material;
- uncertainty semantics: what the interval, band, or error bar means;
- one bounded takeaway, plus the reading it does not support when the display is
  easy to over-read.

Require also:

- axis labels name the measured quantity. Evaluation-split, protocol,
  aggregation, normalization, or transformation terms such as `held-out`,
  `cross-task`, or `normalized` may qualify that quantity when needed, but must
  not replace it. When shortening a label, preserve what is measured and move
  procedural detail to the caption;
- symbols, line styles, and abbreviations are decodable from the display and its
  caption, never only from body text;
- no color-only encoding. Pair color with shape, line style, position, direct
  labels, or annotation;
- check every display in grayscale and under a color-vision-deficiency
  simulation before submission;
- type is legible at final printed size, not only when zoomed;
- labels, legends, and annotations do not occlude data;
- comparable panels keep stable axes, panel order, and legend placement;
- every display traces to actual data through a named source artifact or
  generating script. Schematic or illustrative displays are labeled as such.

### G7. Table Semantics Gate

Require:

- one row and one column carry one comparison identity: the same split, budget,
  comparator, and aggregation along that axis;
- do not mix absolute endpoints with matched deltas in the same visual grammar.
  Separate them into different tables, separated blocks, or explicitly labeled
  column groups;
- group rows or columns explicitly whenever splits, budgets, checkpoints, or
  seed sets differ. An unlabeled mixture is a defect even when every number is
  correct;
- no interpretation or status columns such as reading, decision, verdict, or
  claim status. Interpretation belongs in the caption or adjacent prose;
- no table that must be resized below the body text's readable size to fit.
  Split it, transpose it, promote the discriminative subset, or move the full
  grid to the appendix;
- numeric precision is consistent within a column and justified by the
  measurement's uncertainty;
- dense exact-value grids belong in the appendix. Main text keeps the subset
  that changes the reader's decision.

### G8. Public-Manuscript Boundary Gate

The manuscript is a reader-facing argument. The repository is the record.

Keep out of the paper, main text and appendix alike:

- run dates, run identifiers, job names, and internal ticket numbers;
- artifact hashes, file paths, storage locations, and artifact manifests;
- TODO ledgers, remediation history, and decision logs;
- reviewer-response bookkeeping and internal status vocabulary.

An item may appear only when it is genuinely required for reproducibility or
attribution, such as a dataset version, a released code or model identifier, or
a protocol constant a replicator must match. Prefer one reproducibility
subsection or artifact note over scattering such details through result
paragraphs.

The appendix is still reader-facing paper, not storage. Appendix material must
extend evidence, prove or calibrate a claim, document essential protocol, or
support reproduction. Everything else stays in repository evidence and
provenance records, which remain the authoritative location under CairnLab
governance.

### G9. Single-Source-Of-Truth Numbers Gate

Require:

- every quantitative table and figure is generated from, or mechanically checked
  against, a machine-readable artifact. Hand-transcribed numbers are a defect;
- each number appearing in more than one place, including duplicated Markdown
  and LaTeX renderings of the same manuscript, names one source of truth and one
  rerunnable consistency check;
- rounding, units, and normalization conventions are defined once and applied
  everywhere;
- when the source artifact changes, the check fails loudly instead of leaving
  stale numbers in prose;
- if no check exists yet, record the duplication as a known risk in the writing
  ledger rather than asserting consistency.

### G10. Reader-Test Review Pass

Run this pass on the compiled artifact before calling a draft ready. It is a
reading exercise, not a source review.

1. Unfamiliar-reader summary test: someone who has not seen the work reads only
   the title and abstract, then states problem, contribution, and boundary.
   Failing to recover all three is an abstract defect, not a reader defect.
2. Acronym and notation scan: list every acronym, symbol, and named concept in
   order of first appearance, then check single definition, single name, and
   defined before use.
3. Claim-evidence audit: for every claim sentence, name the display or artifact
   that supports it and test it against G4. Mark unsupported sentences blocked.
4. Display-only scan: read figures, tables, and captions in order with the body
   text covered. The evidence chain should still be followable.
5. Compiled-PDF inspection at normal zoom, plus a grayscale pass and a
   color-vision-deficiency pass over every display.
6. Cross-reference and link validation: every section, figure, table, equation,
   citation, and external link resolves to the intended target.

Record each step as pass, warn, or fail in the writing-quality ledger.

### Configurable Gate Thresholds

The gates above are policy. The values below are defaults that a venue,
audience, or host project may override in the writing contract. Record every
override and its rationale in the contract and the ledger.

```yaml
audience_level: newcomer_to_subfield
abstract_max_nonstandard_acronyms: 3
new_symbols_or_acronyms_per_paragraph_max: 3
caveat_repetition_budget: 3
main_text_must_read_evidence_layers_max: 4
uncertainty_reporting: required_when_sampling_changes_interpretation
display_accessibility_checks: [grayscale, color_vision_deficiency, final_size_legibility]
numeric_source_of_truth: required_for_every_quantitative_display
```

Do not hard-code other numeric budgets into prose. If a limit matters, name it
here so it can be reviewed and changed deliberately.

## Reusable Evidence-Pattern Constraints

The constraints below recur across scientific manuscripts, but they are
conditional rather than universal. Apply a subsection only when the named
evidence pattern occurs; do not import its vocabulary or assumptions into an
unrelated paper. Project-specific terminology and thresholds belong in the
writing contract, not in this shared reference.

### A. Claim Framing Constraints

Every paper must contain one explicit claim-frame sentence:

```text
This paper is X, not Y.
```

Examples:

```text
This is a diagnostic study, not a method paper.
The screen enriches plateau members, but does not rank inside plateau.
```

Require:

- the abstract contains both a positive claim and a boundary claim: the
  positive claim states what the contribution is, and the boundary claim states
  what the paper does not claim;
- if results are a plateau or range, avoid point-best selector language;
- use plateau, range, membership, region, and screen for plateau behavior;
- avoid optimal, best selector, and rank inside plateau unless the evidence
  directly supports ranking;
- when evaluation variance can cover small differences, do not write a ranking
  conclusion; write "within plateau" or "not treated as meaningful ordering";
- do not use caveats to hide the main line. If too many caveats are needed, the
  claim must be rewritten instead of padded with disclaimers.
- avoid internal audit phrasing in the abstract and contribution paragraph:
  phrases such as "The claim is", "paper-facing claim", "readouts:",
  "scope decision", "legacy", and "provenance" are review notes, not polished
  abstract prose. Rewrite them as direct contribution, evidence, and boundary
  sentences.
- run an internal diagnostic-engineering prose pass across the whole
  manuscript. If wording sounds like an internal review ledger, project
  management note, or diagnostic engineering document, rewrite it as
  reader-facing science. Treat "readout", "paper-facing", "scope decision",
  "legacy audit", "provenance record", "release package", "manifest",
  "debug", and "we retain" as suspect outside artifact documentation unless
  the term is a standard scientific object that the reader must know.

### A2. Top-Conference Structure And Caption Gate

Run this Top-conference structure pass across the title, abstract,
contribution list, table-of-contents structure,
section/subsection/subsubsection/paragraph headings, figure captions, table
captions, appendix headings, and table labels. The manuscript must read like a top-conference paper, not an internal review plan, remediation log, or engineering record.

Require:

- complete this structure pass before generating new plots, converting tables,
  or adding appendix blocks;
- section, subsection, paragraph headings, figure/table captions, and appendix
  titles must expose the scientific argument or evidence role, not the history
  of review, debugging, or artifact retention;
- rewrite process titles into scientific titles. Examples: "Remediation audit tables" -> "Supplementary diagnostic analyses"; "Bounded unseen-stressor check" -> "Evaluation under bounded held-out stressors"; "Retained-summary fixed-pool top-1 audit" -> "Fixed-pool candidate-stability analysis"; "Full-sweep sample-level fixed-pool event-rate audit" -> "Full-sweep fixed-pool event-rate calibration";
- treat audit, check, remediation, retained, recorded, legacy, provenance,
  manifest, artifact, debug, and we retain as suspect in headings and captions.
  Keep them only when they are standard field terms or essential
  reproducibility terms, and prefer analysis, validation, calibration,
  evaluation, sensitivity, evidence, measured, available summary, or
  supplementary detail when those are the actual scientific roles;
- captions should state the measured object, protocol, aggregation, and
  supported interpretation. They should not describe why a table was added,
  which reviewer concern it addresses, or which old artifact survived cleanup;
- appendix headings must still read as paper sections. Use "Supplementary
  diagnostic analyses", "Additional evaluation details", or "Empirical
  risk calibration" rather than "remediation", "legacy audit",
  "scope decision", or "internal checks";
- organize appendices by reader purpose, such as proofs, experimental protocol,
  additional primary tables, diagnostic validation details, mechanism analyses,
  boundary checks, and reproducibility/source-release details. Do not preserve
  historical artifact order when it hides the reader path;
- after edits, scan the compiled PDF's visible title, heading hierarchy,
  captions, and table of contents if present. If a reader could mistake the
  paper for an internal engineering report or reviewer-response document, the
  structure gate fails.

### B. Terminology Consistency Constraints

One concept gets one primary name.

Require:

- do not use rule, score, selector, screen, and policy interchangeably for the
  same object;
- recommended usage: score is the numeric evaluation; screen returns a set;
  view is a reporting perspective;
- every field-specific abbreviation is expanded at first use unless it is
  universally standard for the target venue;
- keep internal engineering and project-management terms out of the main paper.
  Avoid legacy, provenance, archived, remediation, release package, manifest,
  artifact hash, rendering command, debug, audit ledger, retained-summary,
  recorded artifact, and old path in the main narrative;
- use "readout" only when it is a standard field term or a defined measured
  object. Otherwise prefer the concrete object: projection, metric, signal,
  output, score, planner cost, representation, or diagnostic;
- if artifact history matters, move it to an appendix artifact note rather than
  making it part of the main contribution story.

### C. Experiment Narrative Constraints

Every experiment paragraph should answer one question.

Require:

- separate behavior endpoints, diagnostic measurements, scope checks, and
  selectivity or safety guards instead of piling them into one paragraph;
- order each experiment paragraph as question, protocol, result, boundary;
- do not begin by stacking metrics before the experiment question is clear;
- introduce the theoretical or conceptual object before operational metrics
  when the paper claims theory-aligned evidence;
- when baselines or reference screens are weak, report that transparently and
  narrow the claim. A coarse intervention or single-metric reference should
  narrow the claim instead of supporting a broad dominance statement;
- if a metric is saturated, do not place it in the most prominent table or
  sentence position. For example, if presence hit is saturated, put
  precision/recall before presence;
- replace self-evaluation with facts. Prefer "X changes from a to b; Y remains unchanged" over "this strengthens", "useful", or "compelling".

### D. Caveat Constraints

Caveats should calibrate claims, not become the paper's main structure.

Require:

- the same caveat appears at most three times: abstract, main result, and
  discussion;
- every caveat is bound to a positive claim rather than placed as a standalone
  disclaimer block;
- if caveats exceed the budget, rewrite the claim instead of adding more
  disclaimers;
- do not use caveats to mask missing evidence, weak baselines, or unclear
  contribution type.

### E. Table And Caption Constraints

Tables should make the intended reading hard to miss.

Require:

- the first table column or first metric is the most discriminative metric for
  the method claim;
- saturated metrics do not appear first;
- captions state the metric's intended use. Example: "Precision/recall are the primary measurements; presence is reported for block coverage";
- captions and table titles must not sound like internal issue closure. Replace
  "audit", "check", "remediation", "retained summary", and "recorded
  artifact" wording with reader-facing names for the analysis, validation,
  calibration, evaluation, or evidence when possible;
- every table reading includes how not to read it. Example: "This is not evidence of selector dominance";
- every reference baseline explains what it is and what it is not;
- a coarse high-intensity reference is an intervention-order screen, not a plateau-internal ranker.
- table headers must identify the compared checkpoint, method, condition, or
  evaluation stressor without relying on internal shorthand. Replace ambiguous
  headers such as "baseline stress" with reader-facing labels such as
  "no-noise checkpoint score under blur";
- remove interpretation columns such as "reading", "decision", or "claim
  status" unless they are measured data. Put the interpretation in the caption
  or adjacent prose;
- remove redundant difference columns when the table's purpose is a compact
  scope or boundary check and the prose can state the direction of change.

### F. Appendix Constraints

The appendix extends evidence; it is not a main-text junk drawer.

Require:

- every appendix subsection opens with a reader-facing orientation sentence
  that states how the supplementary material should be used;
- appendix section titles must not expose internal review or remediation state.
  They should name the supplementary evidence role, such as additional
  evaluation details, sensitivity analysis, calibration, or proof;
- appendix material is either evidence expansion, proof/calibration,
  reproducibility support, or scoped audit context. Otherwise remove it;
- artifact provenance, failed attempts, old experiments, old audits, debug logs,
  and reviewer-defense material belong in repository/internal documentation
  unless they are necessary to support the paper-facing claim;
- when non-core appendix material is retained, mark the intended reading and
  mark it as "not paper-facing evidence" when it only provides scoped support
  or provenance.
- short rule: appendix extends evidence; it is not a main-text junk drawer.
- if an appendix table repeats the same rows, columns, and conclusion already
  present in the main text, delete or merge it. A table may remain in the
  appendix only when it adds distinct evidence, protocol detail, calibration,
  or reproducibility support;
- method-family replication, alternative-model rows, or old diagnostic
  artifacts belong in the appendix only when they use the current evidence
  standard for the paper's claim. Single-run, old-metric, unmatched-scope, or
  development-grid artifacts should stay in repository provenance or
  future-work notes.

### Abstract Gate

The abstract is not a mini introduction plus a promise list. It must be a
compressed decision path.

Require:

- one sentence for problem/context, unless the venue explicitly expects more;
- one sentence for the actual contribution type;
- a positive claim and a boundary claim;
- one sentence for the evidence scale and strongest supported result;
- one sentence for the key limitation or scope condition when material;
- no unrelated motivation, implementation history, artifact bookkeeping, or
  reviewer-defense prose;
- no internal claim-audit wording such as "The claim is", "readouts:",
  "paper-facing claim", "scope decision", "legacy", "provenance", "we
  retain", "manifest", or "release package";
- no sentence explains the review process, scope ledger, or diagnostic
  engineering bookkeeping when it should state the scientific object, evidence,
  and boundary directly;
- no unsupported superlatives;
- no more than three nonstandard acronyms unless the venue or field makes them
  unavoidable;
- all numbers trace to current result artifacts.

### Accepted-Abstract Baseline And Reversion Gate

Treat an accepted abstract as a versioned baseline, not as a draft that must be
refreshed whenever the paper gains a supporting experiment. More complete is
not automatically better.

If the user asks to restore, revert, or return to a named version, this is an
exact-restoration task:

1. resolve the requested baseline from version control, released source, or the
   accepted artifact;
2. restore its wording and sentence order verbatim, except for mechanical
   metadata changes that the user explicitly keeps;
3. do not merge in clauses from the rejected candidate, opportunistically
   polish the baseline, or reinterpret the request as a compromise rewrite;
4. verify the restored abstract against the baseline with a source diff and
   record any authorized exception.

Without an explicit reversion request, choose and record `preserve`, `compress`,
`replace`, or `expand`. Default to `preserve` unless verified new evidence
changes the central contribution, the strongest headline result, or a material
boundary needed to interpret that result correctly.

A supporting experiment, factorial analysis, protocol refinement, adjustment,
robustness check, or defensive boundary does not automatically earn abstract
space. Such material usually belongs in the main text or appendix when it
strengthens an existing claim without changing the abstract's decision path.
Adding it can make an abstract worse even when every sentence is accurate: it
can flatten the contribution hierarchy, replace the scientific takeaway with
analysis plumbing, inflate reviewer expectations, and enlarge the claim surface
without adding a new primary contribution.

For a proposed non-reversion edit, compare baseline and candidate side by side.
Check sentence roles, word count, new acronyms, new numbers, protocol terms,
statistical terms, and caveats. Every added or replaced clause must satisfy a
previously unmet abstract obligation. Prefer a length-neutral or shorter
revision; expansion requires a genuinely new primary contribution or material
boundary plus an explicit writing-contract rationale. If the candidate weakens
concision, contribution hierarchy, or first-read comprehension, restore the
accepted baseline.

### Introduction Gate

Require:

- problem importance;
- gap or failure mode;
- why existing work is insufficient;
- the paper's positive contribution, not only what it is not;
- contribution list only when venue-appropriate;
- each contribution paired with evidence or a section where evidence appears;
- no fake novelty from stacking qualifiers around a natural idea.

### Related Work Gate

Require:

- grouped comparison by technical axis, not a citation dump;
- primary sources for closest methods and baselines;
- no placeholder citation for a claim the cited work does not support;
- clear distinction between prior art, baseline, concurrent work, and
  background.

### Methods Or System Gate

Require:

- definitions before formulas and formulas before claims relying on them;
- notation table or glossary when symbols/acronyms are dense;
- algorithm steps tied to reproducible inputs and outputs;
- training, evaluation, and data scope separated;
- no method claim that implies a trained intervention when only diagnostics
  were run.

### Results Gate

Require:

- main result first, secondary diagnostics later;
- tables and figures interpreted in text without duplicating all entries;
- experiment paragraphs follow question, protocol, result, boundary;
- uncertainty and aggregation conventions when sampling can affect the
  conclusion;
- exact winners de-emphasized when differences are within noise;
- every comparison has an appropriate baseline or is labeled as a scope check;
- result labels match evidence strength: main result, sanity check, ablation,
  failure case, audit, or appendix diagnostic.
- metrics that do not close the theory-to-evidence loop are demoted to audits,
  internal provenance, or removed from the paper;
- negative ablations are included only when they rule out a key alternative
  explanation without changing the paper into a broader method-comparison
  paper.

### Discussion And Limitation Gate

Require:

- interpretation tied to the research question;
- limitations that affect conclusions, not generic apologies;
- future work separated from current evidence;
- no repeated claim from the abstract/introduction unless it is now qualified
  by evidence;
- no attempt to repair weak evidence with rhetorical confidence.

## Phase 3: Style And Prose Constraints

Apply these constraints during writing and revision:

- prefer concrete verbs over generic AI-writing verbs such as "delve",
  "leverage" where "use" is more precise, "robustly" without evidence, or
  "comprehensive" without scope;
- keep sentence subjects close to verbs;
- avoid stacked noun phrases when a short clause is clearer;
- split sentences that contain multiple claims, multiple abbreviations, and a
  formula reference at once;
- use active voice when it clarifies agency, except where venue conventions
  prefer passive methods reporting;
- avoid repeating the same thesis sentence in the abstract, introduction,
  conclusion, and limitations;
- use hedging according to evidence strength, not as a blanket tone;
- avoid defensive caveat chains such as "not a method, not a predictor, not a
  guarantee" unless the paper also states the positive contribution.

## Phase 4: Acronym, Notation, And Formula Gate

A dense paper can still be readable, but abbreviation and notation load must be
budgeted.

Require:

- every acronym is defined at first use in abstract and main text unless it is
  universally standard for the target venue;
- abstract has no more than three nonstandard acronyms;
- no paragraph introduces more than three new acronyms or symbols;
- notation is reused consistently across text, equations, figures, and tables;
- formula symbols are introduced before or immediately after the display;
- equation labels are referenced only when the equation is important enough to
  revisit;
- long equations are broken, aligned, or moved to appendix so they fit in the
  compiled PDF;
- theorem, lemma, proposition, and definition names match actual claim
  strength.

## Phase 5: Figures, Tables, Captions, And PDF Layout

Inspect the compiled PDF. Source review is insufficient.

Caption self-containment, display accessibility, and table comparison identity
are defined once in `G6. Self-Contained Display Gate` and `G7. Table Semantics
Gate`. Apply them here rather than restating them.

Require:

- no overfull text or formulas that visibly collide with margins or columns;
- fonts match the venue template and remain readable in figures;
- captions state what is averaged, what units are used, and what conclusion the
  reader should take;
- figures and tables support the argument rather than decorate it;
- figure panels should use consistent axes, panel order, and legend placement
  for comparable conditions; legends and metric annotations must not occlude
  the data;
- omit redundant plot-internal titles when axis labels, panel labels, and the
  caption already identify the condition;
- when a result is defined by paired diagnostics or a guard-plus-primary
  measurement, show both components in the same figure, adjacent panels, or
  immediately adjacent prose;
- prefer figures for trends, robust regions, event rates with uncertainty,
  mechanism ratios, and before/after contrasts; prefer tables for exact lookup
  values, dense threshold grids, and supplementary numeric inventories;
- table precision is justified and consistent;
- appendix displays are demoted unless they answer a necessary reviewer or
  reproducibility question;
- figure text does not contain stale implementation names, old terminology, or
  unexpanded acronyms;
- shortened axis labels still say what is measured; do not introduce a split or
  protocol adjective merely as a shorter substitute for the metric or target;
- line breaks do not separate symbols from definitions in a confusing way;
- every promoted figure/table is generated from an existing artifact or
  reproducible script, not hand-entered numbers;
- source-bundle or release scripts include every figure/table referenced by the
  manuscript when a source bundle is part of the target release;
- final PDF satisfies page, font, margin, anonymization, and reference
  requirements for the target venue.

## Phase 6: Independent Writing Review

After drafting and compiling, run a fresh review that did not write the section.
The reviewer must judge the current artifact, not the improvement delta. Begin
with `G10. Reader-Test Review Pass` on the compiled artifact.

The review output must include:

- recommendation for writing readiness: `ready`, `minor_revision`,
  `major_revision`, or `blocked`;
- score by track if the paper could be submitted to more than one track;
- section-level findings ordered by decision impact;
- writing-only fixes;
- remove/merge/demote candidates;
- artifact-blocked claims;
- formatting and PDF issues;
- citation and source-verification risks;
- whether each writing-contract assertion is satisfied, violated, waived, or
  blocked.

Writing-only work can improve clarity, trust, and fit. It cannot turn missing
evidence into a supported claim.

## Failure Modes This Module Must Catch

The module is incomplete if it does not catch:

- abstract contains irrelevant setup, misses the central result, or reads like
  a status report;
- an explicit request to restore an accepted abstract produces a hybrid,
  polished, or partially preserved candidate instead of an exact restoration;
- an accepted abstract is expanded to inventory supporting experiments,
  protocol details, statistics, or caveats without changing the central
  contribution, strongest headline result, or a material boundary;
- paragraphs repeat the same claim or exist only as reviewer-defense text;
- too many acronyms, symbols, or named diagnostics are introduced together;
- section titles expose artifact bookkeeping instead of the paper's argument;
- formulas are introduced without symbols or overflow in the PDF;
- tables are too wide, overprecise, or disconnected from the main claim;
- figure captions do not explain units, seeds, sample count, or conclusion;
- headings, appendix titles, table names, or captions read like internal review
  expectations, remediation logs, engineering records, or artifact bookkeeping;
- writing uses AI-generic phrasing instead of precise scientific language;
- discussion inflates results beyond evidence strength;
- limitations are long but do not change reader interpretation;
- related work is a citation dump;
- prose sounds like an internal diagnostic engineering document or internal
  review ledger instead of reader-facing scientific prose;
- a newcomer cannot state problem, contribution, and boundary after the title
  and abstract;
- notation or formalism arrives before any concrete instance of what it denotes;
- one concept carries several names, or one name carries several concepts;
- a missing experiment is reported in wording that reads like a negative result;
- an absolute endpoint is presented as a method effect without a matched
  comparator;
- a display cannot be read without the body text, or encodes its comparison in
  color alone;
- a table mixes splits, budgets, or absolute and delta quantities in one
  unlabeled grammar;
- run identifiers, hashes, paths, dates, or remediation history appear in the
  manuscript instead of repository provenance;
- the same number appears in several places with no named source artifact or
  consistency check;
- venue style, page limit, anonymization, or citation format is ignored.

## Interaction With Review And Remediation

Use this module before and during manuscript creation. Then use
`paper-review-remediation-protocol.md` for independent scoring, claim
calibration, citation audit, subtractive remediation, and multi-round
remediation.

If the two disagree, prefer the stricter gate:

- writing module says the draft is unclear: revise before scoring up;
- review protocol says evidence is weak: do not hide that weakness with better
  prose;
- citation audit says source support is missing: block or rewrite the sentence;
- PDF inspection fails: do not call the draft submission-ready.

## Final Writing Quality Ledger

Every substantial writing pass should leave a ledger with:

```yaml
target_venue: ""
track: ""
paper_type: ""
central_thesis: ""
writing_readiness: ready | minor_revision | major_revision | blocked
contract_status: accepted | contested | missing
claim_evidence_matrix_checked: true
abstract_gate: pass | warn | fail
accepted_abstract_baseline_gate: pass | warn | fail | not_applicable
abstract_revision_decision: preserve | compress | replace | expand | not_applicable
abstract_baseline_source: ""
abstract_exact_reversion: true | false | not_applicable
claim_frame_gate: pass | warn | fail
paragraph_gate: pass | warn | fail
terminology_gate: pass | warn | fail
experiment_narrative_gate: pass | warn | fail
acronym_notation_gate: pass | warn | fail
formula_layout_gate: pass | warn | fail
figure_table_caption_gate: pass | warn | fail
claim_compression_gate: pass | warn | fail
theory_metric_mapping_gate: pass | warn | fail
appendix_provenance_split_gate: pass | warn | fail
appendix_reading_gate: pass | warn | fail
pdf_layout_gate: pass | warn | fail
citation_source_gate: pass | warn | fail
progressive_disclosure_gate: pass | warn | fail
concrete_before_abstract_gate: pass | warn | fail
one_job_gate: pass | warn | fail
claim_evidence_identity_gate: pass | warn | fail
conceptual_precision_gate: pass | warn | fail
display_self_containment_gate: pass | warn | fail
table_semantics_gate: pass | warn | fail
public_manuscript_boundary_gate: pass | warn | fail
numeric_source_of_truth_gate: pass | warn | fail
reader_test_gate: pass | warn | fail
threshold_overrides:
  - parameter: ""
    value: ""
    rationale: ""
remove_merge_demote:
  - location: ""
    action: remove | merge | demote | move_to_appendix
    reason: ""
blocked_claims:
  - claim: ""
    missing_artifact: ""
waivers:
  - assertion: ""
    actor: ""
    rationale: ""
```

The ledger is review evidence. It is not a verifier certificate or transition event.
