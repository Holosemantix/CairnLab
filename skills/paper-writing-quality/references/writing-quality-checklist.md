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

The plan must include:

- central thesis;
- reader promise: what the reader should know after the paper;
- claim/evidence matrix;
- section map with each section's single job;
- figure/table map with the question each display answers;
- appendix demotion plan for extra audits, ledgers, diagnostic families, and
  reproducibility material;
- citation scaffold with primary sources for novelty, methods, baselines, and
  theory;
- known blockers that require new experiments, retraining, source lookup, or
  human metadata.

Reject plans that are a list of topics without an argument.

## Phase 2: Section Drafting Standards

Draft section by section from the plan. Final manuscript text must be flowing
prose except where the venue or method section explicitly expects lists.

Every section must satisfy:

- the section title matches the section's actual job;
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

## Project-Specific Writing Constraints

These constraints come from CairnLab paper-writing experience and should be
treated as default checklist items for future papers.

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

### B. Terminology Consistency Constraints

One concept gets one primary name.

Require:

- do not use rule, score, selector, screen, and policy interchangeably for the
  same object;
- recommended usage: score is the numeric evaluation; screen returns a set;
  view is a reporting perspective;
- every abbreviation is expanded at first use, including PCC, CRA, MAF, and
  ACPC-H/trans when they appear;
- keep internal engineering terms out of the main paper. Avoid legacy,
  provenance, archived, remediation, and old path in the main narrative;
- if artifact history matters, move it to an appendix artifact note rather than
  making it part of the main contribution story.

### C. Experiment Narrative Constraints

Every experiment paragraph should answer one question.

Require:

- separate behavioral recovery, plateau membership, planner-side sensitivity,
  and selectivity guard instead of piling them into one paragraph;
- order each experiment paragraph as question, protocol, result, boundary;
- do not begin by stacking metrics before the experiment question is clear;
- when baselines are weak, report that transparently and narrow the claim. A
  high-std or MAF-only reference should narrow the claim instead of supporting a
  broad dominance statement;
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
- captions state the metric's intended use. Example: "Precision/recall are the primary readouts; presence is reported for block coverage";
- every table reading includes how not to read it. Example: "This is not evidence of selector dominance";
- every reference baseline explains what it is and what it is not;
- the high-std reference is a coarse intervention-order screen, not a plateau-internal ranker.

### F. Appendix Constraints

The appendix extends evidence; it is not a main-text junk drawer.

Require:

- every appendix subsection has a sentence beginning with "Reading:";
- artifact provenance can be preserved in the appendix, but provenance should
  not drive the main narrative;
- old experiments and old audits may be retained only when marked "not paper-facing evidence";
- appendix material is either evidence expansion, reproducibility support, or
  scoped audit context. Otherwise remove it.
- short rule: appendix extends evidence; it is not a main-text junk drawer.

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
- no unsupported superlatives;
- no more than three nonstandard acronyms unless the venue or field makes them
  unavoidable;
- all numbers trace to current result artifacts.

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

Require:

- no overfull text or formulas that visibly collide with margins or columns;
- fonts match the venue template and remain readable in figures;
- captions state what is averaged, what units are used, and what conclusion the
  reader should take;
- figures and tables support the argument rather than decorate it;
- table precision is justified and consistent;
- appendix displays are demoted unless they answer a necessary reviewer or
  reproducibility question;
- figure text does not contain stale implementation names, old terminology, or
  unexpanded acronyms;
- line breaks do not separate symbols from definitions in a confusing way;
- final PDF satisfies page, font, margin, anonymization, and reference
  requirements for the target venue.

## Phase 6: Independent Writing Review

After drafting and compiling, run a fresh review that did not write the section.
The reviewer must judge the current artifact, not the improvement delta.

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
- paragraphs repeat the same claim or exist only as reviewer-defense text;
- too many acronyms, symbols, or named diagnostics are introduced together;
- section titles expose artifact bookkeeping instead of the paper's argument;
- formulas are introduced without symbols or overflow in the PDF;
- tables are too wide, overprecise, or disconnected from the main claim;
- figure captions do not explain units, seeds, sample count, or conclusion;
- writing uses AI-generic phrasing instead of precise scientific language;
- discussion inflates results beyond evidence strength;
- limitations are long but do not change reader interpretation;
- related work is a citation dump;
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
claim_frame_gate: pass | warn | fail
paragraph_gate: pass | warn | fail
terminology_gate: pass | warn | fail
experiment_narrative_gate: pass | warn | fail
acronym_notation_gate: pass | warn | fail
formula_layout_gate: pass | warn | fail
figure_table_caption_gate: pass | warn | fail
appendix_reading_gate: pass | warn | fail
pdf_layout_gate: pass | warn | fail
citation_source_gate: pass | warn | fail
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
