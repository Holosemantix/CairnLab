---
name: paper-writing-quality
description: Use this skill when drafting, rewriting, polishing, or reviewing scientific manuscripts, LaTeX papers, abstracts, introductions, experiment sections, tables/captions, appendices, section headings, table-of-contents structure, or top-conference papers. Applies when Codex must enforce paper-writing constraints for paper-facing claim framing, evidence compression, metric scope control, theory-to-metric mapping, positive and boundary claims, plateau/range language, terminology consistency, acronym expansion, experiment narrative, caveat discipline, table attack-surface review, reader-facing section/caption structure, appendix/provenance separation, citation/source grounding, compiled-PDF layout checks, progressive disclosure and layered readability, concrete-before-abstract exposition, claim/evidence identity, display self-containment and accessibility, table comparison semantics, public-manuscript versus internal-record separation, single-source-of-truth numbers, and a final reader-test pass.
---

# Paper Writing Quality

## Core Rule

Use this skill as a writing-quality gate, not as scientific authority.

LLM writers propose text. Evidence artifacts, verifier certificates, provenance
records, and human approval decide scientific claims when a host project has
that governance model.

## Required Reference

Before drafting, rewriting, polishing, or reviewing a paper, read:

```text
references/writing-quality-checklist.md
```

Use it as the checklist for:

- claim framing: `This paper is X, not Y`;
- evidence compression: main text and appendix only keep material needed for
  the paper-facing claim;
- theory-to-metric mapping and metric scope control;
- structure-first remediation: fix the title, abstract, section order,
  section jobs, and main/appendix split before adding or converting displays;
- theory-to-evidence mapping: connect theoretical quantities to empirical
  audits, observed results, and explicit limitations;
- abstract positive claim plus boundary claim;
- plateau/range language instead of point-best selector language;
- terminology consistency: score, screen, view;
- acronym expansion and removal of internal engineering terms from main text;
- internal diagnostic-engineering prose rewritten as reader-facing science;
- top-conference structure pass: section, subsection, paragraph headings,
  figure/table captions, appendix titles, and table-of-contents entries must
  read as scientific argument, not internal review, remediation, or engineering
  records;
- experiment paragraphs ordered as question, protocol, result, boundary;
- caveat budget and caveat-to-positive-claim binding;
- table metric ordering, caption purpose, and "how not to read this" sentence;
- figure/table conversion rules: use figures for trends, regions, uncertainty,
  mechanisms, or before/after contrasts; keep dense exact values and
  sensitivity grids in tables or appendix;
- display provenance: every promoted figure/table must trace to an artifact or
  reproducible script, and source bundles must include referenced displays;
- table headers that name the checkpoint, method, condition, or evaluation
  stressor without internal shorthand;
- no interpretation-only table columns such as "reading" or "claim status";
- reader-facing appendix orientation sentences and `not paper-facing evidence` markings when scoped provenance must remain visible;
- deletion or merge of appendix tables that duplicate main-text rows and
  conclusions;
- external-family or old-metric artifacts kept out of the appendix unless they
  meet the current core-evidence standard;
- compiled-PDF checks for formulas, tables, fonts, layout, anonymity, and page
  limits.

## Generic Manuscript Gates

The checklist defines ten venue- and domain-independent gates. Cite them by name
in plans, reviews, and ledgers; read the checklist for the operational detail.

- `G1. Progressive Disclosure Gate`: title plus abstract gives problem,
  contribution, and boundary; intro plus figures gives the argument; details
  come later, and completeness comes from artifacts, not transcribed logs.
- `G2. Concrete-Before-Abstract Gate`: a plain-language instance before the
  notation, every nonstandard term defined once, one concept one stable name.
- `G3. One-Job Gate`: one job per section, paragraph, and display; topic
  sentence first; claim, evidence, boundary order; no chronology or
  reviewer-response prose.
- `G4. Claim-Evidence Identity Gate`: separate not-run, unavailable,
  inconclusive, and failed; identify split, seeds versus episodes, budget,
  comparator, aggregation, and uncertainty; no method effect from an absolute
  endpoint without a matched comparator.
- `G5. Conceptual And Statistical Precision Gate`: identifiability and related
  properties belong to assumptions and data, not to observed model scores;
  separate diagnostics, endpoints, mechanism evidence, and necessary or
  sufficient conditions; match verbs to evidence.
- `G6. Self-Contained Display Gate`: captions carry question, comparison,
  protocol, units, sample or aggregation, uncertainty, and one bounded takeaway;
  no color-only encoding; grayscale, color-vision, and final-size checks.
- `G7. Table Semantics Gate`: one comparison identity per row and column; never
  mix absolute endpoints with matched deltas in one grammar; no resize-to-fit
  tables and no interpretation or status columns.
- `G8. Public-Manuscript Boundary Gate`: dates, run IDs, hashes, manifests,
  ledgers, and decision logs stay in repository evidence and provenance; the
  appendix is reader-facing paper, not storage.
- `G9. Single-Source-Of-Truth Numbers Gate`: quantitative displays are generated
  or checked against machine-readable artifacts, and duplicated numbers name a
  source plus a rerunnable consistency check.
- `G10. Reader-Test Review Pass`: unfamiliar-reader summary test, acronym and
  notation scan, claim-evidence audit, display-only scan, compiled PDF at normal
  zoom with grayscale and color-vision inspection, and cross-reference and link
  validation.

Numeric limits are not policy. They live in the checklist's
`Configurable Gate Thresholds` block and may be overridden per venue or audience
in the writing contract, with the override recorded in the ledger.

## Workflow

1. Collect target venue, track, page limit, anonymity rules, bibliography style,
   source files, compiled PDF, figures, tables, appendix, and result artifacts.
2. Create a writing contract before major drafting:
   - paper type;
   - central thesis;
   - `This paper is X, not Y`;
   - allowed claim strength and disallowed over-claims;
   - positive claim and boundary claim for the abstract;
   - structure-first section order, display plan, and table/caption/appendix
     constraints;
   - any override of the configurable gate thresholds, with rationale.
3. Draft or edit section by section. Keep final manuscript prose flowing except
   where the venue or methods section expects lists.
4. Rebuild or inspect the compiled PDF when LaTeX/PDF artifacts are available.
5. Run `G10. Reader-Test Review Pass` on the compiled artifact.
6. Produce a writing-quality ledger with pass/warn/fail status for the gates in
   the reference checklist, including `G1` through `G10`.

## Do Not Do

- Do not inflate a claim because the prose is polished.
- Do not use caveats to hide a weak or confused main claim.
- Do not treat reviewer scores, LLM critique, or a clean PDF as scientific
  release authority.
- Do not move main-text clutter into the appendix unless it extends evidence,
  reproducibility, or scoped audit context.
- Do not report an absolute endpoint as a method effect without a matched
  comparator.
- Do not describe a missing experiment in wording that reads like a result.
- Do not put run identifiers, hashes, paths, dates, or remediation history in
  the manuscript when repository provenance is the right home.

## Output Pattern

For reviews, return:

- writing readiness: `ready`, `minor_revision`, `major_revision`, or `blocked`;
- highest-impact writing failures first;
- concrete edits grouped by claim framing, terminology, experiment narrative,
  caveats, tables/captions, appendix, citations, and layout;
- pass/warn/fail for `G1` through `G10` with the failing evidence named;
- blocked claims whose source artifacts are missing;
- a short usage note if a host project has stricter claim/evidence authority.
