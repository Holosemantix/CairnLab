---
name: paper-writing-quality
description: Use this skill when drafting, rewriting, polishing, or reviewing scientific manuscripts, LaTeX papers, abstracts, introductions, experiment sections, tables/captions, appendices, or top-conference papers. Applies when Codex must enforce paper-writing constraints for paper-facing claim framing, evidence compression, metric scope control, theory-to-metric mapping, positive and boundary claims, plateau/range language, terminology consistency, acronym expansion, experiment narrative, caveat discipline, table attack-surface review, appendix/provenance separation, citation/source grounding, and compiled-PDF layout checks.
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
- abstract positive claim plus boundary claim;
- plateau/range language instead of point-best selector language;
- terminology consistency: score, screen, view;
- acronym expansion and removal of internal engineering terms from main text;
- experiment paragraphs ordered as question, protocol, result, boundary;
- caveat budget and caveat-to-positive-claim binding;
- table metric ordering, caption purpose, and "how not to read this" sentence;
- table headers that name the checkpoint, method, condition, or evaluation
  stressor without internal shorthand;
- no interpretation-only table columns such as "reading" or "claim status";
- appendix `Reading:` sentences and `not paper-facing evidence` markings;
- deletion or merge of appendix tables that duplicate main-text rows and
  conclusions;
- external-family or old-metric artifacts kept out of the appendix unless they
  meet the current core-evidence standard;
- compiled-PDF checks for formulas, tables, fonts, layout, anonymity, and page
  limits.

## Workflow

1. Collect target venue, track, page limit, anonymity rules, bibliography style,
   source files, compiled PDF, figures, tables, appendix, and result artifacts.
2. Create a writing contract before major drafting:
   - paper type;
   - central thesis;
   - `This paper is X, not Y`;
   - allowed claim strength and disallowed over-claims;
   - positive claim and boundary claim for the abstract;
   - table/caption/appendix constraints.
3. Draft or edit section by section. Keep final manuscript prose flowing except
   where the venue or methods section expects lists.
4. Rebuild or inspect the compiled PDF when LaTeX/PDF artifacts are available.
5. Produce a writing-quality ledger with pass/warn/fail status for the gates in
   the reference checklist.

## Do Not Do

- Do not inflate a claim because the prose is polished.
- Do not use caveats to hide a weak or confused main claim.
- Do not treat reviewer scores, LLM critique, or a clean PDF as scientific
  release authority.
- Do not move main-text clutter into the appendix unless it extends evidence,
  reproducibility, or scoped audit context.

## Output Pattern

For reviews, return:

- writing readiness: `ready`, `minor_revision`, `major_revision`, or `blocked`;
- highest-impact writing failures first;
- concrete edits grouped by claim framing, terminology, experiment narrative,
  caveats, tables/captions, appendix, citations, and layout;
- blocked claims whose source artifacts are missing;
- a short usage note if a host project has stricter claim/evidence authority.
