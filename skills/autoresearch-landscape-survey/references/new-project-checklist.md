# New Project Checklist

Use this checklist when adding a new AutoResearch-related project.

## 1. Source Reading Checklist

Do not rely only on the repository description. Inspect:

- README / README_CN;
- paper or technical report;
- `docs/`, `examples/`, `tests/`, `benchmarks/`;
- `skills/`, `.claude/skills/`, `prompts/`, `agents/`;
- pipeline/orchestrator code;
- artifact formats and sample run outputs;
- config schema and environment requirements;
- security warnings, API keys, network use, sandbox policy;
- license and data/model dependency claims.

## 2. Distinguish Claims from Evidence

For each advertised ability, classify it as:

| Level | Meaning |
| --- | --- |
| advertised | Only claimed in README or paper. |
| documented | Has docs or protocol but no runnable evidence inspected. |
| runnable | Has code/examples that appear executable. |
| tested | Has tests, CI, benchmark, or reproduction scripts. |
| externally_validated | Has independent benchmark, paper review, or third-party replication. |

## 3. Fill Facets

Record facets under:

- research fields;
- starting points;
- outputs;
- workflow scopes;
- execution depth;
- verification models;
- accountability features;
- agent topology;
- integration style;
- maturity signals;
- risk flags;
- fit to our target.

## 4. Compare to Existing Projects

Run:

```bash
python skills/autoresearch-landscape-survey/scripts/check_new_project.py   --input path/to/intake.yaml --format markdown
```

Then ask:

- Is this already in the registry under an alias or fork?
- Which 3-5 existing projects are closest?
- What is genuinely new?
- Does it add a new facet?
- Does it change our architecture assumptions?

## 5. Decide Depth

Deep dive if the project is highly relevant to:

- paper-to-code or paper-to-reproduction;
- claim ledger / run ledger / provenance graph;
- multi-agent producer/verifier/judge separation;
- review-finding-to-issue/rerun closure;
- fresh-container reproduction;
- evidence-bound audit or chain-of-evidence.

Otherwise add a brief card or watchlist entry.


## v0.3 CairnLab Kernel Checklist

Before adding a new project, check whether it includes any of the following:

- claim lifecycle state machine;
- evidence policy DSL;
- verifier-issued transition certificates;
- append-only state transition log;
- risk-tiered controls;
- lifecycle-context-aware governance;
- responsibility assignment / RACI-like model;
- human liability gates;
- decision trace package;
- external kernel/API that other research agents can call.

If yes, treat it as a potential core competitor or protocol competitor, not a generic adjacent project.
