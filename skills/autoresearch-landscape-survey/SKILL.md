---
name: autoresearch-landscape-survey
description: Use this skill to maintain a multi-facet survey map of AutoResearch, AI scientist, paper-to-code, reproducibility, evidence/provenance, and multi-agent scientific discovery projects, and to apply reusable research-paper review/remediation protocols without turning reviewer prose into claim authority.
license: Apache-2.0
allowed-tools: Read Write Edit Bash WebSearch WebFetch
compatibility: Requires Python 3.10+ and PyYAML. Web/GitHub access is recommended for new-project deep dives.
metadata:
  version: "0.2.1"
  skill-author: Holosemantix research mapping workflow
---

# AutoResearch Landscape Survey Skill

## When to Use

Use this skill when the task is to survey, compare, classify, or update knowledge about AutoResearch-related systems, including:

- AI Scientist / idea-to-paper systems;
- ARIS / AutoResearchClaw / skill-native research harnesses;
- paper-to-code and paper reproduction benchmarks;
- multi-agent scientific discovery systems such as Robin;
- evidence, provenance, claim audit, and review-integrity systems;
- execution infrastructure such as MLflow, DVC, W&B, ReproZip, RO-Crate, Snakemake, and Nextflow.

Also use this skill when maintaining CairnLab-facing research-paper review,
writing, citation-audit, or remediation workflows. In that mode, read
`references/paper-review-remediation-protocol.md` before judging or editing a
paper. Reviewer notes, LLM critiques, and manuscript scores remain
non-authoritative evidence; they do not decide CairnLab claim lifecycle
transitions.

## Core Principle

Do **not** force projects into one orthogonal layer. A project can simultaneously be:

- an AI algorithm research system;
- a skill-native workflow;
- an idea-to-paper pipeline;
- a multi-agent debate system;
- a sandbox execution harness;
- a weak or strong accountability system.

Describe the project by **facets**. If a new project introduces a new feature not covered by the taxonomy, record it as a proposed new facet instead of dropping it.

## Main Files

| Path | Purpose |
| --- | --- |
| `data/taxonomy.yaml` | Multi-facet attribute taxonomy and open-vocabulary rules. |
| `data/project_registry.yaml` | Existing surveyed projects, including facets, legacy tags, sources, gaps, and deep-dive links. |
| `reports/project_overview.md` | Continuously updated overall project overview. |
| `reports/landscape_report.md` | Longer generated landscape report. |
| `references/project-map.md` | Human-readable project map and maintenance notes. |
| `references/paper-review-remediation-protocol.md` | Top-conference paper writing, review, citation-audit, and multi-round remediation protocol. |
| `references/deep-dives/` | Detailed analysis for highly relevant projects. |
| `scripts/check_new_project.py` | Check if a new project has already been surveyed and find closest projects. |
| `scripts/classify_project_from_yaml.py` | Infer likely facets from a project intake YAML. |
| `scripts/prepare_ai_project_review.py` | Generate a prompt bundle for an AI colleague to deeply read a new GitHub repo and produce analysis. |
| `scripts/prepare_ai_report_prompt.py` | Generate a prompt for AI-assisted landscape report writing from structured data. |
| `scripts/render_project_overview.py` | Render the overall project overview Markdown. |
| `scripts/render_landscape_report.py` | Render the landscape report Markdown. |
| `prompts/` | Reusable prompts for AI analysts and report writers. |

## Recommended New-Project Workflow

1. Create an intake file from `assets/templates/new_project_intake.yaml`.
2. Run a quick duplicate and similarity check:

```bash
python skills/autoresearch-landscape-survey/scripts/check_new_project.py   --input path/to/new_project.yaml --format markdown
```

3. Infer likely facets from the YAML:

```bash
python skills/autoresearch-landscape-survey/scripts/classify_project_from_yaml.py   --input path/to/new_project.yaml --format yaml   --output reports/intake/new_project_facets.yaml
```

4. Prepare an AI colleague prompt bundle for repository reading:

```bash
python skills/autoresearch-landscape-survey/scripts/prepare_ai_project_review.py   --input path/to/new_project.yaml   --output reports/intake/new_project_ai_review_prompt.md
```

5. Ask the AI colleague to read the project repository and return the requested structured report.
6. Update `data/project_registry.yaml` with verified attributes, sources, gaps, and `custom_facets` if needed.
7. Regenerate overview reports:

```bash
python skills/autoresearch-landscape-survey/scripts/render_project_overview.py
python skills/autoresearch-landscape-survey/scripts/render_landscape_report.py
```

## New-Project Analysis Rules

- Read the repo, not just the README. Inspect docs, examples, tests, pipeline code, skill folders, agent prompts, config files, artifact formats, and sample runs.
- Separate advertised capability from demonstrated capability.
- Record starting point, outputs, workflow scope, execution depth, verification model, accountability features, agent topology, integration style, maturity signals, and risks.
- Always compare with the closest existing projects rather than with an abstract category.
- For our target, the most important gap is usually whether the system has: `claim_ledger`, `experiment_spec`, `run_ledger`, `artifact_lineage`, `provenance_graph`, `judge_panel`, and `review_issue_loop`.
- If the project adds a new attribute, propose a new facet with rationale and source evidence.

## Detailed Analysis Policy

Create or update a deep dive when a project is highly relevant to CairnLab Research Claim Kernel, especially if it has any of the following:

- claim-level evidence or chain-of-evidence;
- paper-to-code or paper-to-reproduction execution;
- sandbox/fresh-container reproduction;
- multi-agent producer/verifier/judge separation;
- human-in-the-loop gates;
- review finding to issue/patch/rerun closure;
- provenance graph, attestation protocol, or run ledger.

For low-relevance projects, use a brief card but still record facets and sources.

## Paper Review and Remediation Policy

When the task is to improve or review a research paper, run the manuscript as a
fresh submission through `references/paper-review-remediation-protocol.md`. The
protocol is intentionally broader than numerical consistency: it covers writing
quality, contribution framing, claim strength, theory-to-experiment alignment,
figures/tables/formulas, sampling variance, reference auditing, and independent
multi-round scoring. Do not inflate scores because earlier rounds improved the
paper. Do not over-fit the manuscript into a pile of defensive caveats. Every
round must judge the current artifact on its own evidence. For top-conference
remediation, continue rounds until the current artifact honestly reaches the
`strong_accept_baseline`, the user pauses or narrows scope, or the only
decision-changing blockers require retraining, new large-scale experiments,
private author metadata, or other user authorization outside the current task.


## v0.3 CairnLab Alignment

This skill now supports competitive intelligence for CairnLab. The key question is no longer simply whether a project supports accountable research CI. The key question is whether it defines or threatens CairnLab's claim lifecycle kernel:

- claim state machine;
- evidence policy;
- verifier certificate;
- append-only transition ledger;
- governance/risk/responsibility model;
- human liability gate;
- decision trace package.
