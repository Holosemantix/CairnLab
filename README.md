# CairnLab

**Reliable paths for AI-assisted research.**

CairnLab is currently a validation-first design package for a possible **Research Claim Kernel**.
It tests whether existing AutoResearch systems need an independent claim lifecycle control layer before committing to a full kernel implementation.

LLMs propose.  
Verifiers decide.  
Provenance records.  
Humans approve.

**No artifact, no claim.**

## AutoResearch Adapters

Built-in manifest adapters can translate external project metadata into CairnLab
claim cases without importing those projects:

```bash
cairn adapter detect path/to/project --json
cairn import-external path/to/project --adapter auto --path .
```

Auto-selection succeeds only when exactly one adapter matches. The adapter layer
does not decide claim release, downgrade, or retraction; it only emits portable
`ClaimCase` objects for the kernel runtime.

## Why this repository exists

The AutoResearch ecosystem is now crowded: ARIS, AutoResearchClaw, ScientistOne, EviBound, DeepScientist, Claw AI Lab, AI Scientist, Paper2Agent, Paper2Code, MLflow/DVC/PROV-based infrastructure, and many benchmarks all cover important parts of autonomous research. CairnLab must therefore **not** become another idea-to-paper agent, research dashboard, or thin integration layer.

CairnLab's candidate contribution is narrower and deeper:

> enforceable scientific claim lifecycle transitions.

A claim cannot move from draft to release just because a model wrote a convincing paper, a reviewer agreed, or a dashboard shows a metric. It moves only when the kernel receives the required evidence items, verifier certificates, governance checks, risk controls, and human responsibility gates.

This contribution is not assumed. The current phase is to validate it by running existing AutoResearch systems on real tasks and recording whether repeated claim/evidence/governance failures appear across systems.

## What CairnLab is

CairnLab is currently:

- a validation program for testing whether AutoResearch needs an independent claim lifecycle control layer;
- a benchmark-lite harness concept that reuses existing public tasks and adds claim/evidence/governance rubrics;
- a decoupled semantic invalidation harness for testing whether invalidated evidence propagates to downstream claim states;
- a set of kernel design documents that should be implemented only if validation shows a repeated cross-system need.

CairnLab is designed to become:

- a claim-first object model;
- a claim lifecycle state machine;
- evidence policies for each claim type;
- verifier certificates as state-transition authority;
- append-only state-transition logs;
- provenance records for runs, artifacts, prompts, patches, and human decisions;
- role and responsibility assignments;
- risk-tiered governance controls;
- dissent-aware multi-agent governance;
- human liability gates before consequential release;
- exportable decision trace packages for review, reproduction, and archival.

## What CairnLab is not

CairnLab is not another autonomous scientist.

It is not a topic-to-paper generator.

It is not a paper-to-code generator.

It is not a generic experiment tracker.

It is not a generic AutoResearch benchmark or public leaderboard.

If the validation phase succeeds, it becomes the kernel that asks, for every research claim:

1. What exactly is being claimed?
2. What evidence is required?
3. What artifacts exist?
4. Which verifier judged them?
5. What state transition was allowed or blocked?
6. What risk tier and lifecycle stage apply?
7. Who is responsible and who is accountable?
8. Can a later reviewer reconstruct the path?

## Repository contents

This repository contains three connected parts:

1. **Validation-first execution plan** under `docs/`.
2. **Candidate CairnLab kernel design documents** under `docs/`.
3. **AutoResearch landscape survey skill** under `skills/autoresearch-landscape-survey/`, preserved and upgraded from v0.2 to support the Claim Kernel positioning.

Current implementation design lives under `docs/design/`:

- `docs/design/SYSTEM_DESIGN.md` for system architecture and dependency direction;
- `docs/design/modules/` for module-level ownership, extension rules, and tests.

Any code change that modifies public APIs, CLI commands, storage layout, adapter
contracts, model fields, or transition semantics should update the matching
design document in the same change.

Current verification scope is tracked in `docs/VERIFICATION_SCOPE.md`.

## Key documents

| Path | Purpose |
| --- | --- |
| `docs/PROJECT_POSITIONING.md` | Strategic positioning and anti-copycat boundary. |
| `docs/VALIDATION_FIRST_EXECUTION_PLAN.md` | Current validation-first execution plan and claim-case archive. |
| `docs/SEMANTIC_INVALIDATION_HARNESS.md` | Reusable claim-state invalidation module for AutoResearch systems. |
| `docs/ADAPTER_CONTRACT.md` | Minimal contract for plugging external AutoResearch systems into CairnLab. |
| `docs/KERNEL_SPEC.md` | Core claim lifecycle kernel primitives. |
| `docs/ARCHITECTURE.md` | System architecture and module boundaries. |
| `docs/GOVERNANCE_ALIGNMENT.md` | Traceability, accountability, role/risk model, and governance mapping. |
| `docs/EU_AI_ACT_PRACTICE_NOTES.md` | Practical governance lessons from EU AI Act-style logging and human oversight requirements. |
| `docs/FILE_UPGRADE_AUDIT.md` | File-by-file keep/modify/add/remove audit from v0.2 to v0.3. |
| `docs/ROADMAP.md` | Development phases and acceptance criteria. |
| `docs/CODEX_BOOTSTRAP_PROMPT.md` | Validation-first bootstrap prompt for Codex. |
| `reports/landscape_report.md` | Regenerated landscape report using the upgraded taxonomy. |
| `data/taxonomy.yaml` | Facet taxonomy upgraded with claim-kernel and governance axes. |
| `data/project_registry.yaml` | Landscape inventory, including newly added high-risk adjacent work. |

## Core loop

Current validation loop:

```text
existing AutoResearch system
  -> controlled benchmark or manifest-style task
  -> material claim extraction
  -> claim/evidence/artifact/verdict/human/dissent record
  -> repeated failure taxonomy
  -> Go / No-Go decision for CairnLab kernel
```

Semantic invalidation harness loop:

```text
external AutoResearch metadata
  -> claims / evidence / relations / verifier outputs / human gates
  -> dependency graph
  -> plan invalidation effects
  -> append transition events
  -> recompute projected claim authority
```

Candidate kernel loop:

```text
claim
  -> lifecycle context
  -> risk assessment
  -> responsibility assignment
  -> evidence policy
  -> evidence item
  -> verifier certificate
  -> claim state transition
  -> governance check
  -> human gate
  -> decision trace package
```

## Example CLI direction

0.3 semantic invalidation harness:

```bash
cairn init
cairn import-case examples/cases/case_wrong_metric.yaml
cairn validate
cairn trace claim:C1
cairn revert run:exp_007 --reason "metric computed on wrong split" --plan-only
cairn revert run:exp_007 --reason "metric computed on wrong split" --apply --actor user:alice
cairn decision-trace claim:C1 --transition release --json
```

Candidate full-kernel direction:

```bash
cairn init
cairn claim add "Method A improves baseline B by 2.3% on Dataset X"
cairn evidence attach C1 --run runs/exp-001 --artifact metrics.json
cairn verify C1 --policy empirical-score-v1
cairn gate request C1 --type human-release
cairn report C1
```

## Landscape survey maintenance

The original v0.2 landscape skill is preserved. To review a new project:

```bash
python skills/autoresearch-landscape-survey/scripts/check_new_project.py --input path/to/intake.yaml --format markdown
python skills/autoresearch-landscape-survey/scripts/classify_project_from_yaml.py --input path/to/intake.yaml --format yaml
python skills/autoresearch-landscape-survey/scripts/prepare_ai_project_review.py --input path/to/intake.yaml
python skills/autoresearch-landscape-survey/scripts/render_project_overview.py
python skills/autoresearch-landscape-survey/scripts/render_landscape_report.py
```
