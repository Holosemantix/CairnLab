# Codex Bootstrap Prompt

You are helping maintain CairnLab in its validation-first phase.

Project:

- Repo: `cairnlab`
- Package: `cairnlab`
- CLI: `cairn`
- Tagline: `Reliable paths for AI-assisted research.`

CairnLab is not currently implementing a full AutoResearch system or a full claim lifecycle kernel. The current task is to validate whether existing AutoResearch systems need an independent claim lifecycle control layer.

## Current Strategic Rule

Do not assume that the kernel is already justified.

Use existing AutoResearch systems as failure-sampling instruments. Build the transition engine only if repeated cross-system failures show that in-system evidence chains, reviewer loops, and governance gates are insufficient for claim release control.

## Core Principles Under Test

No artifact, no claim.

LLMs propose. Verifiers decide. Provenance records. Humans approve.

These principles should currently be used as validation rubrics, not as evidence that the kernel must already be built.

## First Validation Target

Start with AutoResearchClaw.

Recommended sequence:

1. Run a documented smoke test or a lightweight ARC-Bench task.
2. Confirm setup, model calls, sandbox execution, logs, artifacts, and final report generation.
3. Run two or three controlled ARC-Bench tasks.
4. Extract five to ten material claims per task.
5. Record claim/evidence/artifact/verdict/human/dissent information using `docs/VALIDATION_FIRST_EXECUTION_PLAN.md`.
6. Only after controlled benchmark tasks run, convert a custom research topic into a manifest-style task and run it.

## Validation Artifacts

Maintain:

- claim case records;
- failure taxonomy labels;
- task manifests;
- run and artifact references;
- a Go / No-Go report.

Do not build a public benchmark or leaderboard in this phase.

## Do Not Implement Yet

- Full autonomous research pipeline.
- Paper generator.
- Research OS dashboard.
- Full claim transition engine.
- MLflow / DVC / PROV integration layer.
- Large benchmark suite or public leaderboard.

## Allowed 0.3 Work

It is acceptable to build a small semantic invalidation harness before the full kernel gate if it remains a validation instrument and reusable module.

Allowed commands:

```bash
cairn init
cairn import-case examples/cases/case_wrong_metric.yaml
cairn validate
cairn trace claim:C1
cairn affected run:exp_007
cairn revert run:exp_007 --reason "metric computed on wrong split" --plan-only
cairn revert run:exp_007 --reason "metric computed on wrong split" --apply --actor user:alice
```

Implementation rule: keep `models`, `store`, `graph`, `projection`, `planner`, `validation`, and `cli` loosely coupled. The invalidation planner should be usable by other AutoResearch projects without adopting CairnLab's CLI or storage layout.

## Kernel Implementation Gate

Proceed to a transition engine MVP only if all conditions below are met:

1. At least three systems or workflows are sampled.
2. At least six real tasks are run or replayed.
3. At least thirty material claims are recorded.
4. At least three failure classes recur across systems.
5. At least one recurring failure class directly affects claim release control.
6. Existing benchmark metrics do not already capture the failure.
7. The failure can plausibly be reduced by an external claim transition authority.

If the gate is met, the kernel MVP should implement:

- Claim
- EvidencePolicy
- EvidenceItem
- VerifierCertificate
- StateTransition
- HumanGate
- LifecycleContext
- RiskAssessment
- ResponsibilityAssignment
- DecisionTracePackage

Required kernel behavior after the gate:

- A claim cannot become `verified` without required verifier certificates.
- A claim cannot become `released` without required human gate and accountable party.
- A failed verifier blocks transition.
- Material dissent blocks release unless resolved by verifier or explicit human override.
- Human gates record actor, authority, scope, and rationale.
- State is derived from append-only events, not mutable fields.
