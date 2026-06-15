---
name: dev-quality-review
description: >
  Use after CairnLab code changes or before a pull request to review
  architecture, coupling, cohesion, reliability, tests, dependency drift,
  minimalism, and contract compliance.
---

# Dev Quality Review Skill

## Purpose

Review a change against the repository's design contract. Treat review as a
blocking quality gate when the change violates reliability, coupling, API, or
test expectations.

## Required Discovery

1. Read `AGENTS.md`.
2. Read `.cairndev/contract.yaml` if present.
3. Inspect the changed files and relevant surrounding code.
4. For CairnLab code or development changes, inspect `docs/design/README.md`
   and affected module design docs under `docs/design/modules/`.
5. If third-party AutoResearch systems are invoked or adapted, inspect the
   adapter-system boundary and confirm their outputs remain external,
   non-authoritative artifact/evidence inputs.
6. Identify public API changes, new dependencies, new abstractions, data
   boundary changes, I/O boundary changes, design-doc obligations, and test
   coverage.
7. Run the declared test command and `cairndev check .` when available, unless
   the user only asked for a static review.

## Blocking Criteria

Request changes when any of these are true:

- public behavior changed without deterministic tests;
- module boundaries became less explicit or more circular;
- I/O, business logic, and presentation were mixed without justification;
- a broad manager/orchestrator object was introduced;
- a new dependency was added without a concrete current need;
- an abstraction was added for speculative future use;
- CairnLab's current Research Claim Kernel requirements were weakened,
  including authority-only transitions, append-only state, thin CLI/adapters,
  governance gates, provenance, or material-dissent handling;
- third-party AutoResearch integration can mutate host state, import host
  runtime packages into CairnLab, use external outputs as authority, or bypass
  CairnLab evidence, verifier, governance, provenance, or transition-event
  requirements;
- public APIs, CLI behavior, schemas, storage layout, adapter contracts,
  third-party AutoResearch calls, or transition semantics changed without
  matching design-doc updates;
- errors became silent, ambiguous, or hard to diagnose;
- the change violates a contract budget and does not justify the violation;
- verification failed and the failure is relevant to the change.

## Review Rubric

Evaluate each area as `pass`, `warn`, or `fail`, using concrete file and line
evidence:

```text
Coupling:
Cohesion:
CairnLab requirements:
Minimalism:
Reliability:
Testability:
Extensibility:
Dependency discipline:
Observability:
```

## Output Format

Lead with findings, ordered by severity:

```text
Blocking findings:
- ...

Non-blocking findings:
- ...

Open questions:
- ...

Verification:
- ...

Design summary:
- ...
```

If there are no blocking findings, say so clearly and still mention residual
test or design risk.
