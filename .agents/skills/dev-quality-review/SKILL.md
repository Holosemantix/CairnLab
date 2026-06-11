---
name: dev-quality-review
description: >
  Use after code changes or before a pull request to review architecture,
  coupling, cohesion, reliability, tests, dependency drift, minimalism, and
  contract compliance.
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
4. Identify public API changes, new dependencies, new abstractions, data
   boundary changes, I/O boundary changes, and test coverage.
5. Run the declared test command and `cairndev check .` when available, unless
   the user only asked for a static review.

## Blocking Criteria

Request changes when any of these are true:

- public behavior changed without deterministic tests;
- module boundaries became less explicit or more circular;
- I/O, business logic, and presentation were mixed without justification;
- a broad manager/orchestrator object was introduced;
- a new dependency was added without a concrete current need;
- an abstraction was added for speculative future use;
- errors became silent, ambiguous, or hard to diagnose;
- the change violates a contract budget and does not justify the violation;
- verification failed and the failure is relevant to the change.

## Review Rubric

Evaluate each area as `pass`, `warn`, or `fail`, using concrete file and line
evidence:

```text
Coupling:
Cohesion:
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
