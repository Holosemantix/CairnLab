---
name: dev-quality-control
description: >
  Use before and during implementation when architecture quality, low coupling,
  extensibility, reusability, reliability, minimal code, design review, or
  Codex task planning matters.
---

# Dev Quality Control Skill

## Purpose

Make the repository's design contract executable during implementation.
Do not rely on chat history or memory. Treat `AGENTS.md` and
`.cairndev/contract.yaml` as the source of truth for the current project.

## Required Discovery

Before editing non-trivial code:

1. Read `AGENTS.md`.
2. Read `.cairndev/contract.yaml` if present.
3. Inspect the local code paths that the task may touch.
4. Identify affected modules, public APIs, data boundaries, I/O boundaries,
   runtime dependencies, and expected test surface.
5. If the task changes architecture, extension points, or cross-module
   ownership, inspect existing ADRs before deciding.

## Implementation Plan

Before editing, provide a concise plan:

```text
Plan:
1. ...
2. ...

Design constraints:
- ...

Smallest viable change:
- ...

Tests:
- ...

Risks:
- ...
```

The plan must explicitly state whether the change adds or changes public
behavior, introduces an abstraction, adds a dependency, or crosses a module
boundary.

## Implementation Rules

- Prefer narrow functions, explicit data contracts, and clear module ownership.
- Keep I/O, domain logic, presentation, and orchestration separate.
- Do not introduce broad manager/orchestrator classes.
- Do not add runtime dependencies without a current, concrete need and explicit
  justification.
- Do not add an abstraction unless it removes real complexity or serves at
  least one current use.
- Preserve existing public APIs unless the task requires a breaking change.
- Cover public behavior changes with deterministic tests.
- Use explicit errors and actionable diagnostics instead of silent fallback.
- Keep edits scoped to the smallest reversible change that satisfies the task.
- If existing code violates the contract, fix only the part needed for the task
  unless the user asks for a broader refactor.

## Decision Gates

Stop and revise the plan, or ask the user when needed, if:

- the implementation would mix responsibilities across module boundaries;
- public behavior would change without tests;
- a new abstraction layer would be added without an ADR when the contract
  requires one;
- a new dependency is avoidable;
- the requested change conflicts with `AGENTS.md` or `.cairndev/contract.yaml`;
- verification cannot be run and the residual risk is material.

## Verification

After editing:

1. Run the project's declared test command when available.
2. Run `cairndev check .` when available.
3. If the `cairndev` executable is not installed, use the documented local
   module entry point when this repository provides one.
4. Fix failures or report a concrete reason they remain.

## Final Report

Summarize:

- files changed;
- design impact;
- tests and checks run;
- CairnDev findings or contract violations;
- dependency changes;
- ADR changes or why none were needed;
- remaining risks.
