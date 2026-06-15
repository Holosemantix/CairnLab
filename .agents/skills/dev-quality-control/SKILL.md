---
name: dev-quality-control
description: >
  Use before and during implementation when CairnLab code changes, architecture
  quality, low coupling, extensibility, reusability, reliability, minimal code,
  design review, or Codex task planning matters.
---

# Dev Quality Control Skill

## Purpose

Make the repository's design contract executable during implementation.
Do not rely on chat history or memory. Treat `AGENTS.md` and
`.cairndev/contract.yaml` as the source of truth for the current project.

## Required Discovery

Before editing non-trivial code or development workflow:

1. Read `AGENTS.md`.
2. Read `.cairndev/contract.yaml` if present.
3. For CairnLab code or development changes, read `docs/design/README.md` and
   the affected module design document under `docs/design/modules/` before
   editing.
4. Inspect the local code paths that the task may touch.
5. Identify affected modules, public APIs, data boundaries, I/O boundaries,
   runtime dependencies, and expected test surface.
6. Identify the CairnLab-specific requirement being protected, such as
   authority-only transitions, append-only authority state, thin CLI/adapters,
   governance gates, provenance, or material-dissent handling.
7. If the task invokes, wraps, imports from, or adapts a third-party
   AutoResearch system, identify the exact artifact/evidence boundary and how
   the external output remains non-authoritative inside CairnLab.
8. If the task changes architecture, extension points, or cross-module
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
- For CairnLab code or development changes, preserve the Research Claim Kernel
  boundary: deterministic authority code decides claim lifecycle transitions;
  CLI, adapters, reports, and imported upstream state do not decide release or
  verification; authority state is derived from append-only events.
- Treat third-party AutoResearch systems as external artifact/evidence
  producers. Any call, wrapper, script, or adapter must keep their outputs
  non-authoritative until CairnLab verifier certificates, governance gates,
  provenance, material-dissent rules, and append-only transition events allow a
  state transition.
- Keep I/O, domain logic, presentation, and orchestration separate.
- Do not introduce broad manager/orchestrator classes.
- Do not add runtime dependencies without a current, concrete need and explicit
  justification.
- Do not add an abstraction unless it removes real complexity or serves at
  least one current use.
- Preserve existing public APIs unless the task requires a breaking change.
- Cover public behavior changes with deterministic tests.
- Update the matching design document when public APIs, CLI behavior, schemas,
  storage layout, adapter contracts, third-party AutoResearch calls, or
  transition semantics change.
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
- the implementation weakens CairnLab's current Research Claim Kernel
  requirements, including authority-only transitions, append-only state, thin
  CLI/adapters, governance gates, provenance, or material-dissent handling;
- third-party AutoResearch integration can mutate host state, import host
  runtime packages into CairnLab, use external outputs as authority, or bypass
  CairnLab evidence, verifier, governance, provenance, or transition-event
  requirements;
- a code change modifies public APIs, CLI behavior, schemas, storage layout,
  adapter contracts, third-party AutoResearch calls, or transition semantics
  without matching tests and design docs;
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
- whether CairnLab-specific requirements were preserved;
- dependency changes;
- ADR changes or why none were needed;
- remaining risks.
