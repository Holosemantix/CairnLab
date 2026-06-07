# CairnLab System Design

CairnLab is implemented as a Research Claim Kernel, starting with a reusable
semantic invalidation harness. The current system is not an autonomous research
agent and not an experiment runner. It answers a narrower question:

```text
When an artifact, verifier, human gate, or supporting object loses authority,
which claim lifecycle states must change, and what append-only events record it?
```

## Current Scope

The implemented system covers Phase 0.6 plus a minimal Phase 0.7 transition
authority seed:

- portable `ClaimCase` import;
- in-memory runtime for claim invalidation planning;
- local `.cairn/` project storage;
- append-only transition event projection;
- deterministic transition authority for verification and release requests;
- deterministic adapter detection for external manifest exports;
- thin CLI commands over reusable Python APIs.

It does not yet implement the full EvidencePolicy or VerifierCertificate
execution engine described in `docs/KERNEL_SPEC.md`.

## Architecture

```text
External project metadata
        |
        v
Adapter registry
        |
        v
AutoResearchAdapter.export_case()
        |
        v
ClaimCase
   |                 |
   v                 v
CairnRuntime      CairnProject
in memory         local .cairn/ store
   |                 |
   +--------+--------+
            v
  RelationGraph + EventProjection
            |
            +------------------------+
            |                        |
            v                        v
   InvalidationPlanner      TransitionAuthority
            |                        |
            v                        v
 RevertPlan + Events        TransitionDecision + Event
```

## Dependency Direction

Allowed dependency direction:

```text
models
  <- builder
  <- graph / projection / planner / authority / validation
  <- runtime
  <- store
  <- engine
  <- cli

adapters/base + adapters/*
  -> models + builder
  -> no store, no engine, no cli
```

Key rules:

- `models` must remain portable and must not depend on storage, CLI, or adapters.
- `planner` must not write events directly.
- `authority` must not write events directly or call external reviewers.
- `store` must not decide transition semantics.
- `engine` is a composition facade, not a policy layer.
- `cli` is a command facade, not a kernel logic layer.
- adapters translate host metadata into `ClaimCase`; they never decide release,
  downgrade, retraction, or verification authority.

## State Model

State is projected:

```text
current_state = imported base objects + ordered append-only TransitionEvents
```

Imported claim and evidence YAML files are not rewritten by revert operations.
Invalidation appends events such as:

- `RevertRequested`;
- `MetricInvalidated`;
- `ClaimDowngraded`;
- `HumanReapprovalRequired`;
- `ReleaseDecisionReopened`.

This preserves auditability and allows later review of why a claim changed state.

## Reusable Surfaces

CairnLab has two public integration surfaces:

1. In-memory library API:

```python
from cairnlab import CairnRuntime
from cairnlab.adapters import export_case

export = export_case(project_path)
runtime = CairnRuntime.from_case(export.case)
plan = runtime.plan_revert("run:exp_007", reason="wrong metric split")
events = runtime.events_from_plan(plan)
```

2. Transition authority API through the local project facade:

```python
from cairnlab import Actor, CairnProject
from cairnlab.models import ClaimState

project = CairnProject.open(".")
decision = project.request_transition(
    "claim:C1",
    ClaimState.RELEASED,
    Actor(id="human:pi", role="principal_investigator"),
    reason="release review",
)
```

3. Local project CLI:

```bash
cairn adapter detect path/to/project --json
cairn import-external path/to/project --adapter auto --path .
cairn affected run:exp_007 --json
cairn revert run:exp_007 --reason "wrong metric split" --apply
```

The library API is the primary reusable surface. The CLI is optional.

## Governance Invariants

The implementation must preserve these repository-level constraints:

- consequential claim transitions require `RiskAssessment`;
- released claims require `ResponsibilityAssignment` with an accountable party;
- high-impact claims require `DecisionTracePackage`;
- human gates record actor, authority, scope, and rationale;
- material dissent blocks release unless resolved by verifier or explicit human override;
- state is derived from append-only events, not mutable fields.

The Phase 0.6 harness already preserves these objects as structured model fields
or evidence metadata. Full enforcement belongs to later kernel phases.

## Extension Points

Preferred extension points:

- add new model fields in `models.py` with schema and design doc updates;
- add new relation semantics in `graph.py` and planner tests;
- add new affected-object actions or event mappings in `planner.py`;
- add new deterministic transition gates in `authority.py`;
- add new manifest adapters under `src/cairnlab/adapters/`;
- add storage backends by replacing `CairnProjectStore`, not planner logic;
- add new CLI commands as wrappers over `engine` or library APIs.

Avoid extension through hidden globals, dynamic imports of host systems, mutable
claim status fields, or adapter-specific logic inside planner or store modules.

## Current Refactor Watchpoints

These are known areas to revisit as functionality grows:

- split full verifier policy enforcement from Phase 0.6 validation reports;
- keep adapter registry static until there is a real plugin packaging need;
- move CLI JSON payload shapes into tested response models if commands expand;
- add schema-driven contract tests when external adapters become separate packages;
- consider an abstract event sink only after a second storage backend exists.

Do not add these abstractions early. They should appear only when they remove
real coupling or support a concrete external integration.
