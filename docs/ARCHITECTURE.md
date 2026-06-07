# CairnLab Candidate Architecture

This document describes the candidate Research Claim Kernel architecture.

The architecture should be implemented only if validation-first failure sampling shows a repeated cross-system need for an independent claim lifecycle control layer.

It integrates with agent systems, experiment trackers, code runners, and provenance tools, but it must not become a thin wrapper around them. Its responsibility is enforcing claim lifecycle transitions.

## High-level architecture

```text
External agent / workflow / research OS
        |
        v
Cairn API / CLI
        |
        v
Research Claim Kernel
  ├── Claim Graph
  ├── Evidence Policy Engine
  ├── Verifier Engine
  ├── Transition Ledger
  ├── Governance Module
  └── Human Gate Module
        |
        v
Exports / Reports / Issues / Release Bundles
```

## Phase 0.6 Semantic Invalidation Harness

Before the full Research Claim Kernel is justified, CairnLab provides a smaller reusable harness for claim-state invalidation propagation.

```text
External AutoResearch metadata
        |
        v
ClaimCase Import Contract
        |
        v
Portable Modules
  ├── Store
  ├── Relation Graph
  ├── Event Projection
  ├── Invalidation Planner
  ├── Validation Reporter
  └── CLI / Python API facade
        |
        v
RevertPlan / TransitionEvents / TraceResult
```

The harness is designed so that another AutoResearch project can import only the modules it needs. The CLI is optional. The local `.cairn/` store is the default adapter, not the core abstraction.

### Runtime Layout

```text
.cairn/
  project.yaml
  objects/
    claims/*.yaml
    evidence/*.yaml
    relations/*.yaml
    cases/*.yaml
  events/
    events.jsonl
  reports/
    validation_report.json
    validation_report.md
  cache/
    graph.json
```

### Module Boundaries

`models` defines portable Pydantic objects and enums. It must not depend on the local store or CLI.

`store` loads and writes claim cases, object YAML, and append-only event JSONL. It must not decide transition semantics.

`graph` builds dependency edges and downstream traces from relations. It must not know about file paths or CLI options.

`projection` derives current object and claim state from imported base objects plus ordered events. It must not mutate base objects.

`planner` computes `RevertPlan` objects from a graph, projection, and target object. It must not append events directly.

`validation` scores claim cases and failure taxonomies. It must not execute experiments or call LLM reviewers.

`engine` is a thin facade that composes the modules for local project use.

`cli` is a command wrapper over `engine`. No kernel logic should live in CLI handlers.

### Semantic Revert Contract

```python
plan = project.plan_revert(
    target_id="run:exp_007",
    reason="metric computed on wrong split",
    actor=Actor(id="user:alice", role="maintainer"),
)
events = project.apply_plan(plan)
trace = project.trace("claim:C1")
```

`plan_revert` must be safe to run as a pure preview. `apply_plan` appends a root `RevertRequested` event and one derived event per affected object. No imported claim or evidence YAML may be rewritten by revert.

### Propagation Semantics

For ordinary dependency relations, `source` is the upstream object and `target` is the dependent object. For authority relations such as `approved_by`, `verified_by`, and `released_by`, the graph also supports reverse propagation from an affected claim to the gate, certificate, or release decision that may need to be reopened.

This makes both directions operational:

```text
invalidated gate -> dependent claim challenged
invalidated evidence -> dependent claim changed -> gate/release decision reapproval required
```

The planner emits affected-object actions rather than editing state directly:

- run, metric, artifact, dataset, code, citation -> `invalidate`;
- verifier certificate -> `invalidate`;
- released claim losing critical support -> `downgrade`;
- unreleased verified claim losing critical support -> `challenge`;
- report or paper section -> `mark_stale`;
- human gate -> `require_reapproval`;
- release decision -> `reopen_release_decision`.

## Claim Graph

The Claim Graph stores claims and their relationships to evidence, experiments, artifacts, references, reviewers, gates, and state transitions.

It is not merely a citation graph.

It is a state graph that answers:

```text
What is the current status of this claim, and why?
```

## Evidence Policy Engine

The Evidence Policy Engine maps claim types to required evidence.

Examples:

- empirical score claim,
- method implementation claim,
- novelty claim,
- citation claim,
- ablation claim,
- robustness claim,
- safety/ethics claim,
- negative result claim.

Each type can require different evidence and verifier sets.

## Verifier Engine

The Verifier Engine runs registered verifier plugins and emits `VerifierCertificate` objects.

Verifier categories:

- artifact completeness,
- metric schema,
- metric threshold,
- statistical validity,
- reference existence,
- citation support,
- method-code alignment,
- environment completeness,
- data identity,
- replayability,
- governance compliance.

A verifier certificate can authorize, block, or request more evidence for a transition.

## Transition Ledger

The Transition Ledger is append-only.

It stores:

- claim creation,
- evidence attachment,
- verifier certificates,
- state transitions,
- role actions,
- position statements,
- human gates,
- release bundles,
- downgrades and retractions.

The current claim state is derived from the ledger. It is not edited in-place.

## Governance Module

The Governance Module enforces multi-agent and human accountability rules.

Examples:

- generator and verifier role separation,
- sealed initial judgments,
- dissent preservation,
- material dissent blocks release,
- social consensus cannot verify claims,
- role permissions,
- human override requires liability event.

## Human Gate Module

Human gates are not UI confirmations. They are responsibility-bearing events.

Human gates can approve:

- expensive runs,
- proxy datasets,
- unresolved limitations,
- release of verified claims,
- retraction or downgrade decisions.

## External integrations

CairnLab can integrate with:

- MLflow for runs and metrics,
- DVC for data/model versioning,
- Git for code state,
- Hydra for experiment configuration,
- Snakemake/Nextflow for workflows,
- RO-Crate for release packaging,
- W3C PROV / PROV-AGENT for provenance export,
- in-toto / SLSA for artifact attestation,
- OpenHands / Codex / Claude Code for patch generation,
- ARIS / AutoResearchClaw / DeepScientist-style systems as upstream workflows.

## Architectural non-goals

CairnLab v0.1 should not implement:

- full autonomous research,
- paper generation,
- deep research browsing,
- a large lab dashboard,
- multi-agent chat orchestration,
- automatic novelty discovery.

Those are upstream systems. CairnLab is the kernel that enforces whether their claims can be promoted.


## Risk & Responsibility Module

The Risk & Responsibility Module is part of the kernel, not an external compliance add-on.

It consumes:

- `LifecycleContext`
- `RiskAssessment`
- `ResponsibilityAssignment`
- `AssessmentRecord`
- `GovernancePolicy`

It emits:

- `risk_control_required`
- `transition_blocked_by_missing_accountability`
- `human_gate_required`
- `decision_trace_required`
- `release_allowed`

This module operationalizes governance requirements such as lifetime logging, role-based risk assignment, human oversight, and traceable decision processes.

## Decision Trace Exporter

The Decision Trace Exporter creates a reviewable package for any released or high-impact claim.

It can export native JSON, Markdown, W3C PROV-inspired graphs, RO-Crate-compatible metadata, and future in-toto/SLSA-style attestations.

The exporter is not the source of truth; it is a view over the append-only transition ledger.
