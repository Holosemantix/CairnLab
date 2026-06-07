# Semantic Invalidation Harness

CairnLab's first reusable module is a decoupled semantic invalidation harness for AutoResearch systems.

It is intentionally narrower than workspace rollback, experiment replay, model registry rollback, or provenance export. It consumes metadata from external research systems and answers one transition-authority question:

```text
If this supporting object is invalidated, which downstream claims, approvals,
release decisions, verifier certificates, and report sections can no longer
keep their current lifecycle authority?
```

## Position In CairnLab

The harness is a Phase 0.6 bridge between validation-first failure sampling and the full Research Claim Kernel.

It is not proof that the full kernel is already justified. It is a portable way to test whether existing systems repeatedly lack claim-state invalidation propagation.

```text
Phase 0.5: sample real AutoResearch failures
Phase 0.6: replay those failures through a semantic invalidation harness
Phase 1: build the full claim lifecycle kernel only if the evidence warrants it
```

## Reusable Boundary

The harness must be independently reusable by other AutoResearch projects.

It exposes a small contract:

- import claims, evidence objects, relations, verifier outputs, human gates, and release decisions;
- build a dependency graph over those objects;
- preview downstream effects with `plan_revert`;
- apply effects only by appending transition events;
- reconstruct projected state from base objects plus append-only events;
- export traces and validation reports.

It must not require the host project to use CairnLab's execution stack, UI, storage backend, LLM prompts, experiment runner, or model registry.

The adapter contract is specified in `docs/ADAPTER_CONTRACT.md`. Adapters translate host metadata into `ClaimCase`; they do not decide claim lifecycle authority.

## Non-Goals

The harness does not:

- restore files or workspaces;
- version datasets or object stores;
- roll back MLflow model aliases;
- replay LangGraph or Temporal workflows;
- run experiments;
- call LLM reviewers;
- parse PDFs;
- replace W3C PROV, RO-Crate, in-toto, SLSA, DVC, lakeFS, MLflow, Git, or workflow engines.

Those systems can provide object identifiers, hashes, runs, artifacts, and provenance records. CairnLab decides only how claim lifecycle authority changes.

## Interchange Shape

The minimal import unit is a claim case:

```yaml
case_id: case_wrong_metric
source_system: synthetic_mlflow_like
stress_scenario: wrong_metric_path
claims: []
evidence: []
relations: []
native_system_behavior: {}
expected_cairnlab_behavior: {}
failure_classes: []
```

The graph relation contract treats `source` as the upstream object and `target` as the object whose authority depends on it. Some authority relations also support reverse propagation from a changed claim to the gate, certificate, or release decision that must be reopened.

## Runtime Layout

Local-first projects use `.cairn/`:

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
```

## CLI Surface

The harness commands are:

```bash
cairn init
cairn import-case examples/cases/case_wrong_metric.yaml
cairn validate
cairn trace claim:C1
cairn affected run:exp_007
cairn revert run:exp_007 --reason "metric computed on wrong split" --plan-only
cairn revert run:exp_007 --reason "metric computed on wrong split" --apply --actor user:alice
```

These commands are a validation and invalidation surface. The later full kernel can add lifecycle authoring commands such as `claim add`, `evidence attach`, `verify`, `gate request`, and `report`.

## State Rule

State is a projection:

```text
current_state = imported_base_state + ordered_append_only_events
```

No revert deletes or rewrites the imported object YAML. A revert appends a root `RevertRequested` event and one derived event per affected object.

## Governance Invariants

The harness must preserve the repository-level governance rules:

- consequential claim transitions require `RiskAssessment`;
- released claims require `ResponsibilityAssignment` with an accountable party;
- high-impact claims require `DecisionTracePackage`;
- human gates record actor, authority, scope, and rationale;
- material dissent blocks release unless resolved by verifier or explicit human override;
- state is derived from append-only events, not mutable fields.

In Phase 0.6, these objects may be imported as structured metadata. They must not be collapsed into free-form prose when the project advances to the full kernel.

## First Stress Cases

The harness should initially support these stress scenarios:

- wrong metric path;
- wrong dataset split;
- underpowered seed claim;
- citation or support withdrawn;
- human approval scope drift;
- verifier verdict not operational;
- dashboard-only mutable status;
- artifact overwritten or hash mismatch.

The first proof point is not broad functionality. It is whether the harness can show that a host AutoResearch system's existing rollback, traceability, or approval mechanism fails to propagate claim-state consequences.
