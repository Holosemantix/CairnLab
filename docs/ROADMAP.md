# CairnLab Roadmap

## Phase 0: Repository alignment

- Rename all previous ReproLedger references to CairnLab.
- Confirm repo: `cairnlab`.
- Confirm package: `cairnlab`.
- Confirm CLI: `cairn`.
- Adopt tagline: `Reliable paths for AI-assisted research.`
- Adopt core positioning: `Research Claim Kernel`.

## Phase 0.5: Validation-first failure sampling

Before building the kernel, validate whether an independent claim lifecycle control layer is actually needed.

Inputs:

- existing AutoResearch systems;
- public or system-provided benchmark tasks;
- manifest-style custom tasks only after controlled benchmark tasks run;
- material claims extracted from final outputs.

Initial target:

- AutoResearchClaw, starting with documented smoke tests and ARC-Bench tasks.

Validation artifacts:

- claim case records;
- failure taxonomy labels;
- task manifests;
- run and artifact references;
- Go / No-Go report.

Acceptance criteria:

- At least three systems or workflows are sampled.
- At least six real tasks are run or replayed.
- At least thirty material claims are recorded.
- At least three failure classes recur across systems.
- At least one recurring failure class directly affects claim release control.
- Existing benchmark metrics do not already capture the failure.
- The failure can plausibly be reduced by an external claim transition authority.

If these criteria are not met, continue with validation harness work and do not proceed to a full transition engine.

## Phase 0.6: Semantic invalidation harness

Build a decoupled, local-first harness that can be reused by other AutoResearch projects to test claim-state invalidation propagation.

This phase is allowed before the full kernel because it is a validation instrument, not a product-thesis commitment. It should remain independently extractable from the rest of CairnLab.

Inputs:

- hand-authored or exported claim cases;
- claims, evidence objects, verifier outputs, human gates, release decisions, and typed relations;
- native-system behavior fields that record whether the host system can propagate invalidation itself.

Core modules:

- `models`: portable claim-case, event, relation, and plan models;
- `store`: local `.cairn/` object store and append-only event ledger;
- `graph`: dependency traversal over imported objects;
- `projection`: current state derived from base objects plus events;
- `planner`: `plan_revert` and affected-object action mapping;
- `validation`: failure-taxonomy and Go / No-Go reports;
- `cli`: thin command wrapper over reusable library APIs.

Commands:

```bash
cairn init
cairn import-case examples/cases/case_wrong_metric.yaml
cairn validate
cairn trace claim:C1
cairn affected run:exp_007
cairn revert run:exp_007 --reason "metric computed on wrong split" --plan-only
cairn revert run:exp_007 --reason "metric computed on wrong split" --apply --actor user:alice
```

Acceptance criteria:

- The harness does not depend on any host AutoResearch runtime.
- A host project can use the planner from Python without using the CLI.
- `--plan-only` does not mutate event logs.
- `--apply` appends a root invalidation event plus derived events.
- Original imported YAML is not rewritten by revert.
- Wrong metric and human scope drift cases produce affected claim, section, gate, and release-decision changes.
- State is projected from append-only events.
- Governance fields are preserved as structured objects or metadata and are not collapsed into prose.

## Phase 0.7: Minimal transition authority seed

Add a small deterministic authority module before the full kernel MVP, so the
project's core difference is exercised early without introducing a policy DSL or
verifier plugin system.

Core module:

- `authority`: request-time transition gate for `verified` and `released` claim states.

Acceptance criteria:

- The module is storage-free and can be reused outside the CLI.
- `verified` requires machine-addressable evidence and a passing verifier certificate.
- `released` rechecks evidence and verifier certificates instead of trusting imported status.
- `released` requires a human gate with actor, authority, scope, and rationale.
- `released` requires `ResponsibilityAssignment` with an accountable party.
- Consequential transitions require `RiskAssessment`.
- High-impact releases require `DecisionTracePackage`.
- Unresolved material dissent blocks release unless explicitly overridden.
- `engine` remains a thin facade over the authority module.

## Phase 1: Kernel MVP

This phase is gated by Phase 0.5 evidence and informed by Phase 0.6 counterfactual results.

Implement the minimum claim state machine.

Objects:

- Claim
- EvidencePolicy
- EvidenceItem
- VerifierCertificate
- StateTransition
- HumanGate
- RoleContract
- PositionStatement

Commands:

```bash
cairn init
cairn claim add
cairn policy list
cairn evidence attach
cairn verify
cairn status
cairn gate request
cairn report
```

Acceptance criteria:

- A claim cannot become `verified` without required verifier certificates.
- A claim cannot become `released` without required human gate.
- A failed verifier blocks transition.
- A material dissent blocks transition.
- State is derived from append-only events.

## Phase 2: Verifier plugins

Implement initial verifiers:

- artifact_exists
- artifact_hash
- metric_json_schema
- metric_threshold
- reference_exists
- provenance_complete
- role_permission
- material_dissent

Acceptance criteria:

- Verifiers emit certificates.
- Certificates can authorize or block transitions.
- Reports cite certificates rather than free-form LLM judgments.

## Phase 3: Experiment and run adapters

Integrate with minimal external systems:

- local command runner,
- Git commit capture,
- MLflow run import,
- DVC data pointer import,
- JSON artifact import.

Acceptance criteria:

- Existing runs can be attached to claims.
- CairnLab can verify claims against imported artifacts.
- CairnLab does not need to execute everything itself.

## Phase 4: Upstream agent adapters

Add adapters for systems that produce claims or artifacts:

- generic Markdown report adapter,
- ARIS-style workflow adapter,
- AutoResearchClaw-style artifact adapter,
- paper-to-code output adapter,
- OpenHands / Codex patch adapter.

Acceptance criteria:

- External agents can submit candidate claims.
- CairnLab can generate blocking issues when evidence is insufficient.
- CairnLab can return claim status to upstream systems.

## Phase 5: Release bundle and provenance export

Add export formats:

- report.md,
- claims.json,
- evidence.json,
- verifier_certificates.json,
- transition_log.jsonl,
- RO-Crate package,
- W3C PROV export,
- optional in-toto/SLSA attestations.

Acceptance criteria:

- A third-party reviewer can reconstruct why a claim was released.
- A released claim can be challenged and downgraded without mutating history.

## Phase 6: Governance hardening

Implement:

- sealed initial positions,
- role separation checks,
- position change events,
- social-consensus rejection,
- human liability scopes,
- challenge / downgrade / retract flows.

Acceptance criteria:

- Agent consensus cannot verify a claim.
- A changed agent position must cite a reason and evidence delta.
- Dissent cannot disappear silently.


## Phase 1.5: Governance alignment MVP

Add lifecycle and accountability primitives before integrating large agent workflows.

Objects:

- LifecycleContext
- ResponsibilityAssignment
- RiskAssessment
- AssessmentRecord
- DecisionTracePackage

Acceptance criteria:

- A released claim without accountable party is blocked.
- A high-impact claim without DecisionTracePackage is blocked.
- A consequential transition without RiskAssessment is blocked.
- A transition missing required algorithm/data/design assessment records is blocked.
- Human override creates a new transition event with liability scope.

## Phase 1.6: EU-style logging and human oversight mapping

Implement practice-inspired controls:

- automatic event logging for all transition events;
- lifetime log retention policy field;
- human oversight assignment with competence/authority fields;
- automation-bias warning field for high-autonomy contexts;
- natural-person verification field where applicable;
- machine-readable system and claim registry metadata.

These are not compliance claims. They are kernel design requirements inspired by public AI governance practice.
