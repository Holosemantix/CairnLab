# Domain Model Design

## Purpose

The domain model defines portable, machine-readable objects for claim lifecycle
control. It is the lowest-level shared contract used by the runtime, adapters,
store, validation code, and CLI.

Primary source files:

- `src/cairnlab/models.py`
- `src/cairnlab/builder.py`
- `schemas/*.schema.json`

## Owns

This module owns:

- claim, evidence, relation, event, and plan data shapes;
- lifecycle and evidence enums;
- governance primitives such as `RiskAssessment`, `ResponsibilityAssignment`,
  `GovernancePolicy`, `HumanGate` metadata, and `DecisionTracePackage`;
- `ClaimCase`, the portable import/export unit;
- `ClaimCaseBuilder`, a convenience API for adapters and tests, including
  governance helper methods for risk, accountability, and decision trace data.

## Does Not Own

This module must not own:

- file layout;
- CLI behavior;
- transition policy execution;
- graph traversal;
- external project detection;
- experiment execution.

## Public Contracts

The most important contracts are:

- `Claim`
- `EvidenceItem`
- `Relation`
- `TransitionEvent`
- `VerifierCertificate`
- `VerificationRequest`
- `DecisionTracePackage`
- `DecisionTracePackageExport`
- `RevertPlan`
- `TraceResult`
- `ClaimCase`
- `ClaimCaseBuilder`

Object IDs must remain stable and meaningful because they are used as graph
nodes, event targets, trace anchors, and adapter interoperability keys.

`VerifierCertificate` is the structured output of deterministic verifier
execution. For current storage compatibility, certificates are attached to claim
cases as `EvidenceItem(type=verifier_certificate)` through the builder helper.

`DecisionTracePackageExport` is the structured review bundle emitted by
`DecisionTracePackager`. It is not transition authority; it packages the evidence
and governance context needed to inspect a consequential decision.

## Dependency Rules

`models.py` may depend on Pydantic and the Python standard library. It must not
import `store`, `engine`, `cli`, `runtime`, or adapters.

`builder.py` may depend on `models.py`. It should remain a thin helper and must
not hide claim transition authority.

## Extension Rules

When adding or changing model fields:

- preserve backwards-compatible aliases where existing case files use them;
- update related JSON schemas;
- update affected module docs;
- add tests for validation, serialization, and at least one runtime path if the
  field affects transition behavior.

## Tests

Current coverage is indirect through:

- `tests/test_semantic_invalidation.py`
- `tests/test_runtime_and_adapters.py`
- adapter fixture tests

Add direct model tests when validation rules become stricter or schema
compatibility becomes a risk.
