# EU AI Act Practice Notes for CairnLab

CairnLab is not a compliance product. This document records which public AI governance practices are structurally useful for the Research Claim Kernel.

## Useful practice 1: lifetime event logging

The EU AI Act requires high-risk AI systems to technically allow automatic recording of events over the lifetime of the system, with logging sufficient for traceability and post-market monitoring.

CairnLab translation:

- all claim state transitions are append-only events;
- events include actor, role, action, inputs, outputs, verifier certificate, risk context, and previous hash;
- the transition ledger is the source of truth for claim status;
- log retention and export policy must be explicit.

## Useful practice 2: human oversight proportional to risk and autonomy

The EU AI Act ties human oversight to risks, autonomy level, and context of use, and expects oversight actors to understand capabilities and limitations, detect automation bias, interpret outputs, and intervene or stop system operation.

CairnLab translation:

- `HumanGate` must include authority, competence, rationale, and liability scope;
- high-risk or high-autonomy claims require explicit human gate;
- automation-bias risk can be recorded in `RiskAssessment`;
- human approval cannot erase verifier failure; it creates a liability-bearing override event.

## Useful practice 3: technical documentation and assessment records

The AI Act's high-risk system documentation model emphasizes validation/testing procedures, data characteristics, performance metrics, monitoring, and human oversight measures.

CairnLab translation:

- `AssessmentRecord` captures algorithm, data, design, deployment, and governance assessments;
- `DecisionTracePackage` bundles these records for released or high-impact claims;
- every released claim must be reproducible from the package or explicitly disclose limitations.

## Useful practice 4: machine-readable AI system registry metadata

AICat proposes machine-readable cataloguing for AI systems to support EU AI Act transparency and traceability requirements.

CairnLab translation:

- maintain machine-readable metadata for each project, system, claim, and release bundle;
- include risk tier, responsible parties, system scope, model/tool dependencies, and evidence policy identifiers;
- keep this as an exportable registry manifest, not a marketing document.

## What not to import

Do not import legal categories mechanically. Most research prototypes are not directly regulated as high-risk AI systems. CairnLab should adopt the design lessons only where they improve claim traceability, role accountability, and reviewability.
