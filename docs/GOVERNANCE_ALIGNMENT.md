# Governance Alignment

CairnLab is not a legal compliance product, but its kernel model should be compatible with mainstream AI governance expectations around traceability, accountability, human oversight, and role-based risk management.

This document records how CairnLab maps those expectations into concrete kernel primitives.

## Governance sources considered

CairnLab's governance design is aligned with the following public frameworks and regulatory concepts:

- **NIST AI Risk Management Framework (AI RMF 1.0)**: emphasizes risk management throughout the AI lifecycle; separates actors involved in design, development, deployment, evaluation, and use; highlights test, evaluation, verification, and validation (TEVV) across the lifecycle; and treats GOVERN as a cross-cutting function.
- **EU AI Act, Articles 12 and 14**: requires high-risk AI systems to support automatic event logging over the system lifetime, and requires human oversight proportionate to risk, autonomy, and context of use.
- **ISO/IEC 42001:2023**: establishes an AI management system approach with roles, responsibilities, risk assessment, impact assessment, lifecycle processes, operational control, and documented information.
- **RACI-style responsibility models**: distinguish Responsible, Accountable, Consulted, and Informed roles for tasks and deliverables.

## What was already covered

The existing CairnLab design already includes:

- append-only state-transition logs;
- claim state derived from event history rather than mutable fields;
- verifier certificates as transition authority;
- human gates as liability-bearing events;
- role contracts that prevent implementation agents from verifying or approving claims;
- dissent-aware governance;
- provenance for claims, evidence, artifacts, verifier certificates, and human decisions.

These already satisfy much of the traceability and accountability intent.

## Governance gaps addressed in this update

The governance requirements add four missing emphases:

1. **Lifecycle accountability**: responsibility should cover design, development, evaluation, deployment/use, monitoring, release, downgrade, and retirement/retraction.
2. **Responsible entity per action**: every major action should carry not only an actor but also a responsible role, accountable party, consulted parties, and informed parties.
3. **Risk-tiered controls**: risk controls should scale with claim impact, autonomy level, domain sensitivity, and release scope.
4. **Assessment records**: algorithm, data, design, and deployment assumptions should have explicit assessment records rather than being implied by artifacts.

## Added kernel governance objects

### LifecycleContext

`LifecycleContext` binds a claim or transition to a lifecycle stage.

```yaml
lifecycle_context:
  stage: design | development | evaluation | deployment_use | publication | monitoring | downgrade_retraction
  system_scope: local_research | shared_lab | public_release | high_impact_decision
  autonomy_level: assistive | supervised_agentic | autonomous_agentic
  affected_parties:
    - researchers
    - downstream_users
    - public_readers
```

### ResponsibilityAssignment

`ResponsibilityAssignment` turns role separation into a concrete responsibility model.

```yaml
responsibility_assignment:
  object: claim:C1
  action: release_claim
  responsible:
    - role: verifier
      actor: system:metric-threshold@0.1.0
  accountable:
    - role: human_pi
      actor: human:pi@example.org
  consulted:
    - role: domain_reviewer
      actor: human:reviewer@example.org
  informed:
    - role: project_maintainer
      actor: human:maintainer@example.org
```

Rules:

- every released claim must have exactly one accountable party or explicitly documented shared accountability;
- implementation agents cannot be accountable for scientific acceptance;
- verifier agents can be responsible for checks but not accountable for publication;
- human gates must include a liability scope.

### RiskAssessment

`RiskAssessment` records layered and role-based risk identification.

```yaml
risk_assessment:
  id: RA1
  object: claim:C1
  lifecycle_stage: publication
  risk_tier: low | medium | high | critical
  dimensions:
    scientific_validity: high
    reproducibility: medium
    data_sensitivity: low
    downstream_impact: high
    autonomy_level: supervised_agentic
  identified_risks:
    - missing_external_benchmark_server
    - single_dataset_generalization
  controls:
    - require_human_gate
    - require_limitation_disclosure
    - block_if_material_dissent_unresolved
```

### AssessmentRecord

`AssessmentRecord` captures assessment of algorithm, data, design, and deployment assumptions.

```yaml
assessment_record:
  id: AR1
  target: claim:C1
  assessment_type: algorithm | data | design | deployment_context | governance
  assessor:
    role: domain_reviewer
    actor: human:reviewer@example.org
  result: pass | fail | needs_more_evidence | not_applicable
  evidence_refs:
    - evidence:EVID1
    - verifier_certificate:VC1
  notes: "Dataset identity is complete, but external benchmark server was unavailable."
```

### DecisionTracePackage

`DecisionTracePackage` is the exportable trace for consequential decisions.

```yaml
decision_trace_package:
  id: DTP1
  claim: C1
  transition: ST5
  trace_includes:
    - claim_text
    - lifecycle_context
    - evidence_policy
    - evidence_items
    - verifier_certificates
    - risk_assessment
    - responsibility_assignment
    - human_gate
    - unresolved_limitations
    - transition_log_hash
```

A `DecisionTracePackage` is required when a claim reaches `released`, `downgraded`, or `retracted`, and may be required earlier for high-risk claims.

## Risk-tiered transition rules

CairnLab should support risk-tiered controls:

| Risk tier | Example | Required controls |
| --- | --- | --- |
| low | internal note, non-public hypothesis | evidence policy + basic ledger |
| medium | lab report, public benchmark note | verifier certificates + provenance completeness |
| high | public paper claim, medical/clinical claim, safety claim | human gate + risk assessment + limitation disclosure + dissent check |
| critical | claim affecting health, rights, safety, or major resource allocation | dual human gate or independent reviewer gate + full decision trace package |

## Kernel rule update

The original kernel rule remains:

```text
transition_requested
AND required evidence exists
AND required verifier certificates pass
AND no blocking governance rule applies
AND required human gates are satisfied
```

The governance-aligned kernel rule adds:

```text
AND lifecycle context is recorded
AND risk tier is assigned
AND responsibility assignment exists for consequential transitions
AND required assessment records exist for algorithm/data/design/deployment context
AND decision trace package can be generated for high-impact transitions
```

## Why this matters for CairnLab

This update strengthens CairnLab's differentiation.

CairnLab is not merely a provenance logger. It is a governance-aware claim-state kernel. It can answer not only:

```text
Why was this claim verified?
```

but also:

```text
At which lifecycle stage was this decision made?
Who was responsible?
Who was accountable?
Which risk tier applied?
Which controls were required?
Which algorithm/data/design assessments were performed?
Can the decision process be reconstructed later?
```

That is closer to the traceability and accountability standard expected of consequential AI systems.


## Practical governance mapping used in v0.3

CairnLab's kernel design borrows structurally useful ideas from public governance practice:

- lifetime logging and log interpretability;
- human oversight proportionate to risk, autonomy, and context;
- human operators with competence, training, and authority;
- documentation of validation and testing procedures;
- risk management across design, development, use, and evaluation;
- AI management system concepts such as roles, responsibilities, documented information, monitoring, and continual improvement;
- machine-readable cataloguing of AI systems and their metadata.

CairnLab does **not** claim legal compliance. It translates these ideas into kernel primitives so that research claims become traceable and reviewable.
