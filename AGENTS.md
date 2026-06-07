# Instructions for AI Agents Maintaining This Repository

## Strategic boundary

This repository is now aligned around **CairnLab: Research Claim Kernel**.

Do not describe the project as another autonomous scientist, idea-to-paper agent, paper-to-code system, or research OS dashboard. The central abstraction is enforceable scientific claim lifecycle transition.

## Core rule

No artifact, no claim.

LLMs propose. Verifiers decide. Provenance records. Humans approve.

## Do not collapse into integration glue

It is not enough to connect MLflow, DVC, PROV, RO-Crate, EviBound-style gates, and reviewer prompts. A change is core only if it affects whether a claim is allowed to transition to a stronger state.

Core primitives:

- Claim
- LifecycleContext
- RiskAssessment
- ResponsibilityAssignment
- EvidencePolicy
- EvidenceItem
- VerifierCertificate
- StateTransition
- GovernancePolicy
- HumanGate
- DecisionTracePackage

## New project workflow

1. Create an intake YAML from `skills/autoresearch-landscape-survey/assets/templates/new_project_intake.yaml`.
2. Run `check_new_project.py`.
3. Run `classify_project_from_yaml.py`.
4. Run `prepare_ai_project_review.py` and use the generated prompt to read the repository deeply.
5. Update `data/project_registry.yaml` only after reading sources.
6. If the project introduces a new property, add it to `facets.custom_facets` first.
7. Regenerate `reports/project_overview.md` and `reports/landscape_report.md`.
8. If the project weakens our strategic differentiation, update `docs/LANDSCAPE_IMPLICATIONS.md` and `docs/PROJECT_POSITIONING.md`.

## Required analytical distinctions

- idea-to-paper vs claim lifecycle kernel;
- evidence chain inside one system vs cross-system enforceable state machine;
- provenance logging vs transition authority;
- LLM reviewer vs deterministic verifier certificate;
- claim attestation format vs claim state runtime;
- run tracking vs scientific release decision;
- human approval button vs liability-bearing human gate;
- role labels vs enforceable role permissions;
- consensus vs material dissent resolution.

## Governance requirements

When modifying docs or code, preserve these properties:

- consequential claim transitions require `RiskAssessment`;
- released claims require `ResponsibilityAssignment` with an accountable party;
- high-impact claims require `DecisionTracePackage`;
- human gates must record actor, authority, scope, and rationale;
- material dissent must block release unless resolved by verifier or explicit human override;
- state is derived from append-only events, not mutable fields.
