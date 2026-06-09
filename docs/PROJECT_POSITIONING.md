# Project Positioning

## Name

- Project: **CairnLab**
- Repository: `cairnlab`
- Package: `cairnlab`
- CLI: `cairn`
- Tagline: **Reliable paths for AI-assisted research.**

A cairn is a human-made stack of stones used to mark a path. This is exactly the metaphor for the system: each claim, run, artifact, verifier verdict, and human gate is a stone. The final output is not just a paper or report; it is a reliable path that later researchers can follow, inspect, challenge, and extend.

## Strategic positioning

CairnLab should not compete head-on with end-to-end autonomous research systems.

The market already contains strong systems and near-competitors:

- ARIS / Auto-claude-code-research-in-sleep: skill/workflow layer for autonomous ML research.
- AutoResearchClaw: idea-to-paper multi-agent pipeline with sandboxing, review, self-healing, and HITL modes.
- ScientistOne: Chain-of-Evidence and claim verification inside an end-to-end autonomous research system.
- EviBound: evidence-bound gates to prevent unsupported claims.
- DeepScientist and Claw AI Lab: research OS / lab workspace directions.
- Paper2Agent / Paper2Code / AutoP2C / RePro: paper-to-agent and paper-to-code verification directions.
- MLflow / DVC / W3C PROV / RO-Crate / OpenLineage / in-toto / SLSA: mature tracking, provenance, packaging, lineage, and attestation components.

Therefore, CairnLab's defensible position is not:

```text
another AI scientist
another autoresearch loop
another research OS dashboard
another evidence chain inside one pipeline
```

The defensible position is:

```text
Research Claim Kernel for accountable AI-assisted research
```

## Current project stance

CairnLab is in a validation-first phase.

The project should not immediately assume that an independent claim lifecycle kernel is needed. Existing systems such as ARIS, AutoResearchClaw, ScientistOne-style systems, EviBound, DeepScientist-style research OSes, and paper-to-code systems already include many forms of claim traceability, evidence chains, reviewer loops, and governance gates.

The immediate task is therefore to test whether those in-system mechanisms repeatedly fail in ways that justify an external claim transition authority.

## One-sentence definition

CairnLab is a validation-first effort to determine whether AutoResearch systems need a reusable claim-state enforcement runtime with policy-bound state transitions, verifier certificates, provenance records, and human responsibility gates.

## The candidate core contribution

The core contribution is not stitching together existing modules.

The candidate core contribution is a formal and executable model for **scientific claim lifecycle transitions**.

A claim cannot move from draft to release just because an agent generated a convincing paper, a reviewer agreed, or a dashboard shows a nice metric. A claim moves only when the kernel receives the required evidence, verifier certificates, governance checks, and human gates.

This candidate contribution should be implemented only after validation shows repeated cross-system failures that cannot be handled by existing in-system gates or public benchmark scores.

The adapter layer is the practical boundary test for this position. Strong
systems such as ARIS and AutoResearchClaw may produce rich run artifacts,
review sidecars, audit verdicts, HITL records, and verifier reports. CairnLab
should import those objects as portable evidence, but the claim lifecycle
transition remains a CairnLab authority decision rather than an upstream agent
or reviewer conclusion.

This boundary is stage-neutral. Idea generation, paper-to-code, review,
experiment execution, paper writing, and verification can all be delegated to
external frameworks. CairnLab's reusable surface is the external evidence
manifest and transition authority, not ownership of those upstream stages.

## Competitive boundary

CairnLab should be embedded by systems like ARIS, AutoResearchClaw, ScientistOne-style systems, DeepScientist-style research OSes, and paper-to-code systems.

It should not try to replace them.

The validation evidence ledger keeps this positioning honest: fixtures and
adapter contracts demonstrate interoperability, while real cross-system release
control failures are required before treating the kernel as validated.

If the validation phase succeeds, CairnLab should answer the question they all need to answer:

> Can this claim be safely promoted to a stronger scientific state?

During validation, CairnLab should instead ask:

> Do existing systems repeatedly fail to make claim lifecycle decisions portable, operational, and accountable across system boundaries?

## Non-negotiable positioning rules

1. Do not market CairnLab as a paper generator.
2. Do not market CairnLab as a general research OS UI.
3. Do not make LLM review the final authority.
4. Do not make provenance an afterthought.
5. Do not let consensus override verifier failure.
6. Do not treat human approval as a button; it is a liability-bearing event.
7. Do not let claim status mutate without an append-only transition event.

## Tagline interpretation

**Reliable paths for AI-assisted research** means:

- paths, not isolated artifacts;
- reliable, not merely automated;
- AI-assisted, not AI-unaccountable;
- research, not just code execution.


## Final strategic claim

CairnLab's core claim is not that it integrates evidence, provenance, verifiers, and human gates. The candidate core claim is that AutoResearch may need a **scientific claim lifecycle state machine** that upstream agents cannot bypass.

A feature belongs in CairnLab only if it either:

- helps validate whether that candidate core claim is true; or
- affects claim state transitions or the evidence, governance, risk, or responsibility conditions for those transitions.

## Non-derivative boundary

CairnLab should be judged against these anti-copycat tests:

1. If an upstream system can implement the feature as a prompt or dashboard widget, it is not kernel core.
2. If a feature does not affect claim state transition authority, it is not kernel core.
3. If provenance is only a report after the fact, it is not kernel core.
4. If human approval is only a UI button without liability scope, it is not kernel core.
5. If consensus can change claim state without a verifier certificate or human override, it is not kernel core.
