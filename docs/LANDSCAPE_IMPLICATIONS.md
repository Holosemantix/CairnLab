# Landscape Implications for CairnLab

The landscape report shows that adjacent and core competitors already cover many capabilities CairnLab might otherwise be tempted to build:

- ARIS: skill/workflow orchestration, experiment bridge, result-to-claim, research wiki.
- AutoResearchClaw: idea-to-paper pipeline, sandbox experiments, multi-agent review, VerifiedRegistry, HITL, self-healing.
- ScientistOne: Chain-of-Evidence, CoE Audit, claim verification, reference and method-code alignment.
- EviBound: approval gates and MLflow-backed verification gates.
- DeepScientist / Claw AI Lab: research OS and lab dashboard directions.
- DVC / MLflow / PROV / RO-Crate / ReproZip / W&B: infrastructure for tracking, provenance, packaging, and artifacts.

Therefore CairnLab must not compete as another end-to-end agent.

CairnLab must first validate whether an enforcement kernel is needed.

The current implication is not "build the kernel immediately." The current implication is:

```text
use existing AutoResearch systems as failure-sampling instruments
before committing to a reusable claim lifecycle control layer
```

Only if repeated cross-system failures appear should CairnLab compete as an enforcement kernel.

## Differentiation test

A feature is not core to CairnLab if an upstream system can add it as a prompt, plugin, or dashboard widget.

A feature is core to CairnLab if it changes the rules by which a claim may move to a stronger scientific state.

## Core innovation boundary

CairnLab's candidate defensible contribution is:

```text
scientific claim lifecycle enforcement
```

This includes:

- state machine,
- evidence policy DSL,
- verifier certificates,
- append-only transition log,
- governance policies,
- human liability gates,
- external adapter protocol.

During validation, the core artifact is not a public benchmark or leaderboard. It is a benchmark-lite harness that reuses existing tasks and records whether claim/evidence/governance failures recur across systems.

## What to reuse

Reuse external systems aggressively:

- MLflow for run and metric import,
- DVC for datasets and large artifacts,
- Git for code state,
- Hydra for experiment configuration,
- RO-Crate for export,
- W3C PROV / PROV-AGENT for provenance semantics,
- in-toto / SLSA for attestation,
- OpenHands / Codex / Claude Code for code patches.

But do not outsource claim state transitions.

That is CairnLab's candidate kernel, contingent on validation evidence.

## What remains uncertain

ScientistOne, EviBound, ARIS, AutoResearchClaw, DeepScientist-style systems, and paper-to-code systems already cover many reliability mechanisms inside their own workflows. Therefore CairnLab should not claim that existing systems lack traceability, evidence chains, or reviewer gates.

The open question is narrower:

```text
Do those in-system mechanisms fail to provide portable,
operational, release-relevant claim lifecycle decisions
across system boundaries?
```

The answer must come from real failure sampling, not positioning language.

The v0.4 adapter work operationalizes that test: AutoResearchClaw e2e artifacts
and ARIS review/audit artifacts can enter CairnLab without importing either
runtime. This keeps the coupling light and makes the competitive boundary
observable: external systems provide evidence; CairnLab tests whether a claim is
allowed to transition.

The generic `cairn.external_run.v1` manifest extends the same boundary beyond
named competitors. Idea generation, paper-to-code, review, experiment execution,
paper writing, and verification may all be handled by external frameworks. The
manifest is only an evidence bridge; it does not turn those frameworks into
CairnLab dependencies or give them release authority.

## v0.3 implication from governance practice

EU AI Act-style record keeping and human oversight requirements strengthen CairnLab's direction because they show that logging, interpretability of decisions, competent human oversight, and lifetime monitoring are not optional niceties for consequential AI systems. They should become kernel transition conditions when claims have high impact.

The practical takeaway is not to build a compliance product. The practical takeaway is to treat traceability, risk tiering, role accountability, and human authority as part of the claim lifecycle runtime.
