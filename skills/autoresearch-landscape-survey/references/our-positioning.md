# Our Positioning: CairnLab

## From Accountable Research CI to Research Claim Kernel

The v0.2 target was Accountable Research CI:

```text
paper/claim -> experiment spec -> run/artifact -> evidence-bound review -> issue/patch/rerun/closure
```

The v0.3 target is stricter:

```text
Research Claim Kernel
= claim lifecycle state machine
+ evidence policy DSL
+ verifier-issued transition certificates
+ append-only transition log
+ governance/risk/responsibility controls
+ human liability gates
+ decision trace packages
```

## Why the change matters

ARIS, AutoResearchClaw, ScientistOne, EviBound, DeepScientist, and Claw AI Lab already cover major parts of autonomous research. CairnLab cannot win by adding another pipeline or dashboard.

CairnLab should own the question:

> Can this scientific claim be promoted to a stronger state, and why?

## Non-negotiable boundary

Do not treat these as core innovation by themselves:

- claim extraction;
- MLflow/DVC integration;
- provenance export;
- evidence-bound report generation;
- human approval UI;
- multi-agent review prompts.

They become CairnLab core only when they participate in enforceable claim-state transition rules.
