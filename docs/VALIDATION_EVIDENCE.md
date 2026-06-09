# Validation Evidence Ledger

CairnLab stays validation-first until real external systems show repeated
release-control failures that a claim lifecycle transition authority can reduce.
Fixture and adapter tests are necessary contract verification, but they are not
market or design validation by themselves.

The machine-readable ledger lives at:

```text
data/validation_evidence.yaml
```

## What Counts

The ledger separates:

- `real`: an observed run or replay from an external system;
- `fixture_contract`: a fixture that preserves a real external artifact shape;
- `synthetic`: a hand-built stress case for semantic invalidation.

`cairn validate` reports these separately. A `go` recommendation requires real
cross-system evidence, not only synthetic cases or adapter contract tests.

## Current Evidence

Current real evidence is intentionally conservative:

- AutoResearchClaw ML01 and ML03 runs provide real external-system evidence for
  paper/experiment/provenance gaps that affect whether claims should be released;
- ARIS evidence currently includes fixture/contract coverage for real helper
  output shapes, but it is not counted as a real framework run in the go/no-go
  gate.

This means CairnLab should continue sampling. The current evidence supports the
boundary and adapter design, but not a claim that a full kernel has already been
validated across the landscape.

## Why This Matters

The core differentiation is not another AutoResearch agent. External systems can
generate ideas, paper-to-code projects, experiments, reviews, and paper drafts.
CairnLab records their artifacts as evidence and enforces claim lifecycle
transitions with verifier certificates, governance records, material dissent,
and append-only events.
