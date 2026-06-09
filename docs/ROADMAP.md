# CairnLab Roadmap

This roadmap is organized by tracks and milestones. The numbering is hierarchical
instead of version-like: `0.3` means the third milestone in Track 0, not an
incomplete product version.

## 0. Validation And Kernel Seed Track

This track validates CairnLab's strategic thesis before the project commits to a
full kernel.

### 0.1 Repository Alignment

- Rename previous ReproLedger references to CairnLab.
- Confirm repo, package, and CLI names.
- Adopt **Research Claim Kernel** as the strategic boundary.
- Preserve the core rule: no artifact, no claim.

### 0.2 Failure Sampling

Validate whether an independent claim lifecycle control layer is actually needed.

Acceptance:

- at least three systems or workflows are sampled;
- at least six real tasks are run or replayed;
- at least thirty material claims are recorded;
- at least three failure classes recur across systems;
- at least one recurring failure class directly affects claim release control;
- existing benchmark metrics do not already capture the failure;
- the failure can plausibly be reduced by an external claim transition authority.

If these criteria are not met, keep CairnLab as a validation harness rather than
promoting it into a full kernel.

### 0.3 Semantic Invalidation Harness

Build a decoupled, local-first harness that can be reused by other AutoResearch
projects to test claim-state invalidation propagation.

Core modules:

- `models`, `store`, `graph`, `projection`, `planner`, `validation`, `cli`;
- reusable Python API first, CLI second.

Acceptance:

- no dependency on a host AutoResearch runtime;
- `plan_revert` is usable without `.cairn/` storage;
- plan-only mode does not mutate events;
- apply mode appends events and never rewrites imported YAML;
- wrong metric, stale section, human scope drift, and release reopen cases work;
- governance fields remain structured.

### 0.4 External Run Evidence Adapters

Import real AutoResearch outputs without importing host runtime packages.

Acceptance:

- AutoResearchClaw manifest and e2e runs import as `ClaimCase`;
- ARIS manifests import as `ClaimCase`;
- adapter diagnostics preserve degraded, paused, verifier-rejected, and missing
  human-gate signals;
- adapters remain translators, not transition authorities.

Closure note:

- AutoResearchClaw e2e evidence now covers real stage/pipeline artifacts such
  as result analysis, downstream pauses, quality reports, paper verification,
  and sanitization signals.
- ARIS evidence now covers research-wiki manifests, audit JSON, real
  `*.review.json` sidecars, and submission verifier reports.
- ARIS smoke validation now exercises real deterministic helper boundaries:
  `research_wiki.py`, `evidence_check.py`, and `verify_paper_audits.sh`.
- LLM reviewer sidecars are imported as evidence context; deterministic
  verifier reports are imported as verifier evidence; neither bypasses the
  CairnLab transition authority.

## 1. Claim Kernel Track

This track turns the validated harness into enforceable claim lifecycle control.

### 1.1 Transition Authority Seed

Add a deterministic transition gate for `verified` and `released` states.

Acceptance:

- `verified` requires machine-addressable evidence and a passing verifier certificate;
- `released` requires verified evidence, human gate, accountable party, and risk controls;
- high-impact release requires a decision trace package;
- material dissent blocks release unless resolved or explicitly overridden;
- the module is storage-free and reusable outside the CLI.

### 1.2 Verifier Certificate Execution

Generate deterministic verifier certificates that can feed transition authority
without relying on LLM reviewer prose.

Acceptance:

- `artifact_hash` and `metric_threshold` emit pass/fail/error certificates;
- certificates are attachable as evidence;
- passing certificates can authorize `verified`;
- failed certificates cannot authorize stronger claim states.

### 1.3 Decision Trace Package Export

Produce a reviewable package for consequential claim decisions.

Acceptance:

- export includes claim, evidence, certificates, gates, dissent, governance records,
  relations, and transition events;
- package generation is storage-free;
- export includes a stable hash;
- local projects expose it through a thin CLI command.

### 1.4 Kernel MVP

Implement the minimum claim lifecycle state machine if 0.2 and 0.3 show repeated
release-control gaps.

Acceptance:

- a claim cannot become `verified` without required verifier certificates;
- a claim cannot become `released` without required human gate and accountability;
- failed verifiers and unresolved material dissent block transition;
- state is derived from append-only events.

## 2. Governance Track

This track makes responsibility, risk, oversight, dissent, and release decisions
operational.

### 2.1 Governance Alignment MVP

- block released claims without accountable parties;
- block high-impact claims without decision trace packages;
- block consequential transitions without risk assessment;
- record actor, authority, scope, rationale, and liability scope for human override.

### 2.2 Logging And Human Oversight Mapping

- log transition events automatically;
- represent retention, oversight assignment, competence, authority, and
  automation-bias fields where relevant;
- keep these as design controls, not legal compliance claims.

### 2.3 Governance Hardening

- prevent agent consensus from verifying a claim;
- require changed positions to cite reason and evidence delta;
- preserve dissent as append-only state;
- keep challenge, downgrade, and retract flows event-derived.

## 3. Ecosystem Track

This track expands interoperability without collapsing CairnLab into integration
glue.

### 3.1 Verifier Plugins

Initial plugin candidates:

- artifact existence and hash;
- metric schema and threshold;
- reference existence;
- provenance completeness;
- role permission;
- material dissent.

### 3.2 Experiment And Run Adapters

Initial integrations:

- local command runner metadata;
- Git commit capture;
- MLflow run import;
- DVC data pointer import;
- JSON artifact import.

### 3.3 Upstream Agent Adapters

Initial adapters:

- Markdown report;
- ARIS workflow;
- AutoResearchClaw artifacts and e2e runs;
- paper-to-code output;
- OpenHands or Codex patch metadata.

### 3.4 Release Bundle And Provenance Export

Export candidates:

- `report.md`, `claims.json`, `evidence.json`, `verifier_certificates.json`;
- `transition_log.jsonl`;
- RO-Crate and W3C PROV-compatible metadata;
- optional in-toto or SLSA-style attestations.
