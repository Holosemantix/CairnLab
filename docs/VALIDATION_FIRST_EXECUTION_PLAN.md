# Validation-First Execution Plan

This document records the current execution plan for CairnLab.

CairnLab should not immediately build a full AutoResearch system, a generic benchmark, or a full claim lifecycle kernel. The near-term goal is to validate whether existing AutoResearch systems repeatedly fail in ways that justify an independent claim lifecycle control layer.

## Current Decision

The project is in a validation-first phase.

The thesis under test is:

```text
Existing AutoResearch systems increasingly provide claim traceability,
evidence chains, reviewer loops, and governance gates inside their own
pipelines. The remaining question is whether there is a repeated
cross-system need for an external claim lifecycle transition authority.
```

The kernel should be built only if real usage shows that in-system evidence chains and governance gates are insufficient for claim release control.

## Two-Stage Route

### Phase A: Failure Sampling

Use existing AutoResearch systems as instruments for discovering real failures.

Initial systems:

- AutoResearchClaw, starting with ARC-Bench tasks.
- ARIS / Auto-claude-code-research-in-sleep, after the AutoResearchClaw pass.
- A paper-to-code or paper-to-reproduction workflow, if resources allow.

The goal is not to prove that one system is good or bad. The goal is to collect comparable claim cases and identify whether the same governance failures recur across systems.

### Phase B: Claim Lifecycle Control Layer

Build the CairnLab transition engine only if Phase A produces repeated failures that are:

- cross-system rather than tool-specific;
- material to whether a claim may be verified, released, challenged, downgraded, or retracted;
- not already captured by existing benchmark scores or in-system gates;
- addressable by an external claim transition authority.

### Phase A.5: Semantic Invalidation Harness

Before committing to the full kernel, CairnLab can build a small reusable harness that replays invalidation scenarios against imported claim cases.

This harness is not a workspace rollback tool. It is a counterfactual claim-state module:

```text
invalidated run / metric / artifact / approval
  -> dependency graph
  -> affected claims, verifier certificates, human gates, release decisions
  -> append-only transition events
  -> projected claim authority
```

The harness must stay decoupled from any one AutoResearch stack. External projects should be able to export claims, evidence, relations, verifier outputs, human gates, and release decisions as YAML/JSON, then call the planner without adopting CairnLab's CLI or storage layout.

## First Target: AutoResearchClaw

The first validation target is AutoResearchClaw because it already includes ARC-Bench, benchmark-style manifests, metrics, datasets, rubrics, and verifiable result reporting.

Recommended sequence:

1. **Smoke test**
   - Run the smallest documented demo or a lightweight ARC-Bench task.
   - Validate setup, model calls, sandbox execution, logs, artifacts, and final report generation.
   - Do not use a custom research topic yet.

2. **Controlled ARC-Bench sampling**
   - Select two or three ARC-Bench tasks.
   - Include one lightweight ML task and one task close to the intended research direction.
   - Extract five to ten material claims from each final report.
   - Record evidence, run, artifact, code, dataset, reviewer, and release-support information for each claim.

3. **Own-topic externalization**
   - Convert the custom research topic into an ARC-Bench-style manifest before running it.
   - Do not submit a vague open-ended topic directly.
   - Require explicit conditions, datasets, metrics, compute budget, expected artifacts, failure criteria, and claims to validate.

Example custom manifest shape:

```yaml
research_question: ""
conditions: []
allowed_datasets: []
baseline: ""
success_metric: ""
compute_budget: ""
expected_artifacts: []
failure_criteria: []
claims_to_validate: []
```

## Claim Case Record

Each material claim should be recorded as a claim case. This is a validation artifact, not yet a kernel object.

```yaml
case_id: ""
source_system: "autoresearchclaw"
task_id: ""
model_config: ""
claim:
  text: ""
  type: "empirical_score | method | citation | novelty | robustness | safety | limitation"
  source_location: ""
evidence:
  evidence_items: []
  run_refs: []
  artifact_refs: []
  code_refs: []
  dataset_refs: []
  environment_refs: []
verdict:
  reviewer_or_verifier: ""
  verdict_text: ""
  operational_decision: "allow | block | needs_more_evidence | unclear"
  reason_codes: []
human_decision:
  actor: ""
  authority: ""
  scope: []
  rationale: ""
dissent:
  material_dissent_present: false
  dissent_refs: []
  resolved: false
lifecycle_support:
  can_be_verified: "yes | no | unclear"
  can_be_released: "yes | no | unclear"
  can_be_challenged: "yes | no | unclear"
  can_be_downgraded_or_retracted: "yes | no | unclear"
notes: ""
```

## Failure Taxonomy

Use these labels during Phase A. Add new labels only when an observed failure cannot be represented here.

- `claim_without_artifact`
- `artifact_not_machine_checkable`
- `run_not_queryable`
- `metric_not_bound_to_claim`
- `method_code_mismatch`
- `verdict_non_operational`
- `reviewer_consensus_without_verifier`
- `material_dissent_suppressed`
- `human_decision_under_specified`
- `release_state_mutable_or_implicit`
- `challenge_or_retraction_missing`
- `cross_system_trace_loss`
- `benchmark_score_hides_release_risk`
- `rollback_without_invalidation`
- `downstream_claim_not_invalidated`
- `paper_section_not_marked_stale`
- `release_decision_not_reopened`
- `human_scope_not_reopened`
- `state_not_event_derived`
- `system_local_traceability_only`
- `side_effect_unaccounted`

## Semantic Invalidation Stress Scenarios

Phase A.5 should start with synthetic cases before relying on expensive real-system runs. These cases are validation fixtures for the reusable harness.

| Scenario | Invalidated object | Expected claim-state consequence |
| --- | --- | --- |
| wrong metric path | run or metric | released claim challenged or downgraded; result section stale; release reopened |
| wrong dataset split | dataset or run | dependent claims challenged; verifier certificates depending on the run invalidated |
| underpowered seed claim | material dissent | release blocked or claim scope downgraded |
| citation/support withdrawn | citation | literature claim challenged or marked stale |
| human approval scope drift | human gate | reapproval required; release decision reopened |
| verifier verdict not operational | reviewer verdict | unresolved material dissent blocks release or creates challenge event |
| dashboard-only status | mutable claim status | state provenance failure recorded |
| artifact overwritten | artifact hash | evidence invalidated; downstream claims challenged |

Each semantic invalidation case should record:

```yaml
native_system_behavior:
  can_restore_workspace_state: true | false | unknown
  can_replay_or_fork_agent_state: true | false | unknown
  can_identify_downstream_claims: true | false | partial | unknown
  can_invalidate_claims: true | false | partial | unknown
  can_downgrade_or_retract_released_claims: true | false | partial | unknown
  can_preserve_old_history_append_only: true | false | unknown
  can_require_reapproval: true | false | partial | unknown
  can_export_machine_readable_revert_trace: true | false | unknown
cairnlab_counterfactual:
  would_change_final_claim_state: true | false
  affected_objects_detected: []
  proposed_transitions: []
```

## Go / No-Go Criteria

Move from Phase A to a CairnLab transition engine MVP only if all of the following hold:

- At least three systems or workflows have been sampled.
- At least six real tasks have been run or replayed.
- At least thirty material claims have been recorded.
- At least three failure classes recur across systems.
- At least one recurring failure class directly affects release control.
- Existing benchmark metrics do not already capture the failure.
- The failure can plausibly be reduced by an external claim transition authority.

If these conditions are not met, continue using existing AutoResearch systems and keep CairnLab as a validation harness rather than a kernel project.

For the semantic invalidation harness, a Go signal requires not only recurring failures but also a CairnLab counterfactual that changes a release-relevant state. If the harness only generates a report without changing claim authority, it is not sufficient differentiation.

## Benchmark-Lite Scope

The current benchmark direction is benchmark-lite, not a public leaderboard.

Benchmark-lite means:

- reuse public tasks such as ARC-Bench, PaperBench, MLE-bench, MLGym, BrowseComp-style tasks, or AutoResearchBench-style tasks;
- add a claim/evidence/governance rubric overlay;
- record claim cases and repeated failure modes;
- report whether the overlay changes release decisions or system rankings.

Benchmark-lite does not mean:

- building a general AutoResearch benchmark from scratch;
- building a public leaderboard now;
- replacing existing capability benchmarks;
- using LLM reviewer text as final authority.

## Current Execution Log

### 2026-06-04

- Strategy updated to validation-first.
- First target selected: AutoResearchClaw.
- Initial recommendation: run official or ARC-Bench tasks before custom research topics.
- Custom research topics must be converted to manifest-style tasks before use.
- CairnLab transition engine remains gated behind cross-system failure evidence.
