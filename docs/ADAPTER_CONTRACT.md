# Adapter Contract

CairnLab adapters translate external AutoResearch project metadata into a portable `ClaimCase`.

The adapter contract is deliberately small. A host project should not need to adopt CairnLab's CLI, `.cairn/` store, execution runtime, LLM prompts, or report layout.

## Design Rule

Adapters are translators, not authorities.

They may read host artifacts and emit:

- claims;
- evidence items;
- verifier or audit outputs;
- human approval gates;
- release decisions;
- typed relations between those objects;
- diagnostics about missing metadata.

They must not decide whether a claim is released, downgraded, or retracted. That decision remains in the runtime and planner.

## Minimal Python Flow

```python
from cairnlab import CairnRuntime
from cairnlab.adapters import AutoResearchAdapter

adapter: AutoResearchAdapter = ...
export = adapter.export_case(project_path)

runtime = CairnRuntime.from_case(export.case)
plan = runtime.plan_revert(
    "run:exp_007",
    reason="metric computed on wrong split",
)
events = runtime.events_from_plan(plan)
```

This flow does not touch the filesystem after the adapter returns its `ClaimCase`.

## Protocol

```python
class AutoResearchAdapter(Protocol):
    name: str

    def detect(self, path: Path) -> bool:
        ...

    def export_case(self, path: Path) -> AdapterExportResult:
        ...
```

`detect` should be conservative. It should return true only when the adapter sees enough host metadata to build at least one claim or evidence object.

`export_case` returns:

```python
class AdapterExportResult(BaseModel):
    case: ClaimCase
    diagnostics: list[AdapterDiagnostic]
```

Diagnostics should report missing or ambiguous source metadata without blocking export unless the case would be structurally invalid.

## Builder API

Adapters should usually use `ClaimCaseBuilder` instead of manually constructing every model.

```python
from cairnlab import ClaimCaseBuilder

case = (
    ClaimCaseBuilder(
        case_id="autoresearchclaw-run-001",
        source_system="autoresearchclaw",
        stress_scenario="imported_run",
    )
    .add_claim(
        "claim:C1",
        "Method A improves baseline B on Dataset X.",
        state="released",
    )
    .add_evidence(
        "run:exp_007",
        "run",
        uri="file://artifacts/rc-001/experiment_summary.json",
        hash="sha256:...",
    )
    .add_evidence(
        "metric:exp_007.primary",
        "metric",
        metadata={"metric_name": "primary_metric", "value": 0.923},
    )
    .add_relation("run:exp_007", "metric:exp_007.primary", "computed", criticality="critical")
    .add_support("metric:exp_007.primary", "claim:C1")
    .add_human_gate(
        "human_gate:H1",
        "claim:C1",
        "human:alice",
        authority="project_owner",
        scope={"claim": "claim:C1", "run": "run:exp_007"},
        rationale="Approved after reviewing experiment summary.",
    )
    .add_release_decision("release_decision:R1", "claim:C1", "human:alice")
    .build()
)
```

## Relation Direction

For ordinary dependencies, `source` is upstream and `target` depends on it:

```text
run -> metric -> claim -> paper_section
```

Authority objects also point to the claim they authorize:

```text
human_gate -> claim
release_decision -> claim
verifier_certificate -> claim
```

The runtime propagates both:

- upstream invalidation to dependent claims;
- affected claim state back to gates, release decisions, and verifier certificates that must be reopened or invalidated.

## AutoResearchClaw Mapping

The first AutoResearchClaw adapter should treat the following as metadata sources, not runtime dependencies:

| Host object | CairnLab object |
| --- | --- |
| `experiment_summary.json` | `run`, `metric`, `artifact` evidence |
| `VerifiedRegistry` exported values | `verifier_certificate` or metric evidence |
| result tables | `paper_section` evidence |
| HITL intervention records | `human_gate` evidence |
| final deliverable decision | `release_decision` evidence |
| stage artifacts | `artifact` evidence |

The adapter should not import AutoResearchClaw classes in the core package. If needed later, a plugin package can depend on AutoResearchClaw.

### Current Manifest Adapter

The in-tree `AutoResearchClawManifestAdapter` is the first minimal implementation.

It detects and reads:

```text
experiment_summary.json
experiment_summary_best.json
results/experiment_summary.json
deliverables/experiment_summary.json
hitl/approval.json              optional
release_decision.json           optional
```

It maps:

- the experiment summary file to a `run` evidence item;
- the primary metric to a `metric` evidence item;
- listed artifacts to `artifact` evidence items;
- an exported claim object, if present, to `Claim`;
- `hitl/approval.json` to a `human_gate`;
- `release_decision.json` to a `release_decision`.

It does not import AutoResearchClaw, run its pipeline, repair experiments, parse papers, or infer human approval when approval metadata is missing.

## ARIS Mapping

The first ARIS adapter should treat these sources as metadata:

| Host object | CairnLab object |
| --- | --- |
| `research-wiki/claims` | `Claim` |
| experiment pages | `run`, `metric`, `artifact` evidence |
| `/result-to-claim` status | claim base state or transition seed |
| `EXPERIMENT_AUDIT.json` | `verifier_certificate` evidence |
| `PAPER_CLAIM_AUDIT.json` | `verifier_certificate` evidence |
| `CITATION_AUDIT.json` | citation verifier evidence |
| `.aris/meta/events.jsonl` | optional source event evidence |
| human checkpoint metadata | `human_gate` evidence |

The adapter should preserve ARIS audit verdicts as machine-readable metadata and avoid turning them into prose-only notes.

### Current ARIS Manifest Adapter

The in-tree `ArisManifestAdapter` is the first minimal ARIS implementation.

It detects and reads:

```text
research-wiki/claims/*.json
research-wiki/experiments/*.json
research-wiki/graph/edges.jsonl
EXPERIMENT_AUDIT.json            optional
PAPER_CLAIM_AUDIT.json           optional
CITATION_AUDIT.json              optional
KILL_ARGUMENT.json               optional
.aris/human_gate.json            optional
```

It maps:

- `research-wiki/claims/*.json` to `Claim`;
- `research-wiki/experiments/*.json` to `run`, `metric`, and `artifact` evidence;
- `research-wiki/graph/edges.jsonl` to typed CairnLab relations;
- ARIS audit JSON files to `verifier_certificate` evidence;
- `.aris/human_gate.json` to a `human_gate`.

It does not parse arbitrary wiki Markdown, run ARIS skills, call reviewers, mutate `.aris/meta/events.jsonl`, infer missing human approvals, or decide claim lifecycle authority.

## Reliability Requirements

Adapters should:

- produce stable object IDs;
- attach stable URIs or hashes whenever possible;
- preserve actor, authority, scope, and rationale for human gates;
- preserve verifier status and input references;
- emit diagnostics for missing evidence bindings;
- avoid network access by default;
- never mutate the host project while exporting a case.

## Non-Goals

Adapters must not:

- run new experiments;
- parse or rewrite papers;
- call LLM reviewers;
- restore workspaces;
- hide material dissent;
- silently drop unsupported claims;
- change host project state.
