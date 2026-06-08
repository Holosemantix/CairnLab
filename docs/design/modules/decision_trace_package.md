# Decision Trace Package Design

## Purpose

Decision trace packaging creates a reviewable bundle for consequential claim
decisions. It is the answer to a release-review question:

```text
What exactly allowed, blocked, or changed this claim state?
```

Primary source file:

- `src/cairnlab/trace_package.py`

## Owns

This module owns:

- collecting claim-local evidence;
- collecting verifier certificates, human gates, release decisions, and material dissent;
- collecting governance records such as risk and accountability;
- collecting relevant transition events;
- computing a stable export hash.

## Does Not Own

This module must not own:

- transition decisions;
- verifier execution;
- event persistence;
- adapter detection;
- experiment execution;
- LLM review.

It packages evidence for review. `TransitionAuthority` remains the gate.

## Flow

```mermaid
flowchart TD
    case["ClaimCase or .cairn/ project"]
    graph["Relations touching claim"]
    events["TransitionEvents"]
    governance["Risk, accountability,<br/>decision trace descriptors"]
    packager["DecisionTracePackager"]
    export["DecisionTracePackageExport<br/>stable export_hash"]
    reviewer["Human or external reviewer"]

    case --> graph --> packager
    case --> governance --> packager
    events --> packager
    packager --> export --> reviewer
```

## Public Contract

```python
from cairnlab import DecisionTracePackager

package = DecisionTracePackager.from_case(case).build(
    "claim:C1",
    transition="release",
)
```

Local project facade:

```python
package = CairnProject.open(".").decision_trace_package(
    "claim:C1",
    transition="release",
)
```

CLI:

```bash
cairn decision-trace claim:C1 --transition release --json
```

## Contents

The exported package includes:

- package descriptor with `export_hash`;
- claim payload;
- directly related evidence;
- verifier certificate evidence;
- human gates and release decisions;
- material dissent records;
- relations touching the claim or included evidence;
- transition events touching the claim or included evidence;
- governance objects from claim-level or case-level metadata.

## Dependency Rules

`trace_package.py` may depend on:

- `models`;
- `graph`;
- deterministic hash utilities.

It must not depend on:

- `store`;
- `engine`;
- `cli`;
- adapters;
- verifiers;
- transition authority.

## Extension Rules

When adding package content:

- keep it structured;
- include stable object IDs;
- preserve machine-readable evidence and governance fields;
- update the design doc and tests;
- avoid fetching data from host runtimes.

## Tests

Current coverage:

- package includes release authority inputs;
- export hash is stable for identical inputs;
- local project package includes transition events;
- CLI JSON output includes claim and export hash.
