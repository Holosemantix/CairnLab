# Local Project Store Design

## Purpose

The local project store provides a simple `.cairn/` persistence adapter for
claim cases, imported objects, append-only transition events, and validation
reports.

Primary source files:

- `src/cairnlab/store.py`
- `src/cairnlab/engine.py`

## Owns

This module owns:

- `.cairn/` directory initialization;
- YAML storage for imported claims, evidence, relations, and cases;
- JSONL append-only event storage;
- loading stored objects for graph and projection modules;
- writing validation report artifacts.

## Does Not Own

This module must not own:

- claim transition policy;
- invalidation propagation;
- adapter detection;
- CLI formatting;
- verifier logic.

## Storage Layout

```text
.cairn/
  project.yaml
  objects/
    claims/*.yaml
    evidence/*.yaml
    relations/*.yaml
    cases/*.yaml
    policies/
  events/
    events.jsonl
  reports/
    validation_report.json
    validation_report.md
  cache/
```

## Import Paths

The store supports two import entry points:

- `import_case(path)`: load a YAML case file and persist it;
- `import_claim_case(case)`: persist an already-created in-memory `ClaimCase`.

The second path exists so adapters can remain filesystem-independent after they
return a case.

## Event Rule

Events are append-only JSONL records. Revert or transition operations must append
new `TransitionEvent` records rather than rewriting imported object YAML.

## Dependency Rules

`store.py` may depend on `models`, YAML, JSON, and path utilities. It must not
import `planner`, `graph`, adapters, or CLI.

`engine.py` may compose store, graph, projection, planner, validation, and
transition request logic for local project use. It should remain a thin facade.

## Extension Rules

When changing storage layout:

- update this document;
- update `docs/design/SYSTEM_DESIGN.md` if dependency direction changes;
- add migration or backward-compatibility tests if existing files are affected;
- keep imported object state immutable under revert operations.

Add a storage abstraction only after a second real backend exists.

## Tests

Current coverage:

- project initialization;
- case import;
- plan-only behavior does not write events;
- apply behavior appends events and projects state.
