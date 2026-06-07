# Adapter System Design

## Purpose

Adapters translate external AutoResearch project metadata into portable
`ClaimCase` objects. They make CairnLab reusable without requiring host projects
to adopt CairnLab storage, CLI, prompts, or execution runtime.

Primary source files:

- `src/cairnlab/adapters/base.py`
- `src/cairnlab/adapters/registry.py`
- `src/cairnlab/adapters/autoresearchclaw_manifest.py`
- `src/cairnlab/adapters/aris_manifest.py`
- `docs/ADAPTER_CONTRACT.md`

## Owns

This module owns:

- the `AutoResearchAdapter` protocol;
- adapter diagnostics;
- dependency-free manifest export into `ClaimCase`;
- conservative adapter detection;
- deterministic adapter selection.

## Does Not Own

This module must not own:

- claim transition authority;
- invalidation planning;
- verifier execution;
- host project mutation;
- network access;
- external project package imports.

Adapters are translators, not authorities.

## Registry Behavior

The registry is intentionally static:

```python
from cairnlab.adapters import detect_adapters, select_adapter, export_case

matches = detect_adapters(path)
adapter = select_adapter(path, adapter_name="auto")
export = export_case(path)
```

`adapter_name="auto"` succeeds only when exactly one adapter matches. Zero
matches or multiple matches raise `AdapterSelectionError`.

This avoids silent imports from ambiguous projects and keeps detection behavior
scriptable for external users.

## Built-In Adapters

Current built-ins:

- `autoresearchclaw-manifest`
- `aris-manifest`

Both read structured JSON or JSONL manifests only. They do not import
AutoResearchClaw, ARIS, or any host runtime.

## CLI Surface

```bash
cairn adapter detect path/to/project --json
cairn import-external path/to/project --adapter auto --path .
```

CLI import calls the adapter, receives a `ClaimCase`, and passes it to the same
local project import path used by ordinary case files.

## Dependency Rules

Adapters may depend on:

- `models`;
- `builder`;
- standard-library file and JSON utilities.

Adapters must not depend on:

- `store`;
- `engine`;
- `cli`;
- host project packages;
- planner internals.

## Extension Rules

When adding an adapter:

- implement `AutoResearchAdapter`;
- make `detect()` conservative;
- preserve stable IDs, URIs, hashes, actors, authority, scope, and rationale;
- emit diagnostics for missing structured metadata;
- add fixture-based tests;
- update `docs/ADAPTER_CONTRACT.md` and this document.

Do not add a plugin mechanism until at least one adapter lives outside this
repository and needs packaging support.

## Tests

Current coverage:

- AutoResearchClaw manifest detection and export;
- ARIS manifest detection and export;
- registry name listing;
- auto-detection success;
- no-match failure;
- ambiguous match failure;
- CLI `adapter detect`;
- CLI `import-external`.
