# Verification Scope

This document records what the current repository verification does and does not
prove.

## Current Verification

The current engineering verification command is:

```bash
pytest -q -p no:rerunfailures
```

It verifies:

- domain models and builder helpers;
- synthetic semantic invalidation cases;
- in-memory `CairnRuntime`;
- local `.cairn/` store behavior;
- AutoResearchClaw and ARIS manifest adapter fixtures;
- deterministic adapter registry detection;
- transition authority blocking and allow paths;
- verifier certificate execution;
- decision trace package generation;
- selected CLI entrypoints.

It does not run real AutoResearchClaw, ARIS, or another AutoResearch runtime.
Fixture tests prove the CairnLab contracts and adapter mappings. They do not
prove that a live upstream framework emits complete metadata under real load.

## Real AutoResearch Validation

Real-framework validation should be a separate integration layer:

For AutoResearchClaw ACP/Codex runs, pin the Codex model with
`scripts/acpx-codex-gpt55-medium.sh` so recreated sessions do not fall back to a
local default before CairnLab imports evidence. See
`docs/AUTORESEARCHCLAW_CODEX_PINNING.md`.

On 2026-06-08, the AutoResearchClaw ML01 e2e run was resumed from Stage 15 with
`gpt-5.5[medium]` pinned through that wrapper. The run advanced through Stage 23
and exited with `Pipeline complete: 9/9 stages done, 0 failed`. Its final
`pipeline_summary.json` still reported `degraded=true`; Stage 20 reported
quality verdict `FAIL`; Stage 22 reported paper verification severity `REJECT`
and sanitization replaced 79 numbers. The e2e adapter now imports those signals
as evidence diagnostics and validation `failure_classes`, not as release
authority.

```mermaid
flowchart TD
    run["Run real AutoResearch task"]
    artifacts["Collect exported manifests<br/>runs, metrics, claims, gates"]
    adapter["CairnLab manifest adapter"]
    case["ClaimCase"]
    authority["TransitionAuthority"]
    trace["DecisionTracePackage"]
    finding["Gap or no-gap finding"]

    run --> artifacts --> adapter --> case
    case --> authority
    case --> trace
    authority --> finding
    trace --> finding
```

Recommended order:

1. Run the smallest documented AutoResearchClaw task.
2. Export or hand-normalize its manifest files into the current adapter shape.
3. Import with `cairn import-external`.
4. Run `cairn transition request` on material claims.
5. Run `cairn decision-trace` for any claim that appears release-ready.
6. Record whether CairnLab blocks or changes a release-relevant state that the
   upstream system leaves implicit.

This is required to validate the market/design thesis. It is not required for
unit-level correctness of the kernel modules.
