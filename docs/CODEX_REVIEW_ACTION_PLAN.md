# Codex Review Action Plan

This document tracks hardening work from the external CairnLab review. It is a
design-control checklist, not a product roadmap.

## Applied In This Pass

- Added a validation evidence ledger so real framework runs, fixture/contract
  tests, synthetic cases, material claims, and release-control failures are
  counted separately.
- Split upstream `observed_state` from CairnLab `authority_state`; legacy
  imported `state` and `status` map to `observed_state`.
- Added transition apply persistence through the project facade and CLI. Requests
  remain plan-only unless `--apply` is set; blocked decisions require
  `--record-blocked` to enter the event log.
- Hardened verifier certificates so naked `metadata.verdict: pass`, wrong-claim
  certificates, missing inputs, and invalidated inputs cannot authorize
  verification.
- Required explicit actor override authority for `force=True` release overrides.
- Updated schemas, fixtures, examples, tests, and design docs for the changed
  public semantics.

## Still Deferred

These items remain intentionally deferred until the lighter foundations are
stable:

- minimal evidence policy registry;
- relation propagation semantics table;
- event hash chain verification;
- manifest hygiene scripts;
- schema generation or schema drift automation.

The deferral is deliberate. CairnLab should stay a lightweight, reusable claim
lifecycle transition authority. It should not grow into a plugin-heavy research
OS or another AutoResearch agent.
