# ARIS E2E Smoke

This smoke test validates CairnLab's ARIS adapter against a minimal external
artifact chain without importing ARIS as a runtime dependency.

It is intentionally not a full ARIS paper-generation run. Full ARIS workflows
need a host agent CLI, reviewer routing, and model credentials. The smoke test
uses ARIS deterministic helpers to exercise the artifact contract CairnLab
cares about:

- `research_wiki.py init`;
- `research_wiki.py add_edge`;
- `evidence_check.py`;
- `verify_paper_audits.sh`;
- ARIS research-wiki claim and experiment pages;
- paper audit JSON files;
- `.aris/audit-verifier-report.json`;
- `.review.json` sidecar;
- `.aris/human_gate.json`.

Run:

```bash
python scripts/run_aris_e2e_smoke.py \
  --aris-repo /path/to/Auto-claude-code-research-in-sleep
```

Expected result:

- ARIS verifier report exits cleanly at `assurance=submission`;
- CairnLab imports one claim, run evidence, metric evidence, audit verifier
  evidence, reviewer evidence, and a human gate;
- invalidating `run:exp_001` challenges the claim, invalidates verifier
  evidence, and requires human reapproval.

The boundary remains strict: ARIS produces external evidence, but CairnLab
decides lifecycle transitions through its own authority layer.
