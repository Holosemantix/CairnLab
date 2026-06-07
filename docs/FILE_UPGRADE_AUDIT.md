# File Upgrade Audit: v0.2 → v0.3

This audit records how `autoresearch_landscape_skill_repo_v0_2.zip` was transformed into the CairnLab v0.3 package.

## Summary

The v0.2 archive was a landscape survey skill repo aimed at **Accountable Research CI**. v0.3 preserves the survey skill but upgrades the repository into a CairnLab design package centered on a **Research Claim Kernel**.

## Keep unchanged or nearly unchanged

| Path | Decision | Reason |
| --- | --- | --- |
| `skills/autoresearch-landscape-survey/scripts/check_new_project.py` | Modify lightly | Keep duplicate/similarity logic; extend facet axes. |
| `skills/autoresearch-landscape-survey/scripts/classify_project_from_yaml.py` | Modify lightly | Keep heuristic classifier; add claim-kernel/governance keyword rules. |
| `skills/autoresearch-landscape-survey/scripts/prepare_ai_project_review.py` | Keep | Still useful for deep source reading. |
| `skills/autoresearch-landscape-survey/scripts/prepare_ai_report_prompt.py` | Keep | Still useful for AI-assisted synthesis. |
| `skills/autoresearch-landscape-survey/scripts/render_project_overview.py` | Modify lightly | Keep report generator; add new axes. |
| `skills/autoresearch-landscape-survey/scripts/render_landscape_report.py` | Modify | Update executive summary and target positioning. |
| `skills/autoresearch-landscape-survey/tests/test_check_new_project.py` | Keep | Existing tests remain valid. |
| `skills/autoresearch-landscape-survey/references/deep-dives/*` | Keep | Prior research remains useful. |
| `reports/intake/*` | Keep | Useful examples for future intake review. |
| `THIRD_PARTY_LICENSES.md` | Keep | License notes still needed. |

## Modify

| Path | Change |
| --- | --- |
| `README.md` | Rewritten from landscape-skill repo to CairnLab Research Claim Kernel package. |
| `AGENTS.md` | Rewritten with strategic boundary, role/risk/accountability instructions. |
| `pyproject.toml` | Updated package metadata to `cairnlab`, CLI `cairn`, and dependencies. |
| `data/taxonomy.yaml` | Upgraded to v0.3 with kernel primitives, lifecycle stages, governance controls, risk tiers, and governance alignment axes. |
| `data/project_registry.yaml` | Added/updated high-priority projects and claim-kernel facets. |
| `data/project_registry.json` | Regenerated from YAML. |
| `reports/landscape_report.md` | Regenerated with Research Claim Kernel target. |
| `reports/project_overview.md` | Regenerated with new axes. |
| `skills/autoresearch-landscape-survey/SKILL.md` | Updated to explain CairnLab-oriented landscape maintenance. |
| `skills/autoresearch-landscape-survey/references/our-positioning.md` | Replaced Accountable Research CI with Research Claim Kernel positioning. |
| `skills/autoresearch-landscape-survey/references/taxonomy.md` | Updated with new axes. |
| `skills/autoresearch-landscape-survey/references/comparison-axes.md` | Updated with kernel/governance comparison questions. |
| `skills/autoresearch-landscape-survey/references/new-project-checklist.md` | Updated to detect claim-state-kernel competitors. |

## Add

| Path | Purpose |
| --- | --- |
| `docs/PROJECT_POSITIONING.md` | Strategic non-derivative boundary. |
| `docs/KERNEL_SPEC.md` | Claim lifecycle kernel primitives and state machine. |
| `docs/ARCHITECTURE.md` | CairnLab system architecture. |
| `docs/GOVERNANCE_ALIGNMENT.md` | Mapping of traceability/accountability/risk/role requirements into kernel objects. |
| `docs/EU_AI_ACT_PRACTICE_NOTES.md` | Practical lessons from EU AI Act-style logging and human oversight. |
| `docs/LANDSCAPE_IMPLICATIONS.md` | Why CairnLab should be a kernel, not another agent. |
| `docs/ROADMAP.md` | Development plan and acceptance criteria. |
| `docs/CODEX_BOOTSTRAP_PROMPT.md` | First implementation prompt. |
| `docs/FILE_UPGRADE_AUDIT.md` | This audit. |
| `examples/minimal/claim.yaml` | Minimal claim/evidence/policy/governance example. |
| `schemas/*.schema.json` | Initial schema placeholders for kernel objects. |
| `skills/autoresearch-landscape-survey/examples/new_project_cairnlab_claim_kernel.yaml` | Intake example for our own positioning. |

## Remove

No files were physically removed in v0.3. The previous survey assets remain valuable for competitive intelligence. Several concepts are deprecated in wording:

- `Accountable Research CI` is now a secondary description.
- `Research OS` is avoided as primary positioning because DeepScientist and Claw AI Lab already occupy OS/workspace/dashboard narratives.
- `claim ledger` alone is not treated as sufficient differentiation.

## Final target after upgrade

```text
CairnLab = Research Claim Kernel
= claim lifecycle state machine
+ evidence policy DSL
+ verifier-issued transition certificates
+ append-only state transition log
+ governance/risk/responsibility controls
+ human liability gates
+ decision trace packages
```
