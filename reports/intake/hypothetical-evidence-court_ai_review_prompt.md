# AI Project Review Prompt Bundle

## Base Analyst Prompt

# AI New Project Analyst Prompt

You are an AI research-systems analyst helping maintain an AutoResearch landscape map.

Your task is to deeply inspect a new project and decide:

1. whether it has already been surveyed under another name, fork, alias, or related paper;
2. what multi-facet attributes it has;
3. which known projects are closest;
4. what is genuinely new;
5. whether it changes our target architecture for Accountable Research CI.

## Required Reading

Read more than the README. Inspect as many of the following as available:

- README / README_CN;
- paper / arXiv / Nature / blog / project page;
- `docs/`, `examples/`, `tests/`, `benchmarks/`;
- skill directories such as `skills/`, `.claude/skills/`, `prompts/`, `agents/`;
- pipeline/orchestrator code;
- artifact schemas, sample run folders, logs, reports;
- config files and environment setup;
- security warnings, API key requirements, network assumptions;
- license and third-party dependencies.

## Important Rule

Separate **advertised claims** from **demonstrated mechanisms**. A README promise is not the same as a runnable verifier, benchmark, or run ledger.

## Output Schema

Return a structured Markdown report with a YAML block at the end:

```yaml
project_id:
name:
aliases: []
urls: []
source_reading:
  read:
    - path_or_url:
      finding:
  not_accessible: []
duplicate_assessment:
  status: new | already_surveyed | fork_or_variant | uncertain
  matching_existing_projects: []
facets:
  research_fields: []
  starting_points: []
  primary_outputs: []
  workflow_scopes: []
  execution_depth: []
  verification_models: []
  accountability_features: []
  agent_topologies: []
  integration_styles: []
  maturity_signals: []
  risk_flags: []
  fit_to_our_target: []
  custom_facets: []
capability_evidence:
  advertised: []
  documented: []
  runnable: []
  tested: []
  externally_validated: []
closest_projects:
  - id:
    similarity:
    shared_facets: []
    differences: []
what_is_new:
  mechanisms: []
  process_design: []
  accountability_design: []
  domain_coverage: []
gaps_vs_our_target: []
architecture_lessons_for_us: []
recommended_registry_action: add_brief | add_medium | add_deep_dive | update_existing | watch_only
recommended_new_facets: []
```

## Our Target

Our target is not merely idea-to-paper. It is Accountable Research CI:

```text
paper / claim / method
→ experiment spec
→ code + data + environment + run
→ metrics + artifacts
→ claim verification
→ evidence-bound review
→ issue / patch / rerun / closure
```

Prioritize whether the project has claim ledger, experiment spec, run ledger, artifact lineage, provenance graph, producer/verifier/judge separation, and review issue loop.


## Project Intake

```yaml
id: hypothetical-evidence-court
name: Evidence Court Agent
aliases:
- EvidenceCourt
repo_url: https://github.com/example/evidence-court-agent
short_description: Hypothetical multi-agent court that turns paper claims into evidence objects, assigns prosecutors/defenders/judges,
  runs experiments, and closes review findings only after reruns pass.
claimed_capabilities:
- claim ledger
- run ledger
- judge panel
- review finding to issue and rerun closure
facets:
  research_fields:
  - metascience_integrity_review
  - ai_agent_systems
  starting_points:
  - paper_pdf_or_preprint
  - existing_code_repo
  primary_outputs:
  - verified_claims
  - review_or_rebuttal
  - reproduction_bundle
  workflow_scopes:
  - paper_to_reproduction
  - experiment_to_claim
  - claim_to_review_issue
  execution_depth:
  - fresh_container_reproduction
  verification_models:
  - claim_evidence_mapping
  - deterministic_verifier
  - adversarial_reviewer
  - artifact_hashing
  accountability_features:
  - claim_ledger
  - experiment_spec
  - run_ledger
  - artifact_lineage
  - provenance_graph
  - review_issue_loop
  agent_topologies:
  - producer_verifier_split
  - judge_panel
  - inspector_agent
  integration_styles:
  - cli
  - container_runtime
  maturity_signals:
  - early_research_prototype
  risk_flags:
  - llm_generated_code_execution
  - expensive_compute
  fit_to_our_target:
  - core_competitor
  custom_facets:
  - axis: agent_topologies
    value: prosecutor_defender_judge_court
    rationale: Uses legal-style adversarial roles beyond ordinary reviewer panels.
center_object: paper_claim
```

## First-Pass Heuristic Facets

```yaml
research_fields:
- ai_agent_systems
- metascience_integrity_review
- scientific_infrastructure
- software_engineering
starting_points:
- existing_code_repo
- paper_pdf_or_preprint
primary_outputs:
- code_repo_or_patch
- paper_draft_or_tex
- reproduction_bundle
- review_or_rebuttal
- training_run_results
- verified_claims
workflow_scopes:
- agent_runtime
- claim_to_review_issue
- experiment_to_claim
- literature_to_report
- paper_to_reproduction
- repo_to_experiment
- tracking_infrastructure
execution_depth:
- fresh_container_reproduction
verification_models:
- adversarial_reviewer
- artifact_hashing
- claim_evidence_mapping
- deterministic_verifier
- llm_reviewer
accountability_features:
- artifact_lineage
- claim_ledger
- experiment_spec
- provenance_graph
- review_issue_loop
- run_ledger
agent_topologies:
- human_ai_team
- inspector_agent
- judge_panel
- multi_agent_debate
- producer_verifier_split
integration_styles:
- cli
- container_runtime
- github_native
maturity_signals:
- early_research_prototype
- peer_reviewed_paper
- reproducible_examples
risk_flags:
- expensive_compute
- llm_generated_code_execution
- unverified_claim_generation
custom_facets:
- axis: agent_topologies
  value: prosecutor_defender_judge_court
  rationale: Uses legal-style adversarial roles beyond ordinary reviewer panels.
```

## Duplicate / Similarity Precheck

```json
{
  "input": {
    "id": "hypothetical-evidence-court",
    "name": "Evidence Court Agent",
    "aliases": [
      "EvidenceCourt"
    ],
    "repo_url": "https://github.com/example/evidence-court-agent",
    "short_description": "Hypothetical multi-agent court that turns paper claims into evidence objects, assigns prosecutors/defenders/judges, runs experiments, and closes review findings only after reruns pass.",
    "claimed_capabilities": [
      "claim ledger",
      "run ledger",
      "judge panel",
      "review finding to issue and rerun closure"
    ],
    "facets": {
      "research_fields": [
        "metascience_integrity_review",
        "ai_agent_systems"
      ],
      "starting_points": [
        "paper_pdf_or_preprint",
        "existing_code_repo"
      ],
      "primary_outputs": [
        "verified_claims",
        "review_or_rebuttal",
        "reproduction_bundle"
      ],
      "workflow_scopes": [
        "paper_to_reproduction",
        "experiment_to_claim",
        "claim_to_review_issue"
      ],
      "execution_depth": [
        "fresh_container_reproduction"
      ],
      "verification_models": [
        "claim_evidence_mapping",
        "deterministic_verifier",
        "adversarial_reviewer",
        "artifact_hashing"
      ],
      "accountability_features": [
        "claim_ledger",
        "experiment_spec",
        "run_ledger",
        "artifact_lineage",
        "provenance_graph",
        "review_issue_loop"
      ],
      "agent_topologies": [
        "producer_verifier_split",
        "judge_panel",
        "inspector_agent"
      ],
      "integration_styles": [
        "cli",
        "container_runtime"
      ],
      "maturity_signals": [
        "early_research_prototype"
      ],
      "risk_flags": [
        "llm_generated_code_execution",
        "expensive_compute"
      ],
      "fit_to_our_target": [
        "core_competitor"
      ],
      "custom_facets": [
        {
          "axis": "agent_topologies",
          "value": "prosecutor_defender_judge_court",
          "rationale": "Uses legal-style adversarial roles beyond ordinary reviewer panels."
        }
      ]
    },
    "center_object": "paper_claim"
  },
  "duplicate_status": "not_exact_duplicate",
  "exact_matches": [],
  "closest_projects": [
    {
      "score": 0.3689,
      "id": "evibound",
      "name": "EviBound",
      "depth": "deep",
      "summary": "通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。",
      "shared_legacy_domains": [],
      "shared_legacy_mechanisms": [],
      "shared_facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "verified_claims"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "verification_models": [
          "artifact_hashing",
          "claim_evidence_mapping",
          "deterministic_verifier"
        ],
        "accountability_features": [
          "artifact_lineage",
          "claim_ledger",
          "run_ledger"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "core_competitor"
        ]
      },
      "gaps_vs_our_target": [
        "不是完整 paper-to-code-to-review pipeline",
        "没有 reviewer-to-issue/rerun loop"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/accountability-layer.md"
    },
    {
      "score": 0.3199,
      "id": "scientistone",
      "name": "ScientistOne / Chain-of-Evidence",
      "depth": "deep",
      "summary": "Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。",
      "shared_legacy_domains": [],
      "shared_legacy_mechanisms": [],
      "shared_facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "verified_claims"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "verification_models": [
          "claim_evidence_mapping",
          "deterministic_verifier"
        ],
        "accountability_features": [
          "claim_ledger",
          "provenance_graph"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "core_competitor"
        ]
      },
      "gaps_vs_our_target": [
        "仍需扩到 run ledger、training reproduction 和 reviewer issue closure"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/accountability-layer.md"
    },
    {
      "score": 0.3136,
      "id": "march",
      "name": "MARCH",
      "depth": "brief",
      "summary": "Solver/Proposer/Checker 信息隔离，可减少 verifier 重复 generator 错误的自我确认偏差。",
      "shared_legacy_domains": [],
      "shared_legacy_mechanisms": [],
      "shared_facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "verified_claims"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "verification_models": [
          "adversarial_reviewer",
          "claim_evidence_mapping",
          "deterministic_verifier"
        ],
        "agent_topologies": [
          "judge_panel"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ]
      },
      "gaps_vs_our_target": [
        "不执行实验"
      ],
      "detailed_analysis": null
    },
    {
      "score": 0.281,
      "id": "pramana",
      "name": "Pramana / ClaimAttestation",
      "depth": "medium",
      "summary": "协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。",
      "shared_legacy_domains": [],
      "shared_legacy_mechanisms": [],
      "shared_facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "verified_claims"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "verification_models": [
          "claim_evidence_mapping",
          "deterministic_verifier"
        ],
        "accountability_features": [
          "provenance_graph"
        ]
      },
      "gaps_vs_our_target": [
        "不是完整自动科研系统"
      ],
      "detailed_analysis": null
    },
    {
      "score": 0.1536,
      "id": "catfish-agent",
      "name": "Catfish Agent",
      "depth": "brief",
      "summary": "通过故意注入结构化异议缓解多 agent silent agreement。",
      "shared_legacy_domains": [],
      "shared_legacy_mechanisms": [],
      "shared_facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "primary_outputs": [
          "review_or_rebuttal"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "verification_models": [
          "adversarial_reviewer"
        ],
        "agent_topologies": [
          "judge_panel"
        ]
      },
      "gaps_vs_our_target": [
        "不做研究执行"
      ],
      "detailed_analysis": null
    },
    {
      "score": 0.1392,
      "id": "reprozip-rocrate-prov",
      "name": "ReproZip / RO-Crate / W3C PROV",
      "depth": "medium",
      "summary": "环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。",
      "shared_legacy_domains": [],
      "shared_legacy_mechanisms": [],
      "shared_facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "primary_outputs": [
          "reproduction_bundle"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "verification_models": [
          "artifact_hashing",
          "claim_evidence_mapping"
        ],
        "accountability_features": [
          "artifact_lineage",
          "provenance_graph"
        ]
      },
      "gaps_vs_our_target": [
        "标准/工具层，不自动理解 paper/review"
      ],
      "detailed_analysis": null
    },
    {
      "score": 0.1295,
      "id": "prov-agent",
      "name": "PROV-AGENT",
      "depth": "medium",
      "summary": "扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。",
      "shared_legacy_domains": [],
      "shared_legacy_mechanisms": [],
      "shared_facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "primary_outputs": [
          "reproduction_bundle"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "accountability_features": [
          "provenance_graph"
        ]
      },
      "gaps_vs_our_target": [
        "provenance 标准层，不含研究执行语义"
      ],
      "detailed_analysis": null
    },
    {
      "score": 0.1277,
      "id": "agenthallu",
      "name": "AgentHallu",
      "depth": "brief",
      "summary": "评估多步 agent hallucination 及 step localization，适合设计错误归因机制。",
      "shared_legacy_domains": [],
      "shared_legacy_mechanisms": [],
      "shared_facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "primary_outputs": [
          "reproduction_bundle"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "accountability_features": [
          "provenance_graph"
        ]
      },
      "gaps_vs_our_target": [
        "benchmark 而非系统"
      ],
      "detailed_analysis": null
    }
  ],
  "novelty": {
    "novel_facets_vs_top5": {
      "research_fields": [],
      "starting_points": [
        "existing_code_repo"
      ],
      "primary_outputs": [
        "reproduction_bundle"
      ],
      "workflow_scopes": [
        "paper_to_reproduction"
      ],
      "execution_depth": [
        "fresh_container_reproduction"
      ],
      "verification_models": [],
      "accountability_features": [
        "experiment_spec",
        "review_issue_loop"
      ],
      "agent_topologies": [
        "inspector_agent",
        "producer_verifier_split"
      ],
      "integration_styles": [
        "cli",
        "container_runtime"
      ],
      "maturity_signals": [
        "early_research_prototype"
      ],
      "risk_flags": [
        "expensive_compute"
      ],
      "fit_to_our_target": [],
      "custom_facets": [
        {
          "axis": "agent_topologies",
          "value": "prosecutor_defender_judge_court",
          "rationale": "Uses legal-style adversarial roles beyond ordinary reviewer panels."
        }
      ]
    },
    "shared_facets_with_top5": {
      "research_fields": [
        "ai_agent_systems",
        "metascience_integrity_review"
      ],
      "starting_points": [
        "paper_pdf_or_preprint"
      ],
      "primary_outputs": [
        "review_or_rebuttal",
        "verified_claims"
      ],
      "workflow_scopes": [
        "claim_to_review_issue",
        "experiment_to_claim"
      ],
      "execution_depth": [],
      "verification_models": [
        "adversarial_reviewer",
        "artifact_hashing",
        "claim_evidence_mapping",
        "deterministic_verifier"
      ],
      "accountability_features": [
        "artifact_lineage",
        "claim_ledger",
        "provenance_graph",
        "run_ledger"
      ],
      "agent_topologies": [
        "judge_panel"
      ],
      "integration_styles": [],
      "maturity_signals": [],
      "risk_flags": [
        "llm_generated_code_execution"
      ],
      "fit_to_our_target": [
        "core_competitor"
      ]
    }
  },
  "recommended_depth": "deep_or_medium"
}
```

## Suggested Repository Reading Plan

For each accessible repository URL, inspect these paths if present:

- README.md / README_CN.md
- paper links, arXiv, Nature, blog, project docs
- docs/
- skills/ and .claude/skills/
- prompts/ and agents/
- pipeline/orchestrator source files
- schemas, contracts, artifacts, sample runs
- examples/ and tests/
- pyproject.toml, setup.py, requirements, Dockerfile
- security notes, API key requirements, network policy
- issues/releases if needed for maturity signals

Repository/source URLs to start from:
- https://github.com/example/evidence-court-agent

## Taxonomy Reminder

- `research_fields`: ai_ml_algorithm_research, ai_agent_systems, biomedicine_drug_discovery, wetlab_lab_automation, materials_chemistry_physics, mathematical_scientific_discovery, social_science_reproducibility, software_engineering, scientific_infrastructure, metascience_integrity_review, general_literature_research
- `starting_points`: topic_or_idea, research_question, paper_pdf_or_preprint, draft_paper, existing_code_repo, dataset_or_benchmark, experiment_plan, disease_or_candidate, workflow_task
- `primary_outputs`: literature_report, hypothesis_or_idea, experiment_plan, code_repo_or_patch, training_run_results, paper_draft_or_tex, review_or_rebuttal, verified_claims, reproduction_bundle, dashboard_or_lab_ui, benchmark_scores
- `workflow_scopes`: literature_to_report, idea_to_paper, idea_to_experiment, paper_to_code, paper_to_reproduction, repo_to_experiment, experiment_to_claim, claim_to_review_issue, lab_discovery_loop, agent_runtime, tracking_infrastructure
- `execution_depth`: text_only, code_generation_only, smoke_test, full_training_or_simulation, fresh_container_reproduction, wetlab_human_in_loop, benchmark_grading, tracking_only
- `verification_models`: citation_grounding, claim_evidence_mapping, deterministic_verifier, llm_reviewer, cross_model_review, adversarial_reviewer, human_gate, fresh_container_reproduction, artifact_hashing, attestation_protocol, rubric_grading
- `accountability_features`: claim_ledger, method_spec, experiment_spec, run_ledger, artifact_lineage, provenance_graph, agent_trace, decision_log, review_issue_loop, release_bundle, failure_registry
- `agent_topologies`: single_agent, orchestrator_workers, specialist_agents, multi_agent_debate, producer_verifier_split, judge_panel, inspector_agent, dynamic_roles, human_ai_team
- `integration_styles`: markdown_skills, python_package, cli, web_dashboard, mcp_or_tool_protocol, github_native, notebook_or_script, external_platform_dependency, container_runtime, mlops_tracking_stack
- `maturity_signals`: popular_github, peer_reviewed_paper, benchmark_or_leaderboard, reproducible_examples, active_development, external_evaluation, early_research_prototype, closed_or_gated_dependency, security_sensitive
- `risk_flags`: llm_generated_code_execution, unrestricted_network, secret_or_api_key_required, unverified_claim_generation, data_license_or_privacy, expensive_compute, supply_chain_risk, common_mode_model_failure
- `fit_to_our_target`: core_competitor, adjacent_competitor, plugin_layer, runtime_component, benchmark_reference, infrastructure_component, risk_model, domain_inspiration, watch_only

## Final Instruction

After reading the repo, produce the structured Markdown + YAML report requested in the base prompt. Explicitly mark any new attributes as `custom_facets` and say whether they should become taxonomy entries.
