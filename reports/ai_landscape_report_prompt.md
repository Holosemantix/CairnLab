# AI Landscape Report Writing Bundle

# AI Landscape Report Synthesis Prompt

You are helping synthesize an AutoResearch ecosystem report from structured registry data.

Use the provided project registry, taxonomy, overview tables, and deep-dive notes. Produce a report that:

- treats categories as non-orthogonal facets;
- groups projects from several perspectives: field, workflow scope, verification/accountability, agent topology, and maturity;
- clearly identifies which projects are direct competitors, adjacent systems, plugin layers, infrastructure, benchmarks, and risk-model references;
- emphasizes our differentiator: Accountable Research CI with claim ledger, run ledger, artifact lineage, provenance graph, judge separation, and review issue loop;
- identifies new/open facets that need taxonomy updates;
- distinguishes brief survey entries from detailed deep dives.

Avoid overclaiming. If a project only advertises a feature but does not demonstrate it with code, tests, artifacts, or benchmark, say so.

Recommended sections:

1. Executive Summary
2. Why Facets Instead of Layers
3. Field-Based Map
4. Workflow/Process Map
5. Verification and Accountability Map
6. Multi-Agent Architecture Map
7. Detailed Deep-Dive Projects
8. Brief/Watchlist Projects
9. Gaps and Opportunities for Accountable Research CI
10. Update Log and Next Projects to Investigate


## Structured Registry Snapshot

```json
{
  "taxonomy_version": "0.2.0",
  "registry_version": "0.2.0",
  "project_count": 49,
  "projects": [
    {
      "id": "aris",
      "name": "ARIS / Auto-claude-code-research-in-sleep",
      "depth": "deep",
      "summary": "Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "code_repo_or_patch",
          "experiment_plan",
          "review_or_rebuttal",
          "training_run_results"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "experiment_to_claim",
          "idea_to_experiment",
          "idea_to_paper",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "full_training_or_simulation"
        ],
        "verification_models": [
          "adversarial_reviewer",
          "citation_grounding",
          "claim_evidence_mapping",
          "cross_model_review",
          "deterministic_verifier"
        ],
        "accountability_features": [
          "agent_trace"
        ],
        "agent_topologies": [
          "judge_panel",
          "orchestrator_workers"
        ],
        "integration_styles": [
          "cli",
          "markdown_skills",
          "python_package"
        ],
        "maturity_signals": [
          "active_development",
          "reproducible_examples"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "adjacent_competitor"
        ],
        "custom_facets": []
      },
      "our_relevance": 5,
      "differentiators": [
        "轻量可移植，无固定 runtime stage engine",
        "Research Wiki 记录 Paper/Idea/Experiment/Claim 图",
        "reviewer 和 verifier 与 executor 分权"
      ],
      "gaps_vs_our_target": [
        "更偏自己的研究推进和 paper improvement",
        "claim/evidence/provenance 多依赖 Markdown/wiki 约定，强类型 ledger 还不够硬",
        "外部 paper-to-reproduction 不是核心入口"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/aris.md"
    },
    {
      "id": "autoresearchclaw",
      "name": "AutoResearchClaw",
      "depth": "deep",
      "summary": "万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research",
          "metascience_integrity_review",
          "software_engineering"
        ],
        "starting_points": [
          "topic_or_idea"
        ],
        "primary_outputs": [
          "hypothesis_or_idea",
          "paper_draft_or_tex",
          "review_or_rebuttal",
          "training_run_results"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "claim_to_review_issue",
          "idea_to_experiment",
          "idea_to_paper",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [
          "artifact_hashing",
          "citation_grounding",
          "claim_evidence_mapping",
          "cross_model_review",
          "human_gate"
        ],
        "accountability_features": [
          "artifact_lineage",
          "claim_ledger",
          "failure_registry"
        ],
        "agent_topologies": [
          "human_ai_team",
          "orchestrator_workers"
        ],
        "integration_styles": [
          "cli",
          "container_runtime",
          "python_package"
        ],
        "maturity_signals": [
          "active_development",
          "reproducible_examples"
        ],
        "risk_flags": [
          "llm_generated_code_execution",
          "unverified_claim_generation"
        ],
        "fit_to_our_target": [
          "adjacent_competitor"
        ],
        "custom_facets": []
      },
      "our_relevance": 5,
      "differentiators": [
        "stage/context skill matcher",
        "6 类 HITL 模式",
        "PIVOT/REFINE 决策",
        "多领域 execution agents 与 ARC-Bench"
      ],
      "gaps_vs_our_target": [
        "中心对象仍是 topic/idea 到 paper，不是外部论文 claim-by-claim 复现审计",
        "claim verification 更偏防生成文本幻觉，不等同于 run-level evidence ledger",
        "review finding 尚未系统化为 issue/patch/rerun/closure"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/autoresearchclaw.md"
    },
    {
      "id": "evibound",
      "name": "EviBound",
      "depth": "deep",
      "summary": "通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。",
      "facets": {
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
        "execution_depth": [
          "smoke_test"
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
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper",
          "reproducible_examples"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "core_competitor",
          "risk_model"
        ],
        "custom_facets": []
      },
      "our_relevance": 5,
      "differentiators": [
        "research integrity as architecture",
        "claim 必须有 queryable run/artifact/status"
      ],
      "gaps_vs_our_target": [
        "不是完整 paper-to-code-to-review pipeline",
        "没有 reviewer-to-issue/rerun loop"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/accountability-layer.md"
    },
    {
      "id": "paperbench",
      "name": "PaperBench",
      "depth": "deep",
      "summary": "评估 agent 复现 ICML 论文的关键 benchmark；证明 paper reproduction 仍很难。",
      "facets": {
        "research_fields": [
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "literature_report",
          "paper_draft_or_tex",
          "reproduction_bundle",
          "training_run_results"
        ],
        "workflow_scopes": [
          "paper_to_code",
          "paper_to_reproduction",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "benchmark_grading",
          "fresh_container_reproduction",
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [
          "rubric_grading"
        ],
        "accountability_features": [
          "release_bundle"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "container_runtime"
        ],
        "maturity_signals": [
          "benchmark_or_leaderboard",
          "reproducible_examples"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "benchmark_reference"
        ],
        "custom_facets": []
      },
      "our_relevance": 5,
      "differentiators": [
        "author co-developed rubrics",
        "fresh container reproduction",
        "细粒度评分"
      ],
      "gaps_vs_our_target": [
        "是 benchmark，不是平台",
        "不提供 reviewer-to-issue loop"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/paperbench-and-reproduction-benchmarks.md"
    },
    {
      "id": "scientistone",
      "name": "ScientistOne / Chain-of-Evidence",
      "depth": "deep",
      "summary": "Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。",
      "facets": {
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
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "attestation_protocol",
          "claim_evidence_mapping",
          "deterministic_verifier"
        ],
        "accountability_features": [
          "agent_trace",
          "claim_ledger",
          "decision_log",
          "provenance_graph"
        ],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper",
          "reproducible_examples"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "core_competitor",
          "risk_model"
        ],
        "custom_facets": []
      },
      "our_relevance": 5,
      "differentiators": [
        "最接近 claim-level evidence 的新竞品/研究方向"
      ],
      "gaps_vs_our_target": [
        "仍需扩到 run ledger、training reproduction 和 reviewer issue closure"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/accountability-layer.md"
    },
    {
      "id": "ablationbench",
      "name": "AblationBench",
      "depth": "brief",
      "summary": "评估 LM 系统找回/规划 ablation 的能力，可作为 reviewer-to-experiment 机制参考。",
      "facets": {
        "research_fields": [
          "ai_ml_algorithm_research",
          "metascience_integrity_review",
          "software_engineering"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "experiment_plan",
          "review_or_rebuttal",
          "training_run_results"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "repo_to_experiment"
        ],
        "execution_depth": [],
        "verification_models": [
          "adversarial_reviewer",
          "rubric_grading"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "judge_panel"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "benchmark_or_leaderboard",
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "ablation planning benchmark"
      ],
      "gaps_vs_our_target": [
        "没有 execution/provenance 层"
      ],
      "detailed_analysis": null
    },
    {
      "id": "ai-scientist-v2",
      "name": "The AI Scientist v2",
      "depth": "medium",
      "summary": "自动科学发现代表项目，v2 减少模板依赖，强调 agentic tree search；需要重点关注安全和可验证性不足。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research"
        ],
        "starting_points": [
          "topic_or_idea"
        ],
        "primary_outputs": [
          "hypothesis_or_idea",
          "paper_draft_or_tex",
          "training_run_results"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "idea_to_paper",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [
          "citation_grounding"
        ],
        "accountability_features": [
          "failure_registry"
        ],
        "agent_topologies": [
          "multi_agent_debate",
          "orchestrator_workers"
        ],
        "integration_styles": [
          "cli",
          "container_runtime",
          "python_package"
        ],
        "maturity_signals": [
          "active_development",
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "adjacent_competitor"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "假设/实验/论文全链条",
        "强调去模板化"
      ],
      "gaps_vs_our_target": [
        "不是 paper-first reproduction",
        "缺强 claim/run/evidence/review issue ledger"
      ],
      "detailed_analysis": null
    },
    {
      "id": "autoexperiment",
      "name": "AutoExperiment",
      "depth": "medium",
      "summary": "通过 progressive code masking 测 agent 从 paper + masked codebase 中补全代码、执行实验、复现结果。",
      "facets": {
        "research_fields": [
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "experiment_plan",
          "training_run_results"
        ],
        "workflow_scopes": [
          "paper_to_code",
          "paper_to_reproduction",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "benchmark_grading",
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [
          "rubric_grading"
        ],
        "accountability_features": [
          "failure_registry"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "container_runtime"
        ],
        "maturity_signals": [
          "benchmark_or_leaderboard"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "benchmark_reference"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "动态交互/debugging agent 明显优于固定 harness"
      ],
      "gaps_vs_our_target": [
        "benchmark 而非系统，缺 review/accountability layer"
      ],
      "detailed_analysis": null
    },
    {
      "id": "autop2c",
      "name": "AutoP2C",
      "depth": "medium",
      "summary": "从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。",
      "facets": {
        "research_fields": [
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "code_repo_or_patch",
          "experiment_plan",
          "literature_report",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "paper_to_code",
          "paper_to_reproduction",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [
          "failure_registry"
        ],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "blueprint extraction",
        "hierarchical decomposition",
        "iterative debugging"
      ],
      "gaps_vs_our_target": [
        "可运行 repo 不等于 claim-level verified reproduction"
      ],
      "detailed_analysis": null
    },
    {
      "id": "claw-ai-lab",
      "name": "Claw AI Lab",
      "depth": "medium",
      "summary": "把自动科研变成 dashboard 化 AI lab，突出 artifact inspector、rollback/resume、reproduce mode。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "experiment_plan",
          "review_or_rebuttal",
          "training_run_results"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "idea_to_paper",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [
          "artifact_hashing",
          "citation_grounding",
          "human_gate"
        ],
        "accountability_features": [
          "artifact_lineage"
        ],
        "agent_topologies": [
          "human_ai_team",
          "multi_agent_debate"
        ],
        "integration_styles": [
          "cli",
          "container_runtime",
          "python_package"
        ],
        "maturity_signals": [
          "active_development",
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "adjacent_competitor"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "交互式实验室",
        "Claw-Code Harness",
        "可视化监控"
      ],
      "gaps_vs_our_target": [
        "UI/协作强，claim-level accountability 不够硬"
      ],
      "detailed_analysis": null
    },
    {
      "id": "dvc",
      "name": "DVC",
      "depth": "brief",
      "summary": "数据/模型/管线版本化，适合作为数据和大 artifact 版本层。",
      "facets": {
        "research_fields": [
          "scientific_infrastructure",
          "software_engineering"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "reproduction_bundle",
          "training_run_results"
        ],
        "workflow_scopes": [
          "tracking_infrastructure"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "artifact_hashing"
        ],
        "accountability_features": [
          "artifact_lineage",
          "release_bundle",
          "run_ledger"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "mlops_tracking_stack"
        ],
        "maturity_signals": [],
        "risk_flags": [],
        "fit_to_our_target": [
          "infrastructure_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "Git-like data/model versioning"
      ],
      "gaps_vs_our_target": [
        "不理解 paper claim"
      ],
      "detailed_analysis": null
    },
    {
      "id": "march",
      "name": "MARCH",
      "depth": "brief",
      "summary": "Solver/Proposer/Checker 信息隔离，可减少 verifier 重复 generator 错误的自我确认偏差。",
      "facets": {
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
        "execution_depth": [],
        "verification_models": [
          "adversarial_reviewer",
          "claim_evidence_mapping",
          "deterministic_verifier",
          "llm_reviewer"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "judge_panel"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "risk_model"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "checker 不读 solver 叙事，只验证 atomic propositions"
      ],
      "gaps_vs_our_target": [
        "不执行实验"
      ],
      "detailed_analysis": null
    },
    {
      "id": "mlflow",
      "name": "MLflow",
      "depth": "brief",
      "summary": "run tracking、params/metrics/artifacts、model registry；适合作为 Run Ledger 后端。",
      "facets": {
        "research_fields": [
          "scientific_infrastructure",
          "software_engineering"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "training_run_results"
        ],
        "workflow_scopes": [
          "tracking_infrastructure"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "artifact_hashing"
        ],
        "accountability_features": [
          "artifact_lineage",
          "run_ledger"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "mlops_tracking_stack"
        ],
        "maturity_signals": [],
        "risk_flags": [],
        "fit_to_our_target": [
          "infrastructure_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "实验 tracking 成熟"
      ],
      "gaps_vs_our_target": [
        "无 paper/claim/review 语义"
      ],
      "detailed_analysis": null
    },
    {
      "id": "pramana",
      "name": "Pramana / ClaimAttestation",
      "depth": "medium",
      "summary": "协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。",
      "facets": {
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
        "execution_depth": [],
        "verification_models": [
          "attestation_protocol",
          "claim_evidence_mapping",
          "deterministic_verifier"
        ],
        "accountability_features": [
          "agent_trace",
          "decision_log",
          "provenance_graph"
        ],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [
          "risk_model"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "measurement/inference/analogy/citation attestation"
      ],
      "gaps_vs_our_target": [
        "不是完整自动科研系统"
      ],
      "detailed_analysis": null
    },
    {
      "id": "prov-agent",
      "name": "PROV-AGENT",
      "depth": "medium",
      "summary": "扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "reproduction_bundle"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "execution_depth": [],
        "verification_models": [
          "attestation_protocol"
        ],
        "accountability_features": [
          "agent_trace",
          "decision_log",
          "provenance_graph"
        ],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [
          "infrastructure_component",
          "risk_model"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "agent decision-level provenance"
      ],
      "gaps_vs_our_target": [
        "provenance 标准层，不含研究执行语义"
      ],
      "detailed_analysis": null
    },
    {
      "id": "reprozip-rocrate-prov",
      "name": "ReproZip / RO-Crate / W3C PROV",
      "depth": "medium",
      "summary": "环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review",
          "scientific_infrastructure",
          "software_engineering"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "reproduction_bundle"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim",
          "tracking_infrastructure"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "artifact_hashing",
          "claim_evidence_mapping"
        ],
        "accountability_features": [
          "agent_trace",
          "artifact_lineage",
          "provenance_graph",
          "release_bundle"
        ],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [],
        "risk_flags": [],
        "fit_to_our_target": [
          "infrastructure_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "自包含复现包",
        "W3C PROV semantics"
      ],
      "gaps_vs_our_target": [
        "标准/工具层，不自动理解 paper/review"
      ],
      "detailed_analysis": null
    },
    {
      "id": "robin",
      "name": "Robin / Future House",
      "depth": "deep",
      "summary": "多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "biomedicine_drug_discovery",
          "general_literature_research",
          "wetlab_lab_automation"
        ],
        "starting_points": [
          "disease_or_candidate",
          "topic_or_idea"
        ],
        "primary_outputs": [
          "hypothesis_or_idea"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "lab_discovery_loop"
        ],
        "execution_depth": [
          "smoke_test",
          "wetlab_human_in_loop"
        ],
        "verification_models": [
          "citation_grounding",
          "deterministic_verifier",
          "human_gate"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "human_ai_team",
          "inspector_agent",
          "multi_agent_debate"
        ],
        "integration_styles": [
          "cli",
          "python_package"
        ],
        "maturity_signals": [
          "active_development",
          "peer_reviewed_paper",
          "reproducible_examples"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "domain_inspiration"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "Crow/Falcon/Finch 等专用 agent 编排",
        "湿实验由人执行，AI 负责 intellectual loop",
        "GitHub 依赖 Edison 平台 key，完整本地复现受限"
      ],
      "gaps_vs_our_target": [
        "中心是疾病候选发现，不是论文复现 CI",
        "没有通用 paper claim/run/review ledger",
        "开源 repo 依赖外部平台访问"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/robin.md"
    },
    {
      "id": "scientific-agent-skills",
      "name": "scientific-agent-skills / ML-AI skill authoring",
      "depth": "deep",
      "summary": "高质量 skill 编写范式：SKILL.md 做路由，references 放长决策，scripts 放稳定 scaffold，tests 放验证。适合作为我们调查 skill 仓库的结构样板。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "software_engineering"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "experiment_plan",
          "review_or_rebuttal"
        ],
        "workflow_scopes": [
          "agent_runtime"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [],
        "agent_topologies": [],
        "integration_styles": [
          "container_runtime",
          "markdown_skills"
        ],
        "maturity_signals": [
          "reproducible_examples"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "plugin_layer"
        ],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "强调工具级 SOP 与可验证模板",
        "防 agent 猜 API/数据格式/版本约束",
        "适合作为局部专家层，不是研究操作系统"
      ],
      "gaps_vs_our_target": [
        "没有 claim ledger、run ledger、review issue loop",
        "不负责端到端 paper-to-run-to-review"
      ],
      "detailed_analysis": "skills/autoresearch-landscape-survey/references/deep-dives/scientific-agent-skills-ml-ai.md"
    },
    {
      "id": "sciintegrity-bench",
      "name": "SciIntegrity-Bench",
      "depth": "medium",
      "summary": "科研诚信 benchmark，强调缺失数据、伪造结果、数据合成等问题。",
      "facets": {
        "research_fields": [
          "metascience_integrity_review"
        ],
        "starting_points": [
          "draft_paper"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "review_or_rebuttal"
        ],
        "workflow_scopes": [
          "claim_to_review_issue"
        ],
        "execution_depth": [],
        "verification_models": [
          "adversarial_reviewer",
          "claim_evidence_mapping",
          "rubric_grading"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "judge_panel"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "benchmark_or_leaderboard",
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 4,
      "differentiators": [
        "integrity problem rate 评估"
      ],
      "gaps_vs_our_target": [
        "benchmark，不提供流水线"
      ],
      "detailed_analysis": null
    },
    {
      "id": "agent-laboratory",
      "name": "Agent Laboratory",
      "depth": "brief",
      "summary": "human-in-the-loop 自动科研：literature review、experimentation、report writing。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research"
        ],
        "starting_points": [
          "topic_or_idea"
        ],
        "primary_outputs": [
          "hypothesis_or_idea",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "idea_to_paper"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "citation_grounding",
          "human_gate"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "human_ai_team",
          "orchestrator_workers"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "强调人类反馈提升质量"
      ],
      "gaps_vs_our_target": [
        "未聚焦 claim/run ledger"
      ],
      "detailed_analysis": null
    },
    {
      "id": "agenthallu",
      "name": "AgentHallu",
      "depth": "brief",
      "summary": "评估多步 agent hallucination 及 step localization，适合设计错误归因机制。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "reproduction_bundle"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "execution_depth": [],
        "verification_models": [
          "rubric_grading"
        ],
        "accountability_features": [
          "agent_trace",
          "provenance_graph"
        ],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "benchmark_or_leaderboard",
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "unverified_claim_generation"
        ],
        "fit_to_our_target": [
          "risk_model"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "Planning/Retrieval/Reasoning/Tool-use 等 hallucination 分类"
      ],
      "gaps_vs_our_target": [
        "benchmark 而非系统"
      ],
      "detailed_analysis": null
    },
    {
      "id": "ai-scientist-v1",
      "name": "The AI Scientist",
      "depth": "medium",
      "summary": "早期端到端自动科研标志项目，强依赖模板和领域约束，是 idea-to-paper 叙事起点。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research"
        ],
        "starting_points": [
          "topic_or_idea"
        ],
        "primary_outputs": [
          "hypothesis_or_idea",
          "paper_draft_or_tex",
          "training_run_results"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "idea_to_paper"
        ],
        "execution_depth": [
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [
          "citation_grounding",
          "llm_reviewer"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "judge_panel",
          "orchestrator_workers"
        ],
        "integration_styles": [
          "cli",
          "container_runtime",
          "python_package"
        ],
        "maturity_signals": [
          "active_development",
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "adjacent_competitor"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "开放式科研 loop",
        "模拟评审"
      ],
      "gaps_vs_our_target": [
        "独立评估显示 coding/novelty/manuscript 问题",
        "证据链薄弱"
      ],
      "detailed_analysis": null
    },
    {
      "id": "aide-ml",
      "name": "AIDE ML",
      "depth": "brief",
      "summary": "面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "existing_code_repo"
        ],
        "primary_outputs": [
          "code_repo_or_patch",
          "experiment_plan",
          "training_run_results"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "idea_to_experiment",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [
          "failure_registry"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "cli",
          "container_runtime",
          "python_package"
        ],
        "maturity_signals": [
          "active_development"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "runtime_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "tree-search coding + metric loop"
      ],
      "gaps_vs_our_target": [
        "不做 paper claim audit"
      ],
      "detailed_analysis": null
    },
    {
      "id": "autoscientists",
      "name": "AutoScientists",
      "depth": "brief",
      "summary": "去中心化实验团队，共享 experimental state 和成功/失败记忆。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research",
          "general_literature_research"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "experiment_plan",
          "training_run_results"
        ],
        "workflow_scopes": [
          "experiment_to_claim",
          "idea_to_experiment",
          "idea_to_paper",
          "lab_discovery_loop"
        ],
        "execution_depth": [
          "full_training_or_simulation"
        ],
        "verification_models": [],
        "accountability_features": [
          "agent_trace"
        ],
        "agent_topologies": [
          "dynamic_roles",
          "multi_agent_debate"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "算力使用前 critique proposals",
        "失败共享减少重复探索"
      ],
      "gaps_vs_our_target": [
        "更偏长期实验探索，不是 paper audit"
      ],
      "detailed_analysis": null
    },
    {
      "id": "biomars",
      "name": "BioMARS",
      "depth": "brief",
      "summary": "Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "biomedicine_drug_discovery",
          "general_literature_research",
          "wetlab_lab_automation"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "experiment_plan",
          "training_run_results"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "lab_discovery_loop"
        ],
        "execution_depth": [
          "smoke_test",
          "wetlab_human_in_loop"
        ],
        "verification_models": [
          "deterministic_verifier",
          "human_gate"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "human_ai_team",
          "inspector_agent"
        ],
        "integration_styles": [
          "container_runtime"
        ],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "domain_inspiration"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "procedural integrity inspector"
      ],
      "gaps_vs_our_target": [
        "湿实验/机器人域，不是通用 research CI"
      ],
      "detailed_analysis": null
    },
    {
      "id": "catfish-agent",
      "name": "Catfish Agent",
      "depth": "brief",
      "summary": "通过故意注入结构化异议缓解多 agent silent agreement。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "draft_paper"
        ],
        "primary_outputs": [
          "review_or_rebuttal"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "experiment_to_claim"
        ],
        "execution_depth": [],
        "verification_models": [
          "adversarial_reviewer",
          "llm_reviewer"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "judge_panel"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [
          "risk_model"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "Dissent / Refutation agent"
      ],
      "gaps_vs_our_target": [
        "不做研究执行"
      ],
      "detailed_analysis": null
    },
    {
      "id": "era",
      "name": "ERA / Empirical Research Assistance",
      "depth": "brief",
      "summary": "LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "general_literature_research",
          "software_engineering"
        ],
        "starting_points": [
          "existing_code_repo"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "code_repo_or_patch"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "idea_to_experiment",
          "lab_discovery_loop",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "rubric_grading"
        ],
        "accountability_features": [
          "failure_registry"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "container_runtime"
        ],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "tree search over scientific software"
      ],
      "gaps_vs_our_target": [
        "不做 paper claim/review ledger"
      ],
      "detailed_analysis": null
    },
    {
      "id": "google-co-scientist",
      "name": "Google Co-Scientist",
      "depth": "medium",
      "summary": "基于 Gemini 的多 agent hypothesis generation，使用 tournament evolution 和 test-time compute。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "biomedicine_drug_discovery",
          "general_literature_research"
        ],
        "starting_points": [
          "disease_or_candidate",
          "topic_or_idea"
        ],
        "primary_outputs": [
          "hypothesis_or_idea"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "lab_discovery_loop"
        ],
        "execution_depth": [],
        "verification_models": [
          "citation_grounding",
          "llm_reviewer"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "dynamic_roles",
          "judge_panel",
          "multi_agent_debate"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [
          "domain_inspiration"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "hypothesis tournament/evolution",
        "biomedical validation"
      ],
      "gaps_vs_our_target": [
        "不是 paper reproduction CI"
      ],
      "detailed_analysis": null
    },
    {
      "id": "hiras",
      "name": "HiRAS",
      "depth": "brief",
      "summary": "层级多 agent paper-to-code/execution framework。",
      "facets": {
        "research_fields": [
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "code_repo_or_patch",
          "literature_report",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "paper_to_code",
          "paper_to_reproduction"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [],
        "agent_topologies": [
          "multi_agent_debate"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "manager agents 协调 specialized agents"
      ],
      "gaps_vs_our_target": [
        "仍偏 code generation/execution，未形成 review issue ledger"
      ],
      "detailed_analysis": null
    },
    {
      "id": "hydra",
      "name": "Hydra",
      "depth": "brief",
      "summary": "配置组合、CLI override、multirun/sweeps，适合 Experiment Spec 编译。",
      "facets": {
        "research_fields": [
          "scientific_infrastructure",
          "software_engineering"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "code_repo_or_patch",
          "experiment_plan",
          "training_run_results"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "repo_to_experiment",
          "tracking_infrastructure"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [],
        "risk_flags": [],
        "fit_to_our_target": [
          "infrastructure_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "配置层成熟"
      ],
      "gaps_vs_our_target": [
        "无 artifact/provenance/claim"
      ],
      "detailed_analysis": null
    },
    {
      "id": "mle-bench",
      "name": "MLE-bench",
      "depth": "brief",
      "summary": "Kaggle-style ML engineering benchmark，可作为训练/调参能力测量参考。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "dataset_or_benchmark"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "training_run_results"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [
          "rubric_grading"
        ],
        "accountability_features": [],
        "agent_topologies": [],
        "integration_styles": [
          "container_runtime"
        ],
        "maturity_signals": [
          "benchmark_or_leaderboard"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "benchmark_reference"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "ML engineering 能力评估"
      ],
      "gaps_vs_our_target": [
        "非 paper-faithful reproduction"
      ],
      "detailed_analysis": null
    },
    {
      "id": "openhands",
      "name": "OpenHands",
      "depth": "brief",
      "summary": "通用 coding agent/agent runtime，可作为代码修复和 repo patch 执行底座。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "software_engineering"
        ],
        "starting_points": [
          "existing_code_repo"
        ],
        "primary_outputs": [
          "code_repo_or_patch"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [
          "failure_registry"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "cli",
          "container_runtime",
          "markdown_skills",
          "python_package"
        ],
        "maturity_signals": [
          "active_development"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "plugin_layer",
          "runtime_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "SDK/CLI/GUI",
        "大生态"
      ],
      "gaps_vs_our_target": [
        "无研究语义/claim ledger"
      ],
      "detailed_analysis": null
    },
    {
      "id": "papercoder-paper2code",
      "name": "PaperCoder / Paper2Code",
      "depth": "brief",
      "summary": "把 ML paper 转成 functional code repository 的研究原型。",
      "facets": {
        "research_fields": [
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "code_repo_or_patch",
          "literature_report",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "paper_to_code",
          "paper_to_reproduction"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "planning-analysis-generation pipeline"
      ],
      "gaps_vs_our_target": [
        "不覆盖审稿反馈和 run provenance"
      ],
      "detailed_analysis": null
    },
    {
      "id": "paperqa2",
      "name": "PaperQA2",
      "depth": "brief",
      "summary": "科学文献 RAG / QA / contradiction detection，可作为 paper understanding 与 evidence retrieval 层。",
      "facets": {
        "research_fields": [
          "general_literature_research",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "literature_report",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "literature_to_report"
        ],
        "execution_depth": [],
        "verification_models": [
          "citation_grounding",
          "claim_evidence_mapping",
          "deterministic_verifier"
        ],
        "accountability_features": [],
        "agent_topologies": [],
        "integration_styles": [
          "cli",
          "python_package"
        ],
        "maturity_signals": [
          "active_development"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "plugin_layer"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "scientific QA",
        "PDF/文档/代码文档 RAG"
      ],
      "gaps_vs_our_target": [
        "不执行实验"
      ],
      "detailed_analysis": null
    },
    {
      "id": "researchcodebench",
      "name": "ResearchCodeBench",
      "depth": "brief",
      "summary": "测试 LLM 实现 ML 论文中新研究 idea 的 coding benchmark。",
      "facets": {
        "research_fields": [
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "code_repo_or_patch"
        ],
        "workflow_scopes": [
          "paper_to_code",
          "paper_to_reproduction",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "rubric_grading"
        ],
        "accountability_features": [],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "benchmark_or_leaderboard"
        ],
        "risk_flags": [],
        "fit_to_our_target": [
          "benchmark_reference"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "来自 2024-2025 ML papers 的 coding challenges"
      ],
      "gaps_vs_our_target": [
        "不跑完整训练复现"
      ],
      "detailed_analysis": null
    },
    {
      "id": "scientific-skills-skillfoundry",
      "name": "SkillFoundry",
      "depth": "brief",
      "summary": "从异构科学资源挖掘并验证 skill 的 self-evolving framework；可作为未来扩展调查 skill 的方法参考。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "software_engineering"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "experiment_plan",
          "review_or_rebuttal"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "attestation_protocol"
        ],
        "accountability_features": [
          "decision_log",
          "failure_registry"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "markdown_skills"
        ],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "domain knowledge tree",
        "skill mining/repair/merge/prune"
      ],
      "gaps_vs_our_target": [
        "skill 生成，不是研究流水线"
      ],
      "detailed_analysis": null
    },
    {
      "id": "snakemake-nextflow",
      "name": "Snakemake / Nextflow",
      "depth": "brief",
      "summary": "可复现 workflow DAG，适合复杂数据分析 pipeline。",
      "facets": {
        "research_fields": [
          "scientific_infrastructure",
          "software_engineering"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "experiment_plan",
          "reproduction_bundle",
          "training_run_results"
        ],
        "workflow_scopes": [
          "tracking_infrastructure"
        ],
        "execution_depth": [
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [
          "release_bundle",
          "run_ledger"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "container_runtime"
        ],
        "maturity_signals": [],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "infrastructure_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "HPC/cloud/local workflow portability"
      ],
      "gaps_vs_our_target": [
        "无 paper claim/reviewer 语义"
      ],
      "detailed_analysis": null
    },
    {
      "id": "toolmaker",
      "name": "ToolMaker",
      "depth": "brief",
      "summary": "把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "existing_code_repo"
        ],
        "primary_outputs": [
          "code_repo_or_patch"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "paper_to_code",
          "paper_to_reproduction",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [
          "failure_registry"
        ],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "plugin_layer",
          "runtime_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "自动安装依赖、生成工具使用代码、自修复"
      ],
      "gaps_vs_our_target": [
        "不覆盖训练复现和审稿追责"
      ],
      "detailed_analysis": null
    },
    {
      "id": "wandb",
      "name": "Weights & Biases",
      "depth": "brief",
      "summary": "实验 dashboard 和 artifact lineage，可作为可视化/协作层。",
      "facets": {
        "research_fields": [
          "scientific_infrastructure",
          "software_engineering"
        ],
        "starting_points": [
          "experiment_plan"
        ],
        "primary_outputs": [
          "training_run_results"
        ],
        "workflow_scopes": [
          "tracking_infrastructure"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "artifact_hashing"
        ],
        "accountability_features": [
          "artifact_lineage",
          "run_ledger"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "mlops_tracking_stack"
        ],
        "maturity_signals": [],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "infrastructure_component"
        ],
        "custom_facets": []
      },
      "our_relevance": 3,
      "differentiators": [
        "artifact graph + dashboard"
      ],
      "gaps_vs_our_target": [
        "在线平台依赖，不负责科研语义"
      ],
      "detailed_analysis": null
    },
    {
      "id": "cascade-skills",
      "name": "CASCADE",
      "depth": "brief",
      "summary": "自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "general_literature_research",
          "software_engineering"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "experiment_plan",
          "review_or_rebuttal"
        ],
        "workflow_scopes": [
          "agent_runtime",
          "experiment_to_claim",
          "idea_to_experiment",
          "lab_discovery_loop",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [],
        "accountability_features": [
          "agent_trace",
          "failure_registry"
        ],
        "agent_topologies": [],
        "integration_styles": [
          "markdown_skills"
        ],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "continuous learning + self-reflection meta-skills"
      ],
      "gaps_vs_our_target": [
        "不以 claim ledger 为中心"
      ],
      "detailed_analysis": null
    },
    {
      "id": "denario",
      "name": "Denario",
      "depth": "brief",
      "summary": "multi-agent scientific assistant，偏数据分析和 LaTeX article 生成。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "ai_ml_algorithm_research"
        ],
        "starting_points": [
          "topic_or_idea"
        ],
        "primary_outputs": [
          "hypothesis_or_idea",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "idea_to_paper"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "citation_grounding"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "multi_agent_debate",
          "orchestrator_workers"
        ],
        "integration_styles": [
          "cli",
          "python_package"
        ],
        "maturity_signals": [
          "active_development",
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "科学数据分析闭环"
      ],
      "gaps_vs_our_target": [
        "缺外部 paper reproduction/audit ledger"
      ],
      "detailed_analysis": null
    },
    {
      "id": "gpt-researcher",
      "name": "GPT Researcher",
      "depth": "brief",
      "summary": "web/local research report agent，适合 literature/source aggregation。",
      "facets": {
        "research_fields": [
          "general_literature_research"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "literature_report",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "idea_to_paper",
          "literature_to_report"
        ],
        "execution_depth": [],
        "verification_models": [
          "citation_grounding"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "orchestrator_workers"
        ],
        "integration_styles": [
          "cli",
          "python_package"
        ],
        "maturity_signals": [
          "active_development"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "plugin_layer"
        ],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "带 citation 的研究报告"
      ],
      "gaps_vs_our_target": [
        "不执行实验"
      ],
      "detailed_analysis": null
    },
    {
      "id": "mlagentbench",
      "name": "MLAgentBench",
      "depth": "brief",
      "summary": "评估 ML experimentation agent 的早期 benchmark。",
      "facets": {
        "research_fields": [
          "ai_ml_algorithm_research",
          "software_engineering"
        ],
        "starting_points": [
          "dataset_or_benchmark"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "training_run_results"
        ],
        "workflow_scopes": [
          "repo_to_experiment"
        ],
        "execution_depth": [
          "full_training_or_simulation",
          "smoke_test"
        ],
        "verification_models": [
          "rubric_grading"
        ],
        "accountability_features": [],
        "agent_topologies": [],
        "integration_styles": [
          "container_runtime"
        ],
        "maturity_signals": [
          "benchmark_or_leaderboard"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "benchmark_reference"
        ],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "端到端 ML 任务"
      ],
      "gaps_vs_our_target": [
        "不绑定 paper claim"
      ],
      "detailed_analysis": null
    },
    {
      "id": "open-deep-research",
      "name": "Open Deep Research",
      "depth": "brief",
      "summary": "开源 deep research agent，可参考 research graph 和 MCP/tool 接入。",
      "facets": {
        "research_fields": [
          "general_literature_research"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "literature_report",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "idea_to_paper",
          "literature_to_report"
        ],
        "execution_depth": [],
        "verification_models": [
          "citation_grounding"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "orchestrator_workers"
        ],
        "integration_styles": [
          "cli",
          "python_package"
        ],
        "maturity_signals": [
          "active_development"
        ],
        "risk_flags": [],
        "fit_to_our_target": [
          "plugin_layer"
        ],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "多模型/多搜索工具"
      ],
      "gaps_vs_our_target": [
        "不做训练复现"
      ],
      "detailed_analysis": null
    },
    {
      "id": "openreviewer",
      "name": "OpenReviewer",
      "depth": "brief",
      "summary": "面向 ML/AI 论文评审的模型/系统，适合 reviewer text style 和 rubric。",
      "facets": {
        "research_fields": [
          "general_literature_research",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "draft_paper"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "review_or_rebuttal"
        ],
        "workflow_scopes": [
          "claim_to_review_issue",
          "literature_to_report"
        ],
        "execution_depth": [],
        "verification_models": [
          "adversarial_reviewer",
          "rubric_grading"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "judge_panel"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [
          "plugin_layer",
          "risk_model"
        ],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "使用专家 review 数据微调"
      ],
      "gaps_vs_our_target": [
        "review 不绑定 run evidence"
      ],
      "detailed_analysis": null
    },
    {
      "id": "openscholar",
      "name": "OpenScholar",
      "depth": "brief",
      "summary": "科学文献 synthesis，citation-backed responses 和 ScholarQABench。",
      "facets": {
        "research_fields": [
          "general_literature_research",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "benchmark_scores",
          "literature_report",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "literature_to_report"
        ],
        "execution_depth": [],
        "verification_models": [
          "citation_grounding",
          "claim_evidence_mapping",
          "deterministic_verifier",
          "rubric_grading"
        ],
        "accountability_features": [],
        "agent_topologies": [],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [
          "plugin_layer"
        ],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "大规模 open-access paper 检索"
      ],
      "gaps_vs_our_target": [
        "不做代码/训练"
      ],
      "detailed_analysis": null
    },
    {
      "id": "stella",
      "name": "STELLA",
      "depth": "brief",
      "summary": "self-evolving biomedical agent，Template Library + Tool Ocean。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "general_literature_research",
          "metascience_integrity_review"
        ],
        "starting_points": [
          "workflow_task"
        ],
        "primary_outputs": [
          "experiment_plan",
          "review_or_rebuttal"
        ],
        "workflow_scopes": [
          "experiment_to_claim",
          "idea_to_experiment",
          "lab_discovery_loop",
          "literature_to_report",
          "repo_to_experiment"
        ],
        "execution_depth": [
          "smoke_test"
        ],
        "verification_models": [
          "citation_grounding"
        ],
        "accountability_features": [
          "agent_trace",
          "failure_registry"
        ],
        "agent_topologies": [
          "dynamic_roles"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [],
        "fit_to_our_target": [],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "工具发现与模板进化"
      ],
      "gaps_vs_our_target": [
        "自进化记忆需可审计，否则会放大错误"
      ],
      "detailed_analysis": null
    },
    {
      "id": "storm",
      "name": "STORM / Co-STORM",
      "depth": "brief",
      "summary": "多视角提问和 outline 生成的知识整理系统，适合 related work/landscape synthesis。",
      "facets": {
        "research_fields": [
          "general_literature_research"
        ],
        "starting_points": [
          "draft_paper",
          "paper_pdf_or_preprint"
        ],
        "primary_outputs": [
          "literature_report",
          "paper_draft_or_tex"
        ],
        "workflow_scopes": [
          "literature_to_report"
        ],
        "execution_depth": [],
        "verification_models": [
          "citation_grounding"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "multi_agent_debate"
        ],
        "integration_styles": [
          "cli",
          "python_package"
        ],
        "maturity_signals": [
          "active_development",
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "plugin_layer"
        ],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "wiki-like report",
        "perspective-driven querying"
      ],
      "gaps_vs_our_target": [
        "不跑实验"
      ],
      "detailed_analysis": null
    },
    {
      "id": "virtual-lab",
      "name": "Virtual Lab",
      "depth": "brief",
      "summary": "模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。",
      "facets": {
        "research_fields": [
          "ai_agent_systems",
          "biomedicine_drug_discovery",
          "general_literature_research",
          "wetlab_lab_automation"
        ],
        "starting_points": [
          "disease_or_candidate",
          "topic_or_idea"
        ],
        "primary_outputs": [
          "hypothesis_or_idea"
        ],
        "workflow_scopes": [
          "idea_to_experiment",
          "lab_discovery_loop"
        ],
        "execution_depth": [],
        "verification_models": [
          "citation_grounding",
          "human_gate"
        ],
        "accountability_features": [],
        "agent_topologies": [
          "human_ai_team",
          "multi_agent_debate"
        ],
        "integration_styles": [],
        "maturity_signals": [
          "peer_reviewed_paper"
        ],
        "risk_flags": [
          "llm_generated_code_execution"
        ],
        "fit_to_our_target": [
          "domain_inspiration"
        ],
        "custom_facets": []
      },
      "our_relevance": 2,
      "differentiators": [
        "实验室组织仿真"
      ],
      "gaps_vs_our_target": [
        "实验室角色不等于可追责证据链"
      ],
      "detailed_analysis": null
    }
  ]
}
```
