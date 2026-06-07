# AutoResearch Project Overview

Last updated: 2026-06-03  
Registry version: 0.3.0  
Projects: 66

## How to Read This Overview

Projects are described with non-orthogonal facets. A project can be an idea-to-paper system, a skill harness, a sandbox runner, and a partial claim-audit system at the same time. Do not read any section as a mutually exclusive category.

## High-Level Matrix

| Project | Fit | Fields | Workflow | Verification | Accountability | Agent Topology | Deep Dive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AAR / Claim-Level Auditability | claim_kernel_reference, protocol_competitor, risk_model | metascience_integrity_review, general_literature_research | literature_to_report, claim_to_review_issue | claim_evidence_mapping, citation_grounding | provenance_graph, claim_ledger, decision_log |  |  |
| ARIS / Auto-claude-code-research-in-sleep | adjacent_competitor | ai_agent_systems, ai_ml_algorithm_research, software_engineering | agent_runtime, experiment_to_claim, idea_to_experiment, idea_to_paper | adversarial_reviewer, citation_grounding, claim_evidence_mapping, cross_model_review | agent_trace | judge_panel, orchestrator_workers | skills/autoresearch-landscape-survey/references/deep-dives/aris.md |
| AutoResearchClaw | adjacent_competitor | ai_agent_systems, ai_ml_algorithm_research, metascience_integrity_review | agent_runtime, claim_to_review_issue, idea_to_experiment, idea_to_paper | artifact_hashing, citation_grounding, claim_evidence_mapping, cross_model_review | artifact_lineage, claim_ledger, failure_registry | human_ai_team, orchestrator_workers | skills/autoresearch-landscape-survey/references/deep-dives/autoresearchclaw.md |
| data-to-paper | claim_kernel_reference, governance_reference | metascience_integrity_review, scientific_infrastructure | experiment_to_claim, claim_to_review_issue | claim_evidence_mapping, human_gate | provenance_graph, artifact_lineage, release_bundle, decision_log |  |  |
| DeepScientist | research_os_competitor, adjacent_competitor | ai_agent_systems, ai_ml_algorithm_research | agent_runtime, idea_to_experiment, idea_to_paper, repo_to_experiment | human_gate | agent_trace, artifact_lineage, decision_log |  |  |
| EviBound | core_competitor, risk_model, kernel_competitor | ai_agent_systems, metascience_integrity_review | claim_to_review_issue, experiment_to_claim | artifact_hashing, claim_evidence_mapping, deterministic_verifier | artifact_lineage, claim_ledger, run_ledger |  | skills/autoresearch-landscape-survey/references/deep-dives/accountability-layer.md |
| PaperBench | benchmark_reference | ai_ml_algorithm_research, software_engineering | paper_to_code, paper_to_reproduction, repo_to_experiment | rubric_grading | release_bundle |  | skills/autoresearch-landscape-survey/references/deep-dives/paperbench-and-reproduction-benchmarks.md |
| RePro / RefP2C | claim_kernel_reference, benchmark_reference | ai_ml_algorithm_research, software_engineering | paper_to_code, paper_to_reproduction | claim_evidence_mapping, deterministic_verifier | method_spec, experiment_spec |  |  |
| ScientistOne / Chain-of-Evidence | core_competitor, risk_model, kernel_competitor | ai_agent_systems, metascience_integrity_review | claim_to_review_issue, experiment_to_claim | attestation_protocol, claim_evidence_mapping, deterministic_verifier | agent_trace, claim_ledger, decision_log, provenance_graph |  | skills/autoresearch-landscape-survey/references/deep-dives/accountability-layer.md |
| Sibyl-AutoResearch | adjacent_competitor, governance_reference, risk_model | ai_agent_systems, metascience_integrity_review | idea_to_experiment, experiment_to_claim, agent_runtime | human_gate, deterministic_verifier | failure_registry, artifact_lineage, decision_log, agent_trace |  |  |
| AAGATE | governance_reference, infrastructure_component | ai_agent_systems, scientific_infrastructure | agent_runtime, tracking_infrastructure | human_gate, deterministic_verifier | provenance_graph, decision_log, agent_trace |  |  |
| AblationBench |  | ai_ml_algorithm_research, metascience_integrity_review, software_engineering | claim_to_review_issue, repo_to_experiment | adversarial_reviewer, rubric_grading |  | judge_panel |  |
| The AI Scientist v2 | adjacent_competitor | ai_agent_systems, ai_ml_algorithm_research | idea_to_experiment, idea_to_paper, repo_to_experiment | citation_grounding | failure_registry | multi_agent_debate, orchestrator_workers |  |
| AutoExperiment | benchmark_reference | ai_ml_algorithm_research, software_engineering | paper_to_code, paper_to_reproduction, repo_to_experiment | rubric_grading | failure_registry |  |  |
| AutoP2C |  | ai_ml_algorithm_research, software_engineering | idea_to_experiment, paper_to_code, paper_to_reproduction, repo_to_experiment |  | failure_registry |  |  |
| AutoReproduce | benchmark_reference, adjacent_competitor | ai_ml_algorithm_research, software_engineering | paper_to_reproduction, paper_to_code, repo_to_experiment | claim_evidence_mapping, deterministic_verifier | artifact_lineage, provenance_graph |  |  |
| Claw AI Lab | adjacent_competitor | ai_agent_systems, ai_ml_algorithm_research | idea_to_experiment, idea_to_paper, repo_to_experiment | artifact_hashing, citation_grounding, human_gate | artifact_lineage | human_ai_team, multi_agent_debate |  |
| DVC | infrastructure_component | scientific_infrastructure, software_engineering | tracking_infrastructure | artifact_hashing | artifact_lineage, release_bundle, run_ledger |  |  |
| Hyperspace AGI | adjacent_competitor, runtime_component | ai_agent_systems, scientific_infrastructure | agent_runtime, idea_to_experiment, idea_to_paper | llm_reviewer | agent_trace, failure_registry |  |  |
| KAIJU | runtime_component, governance_reference | ai_agent_systems, software_engineering | agent_runtime | human_gate, deterministic_verifier | agent_trace, decision_log |  |  |
| MARCH | risk_model | ai_agent_systems, metascience_integrity_review | claim_to_review_issue, experiment_to_claim | adversarial_reviewer, claim_evidence_mapping, deterministic_verifier, llm_reviewer |  | judge_panel |  |
| MLflow | infrastructure_component | scientific_infrastructure, software_engineering | tracking_infrastructure | artifact_hashing | artifact_lineage, run_ledger |  |  |
| Paper2Agent | adjacent_competitor, plugin_layer | ai_agent_systems, software_engineering | paper_to_code, paper_to_reproduction, agent_runtime | deterministic_verifier, claim_evidence_mapping | artifact_lineage |  |  |
| Pramana / ClaimAttestation | risk_model, protocol_competitor | ai_agent_systems, metascience_integrity_review | claim_to_review_issue, experiment_to_claim | attestation_protocol, claim_evidence_mapping, deterministic_verifier | agent_trace, decision_log, provenance_graph |  |  |
| PROV-AGENT | infrastructure_component, risk_model, protocol_competitor | ai_agent_systems, metascience_integrity_review | claim_to_review_issue, experiment_to_claim | attestation_protocol | agent_trace, decision_log, provenance_graph |  |  |
| ReproZip / RO-Crate / W3C PROV | infrastructure_component | ai_agent_systems, metascience_integrity_review, scientific_infrastructure | claim_to_review_issue, experiment_to_claim, tracking_infrastructure | artifact_hashing, claim_evidence_mapping | agent_trace, artifact_lineage, provenance_graph, release_bundle |  |  |
| Robin / Future House | domain_inspiration | ai_agent_systems, biomedicine_drug_discovery, general_literature_research | idea_to_experiment, lab_discovery_loop | citation_grounding, deterministic_verifier, human_gate |  | human_ai_team, inspector_agent, multi_agent_debate | skills/autoresearch-landscape-survey/references/deep-dives/robin.md |
| scientific-agent-skills / ML-AI skill authoring | plugin_layer | ai_agent_systems, software_engineering | agent_runtime |  |  |  | skills/autoresearch-landscape-survey/references/deep-dives/scientific-agent-skills-ml-ai.md |
| SciIntegrity-Bench |  | metascience_integrity_review | claim_to_review_issue | adversarial_reviewer, claim_evidence_mapping, rubric_grading |  | judge_panel |  |
| Agent Laboratory |  | ai_agent_systems, ai_ml_algorithm_research | idea_to_experiment, idea_to_paper | citation_grounding, human_gate |  | human_ai_team, orchestrator_workers |  |
| AgentHallu | risk_model | ai_agent_systems, metascience_integrity_review | claim_to_review_issue, experiment_to_claim | rubric_grading | agent_trace, provenance_graph |  |  |
| AgentRxiv | infrastructure_component, plugin_layer | ai_agent_systems, general_literature_research | literature_to_report, agent_runtime | citation_grounding | release_bundle |  |  |
| The AI Scientist | adjacent_competitor | ai_agent_systems, ai_ml_algorithm_research | idea_to_experiment, idea_to_paper | citation_grounding, llm_reviewer |  | judge_panel, orchestrator_workers |  |
| AICat | governance_reference, infrastructure_component | scientific_infrastructure, metascience_integrity_review | tracking_infrastructure |  | release_bundle, provenance_graph |  |  |
| AIDE ML | runtime_component | ai_agent_systems, ai_ml_algorithm_research, software_engineering | agent_runtime, idea_to_experiment, repo_to_experiment |  | failure_registry |  |  |
| AutoScientists |  | ai_agent_systems, ai_ml_algorithm_research, general_literature_research | experiment_to_claim, idea_to_experiment, idea_to_paper, lab_discovery_loop |  | agent_trace | dynamic_roles, multi_agent_debate |  |
| Bilevel Autoresearch | watch_only, runtime_component | ai_agent_systems, software_engineering | agent_runtime, repo_to_experiment | deterministic_verifier | agent_trace |  |  |
| BioMARS | domain_inspiration | ai_agent_systems, biomedicine_drug_discovery, general_literature_research | idea_to_experiment, lab_discovery_loop | deterministic_verifier, human_gate |  | human_ai_team, inspector_agent |  |
| Catfish Agent | risk_model | ai_agent_systems, metascience_integrity_review | claim_to_review_issue, experiment_to_claim | adversarial_reviewer, llm_reviewer |  | judge_panel |  |
| DeepXiv-SDK | plugin_layer, infrastructure_component | general_literature_research, scientific_infrastructure | literature_to_report, agent_runtime | citation_grounding |  |  |  |
| ERA / Empirical Research Assistance |  | ai_agent_systems, general_literature_research, software_engineering | agent_runtime, idea_to_experiment, lab_discovery_loop, repo_to_experiment | rubric_grading | failure_registry |  |  |
| EvoScientist | adjacent_competitor | ai_agent_systems | idea_to_experiment, idea_to_paper, agent_runtime | llm_reviewer | agent_trace, failure_registry |  |  |
| FM Agent | runtime_component, adjacent_competitor | ai_agent_systems, ai_ml_algorithm_research, software_engineering | agent_runtime, idea_to_experiment, repo_to_experiment | deterministic_verifier | agent_trace, artifact_lineage |  |  |
| Google Co-Scientist | domain_inspiration | ai_agent_systems, biomedicine_drug_discovery, general_literature_research | idea_to_experiment, lab_discovery_loop | citation_grounding, llm_reviewer |  | dynamic_roles, judge_panel, multi_agent_debate |  |
| HiRAS |  | ai_ml_algorithm_research, software_engineering | paper_to_code, paper_to_reproduction |  |  | multi_agent_debate |  |
| Hydra | infrastructure_component | scientific_infrastructure, software_engineering | idea_to_experiment, repo_to_experiment, tracking_infrastructure |  |  |  |  |
| MLE-bench | benchmark_reference | ai_agent_systems, ai_ml_algorithm_research, software_engineering | agent_runtime, repo_to_experiment | rubric_grading |  |  |  |
| OpenAGS | adjacent_competitor | ai_agent_systems | idea_to_paper, idea_to_experiment, agent_runtime | llm_reviewer | agent_trace |  |  |
| OpenHands | plugin_layer, runtime_component | ai_agent_systems, software_engineering | agent_runtime, repo_to_experiment |  | failure_registry |  |  |
| PaperCoder / Paper2Code |  | ai_ml_algorithm_research, software_engineering | paper_to_code, paper_to_reproduction |  |  |  |  |
| PaperQA2 | plugin_layer | general_literature_research, metascience_integrity_review | literature_to_report | citation_grounding, claim_evidence_mapping, deterministic_verifier |  |  |  |
| ResearchCodeBench | benchmark_reference | ai_ml_algorithm_research, software_engineering | paper_to_code, paper_to_reproduction, repo_to_experiment | rubric_grading |  |  |  |
| SkillFoundry |  | ai_agent_systems, software_engineering | agent_runtime, repo_to_experiment | attestation_protocol | decision_log, failure_registry |  |  |
| Snakemake / Nextflow | infrastructure_component | scientific_infrastructure, software_engineering | tracking_infrastructure |  | release_bundle, run_ledger |  |  |
| ToolMaker | plugin_layer, runtime_component | ai_agent_systems, ai_ml_algorithm_research, software_engineering | agent_runtime, paper_to_code, paper_to_reproduction, repo_to_experiment |  | failure_registry |  |  |
| Weights & Biases | infrastructure_component | scientific_infrastructure, software_engineering | tracking_infrastructure | artifact_hashing | artifact_lineage, run_ledger |  |  |
| CASCADE |  | ai_agent_systems, general_literature_research, software_engineering | agent_runtime, experiment_to_claim, idea_to_experiment, lab_discovery_loop |  | agent_trace, failure_registry |  |  |
| Denario |  | ai_agent_systems, ai_ml_algorithm_research | idea_to_experiment, idea_to_paper | citation_grounding |  | multi_agent_debate, orchestrator_workers |  |
| GPT Researcher | plugin_layer | general_literature_research | idea_to_paper, literature_to_report | citation_grounding |  | orchestrator_workers |  |
| MLAgentBench | benchmark_reference | ai_ml_algorithm_research, software_engineering | repo_to_experiment | rubric_grading |  |  |  |
| Open Deep Research | plugin_layer | general_literature_research | idea_to_paper, literature_to_report | citation_grounding |  | orchestrator_workers |  |
| OpenReviewer | plugin_layer, risk_model | general_literature_research, metascience_integrity_review | claim_to_review_issue, literature_to_report | adversarial_reviewer, rubric_grading |  | judge_panel |  |
| OpenScholar | plugin_layer | general_literature_research, metascience_integrity_review | literature_to_report | citation_grounding, claim_evidence_mapping, deterministic_verifier, rubric_grading |  |  |  |
| STELLA |  | ai_agent_systems, general_literature_research, metascience_integrity_review | experiment_to_claim, idea_to_experiment, lab_discovery_loop, literature_to_report | citation_grounding | agent_trace, failure_registry | dynamic_roles |  |
| STORM / Co-STORM | plugin_layer | general_literature_research | literature_to_report | citation_grounding |  | multi_agent_debate |  |
| Virtual Lab | domain_inspiration | ai_agent_systems, biomedicine_drug_discovery, general_literature_research | idea_to_experiment, lab_discovery_loop | citation_grounding, human_gate |  | human_ai_team, multi_agent_debate |  |

## By Fit to Our Target

### `adjacent_competitor`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **The AI Scientist v2** (`ai-scientist-v2`): 自动科学发现代表项目，v2 减少模板依赖，强调 agentic tree search；需要重点关注安全和可验证性不足。
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **Claw AI Lab** (`claw-ai-lab`): 把自动科研变成 dashboard 化 AI lab，突出 artifact inspector、rollback/resume、reproduce mode。
- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **Paper2Agent** (`paper2agent`): Converts papers and codebases into interactive MCP agents; useful paper-to-agent direction but not claim-state governance.
- **The AI Scientist** (`ai-scientist-v1`): 早期端到端自动科研标志项目，强依赖模板和领域约束，是 idea-to-paper 叙事起点。
- **EvoScientist** (`evoscientist`): Self-evolving AI scientist with persistent memory and long-term improvement.
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **OpenAGS** (`openags`): Open Autonomous Generalist Scientist / auto-research framework; adjacent broad autonomous research project.

### `benchmark_reference`

- **PaperBench** (`paperbench`): 评估 agent 复现 ICML 论文的关键 benchmark；证明 paper reproduction 仍很难。
- **RePro / RefP2C** (`repro-refp2c`): Fine-grained paper fingerprint extraction and iterative verification/refinement for paper-to-code; important verifier-design reference.
- **AutoExperiment** (`autoexperiment`): 通过 progressive code masking 测 agent 从 paper + masked codebase 中补全代码、执行实验、复现结果。
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **MLE-bench** (`mle-bench`): Kaggle-style ML engineering benchmark，可作为训练/调参能力测量参考。
- **ResearchCodeBench** (`researchcodebench`): 测试 LLM 实现 ML 论文中新研究 idea 的 coding benchmark。
- **MLAgentBench** (`mlagentbench`): 评估 ML experimentation agent 的早期 benchmark。

### `claim_kernel_reference`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **RePro / RefP2C** (`repro-refp2c`): Fine-grained paper fingerprint extraction and iterative verification/refinement for paper-to-code; important verifier-design reference.
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。

### `core_competitor`

- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。

### `domain_inspiration`

- **Robin / Future House** (`robin`): 多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。
- **BioMARS** (`biomars`): Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。
- **Google Co-Scientist** (`google-co-scientist`): 基于 Gemini 的多 agent hypothesis generation，使用 tournament evolution 和 test-time compute。
- **Virtual Lab** (`virtual-lab`): 模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。

### `governance_reference`

- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **KAIJU** (`kaiju`): Execution kernel for LLM agents separating reasoning from tool execution with scheduling/tool dispatch/security gates.
- **AICat** (`aicat`): AI catalogue vocabulary/approach for machine-readable AI system metadata to support EU AI Act-style registration and transparency.

### `infrastructure_component`

- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **DVC** (`dvc`): 数据/模型/管线版本化，适合作为数据和大 artifact 版本层。
- **MLflow** (`mlflow`): run tracking、params/metrics/artifacts、model registry；适合作为 Run Ledger 后端。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **AgentRxiv** (`agentrxiv`): Shared preprint / report server for autonomous research agents; useful collaboration bus but not accountability kernel.
- **AICat** (`aicat`): AI catalogue vocabulary/approach for machine-readable AI system metadata to support EU AI Act-style registration and transparency.
- **DeepXiv-SDK** (`deepxiv-sdk`): Scientific literature data interface / SDK / MCP layer for agents.
- **Hydra** (`hydra`): 配置组合、CLI override、multirun/sweeps，适合 Experiment Spec 编译。
- **Snakemake / Nextflow** (`snakemake-nextflow`): 可复现 workflow DAG，适合复杂数据分析 pipeline。
- **Weights & Biases** (`wandb`): 实验 dashboard 和 artifact lineage，可作为可视化/协作层。

### `kernel_competitor`

- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。

### `plugin_layer`

- **Paper2Agent** (`paper2agent`): Converts papers and codebases into interactive MCP agents; useful paper-to-agent direction but not claim-state governance.
- **scientific-agent-skills / ML-AI skill authoring** (`scientific-agent-skills`): 高质量 skill 编写范式：SKILL.md 做路由，references 放长决策，scripts 放稳定 scaffold，tests 放验证。适合作为我们调查 skill 仓库的结构样板。
- **AgentRxiv** (`agentrxiv`): Shared preprint / report server for autonomous research agents; useful collaboration bus but not accountability kernel.
- **DeepXiv-SDK** (`deepxiv-sdk`): Scientific literature data interface / SDK / MCP layer for agents.
- **OpenHands** (`openhands`): 通用 coding agent/agent runtime，可作为代码修复和 repo patch 执行底座。
- **PaperQA2** (`paperqa2`): 科学文献 RAG / QA / contradiction detection，可作为 paper understanding 与 evidence retrieval 层。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。
- **GPT Researcher** (`gpt-researcher`): web/local research report agent，适合 literature/source aggregation。
- **Open Deep Research** (`open-deep-research`): 开源 deep research agent，可参考 research graph 和 MCP/tool 接入。
- **OpenReviewer** (`openreviewer`): 面向 ML/AI 论文评审的模型/系统，适合 reviewer text style 和 rubric。
- **OpenScholar** (`openscholar`): 科学文献 synthesis，citation-backed responses 和 ScholarQABench。
- **STORM / Co-STORM** (`storm`): 多视角提问和 outline 生成的知识整理系统，适合 related work/landscape synthesis。

### `protocol_competitor`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。

### `research_os_competitor`

- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.

### `risk_model`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **MARCH** (`march`): Solver/Proposer/Checker 信息隔离，可减少 verifier 重复 generator 错误的自我确认偏差。
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **AgentHallu** (`agenthallu`): 评估多步 agent hallucination 及 step localization，适合设计错误归因机制。
- **Catfish Agent** (`catfish-agent`): 通过故意注入结构化异议缓解多 agent silent agreement。
- **OpenReviewer** (`openreviewer`): 面向 ML/AI 论文评审的模型/系统，适合 reviewer text style 和 rubric。

### `runtime_component`

- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **KAIJU** (`kaiju`): Execution kernel for LLM agents separating reasoning from tool execution with scheduling/tool dispatch/security gates.
- **AIDE ML** (`aide-ml`): 面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。
- **Bilevel Autoresearch** (`bilevel-autoresearch`): Meta-autoresearch loop where outer loop modifies inner autoresearch mechanism.
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **OpenHands** (`openhands`): 通用 coding agent/agent runtime，可作为代码修复和 repo patch 执行底座。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。

### `unclassified`

- **AblationBench** (`ablationbench`): 评估 LM 系统找回/规划 ablation 的能力，可作为 reviewer-to-experiment 机制参考。
- **AutoP2C** (`autop2c`): 从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。
- **SciIntegrity-Bench** (`sciintegrity-bench`): 科研诚信 benchmark，强调缺失数据、伪造结果、数据合成等问题。
- **Agent Laboratory** (`agent-laboratory`): human-in-the-loop 自动科研：literature review、experimentation、report writing。
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **HiRAS** (`hiras`): 层级多 agent paper-to-code/execution framework。
- **PaperCoder / Paper2Code** (`papercoder-paper2code`): 把 ML paper 转成 functional code repository 的研究原型。
- **SkillFoundry** (`scientific-skills-skillfoundry`): 从异构科学资源挖掘并验证 skill 的 self-evolving framework；可作为未来扩展调查 skill 的方法参考。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **Denario** (`denario`): multi-agent scientific assistant，偏数据分析和 LaTeX article 生成。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。

### `watch_only`

- **Bilevel Autoresearch** (`bilevel-autoresearch`): Meta-autoresearch loop where outer loop modifies inner autoresearch mechanism.

## By Research Field

### `ai_agent_systems`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **The AI Scientist v2** (`ai-scientist-v2`): 自动科学发现代表项目，v2 减少模板依赖，强调 agentic tree search；需要重点关注安全和可验证性不足。
- **Claw AI Lab** (`claw-ai-lab`): 把自动科研变成 dashboard 化 AI lab，突出 artifact inspector、rollback/resume、reproduce mode。
- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **KAIJU** (`kaiju`): Execution kernel for LLM agents separating reasoning from tool execution with scheduling/tool dispatch/security gates.
- **MARCH** (`march`): Solver/Proposer/Checker 信息隔离，可减少 verifier 重复 generator 错误的自我确认偏差。
- **Paper2Agent** (`paper2agent`): Converts papers and codebases into interactive MCP agents; useful paper-to-agent direction but not claim-state governance.
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **Robin / Future House** (`robin`): 多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。
- **scientific-agent-skills / ML-AI skill authoring** (`scientific-agent-skills`): 高质量 skill 编写范式：SKILL.md 做路由，references 放长决策，scripts 放稳定 scaffold，tests 放验证。适合作为我们调查 skill 仓库的结构样板。
- **Agent Laboratory** (`agent-laboratory`): human-in-the-loop 自动科研：literature review、experimentation、report writing。
- **AgentHallu** (`agenthallu`): 评估多步 agent hallucination 及 step localization，适合设计错误归因机制。
- **AgentRxiv** (`agentrxiv`): Shared preprint / report server for autonomous research agents; useful collaboration bus but not accountability kernel.
- **The AI Scientist** (`ai-scientist-v1`): 早期端到端自动科研标志项目，强依赖模板和领域约束，是 idea-to-paper 叙事起点。
- **AIDE ML** (`aide-ml`): 面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **Bilevel Autoresearch** (`bilevel-autoresearch`): Meta-autoresearch loop where outer loop modifies inner autoresearch mechanism.
- **BioMARS** (`biomars`): Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。
- **Catfish Agent** (`catfish-agent`): 通过故意注入结构化异议缓解多 agent silent agreement。
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **EvoScientist** (`evoscientist`): Self-evolving AI scientist with persistent memory and long-term improvement.
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **Google Co-Scientist** (`google-co-scientist`): 基于 Gemini 的多 agent hypothesis generation，使用 tournament evolution 和 test-time compute。
- **MLE-bench** (`mle-bench`): Kaggle-style ML engineering benchmark，可作为训练/调参能力测量参考。
- **OpenAGS** (`openags`): Open Autonomous Generalist Scientist / auto-research framework; adjacent broad autonomous research project.
- **OpenHands** (`openhands`): 通用 coding agent/agent runtime，可作为代码修复和 repo patch 执行底座。
- **SkillFoundry** (`scientific-skills-skillfoundry`): 从异构科学资源挖掘并验证 skill 的 self-evolving framework；可作为未来扩展调查 skill 的方法参考。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **Denario** (`denario`): multi-agent scientific assistant，偏数据分析和 LaTeX article 生成。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。
- **Virtual Lab** (`virtual-lab`): 模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。

### `ai_ml_algorithm_research`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **PaperBench** (`paperbench`): 评估 agent 复现 ICML 论文的关键 benchmark；证明 paper reproduction 仍很难。
- **RePro / RefP2C** (`repro-refp2c`): Fine-grained paper fingerprint extraction and iterative verification/refinement for paper-to-code; important verifier-design reference.
- **AblationBench** (`ablationbench`): 评估 LM 系统找回/规划 ablation 的能力，可作为 reviewer-to-experiment 机制参考。
- **The AI Scientist v2** (`ai-scientist-v2`): 自动科学发现代表项目，v2 减少模板依赖，强调 agentic tree search；需要重点关注安全和可验证性不足。
- **AutoExperiment** (`autoexperiment`): 通过 progressive code masking 测 agent 从 paper + masked codebase 中补全代码、执行实验、复现结果。
- **AutoP2C** (`autop2c`): 从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **Claw AI Lab** (`claw-ai-lab`): 把自动科研变成 dashboard 化 AI lab，突出 artifact inspector、rollback/resume、reproduce mode。
- **Agent Laboratory** (`agent-laboratory`): human-in-the-loop 自动科研：literature review、experimentation、report writing。
- **The AI Scientist** (`ai-scientist-v1`): 早期端到端自动科研标志项目，强依赖模板和领域约束，是 idea-to-paper 叙事起点。
- **AIDE ML** (`aide-ml`): 面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **HiRAS** (`hiras`): 层级多 agent paper-to-code/execution framework。
- **MLE-bench** (`mle-bench`): Kaggle-style ML engineering benchmark，可作为训练/调参能力测量参考。
- **PaperCoder / Paper2Code** (`papercoder-paper2code`): 把 ML paper 转成 functional code repository 的研究原型。
- **ResearchCodeBench** (`researchcodebench`): 测试 LLM 实现 ML 论文中新研究 idea 的 coding benchmark。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。
- **Denario** (`denario`): multi-agent scientific assistant，偏数据分析和 LaTeX article 生成。
- **MLAgentBench** (`mlagentbench`): 评估 ML experimentation agent 的早期 benchmark。

### `biomedicine_drug_discovery`

- **Robin / Future House** (`robin`): 多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。
- **BioMARS** (`biomars`): Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。
- **Google Co-Scientist** (`google-co-scientist`): 基于 Gemini 的多 agent hypothesis generation，使用 tournament evolution 和 test-time compute。
- **Virtual Lab** (`virtual-lab`): 模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。

### `general_literature_research`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **Robin / Future House** (`robin`): 多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。
- **AgentRxiv** (`agentrxiv`): Shared preprint / report server for autonomous research agents; useful collaboration bus but not accountability kernel.
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **BioMARS** (`biomars`): Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。
- **DeepXiv-SDK** (`deepxiv-sdk`): Scientific literature data interface / SDK / MCP layer for agents.
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **Google Co-Scientist** (`google-co-scientist`): 基于 Gemini 的多 agent hypothesis generation，使用 tournament evolution 和 test-time compute。
- **PaperQA2** (`paperqa2`): 科学文献 RAG / QA / contradiction detection，可作为 paper understanding 与 evidence retrieval 层。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **GPT Researcher** (`gpt-researcher`): web/local research report agent，适合 literature/source aggregation。
- **Open Deep Research** (`open-deep-research`): 开源 deep research agent，可参考 research graph 和 MCP/tool 接入。
- **OpenReviewer** (`openreviewer`): 面向 ML/AI 论文评审的模型/系统，适合 reviewer text style 和 rubric。
- **OpenScholar** (`openscholar`): 科学文献 synthesis，citation-backed responses 和 ScholarQABench。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。
- **STORM / Co-STORM** (`storm`): 多视角提问和 outline 生成的知识整理系统，适合 related work/landscape synthesis。
- **Virtual Lab** (`virtual-lab`): 模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。

### `metascience_integrity_review`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **AblationBench** (`ablationbench`): 评估 LM 系统找回/规划 ablation 的能力，可作为 reviewer-to-experiment 机制参考。
- **MARCH** (`march`): Solver/Proposer/Checker 信息隔离，可减少 verifier 重复 generator 错误的自我确认偏差。
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **SciIntegrity-Bench** (`sciintegrity-bench`): 科研诚信 benchmark，强调缺失数据、伪造结果、数据合成等问题。
- **AgentHallu** (`agenthallu`): 评估多步 agent hallucination 及 step localization，适合设计错误归因机制。
- **AICat** (`aicat`): AI catalogue vocabulary/approach for machine-readable AI system metadata to support EU AI Act-style registration and transparency.
- **Catfish Agent** (`catfish-agent`): 通过故意注入结构化异议缓解多 agent silent agreement。
- **PaperQA2** (`paperqa2`): 科学文献 RAG / QA / contradiction detection，可作为 paper understanding 与 evidence retrieval 层。
- **OpenReviewer** (`openreviewer`): 面向 ML/AI 论文评审的模型/系统，适合 reviewer text style 和 rubric。
- **OpenScholar** (`openscholar`): 科学文献 synthesis，citation-backed responses 和 ScholarQABench。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。

### `scientific_infrastructure`

- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **DVC** (`dvc`): 数据/模型/管线版本化，适合作为数据和大 artifact 版本层。
- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **MLflow** (`mlflow`): run tracking、params/metrics/artifacts、model registry；适合作为 Run Ledger 后端。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **AICat** (`aicat`): AI catalogue vocabulary/approach for machine-readable AI system metadata to support EU AI Act-style registration and transparency.
- **DeepXiv-SDK** (`deepxiv-sdk`): Scientific literature data interface / SDK / MCP layer for agents.
- **Hydra** (`hydra`): 配置组合、CLI override、multirun/sweeps，适合 Experiment Spec 编译。
- **Snakemake / Nextflow** (`snakemake-nextflow`): 可复现 workflow DAG，适合复杂数据分析 pipeline。
- **Weights & Biases** (`wandb`): 实验 dashboard 和 artifact lineage，可作为可视化/协作层。

### `software_engineering`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **PaperBench** (`paperbench`): 评估 agent 复现 ICML 论文的关键 benchmark；证明 paper reproduction 仍很难。
- **RePro / RefP2C** (`repro-refp2c`): Fine-grained paper fingerprint extraction and iterative verification/refinement for paper-to-code; important verifier-design reference.
- **AblationBench** (`ablationbench`): 评估 LM 系统找回/规划 ablation 的能力，可作为 reviewer-to-experiment 机制参考。
- **AutoExperiment** (`autoexperiment`): 通过 progressive code masking 测 agent 从 paper + masked codebase 中补全代码、执行实验、复现结果。
- **AutoP2C** (`autop2c`): 从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **DVC** (`dvc`): 数据/模型/管线版本化，适合作为数据和大 artifact 版本层。
- **KAIJU** (`kaiju`): Execution kernel for LLM agents separating reasoning from tool execution with scheduling/tool dispatch/security gates.
- **MLflow** (`mlflow`): run tracking、params/metrics/artifacts、model registry；适合作为 Run Ledger 后端。
- **Paper2Agent** (`paper2agent`): Converts papers and codebases into interactive MCP agents; useful paper-to-agent direction but not claim-state governance.
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **scientific-agent-skills / ML-AI skill authoring** (`scientific-agent-skills`): 高质量 skill 编写范式：SKILL.md 做路由，references 放长决策，scripts 放稳定 scaffold，tests 放验证。适合作为我们调查 skill 仓库的结构样板。
- **AIDE ML** (`aide-ml`): 面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。
- **Bilevel Autoresearch** (`bilevel-autoresearch`): Meta-autoresearch loop where outer loop modifies inner autoresearch mechanism.
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **HiRAS** (`hiras`): 层级多 agent paper-to-code/execution framework。
- **Hydra** (`hydra`): 配置组合、CLI override、multirun/sweeps，适合 Experiment Spec 编译。
- **MLE-bench** (`mle-bench`): Kaggle-style ML engineering benchmark，可作为训练/调参能力测量参考。
- **OpenHands** (`openhands`): 通用 coding agent/agent runtime，可作为代码修复和 repo patch 执行底座。
- **PaperCoder / Paper2Code** (`papercoder-paper2code`): 把 ML paper 转成 functional code repository 的研究原型。
- **ResearchCodeBench** (`researchcodebench`): 测试 LLM 实现 ML 论文中新研究 idea 的 coding benchmark。
- **SkillFoundry** (`scientific-skills-skillfoundry`): 从异构科学资源挖掘并验证 skill 的 self-evolving framework；可作为未来扩展调查 skill 的方法参考。
- **Snakemake / Nextflow** (`snakemake-nextflow`): 可复现 workflow DAG，适合复杂数据分析 pipeline。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。
- **Weights & Biases** (`wandb`): 实验 dashboard 和 artifact lineage，可作为可视化/协作层。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **MLAgentBench** (`mlagentbench`): 评估 ML experimentation agent 的早期 benchmark。

### `wetlab_lab_automation`

- **Robin / Future House** (`robin`): 多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。
- **BioMARS** (`biomars`): Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。
- **Virtual Lab** (`virtual-lab`): 模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。

## By Workflow Scope

### `agent_runtime`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **KAIJU** (`kaiju`): Execution kernel for LLM agents separating reasoning from tool execution with scheduling/tool dispatch/security gates.
- **Paper2Agent** (`paper2agent`): Converts papers and codebases into interactive MCP agents; useful paper-to-agent direction but not claim-state governance.
- **scientific-agent-skills / ML-AI skill authoring** (`scientific-agent-skills`): 高质量 skill 编写范式：SKILL.md 做路由，references 放长决策，scripts 放稳定 scaffold，tests 放验证。适合作为我们调查 skill 仓库的结构样板。
- **AgentRxiv** (`agentrxiv`): Shared preprint / report server for autonomous research agents; useful collaboration bus but not accountability kernel.
- **AIDE ML** (`aide-ml`): 面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。
- **Bilevel Autoresearch** (`bilevel-autoresearch`): Meta-autoresearch loop where outer loop modifies inner autoresearch mechanism.
- **DeepXiv-SDK** (`deepxiv-sdk`): Scientific literature data interface / SDK / MCP layer for agents.
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **EvoScientist** (`evoscientist`): Self-evolving AI scientist with persistent memory and long-term improvement.
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **MLE-bench** (`mle-bench`): Kaggle-style ML engineering benchmark，可作为训练/调参能力测量参考。
- **OpenAGS** (`openags`): Open Autonomous Generalist Scientist / auto-research framework; adjacent broad autonomous research project.
- **OpenHands** (`openhands`): 通用 coding agent/agent runtime，可作为代码修复和 repo patch 执行底座。
- **SkillFoundry** (`scientific-skills-skillfoundry`): 从异构科学资源挖掘并验证 skill 的 self-evolving framework；可作为未来扩展调查 skill 的方法参考。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。

### `claim_to_review_issue`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。
- **AblationBench** (`ablationbench`): 评估 LM 系统找回/规划 ablation 的能力，可作为 reviewer-to-experiment 机制参考。
- **MARCH** (`march`): Solver/Proposer/Checker 信息隔离，可减少 verifier 重复 generator 错误的自我确认偏差。
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **SciIntegrity-Bench** (`sciintegrity-bench`): 科研诚信 benchmark，强调缺失数据、伪造结果、数据合成等问题。
- **AgentHallu** (`agenthallu`): 评估多步 agent hallucination 及 step localization，适合设计错误归因机制。
- **Catfish Agent** (`catfish-agent`): 通过故意注入结构化异议缓解多 agent silent agreement。
- **OpenReviewer** (`openreviewer`): 面向 ML/AI 论文评审的模型/系统，适合 reviewer text style 和 rubric。

### `experiment_to_claim`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **MARCH** (`march`): Solver/Proposer/Checker 信息隔离，可减少 verifier 重复 generator 错误的自我确认偏差。
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **AgentHallu** (`agenthallu`): 评估多步 agent hallucination 及 step localization，适合设计错误归因机制。
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **Catfish Agent** (`catfish-agent`): 通过故意注入结构化异议缓解多 agent silent agreement。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。

### `idea_to_experiment`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **The AI Scientist v2** (`ai-scientist-v2`): 自动科学发现代表项目，v2 减少模板依赖，强调 agentic tree search；需要重点关注安全和可验证性不足。
- **AutoP2C** (`autop2c`): 从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。
- **Claw AI Lab** (`claw-ai-lab`): 把自动科研变成 dashboard 化 AI lab，突出 artifact inspector、rollback/resume、reproduce mode。
- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **Robin / Future House** (`robin`): 多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。
- **Agent Laboratory** (`agent-laboratory`): human-in-the-loop 自动科研：literature review、experimentation、report writing。
- **The AI Scientist** (`ai-scientist-v1`): 早期端到端自动科研标志项目，强依赖模板和领域约束，是 idea-to-paper 叙事起点。
- **AIDE ML** (`aide-ml`): 面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **BioMARS** (`biomars`): Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **EvoScientist** (`evoscientist`): Self-evolving AI scientist with persistent memory and long-term improvement.
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **Google Co-Scientist** (`google-co-scientist`): 基于 Gemini 的多 agent hypothesis generation，使用 tournament evolution 和 test-time compute。
- **Hydra** (`hydra`): 配置组合、CLI override、multirun/sweeps，适合 Experiment Spec 编译。
- **OpenAGS** (`openags`): Open Autonomous Generalist Scientist / auto-research framework; adjacent broad autonomous research project.
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **Denario** (`denario`): multi-agent scientific assistant，偏数据分析和 LaTeX article 生成。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。
- **Virtual Lab** (`virtual-lab`): 模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。

### `idea_to_paper`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **The AI Scientist v2** (`ai-scientist-v2`): 自动科学发现代表项目，v2 减少模板依赖，强调 agentic tree search；需要重点关注安全和可验证性不足。
- **Claw AI Lab** (`claw-ai-lab`): 把自动科研变成 dashboard 化 AI lab，突出 artifact inspector、rollback/resume、reproduce mode。
- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **Agent Laboratory** (`agent-laboratory`): human-in-the-loop 自动科研：literature review、experimentation、report writing。
- **The AI Scientist** (`ai-scientist-v1`): 早期端到端自动科研标志项目，强依赖模板和领域约束，是 idea-to-paper 叙事起点。
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **EvoScientist** (`evoscientist`): Self-evolving AI scientist with persistent memory and long-term improvement.
- **OpenAGS** (`openags`): Open Autonomous Generalist Scientist / auto-research framework; adjacent broad autonomous research project.
- **Denario** (`denario`): multi-agent scientific assistant，偏数据分析和 LaTeX article 生成。
- **GPT Researcher** (`gpt-researcher`): web/local research report agent，适合 literature/source aggregation。
- **Open Deep Research** (`open-deep-research`): 开源 deep research agent，可参考 research graph 和 MCP/tool 接入。

### `lab_discovery_loop`

- **Robin / Future House** (`robin`): 多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **BioMARS** (`biomars`): Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **Google Co-Scientist** (`google-co-scientist`): 基于 Gemini 的多 agent hypothesis generation，使用 tournament evolution 和 test-time compute。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。
- **Virtual Lab** (`virtual-lab`): 模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。

### `literature_to_report`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **AgentRxiv** (`agentrxiv`): Shared preprint / report server for autonomous research agents; useful collaboration bus but not accountability kernel.
- **DeepXiv-SDK** (`deepxiv-sdk`): Scientific literature data interface / SDK / MCP layer for agents.
- **PaperQA2** (`paperqa2`): 科学文献 RAG / QA / contradiction detection，可作为 paper understanding 与 evidence retrieval 层。
- **GPT Researcher** (`gpt-researcher`): web/local research report agent，适合 literature/source aggregation。
- **Open Deep Research** (`open-deep-research`): 开源 deep research agent，可参考 research graph 和 MCP/tool 接入。
- **OpenReviewer** (`openreviewer`): 面向 ML/AI 论文评审的模型/系统，适合 reviewer text style 和 rubric。
- **OpenScholar** (`openscholar`): 科学文献 synthesis，citation-backed responses 和 ScholarQABench。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。
- **STORM / Co-STORM** (`storm`): 多视角提问和 outline 生成的知识整理系统，适合 related work/landscape synthesis。

### `paper_to_code`

- **PaperBench** (`paperbench`): 评估 agent 复现 ICML 论文的关键 benchmark；证明 paper reproduction 仍很难。
- **RePro / RefP2C** (`repro-refp2c`): Fine-grained paper fingerprint extraction and iterative verification/refinement for paper-to-code; important verifier-design reference.
- **AutoExperiment** (`autoexperiment`): 通过 progressive code masking 测 agent 从 paper + masked codebase 中补全代码、执行实验、复现结果。
- **AutoP2C** (`autop2c`): 从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **Paper2Agent** (`paper2agent`): Converts papers and codebases into interactive MCP agents; useful paper-to-agent direction but not claim-state governance.
- **HiRAS** (`hiras`): 层级多 agent paper-to-code/execution framework。
- **PaperCoder / Paper2Code** (`papercoder-paper2code`): 把 ML paper 转成 functional code repository 的研究原型。
- **ResearchCodeBench** (`researchcodebench`): 测试 LLM 实现 ML 论文中新研究 idea 的 coding benchmark。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。

### `paper_to_reproduction`

- **PaperBench** (`paperbench`): 评估 agent 复现 ICML 论文的关键 benchmark；证明 paper reproduction 仍很难。
- **RePro / RefP2C** (`repro-refp2c`): Fine-grained paper fingerprint extraction and iterative verification/refinement for paper-to-code; important verifier-design reference.
- **AutoExperiment** (`autoexperiment`): 通过 progressive code masking 测 agent 从 paper + masked codebase 中补全代码、执行实验、复现结果。
- **AutoP2C** (`autop2c`): 从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **Paper2Agent** (`paper2agent`): Converts papers and codebases into interactive MCP agents; useful paper-to-agent direction but not claim-state governance.
- **HiRAS** (`hiras`): 层级多 agent paper-to-code/execution framework。
- **PaperCoder / Paper2Code** (`papercoder-paper2code`): 把 ML paper 转成 functional code repository 的研究原型。
- **ResearchCodeBench** (`researchcodebench`): 测试 LLM 实现 ML 论文中新研究 idea 的 coding benchmark。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。

### `repo_to_experiment`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **PaperBench** (`paperbench`): 评估 agent 复现 ICML 论文的关键 benchmark；证明 paper reproduction 仍很难。
- **AblationBench** (`ablationbench`): 评估 LM 系统找回/规划 ablation 的能力，可作为 reviewer-to-experiment 机制参考。
- **The AI Scientist v2** (`ai-scientist-v2`): 自动科学发现代表项目，v2 减少模板依赖，强调 agentic tree search；需要重点关注安全和可验证性不足。
- **AutoExperiment** (`autoexperiment`): 通过 progressive code masking 测 agent 从 paper + masked codebase 中补全代码、执行实验、复现结果。
- **AutoP2C** (`autop2c`): 从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **Claw AI Lab** (`claw-ai-lab`): 把自动科研变成 dashboard 化 AI lab，突出 artifact inspector、rollback/resume、reproduce mode。
- **AIDE ML** (`aide-ml`): 面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。
- **Bilevel Autoresearch** (`bilevel-autoresearch`): Meta-autoresearch loop where outer loop modifies inner autoresearch mechanism.
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **Hydra** (`hydra`): 配置组合、CLI override、multirun/sweeps，适合 Experiment Spec 编译。
- **MLE-bench** (`mle-bench`): Kaggle-style ML engineering benchmark，可作为训练/调参能力测量参考。
- **OpenHands** (`openhands`): 通用 coding agent/agent runtime，可作为代码修复和 repo patch 执行底座。
- **ResearchCodeBench** (`researchcodebench`): 测试 LLM 实现 ML 论文中新研究 idea 的 coding benchmark。
- **SkillFoundry** (`scientific-skills-skillfoundry`): 从异构科学资源挖掘并验证 skill 的 self-evolving framework；可作为未来扩展调查 skill 的方法参考。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **MLAgentBench** (`mlagentbench`): 评估 ML experimentation agent 的早期 benchmark。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。

### `tracking_infrastructure`

- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **DVC** (`dvc`): 数据/模型/管线版本化，适合作为数据和大 artifact 版本层。
- **MLflow** (`mlflow`): run tracking、params/metrics/artifacts、model registry；适合作为 Run Ledger 后端。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **AICat** (`aicat`): AI catalogue vocabulary/approach for machine-readable AI system metadata to support EU AI Act-style registration and transparency.
- **Hydra** (`hydra`): 配置组合、CLI override、multirun/sweeps，适合 Experiment Spec 编译。
- **Snakemake / Nextflow** (`snakemake-nextflow`): 可复现 workflow DAG，适合复杂数据分析 pipeline。
- **Weights & Biases** (`wandb`): 实验 dashboard 和 artifact lineage，可作为可视化/协作层。

## By Accountability Feature

### `agent_trace`

- **ARIS / Auto-claude-code-research-in-sleep** (`aris`): Markdown-only skill harness，强调 cross-model review、experiment bridge、research wiki、result-to-claim、paper claim audit。
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **KAIJU** (`kaiju`): Execution kernel for LLM agents separating reasoning from tool execution with scheduling/tool dispatch/security gates.
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **AgentHallu** (`agenthallu`): 评估多步 agent hallucination 及 step localization，适合设计错误归因机制。
- **AutoScientists** (`autoscientists`): 去中心化实验团队，共享 experimental state 和成功/失败记忆。
- **Bilevel Autoresearch** (`bilevel-autoresearch`): Meta-autoresearch loop where outer loop modifies inner autoresearch mechanism.
- **EvoScientist** (`evoscientist`): Self-evolving AI scientist with persistent memory and long-term improvement.
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **OpenAGS** (`openags`): Open Autonomous Generalist Scientist / auto-research framework; adjacent broad autonomous research project.
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。

### `artifact_lineage`

- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **Claw AI Lab** (`claw-ai-lab`): 把自动科研变成 dashboard 化 AI lab，突出 artifact inspector、rollback/resume、reproduce mode。
- **DVC** (`dvc`): 数据/模型/管线版本化，适合作为数据和大 artifact 版本层。
- **MLflow** (`mlflow`): run tracking、params/metrics/artifacts、model registry；适合作为 Run Ledger 后端。
- **Paper2Agent** (`paper2agent`): Converts papers and codebases into interactive MCP agents; useful paper-to-agent direction but not claim-state governance.
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **FM Agent** (`fm-agent`): Distributed multi-agent/evolutionary R&D execution framework across OR/ML/GPU/mathematics; relevant execution competitor.
- **Weights & Biases** (`wandb`): 实验 dashboard 和 artifact lineage，可作为可视化/协作层。

### `claim_ledger`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。

### `decision_log`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **DeepScientist** (`deep-scientist`): Local-first research operating system / workspace; strong research-map, memory, takeover, and artifact workflow, but not a claim-state transition kernel.
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **KAIJU** (`kaiju`): Execution kernel for LLM agents separating reasoning from tool execution with scheduling/tool dispatch/security gates.
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **SkillFoundry** (`scientific-skills-skillfoundry`): 从异构科学资源挖掘并验证 skill 的 self-evolving framework；可作为未来扩展调查 skill 的方法参考。

### `experiment_spec`

- **RePro / RefP2C** (`repro-refp2c`): Fine-grained paper fingerprint extraction and iterative verification/refinement for paper-to-code; important verifier-design reference.

### `failure_registry`

- **AutoResearchClaw** (`autoresearchclaw`): 万星级 idea-to-paper 自动科研流水线；23-stage pipeline、skills、sandbox experiments、multi-agent review、VerifiedRegistry/HITL/self-healing 是重要对标。
- **Sibyl-AutoResearch** (`sibyl-autoresearch`): Scientific trial-and-error harness emphasizing bounded trials, failure registry, artifact traces, roles, memory, and gates.
- **The AI Scientist v2** (`ai-scientist-v2`): 自动科学发现代表项目，v2 减少模板依赖，强调 agentic tree search；需要重点关注安全和可验证性不足。
- **AutoExperiment** (`autoexperiment`): 通过 progressive code masking 测 agent 从 paper + masked codebase 中补全代码、执行实验、复现结果。
- **AutoP2C** (`autop2c`): 从多模态论文内容生成可执行代码仓库的 paper-to-code 原型。
- **Hyperspace AGI** (`hyperspace-agi`): Distributed P2P autonomous research/swarm system with shared state and paper output.
- **AIDE ML** (`aide-ml`): 面向 ML 代码、debug、benchmark、metric optimization 的 agent，可作为 Experiment Implementer 参考。
- **ERA / Empirical Research Assistance** (`era`): LLM + tree search 生成 expert-level scientific software；说明程序化 objective 比自然语言 judge 更可靠。
- **EvoScientist** (`evoscientist`): Self-evolving AI scientist with persistent memory and long-term improvement.
- **OpenHands** (`openhands`): 通用 coding agent/agent runtime，可作为代码修复和 repo patch 执行底座。
- **SkillFoundry** (`scientific-skills-skillfoundry`): 从异构科学资源挖掘并验证 skill 的 self-evolving framework；可作为未来扩展调查 skill 的方法参考。
- **ToolMaker** (`toolmaker`): 把 papers-with-code repo 适配为 agent 可调用 tool 的思路，可作为 repo adapter 层参考。
- **CASCADE** (`cascade-skills`): 自主演化 skill acquisition，从 LLM+tool use 走向 LLM+skill acquisition。
- **STELLA** (`stella`): self-evolving biomedical agent，Template Library + Tool Ocean。

### `method_spec`

- **RePro / RefP2C** (`repro-refp2c`): Fine-grained paper fingerprint extraction and iterative verification/refinement for paper-to-code; important verifier-design reference.

### `provenance_graph`

- **AAR / Claim-Level Auditability** (`aar-claim-level-auditability`): Auditable Autonomous Research proposal emphasizing persistent queryable provenance graphs, provenance coverage/soundness, contradiction transparency, and audit effort.
- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **ScientistOne / Chain-of-Evidence** (`scientistone`): Chain-of-Evidence 和 CoE Audit：score verification、spec violation、reference verification、method-code alignment。
- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **AutoReproduce** (`autoreproduce`): Paper-lineage-based automated experimental reproduction; important for implicit knowledge extraction and reproduction benchmarks.
- **Pramana / ClaimAttestation** (`pramana`): 协议层：每个 consequential agent output 包装为 typed ClaimAttestation，并提供 verify()。
- **PROV-AGENT** (`prov-agent`): 扩展 W3C PROV 以记录 prompts、responses、decisions、tool calls 与 downstream impact。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **AgentHallu** (`agenthallu`): 评估多步 agent hallucination 及 step localization，适合设计错误归因机制。
- **AICat** (`aicat`): AI catalogue vocabulary/approach for machine-readable AI system metadata to support EU AI Act-style registration and transparency.

### `release_bundle`

- **data-to-paper** (`data-to-paper`): Backward-traceable AI-driven scientific manuscript pipeline; very relevant for data-to-claim traceability and human-verifiable manuscripts.
- **PaperBench** (`paperbench`): 评估 agent 复现 ICML 论文的关键 benchmark；证明 paper reproduction 仍很难。
- **DVC** (`dvc`): 数据/模型/管线版本化，适合作为数据和大 artifact 版本层。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **AgentRxiv** (`agentrxiv`): Shared preprint / report server for autonomous research agents; useful collaboration bus but not accountability kernel.
- **AICat** (`aicat`): AI catalogue vocabulary/approach for machine-readable AI system metadata to support EU AI Act-style registration and transparency.
- **Snakemake / Nextflow** (`snakemake-nextflow`): 可复现 workflow DAG，适合复杂数据分析 pipeline。

### `run_ledger`

- **EviBound** (`evibound`): 通过 pre-execution approval gate 和 post-execution MLflow verification gate 阻止 unsupported claim 传播。
- **DVC** (`dvc`): 数据/模型/管线版本化，适合作为数据和大 artifact 版本层。
- **MLflow** (`mlflow`): run tracking、params/metrics/artifacts、model registry；适合作为 Run Ledger 后端。
- **Snakemake / Nextflow** (`snakemake-nextflow`): 可复现 workflow DAG，适合复杂数据分析 pipeline。
- **Weights & Biases** (`wandb`): 实验 dashboard 和 artifact lineage，可作为可视化/协作层。

## Open / Proposed New Facets

No custom facets currently recorded. If a new project brings a new attribute, add it under `facets.custom_facets` and regenerate this overview.

## Maintenance Commands

```bash
python skills/autoresearch-landscape-survey/scripts/check_new_project.py --input path/to/intake.yaml --format markdown
python skills/autoresearch-landscape-survey/scripts/classify_project_from_yaml.py --input path/to/intake.yaml --format yaml
python skills/autoresearch-landscape-survey/scripts/prepare_ai_project_review.py --input path/to/intake.yaml
python skills/autoresearch-landscape-survey/scripts/render_project_overview.py
python skills/autoresearch-landscape-survey/scripts/prepare_ai_report_prompt.py
```
