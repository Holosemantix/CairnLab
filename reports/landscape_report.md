# AutoResearch Landscape Report

Last updated: 2026-06-03

## Executive Summary

This report maps AutoResearch-related projects by non-orthogonal facets rather than by rigid layers. The central distinction for our work is not whether a project is an agent, a skill library, or a benchmark, but whether it can make research claims executable, verifiable, and accountable.

Our target is CairnLab Research Claim Kernel: claim → lifecycle context → risk/responsibility → evidence policy → evidence item → verifier certificate → state transition → governance/human gate → decision trace package.

## Deep-Dive Projects

| Project | Fit | Workflow | Accountability | Relevance | Detailed Analysis |
| --- | --- | --- | --- | ---: | --- |
| AAR / Claim-Level Auditability | claim_kernel_reference, protocol_competitor, risk_model | literature_to_report, claim_to_review_issue | provenance_graph, claim_ledger, decision_log | 5 |  |
| ARIS / Auto-claude-code-research-in-sleep | adjacent_competitor | agent_runtime, experiment_to_claim, idea_to_experiment, idea_to_paper | agent_trace | 5 | skills/autoresearch-landscape-survey/references/deep-dives/aris.md |
| AutoResearchClaw | adjacent_competitor | agent_runtime, claim_to_review_issue, idea_to_experiment, idea_to_paper | artifact_lineage, claim_ledger, failure_registry | 5 | skills/autoresearch-landscape-survey/references/deep-dives/autoresearchclaw.md |
| data-to-paper | claim_kernel_reference, governance_reference | experiment_to_claim, claim_to_review_issue | provenance_graph, artifact_lineage, release_bundle, decision_log | 5 |  |
| DeepScientist | research_os_competitor, adjacent_competitor | agent_runtime, idea_to_experiment, idea_to_paper, repo_to_experiment | agent_trace, artifact_lineage, decision_log | 5 |  |
| EviBound | core_competitor, risk_model, kernel_competitor | claim_to_review_issue, experiment_to_claim | artifact_lineage, claim_ledger, run_ledger | 5 | skills/autoresearch-landscape-survey/references/deep-dives/accountability-layer.md |
| PaperBench | benchmark_reference | paper_to_code, paper_to_reproduction, repo_to_experiment | release_bundle | 5 | skills/autoresearch-landscape-survey/references/deep-dives/paperbench-and-reproduction-benchmarks.md |
| ScientistOne / Chain-of-Evidence | core_competitor, risk_model, kernel_competitor | claim_to_review_issue, experiment_to_claim | agent_trace, claim_ledger, decision_log, provenance_graph | 5 | skills/autoresearch-landscape-survey/references/deep-dives/accountability-layer.md |
| Sibyl-AutoResearch | adjacent_competitor, governance_reference, risk_model | idea_to_experiment, experiment_to_claim, agent_runtime | failure_registry, artifact_lineage, decision_log, agent_trace | 5 |  |
| Robin / Future House | domain_inspiration | idea_to_experiment, lab_discovery_loop |  | 4 | skills/autoresearch-landscape-survey/references/deep-dives/robin.md |
| scientific-agent-skills / ML-AI skill authoring | plugin_layer | agent_runtime |  | 4 | skills/autoresearch-landscape-survey/references/deep-dives/scientific-agent-skills-ml-ai.md |

## Why Facets Instead of Layers

The same project can cover several concerns: ARIS is a Markdown skill harness, a workflow orchestrator, an experiment bridge, and a partial result-to-claim system. AutoResearchClaw is an idea-to-paper pipeline, a skill matcher, a sandbox runner, a review loop, and a partial anti-fabrication system. Therefore the registry records multiple facets for each project.

## Field-Based Map

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

### `wetlab_lab_automation`

- **Robin / Future House** (`robin`): 多 agent 生物医学发现系统，编排文献 agent 与数据分析 agent，提出假设、实验、结果解释与 follow-up；证明 lab-in-the-loop 科学发现可行。
- **BioMARS** (`biomars`): Biologist/Technician/Inspector 三层结构，Inspector role 对我们很有启发。
- **Virtual Lab** (`virtual-lab`): 模拟 PI/senior scientist 角色的 virtual lab，适合研究多角色组织。

## Workflow / Process Map

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

### `tracking_infrastructure`

- **AAGATE** (`aagate`): NIST AI RMF-aligned Kubernetes-native governance control plane for agentic AI; useful governance practice reference.
- **DVC** (`dvc`): 数据/模型/管线版本化，适合作为数据和大 artifact 版本层。
- **MLflow** (`mlflow`): run tracking、params/metrics/artifacts、model registry；适合作为 Run Ledger 后端。
- **ReproZip / RO-Crate / W3C PROV** (`reprozip-rocrate-prov`): 环境捕获、研究对象打包与通用 provenance 标准，适合 release/replay 层。
- **AICat** (`aicat`): AI catalogue vocabulary/approach for machine-readable AI system metadata to support EU AI Act-style registration and transparency.
- **Hydra** (`hydra`): 配置组合、CLI override、multirun/sweeps，适合 Experiment Spec 编译。
- **Snakemake / Nextflow** (`snakemake-nextflow`): 可复现 workflow DAG，适合复杂数据分析 pipeline。
- **Weights & Biases** (`wandb`): 实验 dashboard 和 artifact lineage，可作为可视化/协作层。

## Fit to Our Target

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

## Projects Most Similar to CairnLab Research Claim Kernel

| Project | Starting Points | Outputs | Key Gap |
| --- | --- | --- | --- |
| AAR / Claim-Level Auditability |  |  | Strong auditability theory; does not implement a reusable claim-state runtime. |
| ARIS / Auto-claude-code-research-in-sleep | workflow_task | code_repo_or_patch, experiment_plan, review_or_rebuttal, training_run_results | 更偏自己的研究推进和 paper improvement |
| AutoResearchClaw | topic_or_idea | hypothesis_or_idea, paper_draft_or_tex, review_or_rebuttal, training_run_results | 中心对象仍是 topic/idea 到 paper，不是外部论文 claim-by-claim 复现审计 |
| data-to-paper |  |  | Strong backward traceability, but not a reusable claim lifecycle kernel. |
| DeepScientist |  |  | Occupies Research OS/workspace narrative; claim-state semantics and verifier certificates are not the core object. |
| EviBound | paper_pdf_or_preprint | verified_claims | 不是完整 paper-to-code-to-review pipeline |
| PaperBench | draft_paper, paper_pdf_or_preprint | benchmark_scores, literature_report, paper_draft_or_tex, reproduction_bundle | 是 benchmark，不是平台 |
| RePro / RefP2C |  |  | Fine-grained criteria are close to evidence policy, but system scope is paper-to-code rather than lifecycle kernel. |
| ScientistOne / Chain-of-Evidence | paper_pdf_or_preprint | verified_claims | 仍需扩到 run ledger、training reproduction 和 reviewer issue closure |
| Sibyl-AutoResearch |  |  | Very close process-governance work; still not a reusable claim lifecycle transition kernel. |
| AAGATE |  |  | Governance control plane, not scientific claim lifecycle kernel. |
| AblationBench | experiment_plan | benchmark_scores, experiment_plan, review_or_rebuttal, training_run_results | 没有 execution/provenance 层 |
| The AI Scientist v2 | topic_or_idea | hypothesis_or_idea, paper_draft_or_tex, training_run_results | 不是 paper-first reproduction |
| AutoExperiment | experiment_plan | benchmark_scores, experiment_plan, training_run_results | benchmark 而非系统，缺 review/accountability layer |
| AutoP2C | draft_paper, paper_pdf_or_preprint | code_repo_or_patch, experiment_plan, literature_report, paper_draft_or_tex | 可运行 repo 不等于 claim-level verified reproduction |

## Maintenance Protocol

Run `check_new_project.py` and `prepare_ai_project_review.py` before adding a new item. Projects with claim ledger, evidence gate, run ledger, provenance graph, verifier/judge separation, or paper-to-code reproduction should be considered for deep dive. If a project introduces a new property, record it under `facets.custom_facets` before changing the taxonomy.
