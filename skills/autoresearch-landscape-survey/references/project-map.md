# Project Map

本页是 `data/project_registry.yaml` 的人类可读地图。v0.2 起，不再把生态强行拆成四个正交层；改用多维 facet 描述项目。

一个项目可以同时拥有多个属性，例如：

- AutoResearchClaw：idea-to-paper、skill runtime matcher、sandbox runner、HITL、multi-agent review、partial claim verification。
- ARIS：Markdown skill harness、workflow orchestrator、experiment bridge、research wiki、result-to-claim、paper claim audit。
- Robin：multi-agent scientific discovery、biomedicine、lab-in-the-loop、specialist agents、human wetlab execution。
- EviBound / ScientistOne：evidence gate、claim/evidence audit、accountability layer。

## 主要 deep-dive 项目

| 项目 | 关键 facets | 为什么重要 |
| --- | --- | --- |
| AutoResearchClaw | idea-to-paper, stage pipeline, skill matcher, sandbox, HITL, self-healing, partial claim verification | 最强 idea-to-paper 工程对标；但中心对象仍是 topic/idea，不是外部论文复现审计。 |
| ARIS | Markdown skills, orchestrator workflow, research wiki, experiment bridge, result-to-claim, cross-model review | 与我们的 review/claim 思路最接近；但更多依赖 Markdown/wiki 约定，强 typed ledger 仍不够。 |
| scientific-agent-skills | skill library, tool SOP, references/scripts/tests, validation scaffold | 局部专家层样板；告诉我们 skill 应如何降低 agent 猜 API/参数/接口。 |
| Robin | biomedical discovery, specialist agents, lab-in-the-loop, hypothesis/candidate loop | 真实多 agent 科学发现样板；说明人类实验与 AI 推理如何组合。 |
| PaperBench | paper-to-reproduction benchmark, fresh container, rubric grading | 外部论文复现的现实标尺；告诉我们 paper reproduction 远未成熟。 |
| EviBound / ScientistOne / Pramana / PROV-AGENT | claim ledger, evidence gate, attestation, provenance graph | 我们可追责层的核心参考。 |

## 关键观察

1. 生态里已经有强 idea-to-paper 项目，但 paper-first reproduction CI 仍未形成事实标准。
2. skill 系统适合作为局部专家层，但不等于研究操作系统。
3. 多 agent 不能天然提高可信度；需要 producer、verifier、adversary、judge、inspector、provenance store 分权。
4. 下一阶段的竞争点不是“谁写论文更流畅”，而是“谁的 claim 能被低成本审计、复跑和归责”。
5. 新项目如果带来新属性，不应被硬塞进旧分类；应先写入 `facets.custom_facets`，再决定是否提升为 taxonomy 正式值。

## 维护入口

- 总览：`reports/project_overview.md`
- 长报告：`reports/landscape_report.md`
- 项目注册表：`data/project_registry.yaml`
- 属性体系：`data/taxonomy.yaml`
- AI 新项目调研 prompt：`skills/autoresearch-landscape-survey/prompts/ai_new_project_analysis_prompt.md`
