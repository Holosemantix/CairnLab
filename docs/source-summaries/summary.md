# Source Summaries

## Attached deep research report

用户附件的核心结论：autoresearch 已经明显超过“只写综述/demo”的阶段，但仍未成熟到通用、稳定、可追责；skill 更适合作为局部专家层，而不是完整科研操作系统。推荐方向是以 Claim Ledger、Experiment Spec、Sandbox Runner、Artifact/Provenance Registry、Review/HITL Gate 为核心的研究操作系统。

## AutoResearchClaw skill report

AutoResearchClaw 的 skill 系统面向 23-stage autonomous research pipeline；每个 skill 用 SKILL.md 表达触发、分类、适用 stage、优先级和反模式；运行时按 stage/topic/context 自动匹配。

## ARIS skill report

ARIS 是 Markdown-only skill harness，不是固定 stage runtime engine。它依赖 orchestrator skill 串联 `/idea-discovery -> /experiment-bridge -> /auto-review-loop -> /paper-writing`，并用 research-wiki、traces 和跨模型 verifier 做质量门。

## ML-AI skill authoring report

高质量 skill 不是文档堆砌，而是 `SKILL.md` 做入口，`references/` 拆长决策，`scripts/` 提供稳定 scaffold，`examples/` 和 `tests/` 给出成功标准。
