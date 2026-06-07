# Project Deep Dive: scientific-agent-skills / ML-AI skill authoring

## One-line positioning

scientific-agent-skills 类型的 skill 仓库不是自动科研系统，而是专业工具/工作流的可复用 SOP、脚手架和验证门槛。

## Why it matters

用户附件与 ML/AI skill authoring 报告说明，高质量 skill 的核心结构是：

```text
skills/<skill-name>/
  ├── SKILL.md
  ├── references/
  ├── scripts/
  ├── assets/
  ├── examples/
  └── tests/
```

`SKILL.md` 是入口和路由；`references/` 承载长决策、边界条件和 troubleshooting；`scripts/` 承载易错流程的稳定模板；`tests/` 承载验证。

## Mechanism breakdown

| 机制 | 评价 |
| --- | --- |
| SKILL.md frontmatter | 让 agent 知道何时使用、权限、兼容性、边界。 |
| references | 防止把复杂 API/决策表塞进主文件。 |
| scripts | 用稳定 scaffold 替代脆弱长 prompt。 |
| examples/tests | 给 agent 成功标准和 smoke test。 |
| boundary notes | 明确何时不要使用该 skill。 |

## What it already solves

- 减少 agent 猜 API、版本、参数、数据格式。
- 把专家实践做成可复用局部知识单元。
- 给训练/评估/环境验证提供模板。

## What it does not solve

- 不负责 paper -> method -> experiment -> run -> review 的端到端语义。
- 没有 claim ledger、run ledger、review issue closure。
- 它是我们系统的 plugin/skill 层，而不是 orchestrator/provenance 层。

## Reusable design patterns for this repo

本调研仓库沿用其结构：

- `SKILL.md`：调研入口。
- `references/`：分类法、对比轴、deep-dive 协议。
- `scripts/`：新项目查重/比较脚本。
- `assets/templates/`：项目卡、deep dive、intake 模板。
- `examples/`：新项目输入示例。
- `tests/`：保证脚本可跑。

## Registry note

这是本仓库的结构样板，不是 AutoResearch 竞品。
