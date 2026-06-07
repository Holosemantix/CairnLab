# Project Deep Dive: ARIS / Auto-Research-In-Sleep

## One-line positioning

ARIS 是 Markdown-only skill harness：不以固定 runtime engine 为核心，而以一组可组合 `SKILL.md` workflow 驱动 autonomous ML research。

## Why it matters

ARIS 是最接近“skill 组合成研究工作流”的项目之一。它的关键特征：

```text
/research-pipeline
  -> /idea-discovery
  -> /experiment-bridge
  -> /auto-review-loop
  -> /paper-writing
```

它还提供 Research Wiki，记录 Paper / Idea / Experiment / Claim 及关系：

```text
experiment --supports--> claim
experiment --invalidates--> claim
```

## Mechanism breakdown

| 机制 | 评价 |
| --- | --- |
| markdown_skill | 极轻量、可 fork、适配 Claude Code/Codex/OpenClaw 等。 |
| orchestrator_skill | 长流程由 `/research-pipeline` 等显式串联。 |
| experiment_bridge | 从 EXPERIMENT_PLAN 到代码、code review、sanity run、full deployment。 |
| auto_review_loop | review -> fix -> experiment -> re-review，接近 reviewer-to-action loop。 |
| research_wiki | 长期记忆和 claim graph，是我们需要重点借鉴的机制。 |
| external verifier | ARIS 强调 verifier exit code 才是 source of truth，执行者不能自判。 |

## What it already solves

- 把自动科研流程 skill 化。
- 用跨模型 reviewer 和外部 verifier 降低单模型幻觉。
- 将结果映射回 claim status。
- 用 persistent wiki 记录失败 idea 和实验经验。

## What it does not solve

- 它更偏“自己的研究推进/论文改进”，不是外部 paper reproduction CI。
- Markdown/wiki 很灵活，但强类型 ledger、数据库查询、release-grade provenance 仍不足。
- 对 commit/env/data/seed/hardware/artifact 的强制绑定不是核心。

## Reusable design patterns

- Research Wiki 的 Paper/Idea/Experiment/Claim schema。
- result-to-claim update。
- reviewer 与 executor 分权。
- assurance mode：submission 前外部 verifier 不通过则不能标记 ready。

## Questions to verify by running code

1. `research-wiki` 的 claim 状态是否机器可查询？
2. `/result-to-claim` 的 evidence 是否能追溯到 run artifact？
3. `/auto-review-loop` 的 reviewer finding 是否会生成结构化 task？
4. Codex/Claude/Gemini reviewer overlay 是否只改 reviewer backend，不漂移语义？

## Registry note

ARIS 是最重要对标之一。我们的差异应是：更强 schema、更强 provenance、更 paper-first。 
