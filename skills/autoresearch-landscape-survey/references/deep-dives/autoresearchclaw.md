# Project Deep Dive: AutoResearchClaw

## One-line positioning

AutoResearchClaw 是目前最强的开源 idea-to-paper 工程化竞品之一：从一个 topic 出发，走 23-stage pipeline，生成论文、代码、实验、图表、reviews 和 verification report。

## Why it matters

它说明“自动科研流水线”已经不是空白。它有：

- 23-stage pipeline；
- stage/context runtime skill matcher；
- experiment execution 和 sandbox；
- self-healing repair；
- PIVOT / REFINE loop；
- 6 种 HITL 模式；
- multi-agent peer review；
- VerifiedRegistry / anti-fabrication / citation verification；
- 多领域 execution agents 与 ARC-Bench。

## Mechanism breakdown

| 机制 | 评价 |
| --- | --- |
| stage pipeline | 很强，23 stage 的 I/O contract 适合做系统级参照。 |
| runtime skill matcher | 与 ARIS 的手动/orchestrator skill 不同，AutoResearchClaw 根据 stage/context 自动注入 skill。 |
| HITL | 很强，full-auto、gate-only、checkpoint、step-by-step、co-pilot、custom 等模式值得借鉴。 |
| self-healing | 对失败实验进行 diagnosis & repair，是实验执行层关键机制。 |
| claim verification | 主要防 AI 生成文本里的 fabricated numbers/ungrounded citations。 |
| provenance | 有 artifact versioning 和 structured metrics，但不是我们要的强 claim/run/review ledger。 |

## What it already solves

- 一条 research topic 到论文的端到端体验。
- 多 agent peer review 和 paper-evidence consistency check。
- 实验失败后的修复和 pivot/refine。
- 通过 skills 扩展领域知识。

## What it does not solve

- 外部论文逐 claim 复现审计不是主入口。
- `paper claim -> experiment spec -> code commit -> data hash -> run -> artifact -> review issue` 不是强类型核心对象。
- review feedback 仍偏质量优化，不是 issue/patch/rerun/closure 的工程闭环。

## Reusable design patterns

- 23-stage pipeline 可用于构建我们的对照分类。
- 每个 stage 有输入/输出/失败/重试/gate 约束。
- skill 可以按 stage/context 自动匹配。
- HITL 不应是 afterthought，而应是一等设计。

## Questions to verify by running code

1. 新论文/repo 输入时，是否能从外部 paper 抽 claim 和 experiment？
2. 实验 metric 是否有 artifact hash 和 run metadata？
3. verification_report 能否追到具体 run 和 source？
4. peer review 的每条问题是否能触发 patch/rerun？
5. stage 失败时是否完整记录原因和修改 diff？

## Registry note

保留 deep dive。它是 idea-to-paper 竞品，不是我们 Research CI 的直接替代。
