# Project Deep Dive: Robin / Future House

## One-line positioning

Robin 是多 agent 科学发现系统，面向生物医学 disease-driven discovery，整合文献搜索 agent 与数据分析 agent，提出假设、实验、解释结果并生成 follow-up hypothesis。

## Why it matters

Robin 不是 paper reproduction CI，但它说明“多专用 agent + lab-in-the-loop”已经能驱动真实科研发现。它对我们的多角色架构有重要启发。

## Mechanism breakdown

| 机制 | 评价 |
| --- | --- |
| specialized agents | Crow/Falcon/Finch 等专用 agent 编排，而非一个万能 agent。 |
| lab-in-the-loop | 人类执行湿实验，AI 负责 hypothesis、experimental direction、data analysis。 |
| data analysis loop | 实验结果进入下一轮假设生成。 |
| platform dependency | GitHub 运行依赖 Edison 平台 API，完整可复现受限。 |

## What it already solves

- 多 agent 端到端科学发现轨迹。
- 文献到候选再到实验解释的闭环。
- 人机分工现实：AI 做 intellectual loop，人做物理实验。

## What it does not solve

- 不以外部论文 claim reproduction 为目标。
- 没有通用 claim/run/artifact/review issue ledger。
- 不适合作为我们的直接架构底座，但适合多 agent role 分工参考。

## Reusable design patterns

- 专用 agent 角色比“万能研究员”更可靠。
- lab-in-the-loop 比盲目全自动更现实。
- 数据分析 agent 与文献 agent 应分权。
- 对我们来说还需要额外引入 Inspector、Verifier、Judge、Provenance Store。

## Questions to verify by running code

1. 没有 Edison API 时，哪些模块可运行？
2. Notebook 产物是否记录每步 agent decision？
3. 文献检索、候选筛选、数据分析之间的 evidence trail 如何保存？
4. 是否能导出可审计 run bundle？

## Registry note

Robin 是 multi-agent scientific discovery 的代表，不是 Research CI 竞品。
