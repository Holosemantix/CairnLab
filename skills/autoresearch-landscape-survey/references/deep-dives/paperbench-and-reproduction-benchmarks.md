# Deep Dive: PaperBench and Reproduction Benchmarks

## One-line positioning

PaperBench 是外部论文复现能力的关键 benchmark；它证明了 agent 从 paper 到可运行复现实验仍远未成熟。

## Why it matters

我们的目标不是生成漂亮 paper，而是把 paper claim 绑定到 experiment/run/artifact/review。PaperBench、AutoExperiment、ResearchCodeBench、MLE-bench、MLAgentBench 给出了现实标尺。

## Mechanisms

| 项目 | 机制 | 对我们的启发 |
| --- | --- | --- |
| PaperBench | author co-developed rubric、fresh container reproduction、细粒度评分 | 我们的 benchmark/CI 应该使用 fresh environment 和 rubric，不听 executor 自述。 |
| AutoExperiment | progressive code masking、interactive debugging | verifier + repair loop 重要。 |
| ResearchCodeBench | paper method coding challenges | method-to-code 是瓶颈。 |
| MLE-bench | Kaggle-style ML engineering | 衡量代码/训练/调参能力，但不是 paper-faithful。 |
| MLAgentBench | end-to-end ML experiment tasks | 可作为 runner/debugger 回归集。 |

## What they do not solve

- 多数是 benchmark，不是可持续调研/执行平台。
- 评测 run 不自动变成 reviewer issue / patch / rerun closure。
- 不提供完整 claim ledger 作为系统中心对象。

## Reusable design patterns

- Fresh container rerun。
- Rubric grading。
- 多层任务拆分。
- 失败归因：理解论文、写代码、环境、训练、metric、artifact。

## Registry note

PaperBench 应作为我们验证系统价值的核心对照，而不是架构终点。
