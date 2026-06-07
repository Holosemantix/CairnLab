# Deep-Dive Protocol

深度项目分析必须服务于未来架构设计，而不是简单复述 README。

## 模板结构

```markdown
# Project Deep Dive: <name>

## One-line positioning
## Why it matters to our map
## Mechanism breakdown
## Architecture sketch
## What it already solves
## What it does not solve
## Differences vs AutoResearchClaw / ARIS / Robin / PaperBench / EviBound
## Reusable design patterns
## Risks and failure modes
## Questions to verify by running the code
## Suggested registry update
```

## 必须回答的问题

1. 它是否有可运行代码？如何安装？是否依赖私有 API？
2. 它是否能处理外部论文，还是只处理自己的 topic/idea？
3. 它的 claim 是自然语言还是结构化对象？
4. 它的 run/artifact 是日志还是可查询 ledger？
5. reviewer 是文本生成器还是 evidence-bound judge？
6. 多 agent 是否分权？执行者、验证者、审判者是否隔离？
7. 失败是否归因并沉淀为可审计记忆？
8. 和我们目标相比，最值得借鉴的机制是什么？

## 重要原则

不要把项目自身宣传视为事实。对 claims、benchmarks、运行能力、star 数、release 状态都要保留 `last_reviewed` 并在后续调研时刷新。
