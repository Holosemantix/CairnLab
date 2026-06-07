# Deep Dive: Accountability Layer Projects

## One-line positioning

EviBound、ScientistOne、Pramana、PROV-AGENT、MARCH、AgentHallu、Catfish Agent 共同说明：下一代 AutoResearch 的关键不是更多 agent，而是 evidence-bound、attested、auditable、role-separated 的研究过程。

## Project notes

### EviBound

核心思想：claim 传播前必须经过 pre-execution approval gate 和 post-execution verification gate。Verification Gate 查询 MLflow run、artifact、metric、status。它是我们 `EvidenceGate` 的直接参考。

### ScientistOne / Chain-of-Evidence

核心思想：每个 claim 都追溯到 evidence source，并通过 CoE Audit 检查 score verification、specification violation、reference verification、method-code alignment。它是最接近我们 claim-level evidence 目标的研究方向之一。

### Pramana

核心思想：把 consequential agent output 包装为 typed ClaimAttestation，例如 measurement、inference、analogy、citation，并提供 verify()。它适合作为 agent 输出协议参考。

### PROV-AGENT

核心思想：扩展 W3C PROV 记录 prompts、responses、decisions、tool calls 和 downstream impact。它适合我们的 agent trace/provenance layer。

### MARCH

核心思想：Solver/Proposer/Checker 信息不对称。Checker 不读 generator 的完整叙事，只验证 atomic claims 和 evidence。它适合设计 Judge/Verifier 信息隔离。

### AgentHallu

核心思想：多步 agent hallucination 会沿 trajectory 传播，需要 step-level attribution。适合我们的错误归因 schema。

### Catfish Agent

核心思想：多 agent 容易 silent agreement，需要专门 Dissent/Catfish agent 注入结构化异议。

## Architecture implication for us

我们的多 agent 架构应该是：

```text
Producer agents
  -> structured claims / specs / code / runs
Independent verifiers
  -> method-code, metric, data, artifact, citation checks
Adversarial reviewers
  -> refutation, leakage, baseline unfairness, post-hoc selection
Evidence judge
  -> verdict based only on evidence objects
Provenance store
  -> prompt/tool/decision/run/artifact graph
```

## What remains open

这些项目大多是机制或论文，而不是完整开源 Research CI。我们的空间是把它们组合成可运行系统。
