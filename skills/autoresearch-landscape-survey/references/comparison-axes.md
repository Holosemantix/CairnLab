# Comparison Axes

Use these questions to compare projects without forcing them into a single layer.

## Field and Scope

- Which scientific or engineering domain does it target?
- Is the domain generic or specialized, such as AI algorithms, biomedicine, wetlab automation, materials science, math discovery, or social-science reproducibility?
- Does the project operate on ideas, papers, code repos, datasets, experiments, claims, reviews, or provenance traces?

## Process Nature

- Does it start from a topic, paper, repo, dataset, or review comment?
- Does it only produce text, or does it execute code/training/simulation/wetlab loops?
- Does it optimize a metric, reproduce an existing result, or generate a new paper?
- Does it have a fixed pipeline, a skill orchestrator, a dynamic multi-agent debate, or a CI-like run graph?

## Verification and Accountability

- What does the system require before it says a claim is supported?
- Are metrics linked to run IDs, artifact hashes, environment digests, seeds, and commits?
- Is there a deterministic verifier, a human gate, a judge agent, or only LLM self-review?
- Can review feedback become an issue, patch, rerun, and closure condition?
- Can a third party reproduce or audit the result from the release bundle?

## Multi-Agent Reliability

- Are producer, verifier, reviewer, and judge roles separated?
- Does the judge see executor narratives or only evidence objects?
- Is there adversarial dissent to avoid silent agreement?
- Are different models used to reduce common-mode hallucination?
- Is there step-level attribution when an error occurs?

## Maturity and Risk

- Is the repository active and runnable?
- Is there a paper, benchmark, CI, or external evaluation?
- Does it require closed APIs, large GPU budgets, secret tokens, or unrestricted network?
- Does it execute LLM-generated code or install untrusted dependencies?
- Are limitations and failure modes explicitly documented?


## Claim Kernel Axes v0.3

When evaluating a project against CairnLab, ask:

- Is `Claim` a first-class object or just generated text?
- Does the system define a claim state machine?
- Are state transitions machine-enforced?
- Does each claim have an evidence policy?
- Are verifier outputs certificates that authorize or block transitions?
- Is the transition log append-only and tamper-evident?
- Can material dissent block release?
- Does the system record responsibility assignment, not just actor IDs?
- Are risk tier and lifecycle stage used to change required controls?
- Can the system export a decision trace package for consequential claims?
- Can an external system use the mechanism as a kernel, or is it trapped inside one pipeline?
