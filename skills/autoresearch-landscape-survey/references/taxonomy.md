# Multi-Facet Taxonomy for AutoResearch Projects

This survey no longer treats the ecosystem as four orthogonal layers. Most projects are hybrids. For example, ARIS is simultaneously a Markdown skill harness, an idea-to-experiment workflow, a review loop, and a partial claim-audit system. AutoResearchClaw is an idea-to-paper pipeline, a skill matcher, a sandbox runner, a HITL system, and a partial anti-fabrication system.

The correct unit is therefore a **facet**, not a layer.

## Required Facet Axes

The canonical schema lives in `data/taxonomy.yaml`. Current axes are:

- `research_fields`: AI/ML, biomedical discovery, wetlab automation, materials/physics, math/formal science, social science reproducibility, software engineering, scientific infrastructure, metascience/integrity, literature research.
- `starting_points`: topic, question, paper PDF, draft paper, existing repo, dataset/benchmark, experiment plan, disease/candidate, workflow task.
- `primary_outputs`: literature report, hypothesis, experiment plan, code repo, training run, paper draft, review/rebuttal, verified claims, reproduction bundle, dashboard, benchmark scores.
- `workflow_scopes`: literature-to-report, idea-to-paper, paper-to-code, paper-to-reproduction, repo-to-experiment, experiment-to-claim, claim-to-review-issue, lab discovery loop, runtime, tracking infrastructure.
- `execution_depth`: text-only, code generation, smoke test, full training/simulation, fresh-container reproduction, wetlab human-in-loop, benchmark grading, tracking only.
- `verification_models`: citation grounding, claim-evidence mapping, deterministic verifier, LLM reviewer, cross-model review, adversarial reviewer, human gate, fresh-container reproduction, artifact hashing, attestation protocol, rubric grading.
- `accountability_features`: claim ledger, method spec, experiment spec, run ledger, artifact lineage, provenance graph, agent trace, decision log, review issue loop, release bundle, failure registry.
- `agent_topologies`: single agent, orchestrator-workers, specialist agents, multi-agent debate, producer/verifier split, judge panel, inspector agent, dynamic roles, human-AI team.
- `integration_styles`: Markdown skills, Python package, CLI, web dashboard, MCP/tools, GitHub-native, notebook/script, external platform, container runtime, MLOps tracking stack.
- `maturity_signals`: GitHub popularity, peer-reviewed paper, benchmark/leaderboard, reproducible examples, active development, external evaluation, early prototype, gated dependency, security sensitivity.
- `risk_flags`: LLM code execution, unrestricted network, API key requirement, unverified claim generation, data/license/privacy, expensive compute, supply-chain risk, common-mode model failure.
- `fit_to_our_target`: core competitor, adjacent competitor, plugin layer, runtime component, benchmark reference, infrastructure component, risk model, domain inspiration, watch only.

## Open Vocabulary Rule

If a new project does not fit the current schema, do not discard the attribute. Add it to:

```yaml
facets:
  custom_facets:
    - axis: verification_models
      value: causal_trace_counterfactual_judge
      rationale: "The project verifies claims by generating counterfactual causal traces."
      source: "README section ..."
```

Then mention it in the AI analysis report and decide whether to promote it into `data/taxonomy.yaml`.

## Legacy Fields

The registry still keeps `domain_categories`, `mechanism_tags`, and `center_object` for backward compatibility. New reports should prefer `facets`.


## v0.3 Added Axes

The v0.3 taxonomy adds axes that are specific to CairnLab's Research Claim Kernel strategy:

- `kernel_primitives`
- `claim_state_semantics`
- `lifecycle_stages`
- `governance_controls`
- `risk_tiers`
- `governance_alignment`

These axes exist to detect whether a project merely tracks evidence or actually enforces claim lifecycle transitions.
