#!/usr/bin/env python3
"""Infer multi-facet attributes from a new project intake YAML.

This script is intentionally heuristic. It prepares a first-pass attribute proposal
for an AI colleague or human analyst; it does not replace repository reading.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from e

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY = ROOT / "data" / "taxonomy.yaml"

KEYWORD_RULES = {
    "research_fields": {
        "ai_ml_algorithm_research": ["machine learning", "ml", "neural", "training", "benchmark", "ablation", "model"],
        "ai_agent_systems": ["agent", "multi-agent", "claude code", "codex", "tool", "mcp", "workflow"],
        "biomedicine_drug_discovery": ["bio", "biomedical", "drug", "disease", "rna", "cell", "protein", "therapeutic"],
        "wetlab_lab_automation": ["wetlab", "wet lab", "robot", "laboratory", "assay", "phagocytosis"],
        "materials_chemistry_physics": ["material", "chemistry", "physics", "dft", "hep", "quantum", "collider"],
        "mathematical_scientific_discovery": ["math", "theorem", "formal", "proof", "symbolic"],
        "social_science_reproducibility": ["social science", "r package", "stata", "regression table", "replication package"],
        "software_engineering": ["repo", "pull request", "ci", "test", "codebase", "patch"],
        "scientific_infrastructure": ["mlflow", "dvc", "wandb", "snakemake", "nextflow", "reprozip", "ro-crate", "provenance"],
        "metascience_integrity_review": ["integrity", "review", "claim", "evidence", "audit", "hallucination", "provenance"],
        "general_literature_research": ["literature", "deep research", "survey", "rag", "citation", "paper qa"],
    },
    "starting_points": {
        "topic_or_idea": ["idea", "topic", "research direction"],
        "research_question": ["question", "query"],
        "paper_pdf_or_preprint": ["paper", "pdf", "preprint", "arxiv"],
        "draft_paper": ["draft", "latex", "submission"],
        "existing_code_repo": ["github", "repo", "codebase", "repository"],
        "dataset_or_benchmark": ["dataset", "benchmark", "leaderboard"],
        "experiment_plan": ["experiment plan", "protocol"],
        "disease_or_candidate": ["disease", "candidate", "compound", "drug", "target"],
    },
    "primary_outputs": {
        "literature_report": ["report", "survey", "synthesis"],
        "hypothesis_or_idea": ["hypothesis", "idea", "candidate"],
        "experiment_plan": ["experiment design", "experiment plan", "baseline", "ablation"],
        "code_repo_or_patch": ["code", "repo", "patch", "adapter", "implementation"],
        "training_run_results": ["run", "metrics", "checkpoint", "training", "result"],
        "paper_draft_or_tex": ["paper", "latex", "tex", "manuscript"],
        "review_or_rebuttal": ["review", "rebuttal", "critique"],
        "verified_claims": ["verified", "claim", "evidence", "verification report"],
        "reproduction_bundle": ["bundle", "capsule", "reproduce", "artifact"],
        "dashboard_or_lab_ui": ["dashboard", "ui", "lab"],
        "benchmark_scores": ["score", "rubric", "leaderboard"],
    },
    "workflow_scopes": {
        "literature_to_report": ["literature", "search", "survey", "synthesis"],
        "idea_to_paper": ["idea to paper", "from idea to paper", "paper generation"],
        "idea_to_experiment": ["hypothesis", "experiment design", "experiment run"],
        "paper_to_code": ["paper-to-code", "paper to code", "implement paper"],
        "paper_to_reproduction": ["reproduce paper", "replication", "reproduction"],
        "repo_to_experiment": ["repo", "codebase", "run experiment"],
        "experiment_to_claim": ["result-to-claim", "claim status", "claim evidence"],
        "claim_to_review_issue": ["review finding", "issue", "rerun", "closure"],
        "lab_discovery_loop": ["wetlab", "lab", "disease", "candidate"],
        "agent_runtime": ["runtime", "agent", "tool", "codex", "claude code"],
        "tracking_infrastructure": ["tracking", "provenance", "mlflow", "dvc", "wandb"],
    },
    "execution_depth": {
        "text_only": ["text-only", "report only"],
        "code_generation_only": ["code generation"],
        "smoke_test": ["smoke", "sanity", "test"],
        "full_training_or_simulation": ["full experiment", "training", "simulation", "sweep"],
        "fresh_container_reproduction": ["fresh container", "clean container", "docker reproduction"],
        "wetlab_human_in_loop": ["wetlab", "human experiment", "assay"],
        "benchmark_grading": ["benchmark", "rubric", "grading"],
        "tracking_only": ["tracking only", "experiment tracking"],
    },
    "verification_models": {
        "citation_grounding": ["citation", "reference", "literature grounding"],
        "claim_evidence_mapping": ["claim", "evidence", "chain-of-evidence"],
        "deterministic_verifier": ["verifier", "exit code", "schema", "pytest", "test"],
        "llm_reviewer": ["reviewer", "critic", "peer review"],
        "cross_model_review": ["cross-model", "claude reviewer", "gemini reviewer", "gpt reviewer"],
        "adversarial_reviewer": ["adversarial", "catfish", "refutation", "reviewer 2"],
        "human_gate": ["human", "hitl", "approval"],
        "fresh_container_reproduction": ["fresh container", "clean container"],
        "artifact_hashing": ["hash", "checksum", "artifact lineage"],
        "attestation_protocol": ["attestation", "typed claim"],
        "rubric_grading": ["rubric", "grading"],
    },
    "accountability_features": {
        "claim_ledger": ["claim ledger", "claim registry"],
        "method_spec": ["method spec"],
        "experiment_spec": ["experiment spec", "experiment plan"],
        "run_ledger": ["run ledger", "run id", "mlflow"],
        "artifact_lineage": ["artifact lineage", "artifact", "checksum", "hash"],
        "provenance_graph": ["provenance", "prov", "trace graph"],
        "agent_trace": ["trace", "trajectory", "prompt", "tool call"],
        "decision_log": ["decision log", "approval", "verdict"],
        "review_issue_loop": ["review finding", "issue", "patch", "rerun", "closure"],
        "release_bundle": ["release bundle", "repro pack", "capsule"],
        "failure_registry": ["failure registry", "self-healing", "error collector"],
    },
    "agent_topologies": {
        "single_agent": ["single agent"],
        "orchestrator_workers": ["orchestrator", "worker", "planner"],
        "specialist_agents": ["specialist", "crow", "falcon", "finch", "domain agent"],
        "multi_agent_debate": ["debate", "tournament", "multi-agent"],
        "producer_verifier_split": ["executor", "verifier", "producer", "checker"],
        "judge_panel": ["judge", "court", "verdict"],
        "inspector_agent": ["inspector"],
        "dynamic_roles": ["dynamic role", "role assignment"],
        "human_ai_team": ["human", "hitl", "pi", "lab"],
    },
    "integration_styles": {
        "markdown_skills": ["skill.md", "markdown skill", ".claude/skills"],
        "python_package": ["python", "pip", "pyproject", "package"],
        "cli": ["cli", "command line"],
        "web_dashboard": ["dashboard", "web ui"],
        "mcp_or_tool_protocol": ["mcp", "tool protocol"],
        "github_native": ["github", "pull request", "actions"],
        "notebook_or_script": ["notebook", "script"],
        "external_platform_dependency": ["api key", "platform", "edison"],
        "container_runtime": ["docker", "container", "sandbox"],
        "mlops_tracking_stack": ["mlflow", "dvc", "wandb"],
    },
    "maturity_signals": {
        "popular_github": ["stars", "forks", "popular"],
        "peer_reviewed_paper": ["nature", "paper", "arxiv", "published"],
        "benchmark_or_leaderboard": ["benchmark", "leaderboard"],
        "reproducible_examples": ["example", "tests", "demo"],
        "active_development": ["commits", "release", "active"],
        "external_evaluation": ["independent", "third-party", "external evaluation"],
        "early_research_prototype": ["prototype", "early"],
        "closed_or_gated_dependency": ["api key", "gated", "platform"],
        "security_sensitive": ["dangerously", "execute code", "secret"],
    },
    "risk_flags": {
        "llm_generated_code_execution": ["execute code", "generated code", "run code"],
        "unrestricted_network": ["network", "web access"],
        "secret_or_api_key_required": ["api key", "token", "secret", "edison"],
        "unverified_claim_generation": ["claim", "hallucination", "fabrication"],
        "data_license_or_privacy": ["license", "privacy", "pii"],
        "expensive_compute": ["gpu", "a100", "cluster", "expensive"],
        "supply_chain_risk": ["pip install", "dependencies", "remote code"],
        "common_mode_model_failure": ["same model", "common-mode"],
    },
}

CLAIM_KERNEL_EXTRA_RULES = {
    "kernel_primitives": {
        "claim_state_machine": ["claim state", "state machine", "lifecycle transition"],
        "evidence_policy_dsl": ["evidence policy", "acceptance criteria", "claim policy"],
        "verifier_certificate": ["certificate", "attestation", "verifier output"],
        "transition_authority": ["transition", "authorize", "block transition"],
        "append_only_transition_log": ["append-only", "tamper", "ledger", "event log"],
        "decision_trace_package": ["decision trace", "audit package", "trace package"],
        "human_liability_gate": ["liability", "human gate", "responsibility"],
    },
    "governance_controls": {
        "role_contract": ["role contract", "responsibility model", "raci"],
        "producer_verifier_isolation": ["producer verifier", "checker", "solver proposer checker"],
        "dissent_persistence": ["dissent", "minority", "catfish"],
        "material_dissent_block": ["blocking dissent", "material dissent"],
        "social_consensus_block": ["consensus cannot", "majority cannot"],
        "risk_tiered_controls": ["risk tier", "risk based", "high-risk"],
        "human_oversight_gate": ["human oversight", "human approval", "human gate"],
        "liability_scope": ["liability scope", "accountable"],
    },
    "lifecycle_stages": {
        "design": ["design"], "development": ["development"], "evaluation": ["evaluation", "test"],
        "deployment_use": ["deployment", "use"], "publication": ["publication", "release"],
        "monitoring": ["monitoring"], "downgrade_retraction": ["retraction", "downgrade"],
    },
    "governance_alignment": {
        "eu_ai_act_logging": ["eu ai act", "automatic recording", "logs"],
        "eu_ai_act_human_oversight": ["human oversight", "automation bias"],
        "nist_lifecycle_rmf": ["nist", "ai rmf"],
        "iso42001_aims": ["iso 42001", "ai management system"],
        "aicat_machine_readable_registry": ["aicat", "catalogue", "machine-readable"],
    },
}
KEYWORD_RULES.update(CLAIM_KERNEL_EXTRA_RULES)


def load(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def add_unique(out: Dict[str, List[str]], axis: str, value: str) -> None:
    out.setdefault(axis, [])
    if value not in out[axis]:
        out[axis].append(value)


def flatten_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return " ".join(flatten_text(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(flatten_text(v) for v in obj)
    return str(obj)


def infer_facets(data: Dict[str, Any], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    text = flatten_text(data).lower()
    out: Dict[str, List[str]] = {axis: [] for axis in KEYWORD_RULES}

    # Preserve explicit fields if already supplied.
    explicit = data.get("facets") or {}
    for axis in out:
        for v in explicit.get(axis, []) or []:
            add_unique(out, axis, v)

    legacy_domain_map = {
        "idea_to_paper_pipeline": ["ai_agent_systems", "ai_ml_algorithm_research"],
        "paper_to_code_reproduction": ["ai_ml_algorithm_research", "software_engineering"],
        "evidence_provenance_accountability": ["metascience_integrity_review", "ai_agent_systems"],
        "multi_agent_scientific_discovery": ["ai_agent_systems"],
        "lab_automation_wetlab": ["wetlab_lab_automation", "biomedicine_drug_discovery"],
        "execution_tracking_infrastructure": ["scientific_infrastructure"],
        "deep_research_literature_synthesis": ["general_literature_research"],
        "scientific_rag_review": ["general_literature_research", "metascience_integrity_review"],
        "skill_library": ["ai_agent_systems", "software_engineering"],
        "coding_agent_runtime": ["software_engineering", "ai_agent_systems"],
        "review_integrity_audit": ["metascience_integrity_review"],
    }
    for d in data.get("observed_domain_categories") or data.get("domain_categories") or []:
        for f in legacy_domain_map.get(d, []):
            add_unique(out, "research_fields", f)

    for axis, rules in KEYWORD_RULES.items():
        for value, keywords in rules.items():
            if any(k.lower() in text for k in keywords):
                add_unique(out, axis, value)

    # Unknown observed tags become custom facets.
    known_axes = set((taxonomy.get("facet_axes") or {}).keys())
    custom = list(explicit.get("custom_facets") or [])
    for k, v in data.items():
        if k.startswith("observed_") and k.replace("observed_", "") not in known_axes:
            custom.append({"axis": k, "value": v, "rationale": "Observed in intake but not in taxonomy."})

    result = {axis: sorted(vals) for axis, vals in out.items() if vals}
    if custom:
        result["custom_facets"] = custom
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--format", choices=["json", "yaml"], default="json")
    parser.add_argument("--output")
    args = parser.parse_args()

    data = load(Path(args.input))
    taxonomy = load(Path(args.taxonomy))
    facets = infer_facets(data, taxonomy)
    payload = {"id": data.get("id"), "name": data.get("name"), "suggested_facets": facets, "note": "Heuristic only. Ask an AI analyst to verify after reading the repository."}
    if args.format == "yaml":
        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120)
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
