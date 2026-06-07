#!/usr/bin/env python3
"""Check whether a new AutoResearch project has already been surveyed.

The similarity model is intentionally transparent and simple. It compares URLs,
aliases, legacy domain/mechanism tags, and the newer multi-facet attributes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from e

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "data" / "project_registry.yaml"
DEFAULT_TAXONOMY = ROOT / "data" / "taxonomy.yaml"
FACET_AXES = [
    "research_fields", "starting_points", "primary_outputs", "workflow_scopes", "execution_depth",
    "verification_models", "accountability_features", "agent_topologies", "integration_styles",
    "maturity_signals", "risk_flags", "fit_to_our_target",
    "kernel_primitives", "claim_state_semantics", "lifecycle_stages", "governance_controls",
    "risk_tiers", "governance_alignment",
]


def load_structured(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def urls(item: Dict[str, Any]) -> set[str]:
    vals = set(item.get("urls") or item.get("sources") or [])
    if item.get("repo_url"):
        vals.add(item["repo_url"])
    if item.get("paper_url"):
        vals.add(item["paper_url"])
    return {u.rstrip("/") for u in vals if isinstance(u, str)}


def names(item: Dict[str, Any]) -> set[str]:
    vals = {item.get("id", ""), item.get("name", "")}
    vals.update(item.get("aliases") or [])
    return {normalize(v) for v in vals if v}


def get_facets(item: Dict[str, Any]) -> Dict[str, List[str]]:
    facets = dict(item.get("facets") or {})
    # Accept intake files using observed_* keys.
    for axis in FACET_AXES:
        if axis not in facets:
            observed_key = f"observed_{axis}"
            if observed_key in item:
                facets[axis] = item.get(observed_key) or []
    # Backward compatibility.
    if "workflow_scopes" not in facets and item.get("observed_domain_categories"):
        facets["workflow_scopes"] = item.get("observed_domain_categories")
    return {k: list(v or []) for k, v in facets.items() if isinstance(v, list)}


def exact_duplicate_reasons(new: Dict[str, Any], existing: Dict[str, Any]) -> List[str]:
    reasons = []
    url_overlap = urls(new) & urls(existing)
    if url_overlap:
        reasons.append("url_overlap:" + ",".join(sorted(url_overlap)))
    name_overlap = names(new) & names(existing)
    if name_overlap:
        reasons.append("alias_or_name_overlap:" + ",".join(sorted(name_overlap)))
    return reasons


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_score(new: Dict[str, Any], existing: Dict[str, Any]) -> float:
    n_facets = get_facets(new)
    e_facets = get_facets(existing)

    # Legacy fields still matter for old intake files.
    legacy_domain = jaccard(set(new.get("observed_domain_categories") or new.get("domain_categories") or []), set(existing.get("domain_categories") or []))
    legacy_mech = jaccard(set(new.get("observed_mechanism_tags") or new.get("mechanism_tags") or []), set(existing.get("mechanism_tags") or []))
    center = 1.0 if new.get("center_object") and new.get("center_object") == existing.get("center_object") else 0.0

    facet_scores = []
    weights = {
        "research_fields": 1.0,
        "starting_points": 0.7,
        "primary_outputs": 0.7,
        "workflow_scopes": 1.3,
        "execution_depth": 1.0,
        "verification_models": 1.4,
        "accountability_features": 1.5,
        "agent_topologies": 1.0,
        "integration_styles": 0.7,
        "maturity_signals": 0.3,
        "risk_flags": 0.3,
        "fit_to_our_target": 0.7,
    }
    total_w = 0.0
    weighted = 0.0
    for axis, w in weights.items():
        s = jaccard(set(n_facets.get(axis, [])), set(e_facets.get(axis, [])))
        if n_facets.get(axis) or e_facets.get(axis):
            weighted += w * s
            total_w += w
    facet_score = weighted / total_w if total_w else 0.0

    return round(0.15 * legacy_domain + 0.15 * legacy_mech + 0.10 * center + 0.60 * facet_score, 4)


def novelty_against_top(new: Dict[str, Any], closest: List[Tuple[float, Dict[str, Any]]]) -> Dict[str, Any]:
    n_facets = get_facets(new)
    known = {axis: set() for axis in FACET_AXES}
    for _, p in closest[:5]:
        pf = get_facets(p)
        for axis in FACET_AXES:
            known[axis].update(pf.get(axis, []))
    novelty = {}
    shared = {}
    for axis in FACET_AXES:
        vals = set(n_facets.get(axis, []))
        if vals:
            novelty[axis] = sorted(vals - known[axis])
            shared[axis] = sorted(vals & known[axis])
    custom = (new.get("facets") or {}).get("custom_facets") or []
    if custom:
        novelty["custom_facets"] = custom
    return {"novel_facets_vs_top5": novelty, "shared_facets_with_top5": shared}


def recommend_depth(new: Dict[str, Any], exact: List[Dict[str, Any]], closest: List[Tuple[float, Dict[str, Any]]]) -> str:
    if exact:
        return "update_existing"
    facets = get_facets(new)
    high_value = {
        "claim_ledger", "method_spec", "experiment_spec", "run_ledger", "artifact_lineage", "provenance_graph",
        "agent_trace", "decision_log", "review_issue_loop", "release_bundle", "failure_registry",
    }
    wf = set(facets.get("workflow_scopes", []))
    acc = set(facets.get("accountability_features", []))
    ver = set(facets.get("verification_models", []))
    if len(acc & high_value) >= 2 or {"paper_to_reproduction", "experiment_to_claim", "claim_to_review_issue"} & wf:
        return "deep_or_medium"
    if {"claim_evidence_mapping", "fresh_container_reproduction", "attestation_protocol"} & ver:
        return "deep_or_medium"
    if closest and closest[0][0] > 0.45:
        return "brief_or_medium"
    return "brief"


def analyze(new: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    projects = registry.get("projects", [])
    exact = []
    exact_reasons = {}
    for p in projects:
        reasons = exact_duplicate_reasons(new, p)
        if reasons:
            exact.append(p)
            exact_reasons[p["id"]] = reasons

    scored = sorted(((overlap_score(new, p), p) for p in projects), key=lambda x: x[0], reverse=True)
    closest = [(s, p) for s, p in scored if s > 0][:10]
    return {
        "input": new,
        "duplicate_status": "already_surveyed" if exact else "not_exact_duplicate",
        "exact_matches": [
            {"id": p["id"], "name": p["name"], "reasons": exact_reasons[p["id"]], "detailed_analysis": p.get("detailed_analysis")}
            for p in exact
        ],
        "closest_projects": [
            {
                "score": s,
                "id": p["id"],
                "name": p["name"],
                "depth": p.get("depth"),
                "summary": p.get("summary"),
                "shared_legacy_domains": sorted(set(new.get("observed_domain_categories") or new.get("domain_categories") or []) & set(p.get("domain_categories") or [])),
                "shared_legacy_mechanisms": sorted(set(new.get("observed_mechanism_tags") or new.get("mechanism_tags") or []) & set(p.get("mechanism_tags") or [])),
                "shared_facets": {
                    axis: sorted(set(get_facets(new).get(axis, [])) & set(get_facets(p).get(axis, [])))
                    for axis in FACET_AXES
                    if set(get_facets(new).get(axis, [])) & set(get_facets(p).get(axis, []))
                },
                "gaps_vs_our_target": p.get("gaps_vs_our_target", []),
                "detailed_analysis": p.get("detailed_analysis"),
            }
            for s, p in closest[:8]
        ],
        "novelty": novelty_against_top(new, closest),
        "recommended_depth": recommend_depth(new, exact, closest),
    }


def to_markdown(result: Dict[str, Any]) -> str:
    new = result["input"]
    lines = [f"# New Project Comparison: {new.get('name', new.get('id'))}", ""]
    lines += ["## Duplicate Check", ""]
    lines.append(f"Status: **{result['duplicate_status']}**")
    if result["exact_matches"]:
        for m in result["exact_matches"]:
            lines.append(f"- `{m['id']}` / {m['name']} — reasons: {', '.join(m['reasons'])}")
            if m.get("detailed_analysis"):
                lines.append(f"  - Detailed analysis: `{m['detailed_analysis']}`")
    else:
        lines.append("- No exact URL or alias match found.")

    lines += ["", "## Closest Existing Projects", ""]
    for p in result["closest_projects"]:
        lines.append(f"- **{p['name']}** (`{p['id']}`), score={p['score']}, depth={p.get('depth')}")
        if p.get("shared_facets"):
            for axis, vals in p["shared_facets"].items():
                lines.append(f"  - shared {axis}: {', '.join(vals)}")
        lines.append(f"  - summary: {p.get('summary','')}")
        if p.get("detailed_analysis"):
            lines.append(f"  - detailed analysis: `{p['detailed_analysis']}`")

    lines += ["", "## Observed Facets", ""]
    facets = get_facets(new)
    for axis in FACET_AXES:
        vals = facets.get(axis, [])
        if vals:
            lines.append(f"- {axis}: {', '.join(vals)}")
    if (new.get("facets") or {}).get("custom_facets"):
        lines.append(f"- custom_facets: {json.dumps((new.get('facets') or {}).get('custom_facets'), ensure_ascii=False)}")

    lines += ["", "## What Looks New", ""]
    novelty = result["novelty"]
    for scope, axes in novelty.items():
        lines.append(f"### {scope}")
        for axis, vals in axes.items():
            if vals:
                lines.append(f"- {axis}: {vals if isinstance(vals, list) else vals}")

    lines += ["", "## Recommendation", ""]
    lines.append(f"- recommended depth: **{result['recommended_depth']}**")
    if result["duplicate_status"] == "already_surveyed":
        lines.append("- Action: update the existing project card/deep dive instead of adding a new project.")
    else:
        lines.append("- Action: add a new registry entry if sources are verified; ask an AI analyst to read the repo and confirm facets before finalizing.")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="YAML/JSON intake file for the new project")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", help="Optional output path")
    args = parser.parse_args(argv)

    new = load_structured(Path(args.input))
    registry = load_structured(Path(args.registry))
    result = analyze(new, registry)
    out = json.dumps(result, ensure_ascii=False, indent=2) if args.format == "json" else to_markdown(result)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out, encoding="utf-8")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
