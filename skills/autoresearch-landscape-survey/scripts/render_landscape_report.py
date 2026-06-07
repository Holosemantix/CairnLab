#!/usr/bin/env python3
"""Render a static landscape report from data/project_registry.yaml."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from e

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "data" / "project_registry.yaml"


def load(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def vals(p: Dict[str, Any], axis: str) -> List[str]:
    return list(((p.get("facets") or {}).get(axis)) or [])


def render(reg: Dict[str, Any]) -> str:
    projects = reg.get("projects", [])
    by_field = defaultdict(list)
    by_workflow = defaultdict(list)
    by_fit = defaultdict(list)
    for p in projects:
        for d in vals(p, "research_fields") or ["unclassified"]:
            by_field[d].append(p)
        for w in vals(p, "workflow_scopes") or ["unclassified"]:
            by_workflow[w].append(p)
        for f in vals(p, "fit_to_our_target") or ["unclassified"]:
            by_fit[f].append(p)

    lines = []
    lines.append("# AutoResearch Landscape Report")
    lines.append("")
    lines.append(f"Last updated: {reg.get('last_updated')}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report maps AutoResearch-related projects by non-orthogonal facets rather than by rigid layers. The central distinction for our work is not whether a project is an agent, a skill library, or a benchmark, but whether it can make research claims executable, verifiable, and accountable.")
    lines.append("")
    lines.append("Our target is CairnLab Research Claim Kernel: claim → lifecycle context → risk/responsibility → evidence policy → evidence item → verifier certificate → state transition → governance/human gate → decision trace package.")
    lines.append("")
    lines.append("## Deep-Dive Projects")
    lines.append("")
    lines.append("| Project | Fit | Workflow | Accountability | Relevance | Detailed Analysis |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")
    for p in sorted([p for p in projects if p.get("depth") == "deep"], key=lambda x: (-x.get("our_relevance",0), x["id"])):
        lines.append(f"| {p['name']} | {', '.join(vals(p, 'fit_to_our_target')[:3])} | {', '.join(vals(p, 'workflow_scopes')[:4])} | {', '.join(vals(p, 'accountability_features')[:4])} | {p.get('our_relevance','')} | {p.get('detailed_analysis','')} |")
    lines.append("")
    lines.append("## Why Facets Instead of Layers")
    lines.append("")
    lines.append("The same project can cover several concerns: ARIS is a Markdown skill harness, a workflow orchestrator, an experiment bridge, and a partial result-to-claim system. AutoResearchClaw is an idea-to-paper pipeline, a skill matcher, a sandbox runner, a review loop, and a partial anti-fabrication system. Therefore the registry records multiple facets for each project.")
    lines.append("")

    def list_section(title, mapping):
        lines.append(f"## {title}")
        lines.append("")
        for key in sorted(mapping):
            ps = sorted(mapping[key], key=lambda x: (-x.get("our_relevance",0), x["id"]))
            lines.append(f"### `{key}`")
            lines.append("")
            for p in ps[:20]:
                lines.append(f"- **{p['name']}** (`{p['id']}`): {p.get('summary','')}")
            lines.append("")

    list_section("Field-Based Map", by_field)
    list_section("Workflow / Process Map", by_workflow)
    list_section("Fit to Our Target", by_fit)

    lines.append("## Projects Most Similar to CairnLab Research Claim Kernel")
    lines.append("")
    ranked = sorted(projects, key=lambda p: (-p.get("our_relevance",0), p["id"]))[:15]
    lines.append("| Project | Starting Points | Outputs | Key Gap |")
    lines.append("| --- | --- | --- | --- |")
    for p in ranked:
        gap = (p.get("gaps_vs_our_target") or [""])[0]
        lines.append(f"| {p['name']} | {', '.join(vals(p, 'starting_points')[:3])} | {', '.join(vals(p, 'primary_outputs')[:4])} | {gap} |")
    lines.append("")
    lines.append("## Maintenance Protocol")
    lines.append("")
    lines.append("Run `check_new_project.py` and `prepare_ai_project_review.py` before adding a new item. Projects with claim ledger, evidence gate, run ledger, provenance graph, verifier/judge separation, or paper-to-code reproduction should be considered for deep dive. If a project introduces a new property, record it under `facets.custom_facets` before changing the taxonomy.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output", default=str(ROOT / "reports" / "landscape_report.md"))
    args = parser.parse_args()
    text = render(load(Path(args.registry)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
