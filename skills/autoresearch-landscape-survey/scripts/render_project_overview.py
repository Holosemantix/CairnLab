#!/usr/bin/env python3
"""Render the continuously updated project overview Markdown."""
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
FACET_AXES = [
    "research_fields", "starting_points", "primary_outputs", "workflow_scopes", "execution_depth",
    "verification_models", "accountability_features", "agent_topologies", "integration_styles",
    "maturity_signals", "risk_flags", "fit_to_our_target",
    "kernel_primitives", "claim_state_semantics", "lifecycle_stages", "governance_controls",
    "risk_tiers", "governance_alignment",
]


def load(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def vals(p: Dict[str, Any], axis: str) -> List[str]:
    return list(((p.get("facets") or {}).get(axis)) or [])


def render(reg: Dict[str, Any]) -> str:
    projects = reg.get("projects", [])
    by_fit = defaultdict(list)
    by_field = defaultdict(list)
    by_workflow = defaultdict(list)
    by_accountability = defaultdict(list)
    custom = []
    for p in projects:
        for f in vals(p, "fit_to_our_target") or ["unclassified"]:
            by_fit[f].append(p)
        for f in vals(p, "research_fields") or ["unclassified"]:
            by_field[f].append(p)
        for f in vals(p, "workflow_scopes") or ["unclassified"]:
            by_workflow[f].append(p)
        for f in vals(p, "accountability_features"):
            by_accountability[f].append(p)
        for cf in vals(p, "custom_facets"):
            custom.append((p, cf))

    lines = []
    lines.append("# AutoResearch Project Overview")
    lines.append("")
    lines.append(f"Last updated: {reg.get('last_updated')}  ")
    lines.append(f"Registry version: {reg.get('version')}  ")
    lines.append(f"Projects: {len(projects)}")
    lines.append("")
    lines.append("## How to Read This Overview")
    lines.append("")
    lines.append("Projects are described with non-orthogonal facets. A project can be an idea-to-paper system, a skill harness, a sandbox runner, and a partial claim-audit system at the same time. Do not read any section as a mutually exclusive category.")
    lines.append("")
    lines.append("## High-Level Matrix")
    lines.append("")
    lines.append("| Project | Fit | Fields | Workflow | Verification | Accountability | Agent Topology | Deep Dive |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in sorted(projects, key=lambda x: (-x.get("our_relevance", 0), x.get("id", ""))):
        lines.append("| {name} | {fit} | {fields} | {workflow} | {verif} | {acc} | {agents} | {deep} |".format(
            name=p.get("name"),
            fit=", ".join(vals(p, "fit_to_our_target")[:3]),
            fields=", ".join(vals(p, "research_fields")[:3]),
            workflow=", ".join(vals(p, "workflow_scopes")[:4]),
            verif=", ".join(vals(p, "verification_models")[:4]),
            acc=", ".join(vals(p, "accountability_features")[:4]),
            agents=", ".join(vals(p, "agent_topologies")[:3]),
            deep=p.get("detailed_analysis", ""),
        ))
    lines.append("")

    def section(title: str, mapping):
        lines.append(f"## {title}")
        lines.append("")
        for key in sorted(mapping):
            ps = sorted(mapping[key], key=lambda x: (-x.get("our_relevance", 0), x.get("id", "")))
            lines.append(f"### `{key}`")
            lines.append("")
            for p in ps:
                lines.append(f"- **{p.get('name')}** (`{p.get('id')}`): {p.get('summary','')}")
            lines.append("")

    section("By Fit to Our Target", by_fit)
    section("By Research Field", by_field)
    section("By Workflow Scope", by_workflow)
    section("By Accountability Feature", by_accountability)

    lines.append("## Open / Proposed New Facets")
    lines.append("")
    if not custom:
        lines.append("No custom facets currently recorded. If a new project brings a new attribute, add it under `facets.custom_facets` and regenerate this overview.")
    else:
        for p, cf in custom:
            lines.append(f"- **{p.get('name')}**: `{cf}`")
    lines.append("")
    lines.append("## Maintenance Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("python skills/autoresearch-landscape-survey/scripts/check_new_project.py --input path/to/intake.yaml --format markdown")
    lines.append("python skills/autoresearch-landscape-survey/scripts/classify_project_from_yaml.py --input path/to/intake.yaml --format yaml")
    lines.append("python skills/autoresearch-landscape-survey/scripts/prepare_ai_project_review.py --input path/to/intake.yaml")
    lines.append("python skills/autoresearch-landscape-survey/scripts/render_project_overview.py")
    lines.append("python skills/autoresearch-landscape-survey/scripts/prepare_ai_report_prompt.py")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output", default=str(ROOT / "reports" / "project_overview.md"))
    args = parser.parse_args()
    text = render(load(Path(args.registry)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
