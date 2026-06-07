#!/usr/bin/env python3
"""Prepare an AI prompt for writing/updating the landscape report."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from e

ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data" / "project_registry.yaml"
DEFAULT_TAXONOMY = ROOT / "data" / "taxonomy.yaml"
DEFAULT_PROMPT = SKILL_DIR / "prompts" / "ai_report_synthesis_prompt.md"


def load(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)


def compact_project(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "depth": p.get("depth"),
        "summary": p.get("summary"),
        "facets": p.get("facets", {}),
        "our_relevance": p.get("our_relevance"),
        "differentiators": p.get("differentiators", []),
        "gaps_vs_our_target": p.get("gaps_vs_our_target", []),
        "detailed_analysis": p.get("detailed_analysis"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output", default=str(ROOT / "reports" / "ai_landscape_report_prompt.md"))
    parser.add_argument("--max-projects", type=int, default=80)
    args = parser.parse_args()

    reg = load(Path(args.registry))
    tax = load(Path(args.taxonomy))
    base = Path(args.prompt).read_text(encoding="utf-8")
    projects = sorted(reg.get("projects", []), key=lambda p: (-p.get("our_relevance", 0), p.get("id", "")))[:args.max_projects]
    payload = {
        "taxonomy_version": tax.get("version"),
        "registry_version": reg.get("version"),
        "project_count": len(reg.get("projects", [])),
        "projects": [compact_project(p) for p in projects],
    }
    text = (
        "# AI Landscape Report Writing Bundle\n\n"
        + base
        + "\n\n## Structured Registry Snapshot\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
