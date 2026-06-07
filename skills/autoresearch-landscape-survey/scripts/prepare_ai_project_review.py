#!/usr/bin/env python3
"""Prepare an AI colleague prompt bundle for deep-reading a new project repo."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from e

import importlib.util

ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data" / "project_registry.yaml"
DEFAULT_TAXONOMY = ROOT / "data" / "taxonomy.yaml"
DEFAULT_PROMPT = SKILL_DIR / "prompts" / "ai_new_project_analysis_prompt.md"
CHECK_SCRIPT = SKILL_DIR / "scripts" / "check_new_project.py"
CLASSIFY_SCRIPT = SKILL_DIR / "scripts" / "classify_project_from_yaml.py"


def load(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_bundle(project: Dict[str, Any], taxonomy: Dict[str, Any], duplicate_result: Dict[str, Any], inferred: Dict[str, Any], base_prompt: str) -> str:
    repo_urls = project.get("repo_url") or project.get("urls") or project.get("sources") or []
    if isinstance(repo_urls, str):
        repo_urls = [repo_urls]
    lines = []
    lines.append("# AI Project Review Prompt Bundle")
    lines.append("")
    lines.append("## Base Analyst Prompt")
    lines.append("")
    lines.append(base_prompt)
    lines.append("")
    lines.append("## Project Intake")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(project, allow_unicode=True, sort_keys=False, width=120).strip())
    lines.append("```")
    lines.append("")
    lines.append("## First-Pass Heuristic Facets")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(inferred, allow_unicode=True, sort_keys=False, width=120).strip())
    lines.append("```")
    lines.append("")
    lines.append("## Duplicate / Similarity Precheck")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(duplicate_result, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Suggested Repository Reading Plan")
    lines.append("")
    lines.append("For each accessible repository URL, inspect these paths if present:")
    lines.append("")
    for path in [
        "README.md / README_CN.md",
        "paper links, arXiv, Nature, blog, project docs",
        "docs/",
        "skills/ and .claude/skills/",
        "prompts/ and agents/",
        "pipeline/orchestrator source files",
        "schemas, contracts, artifacts, sample runs",
        "examples/ and tests/",
        "pyproject.toml, setup.py, requirements, Dockerfile",
        "security notes, API key requirements, network policy",
        "issues/releases if needed for maturity signals",
    ]:
        lines.append(f"- {path}")
    lines.append("")
    if repo_urls:
        lines.append("Repository/source URLs to start from:")
        for u in repo_urls:
            lines.append(f"- {u}")
        lines.append("")
    lines.append("## Taxonomy Reminder")
    lines.append("")
    axes = taxonomy.get("facet_axes", {})
    for axis, obj in axes.items():
        vals = list((obj.get("values") or {}).keys())
        lines.append(f"- `{axis}`: {', '.join(vals[:25])}")
    lines.append("")
    lines.append("## Final Instruction")
    lines.append("")
    lines.append("After reading the repo, produce the structured Markdown + YAML report requested in the base prompt. Explicitly mark any new attributes as `custom_facets` and say whether they should become taxonomy entries.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    project = load(Path(args.input))
    registry = load(Path(args.registry))
    taxonomy = load(Path(args.taxonomy))
    check = load_module(CHECK_SCRIPT, "check_new_project")
    classifier = load_module(CLASSIFY_SCRIPT, "classify_project_from_yaml")
    duplicate_result = check.analyze(project, registry)
    inferred = classifier.infer_facets(project, taxonomy)
    base_prompt = Path(args.prompt).read_text(encoding="utf-8")
    text = render_bundle(project, taxonomy, duplicate_result, inferred, base_prompt)
    out = Path(args.output) if args.output else ROOT / "reports" / "intake" / f"{project.get('id','new_project')}_ai_review_prompt.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
