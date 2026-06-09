#!/usr/bin/env python3
"""Run a minimal ARIS artifact-chain smoke test against CairnLab.

The script intentionally uses ARIS deterministic helpers as external commands
instead of importing ARIS. It creates a tiny project with research-wiki pages,
an evidence pre-check, paper audit artifacts, an ARIS verifier report, and a
human gate, then imports the project through CairnLab's adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cairnlab import ArisManifestAdapter, CairnRuntime  # noqa: E402


def main() -> int:
    args = parse_args()
    aris_repo = resolve_aris_repo(args.aris_repo)
    workdir = prepare_workdir(args.workdir)

    project = workdir / "aris_smoke_project"
    project.mkdir(parents=True, exist_ok=True)

    tools = aris_repo / "tools"
    research_wiki = tools / "research_wiki.py"
    evidence_check = tools / "evidence_check.py"
    verifier = tools / "verify_paper_audits.sh"
    require_file(research_wiki)
    require_file(evidence_check)
    require_file(verifier)

    run([sys.executable, str(research_wiki), "init", str(project / "research-wiki")])
    create_claim_and_experiment(project)
    run(
        [
            sys.executable,
            str(research_wiki),
            "add_edge",
            str(project / "research-wiki"),
            "--from",
            "exp:exp_001",
            "--to",
            "claim:C1",
            "--type",
            "supports",
            "--evidence",
            "accuracy 0.91 from paper/results/metrics.json",
        ]
    )
    create_paper_and_claims(project)
    run_evidence_precheck(project, evidence_check)
    create_audit_chain(project)
    run(
        [
            "bash",
            str(verifier),
            str(project / "paper"),
            "--assurance",
            "submission",
            "--json-out",
            str(project / "paper" / ".aris" / "audit-verifier-report.json"),
        ]
    )
    create_review_sidecar(project)
    create_human_gate(project)

    export = ArisManifestAdapter().export_case(project)
    runtime = CairnRuntime.from_case(export.case)
    plan = runtime.plan_revert("run:exp_001", reason="ARIS smoke result invalidated")
    affected = {item.id: item.action.value for item in plan.affected}
    summary = {
        "project": str(project),
        "claims": len(export.case.claims),
        "evidence": len(export.case.evidence),
        "relations": len(export.case.relations),
        "diagnostics": [item.model_dump(mode="json") for item in export.diagnostics],
        "failure_classes": export.case.failure_classes,
        "affected": affected,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    required = {
        "claim:C1": "challenge",
        "metric:exp_001.accuracy": "invalidate",
        "verifier:paper..aris.audit-verifier-report": "invalidate",
        "human_gate:aris_smoke_release": "require_reapproval",
    }
    missing = {key: value for key, value in required.items() if affected.get(key) != value}
    if missing:
        print(f"missing expected revert effects: {missing}", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aris-repo", help="Path to Auto-claude-code-research-in-sleep")
    parser.add_argument("--workdir", help="Empty directory to receive the smoke project")
    return parser.parse_args()


def resolve_aris_repo(value: str | None) -> Path:
    candidates = [
        Path(value) if value else None,
        Path(os.environ["ARIS_REPO"]) if os.environ.get("ARIS_REPO") else None,
        ROOT.parent / "Auto-claude-code-research-in-sleep",
    ]
    for candidate in candidates:
        if candidate and (candidate / "tools" / "research_wiki.py").exists():
            return candidate.resolve()
    raise SystemExit("ARIS repo not found. Pass --aris-repo or set ARIS_REPO.")


def prepare_workdir(value: str | None) -> Path:
    if not value:
        return Path(tempfile.mkdtemp(prefix="cairnlab-aris-smoke-"))
    workdir = Path(value)
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(f"--workdir must be empty: {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir.resolve()


def create_claim_and_experiment(project: Path) -> None:
    claims_dir = project / "research-wiki" / "claims"
    experiments_dir = project / "research-wiki" / "experiments"
    claims_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    (claims_dir / "C1.md").write_text(
        """---
type: claim
node_id: claim:C1
claim: "ARIS smoke run reports accuracy = 0.91 on the fixture metric file."
status: supported
claim_type: empirical_metric
risk: medium
scope: {"metric": "accuracy", "dataset": "fixture"}
---

# Claim C1

ARIS smoke run reports accuracy = 0.91 on the fixture metric file.
""",
        encoding="utf-8",
    )
    (experiments_dir / "exp_001.md").write_text(
        """---
type: experiment
node_id: exp:exp_001
status: completed
verdict: supports
metrics: {"accuracy": 0.91}
---

# Experiment exp_001

The smoke experiment writes paper/results/metrics.json and validates that the cited accuracy value exists before verifier review.
""",
        encoding="utf-8",
    )


def create_paper_and_claims(project: Path) -> None:
    paper = project / "paper"
    (paper / "results").mkdir(parents=True, exist_ok=True)
    (paper / "main.tex").write_text(
        "The ARIS smoke run reports accuracy = 0.91 on the fixture metric file.\n",
        encoding="utf-8",
    )
    write_json(paper / "results" / "metrics.json", {"accuracy": 0.91})
    (project / ".aris").mkdir(exist_ok=True)
    write_json(
        project / ".aris" / "claims.json",
        [{"id": "claim:C1", "value": "0.91", "source": "paper/results/metrics.json"}],
    )


def run_evidence_precheck(project: Path, evidence_check: Path) -> None:
    result = run(
        [
            sys.executable,
            str(evidence_check),
            str(project),
            "--batch",
            str(project / ".aris" / "claims.json"),
        ],
        capture=True,
    )
    (project / ".aris" / "evidence_precheck.json").write_text(result.stdout, encoding="utf-8")


def create_audit_chain(project: Path) -> None:
    paper = project / "paper"
    for skill in ("proof-checker", "paper-claim-audit", "citation-audit", "kill-argument"):
        trace_dir = paper / ".aris" / "traces" / skill / "2026-06-09_run01"
        trace_dir.mkdir(parents=True, exist_ok=True)
        write_json(trace_dir / "trace.json", {"skill": skill, "status": "smoke"})

    write_json(
        paper / "PROOF_AUDIT.json",
        audit_payload(
            "proof-checker",
            "NOT_APPLICABLE",
            "no_theorems",
            "No formal theorem claims in the smoke paper.",
            {paper / "main.tex": "main.tex"},
        ),
    )
    write_json(
        paper / "PAPER_CLAIM_AUDIT.json",
        audit_payload(
            "paper-claim-audit",
            "PASS",
            "numbers_match",
            "The smoke paper's accuracy claim matches paper/results/metrics.json.",
            {paper / "main.tex": "main.tex", paper / "results" / "metrics.json": "results/metrics.json"},
        ),
    )
    write_json(
        paper / "CITATION_AUDIT.json",
        audit_payload(
            "citation-audit",
            "NOT_APPLICABLE",
            "no_citations",
            "No bibliography claims in the smoke paper.",
            {paper / "main.tex": "main.tex"},
        ),
    )
    write_json(
        paper / "KILL_ARGUMENT.json",
        audit_payload(
            "kill-argument",
            "NOT_APPLICABLE",
            "smoke_not_theory_or_scope_paper",
            "No adversarial theory or scope argument is required for this smoke paper.",
            {paper / "main.tex": "main.tex"},
        ),
    )


def audit_payload(
    skill: str,
    verdict: str,
    reason_code: str,
    summary: str,
    inputs: dict[Path, str],
) -> dict[str, Any]:
    return {
        "audit_skill": skill,
        "verdict": verdict,
        "reason_code": reason_code,
        "summary": summary,
        "audited_input_hashes": {rel: f"sha256:{sha256(path)}" for path, rel in inputs.items()},
        "trace_path": f".aris/traces/{skill}/2026-06-09_run01/",
        "thread_id": f"deterministic-smoke-{skill}",
        "reviewer_model": "deterministic-smoke",
        "reviewer_reasoning": "none",
        "generated_at": "2026-06-09T00:00:00Z",
    }


def create_review_sidecar(project: Path) -> None:
    docs = project / "docs"
    docs.mkdir(exist_ok=True)
    write_json(
        docs / "SMOKE.review.json",
        {
            "skill": "render-html",
            "source": "docs/SMOKE.md",
            "output": "docs/SMOKE.html",
            "reviewer": "deterministic-smoke",
            "verdict": "PASS",
            "checks": {"source_hash_match": "pass", "information_fidelity": "pass"},
            "blocking_issues": [],
            "warnings": [],
            "rendered_at": "2026-06-09",
        },
    )


def create_human_gate(project: Path) -> None:
    write_json(
        project / ".aris" / "human_gate.json",
        {
            "id": "human_gate:aris_smoke_release",
            "claim": "claim:C1",
            "actor": "human:smoke-reviewer@example.org",
            "authority": "validation_owner",
            "scope": {"claim": "claim:C1", "verifier": "verifier:paper..aris.audit-verifier-report"},
            "rationale": "Smoke run records that a human gate is separate from ARIS verifier output.",
        },
    )


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"required file not found: {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
