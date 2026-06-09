from __future__ import annotations

import json
import shutil
from pathlib import Path

from cairnlab import ArisManifestAdapter, CairnRuntime


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "aris_manifest"
E2E_SMOKE_FIXTURE = ROOT / "tests" / "fixtures" / "aris_e2e_smoke"


def test_aris_manifest_adapter_detects_fixture() -> None:
    adapter = ArisManifestAdapter()
    assert adapter.detect(FIXTURE)


def test_aris_manifest_adapter_exports_claim_case() -> None:
    adapter = ArisManifestAdapter()
    result = adapter.export_case(FIXTURE)
    case = result.case

    assert case.source_system == "aris"
    assert [claim.id for claim in case.claims] == ["claim:C1"]
    assert {item.id for item in case.evidence} >= {
        "run:exp_001",
        "metric:exp_001.accuracy",
        "artifact:experiment_log",
        "paper_section:results.table_1",
        "verifier:experiment_audit",
        "verifier:paper_claim_audit",
        "verifier:aris.audit-verifier-report",
        "reviewer:docs.ARIS_INTRO",
        "human_gate:aris_release",
    }
    reviewer = next(item for item in case.evidence if item.id == "reviewer:docs.ARIS_INTRO")
    verifier = next(item for item in case.evidence if item.id == "verifier:aris.audit-verifier-report")
    assert reviewer.type == "reviewer_verdict"
    assert reviewer.metadata["not_transition_authority"] is True
    assert verifier.type == "verifier_certificate"
    assert case.expected_cairnlab_behavior["do_not_treat_llm_review_as_transition_authority"]
    assert any(relation.source == "metric:exp_001.accuracy" and relation.target == "claim:C1" for relation in case.relations)
    assert any("ARIS review sidecar reports WARN" in item.message for item in result.diagnostics)


def test_aris_manifest_adapter_detects_real_aris_review_sidecar_shape(tmp_path: Path) -> None:
    (tmp_path / "AGENT_GUIDE.md").write_text("# ARIS agent guide\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    shutil.copyfile(
        FIXTURE / "docs" / "ARIS_INTRO.review.json",
        tmp_path / "docs" / "ARIS_INTRO.review.json",
    )

    adapter = ArisManifestAdapter()
    result = adapter.export_case(tmp_path)
    case = result.case

    assert adapter.detect(tmp_path)
    assert case.claims == []
    assert [item.id for item in case.evidence] == ["reviewer:docs.ARIS_INTRO"]
    assert case.evidence[0].metadata["artifact_type"] == "aris_review_sidecar"
    assert case.evidence[0].metadata["verdicts"] == ["PASS", "WARN", "PASS"]


def test_aris_manifest_adapter_preserves_rejected_verifier_and_missing_human_gate(tmp_path: Path) -> None:
    claims_dir = tmp_path / "research-wiki" / "claims"
    claims_dir.mkdir(parents=True)
    (claims_dir / "C1.json").write_text(
        json.dumps(
            {
                "id": "claim:C1",
                "text": "The paper is submission-ready.",
                "status": "released",
            }
        ),
        encoding="utf-8",
    )
    aris_dir = tmp_path / ".aris"
    aris_dir.mkdir()
    (aris_dir / "audit-verifier-report.json").write_text(
        json.dumps(
            {
                "verifier": "verify_paper_audits.sh",
                "verdict": "BLOCKED",
                "exit_code": 1,
                "audits": [{"path": "PAPER_CLAIM_AUDIT.json", "verdict": "BLOCKED"}],
            }
        ),
        encoding="utf-8",
    )

    result = ArisManifestAdapter().export_case(tmp_path)
    case = result.case

    assert "aris_submission_verifier_rejected" in case.failure_classes
    assert any(item.id == "verifier:aris.audit-verifier-report" for item in case.evidence)
    assert any("ARIS submission verifier reports BLOCKED" in item.message for item in result.diagnostics)
    assert any("No .aris/human_gate.json" in item.message for item in result.diagnostics)


def test_aris_manifest_adapter_imports_e2e_smoke_contract() -> None:
    result = ArisManifestAdapter().export_case(E2E_SMOKE_FIXTURE)
    case = result.case
    evidence_ids = {item.id for item in case.evidence}
    relation_pairs = {(relation.source, relation.target, str(relation.type)) for relation in case.relations}

    assert [claim.id for claim in case.claims] == ["claim:C1"]
    assert "run:exp_001" in evidence_ids
    assert "metric:exp_001.accuracy" in evidence_ids
    assert "reviewer:docs.SMOKE" in evidence_ids
    assert "verifier:paper.PROOF_AUDIT" in evidence_ids
    assert "verifier:paper.PAPER_CLAIM_AUDIT" in evidence_ids
    assert "verifier:paper..aris.audit-verifier-report" in evidence_ids
    assert "human_gate:aris_smoke_release" in evidence_ids
    assert ("run:exp_001", "claim:C1", "supports") in relation_pairs
    assert not case.failure_classes
    assert case.expected_cairnlab_behavior["require_transition_authority_for_release"]


def test_aris_e2e_smoke_contract_feeds_runtime_plan() -> None:
    result = ArisManifestAdapter().export_case(E2E_SMOKE_FIXTURE)
    runtime = CairnRuntime.from_case(result.case)

    plan = runtime.plan_revert("run:exp_001", reason="ARIS smoke result invalidated")
    affected = {item.id: item.action.value for item in plan.affected}

    assert affected["metric:exp_001.accuracy"] == "invalidate"
    assert affected["claim:C1"] == "challenge"
    assert affected["verifier:paper.PAPER_CLAIM_AUDIT"] == "invalidate"
    assert affected["verifier:paper..aris.audit-verifier-report"] == "invalidate"
    assert affected["human_gate:aris_smoke_release"] == "require_reapproval"


def test_aris_manifest_adapter_feeds_runtime_plan() -> None:
    adapter = ArisManifestAdapter()
    result = adapter.export_case(FIXTURE)
    runtime = CairnRuntime.from_case(result.case)

    plan = runtime.plan_revert("run:exp_001", reason="audit found wrong split")
    affected = {item.id: item.action.value for item in plan.affected}

    assert affected["metric:exp_001.accuracy"] == "invalidate"
    assert affected["claim:C1"] == "downgrade"
    assert affected["paper_section:results.table_1"] == "mark_stale"
    assert affected["verifier:experiment_audit"] == "invalidate"
    assert affected["verifier:paper_claim_audit"] == "invalidate"
    assert affected["human_gate:aris_release"] == "require_reapproval"
