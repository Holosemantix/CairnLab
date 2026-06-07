from __future__ import annotations

from pathlib import Path

from cairnlab import ArisManifestAdapter, CairnRuntime


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "aris_manifest"


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
        "human_gate:aris_release",
    }
    assert any(relation.source == "metric:exp_001.accuracy" and relation.target == "claim:C1" for relation in case.relations)


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
