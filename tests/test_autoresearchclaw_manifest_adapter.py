from __future__ import annotations

from pathlib import Path

from cairnlab import AutoResearchClawManifestAdapter, CairnRuntime


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "autoresearchclaw_manifest"
E2E_FIXTURE = ROOT / "tests" / "fixtures" / "autoresearchclaw_e2e_run"


def test_autoresearchclaw_manifest_adapter_detects_fixture() -> None:
    adapter = AutoResearchClawManifestAdapter()
    assert adapter.detect(FIXTURE)
    assert not adapter.detect(E2E_FIXTURE)
    assert adapter.detect(E2E_FIXTURE / "stage-14")


def test_autoresearchclaw_manifest_adapter_exports_claim_case() -> None:
    adapter = AutoResearchClawManifestAdapter()
    result = adapter.export_case(FIXTURE)
    case = result.case

    assert case.case_id == "arc-manifest-wrong-metric"
    assert case.source_system == "autoresearchclaw"
    assert [claim.id for claim in case.claims] == ["claim:C1"]
    assert {item.id for item in case.evidence} >= {
        "run:exp_007",
        "metric:exp_007.accuracy",
        "artifact:metrics_json_exp_007",
        "human_gate:H1",
        "release_decision:R1",
    }
    assert not result.diagnostics


def test_autoresearchclaw_manifest_adapter_feeds_runtime_plan() -> None:
    adapter = AutoResearchClawManifestAdapter()
    result = adapter.export_case(FIXTURE)
    runtime = CairnRuntime.from_case(result.case)

    plan = runtime.plan_revert("run:exp_007", reason="metric computed on wrong split")
    affected = {item.id: item.action.value for item in plan.affected}

    assert affected["metric:exp_007.accuracy"] == "invalidate"
    assert affected["claim:C1"] == "downgrade"
    assert affected["paper_section:results.table_1"] == "mark_stale"
    assert affected["human_gate:H1"] == "require_reapproval"
    assert affected["release_decision:R1"] == "reopen_release_decision"
