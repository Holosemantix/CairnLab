from __future__ import annotations

from pathlib import Path

from cairnlab import Actor, CairnProject, CairnRuntime, ExternalRunManifestAdapter
from cairnlab.models import ClaimState


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "external_run_manifest"


def test_external_run_manifest_adapter_detects_fixture() -> None:
    adapter = ExternalRunManifestAdapter()

    assert adapter.detect(FIXTURE)
    assert adapter.detect(FIXTURE / "cairn_external_run.yaml")


def test_external_run_manifest_adapter_exports_cross_stage_case() -> None:
    result = ExternalRunManifestAdapter().export_case(FIXTURE)
    case = result.case
    evidence_ids = {item.id for item in case.evidence}
    relation_pairs = {(relation.source, relation.target, str(relation.type)) for relation in case.relations}

    assert case.source_system == "external-paper2code-stack"
    assert case.native_system_behavior["external_stage"] == "paper_to_code"
    assert [claim.id for claim in case.claims] == ["claim:C1"]
    assert {
        "run:idea_generation_001",
        "run:paper_to_code_001",
        "artifact:paper_to_code_patch",
        "metric:repro.accuracy",
        "verifier:metric_threshold_accuracy",
        "reviewer:external_scope_review",
        "dissent:split_scope",
        "human_gate:external_release_owner",
    } <= evidence_ids
    reviewer = next(item for item in case.evidence if item.id == "reviewer:external_scope_review")
    dissent = next(item for item in case.evidence if item.id == "dissent:split_scope")
    verifier = next(item for item in case.evidence if item.id == "verifier:metric_threshold_accuracy")
    assert reviewer.metadata["not_transition_authority"] is True
    assert dissent.type == "material_dissent"
    assert dissent.metadata["resolved"] is False
    assert verifier.metadata["verdict"] == "pass"
    assert ("metric:repro.accuracy", "claim:C1", "supports") in relation_pairs
    assert "external_material_dissent" in case.failure_classes
    assert "external_review_split_scope_warning" in case.failure_classes
    assert not result.diagnostics


def test_external_run_manifest_blocks_release_on_unresolved_dissent(tmp_path: Path) -> None:
    result = ExternalRunManifestAdapter().export_case(FIXTURE)
    project = CairnProject.open(tmp_path)
    project.import_claim_case(result.case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="human:pi", role="principal_investigator", authority="release_owner"),
        reason="release external paper-to-code claim",
    )

    assert decision.decision == "blocked"
    assert decision.blocking_reasons == ["unresolved_material_dissent"]


def test_external_run_manifest_feeds_runtime_revert_plan() -> None:
    result = ExternalRunManifestAdapter().export_case(FIXTURE)
    runtime = CairnRuntime.from_case(result.case)

    plan = runtime.plan_revert("run:paper_to_code_001", reason="external code project invalidated")
    affected = {item.id: item.action.value for item in plan.affected}

    assert affected["artifact:paper_to_code_patch"] == "invalidate"
    assert affected["metric:repro.accuracy"] == "invalidate"
    assert affected["claim:C1"] == "challenge"
    assert affected["verifier:metric_threshold_accuracy"] == "invalidate"
    assert affected["human_gate:external_release_owner"] == "require_reapproval"
