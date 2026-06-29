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


def test_external_run_manifest_hashes_directory_evidence_and_repo_relative_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = project / "research" / "multitask_transfer"
    tables = package / "tables"
    eval_results = package / "eval_results"
    tables.mkdir(parents=True)
    eval_results.mkdir()
    (tables / "eval_summary.csv").write_text("metric,value\nsuccess,1.0\n", encoding="utf-8")
    (eval_results / "b_metrics.txt").write_text("seed=43 success=1\n", encoding="utf-8")
    (eval_results / "a_metrics.txt").write_text("seed=42 success=1\n", encoding="utf-8")
    manifest = package / "cairn_external_run.yaml"
    manifest.write_text(
        """
manifest_type: cairn.external_run.v1
case_id: external-run:directory-evidence
source_system: external-runner
stage: result_analysis
stages:
  - id: run:summarize
    phase: result_analysis
    tool: summarize.py
    status: completed
    path: research/multitask_transfer/tables/eval_summary.csv
claims:
  - id: claim:C1
    text: "External metrics support the claim."
    state: evidence_attached
evidence:
  - id: evidence:eval_summary_csv
    type: table
    path: research/multitask_transfer/tables/eval_summary.csv
  - id: evidence:eval_metric_directory
    type: metric_set
    path: eval_results
relations:
  - source: evidence:eval_summary_csv
    target: claim:C1
    type: supports
    criticality: material
""".lstrip(),
        encoding="utf-8",
    )

    result = ExternalRunManifestAdapter().export_case(manifest)
    second = ExternalRunManifestAdapter().export_case(manifest)
    evidence = {item.id: item for item in result.case.evidence}
    second_evidence = {item.id: item for item in second.case.evidence}
    relation = next(item for item in result.case.relations if item.source == "evidence:eval_summary_csv")

    assert not result.diagnostics
    assert relation.criticality.value == "critical"
    assert evidence["run:summarize"].uri == "research/multitask_transfer/tables/eval_summary.csv"
    assert evidence["run:summarize"].hash.startswith("sha256:")
    assert evidence["evidence:eval_summary_csv"].uri == "research/multitask_transfer/tables/eval_summary.csv"
    assert evidence["evidence:eval_summary_csv"].hash.startswith("sha256:")
    assert evidence["evidence:eval_metric_directory"].uri == "eval_results"
    assert evidence["evidence:eval_metric_directory"].hash.startswith("sha256-tree:")
    assert evidence["evidence:eval_metric_directory"].hash == second_evidence["evidence:eval_metric_directory"].hash


def test_external_run_manifest_resolves_prefixed_relative_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    manifest_dir = project / "manifests" / "paper1"
    artifact_root = project / "external" / "wm_exp"
    artifact_root.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    (artifact_root / "result.json").write_text('{"ok": true}\n', encoding="utf-8")
    manifest = manifest_dir / "cairn_external_run.yaml"
    manifest.write_text(
        """
manifest_type: cairn.external_run.v1
case_id: external-run:prefixed-paths
source_system: external-runner
stage: release_review
path_prefixes:
  wm_exp_repo: ../../external/wm_exp
claims:
  - id: claim:C1
    text: "External result exists."
    state: evidence_attached
evidence:
  - id: artifact:result
    type: artifact
    root: wm_exp_repo
    path: result.json
relations:
  - source: artifact:result
    target: claim:C1
    type: supports
    criticality: critical
""".lstrip(),
        encoding="utf-8",
    )

    result = ExternalRunManifestAdapter().export_case(manifest)
    evidence = {item.id: item for item in result.case.evidence}

    assert not result.diagnostics
    assert evidence["artifact:result"].uri == "root:wm_exp_repo:result.json"
    assert evidence["artifact:result"].hash.startswith("sha256:")


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
    assert decision.blocking_reasons == [
        "release_requires_verified_state",
        "unresolved_material_dissent",
    ]


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
