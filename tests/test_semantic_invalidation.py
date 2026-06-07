from __future__ import annotations

from pathlib import Path

from cairnlab.engine import CairnProject
from cairnlab.models import AffectedAction, Actor, EventType
from cairnlab.store import CairnProjectStore


ROOT = Path(__file__).resolve().parents[1]


def test_load_wrong_metric_case() -> None:
    store = CairnProjectStore(ROOT)
    case = store.load_case_file(ROOT / "examples" / "cases" / "case_wrong_metric.yaml")
    assert case.case_id == "case_wrong_metric"
    assert any(claim.id == "claim:C1" for claim in case.claims)
    assert any(item.id == "run:exp_007" for item in case.evidence)
    assert any(relation.source == "metric:exp_007.accuracy" and relation.target == "claim:C1" for relation in case.relations)


def test_wrong_metric_plan_only_does_not_write_events(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    project.init()
    project.import_case(ROOT / "examples" / "cases" / "case_wrong_metric.yaml")

    plan = project.plan_revert(
        "run:exp_007",
        reason="metric computed on wrong split",
        actor=Actor(id="user:alice", role="maintainer"),
    )

    affected = {item.id: item for item in plan.affected}
    assert affected["metric:exp_007.accuracy"].action == AffectedAction.INVALIDATE
    assert affected["claim:C1"].action == AffectedAction.DOWNGRADE
    assert affected["paper_section:results.table_1"].action == AffectedAction.MARK_STALE
    assert affected["human_gate:H1"].action == AffectedAction.REQUIRE_REAPPROVAL
    assert affected["release_decision:R1"].action == AffectedAction.REOPEN_RELEASE_DECISION
    assert project.store.load_events() == []


def test_apply_wrong_metric_projects_claim_state(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    project.init()
    project.import_case(ROOT / "examples" / "cases" / "case_wrong_metric.yaml")

    plan = project.plan_revert("run:exp_007", reason="metric computed on wrong split")
    events = project.apply_plan(plan)
    trace = project.trace("claim:C1")

    assert events[0].type == EventType.REVERT_REQUESTED
    assert trace.projected_state == "downgraded"
    assert any(event.type == EventType.CLAIM_DOWNGRADED for event in trace.events)


def test_human_scope_drift_reopens_release_decision(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    project.init()
    project.import_case(ROOT / "examples" / "cases" / "case_human_scope_drift.yaml")

    plan = project.plan_revert("human_gate:H2", reason="approval scope drift")
    affected = {item.id: item for item in plan.affected}

    assert affected["human_gate:H2"].action == AffectedAction.REQUIRE_REAPPROVAL
    assert affected["claim:C2"].action == AffectedAction.DOWNGRADE
    assert affected["paper_section:abstract.sentence_3"].action == AffectedAction.MARK_STALE
    assert affected["release_decision:R2"].action == AffectedAction.REOPEN_RELEASE_DECISION
