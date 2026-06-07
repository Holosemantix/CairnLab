from __future__ import annotations

from pathlib import Path

from cairnlab import Actor, CairnProject, ClaimCaseBuilder
from cairnlab.models import ClaimState


def test_verified_transition_requires_passing_verifier_certificate(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        ClaimCaseBuilder(
            case_id="authority-missing-verifier",
            source_system="unit-test",
            stress_scenario="missing_verifier_certificate",
        )
        .add_claim("claim:C1", "A claim with evidence but no verifier.", state="evidence_attached")
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1")
        .add_support("metric:m1", "claim:C1")
        .build()
    )
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="user:alice", role="maintainer"),
        reason="verify claim",
    )

    assert decision.decision == "blocked"
    assert decision.blocking_reasons == ["missing_passing_verifier_certificate"]


def test_high_impact_release_requires_governance_package(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        ClaimCaseBuilder(
            case_id="authority-high-impact-missing-governance",
            source_system="unit-test",
            stress_scenario="missing_release_governance",
        )
        .add_claim("claim:C1", "A high-impact verified claim.", state="verified", risk="high")
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1")
        .add_support("metric:m1", "claim:C1")
        .add_evidence(
            "verifier:V1",
            "verifier_certificate",
            uri="memory://verifiers/V1",
            hash="sha256:v1",
            metadata={"verdict": "pass"},
        )
        .add_relation("verifier:V1", "claim:C1", "verified_by")
        .add_human_gate(
            "human_gate:H1",
            "claim:C1",
            "human:alice",
            authority="project_owner",
            scope={"claim": "claim:C1"},
            rationale="Approved after reviewing structured evidence.",
        )
        .build()
    )
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="user:alice", role="maintainer"),
        reason="release claim",
    )

    assert decision.decision == "blocked"
    assert decision.blocking_reasons == [
        "missing_risk_assessment",
        "missing_accountable_party",
        "missing_decision_trace_package",
    ]


def test_release_rechecks_artifact_and_verifier_even_if_imported_verified(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        ClaimCaseBuilder(
            case_id="authority-release-rechecks-evidence",
            source_system="unit-test",
            stress_scenario="imported_verified_without_artifact",
        )
        .add_claim("claim:C1", "An imported verified claim without artifacts.", state="verified")
        .add_human_gate(
            "human_gate:H1",
            "claim:C1",
            "human:alice",
            authority="project_owner",
            scope={"claim": "claim:C1"},
            rationale="Approved after reviewing structured evidence.",
        )
        .add_risk_assessment("claim:C1", risk_tier="medium")
        .add_responsibility_assignment("claim:C1", accountable=[("human_pi", "human:pi@example.org")])
        .build()
    )
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="user:alice", role="maintainer"),
        reason="release claim",
    )

    assert decision.decision == "blocked"
    assert decision.blocking_reasons == [
        "missing_machine_addressable_evidence",
        "missing_passing_verifier_certificate",
    ]


def test_release_allowed_when_required_governance_is_present(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = _release_ready_case("authority-release-ready", risk="high").build()
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="user:alice", role="maintainer"),
        reason="release claim",
    )

    assert decision.decision == "allowed"
    assert decision.blocking_reasons == []
    assert decision.events[0].new_state == "released"


def test_material_dissent_blocks_release_without_explicit_override(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        _release_ready_case("authority-dissent", risk="medium")
        .add_evidence(
            "dissent:D1",
            "material_dissent",
            uri="cairn://dissent/D1",
            metadata={"severity": "material", "resolved": False},
        )
        .add_relation("dissent:D1", "claim:C1", "challenges", criticality="critical")
        .build()
    )
    project.import_claim_case(case)

    blocked = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="user:alice", role="maintainer"),
        reason="release claim",
    )
    overridden = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="human:pi", role="principal_investigator", authority="release_override"),
        reason="explicitly override material dissent",
        force=True,
    )

    assert blocked.decision == "blocked"
    assert blocked.blocking_reasons == ["unresolved_material_dissent"]
    assert overridden.decision == "allowed"
    assert overridden.events[0].payload["force"] is True


def _release_ready_case(case_id: str, risk: str = "medium") -> ClaimCaseBuilder:
    builder = (
        ClaimCaseBuilder(
            case_id=case_id,
            source_system="unit-test",
            stress_scenario="release_authority",
        )
        .add_claim("claim:C1", "A verified claim ready for release.", state="verified", risk=risk)
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1")
        .add_support("metric:m1", "claim:C1")
        .add_evidence(
            "verifier:V1",
            "verifier_certificate",
            uri="memory://verifiers/V1",
            hash="sha256:v1",
            metadata={"verdict": "pass"},
        )
        .add_relation("verifier:V1", "claim:C1", "verified_by")
        .add_human_gate(
            "human_gate:H1",
            "claim:C1",
            "human:alice",
            authority="project_owner",
            scope={"claim": "claim:C1"},
            rationale="Approved after reviewing structured evidence.",
        )
        .add_release_decision("release_decision:R1", "claim:C1", "human:alice")
        .add_risk_assessment("claim:C1", risk_tier=risk)
        .add_responsibility_assignment(
            "claim:C1",
            accountable=[("human_pi", "human:pi@example.org")],
            responsible=[("verifier", "system:verifier")],
        )
    )
    if risk in {"high", "critical"}:
        builder.add_decision_trace_package(
            "dtp:D1",
            "claim:C1",
            includes=["claim", "evidence", "risk_assessment", "responsibility_assignment", "human_gate"],
        )
    return builder
