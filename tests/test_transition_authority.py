from __future__ import annotations

from pathlib import Path

from cairnlab import Actor, CairnProject, ClaimCaseBuilder
from cairnlab.models import Claim, ClaimState, VerifierCertificate


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
        .add_claim(
            "claim:C1",
            "A high-impact verified claim.",
            state="verified",
            authority_state="verified",
            risk="high",
        )
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1")
        .add_support("metric:m1", "claim:C1")
        .add_verifier_certificate(_passing_certificate(inputs=["metric:m1"]))
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
        "release_requires_verified_state",
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


def test_transition_apply_appends_allowed_event_and_updates_authority_state(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    project.import_claim_case(_release_ready_case("authority-release-apply", risk="medium").build())

    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="human:alice", role="maintainer"),
        reason="release claim",
        apply=True,
    )
    trace = project.trace("claim:C1")

    assert decision.decision == "allowed"
    assert len(project.store.load_events()) == 1
    assert trace.authority_state == "released"
    assert trace.projected_state == "released"


def test_blocked_transition_apply_does_not_append_without_record_blocked(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    project.import_claim_case(
        ClaimCaseBuilder(
            case_id="authority-blocked-apply",
            source_system="unit-test",
            stress_scenario="blocked_transition_apply",
        )
        .add_claim("claim:C1", "A claim without evidence.")
        .build()
    )

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="human:alice", role="maintainer"),
        reason="try verify without evidence",
        apply=True,
    )

    assert decision.decision == "blocked"
    assert project.store.load_events() == []


def test_record_blocked_transition_appends_audit_event(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    project.import_claim_case(
        ClaimCaseBuilder(
            case_id="authority-record-blocked",
            source_system="unit-test",
            stress_scenario="record_blocked_transition",
        )
        .add_claim("claim:C1", "A claim without evidence.")
        .build()
    )

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="human:alice", role="maintainer"),
        reason="record blocked verify",
        record_blocked=True,
    )
    events = project.store.load_events()

    assert decision.decision == "blocked"
    assert len(events) == 1
    assert events[0].type == "TransitionBlocked"


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
    assert overridden.events[0].payload["override"]["authority"] == "release_override"


def test_force_override_requires_actor_authority(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        _release_ready_case("authority-dissent-no-override-authority", risk="medium")
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

    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="human:assistant", role="maintainer"),
        reason="attempt unauthorized force release",
        force=True,
    )

    assert decision.decision == "blocked"
    assert decision.blocking_reasons == ["missing_override_authority"]


def test_legacy_imported_state_is_observed_not_authority(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        ClaimCaseBuilder(
            case_id="authority-observed-state",
            source_system="unit-test",
            stress_scenario="observed_state_not_authority",
        )
        .add_claim("claim:C1", "An upstream released claim.", state="released")
        .build()
    )
    project.import_claim_case(case)

    trace = project.trace("claim:C1")
    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="human:alice", role="maintainer"),
        reason="release imported claim",
    )

    assert trace.observed_state == "released"
    assert trace.authority_state == "draft"
    assert trace.projected_state == "draft"
    assert "release_requires_verified_state" in decision.blocking_reasons


def test_legacy_status_alias_maps_to_observed_state_only() -> None:
    claim = Claim.model_validate(
        {
            "id": "claim:C1",
            "text": "A legacy status claim.",
            "type": "empirical_metric",
            "status": "released",
        }
    )

    assert claim.observed_state == "released"
    assert claim.authority_state == ClaimState.DRAFT


def test_metadata_verdict_alone_cannot_authorize_verified(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        ClaimCaseBuilder(
            case_id="authority-naked-verdict",
            source_system="unit-test",
            stress_scenario="malformed_verifier_certificate",
        )
        .add_claim("claim:C1", "A claim with a naked pass verdict.", state="evidence_attached")
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1")
        .add_support("metric:m1", "claim:C1")
        .add_evidence(
            "verifier:V1",
            "verifier_certificate",
            uri="memory://verifiers/V1",
            metadata={"verdict": "pass"},
        )
        .add_relation("verifier:V1", "claim:C1", "verified_by")
        .build()
    )
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="system:test", role="verifier"),
        reason="verify claim",
    )

    assert decision.decision == "blocked"
    assert decision.blocking_reasons == ["malformed_verifier_certificate"]


def test_wrong_claim_certificate_cannot_authorize_verified(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        ClaimCaseBuilder(
            case_id="authority-wrong-claim-certificate",
            source_system="unit-test",
            stress_scenario="claim_mismatch_verifier_certificate",
        )
        .add_claim("claim:C1", "A claim with someone else's certificate.", state="evidence_attached")
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1")
        .add_support("metric:m1", "claim:C1")
        .add_verifier_certificate(_passing_certificate(claim_id="claim:C2", inputs=["metric:m1"]))
        .add_relation("verifier:V1", "claim:C1", "verified_by")
        .build()
    )
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="system:test", role="verifier"),
        reason="verify claim",
    )

    assert decision.decision == "blocked"
    assert "verifier_certificate_claim_mismatch" in decision.blocking_reasons


def test_missing_certificate_input_cannot_authorize_verified(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        ClaimCaseBuilder(
            case_id="authority-missing-certificate-input",
            source_system="unit-test",
            stress_scenario="missing_verifier_certificate_input",
        )
        .add_claim("claim:C1", "A claim with a certificate missing inputs.", state="evidence_attached")
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1")
        .add_support("metric:m1", "claim:C1")
        .add_verifier_certificate(_passing_certificate(inputs=["metric:missing"]))
        .build()
    )
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="system:test", role="verifier"),
        reason="verify claim",
    )

    assert decision.decision == "blocked"
    assert decision.blocking_reasons == ["verifier_certificate_inputs_missing"]


def test_invalidated_certificate_input_cannot_authorize_verified(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    case = (
        ClaimCaseBuilder(
            case_id="authority-invalidated-certificate-input",
            source_system="unit-test",
            stress_scenario="invalidated_verifier_certificate_input",
        )
        .add_claim("claim:C1", "A claim with an invalidated certificate input.", state="evidence_attached")
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1", status="invalidated")
        .add_support("metric:m1", "claim:C1")
        .add_verifier_certificate(_passing_certificate(inputs=["metric:m1"]))
        .build()
    )
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="system:test", role="verifier"),
        reason="verify claim",
    )

    assert decision.decision == "blocked"
    assert "verifier_certificate_inputs_invalidated" in decision.blocking_reasons


def _release_ready_case(case_id: str, risk: str = "medium") -> ClaimCaseBuilder:
    builder = (
        ClaimCaseBuilder(
            case_id=case_id,
            source_system="unit-test",
            stress_scenario="release_authority",
        )
        .add_claim(
            "claim:C1",
            "A verified claim ready for release.",
            state="verified",
            authority_state="verified",
            risk=risk,
        )
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1")
        .add_support("metric:m1", "claim:C1")
        .add_verifier_certificate(_passing_certificate(inputs=["metric:m1"]))
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


def _passing_certificate(
    claim_id: str = "claim:C1",
    inputs: list[str] | None = None,
) -> VerifierCertificate:
    return VerifierCertificate(
        id="verifier:V1",
        verifier="unit-test@0.1.0",
        claim=claim_id,
        status="pass",
        inputs=inputs or [],
        result={"status": "ok"},
        can_authorize=["evidence_attached -> verified"],
    )
