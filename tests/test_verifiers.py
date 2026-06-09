from __future__ import annotations

from pathlib import Path

from cairnlab import (
    Actor,
    ArtifactHashVerifier,
    CairnProject,
    ClaimCaseBuilder,
    MetricThresholdVerifier,
    VerificationRequest,
)
from cairnlab.models import ClaimState, EvidenceItem


def test_artifact_hash_certificate_can_authorize_verified_transition(tmp_path: Path) -> None:
    artifact = EvidenceItem(
        id="artifact:metrics",
        type="artifact",
        uri="memory://artifacts/metrics.json",
        hash="sha256:metrics",
    )
    certificate = ArtifactHashVerifier().verify(
        VerificationRequest(
            claim_id="claim:C1",
            evidence=[artifact],
            parameters={"expected_hash": "sha256:metrics"},
        )
    )
    case = (
        ClaimCaseBuilder(
            case_id="verifier-artifact-hash-pass",
            source_system="unit-test",
            stress_scenario="artifact_hash_pass",
        )
        .add_claim("claim:C1", "Artifact hash matches expected digest.", state="evidence_attached")
        .add_evidence(artifact.id, artifact.type, uri=artifact.uri, hash=artifact.hash)
        .add_support(artifact.id, "claim:C1")
        .add_verifier_certificate(certificate)
        .build()
    )
    project = CairnProject.open(tmp_path)
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="system:test", role="verifier"),
        reason="artifact hash verified",
    )

    assert certificate.status == "pass"
    assert decision.decision == "allowed"
    assert decision.events[0].new_state == "verified"


def test_failed_metric_threshold_certificate_does_not_authorize_verified(tmp_path: Path) -> None:
    metric = EvidenceItem(
        id="metric:accuracy",
        type="metric",
        uri="memory://metrics/accuracy",
        hash="sha256:accuracy",
        metadata={"metric_name": "accuracy", "value": 0.81},
    )
    certificate = MetricThresholdVerifier().verify(
        VerificationRequest(
            claim_id="claim:C1",
            evidence=[metric],
            parameters={"metric_name": "accuracy", "min_value": 0.9},
        )
    )
    case = (
        ClaimCaseBuilder(
            case_id="verifier-metric-threshold-fail",
            source_system="unit-test",
            stress_scenario="metric_threshold_fail",
        )
        .add_claim("claim:C1", "Accuracy is at least 0.90.", state="evidence_attached")
        .add_evidence(metric.id, metric.type, uri=metric.uri, hash=metric.hash, metadata=metric.metadata)
        .add_support(metric.id, "claim:C1")
        .add_verifier_certificate(certificate)
        .build()
    )
    project = CairnProject.open(tmp_path)
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="system:test", role="verifier"),
        reason="metric threshold checked",
    )

    assert certificate.status == "fail"
    assert certificate.reason == "metric_threshold_failed"
    assert decision.decision == "blocked"
    assert decision.blocking_reasons == ["missing_passing_verifier_certificate"]


def test_metric_threshold_certificate_feeds_release_authority(tmp_path: Path) -> None:
    metric = EvidenceItem(
        id="metric:accuracy",
        type="metric",
        uri="memory://metrics/accuracy",
        hash="sha256:accuracy",
        metadata={"metric_name": "accuracy", "value": 0.93},
    )
    certificate = MetricThresholdVerifier().verify(
        VerificationRequest(
            claim_id="claim:C1",
            evidence=[metric],
            parameters={"metric_name": "accuracy", "min_value": 0.9},
        )
    )
    case = (
        ClaimCaseBuilder(
            case_id="verifier-metric-threshold-release",
            source_system="unit-test",
            stress_scenario="metric_threshold_release",
        )
        .add_claim(
            "claim:C1",
            "Accuracy is at least 0.90.",
            state="verified",
            authority_state="verified",
            risk="medium",
        )
        .add_evidence(metric.id, metric.type, uri=metric.uri, hash=metric.hash, metadata=metric.metadata)
        .add_support(metric.id, "claim:C1")
        .add_verifier_certificate(certificate)
        .add_human_gate(
            "human_gate:H1",
            "claim:C1",
            "human:alice",
            authority="project_owner",
            scope={"claim": "claim:C1"},
            rationale="Reviewed metric threshold certificate.",
        )
        .add_risk_assessment("claim:C1", risk_tier="medium")
        .add_responsibility_assignment("claim:C1", accountable=[("human_pi", "human:pi@example.org")])
        .build()
    )
    project = CairnProject.open(tmp_path)
    project.import_claim_case(case)

    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="human:alice", role="project_owner"),
        reason="release metric claim",
    )

    assert certificate.status == "pass"
    assert decision.decision == "allowed"
    assert decision.events[0].new_state == "released"
