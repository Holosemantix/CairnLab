from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cairnlab import Actor, CairnProject, CairnRuntime, ClaimCaseBuilder
from cairnlab.cli import app
from cairnlab.models import ClaimState, VerifierCertificate


def test_runtime_decision_trace_package_includes_release_authority_inputs() -> None:
    case = _decision_trace_case().build()
    runtime = CairnRuntime.from_case(case)

    package = runtime.decision_trace_package("claim:C1", transition="release")

    evidence_ids = {item["id"] for item in package.evidence}
    relation_types = {relation["type"] for relation in package.relations}

    assert package.package.claim == "claim:C1"
    assert package.package.transition == "release"
    assert package.package.export_hash.startswith("sha256:")
    assert {"metric:m1", "verifier:V1", "human_gate:H1", "release_decision:R1", "dissent:D1"} <= evidence_ids
    assert {"supports", "verified_by", "approved_by", "released_by", "challenges"} <= relation_types
    assert package.governance["risk_assessment"]["object"] == "claim:C1"
    assert package.governance["responsibility_assignment"]["accountable"][0]["actor"] == "human:pi@example.org"
    assert package.governance["material_dissent"][0]["id"] == "dissent:D1"


def test_decision_trace_package_hash_is_stable_for_same_inputs() -> None:
    case = _decision_trace_case().build()
    runtime = CairnRuntime.from_case(case)

    first = runtime.decision_trace_package("claim:C1", transition="release")
    second = runtime.decision_trace_package("claim:C1", transition="release")

    assert first.package.export_hash == second.package.export_hash


def test_project_decision_trace_package_includes_transition_events(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    project.import_claim_case(_decision_trace_case().build())
    decision = project.request_transition(
        "claim:C1",
        ClaimState.RELEASED,
        Actor(id="human:alice", role="project_owner"),
        reason="release claim",
    )
    for event in decision.events:
        project.store.append_event(event)

    package = project.decision_trace_package("claim:C1", transition="release")

    assert package.events[0]["type"] == "TransitionAllowed"
    assert package.events[0]["new_state"] == "released"


def test_decision_trace_package_rejects_unknown_claim() -> None:
    runtime = CairnRuntime.from_case(_decision_trace_case().build())

    try:
        runtime.decision_trace_package("claim:missing")
    except KeyError as exc:
        assert "claim:missing" in str(exc)
    else:
        raise AssertionError("expected KeyError for unknown claim")


def test_decision_trace_cli_outputs_json_package(tmp_path: Path) -> None:
    project = CairnProject.open(tmp_path)
    project.import_claim_case(_decision_trace_case().build())
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "decision-trace",
            "claim:C1",
            "--transition",
            "release",
            "--path",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"claim": "claim:C1"' in result.output
    assert '"export_hash": "sha256:' in result.output


def _decision_trace_case() -> ClaimCaseBuilder:
    return (
        ClaimCaseBuilder(
            case_id="decision-trace-case",
            source_system="unit-test",
            stress_scenario="release_trace",
        )
        .add_claim(
            "claim:C1",
            "A verified claim ready for release.",
            state="verified",
            authority_state="verified",
            risk="high",
        )
        .add_evidence("metric:m1", "metric", uri="memory://metrics/m1", hash="sha256:m1", metadata={"value": 0.93})
        .add_support("metric:m1", "claim:C1")
        .add_verifier_certificate(
            VerifierCertificate(
                id="verifier:V1",
                verifier="unit-test@0.1.0",
                claim="claim:C1",
                status="pass",
                inputs=["metric:m1"],
                result={"status": "ok"},
                can_authorize=["evidence_attached -> verified"],
            )
        )
        .add_human_gate(
            "human_gate:H1",
            "claim:C1",
            "human:alice",
            authority="project_owner",
            scope={"claim": "claim:C1"},
            rationale="Reviewed verifier certificate and metric evidence.",
        )
        .add_release_decision("release_decision:R1", "claim:C1", "human:alice")
        .add_evidence(
            "dissent:D1",
            "material_dissent",
            uri="cairn://dissent/D1",
            metadata={"severity": "material", "resolved": True},
        )
        .add_relation("dissent:D1", "claim:C1", "challenges", criticality="critical")
        .add_risk_assessment("claim:C1", risk_tier="high")
        .add_responsibility_assignment(
            "claim:C1",
            accountable=[("human_pi", "human:pi@example.org")],
            responsible=[("verifier", "system:verifier")],
        )
        .add_decision_trace_package(
            "dtp:D1",
            "claim:C1",
            includes=["claim", "evidence", "verifier_certificates", "human_gate", "risk_assessment"],
        )
    )
