from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cairnlab import Actor, ClaimCaseBuilder
from cairnlab.cli import app
from cairnlab.engine import CairnProject
from cairnlab.models import ClaimState
from cairnlab.transition_explain import (
    explain_transition_decision,
    render_transition_explanation_text,
)


def test_transition_explanation_summarizes_blocked_decision(tmp_path: Path) -> None:
    project = _project_with_unverified_claim(tmp_path)
    decision = project.request_transition(
        "claim:C1",
        ClaimState.VERIFIED,
        Actor(id="system:test", role="verifier"),
        reason="verify claim",
    )

    explanation = explain_transition_decision(decision)
    rendered = render_transition_explanation_text(explanation)

    assert explanation.decision == "blocked"
    assert explanation.current_state == "draft"
    assert explanation.requested_state == "verified"
    assert explanation.event_type == "TransitionBlocked"
    assert "missing_machine_addressable_evidence" in explanation.blocking_reasons
    assert "attach evidence with stable URI or hash" in explanation.required_actions
    assert "claim:C1 cannot transition from draft to verified." in rendered


def test_transition_explain_cli_json_is_plan_only(tmp_path: Path) -> None:
    project = _project_with_unverified_claim(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "transition",
            "explain",
            "claim:C1",
            "--to",
            "verified",
            "--reason",
            "verify claim",
            "--path",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"decision": "blocked"' in result.output
    assert '"missing_machine_addressable_evidence"' in result.output
    assert project.store.load_events() == []


def test_transition_explain_cli_text_lists_required_actions(tmp_path: Path) -> None:
    _project_with_unverified_claim(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "transition",
            "explain",
            "claim:C1",
            "--to",
            "verified",
            "--reason",
            "verify claim",
            "--path",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Blocking reasons:" in result.output
    assert "Required actions:" in result.output
    assert "Proposed event: TransitionBlocked" in result.output


def _project_with_unverified_claim(path: Path) -> CairnProject:
    project = CairnProject.open(path)
    project.import_claim_case(
        ClaimCaseBuilder(
            case_id="transition-explain-blocked",
            source_system="unit-test",
            stress_scenario="missing_verifier",
        )
        .add_claim("claim:C1", "A claim with no evidence.")
        .build()
    )
    return project
