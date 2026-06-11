from __future__ import annotations

from pydantic import BaseModel, Field

from .models import TransitionDecision
from .utils import enum_value


class TransitionExplanation(BaseModel):
    claim_id: str
    decision: str
    current_state: str
    requested_state: str
    summary: str
    blocking_reasons: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    would_append_event: bool
    event_type: str | None = None


def explain_transition_decision(decision: TransitionDecision) -> TransitionExplanation:
    event = decision.events[0] if decision.events else None
    current_state = enum_value(decision.current_state)
    requested_state = enum_value(decision.requested_state)
    return TransitionExplanation(
        claim_id=decision.claim_id,
        decision=decision.decision,
        current_state=current_state,
        requested_state=requested_state,
        summary=_summary(
            decision.claim_id,
            decision.decision,
            current_state,
            requested_state,
        ),
        blocking_reasons=list(decision.blocking_reasons),
        required_actions=list(decision.required_actions),
        would_append_event=event is not None,
        event_type=enum_value(event.type) if event else None,
    )


def render_transition_explanation_text(explanation: TransitionExplanation) -> str:
    lines = [
        explanation.summary,
        f"Decision: {explanation.decision}",
        f"Current state: {explanation.current_state}",
        f"Requested state: {explanation.requested_state}",
    ]
    if explanation.blocking_reasons:
        lines.append("Blocking reasons:")
        lines.extend(f"- {reason}" for reason in explanation.blocking_reasons)
    if explanation.required_actions:
        lines.append("Required actions:")
        lines.extend(f"- {action}" for action in explanation.required_actions)
    if explanation.event_type:
        lines.append(f"Proposed event: {explanation.event_type}")
    return "\n".join(lines)


def _summary(claim_id: str, decision: str, current_state: str, requested_state: str) -> str:
    if decision == "allowed":
        return f"{claim_id} can transition from {current_state} to {requested_state}."
    return f"{claim_id} cannot transition from {current_state} to {requested_state}."
