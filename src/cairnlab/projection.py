from __future__ import annotations

from .models import Claim, ClaimState, EventType, EvidenceItem, EvidenceStatus, TransitionEvent
from .utils import enum_value


CLAIM_EVENT_STATES = {
    EventType.CLAIM_CHALLENGED.value: ClaimState.CHALLENGED.value,
    EventType.CLAIM_DOWNGRADED.value: ClaimState.DOWNGRADED.value,
    EventType.CLAIM_RETRACTED.value: ClaimState.RETRACTED.value,
    EventType.CLAIM_INVALIDATED.value: ClaimState.INVALIDATED.value,
    EventType.OBJECT_MARKED_STALE.value: ClaimState.STALE.value,
}

EVIDENCE_EVENT_STATES = {
    EventType.EVIDENCE_INVALIDATED.value: EvidenceStatus.INVALIDATED.value,
    EventType.METRIC_INVALIDATED.value: EvidenceStatus.INVALIDATED.value,
    EventType.ARTIFACT_INVALIDATED.value: EvidenceStatus.INVALIDATED.value,
    EventType.DATASET_INVALIDATED.value: EvidenceStatus.INVALIDATED.value,
    EventType.CODE_INVALIDATED.value: EvidenceStatus.INVALIDATED.value,
    EventType.CITATION_INVALIDATED.value: EvidenceStatus.INVALIDATED.value,
    EventType.PAPER_SECTION_MARKED_STALE.value: EvidenceStatus.STALE.value,
    EventType.OBJECT_MARKED_STALE.value: EvidenceStatus.STALE.value,
    EventType.VERIFIER_CERTIFICATE_INVALIDATED.value: EvidenceStatus.INVALIDATED.value,
    EventType.HUMAN_REAPPROVAL_REQUIRED.value: EvidenceStatus.STALE.value,
    EventType.RELEASE_DECISION_REOPENED.value: EvidenceStatus.STALE.value,
}


class EventProjection:
    """Derives current state from imported base objects plus ordered events."""

    def __init__(
        self,
        claims: dict[str, Claim],
        evidence: dict[str, EvidenceItem],
        events: list[TransitionEvent],
    ):
        self.claims = claims
        self.evidence = evidence
        self.events = events

    def object_state(self, object_id: str) -> str | None:
        if object_id in self.claims:
            return self.claim_state(object_id)
        if object_id in self.evidence:
            return self.evidence_status(object_id)
        return None

    def claim_state(self, claim_id: str) -> str | None:
        claim = self.claims.get(claim_id)
        if not claim:
            return None
        state = enum_value(claim.state)
        for event in self.events:
            if event.target != claim_id:
                continue
            event_type = enum_value(event.type)
            if event.new_state:
                state = event.new_state
            elif event_type in CLAIM_EVENT_STATES:
                state = CLAIM_EVENT_STATES[event_type]
        return state

    def evidence_status(self, evidence_id: str) -> str | None:
        item = self.evidence.get(evidence_id)
        if not item:
            return None
        status = enum_value(item.status)
        for event in self.events:
            if event.target != evidence_id:
                continue
            event_type = enum_value(event.type)
            if event.new_state:
                status = event.new_state
            elif event_type in EVIDENCE_EVENT_STATES:
                status = EVIDENCE_EVENT_STATES[event_type]
        return status

    def events_for(self, object_id: str) -> list[TransitionEvent]:
        return [event for event in self.events if event.target == object_id]
