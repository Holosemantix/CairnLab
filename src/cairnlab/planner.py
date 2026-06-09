from __future__ import annotations

from .graph import RelationGraph
from .models import (
    Actor,
    AffectedAction,
    AffectedObject,
    ClaimState,
    EventType,
    EvidenceItem,
    EvidenceType,
    RevertPlan,
    RevertRequest,
    TransitionEvent,
)
from .projection import EventProjection
from .utils import enum_value, stable_hash, utc_event_id


class InvalidationPlanner:
    """Pure planner for claim-state invalidation propagation."""

    def __init__(
        self,
        evidence: dict[str, EvidenceItem],
        projection: EventProjection,
        graph: RelationGraph,
    ):
        self.evidence = evidence
        self.projection = projection
        self.graph = graph

    def plan_revert(
        self,
        target_id: str,
        reason: str,
        actor: Actor | None = None,
        mode: str = "invalidate_only",
    ) -> RevertPlan:
        request = RevertRequest(
            target=target_id,
            reason=reason,
            mode=mode,
            actor=actor or Actor(id="user:unknown"),
        )
        affected: list[AffectedObject] = []

        root = self._affected_for_object(target_id, reason, relation_path=[])
        if root:
            affected.append(root)

        seen = {target_id}
        for step in self.graph.downstream_of(target_id):
            if step.object_id in seen:
                continue
            seen.add(step.object_id)
            item = self._affected_for_object(step.object_id, reason, list(step.relation_path))
            if item and item.action != AffectedAction.NO_ACTION:
                affected.append(item)

        required_actions = self._required_actions(affected)
        return RevertPlan(request=request, affected=affected, required_actions=required_actions)

    def events_from_plan(self, plan: RevertPlan) -> list[TransitionEvent]:
        root_event = TransitionEvent(
            id=utc_event_id("revert", plan.request.target),
            type=EventType.REVERT_REQUESTED,
            target=plan.request.target,
            actor=plan.request.actor,
            reason=plan.request.reason,
            input_hash=stable_hash(plan.model_dump(mode="json")),
            payload={"mode": plan.request.mode},
        )
        events = [root_event]
        for item in plan.affected:
            if not item.proposed_event_type:
                continue
            events.append(
                TransitionEvent(
                    id=utc_event_id("derived", item.id),
                    type=item.proposed_event_type,
                    target=item.id,
                    actor=plan.request.actor,
                    reason=item.reason,
                    previous_state=item.previous_state,
                    new_state=item.new_state,
                    caused_by=root_event.id,
                    input_hash=stable_hash(item.model_dump(mode="json")),
                    payload={"action": item.action, "relation_path": item.relation_path},
                )
            )
        return events

    def _affected_for_object(
        self,
        object_id: str,
        reason: str,
        relation_path: list[str],
    ) -> AffectedObject | None:
        previous_state = self.projection.object_state(object_id)
        if object_id.startswith("claim:"):
            previous_state = self._claim_invalidation_state(object_id)
            return self._claim_effect(object_id, reason, relation_path, previous_state)

        item = self.evidence.get(object_id)
        if item:
            return self._evidence_effect(item, reason, relation_path, previous_state)

        return AffectedObject(
            id=object_id,
            action=AffectedAction.NEEDS_REVIEW,
            reason=f"Object is downstream of invalidated target: {reason}",
            relation_path=relation_path,
            proposed_event_type=EventType.OBJECT_MARKED_STALE,
            previous_state=previous_state,
            new_state="stale",
        )

    def _claim_invalidation_state(self, claim_id: str) -> str | None:
        authority_state = self.projection.claim_authority_state(claim_id)
        if self.projection.events_for(claim_id) or authority_state != ClaimState.DRAFT.value:
            return authority_state
        return self.projection.claim_observed_state(claim_id) or authority_state

    def _claim_effect(
        self,
        claim_id: str,
        reason: str,
        relation_path: list[str],
        previous_state: str | None,
    ) -> AffectedObject:
        if previous_state == ClaimState.RELEASED.value:
            action = AffectedAction.DOWNGRADE
            event_type = EventType.CLAIM_DOWNGRADED
            new_state = ClaimState.DOWNGRADED.value
        else:
            action = AffectedAction.CHALLENGE
            event_type = EventType.CLAIM_CHALLENGED
            new_state = ClaimState.CHALLENGED.value
        return AffectedObject(
            id=claim_id,
            action=action,
            reason=f"Claim authority depends on invalidated support: {reason}",
            relation_path=relation_path,
            proposed_event_type=event_type,
            previous_state=previous_state,
            new_state=new_state,
        )

    def _evidence_effect(
        self,
        item: EvidenceItem,
        reason: str,
        relation_path: list[str],
        previous_state: str | None,
    ) -> AffectedObject:
        item_type = enum_value(item.type)
        if item_type == EvidenceType.METRIC.value:
            return self._effect(item.id, AffectedAction.INVALIDATE, EventType.METRIC_INVALIDATED, reason, relation_path, previous_state, "invalidated")
        if item_type == EvidenceType.ARTIFACT.value:
            return self._effect(item.id, AffectedAction.INVALIDATE, EventType.ARTIFACT_INVALIDATED, reason, relation_path, previous_state, "invalidated")
        if item_type == EvidenceType.DATASET.value:
            return self._effect(item.id, AffectedAction.INVALIDATE, EventType.DATASET_INVALIDATED, reason, relation_path, previous_state, "invalidated")
        if item_type == EvidenceType.CODE_COMMIT.value:
            return self._effect(item.id, AffectedAction.INVALIDATE, EventType.CODE_INVALIDATED, reason, relation_path, previous_state, "invalidated")
        if item_type == EvidenceType.CITATION.value:
            return self._effect(item.id, AffectedAction.INVALIDATE, EventType.CITATION_INVALIDATED, reason, relation_path, previous_state, "invalidated")
        if item_type == EvidenceType.PAPER_SECTION.value:
            return self._effect(item.id, AffectedAction.MARK_STALE, EventType.PAPER_SECTION_MARKED_STALE, reason, relation_path, previous_state, "stale")
        if item_type == EvidenceType.VERIFIER_CERTIFICATE.value:
            return self._effect(item.id, AffectedAction.INVALIDATE, EventType.VERIFIER_CERTIFICATE_INVALIDATED, reason, relation_path, previous_state, "invalidated")
        if item_type == EvidenceType.HUMAN_GATE.value:
            return self._effect(item.id, AffectedAction.REQUIRE_REAPPROVAL, EventType.HUMAN_REAPPROVAL_REQUIRED, reason, relation_path, previous_state, "stale")
        if item_type == EvidenceType.RELEASE_DECISION.value:
            return self._effect(item.id, AffectedAction.REOPEN_RELEASE_DECISION, EventType.RELEASE_DECISION_REOPENED, reason, relation_path, previous_state, "stale")
        if item_type == EvidenceType.MATERIAL_DISSENT.value:
            return self._effect(item.id, AffectedAction.NEEDS_REVIEW, EventType.OBJECT_MARKED_STALE, reason, relation_path, previous_state, "stale")
        return self._effect(item.id, AffectedAction.INVALIDATE, EventType.EVIDENCE_INVALIDATED, reason, relation_path, previous_state, "invalidated")

    def _effect(
        self,
        object_id: str,
        action: AffectedAction,
        event_type: EventType,
        reason: str,
        relation_path: list[str],
        previous_state: str | None,
        new_state: str,
    ) -> AffectedObject:
        return AffectedObject(
            id=object_id,
            action=action,
            reason=f"Object authority affected by invalidation: {reason}",
            relation_path=relation_path,
            proposed_event_type=event_type,
            previous_state=previous_state,
            new_state=new_state,
        )

    def _required_actions(self, affected: list[AffectedObject]) -> list[str]:
        actions: list[str] = []
        if any(item.action in {AffectedAction.DOWNGRADE, AffectedAction.CHALLENGE} for item in affected):
            actions.append("refresh or replace invalidated evidence before stronger claim state")
        if any(item.action == AffectedAction.REQUIRE_REAPPROVAL for item in affected):
            actions.append("record a new human gate with actor, authority, scope, and rationale")
        if any(item.action == AffectedAction.REOPEN_RELEASE_DECISION for item in affected):
            actions.append("reopen release decision until affected claims are revalidated")
        return actions
