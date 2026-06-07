from __future__ import annotations

from pathlib import Path

from .graph import RelationGraph
from .models import (
    Actor,
    ClaimCase,
    ClaimState,
    EventType,
    ImportResult,
    TraceResult,
    TransitionDecision,
    TransitionEvent,
)
from .planner import InvalidationPlanner
from .projection import EventProjection
from .store import CairnProjectStore
from .utils import enum_value, stable_hash, utc_event_id
from .validation import build_validation_report, validation_report_markdown


class CairnProject:
    """Thin local-project facade over reusable CairnLab modules."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.store = CairnProjectStore(self.root)

    @classmethod
    def open(cls, root: str | Path = ".") -> "CairnProject":
        return cls(Path(root))

    def init(self) -> None:
        self.store.init()

    def import_case(self, path: str | Path) -> ImportResult:
        return self.store.import_case(Path(path))

    def import_claim_case(self, case: ClaimCase) -> ImportResult:
        return self.store.import_claim_case(case)

    def plan_revert(
        self,
        target_id: str,
        reason: str,
        actor: Actor | None = None,
        mode: str = "invalidate_only",
    ):
        planner = self._planner()
        return planner.plan_revert(target_id=target_id, reason=reason, actor=actor, mode=mode)

    def apply_plan(self, plan):
        planner = self._planner()
        events = planner.events_from_plan(plan)
        for event in events:
            self.store.append_event(event)
        return events

    def trace(self, object_id: str) -> TraceResult:
        graph = RelationGraph(self.store.load_relations())
        projection = self._projection()
        payloads = self.store.load_object_payloads()
        return TraceResult(
            object_id=object_id,
            object=payloads.get(object_id),
            projected_state=projection.object_state(object_id),
            incoming_relations=graph.incoming(object_id),
            outgoing_relations=graph.outgoing(object_id),
            events=projection.events_for(object_id),
            downstream_objects=[step.object_id for step in graph.downstream_of(object_id)],
        )

    def validate(self):
        report = build_validation_report(self.store.load_cases())
        self.store.write_validation_report(
            report.model_dump(mode="json"),
            validation_report_markdown(report),
        )
        return report

    def request_transition(
        self,
        claim_id: str,
        target_state: ClaimState,
        actor: Actor,
        reason: str,
        force: bool = False,
    ) -> TransitionDecision:
        claims = self.store.load_claims()
        graph = RelationGraph(self.store.load_relations())
        projection = self._projection()
        current_state_value = projection.claim_state(claim_id)
        current_state = ClaimState(current_state_value) if current_state_value else ClaimState.DRAFT

        blocking_reasons: list[str] = []
        required_actions: list[str] = []
        if claim_id not in claims:
            blocking_reasons.append("claim_not_found")
        if target_state == ClaimState.VERIFIED and not self._has_machine_addressable_evidence(claim_id, graph):
            blocking_reasons.append("missing_machine_addressable_evidence")
            required_actions.append("attach evidence with stable URI or hash")
        if target_state == ClaimState.RELEASED:
            if current_state != ClaimState.VERIFIED and not force:
                blocking_reasons.append("release_requires_verified_state")
                required_actions.append("verify claim before release or record explicit force")
            if not self._has_human_gate(claim_id, graph):
                blocking_reasons.append("missing_human_gate")
                required_actions.append("record human gate with actor, authority, scope, and rationale")
            if self._has_unresolved_material_dissent(claim_id, graph):
                blocking_reasons.append("unresolved_material_dissent")
                required_actions.append("resolve dissent through verifier or explicit human override")

        event_type = EventType.TRANSITION_BLOCKED if blocking_reasons else EventType.TRANSITION_ALLOWED
        event = TransitionEvent(
            id=utc_event_id("transition", claim_id),
            type=event_type,
            target=claim_id,
            actor=actor,
            reason=reason,
            previous_state=current_state.value,
            new_state=target_state.value if not blocking_reasons else current_state.value,
            input_hash=stable_hash(
                {
                    "claim_id": claim_id,
                    "target_state": target_state.value,
                    "reason": reason,
                    "force": force,
                }
            ),
            payload={"blocking_reasons": blocking_reasons, "force": force},
        )
        return TransitionDecision(
            decision="blocked" if blocking_reasons else "allowed",
            claim_id=claim_id,
            requested_state=target_state,
            current_state=current_state,
            blocking_reasons=blocking_reasons,
            required_actions=required_actions,
            events=[event],
        )

    def _planner(self) -> InvalidationPlanner:
        return InvalidationPlanner(
            evidence=self.store.load_evidence(),
            projection=self._projection(),
            graph=RelationGraph(self.store.load_relations()),
        )

    def _projection(self) -> EventProjection:
        return EventProjection(
            claims=self.store.load_claims(),
            evidence=self.store.load_evidence(),
            events=self.store.load_events(),
        )

    def _has_machine_addressable_evidence(self, claim_id: str, graph: RelationGraph) -> bool:
        evidence = self.store.load_evidence()
        for relation in graph.incoming(claim_id):
            item = evidence.get(relation.source)
            if item and (item.uri or item.hash):
                return True
        return False

    def _has_human_gate(self, claim_id: str, graph: RelationGraph) -> bool:
        evidence = self.store.load_evidence()
        for relation in graph.incoming(claim_id):
            item = evidence.get(relation.source)
            if item and enum_value(item.type) == "human_gate":
                metadata = item.metadata or {}
                return bool(metadata.get("actor") and metadata.get("authority") and metadata.get("scope") and metadata.get("rationale"))
        return False

    def _has_unresolved_material_dissent(self, claim_id: str, graph: RelationGraph) -> bool:
        evidence = self.store.load_evidence()
        for relation in graph.incoming(claim_id):
            item = evidence.get(relation.source)
            if not item or enum_value(item.type) != "material_dissent":
                continue
            metadata = item.metadata or {}
            if metadata.get("severity") == "material" and not metadata.get("resolved", False):
                return True
        return False
