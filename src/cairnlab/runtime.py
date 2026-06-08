from __future__ import annotations

from collections.abc import Iterable

from .graph import RelationGraph
from .models import Actor, Claim, ClaimCase, EvidenceItem, Relation, RevertPlan, TraceResult, TransitionEvent
from .planner import InvalidationPlanner
from .projection import EventProjection
from .trace_package import DecisionTracePackager


class CairnRuntime:
    """Filesystem-free runtime for reusable claim-state invalidation.

    Host AutoResearch projects can use this class without adopting the CairnLab
    CLI or `.cairn/` store. It composes the graph, projection, and planner
    modules around plain model objects.
    """

    def __init__(
        self,
        claims: Iterable[Claim],
        evidence: Iterable[EvidenceItem],
        relations: Iterable[Relation],
        events: Iterable[TransitionEvent] | None = None,
        cases: Iterable[ClaimCase] | None = None,
    ):
        self.claims = {claim.id: claim for claim in claims}
        self.evidence = {item.id: item for item in evidence}
        self.relations = list(relations)
        self.events = list(events or [])
        self.cases = list(cases or [])
        self.graph = RelationGraph(self.relations)
        self.projection = EventProjection(self.claims, self.evidence, self.events)
        self.planner = InvalidationPlanner(self.evidence, self.projection, self.graph)

    @classmethod
    def from_case(
        cls,
        case: ClaimCase,
        events: Iterable[TransitionEvent] | None = None,
    ) -> "CairnRuntime":
        return cls(
            claims=case.claims,
            evidence=case.evidence,
            relations=case.relations,
            events=events,
            cases=[case],
        )

    def plan_revert(
        self,
        target_id: str,
        reason: str,
        actor: Actor | None = None,
        mode: str = "invalidate_only",
    ) -> RevertPlan:
        return self.planner.plan_revert(target_id=target_id, reason=reason, actor=actor, mode=mode)

    def events_from_plan(self, plan: RevertPlan) -> list[TransitionEvent]:
        return self.planner.events_from_plan(plan)

    def with_events(self, events: Iterable[TransitionEvent]) -> "CairnRuntime":
        return CairnRuntime(
            claims=self.claims.values(),
            evidence=self.evidence.values(),
            relations=self.relations,
            events=[*self.events, *events],
            cases=self.cases,
        )

    def trace(self, object_id: str) -> TraceResult:
        payload = None
        if object_id in self.claims:
            payload = self.claims[object_id].model_dump(mode="json", exclude_none=True)
        elif object_id in self.evidence:
            payload = self.evidence[object_id].model_dump(mode="json", exclude_none=True)
        return TraceResult(
            object_id=object_id,
            object=payload,
            projected_state=self.projection.object_state(object_id),
            incoming_relations=self.graph.incoming(object_id),
            outgoing_relations=self.graph.outgoing(object_id),
            events=self.projection.events_for(object_id),
            downstream_objects=[step.object_id for step in self.graph.downstream_of(object_id)],
        )

    def decision_trace_package(self, claim_id: str, transition: str | None = None):
        return DecisionTracePackager(
            claims=self.claims,
            evidence=self.evidence,
            relations=self.relations,
            events=self.events,
            cases=self.cases,
        ).build(claim_id=claim_id, transition=transition)
