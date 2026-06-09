from __future__ import annotations

from pathlib import Path

from .authority import TransitionAuthority
from .graph import RelationGraph
from .models import (
    Actor,
    ClaimCase,
    ClaimState,
    ImportResult,
    TraceResult,
    TransitionDecision,
)
from .planner import InvalidationPlanner
from .projection import EventProjection
from .store import CairnProjectStore
from .trace_package import DecisionTracePackager
from .validation import build_validation_report, validation_report_markdown
from .validation_evidence import load_validation_evidence


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
        claims = self.store.load_claims()
        payloads = self.store.load_object_payloads()
        observed_state = None
        authority_state = None
        if object_id in claims:
            observed_state = projection.claim_observed_state(object_id)
            authority_state = projection.claim_authority_state(object_id)
        return TraceResult(
            object_id=object_id,
            object=payloads.get(object_id),
            observed_state=observed_state,
            authority_state=authority_state,
            projected_state=projection.object_state(object_id),
            incoming_relations=graph.incoming(object_id),
            outgoing_relations=graph.outgoing(object_id),
            events=projection.events_for(object_id),
            downstream_objects=[step.object_id for step in graph.downstream_of(object_id)],
        )

    def validate(self):
        report = build_validation_report(
            self.store.load_cases(),
            ledger=load_validation_evidence(self.root),
        )
        self.store.write_validation_report(
            report.model_dump(mode="json"),
            validation_report_markdown(report),
        )
        return report

    def decision_trace_package(self, claim_id: str, transition: str | None = None):
        packager = DecisionTracePackager(
            claims=self.store.load_claims(),
            evidence=self.store.load_evidence(),
            relations=self.store.load_relations(),
            events=self.store.load_events(),
            cases=self.store.load_cases(),
        )
        return packager.build(claim_id=claim_id, transition=transition)

    def request_transition(
        self,
        claim_id: str,
        target_state: ClaimState,
        actor: Actor,
        reason: str,
        force: bool = False,
        apply: bool = False,
        record_blocked: bool = False,
    ) -> TransitionDecision:
        authority = TransitionAuthority(
            claims=self.store.load_claims(),
            evidence=self.store.load_evidence(),
            graph=RelationGraph(self.store.load_relations()),
            projection=self._projection(),
            cases=self.store.load_cases(),
        )
        decision = authority.request_transition(
            claim_id=claim_id,
            target_state=target_state,
            actor=actor,
            reason=reason,
            force=force,
        )
        if (apply and decision.decision == "allowed") or (record_blocked and decision.decision == "blocked"):
            self.append_transition_events(decision)
        return decision

    def append_transition_events(self, decision: TransitionDecision):
        if decision.decision not in {"allowed", "blocked"}:
            return []
        for event in decision.events:
            self.store.append_event(event)
        return decision.events

    def apply_transition_decision(self, decision: TransitionDecision):
        if decision.decision != "allowed":
            return []
        return self.append_transition_events(decision)

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
