from __future__ import annotations

from collections.abc import Iterable

from .graph import RelationGraph
from .models import (
    Claim,
    ClaimCase,
    DecisionTracePackage,
    DecisionTracePackageExport,
    EvidenceItem,
    Relation,
    TransitionEvent,
)
from .utils import enum_value, stable_hash


DEFAULT_INCLUDES = [
    "claim",
    "evidence",
    "relations",
    "verifier_certificates",
    "human_gates",
    "release_decisions",
    "risk_assessment",
    "responsibility_assignment",
    "decision_trace_package",
    "material_dissent",
    "transition_events",
]


class DecisionTracePackager:
    """Builds reviewable decision traces without deciding transitions."""

    def __init__(
        self,
        claims: dict[str, Claim],
        evidence: dict[str, EvidenceItem],
        relations: Iterable[Relation],
        events: Iterable[TransitionEvent] = (),
        cases: Iterable[ClaimCase] = (),
    ):
        self.claims = claims
        self.evidence = evidence
        self.relations = list(relations)
        self.events = list(events)
        self.cases = list(cases)
        self.graph = RelationGraph(self.relations)

    @classmethod
    def from_case(
        cls,
        case: ClaimCase,
        events: Iterable[TransitionEvent] = (),
    ) -> "DecisionTracePackager":
        return cls(
            claims={claim.id: claim for claim in case.claims},
            evidence={item.id: item for item in case.evidence},
            relations=case.relations,
            events=events,
            cases=[case],
        )

    def build(
        self,
        claim_id: str,
        transition: str | None = None,
        package_id: str | None = None,
    ) -> DecisionTracePackageExport:
        claim = self.claims.get(claim_id)
        if claim is None:
            raise KeyError(f"Claim not found: {claim_id}")

        evidence_ids = self._evidence_ids_for_claim(claim_id)
        relation_ids = self._relation_ids_for_trace(claim_id, evidence_ids)
        event_items = [event for event in self.events if event.target in {claim_id, *evidence_ids}]
        governance = self._governance_for_claim(claim)
        package = self._package_for_claim(
            claim_id,
            transition=transition,
            package_id=package_id,
            existing=governance.get("decision_trace_package"),
        )

        export_without_hash = DecisionTracePackageExport(
            package=package,
            claim=claim.model_dump(mode="json", exclude_none=True),
            evidence=[
                self.evidence[evidence_id].model_dump(mode="json", exclude_none=True)
                for evidence_id in sorted(evidence_ids)
                if evidence_id in self.evidence
            ],
            relations=[
                relation.model_dump(mode="json", exclude_none=True)
                for relation in self.relations
                if relation.id in relation_ids
            ],
            events=[event.model_dump(mode="json", exclude_none=True) for event in event_items],
            governance=governance,
        )
        package.export_hash = stable_hash(
            export_without_hash.model_dump(mode="json", exclude_none=True, exclude={"package": {"export_hash"}})
        )
        return export_without_hash.model_copy(update={"package": package})

    def _evidence_ids_for_claim(self, claim_id: str) -> set[str]:
        evidence_ids: set[str] = set()
        for relation in [*self.graph.incoming(claim_id), *self.graph.outgoing(claim_id)]:
            for object_id in (relation.source, relation.target):
                if object_id in self.evidence:
                    evidence_ids.add(object_id)

        # Include verifier inputs so the trace can be reviewed without relying
        # only on the certificate summary.
        for evidence_id in list(evidence_ids):
            item = self.evidence.get(evidence_id)
            if not item:
                continue
            inputs = item.metadata.get("inputs") if item.metadata else None
            if isinstance(inputs, list):
                evidence_ids.update(str(input_id) for input_id in inputs if str(input_id) in self.evidence)
        return evidence_ids

    def _relation_ids_for_trace(self, claim_id: str, evidence_ids: set[str]) -> set[str]:
        object_ids = {claim_id, *evidence_ids}
        return {
            relation.id
            for relation in self.relations
            if relation.source in object_ids or relation.target in object_ids
        }

    def _governance_for_claim(self, claim: Claim) -> dict[str, object]:
        governance: dict[str, object] = {}
        if claim.lifecycle_context:
            governance["lifecycle_context"] = claim.lifecycle_context.model_dump(mode="json", exclude_none=True)
        if claim.risk_assessment:
            governance["risk_assessment"] = claim.risk_assessment.model_dump(mode="json", exclude_none=True)
        if claim.responsibility_assignment:
            governance["responsibility_assignment"] = claim.responsibility_assignment.model_dump(mode="json", exclude_none=True)
        if claim.decision_trace_package:
            governance["decision_trace_package"] = claim.decision_trace_package.model_dump(mode="json", exclude_none=True)

        for case in self.cases:
            for lifecycle_context in case.lifecycle_contexts:
                governance.setdefault("lifecycle_contexts", []).append(lifecycle_context.model_dump(mode="json", exclude_none=True))
            for risk_assessment in case.risk_assessments:
                if risk_assessment.object == claim.id:
                    governance["risk_assessment"] = risk_assessment.model_dump(mode="json", exclude_none=True)
            for assignment in case.responsibility_assignments:
                if assignment.object == claim.id:
                    governance["responsibility_assignment"] = assignment.model_dump(mode="json", exclude_none=True)
            for package in case.decision_trace_packages:
                if package.claim == claim.id:
                    governance["decision_trace_package"] = package.model_dump(mode="json", exclude_none=True)
        governance["material_dissent"] = [
            item.model_dump(mode="json", exclude_none=True)
            for item in self._typed_incoming_evidence(claim.id, "material_dissent")
        ]
        return governance

    def _typed_incoming_evidence(self, claim_id: str, evidence_type: str) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for relation in self.graph.incoming(claim_id):
            item = self.evidence.get(relation.source)
            if item and enum_value(item.type) == evidence_type:
                items.append(item)
        return items

    def _package_for_claim(
        self,
        claim_id: str,
        *,
        transition: str | None,
        package_id: str | None,
        existing: object | None,
    ) -> DecisionTracePackage:
        if isinstance(existing, dict):
            package = DecisionTracePackage.model_validate(existing)
            if transition and not package.transition:
                package.transition = transition
            return package
        return DecisionTracePackage(
            id=package_id or f"decision_trace:{claim_id.split(':')[-1]}",
            claim=claim_id,
            transition=transition,
            includes=list(DEFAULT_INCLUDES),
        )
