from __future__ import annotations

from typing import Any

from .models import (
    Actor,
    Claim,
    ClaimCase,
    Criticality,
    EvidenceItem,
    EvidenceStatus,
    EvidenceType,
    Relation,
    RelationType,
)


class ClaimCaseBuilder:
    """Small helper for adapter authors building CairnLab claim cases."""

    def __init__(
        self,
        case_id: str,
        source_system: str,
        stress_scenario: str = "imported_case",
        source_task: str | None = None,
    ):
        self.case_id = case_id
        self.source_system = source_system
        self.source_task = source_task
        self.stress_scenario = stress_scenario
        self.claims: list[Claim] = []
        self.evidence: list[EvidenceItem] = []
        self.relations: list[Relation] = []
        self.native_system_behavior: dict[str, Any] = {}
        self.expected_cairnlab_behavior: dict[str, Any] = {}
        self.failure_classes: list[str] = []

    def add_claim(
        self,
        claim_id: str,
        text: str,
        claim_type: str = "empirical_metric",
        state: str = "draft",
        *,
        scope: dict[str, Any] | None = None,
        risk: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> "ClaimCaseBuilder":
        self.claims.append(
            Claim(
                id=claim_id,
                text=text,
                type=claim_type,
                state=state,
                scope=scope or {},
                risk=risk,
                metadata=metadata or {},
            )
        )
        return self

    def add_evidence(
        self,
        evidence_id: str,
        evidence_type: str,
        *,
        uri: str | None = None,
        status: str = EvidenceStatus.VALID.value,
        hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ClaimCaseBuilder":
        self.evidence.append(
            EvidenceItem(
                id=evidence_id,
                type=evidence_type,
                uri=uri,
                status=status,
                hash=hash,
                metadata=metadata or {},
            )
        )
        return self

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        *,
        relation_id: str | None = None,
        criticality: str = Criticality.SUPPORTING.value,
    ) -> "ClaimCaseBuilder":
        self.relations.append(
            Relation(
                id=relation_id or f"rel:{len(self.relations) + 1:04d}",
                source=source,
                target=target,
                type=relation_type,
                criticality=criticality,
            )
        )
        return self

    def add_support(
        self,
        evidence_id: str,
        claim_id: str,
        *,
        criticality: str = Criticality.CRITICAL.value,
    ) -> "ClaimCaseBuilder":
        return self.add_relation(
            evidence_id,
            claim_id,
            RelationType.SUPPORTS.value,
            criticality=criticality,
        )

    def add_human_gate(
        self,
        gate_id: str,
        claim_id: str,
        actor: Actor | str,
        *,
        authority: str,
        scope: dict[str, Any],
        rationale: str,
    ) -> "ClaimCaseBuilder":
        actor_id = actor.id if isinstance(actor, Actor) else actor
        self.add_evidence(
            gate_id,
            EvidenceType.HUMAN_GATE.value,
            uri=f"cairn://human_gates/{gate_id.split(':')[-1]}",
            metadata={
                "actor": actor_id,
                "authority": authority,
                "scope": scope,
                "rationale": rationale,
            },
        )
        return self.add_relation(
            gate_id,
            claim_id,
            RelationType.APPROVED_BY.value,
            criticality=Criticality.CRITICAL.value,
        )

    def add_release_decision(
        self,
        decision_id: str,
        claim_id: str,
        actor: Actor | str,
        *,
        decision: str = "allow",
    ) -> "ClaimCaseBuilder":
        actor_id = actor.id if isinstance(actor, Actor) else actor
        self.add_evidence(
            decision_id,
            EvidenceType.RELEASE_DECISION.value,
            uri=f"cairn://release/{decision_id.split(':')[-1]}",
            metadata={"actor": actor_id, "decision": decision, "claim": claim_id},
        )
        return self.add_relation(
            decision_id,
            claim_id,
            RelationType.RELEASED_BY.value,
            criticality=Criticality.CRITICAL.value,
        )

    def set_native_behavior(self, **values: Any) -> "ClaimCaseBuilder":
        self.native_system_behavior.update(values)
        return self

    def add_failure_class(self, name: str) -> "ClaimCaseBuilder":
        if name not in self.failure_classes:
            self.failure_classes.append(name)
        return self

    def build(self) -> ClaimCase:
        return ClaimCase(
            case_id=self.case_id,
            source_system=self.source_system,
            source_task=self.source_task,
            stress_scenario=self.stress_scenario,
            claims=self.claims,
            evidence=self.evidence,
            relations=self.relations,
            native_system_behavior=self.native_system_behavior,
            expected_cairnlab_behavior=self.expected_cairnlab_behavior,
            failure_classes=self.failure_classes,
        )
