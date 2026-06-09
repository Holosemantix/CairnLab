from __future__ import annotations

from typing import Any

from .models import (
    Actor,
    Claim,
    ClaimState,
    ClaimCase,
    Criticality,
    DecisionTracePackage,
    EvidenceItem,
    EvidenceStatus,
    EvidenceType,
    Relation,
    RelationType,
    ResponsibilityAssignment,
    ResponsibilityEntry,
    RiskAssessment,
    VerifierCertificate,
)
from .utils import safe_id_filename, stable_hash


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
        self.risk_assessments: list[RiskAssessment] = []
        self.responsibility_assignments: list[ResponsibilityAssignment] = []
        self.decision_trace_packages: list[DecisionTracePackage] = []
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
        authority_state: str = ClaimState.DRAFT.value,
        scope: dict[str, Any] | None = None,
        risk: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> "ClaimCaseBuilder":
        self.claims.append(
            Claim(
                id=claim_id,
                text=text,
                type=claim_type,
                observed_state=state,
                authority_state=authority_state,
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

    def add_verifier_certificate(
        self,
        certificate: VerifierCertificate,
        *,
        criticality: str = Criticality.SUPPORTING.value,
        uri: str | None = None,
        hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ClaimCaseBuilder":
        payload = certificate.model_dump(mode="json", exclude_none=True)
        self.add_evidence(
            certificate.id,
            EvidenceType.VERIFIER_CERTIFICATE.value,
            uri=uri or f"cairn://verifier_certificates/{safe_id_filename(certificate.id).removesuffix('.yaml')}",
            hash=hash or stable_hash(payload),
            metadata={
                "verdict": certificate.status,
                **(metadata or {}),
                **payload,
            },
        )
        return self.add_relation(
            certificate.id,
            certificate.claim,
            RelationType.VERIFIED_BY.value,
            criticality=criticality,
        )

    def add_risk_assessment(
        self,
        object_id: str,
        *,
        risk_tier: str = "medium",
        dimensions: dict[str, Any] | None = None,
        controls: list[str] | None = None,
        identified_risks: list[str] | None = None,
    ) -> "ClaimCaseBuilder":
        self.risk_assessments.append(
            RiskAssessment(
                object=object_id,
                risk_tier=risk_tier,
                dimensions=dimensions or {},
                controls=controls or [],
                identified_risks=identified_risks or [],
            )
        )
        return self

    def add_responsibility_assignment(
        self,
        object_id: str,
        *,
        action: str = "release_claim",
        responsible: list[ResponsibilityEntry | dict[str, str] | tuple[str, str]] | None = None,
        accountable: list[ResponsibilityEntry | dict[str, str] | tuple[str, str]] | None = None,
        consulted: list[ResponsibilityEntry | dict[str, str] | tuple[str, str]] | None = None,
        informed: list[ResponsibilityEntry | dict[str, str] | tuple[str, str]] | None = None,
    ) -> "ClaimCaseBuilder":
        self.responsibility_assignments.append(
            ResponsibilityAssignment(
                object=object_id,
                action=action,
                responsible=self._responsibility_entries(responsible or []),
                accountable=self._responsibility_entries(accountable or []),
                consulted=self._responsibility_entries(consulted or []),
                informed=self._responsibility_entries(informed or []),
            )
        )
        return self

    def add_decision_trace_package(
        self,
        package_id: str,
        claim_id: str,
        *,
        transition: str | None = None,
        includes: list[str] | None = None,
        export_hash: str | None = None,
    ) -> "ClaimCaseBuilder":
        self.decision_trace_packages.append(
            DecisionTracePackage(
                id=package_id,
                claim=claim_id,
                transition=transition,
                includes=includes or [],
                export_hash=export_hash,
            )
        )
        return self

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
            risk_assessments=self.risk_assessments,
            responsibility_assignments=self.responsibility_assignments,
            decision_trace_packages=self.decision_trace_packages,
        )

    def _responsibility_entries(
        self,
        values: list[ResponsibilityEntry | dict[str, str] | tuple[str, str]],
    ) -> list[ResponsibilityEntry]:
        entries: list[ResponsibilityEntry] = []
        for value in values:
            if isinstance(value, ResponsibilityEntry):
                entries.append(value)
            elif isinstance(value, dict):
                entries.append(ResponsibilityEntry.model_validate(value))
            else:
                role, actor = value
                entries.append(ResponsibilityEntry(role=role, actor=actor))
        return entries
