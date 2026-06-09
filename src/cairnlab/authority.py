from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .graph import RelationGraph
from .models import (
    Actor,
    Claim,
    ClaimCase,
    ClaimState,
    DecisionTracePackage,
    EventType,
    EvidenceItem,
    EvidenceStatus,
    EvidenceType,
    RelationType,
    ResponsibilityAssignment,
    RiskAssessment,
    TransitionDecision,
    TransitionEvent,
    VerifierCertificate,
)
from .projection import EventProjection
from .utils import enum_value, stable_hash, utc_event_id


CONSEQUENTIAL_STATES = {
    ClaimState.RELEASED.value,
    ClaimState.DOWNGRADED.value,
    ClaimState.RETRACTED.value,
}

HIGH_IMPACT_RISK_TIERS = {"high", "critical"}
OVERRIDE_AUTHORITIES = {
    "release_override",
    "principal_investigator",
    "governance_owner",
}
VERIFIER_AUTHORIZED_TRANSITIONS = {
    "evidence_attached -> verified",
    "verification_pending -> verified",
}
INVALID_EVIDENCE_STATES = {
    EvidenceStatus.INVALIDATED.value,
    EvidenceStatus.STALE.value,
}
NON_MACHINE_EVIDENCE_TYPES = {
    EvidenceType.HUMAN_GATE.value,
    EvidenceType.RELEASE_DECISION.value,
    EvidenceType.MATERIAL_DISSENT.value,
    EvidenceType.RESPONSIBILITY_ASSIGNMENT.value,
    EvidenceType.RISK_ASSESSMENT.value,
    EvidenceType.DECISION_TRACE_PACKAGE.value,
    EvidenceType.VERIFIER_CERTIFICATE.value,
}


@dataclass(frozen=True)
class ClaimGovernance:
    risk_assessment: RiskAssessment | None = None
    responsibility_assignment: ResponsibilityAssignment | None = None
    decision_trace_package: DecisionTracePackage | None = None


class TransitionAuthority:
    """Deterministic claim transition gate over imported CairnLab objects."""

    def __init__(
        self,
        claims: dict[str, Claim],
        evidence: dict[str, EvidenceItem],
        graph: RelationGraph,
        projection: EventProjection,
        cases: Iterable[ClaimCase] = (),
    ):
        self.claims = claims
        self.evidence = evidence
        self.graph = graph
        self.projection = projection
        self.governance = self._index_governance(claims.values(), cases)

    def request_transition(
        self,
        claim_id: str,
        target_state: ClaimState,
        actor: Actor,
        reason: str,
        force: bool = False,
    ) -> TransitionDecision:
        claim = self.claims.get(claim_id)
        current_state = self._current_state(claim_id)
        blocking_reasons: list[str] = []
        required_actions: list[str] = []

        def block(reason_code: str, action: str) -> None:
            if reason_code not in blocking_reasons:
                blocking_reasons.append(reason_code)
            if action not in required_actions:
                required_actions.append(action)

        if claim is None:
            block("claim_not_found", "import or create the claim before requesting a transition")
        else:
            if force and not self._has_override_authority(actor):
                block(
                    "missing_override_authority",
                    "force override requires an authorized actor with release override authority",
                )
            if target_state == ClaimState.VERIFIED:
                self._check_verified_requirements(claim_id, block)

            target_state_value = enum_value(target_state)
            if target_state_value in CONSEQUENTIAL_STATES and not self._has_risk_assessment(claim_id):
                block("missing_risk_assessment", "record RiskAssessment before a consequential transition")

            if target_state == ClaimState.RELEASED:
                if current_state != ClaimState.VERIFIED and not force:
                    block("release_requires_verified_state", "verify claim before release or record explicit override")
                self._check_verified_requirements(claim_id, block)
                self._check_released_requirements(claim, claim_id, force, block)

        event_type = EventType.TRANSITION_BLOCKED if blocking_reasons else EventType.TRANSITION_ALLOWED
        payload: dict[str, object] = {
            "blocking_reasons": blocking_reasons,
            "force": force,
        }
        if force:
            payload["override"] = {
                "actor": actor.id,
                "role": actor.role,
                "authority": actor.authority,
                "liability_scope": [
                    "release_requires_verified_state",
                    "unresolved_material_dissent",
                ],
                "rationale": reason,
            }
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
            payload=payload,
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

    def _check_verified_requirements(self, claim_id: str, block) -> None:
        if not self._has_machine_addressable_evidence(claim_id):
            block("missing_machine_addressable_evidence", "attach evidence with stable URI or hash")
        for reason in self._verifier_certificate_blocking_reasons(claim_id):
            block(reason, self._verifier_certificate_required_action(reason))

    def _check_released_requirements(self, claim: Claim, claim_id: str, force: bool, block) -> None:
        if not self._has_valid_human_gate(claim_id):
            block("missing_human_gate", "record human gate with actor, authority, scope, and rationale")
        if self._has_unresolved_material_dissent(claim_id) and not force:
            block("unresolved_material_dissent", "resolve dissent through verifier or explicit human override")
        if not self._has_accountable_party(claim_id):
            block("missing_accountable_party", "record ResponsibilityAssignment with an accountable party")
        if self._is_high_impact(claim_id, claim) and not self._has_decision_trace_package(claim_id):
            block("missing_decision_trace_package", "record DecisionTracePackage for high-impact release")

    def _current_state(self, claim_id: str) -> ClaimState:
        current_state_value = self.projection.claim_state(claim_id)
        return ClaimState(current_state_value) if current_state_value else ClaimState.DRAFT

    def _has_machine_addressable_evidence(self, claim_id: str) -> bool:
        for item in self._incoming_evidence(claim_id):
            if enum_value(item.type) in NON_MACHINE_EVIDENCE_TYPES:
                continue
            if self._is_invalid_evidence(item.id):
                continue
            if item.uri or item.hash:
                return True
        return False

    def _verifier_certificate_blocking_reasons(self, claim_id: str) -> list[str]:
        reasons: list[str] = []
        saw_candidate = False
        for relation in self.graph.incoming(claim_id):
            if enum_value(relation.type) != RelationType.VERIFIED_BY.value:
                continue
            item = self.evidence.get(relation.source)
            if not item:
                continue
            if enum_value(item.type) != EvidenceType.VERIFIER_CERTIFICATE.value:
                continue
            saw_candidate = True
            if self._is_invalid_evidence(item.id):
                self._append_reason(reasons, "verifier_certificate_inputs_invalidated")
                continue
            try:
                certificate = VerifierCertificate.model_validate(item.metadata or {})
            except Exception:
                self._append_reason(reasons, "malformed_verifier_certificate")
                continue
            if certificate.claim != claim_id:
                self._append_reason(reasons, "verifier_certificate_claim_mismatch")
                continue
            if certificate.status != "pass":
                self._append_reason(reasons, "missing_passing_verifier_certificate")
                continue
            if not any(transition in certificate.can_authorize for transition in VERIFIER_AUTHORIZED_TRANSITIONS):
                self._append_reason(reasons, "verifier_certificate_not_authorized_for_transition")
                continue
            missing_inputs = [input_id for input_id in certificate.inputs if input_id not in self.evidence]
            if missing_inputs:
                self._append_reason(reasons, "verifier_certificate_inputs_missing")
                continue
            invalid_inputs = [
                input_id
                for input_id in certificate.inputs
                if input_id in self.evidence and self._is_invalid_evidence(input_id)
            ]
            if invalid_inputs:
                self._append_reason(reasons, "verifier_certificate_inputs_invalidated")
                continue
            return []
        if not saw_candidate:
            return ["missing_passing_verifier_certificate"]
        return reasons or ["missing_passing_verifier_certificate"]

    def _has_valid_human_gate(self, claim_id: str) -> bool:
        for item in self._incoming_evidence(claim_id):
            if enum_value(item.type) != EvidenceType.HUMAN_GATE.value:
                continue
            if self._is_invalid_evidence(item.id):
                continue
            metadata = item.metadata or {}
            if metadata.get("actor") and metadata.get("authority") and metadata.get("scope") and metadata.get("rationale"):
                return True
        return False

    def _has_unresolved_material_dissent(self, claim_id: str) -> bool:
        for item in self._incoming_evidence(claim_id):
            if enum_value(item.type) != EvidenceType.MATERIAL_DISSENT.value:
                continue
            if self._is_invalid_evidence(item.id):
                continue
            metadata = item.metadata or {}
            if metadata.get("severity") == "material" and not metadata.get("resolved", False):
                return True
        return False

    def _has_risk_assessment(self, claim_id: str) -> bool:
        return self.governance.get(claim_id, ClaimGovernance()).risk_assessment is not None

    def _has_accountable_party(self, claim_id: str) -> bool:
        assignment = self.governance.get(claim_id, ClaimGovernance()).responsibility_assignment
        return bool(assignment and assignment.accountable)

    def _has_decision_trace_package(self, claim_id: str) -> bool:
        return self.governance.get(claim_id, ClaimGovernance()).decision_trace_package is not None

    def _is_high_impact(self, claim_id: str, claim: Claim) -> bool:
        governance = self.governance.get(claim_id, ClaimGovernance())
        if enum_value(claim.risk) in HIGH_IMPACT_RISK_TIERS:
            return True
        if claim.lifecycle_context and claim.lifecycle_context.system_scope == "high_impact_decision":
            return True
        if governance.risk_assessment and governance.risk_assessment.risk_tier in HIGH_IMPACT_RISK_TIERS:
            return True
        return False

    def _incoming_evidence(self, claim_id: str) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for relation in self.graph.incoming(claim_id):
            item = self.evidence.get(relation.source)
            if item:
                items.append(item)
        return items

    def _is_invalid_evidence(self, evidence_id: str) -> bool:
        return self.projection.evidence_status(evidence_id) in INVALID_EVIDENCE_STATES

    def _has_override_authority(self, actor: Actor) -> bool:
        return bool(actor.authority in OVERRIDE_AUTHORITIES)

    def _append_reason(self, reasons: list[str], reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    def _verifier_certificate_required_action(self, reason: str) -> str:
        actions = {
            "malformed_verifier_certificate": "attach a structured VerifierCertificate payload",
            "verifier_certificate_claim_mismatch": "attach a certificate bound to this claim",
            "verifier_certificate_inputs_missing": "attach all evidence inputs referenced by the certificate",
            "verifier_certificate_inputs_invalidated": "refresh invalidated certificate inputs and rerun verifier",
            "verifier_certificate_not_authorized_for_transition": "use a verifier certificate authorized for verification",
            "missing_passing_verifier_certificate": "attach a passing verifier certificate",
        }
        return actions.get(reason, "attach a passing verifier certificate")

    def _index_governance(
        self,
        claims: Iterable[Claim],
        cases: Iterable[ClaimCase],
    ) -> dict[str, ClaimGovernance]:
        index: dict[str, ClaimGovernance] = {}

        def merge(
            claim_id: str,
            *,
            risk_assessment: RiskAssessment | None = None,
            responsibility_assignment: ResponsibilityAssignment | None = None,
            decision_trace_package: DecisionTracePackage | None = None,
        ) -> None:
            current = index.get(claim_id, ClaimGovernance())
            index[claim_id] = ClaimGovernance(
                risk_assessment=risk_assessment or current.risk_assessment,
                responsibility_assignment=responsibility_assignment or current.responsibility_assignment,
                decision_trace_package=decision_trace_package or current.decision_trace_package,
            )

        for claim in claims:
            merge(
                claim.id,
                risk_assessment=claim.risk_assessment,
                responsibility_assignment=claim.responsibility_assignment,
                decision_trace_package=claim.decision_trace_package,
            )

        for case in cases:
            for risk_assessment in case.risk_assessments:
                merge(risk_assessment.object, risk_assessment=risk_assessment)
            for assignment in case.responsibility_assignments:
                merge(assignment.object, responsibility_assignment=assignment)
            for package in case.decision_trace_packages:
                merge(package.claim, decision_trace_package=package)

        return index
