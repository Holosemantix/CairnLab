from __future__ import annotations

from datetime import datetime, timezone
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CairnModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClaimState(StrEnum):
    DRAFT = "draft"
    SCOPED = "scoped"
    NEEDS_EVIDENCE = "needs_evidence"
    EVIDENCE_REQUIRED = "evidence_required"
    EVIDENCE_ATTACHED = "evidence_attached"
    VERIFICATION_PENDING = "verification_pending"
    VERIFIED = "verified"
    BLOCKED = "blocked"
    CHALLENGED = "challenged"
    HUMAN_ACCEPTED = "human_accepted"
    RELEASED = "released"
    DOWNGRADED = "downgraded"
    RETRACTED = "retracted"
    INVALIDATED = "invalidated"
    STALE = "stale"


class ClaimType(StrEnum):
    DRAFT = "draft"
    EMPIRICAL_SCORE = "empirical_score"
    EMPIRICAL_METRIC = "empirical_metric"
    EMPIRICAL_COMPARISON = "empirical_comparison"
    LITERATURE_CLAIM = "literature_claim"
    METHOD_CLAIM = "method_claim"
    NOVELTY = "novelty"
    ROBUSTNESS = "robustness"
    LIMITATION = "limitation"
    SAFETY_CLAIM = "safety_claim"
    CITATION = "citation"


class EvidenceType(StrEnum):
    RUN = "run"
    ARTIFACT = "artifact"
    METRIC = "metric"
    DATASET = "dataset"
    CODE_COMMIT = "code_commit"
    CITATION = "citation"
    PAPER_SECTION = "paper_section"
    VERIFIER_CERTIFICATE = "verifier_certificate"
    REVIEWER_VERDICT = "reviewer_verdict"
    HUMAN_GATE = "human_gate"
    RELEASE_DECISION = "release_decision"
    MATERIAL_DISSENT = "material_dissent"
    RESPONSIBILITY_ASSIGNMENT = "responsibility_assignment"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION_TRACE_PACKAGE = "decision_trace_package"


class EvidenceStatus(StrEnum):
    VALID = "valid"
    INVALIDATED = "invalidated"
    STALE = "stale"
    UNKNOWN = "unknown"


class RelationType(StrEnum):
    SUPPORTS = "supports"
    COMPUTED_FROM = "computed_from"
    COMPUTED = "computed"
    CONTAINED_IN = "contained_in"
    APPROVED_BY = "approved_by"
    VERIFIED_BY = "verified_by"
    RELEASED_BY = "released_by"
    DEPENDS_ON = "depends_on"
    CHALLENGES = "challenges"
    SUPERSEDES = "supersedes"
    REQUIRES = "requires"


class Criticality(StrEnum):
    CRITICAL = "critical"
    SUPPORTING = "supporting"
    CONTEXTUAL = "contextual"
    WEAK = "weak"


class EventType(StrEnum):
    REVERT_REQUESTED = "RevertRequested"
    EVIDENCE_INVALIDATED = "EvidenceInvalidated"
    METRIC_INVALIDATED = "MetricInvalidated"
    ARTIFACT_INVALIDATED = "ArtifactInvalidated"
    DATASET_INVALIDATED = "DatasetInvalidated"
    CODE_INVALIDATED = "CodeInvalidated"
    CITATION_INVALIDATED = "CitationInvalidated"
    CLAIM_CHALLENGED = "ClaimChallenged"
    CLAIM_DOWNGRADED = "ClaimDowngraded"
    CLAIM_RETRACTED = "ClaimRetracted"
    CLAIM_INVALIDATED = "ClaimInvalidated"
    OBJECT_MARKED_STALE = "ObjectMarkedStale"
    PAPER_SECTION_MARKED_STALE = "PaperSectionMarkedStale"
    VERIFIER_CERTIFICATE_INVALIDATED = "VerifierCertificateInvalidated"
    HUMAN_REAPPROVAL_REQUIRED = "HumanReapprovalRequired"
    RELEASE_DECISION_REOPENED = "ReleaseDecisionReopened"
    TRANSITION_REQUESTED = "TransitionRequested"
    TRANSITION_ALLOWED = "TransitionAllowed"
    TRANSITION_BLOCKED = "TransitionBlocked"


class AffectedAction(StrEnum):
    INVALIDATE = "invalidate"
    CHALLENGE = "challenge"
    DOWNGRADE = "downgrade"
    RETRACT_CANDIDATE = "retract_candidate"
    MARK_STALE = "mark_stale"
    REQUIRE_REAPPROVAL = "require_reapproval"
    REOPEN_RELEASE_DECISION = "reopen_release_decision"
    NEEDS_REVIEW = "needs_review"
    NO_ACTION = "no_action"


class Actor(CairnModel):
    id: str
    role: str = "unknown"
    authority: str | None = None


class LifecycleContext(CairnModel):
    stage: str
    system_scope: str = "local_research"
    autonomy_level: str = "assistive"
    affected_parties: list[str] = Field(default_factory=list)


class ResponsibilityEntry(CairnModel):
    role: str
    actor: str


class ResponsibilityAssignment(CairnModel):
    object: str
    action: str
    responsible: list[ResponsibilityEntry] = Field(default_factory=list)
    accountable: list[ResponsibilityEntry] = Field(default_factory=list)
    consulted: list[ResponsibilityEntry] = Field(default_factory=list)
    informed: list[ResponsibilityEntry] = Field(default_factory=list)


class RiskAssessment(CairnModel):
    object: str
    risk_tier: Literal["low", "medium", "high", "critical"] = "medium"
    dimensions: dict[str, Any] = Field(default_factory=dict)
    controls: list[str] = Field(default_factory=list)
    identified_risks: list[str] = Field(default_factory=list)


class GovernancePolicy(CairnModel):
    id: str
    require_human_gate_for: list[str] = Field(default_factory=list)
    block_if: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)


class DecisionTracePackage(CairnModel):
    id: str
    claim: str
    transition: str | None = None
    includes: list[str] = Field(default_factory=list)
    export_hash: str | None = None


class VerifierCertificate(CairnModel):
    id: str
    verifier: str
    claim: str
    status: Literal["pass", "fail", "error"]
    inputs: list[str] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    can_authorize: list[str] = Field(default_factory=list)
    reason: str | None = None


class VerificationRequest(CairnModel):
    claim_id: str
    evidence: list["EvidenceItem"] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class Claim(CairnModel):
    id: str
    text: str
    type: ClaimType | str
    state: ClaimState = ClaimState.DRAFT
    scope: dict[str, Any] = Field(default_factory=dict)
    risk: Literal["low", "medium", "high", "critical"] = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)
    lifecycle_context: LifecycleContext | None = None
    risk_assessment: RiskAssessment | None = None
    responsibility_assignment: ResponsibilityAssignment | None = None
    decision_trace_package: DecisionTracePackage | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_status_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "state" not in data and "status" in data:
            data = dict(data)
            data["state"] = data["status"]
        return data


class EvidenceItem(CairnModel):
    id: str
    type: EvidenceType | str
    uri: str | None = None
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(CairnModel):
    id: str
    source: str
    target: str
    type: RelationType | str
    criticality: Criticality = Criticality.SUPPORTING


class TransitionEvent(CairnModel):
    id: str
    type: EventType | str
    target: str
    actor: Actor
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    previous_state: str | None = None
    new_state: str | None = None
    caused_by: str | None = None
    input_hash: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TransitionDecision(CairnModel):
    decision: Literal["allowed", "blocked"]
    claim_id: str
    requested_state: ClaimState
    current_state: ClaimState
    blocking_reasons: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    events: list[TransitionEvent] = Field(default_factory=list)


class RevertRequest(CairnModel):
    target: str
    reason: str
    mode: str = "invalidate_only"
    actor: Actor = Field(default_factory=lambda: Actor(id="user:unknown"))


class AffectedObject(CairnModel):
    id: str
    action: AffectedAction
    reason: str
    relation_path: list[str] = Field(default_factory=list)
    proposed_event_type: EventType | str | None = None
    previous_state: str | None = None
    new_state: str | None = None


class RevertPlan(CairnModel):
    request: RevertRequest
    affected: list[AffectedObject] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)


class ClaimCase(CairnModel):
    case_id: str
    source_system: str
    source_task: str | None = None
    stress_scenario: str
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    native_system_behavior: dict[str, Any] = Field(default_factory=dict)
    expected_cairnlab_behavior: dict[str, Any] = Field(default_factory=dict)
    failure_classes: list[str] = Field(default_factory=list)
    lifecycle_contexts: list[LifecycleContext] = Field(default_factory=list)
    risk_assessments: list[RiskAssessment] = Field(default_factory=list)
    responsibility_assignments: list[ResponsibilityAssignment] = Field(default_factory=list)
    governance_policies: list[GovernancePolicy] = Field(default_factory=list)
    decision_trace_packages: list[DecisionTracePackage] = Field(default_factory=list)


class ImportResult(CairnModel):
    case_id: str
    claims: int
    evidence: int
    relations: int


class TraceResult(CairnModel):
    object_id: str
    object: dict[str, Any] | None = None
    projected_state: str | None = None
    incoming_relations: list[Relation] = Field(default_factory=list)
    outgoing_relations: list[Relation] = Field(default_factory=list)
    events: list[TransitionEvent] = Field(default_factory=list)
    downstream_objects: list[str] = Field(default_factory=list)


class ValidationReport(CairnModel):
    cases_sampled: int
    systems_sampled: int
    claims_sampled: int
    stress_cases: int
    failure_class_counts: dict[str, int] = Field(default_factory=dict)
    recommendation: Literal["go", "no_go", "continue_sampling"] = "continue_sampling"
    reasons: list[str] = Field(default_factory=list)
