from __future__ import annotations

from typing import Any, Protocol

from .models import (
    EvidenceItem,
    EvidenceType,
    VerificationRequest,
    VerifierCertificate,
)
from .utils import enum_value, stable_hash


class Verifier(Protocol):
    """Deterministic verifier that emits a machine-readable certificate."""

    name: str
    version: str

    def verify(self, request: VerificationRequest) -> VerifierCertificate:
        """Return a certificate for the requested claim and evidence."""


class ArtifactHashVerifier:
    name = "artifact_hash"
    version = "0.1.0"

    def verify(self, request: VerificationRequest) -> VerifierCertificate:
        evidence, error = _select_evidence(request.evidence, request.parameters.get("evidence_id"))
        expected_hash = request.parameters.get("expected_hash")
        if error:
            return self._certificate(request, "error", [], {"error": error}, error)
        if not expected_hash:
            return self._certificate(request, "error", [evidence.id], {"error": "missing_expected_hash"}, "missing_expected_hash")

        observed_hash = evidence.hash
        status = "pass" if observed_hash == expected_hash else "fail"
        return self._certificate(
            request,
            status,
            [evidence.id],
            {
                "expected_hash": expected_hash,
                "observed_hash": observed_hash,
            },
            None if status == "pass" else "artifact_hash_mismatch",
        )

    def _certificate(
        self,
        request: VerificationRequest,
        status: str,
        inputs: list[str],
        result: dict[str, Any],
        reason: str | None,
    ) -> VerifierCertificate:
        return _certificate(
            verifier=f"{self.name}@{self.version}",
            claim_id=request.claim_id,
            status=status,
            inputs=inputs,
            result=result,
            reason=reason,
            can_authorize=["evidence_attached -> verified"] if status == "pass" else [],
        )


class MetricThresholdVerifier:
    name = "metric_threshold"
    version = "0.1.0"

    def verify(self, request: VerificationRequest) -> VerifierCertificate:
        evidence, error = _select_metric(request.evidence, request.parameters)
        if error:
            return self._certificate(request, "error", [], {"error": error}, error)

        value = _metric_value(evidence)
        min_value = request.parameters.get("min_value")
        max_value = request.parameters.get("max_value")
        if min_value is None and max_value is None:
            return self._certificate(request, "error", [evidence.id], {"error": "missing_threshold"}, "missing_threshold")

        try:
            numeric_value = float(value)
            min_ok = min_value is None or numeric_value >= float(min_value)
            max_ok = max_value is None or numeric_value <= float(max_value)
        except (TypeError, ValueError):
            return self._certificate(
                request,
                "fail",
                [evidence.id],
                {
                    "observed_value": value,
                    "min_value": min_value,
                    "max_value": max_value,
                },
                "metric_value_not_numeric",
            )

        status = "pass" if min_ok and max_ok else "fail"
        return self._certificate(
            request,
            status,
            [evidence.id],
            {
                "observed_value": numeric_value,
                "min_value": min_value,
                "max_value": max_value,
            },
            None if status == "pass" else "metric_threshold_failed",
        )

    def _certificate(
        self,
        request: VerificationRequest,
        status: str,
        inputs: list[str],
        result: dict[str, Any],
        reason: str | None,
    ) -> VerifierCertificate:
        return _certificate(
            verifier=f"{self.name}@{self.version}",
            claim_id=request.claim_id,
            status=status,
            inputs=inputs,
            result=result,
            reason=reason,
            can_authorize=["evidence_attached -> verified"] if status == "pass" else [],
        )


def _certificate(
    *,
    verifier: str,
    claim_id: str,
    status: str,
    inputs: list[str],
    result: dict[str, Any],
    reason: str | None,
    can_authorize: list[str],
) -> VerifierCertificate:
    payload = {
        "verifier": verifier,
        "claim": claim_id,
        "status": status,
        "inputs": inputs,
        "result": result,
        "reason": reason,
        "can_authorize": can_authorize,
    }
    certificate_id = f"verifier:{verifier.split('@', 1)[0]}:{stable_hash(payload).split(':', 1)[1][:12]}"
    return VerifierCertificate(
        id=certificate_id,
        verifier=verifier,
        claim=claim_id,
        status=status,
        inputs=inputs,
        result=result,
        reason=reason,
        can_authorize=can_authorize,
    )


def _select_evidence(evidence: list[EvidenceItem], evidence_id: Any) -> tuple[EvidenceItem, str | None]:
    if evidence_id:
        for item in evidence:
            if item.id == evidence_id:
                return item, None
        return EvidenceItem(id="missing", type="artifact"), f"evidence_not_found:{evidence_id}"
    if len(evidence) == 1:
        return evidence[0], None
    return EvidenceItem(id="missing", type="artifact"), "evidence_id_required"


def _select_metric(evidence: list[EvidenceItem], parameters: dict[str, Any]) -> tuple[EvidenceItem, str | None]:
    evidence_id = parameters.get("evidence_id")
    metric_name = parameters.get("metric_name")
    candidates = [
        item
        for item in evidence
        if enum_value(item.type) == EvidenceType.METRIC.value
        and (not evidence_id or item.id == evidence_id)
        and (not metric_name or item.metadata.get("metric_name") == metric_name)
    ]
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return EvidenceItem(id="missing", type="metric"), "metric_evidence_not_found"
    return EvidenceItem(id="ambiguous", type="metric"), "metric_evidence_ambiguous"


def _metric_value(evidence: EvidenceItem) -> Any:
    metadata = evidence.metadata or {}
    if "value" in metadata:
        return metadata["value"]
    return metadata.get("observed_value")
