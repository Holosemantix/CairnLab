from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..builder import ClaimCaseBuilder
from ..models import DecisionTracePackage, ResponsibilityAssignment, RiskAssessment, VerifierCertificate
from .base import AdapterDiagnostic, AdapterExportResult


class ExternalRunManifestAdapter:
    """Translate a generic external-stage manifest into a CairnLab ClaimCase.

    This adapter is for tools that CairnLab should not know about directly:
    idea generators, paper-to-code systems, reviewers, experiment runners, and
    verifier wrappers can all emit the same small manifest.
    """

    name = "external-run-manifest"
    manifest_type = "cairn.external_run.v1"
    manifest_names = (
        "cairn_external_run.yaml",
        "cairn_external_run.yml",
        "cairn_external_run.json",
        "external_run_manifest.yaml",
        "external_run_manifest.yml",
        "external_run_manifest.json",
        ".cairn/external_run_manifest.yaml",
        ".cairn/external_run_manifest.yml",
        ".cairn/external_run_manifest.json",
    )

    def detect(self, path: Path) -> bool:
        manifest_path = self._manifest_path(Path(path))
        if manifest_path is None:
            return False
        manifest = self._read_mapping(manifest_path, diagnostics=[])
        return bool(manifest and manifest.get("manifest_type") == self.manifest_type)

    def export_case(self, path: Path) -> AdapterExportResult:
        root = Path(path)
        manifest_path = self._manifest_path(root)
        if manifest_path is None:
            raise FileNotFoundError(f"No external run manifest found under {root}")

        diagnostics: list[AdapterDiagnostic] = []
        manifest = self._read_mapping(manifest_path, diagnostics)
        if not manifest:
            raise ValueError(f"External run manifest is empty or invalid: {manifest_path}")
        if manifest.get("manifest_type") != self.manifest_type:
            raise ValueError(f"External run manifest must set manifest_type={self.manifest_type!r}")

        project_root = manifest_path.parent if root.is_file() else root
        source_system = str(manifest.get("source_system") or "external")
        stage = str(manifest.get("stage") or "external_stage")
        builder = ClaimCaseBuilder(
            case_id=str(manifest.get("case_id") or f"external-run:{project_root.name or 'project'}"),
            source_system=source_system,
            source_task=manifest.get("source_task"),
            stress_scenario=str(manifest.get("stress_scenario") or f"external_stage:{stage}"),
        )
        builder.set_native_behavior(
            can_identify_downstream_claims=manifest.get("can_identify_downstream_claims", "unknown"),
            can_invalidate_claims=manifest.get("can_invalidate_claims", "unknown"),
            can_downgrade_or_retract_released_claims=manifest.get("can_downgrade_or_retract_released_claims", "unknown"),
            can_preserve_old_history_append_only=bool(manifest.get("can_preserve_old_history_append_only", False)),
            can_require_reapproval=manifest.get("can_require_reapproval", "unknown"),
            can_export_machine_readable_revert_trace=bool(manifest.get("can_export_machine_readable_revert_trace", True)),
            external_stage=stage,
            adapter=self.name,
        )
        builder.native_system_behavior.update(self._mapping(manifest.get("native_system_behavior")))
        builder.expected_cairnlab_behavior.update(
            {
                "external_tools_provide_evidence_not_authority": True,
                "require_transition_authority_for_release": True,
            }
        )
        builder.expected_cairnlab_behavior.update(self._mapping(manifest.get("expected_cairnlab_behavior")))

        self._add_claims(builder, manifest, diagnostics)
        self._add_stages(builder, manifest, project_root, manifest_path)
        self._add_evidence(builder, manifest, project_root, manifest_path, diagnostics)
        self._add_verifier_certificates(builder, manifest, diagnostics)
        self._add_reviewers(builder, manifest, project_root, manifest_path, diagnostics)
        self._add_material_dissent(builder, manifest, project_root, manifest_path, diagnostics)
        self._add_human_gates(builder, manifest, diagnostics)
        self._add_release_decisions(builder, manifest, diagnostics)
        self._add_governance(builder, manifest, diagnostics)
        self._add_relations(builder, manifest, diagnostics)

        for failure_class in manifest.get("failure_classes") or []:
            if isinstance(failure_class, str):
                builder.add_failure_class(failure_class)

        if not builder.claims:
            diagnostics.append(AdapterDiagnostic(level="warning", message="External run manifest contains no claims.", path=str(manifest_path)))
        if not builder.evidence:
            diagnostics.append(AdapterDiagnostic(level="warning", message="External run manifest contains no evidence.", path=str(manifest_path)))

        return AdapterExportResult(case=builder.build(), diagnostics=diagnostics)

    def _add_claims(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, claim in enumerate(self._list_of_mappings(manifest.get("claims")), start=1):
            claim_id = claim.get("id")
            if not claim_id:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping claim without id at claims[{index}]."))
                continue
            builder.add_claim(
                str(claim_id),
                str(claim.get("text") or claim.get("claim") or claim_id),
                claim_type=str(claim.get("type") or claim.get("claim_type") or "empirical_metric"),
                state=str(claim.get("state") or claim.get("status") or "draft"),
                scope=self._mapping(claim.get("scope")),
                risk=str(claim.get("risk") or "medium"),
                metadata=self._mapping(claim.get("metadata")),
            )

    def _add_stages(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        project_root: Path,
        manifest_path: Path,
    ) -> None:
        for index, stage in enumerate(self._list_of_mappings(manifest.get("stages")), start=1):
            phase = str(stage.get("phase") or stage.get("name") or f"stage_{index}")
            stage_id = str(stage.get("id") or f"run:{self._slug(phase)}")
            uri, digest = self._uri_and_hash(project_root, stage, manifest_path)
            metadata = {
                "adapter": self.name,
                "source_system": manifest.get("source_system"),
                "phase": phase,
                "tool": stage.get("tool"),
                "status": stage.get("status"),
                "inputs": stage.get("inputs") or [],
                "outputs": stage.get("outputs") or [],
                **self._mapping(stage.get("metadata")),
            }
            builder.add_evidence(
                stage_id,
                str(stage.get("type") or stage.get("evidence_type") or "run"),
                uri=uri,
                hash=str(stage.get("hash") or stage.get("sha256") or digest) if stage.get("hash") or stage.get("sha256") or digest else None,
                status=str(stage.get("evidence_status") or "valid"),
                metadata=metadata,
            )

    def _add_evidence(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        project_root: Path,
        manifest_path: Path,
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, evidence in enumerate(self._list_of_mappings(manifest.get("evidence")), start=1):
            evidence_id = evidence.get("id")
            if not evidence_id:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping evidence without id at evidence[{index}]."))
                continue
            uri, digest = self._uri_and_hash(project_root, evidence, manifest_path)
            builder.add_evidence(
                str(evidence_id),
                str(evidence.get("type") or evidence.get("evidence_type") or "artifact"),
                uri=uri,
                hash=str(evidence.get("hash") or evidence.get("sha256") or digest) if evidence.get("hash") or evidence.get("sha256") or digest else None,
                status=str(evidence.get("status") or "valid"),
                metadata={
                    "adapter": self.name,
                    "source_system": manifest.get("source_system"),
                    **self._mapping(evidence.get("metadata")),
                },
            )

    def _add_verifier_certificates(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, certificate in enumerate(self._list_of_mappings(manifest.get("verifier_certificates")), start=1):
            try:
                verifier = VerifierCertificate.model_validate(certificate)
            except Exception as exc:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping invalid verifier certificate at verifier_certificates[{index}]: {exc}"))
                continue
            builder.add_verifier_certificate(
                verifier,
                criticality=str(certificate.get("criticality") or "supporting"),
            )

    def _add_reviewers(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        project_root: Path,
        manifest_path: Path,
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, review in enumerate(self._list_of_mappings(manifest.get("reviewer_verdicts")), start=1):
            review_id = review.get("id")
            if not review_id:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping reviewer verdict without id at reviewer_verdicts[{index}]."))
                continue
            uri, digest = self._uri_and_hash(project_root, review, manifest_path)
            builder.add_evidence(
                str(review_id),
                "reviewer_verdict",
                uri=uri,
                hash=str(review.get("hash") or review.get("sha256") or digest) if review.get("hash") or review.get("sha256") or digest else None,
                metadata={
                    "adapter": self.name,
                    "source_system": manifest.get("source_system"),
                    "reviewer": review.get("reviewer"),
                    "verdict": review.get("verdict") or review.get("status"),
                    **self._mapping(review.get("metadata")),
                    "not_transition_authority": True,
                },
            )
            claim_id = review.get("claim") or review.get("claim_id")
            if claim_id:
                builder.add_relation(str(review_id), str(claim_id), str(review.get("relation_type") or "depends_on"), criticality=str(review.get("criticality") or "contextual"))

    def _add_material_dissent(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        project_root: Path,
        manifest_path: Path,
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, dissent in enumerate(self._list_of_mappings(manifest.get("material_dissent")), start=1):
            dissent_id = dissent.get("id")
            claim_id = dissent.get("claim") or dissent.get("claim_id")
            if not dissent_id or not claim_id:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping material dissent without id or claim at material_dissent[{index}]."))
                continue
            uri, digest = self._uri_and_hash(project_root, dissent, manifest_path)
            builder.add_evidence(
                str(dissent_id),
                "material_dissent",
                uri=uri,
                hash=str(dissent.get("hash") or dissent.get("sha256") or digest) if dissent.get("hash") or dissent.get("sha256") or digest else None,
                metadata={
                    "adapter": self.name,
                    "source_system": manifest.get("source_system"),
                    "severity": dissent.get("severity") or "material",
                    "resolved": bool(dissent.get("resolved", False)),
                    "summary": dissent.get("summary") or dissent.get("reason"),
                    **self._mapping(dissent.get("metadata")),
                },
            )
            builder.add_relation(str(dissent_id), str(claim_id), "challenges", criticality="critical")
            if dissent.get("severity", "material") == "material" and not dissent.get("resolved", False):
                builder.add_failure_class("external_material_dissent")

    def _add_human_gates(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, gate in enumerate(self._list_of_mappings(manifest.get("human_gates")), start=1):
            claim_id = gate.get("claim") or gate.get("claim_id")
            actor = gate.get("actor")
            authority = gate.get("authority")
            rationale = gate.get("rationale")
            if not claim_id or not actor or not authority or not rationale:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping incomplete human gate at human_gates[{index}]."))
                continue
            builder.add_human_gate(
                str(gate.get("id") or f"human_gate:{self._slug(str(claim_id))}"),
                str(claim_id),
                str(actor),
                authority=str(authority),
                scope=self._mapping(gate.get("scope")) or {"claim": str(claim_id)},
                rationale=str(rationale),
            )

    def _add_release_decisions(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, decision in enumerate(self._list_of_mappings(manifest.get("release_decisions")), start=1):
            claim_id = decision.get("claim") or decision.get("claim_id")
            actor = decision.get("actor")
            if not claim_id or not actor:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping incomplete release decision at release_decisions[{index}]."))
                continue
            builder.add_release_decision(
                str(decision.get("id") or f"release_decision:{self._slug(str(claim_id))}"),
                str(claim_id),
                str(actor),
                decision=str(decision.get("decision") or "allow"),
            )

    def _add_governance(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, payload in enumerate(self._list_of_mappings(manifest.get("risk_assessments")), start=1):
            try:
                builder.risk_assessments.append(RiskAssessment.model_validate(payload))
            except Exception as exc:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping invalid risk assessment at risk_assessments[{index}]: {exc}"))
        for index, payload in enumerate(self._list_of_mappings(manifest.get("responsibility_assignments")), start=1):
            try:
                builder.responsibility_assignments.append(ResponsibilityAssignment.model_validate(payload))
            except Exception as exc:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping invalid responsibility assignment at responsibility_assignments[{index}]: {exc}"))
        for index, payload in enumerate(self._list_of_mappings(manifest.get("decision_trace_packages")), start=1):
            try:
                builder.decision_trace_packages.append(DecisionTracePackage.model_validate(payload))
            except Exception as exc:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping invalid decision trace package at decision_trace_packages[{index}]: {exc}"))

    def _add_relations(
        self,
        builder: ClaimCaseBuilder,
        manifest: dict[str, Any],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for index, relation in enumerate(self._list_of_mappings(manifest.get("relations")), start=1):
            source = relation.get("source") or relation.get("from")
            target = relation.get("target") or relation.get("to")
            if not source or not target:
                diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping relation without source or target at relations[{index}]."))
                continue
            builder.add_relation(
                str(source),
                str(target),
                str(relation.get("type") or relation.get("relation_type") or "depends_on"),
                relation_id=str(relation.get("id")) if relation.get("id") else None,
                criticality=str(relation.get("criticality") or "supporting"),
            )

    def _manifest_path(self, root: Path) -> Path | None:
        if root.is_file():
            return root if root.suffix.lower() in {".json", ".yaml", ".yml"} else None
        for name in self.manifest_names:
            path = root / name
            if path.exists():
                return path
        return None

    def _read_mapping(self, path: Path, diagnostics: list[AdapterDiagnostic]) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            diagnostics.append(AdapterDiagnostic(level="warning", message=f"Skipping unreadable external run manifest: {exc}", path=str(path)))
            return None
        if not isinstance(data, dict):
            diagnostics.append(AdapterDiagnostic(level="warning", message="External run manifest is not a mapping.", path=str(path)))
            return None
        return data

    def _uri_and_hash(self, root: Path, payload: dict[str, Any], manifest_path: Path) -> tuple[str | None, str | None]:
        del manifest_path
        uri = payload.get("uri")
        path_value = payload.get("path")
        if not path_value:
            return str(uri) if uri else None, None
        path = Path(str(path_value))
        path = path if path.is_absolute() else root / path
        if not path.exists():
            return str(uri) if uri else None, None
        try:
            relative_uri = path.relative_to(root).as_posix()
        except ValueError:
            relative_uri = path.resolve().as_uri()
        return str(uri or relative_uri), self._file_sha256(path)

    def _list_of_mappings(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _mapping(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"sha256:{digest}"

    def _slug(self, value: str) -> str:
        return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("_") or "external"
