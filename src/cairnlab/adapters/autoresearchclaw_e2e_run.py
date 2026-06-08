from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..models import Claim, ClaimCase, Criticality, EvidenceItem, Relation, RelationType
from .autoresearchclaw_manifest import AutoResearchClawManifestAdapter
from .base import AdapterDiagnostic, AdapterExportResult


class AutoResearchClawE2ERunAdapter:
    """Translate an AutoResearchClaw e2e run directory into a ClaimCase.

    The resolver is intentionally thin: it finds the result-analysis stage from
    a real AutoResearchClaw e2e run, delegates base claim/evidence mapping to
    the manifest adapter, and adds stage-level diagnostics as evidence context.
    It never imports or executes AutoResearchClaw.
    """

    name = "autoresearchclaw-e2e-run"
    result_stage = "stage-14"

    def __init__(self, manifest_adapter: AutoResearchClawManifestAdapter | None = None):
        self._manifest_adapter = manifest_adapter or AutoResearchClawManifestAdapter()

    def detect(self, path: Path) -> bool:
        root = Path(path)
        return self._result_summary_path(root).exists() and self._looks_like_e2e_root(root)

    def export_case(self, path: Path) -> AdapterExportResult:
        root = Path(path)
        if not self.detect(root):
            raise FileNotFoundError(f"No AutoResearchClaw e2e result stage found under {root}")

        stage_dir = root / self.result_stage
        result = self._manifest_adapter.export_case(stage_dir)
        case = result.case.model_copy(deep=True)
        diagnostics = list(result.diagnostics)

        topic = self._read_optional_json(root / "topic_manifest.json")
        case.case_id = f"autoresearchclaw-e2e:{root.name}"
        case.source_task = case.source_task or self._source_task(topic)
        case.stress_scenario = "imported_autoresearchclaw_e2e_run"
        case.native_system_behavior.update(
            {
                "adapter": self.name,
                "source_root": str(root),
                "selected_result_stage": self.result_stage,
                "can_restore_workspace_state": True,
                "can_export_machine_readable_revert_trace": False,
            }
        )
        case.expected_cairnlab_behavior.update(
            {
                "invalidate_selected_stage_summary": "challenge all claims supported by derived metrics",
                "do_not_treat_pipeline_pause_as_release_authority": True,
            }
        )

        summary_path = self._result_summary_path(root)
        summary = self._read_json(summary_path)
        run_evidence_id = self._run_evidence_id(case.evidence)
        self._tag_run_evidence(case.evidence, root, summary_path)
        self._add_run_artifacts(case, root, run_evidence_id)
        self._add_condition_metric_claims(case, summary, summary_path, run_evidence_id)
        self._add_stage_artifacts(case, root, run_evidence_id)
        self._add_failure_classes(case, root)
        diagnostics.extend(self._pipeline_diagnostics(root))
        diagnostics.extend(self._stage_diagnostics(root))
        diagnostics.extend(self._verifier_diagnostics(root))
        diagnostics.extend(self._structured_decision_diagnostics(root))

        return AdapterExportResult(case=case, diagnostics=diagnostics)

    def _looks_like_e2e_root(self, root: Path) -> bool:
        if root.name == self.result_stage:
            return False
        return (root / "topic_manifest.json").exists() or (root / "pipeline_summary.json").exists()

    def _result_summary_path(self, root: Path) -> Path:
        return root / self.result_stage / "experiment_summary.json"

    def _source_task(self, topic: dict[str, Any] | None) -> str | None:
        if not topic:
            return None
        for key in ("title", "topic", "research_question", "id"):
            value = topic.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _run_evidence_id(self, evidence: list[EvidenceItem]) -> str:
        for item in evidence:
            if self._value(item.type) == "run":
                return item.id
        return f"run:{self.result_stage}"

    def _tag_run_evidence(self, evidence: list[EvidenceItem], root: Path, summary_path: Path) -> None:
        for item in evidence:
            if self._value(item.type) != "run":
                continue
            item.metadata.update(
                {
                    "adapter": self.name,
                    "source_root": str(root),
                    "selected_stage": self.result_stage,
                    "source_file": str(summary_path),
                }
            )

    def _add_run_artifacts(self, case: ClaimCase, root: Path, run_evidence_id: str) -> None:
        pipeline_path = root / "pipeline_summary.json"
        if not pipeline_path.exists():
            return
        summary = self._read_json(pipeline_path)
        artifact_id = "artifact:pipeline_summary"
        if any(item.id == artifact_id for item in case.evidence):
            return
        case.evidence.append(
            EvidenceItem(
                id=artifact_id,
                type="artifact",
                uri=pipeline_path.resolve().as_uri(),
                status="valid",
                hash=self._file_sha256(pipeline_path),
                metadata={
                    "run_id": summary.get("run_id"),
                    "final_status": summary.get("final_status"),
                    "final_stage": summary.get("final_stage"),
                    "degraded": summary.get("degraded"),
                    "stages_executed": summary.get("stages_executed"),
                    "stages_done": summary.get("stages_done"),
                    "stages_paused": summary.get("stages_paused"),
                    "stages_failed": summary.get("stages_failed"),
                    "generated": summary.get("generated"),
                },
            )
        )
        case.relations.append(
            Relation(
                id=self._next_relation_id(case),
                source=artifact_id,
                target=run_evidence_id,
                type=RelationType.SUPPORTS.value,
                criticality=Criticality.CONTEXTUAL.value,
            )
        )

    def _add_condition_metric_claims(
        self,
        case: ClaimCase,
        summary: dict[str, Any],
        summary_path: Path,
        run_evidence_id: str,
    ) -> None:
        existing_evidence = {item.id for item in case.evidence}
        existing_claims = {claim.id for claim in case.claims}
        for condition, metric_name, metric_value in self._condition_metrics(summary):
            metric_key = f"{condition}/{metric_name}"
            metric_id = f"metric:{self.result_stage}.{metric_key}"
            claim_id = f"claim:{self.result_stage}.{metric_key}"
            if metric_id not in existing_evidence:
                case.evidence.append(
                    EvidenceItem(
                        id=metric_id,
                        type="metric",
                        uri=summary_path.resolve().as_uri()
                        + f"#/condition_summaries/{condition}/metrics/{metric_name}",
                        status="valid",
                        metadata={
                            "condition": condition,
                            "metric_name": metric_name,
                            "metric_key": metric_key,
                            "value": metric_value,
                            "source_stage": self.result_stage,
                        },
                    )
                )
                existing_evidence.add(metric_id)
                case.relations.append(
                    Relation(
                        id=self._next_relation_id(case),
                        source=run_evidence_id,
                        target=metric_id,
                        type=RelationType.COMPUTED.value,
                        criticality=Criticality.CRITICAL.value,
                    )
                )
            if claim_id in existing_claims:
                continue
            case.claims.append(
                Claim(
                    id=claim_id,
                    text=f"AutoResearchClaw e2e run reports `{metric_key}` = {metric_value}.",
                    type="empirical_metric",
                    state="verified",
                    scope={"metric": metric_key, "condition": condition},
                    risk="medium",
                    metadata={"source_stage": self.result_stage, "source_summary": str(summary_path)},
                )
            )
            existing_claims.add(claim_id)
            case.relations.append(
                Relation(
                    id=self._next_relation_id(case),
                    source=metric_id,
                    target=claim_id,
                    type=RelationType.SUPPORTS.value,
                    criticality=Criticality.CRITICAL.value,
                )
            )

    def _condition_metrics(self, summary: dict[str, Any]) -> list[tuple[str, str, Any]]:
        condition_summaries = summary.get("condition_summaries")
        if not isinstance(condition_summaries, dict):
            return []
        metrics: list[tuple[str, str, Any]] = []
        for condition, payload in condition_summaries.items():
            if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
                continue
            for metric_name, value in payload["metrics"].items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    metrics.append((str(condition), str(metric_name), value))
        return metrics

    def _add_stage_artifacts(self, case: ClaimCase, root: Path, run_evidence_id: str) -> None:
        for stage_name in self._stage_names(root):
            for filename in self._stage_artifact_filenames():
                path = root / stage_name / filename
                if not path.exists():
                    continue
                payload = self._read_json(path)
                artifact_id = f"artifact:{stage_name}.{path.stem}"
                if any(item.id == artifact_id for item in case.evidence):
                    continue
                case.evidence.append(
                    EvidenceItem(
                        id=artifact_id,
                        type="artifact",
                        uri=path.resolve().as_uri(),
                        status="valid",
                        hash=self._file_sha256(path),
                        metadata={
                            "stage": stage_name,
                            "artifact": filename,
                            "stage_id": payload.get("stage_id"),
                            "status": payload.get("status"),
                            "decision": payload.get("decision"),
                            "decision_parse_failed": payload.get("decision_parse_failed"),
                            "raw_text_excerpt": payload.get("raw_text_excerpt"),
                            "error": payload.get("error"),
                            "verdict": payload.get("verdict"),
                            "score_1_to_10": payload.get("score_1_to_10"),
                            "passed": payload.get("passed"),
                            "severity": payload.get("severity"),
                            "strict_violations": payload.get("strict_violations"),
                            "fabrication_rate": payload.get("fabrication_rate"),
                            "sanitized": payload.get("sanitized"),
                            "numbers_replaced": payload.get("numbers_replaced"),
                        },
                    )
                )
                case.relations.append(
                    Relation(
                        id=self._next_relation_id(case),
                        source=artifact_id,
                        target=run_evidence_id,
                        type=RelationType.SUPPORTS.value,
                        criticality=Criticality.CONTEXTUAL.value,
                    )
                )

    def _stage_artifact_filenames(self) -> tuple[str, ...]:
        return (
            "stage_health.json",
            "decision.json",
            "decision_structured.json",
            "quality_report.json",
            "paper_verification.json",
            "sanitization_report.json",
        )

    def _pipeline_diagnostics(self, root: Path) -> list[AdapterDiagnostic]:
        path = root / "pipeline_summary.json"
        if not path.exists():
            return []
        summary = self._read_json(path)
        diagnostics: list[AdapterDiagnostic] = []
        if summary.get("degraded") is True:
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message="AutoResearchClaw pipeline is degraded; imported claims are not release-authorized.",
                    path=str(path),
                )
            )
        final_status = str(summary.get("final_status") or "")
        if final_status in {"failed", "paused", "blocked"}:
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message=f"AutoResearchClaw pipeline final_status is {final_status}; release requires review.",
                    path=str(path),
                )
            )
        return diagnostics

    def _stage_diagnostics(self, root: Path) -> list[AdapterDiagnostic]:
        diagnostics = [
            AdapterDiagnostic(
                level="info",
                message="Selected AutoResearchClaw e2e result stage stage-14/experiment_summary.json.",
                path=str(self._result_summary_path(root)),
            )
        ]
        for stage_name in self._stage_names(root):
            health_path = root / stage_name / "stage_health.json"
            if not health_path.exists():
                continue
            health = self._read_json(health_path)
            status = str(health.get("status") or "")
            error = health.get("error")
            if status in {"failed", "paused", "blocked"}:
                diagnostics.append(
                    AdapterDiagnostic(
                        level="warning",
                        message=f"{stage_name} status is {status}; imported claims are not release-authorized.",
                        path=str(health_path),
                    )
                )
                if error:
                    diagnostics.append(
                        AdapterDiagnostic(level="warning", message=str(error), path=str(health_path))
                    )
        return diagnostics

    def _verifier_diagnostics(self, root: Path) -> list[AdapterDiagnostic]:
        diagnostics: list[AdapterDiagnostic] = []
        quality_path = root / "stage-20" / "quality_report.json"
        if quality_path.exists():
            quality = self._read_json(quality_path)
            verdict = str(quality.get("verdict") or "").upper()
            if verdict and verdict != "PASS":
                score = quality.get("score_1_to_10")
                suffix = f" score={score}." if score is not None else ""
                diagnostics.append(
                    AdapterDiagnostic(
                        level="warning",
                        message=f"stage-20 quality gate verdict is {verdict}; imported claims are not release-authorized.{suffix}",
                        path=str(quality_path),
                    )
                )

        verification_path = root / "stage-22" / "paper_verification.json"
        if verification_path.exists():
            verification = self._read_json(verification_path)
            severity = str(verification.get("severity") or "").upper()
            passed = verification.get("passed")
            if passed is False or severity in {"FAIL", "REJECT"}:
                details = self._paper_verification_details(verification)
                suffix = f" ({', '.join(details)})." if details else "."
                diagnostics.append(
                    AdapterDiagnostic(
                        level="warning",
                        message=f"stage-22 paper verification severity is {severity or 'FAIL'}{suffix}",
                        path=str(verification_path),
                    )
                )

        sanitization_path = root / "stage-22" / "sanitization_report.json"
        if sanitization_path.exists():
            sanitization = self._read_json(sanitization_path)
            numbers_replaced = sanitization.get("numbers_replaced")
            if sanitization.get("sanitized") is True and self._positive_int(numbers_replaced):
                diagnostics.append(
                    AdapterDiagnostic(
                        level="warning",
                        message=f"stage-22 sanitization replaced {numbers_replaced} numbers; numerical claims require re-verification.",
                        path=str(sanitization_path),
                    )
                )
        return diagnostics

    def _add_failure_classes(self, case: ClaimCase, root: Path) -> None:
        self._add_pipeline_failure_classes(case, root)
        self._add_stage_failure_classes(case, root)
        self._add_verifier_failure_classes(case, root)

    def _add_pipeline_failure_classes(self, case: ClaimCase, root: Path) -> None:
        path = root / "pipeline_summary.json"
        if not path.exists():
            return
        summary = self._read_json(path)
        if summary.get("degraded") is True:
            self._append_failure_class(case, "autoresearchclaw_pipeline_degraded")
        final_status = str(summary.get("final_status") or "")
        if final_status in {"failed", "paused", "blocked"}:
            self._append_failure_class(case, f"autoresearchclaw_pipeline_{final_status}")

    def _add_stage_failure_classes(self, case: ClaimCase, root: Path) -> None:
        for stage_name in self._stage_names(root):
            health_path = root / stage_name / "stage_health.json"
            if not health_path.exists():
                continue
            status = str(self._read_json(health_path).get("status") or "")
            if status not in {"failed", "paused", "blocked"}:
                continue
            failure_class = f"autoresearchclaw_{stage_name}_{status}"
            self._append_failure_class(case, failure_class)
            decision_path = root / stage_name / "decision_structured.json"
            if not decision_path.exists():
                continue
            decision = self._read_json(decision_path)
            if decision.get("decision_parse_failed") is True:
                self._append_failure_class(case, f"autoresearchclaw_{stage_name}_decision_parse_failed")
            if "invalid_request_error" in str(decision.get("raw_text_excerpt") or ""):
                self._append_failure_class(case, f"autoresearchclaw_{stage_name}_model_error")

    def _add_verifier_failure_classes(self, case: ClaimCase, root: Path) -> None:
        quality_path = root / "stage-20" / "quality_report.json"
        if quality_path.exists():
            quality = self._read_json(quality_path)
            verdict = str(quality.get("verdict") or "").upper()
            if verdict and verdict != "PASS":
                self._append_failure_class(case, f"autoresearchclaw_stage-20_quality_gate_{verdict.lower()}")

        verification_path = root / "stage-22" / "paper_verification.json"
        if verification_path.exists():
            verification = self._read_json(verification_path)
            severity = str(verification.get("severity") or "").upper()
            passed = verification.get("passed")
            if passed is False or severity in {"FAIL", "REJECT"}:
                suffix = severity.lower() if severity else "failed"
                self._append_failure_class(case, f"autoresearchclaw_stage-22_paper_verification_{suffix}")

        sanitization_path = root / "stage-22" / "sanitization_report.json"
        if sanitization_path.exists():
            sanitization = self._read_json(sanitization_path)
            if sanitization.get("sanitized") is True and self._positive_int(sanitization.get("numbers_replaced")):
                self._append_failure_class(case, "autoresearchclaw_stage-22_sanitized_numbers")

    def _structured_decision_diagnostics(self, root: Path) -> list[AdapterDiagnostic]:
        diagnostics: list[AdapterDiagnostic] = []
        for stage_name in self._stage_names(root):
            path = root / stage_name / "decision_structured.json"
            if not path.exists():
                continue
            decision = self._read_json(path)
            if decision.get("decision_parse_failed") is not True:
                continue
            excerpt = str(decision.get("raw_text_excerpt") or "").strip()
            message = f"{stage_name} structured decision parsing failed."
            if excerpt:
                message += f" Raw excerpt: {excerpt[:240]}"
            diagnostics.append(AdapterDiagnostic(level="warning", message=message, path=str(path)))
        return diagnostics

    def _append_failure_class(self, case: ClaimCase, failure_class: str) -> None:
        if failure_class not in case.failure_classes:
            case.failure_classes.append(failure_class)

    def _paper_verification_details(self, verification: dict[str, Any]) -> list[str]:
        details = []
        strict = verification.get("strict_violations")
        fabrication_rate = verification.get("fabrication_rate")
        if strict is not None:
            details.append(f"strict_violations={strict}")
        if fabrication_rate is not None:
            details.append(f"fabrication_rate={fabrication_rate}")
        return details

    def _positive_int(self, value: Any) -> bool:
        return isinstance(value, int) and value > 0

    def _stage_names(self, root: Path) -> list[str]:
        return sorted(path.name for path in root.glob("stage-*") if path.is_dir())

    def _next_relation_id(self, case: ClaimCase) -> str:
        return f"rel:{len(case.relations) + 1:04d}"

    def _read_optional_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return self._read_json(path)

    def _read_json(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return data

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"sha256:{digest}"

    def _value(self, value: Any) -> Any:
        return getattr(value, "value", value)
