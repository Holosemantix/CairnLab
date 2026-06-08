from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..builder import ClaimCaseBuilder
from ..models import ClaimCase
from .base import AdapterDiagnostic, AdapterExportResult


class AutoResearchClawManifestAdapter:
    """Translate AutoResearchClaw-style exported metadata into a ClaimCase.

    This adapter reads JSON files only. It intentionally does not import
    AutoResearchClaw or assume its runtime is installed.
    """

    name = "autoresearchclaw-manifest"

    def detect(self, path: Path) -> bool:
        root = Path(path)
        return not self._looks_like_e2e_root(root) and self._summary_path(root) is not None

    def export_case(self, path: Path) -> AdapterExportResult:
        root = Path(path)
        summary_path = self._summary_path(root)
        if summary_path is None:
            raise FileNotFoundError(f"No experiment_summary.json found under {root}")

        diagnostics: list[AdapterDiagnostic] = []
        summary = self._read_json(summary_path)
        run_id = str(summary.get("run_id") or summary.get("id") or root.name or "run")
        claim = self._claim_payload(summary)
        claim_id = str(claim.get("id") or "claim:C1")
        metric_key, metric_value = self._primary_metric(summary)

        builder = ClaimCaseBuilder(
            case_id=str(summary.get("case_id") or f"autoresearchclaw:{run_id}"),
            source_system="autoresearchclaw",
            source_task=summary.get("task_id") or summary.get("research_question"),
            stress_scenario=str(summary.get("stress_scenario") or "imported_autoresearchclaw_run"),
        )
        builder.add_claim(
            claim_id,
            str(claim.get("text") or self._default_claim_text(metric_key, metric_value)),
            claim_type=str(claim.get("type") or "empirical_metric"),
            state=str(claim.get("state") or claim.get("status") or "verified"),
            scope=claim.get("scope") or {"metric": metric_key},
            risk=str(claim.get("risk") or "medium"),
            metadata=claim.get("metadata") or {},
        )

        run_evidence_id = f"run:{run_id}"
        builder.add_evidence(
            run_evidence_id,
            "run",
            uri=summary_path.resolve().as_uri(),
            hash=self._file_sha256(summary_path),
            metadata={
                "adapter": self.name,
                "source_file": str(summary_path),
                "metric_direction": summary.get("metric_direction"),
            },
        )

        metric_id = f"metric:{run_id}.{metric_key}"
        builder.add_evidence(
            metric_id,
            "metric",
            uri=summary_path.resolve().as_uri() + f"#/metrics/{metric_key}",
            metadata={"metric_name": metric_key, "value": metric_value},
        )
        builder.add_relation(run_evidence_id, metric_id, "computed", criticality="critical")
        builder.add_support(metric_id, claim_id)

        for artifact in self._artifact_payloads(summary, root):
            artifact_id = str(artifact.get("id") or f"artifact:{Path(str(artifact.get('path', 'artifact'))).stem}")
            artifact_path = self._resolve_artifact_path(root, artifact)
            builder.add_evidence(
                artifact_id,
                "artifact",
                uri=artifact_path.resolve().as_uri() if artifact_path and artifact_path.exists() else artifact.get("uri"),
                hash=artifact.get("sha256") or (self._file_sha256(artifact_path) if artifact_path and artifact_path.exists() else None),
                metadata={key: value for key, value in artifact.items() if key not in {"id", "sha256"}},
            )
            builder.add_relation(artifact_id, metric_id, "supports", criticality="supporting")

        section_id = self._paper_section_id(summary, claim)
        if section_id:
            builder.add_evidence(section_id, "paper_section", metadata={"source": "autoresearchclaw_manifest"})
            builder.add_relation(claim_id, section_id, "contained_in")

        approval_path = root / "hitl" / "approval.json"
        if approval_path.exists():
            approval = self._read_json(approval_path)
            builder.add_human_gate(
                str(approval.get("id") or "human_gate:H1"),
                claim_id,
                str(approval.get("actor") or "human:unknown"),
                authority=str(approval.get("authority") or "project_owner"),
                scope=approval.get("scope") or {"claim": claim_id, "run": run_evidence_id},
                rationale=str(approval.get("rationale") or "Imported AutoResearchClaw HITL approval."),
            )
        else:
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message="No hitl/approval.json found; released claims may require a human gate.",
                    path=str(root / "hitl"),
                )
            )

        release_path = root / "release_decision.json"
        if release_path.exists():
            decision = self._read_json(release_path)
            builder.add_release_decision(
                str(decision.get("id") or "release_decision:R1"),
                claim_id,
                str(decision.get("actor") or "human:unknown"),
                decision=str(decision.get("decision") or "allow"),
            )
        elif str(claim.get("state") or claim.get("status")) == "released":
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message="Released claim has no release_decision.json.",
                    path=str(release_path),
                )
            )

        builder.set_native_behavior(
            can_restore_workspace_state=True,
            can_identify_downstream_claims="unknown",
            can_invalidate_claims="unknown",
            can_downgrade_or_retract_released_claims="unknown",
            can_require_reapproval="unknown",
            can_export_machine_readable_revert_trace=False,
        )
        return AdapterExportResult(case=builder.build(), diagnostics=diagnostics)

    def _summary_path(self, root: Path) -> Path | None:
        candidates = [
            root / "experiment_summary.json",
            root / "experiment_summary_best.json",
            root / "results" / "experiment_summary.json",
            root / "deliverables" / "experiment_summary.json",
        ]
        return next((path for path in candidates if path.exists()), None)

    def _looks_like_e2e_root(self, root: Path) -> bool:
        return (
            (root / "stage-14" / "experiment_summary.json").exists()
            and ((root / "topic_manifest.json").exists() or (root / "pipeline_summary.json").exists())
        )

    def _claim_payload(self, summary: dict[str, Any]) -> dict[str, Any]:
        if isinstance(summary.get("claim"), dict):
            return summary["claim"]
        claims = summary.get("claims")
        if isinstance(claims, list) and claims and isinstance(claims[0], dict):
            return claims[0]
        return {}

    def _primary_metric(self, summary: dict[str, Any]) -> tuple[str, Any]:
        metric_key = summary.get("metric_key")
        metrics = {}
        best_run = summary.get("best_run")
        if isinstance(best_run, dict) and isinstance(best_run.get("metrics"), dict):
            metrics = best_run["metrics"]
        if metric_key is not None:
            key = str(metric_key)
            if key in metrics:
                return key, metrics[key]
            if key in summary:
                return key, summary[key]
        for preferred_key in ("test_accuracy", "accuracy", "primary_metric"):
            if preferred_key in metrics:
                return preferred_key, metrics[preferred_key]
            if preferred_key in summary:
                return preferred_key, summary[preferred_key]
        if metrics:
            first_key = next(iter(metrics))
            return str(first_key), metrics[first_key]
        return "primary_metric", None

    def _artifact_payloads(self, summary: dict[str, Any], root: Path) -> list[dict[str, Any]]:
        artifacts = summary.get("artifacts")
        if isinstance(artifacts, list):
            return [item for item in artifacts if isinstance(item, dict)]
        default_path = root / "metrics.json"
        if default_path.exists():
            return [{"id": "artifact:metrics_json", "path": "metrics.json"}]
        return []

    def _resolve_artifact_path(self, root: Path, artifact: dict[str, Any]) -> Path | None:
        path_value = artifact.get("path")
        if not path_value:
            return None
        path = Path(str(path_value))
        return path if path.is_absolute() else root / path

    def _paper_section_id(self, summary: dict[str, Any], claim: dict[str, Any]) -> str | None:
        metadata = claim.get("metadata")
        if isinstance(metadata, dict) and metadata.get("paper_section"):
            return str(metadata["paper_section"])
        section = summary.get("paper_section")
        if isinstance(section, str):
            return section
        if isinstance(section, dict) and section.get("id"):
            return str(section["id"])
        return None

    def _default_claim_text(self, metric_key: str, metric_value: Any) -> str:
        if metric_value is None:
            return f"AutoResearchClaw run reports metric `{metric_key}`."
        return f"AutoResearchClaw run reports `{metric_key}` = {metric_value}."

    def _read_json(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return data

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"sha256:{digest}"
