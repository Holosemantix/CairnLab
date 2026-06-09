from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..builder import ClaimCaseBuilder
from .base import AdapterDiagnostic, AdapterExportResult


class ArisManifestAdapter:
    """Translate ARIS-style exported metadata into a CairnLab ClaimCase.

    The adapter reads structured JSON/JSONL exports and ARIS wiki Markdown
    frontmatter only. It does not import ARIS, invoke skills, parse papers, or
    mutate the host project.
    """

    name = "aris-manifest"

    def detect(self, path: Path) -> bool:
        root = Path(path)
        has_manifest = (root / "research-wiki" / "claims").exists() or any(
            (root / name).exists()
            for name in (
                "PROOF_AUDIT.json",
                "EXPERIMENT_AUDIT.json",
                "PAPER_CLAIM_AUDIT.json",
                "CITATION_AUDIT.json",
                "KILL_ARGUMENT.json",
            )
        )
        return has_manifest or (self._has_aris_marker(root) and bool(self._review_sidecar_paths(root)))

    def export_case(self, path: Path) -> AdapterExportResult:
        root = Path(path)
        diagnostics: list[AdapterDiagnostic] = []
        builder = ClaimCaseBuilder(
            case_id=f"aris:{root.name or 'project'}",
            source_system="aris",
            stress_scenario="imported_aris_project",
        )

        claim_ids = self._add_claims(root, builder, diagnostics)
        experiment_ids = self._add_experiments(root, builder, diagnostics)
        self._add_edges(root, builder, diagnostics)
        self._add_audits(root, builder, claim_ids, diagnostics)
        self._add_review_sidecars(root, builder, diagnostics)
        self._add_submission_verifier_reports(root, builder, claim_ids, diagnostics)
        self._add_optional_human_gate(root, builder, claim_ids, diagnostics)

        if not claim_ids:
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message="No structured claim JSON or Markdown files found under research-wiki/claims.",
                    path=str(root / "research-wiki" / "claims"),
                )
            )
        if not experiment_ids:
            diagnostics.append(
                AdapterDiagnostic(
                    level="info",
                    message="No structured experiment JSON or Markdown files found under research-wiki/experiments.",
                    path=str(root / "research-wiki" / "experiments"),
                )
            )

        builder.set_native_behavior(
            can_identify_downstream_claims="partial",
            can_invalidate_claims="partial",
            can_downgrade_or_retract_released_claims="unknown",
            can_preserve_old_history_append_only=True,
            can_require_reapproval="unknown",
            can_export_machine_readable_revert_trace=True,
        )
        builder.expected_cairnlab_behavior.update(
            {
                "do_not_treat_llm_review_as_transition_authority": True,
                "require_transition_authority_for_release": True,
            }
        )
        return AdapterExportResult(case=builder.build(), diagnostics=diagnostics)

    def _add_claims(
        self,
        root: Path,
        builder: ClaimCaseBuilder,
        diagnostics: list[AdapterDiagnostic],
    ) -> list[str]:
        claims_dir = root / "research-wiki" / "claims"
        claim_ids: list[str] = []
        for claim_path in self._structured_paths(claims_dir):
            claim = self._read_structured(claim_path, diagnostics)
            if not claim:
                continue
            claim_id = str(claim.get("id") or claim.get("node_id") or f"claim:{claim_path.stem}")
            builder.add_claim(
                claim_id,
                str(claim.get("text") or claim.get("claim") or claim_path.stem),
                claim_type=str(claim.get("claim_type") or claim.get("cairn_type") or "empirical_metric"),
                state=self._claim_state(claim.get("state") or claim.get("status")),
                scope=claim.get("scope") or {},
                risk=str(claim.get("risk") or "medium"),
                metadata={
                    "source_file": str(claim_path),
                    **(claim.get("metadata") or {}),
                },
            )
            claim_ids.append(claim_id)
        return claim_ids

    def _add_experiments(
        self,
        root: Path,
        builder: ClaimCaseBuilder,
        diagnostics: list[AdapterDiagnostic],
    ) -> list[str]:
        experiments_dir = root / "research-wiki" / "experiments"
        experiment_ids: list[str] = []
        for experiment_path in self._structured_paths(experiments_dir):
            experiment = self._read_structured(experiment_path, diagnostics)
            if not experiment:
                continue
            experiment_id = str(experiment.get("id") or experiment.get("node_id") or f"experiment:{experiment_path.stem}")
            run_id = experiment_id if experiment_id.startswith("run:") else f"run:{experiment_id.split(':')[-1]}"
            builder.add_evidence(
                run_id,
                "run",
                uri=experiment_path.resolve().as_uri(),
                hash=self._file_sha256(experiment_path),
                metadata={
                    "source_file": str(experiment_path),
                    "status": experiment.get("status"),
                    "verdict": experiment.get("verdict"),
                },
            )
            experiment_ids.append(run_id)
            metrics = experiment.get("metrics") or {}
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    metric_id = f"metric:{run_id.removeprefix('run:')}.{metric_name}"
                    builder.add_evidence(
                        metric_id,
                        "metric",
                        uri=experiment_path.resolve().as_uri() + f"#/metrics/{metric_name}",
                        metadata={"metric_name": metric_name, "value": value},
                    )
                    builder.add_relation(run_id, metric_id, "computed", criticality="critical")
            for artifact in experiment.get("artifacts") or []:
                if not isinstance(artifact, dict):
                    continue
                artifact_id = str(artifact.get("id") or f"artifact:{Path(str(artifact.get('path', 'artifact'))).stem}")
                builder.add_evidence(
                    artifact_id,
                    "artifact",
                    uri=artifact.get("uri"),
                    hash=artifact.get("sha256"),
                    metadata={key: value for key, value in artifact.items() if key not in {"id", "sha256"}},
                )
        return experiment_ids

    def _add_edges(
        self,
        root: Path,
        builder: ClaimCaseBuilder,
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        edges_path = root / "research-wiki" / "graph" / "edges.jsonl"
        if not edges_path.exists():
            diagnostics.append(
                AdapterDiagnostic(
                    level="info",
                    message="No research-wiki graph/edges.jsonl found; relation coverage may be incomplete.",
                    path=str(edges_path),
                )
            )
            return
        for index, line in enumerate(edges_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                edge = json.loads(line)
            except json.JSONDecodeError as exc:
                diagnostics.append(
                    AdapterDiagnostic(
                        level="warning",
                        message=f"Skipping invalid edge JSON on line {index}: {exc}",
                        path=str(edges_path),
                    )
                )
                continue
            if not isinstance(edge, dict):
                diagnostics.append(
                    AdapterDiagnostic(
                        level="warning",
                        message=f"Skipping edge JSON that is not an object on line {index}.",
                        path=str(edges_path),
                    )
                )
                continue
            source = edge.get("source") or edge.get("from")
            target = edge.get("target") or edge.get("to")
            if not source or not target:
                diagnostics.append(
                    AdapterDiagnostic(
                        level="warning",
                        message=f"Skipping incomplete edge on line {index}.",
                        path=str(edges_path),
                    )
                )
                continue
            source_id = self._normalize_node_id(str(source))
            target = self._normalize_node_id(str(target))
            if target.startswith("paper_section:"):
                builder.add_evidence(
                    target,
                    "paper_section",
                    metadata={"source": "aris_research_wiki_edge"},
                )
            builder.add_relation(
                source_id,
                target,
                self._relation_type(str(edge.get("type") or "depends_on")),
                relation_id=str(edge.get("id") or f"rel:aris:{index:04d}"),
                criticality=str(edge.get("criticality") or "supporting"),
            )

    def _add_audits(
        self,
        root: Path,
        builder: ClaimCaseBuilder,
        claim_ids: list[str],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        audit_files = [
            ("PROOF_AUDIT.json", "proof_audit"),
            ("EXPERIMENT_AUDIT.json", "experiment_audit"),
            ("PAPER_CLAIM_AUDIT.json", "paper_claim_audit"),
            ("CITATION_AUDIT.json", "citation_audit"),
            ("KILL_ARGUMENT.json", "kill_argument"),
        ]
        for file_name, audit_type in audit_files:
            for audit_path in self._audit_paths(root, file_name):
                audit = self._read_json(audit_path, diagnostics)
                if not audit:
                    continue
                audit_id = str(audit.get("id") or self._default_audit_id(root, audit_path, audit_type))
                builder.add_evidence(
                    audit_id,
                    "verifier_certificate",
                    uri=audit_path.resolve().as_uri(),
                    hash=self._file_sha256(audit_path),
                    metadata={
                        "audit_type": audit_type,
                        "verdict": audit.get("verdict") or audit.get("status"),
                        "source_file": str(audit_path),
                        "inputs": audit.get("inputs") or audit.get("evidence_refs") or [],
                    },
                )
                for claim_id in claim_ids:
                    builder.add_relation(audit_id, claim_id, "verified_by", criticality="supporting")
                self._record_verdict_diagnostic(
                    builder,
                    diagnostics,
                    path=audit_path,
                    verdict=audit.get("verdict") or audit.get("status"),
                    source=f"ARIS {audit_type}",
                    failure_class=f"aris_{audit_type}_rejected",
                )

    def _add_review_sidecars(
        self,
        root: Path,
        builder: ClaimCaseBuilder,
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for review_path in self._review_sidecar_paths(root):
            review = self._read_json(review_path, diagnostics)
            if not review:
                continue
            evidence_id = f"reviewer:{self._path_id(root, review_path)}"
            verdicts = self._verdicts(review)
            builder.add_evidence(
                evidence_id,
                "reviewer_verdict",
                uri=review_path.resolve().as_uri(),
                hash=self._file_sha256(review_path),
                metadata={
                    "adapter": self.name,
                    "artifact_type": "aris_review_sidecar",
                    "source_file": str(review_path),
                    "skill": review.get("skill"),
                    "source": review.get("source"),
                    "output": review.get("output"),
                    "reviewer": review.get("reviewer"),
                    "source_sha256": review.get("source_sha256"),
                    "verdict": self._primary_verdict(review),
                    "verdicts": verdicts,
                    "blocking_issues_count": self._len_if_list(review.get("blocking_issues")),
                    "warnings_count": self._len_if_list(review.get("warnings")),
                    "rendered_at": review.get("rendered_at"),
                    "generated_at": review.get("generated_at"),
                    "not_transition_authority": True,
                },
            )
            self._record_verdict_diagnostic(
                builder,
                diagnostics,
                path=review_path,
                verdict=self._worst_verdict(verdicts),
                source="ARIS review sidecar",
                failure_class="aris_review_sidecar_rejected",
                llm_review=True,
            )

    def _add_submission_verifier_reports(
        self,
        root: Path,
        builder: ClaimCaseBuilder,
        claim_ids: list[str],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        for report_path in self._submission_verifier_report_paths(root):
            report = self._read_json(report_path, diagnostics)
            if not report:
                continue
            evidence_id = f"verifier:{self._path_id(root, report_path)}"
            verdict = self._verifier_report_verdict(report)
            exit_code = report.get("exit_code")
            if verdict == "UNKNOWN" and isinstance(exit_code, int):
                verdict = "PASS" if exit_code == 0 else "FAIL"
            builder.add_evidence(
                evidence_id,
                "verifier_certificate",
                uri=report_path.resolve().as_uri(),
                hash=self._file_sha256(report_path),
                metadata={
                    "adapter": self.name,
                    "artifact_type": "aris_submission_verifier_report",
                    "source_file": str(report_path),
                    "verifier": report.get("verifier") or "verify_paper_audits.sh",
                    "verdict": verdict,
                    "exit_code": exit_code,
                    "assurance": report.get("assurance"),
                    "any_problem": report.get("any_problem"),
                    "submission_blocking": report.get("submission_blocking"),
                    "paper_dir": report.get("paper_dir"),
                    "audits": report.get("audits") or [],
                    "generated_at": report.get("generated_at"),
                },
            )
            for claim_id in claim_ids:
                builder.add_relation(evidence_id, claim_id, "verified_by", criticality="critical")
            self._record_verdict_diagnostic(
                builder,
                diagnostics,
                path=report_path,
                verdict=verdict,
                source="ARIS submission verifier",
                failure_class="aris_submission_verifier_rejected",
            )

    def _add_optional_human_gate(
        self,
        root: Path,
        builder: ClaimCaseBuilder,
        claim_ids: list[str],
        diagnostics: list[AdapterDiagnostic],
    ) -> None:
        gate_path = root / ".aris" / "human_gate.json"
        if not claim_ids:
            return
        if not gate_path.exists():
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message="No .aris/human_gate.json found; released ARIS claims may require a liability-bearing human gate.",
                    path=str(gate_path),
                )
            )
            return
        gate = self._read_json(gate_path, diagnostics)
        if not gate:
            return
        claim_id = str(gate.get("claim") or claim_ids[0])
        builder.add_human_gate(
            str(gate.get("id") or "human_gate:aris"),
            claim_id,
            str(gate.get("actor") or "human:unknown"),
            authority=str(gate.get("authority") or "project_owner"),
            scope=gate.get("scope") or {"claim": claim_id},
            rationale=str(gate.get("rationale") or "Imported ARIS human gate."),
        )

    def _has_aris_marker(self, root: Path) -> bool:
        return any(
            path.exists()
            for path in (
                root / "AGENT_GUIDE.md",
                root / "skills" / "research-wiki" / "SKILL.md",
                root / "skills" / "skills-codex" / "shared-references" / "assurance-contract.md",
                root / ".aris" / "installed-skills-codex.txt",
            )
        )

    def _review_sidecar_paths(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        return sorted(path for path in root.rglob("*.review.json") if path.is_file())

    def _submission_verifier_report_paths(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        return sorted(path for path in root.rglob("audit-verifier-report.json") if path.is_file())

    def _audit_paths(self, root: Path, file_name: str) -> list[Path]:
        candidates = [root / file_name]
        for report_path in self._submission_verifier_report_paths(root):
            candidates.append(report_path.parent.parent / file_name)
        seen: set[Path] = set()
        paths: list[Path] = []
        for path in candidates:
            if path.exists() and path not in seen:
                paths.append(path)
                seen.add(path)
        return paths

    def _record_verdict_diagnostic(
        self,
        builder: ClaimCaseBuilder,
        diagnostics: list[AdapterDiagnostic],
        *,
        path: Path,
        verdict: Any,
        source: str,
        failure_class: str,
        llm_review: bool = False,
    ) -> None:
        normalized = self._normalize_verdict(verdict)
        if normalized in {"PASS", "NOT_APPLICABLE", "UNKNOWN"}:
            return
        if normalized == "WARN":
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message=f"{source} reports WARN; imported evidence needs review before any stronger claim transition.",
                    path=str(path),
                )
            )
            return
        builder.add_failure_class(failure_class)
        authority_note = " LLM review is evidence context, not transition authority." if llm_review else ""
        diagnostics.append(
            AdapterDiagnostic(
                level="warning",
                message=f"{source} reports {normalized}; imported claims are not release-authorized.{authority_note}",
                path=str(path),
            )
        )

    def _verdicts(self, value: Any) -> list[str]:
        verdicts: list[str] = []
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "verdict":
                    verdicts.append(self._normalize_verdict(child))
                else:
                    verdicts.extend(self._verdicts(child))
        elif isinstance(value, list):
            for child in value:
                verdicts.extend(self._verdicts(child))
        return verdicts

    def _primary_verdict(self, payload: dict[str, Any]) -> str:
        return self._normalize_verdict(payload.get("verdict") or payload.get("status") or "unknown")

    def _verifier_report_verdict(self, report: dict[str, Any]) -> str:
        verdict = self._primary_verdict(report)
        if verdict != "UNKNOWN":
            return verdict
        if report.get("submission_blocking") is True:
            return "FAIL"
        if report.get("any_problem") is True:
            return "WARN"
        if "submission_blocking" in report or "any_problem" in report:
            return "PASS"
        return "UNKNOWN"

    def _worst_verdict(self, verdicts: list[str]) -> str:
        order = ["ERROR", "BLOCKED", "FAIL", "WARN", "UNKNOWN", "PASS", "NOT_APPLICABLE"]
        present = {self._normalize_verdict(verdict) for verdict in verdicts}
        return next((verdict for verdict in order if verdict in present), "UNKNOWN")

    def _normalize_verdict(self, verdict: Any) -> str:
        value = str(verdict or "unknown").strip().upper()
        if "WARN" in value:
            return "WARN"
        if value.startswith("PASS"):
            return "PASS"
        if value.startswith("FAIL") or value in {"REJECT", "REJECTED"}:
            return "FAIL"
        if value in {"BLOCKED", "ERROR", "NOT_APPLICABLE"}:
            return value
        return "UNKNOWN"

    def _len_if_list(self, value: Any) -> int | None:
        return len(value) if isinstance(value, list) else None

    def _path_id(self, root: Path, path: Path) -> str:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path.name
        stem = str(relative).removesuffix(".json").removesuffix(".review")
        return re.sub(r"[^A-Za-z0-9_.-]+", ".", stem).strip(".") or "artifact"

    def _default_audit_id(self, root: Path, path: Path, audit_type: str) -> str:
        if path.parent == root:
            return f"verifier:{audit_type}"
        return f"verifier:{self._path_id(root, path)}"

    def _structured_paths(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted([*directory.glob("*.json"), *directory.glob("*.md")])

    def _read_structured(
        self,
        path: Path,
        diagnostics: list[AdapterDiagnostic],
    ) -> dict[str, Any] | None:
        if path.suffix == ".json":
            return self._read_json(path, diagnostics)
        return self._read_markdown_frontmatter(path, diagnostics)

    def _read_markdown_frontmatter(
        self,
        path: Path,
        diagnostics: list[AdapterDiagnostic],
    ) -> dict[str, Any] | None:
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
        if not match:
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message="Skipping Markdown without frontmatter.",
                    path=str(path),
                )
            )
            return None
        payload = self._parse_frontmatter(match.group(1))
        body = match.group(2).strip()
        payload.setdefault("metadata", {})
        payload["metadata"]["source_format"] = "aris_markdown_frontmatter"
        if "text" not in payload and "claim" not in payload:
            first_paragraph = self._first_paragraph(body)
            if first_paragraph:
                payload["text"] = first_paragraph
        return payload

    def _parse_frontmatter(self, raw: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for line in raw.splitlines():
            if ":" not in line or line.startswith((" ", "\t", "-")):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            payload[key] = self._parse_frontmatter_value(value)
        return payload

    def _parse_frontmatter_value(self, value: str) -> Any:
        if value in {"", "null", "None"}:
            return None
        if value in {"true", "false"}:
            return value == "true"
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _first_paragraph(self, body: str) -> str | None:
        lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                if lines:
                    break
                continue
            if stripped.startswith("#"):
                continue
            lines.append(stripped)
        return " ".join(lines) if lines else None

    def _normalize_node_id(self, value: str) -> str:
        if value.startswith("exp:"):
            return f"run:{value.split(':', 1)[1]}"
        return value

    def _claim_state(self, value: Any) -> str:
        mapping = {
            "supported": "verified",
            "partial": "challenged",
            "invalidated": "invalidated",
            "reported": "evidence_attached",
            "pending": "verification_pending",
        }
        state = str(value or "draft")
        return mapping.get(state, state)

    def _relation_type(self, aris_type: str) -> str:
        mapping = {
            "supports": "supports",
            "invalidates": "challenges",
            "tested_by": "depends_on",
            "verified_by": "verified_by",
            "reported_in": "contained_in",
            "contained_in": "contained_in",
            "challenges": "challenges",
        }
        return mapping.get(aris_type, "depends_on")

    def _read_json(
        self,
        path: Path,
        diagnostics: list[AdapterDiagnostic],
    ) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message=f"Skipping invalid JSON: {exc}",
                    path=str(path),
                )
            )
            return None
        if not isinstance(data, dict):
            diagnostics.append(
                AdapterDiagnostic(
                    level="warning",
                    message="Skipping JSON that is not an object.",
                    path=str(path),
                )
            )
            return None
        return data

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"sha256:{digest}"
