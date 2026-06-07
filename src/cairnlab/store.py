from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import Claim, ClaimCase, EvidenceItem, ImportResult, Relation, TransitionEvent
from .utils import safe_id_filename


class CairnProjectStore:
    """Local `.cairn/` storage adapter.

    The store is deliberately boring: it loads and writes imported objects plus
    append-only events. It does not decide policy or graph semantics.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.cairn_dir = self.root / ".cairn"
        self.objects_dir = self.cairn_dir / "objects"
        self.claims_dir = self.objects_dir / "claims"
        self.evidence_dir = self.objects_dir / "evidence"
        self.relations_dir = self.objects_dir / "relations"
        self.cases_dir = self.objects_dir / "cases"
        self.events_file = self.cairn_dir / "events" / "events.jsonl"
        self.reports_dir = self.cairn_dir / "reports"

    def init(self) -> None:
        for subdir in [
            self.claims_dir,
            self.evidence_dir,
            self.relations_dir,
            self.objects_dir / "policies",
            self.cases_dir,
            self.cairn_dir / "events",
            self.reports_dir,
            self.cairn_dir / "cache",
        ]:
            subdir.mkdir(parents=True, exist_ok=True)
        project_file = self.cairn_dir / "project.yaml"
        if not project_file.exists():
            self._write_yaml(project_file, {"project": {"name": self.root.name, "version": 1}})
        if not self.events_file.exists():
            self.events_file.write_text("", encoding="utf-8")

    def load_case_file(self, path: Path) -> ClaimCase:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Claim case must be a mapping: {path}")
        return ClaimCase.model_validate(data)

    def import_case(self, path: Path) -> ImportResult:
        self.init()
        case = self.load_case_file(path)
        self._write_yaml(self.cases_dir / safe_id_filename(case.case_id), case.model_dump(mode="json", exclude_none=True))
        for claim in case.claims:
            self._write_yaml(self.claims_dir / safe_id_filename(claim.id), claim.model_dump(mode="json", exclude_none=True))
        for evidence in case.evidence:
            self._write_yaml(self.evidence_dir / safe_id_filename(evidence.id), evidence.model_dump(mode="json", exclude_none=True))
        for relation in case.relations:
            self._write_yaml(self.relations_dir / safe_id_filename(relation.id), relation.model_dump(mode="json", exclude_none=True))
        return ImportResult(
            case_id=case.case_id,
            claims=len(case.claims),
            evidence=len(case.evidence),
            relations=len(case.relations),
        )

    def append_event(self, event: TransitionEvent) -> None:
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def load_events(self) -> list[TransitionEvent]:
        if not self.events_file.exists():
            return []
        events: list[TransitionEvent] = []
        for line in self.events_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(TransitionEvent.model_validate(json.loads(line)))
        return events

    def load_claims(self) -> dict[str, Claim]:
        return {claim.id: claim for claim in self._load_many(self.claims_dir, Claim)}

    def load_evidence(self) -> dict[str, EvidenceItem]:
        return {item.id: item for item in self._load_many(self.evidence_dir, EvidenceItem)}

    def load_relations(self) -> list[Relation]:
        return self._load_many(self.relations_dir, Relation)

    def load_cases(self) -> list[ClaimCase]:
        return self._load_many(self.cases_dir, ClaimCase)

    def load_object_payloads(self) -> dict[str, dict[str, Any]]:
        payloads: dict[str, dict[str, Any]] = {}
        for claim in self.load_claims().values():
            payloads[claim.id] = claim.model_dump(mode="json", exclude_none=True)
        for evidence in self.load_evidence().values():
            payloads[evidence.id] = evidence.model_dump(mode="json", exclude_none=True)
        return payloads

    def write_validation_report(self, report_json: dict[str, Any], report_markdown: str) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "validation_report.json").write_text(
            json.dumps(report_json, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.reports_dir / "validation_report.md").write_text(report_markdown, encoding="utf-8")

    def _load_many(self, directory: Path, model_type):
        if not directory.exists():
            return []
        objects = []
        for path in sorted(directory.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if data is not None:
                objects.append(model_type.model_validate(data))
        return objects

    def _write_yaml(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
