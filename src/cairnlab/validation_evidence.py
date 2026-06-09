from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from .models import CairnModel


class ValidationEvidenceRun(CairnModel):
    id: str
    task: str | None = None
    real_or_fixture: str = "fixture"
    artifacts: list[str] = Field(default_factory=list)
    notes: str | None = None


class ValidationEvidenceSystem(CairnModel):
    name: str
    real_or_fixture: str = "fixture"
    runs: list[ValidationEvidenceRun] = Field(default_factory=list)


class ValidationEvidenceTask(CairnModel):
    id: str
    source_system: str
    real_or_fixture: str = "fixture"
    claims_recorded: int = 0
    material_claims_recorded: int = 0


class ValidationEvidenceClaim(CairnModel):
    id: str
    source_system: str
    task: str
    real_or_fixture: str = "fixture"
    materiality: str = "medium"
    upstream_observed_state: str | None = None
    cairnlab_counterfactual_state: str | None = None
    changed_release_decision: bool = False
    benchmark_would_capture: bool | None = None
    failure_classes: list[str] = Field(default_factory=list)


class ValidationEvidenceFailureClass(CairnModel):
    name: str
    real_occurrences: int = 0
    systems_seen: list[str] = Field(default_factory=list)
    release_control_relevant: bool = False
    plausibly_reduced_by_cairnlab: bool = False


class ValidationEvidenceLedger(CairnModel):
    version: int = 1
    updated_at: str | None = None
    systems_sampled: list[ValidationEvidenceSystem] = Field(default_factory=list)
    tasks: list[ValidationEvidenceTask] = Field(default_factory=list)
    claims: list[ValidationEvidenceClaim] = Field(default_factory=list)
    failure_classes: list[ValidationEvidenceFailureClass] = Field(default_factory=list)


def load_validation_evidence(root: Path) -> ValidationEvidenceLedger:
    """Load the optional real-evidence ledger for validation-first reporting."""

    for path in (
        root / "data" / "validation_evidence.yaml",
        root / ".cairn" / "validation_evidence.yaml",
    ):
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return ValidationEvidenceLedger.model_validate(data)
    return ValidationEvidenceLedger()


def evidence_counts(ledger: ValidationEvidenceLedger) -> dict[str, Any]:
    real_systems = {
        system.name
        for system in ledger.systems_sampled
        if system.real_or_fixture == "real" or any(run.real_or_fixture == "real" for run in system.runs)
    }
    real_runs = sum(
        1
        for system in ledger.systems_sampled
        for run in system.runs
        if run.real_or_fixture == "real"
    )
    real_tasks = [task for task in ledger.tasks if task.real_or_fixture == "real"]
    real_claims = [claim for claim in ledger.claims if claim.real_or_fixture == "real"]
    real_material_claims = [
        claim
        for claim in real_claims
        if claim.materiality in {"medium", "high", "critical"}
    ]
    real_failure_counts = {
        failure.name: failure.real_occurrences
        for failure in ledger.failure_classes
        if failure.real_occurrences > 0
    }
    release_control_failures = sum(
        failure.real_occurrences
        for failure in ledger.failure_classes
        if failure.release_control_relevant and failure.plausibly_reduced_by_cairnlab
    )
    if release_control_failures == 0:
        release_control_failures = sum(
            1
            for claim in real_claims
            if claim.changed_release_decision
            and any(_release_relevant_failure(name) for name in claim.failure_classes)
        )
    recurring_real_failures = [
        name for name, count in real_failure_counts.items() if count >= 2
    ]
    return {
        "real_systems_sampled": len(real_systems),
        "real_framework_runs": real_runs,
        "real_tasks": len(real_tasks),
        "real_material_claims": sum(task.material_claims_recorded for task in real_tasks)
        or len(real_material_claims),
        "real_release_control_failures": release_control_failures,
        "real_failure_class_counts": dict(sorted(real_failure_counts.items())),
        "recurring_real_failure_classes": recurring_real_failures,
    }


def _release_relevant_failure(name: str) -> bool:
    return any(
        token in name
        for token in (
            "release",
            "verifier",
            "dissent",
            "claim",
            "authority",
        )
    )
