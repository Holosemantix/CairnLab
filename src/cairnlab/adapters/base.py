from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from ..models import ClaimCase


class AdapterDiagnostic(BaseModel):
    level: str = "info"
    message: str
    path: str | None = None


class AdapterExportResult(BaseModel):
    case: ClaimCase
    diagnostics: list[AdapterDiagnostic] = Field(default_factory=list)


class AutoResearchAdapter(Protocol):
    """Protocol for translating host project metadata into CairnLab cases."""

    name: str

    def detect(self, path: Path) -> bool:
        """Return True when this adapter can read metadata under path."""

    def export_case(self, path: Path) -> AdapterExportResult:
        """Translate host metadata into a CairnLab ClaimCase."""
