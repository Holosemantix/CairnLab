from .base import AdapterDiagnostic, AdapterExportResult, AutoResearchAdapter
from .autoresearchclaw_manifest import AutoResearchClawManifestAdapter
from .aris_manifest import ArisManifestAdapter
from .registry import (
    AdapterSelectionError,
    adapter_by_name,
    adapter_names,
    available_adapters,
    detect_adapters,
    export_case,
    select_adapter,
)

__all__ = [
    "AdapterDiagnostic",
    "AdapterExportResult",
    "AdapterSelectionError",
    "AutoResearchAdapter",
    "AutoResearchClawManifestAdapter",
    "ArisManifestAdapter",
    "adapter_by_name",
    "adapter_names",
    "available_adapters",
    "detect_adapters",
    "export_case",
    "select_adapter",
]
