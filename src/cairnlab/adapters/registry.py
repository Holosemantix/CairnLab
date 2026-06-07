from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .aris_manifest import ArisManifestAdapter
from .autoresearchclaw_manifest import AutoResearchClawManifestAdapter
from .base import AdapterExportResult, AutoResearchAdapter


class AdapterSelectionError(ValueError):
    """Raised when an adapter cannot be selected deterministically."""


def builtin_adapters() -> tuple[AutoResearchAdapter, ...]:
    return (
        AutoResearchClawManifestAdapter(),
        ArisManifestAdapter(),
    )


def available_adapters(
    adapters: Iterable[AutoResearchAdapter] | None = None,
) -> tuple[AutoResearchAdapter, ...]:
    return tuple(adapters) if adapters is not None else builtin_adapters()


def adapter_names(adapters: Iterable[AutoResearchAdapter] | None = None) -> tuple[str, ...]:
    return tuple(adapter.name for adapter in available_adapters(adapters))


def adapter_by_name(
    name: str,
    adapters: Iterable[AutoResearchAdapter] | None = None,
) -> AutoResearchAdapter:
    for adapter in available_adapters(adapters):
        if adapter.name == name:
            return adapter
    known = ", ".join(adapter_names(adapters)) or "none"
    raise AdapterSelectionError(f"Unknown adapter '{name}'. Available adapters: {known}")


def detect_adapters(
    path: str | Path,
    adapters: Iterable[AutoResearchAdapter] | None = None,
) -> list[AutoResearchAdapter]:
    root = Path(path)
    return [adapter for adapter in available_adapters(adapters) if adapter.detect(root)]


def select_adapter(
    path: str | Path,
    adapter_name: str = "auto",
    adapters: Iterable[AutoResearchAdapter] | None = None,
) -> AutoResearchAdapter:
    if adapter_name != "auto":
        return adapter_by_name(adapter_name, adapters)

    matches = detect_adapters(path, adapters)
    if not matches:
        known = ", ".join(adapter_names(adapters)) or "none"
        raise AdapterSelectionError(f"No adapter detected for {Path(path)}. Available adapters: {known}")
    if len(matches) > 1:
        names = ", ".join(adapter.name for adapter in matches)
        raise AdapterSelectionError(f"Multiple adapters detected for {Path(path)}: {names}. Pass --adapter explicitly.")
    return matches[0]


def export_case(
    path: str | Path,
    adapter_name: str = "auto",
    adapters: Iterable[AutoResearchAdapter] | None = None,
) -> AdapterExportResult:
    adapter = select_adapter(path, adapter_name=adapter_name, adapters=adapters)
    return adapter.export_case(Path(path))
