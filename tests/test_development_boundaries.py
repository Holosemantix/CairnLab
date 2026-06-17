from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "cairnlab"
ADAPTERS = SRC / "adapters"

FORBIDDEN_ADAPTER_INTERNAL_IMPORTS = {
    "cairnlab.authority",
    "cairnlab.cli",
    "cairnlab.engine",
    "cairnlab.graph",
    "cairnlab.planner",
    "cairnlab.projection",
    "cairnlab.runtime",
    "cairnlab.store",
    "cairnlab.trace_package",
    "cairnlab.transition_explain",
    "cairnlab.validation",
    "cairnlab.verifiers",
}
FORBIDDEN_HOST_RUNTIME_IMPORT_ROOTS = {
    "aris",
    "auto_research_claw",
    "autoresearch",
    "autoresearchclaw",
    "dvc",
    "mlflow",
    "paper2agent",
    "paper2code",
    "prov",
    "research_wiki",
    "rocrate",
}
FORBIDDEN_ADAPTER_CALLS = {
    "append_event",
    "append_transition_events",
    "apply_transition_decision",
    "request_transition",
}
FORBIDDEN_CLI_IMPORTS = {
    "cairnlab.authority",
    "cairnlab.graph",
    "cairnlab.planner",
    "cairnlab.projection",
    "cairnlab.store",
    "cairnlab.verifiers",
}
EVENT_APPEND_ALLOWED_FILES = {
    "src/cairnlab/engine.py",
    "src/cairnlab/store.py",
}
AUTHORITY_ALLOWED_IMPORT_FILES = {
    "src/cairnlab/__init__.py",
    "src/cairnlab/engine.py",
}


def test_adapters_stay_outside_authority_and_runtime_layers() -> None:
    violations: list[str] = []
    for path in _python_files(ADAPTERS):
        imports = _imported_modules(path)
        forbidden_imports = sorted(
            imported
            for imported in imports
            if _matches_any(imported, FORBIDDEN_ADAPTER_INTERNAL_IMPORTS)
        )
        forbidden_host_imports = sorted(
            imported
            for imported in imports
            if imported.split(".", 1)[0] in FORBIDDEN_HOST_RUNTIME_IMPORT_ROOTS
        )
        forbidden_calls = sorted(_called_attributes(path) & FORBIDDEN_ADAPTER_CALLS)
        if forbidden_imports:
            violations.append(
                f"{_rel(path)} imports CairnLab runtime/authority modules: "
                f"{forbidden_imports}"
            )
        if forbidden_host_imports:
            violations.append(
                f"{_rel(path)} imports host AutoResearch runtime modules: "
                f"{forbidden_host_imports}"
            )
        if forbidden_calls:
            violations.append(f"{_rel(path)} calls authority/event methods: {forbidden_calls}")

    assert violations == []


def test_cli_stays_a_facade_over_engine_and_public_apis() -> None:
    path = SRC / "cli.py"
    imports = _imported_modules(path)
    forbidden_imports = sorted(
        imported for imported in imports if _matches_any(imported, FORBIDDEN_CLI_IMPORTS)
    )
    forbidden_calls = sorted(
        _called_attributes(path) & {"append_event", "append_transition_events"}
    )
    names = _loaded_names(path)

    assert forbidden_imports == []
    assert forbidden_calls == []
    assert "TransitionAuthority" not in names


def test_transition_events_are_appended_only_by_store_or_engine_facade() -> None:
    violations: list[str] = []
    for path in _python_files(SRC):
        rel = _rel(path)
        if rel in EVENT_APPEND_ALLOWED_FILES:
            continue
        if "append_event" in _called_attributes(path):
            violations.append(rel)

    assert violations == []


def test_transition_authority_is_composed_only_by_engine_or_public_export() -> None:
    violations: list[str] = []
    for path in _python_files(SRC):
        rel = _rel(path)
        imports_authority = any(
            _matches_any(imported, {"cairnlab.authority"}) for imported in _imported_modules(path)
        )
        uses_authority_name = "TransitionAuthority" in _loaded_names(path)
        if rel not in AUTHORITY_ALLOWED_IMPORT_FILES and (imports_authority or uses_authority_name):
            violations.append(rel)

    assert violations == []


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT / "src").with_suffix("")
    return ".".join(relative.parts)


def _imported_modules(path: Path) -> set[str]:
    imports: set[str] = set()
    current_module = _module_name(path)
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(_resolve_import_from(current_module, node))
    return {item for item in imports if item}


def _resolve_import_from(current_module: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = current_module.split(".")[:-1]
    if node.level > 1:
        package_parts = package_parts[: -(node.level - 1)]
    module_parts = node.module.split(".") if node.module else []
    return ".".join([*package_parts, *module_parts])


def _called_attributes(path: Path) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _loaded_names(path: Path) -> set[str]:
    return {
        node.id
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _matches_any(module: str, forbidden_modules: set[str]) -> bool:
    return any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for forbidden in forbidden_modules
    )
