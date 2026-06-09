from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cairnlab.adapters import (
    AdapterSelectionError,
    adapter_names,
    detect_adapters,
    export_case,
    select_adapter,
)
from cairnlab.cli import app
from cairnlab.engine import CairnProject


ROOT = Path(__file__).resolve().parents[1]
ARC_FIXTURE = ROOT / "tests" / "fixtures" / "autoresearchclaw_manifest"
ARC_E2E_FIXTURE = ROOT / "tests" / "fixtures" / "autoresearchclaw_e2e_run"
ARIS_FIXTURE = ROOT / "tests" / "fixtures" / "aris_manifest"
EXTERNAL_FIXTURE = ROOT / "tests" / "fixtures" / "external_run_manifest"


def test_registry_lists_builtin_manifest_adapters() -> None:
    assert adapter_names() == (
        "external-run-manifest",
        "autoresearchclaw-e2e-run",
        "autoresearchclaw-manifest",
        "aris-manifest",
    )


def test_registry_detects_autoresearchclaw_fixture() -> None:
    matches = detect_adapters(ARC_FIXTURE)

    assert [adapter.name for adapter in matches] == ["autoresearchclaw-manifest"]
    assert select_adapter(ARC_FIXTURE).name == "autoresearchclaw-manifest"


def test_registry_detects_autoresearchclaw_e2e_fixture() -> None:
    matches = detect_adapters(ARC_E2E_FIXTURE)

    assert [adapter.name for adapter in matches] == ["autoresearchclaw-e2e-run"]
    assert select_adapter(ARC_E2E_FIXTURE).name == "autoresearchclaw-e2e-run"


def test_registry_detects_aris_fixture() -> None:
    matches = detect_adapters(ARIS_FIXTURE)

    assert [adapter.name for adapter in matches] == ["aris-manifest"]
    assert export_case(ARIS_FIXTURE).case.source_system == "aris"


def test_registry_detects_external_run_manifest_fixture() -> None:
    matches = detect_adapters(EXTERNAL_FIXTURE)

    assert [adapter.name for adapter in matches] == ["external-run-manifest"]
    assert export_case(EXTERNAL_FIXTURE).case.source_system == "external-paper2code-stack"


def test_registry_rejects_no_match(tmp_path: Path) -> None:
    assert detect_adapters(tmp_path) == []
    with pytest.raises(AdapterSelectionError, match="No adapter detected"):
        select_adapter(tmp_path)


def test_registry_rejects_ambiguous_auto_detection(tmp_path: Path) -> None:
    (tmp_path / "experiment_summary.json").write_text('{"run_id": "mixed"}\n', encoding="utf-8")
    (tmp_path / "research-wiki" / "claims").mkdir(parents=True)

    matches = detect_adapters(tmp_path)

    assert [adapter.name for adapter in matches] == ["autoresearchclaw-manifest", "aris-manifest"]
    with pytest.raises(AdapterSelectionError, match="Multiple adapters detected"):
        select_adapter(tmp_path)


def test_import_external_cli_imports_detected_case(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "import-external",
            str(ARC_FIXTURE),
            "--path",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"adapter": "autoresearchclaw-manifest"' in result.output
    assert CairnProject.open(tmp_path).trace("claim:C1").object_id == "claim:C1"


def test_adapter_detect_cli_reports_matches() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["adapter", "detect", str(ARIS_FIXTURE), "--json"])

    assert result.exit_code == 0, result.output
    assert '"aris-manifest"' in result.output
