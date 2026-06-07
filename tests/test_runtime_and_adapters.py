from __future__ import annotations

from pathlib import Path

from cairnlab import CairnRuntime, ClaimCaseBuilder
from cairnlab.adapters import AdapterDiagnostic, AdapterExportResult


class FakeAutoResearchAdapter:
    name = "fake-autoresearch"

    def detect(self, path: Path) -> bool:
        return (path / "experiment_summary.json").exists()

    def export_case(self, path: Path) -> AdapterExportResult:
        builder = ClaimCaseBuilder(
            case_id="fake-autoresearch-run",
            source_system=self.name,
            stress_scenario="wrong_metric_path",
        )
        case = (
            builder.add_claim(
                "claim:C1",
                "Method A improves baseline B on Dataset X.",
                state="released",
                metadata={"paper_section": "paper_section:results.table_1"},
            )
            .add_evidence(
                "run:exp_001",
                "run",
                uri=(path / "experiment_summary.json").as_uri(),
                hash="sha256:run001",
            )
            .add_evidence(
                "metric:exp_001.primary",
                "metric",
                metadata={"metric_name": "primary_metric", "value": 0.91},
            )
            .add_evidence("paper_section:results.table_1", "paper_section")
            .add_relation("run:exp_001", "metric:exp_001.primary", "computed", criticality="critical")
            .add_support("metric:exp_001.primary", "claim:C1")
            .add_relation("claim:C1", "paper_section:results.table_1", "contained_in")
            .add_human_gate(
                "human_gate:H1",
                "claim:C1",
                "human:alice",
                authority="project_owner",
                scope={"claim": "claim:C1", "run": "run:exp_001"},
                rationale="Approved after reviewing exported experiment summary.",
            )
            .add_release_decision("release_decision:R1", "claim:C1", "human:alice")
            .build()
        )
        return AdapterExportResult(
            case=case,
            diagnostics=[AdapterDiagnostic(message="fake export completed")],
        )


def test_runtime_from_builder_is_filesystem_free() -> None:
    case = (
        ClaimCaseBuilder(
            case_id="memory-case",
            source_system="unit-test",
            stress_scenario="wrong_metric_path",
        )
        .add_claim("claim:C1", "A test claim.", state="released")
        .add_evidence("run:exp_001", "run", uri="memory://run/exp_001")
        .add_evidence("metric:exp_001.primary", "metric", metadata={"value": 1.0})
        .add_relation("run:exp_001", "metric:exp_001.primary", "computed", criticality="critical")
        .add_support("metric:exp_001.primary", "claim:C1")
        .build()
    )

    runtime = CairnRuntime.from_case(case)
    plan = runtime.plan_revert("run:exp_001", reason="wrong metric")
    events = runtime.events_from_plan(plan)
    updated = runtime.with_events(events)

    assert {item.id for item in plan.affected} == {
        "run:exp_001",
        "metric:exp_001.primary",
        "claim:C1",
    }
    assert updated.trace("claim:C1").projected_state == "downgraded"


def test_fake_adapter_contract_exports_case(tmp_path: Path) -> None:
    (tmp_path / "experiment_summary.json").write_text('{"primary_metric": 0.91}\n', encoding="utf-8")
    adapter = FakeAutoResearchAdapter()

    assert adapter.detect(tmp_path)
    export = adapter.export_case(tmp_path)
    runtime = CairnRuntime.from_case(export.case)
    plan = runtime.plan_revert("run:exp_001", reason="metric computed on wrong split")
    affected = {item.id: item.action.value for item in plan.affected}

    assert export.diagnostics[0].message == "fake export completed"
    assert affected["claim:C1"] == "downgrade"
    assert affected["human_gate:H1"] == "require_reapproval"
    assert affected["release_decision:R1"] == "reopen_release_decision"
