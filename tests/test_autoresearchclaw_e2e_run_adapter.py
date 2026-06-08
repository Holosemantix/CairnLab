from __future__ import annotations

import json
import shutil
from pathlib import Path

from cairnlab import AutoResearchClawE2ERunAdapter, CairnRuntime


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "autoresearchclaw_e2e_run"


def test_autoresearchclaw_e2e_run_adapter_detects_real_run_layout() -> None:
    adapter = AutoResearchClawE2ERunAdapter()

    assert adapter.detect(FIXTURE)
    assert not adapter.detect(FIXTURE / "stage-14")


def test_autoresearchclaw_e2e_run_adapter_exports_stage_diagnostics_and_metrics() -> None:
    adapter = AutoResearchClawE2ERunAdapter()
    result = adapter.export_case(FIXTURE)
    case = result.case

    assert case.case_id == "autoresearchclaw-e2e:autoresearchclaw_e2e_run"
    assert case.source_system == "autoresearchclaw"
    assert case.native_system_behavior["selected_result_stage"] == "stage-14"
    assert case.expected_cairnlab_behavior["do_not_treat_pipeline_pause_as_release_authority"]
    assert "autoresearchclaw_stage-15_paused" in case.failure_classes
    assert "autoresearchclaw_stage-15_decision_parse_failed" in case.failure_classes
    assert "autoresearchclaw_stage-15_model_error" in case.failure_classes
    assert "claim:C1" in {claim.id for claim in case.claims}
    assert "claim:stage-14.no_dropout/test_accuracy" in {claim.id for claim in case.claims}
    assert "metric:stage-14.spatial_dropout_p30/ece" in {item.id for item in case.evidence}
    assert "artifact:stage-15.stage_health" in {item.id for item in case.evidence}
    assert "artifact:stage-15.decision_structured" in {item.id for item in case.evidence}
    assert any("stage-15 status is paused" in item.message for item in result.diagnostics)
    assert any("structured decision parsing failed" in item.message for item in result.diagnostics)
    assert any("No hitl/approval.json" in item.message for item in result.diagnostics)


def test_autoresearchclaw_e2e_run_adapter_flags_degraded_done_run(tmp_path: Path) -> None:
    run_root = tmp_path / "degraded_done_run"
    shutil.copytree(FIXTURE, run_root)
    (run_root / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "run_id": "rc-degraded",
                "stages_executed": 9,
                "stages_done": 9,
                "stages_paused": 0,
                "stages_failed": 0,
                "degraded": True,
                "from_stage": 15,
                "final_stage": 23,
                "final_status": "done",
                "generated": "2026-06-08T09:18:18+00:00",
            }
        ),
        encoding="utf-8",
    )
    stage_20 = run_root / "stage-20"
    stage_20.mkdir()
    (stage_20 / "quality_report.json").write_text(
        json.dumps({"score_1_to_10": 2, "verdict": "FAIL"}),
        encoding="utf-8",
    )
    stage_22 = run_root / "stage-22"
    stage_22.mkdir()
    (stage_22 / "paper_verification.json").write_text(
        json.dumps(
            {
                "passed": False,
                "severity": "REJECT",
                "strict_violations": 5,
                "fabrication_rate": 0.0507,
            }
        ),
        encoding="utf-8",
    )
    (stage_22 / "sanitization_report.json").write_text(
        json.dumps({"sanitized": True, "numbers_replaced": 79}),
        encoding="utf-8",
    )

    result = AutoResearchClawE2ERunAdapter().export_case(run_root)
    case = result.case
    evidence_ids = {item.id for item in case.evidence}

    assert "artifact:pipeline_summary" in evidence_ids
    assert "artifact:stage-20.quality_report" in evidence_ids
    assert "artifact:stage-22.paper_verification" in evidence_ids
    assert "artifact:stage-22.sanitization_report" in evidence_ids
    assert "autoresearchclaw_pipeline_degraded" in case.failure_classes
    assert "autoresearchclaw_stage-20_quality_gate_fail" in case.failure_classes
    assert "autoresearchclaw_stage-22_paper_verification_reject" in case.failure_classes
    assert "autoresearchclaw_stage-22_sanitized_numbers" in case.failure_classes
    assert any("pipeline is degraded" in item.message for item in result.diagnostics)
    assert any("quality gate verdict is FAIL" in item.message for item in result.diagnostics)
    assert any("paper verification severity is REJECT" in item.message for item in result.diagnostics)
    assert any("sanitization replaced 79 numbers" in item.message for item in result.diagnostics)


def test_autoresearchclaw_e2e_run_adapter_feeds_runtime_revert_plan() -> None:
    adapter = AutoResearchClawE2ERunAdapter()
    runtime = CairnRuntime.from_case(adapter.export_case(FIXTURE).case)

    plan = runtime.plan_revert("run:stage-14", reason="real ARC result summary invalidated")
    affected = {item.id: item.action.value for item in plan.affected}

    assert affected["metric:stage-14.test_accuracy"] == "invalidate"
    assert affected["claim:C1"] == "challenge"
    assert affected["metric:stage-14.no_dropout/test_accuracy"] == "invalidate"
    assert affected["claim:stage-14.no_dropout/test_accuracy"] == "challenge"
