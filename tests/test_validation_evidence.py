from __future__ import annotations

from cairnlab import ClaimCaseBuilder
from cairnlab.validation import build_validation_report
from cairnlab.validation_evidence import ValidationEvidenceLedger


def test_empty_validation_evidence_keeps_recommendation_in_sampling() -> None:
    report = build_validation_report([])

    assert report.recommendation == "continue_sampling"
    assert report.real_framework_runs == 0
    assert "fewer than 3 real systems sampled" in report.reasons
    assert "no real release-control failure recorded" in report.reasons


def test_fixture_only_cases_do_not_count_as_real_validation() -> None:
    cases = [
        ClaimCaseBuilder(
            case_id="fixture-only",
            source_system="synthetic_mlflow_like",
            stress_scenario="wrong_metric_path",
        )
        .add_claim("claim:C1", "A synthetic claim.")
        .add_failure_class("release_decision_not_reopened")
        .build()
    ]

    report = build_validation_report(cases)

    assert report.synthetic_cases == 1
    assert report.real_framework_runs == 0
    assert report.recommendation == "continue_sampling"


def test_real_evidence_ledger_can_satisfy_go_criteria_when_contract_coverage_is_present() -> None:
    report = build_validation_report(_contract_cases(), ledger=_sufficient_real_ledger())

    assert report.real_framework_runs == 6
    assert report.real_tasks == 6
    assert report.real_material_claims == 30
    assert report.real_release_control_failures == 6
    assert report.recommendation == "go"


def _contract_cases():
    cases = []
    failure_classes = [
        "release_decision_not_reopened",
        "human_scope_not_reopened",
        "downstream_claim_not_invalidated",
    ]
    for index in range(18):
        builder = ClaimCaseBuilder(
            case_id=f"contract-{index}",
            source_system=f"system-{index % 3}",
            stress_scenario=f"stress-{index}",
        )
        for claim_index in range(2):
            builder.add_claim(f"claim:{index}:{claim_index}", "A contract validation claim.")
        for failure_class in failure_classes:
            builder.add_failure_class(failure_class)
        cases.append(builder.build())
    return cases


def _sufficient_real_ledger() -> ValidationEvidenceLedger:
    return ValidationEvidenceLedger.model_validate(
        {
            "systems_sampled": [
                {
                    "name": f"real-system-{system_index}",
                    "real_or_fixture": "real",
                    "runs": [
                        {
                            "id": f"run-{system_index}-{run_index}",
                            "real_or_fixture": "real",
                        }
                        for run_index in range(2)
                    ],
                }
                for system_index in range(3)
            ],
            "tasks": [
                {
                    "id": f"task:{index}",
                    "source_system": f"real-system-{index % 3}",
                    "real_or_fixture": "real",
                    "claims_recorded": 5,
                    "material_claims_recorded": 5,
                }
                for index in range(6)
            ],
            "failure_classes": [
                {
                    "name": "release_decision_not_reopened",
                    "real_occurrences": 2,
                    "systems_seen": ["real-system-0", "real-system-1"],
                    "release_control_relevant": True,
                    "plausibly_reduced_by_cairnlab": True,
                },
                {
                    "name": "reviewer_consensus_without_verifier",
                    "real_occurrences": 2,
                    "systems_seen": ["real-system-1", "real-system-2"],
                    "release_control_relevant": True,
                    "plausibly_reduced_by_cairnlab": True,
                },
                {
                    "name": "material_dissent_suppressed",
                    "real_occurrences": 2,
                    "systems_seen": ["real-system-0", "real-system-2"],
                    "release_control_relevant": True,
                    "plausibly_reduced_by_cairnlab": True,
                },
            ],
        }
    )
