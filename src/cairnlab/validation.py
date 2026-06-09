from __future__ import annotations

from collections import Counter

from .models import ClaimCase, ValidationReport
from .validation_evidence import ValidationEvidenceLedger, evidence_counts


def build_validation_report(
    cases: list[ClaimCase],
    ledger: ValidationEvidenceLedger | None = None,
) -> ValidationReport:
    systems = {case.source_system for case in cases}
    stress = {case.stress_scenario for case in cases}
    claims_sampled = sum(len(case.claims) for case in cases)
    synthetic_cases = sum(1 for case in cases if _is_synthetic_case(case))
    contract_cases = len(cases) - synthetic_cases
    counts: Counter[str] = Counter()
    for case in cases:
        counts.update(case.failure_classes)
    real_counts = evidence_counts(ledger or ValidationEvidenceLedger())

    reasons: list[str] = []
    if real_counts["real_systems_sampled"] < 3:
        reasons.append("fewer than 3 real systems sampled")
    if real_counts["real_framework_runs"] < 6:
        reasons.append("fewer than 6 real framework runs recorded")
    if real_counts["real_tasks"] < 6:
        reasons.append("fewer than 6 real tasks recorded")
    if real_counts["real_material_claims"] < 30:
        reasons.append("fewer than 30 real material claims recorded")
    if len(real_counts["recurring_real_failure_classes"]) < 3:
        reasons.append("fewer than 3 recurring real failure classes")
    if real_counts["real_release_control_failures"] < 1:
        reasons.append("no real release-control failure recorded")

    contract_reasons: list[str] = []
    if len(systems) < 3:
        contract_reasons.append("fewer than 3 systems covered by fixtures/contracts")
    if claims_sampled < 30:
        contract_reasons.append("fewer than 30 fixture/contract claims sampled")
    if len(stress) < 18:
        contract_reasons.append("fewer than 18 stress cases covered")
    recurring = [name for name, count in counts.items() if count >= 2]
    if len(recurring) < 3:
        contract_reasons.append("fewer than 3 recurring fixture/contract failure classes")

    release_relevant = {
        "release_decision_not_reopened",
        "human_scope_not_reopened",
        "downstream_claim_not_invalidated",
        "challenge_or_retraction_missing",
        "release_state_mutable_or_implicit",
    }
    if not any(name in counts for name in release_relevant):
        contract_reasons.append("no fixture/contract release-control failure recorded")
    if contract_reasons:
        reasons.extend(f"contract verification only: {reason}" for reason in contract_reasons)

    recommendation = "go" if not reasons else "continue_sampling"
    return ValidationReport(
        cases_sampled=len(cases),
        systems_sampled=len(systems),
        claims_sampled=claims_sampled,
        stress_cases=len(stress),
        synthetic_cases=synthetic_cases,
        contract_verification_cases=contract_cases,
        real_framework_runs=real_counts["real_framework_runs"],
        real_tasks=real_counts["real_tasks"],
        real_material_claims=real_counts["real_material_claims"],
        real_release_control_failures=real_counts["real_release_control_failures"],
        failure_class_counts=dict(sorted(counts.items())),
        real_failure_class_counts=real_counts["real_failure_class_counts"],
        recommendation=recommendation,
        reasons=reasons,
    )


def validation_report_markdown(report: ValidationReport) -> str:
    lines = [
        "# CairnLab Validation Report",
        "",
        f"- Cases sampled: {report.cases_sampled}",
        f"- Systems sampled: {report.systems_sampled}",
        f"- Claims sampled: {report.claims_sampled}",
        f"- Stress cases: {report.stress_cases}",
        f"- Synthetic cases: {report.synthetic_cases}",
        f"- Contract verification cases: {report.contract_verification_cases}",
        f"- Real framework runs: {report.real_framework_runs}",
        f"- Real tasks: {report.real_tasks}",
        f"- Real material claims: {report.real_material_claims}",
        f"- Real release-control failures: {report.real_release_control_failures}",
        f"- Recommendation: {report.recommendation}",
        "",
        "## Failure Classes",
        "",
    ]
    if report.failure_class_counts:
        for name, count in report.failure_class_counts.items():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Real Failure Classes", ""])
    if report.real_failure_class_counts:
        for name, count in report.real_failure_class_counts.items():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Reasons", ""])
    if report.reasons:
        for reason in report.reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- Go criteria satisfied.")
    lines.append("")
    return "\n".join(lines)


def _is_synthetic_case(case: ClaimCase) -> bool:
    source = case.source_system.lower()
    return source.startswith("synthetic") or source in {"unit-test", "test", "fake-autoresearch"}
