from __future__ import annotations

from collections import Counter

from .models import ClaimCase, ValidationReport


def build_validation_report(cases: list[ClaimCase]) -> ValidationReport:
    systems = {case.source_system for case in cases}
    stress = {case.stress_scenario for case in cases}
    claims_sampled = sum(len(case.claims) for case in cases)
    counts: Counter[str] = Counter()
    for case in cases:
        counts.update(case.failure_classes)

    reasons: list[str] = []
    if len(systems) < 3:
        reasons.append("fewer than 3 systems sampled")
    if claims_sampled < 30:
        reasons.append("fewer than 30 material claims sampled")
    if len(stress) < 18:
        reasons.append("fewer than 18 stress cases sampled")
    recurring = [name for name, count in counts.items() if count >= 2]
    if len(recurring) < 3:
        reasons.append("fewer than 3 recurring failure classes")

    release_relevant = {
        "release_decision_not_reopened",
        "human_scope_not_reopened",
        "downstream_claim_not_invalidated",
        "challenge_or_retraction_missing",
        "release_state_mutable_or_implicit",
    }
    if not any(name in counts for name in release_relevant):
        reasons.append("no release-control failure recorded")

    recommendation = "go" if not reasons else "continue_sampling"
    return ValidationReport(
        cases_sampled=len(cases),
        systems_sampled=len(systems),
        claims_sampled=claims_sampled,
        stress_cases=len(stress),
        failure_class_counts=dict(sorted(counts.items())),
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
    lines.extend(["", "## Reasons", ""])
    if report.reasons:
        for reason in report.reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- Go criteria satisfied.")
    lines.append("")
    return "\n".join(lines)
