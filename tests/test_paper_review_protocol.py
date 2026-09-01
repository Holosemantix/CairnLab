from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "autoresearch-landscape-survey" / "SKILL.md"
PROTOCOL = (
    ROOT
    / "skills"
    / "autoresearch-landscape-survey"
    / "references"
    / "paper-review-remediation-protocol.md"
)


def test_autoresearch_skill_exposes_paper_review_protocol() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")

    assert "paper-review-remediation-protocol.md" in skill_text
    assert "non-authoritative evidence" in skill_text


def test_paper_review_protocol_covers_required_review_gates() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    required_phrases = [
        "No artifact, no claim",
        "non-authoritative review evidence",
        "Every round is an independent review",
        "Reset the score",
        "newly submitted today",
        "Round Execution Order",
        "assign the recommendation, score, and confidence before proposing fixes",
        "classify each fix as `no-retraining`, `new evaluation/diagnostic with fixed checkpoints`, `retraining-required`, or `writing-only`",
        "list deletion candidates for content that should be removed, merged, demoted, or moved to appendix",
        "Execute the feasible `no-retraining`, `new evaluation/diagnostic with fixed checkpoints`, `writing-only`, and subtractive cleanup items",
        "Subtractive Remediation Gate",
        "what should be removed, not only what can be added",
        "remove/merge/demote",
        "A remediation round is incomplete if it only lists additions",
        "Internal diagnostic-engineering prose gate",
        "Top-conference structure pass",
        "not an internal review plan, remediation log, or",
        "Remediation audit tables",
        "Supplementary diagnostic analyses",
        "Fixed-pool candidate-stability analysis",
        "diagnostic engineering document",
        "readout",
        "metric, projection, signal, output, score, planner cost",
        "Leave `retraining-required` items as explicit blockers or future work",
        "re-review the changed manuscript as a fresh submission before updating the score",
        "abstract",
        "paragraph",
        "sentence",
        "non-AI-generated",
        "fake novelty",
        "over-claim",
        "under-claim",
        "theory-to-experiment bridge",
        "finite-sample",
        "sampling randomness",
        "do not emphasize exact best",
        "fixed checkpoints",
        "new diagnostics",
        "new evaluation",
        "author order",
        "web lookup",
        "primary sources",
        "double-blind",
        "reviewer finding -> concrete issue -> patch/eval/diagnostic -> re-review",
        "producer, reviewer, verifier, and human-gate roles",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in text]

    assert missing == []


def test_review_protocol_requires_exact_accepted_abstract_reversion() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for phrase in (
        "## Accepted-Abstract Baseline And Reversion Gate",
        "More complete is not automatically better",
        "An explicit restore, revert, or return-to-version request",
        "wording and sentence order match it",
        "authorized mechanical exception",
        "A hybrid, polished, or partially preserved candidate fails",
        "inflates reviewer expectations",
        "enlarges claim surface",
    ):
        assert phrase in normalized
