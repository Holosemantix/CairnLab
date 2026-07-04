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
