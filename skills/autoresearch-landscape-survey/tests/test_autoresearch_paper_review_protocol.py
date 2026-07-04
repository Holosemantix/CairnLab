from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "skills" / "autoresearch-landscape-survey" / "references" / "paper-review-remediation-protocol.md"
SKILL = ROOT / "skills" / "autoresearch-landscape-survey" / "SKILL.md"


def test_protocol_requires_score_drift_calibration_gate():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "## Score-Drift Calibration Gate" in text
    for root_cause in (
        "improvement bias",
        "track mismatch",
        "evidence substitution",
        "local overfitting",
        "future-work optimism",
        "plateau/selector optimism",
        "matched-stressor optimism",
    ):
        assert root_cause in text


def test_protocol_forces_track_separated_scores_and_disagreement_ledger():
    text = PROTOCOL.read_text(encoding="utf-8")
    required = (
        "main-conference method/general track",
        "diagnostic, empirical-analysis, or benchmark track",
        "workshop/resource positioning",
        "score-disagreement ledger",
        "prior internal recommendation and score",
        "fresh external/user recommendation and score",
        "target track assumed by each score",
        "corrected score ceiling",
    )
    for phrase in required:
        assert phrase in text


def test_protocol_prevents_large_unverified_score_jump():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "must not exceed the lower fresh-review score by" in text
    assert "more than 0.5 points" in text
    assert "unless a verified new artifact" in text
    assert "Writing-only claim calibration" in text
    assert "cannot by itself convert a weak-reject method-track paper" in text


def test_skill_entry_mentions_score_disagreement_gate():
    text = SKILL.read_text(encoding="utf-8")
    assert 'version: "0.2.2"' in text
    assert "score-disagreement ledger" in text
    assert "separate main-track and diagnostic" in text
    assert "corrected score ceiling" in text
