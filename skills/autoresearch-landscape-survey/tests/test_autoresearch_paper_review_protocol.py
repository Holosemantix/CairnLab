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
    assert 'version: "0.2.6"' in text
    assert "score-disagreement ledger" in text
    assert "separate main-track and diagnostic" in text
    assert "corrected score ceiling" in text
    assert "Weak-Reject Diagnostic-Paper Calibration Gate" in text


def test_protocol_has_weak_reject_diagnostic_paper_calibration_gate():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "## Weak-Reject Diagnostic-Paper Calibration Gate" in text
    for phrase in (
        "Method-paper ceiling",
        "Matched-stressor ceiling",
        "Selector-increment ceiling",
        "Theory-link ceiling",
        "Semantic-proxy ceiling",
        "External-baseline ceiling",
        "Appendix-burden ceiling",
        "Fixed-checkpoint ceiling",
    ):
        assert phrase in text


def test_protocol_forces_remediation_classification_before_edits():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "Classification is mandatory before edits" in text
    for bucket in (
        "`no-retraining`",
        "`new evaluation/diagnostic with fixed checkpoints`",
        "`retraining-required`",
        "`writing-only`",
    ):
        assert bucket in text


def test_protocol_prevents_fixed_checkpoint_strong_method_inflation():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "with fixed checkpoints and no retraining" in text
    assert "blockers instead of inflating the score" in text
    assert "should normally remain at `weak_reject` to" in text
    assert "track score may be higher" in text

def test_protocol_requires_subtractive_remediation_gate():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "## Subtractive Remediation Gate" in text
    required = (
        "what should be removed, not only what can be added",
        "list deletion candidates",
        "removed, merged, demoted, or moved to appendix",
        "main-text burden",
        "defensive patching",
        "fake novelty",
        "remove/merge/demote",
        "A remediation round is incomplete if it only lists additions",
    )
    for phrase in required:
        assert phrase in text


def test_skill_entry_mentions_subtractive_remediation_gate():
    text = SKILL.read_text(encoding="utf-8")
    assert "Subtractive Remediation Gate" in text
    assert "deleted, merged, demoted, or moved to appendix" in text
    assert "before adding more caveats, diagnostics, tables, or reviewer patches" in text


def test_protocol_has_claim_hygiene_writing_gate():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "## Claim-Hygiene Writing Gate" in text
    required = (
        "The claim is",
        "readouts:",
        "table headers",
        "No table self-interpretation columns",
        "Main/appendix duplication check",
        "External-family evidence gate",
        "Old metric quarantine",
        "old-metric",
        "current core-evidence standard",
    )
    for phrase in required:
        assert phrase in text


def test_skill_entry_mentions_claim_hygiene_writing_gate():
    text = SKILL.read_text(encoding="utf-8")
    assert "Claim-Hygiene Writing Gate" in text
    assert "abstracts must avoid internal audit phrases" in text
    assert "duplicated main/appendix tables" in text
    assert "interpretation columns must move to prose" in text
    assert "external-family or old-metric artifacts" in text
