from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "skills" / "autoresearch-landscape-survey" / "references" / "paper-review-remediation-protocol.md"
SKILL = ROOT / "skills" / "autoresearch-landscape-survey" / "SKILL.md"
WRITING_MODULE = (
    ROOT / "skills" / "autoresearch-landscape-survey" / "references" / "paper-writing-quality-module.md"
)

GENERIC_GATES = (
    "G1. Progressive Disclosure Gate",
    "G2. Concrete-Before-Abstract Gate",
    "G3. One-Job Gate",
    "G4. Claim-Evidence Identity Gate",
    "G5. Conceptual And Statistical Precision Gate",
    "G6. Self-Contained Display Gate",
    "G7. Table Semantics Gate",
    "G8. Public-Manuscript Boundary Gate",
    "G9. Single-Source-Of-Truth Numbers Gate",
    "G10. Reader-Test Review Pass",
)


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
    assert 'version: "0.2.9"' in text
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
        "Internal diagnostic-engineering prose gate",
        "diagnostic engineering document",
        "metric, projection, signal, output, score, planner cost",
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
    assert "internal diagnostic-engineering prose" in text
    assert "external-family or old-metric artifacts" in text

def test_protocol_has_structure_first_manuscript_gate():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "## Structure-First Manuscript Remediation Gate" in text
    for phrase in (
        "Structure remediation precedes display work",
        "theory terms mapped to empirical audits",
        "figures used for trends, regions, event rates with uncertainty",
        "while tables retain exact lookup",
        "two diagnostics or a primary metric plus a guard",
        "appendix titles read as supplementary paper sections",
        "main figure/table order mirror the evidence chain",
        "trace to an existing artifact or reproducible",
    ):
        assert phrase in text


def test_skill_entry_mentions_structure_first_gate():
    text = SKILL.read_text(encoding="utf-8")
    assert "Structure-First Manuscript" in text
    assert "structure-before-display" in text
    assert "figure/table conversion mistakes" in text


def test_protocol_has_generic_manuscript_gate_review_pass():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "## Generic Manuscript Gate Review Pass" in text
    for gate in GENERIC_GATES:
        assert gate in text


def test_skill_entry_routes_reviews_through_generic_manuscript_gates():
    text = SKILL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "ten generic manuscript gates" in normalized
    assert "progressive disclosure" in normalized
    assert "display accessibility" in normalized
    assert "numeric traceability" in normalized


def test_protocol_generic_gate_pass_states_review_obligations():
    text = PROTOCOL.read_text(encoding="utf-8")
    for phrase in (
        "Report pass, warn, or fail for each",
        "are not-run, unavailable, inconclusive,",
        "training seeds versus evaluation episodes",
        "Flag every absolute endpoint that is written as",
        "grayscale, color-vision, final-size, and",
        "each row and column carry one comparison",
        "repository evidence and provenance rather than in the paper",
        "machine-readable artifact",
        "unfamiliar-reader summary test",
        "cross-reference and link validation",
        "is not a completed pass",
    ):
        assert phrase in text


def test_protocol_classifies_generic_gate_failures():
    text = PROTOCOL.read_text(encoding="utf-8")
    for phrase in (
        "are claim-calibration failures",
        "Do not resolve them by adding hedging language",
        "block the numeric claims they touch",
        "configurable thresholds in the",
        "record the override",
    ):
        assert phrase in text
    assert "pass, warn, or fail for `G1` through `G10`" in text


def test_generic_gate_detail_lives_in_one_reference():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    module = WRITING_MODULE.read_text(encoding="utf-8")
    for text in (protocol, module):
        assert "writing-quality-checklist.md" in text
        for gate in GENERIC_GATES:
            assert gate in text
    assert "Do not restate their text here" in protocol
    assert "Configurable Gate Thresholds" in module
