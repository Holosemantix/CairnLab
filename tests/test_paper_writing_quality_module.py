from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "autoresearch-landscape-survey" / "SKILL.md"
MODULE = (
    ROOT
    / "skills"
    / "autoresearch-landscape-survey"
    / "references"
    / "paper-writing-quality-module.md"
)
STANDALONE_SKILL = ROOT / "skills" / "paper-writing-quality" / "SKILL.md"
STANDALONE_REFERENCE = (
    ROOT / "skills" / "paper-writing-quality" / "references" / "writing-quality-checklist.md"
)
STANDALONE_OPENAI_YAML = ROOT / "skills" / "paper-writing-quality" / "agents" / "openai.yaml"
PROTOCOL = (
    ROOT
    / "skills"
    / "autoresearch-landscape-survey"
    / "references"
    / "paper-review-remediation-protocol.md"
)


def test_skill_routes_paper_writing_to_quality_module() -> None:
    text = SKILL.read_text(encoding="utf-8")

    assert 'version: "0.2.7"' in text
    assert "paper-writing-quality-module.md" in text
    assert "writing-quality ledgers" in text.lower()
    assert "non-authoritative evidence" in text
    assert "claim lifecycle transitions" in text


def test_standalone_skill_routes_to_quality_checklist() -> None:
    skill_text = STANDALONE_SKILL.read_text(encoding="utf-8")
    reference_text = STANDALONE_REFERENCE.read_text(encoding="utf-8")
    openai_yaml = STANDALONE_OPENAI_YAML.read_text(encoding="utf-8")

    required_skill_phrases = [
        "name: paper-writing-quality",
        "Use this skill when drafting, rewriting, polishing, or reviewing scientific manuscripts",
        "references/writing-quality-checklist.md",
        "This paper is X, not Y",
        "positive claim plus boundary claim",
        "internal diagnostic-engineering prose",
        "top-conference structure pass",
        "writing-quality ledger",
    ]
    missing_skill = [phrase for phrase in required_skill_phrases if phrase not in skill_text]

    required_reference_phrases = [
        "Project-Specific Writing Constraints",
        "The screen enriches plateau members, but does not rank inside plateau",
        "Precision/recall are the primary readouts; presence is reported for block coverage",
        "not paper-facing evidence",
    ]
    missing_reference = [
        phrase for phrase in required_reference_phrases if phrase not in reference_text
    ]

    assert missing_skill == []
    assert missing_reference == []
    assert "Use $paper-writing-quality" in openai_yaml


def test_writing_module_preserves_cairnlab_authority_boundary() -> None:
    text = MODULE.read_text(encoding="utf-8")

    required = [
        "No artifact, no claim",
        "non-authoritative evidence",
        "verifier certificates",
        "human gates",
        "append-only transition events",
        "not a verifier certificate or transition event",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_writing_module_records_reusable_autoresearch_patterns() -> None:
    text = MODULE.read_text(encoding="utf-8")

    required = [
        "scientific-writing",
        "venue-templates",
        "PaperOrchestra",
        "ARIS paper-writing",
        "AutoResearchClaw",
        "data-to-paper",
        "Agent Laboratory",
        "AI Scientist",
        "negotiated acceptance contract",
        "paper-claim audit",
        "backward traceability",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_writing_module_makes_direct_reuse_decision() -> None:
    text = MODULE.read_text(encoding="utf-8")

    required = [
        "Open-Source Reuse Decision",
        "The closest direct open-source base is PaperOrchestra",
        "external artifact producer",
        "Do not vendor PaperOrchestra directly into CairnLab authority code",
        "ARIS is the best direct base for a Markdown skill workflow",
        "Current implementation decision",
        "future executable-engine candidate",
        "non-authoritative artifact producer",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_writing_module_covers_user_reported_failure_modes() -> None:
    text = MODULE.read_text(encoding="utf-8")

    required = [
        "Abstract Gate",
        "no unrelated motivation",
        "paragraph",
        "reviewer-defense prose",
        "acronym",
        "no more than three nonstandard acronyms",
        "formula",
        "compiled PDF",
        "fonts",
        "overfull",
        "table precision",
        "AI-generic phrasing",
        "repeating the same thesis",
        "venue",
        "page limit",
        "anonymization",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_writing_module_requires_checkable_contract_and_ledger() -> None:
    text = MODULE.read_text(encoding="utf-8")

    required = [
        "Phase 0: Writing Contract",
        "checkable after compilation",
        "Every acronym used in the abstract is defined before first use",
        "Final Writing Quality Ledger",
        "writing_readiness: ready | minor_revision | major_revision | blocked",
        "abstract_gate: pass | warn | fail",
        "claim_frame_gate: pass | warn | fail",
        "terminology_gate: pass | warn | fail",
        "experiment_narrative_gate: pass | warn | fail",
        "formula_layout_gate: pass | warn | fail",
        "appendix_reading_gate: pass | warn | fail",
        "pdf_layout_gate: pass | warn | fail",
        "remove_merge_demote",
        "blocked_claims",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_writing_module_codifies_claim_framing_and_plateau_language() -> None:
    text = MODULE.read_text(encoding="utf-8")

    required = [
        "This paper is X, not Y",
        "This is a diagnostic study, not a method paper",
        "The screen enriches plateau members, but does not rank inside plateau",
        "positive claim and a boundary claim",
        "plateau, range, membership, region, and screen",
        "avoid optimal, best selector, and rank inside plateau",
        "within plateau",
        "not treated as meaningful ordering",
        "claim must be rewritten instead of padded with disclaimers",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_writing_module_codifies_terminology_and_experiment_narrative() -> None:
    text = MODULE.read_text(encoding="utf-8")

    required = [
        "score is the numeric evaluation",
        "screen returns a set",
        "view is a reporting perspective",
        "PCC, CRA, MAF, and",
        "ACPC-H/trans",
        "Avoid legacy",
        "artifact history",
        "behavioral recovery, plateau membership, planner-side sensitivity",
        "selectivity guard",
        "question, protocol, result, boundary",
        "high-std or MAF-only reference",
        "presence hit is saturated",
        "precision/recall before presence",
        "X changes from a to b; Y remains unchanged",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_writing_module_codifies_caveat_table_and_appendix_rules() -> None:
    text = MODULE.read_text(encoding="utf-8")

    required = [
        "same caveat appears at most three times",
        "bound to a positive claim",
        "saturated metrics do not appear first",
        "Precision/recall are the primary readouts; presence is reported for block coverage",
        "This is not evidence of selector dominance",
        "high-std reference is a coarse intervention-order screen",
        "not a plateau-internal ranker",
        "Appendix Constraints",
        "Reading:",
        "not paper-facing evidence",
        "appendix extends evidence",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_review_protocol_integrates_pure_writing_quality_gate() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")

    required = [
        "paper-writing-quality-module.md",
        "Pure Writing Quality Gate",
        "checkable writing contract",
        "one central thesis",
        "no unrelated content",
        "no acronym or notation overload",
        "no more than three nonstandard acronyms in the abstract",
        "no visible PDF overflow",
        "compiled-PDF inspection",
        "Writing-only remediation",
        "cannot raise the evidence ceiling",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_review_protocol_checks_project_specific_writing_constraints() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")

    required = [
        "This paper is X, not Y",
        "positive claim and a boundary claim",
        "plateau/range results",
        "point-best selector language",
        "score means numeric evaluation",
        "screen returns a set",
        "question, protocol, result, boundary",
        "most discriminative metric first",
        "not saturated metrics",
        "This is not evidence of selector dominance",
        "high-std",
        "not a plateau-internal",
        "same caveat appears at most three times",
        "Reading:",
        "not paper-facing evidence",
        "Top-conference structure pass",
        "Remediation audit tables",
        "Supplementary diagnostic analyses",
        "Fixed-pool candidate-stability analysis",
    ]

    missing = [phrase for phrase in required if phrase not in text]

    assert missing == []


def test_writing_module_catches_claim_hygiene_table_and_appendix_failures() -> None:
    module_text = MODULE.read_text(encoding="utf-8")
    standalone_text = STANDALONE_REFERENCE.read_text(encoding="utf-8")

    required = [
        "The claim is",
        "readouts:",
        "internal diagnostic-engineering prose pass",
        "diagnostic engineering document",
        "projection, metric, signal",
        "output, score, planner cost",
        "baseline stress",
        "no-noise checkpoint score under blur",
        "remove interpretation columns",
        "appendix table repeats the same rows",
        "method-family replication",
        "old-metric",
        "development-grid artifacts",
    ]

    for text in (module_text, standalone_text):
        missing = [phrase for phrase in required if phrase not in text]
        assert missing == []


def test_writing_module_catches_internal_structure_and_caption_language() -> None:
    module_text = MODULE.read_text(encoding="utf-8")
    standalone_text = STANDALONE_REFERENCE.read_text(encoding="utf-8")
    skill_text = STANDALONE_SKILL.read_text(encoding="utf-8")
    openai_yaml = STANDALONE_OPENAI_YAML.read_text(encoding="utf-8")

    required = [
        "Top-Conference Structure And Caption Gate",
        "Top-conference structure pass",
        "table-of-contents structure",
        "section, subsection, paragraph headings",
        "figure/table captions",
        "appendix titles",
        "not an internal review plan, remediation log, or engineering record",
        "Remediation audit tables",
        "Supplementary diagnostic analyses",
        "Bounded unseen-stressor check",
        "Evaluation under bounded non-Gaussian stressors",
        "Retained-summary fixed-pool top-1 audit",
        "Fixed-pool candidate-stability analysis",
        "Full-sweep sample-level fixed-pool event-rate audit",
        "Full-sweep fixed-pool event-rate calibration",
        "reader-facing section/caption structure",
    ]

    for source in (module_text, standalone_text):
        missing = [phrase for phrase in required[:-1] if phrase not in source]
        assert missing == []

    assert "reader-facing section/caption structure" in skill_text
    assert "section/TOC structure" in openai_yaml
