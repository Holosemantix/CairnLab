from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / "skills" / "paper-writing-quality" / "SKILL.md"
CHECKLIST = (
    ROOT / "skills" / "paper-writing-quality" / "references" / "writing-quality-checklist.md"
)
MODULE = (
    ROOT
    / "skills"
    / "autoresearch-landscape-survey"
    / "references"
    / "paper-writing-quality-module.md"
)
PROTOCOL = (
    ROOT
    / "skills"
    / "autoresearch-landscape-survey"
    / "references"
    / "paper-review-remediation-protocol.md"
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


def _generic_gate_section(text: str) -> str:
    start = text.index("## Generic Manuscript Gates")
    end = text.index("## Reusable Evidence-Pattern Constraints", start)
    return text[start:end]


def test_skill_mentions_structure_first_and_display_conversion():
    text = SKILL.read_text(encoding="utf-8")
    assert "structure-first remediation" in text
    assert "theory-to-evidence mapping" in text
    assert "figure/table conversion rules" in text


def test_checklist_has_structure_first_plan_gate():
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "### Structure-First Manuscript Plan Gate" in text
    for phrase in (
        "before creating new figures, converting tables",
        "theory-to-evidence map",
        "main text displays answer must-read questions",
        "convert tables to figures only",
        "the claim requires two diagnostics or conditions jointly",
        "order main figures and tables by the evidence chain",
        "every promoted figure/table",
    ):
        assert phrase in text


def test_checklist_keeps_paper_specific_names_out_of_generic_rules():
    text = CHECKLIST.read_text(encoding="utf-8")
    forbidden = (
        "ACPC",
        "ATR",
        "SMPR",
        "LeWM",
        "PLDM",
        "PushT",
        "TwoRoom",
        "Reacher",
        "Cube",
        "PCC",
        "CRA",
        "MAF",
    )
    for token in forbidden:
        assert token not in text


def test_checklist_defines_every_named_generic_gate():
    section = _generic_gate_section(CHECKLIST.read_text(encoding="utf-8"))
    for gate in GENERIC_GATES:
        assert f"### {gate}" in section


def test_generic_gates_state_their_required_standards():
    section = _generic_gate_section(CHECKLIST.read_text(encoding="utf-8"))
    required = (
        # G1 progressive disclosure and layered readability
        "a newcomer can state the problem, the",
        "technical completeness never requires every internal log",
        # G2 concrete before abstract
        "before or immediately after its definition",
        "one concept has one stable name",
        # G3 one job per unit
        "one job per section, per paragraph, and per display",
        "claim or reason, then evidence, then boundary",
        "project chronology is removed",
        "reviewer-response prose is removed",
        # G4 claim/evidence identity
        "`not-run`",
        "`unavailable`",
        "`inconclusive`",
        "`failed`",
        "training seeds and evaluation",
        "checkpoint-selection rule",
        "never convert absolute endpoint performance into a method effect",
        # G5 conceptual and statistical precision
        "Identifiability, realizability, sufficiency, and well-posedness are properties",
        "necessary condition is not a sufficient condition",
        "the verb to the evidence: observe, is consistent with, localizes",
        # G6 self-contained displays
        "the question the display answers",
        "uncertainty semantics",
        "no color-only encoding",
        "color-vision-deficiency",
        "legible at final printed size",
        "stable axes, panel order, and legend placement",
        # G7 table semantics
        "one row and one column carry one comparison identity",
        "absolute endpoints with matched deltas in the same visual grammar",
        "resized below the body text's readable size",
        "no interpretation or status columns",
        # G8 public manuscript vs internal record
        "run dates, run identifiers, job names",
        "artifact hashes, file paths, storage locations, and artifact manifests",
        "The appendix is still reader-facing paper, not storage",
        # G9 single source of truth
        "Hand-transcribed numbers are a defect",
        "rerunnable consistency check",
        # G10 reader test
        "Unfamiliar-reader summary test",
        "Acronym and notation scan",
        "Display-only scan",
        "Cross-reference and link validation",
    )
    for phrase in required:
        assert phrase in section


def test_generic_gates_separate_policy_from_configurable_thresholds():
    text = CHECKLIST.read_text(encoding="utf-8")
    section = _generic_gate_section(text)
    assert "### Configurable Gate Thresholds" in section
    assert "The gates above are policy" in section
    for parameter in (
        "audience_level:",
        "abstract_max_nonstandard_acronyms:",
        "caveat_repetition_budget:",
        "main_text_must_read_evidence_layers_max:",
        "display_accessibility_checks:",
        "numeric_source_of_truth:",
    ):
        assert parameter in section
    assert "threshold_overrides:" in text


def test_generic_gates_stay_domain_agnostic():
    section = _generic_gate_section(CHECKLIST.read_text(encoding="utf-8"))
    for token in ("ContextWorld", "COJA", "world model", "plateau", "selector"):
        assert token not in section


def test_conditional_constraints_do_not_claim_universal_scope():
    text = CHECKLIST.read_text(encoding="utf-8")
    assert "## Reusable Evidence-Pattern Constraints" in text
    assert "conditional rather than universal" in text
    assert "do not import its vocabulary or assumptions" in text


def test_ledger_exposes_generic_gate_status_keys():
    text = CHECKLIST.read_text(encoding="utf-8")
    for key in (
        "progressive_disclosure_gate: pass | warn | fail",
        "concrete_before_abstract_gate: pass | warn | fail",
        "one_job_gate: pass | warn | fail",
        "claim_evidence_identity_gate: pass | warn | fail",
        "conceptual_precision_gate: pass | warn | fail",
        "display_self_containment_gate: pass | warn | fail",
        "table_semantics_gate: pass | warn | fail",
        "public_manuscript_boundary_gate: pass | warn | fail",
        "numeric_source_of_truth_gate: pass | warn | fail",
        "reader_test_gate: pass | warn | fail",
    ):
        assert key in text


def test_accepted_abstract_baseline_gate_is_operational():
    text = CHECKLIST.read_text(encoding="utf-8")

    for phrase in (
        "### Accepted-Abstract Baseline And Reversion Gate",
        "exact-restoration task",
        "source diff",
        "rejected candidate",
        "supporting experiment",
        "factorial analysis",
        "defensive boundary",
        "analysis plumbing",
        "accepted_abstract_baseline_gate:",
        "abstract_revision_decision:",
        "abstract_baseline_source:",
        "abstract_exact_reversion:",
    ):
        assert phrase in text


def test_skill_advertises_generic_gates_and_reader_test_step():
    text = SKILL.read_text(encoding="utf-8")
    assert "## Generic Manuscript Gates" in text
    for gate in GENERIC_GATES:
        assert gate in text
    assert "Configurable Gate Thresholds" in text
    assert "Run `G10. Reader-Test Review Pass` on the compiled artifact." in text
    assert "including `G1` through `G10`" in text


def test_generic_gate_names_are_consistent_across_files():
    checklist = CHECKLIST.read_text(encoding="utf-8")
    module = MODULE.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    skill = SKILL.read_text(encoding="utf-8")

    for text in (checklist, module, protocol, skill):
        assert "Generic Manuscript Gate" in text
        for gate in GENERIC_GATES:
            assert gate in text

    # The operational detail lives in one file; the others point at it.
    for pointer_text in (module, protocol):
        assert "writing-quality-checklist.md" in pointer_text


def test_display_labels_preserve_the_measured_quantity():
    checklist = CHECKLIST.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    skill = SKILL.read_text(encoding="utf-8")

    assert "axis labels that name the measured quantity" in skill
    assert "axis labels name the measured quantity" in checklist
    assert "preserve what is measured" in checklist
    assert "A shortened label" in protocol
    assert "procedural detail belongs in the caption" in protocol
