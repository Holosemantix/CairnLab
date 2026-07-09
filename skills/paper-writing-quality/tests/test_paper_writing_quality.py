from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / "skills" / "paper-writing-quality" / "SKILL.md"
CHECKLIST = (
    ROOT / "skills" / "paper-writing-quality" / "references" / "writing-quality-checklist.md"
)


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
