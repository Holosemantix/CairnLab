from pathlib import Path
import importlib.util
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "skills" / "autoresearch-landscape-survey" / "scripts" / "check_new_project.py"
CLASSIFY = ROOT / "skills" / "autoresearch-landscape-survey" / "scripts" / "classify_project_from_yaml.py"
REGISTRY = ROOT / "data" / "project_registry.yaml"
TAXONOMY = ROOT / "data" / "taxonomy.yaml"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_autoresearchclaw_duplicate():
    mod = load_module(SCRIPT, "check_new_project")
    registry = mod.load_structured(REGISTRY)
    new = mod.load_structured(ROOT / "skills" / "autoresearch-landscape-survey" / "examples" / "new_project_autoresearchclaw.yaml")
    result = mod.analyze(new, registry)
    assert result["duplicate_status"] == "already_surveyed"
    assert any(m["id"] == "autoresearchclaw" for m in result["exact_matches"])


def test_hypothetical_core_recommended_deep_or_medium():
    mod = load_module(SCRIPT, "check_new_project")
    registry = mod.load_structured(REGISTRY)
    new = mod.load_structured(ROOT / "skills" / "autoresearch-landscape-survey" / "examples" / "new_project_hypothetical_evidence_court.yaml")
    result = mod.analyze(new, registry)
    assert result["recommended_depth"] in {"deep_or_medium", "update_existing"}
    assert any(p["id"] in {"evibound", "scientistone", "aris", "paperbench"} for p in result["closest_projects"])


def test_classifier_infers_facets():
    mod = load_module(CLASSIFY, "classify_project_from_yaml")
    taxonomy = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    project = yaml.safe_load((ROOT / "skills" / "autoresearch-landscape-survey" / "examples" / "new_project_hypothetical_evidence_court.yaml").read_text(encoding="utf-8"))
    facets = mod.infer_facets(project, taxonomy)
    assert "claim_ledger" in facets.get("accountability_features", [])
    assert "paper_to_reproduction" in facets.get("workflow_scopes", [])


def test_registry_entries_have_facets():
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert reg["version"] in {"0.2.0", "0.3.0"}
    assert all("facets" in p for p in reg["projects"])
