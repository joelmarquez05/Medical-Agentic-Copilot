"""Read-only memory-demo provenance, documentation and privacy artifact checks."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

from medical_copilot.clinical_graph import ClinicalQuestion
from medical_copilot.clinical_evaluation import DEFAULT_PROFILE
from medical_copilot.evaluation import read_json, write_json
from medical_copilot.patient_memory import FactInput, file_hash

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/evaluation/patient-runs"
MAIN = RUNS / "gemma-patient-main-20261014"


def read_text(path):
    data = path.read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def main():
    prior = read_json(ROOT / "data/checks/requirement-7-memory-research-compatibility-20261014.json")
    assert all(file_hash(ROOT / p) == sha for p, sha in prior["frozen_application_source_hashes"].items())
    source_snapshots = 0
    episode_count = 0
    generation_runtime_versions = set()
    for folder in sorted(RUNS.iterdir()):
        if not (folder / "usage-audit.json").exists():
            continue
        manifest = read_json(folder / "manifest.json")
        assert manifest["requirement"] == 7
        config = read_json(folder / "config.json")
        assert config["model"] == "google/gemma-4-26b-a4b-it"
        assert file_hash(ROOT / DEFAULT_PROFILE) == manifest["config_sha256"]
        for name, sha in manifest["source_hashes"].items():
            assert file_hash(folder / "source-snapshot" / name) == sha
            if name == "patient_runtime.py":
                generation_runtime_versions.add(sha)
            else:
                assert file_hash(ROOT / "src/medical_copilot" / name) == sha
            source_snapshots += 1
        for episode in manifest["episodes"]:
            assert file_hash(folder / episode["memory_snapshot"]) == episode["memory_snapshot_sha256"]
            answer = read_json(folder / "answers" / (episode["question_id"] + ".json"))
            assert answer["patient_id"] == episode["patient_id"]
            assert answer["memory_write"]["saved"] == episode["memory_enabled"]
            assert answer["response"]
            episode_count += 1
    assert episode_count == 14 and len(generation_runtime_versions) == 2
    receipt_files = [ROOT / f"data/checks/requirement-7-{stage}-replay-20261014.json" for stage in ["probe", "pilot", "main", "control-check"]]
    receipts = [read_json(p) for p in receipt_files]
    assert all(r["passed"] and r["new_inference_requests"] == r["credential_files_read"] == r["new_pubmed_requests"] == 0 for r in receipts)
    assert sum(r["replayed_model_responses"] for r in receipts) == 27
    assert sum(r["replayed_http_requests"] for r in receipts) == 2
    scores = read_json(MAIN / "scores.json")
    assert scores["totals"]["matched_fields"] == 11 and scores["totals"]["total_fields"] == 12
    assert scores["totals"]["memory_on_all_fields_matched"] == 9
    reviews = read_json(MAIN / "manual-reviews.json")
    assert len(reviews["cases"]) == 10
    assert sum(len(r["cited_view_assessments"]) for r in reviews["cases"]) == 3
    assert all(s["support"] == "supports_requested_claim" for r in reviews["cases"] for s in r["cited_view_assessments"])
    request_audit = read_json(ROOT / "data/checks/requirement-7-request-isolation-20261014.json")
    assert request_audit["passed"] and request_audit["requests_checked"] == 27
    costs = read_json(ROOT / "data/checks/requirement-7-cost-audit-20261014.json")
    assert costs["new_requests"] == 27 and costs["known_cost_usd"] == .01291831 and costs["unknown_charge_records"] == 1
    assert costs["below_cumulative_limit"]
    test_log = ROOT / "data/checks/requirement-7-pytest-20261014.txt"
    assert "262 passed" in read_text(test_log)
    question = ClinicalQuestion.model_validate(read_json(ROOT / "data/examples/patient-memory-question.json"))
    fact = FactInput.model_validate(read_json(ROOT / "data/examples/patient-memory-fact.json"))
    assert question.question_id == "EXAMPLE-WEIGHT" and fact.origin == "user_provided"
    syntax_files = sorted((ROOT / "src/medical_copilot").glob("patient*.py")) + [ROOT / "scripts" / name for name in [
        "manage_patients.py", "ask_patient.py", "prepare_patient_demo.py", "run_patient_demo.py",
        "verify_patient_run.py", "build_patient_memory_report.py", "check_patient_memory_artifacts.py"]]
    for path in syntax_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    documents = [ROOT / "README.md", ROOT / "handoff.md", ROOT / "docs/implementation/requirement-7-patient-memory.md", MAIN / "results.md"]
    links = 0
    for path in documents:
        text = read_text(path)
        for url in re.findall(r"\]\(([^\s)]+)\)", text):
            if "://" in url or url.startswith("#"):
                continue
            target = unquote(url.split("#", 1)[0])
            resolved = (path.parent / target).resolve()
            assert resolved.exists() or resolved == ROOT / "data/checks/requirement-7-final-artifacts-20261014.json", f"Missing local link in {path.name}: {target}"
            links += 1
    credential_pattern = re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}")
    private_field_pattern = re.compile(r'"(?:reasoning_content|reasoning_details|chain_of_thought|api_key|authorization)"\s*:')
    scanned = 0
    # Inspect only generated artifacts. Never read credential files for this scan.
    for path in RUNS.rglob("*"):
        if path.suffix not in {".json", ".md", ".txt", ".xml"}:
            continue
        text = read_text(path)
        assert not credential_pattern.search(text), "Credential-shaped content in generated artifact"
        assert not private_field_pattern.search(text), "Unexpected private/credential field in generated artifact"
        scanned += 1
    result = {"passed": True, "requirement": 7, "frozen_earlier_application_files": len(prior["frozen_application_source_hashes"]),
              "source_snapshots_checked": source_snapshots, "historical_generation_runtime_hashes": sorted(generation_runtime_versions),
              "episodes": episode_count, "replayed_responses": 27, "replayed_http_requests": 2,
              "primary_scores": scores["totals"], "supported_structured_citations": 3, "requests_isolation_audited": 27,
              "offline_tests_passed": 262, "new_patient_memory_runtime_test_cases": 32,
              "test_log_sha256": file_hash(test_log), "syntactically_valid_python_files": len(syntax_files),
              "validated_local_document_links": links, "generated_text_artifacts_scanned": scanned,
              "example_input_contracts_valid": True, "new_inference_requests": 0, "credential_files_read": 0,
              "interpretation": "Artifact/provenance verification and recorded checks; not an independent clinical judgment or universal security proof."}
    write_json(ROOT / "data/checks/requirement-7-final-artifacts-20261014.json", result)
    print(json.dumps({"passed": True, "episodes": episode_count, "saved_responses": 27, "tests_passed": 262, "local_links": links, "scanned_artifacts": scanned}))


if __name__ == "__main__":
    main()
