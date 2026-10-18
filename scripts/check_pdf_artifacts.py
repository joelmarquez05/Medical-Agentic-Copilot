"""Check frozen PDF artifacts, appraisals, links, source preservation and privacy."""
import hashlib
import argparse
import json
import py_compile
import re
from pathlib import Path
from urllib.parse import unquote
from medical_copilot.clinical_evaluation import score_record
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.pdf_agent import CONDITIONS
from medical_copilot.pdf_evaluation import source_hashes, validate_set
from build_pdf_report import file_hash_view

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="Optional fresh verification receipt; repeat checks otherwise only print")
args = parser.parse_args()
RUN = ROOT / "data/evaluation/pdf-runs/gemma-pdf-main-20261012"
questions, keys = validate_set(ROOT)
assert read_json(RUN / "questions.json")["questions"] == questions
assert read_json(RUN / "answer-key.json")["keys"] == keys
manifest = read_json(RUN / "manifest.json")
assert manifest["source_hashes"] == source_hashes(ROOT)
for name, digest in manifest["source_hashes"].items():
    assert file_hash(RUN / "source-snapshot" / name) == digest
reviews = read_json(RUN / "manual-reviews.json")["reviews"]
assert len(reviews) == 18 and len({(r["condition"], r["question_id"]) for r in reviews}) == 18
rendered = (RUN / "results.md").read_text(encoding="utf-8")
for review in reviews:
    row = read_json(RUN / review["condition"] / "answers" / (review["question_id"] + ".json"))
    assert review["answer_sha256"] == file_hash(RUN / review["condition"] / "answers" / (review["question_id"] + ".json"))
    assert row["response"]["answer"] in rendered and row["response"]["justification"] in rendered
    for view in review["citations"] + review["inline_only_citations"] + review["retrieval_views"]:
        sid = view["source_id"]
        if sid in row["evidence"]:
            assert view["view_sha256"] == file_hash_view(row["evidence"][sid])
        else:
            assert view["grade"] == "invalid-id"
receipts = [ROOT / "data/checks" / (f"requirement-6-{stage}-replay-20261012.json") for stage in ["probe-v3", "pilot-v3", "main", "application"]]
for receipt in receipts:
    assert read_json(receipt)["passed"]
replays = [case for receipt in receipts for case in read_json(receipt)["episodes"]]
assert len(replays) == 23 and sum(case["model_responses"] for case in replays) == 37
cost = read_json(ROOT / "data/checks/requirement-6-cost-audit-20261012.json")
assert cost["passed"] and cost["saved_provider_requests"] == 51 and cost["unknown_charge_records"] == 1
assert cost["latest_ledger_usage_usd"] < cost["account_limit_usd"] == 6
docs = [ROOT / "README.md", ROOT / "handoff.md", ROOT / ".agents/journal.md", ROOT / "docs/implementation/requirement-6-pdf.md",
        ROOT / "docs/implementation/requirement-6-pdf-protocol.md", RUN / "results.md", ROOT / "data/evaluation/pdf-set-v2-20261012/questions.md"]
checked_links, historical_missing = 0, []
for path in docs:
    content = path.read_text(encoding="utf-8-sig")
    for target in re.findall(r"\[[^\]\n]+\]\(([^()\n]+)\)", content):
        target = target.strip().strip("<>")
        if target.startswith(("https://", "http://", "#", "mailto:")):
            continue
        base = unquote(target.split("#")[0])
        if not base:
            continue
        resolved = (path.parent / base).resolve()
        if not resolved.exists():
            if path == ROOT / ".agents/journal.md" and base == "../Medical Agent Copilot.pdf":
                historical_missing.append(target)
                continue
            raise AssertionError((path.relative_to(ROOT).as_posix(), target))
        checked_links += 1
scripts = ["ingest_medical_pdf.py", "prepare_pdf_question_set.py", "run_pdf_evaluation.py", "ask_medical_pdf.py",
           "verify_pdf_evaluation.py", "check_pdf_extraction.py", "record_pdf_sources.py", "inspect_pdf_answers.py",
           "appraise_pdf_answers.py", "build_pdf_report.py", "audit_pdf_cost.py", "check_pdf_artifacts.py"]
for name in scripts:
    py_compile.compile(str(ROOT / "scripts" / name), doraise=True)
scan_files = {path for folder in [ROOT / "data/evaluation/pdf-runs", ROOT / "data/evaluation/pdf-set-v2-20261012",
                                ROOT / "data/documents/ingested/espen-cancer-v2-20261012"] for path in folder.rglob("*")
              if path.is_file() and path.suffix in {".json", ".md", ".py"}}
scan_files.update(docs)
suspicious = re.compile(r"sk-or-v1-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9_-]{15,}")
for path in scan_files:
    value = path.read_text(encoding="utf-8-sig")
    assert not suspicious.search(value), "Credential-shaped artifact content withheld."
    if path.suffix == ".json":
        assert '"reasoning_content"' not in value and '"reasoning_text"' not in value
result = {"passed": True, "primary_episodes": 18, "reviewed_explanations": len(reviews),
    "structured_cited_views": sum(len(r["citations"]) for r in reviews), "additional_inline_only_cited_views": sum(len(r["inline_only_citations"]) for r in reviews),
    "unique_retrieval_pairs": sum(len(r["retrieval_views"]) for r in reviews), "questions_and_keys_unchanged": True,
    "earlier_operational_sources_and_gemma_profile_unchanged": True, "saved_active_replay_episodes": len(replays),
    "saved_active_replay_model_responses": sum(r["model_responses"] for r in replays),
    "compiled_scripts": len(scripts), "local_links_checked": checked_links, "privacy_scanned_files": len(scan_files),
    "historical_missing_journal_links_preserved": historical_missing,
    "observed_full_test_suite": "218 passed before excluded-pilot free-form revision; 11 targeted runner tests passed after that revision",
    "new_model_or_network_requests": 0, "patient_memory_implemented": False,
    "report_sha256": file_hash(RUN / "results.md"), "manual_reviews_sha256": file_hash(RUN / "manual-reviews.json"),
    "cost_audit_sha256": file_hash(ROOT / "data/checks/requirement-6-cost-audit-20261012.json")}
destination = args.output or ROOT / "data/checks/requirement-6-final-artifacts-20261012.json"
if args.output or not destination.exists():
    save_trace(destination, result)
print(json.dumps(result))
