"""Check completed comparison artifacts locally; never invoke a model or network."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from medical_copilot.clinical_evaluation import score_record
from medical_copilot.clinical_graph import save_trace
from medical_copilot.clinical_model_comparison import MODELS, PROFILES, validate_reference
from medical_copilot.clinical_resilient import RESILIENT_SYSTEM
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json
from appraise_clinical_model_comparison import fingerprint
from clinical_comparison_review_helpers import extra_reference_mentions

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"
OUTPUT = ROOT / "data/checks/requirement-5-final-artifacts-20261011.json"
STAGES = (
    ("medqa30-native-probes-20261011", "native-probes", 4, 4, 0),
    ("medqa30-agent-pilots-20261011", "agent-pilots", 4, 7, 5),
    (RUN.name, "main", 120, 264, 239),
)


def assert_private_free(value):
    if isinstance(value, dict):
        for name, item in value.items():
            assert name.lower() not in {"api_key", "authorization", "reasoning_content", "reasoning_details", "thoughts"}, "Sensitive field present; value not printed."
            if name.lower() == "reasoning":
                allowed = {"effort", "enabled", "exclude", "max_tokens", "mandatory", "default_enabled", "supported_efforts", "default_effort"}
                assert item is None or isinstance(item, (int, float)) or isinstance(item, dict) and set(item) <= allowed, "Unexpected reasoning payload; value not printed."
            assert_private_free(item)
    elif isinstance(value, list):
        for item in value:
            assert_private_free(item)


def check_stage(directory, receipt):
    manifest = read_json(directory / "manifest.json")
    assert manifest["finished_at_utc"]
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(directory / "source-snapshot" / name) == file_hash(ROOT / "src/medical_copilot" / name) == digest
    fields = {
        "questions_sha256": "frozen-benchmark-questions.json", "answer_key_sha256": "answer-key.json",
        "selection_manifest_sha256": "selection-manifest.json", "protocol_sha256": "protocol.md",
        "rubric_sha256": "rubric.json", "retrieval_rubric_sha256": "retrieval-rubric.json",
        "policy_sha256": "policy.json", "pubmed_config_sha256": "pubmed-config.json",
    }
    for field, name in fields.items():
        assert file_hash(directory / name) == manifest[field]
    assert file_hash(ROOT / manifest["protocol_origin"]) == manifest["protocol_sha256"]
    assert hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest() == manifest["effective_system_sha256"]
    assert read_json(directory / "system-instructions.json")["effective_system"] == RESILIENT_SYSTEM
    assert file_hash(ROOT / manifest["reference_origin"] / "manifest.json") == manifest["reference_manifest_sha256"]
    for slug, digest in manifest["profile_hashes"].items():
        profile = ROOT / PROFILES / (slug + ".json")
        assert file_hash(profile) == digest
        assert read_json(directory / slug / "config.json") == load_config(profile).public_dict()
    assert receipt["passed"] and receipt["new_model_requests"] == receipt["new_pubmed_requests"] == receipt["credential_files_read"] == 0
    for case in receipt["cases"]:
        assert file_hash(directory / case["model_profile"] / "answers" / (case["question_id"] + ".json")) == case["answer_sha256"]


def main():
    reference, selection, questions, keys = validate_reference(ROOT)
    key_map = {k["question_id"]: k for k in keys}
    question_map = {q["question_id"]: q for q in questions}
    rubric = read_json(RUN / "rubric.json")
    summary = read_json(RUN / "assessment-summary.json")["models"]
    totals = Counter()
    citations, relevance = Counter(), Counter()
    for slug in MODELS:
        directory = RUN / slug
        assessed = read_json(directory / "answer-assessments.json")
        retrieved = read_json(directory / "retrieval-assessments.json")
        reviews = {r["question_id"]: r for r in assessed["reviews"]}
        episodes = {r["question_id"]: r for r in retrieved["episodes"]}
        scores = {s["question_id"]: s for s in read_json(directory / "scores.json")["scores"]}
        notes = {p.stem: p for p in (RUN / "manual-reviews" / slug).glob("*.json")}
        assert len(reviews) == len(episodes) == len(scores) == len(notes) == 30
        exact = (directory / "answers.md").read_text(encoding="utf-8")
        for identifier in question_map:
            path = directory / "answers" / (identifier + ".json")
            row, note = read_json(path), read_json(notes[identifier])
            digest = file_hash(path)
            assert row["question"] == question_map[identifier]
            assert row["output_origin"] == "model" and row["response"]["type"] == "final"
            assert row["provider_failures"] == 0 and row["error_type"] is None
            assert reviews[identifier]["answer_sha256"] == episodes[identifier]["answer_sha256"] == note["answer_sha256"] == digest
            assert note["ratings"] == reviews[identifier]["dimension_ratings"]
            assert set(note["ratings"].values()) <= set(rubric["answer_ratings"])
            assert assessed["manual_review_sha256"][identifier + ".json"] == file_hash(notes[identifier])
            assert scores[identifier] == score_record(row, key_map[identifier])
            block = exact.split("## " + identifier + "\n", 1)[1].split("\n## BM", 1)[0]
            response = row["response"]
            for text in [row["question"]["question"], *row["question"]["options"].values(), response["answer"], response["justification"], *response["limitations"], *response["citations"]]:
                assert text in block, (slug, identifier, "Rendered text mismatch")
            events = row["literature"]["events"]
            views = {v["id"]: v for e in events for v in e["generation_view"]["sources"]}
            papers = {p["pmid"]: p for e in events for p in e["papers"]}
            episode = episodes[identifier]
            assert {p["pmid"] for p in episode["paper_assessments"]} == set(papers) == set(note["papers"])
            for paper in episode["paper_assessments"]:
                assert paper["paper_relevance"] == note["papers"][paper["pmid"]]["rating"]
                expected = {sid: fingerprint(v) for sid, v in views.items() if v["pmid"] == paper["pmid"]}
                assert paper["generation_view_sha256"] == expected
                relevance[paper["paper_relevance"]] += 1
            assert len(episode["query_assessments"]) == len(note["queries"]) == sum(e["name"] == "search_pubmed" for e in events)
            assert [c["source_id"] for c in episode["citation_assessments"]] == response["citations"]
            for c in episode["citation_assessments"]:
                assert c["view_exists"] and c["generation_view_sha256"] == fingerprint(views[c["source_id"]])
                assert c["rating"] == note["citations"][c["source_id"]]["rating"]
                citations[c["rating"]] += 1
            inline = episode["supplementary_inline_reference_assessments"]
            assert {c["source_id"] for c in inline} == extra_reference_mentions(row)
            for c in inline:
                assert c["view_exists"] and c["generation_view_sha256"] == fingerprint(views[c["source_id"]])
            totals.update(answers=1, paper_pairs=len(papers), structured_citations=len(response["citations"]), inline_references=len(inline))
        assert summary[slug] == read_json(directory / "assessment-summary.json")
    assert totals == {"answers": 120, "paper_pairs": 348, "structured_citations": 170, "inline_references": 1}

    stages, cost_total, request_total = [], 0.0, 0
    for dirname, tag, expected_cases, expected_models, expected_http in STAGES:
        directory = RUN.parent / dirname
        receipt_path = ROOT / "data/checks" / ("requirement-5-" + tag + "-verification-20261011.json")
        receipt = read_json(receipt_path)
        check_stage(directory, receipt)
        assert len(receipt["cases"]) == expected_cases
        assert sum(c["model_responses"] for c in receipt["cases"]) == expected_models
        assert sum(c["http_responses"] for c in receipt["cases"]) == expected_http
        for slug in MODELS:
            audit = read_json(directory / slug / "usage-audit.json")["records"]
            assert all(a["status"] == "ok" and a["request_sent"] and a["cost_basis"] == "Provider-reported usage.cost." for a in audit)
            cost_total += sum(a["cost_usd"] for a in audit)
            request_total += len(audit)
        stages.append({"run_id": dirname, "replay_receipt_sha256": file_hash(receipt_path), "episodes": expected_cases, "model_responses": expected_models, "http_responses": expected_http})
    cost_path = ROOT / "data/checks/requirement-5-cost-audit-20261011.json"
    cost = read_json(cost_path)
    assert request_total == cost["saved_provider_requests"] == 275
    assert math.isclose(cost_total, cost["known_reported_or_estimated_total_usd"], abs_tol=1e-12)
    assert cost["agreement_within_1e_7_usd"] and cost["unknown_new_request_reserve_usd"] == 0
    assert cost["latest_ledger"]["total_usage"] < cost["standing_account_limit_usd"] == 6
    assert not cost["gemma_reference_regenerated"]

    catalogues = [ROOT / "data/research" / name for name in (
        "requirement-5-comparison-method-sources-20261011.json", "requirement-5-clinical-review-sources-20261011.json", "requirement-5-observed-papers-20261011.json")]
    for path in catalogues:
        for source in read_json(path)["sources"]:
            assert source["title"] and source["authors_or_organization"] and source["year"] and source["url"] and source["supports"]
            if path.name != "requirement-5-comparison-method-sources-20261011.json":
                assert isinstance(source["preprint"], bool)
            assert source.get("read_level") or source.get("origins")
    papers = read_json(catalogues[-1])
    assert papers["count"] == len(papers["sources"]) == 277
    assert sum(len(p["origins"]) for p in papers["sources"]) == 348

    scripts = [ROOT / "scripts" / name for name in (
        "run_clinical_model_comparison.py", "verify_clinical_model_comparison.py", "inspect_clinical_model_comparison.py",
        "inspect_comparison_question.py", "clinical_comparison_review_helpers.py", "appraise_clinical_model_comparison.py",
        "build_clinical_model_comparison_report.py", "audit_clinical_model_comparison_cost.py", Path(__file__).name)]
    for script in scripts:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
    documents = [ROOT / "README.md", ROOT / "handoff.md", ROOT / ".agents/journal.md",
                 ROOT / "docs/implementation/three-evaluation-plan.md", ROOT / "docs/implementation/requirement-5-medqa30-protocol.md", *RUN.rglob("*.md")]
    local_links, missing = 0, set()
    for document in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            target = target.strip().strip("<>")
            if not target or target.startswith("#") or urlsplit(target).scheme:
                continue
            resolved = (document.parent / unquote(target.split("#", 1)[0])).resolve()
            local_links += 1
            if not resolved.exists():
                missing.add((document.relative_to(ROOT).as_posix(), target))
    allowed = {(".agents/journal.md", "../Medical%20Agent%20Copilot.pdf")}
    assert missing <= allowed, ("Missing new local links", sorted(missing - allowed))

    files = set(catalogues)
    for dirname, *_ in STAGES:
        files.update(p for p in (RUN.parent / dirname).rglob("*") if p.suffix.lower() in {".json", ".xml", ".md"})
    files.update(ROOT.glob("data/checks/requirement-5-*.json"))
    files.update(p for p in (ROOT / "data/research/requirement-5-review-pubmed-20261011").rglob("*") if p.suffix in {".json", ".xml"})
    for path in files:
        raw = path.read_text(encoding="utf-8-sig")
        assert not re.search(r"sk-(?:or-v1-)?[A-Za-z0-9_-]{24,}|Bearer\s+[A-Za-z0-9._-]{20,}", raw), "Possible credential detected; content not printed."
        if path.suffix == ".json":
            assert_private_free(json.loads(raw))
    result = {"passed": True, "completed_on": "2026-10-12", "primary_coverage": dict(totals),
              "citation_support": dict(citations), "paper_relevance": dict(relevance), "unique_primary_papers": 277,
              "frozen_agent_prompt_data_profiles_rubrics_verified": True, "gemma_reference_preserved": True,
              "exact_rendered_answers_and_all_view_note_fingerprints_verified": True, "replay_receipts": stages,
              "provider_reported_charges_verified": request_total, "new_stages_cost_usd": cost_total,
              "cumulative_usage_snapshot_usd": cost["latest_ledger"]["total_usage"], "cost_audit_sha256": file_hash(cost_path),
              "scripts_compiled": len(scripts), "privacy_files_scanned": len(files), "local_links_checked": local_links,
              "preserved_historical_missing_links": [{"document": d, "target": t} for d, t in sorted(missing)],
              "full_test_suite": {"previously_observed_passed": 184, "rerun_by_this_check": False},
              "new_model_requests": 0, "new_pubmed_requests": 0, "credential_files_read": 0,
              "interpretation": "Local artifact consistency, not independent clinical validation. Old point-4 in-flight uncertainty remains explicit."}
    if not OUTPUT.exists():
        save_trace(OUTPUT, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
