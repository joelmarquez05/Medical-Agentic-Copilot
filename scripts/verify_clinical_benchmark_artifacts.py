"""Check saved benchmark reports/appraisals/provenance without network or credentials."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

from medical_copilot.clinical_benchmark import BENCHMARK, PROTOCOL, RUBRIC, validate_benchmark
from medical_copilot.clinical_evaluation import DEFAULT_PROFILE
from medical_copilot.clinical_graph import save_trace
from medical_copilot.clinical_resilient import RESILIENT_SYSTEM
from medical_copilot.clinical_resilient_evaluation import POLICY
from medical_copilot.evaluation import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"


def fingerprint(view):
    return hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify(output=None):
    selection, questions, keys = validate_benchmark(ROOT)
    manifest = read_json(RUN / "manifest.json")
    notes = read_json(RUN / "review-notes-v2.json")
    answers = read_json(RUN / "answer-assessments-v2.json")
    retrieval = read_json(RUN / "retrieval-assessments-v2.json")
    assessment = read_json(RUN / "assessment-summary-v2.json")
    records = [read_json(p) for p in sorted((RUN / "answers").glob("*.json"))]
    assert len(records) == len(questions) == len(keys) == 30
    assert file_hash(ROOT / BENCHMARK / "manifest.json") == manifest["selection_manifest_sha256"]
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(ROOT / "src/medical_copilot" / name) == file_hash(RUN / "source-snapshot" / name) == digest
    for path, field in [(DEFAULT_PROFILE, "config_sha256"), (POLICY, "policy_sha256"),
                        (PROTOCOL, "protocol_sha256"), (RUBRIC, "rubric_sha256")]:
        assert file_hash(ROOT / path) == manifest[field]
    assert hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest() == manifest["effective_system_sha256"]
    reference = read_json(ROOT / manifest["reference_origin"] / "manifest.json")
    for name, digest in reference["source_hashes"].items():
        assert manifest["source_hashes"][name] == digest
    for field in ["config_sha256", "policy_sha256", "effective_system_sha256"]:
        assert manifest[field] == reference[field]
    for origin in read_json(RUN / "history-exclusions.json")["origins"]:
        assert file_hash(ROOT / origin["path"]) == origin["sha256"]
    for review in [answers, retrieval]:
        assert review["review_notes_sha256"] == file_hash(RUN / "review-notes-v2.json")
    assert answers["rubric_sha256"] == file_hash(RUN / "rubric.json")
    assert retrieval["rubric_sha256"] == file_hash(RUN / "retrieval-rubric.json")
    by_id = {r["question"]["question_id"]: r for r in records}
    assert set(notes["answers"]) == set(by_id)
    assert {r["question_id"] for r in answers["reviews"]} == {r["question_id"] for r in retrieval["episodes"]} == set(by_id)
    exact = (RUN / "answers.md").read_text(encoding="utf-8")
    for row in records:
        identifier, response = row["question"]["question_id"], row["response"]
        block = exact.split("## " + identifier + "\n", 1)[1].split("\n## BM", 1)[0]
        for text in [row["question"]["question"], *row["question"]["options"].values(),
                     response["answer"], response["justification"], *response["limitations"], *response["citations"]]:
            assert text in block, identifier
    for r in answers["reviews"] + retrieval["episodes"]:
        assert r["answer_sha256"] == file_hash(RUN / "answers" / (r["question_id"] + ".json"))
    paper_pairs, cited_ids, visible_pmids = set(), set(), set()
    for episode in retrieval["episodes"]:
        row = by_id[episode["question_id"]]
        events = row["literature"]["events"]
        actual_papers = {p["pmid"] for e in events for p in e["papers"]}
        assert {p["pmid"] for p in episode["paper_assessments"]} == actual_papers
        views = {v["id"]: v for e in events for v in e["generation_view"]["sources"]}
        for paper in episode["paper_assessments"]:
            paper_pairs.add(episode["question_id"] + ":" + paper["pmid"])
            visible_pmids.add(paper["pmid"])
            assert set(paper["generation_view_ids"]) == {sid for sid, v in views.items() if v["pmid"] == paper["pmid"]}
            for sid, digest in paper["generation_view_sha256"].items():
                assert fingerprint(views[sid]) == digest
        assert [c["source_id"] for c in episode["citation_assessments"]] == row["response"]["citations"]
        assert len(episode["query_assessments"]) == sum(e["name"] == "search_pubmed" for e in events)
        for c in episode["citation_assessments"]:
            cited_ids.add(episode["question_id"] + ":" + c["source_id"])
            assert c["view_exists"] == (c["source_id"] in views)
            if c["view_exists"]:
                assert fingerprint(views[c["source_id"]]) == c["generation_view_sha256"]
    assert paper_pairs == set(notes["papers"]) and cited_ids == set(notes["citations"])
    assert len(paper_pairs) == len(visible_pmids) == 42 and len(cited_ids) == 10
    assert Counter(p["paper_relevance"] for e in retrieval["episodes"] for p in e["paper_assessments"]) == Counter(assessment["paper_relevance"])
    assert Counter(c["rating"] for e in retrieval["episodes"] for c in e["citation_assessments"]) == Counter(assessment["citation_support"])
    assert sum(not c["view_exists"] for e in retrieval["episodes"] for c in e["citation_assessments"]) == 2
    material = [r["question_id"] for r in answers["reviews"] if "material_issue" in r["dimension_ratings"].values()]
    assert material == ["BM15", "BM21", "BM26"]
    catalogue_counts = {}
    for name, expected in [("requirement-4-benchmark-sources-20261011.json", 3),
                           ("requirement-4-clinical-review-sources-20261011.json", 12),
                           ("requirement-4-observed-papers-20261011.json", 42),
                           ("requirement-4-interrupted-observed-papers-20261011.json", 5)]:
        sources = read_json(ROOT / "data/research" / name)["sources"]
        assert len(sources) == expected
        for source in sources:
            assert all(k in source and source[k] for k in ["id", "title", "authors_or_organization", "year", "url"])
            assert "supports" in source and ("read_level" in source or "origins" in source)
            # Repository/dataset provenance records are not articles or preprints.
            if name != "requirement-4-benchmark-sources-20261011.json":
                assert "preprint" in source
        catalogue_counts[name] = len(sources)
    clinical = read_json(ROOT / "data/research/requirement-4-clinical-review-sources-20261011.json")["sources"]
    used_refs = {sid for r in answers["reviews"] for sid in r["researcher_reference_ids"]}
    assert used_refs == {s["id"] for s in clinical}
    observed = read_json(ROOT / "data/research/requirement-4-observed-papers-20261011.json")["sources"]
    assert {s["pmid"] for s in observed} == visible_pmids
    cost = read_json(ROOT / "data/checks/requirement-4-cost-audit-20261011.json")
    assert cost["saved_provider_requests"] == 59 and cost["possible_unrecorded_inflight_requests"] == 1
    assert math.isclose(sum(r["provider_reported_usd"] for r in cost["runs"]), cost["provider_reported_known_total_usd"], abs_tol=1e-12)
    assert cost["agreement_within_1e_7_usd"] and cost["latest_ledger"]["total_usage"] < 6
    primary = read_json(ROOT / "data/checks/requirement-4-main-verification-20261011.json")
    pilot = read_json(ROOT / "data/checks/requirement-4-pilot-v3-verification-20261011.json")
    interrupted = read_json(ROOT / "data/checks/requirement-4-interruption-verification-20261011.json")
    assert primary["passed"] and pilot["passed"] and interrupted["passed"]
    for case in primary["cases"]:
        assert case["answer_sha256"] == file_hash(RUN / "answers" / (case["question_id"] + ".json"))
    replayed_model = sum(c["replayed_model_responses"] for r in [primary, pilot] for c in r["cases"])
    replayed_http = sum(c["replayed_http_responses"] for r in [primary, pilot] for c in r["cases"])
    assert replayed_model == 54 and replayed_http == 34
    scripts = ["prepare_clinical_benchmark.py", "run_clinical_benchmark.py", "verify_clinical_benchmark.py",
               "inspect_clinical_benchmark.py", "author_clinical_benchmark_notes.py", "appraise_clinical_benchmark.py",
               "catalogue_clinical_benchmark_review.py", "audit_clinical_benchmark_cost.py",
               "build_clinical_benchmark_report.py", "verify_clinical_benchmark_artifacts.py"]
    for name in scripts:
        path = ROOT / "scripts" / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    compile((ROOT / "src/medical_copilot/clinical_benchmark.py").read_text(encoding="utf-8"), "clinical_benchmark.py", "exec")
    privacy_files = 0
    for folder in [RUN, ROOT / "data/evaluation/benchmark-runs/gemma-full-agent-pilot-v3-20261011"]:
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".xml", ".md"}:
                text = path.read_text(encoding="utf-8")
                assert not re.search(r"sk-or-v1-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9_-]{15,}", text)
                assert '"reasoning_content"' not in text and '"reasoning_text"' not in text
                privacy_files += 1
    mdpaths = [ROOT / "README.md", ROOT / "handoff.md", ROOT / ".agents/journal.md",
               ROOT / "docs/implementation/three-evaluation-plan.md", ROOT / PROTOCOL,
               RUN / "results.md", RUN / "answers.md", RUN / "answer-assessment.md", RUN / "retrieval-assessment.md"]
    links, missing = 0, []
    for path in mdpaths:
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip().strip("<>")
            if target.startswith("#") or urlparse(target).scheme:
                continue
            dest = (path.parent / unquote(target.split("#", 1)[0])).resolve()
            if output and dest == output:
                continue
            links += 1
            if not dest.exists():
                missing.append([path.relative_to(ROOT).as_posix(), target])
    # Preserve the old journal's source citation even though that supplied slide
    # PDF is no longer present. It is unrelated to the benchmark's artifacts.
    historical_missing = [link for link in missing if link == [".agents/journal.md", "../Medical%20Agent%20Copilot.pdf"]]
    active_missing = [link for link in missing if link not in historical_missing]
    assert not active_missing, active_missing
    return {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": RUN.name,
            "question_key_count": 30, "frozen_sampling_source_keys_history_verified": True,
            "unchanged_graph_prompt_profile_policy": True, "source_modules_verified": len(manifest["source_hashes"]),
            "exact_rendered_answer_count": 30, "answer_and_retrieval_fingerprints_verified": 30,
            "paper_pairs_and_generation_views_verified": len(paper_pairs), "citation_assessments_verified": len(cited_ids),
            "unknown_primary_citation_ids_preserved": 2, "material_answer_flags": material,
            "reference_catalogue_counts": catalogue_counts, "clinical_references_resolved": len(used_refs),
            "primary_and_matching_pilot_saved_replay_model_responses": replayed_model,
            "primary_and_matching_pilot_saved_replay_http_responses": replayed_http,
            "complete_interrupted_prefix_replay_preserved": True, "interrupted_orphan_charged_response": 1,
            "possible_unrecorded_inflight_requests": 1, "known_point_4_provider_requests": 59,
            "known_point_4_cost_usd": cost["provider_reported_known_total_usd"],
            "saved_account_usage_usd": cost["latest_ledger"]["total_usage"],
            "account_snapshot_utc": cost["latest_ledger"]["checked_at_utc"], "cost_arithmetic_verified": True,
            "compiled_new_scripts": scripts, "benchmark_module_syntax_compiled": True,
            "last_observed_offline_suite_passes": 172, "test_suite_rerun_this_check": False,
            "privacy_artifacts_checked": privacy_files, "local_links_checked_before_receipt": links,
            "missing_current_artifact_links": active_missing, "historical_missing_source_links": historical_missing,
            "new_model_requests": 0, "new_pubmed_requests": 0, "credential_files_read": 0,
            "point_5_status": "Additional models/conditions deferred for user discussion; no other model run.",
            "interpretation": "Saved-record provenance/rendering/appraisal/reference verification; not independent clinical validation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else None
    if output and output.exists():
        raise ValueError("Do not overwrite an existing verification receipt.")
    result = verify(output)
    if output:
        save_trace(output, result)
    print(json.dumps(result, indent=2))
