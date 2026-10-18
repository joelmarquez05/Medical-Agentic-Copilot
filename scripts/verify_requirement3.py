"""Verify real-response provenance, frozen inputs, reviews and report locally."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import pymupdf
from medical_copilot.config import load_config, resolve_credential, ConfigurationError
from medical_copilot.evaluation import read_json, file_hash, write_json
from medical_copilot.pubmed import parse_pubmed_xml, utc_now

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evaluation/retrieval-runs"


def main():
    original = read_json(ROOT / "data/evaluation/runs/gemini-internal-final-20261010/manifest.json")
    question_path = ROOT / "data/evaluation/oncology_nutrition_questions.json"
    dataset_hash = file_hash(question_path)
    assert dataset_hash == original["dataset_sha256"]
    ids = [item["id"] for item in read_json(question_path)["questions"]]
    final = BASE / "gemini-pubmed-final-20261010"
    initial = BASE / "gemini-pubmed-internal-20261010"
    pair_count = 0
    current_source_changes = {}
    for run in [initial, final]:
        manifest = read_json(run / "manifest.json")
        assert manifest["status"] == "completed" and manifest["question_ids"] == ids
        assert manifest["dataset_sha256"] == dataset_hash
        assert manifest["rubric_sha256"] == file_hash(ROOT / "data/evaluation/retrieval_rubric.json")
        assert manifest["guidance_sha256"] == original["guidance_sha256"]
        for name, expected in manifest["code_sha256"].items():
            assert file_hash(run / "source-snapshot" / name) == expected
            if run == final:
                current = file_hash(ROOT / "src/medical_copilot" / name)
                if current != expected:
                    # Requirement 5 adds CLI commands; the original workflow
                    # remains verified by archived replay and unchanged modules.
                    comparison_manifest = read_json(ROOT / "data/evaluation/comparison-runs/selected-five-main-20261010/manifest.json")
                    assert name == "cli.py"
                    assert current == comparison_manifest["source_code_sha256"][name]
                    current_source_changes[name] = {"archived_sha256": expected, "current_sha256": current, "reason": "Requirement-5 model-comparison CLI commands; archived retrieval behavior remains unchanged."}
        assessments = read_json(run / "assessments.json")["assessments"]
        assert {item["question_id"] for item in assessments} == set(ids)
        for item in assessments:
            record_path = run / "retrieval" / (item["question_id"] + ".json")
            assert file_hash(record_path) == item["retrieval_file_sha256"]
            record = read_json(record_path)
            papers = record["unique_papers"]
            pair_count += len(papers)
            ratings = item["paper_assessments"]
            assert len(ratings) == len(papers)
            assert {paper["pmid"] for paper in papers} == {paper["pmid"] for paper in ratings}
            assert all(type(paper["relevance"]) is int and paper["relevance"] in {0, 1, 2} and paper["rationale"] for paper in ratings)
            answer = read_json(run / "answers" / (item["question_id"] + ".json"))
            assert answer["answer"] and answer["question"] == record["question"]
            for call in answer["model_calls"]:
                for field in ["expected_key_points", "critical_failures", "reference_ids"]:
                    assert ('"' + field + '"') not in call["user_prompt"]
            for search in record["searches"]:
                assert search["status"] == "ok"
                output = search["output"]
                requests = output["http_requests"]
                raw_search = json.loads((run / "http" / requests[0]["snapshot_file"]).read_bytes())["esearchresult"]
                assert raw_search["idlist"] == output["esearch_pmids"]
                assert int(raw_search["count"]) == output["total_matches"]
                if len(requests) == 2:
                    raw_xml = (run / "http" / requests[1]["snapshot_file"]).read_bytes()
                    parsed = {paper["pmid"]: paper for paper in parse_pubmed_xml(raw_xml, manifest["pubmed_config"]["max_abstract_characters"])}
                    for paper in output["sources"]:
                        assert paper["id"] == "P" + paper["pmid"]
                        assert paper["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
                        for key in ["title", "authors", "year", "doi", "abstract", "abstract_truncated", "read_level", "publication_types"]:
                            assert paper[key] == parsed[paper["pmid"]][key]
        for request in read_json(run / "http-requests.json"):
            assert request["endpoint"].startswith("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
            assert request["status"] == "ok" and not request["cache_hit"]
            assert file_hash(run / "http" / request["snapshot_file"]) == request["response_sha256"]
            assert "email" not in request["parameters"] and "api_key" not in request["parameters"]
    assert pair_count == 76
    a, b = read_json(initial / "manifest.json")["code_sha256"], read_json(final / "manifest.json")["code_sha256"]
    assert [name for name in a if a[name] != b[name]] == ["literature.py"]
    replay = read_json(ROOT / "data/checks/requirement-3-replay.json")
    assert len(replay["cases"]) == 20 and replay["live_model_calls"] == replay["live_pubmed_requests"] == 0
    probe = read_json(ROOT / "data/checks/requirement-3-attribution-probe.json")
    assert probe["source_answer_sha256"] == file_hash(ROOT / probe["source_answer_file"])
    assert probe["pubmed_http_calls"] == 0 and len(probe["model_calls"]) == 1
    # Preserve the observed false negative rather than passing it as repaired.
    assert next(item for item in probe["review"]["paper_checks"] if item["source_id"] == "P41406403")["draft_claim_supported"]
    tests = ET.parse(ROOT / "data/checks/requirement-3-tests.xml").getroot().find("testsuite")
    assert int(tests.get("tests")) == 58 and int(tests.get("failures")) == int(tests.get("errors")) == 0
    catalogue = read_json(ROOT / "data/research/requirement-3-source-catalogue.json")
    sources = catalogue["papers"] + catalogue["technical_sources"]
    assert len(sources) == len({source["id"] for source in sources}) == 70
    pdf_path = ROOT / "docs/implementation/requirement-3-report.pdf"
    tex = pdf_path.with_suffix(".tex").read_text(encoding="utf-8")
    assert set(re.findall(r"\\bibitem\[[^\]]+\]\{([^}]+)\}", tex)) == {source["id"] for source in sources}
    log = pdf_path.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"Overfull \\hbox|Missing character|(?:LaTeX|Package[^\n]*) Warning", log)
    pdf = pymupdf.open(pdf_path)
    text = "\n".join(page.get_text() for page in pdf)
    assert pdf.page_count >= 15 and all(identifier in text for identifier in ids)
    urls = {link["uri"] for page in pdf for link in page.get_links() if link.get("uri")}
    assert {source["url"] for source in sources}.issubset(urls)
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    assert x0 >= 35 and x1 <= page.rect.width - 35 and y0 >= 20 and y1 <= page.rect.height - 20
    keys = []
    for name in ["gemini.json", "nvidia-kimi-k3.json", "openrouter-claude-opus-5.5.json"]:
        try:
            key, _ = resolve_credential(load_config(ROOT / "config/models" / name), ROOT)
            if key:
                keys.append(key)
        except ConfigurationError:
            pass
    files = []
    for directory in ["src", "scripts", "tests", "data", "docs", "config", ".agents"]:
        files += [path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts and not any(part.endswith(".egg-info") for part in path.parts)]
    files += [ROOT / name for name in ["README.md", "AGENTS.md", ".env.example", "pyproject.toml", "requirements.lock.txt", ".gitignore"]]
    for path in files:
        raw = path.read_bytes()
        assert not any(key.encode() in raw for key in keys), "Known credential detected in a public artifact."
    assert not any(key in text for key in keys)
    links = 0
    for document in [ROOT / "README.md", ROOT / ".agents/journal.md", ROOT / "docs/implementation/requirement-3.md", ROOT / "docs/implementation/requirement-3-answer-observations.md"]:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text(encoding="utf-8-sig")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = unquote(target.split("#")[0].strip("<>"))
            if path:
                links += 1
                assert (document.parent / path).resolve().exists(), "Broken local documentation link."
    report = {"verified_at_utc": utc_now(), "live_model_calls": 0, "live_pubmed_requests": 0,
              "question_count_per_run": 10, "retrieval_records_verified": 20, "question_paper_pairs_verified": pair_count,
              "raw_response_metadata_and_hashes_match": True, "original_question_fingerprint_unchanged": True,
              "assessment_hashes_match": True, "source_snapshots_match": True,
              "documented_current_source_changes": current_source_changes,
              "initial_to_final_code_changes": ["literature.py"], "gold_fields_absent": True,
              "offline_tests_passed": 58, "replayed_cases": 20,
              "unresolved_attribution_false_negative_preserved": True,
              "source_count": 70, "pdf_pages": pdf.page_count,
              "pdf_link_annotations": sum(len(page.get_links()) for page in pdf), "latex_warnings": 0,
              "all_source_urls_present": True, "local_links_checked": links,
              "known_credentials_scanned": len(keys), "public_files_scanned": len(files),
              "credentials_absent": True if keys else None,
              "pdf_sha256": file_hash(pdf_path), "tex_sha256": file_hash(pdf_path.with_suffix('.tex')),
              "interpretation": "Local engineering/artifact checks only. No new inference/retrieval, clinical adjudication or attribution repair is claimed."}
    write_json(ROOT / "data/checks/requirement-3-verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
