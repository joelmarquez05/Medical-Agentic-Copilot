"""Verify requirement-2 artifacts and credentials locally without inference."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pymupdf

from medical_copilot.config import ConfigurationError, load_config, resolve_credential
from medical_copilot.evaluation import DIMENSIONS, file_hash, read_json, write_json
from medical_copilot.graph import citation_issues, source_ids

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/runs/gemini-internal-final-20261010"


def main():
    manifest = read_json(RUN / "manifest.json")
    dataset_path = ROOT / "data/evaluation/oncology_nutrition_questions.json"
    rubric_path = ROOT / "data/evaluation/qualitative_rubric.json"
    guidance_path = ROOT / "src/medical_copilot/resources/oncology_guidance.json"
    dataset = read_json(dataset_path)
    questions = dataset["questions"]
    ids = [question["id"] for question in questions]
    assert len(ids) == len(set(ids)) == 10
    assert manifest["question_ids"] == ids and manifest["status"] == "completed"
    assert file_hash(dataset_path) == manifest["dataset_sha256"]
    assert file_hash(rubric_path) == manifest["rubric_sha256"]
    assert file_hash(guidance_path) == manifest["guidance_sha256"]
    sources = read_json(ROOT / "data/research/requirement-2-source-catalogue.json")["sources"]
    source_identifiers = {source["id"] for source in sources}
    assert len(source_identifiers) == len(sources) == 25
    assert all(set(question["reference_ids"]).issubset(source_identifiers) for question in questions)

    replay = read_json(ROOT / "data/checks/requirement-2-replay.json")
    assert replay["live_api_calls"] == 0 and len(replay["cases"]) == 13
    changes = []
    for name, expected in manifest["code_sha256"].items():
        assert file_hash(RUN / "source-snapshot" / name) == expected
        current = file_hash(ROOT / "src/medical_copilot" / name)
        if current != expected:
            assert name in replay["source_changes_since_main_run"]
            assert current == replay["source_changes_since_main_run"][name]["current_sha256"]
            changes.append(name)
    assert changes == ["cli.py", "graph.py"]

    assessments = read_json(RUN / "assessments.json")["assessments"]
    assert {item["question_id"] for item in assessments} == set(ids)
    for item in assessments:
        path = RUN / "answers" / (item["question_id"] + ".json")
        answer = read_json(path)
        assert file_hash(path) == item["answer_file_sha256"]
        assert answer["status"] != "error" and answer["answer"]
        assert set(item["ratings"]) == set(item["rationales"]) == set(DIMENSIONS)
        assert all(type(value) is int and value in {0, 1, 2} for value in item["ratings"].values())
        assert all(excerpt in answer["answer"] for excerpt in item["observed_excerpts"])
        body = answer["answer"].split("\n\nSources consulted from the local guidance collection:")[0]
        issues = citation_issues(body, source_ids(answer), True)
        assert issues == answer["mechanical_issues"]
        if issues:
            assert answer["status"] == "needs_human_review"
        # Runtime metadata contains source_ids, but never evaluator gold fields.
        for call in answer["model_calls"]:
            for field in ["expected_key_points", "critical_failures", "reference_ids"]:
                assert ('"' + field + '"') not in call["user_prompt"]

    profiles = sorted((ROOT / "config/models").glob("*.json"))
    for path in profiles:
        load_config(path)
    secrets = []
    missing_credentials = []
    for path in [ROOT / "config/models/gemini.json", ROOT / "config/models/nvidia-kimi-k3.json", ROOT / "config/models/openrouter-claude-opus-5.5.json"]:
        try:
            key, _ = resolve_credential(load_config(path), ROOT)
        except ConfigurationError:
            # Public artifact checks must also run in a clone without private keys.
            missing_credentials.append(path.name)
            continue
        if key:
            secrets.append(key)
    public_files = []
    for directory in ["src", "tests", "scripts", "data", "docs", "config", ".agents"]:
        public_files += [path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts and not any(part.endswith(".egg-info") for part in path.parts)]
    public_files += [ROOT / name for name in ["README.md", "AGENTS.md", "pyproject.toml", "requirements.lock.txt", ".env.example", ".gitignore"]]
    for path in public_files:
        content = path.read_bytes()
        assert not any(key.encode("utf-8") in content for key in secrets), "Credential detected in a public artifact."

    broken_links = []
    link_count = 0
    documents = [ROOT / "README.md", ROOT / "docs/implementation/requirement-2.md", ROOT / ".agents/journal.md"]
    for document in documents:
        text = document.read_text(encoding="utf-8-sig")
        for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if destination.startswith(("http://", "https://", "#", "mailto:")):
                continue
            local = destination.split("#")[0].strip("<>")
            if local:
                link_count += 1
                if not (document.parent / unquote(local)).resolve().exists():
                    broken_links.append({"document": str(document.relative_to(ROOT)), "target": local})
    assert not broken_links, "Broken local documentation links: " + json.dumps(broken_links)

    pdf_path = ROOT / "docs/implementation/requirement-2-report.pdf"
    tex_path = pdf_path.with_suffix(".tex")
    log_path = pdf_path.with_suffix(".log")
    tex = tex_path.read_text(encoding="utf-8")
    bibitems = re.findall(r"\\bibitem\[[^\]]+\]\{([^}]+)\}", tex)
    assert set(bibitems) == source_identifiers and len(bibitems) == len(sources)
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"Overfull \\hbox|Missing character|(?:LaTeX|Package[^\n]*) Warning", log)
    pdf = pymupdf.open(pdf_path)
    assert pdf.page_count >= 30
    full_text = "\n".join(page.get_text() for page in pdf)
    for qid in ids:
        assert qid in full_text
    assert all(secret not in full_text for secret in secrets)
    urls = {link["uri"] for page in pdf for link in page.get_links() if link.get("uri")}
    assert {source["url"] for source in sources}.issubset(urls)
    edge_spans = []
    for index, page in enumerate(pdf):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < 38 or x1 > page.rect.width - 38 or y0 < 20 or y1 > page.rect.height - 20:
                        edge_spans.append({"page": index + 1, "bbox": span["bbox"]})
    assert not edge_spans, "PDF text approaches the page edge: " + json.dumps(edge_spans[:5])
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "live_api_calls": 0, "question_count": 10, "source_count": len(sources), "model_profiles_validated": len(profiles), "study_fingerprints_match": True, "archived_modules_verified": len(manifest["code_sha256"]), "documented_source_changes": changes, "assessment_hashes_and_excerpts_match": True, "gold_fields_absent_from_agent_prompts": True, "credentials_absent_from_public_artifacts": True if secrets else None, "known_credentials_scanned": len(secrets), "unavailable_credential_profiles": missing_credentials, "files_scanned_for_credentials": len(public_files), "local_documentation_links_checked": link_count, "broken_local_links": [], "pdf_pages": pdf.page_count, "pdf_link_annotations": sum(len(page.get_links()) for page in pdf), "all_source_urls_present": True, "pdf_edge_spans": [], "latex_warnings": 0, "pdf_sha256": file_hash(pdf_path), "tex_sha256": file_hash(tex_path), "interpretation": "Engineering/artifact verification only; not clinical validation or a new provider inference run. Credential scanning checks only locally available known keys; no values are saved."}
    write_json(ROOT / "data/checks/requirement-2-verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
