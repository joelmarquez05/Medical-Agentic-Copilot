"""Preserve live retrieval and validate a separately authored relevance review."""

from __future__ import annotations

import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from medical_copilot.evaluation import (engineering_metrics, file_hash, load_questions,
                                        package_versions, read_json, write_answers_markdown,
                                        write_json)
from medical_copilot.literature import LiteratureCopilot, PROMPT_VERSION
from medical_copilot.models import ModelClient, create_backend
from medical_copilot.pubmed import PubMedClient, utc_now


def retrieval_record(answer: dict[str, Any]) -> dict[str, Any]:
    searches = [result for result in answer.get("tool_results", []) if result["name"] == "search_pubmed"]
    papers = {}
    for search in searches:
        if search["status"] == "ok":
            for source in search["output"]["sources"]:
                papers.setdefault(source["pmid"], source)
    first = searches[0]["output"] if searches and searches[0]["status"] == "ok" else None
    return {"question_id": answer["question_id"], "question": answer["question"],
            "answer_status": answer["status"], "searches": searches,
            "initial_search_succeeded": first is not None,
            "initial_search_pmids": first["esearch_pmids"] if first else [],
            "initial_returned_pmids": [source["pmid"] for source in first["sources"]] if first else [],
            "unique_papers": list(papers.values()),
            "cited_pubmed_ids": [source["id"] for source in answer.get("sources", []) if source["id"].startswith("P")],
            "interpretation": "Retrieval observations only. Relevance ratings are authored separately; citation inclusion and self-review are not quality measures."}


def run_retrieval_evaluation(config, pubmed_config, root: Path, questions_path: Path,
                             rubric_path: Path, run_dir: Path, ids=None,
                             delay_seconds: float = 4) -> dict[str, Any]:
    dataset = load_questions(questions_path)
    selected = [item for item in dataset["questions"] if ids is None or item["id"] in ids]
    if not selected or (ids and set(ids) != {item["id"] for item in selected}):
        raise ValueError("Requested question IDs must exist in the internal set.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Choose an empty run directory; previous results are never overwritten.")
    rubric = read_json(rubric_path)
    if rubric["rubric_id"] != "oncology-pubmed-retrieval-v1":
        raise ValueError("Use the prespecified retrieval relevance rubric.")
    backend, redactor, credential_source = create_backend(config, root)
    run_dir.mkdir(parents=True, exist_ok=True)
    pubmed = PubMedClient(pubmed_config, run_dir / "http")
    agent = LiteratureCopilot(ModelClient(backend, config, redactor), pubmed)
    code = sorted((root / "src/medical_copilot").glob("*.py"))
    for path in code:
        snapshot = run_dir / "source-snapshot" / path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    manifest = {"run_id": run_dir.name, "evaluation": "requirement_3_internal_retrieval",
                "started_at_utc": utc_now(), "question_ids": [item["id"] for item in selected],
                "dataset_id": dataset["dataset_id"], "dataset_sha256": file_hash(questions_path),
                "rubric_id": rubric["rubric_id"], "rubric_sha256": file_hash(rubric_path),
                "guidance_sha256": file_hash(root / "src/medical_copilot/resources/oncology_guidance.json"),
                "code_sha256": {path.name: file_hash(path) for path in code},
                "model_config": config.public_dict(), "pubmed_config": pubmed_config.model_dump(),
                "package_versions": package_versions(), "prompt_version": PROMPT_VERSION,
                "credential_source": credential_source, "credentials_saved": False,
                "external_tracing": False, "question_delay_seconds": delay_seconds,
                "search_policy": "One model-formulated initial PubMed query with 5 relevance-ranked results, and at most one reviewer-requested query. Local guidance/calculation may supplement the answer.",
                "assessment_policy": "Separate unblinded relevance review of titles and available abstracts. No full-text appraisal, exhaustive gold set, recall estimate, or new answer-quality grade.",
                "status": "running", "completed_questions": 0}
    write_json(run_dir / "manifest.json", redactor.object(manifest))
    (run_dir / "graph.mmd").write_text(agent.graph.get_graph().draw_mermaid(), encoding="utf-8")
    results, records = [], []
    for index, item in enumerate(selected):
        # Gold answers, clinical reference IDs, and evaluation criteria never enter the graph.
        answer = agent.run(item["question"])
        answer.update(question_id=item["id"], topic=item["topic"])
        record = retrieval_record(answer)
        write_json(run_dir / "answers" / (item["id"] + ".json"), answer)
        write_json(run_dir / "retrieval" / (item["id"] + ".json"), record)
        results.append(answer)
        records.append(record)
        manifest["completed_questions"] = len(results)
        write_json(run_dir / "manifest.json", redactor.object(manifest))
        print(f"{item['id']}: {answer['status']}; {len(record['searches'])} PubMed searches; {len(record['unique_papers'])} unique papers; {len(answer['model_calls'])} model calls", flush=True)
        if index + 1 < len(selected) and delay_seconds:
            time.sleep(delay_seconds)
    manifest.update(status="completed" if all(answer["status"] != "error" for answer in results) else "completed_with_errors", finished_at_utc=utc_now())
    write_json(run_dir / "manifest.json", redactor.object(manifest))
    metrics = engineering_metrics(results)
    metrics["retrieval"] = {"pubmed_search_count": sum(len(record["searches"]) for record in records),
                            "successful_search_count": sum(search["status"] == "ok" for record in records for search in record["searches"]),
                            "questions_with_papers": sum(bool(record["unique_papers"]) for record in records),
                            "question_paper_pairs": sum(len(record["unique_papers"]) for record in records),
                            "globally_unique_pmids": len({source["pmid"] for record in records for source in record["unique_papers"]}),
                            "http_request_count": len(pubmed.requests),
                            "http_attempt_count": sum(len(request["attempts"]) for request in pubmed.requests),
                            "automatic_relevance_grading": False}
    write_json(run_dir / "engineering-metrics.json", metrics)
    write_json(run_dir / "http-requests.json", pubmed.requests)
    write_answers_markdown(run_dir, results)
    return metrics


def summarize_retrieval_assessments(run_dir: Path, assessments_path: Path, rubric_path: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    review = read_json(assessments_path)
    rubric = read_json(rubric_path)
    if review["run_id"] != manifest["run_id"] or review["rubric_id"] != rubric["rubric_id"] or file_hash(rubric_path) != manifest["rubric_sha256"]:
        raise ValueError("Retrieval assessment run/rubric identifiers or fingerprints do not match.")
    items = review["assessments"]
    ids = [item["question_id"] for item in items]
    if len(ids) != len(set(ids)) or set(ids) != set(manifest["question_ids"]):
        raise ValueError("Exactly one retrieval assessment is required for each question.")
    counts = Counter()
    rows = []
    question_paper_pairs = 0
    for item in items:
        path = run_dir / "retrieval" / (item["question_id"] + ".json")
        record = read_json(path)
        if file_hash(path) != item["retrieval_file_sha256"]:
            raise ValueError("An assessed retrieval record changed after review.")
        expected = {source["pmid"] for source in record["unique_papers"]}
        actual = [paper["pmid"] for paper in item["paper_assessments"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Every retrieved unique paper must have exactly one relevance assessment.")
        if type(item["coverage_rating"]) is not int or item["coverage_rating"] not in {0, 1, 2} or not item["coverage_rationale"].strip():
            raise ValueError("Coverage requires an ordinal 0-2 rating and rationale.")
        ratings = {}
        for paper in item["paper_assessments"]:
            score = paper["relevance"]
            if type(score) is not int or score not in {0, 1, 2} or not paper["rationale"].strip():
                raise ValueError("Paper relevance requires an ordinal 0-2 rating and rationale.")
            ratings[paper["pmid"]] = score
            counts[score] += 1
        question_paper_pairs += len(ratings)
        initial = record["initial_returned_pmids"]
        first_counts = {str(score): sum(ratings[pmid] == score for pmid in initial) for score in [0, 1, 2]}
        rows.append({"question_id": item["question_id"], "coverage_rating": item["coverage_rating"],
                     "initial_returned_count": len(initial), "initial_relevance_counts": first_counts,
                     "pooled_unique_paper_count": len(ratings),
                     "pooled_relevance_counts": {str(score): sum(value == score for value in ratings.values()) for score in [0, 1, 2]}})
    summary = {"run_id": manifest["run_id"], "rubric_id": rubric["rubric_id"],
               "question_count": len(items), "question_paper_pairs": question_paper_pairs,
               "relevance_counts": {str(score): counts[score] for score in [0, 1, 2]},
               "coverage_counts": {str(score): sum(item["coverage_rating"] == score for item in items) for score in [0, 1, 2]},
               "questions": rows, "assessor": review["assessor"],
               "blinded": review["blinded"], "clinician_review": review["clinician_review"],
               "interpretation": "Descriptive title/abstract relevance in a small constructed set. Not recall, retrieval accuracy, clinical validation, or a controlled comparison with requirement 2."}
    write_json(run_dir / "retrieval-summary.json", summary)
    text = ["# Separate PubMed retrieval relevance assessment\n",
            f"Run: `{manifest['run_id']}`. Rubric: `{rubric['rubric_id']}`. Reviewer: {review['assessor']}. Blinded: {review['blinded']}. Clinician review: {review['clinician_review']}.\n",
            "Paper relevance: 2 directly useful, 1 partially related, 0 irrelevant for the question. Coverage is a separate ordinal judgment of important question concepts in the returned title/abstract material. Answers are preserved separately and are not assigned a new answer-quality score.\n",
            "| Question | Initial direct / partial / irrelevant | All unique papers | Coverage (0-2) |\n|---|---|---|---|\n"]
    for row in rows:
        values = row["initial_relevance_counts"]
        text.append(f"| {row['question_id']} | {values['2']} / {values['1']} / {values['0']} | {row['pooled_unique_paper_count']} | {row['coverage_rating']} |")
    for item in items:
        record = read_json(run_dir / "retrieval" / (item["question_id"] + ".json"))
        papers = {source["pmid"]: source for source in record["unique_papers"]}
        text.extend([f"\n## {item['question_id']}\n", record["question"] + "\n",
                     f"Coverage {item['coverage_rating']}/2: {item['coverage_rationale']}\n",
                     "Search findings: " + item["search_findings"] + "\n"])
        for assessment in item["paper_assessments"]:
            paper = papers[assessment["pmid"]]
            text.append(f"- **Relevance {assessment['relevance']}/2** - [{paper['title']}]({paper['url']}) (PMID {paper['pmid']}, {paper['year']}; {paper['read_level']}): {assessment['rationale']}")
    text.extend(["\n## Limits\n", summary["interpretation"], review["limitations"]])
    (run_dir / "retrieval-assessment.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary
