"""Validate and preserve explicit assistant-authored benchmark review notes; no API judge."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"


def fingerprint(view):
    return hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    notes = read_json(RUN / "review-notes-v2.json")
    rubric = read_json(RUN / "rubric.json")
    records = [read_json(p) for p in sorted((RUN / "answers").glob("*.json"))]
    assert len(records) == 30 and set(notes["answers"]) == {r["question"]["question_id"] for r in records}
    answers, retrieval, catalogue = [], [], {}
    used_papers, used_citations = set(), set()
    for row in records:
        identifier = row["question"]["question_id"]
        digest = file_hash(RUN / "answers" / (identifier + ".json"))
        note = notes["answers"][identifier]
        assert set(note["ratings"]) == set(rubric["answer_dimensions"])
        assert set(note["ratings"].values()) <= set(rubric["answer_ratings"]) and note["notes"]
        answers.append({"question_id": identifier, "answer_sha256": digest, "option_score_unchanged": True,
            "dimension_ratings": note["ratings"], "brief_observed_notes": note["notes"],
            "researcher_reference_ids": note.get("reference_sources", [])})
        papers, views, queries = {}, {}, []
        for event in row["literature"]["events"]:
            papers.update({p["pmid"]: p for p in event["papers"]})
            views.update({v["id"]: v for v in event["generation_view"]["sources"]})
            if event["name"] == "search_pubmed":
                qnote = notes["queries"][identifier][len(queries)]
                assert qnote["rating"] in {"appropriate", "partially_appropriate", "inappropriate"}
                queries.append({"query": event["arguments"]["query"], "status": event["status"], **qnote})
        assert len(queries) == len(notes["queries"].get(identifier, []))
        assessed = []
        for pmid, paper in papers.items():
            key = identifier + ":" + pmid
            pnote = notes["papers"][key]
            assert pnote["rating"] in {"directly_relevant", "partially_relevant", "irrelevant", "not_assessable"} and pnote["notes"]
            used_papers.add(key)
            pviews = [v for v in views.values() if v["pmid"] == pmid]
            assessed.append({"pmid": pmid, "title": paper["title"], "authors": paper["authors"], "year": paper["year"],
                "doi": paper["doi"], "url": paper["url"], "paper_relevance": pnote["rating"], "brief_reason": pnote["notes"],
                "generation_view_ids": [v["id"] for v in pviews], "generation_view_sha256": {v["id"]: fingerprint(v) for v in pviews},
                "model_visible_text_kinds": sorted({v["text_kind"] for v in pviews})})
            entry = catalogue.setdefault(pmid, {"id": "PMID"+pmid, "pmid": pmid, "title": paper["title"],
                "authors_or_organization": paper["authors"], "year": paper["year"], "doi": paper["doi"], "url": paper["url"],
                "journal": paper["journal"], "preprint": paper["preprint"], "origins": [],
                "supports": "Actual benchmark-returned source record; relevance and delivered-text claim support are separately appraised."})
            entry["origins"].append({"artifact": (RUN / "answers" / (identifier+".json")).relative_to(ROOT).as_posix(),
                "question_id": identifier, "read_level": sorted({v["text_kind"] for v in pviews}), "relevance": pnote["rating"]})
        citations = []
        for sid in (row.get("response") or {}).get("citations", []):
            key = identifier + ":" + sid
            cnote = notes["citations"][key]
            assert cnote["rating"] in rubric["support_ratings"] and cnote["notes"]
            used_citations.add(key)
            view = views.get(sid)
            citations.append({"source_id": sid, "view_exists": view is not None, "pmid": view["pmid"] if view else None,
                "rating": cnote["rating"], "brief_reason": cnote["notes"], "claim_checks": cnote.get("claim_checks", []),
                "generation_view_sha256": fingerprint(view) if view else None})
        retrieval.append({"question_id": identifier, "answer_sha256": digest,
            "search_decision": "searched" if queries else "answered_directly", "query_assessments": queries,
            "paper_assessments": assessed, "citation_assessments": citations,
            "answer_evidence_notes": note["notes"], "tool_errors": sum(e["node"] == "execute_tool" and e["status"] == "error" for e in row["trace"])})
    assert set(notes["papers"]) == used_papers and set(notes["citations"]) == used_citations
    reviewer = "Unblinded implementation assistant; explicit post-generation notes, no paid judge or independent clinician"
    save_trace(RUN / "answer-assessments-v2.json", {"reviewer": reviewer, "rubric_sha256": file_hash(RUN / "rubric.json"),
        "review_notes_sha256": file_hash(RUN / "review-notes-v2.json"), "reviews": answers, "keys_changed": 0})
    save_trace(RUN / "retrieval-assessments-v2.json", {"reviewer": reviewer, "rubric_sha256": file_hash(RUN / "retrieval-rubric.json"),
        "review_notes_sha256": file_hash(RUN / "review-notes-v2.json"), "episodes": retrieval,
        "limit": "All returned question-paper pairs and cited views; actual delivered text only, no full-text/exhaustive recall claim."})
    relevance = Counter(p["paper_relevance"] for e in retrieval for p in e["paper_assessments"])
    support = Counter(c["rating"] for e in retrieval for c in e["citation_assessments"])
    save_trace(RUN / "assessment-summary-v2.json", {"answer_reviews": len(answers), "paper_relevance": dict(relevance),
        "unique_question_paper_pairs": len(used_papers), "citation_support": dict(support),
        "citation_view_count": len(used_citations), "unknown_citation_ids": sum(not c["view_exists"] for e in retrieval for c in e["citation_assessments"]),
        "answer_dimension_counts": {dim: dict(Counter(r["dimension_ratings"][dim] for r in answers)) for dim in rubric["answer_dimensions"]},
        "new_model_requests": 0, "independent_clinician_review": False})
    observed = {"sources": list(catalogue.values()), "count": len(catalogue),
        "read_limit": "Model received delivered bounded excerpts/abstracts; bibliography is not proof of support or full-text access."}
    catalogue_path = ROOT / "data/research/requirement-4-observed-papers-20261011.json"
    if catalogue_path.exists():
        assert read_json(catalogue_path) == observed
    else:
        save_trace(catalogue_path, observed)
    print(json.dumps({"answer_reviews": len(answers), "paper_pairs": len(used_papers), "citation_support": dict(support), "new_model_requests": 0}))


if __name__ == "__main__":
    main()
