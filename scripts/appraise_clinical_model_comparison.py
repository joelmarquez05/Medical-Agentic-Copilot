"""Validate explicitly authored reviews and summarize their grades without an API judge."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.clinical_model_comparison import MODELS
from medical_copilot.evaluation import file_hash, read_json
from clinical_comparison_review_helpers import extra_reference_mentions

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"


def fingerprint(view):
    return hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    manifest = read_json(RUN / "manifest.json")
    assert manifest.get("finished_at_utc")
    rubric = read_json(RUN / "rubric.json")
    known_refs = {s["id"] for s in read_json(ROOT / "data/research/requirement-4-clinical-review-sources-20261011.json")["sources"]}
    known_refs.update(s["id"] for s in read_json(ROOT / "data/research/requirement-5-clinical-review-sources-20261011.json")["sources"])
    summaries, catalogue = {}, {}
    for slug in MODELS:
        rows = [read_json(p) for p in sorted((RUN / slug / "answers").glob("*.json"))]
        paths = sorted((RUN / "manual-reviews" / slug).glob("*.json"))
        notes = {read_json(p)["question_id"]: read_json(p) for p in paths}
        assert len(rows) == len(notes) == 30
        assert set(notes) == {r["question"]["question_id"] for r in rows}
        answers, retrieval = [], []
        for row in rows:
            identifier = row["question"]["question_id"]
            note = notes[identifier]
            digest = file_hash(RUN / slug / "answers" / (identifier + ".json"))
            assert note["answer_sha256"] == digest
            assert set(note["ratings"]) == set(rubric["answer_dimensions"])
            assert set(note["ratings"].values()) <= set(rubric["answer_ratings"])
            assert set(note["researcher_reference_ids"]) <= known_refs and note["answer_notes"]
            answers.append({"question_id": identifier, "answer_sha256": digest,
                            "dimension_ratings": note["ratings"], "brief_observed_notes": note["answer_notes"],
                            "researcher_reference_ids": note["researcher_reference_ids"], "option_score_unchanged": True})
            papers, views = {}, {}
            events = row["literature"]["events"]
            for event in events:
                papers.update({p["pmid"]: p for p in event["papers"]})
                views.update({v["id"]: v for v in event["generation_view"]["sources"]})
            assert set(note["papers"]) == set(papers)
            assert len(note["queries"]) == sum(e["name"] == "search_pubmed" for e in events)
            assessed_papers = []
            for pmid, paper in papers.items():
                pnote = note["papers"][pmid]
                assert pnote["rating"] in {"directly_relevant", "partially_relevant", "irrelevant", "not_assessable"} and pnote["notes"]
                pviews = [v for v in views.values() if v["pmid"] == pmid]
                assessed_papers.append({"pmid": pmid, "title": paper["title"], "paper_relevance": pnote["rating"],
                                       "brief_reason": pnote["notes"], "generation_view_ids": [v["id"] for v in pviews],
                                       "generation_view_sha256": {v["id"]: fingerprint(v) for v in pviews}})
                source = catalogue.setdefault(pmid, {"id": "PMID" + pmid, "pmid": pmid, "title": paper["title"],
                    "authors_or_organization": paper["authors"], "year": paper["year"], "doi": paper["doi"],
                    "url": paper["url"], "journal": paper["journal"], "preprint": paper["preprint"], "origins": [],
                    "supports": "Actual returned reference; topical relevance and claim support remain separate delivered-text appraisals."})
                source["origins"].append({"model_profile": slug, "question_id": identifier,
                    "artifact": (RUN / slug / "answers" / (identifier + ".json")).relative_to(ROOT).as_posix(),
                    "read_level": sorted({v["text_kind"] for v in pviews}), "relevance": pnote["rating"]})
            citations = []
            assert set(note["citations"]) == set((row.get("response") or {}).get("citations", []))
            for sid in (row.get("response") or {}).get("citations", []):
                cnote = note["citations"][sid]
                assert cnote["rating"] in rubric["support_ratings"] and cnote["notes"]
                view = views.get(sid)
                citations.append({"source_id": sid, "view_exists": view is not None, "pmid": view["pmid"] if view else None,
                                  "rating": cnote["rating"], "brief_reason": cnote["notes"],
                                  "generation_view_sha256": fingerprint(view) if view else None})
            assert set(note.get("inline_citations", {})) == extra_reference_mentions(row)
            inline = []
            for sid, cnote in note.get("inline_citations", {}).items():
                assert cnote["rating"] in rubric["support_ratings"] and cnote["notes"]
                view = views.get(sid)
                inline.append({"source_id": sid, "view_exists": view is not None,
                               "rating": cnote["rating"], "brief_reason": cnote["notes"],
                               "generation_view_sha256": fingerprint(view) if view else None})
            queries = []
            for event, qnote in zip([e for e in events if e["name"] == "search_pubmed"], note["queries"], strict=True):
                assert qnote["rating"] in {"appropriate", "partially_appropriate", "inappropriate"} and qnote["notes"]
                queries.append({"query": event["arguments"]["query"], "status": event["status"], **qnote})
            retrieval.append({"question_id": identifier, "answer_sha256": digest,
                              "query_assessments": queries, "paper_assessments": assessed_papers, "citation_assessments": citations,
                              "supplementary_inline_reference_assessments": inline, "answer_evidence_notes": note["answer_notes"]})
        provenance = {"reviewer": "Unblinded implementation assistant; explicit post-generation notes, no paid judge or independent clinician",
                      "rubric_sha256": file_hash(RUN / "rubric.json"),
                      "manual_review_sha256": {p.name: file_hash(p) for p in paths}, "keys_changed": 0}
        save_trace(RUN / slug / "answer-assessments.json", {**provenance, "reviews": answers})
        save_trace(RUN / slug / "retrieval-assessments.json", {**provenance,
            "retrieval_rubric_sha256": file_hash(RUN / "retrieval-rubric.json"), "episodes": retrieval})
        summary = {"answer_reviews": len(answers),
                   "material_issue_cases": [r["question_id"] for r in answers if "material_issue" in r["dimension_ratings"].values()],
                   "answer_dimension_counts": {dim: dict(Counter(r["dimension_ratings"][dim] for r in answers)) for dim in rubric["answer_dimensions"]},
                   "paper_pairs": sum(len(r["paper_assessments"]) for r in retrieval),
                   "paper_relevance": dict(Counter(p["paper_relevance"] for r in retrieval for p in r["paper_assessments"])),
                   "citation_support": dict(Counter(c["rating"] for r in retrieval for c in r["citation_assessments"])),
                   "unknown_citation_ids": sum(not c["view_exists"] for r in retrieval for c in r["citation_assessments"]),
                   "supplementary_inline_reference_count": sum(len(r["supplementary_inline_reference_assessments"]) for r in retrieval),
                   "query_ratings": dict(Counter(q["rating"] for r in retrieval for q in r["query_assessments"])),
                   "new_model_requests": 0, "independent_clinician_review": False}
        save_trace(RUN / slug / "assessment-summary.json", summary)
        summaries[slug] = summary
    save_trace(RUN / "assessment-summary.json", {"models": summaries, "gemma_reviews_reused": True, "new_model_requests": 0})
    save_trace(ROOT / "data/research/requirement-5-observed-papers-20261011.json", {
        "sources": list(catalogue.values()), "count": len(catalogue),
        "limits": "All new primary returned papers; view origins preserve model ownership and read extent. No full-text or exhaustive recall claim."})
    print(json.dumps({"models_reviewed": len(summaries), "answers_reviewed": 120,
                      "unique_primary_papers": len(catalogue), "new_model_requests": 0}))


if __name__ == "__main__":
    main()
