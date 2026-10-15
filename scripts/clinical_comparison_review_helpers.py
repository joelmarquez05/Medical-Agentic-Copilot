"""Persist explicitly authored per-case review notes; never infer medical grades."""
import re
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"
ANSWER_DIMS = ("case_reasoning", "reference_concordance", "calibration_and_provenance")
RATINGS = {"D": "directly_relevant", "P": "partially_relevant", "I": "irrelevant", "N": "not_assessable"}


def extra_reference_mentions(row):
    """Find reference mentions beyond the primary structured citation array."""
    response = row.get("response") or {}
    body = "\n".join([response.get("answer", ""), response.get("justification", ""), *response.get("limitations", [])])
    mentions = set(re.findall(r"\bP[1-9]\d{0,9}(?:_(?:search|read)_[a-f0-9]{12})?\b", body))
    primary = set(response.get("citations", []))
    primary_pmids = {row["evidence"][sid]["pmid"] for sid in primary if sid in row["evidence"]}
    return {sid for sid in mentions - primary if "_" in sid or sid[1:] not in primary_pmids}


def save_review(slug, identifier, text, *, flags=None, queries=(), papers=(), citations=(), refs=(), inline_citations=()):
    """All text and grades are supplied by the reviewer after reading saved output."""
    assert set(flags or {}) <= set(ANSWER_DIMS)
    assert set((flags or {}).values()) <= {"consistent", "minor_limitation", "material_issue", "not_evaluable"}
    path = RUN / slug / "answers" / (identifier + ".json")
    row = read_json(path)
    paper_map = {pmid: {"rating": RATINGS[rating], "notes": reason} for pmid, rating, reason in papers}
    actual_papers = {p["pmid"] for e in row["literature"]["events"] for p in e["papers"]}
    assert set(paper_map) == actual_papers
    query_list = [{"rating": rating, "notes": reason} for rating, reason in queries]
    assert len(query_list) == sum(e["name"] == "search_pubmed" for e in row["literature"]["events"])
    cite_map = {}
    for selector, rating, reason in citations:
        if selector in row["response"]["citations"]:
            sid = selector
        else:
            matches = [sid for sid in row["response"]["citations"] if row["evidence"].get(sid, {}).get("pmid") == selector]
            assert len(matches) == 1, (slug, identifier, selector)
            sid = matches[0]
        cite_map[sid] = {"rating": rating, "notes": reason}
    assert set(cite_map) == set(row["response"]["citations"])
    inline_map = {sid: {"rating": rating, "notes": reason} for sid, rating, reason in inline_citations}
    assert set(inline_map) == extra_reference_mentions(row)
    note = {"question_id": identifier, "model_profile": slug, "answer_sha256": file_hash(path),
            "reviewer": "Unblinded implementation assistant; manual post-generation assessment, no paid judge or independent clinician",
            "ratings": {dim: (flags or {}).get(dim, "consistent") for dim in ANSWER_DIMS},
            "answer_notes": text, "researcher_reference_ids": list(refs), "queries": query_list,
            "papers": paper_map, "citations": cite_map, "inline_citations": inline_map}
    save_trace(RUN / "manual-reviews" / slug / (identifier + ".json"), note)
