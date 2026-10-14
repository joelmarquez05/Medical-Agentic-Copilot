"""Save explicit unblinded assistant notes for the fixed observed v2 cohort.

No model/API request or automatic medical judge is used. Keys are not modified.
"""

import hashlib
import json
from pathlib import Path

from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.clinical_graph import save_trace

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/retrieval-runs/gemma-evidence-main-20261011"

# Case-level observations authored after inspecting exact answers and visible views.
NOTES = {
 "IC01": ("consistent", "consistent", "minor_limitation", "Correct B, uses antibiotic exposure, watery diarrhea and supplied inflammatory findings. Direct knowledge-based answer; no source obtained. Its limitation says no registry was provided rather than clearly saying it chose not to verify with the available tools; no fabricated citation. Most-likely diagnosis is not a confirmed stool result."),
 "IC02": ("consistent", "minor_limitation", "consistent", "Correct B and decisive positional pain/rub. Retains a pleuritic descriptor not specified by the stem. Explicitly acknowledges no PubMed verification; no invented ECG/echo or citation."),
 "IC03": ("minor_limitation", "minor_limitation", "consistent", "Correct A and coherent hypothyroidism/TRH explanation. Now mentions clozapine and pituitary alternatives, but normal visual fields alone cannot exclude a small prolactinoma; the untreated hypothyroidism is the stronger supplied clue. Length/repetition exceeds the normal concise target. Acknowledges unverified evidence. The clozapine attribution is not externally checked by this agent."),
 "IC04": ("minor_limitation", "material_issue", "consistent", "C still differs from released B. Recognizes limb-threatening ischemia, but mentions possible Rutherford III alongside IIb without distinguishing the nonsalvageable category's different management. The guideline supports immediate UFH and emergency viability/revascularization assessment for salvageable ischemia; no Doppler findings establish a complete stage here. Urgent rescue can be clinically appropriate for IIb, while the unexplained III alternative is a material imprecision. No PubMed search was chosen; original key/prioritization ambiguity remains."),
 "IC05": ("consistent", "minor_limitation", "consistent", "Correct B based on previous newborn invasive GBS disease; avoids delaying prophylaxis. Introduces term for the previous infant although its gestational age is not supplied and is not needed for the indication. ACOG/CDC attribution remains knowledge-based, explicitly unverified by PubMed."),
 "IC06": ("consistent", "consistent", "consistent", "Correct A, uses severe hypertension/thrombocytopenia and distinguishes stabilization from delivery. No invented elevated liver enzymes, dose or full delivery plan. Explicitly states absent PubMed verification; no retrieved guideline citation."),
 "IC07": ("consistent", "consistent", "consistent", "Correct D and neutralization mechanism; contrasts alternatives without inventing a reversal dose. Explicitly states no PubMed verification; no source obtained."),
 "IC08": ("not_evaluable", "not_evaluable", "not_evaluable", "No final answer: after a successful search, a nested action object violates the unchanged JSON contract. Its attempted read targets a secondary BALANCE analysis; it is not executed. The original main trial was not among these five results. This is an operational failure and a retrieval-selection limitation, not an observed wrong medical option."),
 "IC09": ("consistent", "consistent", "consistent", "Correct D with DVT/PE rates supported by the actually supplied original-trial abstract and a valid read-view citation. Does not repeat baseline's hip-fracture-only population description; it does not explicitly discuss eligibility either. The key was already correct, so this is better observable evidence support, not an accuracy gain."),
 "IC10": ("not_evaluable", "not_evaluable", "not_evaluable", "No final answer: two near-equivalent searches return the same two related nonprimary papers. At the last permitted turn, tools are absent but Gemma requests another search; the graph stops without an additional operation or model call. No primary SOUL result is retrieved; no incorrect final medical belief is scored.")
}
PAPERS = {
 "40116824": ("partially_relevant", "Related 7/14-day meta-analysis, not the requested original BALANCE trial; visible excerpt lacks its exact mortality result."),
 "27885969": ("irrelevant", "2016 conference compilation and visible sepsis snippets do not address the requested 2024/2025 main trial result."),
 "40239817": ("irrelevant", "Tests beta-lactam infusion schedules rather than the named trial's treatment duration/mortality."),
 "31941546": ("partially_relevant", "Related BALANCE pilot; feasibility work precedes and cannot substitute for the main trial's numerical result."),
 "42371622": ("partially_relevant", "BALANCE secondary procalcitonin analysis; excerpt mentions noninferiority but not requested main-trial mortality rates. Attempted read was invalid and not executed."),
 "36652352": ("directly_relevant", "Original PREVENT CLOT randomized trial. Search excerpt alone omits rates; actual read provides the requested DVT figures and PE distinction."),
 "33762229": ("partially_relevant", "PREVENT CLOT protocol, not published outcome results."),
 "41056444": ("partially_relevant", "Secondary fracture/subpopulation analysis, not the original aggregate trial result."),
 "42355718": ("partially_relevant", "Broader orthopaedic prophylaxis review; visible excerpt is not the specific original result."),
 "38079260": ("partially_relevant", "PREVENT CLOT risk-stratified secondary analysis rather than the original aggregate DVT percentages."),
 "42169198": ("partially_relevant", "SOUL-based population projection, not the requested primary randomized trial; visible excerpt lacks exact HR/CI/event rates."),
 "39115619": ("partially_relevant", "Pre-result forecasting study, not the observed SOUL trial outcome. Predicted findings must not substitute for actual results.")
}


def main():
    if (RUN / "answer-assessments.json").exists() or (RUN / "retrieval-assessments.json").exists():
        raise ValueError("Preserve existing assessments; do not replace implicitly.")
    keys = {row["question_id"]: row for row in read_json(RUN / "answer-key.json")["keys"]}
    answer_reviews, retrieval_reviews = [], []
    dimensions = ["case_reasoning", "reference_concordance", "calibration_and_provenance"]
    for identifier, note in NOTES.items():
        path = RUN / "answers" / (identifier + ".json")
        record = read_json(path)
        fingerprint = file_hash(path)
        answer_reviews.append({"question_id": identifier, "answer_sha256": fingerprint,
            "option_score_unchanged": True, "dimension_ratings": dict(zip(dimensions, note[:3])),
            "brief_observed_notes": note[3], "reference_source_ids": keys[identifier]["reference_source_ids"]})
        events = record["literature"]["events"]
        returned, views, queries = {}, {}, []
        for event in events:
            for paper in event["papers"]:
                returned[paper["pmid"]] = paper
            for view in event.get("generation_view", {}).get("sources", []):
                views[view["id"]] = view
            if event["name"] == "search_pubmed":
                queries.append({"query": event["arguments"]["query"], "status": event["status"],
                    "rating": "appropriate" if identifier == "IC09" else "partially_appropriate",
                    "brief_reason": "Targets the correct trial and intervention." if identifier == "IC09" else
                      "Targets the named trial but includes literal/conjunctive outcome terms that yielded related papers rather than its original report; reordered SOUL terms did not broaden the query."})
        assessments = []
        for pmid, paper in returned.items():
            rating, reason = PAPERS[pmid]
            pviews = [view for view in views.values() if view["pmid"] == pmid]
            assessments.append({"pmid": pmid, "title": paper["title"], "authors": paper["authors"],
                "year": paper["year"], "doi": paper["doi"], "url": paper["url"],
                "generation_view_ids": [view["id"] for view in pviews],
                "generation_view_sha256": {view["id"]: hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False).encode()).hexdigest() for view in pviews},
                "model_visible_text_kinds": sorted({view["text_kind"] for view in pviews}),
                "paper_relevance": rating, "brief_reason": reason,
                "assessment_text_basis": "Actual title and delivered excerpt/abstract; hidden host text is not credited as model reading."})
        citations = []
        for source_id in (record.get("response") or {}).get("citations", []):
            assert source_id in views and identifier == "IC09" and views[source_id]["pmid"] == "36652352"
            text = views[source_id]["text"]
            assert all(number in text for number in ["2.51%", "1.71%", "1.49%"])
            citations.append({"source_id": source_id, "pmid": "36652352", "rating": "supported",
                "claim_checks": [{"claim": "DVT 2.51% aspirin versus 1.71% LMWH", "rating": "supported", "locator": "Delivered abstract RESULTS"},
                                 {"claim": "PE 1.49% in each group", "rating": "supported", "locator": "Delivered abstract RESULTS"}],
                "limits": "Abstract support for these specific outcome statements, not full-text review or validation of broader clinical recommendations."})
        expected = keys[identifier].get("pmid")
        retrieval_reviews.append({"question_id": identifier, "answer_sha256": fingerprint,
            "search_decision": "searched" if queries else "answered_directly",
            "query_assessments": queries, "paper_assessments": assessments, "citation_assessments": citations,
            "reference_example_found": expected in returned if expected else None,
            "reference_example_pmid": expected, "answer_evidence_notes": note[3] if queries else
                "Direct knowledge-based answer; no papers retrieved or citation support to assess. Choice is allowed; source verification remains absent.",
            "interpretation": "Autonomy/relevance/source-support observation, separate from exact key scoring."})
    assert {p["pmid"] for row in retrieval_reviews for p in row["paper_assessments"]} == set(PAPERS)
    reviewer = "Unblinded implementation assistant; explicit post-generation notes, no paid judge or independent clinician"
    save_trace(RUN / "answer-assessments.json", {"reviewer": reviewer, "rubric_sha256": file_hash(RUN / "answer-rubric.json"), "reviews": answer_reviews, "keys_changed": 0})
    save_trace(RUN / "retrieval-assessments.json", {"reviewer": reviewer, "rubric_sha256": file_hash(RUN / "retrieval-rubric.json"), "episodes": retrieval_reviews,
        "limitation": "Only three episodes execute retrieval. Twelve unique question-paper pairs; no exhaustive recall or general clinical/source-grounding estimate."})
    # Every actually returned source, including preserved v1/smoke engineering records.
    entries = {}
    origins = [ROOT / "data/checks/clinical-pubmed-smoke-20261011/smoke.json"]
    for name in ["gemma-autonomous-pilot-20261011", "gemma-autonomous-main-20261011", "gemma-evidence-pilot-20261011", "gemma-evidence-main-20261011"]:
        origins += list((ROOT / "data/evaluation/retrieval-runs" / name / "answers").glob("*.json"))
    for path in origins:
        row = read_json(path)
        for event in row["literature"]["events"]:
            for paper in event["papers"]:
                entry = entries.setdefault(paper["pmid"], {"id": "PMID" + paper["pmid"], "pmid": paper["pmid"], "title": paper["title"],
                    "authors_or_organization": paper["authors"], "year": paper["year"], "doi": paper["doi"], "url": paper["url"],
                    "journal": paper["journal"], "preprint": paper["preprint"], "supports": "Observed actual PubMed record; clinical relevance/claim support graded separately for the current main cohort", "origins": []})
                entry["origins"].append({"artifact": path.relative_to(ROOT).as_posix(), "tool": event["name"],
                    "read_level": [view["text_kind"] for view in event["generation_view"]["sources"] if view["pmid"] == paper["pmid"]],
                    "model_episode": row.get("question", {}).get("question_id"),
                    "note": "Smoke is researcher-chosen engineering retrieval; model received only its episode's generation views."})
    write_json(ROOT / "data/research/requirement-3-restart-observed-papers-20261011.json", {"sources": list(entries.values()), "count": len(entries),
        "read_limit": "Raw bounded PubMed abstracts in snapshots; delivered search excerpts/read abstracts only. No full-text paper inspection or independent study-quality appraisal."})
    print(json.dumps({"answer_reviews": len(answer_reviews), "retrieval_episodes": len(retrieval_reviews),
                      "current_question_paper_pairs": len(PAPERS), "all_observed_unique_papers": len(entries), "new_inference_requests": 0}))


if __name__ == "__main__":
    main()
