"""Record explicit, unblinded notes on the preserved version-3 study; no API judge."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/retrieval-runs/gemma-resilient-main-20261011"

NOTES = {
    "IC01": ("consistent", "consistent", "minor_limitation", "Correct B; supplied antibiotic exposure, diarrhea and leukocytosis support the most-likely diagnosis. Three real sources are cited, but their 400-character views do not establish every fluoroquinolone/mechanism claim. No stool result is invented. Citation-array IDs are valid; prose uses shortened PMID-like aliases rather than exact view IDs."),
    "IC02": ("consistent", "minor_limitation", "consistent", "Correct B; positional pain and the supplied rub/negative examination findings drive the answer. Pleuritic pain and worsening lying flat are added descriptors rather than exact stem details. Direct knowledge-based answer acknowledges absent verification."),
    "IC03": ("consistent", "minor_limitation", "consistent", "Correct A with untreated hypothyroidism, elevated TSH and hyperprolactinemia. Actual human hypothyroidism excerpt supports the association; the answer explicitly distinguishes the unshown TRH mechanism as background knowledge. Clozapine attribution is not verified by these views. Repetition and stating that TSH alone confirms primary disease reduce precision; option remains unchanged."),
    "IC04": ("minor_limitation", "material_issue", "minor_limitation", "C still differs from released B. The answer recognizes limb-threatening ischemia and acknowledges immediate heparin, but prioritizes definitive thrombectomy over the released initial-treatment choice. This known prioritization ambiguity remains. It calls IIb diagnostic without Doppler/complete viability assessment and labels IIa viable rather than marginally threatened. No current guideline was actually retrieved/read; the two endovascular cohort excerpts cannot validate its preferred next step. Its statement that no evidence was verified is distinct from having performed a search."),
    "IC05": ("consistent", "consistent", "minor_limitation", "Correct B: previous infant invasive GBS disease supports prophylaxis without delaying for screening, consistent with the earlier evaluator ACOG reference. Five background records were actually supplied, so the limitation claiming no evidence registry was provided misstates execution. None of these excerpts verifies the decisive prior-infant guideline indication. No citation is fabricated."),
    "IC06": ("consistent", "consistent", "consistent", "Correct A with severe hypertension/thrombocytopenia and stabilization before delivery. A restrictive search returns zero papers; the model still answers from knowledge and acknowledges lack of PubMed verification. No falsely elevated liver enzymes, dose or retrieved recommendation is asserted."),
    "IC07": ("consistent", "consistent", "consistent", "Correct D and neutralization mechanism, with no invented reversal dose. Direct answer acknowledges absent verification. The broad description of other therapies does not support a patient-specific regimen."),
    "IC08": ("consistent", "minor_limitation", "minor_limitation", "Correct B. An actual original BALANCE search and read supply 14.5% versus 16.1% mortality and noninferiority. The explanation accurately gives the -1.6 percentage-point difference and interval bounds but calls the intention-to-treat interval 95% instead of the abstract's 95.7%; the selected option is unaffected. No normalization/retry was needed in this new live episode."),
    "IC09": ("consistent", "consistent", "consistent", "Correct D, with DVT 2.51% versus 1.71% and PE 1.49% in both groups supported by the actually read original PREVENT CLOT abstract. No hip-fracture-only population description. Correct key and stronger observable attribution, not an option-accuracy gain over baseline."),
    "IC10": ("consistent", "minor_limitation", "minor_limitation", "Correct C after search, a rejected view-ID-instead-of-PMID argument, a corrected read, and reserved fourth-call finalization. The actually read SOUL review supports HR 0.86 and CI 0.77-0.96 but omits 12.0% versus 13.8% rates. The original trial was not found. Event rates in the answer agree with the frozen key but cannot be credited to retrieved evidence; empty limitations overstate complete verification. Its only-choice HR wording is loose since D has the same HR and reverses event rates."),
}

# These grades are authored from actual titles and model-delivered views, not
# computed from an evaluator PMID match or assumed from a full host abstract.
PAPERS = {
    "38387648": ("partially_relevant", "Aspiration-pneumonia antibiotic harms context; visible excerpt does not establish this diarrhea diagnosis."),
    "18067688": ("directly_relevant", "Review explicitly addresses fluoroquinolone predisposition to C. difficile disease; visible start establishes antimicrobial exposure generally."),
    "24907488": ("partially_relevant", "Immunocompromised-patient microbiology context; excerpt supports flora disruption generally, not levofloxacin-specific risk."),
    "17620240": ("partially_relevant", "Fluoroquinolone-resistant C. difficile outbreak; related organism/exposure, not the case's diagnostic discrimination."),
    "28736259": ("partially_relevant", "Molecular epidemiology/susceptibility; introductory antibiotic-associated diarrhea context rather than decisive fluoroquinolone evidence."),
    "39227558": ("irrelevant", "Traumatic-brain-injury pituitary dysfunction differs from this primary untreated hypothyroidism mechanism."),
    "18996577": ("irrelevant", "Gonadectomy/cabergoline/TRH experiments in dogs do not address this human primary-thyroid case."),
    "3344469": ("directly_relevant", "Human primary-hypothyroidism/hyperprolactinemia association and pituitary differential; excerpt does not show the complete TRH explanation."),
    "1832134": ("irrelevant", "Female androgenic hair-loss hormone testing does not explain this hypothyroid galactorrhea case."),
    "7753340": ("directly_relevant", "Human thyroid-status/prolactin/TRH mechanism topic; excerpt begins with a related VIP hypothesis and is insufficient to verify all mechanisms."),
    "38871066": ("partially_relevant", "ALI percutaneous outcomes and sex differences, not the immediate heparin-versus-rescue management question."),
    "33068768": ("partially_relevant", "Endovascular-first ALI management cohort; introductory uncertainty without the decisive next-step recommendation."),
    "35490693": ("partially_relevant", "Global GBS burden rather than prior affected infant prophylaxis indication."),
    "30808831": ("partially_relevant", "Late-pregnancy colonization/neonatal infection cohort rather than this immediate guideline indication."),
    "36917564": ("partially_relevant", "GBS maternal-vaccine impact/cost model, not intrapartum management after a previous affected infant."),
    "29117326": ("partially_relevant", "Infant disease epidemiology mentions IAP but not the decisive prior-infant recommendation."),
    "31830941": ("partially_relevant", "Neonatal infection incidence/etiology over time, not the specific prophylaxis decision."),
    "39565030": ("directly_relevant", "Original BALANCE trial; actual read supplies mortality, noninferiority and adjusted confidence level."),
    "40116824": ("partially_relevant", "Related duration meta-analysis, distinct from requested original BALANCE report."),
    "27885969": ("irrelevant", "2016 conference compilation and sepsis snippets do not address the later original trial result."),
    "41774760": ("partially_relevant", "Pseudomonas post hoc BALANCE subgroup, not original aggregate mortality result."),
    "40754718": ("partially_relevant", "BALANCE clinical-course secondary report; search excerpt omits aggregate primary mortality."),
    "36652352": ("directly_relevant", "Original PREVENT CLOT trial; actual read provides requested DVT and distinct PE rates."),
    "33762229": ("partially_relevant", "PREVENT CLOT protocol, not the published outcomes."),
    "41056444": ("partially_relevant", "Fracture/subpopulation secondary analysis, not original aggregate secondary outcome."),
    "42355718": ("partially_relevant", "Broader orthopaedic prophylaxis review; visible excerpt does not show specific original DVT rates."),
    "38079260": ("partially_relevant", "PREVENT CLOT risk-stratified secondary analysis rather than original aggregate outcomes."),
    "36945734": ("partially_relevant", "SOUL design/baseline report, not observed final outcomes."),
    "42610271": ("irrelevant", "SELECT inflammatory biomarker analysis concerns a different trial/population and preparation."),
    "41370018": ("directly_relevant", "SOUL results review directly provides primary HR/CI in the read abstract; secondary literature, not original trial; crude event rates absent."),
    "39115619": ("partially_relevant", "SOUL pre-result forecasting, not the actual primary trial outcome."),
    "42169198": ("partially_relevant", "SOUL population-impact projection; related results context, not original aggregate trial report."),
}

CITATIONS = {
    "IC01": {
        "18067688": ("partially_supported", "Fluoroquinolone review title and generic antimicrobial risk text support related risk; the short excerpt does not show the full levofloxacin/microbiota claim."),
        "24907488": ("partially_supported", "Normal intestinal flora disruption by antibiotics is discussed, but the excerpt does not establish significant fluoroquinolone risk."),
        "28736259": ("partially_supported", "Generic antibiotic-associated diarrhea is supported; specific fluoroquinolone predisposition/mechanism exceeds this excerpt."),
    },
    "IC03": {"3344469": ("partially_supported", "Hypothyroidism/hyperprolactinemia association is supported; full TRH mechanism is absent and explicitly acknowledged as background knowledge.")},
    "IC08": {"39565030": ("partially_supported", "Mortality 14.5%/16.1%, difference -1.6 points, bounds -4.0/0.8 and noninferiority are supported. The answer's 95% label conflicts with the actual intention-to-treat 95.7% interval.")},
    "IC09": {"36652352": ("supported", "Delivered original abstract explicitly reports DVT 2.51%/1.71% and PE 1.49% in each group.")},
    "IC10": {"41370018": ("partially_supported", "Read review supports HR 0.86 and CI 0.77-0.96, but supplies neither 12.0% nor 13.8% event rates. A valid source does not verify all answer claims.")},
}


def main():
    keys = {k["question_id"]: k for k in read_json(RUN / "answer-key.json")["keys"]}
    reviews, episodes, catalogue = [], [], {}
    for identifier, note in NOTES.items():
        path = RUN / "answers" / (identifier + ".json")
        record = read_json(path)
        fingerprint = file_hash(path)
        reviews.append({"question_id": identifier, "answer_sha256": fingerprint,
            "option_score_unchanged": True, "dimension_ratings": dict(zip(
                ["case_reasoning", "reference_concordance", "calibration_and_provenance"], note[:3])),
            "brief_observed_notes": note[3], "reference_source_ids": keys[identifier]["reference_source_ids"]})
        returned, views, queries = {}, {}, []
        for event in record["literature"]["events"]:
            returned.update({p["pmid"]: p for p in event["papers"]})
            views.update({v["id"]: v for v in event["generation_view"]["sources"]})
            if event["name"] == "search_pubmed":
                queries.append({"query": event["arguments"]["query"], "status": event["status"],
                    "rating": "partially_appropriate" if identifier in {"IC04", "IC05", "IC06"} else "appropriate",
                    "brief_reason": "Targets the correct case/trial topic; actual returned coverage is separately graded." if identifier not in {"IC04", "IC05", "IC06"} else
                        "Related topic but restrictive terms or missing decisive guideline wording limit coverage; no external verification follows."})
        assessments = []
        for pmid, paper in returned.items():
            grade, reason = PAPERS[pmid]
            pviews = [v for v in views.values() if v["pmid"] == pmid]
            assessments.append({"pmid": pmid, "title": paper["title"], "authors": paper["authors"], "year": paper["year"],
                "doi": paper["doi"], "url": paper["url"], "generation_view_ids": [v["id"] for v in pviews],
                "generation_view_sha256": {v["id"]: hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest() for v in pviews},
                "model_visible_text_kinds": sorted({v["text_kind"] for v in pviews}), "paper_relevance": grade, "brief_reason": reason})
            entry = catalogue.setdefault(pmid, {"id": "PMID" + pmid, "pmid": pmid, "title": paper["title"],
                "authors_or_organization": paper["authors"], "year": paper["year"], "doi": paper["doi"], "url": paper["url"],
                "journal": paper["journal"], "preprint": paper["preprint"], "supports": reason, "origins": []})
            entry["origins"].append({"artifact": path.relative_to(ROOT).as_posix(), "question_id": identifier,
                "read_level": sorted({v["text_kind"] for v in pviews}), "note": "Actual supplied views; no full-text inspection."})
        citation_notes = []
        for sid in record["response"].get("citations", []):
            assert sid in views
            view = views[sid]
            grade, reason = CITATIONS[identifier][view["pmid"]]
            citation_notes.append({"source_id": sid, "pmid": view["pmid"], "rating": grade, "brief_reason": reason,
                "model_visible_text_sha256": hashlib.sha256(view["text"].encode()).hexdigest(),
                "interpretation": "Specific delivered-text support, not medical-key judgment or full-text verification."})
        expected = keys[identifier].get("pmid")
        episodes.append({"question_id": identifier, "answer_sha256": fingerprint,
            "search_decision": "searched" if queries else "answered_directly", "query_assessments": queries,
            "paper_assessments": assessments, "citation_assessments": citation_notes,
            "reference_example_found": expected in returned if expected else None, "reference_example_pmid": expected,
            "answer_evidence_notes": note[3], "operational_tool_errors": sum(e["node"] == "execute_tool" and e["status"] == "error" for e in record["trace"])})
    assert set(PAPERS) == {p["pmid"] for e in episodes for p in e["paper_assessments"]}
    reviewer = "Unblinded implementation assistant; explicit post-generation notes, no paid judge or independent clinician"
    save_trace(RUN / "answer-assessments.json", {"reviewer": reviewer, "rubric_sha256": file_hash(RUN / "answer-rubric.json"),
        "reviews": reviews, "keys_changed": 0, "reference_reuse": "Baseline review-sources.json; previous journal entries. No evaluator reference entered generation."})
    save_trace(RUN / "retrieval-assessments.json", {"reviewer": reviewer,
        "rubric_sha256": file_hash(RUN / "retrieval-rubric.json"), "episodes": episodes,
        "limitation": "Title/excerpt/abstract review only; incomplete reference examples, no exhaustive recall or full-text validation."})
    counts = Counter(p["paper_relevance"] for e in episodes for p in e["paper_assessments"])
    supports = Counter(c["rating"] for e in episodes for c in e["citation_assessments"])
    save_trace(RUN / "assessment-summary.json", {"answer_reviews": len(reviews), "retrieval_episodes": len(episodes),
        "unique_question_paper_pairs": sum(counts.values()), "paper_relevance": dict(counts), "citation_support": dict(supports),
        "independent_clinician_review": False, "new_model_requests": 0})
    save_trace(ROOT / "data/research/requirement-3-resilient-observed-papers-20261011.json", {"sources": list(catalogue.values()),
        "count": len(catalogue), "read_limit": "Delivered bounded excerpts/read abstracts only; actual full host text not credited as model reading."})
    save_trace(ROOT / "data/research/requirement-3-resilient-technical-sources-20261011.json", {"sources": [
        {"id": "V3-TECH1", "title": "Graph API overview", "authors_or_organization": "LangChain", "year": "Living documentation, accessed 2026-10-11",
            "url": "https://docs.langchain.com/oss/python/langgraph/graph-api", "read_level": "Official documentation: state, nodes, conditional routing",
            "supports": "Explicit research/finalization graph branches; not clinical quality or native schema compatibility."},
        {"id": "V3-TECH2", "title": "Structured output", "authors_or_organization": "LangChain", "year": "Living documentation, accessed 2026-10-11",
            "url": "https://docs.langchain.com/oss/python/langchain/structured-output", "read_level": "Official ProviderStrategy/error-handling documentation",
            "supports": "Provider-dependent schema support and validation/recovery; this implementation uses explicit bounded graph calls."},
        {"id": "V3-TECH3", "title": "Structured Outputs", "authors_or_organization": "OpenRouter", "year": "Living documentation, accessed 2026-10-11",
            "url": "https://openrouter.ai/docs/guides/features/structured-outputs", "read_level": "Official JSON Schema/strict mode/endpoint-dependent compliance documentation",
            "supports": "Native final-only response_format; actual NextBit probe remains separate from advertised support."},
    ], "clinical_sources": "Reuse prior project-journal trial/guideline entries and baseline review-sources.json; researcher reads never prepopulate the answering agent."})
    print(json.dumps({"reviews": len(reviews), "papers": dict(counts), "citation_support": dict(supports), "new_model_requests": 0}))


if __name__ == "__main__":
    main()
