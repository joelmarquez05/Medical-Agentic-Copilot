"""Persist explicit assistant-authored post-generation notes, without an LLM judge."""
import json
import re
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.pdf_agent import CONDITIONS
from build_pdf_report import file_hash_view

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/pdf-runs/gemma-pdf-main-20261012"
document = read_json(RUN / "document.json")

# These judgments were authored after reading every visible answer and each full
# returned/cited view. They are fixed human-review-style notes, not an automatic judge.
retrieval = {
    "PDF01": {(2, 2): ("direct", "B1-1 gives all three screening items, diagnosis start and clinical-stability repeat timing."),
        (3, 1): ("partial", "Screening/assessment figure caption identifies the topic, but the raster figure is not read and no requested B1-1 facts are present."),
        (2, 1): ("partial", "Introduction supports considering nutrition from diagnosis; abbreviations expand BMI, but screening items/repeat criterion are absent."),
        (12, 1): ("partial", "C6-1 repeats analogous intake/weight/BMI screening for advanced disease and references section 3.1; it is not the requested B1-1 timing text."),
        (5, 1): ("irrelevant", "Nutrition-route/refeeding and exercise recommendations do not establish B1-1 screening items or timing.")},
    "PDF02": {(4, 1): ("direct", "B3-1 commentary contains both percentage/duration examples; 50e75 is the retained numeric glyph ambiguity, independently verified on the original rendering."),
        (4, 2): ("partial", "Continues the nutrition-intervention section and feeding-route rationale, without either asked threshold/duration pair."),
        (2, 2): ("partial", "Provides intake assessment/energy/protein context, not the B3-1 medical-nutrition intake triggers."),
        (5, 1): ("partial", "Continues enteral/parenteral care and prolonged inadequate-intake precautions; not the two precise trigger thresholds.")},
    "PDF03": {(6, 1): ("direct", "B4-2 explicitly supplies weak recommendation/low evidence/strong consensus for resistance plus aerobic exercise."),
        (5, 1): ("direct", "B4-1 explicitly supplies strong recommendation/high evidence/consensus for physical activity."),
        (4, 1): ("irrelevant", "Different vitamin/fat/nutrition recommendations and evidence labels cannot establish the asked two exercise labels."),
        (13, 2): ("irrelevant", "Bibliography includes reference numbers 12/13 but no requested exercise recommendation labels; numerical matching is insufficient.")},
    "PDF04": {(13, 1): ("direct", "C6-3 contains comfort, lack of usual benefit and the short/limited acute-confusion hydration exception."),
        (12, 1): ("direct", "C6-2 contains patient discussion of prognosis, expected quality-of-life/potential-survival benefit and nutritional-care burden."),
        (16, 2): ("partial", "Related hydration/terminal-care article citations only; no abstracts/results or recommendation details are supplied."),
        (2, 1): ("irrelevant", "Introduction/methodology and abbreviations do not provide the advanced-versus-dying care distinction or exception.")},
    "PDF05": {(7, 1): ("direct", "B5-8 gives constipation assessment/treatment before prokinetics and correctly assigns CNS versus cardiac-rhythm risks."),
        (6, 1): ("irrelevant", "Neighboring appetite drugs/fish oil and exercise do not supply prokinetic prerequisite or the named drugs' adverse-effect domains."),
        (15, 1): ("irrelevant", "Bibliography on radiotherapy/feeding support does not establish the asked prokinetic details."),
        (7, 2): ("irrelevant", "Cannabinoids/androgenic steroids and NSAID continuation do not answer the prokinetic/constipation question.")},
    "PDF06": {},
}
notes = {
    "A-no-document": {
        "PDF01": (2, 0, "All asked B1-1 facts match the independently frozen reference. The answer acknowledges existing knowledge and no document verification; no PDF evidence/citation was supplied."),
        "PDF02": (2, 0, "Both exact intake percentages and strictly more-than duration qualifiers match. It explicitly labels them unverified against this PDF; correctness does not establish document access."),
        "PDF03": (2, 0, "Final answer correctly distinguishes both recommendation/evidence/consensus triples and discloses missing text/tools. Two earlier optionless visible answers used the reserved final attempts; no outside-quality rerun. No document verification is credited."),
        "PDF04": (1, 0, "Correct option and hydration/comfort distinction, but the justification omits explicit patient discussion and potential survival benefit from the advanced-cancer criteria. It states existing knowledge rather than actual PDF inspection."),
        "PDF05": (1, 0, "Core prerequisite and CNS/cardiac domains match. Adds extrapyramidal/QT examples not stated in the reference passage; these are unverified additional examples, not established here as medically false. No PDF verification is credited."),
        "PDF06": (2, 0, "Correct absent-regimen option based on title/scope inference, honestly described as highly unlikely and unverified. Imports no external dose. The source was unavailable, so this is not an established document-absence verification."),
    },
    "B-full-context": {
        "PDF01": (2, 2, "Complete B1-1 screening/timing answer and an exact registered page-2 chunk that supports every asked fact."),
        "PDF02": (2, 2, "Both thresholds and strict duration qualifiers match the original-render-verified key and the supplied commentary. Shows 50e75 with a bracketed 50-75 interpretation; the numerical inference is correct here, not proof that extraction generally resolves numeric glyphs or that the model viewed the rendering."),
        "PDF03": (2, 2, "Both triples are stated exactly; two page chunks respectively verify B4-1 and B4-2. Each citation supports its associated recommendation, not both by itself."),
        "PDF04": (2, 1, "All clinical facts and the hydration exception are correct and present in the supplied full registry, but the page-12 citation hash is misconstructed/unregistered. Page-13 citation is valid. Partial auditability grade concerns a broken source token for a real page, not an invented article or wrong clinical answer."),
        "PDF05": (2, 2, "Complete constipation prerequisite and correct CNS/cardiac mapping with exact supporting page-7 ID. One schema-invalid research output was handled by the reserved native final phase; no medical-quality rerun."),
        "PDF06": (2, 2, "Correct no-regimen answer, no external dose and no fabricated positive citation. The entire sixteen-page selectable text was supplied, consistent with the frozen zero-term check. The review does not credit OCR/image/semantic-exhaustive inspection."),
    },
    "C-pdf-tools": {
        "PDF01": (2, 2, "Full B1-1 facts and an exact cited delivered chunk. Actual search returned the gold fact passage plus four lesser results. A schema-invalid subsequent research response was recovered in reserved finalization."),
        "PDF02": (2, 2, "Full threshold/duration pair with a real delivered commentary chunk. Correct interpretation of retained 50e75 glyph matches the independently rendered key; no claim of model image inspection or general numeric parser correction. One unambiguous action wrapper was normalized locally."),
        "PDF03": (2, 2, "Correct two recommendation/evidence/consensus triples verified by the actual page-5/page-6 search results; both exact structured IDs support their respective subclaims. Numerical bibliography noise was not cited."),
        "PDF04": (2, 2, "Correct advanced/dying distinction, patient/benefit/burden criteria and short hydration exception. Two exact delivered IDs are used inline but the structured citations array is empty, so the automatic evidence_status undercounts real references. Inline support is reviewed separately; outputs were not repaired."),
        "PDF05": (2, 2, "Complete prerequisite and correct adverse-effect domains, with an exact delivered page-7 reference. The other three returned chunks do not establish the asked facts; correct retrieval is concentrated in the first result."),
        "PDF06": (1, 1, "Correct no-regimen option and no external dose. Actual exact-term lexical search covered the indexed document and returned zero views. Wording infers the entire document lacks this information from zero hits; the stated lexical limitation does not fully qualify image/semantic absence. Retain the correct objective key and separately mark the overstatement. A later malformed JSON response was closed using the reserved final phase."),
    },
}


def citation_note(sid, identifier, evidence):
    if sid not in evidence:
        return {"source_id": sid, "grade": "invalid-id", "view_sha256": None,
                "note": "This exact ID is not registered. The intended PDF page exists, but its source hash token is wrong; do not silently substitute another ID."}
    locator = document["locators"][sid]
    pair = locator["pdf_page"], locator["chunk"]
    grade, note = retrieval[identifier][pair]
    assert grade == "direct", (identifier, sid)
    return {"source_id": sid, "grade": "supported", "view_sha256": file_hash_view(evidence[sid]),
            "note": "Supports its associated asked subclaim(s) in the actual supplied view. " + note}


reviews = []
for arm in CONDITIONS:
    for identifier, (explanation, grounding, note) in notes[arm].items():
        path = RUN / arm / "answers" / (identifier + ".json")
        row = read_json(path)
        response = row["response"]
        structured = response.get("citations", [])
        inline = sorted(set(re.findall(r"D[0-9a-f]{12}_p\d+_c\d+_[0-9a-f]{12}", response["answer"] + " " + response["justification"])) - set(structured))
        returned = {v["id"]: v for event in row["document_tools"]["events"] for v in event["generation_view"]["sources"]}
        graded = []
        for sid, view in returned.items():
            locator = document["locators"][sid]
            grade, view_note = retrieval[identifier][(locator["pdf_page"], locator["chunk"])]
            graded.append({"source_id": sid, "pdf_page": locator["pdf_page"], "grade": grade, "note": view_note, "view_sha256": file_hash_view(view)})
        if arm == "C-pdf-tools":
            assert {(document["locators"][sid]["pdf_page"], document["locators"][sid]["chunk"]) for sid in returned} == set(retrieval[identifier])
        reviews.append({"condition": arm, "question_id": identifier, "answer_sha256": file_hash(path), "explanation_grade": explanation,
            "document_grounding_grade": grounding, "note": note, "citations": [citation_note(sid, identifier, row["evidence"]) for sid in structured],
            "inline_only_citations": [citation_note(sid, identifier, row["evidence"]) for sid in inline], "retrieval_views": graded})
save_trace(RUN / "manual-reviews.json", {"reviewer": "Unblinded assistant reading actual visible answers and full delivered/cited views; no paid judge or independent clinician",
    "rubric_sha256": file_hash(RUN / "qualitative-rubric.json"), "reviews": reviews, "keys_changed": False,
    "scope": "18 final answers, all 12 structured IDs (one invalid), two additional inline-only IDs and 21 unique C question-view pairs. Earlier visible attempts and all tool events remain in raw records; grading does not modify final answers."})
print(json.dumps({"reviewed_answers": len(reviews), "structured_citations": sum(len(r["citations"]) for r in reviews),
                  "additional_inline_citations": sum(len(r["inline_only_citations"]) for r in reviews), "retrieval_pairs": sum(len(r["retrieval_views"]) for r in reviews), "model_calls": 0}))
