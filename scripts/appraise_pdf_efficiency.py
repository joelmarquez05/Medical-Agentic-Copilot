"""Store explicit assistant notes after inspecting every new answer and actual view."""
import hashlib
import json
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/pdf-runs/gemma-article-efficiency-main-20261012"
document = read_json(RUN / "document.json")
# Fixed manual judgments based on full visible text inspection, not an automatic model judge.
grades = {
    "ART01": {(2,2):("direct","All four requested figures and approximate qualifier are present."), (3,1):("partial","Energy/substrate section caption only, without requested figures; no raster interpretation credited."), (13,1):("partial","Bibliography identifies Lundholm/Cao-related studies but supplies no requested study percentages."), (4,1):("irrelevant","Micronutrient/fat discussion has unrelated numbers, not hypermetabolism/TEE figures.")},
    "ART02": {(4,1):("direct","Exact physician count, E/C daily doses and average follow-up."), (13,2):("partial","Physicians' vitamin-study reference metadata, without the requested quantitative details."), (5,1):("irrelevant","Thiamine/refeeding/exercise are different topics, not the E/C physician trial."), (4,2):("irrelevant","Diet/feeding-route intervention text does not provide this trial's figures.")},
    "ART03": {(7,1):("direct","472 patients, eight weeks, 37% completion and no significant differences."), (8,1):("irrelevant","Arginine-containing perioperative immunonutrition is a different intervention/study."), (10,1):("irrelevant","Glutamine/mucositis studies are not the HMB-mixture cachexia RCT."), (11,1):("irrelevant","HSCT/glutamine and neutropenic diets do not supply HMB-mixture trial details."), (10,2):("irrelevant","Glutamine-controlled subset/HSCT continuation does not answer the HMB question.")},
    "ART04": {(7,2):("direct","Correct multicenter and pilot figures; distinguishes per-day versus per-administration doses."), (14,3):("partial","References 68/69 identify the exact studies, without count/dose/duration details."), (2,1):("irrelevant","Global THC glossary/intro has no requested trial or pilot details.")},
    "ART05": {(7,2):("direct","Contains patient count and first two doses; ends at dexamethasone without its dose."), (14,3):("partial","Reference 72 identifies the three-arm study, without its doses/count."), (6,1):("irrelevant","General corticosteroid/progestin/fish-oil guidance does not give the three-arm trial figures."), (14,2):("irrelevant","Other cachexia reviews/insulin/HMB citations do not report this trial's requested figures."), (7,1):("irrelevant","Adjacent HMB/insulin/prokinetic discussion is not the three-arm trial."), (8,1):("direct","Unread by C; initial 3 mg/day continuation supplies dexamethasone dose when joined to page 7.")},
    "ART06": {(10,1):("direct","Whole-review counts 15/11 present; sentence stops before controlled-subset outcomes."), (10,2):("direct","Unread by C; contains controlled-subset counts two benefit/four no effect."), (15,2):("irrelevant","Other glutamine/RT trial references do not identify/provide the requested review counts."), (15,3):("partial","Reference 112 identifies the exact Sayles review, without its quantitative synthesis."), (11,1):("irrelevant","Other HSCT/glutamine evidence and diets do not supply the requested controlled-subset counts."), (7,2):("irrelevant","Cannabinoid/androgen studies collide with the word six but are unrelated."), (8,1):("irrelevant","Surgery and previous drug-dose continuation are unrelated to the glutamine review."), (11,2):("irrelevant","Survivor activity/BMI continuation is unrelated to the review counts.")},
}
notes = {
    "ART01": "All four article figures match; approximately 50% is preserved in the explanation. No extrapolation to all cancer patients.",
    "ART02": "All four physician-trial details and daily units/average follow-up match the original article. No prescribing recommendation is made.",
    "ART03": "Correct count, duration, completion and false significance flag; no unsupported claim of efficacy/equivalence.",
    "ART04": "Correctly distinguishes the two studies and per-day versus per-administration quantities; bid is explained correctly as twice daily.",
    "ART05": "Full context joins pages 7 and 8 and correctly cites both, recovering all four details.",
    "ART06": "Full context combines the two page-10 chunks, keeping whole-review and controlled-subset counts distinct and citing both.",
}
reviews = []
for arm in ["B-full-context", "C-pdf-tools"]:
    for path in sorted((RUN / arm / "answers").glob("*.json")):
        row = read_json(path)
        identifier = row["question"]["question_id"]
        note, explanation = notes[identifier], 2
        if arm == "C-pdf-tools" and identifier == "ART05":
            explanation, note = 1, "Three requested values correct; dexamethasone dose remains null. Actual search/read obtained page 7, not its page-8 continuation. A malformed third research output led to reserved finalization. Honest missing-value response, but incomplete article access and a long episode; do not credit unread 3 mg/day."
        if arm == "C-pdf-tools" and identifier == "ART06":
            explanation, note = 1, "Whole-review counts correct; both controlled-subset counts remain null. The agent searched the dangling phrase and read page 11 instead of page-10 chunk 2. Claimed truncation concerns delivered excerpts, not absence from the article. Finalization followed three actual research operations; no invented numbers, but incomplete coverage and prolonged generation."
        citations = []
        for sid in row["response"].get("citations", []):
            loc = document["locators"][sid]
            grade, view_note = grades[identifier][(loc["pdf_page"], loc["chunk"])]
            assert grade == "direct"
            view = row["evidence"][sid]
            fingerprint = hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            citations.append({"source_id": sid, "grade": "supported", "note": "Supports the actually stated associated facts. " + view_note, "view_sha256": fingerprint})
        returned = {s["id"]: s for e in row["document_tools"]["events"] for s in e["generation_view"]["sources"]}
        retrieval = []
        for sid, view in returned.items():
            loc = document["locators"][sid]
            grade, view_note = grades[identifier][(loc["pdf_page"], loc["chunk"])]
            retrieval.append({"source_id": sid, "grade": grade, "note": view_note,
                              "view_sha256": hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
        reviews.append({"condition": arm, "question_id": identifier, "answer_sha256": file_hash(path), "explanation_grade": explanation,
            "note": note, "citations": citations, "retrieval_views": retrieval})
save_trace(RUN / "manual-reviews.json", {"reviewer": "Unblinded assistant inspecting all visible answers and full supplied/cited passages; no independent clinician or paid judge",
    "review_scope": "Twelve answers, fourteen actual structured cited views and all unique C case-view pairs; positive source support means associated stated facts, not full requested-case completion.",
    "reference_values_changed": False, "reviews": reviews})
print(json.dumps({"answers_reviewed": len(reviews), "cited_views": sum(len(r["citations"]) for r in reviews), "retrieval_pairs": sum(len(r["retrieval_views"]) for r in reviews), "new_model_calls": 0}))
