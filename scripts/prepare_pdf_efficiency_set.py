"""Freeze six concrete article-extraction tasks and separately verified numeric keys."""
import json
from pathlib import Path
from medical_copilot.clinical_graph import ClinicalQuestion, save_trace
from medical_copilot.evaluation import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/evaluation/pdf-article-details-20261012"
DOC = ROOT / "data/documents/ingested/espen-cancer-v2-20261012/document.json"
if DEST.exists():
    raise ValueError("Use a fresh question-set directory.")
record = read_json(DOC)
rows = [
    ("ART01", "In the energy-requirements section of this ESPEN article, what percentage of weight-losing patients were hypermetabolic in the Lundholm-group study, what percentage of newly diagnosed patients were hypermetabolic in the other study, and what lower/upper kcal/kg/day range does recommendation 3 give for total energy expenditure? Preserve that the first percentage is approximate in your justification.",
     {"weight_losing_hypermetabolic_percent": 50, "newly_diagnosed_hypermetabolic_percent": 47, "tee_lower_kcal_kg_day": 25, "tee_upper_kcal_kg_day": 30},
     {2: ["approximately 50%", "47% were hypermetabolic"]}),
    ("ART02", "In the micronutrient commentary, the article describes an RCT in US physicians combining vitamins E and C. Extract the number of physicians, vitamin E daily dose in IU, vitamin C daily dose in mg, and average follow-up in years. These are the study details reported by this article.",
     {"physicians_count": 14641, "vitamin_e_iu_per_day": 400, "vitamin_c_mg_per_day": 500, "average_followup_years": 10},
     {4: ["14,641 US physicians"]}),
    ("ART03", "The amino-acid/HMB commentary describes a larger RCT comparing an oral HMB, glutamine and arginine mixture with an isonitrogenous control. Extract its patient count, protocol duration in weeks, percentage completing the protocol, and whether a statistically significant between-group difference was observed.",
     {"patients_count": 472, "protocol_weeks": 8, "completion_percent": 37, "significant_between_group_difference": False},
     {7: ["472", "37%"]}),
    ("ART04", "In the cannabinoid commentary, distinguish the multicenter placebo-controlled cannabis-extract/THC trial from the small chemosensory pilot. Extract the multicenter trial's patient count and fixed daily dose in mg; for the small pilot, extract the THC dose per administration in mg and treatment duration in days. Do not mix up the two studies.",
     {"multicenter_patients_count": 164, "multicenter_daily_dose_mg": 5, "pilot_dose_per_administration_mg": 2.5, "pilot_duration_days": 18},
     {7: ["164 patients", "2.5 mg bid"]}),
    ("ART05", "For the three-arm RCT comparing fluoxymesterone, megestrol acetate and dexamethasone in cachectic cancer patients, extract the patient count and each of those three drugs' daily dose in mg as reported in this article. Read the continuation of the commentary if needed.",
     {"patients_count": 475, "fluoxymesterone_mg_per_day": 20, "megestrol_acetate_mg_per_day": 800, "dexamethasone_mg_per_day": 3},
     {7: ["475 cachectic cancer patients"], 8: ["(3 mg/day)"]}),
    ("ART06", "In the commentary on glutamine during conventional cytotoxic or targeted therapy, extract the total number of trials in the cited systematic review, how many of all trials reported positive mucositis effects, and the counts reporting benefit versus no effect among the prospective placebo-controlled subset. Keep the whole review and that subset distinct.",
     {"all_trials_count": 15, "all_trials_with_positive_mucositis_effect_count": 11, "placebo_controlled_trials_with_benefit_count": 2, "placebo_controlled_trials_with_no_effect_count": 4},
     {10: ["15 prospective", "four trials no effect"]}),
]
questions, keys = [], []
for identifier, stem, values, anchors in rows:
    types = {name: "boolean" if type(value) is bool else "number" for name, value in values.items()}
    instructions = " Return the final action's answer as a JSON object encoded inside the answer string, with ONLY these fields and their stated types: " + json.dumps(types) + ". Use null for an unavailable value, not an invented number. Put the document explanation and exact supporting source IDs in justification and citations; selected_option must be null."
    question = ClinicalQuestion(question_id=identifier, question=stem + instructions, options=None).model_dump(mode="json")
    passages = []
    for page, phrases in anchors.items():
        for phrase in phrases:
            hits = [sid for sid in record["pages"][page - 1]["chunk_ids"] if phrase in record["views"][sid]["text"]]
            assert hits, (identifier, page, phrase)
            passages.extend(hits)
    questions.append(question)
    keys.append({"question_id": identifier, "expected_fields": values, "field_types": types, "pdf_pages": list(anchors),
                 "supporting_view_ids": sorted(set(passages)), "source_type": "Facts reported in official fixed ESPEN article; not reanalysis of original trials",
                 "verification": "Original rendered pages and supplied text verified before this new cohort; no model output assigned reference values."})
save_trace(DEST / "questions.json", {"questions": questions})
save_trace(DEST / "answer-key.json", {"keys": keys})
save_trace(DEST / "manifest.json", {"question_count": 6, "field_count": 24, "conditions": ["B-full-context", "C-pdf-tools"],
    "questions_sha256": file_hash(DEST / "questions.json"), "answer_key_sha256": file_hash(DEST / "answer-key.json"),
    "document_origin": DOC.relative_to(ROOT).as_posix(), "document_json_sha256": file_hash(DOC), "pdf_sha256": record["pdf_sha256"],
    "values_frozen_before_new_model_generation": True, "selection": "Six authored positive, open-response concrete article-detail tasks; no multiple-choice options or absent-information case",
    "historical_context": "Earlier ABC study and familiar article retained as historical. This new selection follows the user's changed objective and is not an independent unseen-data evaluation."})
lines = ["# Concrete article-detail questions", "", "The active study compares full document versus local PDF retrieval. All six tasks have positive reference passages. Four fields per task give 24 exact numeric/boolean checks; keys never enter model context. Question instructions specify field names/types, without correct values. These study details are document-extraction facts, not prescribing recommendations.", ""]
for question, key in zip(questions, keys):
    lines += ["## " + question["question_id"], "", question["question"], "", "Reference PDF pages: " + str(key["pdf_pages"]) + ".", ""]
(DEST / "questions.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"questions": len(questions), "exact_reference_fields": 24, "new_model_calls": 0}))
