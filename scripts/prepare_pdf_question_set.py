"""Freeze six authored document-fact questions and separately verified PDF keys."""
import json
import argparse
from pathlib import Path
from medical_copilot.clinical_graph import ClinicalQuestion, save_trace
from medical_copilot.evaluation import file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/pdf-set-v2-20261012")
parser.add_argument("--document", type=Path, default=ROOT / "data/documents/ingested/espen-cancer-v2-20261012/document.json")
args = parser.parse_args()
DESTINATION, DOCUMENT = args.output_dir.resolve(), args.document.resolve()
record = read_json(DOCUMENT)
assert not DESTINATION.exists()

rows = [
    ("PDF01", "According to recommendation 1 (B1-1) in the 2021 ESPEN practical guideline Clinical Nutrition in cancer, which nutritional screening items should be evaluated, when should screening begin, and how should repeat assessment be timed?",
     {"A": "Serum albumin, lymphocyte count and hemoglobin; begin only when chemotherapy starts and repeat once yearly.",
      "B": "Muscle mass, physical performance and inflammation only; begin after severe malnutrition develops and repeat at a fixed monthly interval.",
      "C": "Nutritional intake, weight change and BMI; begin at cancer diagnosis and repeat according to the stability of the clinical situation.",
      "D": "Calorie expenditure and micronutrient concentrations only; begin after treatment ends and repeat only on patient request."},
     "C", [2], ["B1-1"], ["Screen intake, weight change and BMI.", "Start at diagnosis.", "Repeat according to clinical stability, not a mandated fixed interval."],
     {2: ["To detect nutritional disturbances"]}),
    ("PDF02", "In the commentary accompanying recommendation 7 (B3-1) of this PDF, which pair correctly states BOTH examples of inadequate intake indicating medical nutrition, including their duration qualifiers?",
     {"A": "Less than 50% of requirements for more than one week, or only 50-75% of requirements for more than two weeks.",
      "B": "Less than 50% of requirements for more than one day, or only 50-75% for more than two days.",
      "C": "Less than 75% of requirements for more than one week, or 75-100% for more than two weeks.",
      "D": "Exactly 50% of requirements for exactly one week, or exactly 75% for exactly two weeks; no range or longer-duration condition is given."},
     "A", [4], ["B3-1 commentary"], ["First example is strictly less than 50%, for more than one week.", "Second is 50-75%, for more than two weeks."],
     {4: ["Medical nutrition is indicated"]}),
    ("PDF03", "Compare the strength, evidence level and consensus labels explicitly attached to recommendation 12 (B4-1, physical activity) and recommendation 13 (B4-2, individualized resistance exercise added to aerobic exercise). Which pairing matches this PDF?",
     {"A": "Both have strong recommendation strength, high evidence and strong consensus.",
      "B": "Recommendation 12: weak strength, low evidence, consensus. Recommendation 13: strong strength, high evidence, consensus.",
      "C": "Recommendation 12: strong strength, low evidence, strong consensus. Recommendation 13: weak strength, high evidence, consensus.",
      "D": "Recommendation 12: strong strength, high evidence, consensus. Recommendation 13: weak strength, low evidence, strong consensus."},
     "D", [5, 6], ["B4-1", "B4-2"], ["B4-1: strong recommendation/high evidence/consensus.", "B4-2: weak recommendation/low evidence/strong consensus."],
     {5: ["12) We recommend"], 6: ["13) We suggest"]}),
    ("PDF04", "Read recommendations 42 (C6-2) and 43 (C6-3) together. Which answer correctly distinguishes advanced-cancer nutritional decisions from care of dying patients AND preserves the hydration exception stated in the PDF?",
     {"A": "Start nutrition in every advanced-cancer patient regardless of prognosis or burden. In dying patients, routine parenteral nutrition and hydration reliably improve survival; confusion is not discussed.",
      "B": "In advanced cancer, consider prognosis, potential quality-of-life/survival benefit and burden together with the patient. In dying patients, prioritize comfort; parenteral nutrition/hydration are unlikely to benefit most. A short limited hydration trial may assess dehydration as a cause of acute confusion.",
      "C": "Avoid nutrition in every advanced-cancer patient regardless of prognosis. In dying patients, never consider hydration, including when acute confusion might result from dehydration.",
      "D": "Base advanced-cancer nutrition solely on calorie targets without patient discussion. In dying patients, administer prolonged high-volume hydration to all patients with any confusion, without assessing dehydration."},
     "B", [12, 13], ["C6-2", "C6-3"], ["Advanced-cancer choices require patient discussion of prognosis, benefit and care burden.", "Dying-patient goal is comfort; parenteral nutrition/hydration usually lack benefit, not an absolute never rule.", "Acute confusion is the explicit short/limited hydration exception to assess dehydration."],
     {12: ["42) We recommend"], 13: ["43) In dying patients"]}),
    ("PDF05", "According to recommendation 17 (B5-8), what should precede considering prokinetics for early satiety, and which adverse-effect domains are specifically associated with metoclopramide and domperidone?",
     {"A": "Give prokinetics before assessing constipation; metoclopramide and domperidone are described as free from major adverse effects.",
      "B": "Diagnose and treat constipation first; metoclopramide affects cardiac rhythm and domperidone affects the central nervous system.",
      "C": "Treat diarrhea first; the named concerns are renal injury with metoclopramide and bone marrow suppression with domperidone.",
      "D": "Diagnose and treat constipation first; be aware of metoclopramide adverse effects on the central nervous system and domperidone adverse effects on cardiac rhythm."},
     "D", [7], ["B5-8"], ["Diagnose/treat constipation before considering prokinetic treatment.", "Metoclopramide: central nervous system; domperidone: cardiac rhythm."],
     {7: ["17) In patients complaining"]}),
    ("PDF06", "Which exact pembrolizumab dose and administration interval is recommended by this specific 2021 ESPEN cancer-nutrition PDF? Answer about the contents of this document, not a regimen known from other sources.",
     {"A": "200 mg intravenously every three weeks.", "B": "400 mg intravenously every six weeks.",
      "C": "The PDF does not specify a pembrolizumab dose or administration interval.", "D": "2 mg/kg intravenously every week."},
     "C", [], [], ["No pembrolizumab dosing regimen is specified in the fixed document; do not import an external regimen."], {}),
]

questions, keys = [], []
for identifier, stem, options, correct, pages, recommendations, facts, anchors in rows:
    question = ClinicalQuestion(question_id=identifier, question=stem, options=options).model_dump(mode="json")
    supporting = []
    for page, phrases in anchors.items():
        for phrase in phrases:
            ids = [sid for sid in record["pages"][page - 1]["chunk_ids"] if phrase in record["views"][sid]["text"]]
            assert ids, (identifier, page, phrase)
            supporting.extend(ids)
    questions.append(question)
    keys.append({"question_id": identifier, "answer_idx": correct, "key_type": "project-authored-official-document-fact",
                 "reference_facts": facts, "pdf_pages": pages, "printed_pages": [record["pages"][p - 1]["printed_page"] for p in pages],
                 "recommendations": recommendations, "supporting_view_ids": sorted(set(supporting)),
                 "verification": "Read original rendered PDF pages and extracted passages before generation; not a released exam question or LLM-assigned clinical truth."})
full = "\n".join(p["text"] for p in record["pages"])
assert "pembrolizumab" not in full.casefold()
keys[-1]["absence_verification"] = {"exact_term": "pembrolizumab", "text_matches": 0, "scope": "All sixteen extracted text-bearing pages; no image-only or semantic-exhaustive guarantee.", "references_to_regimen_found": False}
save_trace(DESTINATION / "questions.json", {"questions": questions})
save_trace(DESTINATION / "answer-key.json", {"keys": keys})
save_trace(DESTINATION / "manifest.json", {"question_count": 6, "source": "Fixed ESPEN 2021 document; authored four-option fact questions, not MedQA",
    "questions_sha256": file_hash(DESTINATION / "questions.json"), "answer_key_sha256": file_hash(DESTINATION / "answer-key.json"),
    "document_origin": DOCUMENT.relative_to(ROOT).as_posix(), "document_json_sha256": file_hash(DOCUMENT), "pdf_sha256": record["pdf_sha256"],
    "pdf_pages_one_based": True, "keys_frozen_before_model_generation": True, "model_answers_used_to_select_questions": False,
    "limits": "Six convenience document questions with multiple-choice cues; familiar guideline previously used in historical project. No clinical or general retrieval ranking."})
lines = ["# Six fixed PDF questions", "", "Project-authored, four-option document-fact questions. Reference labels come from the official fixed PDF, not an examination board. All arms use identical wording/options. Keys remain separate during generation.", ""]
for question in questions:
    lines += ["## " + question["question_id"], "", question["question"], ""]
    lines += ["- **" + k + ":** " + v for k, v in question["options"].items()]
    lines.append("")
(DESTINATION / "questions.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"prepared_questions": 6, "model_calls": 0, "document_source_verified": True}))
