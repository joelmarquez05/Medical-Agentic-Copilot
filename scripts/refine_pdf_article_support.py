"""Preserve original keys and make a stricter documented per-field chunk correction."""
import copy
import json
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.pdf_efficiency_evaluation import score_fields

root = Path(__file__).resolve().parents[1]
run = root / "data/evaluation/pdf-runs/gemma-article-efficiency-main-20261012"
original = read_json(run / "field-evidence-key.json")
corrected = copy.deepcopy(original)
document = read_json(run / "document.json")
entry = next(k for k in corrected["keys"] if k["question_id"] == "ART06")
for field, ids in entry["field_support"].items():
    chunk = 1 if field in {"all_trials_count", "all_trials_with_positive_mucositis_effect_count"} else 2
    entry["field_support"][field] = [sid for sid in ids if document["locators"][sid]["chunk"] == chunk]
    assert entry["field_support"][field]
corrected["verified_before_generation"] = False
corrected["correction"] = {"original_sha256": file_hash(run / "field-evidence-key.json"),
    "reason": "Source review found ART06's first two numbers in page-10 chunk 1 and controlled-subgroup counts in chunk 2. Initial field eligibility incorrectly allowed either chunk for every field.",
    "scope": "Stricter supporting-passage metadata only. Original numeric/boolean values, questions, original eligibility, prompts, generations and raw scores are preserved unchanged; no passage is supplied retroactively.",
    "timing": "Discovered during full-view review of the ongoing cohort; revised eligibility applied offline after collection, not to answering models or selection."}
save_trace(run / "field-evidence-key-reviewed.json", corrected)
keys = {k["question_id"]: k for k in read_json(run / "answer-key.json")["keys"]}
supports = {k["question_id"]: k for k in corrected["keys"]}
for arm in ["B-full-context", "C-pdf-tools"]:
    rows = [read_json(p) for p in sorted((run / arm / "answers").glob("*.json"))]
    scores = [score_fields(r, keys[r["question"]["question_id"]], supports[r["question"]["question_id"]]) for r in rows]
    save_trace(run / arm / "scores-reviewed.json", {"scores": scores, "source_eligibility_sha256": file_hash(run / "field-evidence-key-reviewed.json"), "values_unchanged": True})
print(json.dumps({"reference_values_changed": False, "questions_or_generations_changed": False, "refined_field_chunk_support": "ART06", "new_model_calls": 0}))
