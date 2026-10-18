"""Freeze exact reference passage eligibility for all 24 extracted article fields."""
import json
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json

root = Path(__file__).resolve().parents[1]
dataset = root / "data/evaluation/pdf-article-details-20261012"
document = read_json(root / "data/documents/ingested/espen-cancer-v2-20261012/document.json")
keys = read_json(dataset / "answer-key.json")["keys"]
rows = []
for key in keys:
    support = {}
    for field in key["expected_fields"]:
        ids = key["supporting_view_ids"]
        if key["question_id"] == "ART05":
            page = 8 if field == "dexamethasone_mg_per_day" else 7
            ids = [sid for sid in ids if document["locators"][sid]["pdf_page"] == page]
        if key["question_id"] == "ART06":
            chunk = 1 if field in {"all_trials_count", "all_trials_with_positive_mucositis_effect_count"} else 2
            ids = [sid for sid in ids if document["locators"][sid]["chunk"] == chunk]
        assert ids
        support[field] = ids
    rows.append({"question_id": key["question_id"], "field_support": support})
save_trace(dataset / "field-evidence-key.json", {"answer_key_sha256": file_hash(dataset / "answer-key.json"),
    "verified_before_generation": True, "keys": rows,
    "criterion": "Report exact value match separately from a match whose independently identified supporting passage is actually cited. Registry membership alone is not sufficient evidence support; additional manual review inspects text/qualifiers."})
print(json.dumps({"field_supports_frozen": 24, "new_model_calls": 0}))
