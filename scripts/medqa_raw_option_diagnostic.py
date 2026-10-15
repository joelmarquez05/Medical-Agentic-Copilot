"""Separate option concordance from strict JSON-format compliance, without inference."""
from pathlib import Path

from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.medqa_option_diagnostic import visible_option

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "data/evaluation/benchmark-runs/medqa-six-model-summary-20261011"

if __name__ == "__main__":
    collection = read_json(COLLECTION / "summary.json")
    keys = {item["question_id"]: item["answer_idx"] for item in read_json(ROOT / "data/evaluation/benchmark-runs/openweight-medqa-high-20261011/answer-key.json")["keys"]}
    records = []
    for origin in collection["records"]:
        path = ROOT / origin["path"]
        assert file_hash(path) == origin["sha256"]
        record = read_json(path)
        answer = visible_option(record)
        records.append({**origin, "primary_status": record["status"], "selected_option": answer["selected_option"] if answer else None, "reference_option": keys[record["question_id"]], "correct": answer["selected_option"] == keys[record["question_id"]] if answer else None, "extraction_basis": answer["basis"] if answer else "No complete unambiguous normally stopped option", "justification": answer["justification"] if answer else None})
    write_json(COLLECTION / "raw-option-diagnostic.json", {"timing": "Post hoc diagnostic after inspecting a nonconforming Ling response; not a prespecified replacement metric", "policy": "Parse exactly one explicit complete JSON answer object in a normally stopped output; no LLM extraction or judgment, no interpretation of truncated answers. Apply to all models/conditions. Preserve strict primary scores and records.", "records": records, "new_inference_calls": 0})
    recovered = [record for record in records if record["primary_status"] != "completed" and record["selected_option"]]
    print(f"Post hoc diagnostic found {len(recovered)} explicit nonconforming option; original scores and responses unchanged.")
