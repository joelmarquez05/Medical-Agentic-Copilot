"""Freeze the requirement-4 MedQA test subset without model or network requests."""
import json
from pathlib import Path
from medical_copilot.clinical_benchmark import prepare

if __name__ == "__main__":
    manifest = prepare(Path(__file__).resolve().parents[1])
    print(json.dumps({"questions": manifest["question_count"], "eligible": manifest["eligible_rows"],
        "seed": manifest["seed"], "indices": manifest["selected_source_row_indices"], "new_model_requests": 0}))
