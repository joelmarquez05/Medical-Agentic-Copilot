"""Inspect four answers and deduplicated actual delivered source views for one case."""
import argparse
import json
from pathlib import Path
from medical_copilot.clinical_model_comparison import MODELS
from medical_copilot.evaluation import read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question_id")
    args = parser.parse_args()
    run = Path(__file__).resolve().parents[1] / "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"
    inputs = {q["question_id"]: q for q in read_json(run / "frozen-benchmark-questions.json")["questions"]}
    print(json.dumps(inputs[args.question_id], ensure_ascii=False))
    views = {}
    for slug in MODELS:
        path = run / slug / "answers" / (args.question_id + ".json")
        if not path.exists():
            continue
        row = read_json(path)
        queries = [{"name": e["name"], "arguments": e["arguments"],
                    "source_ids": [v["id"] for v in e["generation_view"]["sources"]]} for e in row["literature"]["events"]]
        print(json.dumps({"model": slug, "status": row["status"], "response": row["response"], "operations": queries}, ensure_ascii=False))
        for source in row["evidence"].values():
            text = json.dumps(source, sort_keys=True, ensure_ascii=False)
            view = views.setdefault(text, {"models": [], "view": source})
            view["models"].append(slug)
    print("ACTUAL SOURCE VIEWS (identical views printed once; ownership remains separate):")
    for view in views.values():
        print(json.dumps(view, ensure_ascii=False))
