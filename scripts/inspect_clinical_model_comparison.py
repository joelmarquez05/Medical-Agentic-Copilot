"""Show actual saved model answers and delivered literature for manual appraisal."""
import argparse
import json
from pathlib import Path
from medical_copilot.clinical_model_comparison import MODELS
from medical_copilot.evaluation import read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=30)
    args = parser.parse_args()
    run = Path(__file__).resolve().parents[1] / "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"
    keys = {k["question_id"]: k for k in read_json(run / "answer-key.json")["keys"]}
    for number in range(args.first, args.last + 1):
        identifier = f"BM{number:02d}"
        path = run / args.model / "answers" / (identifier + ".json")
        if not path.exists():
            continue
        row = read_json(path)
        print(json.dumps({"model": args.model, "id": identifier, "key": keys[identifier]["answer_idx"],
                          "status": row["status"], "response": row["response"], "model_calls": len(row["model_calls"]),
                          "closing_reason": row["closing_reason"], "events": [
                              {"name": e["name"], "arguments": e["arguments"], "status": e["status"],
                               "views": e["generation_view"]["sources"]} for e in row["literature"]["events"]]},
                         ensure_ascii=False))
