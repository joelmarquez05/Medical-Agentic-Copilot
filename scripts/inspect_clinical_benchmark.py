"""Render saved case answers and exact visible evidence for manual inspection; no network."""
import argparse
import json
from pathlib import Path
from medical_copilot.evaluation import read_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"
    keys = {k["question_id"]: k for k in read_json(run / "answer-key.json")["keys"]}
    for number in range(args.first, args.last+1):
        identifier = f"BM{number:02d}"
        path = run / "answers" / (identifier+".json")
        if not path.exists():
            continue
        row = read_json(path)
        events = [{"name": e["name"], "arguments": e["arguments"], "status": e["status"],
            "views": e["generation_view"]["sources"]} for e in row["literature"]["events"]]
        print(json.dumps({"id": identifier, "key": keys[identifier]["answer_idx"], "status": row["status"],
            "response": row["response"], "model_calls": len(row["model_calls"]), "closing_reason": row["closing_reason"],
            "events": events}, ensure_ascii=False))

if __name__ == "__main__":
    main()
