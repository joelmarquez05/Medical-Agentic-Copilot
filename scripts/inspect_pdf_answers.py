"""Show visible answers and selected immutable views for local manual appraisal."""
import argparse
import json
from pathlib import Path
from medical_copilot.evaluation import read_json
from medical_copilot.pdf_agent import CONDITIONS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--question", required=True)
parser.add_argument("--views", action="store_true", help="Print full unique cited/returned text, never private reasoning")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run = root / "data/evaluation/pdf-runs/gemma-pdf-main-20261012"
views = {}
for arm in CONDITIONS:
    path = run / arm / "answers" / (args.question + ".json")
    if not path.exists():
        continue
    row = read_json(path)
    print(json.dumps({"condition": arm, "status": row["status"], "response": row["response"],
                      "operations": [{"name": e["name"], "arguments": e["arguments"], "returned_view_ids": [v["id"] for v in e["generation_view"]["sources"]]}
                                     for e in row["document_tools"]["events"]]}, ensure_ascii=False))
    for sid in row["response"].get("citations", []):
        if sid in row["evidence"]:
            views[sid] = row["evidence"][sid]
    for event in row["document_tools"]["events"]:
        for view in event["generation_view"]["sources"]:
            views[view["id"]] = view
if args.views:
    for view in views.values():
        print(json.dumps(view, ensure_ascii=False))
