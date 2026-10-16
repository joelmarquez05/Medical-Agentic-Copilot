"""Inspect exact visible article answers and full cited/returned views for appraisal."""
import argparse
import json
from pathlib import Path
from medical_copilot.evaluation import read_json
from medical_copilot.pdf_efficiency_agent import CONDITIONS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--question", required=True)
parser.add_argument("--views", action="store_true")
args = parser.parse_args()
run = Path(__file__).resolve().parents[1] / "data/evaluation/pdf-runs/gemma-article-efficiency-main-20261012"
views = {}
for arm in CONDITIONS:
    path = run / arm / "answers" / (args.question + ".json")
    if not path.exists():
        continue
    row = read_json(path)
    print(json.dumps({"condition": arm, "response": row["response"], "status": row["status"], "operations": [
        {"name": e["name"], "arguments": e["arguments"], "ids": [s["id"] for s in e["generation_view"]["sources"]]} for e in row["document_tools"]["events"]]}, ensure_ascii=False))
    for sid in row["response"].get("citations", []):
        if sid in row["evidence"]:
            views[sid] = row["evidence"][sid]
    for event in row["document_tools"]["events"]:
        for view in event["generation_view"]["sources"]:
            views[view["id"]] = view
if args.views:
    for view in views.values():
        print(json.dumps(view, ensure_ascii=False))
