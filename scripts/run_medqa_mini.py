"""Prepare the frozen dataset or execute the explicitly selected bounded panel."""
import argparse
import json
from pathlib import Path

from medical_copilot.medqa_mini import cohort_bound, prepare_subset, run_evaluation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation/medqa-mini-20261011"
RUN = ROOT / "data/evaluation/benchmark-runs/openweight-medqa-high-20261011"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Freeze questions and evaluator keys; no inference")
    parser.add_argument("--run", action="store_true", help="Execute the user-selected five-model study within the cumulative USD 6 guard")
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare_subset(DATA / "source-test.jsonl", DATA / "subset")
        print(json.dumps({"source_rows": manifest["source_rows"], "eligible_rows": manifest["eligible_rows"], "selected_source_row_indices": manifest["selected_source_row_indices"], "maximum_new_request_allowance_usd": cohort_bound(), "new_inference_calls": 0}, indent=2))
    elif args.run:
        print(json.dumps(run_evaluation(ROOT, DATA / "subset", RUN), indent=2))
    else:
        parser.error("Choose --prepare or --run explicitly.")
