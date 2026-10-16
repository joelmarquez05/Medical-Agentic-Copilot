"""Answer strict medical question JSON about a previously ingested PDF."""
import argparse
import json
from pathlib import Path
from medical_copilot.clinical_evaluation import public_error
from medical_copilot.evaluation import read_json
from medical_copilot.pdf_evaluation import run_stage

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, required=True, help="Ingestion document.json; not the raw PDF")
    parser.add_argument("--questions", type=Path, required=True, help="Object containing questions; only question_id, question and optional A-D options")
    parser.add_argument("--run-dir", type=Path, required=True, help="Fresh direct subdirectory of data/evaluation/pdf-runs")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--condition", choices=["A-no-document", "B-full-context", "C-pdf-tools"], default="C-pdf-tools")
    args = parser.parse_args()
    try:
        result = run_stage(Path(__file__).resolve().parents[1], args.run_dir, "ask", document=args.document,
                           custom_questions=read_json(args.questions)["questions"], condition=args.condition, config_path=args.config)
    except Exception as exc:
        print(json.dumps({"completed": False, **public_error(exc), "details": "Private provider/error content withheld; check input and frozen prerequisites."}))
        raise SystemExit(1) from None
    print(json.dumps(result))

if __name__ == "__main__":
    main()
