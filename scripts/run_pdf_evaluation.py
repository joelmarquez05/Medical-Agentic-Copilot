"""Run a preserved hosted PDF probe, pilot or six-question ABC comparison."""
import argparse
import json
from pathlib import Path
from medical_copilot.clinical_evaluation import public_error
from medical_copilot.pdf_evaluation import run_stage

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["probe", "pilot", "main"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()
    try:
        result = run_stage(Path(__file__).resolve().parents[1], args.run_dir, args.stage,
                           probe=args.probe_dir.resolve() if args.probe_dir else None,
                           pilot=args.pilot_dir.resolve() if args.pilot_dir else None)
    except Exception as exc:
        print(json.dumps({"completed": False, **public_error(exc), "details": "Private provider/error content withheld. Review preserved artifacts or local prerequisite configuration."}))
        raise SystemExit(1) from None
    print(json.dumps(result))

if __name__ == "__main__":
    main()
