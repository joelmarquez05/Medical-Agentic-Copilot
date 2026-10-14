"""Run the separately saved Gemma benchmark pilot or primary requirement-4 study."""
import argparse
from pathlib import Path
from medical_copilot.clinical_benchmark import run_stage
from medical_copilot.clinical_evaluation import public_error

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["pilot", "benchmark"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--smoke-dir", required=True, type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()
    run_stage(Path(__file__).resolve().parents[1], args.run_dir.resolve(), args.stage,
        args.smoke_dir.resolve(), args.pilot_dir.resolve() if args.pilot_dir else None)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print({"operation": "requirement_4_benchmark", "status": "failed", **public_error(exc)})
        raise SystemExit(1) from None
