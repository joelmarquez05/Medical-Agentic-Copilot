"""Run the authorized PubMed smoke, Gemma pilot or autonomous ten-case cohort."""

import argparse
from pathlib import Path

from medical_copilot.clinical_evaluation import public_error
from medical_copilot.clinical_literature_evaluation import retrieval_smoke, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "pilot", "autonomous"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.stage == "smoke":
            if not retrieval_smoke(root, args.run_dir.resolve())["passed"]:
                raise SystemExit(1)
        else:
            if args.smoke_dir is None:
                raise ValueError("--smoke-dir is required before hosted stages.")
            run_stage(root, args.run_dir.resolve(), args.stage, args.smoke_dir.resolve(),
                      args.pilot_dir.resolve() if args.pilot_dir else None)
    except Exception as exc:
        print({"operation": "clinical_literature_runner", "status": "failed", **public_error(exc)})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
