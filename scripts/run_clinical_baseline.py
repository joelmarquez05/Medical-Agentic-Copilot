"""Run the authorized Gemma integration pilot or frozen ten-case baseline."""

import argparse
from pathlib import Path

from medical_copilot.clinical_evaluation import public_error, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "baseline"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        run_stage(root, args.run_dir.resolve(), args.stage, args.pilot_dir.resolve() if args.pilot_dir else None)
    except Exception as exc:
        print({"operation": "clinical_baseline_runner", "status": "failed", **public_error(exc)})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
