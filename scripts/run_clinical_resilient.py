"""Run the native-final probe, full-policy pilot or new resilient ten-case study."""

import argparse
from pathlib import Path

from medical_copilot.clinical_evaluation import public_error
from medical_copilot.clinical_resilient_evaluation import run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["probe", "pilot", "autonomous"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()
    try:
        run_stage(Path(__file__).resolve().parents[1], args.run_dir.resolve(), args.stage,
                  args.smoke_dir.resolve(), args.probe_dir.resolve() if args.probe_dir else None,
                  args.pilot_dir.resolve() if args.pilot_dir else None)
    except Exception as exc:
        print({"operation": "resilient_runner", "status": "failed", **public_error(exc)})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
