"""Execute separately recorded probes, pilots or the four-model benchmark extension."""
import argparse
from pathlib import Path
from medical_copilot.clinical_model_comparison import run_stage
from medical_copilot.clinical_evaluation import public_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["probes", "pilots", "benchmark"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()
    try:
        run_stage(Path(__file__).resolve().parents[1], args.run_dir.resolve(), args.stage,
                  args.probe_dir.resolve() if args.probe_dir else None,
                  args.pilot_dir.resolve() if args.pilot_dir else None)
    except Exception as exc:
        print({"operation": "requirement_5_comparison", "status": "failed", **public_error(exc)})
        raise SystemExit(1) from None
