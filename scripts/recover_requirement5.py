"""Run only failed Google cases after the primary comparison, then collect origins."""
import json
from pathlib import Path
from medical_copilot.comparison_recovery import recover_failed_cases, consolidate_comparison
from medical_copilot.evaluation import read_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evaluation/comparison-runs"

if __name__ == "__main__":
    original = BASE / "selected-five-main-20261010"
    recovery = BASE / "selected-five-recovery-20261010"
    preflight = read_json(ROOT / "data/checks/alternate-google-platform-preflight-20261010.json")["budget"]
    result = recover_failed_cases(ROOT, original, recovery, alternate_platform=True, extra_paid_spend_usd=preflight["reported_or_estimated_spend_usd"], extra_unknown_reserve_usd=preflight["unreported_charge_reserve_usd"])
    print(json.dumps(result, indent=2), flush=True)
    collected = consolidate_comparison(ROOT, original, recovery, BASE / "selected-five-comparison-20261010")
    print(json.dumps(collected, indent=2), flush=True)
