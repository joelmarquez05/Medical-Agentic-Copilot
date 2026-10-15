"""Save a fresh account observation and reconcile all v3 inference charges."""

import json
from pathlib import Path

from medical_copilot.clinical_evaluation import DEFAULT_PROFILE, observe_ledger, public_error
from medical_copilot.clinical_graph import save_trace
from medical_copilot.config import load_config
from medical_copilot.evaluation import read_json


def main():
    root = Path(__file__).resolve().parents[1]
    runs = []
    for name in ["gemma-resilient-probe-20261011", "gemma-resilient-pilot-20261011", "gemma-resilient-main-20261011"]:
        run = root / "data/evaluation/retrieval-runs" / name
        summary = read_json(run / "summary.json")
        runs.append({"run_id": name, "stage": summary["stage"], "requests": summary["provider_requests"],
            "provider_reported_usd": summary["reported_or_estimated_cost_usd"],
            "unknown_charge_records": summary["unknown_charge_records"], "immediate_final_ledger": read_json(run / "final-ledger.json")})
    initial = read_json(root / "data/evaluation/retrieval-runs/gemma-resilient-probe-20261011/manifest.json")["initial_ledger"]
    latest = observe_ledger(load_config(root / DEFAULT_PROFILE), root)
    total = sum(r["provider_reported_usd"] for r in runs)
    delta = latest["total_usage"] - initial["total_usage"]
    result = {"purpose": "User-authorized robustness v3 probe, pilot and complete new ten-case cohort",
        "runs": runs, "initial_ledger": initial, "latest_ledger": latest,
        "provider_reported_total_usd": total, "account_observed_delta_usd": delta,
        "provider_minus_ledger_delta_usd": total - delta, "agreement_within_1e_7_usd": abs(total - delta) < 1e-7,
        "requests_total": sum(r["requests"] for r in runs), "unknown_charge_records": sum(r["unknown_charge_records"] for r in runs),
        "standing_account_limit_usd": 6, "unknown_charge_headroom_usd": 0.15, "inference_requests_during_this_audit": 0,
        "interpretation": "Account totals may lag or include other activity; reported request charges and account observation remain separate."}
    save_trace(root / "data/checks/clinical-resilient-cost-audit-20261011.json", result)
    print(json.dumps({"requests": result["requests_total"], "reported_usd": total, "account_usage_usd": latest["total_usage"],
        "ledger_agreement": result["agreement_within_1e_7_usd"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": public_error(exc)}))
        raise SystemExit(1) from None
