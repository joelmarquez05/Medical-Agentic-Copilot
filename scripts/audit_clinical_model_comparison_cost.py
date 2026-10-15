"""Reconcile all new comparison stages against one fresh account observation."""
from pathlib import Path
from medical_copilot.clinical_evaluation import observe_ledger
from medical_copilot.clinical_graph import save_trace
from medical_copilot.clinical_model_comparison import MODELS, PROFILES
from medical_copilot.config import load_config
from medical_copilot.evaluation import read_json

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    runs, known, unknown, requests = [], 0.0, 0.0, 0
    for name in ["medqa30-native-probes-20261011", "medqa30-agent-pilots-20261011", "medqa30-four-model-main-20261011"]:
        directory = ROOT / "data/evaluation/comparison-runs" / name
        models = []
        for slug in MODELS:
            audit = read_json(directory / slug / "usage-audit.json")["records"]
            cost = sum(a.get("cost_usd") or 0 for a in audit)
            reserve = sum(a["estimated_maximum_usd"] for a in audit if a.get("cost_usd") is None)
            sent = sum(bool(a["request_sent"]) for a in audit)
            models.append({"model_profile": slug, "saved_requests": sent, "reported_or_estimated_usd": cost,
                           "unknown_request_reserve_usd": reserve,
                           "provider_reported_cost_records": sum(a.get("cost_basis") == "Provider-reported usage.cost." for a in audit)})
            known += cost
            unknown += reserve
            requests += sent
        runs.append({"run_id": name, "models": models})
    initial = read_json(ROOT / "data/evaluation/comparison-runs/medqa30-native-probes-20261011/preflight.json")[MODELS[0]]["ledger"]
    latest = observe_ledger(load_config(ROOT / PROFILES / (MODELS[0] + ".json")), ROOT)
    delta = latest["total_usage"] - initial["total_usage"]
    result = {"runs": runs, "saved_provider_requests": requests, "known_reported_or_estimated_total_usd": known,
              "unknown_new_request_reserve_usd": unknown, "initial_ledger": initial, "latest_ledger": latest,
              "account_observed_delta_usd": delta, "ledger_minus_known_usd": delta - known,
              "agreement_within_1e_7_usd": abs(delta - known) <= 1e-7, "standing_account_limit_usd": 6,
              "uncertainty_headroom_usd": 0.15, "new_inference_requests_during_audit": 0,
              "gemma_reference_regenerated": False,
              "interpretation": "Account-wide totals may lag/include other activity. Earlier point-4 in-flight uncertainty remains preserved; no individual missing bill is invented."}
    save_trace(ROOT / "data/checks/requirement-5-cost-audit-20261011.json", result)
    print({k: result[k] for k in ["saved_provider_requests", "known_reported_or_estimated_total_usd", "unknown_new_request_reserve_usd", "agreement_within_1e_7_usd"]})
    print({"saved_account_usage_usd": latest["total_usage"], "new_inference_requests": 0})
