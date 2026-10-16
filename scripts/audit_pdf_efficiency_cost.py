"""Audit the new efficiency cohort separately while retaining all historical reserves."""
import json
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import read_json, file_hash
from medical_copilot.pdf_evaluation import prior_totals

ROOT = Path(__file__).resolve().parents[1]
runs = []
for stage in ["probe", "pilot", "main"]:
    folder = ROOT / "data/evaluation/pdf-runs" / ("gemma-article-efficiency-" + stage + "-20261012")
    audits = [r for p in folder.glob("*/usage-audit.json") for r in read_json(p)["records"]]
    runs.append({"stage": stage, "requests": len(audits), "reported_cost_usd": sum(r.get("cost_usd") or 0 for r in audits),
        "unknown_charge_records": sum(r.get("cost_usd") is None for r in audits), "manifest_sha256": file_hash(folder / "manifest.json")})
ledger_path = ROOT / "data/evaluation/pdf-runs/gemma-article-efficiency-main-20261012/final-ledger.json"
ledger = read_json(ledger_path)
assert ledger["total_usage"] < 6
result = {"passed": True, "new_stages": runs, "new_provider_requests": sum(r["requests"] for r in runs),
    "new_reported_cost_usd": sum(r["reported_cost_usd"] for r in runs), "new_unknown_charge_records": sum(r["unknown_charge_records"] for r in runs),
    "latest_observed_account_usage_usd": ledger["total_usage"], "ledger_sha256": file_hash(ledger_path), "ledger": ledger,
    "all_pdf_stages_accounting": prior_totals(ROOT), "historical_uncertainty_headroom_usd": .15, "account_limit_usd": 6,
    "additional_inference_or_credential_access": 0, "interpretation": "New B/C stages separated from historical ABC; old unreported length charge remains reserved. No attribution of ledger residual as an exact private bill."}
save_trace(ROOT / "data/checks/requirement-6-article-efficiency-cost-20261012.json", result)
print(json.dumps({k: result[k] for k in ["passed", "new_provider_requests", "new_reported_cost_usd", "latest_observed_account_usage_usd"]}))
