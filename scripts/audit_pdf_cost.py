"""Audit all preserved PDF stages, known charges and retained uncertainty offline."""
import json
import math
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.pdf_evaluation import prior_totals

ROOT = Path(__file__).resolve().parents[1]
runs, ledgers, unknown_rows = [], [], []
for path in sorted((ROOT / "data/evaluation/pdf-runs").glob("*/manifest.json")):
    manifest = read_json(path)
    if manifest.get("requirement") != 6:
        continue
    records = [r for audit in path.parent.glob("*/usage-audit.json") for r in read_json(audit)["records"]]
    if manifest.get("finished_at_utc"):
        assert len(records) == read_json(path.parent / "summary.json")["provider_requests"]
    known = sum(r.get("cost_usd") or 0 for r in records)
    reserve = sum(r["estimated_maximum_usd"] for r in records if r.get("cost_usd") is None)
    for r in records:
        if r.get("cost_usd") is None:
            unknown_rows.append({"run": path.parent.name, "question_id": r["question_id"], "error_type": r.get("error_type"),
                                 "estimated_maximum_usd": r["estimated_maximum_usd"], "private_error_content_saved": False})
    runs.append({"run_id": path.parent.name, "stage": manifest["stage"], "saved_provider_requests": len(records),
                 "known_cost_usd": known, "unknown_request_reserve_usd": reserve, "manifest_sha256": file_hash(path)})
    for ledger_path in [path.parent / "final-ledger.json", path.parent / "preflight.json"]:
        if ledger_path.exists():
            value = read_json(ledger_path)
            value = value.get("ledger", value)
            if "total_usage" in value:
                ledgers.append({**value, "artifact": ledger_path.relative_to(ROOT).as_posix(), "artifact_sha256": file_hash(ledger_path)})
latest = max(ledgers, key=lambda r: r["checked_at_utc"])
totals = prior_totals(ROOT)
assert math.isclose(sum(r["known_cost_usd"] for r in runs), totals["known_cost_usd"], abs_tol=1e-12)
assert latest["total_usage"] < 6
result = {"passed": True, "runs": runs, "saved_provider_requests": totals["saved_requests"], "known_cost_usd": totals["known_cost_usd"],
    "unknown_request_reserve_usd": totals["unknown_reserve_usd"], "unknown_charge_records": len(unknown_rows), "unknown_requests": unknown_rows,
    "latest_ledger_usage_usd": latest["total_usage"], "latest_authenticated_observation": latest,
    "account_anchor_usd": totals["account_anchor_usd"], "observed_account_increase_usd": latest["total_usage"] - totals["account_anchor_usd"],
    "account_increase_minus_saved_known_cost_usd": latest["total_usage"] - totals["account_anchor_usd"] - totals["known_cost_usd"],
    "historical_uncertainty_headroom_usd": .15, "account_limit_usd": 6, "new_model_requests": 0, "credentials_read": 0,
    "interpretation": "Covers all saved PDF probes/pilots, primary episodes and application demo. One unreported failed length request remains reserved; ledger residual is not an independently attributable bill. Historical point-4 in-flight uncertainty is still retained. Account observations may lag or include other activity."}
save_trace(ROOT / "data/checks/requirement-6-cost-audit-20261012.json", result)
print(json.dumps({k: result[k] for k in ["passed", "saved_provider_requests", "known_cost_usd", "unknown_request_reserve_usd", "latest_ledger_usage_usd"]}))
