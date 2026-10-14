"""Reconcile Gemma benchmark charges, including preserved interrupted integration work."""
import json
from pathlib import Path
from medical_copilot.clinical_evaluation import DEFAULT_PROFILE, observe_ledger, public_error
from medical_copilot.clinical_graph import save_trace
from medical_copilot.config import load_config
from medical_copilot.evaluation import read_json

def main():
    root=Path(__file__).resolve().parents[1]
    entries=[]
    names=["gemma-full-agent-pilot-20261011", "gemma-full-agent-medqa30-20261011",
           "gemma-full-agent-pilot-v3-20261011", "gemma-full-agent-medqa30-v3-20261011"]
    for name in names:
        run=root/"data/evaluation/benchmark-runs"/name
        audit=read_json(run/"usage-audit.json")["records"]
        entries.append({"run_id":name, "saved_provider_requests":len(audit),
            "provider_reported_usd":sum(a.get("cost_usd") or 0 for a in audit),
            "unknown_cost_audit_records":sum(a.get("cost_usd") is None for a in audit),
            "interrupted":(run/"interruption.json").exists(),
            "completed_episode_records":len(list((run/"answers").glob("*.json")))})
    initial=read_json(root/"data/evaluation/benchmark-runs"/names[0]/"manifest.json")["initial_ledger"]
    latest=observe_ledger(load_config(root/DEFAULT_PROFILE),root)
    total=sum(e["provider_reported_usd"] for e in entries)
    delta=latest["total_usage"]-initial["total_usage"]
    result={"purpose":"All newly authorized requirement-4 Gemma work, including both pilots and the interrupted first integration cohort",
        "runs":entries,"initial_ledger":initial,"latest_ledger":latest,"provider_reported_known_total_usd":total,
        "account_observed_delta_usd":delta,"ledger_minus_known_reported_usd":delta-total,
        "agreement_within_1e_7_usd":abs(delta-total)<1e-7,"saved_provider_requests":sum(e["saved_provider_requests"] for e in entries),
        "possible_unrecorded_inflight_requests":1,"interrupted_inflight_maximum_allowance_usd":0.0128,
        "standing_account_limit_usd":6,"uncertainty_headroom_usd":0.15,"inference_requests_during_audit":0,
        "interpretation":"Interrupted BM03 has one saved response charge without a completed episode, plus a possible unrecorded in-flight request. Account delta is aggregate and may lag/include other activity; do not invent an individual pending bill or treat it as zero. Known charges remain separate from any unattributed ledger difference."}
    save_trace(root/"data/checks/requirement-4-cost-audit-20261011.json",result)
    print(json.dumps({"known_reported_usd":total,"account_usage_usd":latest["total_usage"],
        "unattributed_account_delta_usd":delta-total,"ledger_agreement":result["agreement_within_1e_7_usd"]}))

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(json.dumps({"error":public_error(exc)}))
        raise SystemExit(1) from None
