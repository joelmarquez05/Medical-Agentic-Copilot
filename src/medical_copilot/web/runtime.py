"""Audited live inference for the app, preserving historical evaluation files."""
import copy
import json
from pathlib import Path
import re
import shutil

from medical_copilot.clinical_evaluation import ACCOUNT_LIMIT, UNCERTAINTY_HEADROOM, BaselineAudit, maximum_request_cost, observe_ledger, preflight
from medical_copilot.clinical_graph import BudgetBlocked
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_resilient import PhaseBackend, final_schema
from medical_copilot.comparison_evaluation import BudgetLimitExceeded, RequestBudget
from medical_copilot.config import load_config
from medical_copilot.evaluation import read_json, write_json
from medical_copilot.models import ModelClient, create_backend
from medical_copilot.patient_memory import file_hash, utc_now
from medical_copilot.patient_runtime import prior_usage, recoverable_error
from medical_copilot.pubmed import PubMedClient, load_pubmed_config
from medical_copilot.web.agent import WEB_SYSTEM, WebJSONModel, WorkspaceAgent, WorkspaceLimits
from medical_copilot.web.models import DEFAULT_MODEL, profile_path, public_model


def prior_app_usage(root, directory):
    result = copy.deepcopy(prior_usage(root))
    for path in (directory / "runs").glob("*/usage-audit.json"):
        for row in read_json(path)["records"]:
            result["saved_requests"] += bool(row.get("request_sent"))
            if row.get("cost_usd") is None:
                result["unknown_reserve_usd"] += row["estimated_maximum_usd"]
            else:
                result["known_cost_usd"] += row["cost_usd"]
    # Written before transport entry; survives process loss before response accounting.
    for path in (directory / "runs").glob("*/in-flight.json"):
        item = read_json(path)
        result["unknown_reserve_usd"] += item.get("reserved_usd", 0)
    return result


class DurableAudit(BaselineAudit):
    def invoke(self, messages):
        amount = maximum_request_cost(self.config, sum(len(str(m.content).encode()) for m in messages))
        self.budget.check(amount)
        pending = self.path.parent / "in-flight.json"
        write_json(pending, {"reserved_usd": amount, "recorded_at": utc_now()})
        try:
            return super().invoke(messages)
        finally:
            # Super persists completed/error accounting before removing the crash reserve.
            if self.records and self.path.exists():
                write_json(pending, {"reserved_usd": 0})


def public_result(record, session_id):
    cited = set(record["response"].get("citations", []))
    sources = []
    for source in sorted(record["evidence"].values(), key=lambda view: view["id"] not in cited):
        item = {**source, "cited": source["id"] in cited}
        match = re.fullmatch(r"(D[0-9a-f]{12})_p(\d+)_c\d+_[0-9a-f]{12}", source["id"])
        if match:
            item.update(document_id=match[1], pdf_page=int(match[2]),
                        file_url=f"/api/sessions/{session_id}/documents/{match[1]}/file#page={match[2]}")
        sources.append(item)
    audits = record.get("response_audit", [])
    return {"status": record["status"], "response": record["response"], "sources": sources,
            "model": record.get("workspace_model"),
            "output_origin": record["output_origin"], "memory_saved": record.get("memory_write", {}).get("saved", False),
            "usage": {"model_requests": len(audits), "known_cost_usd": round(sum(r.get("cost_usd") or 0 for r in audits), 8),
                      "unknown_charge_records": sum(r.get("cost_usd") is None for r in audits),
                      "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in audits),
                      "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in audits)}}


def run_live(store, job, emit, root):
    root = Path(root)
    session = store.session(job["session_id"])
    # Validate before credential/provider access.
    with store.memory() as memory:
        memory.context(session["owner_id"], session["id"])
    selected_model = public_model(job.get("model_id", DEFAULT_MODEL))
    selected_profile = profile_path(selected_model["id"])
    config = load_config(root / selected_profile)
    emit("Checking the " + selected_model["name"] + " endpoint", kind="connection", model_id=selected_model["id"])
    observation = preflight(config, root)
    if not any("structured_outputs" in row["supported_parameters"] for row in observation["eligible_endpoints"]):
        raise ValueError("The configured endpoint does not advertise native final output.")
    previous = prior_app_usage(root, store.directory)
    budget = RequestBudget(ACCOUNT_LIMIT, max(observation["ledger"]["total_usage"], previous["account_anchor_usd"] + previous["known_cost_usd"]))
    budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM + previous["unknown_reserve_usd"]
    budget.check(maximum_request_cost(config, 65536 + 4096))
    directory = store.directory / "runs" / job["id"]
    directory.mkdir(parents=True, exist_ok=False)
    snapshots = directory / "source-snapshot"
    snapshots.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, snapshots / source.name)
    backend, redactor, _ = create_backend(config, root)
    phase = PhaseBackend(backend, native_final=True)
    audit = DurableAudit(phase, config, budget, redactor, directory / "usage-audit.json")
    audit.case_id = job["id"]
    client = ModelClient(audit, config, redactor)

    def gate(request):
        count = len((WEB_SYSTEM + json.dumps(request, ensure_ascii=False)).encode())
        if count > 65536:
            raise BudgetBlocked("Request exceeds the bounded workspace context.")
        extra = len(json.dumps(final_schema(request["question"], request["evidence_registry"])).encode()) + 256 if request["phase"] == "finalize" else 0
        try:
            budget.check(maximum_request_cost(config, count + extra))
        except BudgetLimitExceeded:
            raise BudgetBlocked("Configured cumulative request allowance exhausted.") from None

    limits = WorkspaceLimits()
    model = WebJSONModel(client, gate, lambda: bool(audit.records and audit.records[-1]["status"] == "error" and recoverable_error(audit.records[-1])), emit)
    write_json(directory / "manifest.json", {"mode": "web-workspace-v3", "started_at": utc_now(), "session_id": session["id"],
        "workspace_model": selected_model, "profile_path": selected_profile,
        "patient_selected": bool(session["patient_id"]), "profile_sha256": file_hash(root / selected_profile),
        "limits": limits.model_dump(mode="json"), "graph_recursion_limit": limits.graph_recursion_limit,
        "source_hashes": {p.name: file_hash(p) for p in snapshots.glob("*.py")},
        "effective_system": WEB_SYSTEM, "preflight": observation, "prior_accounting": previous})
    literature = ClinicalLiteratureTools(PubMedClient(load_pubmed_config(root / "config/pubmed-clinical.json"), directory / "http-snapshots"))
    with store.memory() as memory:
        memory.backup(directory / "memory-before.sqlite3")
        agent = WorkspaceAgent(model, memory, session["owner_id"], session["id"], literature=literature,
                               patient_selected=bool(session["patient_id"]), emit=emit, limits=limits)
        record = agent.run({"question_id": job["id"], "question": job["question"]})
    record.update(workspace_model=selected_model, model_calls=client.calls, response_audit=audit.records, transport_formats=phase.formats,
                  answer_object_normalizations=model.normalizations)
    write_json(directory / "answer.json", record)
    write_json(directory / "budget.json", budget.public_dict())
    try:
        write_json(directory / "final-ledger.json", observe_ledger(config, root))
    except Exception:
        write_json(directory / "final-ledger.json", {"unavailable": True})
    return public_result(record, session["id"])
