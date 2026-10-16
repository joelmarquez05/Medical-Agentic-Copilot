"""Audited hosted patient-agent episodes using the existing provider adapter."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from medical_copilot.clinical_evaluation import ACCOUNT_LIMIT, DEFAULT_PROFILE, UNCERTAINTY_HEADROOM, BaselineAudit, maximum_request_cost, observe_ledger, preflight, public_error
from medical_copilot.clinical_graph import BudgetBlocked, ClinicalQuestion, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_literature_evaluation import PUBMED_CONFIG
from medical_copilot.clinical_resilient import PhaseBackend, ResilientLimits, final_schema
from medical_copilot.comparison_evaluation import BudgetLimitExceeded, RequestBudget
from medical_copilot.config import load_config
from medical_copilot.evaluation import read_json, write_json
from medical_copilot.models import ModelClient, create_backend
from medical_copilot.patient_agent import PATIENT_SYSTEM, VERSION, PatientAgent, PatientJSONModel
from medical_copilot.patient_memory import file_hash, identifier, utc_now
from medical_copilot.pdf_evaluation import prior_totals as prior_pdf_totals
from medical_copilot.pubmed import PubMedClient, load_pubmed_config

RUNS = "data/evaluation/patient-runs"
NEW_SOURCES = ("patient_memory.py", "patient_tools.py", "patient_agent.py", "patient_runtime.py")
RUNTIME_VERSION = "patient-runtime-v1.1"


def recoverable_error(row):
    """Recognize the connection-error alias actually observed in the first control."""
    return row.get("http_status") in {429, 500, 502, 503, 504} or row.get("error_type") in {
        "APITimeoutError", "APIConnectionError", "OpenAIConnectionError", "TimeoutError", "ConnectTimeout", "ReadTimeout"}


def prior_usage(root):
    result = prior_pdf_totals(root)
    for path in (root / RUNS).glob("*/manifest.json"):
        manifest = read_json(path)
        if manifest.get("requirement") != 7:
            continue
        result["account_anchor_usd"] = max(result["account_anchor_usd"], manifest.get("account_anchor_usd", 0))
        audit = path.parent / "usage-audit.json"
        for row in read_json(audit)["records"] if audit.exists() else []:
            result["saved_requests"] += bool(row.get("request_sent"))
            if row.get("cost_usd") is None:
                result["unknown_reserve_usd"] += row["estimated_maximum_usd"]
            else:
                result["known_cost_usd"] += row["cost_usd"]
    return result


def make_model(config, root, budget, path, input_bound):
    backend, redactor, credential_source = create_backend(config, root)
    phased = PhaseBackend(backend, native_final=True)
    audit = BaselineAudit(phased, config, budget, redactor, path)
    client = ModelClient(audit, config, redactor)

    def gate(request):
        count = len((PATIENT_SYSTEM + json.dumps(request, ensure_ascii=False)).encode())
        if count > input_bound:
            raise BudgetBlocked("Effective patient input bound exceeded.")
        extra = len(json.dumps(final_schema(request["question"], request["evidence_registry"])).encode()) + 256 if request["phase"] == "finalize" else 0
        try:
            budget.check(maximum_request_cost(config, count + extra))
        except BudgetLimitExceeded:
            raise BudgetBlocked("Cumulative request allowance exhausted.") from None

    def transient():
        if not audit.records or audit.records[-1]["status"] != "error":
            return False
        return recoverable_error(audit.records[-1])

    return PatientJSONModel(client, gate, transient), client, audit, phased, credential_source


class PatientRun:
    """Fresh run artifacts, per-request accounting and per-episode DB snapshots."""
    def __init__(self, root, directory, *, mode="ask", config_path=None):
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        if self.directory.parent != self.root / RUNS or self.directory.exists():
            raise ValueError("Use a fresh direct child of data/evaluation/patient-runs.")
        self.config = load_config(config_path or self.root / DEFAULT_PROFILE)
        if self.config.provider != "openai_compatible":
            raise ValueError("This audited CLI uses an OpenRouter-compatible profile.")
        observation = preflight(self.config, self.root)
        if not any("structured_outputs" in row["supported_parameters"] for row in observation["eligible_endpoints"]):
            raise ValueError("Native final output is not advertised by the pinned endpoint.")
        prior = prior_usage(self.root)
        self.budget = RequestBudget(ACCOUNT_LIMIT, max(observation["ledger"]["total_usage"], prior["account_anchor_usd"] + prior["known_cost_usd"]))
        self.budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM + prior["unknown_reserve_usd"]
        self.budget.check(maximum_request_cost(self.config, 65536 + 4096))
        self.directory.mkdir(parents=True)
        save_trace(self.directory / "preflight.json", observation)
        save_trace(self.directory / "config.json", self.config.public_dict())
        save_trace(self.directory / "pubmed-config.json", load_pubmed_config(self.root / PUBMED_CONFIG).model_dump(mode="json"))
        save_trace(self.directory / "system-instructions.json", {"effective_system": PATIENT_SYSTEM})
        compatibility = read_json(self.root / "data/checks/requirement-7-memory-research-compatibility-20261014.json")
        names = [name for name in compatibility["frozen_application_source_hashes"] if Path(name).parent.name == "medical_copilot"]
        all_names = {Path(name).name for name in names} | set(NEW_SOURCES) | {
            "clinical_literature.py", "pubmed.py", "models.py", "config.py", "clinical_evaluation.py",
            "comparison_evaluation.py", "clinical_literature_evaluation.py", "pdf_evaluation.py", "evaluation.py"}
        snapshots = self.directory / "source-snapshot"
        snapshots.mkdir()
        hashes = {}
        for name in sorted(all_names):
            source = self.root / "src/medical_copilot" / name
            shutil.copyfile(source, snapshots / name)
            hashes[name] = file_hash(source)
        self.manifest = {"requirement": 7, "version": VERSION, "runtime_version": RUNTIME_VERSION, "mode": mode, "started_at_utc": utc_now(),
                         "source_hashes": hashes, "config_sha256": file_hash(config_path or self.root / DEFAULT_PROFILE),
                         "effective_system_sha256": hashlib.sha256(PATIENT_SYSTEM.encode()).hexdigest(),
                         "account_anchor_usd": prior["account_anchor_usd"] or observation["ledger"]["total_usage"],
                         "prior_saved_usage": prior, "episodes": [], "budget": self.budget.public_dict()}
        write_json(self.directory / "manifest.json", self.manifest)
        self.model, self.client, self.audit, self.phased, source = make_model(self.config, self.root, self.budget, self.directory / "usage-audit.json", 65536)
        save_trace(self.directory / "credential-resolution.json", {"source": source, "value_saved": False})

    def ask(self, memory, patient_id, conversation_id, question, *, memory_enabled=True, require_reference=False, probe=False):
        question = ClinicalQuestion.model_validate(question).model_dump(mode="json")
        case_id = identifier(question["question_id"])
        if case_id in {item["question_id"] for item in self.manifest["episodes"]}:
            raise ValueError("Choose a unique episode ID; preserve earlier outcomes.")
        memory.conversation(patient_id, conversation_id, create=True)
        if memory_enabled:
            memory.context(patient_id, conversation_id)
        snapshot = self.directory / "memory-snapshots" / (case_id + ".sqlite3")
        memory.backup(snapshot)
        limits = ResilientLimits(max_research_turns=0, max_tool_actions=0, max_model_turns=2) if probe else ResilientLimits()
        client = PubMedClient(load_pubmed_config(self.root / PUBMED_CONFIG), self.directory / "http-snapshots")
        literature = ClinicalLiteratureTools(client)
        interaction_id, stamp = "I" + uuid.uuid4().hex, utc_now()
        episode = {"question_id": case_id, "patient_id": patient_id, "conversation_id": conversation_id,
                   "memory_snapshot": snapshot.relative_to(self.directory).as_posix(), "memory_snapshot_sha256": file_hash(snapshot),
                   "document_root": str(memory.document_root), "interaction_id": interaction_id, "created_at": stamp,
                   "memory_enabled": memory_enabled, "require_reference": require_reference, "limits": limits.model_dump(mode="json")}
        self.manifest["episodes"].append(episode)
        write_json(self.directory / "manifest.json", self.manifest)
        agent = PatientAgent(self.model, memory, patient_id, conversation_id, literature=literature, memory_enabled=memory_enabled,
                             require_reference=require_reference, limits=limits, interaction_id=interaction_id, created_at=stamp)
        start_calls, start_audit, start_formats = len(self.client.calls), len(self.audit.records), len(self.phased.formats)
        self.audit.case_id = case_id
        record = agent.run(question)
        record.update(model_calls=copy.deepcopy(self.client.calls[start_calls:]), response_audit=copy.deepcopy(self.audit.records[start_audit:]),
                      transport_formats=copy.deepcopy(self.phased.formats[start_formats:]), memory_snapshot=episode["memory_snapshot"], memory_snapshot_sha256=episode["memory_snapshot_sha256"])
        record["answer_object_normalizations"] = [copy.deepcopy(item) for item in getattr(self.model, "normalizations", []) if item["question_id"] == case_id]
        save_trace(self.directory / "answers" / (case_id + ".json"), record)
        write_json(self.directory / "budget.json", self.budget.public_dict())
        self.summarize()
        print(case_id + ": " + record["status"] + "; model requests=" + str(len(record["model_calls"])) + "; memory saved=" + str(record.get("memory_write", {}).get("saved", False)), flush=True)
        return record

    def summarize(self):
        rows = [read_json(path) for path in sorted((self.directory / "answers").glob("*.json"))]
        audits = [entry for row in rows for entry in row["response_audit"]]
        result = {"recorded": len(rows), "valid_model_answers": sum(row["status"] == "completed" and row["output_origin"] == "model" for row in rows),
                  "terminal_messages": sum(bool(row["response"]) for row in rows), "saved_interactions": sum(row.get("memory_write", {}).get("saved", False) for row in rows),
                  "provider_requests": len(audits), "reported_or_estimated_cost_usd": sum(a["cost_usd"] or 0 for a in audits),
                  "unknown_charge_records": sum(a["cost_usd"] is None for a in audits),
                  "input_tokens": sum(a.get("usage", {}).get("input_tokens", 0) for a in audits),
                  "output_tokens": sum(a.get("usage", {}).get("output_tokens", 0) for a in audits),
                  "memory_tool_calls": sum(e["status"] == "ok" for row in rows for e in row["patient_tools"]["memory_events"]),
                  "pdf_tool_calls": sum(e["status"] == "ok" for row in rows for e in row["patient_tools"]["pdf_events"]),
                  "literature_tool_calls": sum(e["status"] == "ok" for row in rows for e in row["literature"]["events"]),
                  "interpretation": "Operational counts; fixed synthetic-key checks and qualitative source support are separate."}
        write_json(self.directory / "summary.json", result)
        return result

    def finish(self):
        try:
            ledger = observe_ledger(self.config, self.root)
        except Exception as exc:
            ledger = {"checked_at_utc": utc_now(), "error": public_error(exc)}
        save_trace(self.directory / "final-ledger.json", ledger)
        self.manifest.update(finished_at_utc=utc_now(), budget=self.budget.public_dict())
        write_json(self.directory / "manifest.json", self.manifest)
        return self.summarize()
