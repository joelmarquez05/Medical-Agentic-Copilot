"""Evaluate the user-requested resilient agent as a new, preserved cohort."""

import hashlib
import json
import shutil
from pathlib import Path

from medical_copilot.clinical_evaluation import (
    ACCOUNT_LIMIT, UNCERTAINTY_HEADROOM, DEFAULT_PROFILE, DEFAULT_SET, BaselineAudit,
    maximum_request_cost, observe_ledger, pilot_question, preflight, public_error, utc_now,
)
from medical_copilot.clinical_graph import BudgetBlocked, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_literature_evaluation import (
    BASELINE, PUBMED_CONFIG, RETRIEVAL_RUBRIC, SNAPSHOT_NAMES, check_baseline, summarize,
)
from medical_copilot.clinical_resilient import (
    VERSION, RESILIENT_SYSTEM, PhaseBackend, ResilientClinicalAgent, ResilientJSONModel,
    ResilientLimits, final_schema,
)
from medical_copilot.comparison_evaluation import BudgetLimitExceeded, RequestBudget
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.models import ModelClient, create_backend
from medical_copilot.pubmed import PubMedClient, load_pubmed_config

POLICY = "config/clinical-resilient.json"
PROTOCOL = "docs/implementation/requirement-3-resilient-protocol.md"
PRIOR = "data/evaluation/retrieval-runs/gemma-evidence-main-20261011"


def make_model(config, root, budget, path, policy):
    backend, redactor, credential_source = create_backend(config, root)
    phased = PhaseBackend(backend, policy["native_final_json_schema"])
    audit = BaselineAudit(phased, config, budget, redactor, path)
    client = ModelClient(audit, config, redactor)

    def gate(request):
        count = len((RESILIENT_SYSTEM + json.dumps(request, ensure_ascii=False)).encode())
        if count > policy["max_input_bytes"]:
            raise BudgetBlocked("Effective input bound exceeded; no provider request sent.")
        extra = len(json.dumps(final_schema(request["question"], request["evidence_registry"])).encode()) + 256 if request["phase"] == "finalize" else 0
        try:
            budget.check(maximum_request_cost(config, count + extra))
        except BudgetLimitExceeded:
            raise BudgetBlocked("Cumulative request allowance exhausted.") from None

    def transient():
        if not audit.records or audit.records[-1]["status"] != "error":
            return False
        error = audit.records[-1]
        return error.get("http_status") in {429, 500, 502, 503, 504} or error.get("error_type") in {
            "APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectTimeout", "ReadTimeout"}

    return ResilientJSONModel(client, gate, transient), client, audit, phased, credential_source


def run_stage(root, run_dir, stage, smoke_dir, probe_dir=None, pilot_dir=None):
    if stage not in {"probe", "pilot", "autonomous"} or run_dir.exists():
        raise ValueError("Choose a valid stage and a new run directory.")
    baseline = check_baseline(root)
    policy = read_json(root / POLICY)
    limit_data = {key: value for key, value in policy.items() if key in ResilientLimits.model_fields}
    limits = ResilientLimits.model_validate(limit_data)
    smoke = read_json(smoke_dir / "smoke.json")
    if not smoke["passed"] or smoke["tool_source_sha256"] != file_hash(root / "src/medical_copilot/clinical_literature.py"):
        raise ValueError("A successful check of the unchanged PubMed tools is required.")
    if stage in {"pilot", "autonomous"}:
        if probe_dir is None:
            raise ValueError("Final-only transport probe required.")
        probe = read_json(probe_dir / "answers/PILOT-01.json")
        pm = read_json(probe_dir / "manifest.json")
        if pm["stage"] != "probe" or probe["status"] != "completed" or probe["output_origin"] != "model":
            raise ValueError("Native final transport has not completed the probe.")
        if not probe["transport_formats"] or probe["transport_formats"][0]["response_format"] != "json_schema":
            raise ValueError("Expected a tested native final schema.")
        for key, origin in [("policy_sha256", POLICY), ("protocol_sha256", PROTOCOL)]:
            if pm[key] != file_hash(root / origin):
                raise ValueError("Policy/protocol changed after probe.")
        for name, digest in pm["source_hashes"].items():
            if file_hash(root / "src/medical_copilot" / name) != digest:
                raise ValueError("Code changed after probe.")
    if stage == "autonomous":
        if pilot_dir is None or read_json(pilot_dir / "answers/PILOT-01.json")["status"] != "completed":
            raise ValueError("Successful full-policy pilot required.")
        pilot = read_json(pilot_dir / "manifest.json")
        if pilot["stage"] != "pilot" or pilot["policy_sha256"] != file_hash(root / POLICY) or pilot["effective_system_sha256"] != hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest():
            raise ValueError("Pilot differs from current policy/system.")
    config = load_config(root / DEFAULT_PROFILE)
    observation = preflight(config, root)
    if policy["native_final_json_schema"] and not any("structured_outputs" in endpoint["supported_parameters"] for endpoint in observation["eligible_endpoints"]):
        raise ValueError("Pinned endpoint does not advertise structured outputs.")
    budget = RequestBudget(ACCOUNT_LIMIT, observation["ledger"]["total_usage"])
    budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM
    questions = read_json(root / DEFAULT_SET / "questions.json")["questions"] if stage == "autonomous" else [pilot_question(root)]
    budget.check(len(questions) * limits.max_model_turns * maximum_request_cost(config, limits.max_input_bytes + 4096))
    run_dir.mkdir(parents=True)
    (run_dir / "answers").mkdir()
    save_trace(run_dir / "preflight.json", observation)
    save_trace(run_dir / "config.json", config.public_dict())
    save_trace(run_dir / "system-instructions.json", {"effective_system": RESILIENT_SYSTEM,
        "reason": "User requests completion/recovery for all ten and future cases; autonomous literature remains encouraged.",
        "comparison_limit": "New graph routing, normalizations, larger research budget, explicit closing phase and final native schema; not an isolated retrieval or prompt-effect comparison."})
    for origin, name in [(root / DEFAULT_SET / "questions.json", "frozen-ten-questions.json"),
                         (root / DEFAULT_SET / "answer-key.json", "answer-key.json"),
                         (root / DEFAULT_SET / "manifest.json", "selection-manifest.json"),
                         (root / PROTOCOL, "protocol.md"), (root / RETRIEVAL_RUBRIC, "retrieval-rubric.json"),
                         (root / "data/evaluation/clinical-baseline-rubric.json", "answer-rubric.json"),
                         (root / PUBMED_CONFIG, "pubmed-config.json"), (root / POLICY, "policy.json")]:
        shutil.copyfile(origin, run_dir / name)
    snapshot = run_dir / "source-snapshot"
    snapshot.mkdir()
    hashes = {}
    for name in SNAPSHOT_NAMES + ["clinical_resilient.py", "clinical_resilient_evaluation.py"]:
        origin = root / "src/medical_copilot" / name
        shutil.copyfile(origin, snapshot / name)
        hashes[name] = file_hash(origin)
    manifest = {"run_id": run_dir.name, "stage": stage, "protocol_version": VERSION, "started_at_utc": utc_now(),
        "question_ids": [q["question_id"] for q in questions], "questions_sha256": baseline["questions_sha256"],
        "answer_key_sha256": baseline["answer_key_sha256"], "config_sha256": baseline["config_sha256"],
        "system_sha256": baseline["system_sha256"], "effective_system_sha256": hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest(),
        "source_hashes": hashes, "protocol_sha256": file_hash(root / PROTOCOL), "policy_sha256": file_hash(root / POLICY),
        "retrieval_rubric_sha256": file_hash(root / RETRIEVAL_RUBRIC), "answer_rubric_sha256": file_hash(run_dir / "answer-rubric.json"),
        "pubmed_config_sha256": file_hash(root / PUBMED_CONFIG), "package_versions": package_versions(),
        "baseline_origin": BASELINE, "baseline_manifest_sha256": file_hash(root / BASELINE / "manifest.json"),
        "prior_origin": PRIOR, "prior_manifest_sha256": file_hash(root / PRIOR / "manifest.json"),
        "probe_origin": str(probe_dir) if probe_dir else None, "pilot_origin": str(pilot_dir) if pilot_dir else None,
        "smoke_origin": str(smoke_dir), "smoke_sha256": file_hash(smoke_dir / "smoke.json"),
        "limits": limits.model_dump(mode="json"), "search_policy": "Autonomous with evidence encouragement; reserved closing phase",
        "initial_ledger": observation["ledger"], "budget": budget.public_dict(),
        "probe_policy_note": "Probe sets research turns to zero to exercise the final transport; main/pilot use the shared full policy."}
    save_trace(run_dir / "manifest.json", manifest)
    model, client, audit, phased, credential_source = make_model(config, root, budget, run_dir / "usage-audit.json", policy)
    save_trace(run_dir / "credential-resolution.json", {"source": credential_source, "value_saved": False})
    for number, question in enumerate(questions, 1):
        audit.case_id = question["question_id"]
        start_calls, start_audit, start_formats = len(client.calls), len(audit.records), len(phased.formats)
        tools = ClinicalLiteratureTools(PubMedClient(load_pubmed_config(root / PUBMED_CONFIG), snapshot_dir=run_dir / "http-snapshots"))
        active_limits = limits.model_copy(update={"max_research_turns": 0}) if stage == "probe" else limits
        print(f"[{number}/{len(questions)}] START {stage} {question['question_id']}", flush=True)
        record = ResilientClinicalAgent(model, tools.definitions(), active_limits).run(question)
        record.update(model_calls=client.calls[start_calls:], response_audit=audit.records[start_audit:],
                      transport_formats=phased.formats[start_formats:], literature=tools.public_record(), protocol_version=VERSION)
        save_trace(run_dir / "answers" / (question["question_id"] + ".json"), record)
        write_json(run_dir / "budget.json", budget.public_dict())
        summarize(root, run_dir)
        print(f"[{number}/{len(questions)}] {question['question_id']}: {record['status']}; {len(record['model_calls'])} provider request(s); {record['final_attempts']} final attempt(s)", flush=True)
        # Every accepted case gets a terminal record, even if its generation fails.
    try:
        ledger = observe_ledger(config, root)
    except Exception as exc:
        ledger = {"checked_at_utc": utc_now(), "error": public_error(exc)}
    save_trace(run_dir / "final-ledger.json", ledger)
    manifest.update(finished_at_utc=utc_now(), budget=budget.public_dict())
    write_json(run_dir / "manifest.json", manifest)
    summary = summarize(root, run_dir)
    records = [read_json(path) for path in (run_dir / "answers").glob("*.json")]
    summary.update(terminal_message_coverage=sum(bool(row["response"]) for row in records),
        application_fallbacks=sum(row["output_origin"] == "application" for row in records),
        local_normalizations=sum(len(event.get("normalizations", [])) for row in records for event in row["trace"]),
        closing_phase_episodes=sum(row["final_attempts"] > 0 for row in records))
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary
