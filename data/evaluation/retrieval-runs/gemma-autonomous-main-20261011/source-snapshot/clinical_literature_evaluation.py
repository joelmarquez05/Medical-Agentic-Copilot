"""Frozen, cost-audited autonomous tool evaluation paired with requirement 2."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from medical_copilot.clinical_evaluation import (
    ACCOUNT_LIMIT, UNCERTAINTY_HEADROOM, DEFAULT_SET, DEFAULT_PROFILE, BaselineAudit,
    maximum_request_cost, observe_ledger, pilot_question, preflight, public_error, score_record, utc_now,
)
from medical_copilot.clinical_graph import BudgetBlocked, ClinicalAgent, ClinicalQuestion, JSONActionModel, Limits, SYSTEM, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools, ClinicalSearchArgs, ReadAbstractArgs
from medical_copilot.comparison_evaluation import BudgetLimitExceeded, RequestBudget
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.models import ModelClient, create_backend
from medical_copilot.pubmed import PubMedClient, load_pubmed_config

VERSION = "requirement-3-restart-autonomous-v1"
BASELINE = "data/evaluation/internal-runs/gemma-baseline-restart-20261011"
PROTOCOL = "docs/implementation/requirement-3-autonomous-protocol.md"
RETRIEVAL_RUBRIC = "data/evaluation/clinical-retrieval-rubric.json"
PUBMED_CONFIG = "config/pubmed-clinical.json"
SNAPSHOT_NAMES = ["clinical_graph.py", "clinical_evaluation.py", "clinical_literature.py",
                  "clinical_literature_evaluation.py", "pubmed.py", "models.py", "config.py",
                  "comparison_evaluation.py", "evaluation.py"]


def check_baseline(root):
    manifest = read_json(root / BASELINE / "manifest.json")
    for name, expected in manifest["source_hashes"].items():
        if file_hash(root / "src/medical_copilot" / name) != expected:
            raise ValueError("Core baseline source changed; do not silently mix conditions.")
    if file_hash(root / DEFAULT_PROFILE) != manifest["config_sha256"] or hashlib.sha256(SYSTEM.encode()).hexdigest() != manifest["system_sha256"]:
        raise ValueError("Reference profile/core system changed since baseline.")
    for name, key in [("questions.json", "questions_sha256"), ("answer-key.json", "answer_key_sha256")]:
        if file_hash(root / DEFAULT_SET / name) != manifest[key]:
            raise ValueError("Frozen input/key changed since baseline.")
    return manifest


def make_adapter(config, root, budget, audit_path):
    backend, redactor, credential_source = create_backend(config, root)
    audit = BaselineAudit(backend, config, budget, redactor, audit_path)
    client = ModelClient(audit, config, redactor)

    def gate(request):
        maximum = maximum_request_cost(config, len((SYSTEM + json.dumps(request, ensure_ascii=False)).encode()))
        try:
            budget.check(maximum)
        except BudgetLimitExceeded:
            raise BudgetBlocked("Cumulative request allowance exhausted.") from None

    return JSONActionModel(client, gate), client, audit, credential_source


def retrieval_smoke(root, run_dir):
    if run_dir.exists():
        raise ValueError("Choose a new smoke directory.")
    run_dir.mkdir(parents=True)
    client = PubMedClient(load_pubmed_config(root / PUBMED_CONFIG), snapshot_dir=run_dir / "http-snapshots")
    tools = ClinicalLiteratureTools(client)
    record = {"purpose": "Separate live two-tool engineering check, not model-chosen retrieval or a scored episode",
              "started_at_utc": utc_now(), "model_inference_requests": 0,
              "tool_source_sha256": file_hash(root / "src/medical_copilot/clinical_literature.py"),
              "pubmed_config_sha256": file_hash(root / PUBMED_CONFIG)}
    try:
        search = tools.search(ClinicalSearchArgs(query="naloxone opioid overdose respiratory depression", max_results=3))
        if not search.sources:
            raise ValueError("Smoke search returned no usable records.")
        read = tools.read(ReadAbstractArgs(pmids=[search.sources[0].pmid]))
        if not read.sources:
            raise ValueError("Smoke read returned no usable record.")
        record.update(passed=True, search_source_count=len(search.sources), read_source_count=len(read.sources),
                      read_text_kinds=[source.text_kind for source in read.sources])
    except Exception as exc:
        record.update(passed=False, error=public_error(exc))
    record.update(literature=tools.public_record(), finished_at_utc=utc_now())
    save_trace(run_dir / "smoke.json", record)
    print(json.dumps({key: record[key] for key in ["passed", "model_inference_requests", "finished_at_utc"]}), flush=True)
    return record


def paired_transition(first, second):
    if first is None or second is None:
        return "operationally_incomplete"
    return {(True, True): "both_correct", (False, True): "wrong_to_correct",
            (True, False): "correct_to_wrong", (False, False): "both_wrong"}[(first, second)]


def summarize(root, run_dir):
    manifest = read_json(run_dir / "manifest.json")
    records = [read_json(path) for path in sorted((run_dir / "answers").glob("*.json"))]
    audit = read_json(run_dir / "usage-audit.json")["records"] if (run_dir / "usage-audit.json").exists() else []
    tool_events = [event for record in records for event in record["trace"] if event["node"] == "execute_tool"]
    summary = {"stage": manifest["stage"], "planned": len(manifest["question_ids"]), "recorded": len(records),
               "statuses": dict(Counter(record["status"] for record in records)), "provider_requests": len(audit),
               "reported_or_estimated_cost_usd": sum(row.get("cost_usd") or 0 for row in audit),
               "unknown_charge_records": sum(row.get("cost_usd") is None for row in audit),
               "tool_action_attempts": len(tool_events), "tool_execution_attempts": sum(event["execution_attempted"] for event in tool_events),
               "tool_statuses": dict(Counter(event["status"] for event in tool_events)),
               "tool_names": dict(Counter(event["name"] for event in tool_events)),
               "episodes_with_search": sum(any(event["name"] == "search_pubmed" and event["execution_attempted"] for event in record["trace"] if event["node"] == "execute_tool") for record in records),
               "episodes_without_search": sum(not any(event["name"] == "search_pubmed" and event["execution_attempted"] for event in record["trace"] if event["node"] == "execute_tool") for record in records),
               "pubmed_logical_http_requests": sum(len(record["literature"]["http_requests"]) for record in records),
               "pubmed_http_attempts": sum(len(request["attempts"]) for record in records for request in record["literature"]["http_requests"]),
               "question_paper_pairs": sum(len({paper["pmid"] for event in record["literature"]["events"] for paper in event["papers"]}) for record in records),
               "input_tokens": sum(row.get("usage", {}).get("input_tokens", 0) for row in audit),
               "output_tokens": sum(row.get("usage", {}).get("output_tokens", 0) for row in audit),
               "reasoning_tokens": sum(row.get("provider_usage", {}).get("reasoning_tokens", 0) for row in audit)}
    if manifest["stage"] == "autonomous":
        keys = {key["question_id"]: key for key in read_json(run_dir / "answer-key.json")["keys"]}
        baseline = {score["question_id"]: score for score in read_json(root / BASELINE / "scores.json")["scores"]}
        scores = [score_record(record, keys[record["question"]["question_id"]]) for record in records]
        write_json(run_dir / "scores.json", {"scores": scores})
        pairs = [{"question_id": identifier, "baseline": baseline[identifier], "tool_enabled": score,
                  "transition": paired_transition(baseline[identifier]["correct"], score["correct"])}
                 for score in scores for identifier in [score["question_id"]]]
        write_json(run_dir / "paired-scores.json", {"pairs": pairs})
        summary.update(correct=sum(score["correct"] is True for score in scores),
                       valid_completed=sum(score["valid_completed_answer"] for score in scores),
                       incorrect_completed=sum(score["correct"] is False for score in scores),
                       transitions=dict(Counter(pair["transition"] for pair in pairs)),
                       by_key_type={kind: {"planned": sum(key["key_type"] == kind for key in keys.values()),
                                           "correct": sum(score["key_type"] == kind and score["correct"] is True for score in scores)}
                                    for kind in sorted({key["key_type"] for key in keys.values()})})
    write_json(run_dir / "summary.json", summary)
    return summary


def run_stage(root, run_dir, stage, smoke_dir, pilot_dir=None):
    if stage not in {"pilot", "autonomous"} or run_dir.exists():
        raise ValueError("Choose pilot/autonomous and a new directory.")
    baseline = check_baseline(root)
    smoke = read_json(smoke_dir / "smoke.json")
    if not smoke["passed"] or smoke["tool_source_sha256"] != file_hash(root / "src/medical_copilot/clinical_literature.py") or smoke["pubmed_config_sha256"] != file_hash(root / PUBMED_CONFIG):
        raise ValueError("A successful live smoke check of these tools/configuration is required.")
    config = load_config(root / DEFAULT_PROFILE)
    if config.max_retries or config.rate_limit_retries or config.json_repair_attempts:
        raise ValueError("Model retries/format repairs must remain disabled.")
    if stage == "autonomous":
        if pilot_dir is None or read_json(pilot_dir / "answers/PILOT-01.json")["status"] != "completed":
            raise ValueError("A completed separate pilot is required.")
        pilot = read_json(pilot_dir / "manifest.json")
        if pilot["stage"] != "pilot" or pilot["config_sha256"] != baseline["config_sha256"]:
            raise ValueError("Pilot configuration differs.")
        for name, digest in pilot["source_hashes"].items():
            if file_hash(root / "src/medical_copilot" / name) != digest:
                raise ValueError("Source changed after pilot.")
        for key, path in [("protocol_sha256", PROTOCOL), ("retrieval_rubric_sha256", RETRIEVAL_RUBRIC), ("pubmed_config_sha256", PUBMED_CONFIG)]:
            if pilot[key] != file_hash(root / path):
                raise ValueError("Protocol/tool configuration changed after pilot.")
    observation = preflight(config, root)
    budget = RequestBudget(ACCOUNT_LIMIT, observation["ledger"]["total_usage"])
    budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM
    limits = Limits()
    questions = read_json(root / DEFAULT_SET / "questions.json")["questions"] if stage == "autonomous" else [pilot_question(root)]
    for question in questions:
        ClinicalQuestion.model_validate(question)
    budget.check(len(questions) * limits.max_model_turns * maximum_request_cost(config, limits.max_input_bytes))
    run_dir.mkdir(parents=True)
    (run_dir / "answers").mkdir()
    save_trace(run_dir / "preflight.json", observation)
    save_trace(run_dir / "config.json", config.public_dict())
    for origin, name in [(root / DEFAULT_SET / "questions.json", "frozen-ten-questions.json"),
                         (root / DEFAULT_SET / "answer-key.json", "answer-key.json"),
                         (root / DEFAULT_SET / "manifest.json", "selection-manifest.json"),
                         (root / PROTOCOL, "protocol.md"), (root / RETRIEVAL_RUBRIC, "retrieval-rubric.json"),
                         (root / "data/evaluation/clinical-baseline-rubric.json", "answer-rubric.json"),
                         (root / PUBMED_CONFIG, "pubmed-config.json")]:
        shutil.copyfile(origin, run_dir / name)
    snapshot = run_dir / "source-snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for name in SNAPSHOT_NAMES:
        path = root / "src/medical_copilot" / name
        shutil.copyfile(path, snapshot / name)
        source_hashes[name] = file_hash(path)
    manifest = {"run_id": run_dir.name, "stage": stage, "protocol_version": VERSION,
                "started_at_utc": utc_now(), "question_ids": [item["question_id"] for item in questions],
                "questions_sha256": baseline["questions_sha256"], "answer_key_sha256": baseline["answer_key_sha256"],
                "config_sha256": baseline["config_sha256"], "system_sha256": baseline["system_sha256"],
                "source_hashes": source_hashes, "protocol_sha256": file_hash(root / PROTOCOL),
                "retrieval_rubric_sha256": file_hash(root / RETRIEVAL_RUBRIC),
                "answer_rubric_sha256": file_hash(run_dir / "answer-rubric.json"),
                "pubmed_config_sha256": file_hash(root / PUBMED_CONFIG), "package_versions": package_versions(),
                "baseline_origin": BASELINE, "baseline_manifest_sha256": file_hash(root / BASELINE / "manifest.json"),
                "pilot_origin": str(pilot_dir) if pilot_dir else None, "smoke_origin": str(smoke_dir),
                "smoke_sha256": file_hash(smoke_dir / "smoke.json"), "limits": limits.model_dump(mode="json"),
                "search_policy": "Autonomous; no minimum or forced operation; no direct-answer rerun",
                "initial_ledger": observation["ledger"], "budget": budget.public_dict()}
    save_trace(run_dir / "manifest.json", manifest)
    adapter, client, audit, credential_source = make_adapter(config, root, budget, run_dir / "usage-audit.json")
    save_trace(run_dir / "credential-resolution.json", {"source": credential_source, "value_saved": False})
    for number, question in enumerate(questions, 1):
        identifier = question["question_id"]
        audit.case_id = identifier
        start_call, start_audit = len(client.calls), len(audit.records)
        pubmed = PubMedClient(load_pubmed_config(root / PUBMED_CONFIG), snapshot_dir=run_dir / "http-snapshots")
        tools = ClinicalLiteratureTools(pubmed)
        print(f"[{number}/{len(questions)}] START {stage} {identifier}", flush=True)
        record = ClinicalAgent(adapter, tools.definitions(), limits).run(question)
        record.update(model_calls=client.calls[start_call:], response_audit=audit.records[start_audit:],
                      literature=tools.public_record(), protocol_version=VERSION)
        save_trace(run_dir / "answers" / (identifier + ".json"), record)
        write_json(run_dir / "budget.json", budget.public_dict())
        summarize(root, run_dir)
        print(f"[{number}/{len(questions)}] {identifier}: {record['status']}; {len(record['model_calls'])} model request(s); {record['tool_actions']} tool action(s)", flush=True)
        if record["status"] in {"budget_blocked", "model_error"}:
            print("Stopped after budget/provider failure; records preserved.", flush=True)
            break
    try:
        final = observe_ledger(config, root)
    except Exception as exc:
        final = {"checked_at_utc": utc_now(), "error": public_error(exc)}
    save_trace(run_dir / "final-ledger.json", final)
    manifest.update(finished_at_utc=utc_now(), budget=budget.public_dict())
    write_json(run_dir / "manifest.json", manifest)
    result = summarize(root, run_dir)
    print(json.dumps(result), flush=True)
    return result
