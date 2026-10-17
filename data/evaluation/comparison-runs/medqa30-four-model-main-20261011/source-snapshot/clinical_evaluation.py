"""Audited pilot/baseline execution for the restarted requirement-2 agent."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from medical_copilot.clinical_graph import (
    BudgetBlocked, ClinicalAgent, ClinicalQuestion, JSONActionModel, Limits, SYSTEM, save_trace,
)
from medical_copilot.comparison_evaluation import BudgetLimitExceeded, RequestBudget
from medical_copilot.config import load_config, resolve_credential
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.models import ModelClient, create_backend

PROTOCOL_VERSION = "requirement-2-restart-baseline-v1"
ACCOUNT_LIMIT = 6.0
UNCERTAINTY_HEADROOM = 0.15
DEFAULT_SET = "data/evaluation/internal-pubmed-set-20261011"
DEFAULT_PROFILE = "config/models/openrouter-gemma-4-reference.json"
PROTOCOL = "docs/implementation/requirement-2-baseline-protocol.md"
RUBRIC = "data/evaluation/clinical-baseline-rubric.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_number(value):
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def public_error(exc):
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return {"error_type": type(exc).__name__, "http_status": code if type(code) is int else None}


def request_json(url, credential=None):
    headers = {"User-Agent": "MedicalCopilotBaseline/1.0"}
    if credential:
        headers["Authorization"] = "Bearer " + credential
    with urlopen(Request(url, headers=headers), timeout=25) as response:
        return json.load(response)


def observe_ledger(config, root):
    credential, source = resolve_credential(config, root)
    data = request_json("https://openrouter.ai/api/v1/credits", credential)["data"]
    if not all(finite_number(data.get(key)) for key in ["total_usage", "total_credits"]):
        raise ValueError("Credit ledger lacks valid numeric totals.")
    return {"checked_at_utc": utc_now(), "credential_source": source,
            "total_usage": data["total_usage"], "total_credits": data["total_credits"],
            "interpretation": "Account-wide observation; delayed billing or other account activity remains possible."}


def preflight(config, root):
    public = request_json("https://openrouter.ai/api/v1/models/" + config.model + "/endpoints")["data"]
    provider = config.provider_options["extra_body"]["provider"]["order"][0]
    matching = [endpoint for endpoint in public.get("endpoints", [])
                if endpoint.get("provider_name", "").casefold() == provider.casefold()]
    ceilings = config.provider_options["extra_body"]["provider"]["max_price"]
    eligible = []
    for endpoint in matching:
        supported = endpoint.get("supported_parameters") or []
        prices = endpoint.get("pricing", {})
        if not {"reasoning", "response_format"}.issubset(supported):
            continue
        if any(float(prices[field]) * 1_000_000 > ceilings[field] for field in ["prompt", "completion"]):
            continue
        eligible.append({key: endpoint.get(key) for key in ["provider_name", "name", "tag", "pricing", "supported_parameters", "max_completion_tokens"]})
    if not eligible:
        raise ValueError("Pinned provider does not currently advertise the required parameters/prices.")
    return {"checked_at_utc": utc_now(), "model": config.model, "eligible_endpoints": eligible,
            "ledger": observe_ledger(config, root), "model_inference_requests": 0,
            "interpretation": "Advertised capability and account access, not completed generation."}


def maximum_request_cost(config, input_bytes):
    ceilings = config.provider_options["extra_body"]["provider"]["max_price"]
    return ((input_bytes + 1024) * ceilings["prompt"] + config.max_tokens * ceilings["completion"]) / 1_000_000


class BaselineAudit:
    """Reserve unknown charges, retain numeric usage, omit private reasoning text."""

    def __init__(self, inner, config, budget, redactor, path):
        self.inner, self.config, self.budget, self.redactor, self.path = inner, config, budget, redactor, path
        self.records = []
        self.case_id = None

    def invoke(self, messages):
        maximum = maximum_request_cost(self.config, sum(len(str(message.content).encode("utf-8")) for message in messages))
        self.budget.check(maximum)
        record = {"question_id": self.case_id, "started_at_utc": utc_now(),
                  "estimated_maximum_usd": maximum, "request_sent": True}
        started = time.perf_counter()
        try:
            message = self.inner.invoke(messages)
            metadata, usage = message.response_metadata or {}, message.usage_metadata or {}
            provider_usage = metadata.get("token_usage", {})
            # Store only known numeric fields; no provider thoughts/headers/errors.
            numeric = {key: provider_usage[key] for key in ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]
                       if finite_number(provider_usage.get(key))}
            reason = (provider_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if finite_number(reason):
                numeric["reasoning_tokens"] = reason
            normalized = {key: usage[key] for key in ["input_tokens", "output_tokens", "total_tokens"] if finite_number(usage.get(key))}
            cost = numeric.get("cost")
            if cost is not None:
                basis = "Provider-reported usage.cost."
            elif normalized.get("input_tokens") and "output_tokens" in normalized:
                prices = self.config.provider_options["extra_body"]["provider"]["max_price"]
                cost = (normalized["input_tokens"] * prices["prompt"] + normalized["output_tokens"] * prices["completion"]) / 1_000_000
                basis = "Reported tokens at configured price ceilings; not an observed bill."
            else:
                self.budget.unreported_charge_reserve_usd += maximum
                basis = "No reliable usage/cost; full request allowance retained as unknown-charge reserve."
            if cost is not None:
                self.budget.spent_usd += cost
            record.update(status="ok", usage=normalized, provider_usage=numeric, cost_usd=cost,
                          cost_basis=basis, finish_reason=metadata.get("finish_reason"),
                          model_name=metadata.get("model_name"), provider_route="Configured pin; actual route not independently exposed by this response.")
            return message
        except Exception as exc:
            self.budget.unreported_charge_reserve_usd += maximum
            record.update(status="error", cost_usd=None, cost_basis="Failed request retains full unknown-charge allowance.", **public_error(exc))
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.records.append(self.redactor.object(record))
            write_json(self.path, {"records": self.records, "budget": self.budget.public_dict()})


def score_record(record, key):
    completed = record["status"] in {"completed", "completed_with_citation_issues"}
    selected = (record.get("response") or {}).get("selected_option") if completed else None
    return {"question_id": key["question_id"], "key_type": key["key_type"],
            "reference_option": key["answer_idx"], "selected_option": selected,
            "valid_completed_answer": selected in {"A", "B", "C", "D"},
            "correct": selected == key["answer_idx"] if selected is not None else None,
            "status": record["status"], "explanation_appraisal": "Separate; never determines the option key."}


def pilot_question(root):
    source = read_json(root / DEFAULT_SET / "../internal-candidates-20261011/source-manifest.json")
    training = root / "data/evaluation/internal-candidates-20261011/source-train.jsonl"
    if file_hash(training) != source["sha256"]:
        raise ValueError("Pilot source hash changed.")
    row = json.loads(training.read_text(encoding="utf-8").splitlines()[1397])
    question = {"question_id": "PILOT-01", "question": row["question"], "options": row["options"]}
    if any(question["question"] == item["question"] for item in read_json(root / DEFAULT_SET / "questions.json")["questions"]):
        raise ValueError("Pilot must be outside the ten scored cases.")
    return question


def make_model(config, root, budget, audit_path, limits):
    backend, redactor, credential_source = create_backend(config, root)
    audit = BaselineAudit(backend, config, budget, redactor, audit_path)
    client = ModelClient(audit, config, redactor)

    def gate(request):
        if request["tools"] or request["evidence_registry"] or request["history"]:
            raise ValueError("Baseline cannot receive tools, evidence or earlier answers.")
        maximum = maximum_request_cost(config, len((SYSTEM + json.dumps(request, ensure_ascii=False)).encode("utf-8")))
        try:
            budget.check(maximum)
        except BudgetLimitExceeded:
            raise BudgetBlocked("Cumulative request allowance exhausted.") from None

    return ClinicalAgent(JSONActionModel(client, gate), limits=limits), client, audit, credential_source


def summarize(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    records = [read_json(path) for path in sorted((run_dir / "answers").glob("*.json"))]
    audit = read_json(run_dir / "usage-audit.json")["records"] if (run_dir / "usage-audit.json").exists() else []
    summary = {"stage": manifest["stage"], "planned": len(manifest["question_ids"]),
               "recorded": len(records), "statuses": dict(Counter(record["status"] for record in records)),
               "provider_requests": sum(bool(row["request_sent"]) for row in audit),
               "reported_or_estimated_cost_usd": sum(row.get("cost_usd") or 0 for row in audit),
               "unknown_charge_records": sum(row.get("cost_usd") is None for row in audit),
               "tool_execution_attempts": sum(event.get("execution_attempted", False) for record in records for event in record.get("trace", [])),
               "input_tokens": sum(row.get("usage", {}).get("input_tokens", 0) for row in audit),
               "output_tokens": sum(row.get("usage", {}).get("output_tokens", 0) for row in audit),
               "reasoning_tokens": sum(row.get("provider_usage", {}).get("reasoning_tokens", 0) for row in audit)}
    if manifest["stage"] == "baseline":
        keys = {key["question_id"]: key for key in read_json(run_dir / "answer-key.json")["keys"]}
        scores = [score_record(record, keys[record["question"]["question_id"]]) for record in records]
        write_json(run_dir / "scores.json", {"scores": scores})
        summary["correct"] = sum(score["correct"] is True for score in scores)
        summary["valid_completed"] = sum(score["valid_completed_answer"] for score in scores)
        summary["incorrect_completed"] = sum(score["correct"] is False for score in scores)
        summary["by_key_type"] = {kind: {"planned": sum(key["key_type"] == kind for key in keys.values()),
                                         "correct": sum(score["key_type"] == kind and score["correct"] is True for score in scores)}
                                  for kind in sorted({key["key_type"] for key in keys.values()})}
    write_json(run_dir / "summary.json", summary)
    return summary


def run_stage(root: Path, run_dir: Path, stage: str, pilot_dir: Path | None = None):
    if stage not in {"pilot", "baseline"}:
        raise ValueError("Choose pilot or baseline.")
    if run_dir.exists():
        raise ValueError("Use a new run directory; no automatic resume or replacement.")
    config = load_config(root / DEFAULT_PROFILE)
    if config.max_retries or config.rate_limit_retries or config.json_repair_attempts:
        raise ValueError("Retries and paid format repairs must remain disabled.")
    question_path, key_path = root / DEFAULT_SET / "questions.json", root / DEFAULT_SET / "answer-key.json"
    selection = read_json(root / DEFAULT_SET / "manifest.json")
    for filename, path in [("questions.json", question_path), ("answer-key.json", key_path)]:
        if file_hash(path) != selection[filename + "_sha256"]:
            raise ValueError("Frozen question/key hash changed.")
    questions = read_json(question_path)["questions"] if stage == "baseline" else [pilot_question(root)]
    for question in questions:
        ClinicalQuestion.model_validate(question)
    if stage == "baseline":
        if pilot_dir is None:
            raise ValueError("A successful separately saved pilot is required.")
        pilot = read_json(pilot_dir / "manifest.json")
        result = read_json(pilot_dir / "answers/PILOT-01.json")
        if pilot["stage"] != "pilot" or result["status"] != "completed":
            raise ValueError("Pilot did not produce a valid completed answer.")
        if pilot["config_sha256"] != file_hash(root / DEFAULT_PROFILE) or pilot["system_sha256"] != hashlib.sha256(SYSTEM.encode()).hexdigest():
            raise ValueError("Configuration/system changed since pilot; save a new separate pilot.")
    observation = preflight(config, root)
    initial = observation["ledger"]["total_usage"]
    budget = RequestBudget(ACCOUNT_LIMIT, initial)
    budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM
    limits = Limits()
    budget.check(len(questions) * maximum_request_cost(config, limits.max_input_bytes))
    run_dir.mkdir(parents=True)
    (run_dir / "answers").mkdir()
    save_trace(run_dir / "preflight.json", observation)
    save_trace(run_dir / "config.json", config.public_dict())
    shutil.copyfile(question_path, run_dir / "frozen-ten-questions.json")
    shutil.copyfile(key_path, run_dir / "answer-key.json")
    shutil.copyfile(root / DEFAULT_SET / "manifest.json", run_dir / "selection-manifest.json")
    shutil.copyfile(root / PROTOCOL, run_dir / "protocol.md")
    shutil.copyfile(root / RUBRIC, run_dir / "qualitative-rubric.json")
    snapshot = run_dir / "source-snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for name in ["clinical_graph.py", "clinical_evaluation.py", "models.py", "config.py", "comparison_evaluation.py", "evaluation.py"]:
        path = root / "src/medical_copilot" / name
        shutil.copyfile(path, snapshot / name)
        source_hashes[name] = file_hash(path)
    manifest = {"run_id": run_dir.name, "stage": stage, "protocol_version": PROTOCOL_VERSION,
                "started_at_utc": utc_now(), "question_ids": [item["question_id"] for item in questions],
                "questions_sha256": file_hash(question_path), "answer_key_sha256": file_hash(key_path),
                "config_sha256": file_hash(root / DEFAULT_PROFILE), "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "source_hashes": source_hashes, "protocol_sha256": file_hash(root / PROTOCOL),
                "rubric_sha256": file_hash(root / RUBRIC), "package_versions": package_versions(),
                "pilot_origin": str(pilot_dir) if pilot_dir else None, "limits": limits.model_dump(mode="json"),
                "design": "One fresh state per case; no tools, evidence or previous answer. JSON dispatch; reasoning enabled; no native tool-call claim. One provider request per baseline case; no retries/repairs.",
                "initial_ledger": observation["ledger"], "budget": budget.public_dict(),
                "pilot_selection": "Unchanged training row 1397, outside all ten evaluation inputs." if stage == "pilot" else None}
    save_trace(run_dir / "manifest.json", manifest)
    agent, client, audit, credential_source = make_model(config, root, budget, run_dir / "usage-audit.json", limits)
    save_trace(run_dir / "credential-resolution.json", {"source": credential_source, "value_saved": False})
    for number, question in enumerate(questions, 1):
        identifier = question["question_id"]
        audit.case_id = identifier
        call_start, audit_start = len(client.calls), len(audit.records)
        print(f"[{number}/{len(questions)}] START {stage} {identifier}", flush=True)
        record = agent.run(question)
        record["model_calls"] = client.calls[call_start:]
        record["response_audit"] = audit.records[audit_start:]
        record["protocol_version"] = PROTOCOL_VERSION
        save_trace(run_dir / "answers" / (identifier + ".json"), record)
        write_json(run_dir / "budget.json", budget.public_dict())
        summary = summarize(run_dir)
        print(f"[{number}/{len(questions)}] {identifier}: {record['status']}; {len(record['model_calls'])} request(s)", flush=True)
        if record["status"] in {"budget_blocked", "model_error"}:
            print("Stopped after budget/provider failure; outputs remain preserved.", flush=True)
            break
    try:
        final_ledger = observe_ledger(config, root)
        save_trace(run_dir / "final-ledger.json", final_ledger)
    except Exception as exc:
        save_trace(run_dir / "final-ledger.json", {"checked_at_utc": utc_now(), "error": public_error(exc)})
    manifest.update(finished_at_utc=utc_now(), budget=budget.public_dict())
    write_json(run_dir / "manifest.json", manifest)
    summary = summarize(run_dir)
    print(json.dumps(summary), flush=True)
    return summary
