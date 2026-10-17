"""Audited PDF QA and a separately frozen three-condition demonstration."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from medical_copilot.clinical_evaluation import (ACCOUNT_LIMIT, UNCERTAINTY_HEADROOM, DEFAULT_PROFILE,
    BaselineAudit, maximum_request_cost, observe_ledger, preflight, public_error, score_record, utc_now)
from medical_copilot.clinical_graph import BudgetBlocked, ClinicalQuestion, save_trace
from medical_copilot.clinical_resilient import PhaseBackend, ResilientLimits, final_schema
from medical_copilot.comparison_evaluation import BudgetLimitExceeded, RequestBudget
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.models import ModelClient, create_backend
from medical_copilot.pdf_agent import CONDITIONS, PDF_SYSTEM, PDFClinicalAgent, PDFJSONModel
from medical_copilot.pdf_documents import PDFLibrary, PDFTools

VERSION = "requirement-6-pdf-abc-v1"
SET = "data/evaluation/pdf-set-v2-20261012"
DOCUMENT = "data/documents/ingested/espen-cancer-v2-20261012/document.json"
PROTOCOL = "docs/implementation/requirement-6-pdf-protocol.md"
POLICY = "config/pdf-evaluation.json"
RUBRIC = "data/evaluation/pdf-qualitative-rubric.json"
RUNS = "data/evaluation/pdf-runs"
REFERENCE = "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"


def episode_order(questions, conditions=CONDITIONS):
    return [(q["question_id"], condition) for n, q in enumerate(questions)
            for condition in conditions[n % len(conditions):] + conditions[:n % len(conditions)]]


def pilot_question():
    return {"question_id": "DOC-PILOT", "question": "How many numbered recommendations does this specific ESPEN 2021 cancer-nutrition PDF contain? Verify against document text where available and acknowledge when unverified.",
            "options": {"A": "41", "B": "42", "C": "43", "D": "44"}}


def source_hashes(root):
    prior = read_json(root / REFERENCE / "manifest.json")
    for name, digest in prior["source_hashes"].items():
        if file_hash(root / "src/medical_copilot" / name) != digest:
            raise ValueError("Earlier evaluated operational sources changed.")
    if file_hash(root / DEFAULT_PROFILE) != prior["config_sha256"]:
        raise ValueError("Earlier evaluated Gemma profile changed.")
    names = list(prior["source_hashes"]) + ["pdf_documents.py", "pdf_agent.py", "pdf_evaluation.py"]
    return {name: file_hash(root / "src/medical_copilot" / name) for name in names}


def fingerprint(root, document, config_path):
    return {"source_hashes": source_hashes(root), "config_sha256": file_hash(config_path),
            "effective_system_sha256": hashlib.sha256(PDF_SYSTEM.encode()).hexdigest(),
            "document_sha256": file_hash(document), "selection_manifest_sha256": file_hash(root / SET / "manifest.json"),
            "policy_sha256": file_hash(root / POLICY), "protocol_sha256": file_hash(root / PROTOCOL),
            "rubric_sha256": file_hash(root / RUBRIC)}


def validate_set(root):
    selection = read_json(root / SET / "manifest.json")
    for name, field in [("questions.json", "questions_sha256"), ("answer-key.json", "answer_key_sha256")]:
        if file_hash(root / SET / name) != selection[field]:
            raise ValueError("Frozen question/key file changed.")
    if file_hash(root / DOCUMENT) != selection["document_json_sha256"]:
        raise ValueError("Frozen ingested PDF changed.")
    if file_hash((root / DOCUMENT).parent / "original.pdf") != selection["pdf_sha256"]:
        raise ValueError("Original demonstration PDF changed.")
    questions = read_json(root / SET / "questions.json")["questions"]
    keys = read_json(root / SET / "answer-key.json")["keys"]
    for q in questions:
        ClinicalQuestion.model_validate(q)
    if len(questions) != 6 or len({q["question_id"] for q in questions}) != 6 or {q["question_id"] for q in questions} != {k["question_id"] for k in keys}:
        raise ValueError("Expected six unique matched frozen inputs and keys.")
    return questions, keys


def validate_prerequisite(directory, stage, expected):
    if directory is None:
        raise ValueError("A matching saved prerequisite is required.")
    manifest = read_json(directory / "manifest.json")
    if manifest.get("stage") != stage or not manifest.get("finished_at_utc") or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Prerequisite provenance/configuration differs.")
    conditions = ("B-full-context",) if stage == "probe" else CONDITIONS
    for condition in conditions:
        row = read_json(directory / condition / "answers/DOC-PILOT.json")
        if row["status"] not in {"completed", "completed_with_citation_issues"} or row["output_origin"] != "model":
            raise ValueError("Prerequisite did not produce a model-origin completed answer.")
        if stage == "probe" and (not row["transport_formats"] or row["transport_formats"][0]["response_format"] != "json_schema"):
            raise ValueError("Full-context native final transport not demonstrated.")


def prior_totals(root):
    """Keep every preserved PDF charge, including unfinished runs, despite ledger lag."""
    known, unknown, anchor, count = 0.0, 0.0, 0.0, 0
    for path in (root / RUNS).glob("*/manifest.json"):
        manifest = read_json(path)
        if manifest.get("requirement") != 6:
            continue
        anchor = max(anchor, manifest.get("account_anchor_usd", 0))
        for audit_path in path.parent.glob("*/usage-audit.json"):
            for row in read_json(audit_path)["records"]:
                count += bool(row["request_sent"])
                if row.get("cost_usd") is not None:
                    known += row["cost_usd"]
                else:
                    unknown += row["estimated_maximum_usd"]
    return {"known_cost_usd": known, "unknown_reserve_usd": unknown, "account_anchor_usd": anchor, "saved_requests": count}


def request_budget(ledger_usage, prior):
    initial = max(ledger_usage, prior["account_anchor_usd"] + prior["known_cost_usd"])
    budget = RequestBudget(ACCOUNT_LIMIT, initial)
    budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM + prior["unknown_reserve_usd"]
    return budget


def make_model(config, root, budget, path, policy):
    backend, redactor, credential_source = create_backend(config, root)
    phased = PhaseBackend(backend, policy["native_final_json_schema"])
    audit = BaselineAudit(phased, config, budget, redactor, path)
    client = ModelClient(audit, config, redactor)

    def gate(request):
        count = len((PDF_SYSTEM + json.dumps(request, ensure_ascii=False)).encode())
        if count > policy["max_input_bytes"]:
            raise BudgetBlocked("Effective PDF input bound exceeded.")
        extra = len(json.dumps(final_schema(request["question"], request["evidence_registry"])).encode()) + 256 if request["phase"] == "finalize" else 0
        try:
            budget.check(maximum_request_cost(config, count + extra))
        except BudgetLimitExceeded:
            raise BudgetBlocked("Cumulative request allowance exhausted.") from None

    def transient():
        if not audit.records or audit.records[-1]["status"] != "error":
            return False
        row = audit.records[-1]
        return row.get("http_status") in {429, 500, 502, 503, 504} or row.get("error_type") in {"APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectTimeout", "ReadTimeout"}

    return PDFJSONModel(client, gate, transient), client, audit, phased, credential_source


def summarize(run):
    manifest = read_json(run / "manifest.json")
    keys = {k["question_id"]: k for k in read_json(run / "answer-key.json")["keys"]} if manifest["stage"] == "main" else {}
    summaries = {}
    for condition in manifest["conditions"]:
        directory = run / condition
        rows = [read_json(p) for p in sorted((directory / "answers").glob("*.json"))]
        audit = read_json(directory / "usage-audit.json")["records"] if (directory / "usage-audit.json").exists() else []
        scores = [score_record(row, keys[row["question"]["question_id"]]) for row in rows] if keys else []
        value = {"planned": len(manifest["question_ids"]), "recorded": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
            "valid_completed": sum(r["output_origin"] == "model" and r["status"] in {"completed", "completed_with_citation_issues"} for r in rows),
            "terminal_message_coverage": sum(bool(r.get("response")) for r in rows), "application_fallbacks": sum(r["output_origin"] == "application" for r in rows),
            "provider_requests": len(audit), "cost_usd": sum(r.get("cost_usd") or 0 for r in audit), "unknown_charge_records": sum(r.get("cost_usd") is None for r in audit),
            "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in audit), "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in audit),
            "reasoning_tokens": sum(r.get("provider_usage", {}).get("reasoning_tokens", 0) for r in audit),
            "elapsed_provider_seconds": round(sum(r["elapsed_seconds"] for r in audit), 3),
            "successful_pdf_tool_calls": sum(len(r["document_tools"]["events"]) for r in rows),
            "max_input_bytes": max((e.get("input_bytes", 0) for r in rows for e in r["trace"]), default=0),
            "cited_views": sum(len((r.get("response") or {}).get("citations", [])) for r in rows)}
        if keys:
            value["correct"] = sum(s["correct"] is True for s in scores)
            write_json(directory / "scores.json", {"scores": scores})
        write_json(directory / "summary.json", value)
        summaries[condition] = value
    result = {"stage": manifest["stage"], "conditions": summaries, "provider_requests": sum(v["provider_requests"] for v in summaries.values()),
              "cost_usd": sum(v["cost_usd"] for v in summaries.values()), "interpretation": "Exact option score, explanation/citation support and retrieval efficiency remain separate."}
    write_json(run / "summary.json", result)
    return result


def run_stage(root, run, stage, *, probe=None, pilot=None, custom_questions=None, document=None, condition="C-pdf-tools", config_path=None):
    root, run = Path(root).resolve(), Path(run).resolve()
    if stage not in {"probe", "pilot", "main", "ask"} or run.exists() or run.parent != (root / RUNS).resolve():
        raise ValueError("Use a supported stage and fresh direct subdirectory of data/evaluation/pdf-runs.")
    questions, keys = validate_set(root)
    document = Path(document).resolve() if document else root / DOCUMENT
    config_path = Path(config_path).resolve() if config_path else root / DEFAULT_PROFILE
    if stage != "ask" and (document != root / DOCUMENT or config_path != root / DEFAULT_PROFILE):
        raise ValueError("Primary evaluation uses the frozen document/profile.")
    if stage in {"probe", "pilot"}:
        questions = [pilot_question()]
    if stage == "ask":
        questions = custom_questions or []
        if not questions or condition not in CONDITIONS:
            raise ValueError("Supply valid question inputs and one supported condition.")
    for q in questions:
        ClinicalQuestion.model_validate(q)  # Reject gold answers/evaluator fields before backend/credentials.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", q["question_id"]):
            raise ValueError("Use a simple alphanumeric question identifier for saved output filenames.")
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question identifiers.")
    library = PDFLibrary.load([document])
    current = fingerprint(root, document, config_path)
    if stage in {"pilot", "main"}:
        validate_prerequisite(probe, "probe", current)
    if stage == "main":
        validate_prerequisite(pilot, "pilot", current)
    policy = read_json(root / POLICY)
    limits = ResilientLimits(**{k: v for k, v in policy.items() if k in ResilientLimits.model_fields})
    config = load_config(config_path)
    if config.max_retries or config.rate_limit_retries or config.json_repair_attempts:
        raise ValueError("Paid SDK/format retries are disabled for this workflow.")
    observation = preflight(config, root)
    if not any("structured_outputs" in e["supported_parameters"] for e in observation["eligible_endpoints"]):
        raise ValueError("Pinned provider must advertise native structured outputs.")
    prior = prior_totals(root)
    budget = request_budget(observation["ledger"]["total_usage"], prior)
    budget.check(maximum_request_cost(config, limits.max_input_bytes + 4096))
    conditions = ("B-full-context",) if stage == "probe" else (condition,) if stage == "ask" else CONDITIONS
    order = episode_order(questions, conditions)
    run.mkdir(parents=True)
    save_trace(run / "preflight.json", observation)
    save_trace(run / "config.json", config.public_dict())
    save_trace(run / "questions.json", {"questions": questions})
    save_trace(run / "system-instructions.json", {"effective_system": PDF_SYSTEM})
    for origin, target in [(document, "document.json"), (root / PROTOCOL, "protocol.md"), (root / POLICY, "policy.json"),
                           (root / RUBRIC, "qualitative-rubric.json"), (root / SET / "manifest.json", "selection-manifest.json")]:
        shutil.copyfile(origin, run / target)
    if stage == "main":
        shutil.copyfile(root / SET / "answer-key.json", run / "answer-key.json")
    snapshots = run / "source-snapshot"
    snapshots.mkdir()
    for name in current["source_hashes"]:
        shutil.copyfile(root / "src/medical_copilot" / name, snapshots / name)
    manifest = {"run_id": run.name, "requirement": 6, "stage": stage, "version": VERSION, "started_at_utc": utc_now(), **current,
        "document_origin": str(document.relative_to(root)) if document.is_relative_to(root) else str(document),
        "pdf_sha256": library.catalog()[0]["pdf_sha256"], "question_ids": [q["question_id"] for q in questions],
        "questions_sha256": file_hash(run / "questions.json"), "answer_key_sha256": file_hash(root / SET / "answer-key.json") if stage == "main" else None,
        "conditions": list(conditions), "episode_order": order, "limits": limits.model_dump(mode="json"), "package_versions": package_versions(),
        "parser_version": library.document(library.catalog()[0]["document_id"])["parser_version"], "account_anchor_usd": prior["account_anchor_usd"] or observation["ledger"]["total_usage"],
        "prior_saved_requests": prior, "initial_ledger": observation["ledger"], "budget": budget.public_dict(),
        "probe_origin": str(probe) if probe else None, "pilot_origin": str(pilot) if pilot else None,
        "design": "Fresh per-case LangGraph state; identical PDF extraction B/C; no PubMed, prior guidance, gold labels or prior answers in model requests. No medical-quality reruns."}
    save_trace(run / "manifest.json", manifest)
    clients = {}
    for arm in conditions:
        directory = run / arm
        (directory / "answers").mkdir(parents=True)
        clients[arm] = make_model(config, root, budget, directory / "usage-audit.json", policy)
        save_trace(directory / "credential-resolution.json", {"source": clients[arm][-1], "value_saved": False})
    by_id = {q["question_id"]: q for q in questions}
    for number, (identifier, arm) in enumerate(order, 1):
        model, client, audit, phased, _ = clients[arm]
        audit.case_id = identifier
        offsets = len(client.calls), len(audit.records), len(phased.formats)
        tools = PDFTools(library)
        active_limits = limits.model_copy(update={"max_research_turns": 0}) if stage == "probe" else limits
        print(f"[{number}/{len(order)}] START {stage} {arm} {identifier}", flush=True)
        row = PDFClinicalAgent(model, library, arm, pdf_tools=tools if arm == "C-pdf-tools" else None, limits=active_limits).run(by_id[identifier])
        row.update(model_calls=client.calls[offsets[0]:], response_audit=audit.records[offsets[1]:], transport_formats=phased.formats[offsets[2]:],
                   evaluation_version=VERSION)
        save_trace(run / arm / "answers" / (identifier + ".json"), row)
        write_json(run / "budget.json", budget.public_dict())
        summarize(run)
        print(f"[{number}/{len(order)}] {arm} {identifier}: {row['status']}; {len(row['model_calls'])} requests; {len(row['document_tools']['events'])} PDF operations", flush=True)
    try:
        ledger = observe_ledger(config, root)
    except Exception as exc:
        ledger = {"checked_at_utc": utc_now(), "error": public_error(exc)}
    save_trace(run / "final-ledger.json", ledger)
    manifest.update(finished_at_utc=utc_now(), budget=budget.public_dict())
    write_json(run / "manifest.json", manifest)
    return summarize(run)
