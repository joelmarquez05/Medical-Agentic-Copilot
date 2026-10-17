"""Two-condition article-detail collection with audited efficiency and exact-field checks."""
from __future__ import annotations
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import Counter
from pathlib import Path
from medical_copilot.clinical_evaluation import (DEFAULT_PROFILE, BaselineAudit, maximum_request_cost, observe_ledger, preflight, public_error, utc_now)
from medical_copilot.clinical_graph import BudgetBlocked, ClinicalQuestion, save_trace
from medical_copilot.clinical_resilient import PhaseBackend, ResilientLimits, final_schema
from medical_copilot.comparison_evaluation import BudgetLimitExceeded
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.models import ModelClient, create_backend, parse_object
from medical_copilot.pdf_documents import PDFLibrary, PDFTools
from medical_copilot.pdf_evaluation import DOCUMENT, RUNS, source_hashes, prior_totals, request_budget, recoverable_failure
from medical_copilot.pdf_efficiency_agent import VERSION, CONDITIONS, ARTICLE_SYSTEM, ArticleJSONModel, ArticlePDFAgent

SET = "data/evaluation/pdf-article-details-20261012"
POLICY = "config/pdf-article-efficiency.json"
PROTOCOL = "docs/implementation/requirement-6-article-efficiency-protocol.md"
RUBRIC = "data/evaluation/pdf-qualitative-rubric.json"


def validate_set(root):
    selection = read_json(root / SET / "manifest.json")
    for name, field in [("questions.json", "questions_sha256"), ("answer-key.json", "answer_key_sha256")]:
        if file_hash(root / SET / name) != selection[field]:
            raise ValueError("Frozen article question/key changed.")
    if file_hash(root / DOCUMENT) != selection["document_json_sha256"] or file_hash((root / DOCUMENT).parent / "original.pdf") != selection["pdf_sha256"]:
        raise ValueError("Fixed extraction/original PDF changed.")
    questions = read_json(root / SET / "questions.json")["questions"]
    keys = read_json(root / SET / "answer-key.json")["keys"]
    support = read_json(root / SET / "field-evidence-key.json")
    if support["answer_key_sha256"] != selection["answer_key_sha256"]:
        raise ValueError("Field evidence belongs to another reference key.")
    if len(questions) != 6 or {q["question_id"] for q in questions} != {k["question_id"] for k in keys} or len({q["question_id"] for q in questions}) != 6:
        raise ValueError("Require six unique matched cases.")
    if any(ClinicalQuestion.model_validate(q).options is not None for q in questions) or sum(len(k["expected_fields"]) for k in keys) != 24:
        raise ValueError("Require six open-response tasks and twenty-four checks.")
    return questions, keys


def fingerprint(root, document, config):
    hashes = source_hashes(root)
    for name in ["pdf_efficiency_agent.py", "pdf_efficiency_evaluation.py"]:
        hashes[name] = file_hash(root / "src/medical_copilot" / name)
    return {"source_hashes": hashes, "config_sha256": file_hash(config), "effective_system_sha256": hashlib.sha256(ARTICLE_SYSTEM.encode()).hexdigest(),
        "document_sha256": file_hash(document), "selection_manifest_sha256": file_hash(root / SET / "manifest.json"),
        "answer_key_sha256": file_hash(root / SET / "answer-key.json"), "field_evidence_key_sha256": file_hash(root / SET / "field-evidence-key.json"),
        "policy_sha256": file_hash(root / POLICY), "protocol_sha256": file_hash(root / PROTOCOL), "rubric_sha256": file_hash(root / RUBRIC)}


def order(questions, conditions=CONDITIONS):
    return [(q["question_id"], arm) for n, q in enumerate(questions) for arm in conditions[n % len(conditions):] + conditions[:n % len(conditions)]]


def pilot_question():
    return {"question_id": "ARTICLE-PILOT", "question": "According to the summary of this article, how many numbered recommendations are presented? Return answer as a JSON object encoded in the answer string with only recommendations_count (number), explain the actual source, and give its exact registered citation. selected_option must be null.", "options": None}


def validate_prerequisite(directory, stage, expected):
    if directory is None:
        raise ValueError("Matching saved prerequisite required.")
    manifest = read_json(directory / "manifest.json")
    if manifest.get("stage") != stage or not manifest.get("finished_at_utc") or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Prerequisite differs from frozen article setup.")
    for arm in ("B-full-context",) if stage == "probe" else CONDITIONS:
        row = read_json(directory / arm / "answers/ARTICLE-PILOT.json")
        if row["status"] != "completed" or row["output_origin"] != "model" or not row["response"].get("citations"):
            raise ValueError("Prerequisite must complete with actual passage citations.")
        parse_object(row["response"]["answer"])
        if stage == "probe" and row["transport_formats"][0]["response_format"] != "json_schema":
            raise ValueError("Native final transport not tested.")


def score_fields(row, key, support):
    try:
        fields = parse_object((row.get("response") or {}).get("answer", ""))
        format_status = "exact_fields" if set(fields) == set(key["expected_fields"]) else "field_set_mismatch"
    except (ValueError, TypeError):
        fields, format_status = {}, "not_answer_json_object"
    citations = set((row.get("response") or {}).get("citations", []))
    checks = []
    for name, expected in key["expected_fields"].items():
        observed = fields.get(name)
        typed = type(observed) is bool if type(expected) is bool else type(observed) in {int, float} and math.isfinite(observed)
        matched = typed and observed == expected
        supplied = set(support["field_support"][name]) & set(row["evidence"])
        cited = supplied & citations
        state = "missing" if name not in fields else "null" if observed is None else "wrong_type" if not typed else "match" if matched else "mismatch"
        checks.append({"field": name, "reference_value": expected, "observed_value": observed, "status": state, "correct": matched,
                       "reference_passage_supplied": bool(supplied), "reference_passage_cited": bool(cited), "correct_with_cited_reference_passage": bool(matched and cited)})
    return {"question_id": key["question_id"], "format_status": format_status, "valid_model_answer": row["status"] == "completed" and row["output_origin"] == "model",
            "correct_fields": sum(c["correct"] for c in checks), "correct_fields_with_cited_reference_passage": sum(c["correct_with_cited_reference_passage"] for c in checks),
            "field_count": len(checks), "all_fields_correct": format_status == "exact_fields" and all(c["correct"] for c in checks), "checks": checks}


def make_model(config, root, budget, path, policy):
    backend, redactor, credential_source = create_backend(config, root)
    phased = PhaseBackend(backend, policy["native_final_json_schema"])
    audit = BaselineAudit(phased, config, budget, redactor, path)
    client = ModelClient(audit, config, redactor)
    def gate(request):
        count = len((ARTICLE_SYSTEM + json.dumps(request, ensure_ascii=False)).encode())
        if count > policy["max_input_bytes"]:
            raise BudgetBlocked("Article input bound exceeded.")
        extra = len(json.dumps(final_schema(request["question"], request["evidence_registry"])).encode()) + 256 if request["phase"] == "finalize" else 0
        try:
            budget.check(maximum_request_cost(config, count + extra))
        except BudgetLimitExceeded:
            raise BudgetBlocked("Cumulative request allowance exhausted.") from None
    def transient():
        return bool(audit.records and audit.records[-1]["status"] == "error" and recoverable_failure(audit.records[-1]))
    return ArticleJSONModel(client, gate, transient), client, audit, phased, credential_source


def summarize(run):
    manifest = read_json(run / "manifest.json")
    keys = {k["question_id"]: k for k in read_json(run / "answer-key.json")["keys"]} if manifest["stage"] == "main" else {}
    support = {k["question_id"]: k for k in read_json(run / "field-evidence-key.json")["keys"]} if keys else {}
    summaries = {}
    for arm in manifest["conditions"]:
        directory = run / arm
        rows = [read_json(p) for p in sorted((directory / "answers").glob("*.json"))]
        audit = read_json(directory / "usage-audit.json")["records"] if (directory / "usage-audit.json").exists() else []
        scores = [score_fields(r, keys[r["question"]["question_id"]], support[r["question"]["question_id"]]) for r in rows] if keys else []
        value = {"recorded": len(rows), "planned": len(manifest["question_ids"]), "valid_model_answers": sum(r["status"] == "completed" and r["output_origin"] == "model" for r in rows),
            "terminal_messages": sum(bool(r.get("response")) for r in rows), "statuses": dict(Counter(r["status"] for r in rows)), "provider_requests": len(audit),
            "cost_usd": sum(r.get("cost_usd") or 0 for r in audit), "unknown_charge_records": sum(r.get("cost_usd") is None for r in audit),
            "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in audit), "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in audit),
            "reasoning_tokens": sum(r.get("provider_usage", {}).get("reasoning_tokens", 0) for r in audit),
            "total_episode_seconds": round(sum(r["elapsed_seconds"] for r in rows), 4), "median_episode_seconds": round(statistics.median(r["elapsed_seconds"] for r in rows), 4) if rows else None,
            "successful_pdf_operations": dict(Counter(e["name"] for r in rows for e in r["document_tools"]["events"])),
            "max_input_bytes": max((e.get("input_bytes", 0) for r in rows for e in r["trace"]), default=0), "structured_citations": sum(len(r["response"].get("citations", [])) for r in rows)}
        if keys:
            value.update(correct_fields=sum(s["correct_fields"] for s in scores), correct_fields_with_cited_reference_passage=sum(s["correct_fields_with_cited_reference_passage"] for s in scores),
                         all_fields_correct_cases=sum(s["all_fields_correct"] for s in scores), recorded_fields=sum(s["field_count"] for s in scores))
            write_json(directory / "scores.json", {"scores": scores})
        write_json(directory / "summary.json", value)
        summaries[arm] = value
    result = {"stage": manifest["stage"], "conditions": summaries, "provider_requests": sum(s["provider_requests"] for s in summaries.values()),
              "cost_usd": sum(s["cost_usd"] for s in summaries.values()), "interpretation": "Efficiency experiment with exact extracted-field and actual cited-passage quality checks; no no-document arm."}
    write_json(run / "summary.json", result)
    return result


def run_stage(root, run, stage, *, probe=None, pilot=None, custom_questions=None, document=None, condition="C-pdf-tools", config_path=None):
    root, run = Path(root).resolve(), Path(run).resolve()
    if stage not in {"probe", "pilot", "main", "ask"} or run.exists() or run.parent != (root / RUNS).resolve() or condition not in CONDITIONS:
        raise ValueError("Use a supported document condition/stage and a fresh direct PDF-run subdirectory.")
    questions, keys = validate_set(root)
    document = Path(document).resolve() if document else root / DOCUMENT
    config_path = Path(config_path).resolve() if config_path else root / DEFAULT_PROFILE
    if stage != "ask" and (document != root / DOCUMENT or config_path != root / DEFAULT_PROFILE):
        raise ValueError("Evaluation fixes article and model profile.")
    if stage in {"probe", "pilot"}:
        questions = [pilot_question()]
    if stage == "ask":
        questions = custom_questions or []
    questions = [ClinicalQuestion.model_validate(q).model_dump(mode="json") for q in questions]
    if not questions or len({q["question_id"] for q in questions}) != len(questions) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", q["question_id"]) for q in questions):
        raise ValueError("Supply unique safe question IDs and valid strict inputs.")
    library = PDFLibrary.load([document])
    current = fingerprint(root, document, config_path)
    if stage in {"pilot", "main"}:
        validate_prerequisite(probe, "probe", current)
    if stage == "main":
        validate_prerequisite(pilot, "pilot", current)
    policy, config = read_json(root / POLICY), load_config(config_path)
    if config.max_retries or config.rate_limit_retries or config.json_repair_attempts:
        raise ValueError("SDK/paid format retries must remain disabled.")
    limits = ResilientLimits(**{k: v for k, v in policy.items() if k in ResilientLimits.model_fields})
    observation, prior = preflight(config, root), prior_totals(root)
    if not any("structured_outputs" in e["supported_parameters"] for e in observation["eligible_endpoints"]):
        raise ValueError("Native final transport must be advertised.")
    budget = request_budget(observation["ledger"]["total_usage"], prior)
    budget.check(maximum_request_cost(config, limits.max_input_bytes + 4096))
    conditions = ("B-full-context",) if stage == "probe" else (condition,) if stage == "ask" else CONDITIONS
    schedule = order(questions, conditions)
    run.mkdir(parents=True)
    save_trace(run / "preflight.json", observation)
    save_trace(run / "config.json", config.public_dict())
    save_trace(run / "questions.json", {"questions": questions})
    save_trace(run / "system-instructions.json", {"effective_system": ARTICLE_SYSTEM})
    for origin, target in [(document, "document.json"), (root / POLICY, "policy.json"), (root / PROTOCOL, "protocol.md"), (root / RUBRIC, "qualitative-rubric.json"),
                           (root / SET / "manifest.json", "selection-manifest.json"), (root / SET / "answer-key.json", "answer-key.json"),
                           (root / SET / "field-evidence-key.json", "field-evidence-key.json")]:
        shutil.copyfile(origin, run / target)
    snapshots = run / "source-snapshot"
    snapshots.mkdir()
    for name in current["source_hashes"]:
        shutil.copyfile(root / "src/medical_copilot" / name, snapshots / name)
    manifest = {"run_id": run.name, "requirement": 6, "stage": stage, "version": VERSION, "started_at_utc": utc_now(), **current,
        "document_origin": str(document), "questions_sha256": file_hash(run / "questions.json"), "question_ids": [q["question_id"] for q in questions],
        "conditions": list(conditions), "episode_order": schedule, "limits": limits.model_dump(mode="json"), "package_versions": package_versions(),
        "pdf_sha256": library.catalog()[0]["pdf_sha256"], "account_anchor_usd": prior["account_anchor_usd"] or observation["ledger"]["total_usage"],
        "prior_saved_requests": prior, "initial_ledger": observation["ledger"], "budget": budget.public_dict(), "probe_origin": str(probe) if probe else None, "pilot_origin": str(pilot) if pilot else None,
        "primary_objective": "Efficient access to the same article while preserving exact fields and cited support; no knowledge-only condition or accuracy-focused model ranking."}
    save_trace(run / "manifest.json", manifest)
    clients = {}
    for arm in conditions:
        directory = run / arm
        (directory / "answers").mkdir(parents=True)
        clients[arm] = make_model(config, root, budget, directory / "usage-audit.json", policy)
        save_trace(directory / "credential-resolution.json", {"source": clients[arm][-1], "value_saved": False})
    by_id = {q["question_id"]: q for q in questions}
    for number, (identifier, arm) in enumerate(schedule, 1):
        model, client, audit, phased, _ = clients[arm]
        audit.case_id = identifier
        offsets = len(client.calls), len(audit.records), len(phased.formats), len(model.local_normalizations)
        tools = PDFTools(library)
        active_limits = limits.model_copy(update={"max_research_turns": 0}) if stage == "probe" else limits
        print(f"[{number}/{len(schedule)}] START {stage} {arm} {identifier}", flush=True)
        row = ArticlePDFAgent(model, library, arm, limits=active_limits, pdf_tools=tools if arm == "C-pdf-tools" else None).run(by_id[identifier])
        row.update(model_calls=client.calls[offsets[0]:], response_audit=audit.records[offsets[1]:], transport_formats=phased.formats[offsets[2]:],
            answer_object_normalizations=[{**item, "call_index": item["call_index"] - offsets[0]} for item in model.local_normalizations[offsets[3]:]], evaluation_version=VERSION,
            input_cache_metadata=[c.get("usage", {}).get("input_token_details", {}) for c in client.calls[offsets[0]:]])
        save_trace(run / arm / "answers" / (identifier + ".json"), row)
        write_json(run / "budget.json", budget.public_dict())
        summarize(run)
        print(f"[{number}/{len(schedule)}] {arm} {identifier}: {row['status']}; {len(row['model_calls'])} requests; {row['elapsed_seconds']:.2f} seconds", flush=True)
    try:
        ledger = observe_ledger(config, root)
    except Exception as exc:
        ledger = {"checked_at_utc": utc_now(), "error": public_error(exc)}
    save_trace(run / "final-ledger.json", ledger)
    manifest.update(finished_at_utc=utc_now(), budget=budget.public_dict())
    write_json(run / "manifest.json", manifest)
    return summarize(run)
