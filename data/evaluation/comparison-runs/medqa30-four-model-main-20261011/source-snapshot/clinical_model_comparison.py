"""Compare four hosted models with the frozen full agent and existing Gemma cohort."""

import hashlib
import shutil
from pathlib import Path

from medical_copilot.clinical_benchmark import BENCHMARK, RUBRIC, summarize, validate_benchmark
from medical_copilot.clinical_evaluation import (
    ACCOUNT_LIMIT, UNCERTAINTY_HEADROOM, maximum_request_cost, observe_ledger,
    pilot_question, preflight, public_error, utc_now,
)
from medical_copilot.clinical_graph import save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_literature_evaluation import PUBMED_CONFIG, RETRIEVAL_RUBRIC
from medical_copilot.clinical_resilient import RESILIENT_SYSTEM, VERSION, ResilientClinicalAgent, ResilientLimits
from medical_copilot.clinical_resilient_evaluation import POLICY, make_model
from medical_copilot.comparison_evaluation import RequestBudget
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.pubmed import PubMedClient, load_pubmed_config

MODELS = ("glm-5.3", "kimi-k3", "mimo-v2.6-pro", "qwen3.8-27b")
PROFILES = "config/models/requirement-5-20261011"
PROTOCOL = "docs/implementation/requirement-5-medqa30-protocol.md"
REFERENCE = "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"
SMOKE = "data/checks/clinical-pubmed-smoke-20261011"


def episode_order(question_ids, models=MODELS):
    """Question order stays frozen; only the first model rotates for each case."""
    return [(identifier, model) for number, identifier in enumerate(question_ids)
            for model in models[number % len(models):] + models[:number % len(models)]]


def validate_reference(root):
    selection, questions, keys = validate_benchmark(root)
    reference = read_json(root / REFERENCE / "manifest.json")
    if reference["questions_sha256"] != selection["questions_sha256"] or reference["answer_key_sha256"] != selection["answer_key_sha256"]:
        raise ValueError("Question/key artifacts differ from the completed Gemma condition.")
    for name, digest in reference["source_hashes"].items():
        if file_hash(root / "src/medical_copilot" / name) != digest:
            raise ValueError("Frozen Gemma operational sources changed.")
    for origin, field in [(POLICY, "policy_sha256"), (PUBMED_CONFIG, "pubmed_config_sha256"),
                          (RUBRIC, "rubric_sha256"), (RETRIEVAL_RUBRIC, "retrieval_rubric_sha256")]:
        if file_hash(root / origin) != reference[field]:
            raise ValueError("Frozen policy/tool/rubric changed.")
    if hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest() != reference["effective_system_sha256"]:
        raise ValueError("Effective answering prompt changed.")
    smoke = read_json(root / SMOKE / "smoke.json")
    if not smoke["passed"] or smoke["tool_source_sha256"] != reference["source_hashes"]["clinical_literature.py"]:
        raise ValueError("Matching PubMed smoke is required.")
    return reference, selection, questions, keys


def profile_hashes(root):
    return {model: file_hash(root / PROFILES / (model + ".json")) for model in MODELS}


def validate_prerequisite(root, directory, stage, source_hashes):
    if directory is None:
        raise ValueError("A matching completed prerequisite stage is required.")
    manifest = read_json(directory / "manifest.json")
    expected = {"stage": stage, "protocol_sha256": file_hash(root / PROTOCOL),
                "profile_hashes": profile_hashes(root), "source_hashes": source_hashes,
                "selection_manifest_sha256": file_hash(root / BENCHMARK / "manifest.json")}
    if not manifest.get("finished_at_utc") or any(manifest.get(k) != value for k, value in expected.items()):
        raise ValueError("Prerequisite configuration/provenance is different.")
    for model in MODELS:
        row = read_json(directory / model / "answers/PILOT-01.json")
        if row["status"] not in {"completed", "completed_with_citation_issues"} or row["output_origin"] != "model":
            raise ValueError("A model has not completed the prerequisite integration.")
        if stage == "probes" and row["transport_formats"][0]["response_format"] != "json_schema":
            raise ValueError("Native final schema has not been confirmed for this model.")


def prior_request_totals(root):
    """Retain saved charges/reserves even if a refreshed account ledger lags."""
    known, unknown, count = 0.0, 0.0, 0
    for path in (root / "data/evaluation/comparison-runs").glob("*/manifest.json"):
        manifest = read_json(path)
        if manifest.get("requirement") != 5 or manifest.get("protocol_origin") != PROTOCOL:
            continue
        for audit_path in path.parent.glob("*/usage-audit.json"):
            for row in read_json(audit_path)["records"]:
                count += bool(row["request_sent"])
                if row.get("cost_usd") is not None:
                    known += row["cost_usd"]
                else:
                    unknown += row["estimated_maximum_usd"]
    return {"known_cost_usd": known, "unknown_request_reserve_usd": unknown, "saved_requests": count}


def summarize_panel(run):
    manifest = read_json(run / "manifest.json")
    rows = {model: summarize(run / model) for model in MODELS}
    result = {"stage": manifest["stage"], "models": rows,
              "new_provider_requests": sum(r["provider_requests"] for r in rows.values()),
              "reported_or_estimated_cost_usd": sum(r["reported_or_estimated_cost_usd"] for r in rows.values()),
              "gemma_reference": REFERENCE,
              "interpretation": "New models use identical frozen questions/full agent; Gemma is retained, not regenerated."}
    write_json(run / "summary.json", result)
    return result


def run_stage(root, run, stage, probes=None, pilots=None):
    if stage not in {"probes", "pilots", "benchmark"} or run.exists():
        raise ValueError("Use a supported stage and a fresh directory.")
    reference, selection, questions, keys = validate_reference(root)
    names = list(reference["source_hashes"]) + ["clinical_model_comparison.py"]
    hashes = {name: file_hash(root / "src/medical_copilot" / name) for name in names}
    if stage in {"pilots", "benchmark"}:
        validate_prerequisite(root, probes, "probes", hashes)
    if stage == "benchmark":
        validate_prerequisite(root, pilots, "pilots", hashes)
    policy = read_json(root / POLICY)
    limits = ResilientLimits(**{k: v for k, v in policy.items() if k in ResilientLimits.model_fields})
    if stage != "benchmark":
        questions = [pilot_question(root)]
    configs = {model: load_config(root / PROFILES / (model + ".json")) for model in MODELS}
    observations = {model: preflight(config, root) for model, config in configs.items()}
    for config, observation in zip(configs.values(), observations.values(), strict=True):
        if config.max_tokens != 16384 or config.temperature != 0 or config.timeout_seconds != 180:
            raise ValueError("Keep the shared output/sampling/timeout condition.")
        if not any("structured_outputs" in e["supported_parameters"] for e in observation["eligible_endpoints"]):
            raise ValueError("Pinned provider does not advertise native final JSON Schema.")
    prior = prior_request_totals(root)
    anchor = read_json(root / "data/checks/requirement-4-cost-audit-20261011.json")["latest_ledger"]["total_usage"]
    initial = max(max(o["ledger"]["total_usage"] for o in observations.values()), anchor + prior["known_cost_usd"])
    budget = RequestBudget(ACCOUNT_LIMIT, initial)
    budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM + prior["unknown_request_reserve_usd"]
    # Every actual request has its own full-cap guard. Do not claim that the
    # hypothetical 600-request primary ceiling fits the remaining account limit.
    for config in configs.values():
        budget.check(maximum_request_cost(config, limits.max_input_bytes + 4096))
    run.mkdir(parents=True)
    save_trace(run / "preflight.json", observations)
    for origin, target in [(BENCHMARK + "/questions.json", "frozen-benchmark-questions.json"),
                           (BENCHMARK + "/answer-key.json", "answer-key.json"),
                           (BENCHMARK + "/manifest.json", "selection-manifest.json"),
                           (BENCHMARK + "/history-exclusions.json", "history-exclusions.json"),
                           (PROTOCOL, "protocol.md"), (RUBRIC, "rubric.json"),
                           (RETRIEVAL_RUBRIC, "retrieval-rubric.json"), (POLICY, "policy.json"),
                           (PUBMED_CONFIG, "pubmed-config.json")]:
        shutil.copyfile(root / origin, run / target)
    snapshots = run / "source-snapshot"
    snapshots.mkdir()
    for name in names:
        shutil.copyfile(root / "src/medical_copilot" / name, snapshots / name)
    save_trace(run / "system-instructions.json", {"effective_system": RESILIENT_SYSTEM})
    order = episode_order([q["question_id"] for q in questions])
    manifest = {"run_id": run.name, "stage": stage, "requirement": 5, "agent_version": VERSION,
                "started_at_utc": utc_now(), "protocol_origin": PROTOCOL,
                "protocol_sha256": file_hash(root / PROTOCOL), "source_hashes": hashes,
                "profile_hashes": profile_hashes(root), "selection_manifest_sha256": file_hash(root / BENCHMARK / "manifest.json"),
                "questions_sha256": selection["questions_sha256"], "answer_key_sha256": selection["answer_key_sha256"],
                "rubric_sha256": file_hash(root / RUBRIC), "retrieval_rubric_sha256": file_hash(root / RETRIEVAL_RUBRIC),
                "policy_sha256": file_hash(root / POLICY), "pubmed_config_sha256": file_hash(root / PUBMED_CONFIG),
                "effective_system_sha256": reference["effective_system_sha256"],
                "reference_origin": REFERENCE, "reference_manifest_sha256": file_hash(root / REFERENCE / "manifest.json"),
                "probe_origin": str(probes) if probes else None,
                "probe_manifest_sha256": file_hash(probes / "manifest.json") if probes else None,
                "pilot_origin": str(pilots) if pilots else None,
                "pilot_manifest_sha256": file_hash(pilots / "manifest.json") if pilots else None,
                "question_ids": [q["question_id"] for q in questions], "models": list(MODELS), "episode_order": order,
                "limits": limits.model_dump(mode="json"), "initial_account_usage_or_anchor_usd": initial,
                "prior_saved_requests": prior, "budget": budget.public_dict(), "package_versions": package_versions(),
                "allocation": "Per-request full-cap guard, shared account budget; no entire hypothetical workload reserve.",
                "condition": "Frozen full agent; autonomous live PubMed; new model profiles only; Gemma not rerun."}
    save_trace(run / "manifest.json", manifest)
    clients = {}
    for model, config in configs.items():
        directory = run / model
        directory.mkdir()
        (directory / "answers").mkdir()
        save_trace(directory / "config.json", config.public_dict())
        shutil.copyfile(run / "answer-key.json", directory / "answer-key.json")
        save_trace(directory / "manifest.json", {"run_id": model, "stage": "benchmark" if stage == "benchmark" else stage,
                   "question_ids": manifest["question_ids"], "parent_run": run.name})
        clients[model] = make_model(config, root, budget, directory / "usage-audit.json", policy)
        save_trace(directory / "credential-resolution.json", {"source": clients[model][-1], "value_saved": False})
    by_id = {q["question_id"]: q for q in questions}
    for number, (identifier, slug) in enumerate(order, 1):
        directory = run / slug
        model, client, audit, phased, _ = clients[slug]
        audit.case_id = identifier
        offsets = len(client.calls), len(audit.records), len(phased.formats)
        tools = ClinicalLiteratureTools(PubMedClient(load_pubmed_config(root / PUBMED_CONFIG), snapshot_dir=directory / "http-snapshots"))
        active_limits = limits.model_copy(update={"max_research_turns": 0}) if stage == "probes" else limits
        print(f"[{number}/{len(order)}] START {stage} {slug} {identifier}", flush=True)
        row = ResilientClinicalAgent(model, tools.definitions(), active_limits).run(by_id[identifier])
        row.update(model_calls=client.calls[offsets[0]:], response_audit=audit.records[offsets[1]:],
                   transport_formats=phased.formats[offsets[2]:], literature=tools.public_record(),
                   benchmark_condition="requirement-5-frozen-full-agent", model_profile=slug)
        save_trace(directory / "answers" / (identifier + ".json"), row)
        write_json(run / "budget.json", budget.public_dict())
        summarize_panel(run)
        print(f"[{number}/{len(order)}] {slug} {identifier}: {row['status']}; {len(row['model_calls'])} requests", flush=True)
    try:
        ledger = observe_ledger(next(iter(configs.values())), root)
    except Exception as exc:
        ledger = {"checked_at_utc": utc_now(), "error": public_error(exc)}
    save_trace(run / "final-ledger.json", ledger)
    manifest.update(finished_at_utc=utc_now(), budget=budget.public_dict())
    write_json(run / "manifest.json", manifest)
    return summarize_panel(run)
