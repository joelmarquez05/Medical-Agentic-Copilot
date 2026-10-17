"""Frozen MedQA test sampling and the requirement-4 full-agent Gemma evaluation."""

import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from medical_copilot.clinical_evaluation import (
    ACCOUNT_LIMIT, UNCERTAINTY_HEADROOM, DEFAULT_PROFILE, maximum_request_cost,
    observe_ledger, pilot_question, preflight, public_error, score_record, utc_now,
)
from medical_copilot.clinical_graph import ClinicalQuestion, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_literature_evaluation import PUBMED_CONFIG, RETRIEVAL_RUBRIC, SNAPSHOT_NAMES, check_baseline
from medical_copilot.clinical_resilient import RESILIENT_SYSTEM, VERSION, ResilientClinicalAgent, ResilientLimits
from medical_copilot.clinical_resilient_evaluation import POLICY, make_model
from medical_copilot.comparison_evaluation import RequestBudget
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.pubmed import PubMedClient, load_pubmed_config

BENCHMARK = "data/evaluation/medqa-benchmark-20261011"
SOURCE = "data/evaluation/medqa-mini-20261011/source-test.jsonl"
SOURCE_SHA = "c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb"
REVISION = "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"
PROTOCOL = "docs/implementation/requirement-4-gemma-protocol.md"
RUBRIC = "data/evaluation/clinical-benchmark-rubric.json"
REFERENCE = "data/evaluation/retrieval-runs/gemma-resilient-main-20261011"
PROBE = "data/evaluation/retrieval-runs/gemma-resilient-probe-20261011"
SEED = 202610114
VISUAL = re.compile(r"\b(?:image|photograph|photomicrograph|micrograph|figure|diagram)\b|\bshown\s+(?:below|above)\b", re.I)


def question_hash(text):
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def extract_questions(value):
    if isinstance(value, list):
        return [text for item in value for text in extract_questions(item)]
    if isinstance(value, dict):
        question = value.get("question")
        result = [question] if isinstance(question, str) else extract_questions(question)
        return result + extract_questions(value.get("questions"))
    return []


def historical_questions(root):
    paths = set((root / "data/evaluation").rglob("questions.json"))
    paths.update(path for path in (root / "data/evaluation").rglob("*.json") if path.parent.name == "answers")
    paths = sorted(path for path in paths if BENCHMARK not in path.relative_to(root).as_posix())
    hashes, origins = set(), []
    for path in paths:
        texts = extract_questions(read_json(path))
        if texts:
            digests = sorted({question_hash(text) for text in texts})
            hashes.update(digests)
            origins.append({"path": path.relative_to(root).as_posix(), "sha256": file_hash(path), "question_hashes": digests})
    pilot = pilot_question(root)
    hashes.add(question_hash(pilot["question"]))
    return {"normalization": "Question text only; whitespace collapsed, Unicode casefold, SHA-256; no semantic deduplication.",
            "question_hashes": sorted(hashes), "origins": origins,
            "pilot_question_hash": question_hash(pilot["question"]), "questions_seen_or_prepared": len(hashes)}


def select_rows(rows, history_hashes, seed=SEED, count=30):
    history_hashes, seen, eligible, excluded = set(history_hashes), set(), [], []
    for index, row in enumerate(rows):
        if set(row.get("options", {})) != set("ABCD") or row.get("answer_idx") not in row["options"] or row["options"][row["answer_idx"]] != row.get("answer"):
            raise ValueError("Released source option/label/answer inconsistency; do not choose a replacement key.")
        digest = question_hash(row["question"])
        reason = None
        if digest in history_hashes:
            reason = "Question text already prepared/used in the project or outside-set pilot."
        elif digest in seen:
            reason = "Duplicate normalized question text in this source; retain only the first occurrence."
        elif VISUAL.search(row["question"] + " " + " ".join(row["options"].values())):
            reason = "Conservative unavailable-visual reference screen; no image files are supplied."
        else:
            try:
                ClinicalQuestion(question_id="CHECK", question=row["question"], options=row["options"])
            except ValueError:
                reason = "Question cannot satisfy the existing input contract."
        seen.add(digest)
        if reason:
            excluded.append({"source_row_index": index, "reason": reason, "question_hash": digest})
        else:
            eligible.append(index)
    if len(eligible) < count:
        raise ValueError("Not enough eligible questions.")
    order = list(eligible)
    random.Random(seed).shuffle(order)
    return {"eligible_rows": len(eligible), "excluded_rows": excluded,
            "ordered_candidate_indices": order[:max(count, 50)], "selected_source_row_indices": order[:count]}


def prepare(root):
    directory = root / BENCHMARK
    if directory.exists() or file_hash(root / SOURCE) != SOURCE_SHA:
        raise ValueError("Use the unchanged source and a previously unprepared benchmark directory.")
    rows = [json.loads(line) for line in (root / SOURCE).read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1273:
        raise ValueError("Unexpected frozen source row count.")
    history = historical_questions(root)
    selected = select_rows(rows, history["question_hashes"])
    questions, keys, selection = [], [], []
    for number, index in enumerate(selected["selected_source_row_indices"], 1):
        row, identifier = rows[index], f"BM{number:02d}"
        question = {"question_id": identifier, "question": row["question"], "options": row["options"]}
        questions.append(question)
        keys.append({"question_id": identifier, "key_type": "released_medqa_label", "answer_idx": row["answer_idx"],
                     "answer": row["answer"], "source_zero_based_index": index})
        selection.append({"question_id": identifier, "source_zero_based_index": index, "question_hash": question_hash(row["question"]),
                          "source_meta_info": row.get("meta_info"), "wording_options_key_changed": False})
    directory.mkdir(parents=True)
    shutil.copyfile(root / SOURCE, directory / "source-test.jsonl")
    save_trace(directory / "history-exclusions.json", history)
    save_trace(directory / "questions.json", {"dataset_id": "requirement-4-medqa-test-30", "questions": questions})
    save_trace(directory / "answer-key.json", {"policy": "Unchanged released option labels, not an LLM correctness judgment or independently obtained examination-board key.", "keys": keys})
    manifest = {"prepared_at_utc": utc_now(), "source_revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/{REVISION}/phrases_no_exclude_test.jsonl",
        "source_sha256": SOURCE_SHA, "source_rows": len(rows), "seed": SEED, "question_count": len(questions), **selected,
        "selection": selection, "history_exclusions_sha256": file_hash(directory / "history-exclusions.json"),
        "questions_sha256": file_hash(directory / "questions.json"), "answer_key_sha256": file_hash(directory / "answer-key.json"),
        "screen": VISUAL.pattern, "no_length_word_cap": True, "no_difficulty_score_or_answer_position_filter": True,
        "selection_algorithm": "Filter in original source order, then random.Random(seed).shuffle; first 30 of an independently frozen 50-item candidate order.",
        "held_out_meaning": "Unused for this project's prior development/answers; benchmark pretraining contamination remains possible.",
        "requirement_1_entry": "B01", "openevidence_use_source": "https://doi.org/10.1038/s41591-026-04431-5"}
    save_trace(directory / "manifest.json", manifest)
    return manifest


def validate_benchmark(root):
    directory = root / BENCHMARK
    manifest = read_json(directory / "manifest.json")
    for name, key in [("source-test.jsonl", "source_sha256"), ("questions.json", "questions_sha256"),
                      ("answer-key.json", "answer_key_sha256"), ("history-exclusions.json", "history_exclusions_sha256")]:
        if file_hash(directory / name) != manifest[key]:
            raise ValueError("Frozen benchmark artifact changed.")
    rows = [json.loads(line) for line in (directory / "source-test.jsonl").read_text(encoding="utf-8").splitlines()]
    history = read_json(directory / "history-exclusions.json")
    reproduced = select_rows(rows, history["question_hashes"], manifest["seed"], manifest["question_count"])
    for key, value in reproduced.items():
        if manifest[key] != value:
            raise ValueError("Frozen sampling cannot be reproduced.")
    questions = read_json(directory / "questions.json")["questions"]
    keys = read_json(directory / "answer-key.json")["keys"]
    if len(questions) != len(keys) or len(questions) != manifest["question_count"]:
        raise ValueError("Benchmark question/key count mismatch.")
    for question, key, selected in zip(questions, keys, manifest["selection"], strict=True):
        row = rows[selected["source_zero_based_index"]]
        if question != {"question_id": selected["question_id"], "question": row["question"], "options": row["options"]} or key["answer_idx"] != row["answer_idx"] or key["answer"] != row["answer"] or key["question_id"] != question["question_id"]:
            raise ValueError("Original wording/options/key changed.")
    return manifest, questions, keys


def wilson(correct, count):
    if not count:
        return None
    z, p = 1.959963984540054, correct / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0, center-radius), min(1, center+radius)]


def summarize(run):
    manifest = read_json(run / "manifest.json")
    records = [read_json(p) for p in sorted((run / "answers").glob("*.json"))]
    keys = {k["question_id"]: k for k in read_json(run / "answer-key.json")["keys"]}
    scores = [score_record(row, keys[row["question"]["question_id"]]) for row in records] if manifest["stage"] == "benchmark" else []
    audit = [entry for row in records for entry in row["response_audit"]]
    events = [event for row in records for event in row["trace"] if event["node"] == "execute_tool"]
    http = [request for row in records for request in row["literature"]["http_requests"]]
    valid = sum(s["valid_completed_answer"] for s in scores)
    correct = sum(s["correct"] is True for s in scores)
    planned = len(manifest["question_ids"])
    summary = {"stage": manifest["stage"], "planned": planned, "recorded": len(records),
        "statuses": dict(Counter(row["status"] for row in records)), "provider_requests": len(audit),
        "reported_or_estimated_cost_usd": sum(entry.get("cost_usd") or 0 for entry in audit),
        "unknown_charge_records": sum(entry.get("cost_usd") is None for entry in audit),
        "terminal_message_coverage": sum(bool(row["response"]) for row in records),
        "application_fallbacks": sum(row["output_origin"] == "application" for row in records),
        "local_normalizations": sum(len(event.get("normalizations", [])) for row in records for event in row["trace"]),
        "closing_phase_episodes": sum(row["final_attempts"] > 0 for row in records),
        "tool_action_attempts": len(events), "tool_execution_attempts": sum(e["execution_attempted"] for e in events),
        "tool_statuses": dict(Counter(e["status"] for e in events)), "tool_names": dict(Counter(e["name"] for e in events)),
        "episodes_with_search": sum(any(e["name"] == "search_pubmed" for e in row["literature"]["events"]) for row in records),
        "episodes_with_citations": sum(bool(row["response"].get("citations")) for row in records),
        "pubmed_logical_http_requests": len(http), "pubmed_http_attempts": sum(len(r["attempts"]) for r in http),
        "question_paper_pairs": sum(len({p["pmid"] for e in row["literature"]["events"] for p in e["papers"]}) for row in records),
        "input_tokens": sum(a.get("usage", {}).get("input_tokens", 0) for a in audit),
        "output_tokens": sum(a.get("usage", {}).get("output_tokens", 0) for a in audit),
        "reasoning_tokens": sum(a.get("provider_usage", {}).get("reasoning_tokens", 0) for a in audit)}
    if scores:
        summary.update(correct=correct, valid_completed=valid, incorrect_completed=valid-correct,
            objective_score_planned=correct/planned, objective_score_recorded=correct/len(records),
            accuracy_among_valid=correct/valid if valid else None,
            wilson_95_interval_planned=wilson(correct, planned), unrecorded=planned-len(records),
            interpretation="Primary score is exact released-key matches / all planned cases, including nonanswers; incomplete collection remains explicit. CI does not account for benchmark contamination or clinical deployment.")
        write_json(run / "scores.json", {"scores": scores, "key_policy": "Released key only; qualitative appraisal cannot change correctness."})
    write_json(run / "summary.json", summary)
    return summary


def run_stage(root, run, stage, smoke, pilot=None):
    if stage not in {"pilot", "benchmark"} or run.exists():
        raise ValueError("Use a valid stage and a fresh directory.")
    check_baseline(root)
    selection, questions, keys = validate_benchmark(root)
    reference = read_json(root / REFERENCE / "manifest.json")
    for name, digest in reference["source_hashes"].items():
        if file_hash(root / "src/medical_copilot" / name) != digest:
            raise ValueError("Active agent changed since its completed v3 study.")
    if reference["policy_sha256"] != file_hash(root / POLICY) or reference["config_sha256"] != file_hash(root / DEFAULT_PROFILE):
        raise ValueError("Policy/profile differs from v3.")
    native_probe = read_json(root / PROBE / "answers/PILOT-01.json")
    if native_probe["status"] != "completed" or native_probe["transport_formats"][0]["response_format"] != "json_schema":
        raise ValueError("Confirmed native final transport required.")
    smoke_record = read_json(smoke / "smoke.json")
    if not smoke_record["passed"] or smoke_record["tool_source_sha256"] != file_hash(root / "src/medical_copilot/clinical_literature.py") or smoke_record["pubmed_config_sha256"] != file_hash(root / PUBMED_CONFIG):
        raise ValueError("Matching successful PubMed smoke required.")
    source_names = SNAPSHOT_NAMES + ["clinical_resilient.py", "clinical_resilient_evaluation.py", "clinical_benchmark.py"]
    hashes = {name: file_hash(root / "src/medical_copilot" / name) for name in source_names}
    if stage == "benchmark":
        if pilot is None:
            raise ValueError("Separate benchmark runner pilot required.")
        pm = read_json(pilot / "manifest.json")
        answer = read_json(pilot / "answers/PILOT-01.json")
        if answer["status"] != "completed" or pm["stage"] != "pilot" or pm["source_hashes"] != hashes:
            raise ValueError("Pilot is not a successful matching runner integration.")
        for key, expected in [("selection_manifest_sha256", file_hash(root / BENCHMARK / "manifest.json")),
                              ("protocol_sha256", file_hash(root / PROTOCOL)), ("rubric_sha256", file_hash(root / RUBRIC))]:
            if pm[key] != expected:
                raise ValueError("Pilot protocol/data differs.")
    else:
        questions = [pilot_question(root)]
        if any(question_hash(q["question"]) in {question_hash(item["question"]) for item in questions} for q in read_json(root / BENCHMARK / "questions.json")["questions"]):
            raise ValueError("Pilot overlaps the benchmark.")
    policy = read_json(root / POLICY)
    limits = ResilientLimits(**{key: value for key, value in policy.items() if key in ResilientLimits.model_fields})
    config = load_config(root / DEFAULT_PROFILE)
    observation = preflight(config, root)
    if not any("structured_outputs" in e["supported_parameters"] for e in observation["eligible_endpoints"]):
        raise ValueError("Pinned native schema support no longer advertised.")
    budget = RequestBudget(ACCOUNT_LIMIT, observation["ledger"]["total_usage"])
    budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM
    budget.check(len(questions)*limits.max_model_turns*maximum_request_cost(config, limits.max_input_bytes+4096))
    run.mkdir(parents=True)
    (run / "answers").mkdir()
    save_trace(run / "preflight.json", observation)
    save_trace(run / "config.json", config.public_dict())
    save_trace(run / "system-instructions.json", {"effective_system": RESILIENT_SYSTEM, "unchanged_from": REFERENCE})
    for origin, name in [(root / BENCHMARK / "questions.json", "frozen-benchmark-questions.json"),
                         (root / BENCHMARK / "answer-key.json", "answer-key.json"),
                         (root / BENCHMARK / "manifest.json", "selection-manifest.json"),
                         (root / BENCHMARK / "history-exclusions.json", "history-exclusions.json"),
                         (root / PROTOCOL, "protocol.md"), (root / RUBRIC, "rubric.json"),
                         (root / RETRIEVAL_RUBRIC, "retrieval-rubric.json"),
                         (root / PUBMED_CONFIG, "pubmed-config.json"), (root / POLICY, "policy.json")]:
        shutil.copyfile(origin, run / name)
    snapshots = run / "source-snapshot"
    snapshots.mkdir()
    for name in source_names:
        shutil.copyfile(root / "src/medical_copilot" / name, snapshots / name)
    manifest = {"run_id": run.name, "stage": stage, "requirement": 4, "agent_version": VERSION,
        "started_at_utc": utc_now(), "question_ids": [q["question_id"] for q in questions],
        "source_hashes": hashes, "config_sha256": file_hash(root / DEFAULT_PROFILE), "policy_sha256": file_hash(root / POLICY),
        "questions_sha256": selection["questions_sha256"], "answer_key_sha256": selection["answer_key_sha256"],
        "selection_manifest_sha256": file_hash(root / BENCHMARK / "manifest.json"),
        "protocol_sha256": file_hash(root / PROTOCOL), "rubric_sha256": file_hash(root / RUBRIC),
        "retrieval_rubric_sha256": file_hash(root / RETRIEVAL_RUBRIC), "pubmed_config_sha256": file_hash(root / PUBMED_CONFIG),
        "effective_system_sha256": hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest(),
        "reference_origin": REFERENCE, "reference_manifest_sha256": file_hash(root / REFERENCE / "manifest.json"),
        "native_probe_origin": PROBE, "native_probe_manifest_sha256": file_hash(root / PROBE / "manifest.json"),
        "smoke_origin": str(smoke), "smoke_sha256": file_hash(smoke / "smoke.json"),
        "pilot_origin": str(pilot) if pilot else None, "pilot_manifest_sha256": file_hash(pilot / "manifest.json") if pilot else None,
        "limits": limits.model_dump(mode="json"), "initial_ledger": observation["ledger"], "budget": budget.public_dict(),
        "package_versions": package_versions(), "condition": "Full agent; PubMed available, autonomous; Gemma only.",
        "point_5_status": "Deferred: user will review Gemma results before choosing further models/conditions."}
    save_trace(run / "manifest.json", manifest)
    model, client, audit, phased, credential_source = make_model(config, root, budget, run / "usage-audit.json", policy)
    save_trace(run / "credential-resolution.json", {"source": credential_source, "value_saved": False})
    for number, question in enumerate(questions, 1):
        audit.case_id = question["question_id"]
        offsets = len(client.calls), len(audit.records), len(phased.formats)
        tools = ClinicalLiteratureTools(PubMedClient(load_pubmed_config(root / PUBMED_CONFIG), snapshot_dir=run / "http-snapshots"))
        print(f"[{number}/{len(questions)}] START {stage} {question['question_id']}", flush=True)
        row = ResilientClinicalAgent(model, tools.definitions(), limits).run(question)
        row.update(model_calls=client.calls[offsets[0]:], response_audit=audit.records[offsets[1]:],
            transport_formats=phased.formats[offsets[2]:], literature=tools.public_record(), benchmark_condition="requirement-4-gemma-full-agent")
        save_trace(run / "answers" / (question["question_id"] + ".json"), row)
        write_json(run / "budget.json", budget.public_dict())
        summarize(run)
        print(f"[{number}/{len(questions)}] {question['question_id']}: {row['status']}; {len(row['model_calls'])} provider request(s)", flush=True)
    try:
        ledger = observe_ledger(config, root)
    except Exception as exc:
        ledger = {"checked_at_utc": utc_now(), "error": public_error(exc)}
    save_trace(run / "final-ledger.json", ledger)
    manifest.update(finished_at_utc=utc_now(), budget=budget.public_dict())
    write_json(run / "manifest.json", manifest)
    summary = summarize(run)
    print(json.dumps(summary), flush=True)
    return summary
