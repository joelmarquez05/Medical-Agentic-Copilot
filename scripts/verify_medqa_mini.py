"""Verify released keys, frozen prompts, request bounds and saved study artifacts offline."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

from medical_copilot.config import load_config, resolve_credential
from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.medqa_mini import INPUT_BYTE_CAP, OUTPUT_CAP, SEED, SOURCE_HASH, SOURCE_REVISION, model_configs, question_input, score_answer

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/benchmark-runs/openweight-medqa-high-20261011"
SOURCE = ROOT / "data/evaluation/medqa-mini-20261011/source-test.jsonl"


def forbid_labels(value):
    forbidden = {"answer_idx", "answer", "reference_option", "expected_key_points", "critical_failures", "reference_rationale", "correct", "score"}
    if isinstance(value, dict):
        assert not forbidden.intersection(value)
        for child in value.values():
            forbid_labels(child)
    elif isinstance(value, list):
        for child in value:
            forbid_labels(child)


def main():
    manifest = read_json(RUN / "manifest.json")
    assert manifest.get("completed_at_utc"), "Only verify a completed, saved cohort."
    questions = read_json(RUN / "questions.json")["questions"]
    keys = read_json(RUN / "answer-key.json")["keys"]
    by_id = {item["id"]: item for item in questions}
    key_by_id = {item["question_id"]: item for item in keys}
    assert len(questions) == len(keys) == 10
    assert file_hash(SOURCE) == SOURCE_HASH
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    subset = read_json(RUN / "subset-manifest.json")
    assert subset["source_revision"] == SOURCE_REVISION and subset["seed"] == SEED
    assert len(rows) == subset["source_rows"] == 1273
    assert manifest["question_file_sha256"] == file_hash(RUN / "questions.json")
    assert manifest["answer_key_file_sha256"] == file_hash(RUN / "answer-key.json")
    for question in questions:
        original = rows[question["source_row_index"]]
        assert question_input(question) == question_input(original)
        digest = hashlib.sha256(json.dumps(question_input(question), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert digest == question["question_sha256"] == key_by_id[question["id"]]["question_sha256"]
        assert key_by_id[question["id"]]["answer_idx"] == original["answer_idx"]
        assert key_by_id[question["id"]]["answer"] == original["answer"]
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(RUN / "source-snapshot" / name) == digest
    if manifest.get("primary_cohort"):
        from medical_copilot.medqa_extension import LING_PROFILE, ling_config
        profiles = {LING_PROFILE: ling_config()}
        original = ROOT / manifest["primary_cohort"]
        assert file_hash(original / "manifest.json") == manifest["primary_manifest_sha256"]
        assert file_hash(RUN / "questions.json") == file_hash(original / "questions.json")
        assert file_hash(RUN / "answer-key.json") == file_hash(original / "answer-key.json")
    else:
        profiles = dict(model_configs())
    assert len(profiles) == len(manifest["profiles"])
    assert all("opus" not in item["config"]["model"].lower() for item in manifest["profiles"])
    for item in manifest["profiles"]:
        assert item["config"] == profiles[item["id"]].public_dict()
        assert item["config"]["max_tokens"] == OUTPUT_CAP
    expected = {(item["profile"], item["mode"], item["question_id"]) for item in manifest["case_order"]}
    observed, graded, requests, papers = set(), [], 0, {}
    charges = 0.0
    unknown_reserves = 0.0
    for path in sorted(RUN.glob("models/*/answers/*.json")):
        record = read_json(path)
        identifier = (record["profile"], record["mode"], record["question_id"])
        assert identifier not in observed
        observed.add(identifier)
        question = by_id[record["question_id"]]
        assert record["question_sha256"] == question["question_sha256"]
        calls = record["model_calls"]
        assert len(calls) <= (1 if record["mode"] == "direct" else 2)
        for call in calls:
            payload = json.loads(call["user_prompt"])
            forbid_labels(payload)
            assert question_input(payload) == question_input(question)
            assert len(call["system_prompt"].encode()) + len(call["user_prompt"].encode()) <= INPUT_BYTE_CAP
            assert call["system_prompt"] == manifest["prompts"][{"tool_decision": "plan", "final_answer": record["mode"]}[call["stage"]]]
            assert len(call["attempts"]) == 1
        if record["status"].startswith("not_run"):
            assert not calls and not record["audit_records"]
        if record["status"] == "completed":
            assert record["answer"]["selected_option"] in "ABCD"
            assert all(call["status"] == "ok" for call in calls)
            if record["mode"] == "direct":
                assert record["trace"] == ["answer"] and len(calls) == 1
            else:
                expected_trace = ["decide", "retrieve", "answer"] if record["decision"]["use_pubmed"] else ["decide", "answer"]
                assert record["trace"] == expected_trace and len(calls) == 2
        graded.append(score_answer(record, key_by_id[record["question_id"]]))
        for item in record["audit_records"]:
            if item.get("request_sent"):
                requests += 1
            if item.get("cost_usd") is not None:
                charges += item["cost_usd"]
            if item["status"] == "error" and item.get("request_sent"):
                unknown_reserves += item["estimated_maximum_usd"]
        search_records = [request for request in record["pubmed_requests"] if request["endpoint"].endswith("esearch.fcgi")]
        assert len(search_records) <= 1
        retrieval = record.get("retrieval", {})
        for paper in retrieval.get("sources", []):
            papers[paper["id"]] = paper
            assert paper["id"] == "P" + paper["pmid"] and not paper["retracted"]
        for request in record["pubmed_requests"]:
            if request.get("snapshot_file"):
                snapshot = path.parents[1] / "pubmed-snapshots" / request["snapshot_file"]
                assert file_hash(snapshot) == request["response_sha256"]
    assert observed == expected and len(observed) == 13 * len(profiles) and requests <= manifest["maximum_requests"]
    actual_scores = read_json(RUN / "scores.json")["scores"]
    ordering = lambda item: (item["profile"], item["mode"], item["question_id"])
    assert sorted(graded, key=ordering) == sorted(actual_scores, key=ordering)
    budget = read_json(RUN / "budget.json")
    assert budget["limit_usd"] == 6
    assert abs(budget["reported_or_estimated_spend_usd"] - manifest["initial_account_usage_usd"] - charges) < 1e-6
    assert budget["reported_or_estimated_spend_usd"] + budget["unreported_charge_reserve_usd"] <= 6
    assert budget["unreported_charge_reserve_usd"] + 1e-6 >= 0.15 + unknown_reserves
    key, _ = resolve_credential(load_config(ROOT / "config/models/openrouter-glm-5.3.json"), ROOT)
    scanned = 0
    for path in RUN.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert key.encode() not in content, "Credential found in an evaluation artifact."
            assert b'"reasoning_details"' not in content and b'"reasoning_content"' not in content, "Private reasoning body found."
            scanned += 1
    result = {"run_id": RUN.name, "passed": True, "questions": 10, "case_records": len(observed), "provider_requests": requests, "pubmed_unique_papers": len(papers), "objective_scores_match_released_keys": True, "unchanged_original_stems_options": True, "generation_labels_excluded": True, "source_snapshot_hashes_match": True, "request_and_prompt_bounds_hold": True, "cumulative_budget_guard_holds": True, "artifacts_scanned_without_credentials_or_reasoning_bodies": scanned, "new_model_calls_for_verification": 0}
    filename = "medqa-mini-ling-verification.json" if manifest.get("primary_cohort") else "medqa-mini-verification.json"
    write_json(ROOT / "data/checks" / filename, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", action="store_true")
    if parser.parse_args().extension:
        RUN = ROOT / "data/evaluation/benchmark-runs/ling-sante-medqa-extension-20261011"
    main()
