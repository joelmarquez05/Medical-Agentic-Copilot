"""Verify and replay the saved MedQA full-agent benchmark without APIs/credentials."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from unittest.mock import patch

from medical_copilot.clinical_benchmark import BENCHMARK, validate_benchmark, question_hash, wilson
from medical_copilot.clinical_evaluation import DEFAULT_PROFILE, score_record
from medical_copilot.clinical_graph import BudgetBlocked, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_resilient import (
    RESILIENT_SYSTEM, VERSION, PhaseBackend, ResilientClinicalAgent,
    ResilientJSONModel, ResilientLimits, final_schema,
)
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.models import ModelClient, Redactor
from medical_copilot.pubmed import PubMedClient, PubMedConfig
from verify_clinical_resilient import BoundReplay
from verify_clinical_literature import HTTPReplay, without_transient


def verify(root, run):
    manifest = read_json(run / "manifest.json")
    assert manifest.get("finished_at_utc") and manifest["requirement"] == 4 and manifest["agent_version"] == VERSION
    selection, inputs, keys = validate_benchmark(root)
    assert file_hash(root / BENCHMARK / "manifest.json") == manifest["selection_manifest_sha256"]
    assert manifest["questions_sha256"] == selection["questions_sha256"]
    assert manifest["answer_key_sha256"] == selection["answer_key_sha256"]
    for name, key in [("frozen-benchmark-questions.json", "questions_sha256"), ("answer-key.json", "answer_key_sha256"),
                      ("selection-manifest.json", "selection_manifest_sha256"), ("protocol.md", "protocol_sha256"),
                      ("rubric.json", "rubric_sha256"), ("retrieval-rubric.json", "retrieval_rubric_sha256"),
                      ("policy.json", "policy_sha256"), ("pubmed-config.json", "pubmed_config_sha256")]:
        assert file_hash(run / name) == manifest[key]
    assert file_hash(run / "history-exclusions.json") == selection["history_exclusions_sha256"]
    assert file_hash(root / DEFAULT_PROFILE) == manifest["config_sha256"]
    assert hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest() == manifest["effective_system_sha256"]
    assert read_json(run / "system-instructions.json")["effective_system"] == RESILIENT_SYSTEM
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(run / "source-snapshot" / name) == digest == file_hash(root / "src/medical_copilot" / name)
    reference = root / manifest["reference_origin"]
    assert file_hash(reference / "manifest.json") == manifest["reference_manifest_sha256"]
    rm = read_json(reference / "manifest.json")
    for name, digest in rm["source_hashes"].items():
        assert manifest["source_hashes"][name] == digest
    assert manifest["policy_sha256"] == rm["policy_sha256"]
    assert manifest["config_sha256"] == rm["config_sha256"]
    assert manifest["effective_system_sha256"] == rm["effective_system_sha256"]
    assert file_hash(root / manifest["native_probe_origin"] / "manifest.json") == manifest["native_probe_manifest_sha256"]
    assert file_hash(Path(manifest["smoke_origin"]) / "smoke.json") == manifest["smoke_sha256"]
    if manifest["pilot_origin"]:
        assert file_hash(Path(manifest["pilot_origin"]) / "manifest.json") == manifest["pilot_manifest_sha256"]
    histories = read_json(run / "history-exclusions.json")
    for origin in histories["origins"]:
        assert file_hash(root / origin["path"]) == origin["sha256"]
    input_map = {q["question_id"]: q for q in inputs}
    key_map = {k["question_id"]: k for k in keys}
    config = load_config(root / DEFAULT_PROFILE)
    pm_config = PubMedConfig.model_validate(read_json(run / "pubmed-config.json"))
    policy = read_json(run / "policy.json")
    limits = ResilientLimits(**manifest["limits"])
    records = [read_json(path) for path in sorted((run / "answers").glob("*.json"))]
    assert [r["question"]["question_id"] for r in records] == sorted(manifest["question_ids"])
    audit = read_json(run / "usage-audit.json")["records"]
    assert audit == [a for row in records for a in row["response_audit"]]
    assert len(audit) == sum(len(row["model_calls"]) for row in records)
    results = []
    for row in records:
        identifier = row["question"]["question_id"]
        assert row["response"] and row["version"] == VERSION
        assert row["configuration"]["limits"] == manifest["limits"]
        if manifest["stage"] == "benchmark":
            assert row["question"] == input_map[identifier]
            assert question_hash(row["question"]["question"]) not in histories["question_hashes"]
        else:
            assert question_hash(row["question"]["question"]) not in {question_hash(q["question"]) for q in inputs}
        assert len(row["model_calls"]) <= limits.max_model_turns
        assert row["research_turns"] <= limits.max_research_turns and row["final_attempts"] <= limits.max_final_attempts
        assert row["tool_actions"] <= limits.max_tool_actions
        for number, call in enumerate(row["model_calls"]):
            request = json.loads(call["user_prompt"])
            assert request["question"] == row["question"]
            assert set(request["question"]) == {"question_id", "question", "options"}
            assert call["system_prompt"] == RESILIENT_SYSTEM and len(call["attempts"]) == 1
            assert len((call["system_prompt"] + call["user_prompt"]).encode()) <= limits.max_input_bytes
            if number == 0:
                assert request["history"] == [] and request["evidence_registry"] == {}
            if request["phase"] == "finalize":
                assert request["tools"] == [] and request["remaining"]["tool_actions"] == 0
                assert request["action_schema"] == final_schema(request["question"], request["evidence_registry"])
            else:
                assert [t["name"] for t in request["tools"]] == ["search_pubmed", "read_pubmed_abstract"]
        backend = BoundReplay(row["model_calls"])
        phased = PhaseBackend(backend, policy["native_final_json_schema"])
        client = ModelClient(phased, config, Redactor())

        def gate(request):
            if backend.index >= len(row["model_calls"]) and row["closing_reason"] == "budget_blocked":
                raise BudgetBlocked("Saved pre-call budget denial.")

        def transient():
            failure = row["response_audit"][backend.index-1]
            return failure.get("http_status") in {429, 500, 502, 503, 504} or failure.get("error_type") in {
                "APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectTimeout", "ReadTimeout"}

        adapter = ResilientJSONModel(client, gate, transient)
        http = HTTPReplay(row["literature"]["http_requests"], run / "http-snapshots")
        tools = ClinicalLiteratureTools(PubMedClient(pm_config, transport=http))
        with patch("medical_copilot.pubmed.time.sleep", lambda seconds: None):
            replayed = ResilientClinicalAgent(adapter, tools.definitions(), limits).run(row["question"])
        for field in ["status", "response", "history", "evidence", "tool_actions", "model_turns", "issues", "trace",
                      "research_turns", "final_attempts", "provider_failures", "closing_reason", "output_origin"]:
            assert without_transient(replayed[field]) == without_transient(row[field]), (identifier, field)
        assert phased.formats == row["transport_formats"]
        assert backend.index == len(row["model_calls"]) and http.index == len(row["literature"]["http_requests"])
        assert without_transient(tools.events) == without_transient(row["literature"]["events"])
        results.append({"question_id": identifier, "answer_sha256": file_hash(run / "answers" / (identifier + ".json")),
            "status": row["status"], "replayed_model_responses": backend.index, "replayed_http_responses": http.index,
            "native_final_calls": len(backend.schemas)})
    summary = read_json(run / "summary.json")
    assert len(records) == summary["planned"] == summary["recorded"] == summary["terminal_message_coverage"]
    assert len(audit) == summary["provider_requests"]
    assert math.isclose(sum(a.get("cost_usd") or 0 for a in audit), summary["reported_or_estimated_cost_usd"], abs_tol=1e-12)
    if manifest["stage"] == "benchmark":
        scores = [score_record(row, key_map[row["question"]["question_id"]]) for row in records]
        assert scores == read_json(run / "scores.json")["scores"]
        correct = sum(s["correct"] is True for s in scores)
        assert correct == summary["correct"] and correct / len(records) == summary["objective_score_planned"]
        assert wilson(correct, len(records)) == summary["wilson_95_interval_planned"]
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".xml", ".md"}:
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"sk-or-v1-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9_-]{15,}", text)
            assert '"reasoning_content"' not in text and '"reasoning_text"' not in text
    return {"passed": True, "run_id": run.name, "cases": results,
        "active_v3_graph_prompt_profile_unchanged": True, "sampling_keys_and_wording_verified": True,
        "new_model_requests": 0, "new_pubmed_requests": 0, "credential_files_read": 0,
        "interpretation": "Exact operational replay and objective scoring; substantive answer/source appraisal remains separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[1], args.run_dir.resolve())
    if args.output:
        save_trace(args.output.resolve(), result)
    print(json.dumps(result, indent=2))
