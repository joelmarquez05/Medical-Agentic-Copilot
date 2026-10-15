"""Replay a saved four-model comparison stage without network or credentials."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

from medical_copilot.clinical_benchmark import BENCHMARK, validate_benchmark
from medical_copilot.clinical_evaluation import score_record
from medical_copilot.clinical_graph import BudgetBlocked, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_model_comparison import MODELS, PROFILES, episode_order, validate_reference
from medical_copilot.clinical_resilient import RESILIENT_SYSTEM, PhaseBackend, ResilientClinicalAgent, ResilientJSONModel, ResilientLimits, final_schema
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.models import ModelClient, Redactor
from medical_copilot.pubmed import PubMedClient, PubMedConfig
from verify_clinical_resilient import BoundReplay
from verify_clinical_literature import HTTPReplay, without_transient


def verify(root, run):
    reference, selection, questions, keys = validate_reference(root)
    manifest = read_json(run / "manifest.json")
    assert manifest["finished_at_utc"] and manifest["requirement"] == 5
    assert manifest["models"] == list(MODELS)
    assert manifest["episode_order"] == [list(pair) for pair in episode_order(manifest["question_ids"])]
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(run / "source-snapshot" / name) == file_hash(root / "src/medical_copilot" / name) == digest
    for field, name in [("questions_sha256", "frozen-benchmark-questions.json"), ("answer_key_sha256", "answer-key.json"),
                        ("selection_manifest_sha256", "selection-manifest.json"), ("protocol_sha256", "protocol.md"),
                        ("rubric_sha256", "rubric.json"), ("retrieval_rubric_sha256", "retrieval-rubric.json"),
                        ("policy_sha256", "policy.json"), ("pubmed_config_sha256", "pubmed-config.json")]:
        assert file_hash(run / name) == manifest[field]
    assert manifest["questions_sha256"] == selection["questions_sha256"]
    assert manifest["answer_key_sha256"] == selection["answer_key_sha256"]
    assert file_hash(root / BENCHMARK / "manifest.json") == manifest["selection_manifest_sha256"]
    assert hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest() == manifest["effective_system_sha256"]
    assert read_json(run / "system-instructions.json")["effective_system"] == RESILIENT_SYSTEM
    assert file_hash(root / manifest["reference_origin"] / "manifest.json") == manifest["reference_manifest_sha256"]
    for field, hash_field in [("probe_origin", "probe_manifest_sha256"), ("pilot_origin", "pilot_manifest_sha256")]:
        if manifest[field]:
            assert file_hash(Path(manifest[field]) / "manifest.json") == manifest[hash_field]
    policy = read_json(run / "policy.json")
    pm_config = PubMedConfig.model_validate(read_json(run / "pubmed-config.json"))
    question_map, key_map = {q["question_id"]: q for q in questions}, {k["question_id"]: k for k in keys}
    results = []
    for slug in MODELS:
        directory = run / slug
        config = load_config(root / PROFILES / (slug + ".json"))
        assert manifest["profile_hashes"][slug] == file_hash(root / PROFILES / (slug + ".json"))
        assert read_json(directory / "config.json") == config.public_dict()
        rows = [read_json(p) for p in sorted((directory / "answers").glob("*.json"))]
        assert {r["question"]["question_id"] for r in rows} == set(manifest["question_ids"])
        audit = read_json(directory / "usage-audit.json")["records"] if (directory / "usage-audit.json").exists() else []
        assert audit == [a for row in rows for a in row["response_audit"]]
        for row in rows:
            identifier = row["question"]["question_id"]
            if manifest["stage"] == "benchmark":
                assert row["question"] == question_map[identifier]
            else:
                assert row["question"]["question"] not in {q["question"] for q in questions}
            limits = ResilientLimits(**row["configuration"]["limits"])
            expected = dict(manifest["limits"])
            if manifest["stage"] == "probes":
                expected["max_research_turns"] = 0
            assert limits.model_dump() == expected
            for number, call in enumerate(row["model_calls"]):
                request = json.loads(call["user_prompt"])
                assert request["question"] == row["question"] and set(request["question"]) == {"question_id", "question", "options"}
                assert call["system_prompt"] == RESILIENT_SYSTEM and len(call["attempts"]) == 1
                assert len((call["system_prompt"] + call["user_prompt"]).encode()) <= limits.max_input_bytes
                if number == 0:
                    assert request["history"] == [] and request["evidence_registry"] == {}
                if request["phase"] == "finalize":
                    assert request["tools"] == [] and request["action_schema"] == final_schema(request["question"], request["evidence_registry"])
                else:
                    assert [t["name"] for t in request["tools"]] == ["search_pubmed", "read_pubmed_abstract"]
            backend = BoundReplay(row["model_calls"])
            phased = PhaseBackend(backend, policy["native_final_json_schema"])
            client = ModelClient(phased, config, Redactor())

            def gate(request):
                if backend.index >= len(row["model_calls"]) and row["closing_reason"] == "budget_blocked":
                    raise BudgetBlocked("Saved pre-call budget denial.")

            def transient():
                failure = row["response_audit"][backend.index - 1]
                return failure.get("http_status") in {429, 500, 502, 503, 504} or failure.get("error_type") in {
                    "APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectTimeout", "ReadTimeout"}

            adapter = ResilientJSONModel(client, gate, transient)
            http = HTTPReplay(row["literature"]["http_requests"], directory / "http-snapshots")
            tools = ClinicalLiteratureTools(PubMedClient(pm_config, transport=http))
            with patch("medical_copilot.pubmed.time.sleep", lambda seconds: None):
                replayed = ResilientClinicalAgent(adapter, tools.definitions(), limits).run(row["question"])
            for field in ["status", "response", "history", "evidence", "tool_actions", "model_turns", "issues", "trace",
                          "research_turns", "final_attempts", "provider_failures", "closing_reason", "output_origin"]:
                assert without_transient(replayed[field]) == without_transient(row[field]), (slug, identifier, field)
            assert phased.formats == row["transport_formats"]
            assert backend.index == len(row["model_calls"]) and http.index == len(row["literature"]["http_requests"])
            assert without_transient(tools.events) == without_transient(row["literature"]["events"])
            results.append({"model_profile": slug, "question_id": identifier,
                            "answer_sha256": file_hash(directory / "answers" / (identifier + ".json")),
                            "model_responses": backend.index, "http_responses": http.index, "status": row["status"]})
        summary = read_json(directory / "summary.json")
        assert len(rows) == summary["recorded"] == summary["planned"]
        assert sum(bool(a["request_sent"]) for a in audit) == summary["provider_requests"]
        assert math.isclose(sum(a.get("cost_usd") or 0 for a in audit), summary["reported_or_estimated_cost_usd"], abs_tol=1e-12)
        if manifest["stage"] == "benchmark":
            scores = [score_record(row, key_map[row["question"]["question_id"]]) for row in rows]
            assert scores == read_json(directory / "scores.json")["scores"]
            assert sum(s["correct"] is True for s in scores) == summary["correct"]
    return {"passed": True, "run_id": run.name, "stage": manifest["stage"], "cases": results,
            "frozen_graph_prompt_questions_keys_verified": True, "new_model_requests": 0,
            "new_pubmed_requests": 0, "credential_files_read": 0,
            "interpretation": "Exact engineering replay; substantive explanation/source review remains separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[1], args.run_dir.resolve())
    if args.output:
        save_trace(args.output.resolve(), result)
    print(json.dumps({"passed": True, "episodes": len(result["cases"]),
                      "saved_model_responses": sum(r["model_responses"] for r in result["cases"]),
                      "saved_http_responses": sum(r["http_responses"] for r in result["cases"]), "new_model_requests": 0}))
