"""Replay the resilient cohort with saved model/HTTP responses, without credentials."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from unittest.mock import patch

from medical_copilot.clinical_evaluation import DEFAULT_PROFILE, score_record
from medical_copilot.clinical_graph import SYSTEM, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_resilient import (
    VERSION, RESILIENT_SYSTEM, PhaseBackend, ResilientClinicalAgent,
    ResilientJSONModel, ResilientLimits, final_schema,
)
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.models import ModelClient, Redactor
from medical_copilot.pubmed import PubMedClient, PubMedConfig
from verify_clinical_literature import HTTPReplay, ModelReplay, without_transient


class BoundReplay(ModelReplay):
    def __init__(self, calls):
        super().__init__(calls)
        self.schemas = []

    def bind(self, *, response_format):
        request = json.loads(self.calls[self.index]["user_prompt"])
        assert request["phase"] == "finalize"
        assert response_format == {"type": "json_schema", "json_schema": {
            "name": "medical_final_answer", "strict": True,
            "schema": final_schema(request["question"], request["evidence_registry"])}}
        self.schemas.append(response_format)
        return self


def verify(root, run):
    manifest = read_json(run / "manifest.json")
    assert manifest.get("finished_at_utc") and manifest["protocol_version"] == VERSION
    assert manifest["config_sha256"] == file_hash(root / DEFAULT_PROFILE)
    assert manifest["system_sha256"] == hashlib.sha256(SYSTEM.encode()).hexdigest()
    assert manifest["effective_system_sha256"] == hashlib.sha256(RESILIENT_SYSTEM.encode()).hexdigest()
    assert read_json(run / "system-instructions.json")["effective_system"] == RESILIENT_SYSTEM
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(run / "source-snapshot" / name) == digest
        assert file_hash(root / "src/medical_copilot" / name) == digest
    baseline = root / manifest["baseline_origin"]
    bm = read_json(baseline / "manifest.json")
    assert file_hash(baseline / "manifest.json") == manifest["baseline_manifest_sha256"]
    assert file_hash(root / manifest["prior_origin"] / "manifest.json") == manifest["prior_manifest_sha256"]
    for name, digest in bm["source_hashes"].items():
        assert manifest["source_hashes"][name] == digest
    for name, key in [("frozen-ten-questions.json", "questions_sha256"), ("answer-key.json", "answer_key_sha256"),
                      ("protocol.md", "protocol_sha256"), ("policy.json", "policy_sha256"),
                      ("retrieval-rubric.json", "retrieval_rubric_sha256"), ("answer-rubric.json", "answer_rubric_sha256"),
                      ("pubmed-config.json", "pubmed_config_sha256")]:
        assert file_hash(run / name) == manifest[key]
    assert manifest["questions_sha256"] == bm["questions_sha256"]
    assert manifest["answer_key_sha256"] == bm["answer_key_sha256"]
    config = load_config(root / DEFAULT_PROFILE)
    pm_config = PubMedConfig.model_validate(read_json(run / "pubmed-config.json"))
    policy = read_json(run / "policy.json")
    inputs = {q["question_id"]: q for q in read_json(run / "frozen-ten-questions.json")["questions"]}
    keys = {q["question_id"]: q for q in read_json(run / "answer-key.json")["keys"]}
    records = [read_json(path) for path in sorted((run / "answers").glob("*.json"))]
    assert [r["question"]["question_id"] for r in records] == sorted(manifest["question_ids"])
    audit = read_json(run / "usage-audit.json")["records"]
    assert audit == [entry for row in records for entry in row["response_audit"]]
    assert len(audit) == sum(len(row["model_calls"]) for row in records)
    results = []
    for row in records:
        identifier = row["question"]["question_id"]
        assert row["response"] and row["version"] == VERSION
        limits = ResilientLimits(**row["configuration"]["limits"])
        expected_limits = dict(manifest["limits"])
        if manifest["stage"] == "probe":
            expected_limits["max_research_turns"] = 0
        assert limits.model_dump() == expected_limits
        if manifest["stage"] == "autonomous":
            assert row["question"] == inputs[identifier]
        else:
            assert row["question"]["question"] not in {q["question"] for q in inputs.values()}
        assert len(row["model_calls"]) <= limits.max_model_turns
        assert row["tool_actions"] <= limits.max_tool_actions
        assert row["research_turns"] <= limits.max_research_turns
        assert row["final_attempts"] <= limits.max_final_attempts
        if row["model_calls"]:
            first = json.loads(row["model_calls"][0]["user_prompt"])
            assert first["history"] == [] and first["evidence_registry"] == {}
        for call in row["model_calls"]:
            request = json.loads(call["user_prompt"])
            assert set(request) == {"question", "history", "evidence_registry", "tools", "action_schema", "remaining", "phase", "closing_reason", "execution_policy"}
            assert request["question"] == row["question"]
            assert len(call["attempts"]) == 1
            assert call["system_prompt"] == RESILIENT_SYSTEM
            assert len((call["system_prompt"] + call["user_prompt"]).encode()) <= limits.max_input_bytes
            assert call["stage"] == "clinical_" + request["phase"]
            if request["phase"] == "finalize":
                assert request["tools"] == [] and request["remaining"]["tool_actions"] == 0
                assert request["action_schema"] == final_schema(request["question"], request["evidence_registry"])
            else:
                assert [tool["name"] for tool in request["tools"]] == ["search_pubmed", "read_pubmed_abstract"]
        http = HTTPReplay(row["literature"]["http_requests"], run / "http-snapshots")
        tools = ClinicalLiteratureTools(PubMedClient(pm_config, transport=http))
        backend = BoundReplay(row["model_calls"])
        phased = PhaseBackend(backend, policy["native_final_json_schema"])
        client = ModelClient(phased, config, Redactor())
        # The new cohort contains successful provider responses. Operational-error
        # recovery has separate offline unit coverage, not fabricated live events.
        assert all(call["status"] != "error" for call in row["model_calls"])
        adapter = ResilientJSONModel(client, lambda request: None)
        with patch("medical_copilot.pubmed.time.sleep", lambda seconds: None):
            replayed = ResilientClinicalAgent(adapter, tools.definitions(), limits).run(row["question"])
        for field in ["status", "response", "history", "evidence", "tool_actions", "model_turns", "issues", "trace",
                      "research_turns", "final_attempts", "provider_failures", "closing_reason", "output_origin"]:
            assert without_transient(replayed[field]) == without_transient(row[field]), (identifier, field)
        assert phased.formats == row["transport_formats"]
        assert backend.index == len(row["model_calls"])
        assert http.index == len(row["literature"]["http_requests"])
        assert without_transient(tools.events) == without_transient(row["literature"]["events"])
        results.append({"question_id": identifier, "answer_sha256": file_hash(run / "answers" / (identifier + ".json")),
                        "replayed_model_responses": backend.index, "native_final_schema_calls": len(backend.schemas),
                        "replayed_logical_http_requests": http.index, "status": row["status"], "tool_actions": row["tool_actions"]})
    summary = read_json(run / "summary.json")
    assert len(records) == summary["recorded"] == summary["terminal_message_coverage"]
    assert len(audit) == summary["provider_requests"]
    assert math.isclose(sum(item.get("cost_usd") or 0 for item in audit), summary["reported_or_estimated_cost_usd"], abs_tol=1e-12)
    assert summary["application_fallbacks"] == sum(row["output_origin"] == "application" for row in records)
    if manifest["stage"] == "autonomous":
        calculated = [score_record(row, keys[row["question"]["question_id"]]) for row in records]
        assert calculated == read_json(run / "scores.json")["scores"]
        assert sum(score["correct"] is True for score in calculated) == summary["correct"]
    suspicious = re.compile(r"sk-or-v1-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9_-]{15,}")
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".xml"}:
            raw = path.read_text(encoding="utf-8")
            assert not suspicious.search(raw), "Credential-shaped value found; content withheld."
            assert '"reasoning_content"' not in raw and '"reasoning_text"' not in raw
    return {"passed": True, "run_id": run.name, "cases": results, "baseline_core_source_unchanged": True,
            "question_key_profile_unchanged": True, "new_provider_requests": 0, "new_pubmed_requests": 0,
            "credential_files_read": 0, "interpretation": "Saved prompt/phase/schema/action/HTTP replay; medical quality is separately assessed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[1], args.run_dir.resolve())
    if args.output:
        save_trace(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
