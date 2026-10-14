"""Verify and replay the restarted pilot/baseline without credentials or network."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from langchain_core.messages import AIMessage

from medical_copilot.clinical_evaluation import DEFAULT_PROFILE, score_record
from medical_copilot.clinical_graph import ClinicalAgent, JSONActionModel, Limits, SYSTEM, save_trace
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.models import ModelClient, Redactor


class SavedResponseBackend:
    """An offline backend; it has no credential resolver or network operation."""

    def __init__(self, call):
        self.call, self.invocations = call, 0

    def invoke(self, messages):
        self.invocations += 1
        assert self.invocations == 1, "Replay must not add retries."
        assert messages[0].content == self.call["system_prompt"]
        assert messages[1].content == self.call["user_prompt"]
        if self.call["status"] == "error":
            raise RuntimeError("Offline replay of recorded provider failure.")
        return AIMessage(content=self.call.get("response", ""),
                         response_metadata=self.call.get("response_metadata", {}),
                         usage_metadata=self.call.get("usage") or None)


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def verify(root, run_dir):
    manifest = read_json(run_dir / "manifest.json")
    assert manifest.get("finished_at_utc"), "Collection is still running."
    assert manifest["stage"] in {"pilot", "baseline"}
    assert manifest["config_sha256"] == file_hash(root / DEFAULT_PROFILE)
    assert manifest["system_sha256"] == hashlib.sha256(SYSTEM.encode()).hexdigest()
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(run_dir / "source-snapshot" / name) == digest
        assert file_hash(root / "src/medical_copilot" / name) == digest
    for name, expected in [("frozen-ten-questions.json", "questions_sha256"),
                           ("answer-key.json", "answer_key_sha256"),
                           ("protocol.md", "protocol_sha256"),
                           ("qualitative-rubric.json", "rubric_sha256")]:
        assert file_hash(run_dir / name) == manifest[expected]
    config = load_config(root / DEFAULT_PROFILE)
    assert config.max_retries == config.rate_limit_retries == config.json_repair_attempts == 0
    assert config.provider_options["extra_body"]["reasoning"] == {"enabled": True, "exclude": True}
    assert config.provider_options["extra_body"]["provider"]["allow_fallbacks"] is False
    questions = read_json(run_dir / "frozen-ten-questions.json")["questions"]
    inputs = {question["question_id"]: question for question in questions}
    keys = {key["question_id"]: key for key in read_json(run_dir / "answer-key.json")["keys"]}
    selection = read_json(run_dir / "selection-manifest.json")
    for item in selection["selection"]:
        digest = hashlib.sha256(json.dumps(inputs[item["question_id"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert digest == item["question_sha256"]
    assert len(inputs) == len(keys) == 10
    audits = read_json(run_dir / "usage-audit.json")["records"]
    records = [read_json(path) for path in sorted((run_dir / "answers").glob("*.json"))]
    summary = read_json(run_dir / "summary.json")
    assert len(records) == len(audits) == summary["recorded"] == summary["provider_requests"]
    assert len(records) == len(manifest["question_ids"]), "Incomplete run: report unattempted cases separately."
    results = []
    for record in records:
        question = record["question"]
        identifier = question["question_id"]
        assert set(question) == {"question_id", "question", "options"}
        if manifest["stage"] == "baseline":
            assert question == inputs[identifier]
        else:
            assert question["question"] not in {item["question"] for item in questions}
        assert len(record["model_calls"]) == len(record["response_audit"]) == 1
        assert record["model_turns"] == 1
        assert record["tool_actions"] == 0 and record["evidence"] == {}
        assert record["configuration"]["tools"] == []
        assert not any(event["node"] == "execute_tool" for event in record["trace"])
        call = record["model_calls"][0]
        request = json.loads(call["user_prompt"])
        assert set(request) == {"question", "history", "evidence_registry", "tools", "action_schema", "remaining"}
        assert request["question"] == question
        assert request["history"] == request["tools"] == [] and request["evidence_registry"] == {}
        assert request["remaining"]["tool_actions"] == 0
        assert len(call["attempts"]) == 1
        audit = record["response_audit"][0]
        assert audit == next(item for item in audits if item["question_id"] == identifier)
        if audit["cost_usd"] is not None:
            assert audit["cost_usd"] <= audit["estimated_maximum_usd"]
        backend = SavedResponseBackend(call)
        client = ModelClient(backend, config, Redactor())

        def offline_gate(replay_request):
            assert replay_request == request

        agent = ClinicalAgent(JSONActionModel(client, offline_gate), limits=Limits(**record["configuration"]["limits"]))
        replayed = agent.run(question)
        for field in ["status", "response", "evidence", "tool_actions", "model_turns", "history", "issues", "trace"]:
            assert without_timing(replayed[field]) == without_timing(record[field]), (identifier, field)
        assert backend.invocations == 1
        results.append({"question_id": identifier, "answer_sha256": file_hash(run_dir / "answers" / (identifier + ".json")),
                        "status": record["status"], "replayed_saved_responses": backend.invocations})
    assert math.isclose(sum(item["cost_usd"] or 0 for item in audits), summary["reported_or_estimated_cost_usd"], abs_tol=1e-12)
    if manifest["stage"] == "baseline":
        calculated = [score_record(record, keys[record["question"]["question_id"]]) for record in records]
        assert calculated == read_json(run_dir / "scores.json")["scores"]
        assert sum(score["correct"] is True for score in calculated) == summary["correct"]
        assert sum(score["valid_completed_answer"] for score in calculated) == summary["valid_completed"]
        assert [item["question_id"] for item in results] == manifest["question_ids"]
    suspicious = re.compile(r"sk-or-v1-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9_-]{15,}")
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md"}:
            raw = path.read_text(encoding="utf-8")
            assert not suspicious.search(raw), "Credential-shaped material found; path/value withheld."
            assert '"reasoning_text"' not in raw and '"reasoning_content"' not in raw
    final = read_json(run_dir / "final-ledger.json")
    delta = final.get("total_usage", manifest["initial_ledger"]["total_usage"]) - manifest["initial_ledger"]["total_usage"]
    return {"passed": True, "stage": manifest["stage"], "run_id": manifest["run_id"],
            "checks": "Frozen source/input/key/profile/protocol/rubric hashes; exact replayed prompts/actions/answers/traces; empty evidence/history/tools; separate keys; numeric costs; no credential-shaped strings or private reasoning fields.",
            "cases": results, "new_provider_requests": 0, "new_pubmed_requests": 0,
            "credential_files_read": 0, "tool_execution_attempts": 0,
            "account_ledger_delta_usd": delta,
            "provider_reported_or_estimated_usd": summary["reported_or_estimated_cost_usd"],
            "ledger_delta_matches_recorded_usage": math.isclose(delta, summary["reported_or_estimated_cost_usd"], abs_tol=1e-9)}


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
