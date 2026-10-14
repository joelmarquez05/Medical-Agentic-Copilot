"""Replay actual comparison graphs offline with each archived prompt/schema."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage
from medical_copilot.config import ModelConfig
from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.models import ModelClient

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evaluation/comparison-runs"


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(child) for key, child in value.items() if key not in {"elapsed_seconds", "started_at_utc"}}
    if isinstance(value, list):
        return [canonical(child) for child in value]
    return value


def prompt_value(text):
    try:
        return canonical(json.loads(text))
    except ValueError:
        return text


class RecordedBackend:
    def __init__(self, calls):
        self.calls, self.position, self.attempt = calls, 0, 0

    def invoke(self, messages):
        record = self.calls[self.position]
        assert messages[0].content == record["system_prompt"]
        assert prompt_value(messages[1].content) == prompt_value(record["user_prompt"])
        attempts = record["attempts"]
        current = attempts[self.attempt]
        self.attempt += 1
        if self.attempt == len(attempts):
            self.position += 1
            self.attempt = 0
        if current["status"] == "error":
            error_type = type(current["error_type"], (Exception,), {})
            error = error_type("Recorded provider failure; no network request.")
            if "ratelimit" in current["error_type"].lower():
                error.status_code = 429
            raise error
        return AIMessage(content=record.get("response", ""), usage_metadata=record.get("usage") or None, response_metadata=record.get("response_metadata") or {})


def main():
    cases, skipped = [], []
    for index, run_id in enumerate(["selected-five-pilot-20261010", "selected-five-main-20261010", "selected-five-recovery-20261010"]):
        run = BASE / run_id
        manifest = read_json(run / "manifest.json")
        path = run / "source-snapshot/comparison.py"
        assert file_hash(path) == manifest["source_code_sha256"]["comparison.py"]
        module_name = f"_archived_r5_comparison_{index}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        profiles = {profile["id"]: profile for profile in manifest["profiles"]}
        for answer_path in sorted((run / "models").glob("*/answers/*.json")):
            observed = read_json(answer_path)
            if observed["status"] == "not_run_provider_unavailable":
                assert observed["model_calls"] == [] and observed["answer"] == ""
                skipped.append({"run_id": run_id, "profile": observed["profile"], "question_id": observed["question_id"]})
                continue
            config = ModelConfig.model_validate(profiles[observed["profile"]]["config"]).model_copy(update={"min_interval_seconds": 0, "rate_limit_wait_seconds": 0})
            backend = RecordedBackend(observed["model_calls"])
            evidence = read_json(run / "evidence" / (observed["question_id"] + ".json"))
            result = module.ComparisonCopilot(ModelClient(backend, config), evidence).run(observed["question"])
            assert backend.position == len(observed["model_calls"]) and backend.attempt == 0
            for field in ["status", "answer", "sources", "trace", "revision_count", "mechanical_issues", "drafts", "reviews", "partial_answer", "error"]:
                assert result.get(field) == observed.get(field), f"Replay mismatch: {run_id}/{observed['profile']}/{observed['question_id']}/{field}"
            cases.append({"run_id": run_id, "profile": observed["profile"], "question_id": observed["question_id"], "prompt_version": module.PROMPT_VERSION, "substantive_inputs_match": True, "outputs_and_flags_match": True})
    report = {"live_provider_requests": 0, "live_pubmed_requests": 0, "replayed_graph_cases": len(cases), "cases": cases, "unattempted_records_checked": skipped, "interpretation": "Recorded-output engineering replay with archived prompts/schemas, including failures. Not a new model/medical/retrieval evaluation or independent validation."}
    write_json(ROOT / "data/checks/requirement-5-replay.json", report)
    print(f"Replayed {len(cases)} actual graphs; verified {len(skipped)} unattempted records; zero network calls.")


if __name__ == "__main__":
    main()
