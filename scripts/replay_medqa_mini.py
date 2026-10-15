"""Replay archived direct/agent graphs with recorded responses and no network calls."""
from __future__ import annotations

import copy
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

from medical_copilot.config import ModelConfig
from medical_copilot.evaluation import read_json, write_json
from medical_copilot.models import ModelClient

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/benchmark-runs/openweight-medqa-high-20261011"


class RecordedBackend:
    def __init__(self, calls):
        self.calls, self.index = calls, 0

    def invoke(self, messages):
        call = self.calls[self.index]
        self.index += 1
        assert messages[0].content == call["system_prompt"]
        assert messages[1].content == call["user_prompt"]
        if call["status"] == "error":
            raise type(call.get("error_type", "RecordedFailure"), (Exception,), {})("Archived failure fixture")
        return AIMessage(content=call["response"], usage_metadata=call.get("usage") or None, response_metadata=call.get("response_metadata") or {})


class RecordedPubMed:
    def __init__(self, record):
        self.record, self.calls = record, 0

    def search(self, query, max_results):
        self.calls += 1
        decision = self.record.get("decision", self.record.get("partial_state", {}).get("decision", {}))
        assert query == decision["query"] and max_results == 3 and self.calls == 1
        return copy.deepcopy(self.record.get("retrieval", self.record.get("partial_state", {}).get("retrieval", {})))


def main():
    manifest = read_json(RUN / "manifest.json")
    assert manifest.get("completed_at_utc")
    spec = importlib.util.spec_from_file_location("medical_copilot.medqa_mini_archived", RUN / "source-snapshot/medqa_mini.py")
    archive = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = archive
    spec.loader.exec_module(archive)
    questions = {question["id"]: question for question in read_json(RUN / "questions.json")["questions"]}
    profiles = {profile["id"]: ModelConfig.model_validate(profile["config"]).model_copy(update={"min_interval_seconds": 0}) for profile in manifest["profiles"]}
    results = []
    for path in sorted(RUN.glob("models/*/answers/*.json")):
        record = read_json(path)
        if record["status"].startswith("not_run"):
            assert not record["model_calls"]
            results.append({"profile": record["profile"], "mode": record["mode"], "question_id": record["question_id"], "status": "verified_unattempted"})
            continue
        backend = RecordedBackend(record["model_calls"])
        retrieval = RecordedPubMed(record)
        client = ModelClient(backend, profiles[record["profile"]])
        repeated = archive.MiniCopilot(client, retrieval, record["mode"]).run(questions[record["question_id"]])
        for key in ("status", "answer", "decision", "retrieval", "trace", "error_type", "partial_state"):
            assert repeated.get(key) == record.get(key), (path.name, key)
        assert backend.index == len(record["model_calls"])
        results.append({"profile": record["profile"], "mode": record["mode"], "question_id": record["question_id"], "status": "replayed", "recorded_model_calls": backend.index})
    output = {"passed": True, "cohort": RUN.name, "case_records": len(results), "new_model_calls": 0, "new_pubmed_calls": 0, "results": results}
    filename = "medqa-mini-ling-replay.json" if manifest.get("primary_cohort") else "medqa-mini-replay.json"
    write_json(ROOT / "data/checks" / filename, output)
    print(json.dumps({key: output[key] for key in ("passed", "case_records", "new_model_calls", "new_pubmed_calls")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", action="store_true")
    if parser.parse_args().extension:
        RUN = ROOT / "data/evaluation/benchmark-runs/ling-sante-medqa-extension-20261011"
    main()
